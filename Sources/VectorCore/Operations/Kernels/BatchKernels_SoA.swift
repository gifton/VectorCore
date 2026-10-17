//
//  BatchKernels_SoA.swift
//  VectorCore
//
//  Structure-of-Arrays batch kernels (2-way and 4-way register blocking, and lane-outer
//  tiling for large batches). No cache or speed advantage over AoS is measured in this repository.
//

import Foundation
import simd
#if canImport(Darwin)
import Darwin  // For sqrt, max, min on Apple platforms
#elseif canImport(Glibc)
import Glibc   // For sqrt, max, min on Linux
#elseif canImport(WinSDK)
import WinSDK  // For sqrt, max, min on Windows
#endif

#if canImport(Accelerate)
import Accelerate  // Gates the Manhattan section (#if canImport(Accelerate)); those kernels do not call vDSP
#endif

/// Batch kernels over a Structure-of-Arrays candidate layout.
///
/// In the SoA layout all candidates' SIMD4 data for one lane are stored contiguously
/// (`SoA.lanePointer(i)`), so a kernel that walks candidates for a fixed lane reads one
/// contiguous column. That this reduces cache misses or time relative to AoS is a design
/// rationale, not a measured result; it is not verified here.
public enum BatchKernels_SoA {

    // MARK: - Generic Implementation

    /// Generic Euclidean squared distance kernel using SoA layout with 2-way blocking
    ///
    /// Algorithm:
    /// 1. Process candidates in blocks of 2 (two accumulators)
    /// 2. For each lane, load query data once and process both candidates
    /// 3. Accumulate differences using dual SIMD4 accumulators
    /// 4. Horizontal reduction and scalar output
    @inlinable
    internal static func euclid2_blocked<Vector: SoACompatible>(
        query: Vector,
        soa: SoA<Vector>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        let N = soa.count
        let L = soa.lanes

        guard N > 0 else { return }

        #if DEBUG
        assert(out.count >= N, "Output buffer too small: \(out.count) < \(N)")
        #endif

        let queryStorage = query.storage

        // 2-way blocking configuration
        let blockSize = 2
        let blockedN = (N / blockSize) * blockSize

        // Main loop: Process candidates in blocks of 2
        for j in stride(from: 0, to: blockedN, by: blockSize) {

            // Dual accumulators for 2-way blocking
            var acc0 = SIMD4<Float>.zero
            var acc1 = SIMD4<Float>.zero

            // Inner loop: Process all lanes for current candidate pair
            for i in 0..<L {
                // Load the query lane once and reuse it for both candidates
                let q_i = queryStorage[i]

                // Get pointer to contiguous candidate data for this lane
                let candidateLanePtr = soa.lanePointer(i)

                // Load both candidates' data (contiguous in memory due to SoA layout)
                let c0 = candidateLanePtr[j]       // Candidate j, lane i
                let c1 = candidateLanePtr[j + 1]   // Candidate j+1, lane i

                // Compute differences and accumulate squared distances
                let diff0 = q_i - c0
                let diff1 = q_i - c1

                // addProduct is a fused multiply-add (one rounding per element)
                acc0.addProduct(diff0, diff0)
                acc1.addProduct(diff1, diff1)
            }

            // Horizontal reduction: sum SIMD4 elements to scalars
            out[j] = acc0.sum()
            out[j + 1] = acc1.sum()
        }

        // Tail handling: Process remaining candidate if N is odd
        if blockedN < N {
            let j = N - 1
            var acc = SIMD4<Float>.zero

            for i in 0..<L {
                let q_i = queryStorage[i]
                let candidateLanePtr = soa.lanePointer(i)
                let c = candidateLanePtr[j]

                let diff = q_i - c
                acc.addProduct(diff, diff)
            }

            out[j] = acc.sum()
        }
    }

    /// 4-way blocked squared Euclidean distance (four independent accumulator chains)
    ///
    /// Processes 4 candidates per block with 4 independent accumulator chains and loads
    /// each query lane once per block of 4 candidates. No throughput ratio versus the
    /// 2-way kernel is measured.
    @inlinable
    internal static func euclid2_blocked_4way<Vector: SoACompatible>(
        query: Vector,
        soa: SoA<Vector>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        let N = soa.count
        guard N > 0 else { return }

        #if DEBUG
        assert(out.count >= N, "Output buffer too small")
        #endif

        let lanes = soa.lanes
        let queryStorage = query.storage
        let blockedN = (N / 4) * 4

        // Process 4 candidates simultaneously
        for j in stride(from: 0, to: blockedN, by: 4) {
            var acc0 = SIMD4<Float>.zero
            var acc1 = SIMD4<Float>.zero
            var acc2 = SIMD4<Float>.zero
            var acc3 = SIMD4<Float>.zero

            // Discarded loads of the next block's first lanes (`prefetch` helper; not a prefetch instruction)
            if j + 4 < blockedN {
                for i in stride(from: 0, to: min(8, lanes), by: 2) {
                    prefetch(soa.lanePointer(i), offset: j + 4)
                }
            }

            // Process lanes with 4 independent accumulators
            for i in 0..<lanes {
                let q_i = queryStorage[i]
                let lanePtr = soa.lanePointer(i)

                let c0 = lanePtr[j]
                let c1 = lanePtr[j + 1]
                let c2 = lanePtr[j + 2]
                let c3 = lanePtr[j + 3]

                let diff0 = q_i - c0
                let diff1 = q_i - c1
                let diff2 = q_i - c2
                let diff3 = q_i - c3

                acc0.addProduct(diff0, diff0)
                acc1.addProduct(diff1, diff1)
                acc2.addProduct(diff2, diff2)
                acc3.addProduct(diff3, diff3)
            }

            out[j] = acc0.sum()
            out[j + 1] = acc1.sum()
            out[j + 2] = acc2.sum()
            out[j + 3] = acc3.sum()
        }

        // Handle remaining candidates (1-3 tail elements)
        for j in blockedN..<N {
            var acc = SIMD4<Float>.zero
            for i in 0..<lanes {
                let diff = queryStorage[i] - soa.lanePointer(i)[j]
                acc.addProduct(diff, diff)
            }
            out[j] = acc.sum()
        }
    }

    // MARK: - Cache-Tiling Threshold (Issue 1.2)

    /// Candidate-buffer byte size above which lane-outer **tiling** beats the
    /// register-blocked 4-way kernel.
    ///
    /// The 4-way kernels read each candidate's data once, but they iterate lanes
    /// innermost, producing `L` (= 128…384) interleaved, strided column streams. Above
    /// this threshold the tiled kernels are used instead; they read each lane's column
    /// as a single sequential stream.
    ///
    /// Rationale (heuristic; unverified hardware assumption, no crossover is measured in
    /// this repository): once the candidate buffer exceeds an L2-sized working budget,
    /// the hardware prefetcher may not track `L` streams and demand-miss latency is exposed.
    ///
    /// Set to 8 MiB. `shouldTile` is `count * lanes * 16 > 8 MiB`, so tiling applies for
    /// N > 4096 / 2730 / 1365 candidates at 512 / 768 / 1536 dimensions.
    @usableFromInline
    internal static let tilingByteThreshold = 8 * 1024 * 1024

    /// Whether a batch is large enough that lane-outer tiling should be preferred.
    @inline(__always)
    @usableFromInline
    internal static func shouldTile(count: Int, lanes: Int) -> Bool {
        // candidate buffer bytes = count * lanes * sizeof(SIMD4<Float>)
        count &* lanes &* MemoryLayout<SIMD4<Float>>.stride > tilingByteThreshold
    }

    // MARK: - Candidate-Tiled Euclidean (Issue 1.2)

    /// Lane-outer / candidate-inner squared Euclidean distance for very large batches.
    ///
    /// Restructures the traversal so each lane's candidate column is consumed as ONE
    /// long sequential stream, instead of the 4-way kernel's `L` interleaved strided
    /// streams. Per-candidate SIMD4 partials live in a small tile from
    /// `withUnsafeTemporaryAllocation` (stack or heap, the runtime's choice); the candidate
    /// columns themselves are read once. That the tile stays in L1 is an assumption, not verified.
    ///
    /// - Important: The per-candidate accumulation order over lanes (i = 0…L-1) and the
    ///   single horizontal `sum()` at the end are identical to `euclid2_blocked_4way`,
    ///   so results are **bitwise-identical** to the 4-way path (`tiledEuclid512MatchesFourWayBitwise`,
    ///   `tiledEuclid768MatchesFourWayBitwise`, `tiledEuclid1536MatchesFourWayBitwise`).
    @inlinable
    internal static func euclid2_tiled<Vector: SoACompatible>(
        query: Vector,
        soa: SoA<Vector>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        let N = soa.count
        guard N > 0 else { return }

        #if DEBUG
        assert(out.count >= N, "Output buffer too small: \(out.count) < \(N)")
        #endif

        let L = soa.lanes
        let queryStorage = query.storage

        // 256 candidates/tile → 4 KiB of SIMD4 accumulators, reused across every lane
        // while the candidate columns stream past (L1 residency is assumed, not verified).
        let tileWidth = 256

        var tileStart = 0
        while tileStart < N {
            let tileLen = Swift.min(tileWidth, N - tileStart)

            withUnsafeTemporaryAllocation(of: SIMD4<Float>.self, capacity: tileLen) { accs in
                // Reset this tile's accumulators.
                for t in 0..<tileLen {
                    accs[t] = .zero
                }

                // Lane-outer: stream each lane's candidate column sequentially.
                for i in 0..<L {
                    let q_i = queryStorage[i]
                    let lanePtr = soa.lanePointer(i) + tileStart
                    for t in 0..<tileLen {
                        let diff = q_i - lanePtr[t]
                        accs[t].addProduct(diff, diff)
                    }
                }

                // Horizontal reduction per candidate (single sum at the end → matches 4-way).
                for t in 0..<tileLen {
                    out[tileStart + t] = accs[t].sum()
                }
            }

            tileStart += tileLen
        }
    }

    // MARK: - Dimension-Specific Public APIs

    /// Euclidean squared distance for 512-dimensional vectors using SoA layout
    ///
    /// Tiles when the candidate buffer exceeds `tilingByteThreshold` (8 MiB), uses 4-way
    /// register blocking for N >= 4, and 2-way otherwise.
    @inlinable
    public static func euclid2_512(
        query: Vector512Optimized,
        soa: SoA512,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        if shouldTile(count: soa.count, lanes: soa.lanes) {
            euclid2_tiled(query: query, soa: soa, out: out)
        } else if soa.count >= 4 {
            euclid2_blocked_4way(query: query, soa: soa, out: out)
        } else {
            euclid2_blocked(query: query, soa: soa, out: out)
        }
    }

    /// Euclidean squared distance for 768-dimensional vectors using SoA layout
    ///
    /// Tiles when the candidate buffer exceeds `tilingByteThreshold` (8 MiB), uses 4-way
    /// register blocking for N >= 4, and 2-way otherwise.
    @inlinable
    public static func euclid2_768(
        query: Vector768Optimized,
        soa: SoA768,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        if shouldTile(count: soa.count, lanes: soa.lanes) {
            euclid2_tiled(query: query, soa: soa, out: out)
        } else if soa.count >= 4 {
            euclid2_blocked_4way(query: query, soa: soa, out: out)
        } else {
            euclid2_blocked(query: query, soa: soa, out: out)
        }
    }

    /// Euclidean squared distance for 1536-dimensional vectors using SoA layout
    ///
    /// Tiles when the candidate buffer exceeds `tilingByteThreshold` (8 MiB), uses 4-way
    /// register blocking for N >= 4, and 2-way otherwise.
    @inlinable
    public static func euclid2_1536(
        query: Vector1536Optimized,
        soa: SoA1536,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        if shouldTile(count: soa.count, lanes: soa.lanes) {
            euclid2_tiled(query: query, soa: soa, out: out)
        } else if soa.count >= 4 {
            euclid2_blocked_4way(query: query, soa: soa, out: out)
        } else {
            euclid2_blocked(query: query, soa: soa, out: out)
        }
    }

    // MARK: - Convenience Batch Processing

    /// Convenience method for 512-dimensional euclidean squared batch processing
    public static func batchEuclideanSquared512(
        query: Vector512Optimized,
        candidates: [Vector512Optimized]
    ) -> [Float] {
        // Validate query vector dimensions
        precondition(query.storage.count == 128, "Invalid 512-dim query vector: storage has \(query.storage.count) lanes, expected 128")

        // Validate candidates in DEBUG mode
        #if DEBUG
        for (idx, candidate) in candidates.enumerated() {
            assert(candidate.storage.count == 128, "Invalid candidate[\(idx)] dimensions: storage has \(candidate.storage.count) lanes, expected 128")
        }
        #endif

        let soa = SoA512.build(from: candidates)
        var results = [Float](repeating: 0.0, count: candidates.count)

        results.withUnsafeMutableBufferPointer { buffer in
            euclid2_512(query: query, soa: soa, out: buffer)
        }

        return results
    }

    /// Convenience method for 768-dimensional euclidean squared batch processing
    public static func batchEuclideanSquared768(
        query: Vector768Optimized,
        candidates: [Vector768Optimized]
    ) -> [Float] {
        // Validate query vector dimensions
        precondition(query.storage.count == 192, "Invalid 768-dim query vector: storage has \(query.storage.count) lanes, expected 192")

        // Validate candidates in DEBUG mode
        #if DEBUG
        for (idx, candidate) in candidates.enumerated() {
            assert(candidate.storage.count == 192, "Invalid candidate[\(idx)] dimensions: storage has \(candidate.storage.count) lanes, expected 192")
        }
        #endif

        let soa = SoA768.build(from: candidates)
        var results = [Float](repeating: 0.0, count: candidates.count)

        results.withUnsafeMutableBufferPointer { buffer in
            euclid2_768(query: query, soa: soa, out: buffer)
        }

        return results
    }

    /// Convenience method for 1536-dimensional euclidean squared batch processing
    public static func batchEuclideanSquared1536(
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized]
    ) -> [Float] {
        // Validate query vector dimensions
        precondition(query.storage.count == 384, "Invalid 1536-dim query vector: storage has \(query.storage.count) lanes, expected 384")

        // Validate candidates in DEBUG mode
        #if DEBUG
        for (idx, candidate) in candidates.enumerated() {
            assert(candidate.storage.count == 384, "Invalid candidate[\(idx)] dimensions: storage has \(candidate.storage.count) lanes, expected 384")
        }
        #endif

        let soa = SoA1536.build(from: candidates)
        var results = [Float](repeating: 0.0, count: candidates.count)

        results.withUnsafeMutableBufferPointer { buffer in
            euclid2_1536(query: query, soa: soa, out: buffer)
        }

        return results
    }

    // MARK: - Dot Product Kernels

    /// Discarded pointee load at `ptr + offset`; both architecture branches are the same load. It is
    /// not a prefetch instruction and no prefetch intrinsic is emitted; the optimizer may delete it
    /// (whether it survives optimization is not verified in this repository).
    @inline(__always)
    @usableFromInline
    internal static func prefetch<T>(_ ptr: UnsafePointer<T>, offset: Int = 0) {
        #if arch(arm64)
        // Discarded load; no prefetch instruction is emitted.
        _ = ptr.advanced(by: offset).pointee
        #else
        // Same discarded load as the arm64 branch; not a prefetch instruction.
        _ = ptr.advanced(by: offset).pointee
        #endif
    }

    /// Verify memory alignment for SIMD operations
    @inline(__always)
    @usableFromInline
    internal static func verifyAlignment<T>(_ ptr: UnsafePointer<T>) {
        #if DEBUG
        let address = UInt(bitPattern: ptr)
        assert(address & 0xF == 0, "Pointer not 16-byte aligned for SIMD4 operations")
        #endif
    }

    // MARK: - Generic Dot Product Implementation (2-Way Blocking)

    /// Generic dot product kernel using SoA layout with 2-way blocking
    ///
    /// Structure:
    /// - Reads each lane's candidates from one contiguous SoA column
    /// - Two independent accumulators per block
    /// - Discarded loads of the next block (`prefetch`; not a prefetch instruction)
    /// - Loads each query lane once per block
    @inlinable
    @inline(__always)
    internal static func dot_blocked_2way<Vector: SoACompatible>(
        query: Vector,
        soa: SoA<Vector>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        let N = soa.count
        guard N > 0 else { return }

        #if DEBUG
        assert(out.count >= N, "Output buffer too small: \(out.count) < \(N)")
        #endif

        let lanes = soa.lanes
        let queryStorage = query.storage

        // More explicit boundary calculation
        let blockedN = (N / 2) * 2

        // Main loop with 2-way blocking
        for j in stride(from: 0, to: blockedN, by: 2) {
            var acc0 = SIMD4<Float>.zero
            var acc1 = SIMD4<Float>.zero

            // Discarded loads of the next block's first lanes (`prefetch` helper; not a prefetch instruction)
            if j + 2 < blockedN {
                for i in 0..<min(4, lanes) {
                    let nextPtr = soa.lanePointer(i)
                    prefetch(nextPtr, offset: j + 2)
                }
            }

            for i in 0..<lanes {
                let q_i = queryStorage[i]
                let lanePtr = soa.lanePointer(i)

                #if DEBUG
                if i == 0 { verifyAlignment(lanePtr) }
                #endif

                let c0 = lanePtr[j]
                let c1 = lanePtr[j + 1]

                acc0.addProduct(q_i, c0)
                acc1.addProduct(q_i, c1)
            }

            out[j] = acc0.sum()
            out[j + 1] = acc1.sum()
        }

        // Tail handling for odd N
        if blockedN < N {
            var acc = SIMD4<Float>.zero
            for i in 0..<lanes {
                let q_i = queryStorage[i]
                let c = soa.lanePointer(i)[blockedN]
                acc.addProduct(q_i, c)
            }
            out[blockedN] = acc.sum()
        }
    }

    // MARK: - Advanced 4-Way Blocking Implementation

    /// 4-way blocked dot product (four independent accumulators)
    @inlinable
    @inline(__always)
    internal static func dot_blocked_4way<Vector: SoACompatible>(
        query: Vector,
        soa: SoA<Vector>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        let N = soa.count
        guard N > 0 else { return }

        #if DEBUG
        assert(out.count >= N, "Output buffer too small")
        #endif

        let lanes = soa.lanes
        let queryStorage = query.storage
        let blockedN = (N / 4) * 4

        // Process 4 candidates simultaneously
        for j in stride(from: 0, to: blockedN, by: 4) {
            var acc0 = SIMD4<Float>.zero
            var acc1 = SIMD4<Float>.zero
            var acc2 = SIMD4<Float>.zero
            var acc3 = SIMD4<Float>.zero

            // Discarded loads of the next block's first lanes (`prefetch` helper; not a prefetch instruction)
            if j + 4 < blockedN {
                for i in stride(from: 0, to: min(8, lanes), by: 2) {
                    prefetch(soa.lanePointer(i), offset: j + 4)
                }
            }

            // Process lanes with 4 independent accumulators
            for i in 0..<lanes {
                let q_i = queryStorage[i]
                let lanePtr = soa.lanePointer(i)

                let c0 = lanePtr[j]
                let c1 = lanePtr[j + 1]
                let c2 = lanePtr[j + 2]
                let c3 = lanePtr[j + 3]

                acc0.addProduct(q_i, c0)
                acc1.addProduct(q_i, c1)
                acc2.addProduct(q_i, c2)
                acc3.addProduct(q_i, c3)
            }

            out[j] = acc0.sum()
            out[j + 1] = acc1.sum()
            out[j + 2] = acc2.sum()
            out[j + 3] = acc3.sum()
        }

        // Handle the remaining 1-3 candidates one at a time
        for j in stride(from: blockedN, to: N, by: 1) {
            var acc = SIMD4<Float>.zero
            for i in 0..<lanes {
                let q_i = queryStorage[i]
                let c = soa.lanePointer(i)[j]
                acc.addProduct(q_i, c)
            }
            out[j] = acc.sum()
        }
    }

    // MARK: - Candidate-Tiled Dot Product (Issue 1.2)

    /// Lane-outer / candidate-inner dot product for very large batches.
    ///
    /// Same traversal as `euclid2_tiled`: one sequential candidate stream per lane
    /// instead of `L` interleaved strided streams. Per-candidate SIMD4 partials
    /// accumulate over lanes in the same order as `dot_blocked_4way` and reduce with a
    /// single `sum()`, so results are **bitwise-identical** to the 4-way path
    /// (`tiledDot512MatchesFourWayBitwise`, at 512 dimensions).
    @inlinable
    internal static func dot_tiled<Vector: SoACompatible>(
        query: Vector,
        soa: SoA<Vector>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        let N = soa.count
        guard N > 0 else { return }

        #if DEBUG
        assert(out.count >= N, "Output buffer too small: \(out.count) < \(N)")
        #endif

        let L = soa.lanes
        let queryStorage = query.storage
        let tileWidth = 256

        var tileStart = 0
        while tileStart < N {
            let tileLen = Swift.min(tileWidth, N - tileStart)

            withUnsafeTemporaryAllocation(of: SIMD4<Float>.self, capacity: tileLen) { accs in
                for t in 0..<tileLen {
                    accs[t] = .zero
                }
                for i in 0..<L {
                    let q_i = queryStorage[i]
                    let lanePtr = soa.lanePointer(i) + tileStart
                    for t in 0..<tileLen {
                        accs[t].addProduct(q_i, lanePtr[t])
                    }
                }
                for t in 0..<tileLen {
                    out[tileStart + t] = accs[t].sum()
                }
            }

            tileStart += tileLen
        }
    }

    // MARK: - Public API: Standard Dot Product

    /// Compute dot products between query and all SoA candidates (512-dim)
    @inlinable
    public static func batchDotProduct512(
        query: Vector512Optimized,
        soa: SoA<Vector512Optimized>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        // Choose blocking strategy based on candidate count
        if shouldTile(count: soa.count, lanes: soa.lanes) {
            dot_tiled(query: query, soa: soa, out: out)
        } else if soa.count >= 4 {
            dot_blocked_4way(query: query, soa: soa, out: out)
        } else {
            dot_blocked_2way(query: query, soa: soa, out: out)
        }
    }

    /// Compute dot products between query and all SoA candidates (768-dim)
    @inlinable
    public static func batchDotProduct768(
        query: Vector768Optimized,
        soa: SoA<Vector768Optimized>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        // Choose blocking strategy based on candidate count
        if shouldTile(count: soa.count, lanes: soa.lanes) {
            dot_tiled(query: query, soa: soa, out: out)
        } else if soa.count >= 4 {
            dot_blocked_4way(query: query, soa: soa, out: out)
        } else {
            dot_blocked_2way(query: query, soa: soa, out: out)
        }
    }

    /// Compute dot products between query and all SoA candidates (1536-dim)
    @inlinable
    public static func batchDotProduct1536(
        query: Vector1536Optimized,
        soa: SoA<Vector1536Optimized>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        // Same dispatch as 512/768: tiled above the threshold, 4-way for N >= 4, 2-way otherwise
        if shouldTile(count: soa.count, lanes: soa.lanes) {
            dot_tiled(query: query, soa: soa, out: out)
        } else if soa.count >= 4 {
            dot_blocked_4way(query: query, soa: soa, out: out)
        } else {
            dot_blocked_2way(query: query, soa: soa, out: out)
        }
    }

    // MARK: - Pre-Normalized Entry Points

    /// Dot product for callers whose vectors are pre-normalized (||query|| = ||candidates|| = 1).
    /// It calls `batchDotProduct512` unchanged; the code neither checks nor uses the assumption.
    @inlinable
    public static func batchDotProductNormalized512(
        query: Vector512Optimized,
        soa: SoA<Vector512Optimized>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        // Identical to batchDotProduct512; the name documents the caller's assumption.
        batchDotProduct512(query: query, soa: soa, out: out)
    }

    @inlinable
    public static func batchDotProductNormalized768(
        query: Vector768Optimized,
        soa: SoA<Vector768Optimized>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        batchDotProduct768(query: query, soa: soa, out: out)
    }

    @inlinable
    public static func batchDotProductNormalized1536(
        query: Vector1536Optimized,
        soa: SoA<Vector1536Optimized>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        batchDotProduct1536(query: query, soa: soa, out: out)
    }

    // MARK: - Error-Handling Variants

    /// Error types for batch kernel operations
    public enum BatchKernelError: Error {
        case bufferTooSmall(required: Int, provided: Int)
        case invalidSoAStructure
        case memoryAlignmentError
    }

    /// Checked variant: throws `bufferTooSmall` when `out` is shorter than the candidate count and
    /// `memoryAlignmentError` when lane 0 is not 16-byte aligned; otherwise calls `batchDotProduct512`.
    public static func batchDotProduct512Safe(
        query: Vector512Optimized,
        soa: SoA<Vector512Optimized>,
        out: UnsafeMutableBufferPointer<Float>
    ) throws {
        guard out.count >= soa.count else {
            throw BatchKernelError.bufferTooSmall(
                required: soa.count,
                provided: out.count
            )
        }

        // Verify memory alignment
        if soa.count != 0 {
            let firstPtr = soa.lanePointer(0)
            let address = UInt(bitPattern: firstPtr)
            guard address & 0xF == 0 else {
                throw BatchKernelError.memoryAlignmentError
            }
        }

        batchDotProduct512(query: query, soa: soa, out: out)
    }

    // MARK: - Convenience Methods

    /// Convenience method for 512-dimensional dot product batch processing
    public static func batchDotProduct512(
        query: Vector512Optimized,
        candidates: [Vector512Optimized]
    ) -> [Float] {
        let soa = SoA<Vector512Optimized>.build(from: candidates)
        var results = [Float](repeating: 0.0, count: candidates.count)

        results.withUnsafeMutableBufferPointer { buffer in
            batchDotProduct512(query: query, soa: soa, out: buffer)
        }

        return results
    }

    /// Convenience method for 768-dimensional dot product batch processing
    public static func batchDotProduct768(
        query: Vector768Optimized,
        candidates: [Vector768Optimized]
    ) -> [Float] {
        let soa = SoA<Vector768Optimized>.build(from: candidates)
        var results = [Float](repeating: 0.0, count: candidates.count)

        results.withUnsafeMutableBufferPointer { buffer in
            batchDotProduct768(query: query, soa: soa, out: buffer)
        }

        return results
    }

    /// Convenience method for 1536-dimensional dot product batch processing
    public static func batchDotProduct1536(
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized]
    ) -> [Float] {
        let soa = SoA<Vector1536Optimized>.build(from: candidates)
        var results = [Float](repeating: 0.0, count: candidates.count)

        results.withUnsafeMutableBufferPointer { buffer in
            batchDotProduct1536(query: query, soa: soa, out: buffer)
        }

        return results
    }

    // MARK: - Blocking Helpers

    /// Returns 4 when candidateCount >= 16 and an assumed 32 KiB L1 holds at least 4 candidates of
    /// `dimension` Floats, else 2. A heuristic: the cache size is assumed, not queried.
    @inlinable
    public static func estimateOptimalBlockSize(
        candidateCount: Int,
        dimension: Int
    ) -> Int {
        let l1CacheSize = 32 * 1024  // assumed L1 size (heuristic; not verified)
        let elementSize = 4  // Float size in bytes
        let maxBlockSize = l1CacheSize / (dimension * elementSize)

        // Choose between 2-way and 4-way based on working set size
        if candidateCount >= 16 && maxBlockSize >= 4 {
            return 4
        } else {
            return 2
        }
    }

    // MARK: - Cosine Distance Kernels with Fused Operations

    /// Compute L2 norm (magnitude) of a vector
    @inlinable
    internal static func computeNorm<Vector: SoACompatible>(_ vector: Vector) -> Float {
        var normSqAcc = SIMD4<Float>.zero
        let storage = vector.storage

        for i in 0..<Vector.lanes {
            let v_i = storage[i]
            normSqAcc.addProduct(v_i, v_i)
        }

        let normSq = normSqAcc.sum()
        // A NaN squared norm stays NaN: `max(0, NaN)` would return 0 and turn the query into a zero vector
        return normSq.isNaN ? .nan : sqrt(max(0.0, normSq))
    }

    /// Transforms similarities to cosine distances (1 - s, clamped to [0, 2]), four at a time with SIMD4. A non-finite
    /// similarity gives NaN in both the SIMD4 blocks and the scalar remainder
    /// (T15CosineExceptionalTableTests.everyImplementationFollowsThePolicy, SoA.pre rows).
    @inline(__always)
    @usableFromInline
    internal static func transformToDistance(_ buffer: UnsafeMutableBufferPointer<Float>, count: Int) {
        // Process 4 values at a time using SIMD4 for vectorization
        let blocked = (count / 4) * 4

        for j in stride(from: 0, to: blocked, by: 4) {
            let sim = SIMD4<Float>(buffer[j], buffer[j+1], buffer[j+2], buffer[j+3])
            let dist = SIMD4<Float>(repeating: 1.0) - sim

            // Clamp to the cosine-distance range [0, 2]; rounding can push 1 - s slightly outside it
            let clamped = simd_clamp(dist, SIMD4<Float>.zero, SIMD4<Float>(repeating: 2.0))

            buffer[j] = sim[0].isFinite ? clamped[0] : .nan
            buffer[j+1] = sim[1].isFinite ? clamped[1] : .nan
            buffer[j+2] = sim[2].isFinite ? clamped[2] : .nan
            buffer[j+3] = sim[3].isFinite ? clamped[3] : .nan
        }

        // Handle remainder
        for j in blocked..<count {
            buffer[j] = buffer[j].isFinite ? max(0.0, min(2.0, 1.0 - buffer[j])) : .nan
        }
    }

    // MARK: - Generic Fused Cosine Distance (2-Way Blocking)

    /// Fused cosine distance with 2-way blocking
    /// Computes: 1 - (dot(q,c) / (||q|| * ||c||)) in single pass
    @inlinable
    @inline(__always)
    internal static func cosine_fused_blocked_2way<Vector: SoACompatible>(
        query: Vector,
        queryNorm: Float,
        soa: SoA<Vector>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        let N = soa.count
        guard N > 0 else { return }

        #if DEBUG
        assert(out.count >= N, "Output buffer too small: \(out.count) < \(N)")
        assert(queryNorm.isNaN || queryNorm >= 0, "Invalid query norm: \(queryNorm)")
        #endif

        let lanes = soa.lanes
        let queryStorage = query.storage
        let blockedN = (N / 2) * 2

        // Helper for final distance calculation
        @inline(__always)
        func computeDistance(dotProd: Float, candNormSq: Float) -> Float {
            if queryNorm <= 0 || candNormSq <= 0 {
                return queryNorm.isFinite && candNormSq.isFinite ? 1.0 : .nan  // Maximum distance for zero vector
            }
            let candNorm = sqrt(candNormSq)
            let similarity = dotProd / (queryNorm * candNorm)
            guard similarity.isFinite else { return .nan }
            let distance = 1.0 - similarity
            return max(0.0, min(2.0, distance))  // Clamp to [0, 2]
        }

        // Main loop with 2-way blocking
        for j in stride(from: 0, to: blockedN, by: 2) {
            var dot0 = SIMD4<Float>.zero
            var dot1 = SIMD4<Float>.zero
            var norm0 = SIMD4<Float>.zero
            var norm1 = SIMD4<Float>.zero

            // Discarded loads of the next block's first lanes (`prefetch` helper; not a prefetch instruction)
            if j + 2 < blockedN {
                for i in 0..<min(4, lanes) {
                    prefetch(soa.lanePointer(i), offset: j + 2)
                }
            }

            // Fused computation: dot product and norm in single pass
            for i in 0..<lanes {
                let q_i = queryStorage[i]
                let lanePtr = soa.lanePointer(i)

                #if DEBUG
                if i == 0 { verifyAlignment(lanePtr) }
                #endif

                let c0 = lanePtr[j]
                let c1 = lanePtr[j + 1]

                // Accumulate dot products
                dot0.addProduct(q_i, c0)
                dot1.addProduct(q_i, c1)

                // Accumulate squared norms
                norm0.addProduct(c0, c0)
                norm1.addProduct(c1, c1)
            }

            // Compute final distances
            out[j] = computeDistance(dotProd: dot0.sum(), candNormSq: norm0.sum())
            out[j + 1] = computeDistance(dotProd: dot1.sum(), candNormSq: norm1.sum())
        }

        // Tail handling
        if blockedN < N {
            var dot = SIMD4<Float>.zero
            var norm = SIMD4<Float>.zero

            for i in 0..<lanes {
                let q_i = queryStorage[i]
                let c = soa.lanePointer(i)[blockedN]
                dot.addProduct(q_i, c)
                norm.addProduct(c, c)
            }

            out[blockedN] = computeDistance(dotProd: dot.sum(), candNormSq: norm.sum())
        }
    }

    // MARK: - Generic Fused Cosine Distance (4-Way Blocking)

    /// Fused cosine distance with 4-way blocking (four dot and four norm accumulators)
    @inlinable
    @inline(__always)
    internal static func cosine_fused_blocked_4way<Vector: SoACompatible>(
        query: Vector,
        queryNorm: Float,
        soa: SoA<Vector>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        let N = soa.count
        guard N > 0 else { return }

        #if DEBUG
        assert(out.count >= N, "Output buffer too small")
        #endif

        let lanes = soa.lanes
        let queryStorage = query.storage
        let blockedN = (N / 4) * 4

        @inline(__always)
        func computeDistance(dotProd: Float, candNormSq: Float) -> Float {
            if queryNorm <= 0 || candNormSq <= 0 { return queryNorm.isFinite && candNormSq.isFinite ? 1.0 : .nan }
            let similarity = dotProd / (queryNorm * sqrt(candNormSq))
            guard similarity.isFinite else { return .nan }
            return max(0.0, min(2.0, 1.0 - similarity))
        }

        // Process 4 candidates simultaneously
        for j in stride(from: 0, to: blockedN, by: 4) {
            var dot0 = SIMD4<Float>.zero
            var dot1 = SIMD4<Float>.zero
            var dot2 = SIMD4<Float>.zero
            var dot3 = SIMD4<Float>.zero

            var norm0 = SIMD4<Float>.zero
            var norm1 = SIMD4<Float>.zero
            var norm2 = SIMD4<Float>.zero
            var norm3 = SIMD4<Float>.zero

            // Discarded loads of the next block's first lanes (`prefetch` helper; not a prefetch instruction)
            if j + 4 < blockedN {
                for i in stride(from: 0, to: min(8, lanes), by: 2) {
                    prefetch(soa.lanePointer(i), offset: j + 4)
                }
            }

            for i in 0..<lanes {
                let q_i = queryStorage[i]
                let lanePtr = soa.lanePointer(i)

                let c0 = lanePtr[j]
                let c1 = lanePtr[j + 1]
                let c2 = lanePtr[j + 2]
                let c3 = lanePtr[j + 3]

                // Dot products
                dot0.addProduct(q_i, c0)
                dot1.addProduct(q_i, c1)
                dot2.addProduct(q_i, c2)
                dot3.addProduct(q_i, c3)

                // Squared norms
                norm0.addProduct(c0, c0)
                norm1.addProduct(c1, c1)
                norm2.addProduct(c2, c2)
                norm3.addProduct(c3, c3)
            }

            out[j] = computeDistance(dotProd: dot0.sum(), candNormSq: norm0.sum())
            out[j + 1] = computeDistance(dotProd: dot1.sum(), candNormSq: norm1.sum())
            out[j + 2] = computeDistance(dotProd: dot2.sum(), candNormSq: norm2.sum())
            out[j + 3] = computeDistance(dotProd: dot3.sum(), candNormSq: norm3.sum())
        }

        // Handle remainder with simple loop
        for j in blockedN..<N {
            var dot = SIMD4<Float>.zero
            var norm = SIMD4<Float>.zero

            for i in 0..<lanes {
                let q_i = queryStorage[i]
                let c = soa.lanePointer(i)[j]
                dot.addProduct(q_i, c)
                norm.addProduct(c, c)
            }

            out[j] = computeDistance(dotProd: dot.sum(), candNormSq: norm.sum())
        }
    }

    // MARK: - Candidate-Tiled Fused Cosine (Issue 1.2)

    /// Lane-outer / candidate-inner fused cosine distance for very large batches.
    ///
    /// Accumulates per-candidate dot and squared-norm SIMD4 partials in two small tiles
    /// from `withUnsafeTemporaryAllocation` (L1 residency assumed, not verified) while
    /// streaming each lane's candidate column sequentially. Accumulation order over lanes
    /// and the final `computeDistance` match `cosine_fused_blocked_4way`, so results are
    /// **bitwise-identical** (`tiledCosineFused768MatchesFourWayBitwise`, at 768 dimensions).
    @inlinable
    internal static func cosine_fused_tiled<Vector: SoACompatible>(
        query: Vector,
        queryNorm: Float,
        soa: SoA<Vector>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        let N = soa.count
        guard N > 0 else { return }

        #if DEBUG
        assert(out.count >= N, "Output buffer too small: \(out.count) < \(N)")
        #endif

        let L = soa.lanes
        let queryStorage = query.storage
        let tileWidth = 256

        @inline(__always)
        func computeDistance(dotProd: Float, candNormSq: Float) -> Float {
            if queryNorm <= 0 || candNormSq <= 0 { return queryNorm.isFinite && candNormSq.isFinite ? 1.0 : .nan }
            let similarity = dotProd / (queryNorm * sqrt(candNormSq))
            guard similarity.isFinite else { return .nan }
            return max(0.0, min(2.0, 1.0 - similarity))
        }

        var tileStart = 0
        while tileStart < N {
            let tileLen = Swift.min(tileWidth, N - tileStart)

            withUnsafeTemporaryAllocation(of: SIMD4<Float>.self, capacity: tileLen) { dotAcc in
                withUnsafeTemporaryAllocation(of: SIMD4<Float>.self, capacity: tileLen) { normAcc in
                    for t in 0..<tileLen {
                        dotAcc[t] = .zero
                        normAcc[t] = .zero
                    }
                    for i in 0..<L {
                        let q_i = queryStorage[i]
                        let lanePtr = soa.lanePointer(i) + tileStart
                        for t in 0..<tileLen {
                            let c = lanePtr[t]
                            dotAcc[t].addProduct(q_i, c)
                            normAcc[t].addProduct(c, c)
                        }
                    }
                    for t in 0..<tileLen {
                        out[tileStart + t] = computeDistance(
                            dotProd: dotAcc[t].sum(),
                            candNormSq: normAcc[t].sum()
                        )
                    }
                }
            }

            tileStart += tileLen
        }
    }

    // MARK: - Pre-Normalized Cosine

    /// Pre-normalized cosine distance (assumes ||query|| = ||candidates|| = 1)
    /// Reduces to: distance = 1 - dot(q, c)
    @inlinable
    @inline(__always)
    internal static func cosine_prenorm_blocked<Vector: SoACompatible>(
        query: Vector,
        soa: SoA<Vector>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        // Step 1: dot products through the batchDotProduct dispatch (tiled / 4-way / 2-way)
        if shouldTile(count: soa.count, lanes: soa.lanes) {
            dot_tiled(query: query, soa: soa, out: out)
        } else if soa.count >= 4 {
            dot_blocked_4way(query: query, soa: soa, out: out)
        } else {
            dot_blocked_2way(query: query, soa: soa, out: out)
        }

        // Step 2: Transform similarities to distances with SIMD
        transformToDistance(out, count: soa.count)
    }

    // MARK: - Public API: General Cosine Distance

    /// Batch cosine distance for 512-dimensional vectors
    /// Writes the logical prefix `0..<soa.count`; oversized suffixes are preserved.
    /// Verified by MatrixMetricContractTests.soaLogicalOutputMatrix.
    /// - Precondition: `out.count >= soa.count`.
    @inlinable
    public static func batchCosineDistance512(
        query: Vector512Optimized,
        soa: SoA<Vector512Optimized>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        let queryNorm = computeNorm(query)
        if shouldTile(count: soa.count, lanes: soa.lanes) {
            cosine_fused_tiled(query: query, queryNorm: queryNorm, soa: soa, out: out)
        } else if soa.count >= 4 {
            cosine_fused_blocked_4way(query: query, queryNorm: queryNorm, soa: soa, out: out)
        } else {
            cosine_fused_blocked_2way(query: query, queryNorm: queryNorm, soa: soa, out: out)
        }
    }

    /// Batch cosine distance for 768-dimensional vectors
    /// Writes the logical prefix `0..<soa.count`; oversized suffixes are preserved.
    /// Verified by MatrixMetricContractTests.soaLogicalOutputMatrix.
    /// - Precondition: `out.count >= soa.count`.
    @inlinable
    public static func batchCosineDistance768(
        query: Vector768Optimized,
        soa: SoA<Vector768Optimized>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        let queryNorm = computeNorm(query)
        if shouldTile(count: soa.count, lanes: soa.lanes) {
            cosine_fused_tiled(query: query, queryNorm: queryNorm, soa: soa, out: out)
        } else if soa.count >= 4 {
            cosine_fused_blocked_4way(query: query, queryNorm: queryNorm, soa: soa, out: out)
        } else {
            cosine_fused_blocked_2way(query: query, queryNorm: queryNorm, soa: soa, out: out)
        }
    }

    /// Batch cosine distance for 1536-dimensional vectors
    /// Writes the logical prefix `0..<soa.count`; oversized suffixes are preserved.
    /// Verified by MatrixMetricContractTests.soaLogicalOutputMatrix.
    /// - Precondition: `out.count >= soa.count`.
    @inlinable
    public static func batchCosineDistance1536(
        query: Vector1536Optimized,
        soa: SoA<Vector1536Optimized>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        let queryNorm = computeNorm(query)
        if shouldTile(count: soa.count, lanes: soa.lanes) {
            cosine_fused_tiled(query: query, queryNorm: queryNorm, soa: soa, out: out)
        } else if soa.count >= 4 {
            cosine_fused_blocked_4way(query: query, queryNorm: queryNorm, soa: soa, out: out)
        } else {
            cosine_fused_blocked_2way(query: query, queryNorm: queryNorm, soa: soa, out: out)
        }
    }

    // MARK: - Public API: Pre-Normalized Cosine Distance

    /// Batch cosine distance for 512-dimensional pre-normalized vectors
    /// Assumes ||query|| = ||candidates|| = 1
    @inlinable
    public static func batchCosineDistancePreNorm512(
        query: Vector512Optimized,
        soa: SoA<Vector512Optimized>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        cosine_prenorm_blocked(query: query, soa: soa, out: out)
    }

    /// Batch cosine distance for 768-dimensional pre-normalized vectors
    @inlinable
    public static func batchCosineDistancePreNorm768(
        query: Vector768Optimized,
        soa: SoA<Vector768Optimized>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        cosine_prenorm_blocked(query: query, soa: soa, out: out)
    }

    /// Batch cosine distance for 1536-dimensional pre-normalized vectors
    @inlinable
    public static func batchCosineDistancePreNorm1536(
        query: Vector1536Optimized,
        soa: SoA<Vector1536Optimized>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        cosine_prenorm_blocked(query: query, soa: soa, out: out)
    }

    // MARK: - Convenience Methods

    /// Convenience method for 512-dimensional cosine distance
    public static func batchCosineDistance512(
        query: Vector512Optimized,
        candidates: [Vector512Optimized]
    ) -> [Float] {
        let soa = SoA<Vector512Optimized>.build(from: candidates)
        var results = [Float](repeating: 0.0, count: candidates.count)

        results.withUnsafeMutableBufferPointer { buffer in
            batchCosineDistance512(query: query, soa: soa, out: buffer)
        }

        return results
    }

    /// Convenience method for 768-dimensional cosine distance
    public static func batchCosineDistance768(
        query: Vector768Optimized,
        candidates: [Vector768Optimized]
    ) -> [Float] {
        let soa = SoA<Vector768Optimized>.build(from: candidates)
        var results = [Float](repeating: 0.0, count: candidates.count)

        results.withUnsafeMutableBufferPointer { buffer in
            batchCosineDistance768(query: query, soa: soa, out: buffer)
        }

        return results
    }

    /// Convenience method for 1536-dimensional cosine distance
    public static func batchCosineDistance1536(
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized]
    ) -> [Float] {
        let soa = SoA<Vector1536Optimized>.build(from: candidates)
        var results = [Float](repeating: 0.0, count: candidates.count)

        results.withUnsafeMutableBufferPointer { buffer in
            batchCosineDistance1536(query: query, soa: soa, out: buffer)
        }

        return results
    }

    // MARK: - Manhattan Distance (L1 Norm)

    #if canImport(Accelerate)

    /// Batch Manhattan distance (L1 norm) for 512-dimensional vectors
    ///
    /// Computes: d(query, candidate[i]) = Σ |query[j] - candidate[i][j]|
    ///
    /// **Algorithm**: takes the AoS `[Vector512Optimized]` directly (no SoA transformation and
    /// no vDSP call) and runs `manhattan512Core`, or `manhattan512CoreBlocked` for N > 10,000:
    /// a lane-outer loop over SIMD4 lanes that subtracts, takes a scalar `abs` per element and
    /// accumulates into `distances[j]`. The blocked core walks candidates in blocks of 256
    /// (a heuristic; no cache effect is measured).
    ///
    /// - Parameters:
    ///   - query: Query vector (512-dim)
    ///   - candidates: Array of candidate vectors
    /// - Returns: Array of Manhattan distances
    /// - Complexity: O(N × D) where N = candidates, D = 512
    public static func batchManhattan512(
        query: Vector512Optimized,
        candidates: [Vector512Optimized]
    ) -> [Float] {
        guard !candidates.isEmpty else { return [] }

        // Validate vector dimensions (runs in both DEBUG and RELEASE)
        precondition(query.storage.count == 128, "Invalid 512-dim query vector: storage has \(query.storage.count) lanes, expected 128")

        #if DEBUG
        // Additional debug validation
        for (idx, candidate) in candidates.enumerated() {
            assert(candidate.storage.count == 128, "Invalid candidate[\(idx)] dimensions: storage has \(candidate.storage.count) lanes, expected 128")
        }
        #else
        // Release builds check only the first and last candidate's lane count; DEBUG checks every candidate
        if let first = candidates.first {
            precondition(first.storage.count == 128, "Invalid candidate dimensions detected")
        }
        if candidates.count > 1, let last = candidates.last {
            precondition(last.storage.count == 128, "Invalid candidate dimensions detected")
        }
        #endif

        var distances = [Float](repeating: 0.0, count: candidates.count)

        if candidates.count > 10_000 {
            manhattan512CoreBlocked(
                query: query,
                candidates: candidates,
                distances: &distances
            )
        } else {
            manhattan512Core(
                query: query,
                candidates: candidates,
                distances: &distances
            )
        }

        return distances
    }

    /// Batch Manhattan distance for 768-dimensional vectors
    public static func batchManhattan768(
        query: Vector768Optimized,
        candidates: [Vector768Optimized]
    ) -> [Float] {
        guard !candidates.isEmpty else { return [] }

        // Validate vector dimensions (runs in both DEBUG and RELEASE)
        precondition(query.storage.count == 192, "Invalid 768-dim query vector: storage has \(query.storage.count) lanes, expected 192")

        #if DEBUG
        // Additional debug validation
        for (idx, candidate) in candidates.enumerated() {
            assert(candidate.storage.count == 192, "Invalid candidate[\(idx)] dimensions: storage has \(candidate.storage.count) lanes, expected 192")
        }
        #else
        // Release builds check only the first and last candidate's lane count; DEBUG checks every candidate
        if let first = candidates.first {
            precondition(first.storage.count == 192, "Invalid candidate dimensions detected")
        }
        if candidates.count > 1, let last = candidates.last {
            precondition(last.storage.count == 192, "Invalid candidate dimensions detected")
        }
        #endif

        var distances = [Float](repeating: 0.0, count: candidates.count)

        if candidates.count > 10_000 {
            manhattan768CoreBlocked(
                query: query,
                candidates: candidates,
                distances: &distances
            )
        } else {
            manhattan768Core(
                query: query,
                candidates: candidates,
                distances: &distances
            )
        }

        return distances
    }

    /// Batch Manhattan distance for 1536-dimensional vectors
    public static func batchManhattan1536(
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized]
    ) -> [Float] {
        guard !candidates.isEmpty else { return [] }

        // Validate vector dimensions (runs in both DEBUG and RELEASE)
        precondition(query.storage.count == 384, "Invalid 1536-dim query vector: storage has \(query.storage.count) lanes, expected 384")

        #if DEBUG
        // Additional debug validation
        for (idx, candidate) in candidates.enumerated() {
            assert(candidate.storage.count == 384, "Invalid candidate[\(idx)] dimensions: storage has \(candidate.storage.count) lanes, expected 384")
        }
        #else
        // Release builds check only the first and last candidate's lane count; DEBUG checks every candidate
        if let first = candidates.first {
            precondition(first.storage.count == 384, "Invalid candidate dimensions detected")
        }
        if candidates.count > 1, let last = candidates.last {
            precondition(last.storage.count == 384, "Invalid candidate dimensions detected")
        }
        #endif

        var distances = [Float](repeating: 0.0, count: candidates.count)

        if candidates.count > 10_000 {
            manhattan1536CoreBlocked(
                query: query,
                candidates: candidates,
                distances: &distances
            )
        } else {
            manhattan1536Core(
                query: query,
                candidates: candidates,
                distances: &distances
            )
        }

        return distances
    }

    // MARK: - Manhattan Distance Core Implementations

    @inline(__always)
    @usableFromInline
    internal static func manhattan512Core(
        query: Vector512Optimized,
        candidates: [Vector512Optimized],
        distances: inout [Float]
    ) {
        let N = candidates.count
        let lanes = 128  // 512 / 4 SIMD blocks

        // Zero-initialize distances
        for i in 0..<N {
            distances[i] = 0.0
        }

        let queryStorage = query.storage

        // SIMD-first processing: Process SIMD4 blocks (128 iterations vs 512 scalar)
        for i in 0..<lanes {
            let q_simd = queryStorage[i]  // Load 4 query values at once

            // Process all candidates for this SIMD block
            for j in 0..<N {
                let c_simd = candidates[j].storage[i]  // Load 4 candidate values (SIMD4 load!)

                // Compute Manhattan distance for these 4 dimensions using SIMD operations
                let diff = q_simd - c_simd  // SIMD subtraction

                // Scalar abs per element, then a horizontal sum (no speed comparison is measured)
                let abs_diff = SIMD4<Float>(
                    abs(diff[0]),
                    abs(diff[1]),
                    abs(diff[2]),
                    abs(diff[3])
                )

                // Accumulate to distance (horizontal reduction)
                distances[j] += abs_diff[0] + abs_diff[1] + abs_diff[2] + abs_diff[3]
            }
        }
    }

    @inline(__always)
    @usableFromInline
    internal static func manhattan512CoreBlocked(
        query: Vector512Optimized,
        candidates: [Vector512Optimized],
        distances: inout [Float]
    ) {
        let N = candidates.count
        let lanes = 128
        let blockSize = 256

        // Zero-initialize all distances
        for i in 0..<N {
            distances[i] = 0.0
        }

        let queryStorage = query.storage

        // Process candidates in blocks of 256 (heuristic; no cache effect is measured)
        for blockStart in stride(from: 0, to: N, by: blockSize) {
            let blockEnd = min(blockStart + blockSize, N)

            // SIMD-first processing for this block
            for i in 0..<lanes {
                let q_simd = queryStorage[i]

                // Process all candidates in this block for this SIMD block
                for j in blockStart..<blockEnd {
                    let c_simd = candidates[j].storage[i]
                    let diff = q_simd - c_simd

                    let abs_diff = SIMD4<Float>(
                        abs(diff[0]),
                        abs(diff[1]),
                        abs(diff[2]),
                        abs(diff[3])
                    )

                    distances[j] += abs_diff[0] + abs_diff[1] + abs_diff[2] + abs_diff[3]
                }
            }
        }
    }

    @inline(__always)
    @usableFromInline
    internal static func manhattan768Core(
        query: Vector768Optimized,
        candidates: [Vector768Optimized],
        distances: inout [Float]
    ) {
        let N = candidates.count
        let lanes = 192  // 768 / 4 SIMD blocks

        // Zero-initialize distances
        for i in 0..<N {
            distances[i] = 0.0
        }

        let queryStorage = query.storage

        // SIMD-first processing
        for i in 0..<lanes {
            let q_simd = queryStorage[i]

            for j in 0..<N {
                let c_simd = candidates[j].storage[i]
                let diff = q_simd - c_simd

                let abs_diff = SIMD4<Float>(
                    abs(diff[0]),
                    abs(diff[1]),
                    abs(diff[2]),
                    abs(diff[3])
                )

                distances[j] += abs_diff[0] + abs_diff[1] + abs_diff[2] + abs_diff[3]
            }
        }
    }

    @inline(__always)
    @usableFromInline
    internal static func manhattan768CoreBlocked(
        query: Vector768Optimized,
        candidates: [Vector768Optimized],
        distances: inout [Float]
    ) {
        let N = candidates.count
        let lanes = 192
        let blockSize = 256

        // Zero-initialize all distances
        for i in 0..<N {
            distances[i] = 0.0
        }

        let queryStorage = query.storage

        // Process candidates in blocks
        for blockStart in stride(from: 0, to: N, by: blockSize) {
            let blockEnd = min(blockStart + blockSize, N)

            for i in 0..<lanes {
                let q_simd = queryStorage[i]

                for j in blockStart..<blockEnd {
                    let c_simd = candidates[j].storage[i]
                    let diff = q_simd - c_simd

                    let abs_diff = SIMD4<Float>(
                        abs(diff[0]),
                        abs(diff[1]),
                        abs(diff[2]),
                        abs(diff[3])
                    )

                    distances[j] += abs_diff[0] + abs_diff[1] + abs_diff[2] + abs_diff[3]
                }
            }
        }
    }

    @inline(__always)
    @usableFromInline
    internal static func manhattan1536Core(
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized],
        distances: inout [Float]
    ) {
        let N = candidates.count
        let lanes = 384  // 1536 / 4 SIMD blocks

        // Zero-initialize distances
        for i in 0..<N {
            distances[i] = 0.0
        }

        let queryStorage = query.storage

        // SIMD-first processing
        for i in 0..<lanes {
            let q_simd = queryStorage[i]

            for j in 0..<N {
                let c_simd = candidates[j].storage[i]
                let diff = q_simd - c_simd

                let abs_diff = SIMD4<Float>(
                    abs(diff[0]),
                    abs(diff[1]),
                    abs(diff[2]),
                    abs(diff[3])
                )

                distances[j] += abs_diff[0] + abs_diff[1] + abs_diff[2] + abs_diff[3]
            }
        }
    }

    @inline(__always)
    @usableFromInline
    internal static func manhattan1536CoreBlocked(
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized],
        distances: inout [Float]
    ) {
        let N = candidates.count
        let lanes = 384
        let blockSize = 256

        // Zero-initialize all distances
        for i in 0..<N {
            distances[i] = 0.0
        }

        let queryStorage = query.storage

        // Process candidates in blocks
        for blockStart in stride(from: 0, to: N, by: blockSize) {
            let blockEnd = min(blockStart + blockSize, N)

            for i in 0..<lanes {
                let q_simd = queryStorage[i]

                for j in blockStart..<blockEnd {
                    let c_simd = candidates[j].storage[i]
                    let diff = q_simd - c_simd

                    let abs_diff = SIMD4<Float>(
                        abs(diff[0]),
                        abs(diff[1]),
                        abs(diff[2]),
                        abs(diff[3])
                    )

                    distances[j] += abs_diff[0] + abs_diff[1] + abs_diff[2] + abs_diff[3]
                }
            }
        }
    }

    #endif // canImport(Accelerate)

    // MARK: - Error Handling Variants

    /// Checked variant: throws `bufferTooSmall` when `out` is shorter than the candidate count and
    /// `memoryAlignmentError` when lane 0 is not 16-byte aligned; otherwise calls `batchCosineDistance512`.
    public static func batchCosineDistance512Safe(
        query: Vector512Optimized,
        soa: SoA<Vector512Optimized>,
        out: UnsafeMutableBufferPointer<Float>
    ) throws {
        guard out.count >= soa.count else {
            throw BatchKernelError.bufferTooSmall(
                required: soa.count,
                provided: out.count
            )
        }

        // Verify memory alignment
        if soa.count != 0 {
            let firstPtr = soa.lanePointer(0)
            let address = UInt(bitPattern: firstPtr)
            guard address & 0xF == 0 else {
                throw BatchKernelError.memoryAlignmentError
            }
        }

        batchCosineDistance512(query: query, soa: soa, out: out)
    }
}

// MARK: - Integration with Existing Kernel Infrastructure

extension BatchKernels_SoA {

    /// Heuristic recommendation: true when candidateCount >= 100 and dimension >= 512.
    ///
    /// The thresholds are not derived from a measurement in this repository; whether SoA is
    /// faster than AoS for a given workload, including the transposition cost, is not verified.
    public static func shouldUseSoA(candidateCount: Int, dimension: Int) -> Bool {
        // Heuristic thresholds (unverified): N >= 100 and dimension >= 512
        return candidateCount >= 100 && dimension >= 512
    }

    /// Integration point for AutoTuning system
    ///
    /// Provides kernel kind identifiers for separate SoA calibration
    public static var euclideanKernelKind: String {
        return "euclid2_soa"
    }

    public static var dotProductKernelKind: String {
        return "dot_soa"
    }
}
