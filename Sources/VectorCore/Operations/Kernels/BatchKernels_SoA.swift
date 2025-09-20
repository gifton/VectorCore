//
//  BatchKernels_SoA.swift
//  VectorCore
//
//  Structure-of-Arrays optimized batch kernels with 2-way register blocking
//  Provides superior cache locality for large candidate set processing
//

import Foundation
import simd

/// SoA-optimized batch kernels for high-performance vector similarity computation
///
/// These kernels leverage Structure-of-Arrays memory layout to achieve better
/// cache locality when processing multiple candidates against a single query.
/// The key optimization is that all candidates' data for a given lane are
/// stored contiguously, minimizing cache misses during batch processing.
public enum BatchKernels_SoA {

    // MARK: - Generic Implementation

    /// Generic Euclidean squared distance kernel using SoA layout with 2-way blocking
    ///
    /// Algorithm:
    /// 1. Process candidates in blocks of 2 for register efficiency
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
                // Load query lane once (key optimization: reuse across candidates)
                let q_i = queryStorage[i]

                // Get pointer to contiguous candidate data for this lane
                let candidateLanePtr = soa.lanePointer(i)

                // Load both candidates' data (contiguous in memory due to SoA layout)
                let c0 = candidateLanePtr[j]       // Candidate j, lane i
                let c1 = candidateLanePtr[j + 1]   // Candidate j+1, lane i

                // Compute differences and accumulate squared distances
                let diff0 = q_i - c0
                let diff1 = q_i - c1

                // Fused multiply-add for optimal performance
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

    // MARK: - Dimension-Specific Public APIs

    /// Euclidean squared distance for 512-dimensional vectors using SoA layout
    ///
    /// Performance target: 10-20% improvement over AoS for N >= 1000
    @inlinable
    public static func euclid2_512(
        query: Vector512Optimized,
        soa: SoA512,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        euclid2_blocked(query: query, soa: soa, out: out)
    }

    /// Euclidean squared distance for 768-dimensional vectors using SoA layout
    ///
    /// Performance target: 10-20% improvement over AoS for N >= 1000
    @inlinable
    public static func euclid2_768(
        query: Vector768Optimized,
        soa: SoA768,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        euclid2_blocked(query: query, soa: soa, out: out)
    }

    /// Euclidean squared distance for 1536-dimensional vectors using SoA layout
    ///
    /// Performance target: 10-20% improvement over AoS for N >= 1000
    @inlinable
    public static func euclid2_1536(
        query: Vector1536Optimized,
        soa: SoA1536,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        euclid2_blocked(query: query, soa: soa, out: out)
    }

    // MARK: - Convenience Batch Processing

    /// Convenience method for 512-dimensional euclidean squared batch processing
    public static func batchEuclideanSquared512(
        query: Vector512Optimized,
        candidates: [Vector512Optimized]
    ) -> [Float] {
        let soa = SoA512.build(from: candidates)
        var results = Array<Float>(repeating: 0.0, count: candidates.count)

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
        let soa = SoA768.build(from: candidates)
        var results = Array<Float>(repeating: 0.0, count: candidates.count)

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
        let soa = SoA1536.build(from: candidates)
        var results = Array<Float>(repeating: 0.0, count: candidates.count)

        results.withUnsafeMutableBufferPointer { buffer in
            euclid2_1536(query: query, soa: soa, out: buffer)
        }

        return results
    }

    // MARK: - Performance Analysis Utilities

    /// Estimates memory bandwidth savings from SoA layout
    ///
    /// SoA provides better cache locality but doesn't reduce total memory usage.
    /// The key benefit is reduced cache misses during lane processing.
    public static func estimatePerformanceGain(candidateCount: Int, dimension: Int) -> String {
        let cacheLineSize = 64  // bytes
        let simd4Size = 16      // bytes
        let _ = cacheLineSize / simd4Size  // 4 SIMD4 elements per cache line

        // In AoS: accessing same lane across N candidates hits N/4 cache lines
        // In SoA: accessing same lane across N candidates hits N/4 cache lines but sequentially
        let aosRandomAccess = "AoS: potentially random cache line access pattern"
        let soaSequentialAccess = "SoA: sequential cache line access pattern"

        return """
        Performance Analysis for \(candidateCount) candidates, \(dimension)D:
        - \(aosRandomAccess)
        - \(soaSequentialAccess)
        - Expected improvement: 10-20% for large N due to better cache locality
        - Memory usage: identical to AoS (\(candidateCount * dimension * 4) bytes)
        """
    }
}

// MARK: - Integration with Existing Kernel Infrastructure

extension BatchKernels_SoA {

    /// Check if SoA processing is beneficial for given parameters
    ///
    /// SoA overhead is only worthwhile for larger candidate sets where
    /// cache locality benefits outweigh transposition costs.
    public static func shouldUseSoA(candidateCount: Int, dimension: Int) -> Bool {
        // Heuristic: SoA beneficial for N >= 100 and dimension >= 512
        // For smaller sets, AoS overhead is minimal and transposition cost dominates
        return candidateCount >= 100 && dimension >= 512
    }

    /// Integration point for AutoTuning system
    ///
    /// Provides kernel kind identifier for separate SoA calibration
    public static var kernelKind: String {
        return "euclid2_soa"
    }
}