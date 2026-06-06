//
//  NormalizeKernels.swift
//  VectorCore
//
//  CPU-only, SIMD-optimized normalization helpers for optimized vectors.
//  Provides in-place and out-of-place normalization with zero guards.
//

import Foundation
import simd

public enum NormalizeKernels {

    /// Compute magnitude squared using Kahan's two-pass scaling algorithm
    ///
    /// Uses SIMD operations for performance while preventing overflow through scaling.
    /// This implements the stable algorithm: ||v||² = (M × √(Σ((x/M)²)))²
    ///
    /// - Complexity: O(n) with 2 passes over the data
    /// - Note: Approximately 20-30% slower than naive implementation,
    ///         but prevents overflow for large values (> sqrt(Float.max))
    @inline(__always)
    @usableFromInline
    static func magnitudeSquared(storage: ContiguousArray<SIMD4<Float>>, laneCount: Int) -> Float {
        #if DEBUG
        assert(storage.count == laneCount, "Storage count mismatch")
        assert(laneCount % 16 == 0, "Lane count must be multiple of 16.")
        #endif

        // Phase 1: Find maximum absolute value using SIMD max reduction
        var maxVec = SIMD4<Float>(repeating: 0)
        // Track NaNs explicitly to propagate per IEEE expectations
        var foundNaN = false
        for i in 0..<laneCount {
            let v = storage[i]
            if v[0].isNaN || v[1].isNaN || v[2].isNaN || v[3].isNaN {
                foundNaN = true
                // Continue scanning to keep timing consistent
            }
            let absVec = abs(v)
            maxVec = pointwiseMax(maxVec, absVec)
        }

        // Horizontal max across SIMD4 lanes
        let maxAbs = max(max(maxVec[0], maxVec[1]), max(maxVec[2], maxVec[3]))

        // Handle edge cases
        if foundNaN { return Float.nan }
        guard maxAbs > 0 else { return 0 }  // Zero vector
        if maxAbs.isNaN { return Float.nan }
        guard maxAbs.isFinite else { return Float.infinity }  // Infinite components

        // Phase 2: Scale and compute sum of squares with SIMD
        // By scaling all values by 1/maxAbs, we ensure |scaled| ≤ 1
        // This prevents overflow since 1² = 1
        let scale = 1.0 / maxAbs

        // Subnormal guard: if maxAbs is subnormal (~1e-40), 1/maxAbs overflows to +Inf,
        // which would poison every scaled square with Inf/NaN. maxAbs > 0 and isFinite
        // both pass for subnormals, so they do not catch this. When the reciprocal is
        // non-finite, fall back to a direct (unscaled) sum of squares: for a vector whose
        // largest component is subnormal, every square is ~1e-80 and underflows safely
        // toward 0 without any risk of overflow.
        guard scale.isFinite else {
            var direct: Float = 0
            for i in 0..<laneCount {
                let v = storage[i]
                direct += (v * v).sum()
            }
            return direct
        }

        let simdScale = SIMD4<Float>(repeating: scale)

        var acc0 = SIMD4<Float>()
        var acc1 = SIMD4<Float>()
        var acc2 = SIMD4<Float>()
        var acc3 = SIMD4<Float>()

        for i in stride(from: 0, to: laneCount, by: 16) {
            // Block 0: Scale and square in one operation
            let s0 = storage[i+0] * simdScale
            let s1 = storage[i+1] * simdScale
            let s2 = storage[i+2] * simdScale
            let s3 = storage[i+3] * simdScale

            acc0 += s0 * s0
            acc1 += s1 * s1
            acc2 += s2 * s2
            acc3 += s3 * s3

            // Block 1
            let s4 = storage[i+4] * simdScale
            let s5 = storage[i+5] * simdScale
            let s6 = storage[i+6] * simdScale
            let s7 = storage[i+7] * simdScale

            acc0 += s4 * s4
            acc1 += s5 * s5
            acc2 += s6 * s6
            acc3 += s7 * s7

            // Block 2
            let s8 = storage[i+8] * simdScale
            let s9 = storage[i+9] * simdScale
            let s10 = storage[i+10] * simdScale
            let s11 = storage[i+11] * simdScale

            acc0 += s8 * s8
            acc1 += s9 * s9
            acc2 += s10 * s10
            acc3 += s11 * s11

            // Block 3
            let s12 = storage[i+12] * simdScale
            let s13 = storage[i+13] * simdScale
            let s14 = storage[i+14] * simdScale
            let s15 = storage[i+15] * simdScale

            acc0 += s12 * s12
            acc1 += s13 * s13
            acc2 += s14 * s14
            acc3 += s15 * s15
        }

        let sumSquares = ((acc0 + acc1) + (acc2 + acc3)).sum()

        // Return magnitude squared: (maxAbs × sqrt(sumSquares))²
        // Since magnitude = maxAbs × sqrt(sumSquares)
        // Then magnitude² = maxAbs² × sumSquares
        // Note: This can overflow if maxAbs² > Float.max
        // Callers should use magnitude() instead for large values
        return maxAbs * maxAbs * sumSquares
    }

    /// Compute magnitude using Kahan's two-pass scaling algorithm
    ///
    /// This is the numerically stable version that avoids intermediate overflow.
    /// Use this instead of sqrt(magnitudeSquared()) for large values.
    ///
    /// - Returns: Magnitude (L2 norm) without intermediate overflow
    @inline(__always)
    @usableFromInline
    static func magnitude(storage: ContiguousArray<SIMD4<Float>>, laneCount: Int) -> Float {
        #if DEBUG
        assert(storage.count == laneCount, "Storage count mismatch")
        assert(laneCount % 16 == 0, "Lane count must be multiple of 16.")
        #endif

        // Phase 1: Find maximum absolute value using SIMD max reduction
        var maxVec = SIMD4<Float>(repeating: 0)
        var foundNaN = false
        for i in 0..<laneCount {
            let v = storage[i]
            if v[0].isNaN || v[1].isNaN || v[2].isNaN || v[3].isNaN {
                foundNaN = true
            }
            let absVec = abs(v)
            maxVec = pointwiseMax(maxVec, absVec)
        }

        // Horizontal max across SIMD4 lanes
        let maxAbs = max(max(maxVec[0], maxVec[1]), max(maxVec[2], maxVec[3]))

        // Handle edge cases
        if foundNaN { return Float.nan }
        guard maxAbs > 0 else { return 0 }  // Zero vector
        if maxAbs.isNaN { return Float.nan }
        guard maxAbs.isFinite else { return Float.infinity }  // Infinite components

        // Phase 2: Scale and compute sum of squares with SIMD
        let scale = 1.0 / maxAbs

        // Subnormal guard: if maxAbs is subnormal (~1e-40), 1/maxAbs overflows to +Inf,
        // which would poison every scaled square with Inf/NaN. maxAbs > 0 and isFinite
        // both pass for subnormals, so they do not catch this. When the reciprocal is
        // non-finite, fall back to a direct (unscaled) L2 magnitude: for a vector whose
        // largest component is subnormal, every square is ~1e-80 and underflows safely
        // toward 0 without any risk of overflow.
        guard scale.isFinite else {
            var direct: Float = 0
            for i in 0..<laneCount {
                let v = storage[i]
                direct += (v * v).sum()
            }
            return Foundation.sqrt(direct)
        }

        let simdScale = SIMD4<Float>(repeating: scale)

        var acc0 = SIMD4<Float>()
        var acc1 = SIMD4<Float>()
        var acc2 = SIMD4<Float>()
        var acc3 = SIMD4<Float>()

        for i in stride(from: 0, to: laneCount, by: 16) {
            // Block 0
            let s0 = storage[i+0] * simdScale
            let s1 = storage[i+1] * simdScale
            let s2 = storage[i+2] * simdScale
            let s3 = storage[i+3] * simdScale

            acc0 += s0 * s0
            acc1 += s1 * s1
            acc2 += s2 * s2
            acc3 += s3 * s3

            // Block 1
            let s4 = storage[i+4] * simdScale
            let s5 = storage[i+5] * simdScale
            let s6 = storage[i+6] * simdScale
            let s7 = storage[i+7] * simdScale

            acc0 += s4 * s4
            acc1 += s5 * s5
            acc2 += s6 * s6
            acc3 += s7 * s7

            // Block 2
            let s8 = storage[i+8] * simdScale
            let s9 = storage[i+9] * simdScale
            let s10 = storage[i+10] * simdScale
            let s11 = storage[i+11] * simdScale

            acc0 += s8 * s8
            acc1 += s9 * s9
            acc2 += s10 * s10
            acc3 += s11 * s11

            // Block 3
            let s12 = storage[i+12] * simdScale
            let s13 = storage[i+13] * simdScale
            let s14 = storage[i+14] * simdScale
            let s15 = storage[i+15] * simdScale

            acc0 += s12 * s12
            acc1 += s13 * s13
            acc2 += s14 * s14
            acc3 += s15 * s15
        }

        let sumSquares = ((acc0 + acc1) + (acc2 + acc3)).sum()

        // Return magnitude directly: maxAbs × sqrt(sumSquares)
        // This avoids overflow from squaring maxAbs
        return maxAbs * Foundation.sqrt(sumSquares)
    }

    // Scale in place using broadcasted SIMD factor; loop unrolled by 4 lanes
    @inline(__always)
    @usableFromInline
    static func scaleInPlace(storage: inout ContiguousArray<SIMD4<Float>>, laneCount: Int, scale: Float) {
        let simdScale = SIMD4<Float>(repeating: scale)
        storage.withUnsafeMutableBufferPointer { buf in
            var i = 0
            while i + 3 < laneCount {
                buf[i+0] *= simdScale
                buf[i+1] *= simdScale
                buf[i+2] *= simdScale
                buf[i+3] *= simdScale
                i += 4
            }
            while i < laneCount { buf[i] *= simdScale; i += 1 }
        }
    }

    // MARK: - In-place APIs

    @usableFromInline
    static func normalize384_inplace(_ v: inout Vector384Optimized) -> Result<Void, VectorError> {
        let mag = magnitude(storage: v.storage, laneCount: 96)
        if mag <= 0 { return .failure(.invalidOperation("normalize", reason: "Cannot normalize zero vector")) }
        let inv = 1.0 / mag
        scaleInPlace(storage: &v.storage, laneCount: 96, scale: inv)
        return .success(())
    }

    @usableFromInline
    static func normalize512_inplace(_ v: inout Vector512Optimized) -> Result<Void, VectorError> {
        let mag = magnitude(storage: v.storage, laneCount: 128)
        if mag <= 0 { return .failure(.invalidOperation("normalize", reason: "Cannot normalize zero vector")) }
        let inv = 1.0 / mag
        scaleInPlace(storage: &v.storage, laneCount: 128, scale: inv)
        return .success(())
    }

    @usableFromInline
    static func normalize768_inplace(_ v: inout Vector768Optimized) -> Result<Void, VectorError> {
        let mag = magnitude(storage: v.storage, laneCount: 192)
        if mag <= 0 { return .failure(.invalidOperation("normalize", reason: "Cannot normalize zero vector")) }
        let inv = 1.0 / mag
        scaleInPlace(storage: &v.storage, laneCount: 192, scale: inv)
        return .success(())
    }

    @usableFromInline
    static func normalize1536_inplace(_ v: inout Vector1536Optimized) -> Result<Void, VectorError> {
        let mag = magnitude(storage: v.storage, laneCount: 384)
        if mag <= 0 { return .failure(.invalidOperation("normalize", reason: "Cannot normalize zero vector")) }
        let inv = 1.0 / mag
        scaleInPlace(storage: &v.storage, laneCount: 384, scale: inv)
        return .success(())
    }

    // MARK: - Out-of-place APIs

    @usableFromInline
    static func normalized384(_ a: Vector384Optimized) -> Result<Vector384Optimized, VectorError> {
        var copy = a
        switch normalize384_inplace(&copy) {
        case .success: return .success(copy)
        case .failure(let e): return .failure(e)
        }
    }

    @usableFromInline
    static func normalized512(_ a: Vector512Optimized) -> Result<Vector512Optimized, VectorError> {
        var copy = a
        switch normalize512_inplace(&copy) {
        case .success: return .success(copy)
        case .failure(let e): return .failure(e)
        }
    }

    @usableFromInline
    static func normalized768(_ a: Vector768Optimized) -> Result<Vector768Optimized, VectorError> {
        var copy = a
        switch normalize768_inplace(&copy) {
        case .success: return .success(copy)
        case .failure(let e): return .failure(e)
        }
    }

    @usableFromInline
    static func normalized1536(_ a: Vector1536Optimized) -> Result<Vector1536Optimized, VectorError> {
        var copy = a
        switch normalize1536_inplace(&copy) {
        case .success: return .success(copy)
        case .failure(let e): return .failure(e)
        }
    }

    // MARK: - Unchecked APIs (No zero-vector validation)

    /// Normalize vector without zero-check. Caller guarantees magnitude > 0.
    /// - Precondition: Vector magnitude > 0 (debug assertion only)
    /// - Returns: Normalized copy of the vector
    @usableFromInline @inline(__always)
    static func normalizedUnchecked384(_ a: Vector384Optimized) -> Vector384Optimized {
        var copy = a
        let mag = magnitude(storage: copy.storage, laneCount: 96)
        let inv = 1.0 / mag
        // mag == 0 (zero/underflowed) or deep-subnormal (1/mag overflows) → the
        // vector is not normalizable in FP32; return it unchanged rather than
        // scaling every element by Inf/NaN.
        guard inv.isFinite else { return copy }
        scaleInPlace(storage: &copy.storage, laneCount: 96, scale: inv)
        return copy
    }

    /// Normalize vector without zero-check. Caller guarantees magnitude > 0.
    /// - Precondition: Vector magnitude > 0 (debug assertion only)
    /// - Returns: Normalized copy of the vector
    @usableFromInline @inline(__always)
    static func normalizedUnchecked512(_ a: Vector512Optimized) -> Vector512Optimized {
        var copy = a
        let mag = magnitude(storage: copy.storage, laneCount: 128)
        let inv = 1.0 / mag
        // mag == 0 (zero/underflowed) or deep-subnormal (1/mag overflows) → the
        // vector is not normalizable in FP32; return it unchanged rather than
        // scaling every element by Inf/NaN.
        guard inv.isFinite else { return copy }
        scaleInPlace(storage: &copy.storage, laneCount: 128, scale: inv)
        return copy
    }

    /// Normalize vector without zero-check. Caller guarantees magnitude > 0.
    /// - Precondition: Vector magnitude > 0 (debug assertion only)
    /// - Returns: Normalized copy of the vector
    @usableFromInline @inline(__always)
    static func normalizedUnchecked768(_ a: Vector768Optimized) -> Vector768Optimized {
        var copy = a
        let mag = magnitude(storage: copy.storage, laneCount: 192)
        let inv = 1.0 / mag
        // mag == 0 (zero/underflowed) or deep-subnormal (1/mag overflows) → the
        // vector is not normalizable in FP32; return it unchanged rather than
        // scaling every element by Inf/NaN.
        guard inv.isFinite else { return copy }
        scaleInPlace(storage: &copy.storage, laneCount: 192, scale: inv)
        return copy
    }

    /// Normalize vector without zero-check. Caller guarantees magnitude > 0.
    /// - Precondition: Vector magnitude > 0 (debug assertion only)
    /// - Returns: Normalized copy of the vector
    @usableFromInline @inline(__always)
    static func normalizedUnchecked1536(_ a: Vector1536Optimized) -> Vector1536Optimized {
        var copy = a
        let mag = magnitude(storage: copy.storage, laneCount: 384)
        let inv = 1.0 / mag
        // mag == 0 (zero/underflowed) or deep-subnormal (1/mag overflows) → the
        // vector is not normalizable in FP32; return it unchanged rather than
        // scaling every element by Inf/NaN.
        guard inv.isFinite else { return copy }
        scaleInPlace(storage: &copy.storage, laneCount: 384, scale: inv)
        return copy
    }

    // MARK: - Zero-Copy Pointer API

    /// Normalize a vector in place via raw pointer, for arbitrary dimension.
    ///
    /// Uses Kahan's two-pass scaling algorithm for numerical stability,
    /// operating directly on the pointer without ContiguousArray abstraction.
    /// SIMD4 loads via UnsafeMutableRawPointer avoid memory rebinding issues.
    ///
    /// - Precondition: The vector's magnitude must be > 0 (debug assertion only)
    /// - Parameters:
    ///   - buffer: Mutable pointer to float data (at least `dimension` elements, 16-byte aligned preferred)
    ///   - dimension: Number of float elements in the vector
    /// - Complexity: O(n) with 2 passes over the data
    @inlinable
    public static func normalizeUnchecked(
        _ buffer: UnsafeMutablePointer<Float>,
        dimension: Int
    ) {
        guard dimension > 0 else { return }

        let simdCount = dimension / 4
        let tailStart = simdCount * 4
        let raw = UnsafeMutableRawPointer(buffer)

        // --- Pass 1: Find maximum absolute value (overflow prevention) ---
        var maxVec = SIMD4<Float>(repeating: 0)
        for i in 0..<simdCount {
            let v: SIMD4<Float> = raw.load(fromByteOffset: i * 16, as: SIMD4<Float>.self)
            maxVec = pointwiseMax(maxVec, abs(v))
        }
        var maxAbs = max(max(maxVec[0], maxVec[1]), max(maxVec[2], maxVec[3]))

        for i in tailStart..<dimension {
            let absVal = abs(buffer[i])
            if absVal > maxAbs { maxAbs = absVal }
        }

        assert(maxAbs > 0, "normalizeUnchecked called on zero vector")
        guard maxAbs > 0 else { return }

        // --- Pass 2: Compute scaled sum of squares ---
        // Clamp to leastNormalMagnitude so 1/maxAbs cannot overflow to +Inf for a
        // subnormal-dominated vector (which would poison the scaled squares with
        // Inf/NaN). |scaled| ≤ 1 afterwards, keeping the accumulation finite.
        let scale = 1.0 / Swift.max(maxAbs, Float.leastNormalMagnitude)
        let simdScale = SIMD4<Float>(repeating: scale)

        var acc0 = SIMD4<Float>.zero
        var acc1 = SIMD4<Float>.zero
        var acc2 = SIMD4<Float>.zero
        var acc3 = SIMD4<Float>.zero

        let unrolledCount = (simdCount / 4) * 4
        for i in stride(from: 0, to: unrolledCount, by: 4) {
            let s0: SIMD4<Float> = raw.load(fromByteOffset: (i) * 16, as: SIMD4<Float>.self) * simdScale
            let s1: SIMD4<Float> = raw.load(fromByteOffset: (i + 1) * 16, as: SIMD4<Float>.self) * simdScale
            let s2: SIMD4<Float> = raw.load(fromByteOffset: (i + 2) * 16, as: SIMD4<Float>.self) * simdScale
            let s3: SIMD4<Float> = raw.load(fromByteOffset: (i + 3) * 16, as: SIMD4<Float>.self) * simdScale
            acc0.addProduct(s0, s0)
            acc1.addProduct(s1, s1)
            acc2.addProduct(s2, s2)
            acc3.addProduct(s3, s3)
        }
        for i in unrolledCount..<simdCount {
            let s: SIMD4<Float> = raw.load(fromByteOffset: i * 16, as: SIMD4<Float>.self) * simdScale
            acc0.addProduct(s, s)
        }

        var sumSquares = ((acc0 + acc1) + (acc2 + acc3)).sum()
        for i in tailStart..<dimension {
            let s = buffer[i] * scale
            sumSquares += s * s
        }

        let mag = maxAbs * sqrt(sumSquares)
        assert(mag > 0, "normalizeUnchecked: computed magnitude is zero")
        guard mag > 0 else { return }

        // --- Scale in place ---
        let invMag = 1.0 / mag
        // If the true magnitude is too small to invert in FP32 (deep subnormal),
        // 1/mag overflows to +Inf; leave the buffer unmodified rather than
        // poisoning every element with Inf/NaN.
        guard invMag.isFinite else { return }
        let simdInvMag = SIMD4<Float>(repeating: invMag)

        let scaleUnrolled = (simdCount / 4) * 4
        for i in stride(from: 0, to: scaleUnrolled, by: 4) {
            var v0: SIMD4<Float> = raw.load(fromByteOffset: (i) * 16, as: SIMD4<Float>.self)
            var v1: SIMD4<Float> = raw.load(fromByteOffset: (i + 1) * 16, as: SIMD4<Float>.self)
            var v2: SIMD4<Float> = raw.load(fromByteOffset: (i + 2) * 16, as: SIMD4<Float>.self)
            var v3: SIMD4<Float> = raw.load(fromByteOffset: (i + 3) * 16, as: SIMD4<Float>.self)
            v0 *= simdInvMag; v1 *= simdInvMag; v2 *= simdInvMag; v3 *= simdInvMag
            raw.storeBytes(of: v0, toByteOffset: (i) * 16, as: SIMD4<Float>.self)
            raw.storeBytes(of: v1, toByteOffset: (i + 1) * 16, as: SIMD4<Float>.self)
            raw.storeBytes(of: v2, toByteOffset: (i + 2) * 16, as: SIMD4<Float>.self)
            raw.storeBytes(of: v3, toByteOffset: (i + 3) * 16, as: SIMD4<Float>.self)
        }
        for i in scaleUnrolled..<simdCount {
            var v: SIMD4<Float> = raw.load(fromByteOffset: i * 16, as: SIMD4<Float>.self)
            v *= simdInvMag
            raw.storeBytes(of: v, toByteOffset: i * 16, as: SIMD4<Float>.self)
        }
        for i in tailStart..<dimension {
            buffer[i] *= invMag
        }
    }
}
