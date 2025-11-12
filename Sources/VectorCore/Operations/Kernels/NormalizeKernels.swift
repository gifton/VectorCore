//
//  NormalizeKernels.swift
//  VectorCore
//
//  CPU-only, SIMD-optimized normalization helpers for optimized vectors.
//  Provides in-place and out-of-place normalization with zero guards.
//

import Foundation
import simd

@usableFromInline
internal enum NormalizeKernels {

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
}
