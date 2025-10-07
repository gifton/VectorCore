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

    // Compute magnitude squared using 4 accumulators over SIMD4 lanes
    @inline(__always)
    @usableFromInline
    static func magnitudeSquared(storage: ContiguousArray<SIMD4<Float>>, laneCount: Int) -> Float {
        #if DEBUG
        assert(storage.count == laneCount, "Storage count mismatch")
        assert(laneCount % 16 == 0, "Lane count must be multiple of 16.")
        #endif
        var acc0 = SIMD4<Float>()
        var acc1 = SIMD4<Float>()
        var acc2 = SIMD4<Float>()
        var acc3 = SIMD4<Float>()
        for i in stride(from: 0, to: laneCount, by: 16) {
            // Block 0
            acc0 += storage[i+0] * storage[i+0]
            acc1 += storage[i+1] * storage[i+1]
            acc2 += storage[i+2] * storage[i+2]
            acc3 += storage[i+3] * storage[i+3]
            // Block 1
            acc0 += storage[i+4] * storage[i+4]
            acc1 += storage[i+5] * storage[i+5]
            acc2 += storage[i+6] * storage[i+6]
            acc3 += storage[i+7] * storage[i+7]
            // Block 2
            acc0 += storage[i+8] * storage[i+8]
            acc1 += storage[i+9] * storage[i+9]
            acc2 += storage[i+10] * storage[i+10]
            acc3 += storage[i+11] * storage[i+11]
            // Block 3
            acc0 += storage[i+12] * storage[i+12]
            acc1 += storage[i+13] * storage[i+13]
            acc2 += storage[i+14] * storage[i+14]
            acc3 += storage[i+15] * storage[i+15]
        }
        return ((acc0 + acc1) + (acc2 + acc3)).sum()
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
        let ms = magnitudeSquared(storage: v.storage, laneCount: 128)
        if ms <= 0 { return .failure(.invalidOperation("normalize", reason: "Cannot normalize zero vector")) }
        let inv = 1.0 / sqrt(ms)
        scaleInPlace(storage: &v.storage, laneCount: 128, scale: inv)
        return .success(())
    }

    @usableFromInline
    static func normalize768_inplace(_ v: inout Vector768Optimized) -> Result<Void, VectorError> {
        let ms = magnitudeSquared(storage: v.storage, laneCount: 192)
        if ms <= 0 { return .failure(.invalidOperation("normalize", reason: "Cannot normalize zero vector")) }
        let inv = 1.0 / sqrt(ms)
        scaleInPlace(storage: &v.storage, laneCount: 192, scale: inv)
        return .success(())
    }

    @usableFromInline
    static func normalize1536_inplace(_ v: inout Vector1536Optimized) -> Result<Void, VectorError> {
        let ms = magnitudeSquared(storage: v.storage, laneCount: 384)
        if ms <= 0 { return .failure(.invalidOperation("normalize", reason: "Cannot normalize zero vector")) }
        let inv = 1.0 / sqrt(ms)
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
