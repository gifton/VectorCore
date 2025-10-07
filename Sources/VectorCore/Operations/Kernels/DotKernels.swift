//
//  DotKernels.swift
//  VectorCore
//
//  CPU-only, SIMD-optimized dot product kernels for optimized vector types.
//  Mirrors the 4-accumulator, stride-16 pattern used in optimized vectors,
//  exposed as standalone helpers for reuse by higher-level kernels.
//

import Foundation
import simd

@usableFromInline
internal enum DotKernels {

    // Core implementation: 4 independent accumulators, stride 16 lanes.
    // Accepts storage buffers directly to avoid introducing new protocols.
    @inline(__always)
    private static func dot(storageA: ContiguousArray<SIMD4<Float>>, storageB: ContiguousArray<SIMD4<Float>>, laneCount: Int) -> Float {
        #if DEBUG
        assert(storageA.count == laneCount && storageB.count == laneCount, "Storage count mismatch")
        assert(laneCount % 16 == 0, "Lane count must be multiple of 16.")
        #endif

        var acc0 = SIMD4<Float>()
        var acc1 = SIMD4<Float>()
        var acc2 = SIMD4<Float>()
        var acc3 = SIMD4<Float>()

        for i in stride(from: 0, to: laneCount, by: 16) {
            // Block 0
            acc0 += storageA[i+0] * storageB[i+0]
            acc1 += storageA[i+1] * storageB[i+1]
            acc2 += storageA[i+2] * storageB[i+2]
            acc3 += storageA[i+3] * storageB[i+3]

            // Block 1
            acc0 += storageA[i+4] * storageB[i+4]
            acc1 += storageA[i+5] * storageB[i+5]
            acc2 += storageA[i+6] * storageB[i+6]
            acc3 += storageA[i+7] * storageB[i+7]

            // Block 2
            acc0 += storageA[i+8] * storageB[i+8]
            acc1 += storageA[i+9] * storageB[i+9]
            acc2 += storageA[i+10] * storageB[i+10]
            acc3 += storageA[i+11] * storageB[i+11]

            // Block 3
            acc0 += storageA[i+12] * storageB[i+12]
            acc1 += storageA[i+13] * storageB[i+13]
            acc2 += storageA[i+14] * storageB[i+14]
            acc3 += storageA[i+15] * storageB[i+15]
        }

        let combined = (acc0 + acc1) + (acc2 + acc3)
        return combined.sum()
    }

    // MARK: - Public per-dimension entry points

    @usableFromInline @inline(__always)
    static func dot512(_ a: Vector512Optimized, _ b: Vector512Optimized) -> Float {
        dot(storageA: a.storage, storageB: b.storage, laneCount: 128)
    }

    @usableFromInline @inline(__always)
    static func dot768(_ a: Vector768Optimized, _ b: Vector768Optimized) -> Float {
        dot(storageA: a.storage, storageB: b.storage, laneCount: 192)
    }

    @usableFromInline @inline(__always)
    static func dot1536(_ a: Vector1536Optimized, _ b: Vector1536Optimized) -> Float {
        dot(storageA: a.storage, storageB: b.storage, laneCount: 384)
    }
}
