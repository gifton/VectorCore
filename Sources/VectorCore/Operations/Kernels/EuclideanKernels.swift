//
//  EuclideanKernels.swift
//  VectorCore
//
//  CPU-only, SIMD-optimized Euclidean distance kernels for optimized vectors.
//  Uses 4 accumulators and stride-16 over SIMD4<Float> lanes for high ILP.
//

import Foundation
import simd

@usableFromInline
internal enum EuclideanKernels {

    @inline(__always)
    private static func squared(storageA: ContiguousArray<SIMD4<Float>>, storageB: ContiguousArray<SIMD4<Float>>, laneCount: Int) -> Float {
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
            var d = storageA[i+0] - storageB[i+0]; acc0 += d * d
            d = storageA[i+1] - storageB[i+1]; acc1 += d * d
            d = storageA[i+2] - storageB[i+2]; acc2 += d * d
            d = storageA[i+3] - storageB[i+3]; acc3 += d * d

            // Block 1
            d = storageA[i+4] - storageB[i+4]; acc0 += d * d
            d = storageA[i+5] - storageB[i+5]; acc1 += d * d
            d = storageA[i+6] - storageB[i+6]; acc2 += d * d
            d = storageA[i+7] - storageB[i+7]; acc3 += d * d

            // Block 2
            d = storageA[i+8] - storageB[i+8]; acc0 += d * d
            d = storageA[i+9] - storageB[i+9]; acc1 += d * d
            d = storageA[i+10] - storageB[i+10]; acc2 += d * d
            d = storageA[i+11] - storageB[i+11]; acc3 += d * d

            // Block 3
            d = storageA[i+12] - storageB[i+12]; acc0 += d * d
            d = storageA[i+13] - storageB[i+13]; acc1 += d * d
            d = storageA[i+14] - storageB[i+14]; acc2 += d * d
            d = storageA[i+15] - storageB[i+15]; acc3 += d * d
        }

        let combined = (acc0 + acc1) + (acc2 + acc3)
        return combined.sum()
    }

    // MARK: - Public per-dimension helpers (squared)

    @usableFromInline @inline(__always)
    static func squared384(_ a: Vector384Optimized, _ b: Vector384Optimized) -> Float {
        squared(storageA: a.storage, storageB: b.storage, laneCount: 96)
    }

    @usableFromInline @inline(__always)
    static func squared512(_ a: Vector512Optimized, _ b: Vector512Optimized) -> Float {
        squared(storageA: a.storage, storageB: b.storage, laneCount: 128)
    }

    @usableFromInline @inline(__always)
    static func squared768(_ a: Vector768Optimized, _ b: Vector768Optimized) -> Float {
        squared(storageA: a.storage, storageB: b.storage, laneCount: 192)
    }

    @usableFromInline @inline(__always)
    static func squared1536(_ a: Vector1536Optimized, _ b: Vector1536Optimized) -> Float {
        squared(storageA: a.storage, storageB: b.storage, laneCount: 384)
    }

    // MARK: - Euclidean distance (sqrt wrapper)

    @usableFromInline @inline(__always)
    static func distance384(_ a: Vector384Optimized, _ b: Vector384Optimized) -> Float {
        sqrt(squared384(a, b))
    }

    @usableFromInline @inline(__always)
    static func distance512(_ a: Vector512Optimized, _ b: Vector512Optimized) -> Float {
        sqrt(squared512(a, b))
    }

    @usableFromInline @inline(__always)
    static func distance768(_ a: Vector768Optimized, _ b: Vector768Optimized) -> Float {
        sqrt(squared768(a, b))
    }

    @usableFromInline @inline(__always)
    static func distance1536(_ a: Vector1536Optimized, _ b: Vector1536Optimized) -> Float {
        sqrt(squared1536(a, b))
    }
}
