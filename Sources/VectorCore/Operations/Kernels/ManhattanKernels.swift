//
//  ManhattanKernels.swift
//  VectorCore
//
//  CPU-only, SIMD-optimized L1 (Manhattan) distance kernels for optimized vectors.
//

import Foundation
import simd

public enum ManhattanKernels {

    @inline(__always)
    private static func distance(storageA: ContiguousArray<SIMD4<Float>>, storageB: ContiguousArray<SIMD4<Float>>, laneCount: Int) -> Float {
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
            acc0 += simd.abs(storageA[i+0] - storageB[i+0])
            acc1 += simd.abs(storageA[i+1] - storageB[i+1])
            acc2 += simd.abs(storageA[i+2] - storageB[i+2])
            acc3 += simd.abs(storageA[i+3] - storageB[i+3])

            // Block 1
            acc0 += simd.abs(storageA[i+4] - storageB[i+4])
            acc1 += simd.abs(storageA[i+5] - storageB[i+5])
            acc2 += simd.abs(storageA[i+6] - storageB[i+6])
            acc3 += simd.abs(storageA[i+7] - storageB[i+7])

            // Block 2
            acc0 += simd.abs(storageA[i+8] - storageB[i+8])
            acc1 += simd.abs(storageA[i+9] - storageB[i+9])
            acc2 += simd.abs(storageA[i+10] - storageB[i+10])
            acc3 += simd.abs(storageA[i+11] - storageB[i+11])

            // Block 3
            acc0 += simd.abs(storageA[i+12] - storageB[i+12])
            acc1 += simd.abs(storageA[i+13] - storageB[i+13])
            acc2 += simd.abs(storageA[i+14] - storageB[i+14])
            acc3 += simd.abs(storageA[i+15] - storageB[i+15])
        }

        let combined = (acc0 + acc1) + (acc2 + acc3)
        return combined.sum()
    }

    @inline(__always)
    public static func distance512(_ a: Vector512Optimized, _ b: Vector512Optimized) -> Float {
        distance(storageA: a.storage, storageB: b.storage, laneCount: 128)
    }

    @inline(__always)
    public static func distance768(_ a: Vector768Optimized, _ b: Vector768Optimized) -> Float {
        distance(storageA: a.storage, storageB: b.storage, laneCount: 192)
    }

    @inline(__always)
    public static func distance1536(_ a: Vector1536Optimized, _ b: Vector1536Optimized) -> Float {
        distance(storageA: a.storage, storageB: b.storage, laneCount: 384)
    }
}
