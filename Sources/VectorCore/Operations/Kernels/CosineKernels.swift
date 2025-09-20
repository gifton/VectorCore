//
//  CosineKernels.swift
//  VectorCore
//
//  CPU-only, SIMD-optimized cosine distance kernels.
//  Provides fused one-pass variants and pre-normalized fast paths.
//

import Foundation
import simd

public enum CosineKernels {

    // Helper for safe division/clamping and zero-magnitude guards
    @inline(__always)
    public static func calculateCosineDistance(dot: Float, sumAA: Float, sumBB: Float) -> Float {
        let denomSq = sumAA * sumBB
        let epsilon: Float = 1e-9
        if denomSq <= epsilon {
            // Both near-zero → distance 0; else one is zero → distance 1
            return (sumAA <= epsilon && sumBB <= epsilon) ? 0.0 : 1.0
        }
        let denom = sqrt(denomSq)
        let similarity = dot / denom
        let clamped = max(-1.0, min(1.0, similarity))
        return 1.0 - clamped
    }

    // Fused one-pass cosine: accumulates dot(a,b), dot(a,a), dot(b,b) with 4 accumulators over 16-lane strides
    @inline(__always)
    @usableFromInline
    static func fused(storageA: ContiguousArray<SIMD4<Float>>, storageB: ContiguousArray<SIMD4<Float>>, laneCount: Int) -> Float {
        #if DEBUG
        assert(storageA.count == laneCount && storageB.count == laneCount, "Storage count mismatch")
        assert(laneCount % 16 == 0, "Lane count must be multiple of 16.")
        #endif

        var ab0 = SIMD4<Float>(), ab1 = SIMD4<Float>(), ab2 = SIMD4<Float>(), ab3 = SIMD4<Float>()
        var aa0 = SIMD4<Float>(), aa1 = SIMD4<Float>(), aa2 = SIMD4<Float>(), aa3 = SIMD4<Float>()
        var bb0 = SIMD4<Float>(), bb1 = SIMD4<Float>(), bb2 = SIMD4<Float>(), bb3 = SIMD4<Float>()

        for i in stride(from: 0, to: laneCount, by: 16) {
            // Group 0 (0-3)
            let a0 = storageA[i+0], b0 = storageB[i+0]
            ab0.addProduct(a0, b0); aa0.addProduct(a0, a0); bb0.addProduct(b0, b0)
            let a1 = storageA[i+1], b1 = storageB[i+1]
            ab1.addProduct(a1, b1); aa1.addProduct(a1, a1); bb1.addProduct(b1, b1)
            let a2 = storageA[i+2], b2 = storageB[i+2]
            ab2.addProduct(a2, b2); aa2.addProduct(a2, a2); bb2.addProduct(b2, b2)
            let a3 = storageA[i+3], b3 = storageB[i+3]
            ab3.addProduct(a3, b3); aa3.addProduct(a3, a3); bb3.addProduct(b3, b3)

            // Group 1 (4-7)
            let a4 = storageA[i+4], b4 = storageB[i+4]
            ab0.addProduct(a4, b4); aa0.addProduct(a4, a4); bb0.addProduct(b4, b4)
            let a5 = storageA[i+5], b5 = storageB[i+5]
            ab1.addProduct(a5, b5); aa1.addProduct(a5, a5); bb1.addProduct(b5, b5)
            let a6 = storageA[i+6], b6 = storageB[i+6]
            ab2.addProduct(a6, b6); aa2.addProduct(a6, a6); bb2.addProduct(b6, b6)
            let a7 = storageA[i+7], b7 = storageB[i+7]
            ab3.addProduct(a7, b7); aa3.addProduct(a7, a7); bb3.addProduct(b7, b7)

            // Group 2 (8-11)
            let a8 = storageA[i+8], b8 = storageB[i+8]
            ab0.addProduct(a8, b8); aa0.addProduct(a8, a8); bb0.addProduct(b8, b8)
            let a9 = storageA[i+9], b9 = storageB[i+9]
            ab1.addProduct(a9, b9); aa1.addProduct(a9, a9); bb1.addProduct(b9, b9)
            let a10 = storageA[i+10], b10 = storageB[i+10]
            ab2.addProduct(a10, b10); aa2.addProduct(a10, a10); bb2.addProduct(b10, b10)
            let a11 = storageA[i+11], b11 = storageB[i+11]
            ab3.addProduct(a11, b11); aa3.addProduct(a11, a11); bb3.addProduct(b11, b11)

            // Group 3 (12-15)
            let a12 = storageA[i+12], b12 = storageB[i+12]
            ab0.addProduct(a12, b12); aa0.addProduct(a12, a12); bb0.addProduct(b12, b12)
            let a13 = storageA[i+13], b13 = storageB[i+13]
            ab1.addProduct(a13, b13); aa1.addProduct(a13, a13); bb1.addProduct(b13, b13)
            let a14 = storageA[i+14], b14 = storageB[i+14]
            ab2.addProduct(a14, b14); aa2.addProduct(a14, a14); bb2.addProduct(b14, b14)
            let a15 = storageA[i+15], b15 = storageB[i+15]
            ab3.addProduct(a15, b15); aa3.addProduct(a15, a15); bb3.addProduct(b15, b15)
        }

        let dot = ((ab0 + ab1) + (ab2 + ab3)).sum()
        let sumAA = ((aa0 + aa1) + (aa2 + aa3)).sum()
        let sumBB = ((bb0 + bb1) + (bb2 + bb3)).sum()
        return calculateCosineDistance(dot: dot, sumAA: sumAA, sumBB: sumBB)
    }

    // MARK: - Public fused per-dimension

    public static func distance512_fused(_ a: Vector512Optimized, _ b: Vector512Optimized) -> Float {
        fused(storageA: a.storage, storageB: b.storage, laneCount: 128)
    }
    public static func distance768_fused(_ a: Vector768Optimized, _ b: Vector768Optimized) -> Float {
        fused(storageA: a.storage, storageB: b.storage, laneCount: 192)
    }
    public static func distance1536_fused(_ a: Vector1536Optimized, _ b: Vector1536Optimized) -> Float {
        fused(storageA: a.storage, storageB: b.storage, laneCount: 384)
    }

    // MARK: - Pre-normalized fast paths (1 − dot)

    @inline(__always)
    public static func distance512_preNormalized(_ a: Vector512Optimized, _ b: Vector512Optimized) -> Float {
        let dot = DotKernels.dot512(a, b)
        return 1.0 - max(-1.0, min(1.0, dot))
    }
    @inline(__always)
    public static func distance768_preNormalized(_ a: Vector768Optimized, _ b: Vector768Optimized) -> Float {
        let dot = DotKernels.dot768(a, b)
        return 1.0 - max(-1.0, min(1.0, dot))
    }
    @inline(__always)
    public static func distance1536_preNormalized(_ a: Vector1536Optimized, _ b: Vector1536Optimized) -> Float {
        let dot = DotKernels.dot1536(a, b)
        return 1.0 - max(-1.0, min(1.0, dot))
    }
}
