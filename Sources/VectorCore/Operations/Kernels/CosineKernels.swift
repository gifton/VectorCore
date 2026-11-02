//
//  CosineKernels.swift
//  VectorCore
//
//  CPU-only cosine distance kernels over `SIMD4<Float>` lanes.
//  Provides fused one-pass variants and pre-normalized fast paths.
//

import Foundation
import simd

@usableFromInline
internal enum CosineKernels {

    // Helper for safe division/clamping and zero-magnitude guards.
    //
    // Exceptional values (T15 policy; T15CosineExceptionalTableTests.sharedHelperScalarTable and the helper rows of
    // everyImplementationFollowsThePolicy): a NaN or infinite squared norm on either side gives NaN, even when the
    // other side is zero; a squared norm at or below the floor below is a zero vector, and a zero vector on either
    // side (or both) gives distance 1; a non-finite similarity gives NaN (`Float.nan`). The [-1, 1] clamp applies
    // only to a finite similarity, so a NaN can no longer become distance 0 through `min(1, NaN) == 1`.
    @inline(__always)
    public static func calculateCosineDistance(dot: Float, sumAA: Float, sumBB: Float) -> Float {
        // Absolute floor on the squared magnitudes. `Float.leastNormalMagnitude`
        // (~1.18e-38) distinguishes mathematical zero from valid micro-vectors;
        // the previous hardcoded 1e-9 floor corresponds to a magnitude of ~3.16e-5
        // and wrongly classified small-but-valid vectors as zero (distance 1.0).
        let epsilon = Float.leastNormalMagnitude
        // Guard each factor independently. Forming the product sumAA * sumBB overflows
        // to +Inf once it exceeds Float.greatestFiniteMagnitude (~3.4e38), for example
        // sumAA = sumBB = 3e38, which would make dot/Inf collapse to 0 and yield a spurious
        // distance of 1.0 (CosineKernelsTests.testCosineDistanceNoOverflowForHugeIdenticalDirection).
        if sumAA <= epsilon || sumBB <= epsilon {
            // A NaN or infinite squared norm dominates the zero rule; otherwise one or both zero → distance 1
            guard sumAA.isFinite && sumBB.isFinite else { return .nan }
            return 1.0
        }
        // denom = sqrt(sumAA * sumBB) computed as sqrt(sumAA) * sqrt(sumBB) to avoid overflow.
        let denom = sqrt(sumAA) * sqrt(sumBB)
        let similarity = dot / denom
        guard similarity.isFinite else { return .nan }
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

    // MARK: - The fused arithmetic for range and Top-K loops (T14 phase 3)
    //
    // `fusedSquaredNorm`, `fusedCandidate` and `fusedPair` perform `fused`'s operations: lane group g of each 16-group
    // stride accumulates into accumulator g % 4 with `addProduct`, strides in increasing order, and each quantity
    // reduces as `((x0 + x1) + (x2 + x3)).sum()`. A query's squared norm does not depend on the candidate, so computing it
    // once gives the value `fused` computes in each call, and `fusedPair` keeps two candidates' accumulators apart in one
    // pass. Each result therefore equals `fused(storageA: query, storageB: candidate)` bit for bit; checked by
    // BatchKernelsRangeClosureTests.pairSlotsEqualTheOddTail (pair slots against the odd tail) and
    // MetricWitnessEquivalenceTests (witness against the Top-K and range routes). The search pair loops' arithmetic is
    // checked by MetricSearchWitnessConsistencyTests (search equals witness, k-independence, candidate-count
    // independence) and BatchKernelsRangeClosureTests (the closure and chunk-size tests); the T15 bit groups
    // (T15CosineExceptionalTableTests) do not exercise the pair loops.

    /// The squared norm of `x` in `fused`'s accumulator layout (its `sumAA` for `storageA = x`).
    @inline(__always)
    @usableFromInline
    static func fusedSquaredNorm(_ x: ContiguousArray<SIMD4<Float>>, laneCount: Int) -> Float {
        #if DEBUG
        assert(x.count == laneCount && laneCount % 16 == 0, "Lane count must match and be a multiple of 16.")
        #endif
        var n0 = SIMD4<Float>(), n1 = SIMD4<Float>(), n2 = SIMD4<Float>(), n3 = SIMD4<Float>()
        for i in stride(from: 0, to: laneCount, by: 16) {
            for g in stride(from: i, to: i + 16, by: 4) {
                n0.addProduct(x[g], x[g]); n1.addProduct(x[g + 1], x[g + 1])
                n2.addProduct(x[g + 2], x[g + 2]); n3.addProduct(x[g + 3], x[g + 3])
            }
        }
        return ((n0 + n1) + (n2 + n3)).sum()
    }

    /// `fused(storageA: query, storageB: candidate)` with the query's `fusedSquaredNorm` given (the Top-K kernel).
    @inline(__always)
    @usableFromInline
    static func fusedCandidate(query q: ContiguousArray<SIMD4<Float>>, queryNorm: Float,
                               _ c: ContiguousArray<SIMD4<Float>>, laneCount: Int) -> Float {
        #if DEBUG
        assert(q.count == laneCount && c.count == laneCount && laneCount % 16 == 0, "Storage count mismatch")
        #endif
        var d0 = SIMD4<Float>(), d1 = SIMD4<Float>(), d2 = SIMD4<Float>(), d3 = SIMD4<Float>()
        var n0 = SIMD4<Float>(), n1 = SIMD4<Float>(), n2 = SIMD4<Float>(), n3 = SIMD4<Float>()
        for i in stride(from: 0, to: laneCount, by: 16) {
            for g in stride(from: i, to: i + 16, by: 4) {
                d0.addProduct(q[g], c[g]); n0.addProduct(c[g], c[g])
                d1.addProduct(q[g + 1], c[g + 1]); n1.addProduct(c[g + 1], c[g + 1])
                d2.addProduct(q[g + 2], c[g + 2]); n2.addProduct(c[g + 2], c[g + 2])
                d3.addProduct(q[g + 3], c[g + 3]); n3.addProduct(c[g + 3], c[g + 3])
            }
        }
        let dot = ((d0 + d1) + (d2 + d3)).sum()
        return calculateCosineDistance(dot: dot, sumAA: queryNorm, sumBB: ((n0 + n1) + (n2 + n3)).sum())
    }

    /// `fusedCandidate` for two candidates in one pass, with separate accumulators (the `BatchKernels` fused cosine pair
    /// loops).
    @inline(__always)
    static func fusedPair(query q: ContiguousArray<SIMD4<Float>>, queryNorm: Float,
                          _ c0: ContiguousArray<SIMD4<Float>>, _ c1: ContiguousArray<SIMD4<Float>>,
                          laneCount: Int) -> (Float, Float) {
        #if DEBUG
        assert(q.count == laneCount && c0.count == laneCount && c1.count == laneCount && laneCount % 16 == 0,
               "Storage count mismatch")
        #endif
        var d00 = SIMD4<Float>(), d01 = SIMD4<Float>(), d02 = SIMD4<Float>(), d03 = SIMD4<Float>()
        var n00 = SIMD4<Float>(), n01 = SIMD4<Float>(), n02 = SIMD4<Float>(), n03 = SIMD4<Float>()
        var d10 = SIMD4<Float>(), d11 = SIMD4<Float>(), d12 = SIMD4<Float>(), d13 = SIMD4<Float>()
        var n10 = SIMD4<Float>(), n11 = SIMD4<Float>(), n12 = SIMD4<Float>(), n13 = SIMD4<Float>()
        for i in stride(from: 0, to: laneCount, by: 16) {
            for g in stride(from: i, to: i + 16, by: 4) {
                let q0 = q[g], q1 = q[g + 1], q2 = q[g + 2], q3 = q[g + 3]
                let x0 = c0[g], x1 = c0[g + 1], x2 = c0[g + 2], x3 = c0[g + 3]
                let y0 = c1[g], y1 = c1[g + 1], y2 = c1[g + 2], y3 = c1[g + 3]
                d00.addProduct(q0, x0); n00.addProduct(x0, x0); d10.addProduct(q0, y0); n10.addProduct(y0, y0)
                d01.addProduct(q1, x1); n01.addProduct(x1, x1); d11.addProduct(q1, y1); n11.addProduct(y1, y1)
                d02.addProduct(q2, x2); n02.addProduct(x2, x2); d12.addProduct(q2, y2); n12.addProduct(y2, y2)
                d03.addProduct(q3, x3); n03.addProduct(x3, x3); d13.addProduct(q3, y3); n13.addProduct(y3, y3)
            }
        }
        let first = calculateCosineDistance(dot: ((d00 + d01) + (d02 + d03)).sum(), sumAA: queryNorm,
                                            sumBB: ((n00 + n01) + (n02 + n03)).sum())
        let second = calculateCosineDistance(dot: ((d10 + d11) + (d12 + d13)).sum(), sumAA: queryNorm,
                                             sumBB: ((n10 + n11) + (n12 + n13)).sum())
        return (first, second)
    }

    // MARK: - Public fused per-dimension

    public static func distance384_fused(_ a: Vector384Optimized, _ b: Vector384Optimized) -> Float {
        fused(storageA: a.storage, storageB: b.storage, laneCount: 96)
    }
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
    // A non-finite dot (any NaN or infinite coordinate) gives NaN; the clamp applies only to a finite dot
    // (T15CosineExceptionalTableTests, CK.pre rows).

    @inline(__always)
    public static func distance384_preNormalized(_ a: Vector384Optimized, _ b: Vector384Optimized) -> Float {
        let dot = DotKernels.dot384(a, b)
        return dot.isFinite ? 1.0 - max(-1.0, min(1.0, dot)) : .nan
    }
    @inline(__always)
    public static func distance512_preNormalized(_ a: Vector512Optimized, _ b: Vector512Optimized) -> Float {
        let dot = DotKernels.dot512(a, b)
        return dot.isFinite ? 1.0 - max(-1.0, min(1.0, dot)) : .nan
    }
    @inline(__always)
    public static func distance768_preNormalized(_ a: Vector768Optimized, _ b: Vector768Optimized) -> Float {
        let dot = DotKernels.dot768(a, b)
        return dot.isFinite ? 1.0 - max(-1.0, min(1.0, dot)) : .nan
    }
    @inline(__always)
    public static func distance1536_preNormalized(_ a: Vector1536Optimized, _ b: Vector1536Optimized) -> Float {
        let dot = DotKernels.dot1536(a, b)
        return dot.isFinite ? 1.0 - max(-1.0, min(1.0, dot)) : .nan
    }
}
