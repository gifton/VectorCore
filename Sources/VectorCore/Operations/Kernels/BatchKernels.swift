//
//  BatchKernels.swift
//  VectorCore
//
//  CPU-only batch kernels over `SIMD4<Float>` lanes with 2-way register blocking.
//  Provides euclidean^2, cosine(pre-normalized), and cosine(fused) variants
//  for 384/512/768/1536 optimized vector types.
//

import Foundation
import simd

public enum BatchKernels {

    // MARK: - Euclidean Squared (2-way blocked)

    public static func range_euclid2_384(query: Vector384Optimized, candidates: [Vector384Optimized], range: Range<Int>, out: UnsafeMutableBufferPointer<Float>) {
        let q = query.storage
        let laneCount = 96
        var i = range.lowerBound
        let end = range.upperBound
        while i + 1 < end {
            let c0 = candidates[i].storage
            let c1 = candidates[i+1].storage
            var a0 = SIMD4<Float>(), a1 = SIMD4<Float>(), a2 = SIMD4<Float>(), a3 = SIMD4<Float>()
            var b0 = SIMD4<Float>(), b1 = SIMD4<Float>(), b2 = SIMD4<Float>(), b3 = SIMD4<Float>()
            for k in stride(from: 0, to: laneCount, by: 16) {
                var d = q[k+0] - c0[k+0]; a0 += d * d; d = q[k+0] - c1[k+0]; b0 += d * d
                d = q[k+1] - c0[k+1]; a1 += d * d; d = q[k+1] - c1[k+1]; b1 += d * d
                d = q[k+2] - c0[k+2]; a2 += d * d; d = q[k+2] - c1[k+2]; b2 += d * d
                d = q[k+3] - c0[k+3]; a3 += d * d; d = q[k+3] - c1[k+3]; b3 += d * d

                d = q[k+4] - c0[k+4]; a0 += d * d; d = q[k+4] - c1[k+4]; b0 += d * d
                d = q[k+5] - c0[k+5]; a1 += d * d; d = q[k+5] - c1[k+5]; b1 += d * d
                d = q[k+6] - c0[k+6]; a2 += d * d; d = q[k+6] - c1[k+6]; b2 += d * d
                d = q[k+7] - c0[k+7]; a3 += d * d; d = q[k+7] - c1[k+7]; b3 += d * d

                d = q[k+8] - c0[k+8]; a0 += d * d; d = q[k+8] - c1[k+8]; b0 += d * d
                d = q[k+9] - c0[k+9]; a1 += d * d; d = q[k+9] - c1[k+9]; b1 += d * d
                d = q[k+10] - c0[k+10]; a2 += d * d; d = q[k+10] - c1[k+10]; b2 += d * d
                d = q[k+11] - c0[k+11]; a3 += d * d; d = q[k+11] - c1[k+11]; b3 += d * d

                d = q[k+12] - c0[k+12]; a0 += d * d; d = q[k+12] - c1[k+12]; b0 += d * d
                d = q[k+13] - c0[k+13]; a1 += d * d; d = q[k+13] - c1[k+13]; b1 += d * d
                d = q[k+14] - c0[k+14]; a2 += d * d; d = q[k+14] - c1[k+14]; b2 += d * d
                d = q[k+15] - c0[k+15]; a3 += d * d; d = q[k+15] - c1[k+15]; b3 += d * d
            }
            let s0 = ((a0 + a1) + (a2 + a3)).sum()
            let s1 = ((b0 + b1) + (b2 + b3)).sum()
            out[i - range.lowerBound] = s0
            out[i + 1 - range.lowerBound] = s1
            i += 2
        }
        if i < end {
            out[i - range.lowerBound] = EuclideanKernels.squared384(query, candidates[i])
        }
    }

    public static func range_euclid2_512(query: Vector512Optimized, candidates: [Vector512Optimized], range: Range<Int>, out: UnsafeMutableBufferPointer<Float>) {
        let q = query.storage
        let laneCount = 128
        var i = range.lowerBound
        let end = range.upperBound
        while i + 1 < end {
            let c0 = candidates[i].storage
            let c1 = candidates[i+1].storage
            var a0 = SIMD4<Float>(), a1 = SIMD4<Float>(), a2 = SIMD4<Float>(), a3 = SIMD4<Float>()
            var b0 = SIMD4<Float>(), b1 = SIMD4<Float>(), b2 = SIMD4<Float>(), b3 = SIMD4<Float>()
            for k in stride(from: 0, to: laneCount, by: 16) {
                var d = q[k+0] - c0[k+0]; a0 += d * d; d = q[k+0] - c1[k+0]; b0 += d * d
                d = q[k+1] - c0[k+1]; a1 += d * d; d = q[k+1] - c1[k+1]; b1 += d * d
                d = q[k+2] - c0[k+2]; a2 += d * d; d = q[k+2] - c1[k+2]; b2 += d * d
                d = q[k+3] - c0[k+3]; a3 += d * d; d = q[k+3] - c1[k+3]; b3 += d * d

                d = q[k+4] - c0[k+4]; a0 += d * d; d = q[k+4] - c1[k+4]; b0 += d * d
                d = q[k+5] - c0[k+5]; a1 += d * d; d = q[k+5] - c1[k+5]; b1 += d * d
                d = q[k+6] - c0[k+6]; a2 += d * d; d = q[k+6] - c1[k+6]; b2 += d * d
                d = q[k+7] - c0[k+7]; a3 += d * d; d = q[k+7] - c1[k+7]; b3 += d * d

                d = q[k+8] - c0[k+8]; a0 += d * d; d = q[k+8] - c1[k+8]; b0 += d * d
                d = q[k+9] - c0[k+9]; a1 += d * d; d = q[k+9] - c1[k+9]; b1 += d * d
                d = q[k+10] - c0[k+10]; a2 += d * d; d = q[k+10] - c1[k+10]; b2 += d * d
                d = q[k+11] - c0[k+11]; a3 += d * d; d = q[k+11] - c1[k+11]; b3 += d * d

                d = q[k+12] - c0[k+12]; a0 += d * d; d = q[k+12] - c1[k+12]; b0 += d * d
                d = q[k+13] - c0[k+13]; a1 += d * d; d = q[k+13] - c1[k+13]; b1 += d * d
                d = q[k+14] - c0[k+14]; a2 += d * d; d = q[k+14] - c1[k+14]; b2 += d * d
                d = q[k+15] - c0[k+15]; a3 += d * d; d = q[k+15] - c1[k+15]; b3 += d * d
            }
            let s0 = ((a0 + a1) + (a2 + a3)).sum()
            let s1 = ((b0 + b1) + (b2 + b3)).sum()
            out[i - range.lowerBound] = s0
            out[i + 1 - range.lowerBound] = s1
            i += 2
        }
        if i < end {
            out[i - range.lowerBound] = EuclideanKernels.squared512(query, candidates[i])
        }
    }

    public static func range_euclid2_768(query: Vector768Optimized, candidates: [Vector768Optimized], range: Range<Int>, out: UnsafeMutableBufferPointer<Float>) {
        let q = query.storage
        let laneCount = 192
        var i = range.lowerBound
        let end = range.upperBound
        while i + 1 < end {
            let c0 = candidates[i].storage
            let c1 = candidates[i+1].storage
            var a0 = SIMD4<Float>(), a1 = SIMD4<Float>(), a2 = SIMD4<Float>(), a3 = SIMD4<Float>()
            var b0 = SIMD4<Float>(), b1 = SIMD4<Float>(), b2 = SIMD4<Float>(), b3 = SIMD4<Float>()
            for k in stride(from: 0, to: laneCount, by: 16) {
                var d = q[k+0] - c0[k+0]; a0 += d * d; d = q[k+0] - c1[k+0]; b0 += d * d
                d = q[k+1] - c0[k+1]; a1 += d * d; d = q[k+1] - c1[k+1]; b1 += d * d
                d = q[k+2] - c0[k+2]; a2 += d * d; d = q[k+2] - c1[k+2]; b2 += d * d
                d = q[k+3] - c0[k+3]; a3 += d * d; d = q[k+3] - c1[k+3]; b3 += d * d

                d = q[k+4] - c0[k+4]; a0 += d * d; d = q[k+4] - c1[k+4]; b0 += d * d
                d = q[k+5] - c0[k+5]; a1 += d * d; d = q[k+5] - c1[k+5]; b1 += d * d
                d = q[k+6] - c0[k+6]; a2 += d * d; d = q[k+6] - c1[k+6]; b2 += d * d
                d = q[k+7] - c0[k+7]; a3 += d * d; d = q[k+7] - c1[k+7]; b3 += d * d

                d = q[k+8] - c0[k+8]; a0 += d * d; d = q[k+8] - c1[k+8]; b0 += d * d
                d = q[k+9] - c0[k+9]; a1 += d * d; d = q[k+9] - c1[k+9]; b1 += d * d
                d = q[k+10] - c0[k+10]; a2 += d * d; d = q[k+10] - c1[k+10]; b2 += d * d
                d = q[k+11] - c0[k+11]; a3 += d * d; d = q[k+11] - c1[k+11]; b3 += d * d

                d = q[k+12] - c0[k+12]; a0 += d * d; d = q[k+12] - c1[k+12]; b0 += d * d
                d = q[k+13] - c0[k+13]; a1 += d * d; d = q[k+13] - c1[k+13]; b1 += d * d
                d = q[k+14] - c0[k+14]; a2 += d * d; d = q[k+14] - c1[k+14]; b2 += d * d
                d = q[k+15] - c0[k+15]; a3 += d * d; d = q[k+15] - c1[k+15]; b3 += d * d
            }
            let s0 = ((a0 + a1) + (a2 + a3)).sum()
            let s1 = ((b0 + b1) + (b2 + b3)).sum()
            out[i - range.lowerBound] = s0
            out[i + 1 - range.lowerBound] = s1
            i += 2
        }
        if i < end {
            out[i - range.lowerBound] = EuclideanKernels.squared768(query, candidates[i])
        }
    }

    public static func range_euclid2_1536(query: Vector1536Optimized, candidates: [Vector1536Optimized], range: Range<Int>, out: UnsafeMutableBufferPointer<Float>) {
        let q = query.storage
        let laneCount = 384
        var i = range.lowerBound
        let end = range.upperBound
        while i + 1 < end {
            let c0 = candidates[i].storage
            let c1 = candidates[i+1].storage
            var a0 = SIMD4<Float>(), a1 = SIMD4<Float>(), a2 = SIMD4<Float>(), a3 = SIMD4<Float>()
            var b0 = SIMD4<Float>(), b1 = SIMD4<Float>(), b2 = SIMD4<Float>(), b3 = SIMD4<Float>()
            for k in stride(from: 0, to: laneCount, by: 16) {
                var d = q[k+0] - c0[k+0]; a0 += d * d; d = q[k+0] - c1[k+0]; b0 += d * d
                d = q[k+1] - c0[k+1]; a1 += d * d; d = q[k+1] - c1[k+1]; b1 += d * d
                d = q[k+2] - c0[k+2]; a2 += d * d; d = q[k+2] - c1[k+2]; b2 += d * d
                d = q[k+3] - c0[k+3]; a3 += d * d; d = q[k+3] - c1[k+3]; b3 += d * d

                d = q[k+4] - c0[k+4]; a0 += d * d; d = q[k+4] - c1[k+4]; b0 += d * d
                d = q[k+5] - c0[k+5]; a1 += d * d; d = q[k+5] - c1[k+5]; b1 += d * d
                d = q[k+6] - c0[k+6]; a2 += d * d; d = q[k+6] - c1[k+6]; b2 += d * d
                d = q[k+7] - c0[k+7]; a3 += d * d; d = q[k+7] - c1[k+7]; b3 += d * d

                d = q[k+8] - c0[k+8]; a0 += d * d; d = q[k+8] - c1[k+8]; b0 += d * d
                d = q[k+9] - c0[k+9]; a1 += d * d; d = q[k+9] - c1[k+9]; b1 += d * d
                d = q[k+10] - c0[k+10]; a2 += d * d; d = q[k+10] - c1[k+10]; b2 += d * d
                d = q[k+11] - c0[k+11]; a3 += d * d; d = q[k+11] - c1[k+11]; b3 += d * d

                d = q[k+12] - c0[k+12]; a0 += d * d; d = q[k+12] - c1[k+12]; b0 += d * d
                d = q[k+13] - c0[k+13]; a1 += d * d; d = q[k+13] - c1[k+13]; b1 += d * d
                d = q[k+14] - c0[k+14]; a2 += d * d; d = q[k+14] - c1[k+14]; b2 += d * d
                d = q[k+15] - c0[k+15]; a3 += d * d; d = q[k+15] - c1[k+15]; b3 += d * d
            }
            let s0 = ((a0 + a1) + (a2 + a3)).sum()
            let s1 = ((b0 + b1) + (b2 + b3)).sum()
            out[i - range.lowerBound] = s0
            out[i + 1 - range.lowerBound] = s1
            i += 2
        }
        if i < end {
            out[i - range.lowerBound] = EuclideanKernels.squared1536(query, candidates[i])
        }
    }

    // MARK: - Cosine Pre-Normalized (2-way blocked)
    // A non-finite dot gives NaN; the clamp applies only to a finite dot (T15CosineExceptionalTableTests, BK.pre rows).
    // Pairs use `DotKernels.dotPair` and an odd last candidate `CosineKernels.distance*_preNormalized` (`DotKernels.dot`):
    // the same arithmetic, so a candidate's score does not depend on its position in the range (T14 phase 3;
    // BatchKernelsRangeClosureTests).

    public static func range_cosine_preNorm_384(query: Vector384Optimized, candidates: [Vector384Optimized], range: Range<Int>, out: UnsafeMutableBufferPointer<Float>) {
        let q = query.storage
        var i = range.lowerBound
        let end = range.upperBound
        while i + 1 < end {
            let (d0, d1) = DotKernels.dotPair(query: q, candidates[i].storage, candidates[i + 1].storage, laneCount: 96)
            out[i - range.lowerBound] = d0.isFinite ? 1.0 - max(-1.0, min(1.0, d0)) : .nan
            out[i + 1 - range.lowerBound] = d1.isFinite ? 1.0 - max(-1.0, min(1.0, d1)) : .nan
            i += 2
        }
        if i < end {
            out[i - range.lowerBound] = CosineKernels.distance384_preNormalized(query, candidates[i])
        }
    }

    public static func range_cosine_preNorm_512(query: Vector512Optimized, candidates: [Vector512Optimized], range: Range<Int>, out: UnsafeMutableBufferPointer<Float>) {
        let q = query.storage
        var i = range.lowerBound
        let end = range.upperBound
        while i + 1 < end {
            let (d0, d1) = DotKernels.dotPair(query: q, candidates[i].storage, candidates[i + 1].storage, laneCount: 128)
            out[i - range.lowerBound] = d0.isFinite ? 1.0 - max(-1.0, min(1.0, d0)) : .nan
            out[i + 1 - range.lowerBound] = d1.isFinite ? 1.0 - max(-1.0, min(1.0, d1)) : .nan
            i += 2
        }
        if i < end {
            out[i - range.lowerBound] = CosineKernels.distance512_preNormalized(query, candidates[i])
        }
    }

    public static func range_cosine_preNorm_768(query: Vector768Optimized, candidates: [Vector768Optimized], range: Range<Int>, out: UnsafeMutableBufferPointer<Float>) {
        let q = query.storage
        var i = range.lowerBound
        let end = range.upperBound
        while i + 1 < end {
            let (d0, d1) = DotKernels.dotPair(query: q, candidates[i].storage, candidates[i + 1].storage, laneCount: 192)
            out[i - range.lowerBound] = d0.isFinite ? 1.0 - max(-1.0, min(1.0, d0)) : .nan
            out[i + 1 - range.lowerBound] = d1.isFinite ? 1.0 - max(-1.0, min(1.0, d1)) : .nan
            i += 2
        }
        if i < end {
            out[i - range.lowerBound] = CosineKernels.distance768_preNormalized(query, candidates[i])
        }
    }

    public static func range_cosine_preNorm_1536(query: Vector1536Optimized, candidates: [Vector1536Optimized], range: Range<Int>, out: UnsafeMutableBufferPointer<Float>) {
        let q = query.storage
        var i = range.lowerBound
        let end = range.upperBound
        while i + 1 < end {
            let (d0, d1) = DotKernels.dotPair(query: q, candidates[i].storage, candidates[i + 1].storage, laneCount: 384)
            out[i - range.lowerBound] = d0.isFinite ? 1.0 - max(-1.0, min(1.0, d0)) : .nan
            out[i + 1 - range.lowerBound] = d1.isFinite ? 1.0 - max(-1.0, min(1.0, d1)) : .nan
            i += 2
        }
        if i < end {
            out[i - range.lowerBound] = CosineKernels.distance1536_preNormalized(query, candidates[i])
        }
    }

    // MARK: - Cosine Fused (2-way blocked)
    // Pairs use `CosineKernels.fusedPair` and an odd last candidate `CosineKernels.distance*_fused`: the same arithmetic
    // (`CosineKernels.fused`), so a candidate's score does not depend on its position in the range, and on the 512, 768
    // and 1536 types it is the `CosineDistance` witness's score (T14 phase 3; BatchKernelsRangeClosureTests,
    // MetricSearchWitnessConsistencyTests). The query's squared norm is computed once per call.

    public static func range_cosine_fused_384(query: Vector384Optimized, candidates: [Vector384Optimized], range: Range<Int>, out: UnsafeMutableBufferPointer<Float>) {
        let q = query.storage
        let sumQQ = CosineKernels.fusedSquaredNorm(q, laneCount: 96)
        var i = range.lowerBound
        let end = range.upperBound
        while i + 1 < end {
            let (d0, d1) = CosineKernels.fusedPair(query: q, queryNorm: sumQQ, candidates[i].storage, candidates[i + 1].storage,
                                                   laneCount: 96)
            out[i - range.lowerBound] = d0
            out[i + 1 - range.lowerBound] = d1
            i += 2
        }
        if i < end {
            out[i - range.lowerBound] = CosineKernels.distance384_fused(query, candidates[i])
        }
    }

    public static func range_cosine_fused_512(query: Vector512Optimized, candidates: [Vector512Optimized], range: Range<Int>, out: UnsafeMutableBufferPointer<Float>) {
        let q = query.storage
        let sumQQ = CosineKernels.fusedSquaredNorm(q, laneCount: 128)
        var i = range.lowerBound
        let end = range.upperBound
        while i + 1 < end {
            let (d0, d1) = CosineKernels.fusedPair(query: q, queryNorm: sumQQ, candidates[i].storage, candidates[i + 1].storage,
                                                   laneCount: 128)
            out[i - range.lowerBound] = d0
            out[i + 1 - range.lowerBound] = d1
            i += 2
        }
        if i < end {
            out[i - range.lowerBound] = CosineKernels.distance512_fused(query, candidates[i])
        }
    }

    public static func range_cosine_fused_768(query: Vector768Optimized, candidates: [Vector768Optimized], range: Range<Int>, out: UnsafeMutableBufferPointer<Float>) {
        let q = query.storage
        let sumQQ = CosineKernels.fusedSquaredNorm(q, laneCount: 192)
        var i = range.lowerBound
        let end = range.upperBound
        while i + 1 < end {
            let (d0, d1) = CosineKernels.fusedPair(query: q, queryNorm: sumQQ, candidates[i].storage, candidates[i + 1].storage,
                                                   laneCount: 192)
            out[i - range.lowerBound] = d0
            out[i + 1 - range.lowerBound] = d1
            i += 2
        }
        if i < end {
            out[i - range.lowerBound] = CosineKernels.distance768_fused(query, candidates[i])
        }
    }

    public static func range_cosine_fused_1536(query: Vector1536Optimized, candidates: [Vector1536Optimized], range: Range<Int>, out: UnsafeMutableBufferPointer<Float>) {
        let q = query.storage
        let sumQQ = CosineKernels.fusedSquaredNorm(q, laneCount: 384)
        var i = range.lowerBound
        let end = range.upperBound
        while i + 1 < end {
            let (d0, d1) = CosineKernels.fusedPair(query: q, queryNorm: sumQQ, candidates[i].storage, candidates[i + 1].storage,
                                                   laneCount: 384)
            out[i - range.lowerBound] = d0
            out[i + 1 - range.lowerBound] = d1
            i += 2
        }
        if i < end {
            out[i - range.lowerBound] = CosineKernels.distance1536_fused(query, candidates[i])
        }
    }
}

// MARK: - Euclidean (sqrt wrapper over euclid2)

public extension BatchKernels {
    @inlinable
    static func range_euclid_384(query: Vector384Optimized, candidates: [Vector384Optimized], range: Range<Int>, out: UnsafeMutableBufferPointer<Float>) {
        range_euclid2_384(query: query, candidates: candidates, range: range, out: out)
        applySqrt(out: out, count: range.count)
    }

    @inlinable
    static func range_euclid_512(query: Vector512Optimized, candidates: [Vector512Optimized], range: Range<Int>, out: UnsafeMutableBufferPointer<Float>) {
        range_euclid2_512(query: query, candidates: candidates, range: range, out: out)
        applySqrt(out: out, count: range.count)
    }

    @inlinable
    static func range_euclid_768(query: Vector768Optimized, candidates: [Vector768Optimized], range: Range<Int>, out: UnsafeMutableBufferPointer<Float>) {
        range_euclid2_768(query: query, candidates: candidates, range: range, out: out)
        applySqrt(out: out, count: range.count)
    }

    @inlinable
    static func range_euclid_1536(query: Vector1536Optimized, candidates: [Vector1536Optimized], range: Range<Int>, out: UnsafeMutableBufferPointer<Float>) {
        range_euclid2_1536(query: query, candidates: candidates, range: range, out: out)
        applySqrt(out: out, count: range.count)
    }

    // Vectorized sqrt for the out buffer (length = range.count)
    @inlinable
    static func applySqrt(out: UnsafeMutableBufferPointer<Float>, count: Int) {
        guard count != 0, let base = out.baseAddress else { return }
        var i = 0
        while i + 3 < count {
            let v = SIMD4<Float>(base[i+0], base[i+1], base[i+2], base[i+3])
            let r = v.squareRoot()
            base[i+0] = r.x; base[i+1] = r.y; base[i+2] = r.z; base[i+3] = r.w
            i += 4
        }
        while i < count { base[i] = sqrt(base[i]); i += 1 }
    }
}
