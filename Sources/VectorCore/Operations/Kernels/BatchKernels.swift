//
//  BatchKernels.swift
//  VectorCore
//
//  CPU-only, SIMD-optimized batch kernels with 2-way register blocking.
//  Provides euclidean^2, cosine(pre-normalized), and cosine(fused) variants
//  for 512/768/1536 optimized vector types.
//

import Foundation
import simd

public enum BatchKernels {

    // MARK: - Euclidean Squared (2-way blocked)

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

    public static func range_cosine_preNorm_512(query: Vector512Optimized, candidates: [Vector512Optimized], range: Range<Int>, out: UnsafeMutableBufferPointer<Float>) {
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
                let q0 = q[k+0], q1 = q[k+1], q2 = q[k+2], q3 = q[k+3]
                let q4 = q[k+4], q5 = q[k+5], q6 = q[k+6], q7 = q[k+7]
                let q8 = q[k+8], q9 = q[k+9], q10 = q[k+10], q11 = q[k+11]
                let q12 = q[k+12], q13 = q[k+13], q14 = q[k+14], q15 = q[k+15]

                // swiftlint:disable:next line_length
                // Justification: SIMD fused multiply-add chain - breaking would hurt register allocation and readability
                a0 += q0 * c0[k+0] + q2 * c0[k+2] + q4 * c0[k+4] + q6 * c0[k+6] + q8 * c0[k+8] + q10 * c0[k+10] + q12 * c0[k+12] + q14 * c0[k+14]
                // swiftlint:disable:next line_length
                a1 += q1 * c0[k+1] + q3 * c0[k+3] + q5 * c0[k+5] + q7 * c0[k+7] + q9 * c0[k+9] + q11 * c0[k+11] + q13 * c0[k+13] + q15 * c0[k+15]

                // swiftlint:disable:next line_length
                b0 += q0 * c1[k+0] + q2 * c1[k+2] + q4 * c1[k+4] + q6 * c1[k+6] + q8 * c1[k+8] + q10 * c1[k+10] + q12 * c1[k+12] + q14 * c1[k+14]
                // swiftlint:disable:next line_length
                b1 += q1 * c1[k+1] + q3 * c1[k+3] + q5 * c1[k+5] + q7 * c1[k+7] + q9 * c1[k+9] + q11 * c1[k+11] + q13 * c1[k+13] + q15 * c1[k+15]
            }
            let d0 = ((a0 + a1) + (a2 + a3)).sum()
            let d1 = ((b0 + b1) + (b2 + b3)).sum()
            out[i - range.lowerBound] = 1.0 - max(-1.0, min(1.0, d0))
            out[i + 1 - range.lowerBound] = 1.0 - max(-1.0, min(1.0, d1))
            i += 2
        }
        if i < end {
            out[i - range.lowerBound] = CosineKernels.distance512_preNormalized(query, candidates[i])
        }
    }

    public static func range_cosine_preNorm_768(query: Vector768Optimized, candidates: [Vector768Optimized], range: Range<Int>, out: UnsafeMutableBufferPointer<Float>) {
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
                let q0 = q[k+0], q1 = q[k+1], q2 = q[k+2], q3 = q[k+3]
                let q4 = q[k+4], q5 = q[k+5], q6 = q[k+6], q7 = q[k+7]
                let q8 = q[k+8], q9 = q[k+9], q10 = q[k+10], q11 = q[k+11]
                let q12 = q[k+12], q13 = q[k+13], q14 = q[k+14], q15 = q[k+15]

                a0 += q0 * c0[k+0];  a0 += q2 * c0[k+2];  a0 += q4 * c0[k+4];  a0 += q6 * c0[k+6]
                a0 += q8 * c0[k+8];  a0 += q10 * c0[k+10]; a0 += q12 * c0[k+12]; a0 += q14 * c0[k+14]
                a1 += q1 * c0[k+1];  a1 += q3 * c0[k+3];  a1 += q5 * c0[k+5];  a1 += q7 * c0[k+7]
                a1 += q9 * c0[k+9];  a1 += q11 * c0[k+11]; a1 += q13 * c0[k+13]; a1 += q15 * c0[k+15]

                b0 += q0 * c1[k+0];  b0 += q2 * c1[k+2];  b0 += q4 * c1[k+4];  b0 += q6 * c1[k+6]
                b0 += q8 * c1[k+8];  b0 += q10 * c1[k+10]; b0 += q12 * c1[k+12]; b0 += q14 * c1[k+14]
                b1 += q1 * c1[k+1];  b1 += q3 * c1[k+3];  b1 += q5 * c1[k+5];  b1 += q7 * c1[k+7]
                b1 += q9 * c1[k+9];  b1 += q11 * c1[k+11]; b1 += q13 * c1[k+13]; b1 += q15 * c1[k+15]
            }
            let d0 = ((a0 + a1) + (a2 + a3)).sum()
            let d1 = ((b0 + b1) + (b2 + b3)).sum()
            out[i - range.lowerBound] = 1.0 - max(-1.0, min(1.0, d0))
            out[i + 1 - range.lowerBound] = 1.0 - max(-1.0, min(1.0, d1))
            i += 2
        }
        if i < end {
            out[i - range.lowerBound] = CosineKernels.distance768_preNormalized(query, candidates[i])
        }
    }

    public static func range_cosine_preNorm_1536(query: Vector1536Optimized, candidates: [Vector1536Optimized], range: Range<Int>, out: UnsafeMutableBufferPointer<Float>) {
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
                let q0 = q[k+0], q1 = q[k+1], q2 = q[k+2], q3 = q[k+3]
                let q4 = q[k+4], q5 = q[k+5], q6 = q[k+6], q7 = q[k+7]
                let q8 = q[k+8], q9 = q[k+9], q10 = q[k+10], q11 = q[k+11]
                let q12 = q[k+12], q13 = q[k+13], q14 = q[k+14], q15 = q[k+15]

                a0 += q0 * c0[k+0];  a0 += q2 * c0[k+2];  a0 += q4 * c0[k+4];  a0 += q6 * c0[k+6]
                a0 += q8 * c0[k+8];  a0 += q10 * c0[k+10]; a0 += q12 * c0[k+12]; a0 += q14 * c0[k+14]
                a1 += q1 * c0[k+1];  a1 += q3 * c0[k+3];  a1 += q5 * c0[k+5];  a1 += q7 * c0[k+7]
                a1 += q9 * c0[k+9];  a1 += q11 * c0[k+11]; a1 += q13 * c0[k+13]; a1 += q15 * c0[k+15]

                b0 += q0 * c1[k+0];  b0 += q2 * c1[k+2];  b0 += q4 * c1[k+4];  b0 += q6 * c1[k+6]
                b0 += q8 * c1[k+8];  b0 += q10 * c1[k+10]; b0 += q12 * c1[k+12]; b0 += q14 * c1[k+14]
                b1 += q1 * c1[k+1];  b1 += q3 * c1[k+3];  b1 += q5 * c1[k+5];  b1 += q7 * c1[k+7]
                b1 += q9 * c1[k+9];  b1 += q11 * c1[k+11]; b1 += q13 * c1[k+13]; b1 += q15 * c1[k+15]
            }
            let d0 = ((a0 + a1) + (a2 + a3)).sum()
            let d1 = ((b0 + b1) + (b2 + b3)).sum()
            out[i - range.lowerBound] = 1.0 - max(-1.0, min(1.0, d0))
            out[i + 1 - range.lowerBound] = 1.0 - max(-1.0, min(1.0, d1))
            i += 2
        }
        if i < end {
            out[i - range.lowerBound] = CosineKernels.distance1536_preNormalized(query, candidates[i])
        }
    }

    // MARK: - Cosine Fused (2-way blocked)

    public static func range_cosine_fused_512(query: Vector512Optimized, candidates: [Vector512Optimized], range: Range<Int>, out: UnsafeMutableBufferPointer<Float>) {
        let q = query.storage
        // Precompute sum of squares for query
        var qq = SIMD4<Float>()
        for i in stride(from: 0, to: 128, by: 16) {
            qq += q[i+0] * q[i+0]; qq += q[i+1] * q[i+1]; qq += q[i+2] * q[i+2]; qq += q[i+3] * q[i+3]
            qq += q[i+4] * q[i+4]; qq += q[i+5] * q[i+5]; qq += q[i+6] * q[i+6]; qq += q[i+7] * q[i+7]
            qq += q[i+8] * q[i+8]; qq += q[i+9] * q[i+9]; qq += q[i+10] * q[i+10]; qq += q[i+11] * q[i+11]
            qq += q[i+12] * q[i+12]; qq += q[i+13] * q[i+13]; qq += q[i+14] * q[i+14]; qq += q[i+15] * q[i+15]
        }
        let sumQQ = qq.sum()

        var i = range.lowerBound
        let end = range.upperBound
        while i + 1 < end {
            let c0 = candidates[i].storage
            let c1 = candidates[i+1].storage
            var qc0a = SIMD4<Float>(), qc0b = SIMD4<Float>()
            var cc0a = SIMD4<Float>(), cc0b = SIMD4<Float>()
            var qc1a = SIMD4<Float>(), qc1b = SIMD4<Float>()
            var cc1a = SIMD4<Float>(), cc1b = SIMD4<Float>()
            for k in stride(from: 0, to: 128, by: 16) {
                let q0=q[k+0], q1=q[k+1], q2=q[k+2], q3=q[k+3]
                let q4=q[k+4], q5=q[k+5], q6=q[k+6], q7=q[k+7]
                let q8=q[k+8], q9=q[k+9], q10=q[k+10], q11=q[k+11]
                let q12=q[k+12], q13=q[k+13], q14=q[k+14], q15=q[k+15]

                let c00=c0[k+0], c01=c0[k+1], c02=c0[k+2], c03=c0[k+3]
                let c04=c0[k+4], c05=c0[k+5], c06=c0[k+6], c07=c0[k+7]
                let c08=c0[k+8], c09=c0[k+9], c10=c0[k+10], c11=c0[k+11]
                let c12=c0[k+12], c13=c0[k+13], c14=c0[k+14], c15=c0[k+15]

                let c10a=c1[k+0], c11a=c1[k+1], c12a=c1[k+2], c13a=c1[k+3]
                let c14a=c1[k+4], c15a=c1[k+5], c16a=c1[k+6], c17a=c1[k+7]
                let c18a=c1[k+8], c19a=c1[k+9], c1_10a=c1[k+10], c1_11a=c1[k+11]
                let c1_12a=c1[k+12], c1_13a=c1[k+13], c1_14a=c1[k+14], c1_15a=c1[k+15]

                qc0a += q0*c00; qc0a += q2*c02; qc0a += q4*c04; qc0a += q6*c06; qc0a += q8*c08; qc0a += q10*c10; qc0a += q12*c12; qc0a += q14*c14
                qc0b += q1*c01 + q3*c03 + q5*c05 + q7*c07 + q9*c09 + q11*c11 + q13*c13 + q15*c15
                cc0a += c00*c00; cc0a += c02*c02; cc0a += c04*c04; cc0a += c06*c06; cc0a += c08*c08; cc0a += c10*c10; cc0a += c12*c12; cc0a += c14*c14
                cc0b += c01*c01 + c03*c03 + c05*c05 + c07*c07 + c09*c09 + c11*c11 + c13*c13 + c15*c15

                qc1a += q0*c10a; qc1a += q2*c12a; qc1a += q4*c14a; qc1a += q6*c16a; qc1a += q8*c18a; qc1a += q10*c1_10a; qc1a += q12*c1_12a; qc1a += q14*c1_14a
                qc1b += q1*c11a + q3*c13a + q5*c15a + q7*c17a + q9*c19a + q11*c1_11a + q13*c1_13a + q15*c1_15a
                cc1a += c10a*c10a; cc1a += c12a*c12a; cc1a += c14a*c14a; cc1a += c16a*c16a; cc1a += c18a*c18a; cc1a += c1_10a*c1_10a; cc1a += c1_12a*c1_12a; cc1a += c1_14a*c1_14a
                cc1b += c11a*c11a + c13a*c13a + c15a*c15a + c17a*c17a + c19a*c19a + c1_11a*c1_11a + c1_13a*c1_13a + c1_15a*c1_15a
            }
            let dot0 = (qc0a + qc0b).sum()
            let cc0 = (cc0a + cc0b).sum()
            let dot1 = (qc1a + qc1b).sum()
            let cc1 = (cc1a + cc1b).sum()
            out[i - range.lowerBound] = CosineKernels.calculateCosineDistance(dot: dot0, sumAA: sumQQ, sumBB: cc0)
            out[i + 1 - range.lowerBound] = CosineKernels.calculateCosineDistance(dot: dot1, sumAA: sumQQ, sumBB: cc1)
            i += 2
        }
        if i < end {
            out[i - range.lowerBound] = CosineKernels.distance512_fused(query, candidates[i])
        }
    }

    public static func range_cosine_fused_768(query: Vector768Optimized, candidates: [Vector768Optimized], range: Range<Int>, out: UnsafeMutableBufferPointer<Float>) {
        let q = query.storage
        var qq = SIMD4<Float>()
        for i in stride(from: 0, to: 192, by: 16) {
            qq += q[i+0] * q[i+0]; qq += q[i+1] * q[i+1]; qq += q[i+2] * q[i+2]; qq += q[i+3] * q[i+3]
            qq += q[i+4] * q[i+4]; qq += q[i+5] * q[i+5]; qq += q[i+6] * q[i+6]; qq += q[i+7] * q[i+7]
            qq += q[i+8] * q[i+8]; qq += q[i+9] * q[i+9]; qq += q[i+10] * q[i+10]; qq += q[i+11] * q[i+11]
            qq += q[i+12] * q[i+12]; qq += q[i+13] * q[i+13]; qq += q[i+14] * q[i+14]; qq += q[i+15] * q[i+15]
        }
        let sumQQ = qq.sum()

        var i = range.lowerBound
        let end = range.upperBound
        while i + 1 < end {
            let c0 = candidates[i].storage
            let c1 = candidates[i+1].storage
            var qc0a = SIMD4<Float>(), qc0b = SIMD4<Float>()
            var cc0a = SIMD4<Float>(), cc0b = SIMD4<Float>()
            var qc1a = SIMD4<Float>(), qc1b = SIMD4<Float>()
            var cc1a = SIMD4<Float>(), cc1b = SIMD4<Float>()
            for k in stride(from: 0, to: 192, by: 16) {
                let q0=q[k+0], q1=q[k+1], q2=q[k+2], q3=q[k+3]
                let q4=q[k+4], q5=q[k+5], q6=q[k+6], q7=q[k+7]
                let q8=q[k+8], q9=q[k+9], q10=q[k+10], q11=q[k+11]
                let q12=q[k+12], q13=q[k+13], q14=q[k+14], q15=q[k+15]

                let c00=c0[k+0], c01=c0[k+1], c02=c0[k+2], c03=c0[k+3]
                let c04=c0[k+4], c05=c0[k+5], c06=c0[k+6], c07=c0[k+7]
                let c08=c0[k+8], c09=c0[k+9], c10=c0[k+10], c11=c0[k+11]
                let c12=c0[k+12], c13=c0[k+13], c14=c0[k+14], c15=c0[k+15]

                let d00=c1[k+0], d01=c1[k+1], d02=c1[k+2], d03=c1[k+3]
                let d04=c1[k+4], d05=c1[k+5], d06=c1[k+6], d07=c1[k+7]
                let d08=c1[k+8], d09=c1[k+9], d10=c1[k+10], d11=c1[k+11]
                let d12=c1[k+12], d13=c1[k+13], d14=c1[k+14], d15=c1[k+15]

                qc0a += q0*c00; qc0a += q2*c02; qc0a += q4*c04; qc0a += q6*c06; qc0a += q8*c08; qc0a += q10*c10; qc0a += q12*c12; qc0a += q14*c14
                qc0b += q1*c01 + q3*c03 + q5*c05 + q7*c07 + q9*c09 + q11*c11 + q13*c13 + q15*c15
                cc0a += c00*c00; cc0a += c02*c02; cc0a += c04*c04; cc0a += c06*c06; cc0a += c08*c08; cc0a += c10*c10; cc0a += c12*c12; cc0a += c14*c14
                cc0b += c01*c01 + c03*c03 + c05*c05 + c07*c07 + c09*c09 + c11*c11 + c13*c13 + c15*c15

                qc1a += q0*d00; qc1a += q2*d02; qc1a += q4*d04; qc1a += q6*d06; qc1a += q8*d08; qc1a += q10*d10; qc1a += q12*d12; qc1a += q14*d14
                qc1b += q1*d01 + q3*d03 + q5*d05 + q7*d07 + q9*d09 + q11*d11 + q13*d13 + q15*d15
                cc1a += d00*d00; cc1a += d02*d02; cc1a += d04*d04; cc1a += d06*d06; cc1a += d08*d08; cc1a += d10*d10; cc1a += d12*d12; cc1a += d14*d14
                cc1b += d01*d01 + d03*d03 + d05*d05 + d07*d07 + d09*d09 + d11*d11 + d13*d13 + d15*d15
            }
            let dot0 = (qc0a + qc0b).sum()
            let cc0 = (cc0a + cc0b).sum()
            let dot1 = (qc1a + qc1b).sum()
            let cc1 = (cc1a + cc1b).sum()
            out[i - range.lowerBound] = CosineKernels.calculateCosineDistance(dot: dot0, sumAA: sumQQ, sumBB: cc0)
            out[i + 1 - range.lowerBound] = CosineKernels.calculateCosineDistance(dot: dot1, sumAA: sumQQ, sumBB: cc1)
            i += 2
        }
        if i < end {
            out[i - range.lowerBound] = CosineKernels.distance768_fused(query, candidates[i])
        }
    }

    public static func range_cosine_fused_1536(query: Vector1536Optimized, candidates: [Vector1536Optimized], range: Range<Int>, out: UnsafeMutableBufferPointer<Float>) {
        let q = query.storage
        var qq = SIMD4<Float>()
        for i in stride(from: 0, to: 384, by: 16) {
            qq += q[i+0] * q[i+0]; qq += q[i+1] * q[i+1]; qq += q[i+2] * q[i+2]; qq += q[i+3] * q[i+3]
            qq += q[i+4] * q[i+4]; qq += q[i+5] * q[i+5]; qq += q[i+6] * q[i+6]; qq += q[i+7] * q[i+7]
            qq += q[i+8] * q[i+8]; qq += q[i+9] * q[i+9]; qq += q[i+10] * q[i+10]; qq += q[i+11] * q[i+11]
            qq += q[i+12] * q[i+12]; qq += q[i+13] * q[i+13]; qq += q[i+14] * q[i+14]; qq += q[i+15] * q[i+15]
        }
        let sumQQ = qq.sum()

        var i = range.lowerBound
        let end = range.upperBound
        while i + 1 < end {
            let c0 = candidates[i].storage
            let c1 = candidates[i+1].storage
            var qc0a = SIMD4<Float>(), qc0b = SIMD4<Float>()
            var cc0a = SIMD4<Float>(), cc0b = SIMD4<Float>()
            var qc1a = SIMD4<Float>(), qc1b = SIMD4<Float>()
            var cc1a = SIMD4<Float>(), cc1b = SIMD4<Float>()
            for k in stride(from: 0, to: 384, by: 16) {
                let q0=q[k+0], q1=q[k+1], q2=q[k+2], q3=q[k+3]
                let q4=q[k+4], q5=q[k+5], q6=q[k+6], q7=q[k+7]
                let q8=q[k+8], q9=q[k+9], q10=q[k+10], q11=q[k+11]
                let q12=q[k+12], q13=q[k+13], q14=q[k+14], q15=q[k+15]

                let c00=c0[k+0], c01=c0[k+1], c02=c0[k+2], c03=c0[k+3]
                let c04=c0[k+4], c05=c0[k+5], c06=c0[k+6], c07=c0[k+7]
                let c08=c0[k+8], c09=c0[k+9], c10=c0[k+10], c11=c0[k+11]
                let c12=c0[k+12], c13=c0[k+13], c14=c0[k+14], c15=c0[k+15]

                let d00=c1[k+0], d01=c1[k+1], d02=c1[k+2], d03=c1[k+3]
                let d04=c1[k+4], d05=c1[k+5], d06=c1[k+6], d07=c1[k+7]
                let d08=c1[k+8], d09=c1[k+9], d10=c1[k+10], d11=c1[k+11]
                let d12=c1[k+12], d13=c1[k+13], d14=c1[k+14], d15=c1[k+15]

                qc0a += q0*c00; qc0a += q2*c02; qc0a += q4*c04; qc0a += q6*c06; qc0a += q8*c08; qc0a += q10*c10; qc0a += q12*c12; qc0a += q14*c14
                qc0b += q1*c01 + q3*c03 + q5*c05 + q7*c07 + q9*c09 + q11*c11 + q13*c13 + q15*c15
                cc0a += c00*c00; cc0a += c02*c02; cc0a += c04*c04; cc0a += c06*c06; cc0a += c08*c08; cc0a += c10*c10; cc0a += c12*c12; cc0a += c14*c14
                cc0b += c01*c01 + c03*c03 + c05*c05 + c07*c07 + c09*c09 + c11*c11 + c13*c13 + c15*c15

                qc1a += q0*d00; qc1a += q2*d02; qc1a += q4*d04; qc1a += q6*d06; qc1a += q8*d08; qc1a += q10*d10; qc1a += q12*d12; qc1a += q14*d14
                qc1b += q1*d01 + q3*d03 + q5*d05 + q7*d07 + q9*d09 + q11*d11 + q13*d13 + q15*d15
                cc1a += d00*d00; cc1a += d02*d02; cc1a += d04*d04; cc1a += d06*d06; cc1a += d08*d08; cc1a += d10*d10; cc1a += d12*d12; cc1a += d14*d14
                cc1b += d01*d01 + d03*d03 + d05*d05 + d07*d07 + d09*d09 + d11*d11 + d13*d13 + d15*d15
            }
            let dot0 = (qc0a + qc0b).sum()
            let cc0 = (cc0a + cc0b).sum()
            let dot1 = (qc1a + qc1b).sum()
            let cc1 = (cc1a + cc1b).sum()
            out[i - range.lowerBound] = CosineKernels.calculateCosineDistance(dot: dot0, sumAA: sumQQ, sumBB: cc0)
            out[i + 1 - range.lowerBound] = CosineKernels.calculateCosineDistance(dot: dot1, sumAA: sumQQ, sumBB: cc1)
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
