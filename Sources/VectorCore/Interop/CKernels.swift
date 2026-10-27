// CKernels.swift — Swift wrappers around the VectorCoreC FP32 and INT8 kernels.
// The wrappers call VectorCoreC only when the VC_USE_C_KERNELS compilation condition is defined.
// Package.swift defines no such condition, so every package build runs the Swift loops below
// (VectorCoreCompiledFeatures.useCKernels reports which). The C contracts (compact range output,
// the INT8 modulo law, feature-query semantics) are documented in VectorCoreC.h.

import Foundation
#if canImport(VectorCoreC)
import VectorCoreC
#endif

@inline(__always)
private func assertAligned(_ ptr: UnsafeRawPointer, to alignment: Int) {
    #if DEBUG
    assert(Int(bitPattern: ptr) % alignment == 0, "C kernel received misaligned pointer (expected alignment: \(alignment))")
    #endif
}
@inline(__always)
func vc_hasAVX2() -> Bool {
    #if VC_USE_C_KERNELS
    return vc_has_avx2() != 0
    #else
    return false
    #endif
}

@inline(__always)
func vc_hasNEON() -> Bool {
    #if VC_USE_C_KERNELS
    return vc_has_neon() != 0
    #else
    return false
    #endif
}

// Wrappers for the 512-d FP32 primitives and the INT8 dot product.
// The 16-byte alignment assertions below are wrapper-level Debug checks. Unverified contract: the C
// implementations are not expected to need more than the element type's natural alignment (arm64 FP32 uses
// vld1q_f32, x86_64 FP32 is scalar, INT8 uses vld1q_s8 / _mm_loadu_si128), but no test passes a misaligned pointer.
@usableFromInline
enum CKernels {
    // Dot product 512 (returns scalar)
    static func dot512(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>) -> Float {
        #if VC_USE_C_KERNELS
        assertAligned(UnsafeRawPointer(a), to: 16)
        assertAligned(UnsafeRawPointer(b), to: 16)
        return vc_dot_fp32_512(a, b)
        #else
        // Without VC_USE_C_KERNELS (every package build) this Swift loop runs instead of the C kernel.
        var acc: Float = 0
        for i in 0..<512 { acc += a[i] * b[i] }
        return acc
        #endif
    }

    // L2^2 512 (returns scalar)
    static func l2sq512(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>) -> Float {
        #if VC_USE_C_KERNELS
        assertAligned(UnsafeRawPointer(a), to: 16)
        assertAligned(UnsafeRawPointer(b), to: 16)
        return vc_l2sq_fp32_512(a, b)
        #else
        var acc: Float = 0
        for i in 0..<512 { let d = a[i] - b[i]; acc += d * d }
        return acc
        #endif
    }

    /// INT8 dot product over `lanes` elements: the exact integer dot product reduced modulo 2^32 and
    /// read in the Int32 range (the `vc_dot_int8` law in VectorCoreC.h). The Swift loop below follows
    /// the same law with wrapping `&+=` / `&*`.
    /// With VC_USE_C_KERNELS it calls `vc_dot_int8`: on arm64, SDOT when the compile target defines
    /// `__ARM_FEATURE_DOTPROD`, otherwise the NEON fallback; on x86_64, AVX2 when `vc_has_avx2()`
    /// reports it at run time, otherwise the scalar implementation. Without it (every package build)
    /// the Swift loop runs. `a` and `b` must each point to `lanes` readable values; `lanes == 0`
    /// reads nothing.
    @usableFromInline
    static func dotInt8(_ a: UnsafePointer<Int8>, _ b: UnsafePointer<Int8>, lanes: Int) -> Int32 {
        #if VC_USE_C_KERNELS
        return vc_dot_int8(a, b, lanes)
        #else
        // Scalar fallback when C kernels are disabled
        var acc: Int32 = 0
        for i in 0..<lanes {
            acc &+= Int32(a[i]) &* Int32(b[i])
        }
        return acc
        #endif
    }

    // Range L2^2 512: compact output, written to out[0 ..< end - start] independently of start
    // (VectorCoreC.h states the full caller obligations: start <= end, stride >= 512, live,
    // non-overlapping buffers).
    static func rangeL2sq512(q: UnsafePointer<Float>, base: UnsafePointer<Float>, strideFloats: Int, start: Int, end: Int, out: UnsafeMutablePointer<Float>) {
        #if VC_USE_C_KERNELS
        assertAligned(UnsafeRawPointer(q), to: 16)
        assertAligned(UnsafeRawPointer(base), to: 16)
        vc_range_l2sq_fp32_512(q, base, Int(strideFloats), Int(start), Int(end), out)
        #else
        let D = 512
        var idx = 0
        for row in start..<end {
            let baseRow = base.advanced(by: row * strideFloats)
            var acc: Float = 0
            for i in 0..<D { let d = q[i] - baseRow[i]; acc += d * d }
            out[idx] = acc
            idx += 1
        }
        #endif
    }
}
