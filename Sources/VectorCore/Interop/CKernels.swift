// CKernels.swift — Swift wrappers around C kernels (usage gated)
// This file provides thin Swift helpers to call into VectorCoreC.

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

// Example wrappers for 512-d FP32 primitives
enum CKernels {
    // Dot product 512 (returns scalar)
    static func dot512(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>) -> Float {
        #if VC_USE_C_KERNELS
        assertAligned(UnsafeRawPointer(a), to: 16)
        assertAligned(UnsafeRawPointer(b), to: 16)
        return vc_dot_fp32_512(a, b)
        #else
        // Fallback should never be called when gated; keep a tiny scalar path for debug.
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

    // Range L2^2 512: writes into out[0..(end-start))
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
