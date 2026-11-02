// VectorCoreC.h — public C API of the VectorCoreC target.
//
// This header shares the target's name, so SPM treats it as the module's
// umbrella header: every public header must be included here to be visible
// to Swift.
//
// Supported surface: the four feature queries, the three FP32 512-wide kernels,
// the generic INT8 dot product (below) and the LAPACK shim (vc_lapack.h). The
// object files also export internal vc_scalar_*, vc_arm64_* and vc_x86_* symbols
// (default visibility, declared in no public header); they are not supported API.
// Swift calls the FP32 and INT8 kernels only when VC_USE_C_KERNELS is defined,
// which Package.swift does not do (Sources/VectorCore/Interop/CKernels.swift);
// the LAPACK shim is called by LAPACKLinearAlgebraProvider.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "vc_lapack.h"

// --- Portability & visibility macros ---
#if !defined(VC_EXPORT)
  #if defined(_WIN32)
    #define VC_EXPORT __declspec(dllexport)
  #else
    #define VC_EXPORT __attribute__((visibility("default")))
  #endif
#endif

#if !defined(VC_RESTRICT)
  #if defined(__GNUC__) || defined(__clang__)
    #define VC_RESTRICT __restrict
  #else
    #define VC_RESTRICT
  #endif
#endif

#if !defined(VC_INLINE)
  #if defined(__GNUC__) || defined(__clang__)
    #define VC_INLINE static inline __attribute__((always_inline))
  #else
    #define VC_INLINE static __inline
  #endif
#endif

#if defined(__cplusplus)
extern "C" {
#endif

// --- CPU feature queries ---
// Each returns 1 or 0. They do not all answer the same question:
// - vc_has_avx2 / vc_has_avx512f: on x86_64 builds with GCC/Clang, a RUNTIME
//   query of the executing CPU (__builtin_cpu_supports); 0 on every other build.
// - vc_has_neon / vc_has_dotprod: COMPILE-TARGET macros (__ARM_NEON /
//   __ARM_FEATURE_DOTPROD), fixed when this target is compiled; they report what
//   the code was compiled for, not a runtime probe of the executing CPU. Clang
//   defines __ARM_FEATURE_DOTPROD for the arm64 macOS 14 target this package
//   builds and not for `-march=armv8-a` (D02 evidence script).
VC_EXPORT int vc_has_avx2(void);
VC_EXPORT int vc_has_avx512f(void);
VC_EXPORT int vc_has_neon(void);
VC_EXPORT int vc_has_dotprod(void);

// --- FP32 kernels (fixed dimension 512) ---
// arm64 builds run a 4-accumulator NEON implementation; x86_64 builds run the
// scalar implementation (the x86_64 FP32 entries delegate to it). Summation order
// differs between the two, so results are not promised to be bit-identical
// across architectures. a and b each point to 512 readable floats.
VC_EXPORT float vc_dot_fp32_512(const float* VC_RESTRICT a, const float* VC_RESTRICT b);
VC_EXPORT float vc_l2sq_fp32_512(const float* VC_RESTRICT a, const float* VC_RESTRICT b);

// Range kernel: squared L2 distance between the 512-float query `q` and each row
// `base + row * strideFloats` for row in [start, end), AoS layout.
// Output is COMPACT and independent of start: the result for `row` is written to
// out[row - start], so the call writes exactly out[0 ..< end - start] and nothing
// else. Caller obligations (not checked; this is an unsafe interface):
// - start <= end (other values are outside the contract);
// - strideFloats >= 512 for every referenced row, every row offset
//   row * strideFloats and the address span it implies are representable, and
//   each referenced row holds 512 readable, initialized floats;
// - out has room for end - start floats and does not overlap q or base;
// - all buffers stay live for the duration of the (synchronous) call.
// A valid empty range (start == end) performs no pointer access, so any pointer
// (including NULL) may be passed for it.
VC_EXPORT void vc_range_l2sq_fp32_512(
    const float* VC_RESTRICT q,
    const float* VC_RESTRICT base,
    size_t strideFloats,
    size_t start,
    size_t end,
    float* VC_RESTRICT out
);

// --- INT8 kernel ---
// Dot product of `lanes` int8 pairs. The result is the exact integer dot product
// reduced modulo 2^32 and read in the int32_t range (two's complement): whenever
// the exact sum lies in [INT32_MIN, INT32_MAX] the result is that sum, even if a
// partial sum leaves that range. Every implementation (scalar, arm64 SDOT, arm64
// NEON fallback, x86_64 AVX2) accumulates in unsigned or SIMD integer arithmetic
// with no signed overflow. Evidence: the dispatcher by D02CContractTests; the
// scalar, SDOT and NEON-fallback entries by Scripts/ci/tests/D02CContract.py
// (UndefinedBehaviorSanitizer runs and an LLVM IR scan); x86_64 AVX2 by the IR
// scan and disassembly only, not executed. For lanes <= 131071 the exact sum always fits
// (|a*b| <= 16384); the fixed 512/768/1536 Swift callers stay within
// 1536 * 16384 = 25,165,824. Caller obligations: a and b each point to `lanes`
// readable bytes; lanes == 0 reads nothing (NULL is allowed). There is no lane cap.
VC_EXPORT int32_t vc_dot_int8(const int8_t* VC_RESTRICT a, const int8_t* VC_RESTRICT b, size_t lanes);

#if defined(__cplusplus)
}
#endif
