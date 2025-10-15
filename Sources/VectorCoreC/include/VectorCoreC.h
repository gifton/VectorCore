// VectorCoreC.h — Public C API for VectorCore kernels (scaffold)
#pragma once

#include <stddef.h>
#include <stdint.h>

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

// --- CPU feature queries (scaffold) ---
// Return 1 if the feature is available on this CPU at runtime, 0 otherwise.
// Stubs are conservative; external agent can implement robust detection per-arch.
VC_EXPORT int vc_has_avx2(void);
VC_EXPORT int vc_has_avx512f(void);
VC_EXPORT int vc_has_neon(void);
VC_EXPORT int vc_has_dotprod(void);

// --- FP32 kernels (scaffold) ---
// Dot product for fixed dimensions (baseline scalar stub)
VC_EXPORT float vc_dot_fp32_512(const float* VC_RESTRICT a, const float* VC_RESTRICT b);
VC_EXPORT float vc_l2sq_fp32_512(const float* VC_RESTRICT a, const float* VC_RESTRICT b);

// Range kernel: compute L2^2 between query `q` and rows in `base`
// AoS layout with `strideFloats` elements per row; writes to `out[start..end)`
VC_EXPORT void vc_range_l2sq_fp32_512(
    const float* VC_RESTRICT q,
    const float* VC_RESTRICT base,
    size_t strideFloats,
    size_t start,
    size_t end,
    float* VC_RESTRICT out
);

// --- INT8 kernels (scaffold) ---
// int8 dot product over `lanes` elements; returns int32 accumulator
VC_EXPORT int32_t vc_dot_int8(const int8_t* VC_RESTRICT a, const int8_t* VC_RESTRICT b, size_t lanes);

// ============================================================================
// Proposed public APIs for upcoming kernels (see kernel-specs/ for details)
// These declarations define the surface area to be implemented by arch backends.

// 005 — int8 Euclidean (sum of squared differences)
VC_EXPORT int32_t vc_l2sq_int8(const int8_t* VC_RESTRICT a, const int8_t* VC_RESTRICT b, size_t lanes); // TODO: implement

// 006 — dequantize int8 -> fp32
VC_EXPORT void vc_dequantize_fp32_from_int8(const int8_t* VC_RESTRICT in, float scale, int8_t zp, float* VC_RESTRICT out, size_t lanes); // TODO: implement

// 007 — mixed-precision Euclid (int8 vs fp32, fused)
VC_EXPORT float vc_l2sq_mixed_int8_fp32(const int8_t* VC_RESTRICT a, float scaleA, int8_t zpA, const float* VC_RESTRICT b, size_t lanes); // TODO: implement
// Optional: mixed int8 vs int8 (commented until needed)
// VC_EXPORT float vc_l2sq_mixed_int8_int8(const int8_t* VC_RESTRICT a, float scaleA, int8_t zpA, const int8_t* VC_RESTRICT b, float scaleB, int8_t zpB, size_t lanes);

// 008 — SoA L2^2 blocked (generic dim)
VC_EXPORT void vc_soa_l2sq_blocked(
    const float* VC_RESTRICT q,
    const float* VC_RESTRICT soa,
    size_t rowStride,
    size_t dim,
    size_t startRow,
    size_t endRow,
    float* VC_RESTRICT out
); // TODO: implement

// 009/010 — SoA dot blocked (2-way / 4-way)
VC_EXPORT void vc_soa_dot_blocked_2way(
    const float* VC_RESTRICT q,
    const float* VC_RESTRICT soa,
    size_t rowStride,
    size_t dim,
    size_t startRow,
    size_t endRow,
    float* VC_RESTRICT out
); // TODO: implement

VC_EXPORT void vc_soa_dot_blocked_4way(
    const float* VC_RESTRICT q,
    const float* VC_RESTRICT soa,
    size_t rowStride,
    size_t dim,
    size_t startRow,
    size_t endRow,
    float* VC_RESTRICT out
); // TODO: implement

// 011 — SoA cosine fused (with optional precomputed row norms)
VC_EXPORT void vc_soa_cosine_fused_blocked(
    const float* VC_RESTRICT q,
    float qNorm,
    const float* VC_RESTRICT soa,
    size_t rowStride,
    const float* VC_RESTRICT rowNorms, // nullable
    size_t dim,
    size_t startRow,
    size_t endRow,
    float* VC_RESTRICT out
); // TODO: implement

// 012/013 — AoS range L2^2 for D=768 / D=1536
VC_EXPORT void vc_range_l2sq_fp32_768(
    const float* VC_RESTRICT q,
    const float* VC_RESTRICT base,
    size_t strideFloats,
    size_t start,
    size_t end,
    float* VC_RESTRICT out
); // TODO: implement

VC_EXPORT void vc_range_l2sq_fp32_1536(
    const float* VC_RESTRICT q,
    const float* VC_RESTRICT base,
    size_t strideFloats,
    size_t start,
    size_t end,
    float* VC_RESTRICT out
); // TODO: implement

// 014 — Cosine fused pairwise (optional precomputed norms variant)
VC_EXPORT float vc_cosine_fused_pairwise_fp32(
    const float* VC_RESTRICT a,
    const float* VC_RESTRICT b,
    size_t dim
); // TODO: implement

// Optional precomputed norms variant (uncomment if needed)
// VC_EXPORT float vc_cosine_pairwise_fp32_with_norms(
//     const float* VC_RESTRICT a,
//     const float* VC_RESTRICT b,
//     size_t dim,
//     float aNorm,
//     float bNorm
// );

// 015 — Top-K heap maintenance (small K)
VC_EXPORT void vc_heapifyDown_smallk(
    float* VC_RESTRICT heapScores,
    int32_t* VC_RESTRICT heapIndex,
    size_t K
); // TODO: implement

// 016 — Merge two Top-K buffers (small K)
VC_EXPORT void vc_topk_merge_smallk(
    const float* VC_RESTRICT scoresA, const int32_t* VC_RESTRICT indexA, size_t lenA,
    const float* VC_RESTRICT scoresB, const int32_t* VC_RESTRICT indexB, size_t lenB,
    size_t K,
    int isSimilarity,
    float* VC_RESTRICT outScores, int32_t* VC_RESTRICT outIndex
); // TODO: implement

// 017 — Range Top-K (Euclid, small K, AoS)
VC_EXPORT void vc_range_topk_euclid_smallk(
    const float* VC_RESTRICT q,
    const float* VC_RESTRICT base,
    size_t strideFloats,
    size_t dim,
    size_t start,
    size_t end,
    size_t K,
    float* VC_RESTRICT outScores,
    int32_t* VC_RESTRICT outIndex
); // TODO: implement

// 018 — Range Top-K (Cosine, small K, AoS)
VC_EXPORT void vc_range_topk_cosine_smallk(
    const float* VC_RESTRICT q,
    float qNorm,
    const float* VC_RESTRICT base,
    const float* VC_RESTRICT rowNorms, // nullable
    size_t strideFloats,
    size_t dim,
    size_t start,
    size_t end,
    size_t K,
    float* VC_RESTRICT outScores,
    int32_t* VC_RESTRICT outIndex
); // TODO: implement

// 019 — Range Top-K fused cosine, D=512 (AoS)
VC_EXPORT void vc_range_topk_cosine_fused_512(
    const float* VC_RESTRICT q,
    float qNorm,
    const float* VC_RESTRICT base,
    size_t strideFloats,
    size_t start,
    size_t end,
    size_t K,
    float* VC_RESTRICT outScores,
    int32_t* VC_RESTRICT outIndex
); // TODO: implement

// 020 — Magnitude squared (pairwise and optional AoS range)
VC_EXPORT float vc_magnitude_sq_fp32(const float* VC_RESTRICT a, size_t dim); // TODO: implement
VC_EXPORT void vc_range_magnitude_sq_fp32(
    const float* VC_RESTRICT base,
    size_t strideFloats,
    size_t dim,
    size_t start,
    size_t end,
    float* VC_RESTRICT out
); // TODO: implement

// 021 — Scale in place (and optional fused normalize)
VC_EXPORT void vc_scale_in_place_fp32(float* VC_RESTRICT a, size_t dim, float scale); // TODO: implement
// Optional fused normalize (uncomment if needed)
// VC_EXPORT void vc_normalize_in_place_fp32(float* VC_RESTRICT a, size_t dim, float eps);

#if defined(__cplusplus)
}
#endif
