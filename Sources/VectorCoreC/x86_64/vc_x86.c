// x86_64 implementations — AVX2 intrinsics (INT8); the FP32 entries delegate to the scalar implementations.
#include "VectorCoreC.h"
#include "../common/vc_int8_accum.h"
#include <stddef.h>
#include <stdint.h>

// Scalar references (fallback)
extern float vc_scalar_dot_fp32_512(const float* a, const float* b);
extern float vc_scalar_l2sq_fp32_512(const float* a, const float* b);
extern void  vc_scalar_range_l2sq_fp32_512(const float* q, const float* base, size_t strideFloats, size_t start, size_t end, float* out);

#if defined(__x86_64__)

// FP32 entries — delegate to the scalar implementations (no AVX2 FP32 kernel exists).
float vc_x86_dot_fp32_512(const float* a, const float* b) {
    return vc_scalar_dot_fp32_512(a, b);
}

float vc_x86_l2sq_fp32_512(const float* a, const float* b) {
    return vc_scalar_l2sq_fp32_512(a, b);
}

void vc_x86_range_l2sq_fp32_512(
    const float* q,
    const float* base,
    size_t strideFloats,
    size_t start,
    size_t end,
    float* out
) {
    vc_scalar_range_l2sq_fp32_512(q, base, strideFloats, start, end, out);
}

// --- INT8 Dot Product: AVX2 ---
// Uses cvtepi8_epi16 (sign-extend int8 -> int16) + madd_epi16 (multiply int16 pairs,
// pairwise horizontal add to int32). This approach avoids the signedness trap of
// maddubs_epi16 which treats the first operand as unsigned. Each madd_epi16 lane is
// at most 2 * 16384 in magnitude. Follows the vc_dot_int8 law (VectorCoreC.h): the
// epi32 lane additions and horizontal adds wrap modulo 2^32, the extracted pattern and
// the scalar tail are combined in uint32_t, and vc_int8_accum_to_int32 maps the result.
// Block loops test the remaining count (vc_int8_block_fits), so i <= lanes holds.
// Not executed on the arm64 development host (cross-compiled and disassembled only).
//
// __attribute__((target("avx2"))) enables AVX2 codegen for this function only,
// avoiding .unsafeFlags in Package.swift (which breaks versioned dependency consumption).
// On Clang, target("avx2") handles codegen regardless of -mavx2; the dispatch layer
// gates calls behind vc_has_avx2() at runtime to prevent illegal instructions.
#if defined(__AVX2__) || (defined(__clang__) && __has_attribute(target))

#include <immintrin.h>

__attribute__((target("avx2")))
int32_t vc_x86_dot_int8(const int8_t* VC_RESTRICT a, const int8_t* VC_RESTRICT b, size_t lanes) {
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();

    size_t i = 0;
    // Process 32 int8 elements per iteration (2 x 16 sign-extended to int16)
    for (; vc_int8_block_fits(lanes, i, 32); i += 32) {
        // Load 16 bytes into 128-bit, sign-extend to 16 int16 in 256-bit
        __m128i a_lo_128 = _mm_loadu_si128((const __m128i*)(a + i));
        __m128i a_hi_128 = _mm_loadu_si128((const __m128i*)(a + i + 16));
        __m128i b_lo_128 = _mm_loadu_si128((const __m128i*)(b + i));
        __m128i b_hi_128 = _mm_loadu_si128((const __m128i*)(b + i + 16));

        __m256i a_lo = _mm256_cvtepi8_epi16(a_lo_128);
        __m256i a_hi = _mm256_cvtepi8_epi16(a_hi_128);
        __m256i b_lo = _mm256_cvtepi8_epi16(b_lo_128);
        __m256i b_hi = _mm256_cvtepi8_epi16(b_hi_128);

        // madd_epi16: multiply int16 pairs, horizontal add adjacent pairs to int32
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(a_lo, b_lo));
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(a_hi, b_hi));
    }

    // Handle 16-byte remainder
    for (; vc_int8_block_fits(lanes, i, 16); i += 16) {
        __m128i a_128 = _mm_loadu_si128((const __m128i*)(a + i));
        __m128i b_128 = _mm_loadu_si128((const __m128i*)(b + i));
        __m256i a_wide = _mm256_cvtepi8_epi16(a_128);
        __m256i b_wide = _mm256_cvtepi8_epi16(b_128);
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(a_wide, b_wide));
    }

    // Horizontal sum of 8 int32 lanes
    __m256i sum = _mm256_add_epi32(acc0, acc1);
    __m128i sum_lo = _mm256_castsi256_si128(sum);
    __m128i sum_hi = _mm256_extracti128_si256(sum, 1);
    __m128i sum128 = _mm_add_epi32(sum_lo, sum_hi);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    uint32_t bits = (uint32_t)_mm_extract_epi32(sum128, 0);   // int -> uint32_t is defined (modulo 2^32)

    // Scalar tail for remaining elements, unsigned
    for (; i < lanes; i++) {
        bits += vc_int8_product_bits(a[i], b[i]);
    }

    return vc_int8_accum_to_int32(bits);
}

#else

// Fallback if target attribute not available — delegate to scalar
int32_t vc_x86_dot_int8(const int8_t* VC_RESTRICT a, const int8_t* VC_RESTRICT b, size_t lanes) {
    extern int32_t vc_scalar_dot_int8(const int8_t* a, const int8_t* b, size_t lanes);
    return vc_scalar_dot_int8(a, b, lanes);
}

#endif // __AVX2__ || __has_attribute

#endif // __x86_64__
