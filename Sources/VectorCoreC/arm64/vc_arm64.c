// arm64 implementations — NEON intrinsics (FP32) and SDOT or NEON (INT8), selected at compile time.
#include "VectorCoreC.h"
#include "../common/vc_int8_accum.h"
#include <stddef.h>
#include <stdint.h>

// Scalar references (fallback)
extern float vc_scalar_dot_fp32_512(const float* a, const float* b);
extern float vc_scalar_l2sq_fp32_512(const float* a, const float* b);
extern void  vc_scalar_range_l2sq_fp32_512(const float* q, const float* base, size_t strideFloats, size_t start, size_t end, float* out);

#if defined(__aarch64__)

#include <arm_neon.h>

// --- FP32 Dot Product: 4-accumulator NEON FMA, stride-16 ---
// Processes 16 floats per iteration (4 NEON registers x 4 floats) into 4 independent accumulator chains, which
// are added pairwise at the end. The summation order differs from the scalar loop, so results can differ from it
// in the last bits.
float vc_arm64_dot_fp32_512(const float* VC_RESTRICT a, const float* VC_RESTRICT b) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    for (size_t i = 0; i < 512; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);

        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t b3 = vld1q_f32(b + i + 12);

        acc0 = vfmaq_f32(acc0, a0, b0);
        acc1 = vfmaq_f32(acc1, a1, b1);
        acc2 = vfmaq_f32(acc2, a2, b2);
        acc3 = vfmaq_f32(acc3, a3, b3);
    }

    // Tree reduction
    float32x4_t sum01 = vaddq_f32(acc0, acc1);
    float32x4_t sum23 = vaddq_f32(acc2, acc3);
    float32x4_t sum   = vaddq_f32(sum01, sum23);
    return vaddvq_f32(sum);
}

// --- FP32 L2^2: Same 4-accumulator pattern with difference computation ---
float vc_arm64_l2sq_fp32_512(const float* VC_RESTRICT a, const float* VC_RESTRICT b) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    for (size_t i = 0; i < 512; i += 16) {
        float32x4_t d0 = vsubq_f32(vld1q_f32(a + i),      vld1q_f32(b + i));
        float32x4_t d1 = vsubq_f32(vld1q_f32(a + i + 4),  vld1q_f32(b + i + 4));
        float32x4_t d2 = vsubq_f32(vld1q_f32(a + i + 8),  vld1q_f32(b + i + 8));
        float32x4_t d3 = vsubq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));

        acc0 = vfmaq_f32(acc0, d0, d0);
        acc1 = vfmaq_f32(acc1, d1, d1);
        acc2 = vfmaq_f32(acc2, d2, d2);
        acc3 = vfmaq_f32(acc3, d3, d3);
    }

    float32x4_t sum01 = vaddq_f32(acc0, acc1);
    float32x4_t sum23 = vaddq_f32(acc2, acc3);
    float32x4_t sum   = vaddq_f32(sum01, sum23);
    return vaddvq_f32(sum);
}

// --- Range L2^2: Stride-aware batch computation ---
void vc_arm64_range_l2sq_fp32_512(
    const float* VC_RESTRICT q,
    const float* VC_RESTRICT base,
    size_t strideFloats,
    size_t start,
    size_t end,
    float* VC_RESTRICT out
) {
    size_t idx = 0;
    for (size_t row = start; row < end; ++row) {
        const float* bRow = base + row * strideFloats;
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);

        for (size_t i = 0; i < 512; i += 16) {
            float32x4_t d0 = vsubq_f32(vld1q_f32(q + i),      vld1q_f32(bRow + i));
            float32x4_t d1 = vsubq_f32(vld1q_f32(q + i + 4),  vld1q_f32(bRow + i + 4));
            float32x4_t d2 = vsubq_f32(vld1q_f32(q + i + 8),  vld1q_f32(bRow + i + 8));
            float32x4_t d3 = vsubq_f32(vld1q_f32(q + i + 12), vld1q_f32(bRow + i + 12));

            acc0 = vfmaq_f32(acc0, d0, d0);
            acc1 = vfmaq_f32(acc1, d1, d1);
            acc2 = vfmaq_f32(acc2, d2, d2);
            acc3 = vfmaq_f32(acc3, d3, d3);
        }

        float32x4_t sum01 = vaddq_f32(acc0, acc1);
        float32x4_t sum23 = vaddq_f32(acc2, acc3);
        float32x4_t sum   = vaddq_f32(sum01, sum23);
        out[idx++] = vaddvq_f32(sum);
    }
}

// --- INT8 Dot Product ---
// Two compile-time paths: SDOT when the compile target defines __ARM_FEATURE_DOTPROD (clang defines it for the
// arm64 macOS 14 target this package builds; vc_has_dotprod() reports the same macro, not a runtime probe), and a
// vmull/vpadal NEON fallback otherwise. Both follow the vc_dot_int8 law (VectorCoreC.h): SIMD integer lane
// additions wrap modulo 2^32, the horizontal combination and the scalar tail are unsigned, and the final pattern
// is mapped by vc_int8_accum_to_int32. Block loops test the remaining count (vc_int8_block_fits), so i <= lanes
// holds throughout.

#if defined(__ARM_FEATURE_DOTPROD)

// SDOT path: each vdotq_s32 adds 16 signed int8 products into 4 int32 lanes; 4 independent accumulators.
static int32_t vc_arm64_dot_int8_sdot(const int8_t* VC_RESTRICT a, const int8_t* VC_RESTRICT b, size_t lanes) {
    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);
    int32x4_t acc2 = vdupq_n_s32(0);
    int32x4_t acc3 = vdupq_n_s32(0);

    size_t i = 0;
    // Main loop: 64 int8 elements per iteration (4 x 16-element SDOT)
    for (; vc_int8_block_fits(lanes, i, 64); i += 64) {
        int8x16_t va0 = vld1q_s8(a + i);
        int8x16_t va1 = vld1q_s8(a + i + 16);
        int8x16_t va2 = vld1q_s8(a + i + 32);
        int8x16_t va3 = vld1q_s8(a + i + 48);

        int8x16_t vb0 = vld1q_s8(b + i);
        int8x16_t vb1 = vld1q_s8(b + i + 16);
        int8x16_t vb2 = vld1q_s8(b + i + 32);
        int8x16_t vb3 = vld1q_s8(b + i + 48);

        acc0 = vdotq_s32(acc0, va0, vb0);
        acc1 = vdotq_s32(acc1, va1, vb1);
        acc2 = vdotq_s32(acc2, va2, vb2);
        acc3 = vdotq_s32(acc3, va3, vb3);
    }

    // Handle 16-byte remainder chunks
    for (; vc_int8_block_fits(lanes, i, 16); i += 16) {
        int8x16_t va = vld1q_s8(a + i);
        int8x16_t vb = vld1q_s8(b + i);
        acc0 = vdotq_s32(acc0, va, vb);
    }

    // Scalar tail for remaining elements (< 16), unsigned
    uint32_t tail = 0;
    for (; i < lanes; i++) {
        tail += vc_int8_product_bits(a[i], b[i]);
    }

    // Tree reduction in unsigned lanes (modulo 2^32), then the unsigned final addition and the checked mapping
    uint32x4_t sum01 = vaddq_u32(vreinterpretq_u32_s32(acc0), vreinterpretq_u32_s32(acc1));
    uint32x4_t sum23 = vaddq_u32(vreinterpretq_u32_s32(acc2), vreinterpretq_u32_s32(acc3));
    uint32x4_t sum   = vaddq_u32(sum01, sum23);
    return vc_int8_accum_to_int32(vaddvq_u32(sum) + tail);
}

#else

// NEON fallback, compiled when the target does not define __ARM_FEATURE_DOTPROD (for example `-march=armv8-a`, and
// per `clang -dM` the default arm64 iOS 17, tvOS 17 and visionOS 1 targets).
// Uses vmull_s8 (8 int8 pairs -> 8 int16 products; |product| <= 16384 fits int16) + vpadalq_s16, which widens
// each pair of int16 products to int32 before adding it to the int32 lanes.
static int32_t vc_arm64_dot_int8_neon(const int8_t* VC_RESTRICT a, const int8_t* VC_RESTRICT b, size_t lanes) {
    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);

    size_t i = 0;
    // Process 16 bytes per iteration
    for (; vc_int8_block_fits(lanes, i, 16); i += 16) {
        int8x16_t va = vld1q_s8(a + i);
        int8x16_t vb = vld1q_s8(b + i);

        // Split into low/high halves, multiply to int16
        int16x8_t prod_lo = vmull_s8(vget_low_s8(va), vget_low_s8(vb));
        int16x8_t prod_hi = vmull_s8(vget_high_s8(va), vget_high_s8(vb));

        // Pairwise add int16 pairs into int32 accumulators
        acc0 = vpadalq_s16(acc0, prod_lo);
        acc1 = vpadalq_s16(acc1, prod_hi);
    }

    // Scalar tail, unsigned
    uint32_t tail = 0;
    for (; i < lanes; i++) {
        tail += vc_int8_product_bits(a[i], b[i]);
    }

    // Unsigned lane combination (modulo 2^32), unsigned final addition, checked mapping
    uint32x4_t sum = vaddq_u32(vreinterpretq_u32_s32(acc0), vreinterpretq_u32_s32(acc1));
    return vc_int8_accum_to_int32(vaddvq_u32(sum) + tail);
}

#endif // __ARM_FEATURE_DOTPROD

// Internal arm64 INT8 entry (linker-visible, not declared in a public header; D02 RD-C5) — compile-time
// selection between SDOT and the vmull fallback. The public dispatcher vc_dot_int8 calls it on arm64.
int32_t vc_arm64_dot_int8(const int8_t* VC_RESTRICT a, const int8_t* VC_RESTRICT b, size_t lanes) {
#if defined(__ARM_FEATURE_DOTPROD)
    return vc_arm64_dot_int8_sdot(a, b, lanes);
#else
    return vc_arm64_dot_int8_neon(a, b, lanes);
#endif
}

#endif // __aarch64__
