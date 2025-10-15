// Public API dispatchers choose the best available implementation
#include "VectorCoreC.h"
#include <stddef.h>
#include <stdint.h>

// Scalar references
extern float vc_scalar_dot_fp32_512(const float* a, const float* b);
extern float vc_scalar_l2sq_fp32_512(const float* a, const float* b);
extern void  vc_scalar_range_l2sq_fp32_512(const float* q, const float* base, size_t strideFloats, size_t start, size_t end, float* out);
extern int32_t vc_scalar_dot_int8(const int8_t* a, const int8_t* b, size_t lanes);

// Arch-specific stubs (implemented only on matching arch)
#if defined(__aarch64__)
extern float vc_arm64_dot_fp32_512(const float* a, const float* b);
extern float vc_arm64_l2sq_fp32_512(const float* a, const float* b);
extern void  vc_arm64_range_l2sq_fp32_512(const float* q, const float* base, size_t strideFloats, size_t start, size_t end, float* out);
#endif

#if defined(__x86_64__)
extern float vc_x86_dot_fp32_512(const float* a, const float* b);
extern float vc_x86_l2sq_fp32_512(const float* a, const float* b);
extern void  vc_x86_range_l2sq_fp32_512(const float* q, const float* base, size_t strideFloats, size_t start, size_t end, float* out);
#endif

// --- Public API ---

float vc_dot_fp32_512(const float* a, const float* b) {
#if defined(__aarch64__)
    // Prefer dotprod/NEON when available (stub calls into arch variant)
    if (vc_has_dotprod() || vc_has_neon()) {
        return vc_arm64_dot_fp32_512(a, b);
    }
#elif defined(__x86_64__)
    if (vc_has_avx2() || vc_has_avx512f()) {
        return vc_x86_dot_fp32_512(a, b);
    }
#endif
    return vc_scalar_dot_fp32_512(a, b);
}

float vc_l2sq_fp32_512(const float* a, const float* b) {
#if defined(__aarch64__)
    if (vc_has_dotprod() || vc_has_neon()) {
        return vc_arm64_l2sq_fp32_512(a, b);
    }
#elif defined(__x86_64__)
    if (vc_has_avx2() || vc_has_avx512f()) {
        return vc_x86_l2sq_fp32_512(a, b);
    }
#endif
    return vc_scalar_l2sq_fp32_512(a, b);
}

void vc_range_l2sq_fp32_512(
    const float* q,
    const float* base,
    size_t strideFloats,
    size_t start,
    size_t end,
    float* out
) {
#if defined(__aarch64__)
    if (vc_has_dotprod() || vc_has_neon()) {
        vc_arm64_range_l2sq_fp32_512(q, base, strideFloats, start, end, out);
        return;
    }
#elif defined(__x86_64__)
    if (vc_has_avx2() || vc_has_avx512f()) {
        vc_x86_range_l2sq_fp32_512(q, base, strideFloats, start, end, out);
        return;
    }
#endif
    vc_scalar_range_l2sq_fp32_512(q, base, strideFloats, start, end, out);
}

int32_t vc_dot_int8(const int8_t* a, const int8_t* b, size_t lanes) {
    // For now, call scalar; arch variants will replace later.
    // Runtime detection for SDOT/VNNI will route to optimized backends once available.
    return vc_scalar_dot_int8(a, b, lanes);
}

