// arm64 stubs — call into scalar for now; replace with NEON/dotprod implementations
#include "VectorCoreC.h"
#include <stddef.h>
#include <stdint.h>

// Scalar references
extern float vc_scalar_dot_fp32_512(const float* a, const float* b);
extern float vc_scalar_l2sq_fp32_512(const float* a, const float* b);
extern void  vc_scalar_range_l2sq_fp32_512(const float* q, const float* base, size_t strideFloats, size_t start, size_t end, float* out);

#if defined(__aarch64__)

float vc_arm64_dot_fp32_512(const float* a, const float* b) {
    return vc_scalar_dot_fp32_512(a, b);
}

float vc_arm64_l2sq_fp32_512(const float* a, const float* b) {
    return vc_scalar_l2sq_fp32_512(a, b);
}

void vc_arm64_range_l2sq_fp32_512(
    const float* q,
    const float* base,
    size_t strideFloats,
    size_t start,
    size_t end,
    float* out
) {
    vc_scalar_range_l2sq_fp32_512(q, base, strideFloats, start, end, out);
}

#endif // __aarch64__

