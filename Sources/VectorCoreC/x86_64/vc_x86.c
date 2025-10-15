// x86_64 stubs — call into scalar for now; replace with AVX2/AVX-512 implementations
#include "VectorCoreC.h"
#include <stddef.h>
#include <stdint.h>

// Scalar references
extern float vc_scalar_dot_fp32_512(const float* a, const float* b);
extern float vc_scalar_l2sq_fp32_512(const float* a, const float* b);
extern void  vc_scalar_range_l2sq_fp32_512(const float* q, const float* base, size_t strideFloats, size_t start, size_t end, float* out);

#if defined(__x86_64__)

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

#endif // __x86_64__

