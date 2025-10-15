// Scalar reference implementations (used for fallback and validation)
#include "VectorCoreC.h"
#include <stddef.h>
#include <stdint.h>

// Internal (non-exported) symbols; referenced by dispatchers and arch variants
float vc_scalar_dot_fp32_512(const float* a, const float* b) {
    float acc = 0.0f;
    for (size_t i = 0; i < 512; ++i) {
        acc += a[i] * b[i];
    }
    return acc;
}

float vc_scalar_l2sq_fp32_512(const float* a, const float* b) {
    float acc = 0.0f;
    for (size_t i = 0; i < 512; ++i) {
        float d = a[i] - b[i];
        acc += d * d;
    }
    return acc;
}

void vc_scalar_range_l2sq_fp32_512(
    const float* q,
    const float* base,
    size_t strideFloats,
    size_t start,
    size_t end,
    float* out
) {
    const size_t D = 512;
    size_t idx = 0;
    for (size_t row = start; row < end; ++row) {
        const float* bRow = base + row * strideFloats;
        float acc = 0.0f;
        for (size_t i = 0; i < D; ++i) {
            float d = q[i] - bRow[i];
            acc += d * d;
        }
        out[idx++] = acc;
    }
}

int32_t vc_scalar_dot_int8(const int8_t* a, const int8_t* b, size_t lanes) {
    int32_t acc = 0;
    for (size_t i = 0; i < lanes; ++i) {
        acc += (int32_t)a[i] * (int32_t)b[i];
    }
    return acc;
}

