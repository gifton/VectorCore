// Scalar implementations: the dispatchers' fallback (x86_64 without AVX2) and the x86_64 FP32 entries.
#include "VectorCoreC.h"
#include "vc_int8_accum.h"
#include <stddef.h>
#include <stdint.h>

// Internal symbols: linker-visible (default visibility) but declared in no public header, so they are not
// supported API (D02 RD-C5). Referenced by the dispatchers and the architecture files.
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

// INT8 dot product under the vc_dot_int8 law: unsigned accumulation (wraps modulo 2^32), then the checked
// mapping into the int32_t range (vc_int8_accum.h). lanes == 0 reads nothing.
int32_t vc_scalar_dot_int8(const int8_t* a, const int8_t* b, size_t lanes) {
    uint32_t acc = 0;
    for (size_t i = 0; i < lanes; ++i) {
        acc += vc_int8_product_bits(a[i], b[i]);
    }
    return vc_int8_accum_to_int32(acc);
}

