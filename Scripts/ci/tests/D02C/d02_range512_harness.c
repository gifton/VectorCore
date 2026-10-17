// D02 evidence fixture (not a CI gate; run by Scripts/ci/tests/D02CContract.py, normally under ASan/UBSan).
//
// Exercises the documented vc_range_l2sq_fp32_512 contract through one entry of the ACTUAL objects:
// compact output out[0 ..< end-start] independent of start, padded stride, nonzero start, offset output with NaN
// guards on both sides, an untouched suffix, and an empty range with null pointers (no pointer access).
// Every buffer is an exact-size heap allocation, so AddressSanitizer reports any read or write outside it.
// Reference: a Double sum of (double)q[i] - (double)row[i] squared, with tolerance
//   tol = (gamma_514(2^-24) + 2 * gamma_514(2^-53)) * S + 512 * 2^-149,
// gamma_n(u) = n*u / (1 - n*u): 514 = 512 accumulations plus one rounding each for the difference and the square
// (any summation order, with or without FMA); the Double reference's own error and float underflow are added.
//
// usage: d02_range512_harness <dispatcher|scalar|arch>
#include "VectorCoreC.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void vc_scalar_range_l2sq_fp32_512(const float *q, const float *base, size_t strideFloats, size_t start,
                                          size_t end, float *out);
#if defined(__aarch64__)
extern void vc_arm64_range_l2sq_fp32_512(const float *q, const float *base, size_t strideFloats, size_t start,
                                         size_t end, float *out);
#endif
typedef void (*range_fn)(const float *, const float *, size_t, size_t, size_t, float *);

static double gamma_n(double n, double u) { return n * u / (1.0 - n * u); }

static unsigned long checks = 0, failures = 0;
#define CHECK(cond, ...) do { checks++; if (!(cond)) { failures++; printf("FAIL " __VA_ARGS__); printf("\n"); } } while (0)

static float next_value(uint32_t *s) {
    *s = *s * 1664525u + 1013904223u;
    return (float)((int32_t)(*s >> 8) - (1 << 23)) / (float)(1 << 23);   // in [-1, 1)
}

static void run_case(range_fn fn, size_t stride, size_t start, size_t end, size_t guard_suffix) {
    const size_t D = 512;
    size_t rows = end;                                   // rows 0 ..< end exist; only start ..< end are referenced
    float *base = malloc(sizeof(float) * (rows * stride + (rows == 0 ? 1 : 0)));
    float *q = malloc(sizeof(float) * D);
    size_t n = end - start;
    size_t out_len = 1 + n + guard_suffix + 1;            // leading guard, results, untouched suffix, trailing guard
    float *out_alloc = malloc(sizeof(float) * out_len);
    if (!base || !q || !out_alloc) { fprintf(stderr, "allocation failed\n"); exit(3); }
    uint32_t s = 12345u + (uint32_t)(stride * 7 + start * 13 + end);
    for (size_t i = 0; i < D; ++i) q[i] = next_value(&s);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t i = 0; i < stride; ++i) base[r * stride + i] = i < D ? next_value(&s) : NAN;   // NaN padding
    }
    for (size_t k = 0; k < out_len; ++k) out_alloc[k] = NAN;
    fn(q, base, stride, start, end, out_alloc + 1);
    CHECK(isnan(out_alloc[0]), "leading guard overwritten (stride %zu start %zu end %zu)", stride, start, end);
    for (size_t k = 1 + n; k < out_len; ++k) {
        CHECK(isnan(out_alloc[k]), "suffix/trailing guard %zu overwritten (stride %zu start %zu end %zu)", k, stride,
              start, end);
    }
    const double g = gamma_n(514, ldexp(1, -24)) + 2 * gamma_n(514, ldexp(1, -53));
    for (size_t j = 0; j < n; ++j) {
        const float *row = base + (start + j) * stride;
        double S = 0;
        for (size_t i = 0; i < D; ++i) { double d = (double)q[i] - (double)row[i]; S += d * d; }
        double tol = g * S + 512 * ldexp(1, -149);
        float got = out_alloc[1 + j];
        CHECK(isfinite(got) && fabs((double)got - S) <= tol, "row %zu: got %.9g want %.17g tol %.3g", start + j,
              (double)got, S, tol);
    }
    printf("RANGE stride=%zu start=%zu end=%zu results=%zu\n", stride, start, end, n);
    free(base); free(q); free(out_alloc);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <dispatcher|scalar|arch>\n", argv[0]); return 2; }
    range_fn fn = NULL;
    if (!strcmp(argv[1], "dispatcher")) fn = vc_range_l2sq_fp32_512;
    else if (!strcmp(argv[1], "scalar")) fn = vc_scalar_range_l2sq_fp32_512;
#if defined(__aarch64__)
    else if (!strcmp(argv[1], "arch")) fn = vc_arm64_range_l2sq_fp32_512;
#endif
    if (!fn) { fprintf(stderr, "entry %s not available in this build\n", argv[1]); return 2; }
    printf("ENTRY %s\n", argv[1]);
    // Valid empty range with null pointers: no pointer access.
    fn(NULL, NULL, 512, 7, 7, NULL);
    fn(NULL, NULL, 512, 0, 0, NULL);
    checks += 2;
    run_case(fn, 512, 0, 3, 2);      // dense stride, start 0
    run_case(fn, 512, 5, 9, 3);      // nonzero start: results land at out[0..<4]
    run_case(fn, 520, 3, 8, 1);      // padded stride (NaN padding never read into results)
    run_case(fn, 1031, 1, 2, 0);     // odd padded stride, one row
    printf("RANGE_TESTS checks=%lu failures=%lu\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
