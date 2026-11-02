// D02 evidence fixture: an ordinary C consumer of the retained public VectorCoreC surface (compiled as C11 and,
// through d02_consumer.cpp, as C++17). It includes only the public umbrella header and calls every retained
// function: the four feature queries, the three FP32 512-wide kernels, the generic INT8 dot product and the five
// LAPACK shim entries (zero-size calls, which the shim documents as touching no pointer).
// Exit 0 and print D02_C_CONSUMER_COMPLETE iff every call returns its documented value.
#include "VectorCoreC.h"
#include <stdio.h>
#include <stdlib.h>

#if defined(__cplusplus)
#define D02_CONSUMER_LANGUAGE "C++"
#else
#define D02_CONSUMER_LANGUAGE "C"
#endif

int main(void) {
    int features[4] = {vc_has_avx2(), vc_has_avx512f(), vc_has_neon(), vc_has_dotprod()};
    for (int k = 0; k < 4; ++k) {
        if (features[k] != 0 && features[k] != 1) { printf("feature query %d returned %d\n", k, features[k]); return 1; }
    }
    float *a = (float *)malloc(sizeof(float) * 512), *b = (float *)malloc(sizeof(float) * 512);
    float out[2] = {-1.0f, -1.0f};
    if (!a || !b) return 3;
    for (int i = 0; i < 512; ++i) { a[i] = 1.0f; b[i] = 2.0f; }
    // Small integers: every partial sum is an exactly representable float, so the results are exact.
    if (vc_dot_fp32_512(a, b) != 1024.0f) { printf("dot\n"); return 1; }
    if (vc_l2sq_fp32_512(a, b) != 512.0f) { printf("l2sq\n"); return 1; }
    vc_range_l2sq_fp32_512(a, b, 512, 0, 1, out);   // one row; the result lands at out[0], out[1] is untouched
    if (out[0] != 512.0f || out[1] != -1.0f) { printf("range\n"); return 1; }
    int8_t x[3] = {-128, 127, 5}, y[3] = {-128, -128, 7};
    if (vc_dot_int8(x, y, 3) != 16384 - 16256 + 35) { printf("int8\n"); return 1; }
    if (vc_dot_int8(NULL, NULL, 0) != 0) { printf("int8 zero lanes\n"); return 1; }
    int32_t lapack = vc_lapack_available();
    if (lapack != 0 && lapack != 1) { printf("lapack_available %d\n", lapack); return 1; }
    int32_t infos[4] = {
        vc_lapack_sgeqrf(0, 0, NULL, 1, NULL),
        vc_lapack_sorgqr(0, 0, 0, NULL, 1, NULL),
        vc_lapack_sgesdd(0, 0, NULL, 1, NULL, NULL, 1, NULL, 1),
        vc_lapack_ssyevd('N', 'L', 0, NULL, 1, NULL),
    };
    for (int k = 0; k < 4; ++k) {
        int32_t want = lapack ? 0 : VC_LAPACK_UNAVAILABLE;
        if (infos[k] != want) { printf("lapack entry %d returned %d\n", k, infos[k]); return 1; }
    }
    free(a); free(b);
    printf("features avx2=%d avx512f=%d neon=%d dotprod=%d lapack=%d\n", features[0], features[1], features[2],
           features[3], lapack);
    printf("D02_C_CONSUMER_COMPLETE %s\n", D02_CONSUMER_LANGUAGE);
    return 0;
}
