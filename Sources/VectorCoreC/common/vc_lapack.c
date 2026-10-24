// vc_lapack.c — LAPACK shim implementation.
//
// Apple platforms: routes to Accelerate's modern LAPACK interface
// (ACCELERATE_NEW_LAPACK). Requires macOS 13.3+/iOS 16.4+ at the SDK level;
// the package's platform floor (macOS 14 / iOS 17) already exceeds this.
//
// Other platforms: stubs returning VC_LAPACK_UNAVAILABLE. The Swift
// SwiftLinearAlgebraProvider is the cross-platform fallback; this shim is
// only the fast path. (A future system-LAPACK/OpenBLAS hookup for Linux
// would land here behind the same five entry points.)

#include "../include/vc_lapack.h"

#include <math.h>
#include <stdlib.h>
#include "vc_lapack_workspace.h"

#if defined(__APPLE__)
  // ACCELERATE_NEW_LAPACK is defined via cSettings in Package.swift: under
  // clang modules (SPM default for C targets) the include below is a module
  // import, so an in-source #define would not reach Accelerate's headers.
  // The guarded define keeps plain textual inclusion working too.
  // ACCELERATE_LAPACK_ILP64 intentionally NOT defined — see vc_lapack.h.
  #if !defined(ACCELERATE_NEW_LAPACK)
    #define ACCELERATE_NEW_LAPACK 1
  #endif
  #include <Accelerate/Accelerate.h>
  #define VC_HAS_LAPACK 1
#endif

int32_t vc_lapack_available(void) {
#if defined(VC_HAS_LAPACK)
    return 1;
#else
    return 0;
#endif
}

#if defined(VC_HAS_LAPACK)

_Static_assert(sizeof(__LAPACK_int) == sizeof(int32_t), "VectorCore requires LP64 LAPACK");

int32_t vc_lapack_sgeqrf(int32_t m, int32_t n, float *a, int32_t lda, float *tau) {
    if (m < 0) return -1;
    if (n < 0) return -2;
    if (lda < (m > 1 ? m : 1)) return -4;
    if (m == 0 || n == 0) return 0;
    if (!vc_la_matrix_bytes(lda, n)) return VC_LAPACK_WORKSPACE_OVERFLOW;
    __LAPACK_int mm = m, nn = n, ldaa = lda, info = 0;

    // Workspace query (lwork = -1)
    float query = 0.0f;
    __LAPACK_int lwork = -1;
    sgeqrf_(&mm, &nn, a, &ldaa, tau, &query, &lwork, &info);
    if (info != 0) { return (int32_t)info; }

    size_t work_bytes;
    int32_t status = vc_la_query(query, n > 1 ? n : 1, &lwork, &work_bytes);
    if (status != 0) return status;
    float *work = (float *)malloc(work_bytes);
    if (work == NULL) { return VC_LAPACK_UNAVAILABLE; }

    sgeqrf_(&mm, &nn, a, &ldaa, tau, work, &lwork, &info);
    free(work);
    return (int32_t)info;
}

int32_t vc_lapack_sorgqr(int32_t m, int32_t n, int32_t k,
                         float *a, int32_t lda, const float *tau) {
    if (m < 0) return -1;
    if (n < 0 || n > m) return -2;
    if (k < 0 || k > n) return -3;
    if (lda < (m > 1 ? m : 1)) return -5;
    if (n == 0) return 0;
    if (!vc_la_matrix_bytes(lda, n)) return VC_LAPACK_WORKSPACE_OVERFLOW;
    __LAPACK_int mm = m, nn = n, kk = k, ldaa = lda, info = 0;

    float query = 0.0f;
    __LAPACK_int lwork = -1;
    sorgqr_(&mm, &nn, &kk, a, &ldaa, tau, &query, &lwork, &info);
    if (info != 0) { return (int32_t)info; }

    size_t work_bytes;
    int32_t status = vc_la_query(query, n > 1 ? n : 1, &lwork, &work_bytes);
    if (status != 0) return status;
    float *work = (float *)malloc(work_bytes);
    if (work == NULL) { return VC_LAPACK_UNAVAILABLE; }

    sorgqr_(&mm, &nn, &kk, a, &ldaa, tau, work, &lwork, &info);
    free(work);
    return (int32_t)info;
}

int32_t vc_lapack_sgesdd(int32_t m, int32_t n,
                         float *a, int32_t lda,
                         float *s,
                         float *u, int32_t ldu,
                         float *vt, int32_t ldvt) {
    // Positions include the fixed Fortran JOBZ argument at position 1.
    if (m < 0) return -2;
    if (n < 0) return -3;
    if (lda < (m > 1 ? m : 1)) return -5;
    if (ldu < (m > 1 ? m : 1)) return -8;
    const int32_t k = m < n ? m : n;
    if (ldvt < (k > 1 ? k : 1)) return -10;
    if (m == 0 || n == 0) return 0;
    int32_t minimum, integer_count;
    int32_t status = vc_la_svd_work(k, &minimum, &integer_count);
    size_t integer_bytes;
    if (status != 0) return status;
    if (!vc_la_matrix_bytes(lda, n) || !vc_la_matrix_bytes(ldu, k) ||
        !vc_la_matrix_bytes(ldvt, n) ||
        !vc_la_bytes((size_t)integer_count, sizeof(__LAPACK_int), &integer_bytes))
        return VC_LAPACK_WORKSPACE_OVERFLOW;
    __LAPACK_int mm = m, nn = n, ldaa = lda, lduu = ldu, ldvtt = ldvt, info = 0;
    const char jobz = 'S';  // thin factors only

    __LAPACK_int *iwork = (__LAPACK_int *)malloc(integer_bytes);
    if (iwork == NULL) { return VC_LAPACK_UNAVAILABLE; }

    float query = 0.0f;
    __LAPACK_int lwork = -1;
    sgesdd_(&jobz, &mm, &nn, a, &ldaa, s, u, &lduu, vt, &ldvtt,
            &query, &lwork, iwork, &info);
    if (info != 0) { free(iwork); return (int32_t)info; }

    size_t work_bytes;
    status = vc_la_query(query, minimum, &lwork, &work_bytes);
    if (status != 0) { free(iwork); return status; }
    float *work = (float *)malloc(work_bytes);
    if (work == NULL) { free(iwork); return VC_LAPACK_UNAVAILABLE; }

    sgesdd_(&jobz, &mm, &nn, a, &ldaa, s, u, &lduu, vt, &ldvtt,
            work, &lwork, iwork, &info);
    free(work);
    free(iwork);
    return (int32_t)info;
}

int32_t vc_lapack_ssyevd(char jobz, char uplo, int32_t n,
                         float *a, int32_t lda, float *w) {
    if (jobz != 'V' && jobz != 'v' && jobz != 'N' && jobz != 'n') return -1;
    if (uplo != 'L' && uplo != 'l' && uplo != 'U' && uplo != 'u') return -2;
    if (n < 0) return -3;
    if (lda < (n > 1 ? n : 1)) return -5;
    if (n == 0) return 0;
    int32_t minimum, integer_minimum;
    int32_t status = vc_la_eigen_work(n, jobz == 'V' || jobz == 'v', &minimum, &integer_minimum);
    if (status != 0) return status;
    if (!vc_la_matrix_bytes(lda, n)) return VC_LAPACK_WORKSPACE_OVERFLOW;
    __LAPACK_int nn = n, ldaa = lda, info = 0;

    // Dual workspace query: optimal lwork (float) and liwork (int).
    float fquery = 0.0f;
    __LAPACK_int iquery = 0;
    __LAPACK_int lwork = -1, liwork = -1;
    ssyevd_(&jobz, &uplo, &nn, a, &ldaa, w,
            &fquery, &lwork, &iquery, &liwork, &info);
    if (info != 0) { return (int32_t)info; }

    size_t work_bytes, integer_bytes;
    status = vc_la_query(fquery, minimum, &lwork, &work_bytes);
    if (status != 0) return status;
    status = vc_la_integer_query(iquery, integer_minimum, &integer_bytes);
    if (status != 0) return status;
    liwork = (__LAPACK_int)(integer_bytes / sizeof(__LAPACK_int));  // allocated max(iquery, minimum)
    float *work = (float *)malloc(work_bytes);
    __LAPACK_int *iwork = (__LAPACK_int *)malloc(integer_bytes);
    if (work == NULL || iwork == NULL) {
        free(work);
        free(iwork);
        return VC_LAPACK_UNAVAILABLE;
    }

    ssyevd_(&jobz, &uplo, &nn, a, &ldaa, w,
            work, &lwork, iwork, &liwork, &info);
    free(work);
    free(iwork);
    return (int32_t)info;
}

#else  // !VC_HAS_LAPACK — stubs

int32_t vc_lapack_sgeqrf(int32_t m, int32_t n, float *a, int32_t lda, float *tau) {
    (void)m; (void)n; (void)a; (void)lda; (void)tau;
    return VC_LAPACK_UNAVAILABLE;
}

int32_t vc_lapack_sorgqr(int32_t m, int32_t n, int32_t k,
                         float *a, int32_t lda, const float *tau) {
    (void)m; (void)n; (void)k; (void)a; (void)lda; (void)tau;
    return VC_LAPACK_UNAVAILABLE;
}

int32_t vc_lapack_sgesdd(int32_t m, int32_t n, float *a, int32_t lda, float *s,
                         float *u, int32_t ldu, float *vt, int32_t ldvt) {
    (void)m; (void)n; (void)a; (void)lda; (void)s;
    (void)u; (void)ldu; (void)vt; (void)ldvt;
    return VC_LAPACK_UNAVAILABLE;
}

int32_t vc_lapack_ssyevd(char jobz, char uplo, int32_t n,
                         float *a, int32_t lda, float *w) {
    (void)jobz; (void)uplo; (void)n; (void)a; (void)lda; (void)w;
    return VC_LAPACK_UNAVAILABLE;
}

#endif
