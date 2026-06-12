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

// Helper: convert a float workspace-size query result (work[0]) to an int
// element count. Must round UP with margin: above 2^24, Float spacing
// exceeds 1 and the reported value may sit below the true integer, and
// Accelerate's LAPACK (3.9.x) predates the SROUNDUP_LWORK round-up
// guarantee introduced in LAPACK 3.10. Undershooting lwork yields a clean
// info=-N argument error, but only at panel sizes tests never reach —
// so over-allocate slightly (1/64th) instead.
static int32_t vc__lwork_from_query(float q) {
    int64_t r = (int64_t)ceilf(q);
    r += r / 64 + 1;
    if (r > INT32_MAX) { r = INT32_MAX; }
    return (int32_t)r;
}

int32_t vc_lapack_sgeqrf(int32_t m, int32_t n, float *a, int32_t lda, float *tau) {
    __LAPACK_int mm = m, nn = n, ldaa = lda, info = 0;

    // Workspace query (lwork = -1)
    float query = 0.0f;
    __LAPACK_int lwork = -1;
    sgeqrf_(&mm, &nn, a, &ldaa, tau, &query, &lwork, &info);
    if (info != 0) { return (int32_t)info; }

    lwork = vc__lwork_from_query(query);
    float *work = (float *)malloc((size_t)lwork * sizeof(float));
    if (work == NULL) { return VC_LAPACK_UNAVAILABLE; }

    sgeqrf_(&mm, &nn, a, &ldaa, tau, work, &lwork, &info);
    free(work);
    return (int32_t)info;
}

int32_t vc_lapack_sorgqr(int32_t m, int32_t n, int32_t k,
                         float *a, int32_t lda, const float *tau) {
    __LAPACK_int mm = m, nn = n, kk = k, ldaa = lda, info = 0;

    float query = 0.0f;
    __LAPACK_int lwork = -1;
    sorgqr_(&mm, &nn, &kk, a, &ldaa, tau, &query, &lwork, &info);
    if (info != 0) { return (int32_t)info; }

    lwork = vc__lwork_from_query(query);
    float *work = (float *)malloc((size_t)lwork * sizeof(float));
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
    __LAPACK_int mm = m, nn = n, ldaa = lda, lduu = ldu, ldvtt = ldvt, info = 0;
    const char jobz = 'S';  // thin factors only

    __LAPACK_int minmn = (m < n) ? m : n;
    __LAPACK_int *iwork = (__LAPACK_int *)malloc((size_t)(8 * minmn) * sizeof(__LAPACK_int));
    if (iwork == NULL) { return VC_LAPACK_UNAVAILABLE; }

    float query = 0.0f;
    __LAPACK_int lwork = -1;
    sgesdd_(&jobz, &mm, &nn, a, &ldaa, s, u, &lduu, vt, &ldvtt,
            &query, &lwork, iwork, &info);
    if (info != 0) { free(iwork); return (int32_t)info; }

    lwork = vc__lwork_from_query(query);
    float *work = (float *)malloc((size_t)lwork * sizeof(float));
    if (work == NULL) { free(iwork); return VC_LAPACK_UNAVAILABLE; }

    sgesdd_(&jobz, &mm, &nn, a, &ldaa, s, u, &lduu, vt, &ldvtt,
            work, &lwork, iwork, &info);
    free(work);
    free(iwork);
    return (int32_t)info;
}

int32_t vc_lapack_ssyevd(char jobz, char uplo, int32_t n,
                         float *a, int32_t lda, float *w) {
    __LAPACK_int nn = n, ldaa = lda, info = 0;

    // Dual workspace query: optimal lwork (float) and liwork (int).
    float fquery = 0.0f;
    __LAPACK_int iquery = 0;
    __LAPACK_int lwork = -1, liwork = -1;
    ssyevd_(&jobz, &uplo, &nn, a, &ldaa, w,
            &fquery, &lwork, &iquery, &liwork, &info);
    if (info != 0) { return (int32_t)info; }

    lwork = vc__lwork_from_query(fquery);
    liwork = iquery;
    float *work = (float *)malloc((size_t)lwork * sizeof(float));
    __LAPACK_int *iwork = (__LAPACK_int *)malloc((size_t)liwork * sizeof(__LAPACK_int));
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
