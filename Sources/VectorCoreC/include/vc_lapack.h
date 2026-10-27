// vc_lapack.h — LAPACK shim for VectorCore dense linear algebra.
//
// Why a C shim instead of calling LAPACK from Swift directly:
//   1. Accelerate's modern LAPACK interface is gated behind the
//      ACCELERATE_NEW_LAPACK preprocessor macro, which must be defined
//      *before* <Accelerate/Accelerate.h> is processed. Swift cannot inject
//      C macros into the Clang importer without `unsafeFlags`, which would
//      make the package unconsumable as a dependency. A C translation unit
//      defines the macro locally — zero flags leak into Package.swift.
//   2. The shim owns LAPACK workspace negotiation (lwork=-1 probes, iwork
//      sizing), so the Swift side never touches raw Fortran calling
//      conventions.
//   3. Non-Apple platforms compile the same header against stub
//      implementations; availability is a runtime probe, mirroring the
//      vc_has_* CPU-feature pattern in VectorCoreC.h.
//
// Integer model: signed32 modern LP64; ILP64 is not selected. Intended for
// small covariance/sketch panels. Checked local counts and workspace lengths
// do not prove vendor-internal arithmetic safe for arbitrary extreme shapes.
// Sufficient preflight bounds can reject a shape whose vendor-specific tighter
// workspace path might fit. Representability does not promise allocation success.
//
// Matrix layout: ALL matrices are COLUMN-MAJOR (LAPACK native). Callers are
// responsible for layout; the Swift providers document this contract.

#pragma once

#include <stdint.h>

#if !defined(VC_EXPORT)
  #if defined(_WIN32)
    #define VC_EXPORT __declspec(dllexport)
  #else
    #define VC_EXPORT __attribute__((visibility("default")))
  #endif
#endif

#if defined(__cplusplus)
extern "C" {
#endif

// Returned for an unavailable backend OR local workspace allocation failure.
// Positive/negative values in the small range are LAPACK `info` codes:
//   0 = success; <0 = illegal argument at position -info; >0 = algorithm
//   failure (did not converge / not positive definite, routine-specific).
#define VC_LAPACK_UNAVAILABLE (-1000)
#define VC_LAPACK_INVALID_QUERY (-1001)
#define VC_LAPACK_WORKSPACE_OVERFLOW (-1002)

// Apple scalar validation returns Fortran argument positions (including hidden
// fixed JOBZ for SGESDD): GEQRF m:-1,n:-2,lda:-4; ORGQR m:-1,n:-2,k:-3,lda:-5;
// GESDD m:-2,n:-3,lda:-5,ldu:-8,ldvt:-10; SYEVD jobz:-1,uplo:-2,n:-3,lda:-5.
// Leading dimensions must be >=max(1, corresponding row count). ORGQR requires
// 0<=k<=n<=m. Scalar checks precede valid zero-size returns, which access no
// pointers and allocate nothing. Positive-size buffer capacities, initialization,
// and column-major layout remain the caller's unsafe preconditions. Non-Apple
// stubs always return UNAVAILABLE without touching buffers. Invalid query values
// return INVALID_QUERY; unrepresentable local counts return WORKSPACE_OVERFLOW.

/// 1 if a LAPACK backend is compiled in and callable, 0 otherwise.
VC_EXPORT int32_t vc_lapack_available(void);

/// QR factorization (SGEQRF): A (m×n, col-major, lda≥m) is overwritten with
/// R in the upper triangle and Householder reflectors below the diagonal.
/// tau must hold min(m,n) elements. Returns LAPACK info.
VC_EXPORT int32_t vc_lapack_sgeqrf(int32_t m, int32_t n,
                                   float *a, int32_t lda,
                                   float *tau);

/// Generate explicit Q (SORGQR) from SGEQRF output. On entry `a` holds the
/// reflectors (m×n as left by sgeqrf); on exit `a` is the orthonormal Q (m×n).
/// k is the number of reflectors (= tau length used), with 0<=k<=n<=m.
/// Returns LAPACK info.
VC_EXPORT int32_t vc_lapack_sorgqr(int32_t m, int32_t n, int32_t k,
                                   float *a, int32_t lda,
                                   const float *tau);

/// Thin SVD via divide-and-conquer (SGESDD, jobz='S').
/// A (m×n, col-major) is DESTROYED. With kk = min(m,n):
///   u:  m×kk col-major (ldu≥m)   s: kk singular values, descending
///   vt: kk×n col-major (ldvt≥kk)
/// Returns LAPACK info (>0 means the D&C algorithm failed to converge).
VC_EXPORT int32_t vc_lapack_sgesdd(int32_t m, int32_t n,
                                   float *a, int32_t lda,
                                   float *s,
                                   float *u, int32_t ldu,
                                   float *vt, int32_t ldvt);

/// Symmetric eigendecomposition via divide-and-conquer (SSYEVD).
/// jobz: 'V' = eigenvalues + eigenvectors, 'N' = eigenvalues only.
/// uplo: 'L'/'U' — which triangle of A is referenced.
/// A (n×n, col-major) is overwritten: for jobz='V' its columns become the
/// orthonormal eigenvectors. w receives the n eigenvalues in ASCENDING order.
/// Returns LAPACK info.
VC_EXPORT int32_t vc_lapack_ssyevd(char jobz, char uplo, int32_t n,
                                   float *a, int32_t lda,
                                   float *w);

#if defined(__cplusplus)
}
#endif
