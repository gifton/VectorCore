// Private allocation-free arithmetic, exercised by L01WorkspaceConsumer.c.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include "../include/vc_lapack.h"

static inline int vc_la_mul(size_t a, size_t b, size_t *out) {
    if (b != 0 && a > SIZE_MAX / b) return 0;
    *out = a * b;
    return 1;
}
static inline int vc_la_add(size_t a, size_t b, size_t *out) {
    if (a > SIZE_MAX - b) return 0;
    *out = a + b;
    return 1;
}
static inline int vc_la_bytes(size_t count, size_t width, size_t *out) {
    return width != 0 && vc_la_mul(count, width, out);
}
static inline int vc_la_length(size_t count, int32_t *out) {
    if (count == 0 || count > INT32_MAX) return 0;
    *out = (int32_t)count;
    return 1;
}

// Reference sufficient bounds, not a proof about vendor query internals. Each is an allocation FLOOR (the
// minimum passed to vc_la_query / vc_la_integer_query), never a validity threshold for the vendor's answer.
// Observed (T19LAPACKWorkspaceQueryTests): on some sgesdd path-5 shapes (R < floor(11k/6), k = min(M,N),
// R = max(M,N); in its sweep every one at k = 64, R <= 17 at k = 16, none at k = 8) Accelerate answers
// below 4k^2+7k, and LWORK = max(answer + headroom, floor) succeeds. Basis for not rejecting: reference
// LAPACK sgesdd.f's own JOBZ='S' path-5 minimum is MINWRK = 3k + max(R, 3k^2+4k) = 3k^2+7k there, below
// 4k^2+7k. That Accelerate's answers are sufficient is an unverified contract; the floor is allocated
// either way.
static inline int32_t vc_la_svd_work(int32_t k, int32_t *floating, int32_t *integer) {
    if (k < 0) return VC_LAPACK_WORKSPACE_OVERFLOW;
    if (k == 0) { *floating = *integer = 1; return 0; }
    size_t square, four, seven, total, eight;
    if (!vc_la_mul((size_t)k, (size_t)k, &square) ||
        !vc_la_mul(square, 4, &four) || !vc_la_mul((size_t)k, 7, &seven) ||
        !vc_la_add(four, seven, &total) || !vc_la_mul((size_t)k, 8, &eight) ||
        !vc_la_length(total, floating) || !vc_la_length(eight, integer))
        return VC_LAPACK_WORKSPACE_OVERFLOW;
    return 0;
}
static inline int32_t vc_la_eigen_work(int32_t n, int vectors, int32_t *floating, int32_t *integer) {
    if (n < 0) return VC_LAPACK_WORKSPACE_OVERFLOW;
    if (n <= 1) { *floating = *integer = 1; return 0; }
    size_t square, twice, linear, total, integers;
    if (!vectors) {
        if (!vc_la_mul((size_t)n, 2, &twice) || !vc_la_add(twice, 1, &total) ||
            !vc_la_length(total, floating)) return VC_LAPACK_WORKSPACE_OVERFLOW;
        *integer = 1;
        return 0;
    }
    if (!vc_la_mul((size_t)n, (size_t)n, &square) || !vc_la_mul(square, 2, &twice) ||
        !vc_la_mul((size_t)n, 6, &linear) || !vc_la_add(twice, linear, &total) ||
        !vc_la_add(total, 1, &total) || !vc_la_mul((size_t)n, 5, &integers) ||
        !vc_la_add(integers, 3, &integers) ||
        !vc_la_length(total, floating) || !vc_la_length(integers, integer))
        return VC_LAPACK_WORKSPACE_OVERFLOW;
    return 0;
}

// Allocates max(ceil(query) + ceil(query)/64 + 1, minimum) floats; the conservative ceil + 1/64 + 1 headroom
// accounts for coarse Float queries. `minimum` is a sufficient FLOOR, never a validity threshold: a positive,
// finite vendor answer below it is raised to it, not rejected (T19LAPACKWorkspaceQueryTests sweeps sgesdd path 5
// through svdThin and exact PCA). Rejected: a non-finite or <= 0 query, minimum < 1, and a count or byte total
// that is not representable. No Accelerate version or exact optimal-workspace guarantee is inferred.
static inline int32_t vc_la_query(float query, int32_t minimum, int32_t *count, size_t *bytes) {
    if (!isfinite(query) || query <= 0 || minimum < 1) return VC_LAPACK_INVALID_QUERY;
    double rounded = ceil((double)query);
    if (rounded > INT32_MAX) return VC_LAPACK_INVALID_QUERY;
    size_t value = (size_t)rounded, margin, total;
    if (!vc_la_add(value / 64, 1, &margin) || !vc_la_add(value, margin, &total))
        return VC_LAPACK_INVALID_QUERY;
    if (total < (size_t)minimum) total = (size_t)minimum;
    if (!vc_la_length(total, count) || !vc_la_bytes(total, sizeof(float), bytes))
        return VC_LAPACK_INVALID_QUERY;
    return 0;
}
// Allocates max(query, minimum) integers: the same floor rule, without headroom (no Float rounding to absorb).
// The caller passes the allocated count, bytes / sizeof(int32_t), as LIWORK.
static inline int32_t vc_la_integer_query(int32_t query, int32_t minimum, size_t *bytes) {
    if (minimum < 1 || query < 1 ||
        !vc_la_bytes((size_t)(query > minimum ? query : minimum), sizeof(int32_t), bytes))
        return VC_LAPACK_INVALID_QUERY;
    return 0;
}
static inline int vc_la_matrix_bytes(int32_t leading, int32_t columns) {
    size_t count, bytes;
    return leading >= 0 && columns >= 0 &&
        vc_la_mul((size_t)leading, (size_t)columns, &count) &&
        vc_la_bytes(count, sizeof(float), &bytes);
}
