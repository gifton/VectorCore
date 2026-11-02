// D02 evidence fixture (not a CI gate; run by Scripts/ci/tests/D02CContract.py).
//
// Runs the INT8 dot-product fixture table through one entry of the ACTUAL VectorCoreC objects and compares each
// result with an independent oracle: the exact Int64 sum of the int8 products, reduced modulo 2^32 and read in the
// Int32 range by a mapping written here (it never calls VectorCoreC and does not include the private helper).
//
// usage: d02_int8_harness <dispatcher|scalar|arch> [--oracle-mutant]
//   dispatcher  vc_dot_int8 (public header)
//   scalar      vc_scalar_dot_int8 (internal, linker-visible; declared here by extern)
//   arch        vc_arm64_dot_int8 (arm64 builds only; SDOT or NEON fallback per the compile target)
//   --oracle-mutant  replaces the modulo mapping by saturation; the run must then report >= 1 mismatch
//                    (mutation control: proves the comparison can fail).
// Output: one line per case, then "SUMMARY cases=N mismatches=M". Exit 0 iff mismatches == 0 (or, with
// --oracle-mutant, iff mismatches > 0).
#include "VectorCoreC.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int32_t vc_scalar_dot_int8(const int8_t *a, const int8_t *b, size_t lanes);
#if defined(__aarch64__)
extern int32_t vc_arm64_dot_int8(const int8_t *a, const int8_t *b, size_t lanes);
#endif

typedef int32_t (*dot_fn)(const int8_t *, const int8_t *, size_t);

static int oracle_mutant = 0;

// Independent modulo-2^32 mapping of an exact Int64 sum into the Int32 range (oracle only).
static int32_t oracle_map(int64_t exact) {
    if (oracle_mutant) {   // deliberately wrong: saturation instead of the modulo law
        if (exact > INT32_MAX) return INT32_MAX;
        if (exact < INT32_MIN) return INT32_MIN;
        return (int32_t)exact;
    }
    int64_t r = exact % 4294967296LL;
    if (r < 0) r += 4294967296LL;
    if (r >= 2147483648LL) r -= 4294967296LL;
    return (int32_t)r;   // r is in [INT32_MIN, INT32_MAX] here
}

static int64_t oracle_exact(const int8_t *a, const int8_t *b, size_t lanes) {
    int64_t s = 0;
    for (size_t i = 0; i < lanes; ++i) s += (int64_t)a[i] * (int64_t)b[i];
    return s;
}

typedef enum { P_POS_MAX, P_NEG_MAX, P_POS_127, P_ZERO, P_ALT_SIGN, P_UNEQUAL, P_COUNT } pattern_t;
static const char *pattern_name[P_COUNT] = {"pos_max(-128,-128)", "neg_max(-128,127)", "pos_127(127,127)",
                                             "zero", "alt_sign", "unequal_lcg"};

static void fill(pattern_t p, int8_t *a, int8_t *b, size_t lanes) {
    uint32_t s = 0x2545F491u;
    for (size_t i = 0; i < lanes; ++i) {
        switch (p) {
        case P_POS_MAX: a[i] = -128; b[i] = -128; break;
        case P_NEG_MAX: a[i] = -128; b[i] = 127; break;
        case P_POS_127: a[i] = 127; b[i] = 127; break;
        case P_ZERO: a[i] = 0; b[i] = (int8_t)(i & 0x7f); break;
        case P_ALT_SIGN: a[i] = (i & 1) ? -128 : 127; b[i] = (i % 3 == 0) ? 127 : -128; break;
        default:
            s = s * 1664525u + 1013904223u; a[i] = (int8_t)((int)((s >> 24) & 0xff) - 128);
            s = s * 1664525u + 1013904223u; b[i] = (int8_t)((int)((s >> 16) & 0xff) - 128);
            break;
        }
    }
}

static size_t cases = 0, mismatches = 0;

static void check(const char *name, dot_fn fn, const int8_t *a, const int8_t *b, size_t lanes, int fixed_width) {
    int64_t exact = oracle_exact(a, b, lanes);
    int32_t expected = oracle_map(exact);
    int32_t actual = fn(a, b, lanes);
    int ok = actual == expected;
    // Fixed widths 512/768/1536: |exact| <= 1536 * 16384 = 25,165,824, so the result is the full integer value.
    if (fixed_width && !oracle_mutant) {
        int64_t mag = exact < 0 ? -exact : exact;
        if (mag > 25165824LL || (int64_t)actual != exact) ok = 0;
    }
    cases++;
    if (!ok) mismatches++;
    printf("CASE %s lanes=%zu exact=%lld expected=%d actual=%d %s\n", name, lanes, (long long)exact, expected, actual,
           ok ? "OK" : "MISMATCH");
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "allocation failed\n"); exit(3); }
    return p;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <dispatcher|scalar|arch> [--oracle-mutant]\n", argv[0]); return 2; }
    dot_fn fn = NULL;
    if (!strcmp(argv[1], "dispatcher")) fn = vc_dot_int8;
    else if (!strcmp(argv[1], "scalar")) fn = vc_scalar_dot_int8;
#if defined(__aarch64__)
    else if (!strcmp(argv[1], "arch")) fn = vc_arm64_dot_int8;
#endif
    if (!fn) { fprintf(stderr, "entry %s not available in this build\n", argv[1]); return 2; }
    oracle_mutant = argc > 2 && !strcmp(argv[2], "--oracle-mutant");
    printf("ENTRY %s\n", argv[1]);

    // Zero lanes: no pointer access is permitted (nil inputs).
    {
        int32_t r = fn(NULL, NULL, 0);
        cases++;
        if (r != 0) mismatches++;
        printf("CASE zero-lanes-null lanes=0 exact=0 expected=0 actual=%d %s\n", r, r == 0 ? "OK" : "MISMATCH");
    }

    static const size_t lanes_table[] = {1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 512, 768, 1536,
                                         131071, 131072, 131073, 1048576, 1048579};
    for (size_t k = 0; k < sizeof lanes_table / sizeof lanes_table[0]; ++k) {
        size_t lanes = lanes_table[k];
        int fixed = lanes == 512 || lanes == 768 || lanes == 1536;
        for (int p = 0; p < P_COUNT; ++p) {
            // Exact-size heap buffers so AddressSanitizer reports any read past `lanes`.
            int8_t *a = xmalloc(lanes), *b = xmalloc(lanes);
            fill((pattern_t)p, a, b, lanes);
            check(pattern_name[p], fn, a, b, lanes, fixed);
            free(a); free(b);
        }
    }

    // Individual SIMD accumulator lanes wrap. SDOT gives each of its 16 int32 lanes 4 * floor(lanes / 64) products
    // (4 accumulators x 4 lanes; the +17/+33 tails add a few more to one accumulator and the scalar tail):
    // - (-128,-128): 131072 * 16384 = 2^31 at 2^21 + 17 and 2^32 at 2^22 + 33 per lane, so a lane wraps at both sizes;
    // - (-128,127): 131072 * -16256 = -2,130,706,432 >= -2^31 at 2^21 + 17, so no lane wraps there (only the horizontal
    //   sum does); 262144 * -16256 < -2^31 at 2^22 + 33, so a lane wraps negatively only at that size.
    // So 3 of these 4 fixtures wrap a lane: the saturating-lane mutant of D02 fix round 1 (fix1/c-mutants) mismatches
    // on exactly those 3.
    {
        static const size_t wrap_lanes[] = {(1u << 21) + 17, (1u << 22) + 33};
        for (size_t k = 0; k < 2; ++k) {
            for (int p = P_POS_MAX; p <= P_NEG_MAX; ++p) {
                size_t lanes = wrap_lanes[k];
                int8_t *a = xmalloc(lanes), *b = xmalloc(lanes);
                fill((pattern_t)p, a, b, lanes);
                char name[64];
                snprintf(name, sizeof name, "lane-wrap-%s", pattern_name[p]);
                check(name, fn, a, b, lanes, 0);
                free(a); free(b);
            }
        }
    }

    // Every (a, b) int8 pair exactly once: lanes 65536, exact sum (sum a)(sum b) = (-128)(-128) = 16384.
    {
        size_t lanes = 65536;
        int8_t *a = xmalloc(lanes), *b = xmalloc(lanes);
        for (size_t i = 0; i < lanes; ++i) { a[i] = (int8_t)((int)(i >> 8) - 128); b[i] = (int8_t)((int)(i & 0xff) - 128); }
        check("all-int8-pairs", fn, a, b, lanes, 0);
        free(a); free(b);
    }

    // Cancellation: the final sum fits Int32 (2^24) but the prefix 131072 * 16384 = 2^31 does not.
    {
        size_t lanes = 262144;
        int8_t *a = xmalloc(lanes), *b = xmalloc(lanes);
        for (size_t i = 0; i < lanes; ++i) { a[i] = -128; b[i] = i < 131072 ? -128 : 127; }
        check("cancellation-prefix-overflow", fn, a, b, lanes, 0);
        free(a); free(b);
    }

    // Tail/final-combination witness (positive): lanes 131073; the first 131072 lanes are covered by every SIMD
    // block size (16/32/64) and sum to 131071*16384 + 16129 = 2^31 - 255 (fits Int32); the one tail lane adds 16129,
    // so a signed final addition exceeds INT32_MAX.
    {
        size_t lanes = 131073;
        int8_t *a = xmalloc(lanes), *b = xmalloc(lanes);
        for (size_t i = 0; i < lanes; ++i) { a[i] = -128; b[i] = -128; }
        a[131071] = 127; b[131071] = 127;
        a[131072] = 127; b[131072] = 127;
        check("tail-final-witness-positive", fn, a, b, lanes, 0);
        free(a); free(b);
    }

    // Tail/final-combination witness (negative): lanes 132105 = 132096 + 9; the SIMD part (a multiple of 64 and of
    // 16) sums to -132096*16256 = -2,147,352,576 and the 9 tail lanes add -146,304, below INT32_MIN.
    {
        size_t lanes = 132105;
        int8_t *a = xmalloc(lanes), *b = xmalloc(lanes);
        for (size_t i = 0; i < lanes; ++i) { a[i] = -128; b[i] = 127; }
        check("tail-final-witness-negative", fn, a, b, lanes, 0);
        free(a); free(b);
    }

    printf("SUMMARY cases=%zu mismatches=%zu oracle_mutant=%d\n", cases, mismatches, oracle_mutant);
    if (oracle_mutant) return mismatches > 0 ? 0 : 1;
    return mismatches == 0 ? 0 : 1;
}
