// D02 evidence fixture (not a CI gate; run by Scripts/ci/tests/D02CContract.py with -I <tree>/Sources/VectorCoreC/common).
//
// Pure tests of the private INT8 helper header (vc_int8_accum.h): the modulo mapping and the remaining-count
// block test near SIZE_MAX, without allocating any buffer. Also demonstrates the replaced guard `i + block <= lanes`
// wrapping near SIZE_MAX (pure size_t arithmetic, no library call).
// Exit 0 iff every check passes; prints "HELPER_TESTS checks=N failures=M".
#include "vc_int8_accum.h"
#include <stdint.h>
#include <stdio.h>

static unsigned long checks = 0, failures = 0;
#define CHECK(cond, ...) do { checks++; if (!(cond)) { failures++; printf("FAIL " __VA_ARGS__); printf("\n"); } } while (0)

// Independent reference: exact value of the 32-bit pattern read as two's complement, via Int64 (in-range cast).
static int32_t ref_map(uint32_t bits) {
    int64_t v = (int64_t)bits;
    if (v > INT32_MAX) v -= 4294967296LL;
    return (int32_t)v;
}

int main(void) {
    // Mapping: named boundaries.
    const uint32_t named[] = {0u, 1u, 0x7FFFFFFEu, 0x7FFFFFFFu, 0x80000000u, 0x80000001u, 0xFFFFFFFEu, 0xFFFFFFFFu};
    for (unsigned k = 0; k < sizeof named / sizeof named[0]; ++k) {
        CHECK(vc_int8_accum_to_int32(named[k]) == ref_map(named[k]), "map(%u)", named[k]);
    }
    CHECK(vc_int8_accum_to_int32(0x7FFFFFFFu) == INT32_MAX, "map(INT32_MAX)");
    CHECK(vc_int8_accum_to_int32(0x80000000u) == INT32_MIN, "map(INT32_MAX+1)");
    CHECK(vc_int8_accum_to_int32(0xFFFFFFFFu) == -1, "map(UINT32_MAX)");
    // Mapping: 100000 pseudo-random patterns (xorshift32, fixed seed).
    uint32_t x = 0x9E3779B9u;
    for (int k = 0; k < 100000; ++k) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        CHECK(vc_int8_accum_to_int32(x) == ref_map(x), "map(%u)", x);
    }
    // Product bits: every int8 pair, against the Int64 product reduced modulo 2^32.
    for (int a = -128; a <= 127; ++a) {
        for (int b = -128; b <= 127; ++b) {
            int64_t p = (int64_t)a * b;
            uint32_t want = (uint32_t)(p < 0 ? p + 4294967296LL : p);
            CHECK(vc_int8_product_bits((int8_t)a, (int8_t)b) == want, "product(%d,%d)", a, b);
        }
    }
    // Remaining-count block test near SIZE_MAX: "full block allowed" iff lanes - i >= block, and the index after
    // the block stays <= lanes. Every i here satisfies i <= lanes (the helper's documented precondition).
    const size_t lanes_set[] = {SIZE_MAX, SIZE_MAX - 1, SIZE_MAX - 15};
    const size_t blocks[] = {16, 32, 64};
    for (unsigned l = 0; l < 3; ++l) {
        for (unsigned bk = 0; bk < 3; ++bk) {
            size_t lanes = lanes_set[l], block = blocks[bk];
            const size_t is[] = {lanes - block - 1, lanes - block, lanes - block + 1, lanes};
            for (unsigned k = 0; k < 4; ++k) {
                size_t i = is[k];
                // Independent oracle in 128-bit arithmetic (no wraparound): a full block fits iff i + block <= lanes.
                int expect = (unsigned __int128)i + block <= (unsigned __int128)lanes;
                int got = vc_int8_block_fits(lanes, i, block);
                CHECK(got == expect, "fits(lanes=%zu,i=%zu,block=%zu)", lanes, i, block);
                if (got) {
                    size_t next = i + block;
                    CHECK(next <= lanes && next >= i, "next index stays <= lanes (lanes=%zu,i=%zu,block=%zu)",
                          lanes, i, block);
                }
                // The replaced guard: for i > lanes - block, i + block wraps past SIZE_MAX and admits a block.
                int old_guard = (i + block) <= lanes;
                if (i > lanes - block) {
                    printf("OLD_GUARD lanes=%zu i=%zu block=%zu i+block=%zu admits=%d (new admits=%d)\n", lanes, i,
                           block, i + block, old_guard, got);
                }
            }
        }
    }
    // Small domain, exhaustive: lanes 0..200, i 0..lanes, the helper equals the remaining-count definition.
    for (size_t lanes = 0; lanes <= 200; ++lanes) {
        for (size_t i = 0; i <= lanes; ++i) {
            for (unsigned bk = 0; bk < 3; ++bk) {
                CHECK(vc_int8_block_fits(lanes, i, blocks[bk]) ==
                      ((unsigned __int128)i + blocks[bk] <= (unsigned __int128)lanes), "small %zu %zu", lanes, i);
            }
        }
    }
    printf("HELPER_TESTS checks=%lu failures=%lu\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
