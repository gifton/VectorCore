// vc_int8_accum.h — private helpers for the INT8 dot-product accumulation law (D02).
//
// Not a public header (it is outside include/, so SwiftPM does not export it) and every function is
// `static inline`, so it adds no exported symbol.
//
// The law (see vc_dot_int8 in VectorCoreC.h): the result is the exact integer dot product reduced modulo 2^32 and
// read in the int32_t range. Every implementation accumulates in uint32_t (unsigned wraparound is defined) or in
// SIMD integer lanes (lane additions wrap modulo 2^32), and maps the final 32-bit pattern with
// vc_int8_accum_to_int32, so no signed addition can overflow and no out-of-range unsigned-to-signed conversion
// occurs. Tested by Scripts/ci/tests/D02C/d02_helper_tests.c (run by Scripts/ci/tests/D02CContract.py).
#pragma once

#include <stddef.h>
#include <stdint.h>

// The int32_t value congruent to `bits` modulo 2^32. A pattern <= INT32_MAX converts directly (in range).
// Otherwise UINT32_MAX - bits <= INT32_MAX, so that conversion is in range too, and -1 - (UINT32_MAX - bits)
// lies in [INT32_MIN, -1] with no signed overflow; it equals bits - 2^32.
static inline int32_t vc_int8_accum_to_int32(uint32_t bits) {
    if (bits <= (uint32_t)INT32_MAX) {
        return (int32_t)bits;
    }
    return -1 - (int32_t)(UINT32_MAX - bits);
}

// The product of two int8_t values as a 32-bit pattern. The product is formed in int32_t, where
// |a * b| <= 128 * 128 = 16384 cannot overflow; the int32_t -> uint32_t conversion is defined (modulo 2^32).
static inline uint32_t vc_int8_product_bits(int8_t a, int8_t b) {
    return (uint32_t)((int32_t)a * (int32_t)b);
}

// 1 iff at least `block` elements remain after index `i`. Precondition: i <= lanes, which every loop keeps:
// `lanes - i` then cannot wrap, and after `i += block` (taken only when this returns 1) the index stays <= lanes.
// Replaces the `i + block <= lanes` form, whose sum wraps when i > SIZE_MAX - block.
static inline int vc_int8_block_fits(size_t lanes, size_t i, size_t block) {
    return lanes - i >= block;
}
