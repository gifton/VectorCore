#include <stddef.h>
#include <stdint.h>
/* Single synchronous consumer only. The ordinary dispatcher object is never changed. */
extern int32_t v02_real_vc_dot_int8(const int8_t *, const int8_t *, size_t);
static uint64_t calls;
void v02_reset(void) { calls = 0; }
uint64_t v02_count(void) { return calls; }
int32_t vc_dot_int8(const int8_t *a, const int8_t *b, size_t n) {
    calls += 1;
    return v02_real_vc_dot_int8(a, b, n);
}
