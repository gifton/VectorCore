#include <stdint.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
// ABI/event fields: apple-oss-distributions/libmalloc src/malloc.c and
// private/stack_logging.h. Validation executable only; no library hook.
typedef void logger_t(uint32_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uint32_t);
extern logger_t *malloc_logger;
static _Atomic uint64_t events, bytes, large_events, large_bytes;
static _Atomic uintptr_t minimum;
static _Atomic int active;
static logger_t *previous;
static _Atomic uint64_t sizes[32], counts[32], provider_calls;
uint64_t q04_histogram(int index, int field) { return field ? counts[index] : sizes[index]; }
void q04_provider_call(void) { provider_calls++; }
uint64_t q04_provider_read(void) { return provider_calls; }
static void logger(uint32_t type, uintptr_t arg1, uintptr_t arg2,
                   uintptr_t arg3, uintptr_t result, uint32_t skip) {
    (void)arg1; (void)skip;
    if (!atomic_load_explicit(&active, memory_order_relaxed) || !(type & 2) || !result) return;
    uintptr_t size = type & 4 ? arg3 : arg2;
    for (int i = 0; i < 32; i++) {
        uint64_t seen = atomic_load(&sizes[i]);
        if (seen == size + 1 || (!seen && atomic_compare_exchange_strong(&sizes[i], &seen, size + 1))) {
            atomic_fetch_add(&counts[i], 1); break;
        }
    }
    atomic_fetch_add_explicit(&events, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&bytes, size, memory_order_relaxed);
    if (size >= atomic_load_explicit(&minimum, memory_order_relaxed)) {
        atomic_fetch_add_explicit(&large_events, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&large_bytes, size, memory_order_relaxed);
    }
}
void q04_start(uintptr_t threshold) {
    assert(!active);
    previous = malloc_logger;
    assert(previous == NULL); // Do not replace an existing diagnostic callback.
    events = bytes = large_events = large_bytes = 0;
    minimum = threshold;
    provider_calls = 0;
    for (int i = 0; i < 32; i++) { sizes[i] = 0; counts[i] = 0; }
    malloc_logger = logger;
    active = 1;
}
void q04_stop(void) { active = 0; malloc_logger = previous; }
uint64_t q04_read(int field) {
    switch (field) {
    case 0: return events;
    case 1: return bytes;
    case 2: return large_events;
    default: return large_bytes;
    }
}
void q04_calibrate_filter(void) {
    q04_start(100);
    logger(10, 1, 120, 0, 2, 0); // allocation
    logger(14, 1, 2, 150, 3, 0); // realloc: size is arg3
    logger(10, 1, 1000, 0, 0, 0); // failed allocation
    logger(12, 1, 2, 0, 0, 0); // free
    logger(74, 1, 20, 0, 4, 0); // cleared allocation below threshold
    q04_stop();
    assert(q04_read(0) == 3 && q04_read(1) == 290);
    assert(q04_read(2) == 2 && q04_read(3) == 270);
}
void q04_calibrate_c(void) {
    q04_start(8192);
    void *p = malloc(8192);
    void *q = NULL;
    int status = posix_memalign(&q, 64, 16384);
    q04_stop();
    assert(p && status == 0 && q);
    fprintf(stderr, "C calibration: %llu %llu %llu %llu\n", q04_read(0), q04_read(1), q04_read(2), q04_read(3));
    assert(q04_read(2) == 2 && q04_read(3) == 24576);
    free(p); free(q);
}
