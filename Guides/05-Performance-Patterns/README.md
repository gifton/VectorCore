# Chapter 5: Performance Patterns

> **Measure first. Optimize what matters. Measure again.**

Performance optimization without measurement is guesswork. This chapter teaches you to benchmark correctly, identify bottlenecks, and apply optimization patterns that actually matter.

---

## What You'll Learn

| Guide | Time | What You'll Learn |
|-------|------|-------------------|
| [1. Measuring Performance](./01-Measuring-Performance.md) | 10 min | Benchmarking, variance, warmup |
| [2. Cache-Friendly Code](./02-Cache-Friendly-Code.md) | 10 min | Memory access patterns, prefetching |
| [3. Parallelization Strategy](./03-Parallelization-Strategy.md) | 10 min | When to parallelize, overhead, heuristics |

---

## The Big Picture

Performance optimization is a funnel:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MEASURE                                      │
│  "Where is time actually being spent?"                              │
│                                                                     │
│  Tools: Instruments, custom benchmarks, profiling                   │
└────────────────────────────────────┬────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       ANALYZE                                       │
│  "Why is it slow?"                                                  │
│                                                                     │
│  Common causes: cache misses, branch misprediction, lock            │
│  contention, allocation overhead, poor parallelism                  │
└────────────────────────────────────┬────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       OPTIMIZE                                      │
│  "How do we fix it?"                                                │
│                                                                     │
│  Techniques: SIMD, cache-friendly layout, parallelization,          │
│  algorithmic improvements, inlining                                 │
└────────────────────────────────────┬────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       VALIDATE                                      │
│  "Did it actually help?"                                            │
│                                                                     │
│  Measure again. Compare. Beware of noise.                           │
└─────────────────────────────────────────────────────────────────────┘
```

Most performance bugs are found in step 1—we optimize the wrong thing because we didn't measure first.

---

## Why Swift Developers Should Care

Swift is fast *by default*, but not *automatically* fast:

1. **ARC has overhead.** Reference counting isn't free.
2. **Dynamic dispatch exists.** Protocol methods may use vtables.
3. **Value types aren't always cheap.** Large structs get copied.
4. **The compiler is smart, but not magic.** It can't fix bad algorithms.

Understanding performance patterns helps you write code that the optimizer can actually optimize.

---

## VectorCore Connections

| VectorCore Feature | Performance Pattern |
|-------------------|---------------------|
| Release-only assertions | Zero-cost validation |
| `ContiguousArray` over `Array` | Guaranteed contiguous memory |
| `@inlinable` / `@inline(__always)` | Eliminating call overhead |
| `ParallelHeuristic` | Adaptive parallelization |
| Batch operations | Amortizing overhead |

---

## Start Here

**[→ Measuring Performance](./01-Measuring-Performance.md)**

---

*Chapter 5 of 6 • [← Unsafe Swift](../04-Unsafe-Swift/README.md) | [Next: Capstone →](../06-Capstone/README.md)*
