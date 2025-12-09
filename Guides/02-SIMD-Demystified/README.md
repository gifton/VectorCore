# Chapter 2: SIMD Demystified

> **One instruction, many data points. This is how we get fast.**

SIMD (Single Instruction, Multiple Data) is the secret behind VectorCore's performance. Instead of processing one number at a time, SIMD processes 4, 8, or even 16 numbers with a single instruction. This chapter demystifies how it works and how to use it effectively in Swift.

---

## What You'll Learn

| Guide | Time | What You'll Learn |
|-------|------|-------------------|
| [1. What Is SIMD?](./01-What-Is-SIMD.md) | 8 min | Lanes, vector registers, and why SIMD matters |
| [2. SIMD in Swift](./02-SIMD-In-Swift.md) | 12 min | Using SIMD2, SIMD4, SIMD8 effectively |
| [3. Writing SIMD-Friendly Code](./03-Writing-SIMD-Friendly-Code.md) | 15 min | Loop unrolling, multiple accumulators, and real patterns |

---

## The Big Picture

Modern CPUs have special registers and execution units for SIMD:

```
Traditional (Scalar) Processing:
┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐
│  a  │ + │  b  │ = │  c  │   │ ... │  ← 4 operations for 4 results
└─────┘   └─────┘   └─────┘   └─────┘
  1 op  →   1 op  →   1 op  →   1 op

SIMD Processing:
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│  a0 │ a1 │ a2 │ a3 │ + │  b0 │ b1 │ b2 │ b3 │ = │  c0 │ c1 │ c2 │ c3 │
└─────────────────────┘   └─────────────────────┘   └─────────────────────┘
              1 operation for 4 results
```

With SIMD, we can achieve up to 4x (SIMD4), 8x (SIMD8), or 16x (SIMD16) speedups for operations that map well to this model.

---

## Why Swift Developers Should Care

You might think: "The compiler auto-vectorizes my loops. Why do I need to know this?"

The compiler is good, but it's not magic:

1. **Complex data layouts confuse it.** Arrays of structs don't vectorize well.

2. **Reductions are tricky.** `sum = a[0] + a[1] + a[2] + ...` has data dependencies that limit vectorization.

3. **Manual SIMD can be 5-10x faster** than relying on auto-vectorization for numerical code.

4. **Understanding SIMD helps you write vectorizable code.** Even if you don't use `SIMD4` directly, knowing the patterns helps the compiler help you.

---

## VectorCore Connections

| VectorCore Feature | SIMD Concept |
|-------------------|--------------|
| `ContiguousArray<SIMD4<Float>>` storage | Data layout for 4-wide operations |
| 4 accumulators in kernels | Instruction-level parallelism |
| Stride-16 loops | Processing 64 floats per iteration |
| `@inline(__always)` on kernels | Ensuring SIMD isn't defeated by function calls |

---

## Start Here

**[→ What Is SIMD?](./01-What-Is-SIMD.md)**

---

*Chapter 2 of 6 • [← Memory Fundamentals](../01-Memory-Fundamentals/README.md) | [Next: Numerical Computing →](../03-Numerical-Computing/README.md)*
