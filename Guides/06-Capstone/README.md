# Chapter 6: Capstone

> **Let's read real code together.**

You've learned the concepts. Now let's see them all working together in production code.

This chapter is a guided reading of VectorCore's kernel implementations. We'll trace through actual source files, connecting each line to the concepts you've learned.

---

## What You'll Do

| Guide | Time | What You'll Explore |
|-------|------|-------------------|
| [1. Reading a Real Kernel](./01-Reading-A-Real-Kernel.md) | 20 min | Line-by-line walkthrough of DotKernels.swift |

---

## How This Works

Open the VectorCore source code alongside this guide:

```
📂 VectorCore/
   📂 Sources/VectorCore/Operations/Kernels/
      📄 DotKernels.swift     ← We'll read this
      📄 NormalizeKernels.swift
      📄 EuclideanKernels.swift
```

Each section will reference specific line numbers. Follow along in your editor.

---

## Connecting the Chapters

As you read the kernel, you'll recognize techniques from every chapter:

| Chapter | You'll See |
|---------|-----------|
| [1. Memory](../01-Memory-Fundamentals/README.md) | `ContiguousArray` storage, value type design |
| [2. SIMD](../02-SIMD-Demystified/README.md) | `SIMD4<Float>`, multiple accumulators, lane operations |
| [3. Numerical](../03-Numerical-Computing/README.md) | Kahan's algorithm in NormalizeKernels |
| [4. Unsafe](../04-Unsafe-Swift/README.md) | Pointer access patterns, memory binding |
| [5. Performance](../05-Performance-Patterns/README.md) | `@inline(__always)`, sequential access, unrolling |

---

## Start Here

**[→ Reading a Real Kernel](./01-Reading-A-Real-Kernel.md)**

---

*Chapter 6 of 6 • [← Performance Patterns](../05-Performance-Patterns/README.md) | [Back to Welcome →](../00-Welcome.md)*
