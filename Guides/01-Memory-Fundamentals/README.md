# Chapter 1: Memory Fundamentals

> **Before you can go fast, you need to understand where your data lives.**

Every performance optimization starts with memory. CPUs are incredibly fast at computation, but they spend most of their time *waiting*—waiting for data to arrive from memory. Understanding how memory works is the foundation of writing fast code.

---

## What You'll Learn

This chapter covers the fundamentals of how Swift stores data and why it matters for performance:

| Guide | Time | What You'll Learn |
|-------|------|-------------------|
| [1. How Swift Stores Data](./01-How-Swift-Stores-Data.md) | 10 min | Value types, reference types, and memory layout |
| [2. The Stack and Heap](./02-The-Stack-And-Heap.md) | 12 min | Where allocations happen and what it costs |
| [3. Why Alignment Matters](./03-Why-Alignment-Matters.md) | 10 min | Cache lines, SIMD requirements, and posix_memalign |

---

## The Big Picture

Here's what we're building toward:

```
┌─────────────────────────────────────────────────────────────────────┐
│                            YOUR CODE                                │
│                                                                     │
│   let v = Vector512Optimized(...)                                   │
│   let result = v.dotProduct(other)                                  │
│                                                                     │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        SWIFT RUNTIME                                │
│                                                                     │
│   Value types on stack, reference counting, ARC                     │
│                                                                     │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     MEMORY HIERARCHY                                │
│                                                                     │
│   L1 Cache (~1ns)  →  L2 Cache (~4ns)  →  RAM (~100ns)             │
│                                                                     │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                            CPU                                      │
│                                                                     │
│   Registers, SIMD units, execution pipelines                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

When you write `v.dotProduct(other)`, all of these layers are involved. Understanding each layer helps you write code that flows smoothly through the entire system.

---

## Why This Matters for Swift Developers

Swift does an excellent job of abstracting away memory management. ARC handles object lifetimes. Value semantics prevent unintended sharing. The optimizer is genuinely impressive.

But there are limits to what the compiler can do for you:

1. **The compiler can't change your data layout.** If your structs are poorly organized, you'll pay for it.

2. **The compiler can't guarantee alignment.** Standard Swift arrays might not be aligned for optimal SIMD operations.

3. **The compiler can't fix algorithmic memory patterns.** Cache-unfriendly algorithms stay cache-unfriendly.

This chapter teaches you to think at the memory level—so you can make choices the compiler can't make for you.

---

## Start Here

**[→ How Swift Stores Data](./01-How-Swift-Stores-Data.md)**

---

## VectorCore Connections

As you work through this chapter, you'll understand why VectorCore makes these design choices:

| VectorCore Feature | Memory Concept |
|-------------------|----------------|
| `Vector512Optimized` is a `struct` | Value semantics avoid heap allocation in hot paths |
| Storage uses `ContiguousArray<SIMD4<Float>>` | Contiguous memory enables SIMD and prefetching |
| `AlignedMemory.allocateAligned()` exists | Standard Swift arrays don't guarantee alignment |
| `posix_memalign` + `free()` pattern | Correct allocator/deallocator pairing |

---

*Chapter 1 of 6 • [Next: SIMD Demystified →](../02-SIMD-Demystified/README.md)*
