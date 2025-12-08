# VectorCore Learning Guide

> **Your journey into high-performance Swift programming starts here.**

Welcome to the VectorCore Learning Guide—a comprehensive, hands-on exploration of low-level Swift programming techniques. Whether you've spent years building iOS apps or are comfortable with Swift's standard library, this guide will take you deeper into the machine, showing you how to write code that runs 10x faster than you thought possible.

---

## What You'll Learn

This guide teaches **transferable concepts**—knowledge that applies far beyond VectorCore:

| Chapter | You'll Learn | Why It Matters |
|---------|-------------|----------------|
| [1. Memory Fundamentals](./01-Memory-Fundamentals/README.md) | How data lives in memory | Understanding the machine you're programming |
| [2. SIMD Demystified](./02-SIMD-Demystified/README.md) | Processing multiple values at once | 4-8x speedups with the same hardware |
| [3. Numerical Computing](./03-Numerical-Computing/README.md) | When floating-point math breaks | Writing code that doesn't silently corrupt data |
| [4. Unsafe Swift](./04-Unsafe-Swift/README.md) | Pointers, buffers, and C interop | Breaking out of Swift's safety net (carefully) |
| [5. Performance Patterns](./05-Performance-Patterns/README.md) | Measuring and optimizing | Knowing what to optimize and when |
| [6. Capstone](./06-Capstone/README.md) | Reading production kernel code | Putting it all together |

---

## Prerequisites

This guide assumes you're comfortable with:

- **Swift fundamentals**: structs, classes, protocols, generics
- **Basic concurrency concepts**: what threads are, why race conditions are bad
- **Reading Swift code**: you don't need to be an expert, but you should be able to follow along

You **don't** need experience with:

- Unsafe pointers or manual memory management
- SIMD or low-level CPU operations
- C programming or interoperability
- Assembly or computer architecture

We'll build up from first principles.

---

## How to Use This Guide

### The Guided Path

Each chapter builds on the previous one. If you're new to low-level programming, start at Chapter 1 and work through sequentially:

```
Chapter 1 ──→ Chapter 2 ──→ Chapter 3 ──→ Chapter 4 ──→ Chapter 5 ──→ Chapter 6
  Memory       SIMD        Numerical      Unsafe       Performance    Capstone
```

### The Reference Path

If you're already comfortable with some topics, jump directly to what interests you:

- **"I want to understand SIMD"** → Start at [Chapter 2](./02-SIMD-Demystified/README.md)
- **"I'm debugging NaN values"** → Jump to [Chapter 3](./03-Numerical-Computing/README.md)
- **"I need to use UnsafePointer"** → Go to [Chapter 4](./04-Unsafe-Swift/README.md)
- **"I want to see real optimized code"** → Skip to [Chapter 6](./06-Capstone/README.md)

### Each Guide Follows This Pattern

```
┌─────────────────────────────────────────────────────────────┐
│  THE CONCEPT                                                │
│  What's the idea? Plain English, no code yet.               │
├─────────────────────────────────────────────────────────────┤
│  WHY IT MATTERS                                             │
│  What breaks without this knowledge? Real bugs, real pain.  │
├─────────────────────────────────────────────────────────────┤
│  THE TECHNIQUE                                              │
│  How do we solve it? Code examples, diagrams.               │
├─────────────────────────────────────────────────────────────┤
│  IN VECTORCORE                                              │
│  Where does VectorCore use this? Links to real source.      │
├─────────────────────────────────────────────────────────────┤
│  KEY TAKEAWAYS                                              │
│  What should stick? The transferable lessons.               │
└─────────────────────────────────────────────────────────────┘
```

---

## A Note on Philosophy

VectorCore exists because **Swift can be fast**—really fast. Competitive with C and hand-tuned assembly. But getting there requires understanding what's happening at the hardware level.

This guide doesn't just show you *what* VectorCore does. It teaches you *why* we made these choices, so you can apply the same techniques to your own projects.

The goal isn't to make you memorize patterns. It's to give you a mental model of the machine that lets you reason about performance from first principles.

---

## Let's Begin

Ready? Start with the foundation:

**[→ Chapter 1: Memory Fundamentals](./01-Memory-Fundamentals/README.md)**

---

## Quick Reference

### VectorCore Source Locations

Throughout this guide, we'll reference actual VectorCore source code:

| Topic | File Path |
|-------|-----------|
| SIMD Dot Product | `Sources/VectorCore/Operations/Kernels/DotKernels.swift` |
| Stable Normalization | `Sources/VectorCore/Operations/Kernels/NormalizeKernels.swift` |
| Memory Alignment | `Sources/VectorCore/Storage/AlignedMemory.swift` |
| Optimized Vector | `Sources/VectorCore/Vectors/Vector512Optimized.swift` |
| Protocol Design | `Sources/VectorCore/Protocols/VectorProtocol.swift` |

### Notation Conventions

| Symbol | Meaning |
|--------|---------|
| `📍 See:` | Link to VectorCore source code |
| `⚠️` | Common mistake or pitfall |
| `💡` | Key insight or tip |
| `🔬` | Deep dive into technical details |

---

*VectorCore Learning Guide • Last updated: 2025*
