# Chapter 3: Numerical Computing

> **Floating-point math is an approximation. Here's how to keep it honest.**

Every time you write `1.0 + 2.0`, you're working with approximations. Most of the time, this doesn't matter. But in numerical computing—processing thousands of operations on high-dimensional vectors—small errors compound. Understanding floating-point behavior is essential for writing code that produces correct results.

---

## What You'll Learn

| Guide | Time | What You'll Learn |
|-------|------|-------------------|
| [1. Floating-Point Reality](./01-Floating-Point-Reality.md) | 10 min | IEEE 754, precision limits, representation |
| [2. When Math Breaks](./02-When-Math-Breaks.md) | 12 min | Overflow, underflow, catastrophic cancellation |
| [3. Stable Algorithms](./03-Stable-Algorithms.md) | 12 min | Kahan summation, two-pass algorithms |

---

## The Big Picture

Consider computing the magnitude (length) of a vector:

```
||v|| = √(v₀² + v₁² + v₂² + ... + vₙ²)
```

Seems simple. But what if `v₀ = 1e20`? Then `v₀² = 1e40`, which overflows `Float` (max ~3.4e38). Your result becomes `infinity`, even though the actual magnitude is finite.

VectorCore uses **Kahan's two-pass algorithm** to avoid this:

```
Pass 1: Find max = max(|v₀|, |v₁|, ..., |vₙ|)
Pass 2: Compute √(Σ((vᵢ/max)²)) × max
```

By scaling first, we keep all values in range. The final result is mathematically equivalent but numerically stable.

---

## Why Swift Developers Should Care

Swift's `Float` and `Double` follow IEEE 754, just like every other language. But Swift makes it easy to ignore the implications:

```swift
let a: Float = 0.1
let b: Float = 0.2
let c = a + b

print(c == 0.3)  // false!
print(c)         // 0.30000001192092896
```

This isn't a Swift bug—it's fundamental to binary floating-point. And if you're doing millions of operations (like in vector math), these small errors accumulate.

---

## VectorCore Connections

| VectorCore Feature | Numerical Concept |
|-------------------|-------------------|
| Two-pass magnitude computation | Overflow prevention via scaling |
| `Float` over `Double` | Precision vs. memory/speed tradeoff |
| NaN propagation in kernels | Explicit handling of invalid inputs |
| Clamping in cosine similarity | Range enforcement after floating-point drift |

---

## Start Here

**[→ Floating-Point Reality](./01-Floating-Point-Reality.md)**

---

*Chapter 3 of 6 • [← SIMD Demystified](../02-SIMD-Demystified/README.md) | [Next: Unsafe Swift →](../04-Unsafe-Swift/README.md)*
