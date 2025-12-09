# Floating-Point Reality

> **Reading time:** 10 minutes
> **Prerequisites:** None (but Chapter 1-2 provide helpful context)

---

## The Concept

Floating-point numbers are a way to represent real numbers in binary. Unlike integers (which are exact), floats are **approximations** with finite precision.

The IEEE 754 standard defines how floats work across all modern hardware:

| Type | Swift Type | Bits | Significand | Exponent | Approximate Range |
|------|------------|------|-------------|----------|-------------------|
| Single | `Float` | 32 | 23 bits (~7 digits) | 8 bits | ±3.4 × 10³⁸ |
| Double | `Double` | 64 | 52 bits (~15 digits) | 11 bits | ±1.8 × 10³⁰⁸ |

VectorCore uses `Float` (32-bit) for storage because:
- 4x more vectors fit in memory than with `Double`
- SIMD4<Float> processes twice as many values as SIMD2<Double>
- 7 decimal digits of precision is sufficient for most ML applications

---

## Why It Matters

### The Representation Problem

You might expect `0.1 + 0.2 == 0.3`. Let's see:

```swift
let a: Float = 0.1
let b: Float = 0.2
let c = a + b

print(c)           // 0.30000001
print(c == 0.3)    // false
```

Why? Because 0.1 cannot be represented exactly in binary, just like 1/3 can't be represented exactly in decimal:

```
Decimal: 1/3 = 0.333333... (infinite)
Binary:  0.1 = 0.0001100110011... (infinite)
```

When we store 0.1 in a `Float`, we truncate after 23 bits. The stored value is actually:

```
Stored "0.1" ≈ 0.100000001490116119384765625
```

Close, but not exact. These small errors are called **representation errors**.

### The Precision Limit

`Float` has ~7 significant decimal digits. Beyond that, information is lost:

```swift
let big: Float = 16777216  // 2²⁴ (exactly representable)
let bigger = big + 1       // Should be 16777217

print(bigger == big)       // true! (the +1 was lost)
```

What happened? At this magnitude, the gap between consecutive floats is 2, not 1. Adding 1 rounds back to the original value.

```
Float precision at different magnitudes:

Magnitude        Gap between consecutive values
1               ~0.0000001  (10⁻⁷)
1,000           ~0.0001     (10⁻⁴)
1,000,000       ~0.1        (10⁻¹)
16,777,216      1           (2⁰)
100,000,000     8           (2³)
```

---

## The Technique

### Understanding Float Anatomy

A 32-bit float has three parts:

```
┌───┬─────────────────┬─────────────────────────────────────────────┐
│ S │    Exponent     │               Significand                   │
│ 1 │     8 bits      │                 23 bits                     │
└───┴─────────────────┴─────────────────────────────────────────────┘
```

- **Sign (S):** 0 = positive, 1 = negative
- **Exponent:** Determines the scale (2^exp)
- **Significand:** The actual digits (plus an implicit leading 1)

Value = (-1)^S × 1.significand × 2^(exponent-127)

### Special Values

IEEE 754 defines special values for edge cases:

| Value | Exponent | Significand | Meaning |
|-------|----------|-------------|---------|
| `+0.0` | 0 | 0 | Positive zero |
| `-0.0` | 0 | 0 (sign=1) | Negative zero (yes, it exists!) |
| `Float.infinity` | 255 | 0 | Positive infinity |
| `-Float.infinity` | 255 | 0 (sign=1) | Negative infinity |
| `Float.nan` | 255 | non-zero | Not a Number |

You can create and check for these:

```swift
let inf = Float.infinity
let negInf = -Float.infinity
let nan = Float.nan

print(1.0 / 0.0)           // inf
print(-1.0 / 0.0)          // -inf
print(0.0 / 0.0)           // nan
print(Float.nan == Float.nan)  // false! (NaN is never equal to anything)
```

### Checking for Special Values

```swift
let value: Float = someComputation()

// Check for NaN
if value.isNaN {
    print("Got NaN!")
}

// Check for infinity
if value.isInfinite {
    print("Got infinity!")
}

// Check for finite (not NaN, not infinity)
if value.isFinite {
    print("Normal value: \(value)")
}

// Check for normal (finite and not subnormal)
if value.isNormal {
    print("Normal, non-denormalized: \(value)")
}
```

### Comparing Floats Properly

Never use `==` for floating-point comparison in numerical code:

```swift
// ❌ Bad: Exact equality
if result == expected {
    print("Equal")
}

// ✅ Good: Tolerance-based comparison
let tolerance: Float = 1e-6
if abs(result - expected) < tolerance {
    print("Close enough")
}

// ✅ Better: Relative tolerance for varying magnitudes
func approximatelyEqual(_ a: Float, _ b: Float,
                        relativeTolerance: Float = 1e-5,
                        absoluteTolerance: Float = 1e-8) -> Bool {
    let diff = abs(a - b)
    let largest = max(abs(a), abs(b))
    return diff <= max(relativeTolerance * largest, absoluteTolerance)
}
```

---

## In VectorCore

VectorCore handles floating-point edge cases explicitly:

**📍 See:** `Sources/VectorCore/Protocols/VectorProtocol.swift:100-116`

```swift
var isFinite: Bool {
    withUnsafeBufferPointer { buffer in
        for element in buffer {
            if !element.isFinite { return false }
        }
        return true
    }
}

var isZero: Bool {
    withUnsafeBufferPointer { buffer in
        for element in buffer {
            if element != 0 { return false }
        }
        return true
    }
}
```

The magnitude computation explicitly handles NaN:

**📍 See:** `Sources/VectorCore/Operations/Kernels/NormalizeKernels.swift:31-52`

```swift
// Track NaNs explicitly to propagate per IEEE expectations
var foundNaN = false
for i in 0..<laneCount {
    let v = storage[i]
    if v[0].isNaN || v[1].isNaN || v[2].isNaN || v[3].isNaN {
        foundNaN = true
        // Continue scanning to keep timing consistent
    }
    // ...
}

// Propagate NaN if any component is NaN
if foundNaN { return Float.nan }
```

**Why check each lane explicitly?** SIMD4 comparisons with NaN have subtle behavior. Explicit lane checks are more predictable.

**Why continue scanning after finding NaN?** Constant-time execution prevents timing attacks in security-sensitive contexts. (This may seem paranoid for vector math, but it's a good habit.)

---

## Float vs. Double: The Tradeoff

| Factor | Float (32-bit) | Double (64-bit) |
|--------|---------------|-----------------|
| Precision | ~7 digits | ~15 digits |
| Memory | 4 bytes | 8 bytes |
| SIMD (128-bit register) | SIMD4 = 4 values | SIMD2 = 2 values |
| Performance | Faster (2x parallelism) | Slower (half the values per op) |

VectorCore uses `Float` because:

1. **ML embeddings don't need 15 digits.** Embedding models are trained with float32; storing as float64 adds precision that doesn't exist.

2. **Memory bandwidth is the bottleneck.** Processing 4 floats per SIMD operation vs. 2 doubles is significant.

3. **Error is bounded.** For normalized vectors (magnitude 1), float32 relative error is ~10⁻⁷, which is fine for similarity search.

---

## Key Takeaways

1. **Floats are approximations, not exact values.** Never expect exact equality; always use tolerances.

2. **Precision decreases at larger magnitudes.** A `Float` near 10⁷ can't represent +1 differences.

3. **NaN propagates and is never equal to anything.** One NaN in your computation contaminates the result.

4. **Infinity is the result of overflow, not an error.** But it usually means your algorithm has a problem.

5. **VectorCore uses Float32 for storage.** It's sufficient precision with 2x memory efficiency and better SIMD utilization.

---

## Next Up

Now you know what floats are. Let's see the specific ways they can go wrong:

**[→ When Math Breaks](./02-When-Math-Breaks.md)**
