# When Math Breaks

> **Reading time:** 12 minutes
> **Prerequisites:** [Floating-Point Reality](./01-Floating-Point-Reality.md)

---

## The Concept

Floating-point arithmetic can fail in several specific ways. Understanding these failure modes helps you recognize them in your code—and avoid them.

The main failure modes:

1. **Overflow:** Result too large to represent → `infinity`
2. **Underflow:** Result too small to represent → `0` (or denormalized)
3. **Catastrophic Cancellation:** Subtraction destroys precision
4. **Accumulation Error:** Small errors compound over many operations

---

## Why It Matters

### The NaN Bug That Shipped

Imagine you ship an embedding search feature. Users report that some queries return no results. Investigation reveals:

```swift
func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
    let dot = zip(a, b).map(*).reduce(0, +)
    let magA = sqrt(a.map { $0 * $0 }.reduce(0, +))
    let magB = sqrt(b.map { $0 * $0 }.reduce(0, +))
    return dot / (magA * magB)
}
```

Looks fine. But some embeddings have a component like `1e20`. Then:

```
Component:     1e20
Squared:       1e40  (overflow! Float.max ≈ 3.4e38)
Result:        infinity
magA:          infinity
dot:           also infinity (if both vectors have large values)
∞ / ∞ = NaN

Similarity = NaN  →  Comparison fails  →  No results returned
```

A single large value corrupted everything. This isn't hypothetical—it happens with real embedding models.

---

## The Technique

### Failure Mode 1: Overflow

**What:** Result exceeds `Float.max` (~3.4 × 10³⁸)

**Example:**
```swift
let big: Float = 1e20
let squared = big * big  // 1e40 → overflow!
print(squared)           // inf
print(squared.isInfinite) // true
```

**When it happens in vectors:**
- Computing magnitude: `sqrt(Σx²)` when any `x > 1.8e19`
- Dot products with large values
- Accumulating many large values

**Detection:**
```swift
if result.isInfinite {
    print("Overflow occurred")
}
```

### Failure Mode 2: Underflow

**What:** Result too small to represent, becomes 0 or denormalized

**Example:**
```swift
let tiny: Float = 1e-45
let tinier = tiny * tiny  // 1e-90 → underflow!
print(tinier)             // 0.0
```

**When it happens:**
- Multiplying many small probabilities
- Normalizing vectors with very small magnitudes
- Intermediate calculations in multi-step algorithms

**Why it's subtle:** Unlike infinity (which propagates), zero looks like a valid result. You might not notice the bug.

### Failure Mode 3: Catastrophic Cancellation

**What:** Subtracting nearly-equal numbers destroys significant digits

**Example:**
```swift
let a: Float = 1.000001
let b: Float = 1.000000
let diff = a - b

print(diff)  // 9.536743e-07 (correct: 1e-06)
// We lost 6 digits of precision!
```

What happened:

```
a:            1.000001      (7 significant digits)
b:            1.000000      (7 significant digits)
a - b:        0.000001ish   (1 significant digit left!)

The 6 leading digits cancelled, leaving only the noisy trailing bits.
```

**When it happens in vectors:**
- Computing variance: `E[x²] - E[x]²` (two nearly-equal values)
- Distance between nearby points
- Differences of large sums

### Failure Mode 4: Accumulation Error

**What:** Small errors compound across many operations

**Example:**
```swift
var sum: Float = 0
for _ in 0..<1_000_000 {
    sum += 0.1
}
print(sum)  // 100958.34 (should be 100000.0)
// Error: 0.96%
```

Each addition of `0.1` (which isn't exactly representable) adds a small error. After a million iterations, the error is nearly 1%.

**When it happens:**
- Summing many small values
- Iterative algorithms (gradients, moving averages)
- Long-running simulations

---

## In VectorCore

### The Magnitude Problem

The naive magnitude formula is:

```
||v|| = sqrt(v[0]² + v[1]² + ... + v[n]²)
```

This overflows when any component exceeds `sqrt(Float.max) ≈ 1.84e19`:

```swift
let dangerous: Float = 2e19
let squared = dangerous * dangerous  // 4e38 > Float.max → infinity
```

VectorCore uses **Kahan's two-pass algorithm**:

**📍 See:** `Sources/VectorCore/Operations/Kernels/NormalizeKernels.swift:121-216`

```swift
/// Compute magnitude using Kahan's two-pass scaling algorithm
///
/// This is the numerically stable version that avoids intermediate overflow.
@inline(__always)
@usableFromInline
static func magnitude(storage: ContiguousArray<SIMD4<Float>>, laneCount: Int) -> Float {
    // Phase 1: Find maximum absolute value
    var maxVec = SIMD4<Float>(repeating: 0)
    var foundNaN = false
    for i in 0..<laneCount {
        let v = storage[i]
        if v[0].isNaN || v[1].isNaN || v[2].isNaN || v[3].isNaN {
            foundNaN = true
        }
        let absVec = abs(v)
        maxVec = pointwiseMax(maxVec, absVec)
    }

    let maxAbs = max(max(maxVec[0], maxVec[1]), max(maxVec[2], maxVec[3]))

    // Handle edge cases
    if foundNaN { return Float.nan }
    guard maxAbs > 0 else { return 0 }  // Zero vector
    guard maxAbs.isFinite else { return Float.infinity }

    // Phase 2: Scale, sum squares, scale back
    let scale = 1.0 / maxAbs
    // ... (4-accumulator SIMD pattern) ...

    // Return magnitude: maxAbs × sqrt(Σ((x/maxAbs)²))
    return maxAbs * Foundation.sqrt(sumSquares)
}
```

**How it works:**

1. **Find maximum absolute value (M):** One pass through the data
2. **Scale all values by 1/M:** Now all values are ≤ 1
3. **Sum scaled squares:** Maximum intermediate value is N (vector dimension)
4. **Multiply result by M:** Restore original scale

```
Naive:                          Kahan's Two-Pass:

v = [1e20, 1e20, ...]          v = [1e20, 1e20, ...]

v[0]² = 1e40 → infinity!        max = 1e20
                                scaled = v / max = [1, 1, ...]
                                sum_sq = 1 + 1 + ... = n
                                result = max × sqrt(n) = 1e20 × sqrt(n)  ✓
```

**The tradeoff:** Two passes instead of one (~20-30% slower), but **always correct**.

### NaN Handling in VectorProtocol

VectorCore propagates NaN according to IEEE 754 semantics:

```swift
var magnitude: Scalar {
    // Phase 1: Detect NaN early
    var hasNaN = false
    withUnsafeBufferPointer { buffer in
        for element in buffer {
            if element.isNaN { hasNaN = true; continue }
            // ...
        }
    }

    // Propagate NaN if any component is NaN
    if hasNaN { return Scalar.nan }

    // ... rest of computation ...
}
```

**Why propagate immediately?** Once you have a NaN, further computation is meaningless. Propagating early prevents wasted work.

---

## Defensive Patterns

### Pattern 1: Pre-check Inputs

```swift
func processVector(_ v: Vector512Optimized) throws -> Float {
    guard v.isFinite else {
        throw VectorError.invalidInput("Vector contains non-finite values")
    }
    // Safe to proceed
}
```

### Pattern 2: Clamp Outputs

```swift
func cosineSimilarity(_ a: Vector512Optimized, _ b: Vector512Optimized) -> Float {
    let similarity = a.cosineSimilarity(to: b)
    // Clamp to valid range (floating-point drift might push slightly outside)
    return max(-1.0, min(1.0, similarity))
}
```

### Pattern 3: Use Stable Formulations

```swift
// ❌ Unstable variance: E[x²] - E[x]²
func variance_unstable(_ values: [Float]) -> Float {
    let meanSq = values.map { $0 * $0 }.reduce(0, +) / Float(values.count)
    let mean = values.reduce(0, +) / Float(values.count)
    return meanSq - mean * mean  // Catastrophic cancellation!
}

// ✅ Stable variance: Two-pass algorithm
func variance_stable(_ values: [Float]) -> Float {
    let mean = values.reduce(0, +) / Float(values.count)
    let sumSquaredDiff = values.map { ($0 - mean) * ($0 - mean) }.reduce(0, +)
    return sumSquaredDiff / Float(values.count)
}
```

### Pattern 4: Log-Space for Products

```swift
// ❌ Underflows for many small probabilities
let product = probabilities.reduce(1.0, *)  // 0.01 × 0.01 × ... → 0

// ✅ Work in log-space
let logSum = probabilities.map { log($0) }.reduce(0, +)
let product = exp(logSum)  // Handles very small products
```

---

## Key Takeaways

1. **Overflow turns values into infinity.** Check `.isInfinite` and use scaling to prevent it.

2. **Underflow silently becomes zero.** Work in log-space for products of small values.

3. **Catastrophic cancellation destroys precision.** Avoid subtracting nearly-equal values; reformulate algorithms when possible.

4. **Accumulation error grows with iterations.** Use compensated summation (Kahan) for long sums.

5. **VectorCore uses two-pass algorithms.** ~25% slower, but correct for any input values.

---

## Next Up

Now you understand the failure modes. Let's see the algorithms that avoid them:

**[→ Stable Algorithms](./03-Stable-Algorithms.md)**
