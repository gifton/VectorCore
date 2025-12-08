# Stable Algorithms

> **Reading time:** 12 minutes
> **Prerequisites:** [When Math Breaks](./02-When-Math-Breaks.md)

---

## The Concept

A **numerically stable** algorithm produces accurate results even when intermediate values are very large, very small, or nearly equal. Stability isn't about being correct mathematically—it's about being correct when running on finite-precision hardware.

Key stable algorithms in VectorCore:

1. **Kahan's Magnitude Algorithm:** Prevents overflow in norm computation
2. **Tree Reduction:** Minimizes accumulation error in sums
3. **Guard Values:** Handles edge cases (zero vectors, NaN inputs)

---

## Why It Matters

### The $370 Million Bug

In 1996, the Ariane 5 rocket exploded 37 seconds after launch. The cause? A 64-bit float was converted to a 16-bit integer, and the value was out of range. The software was reused from Ariane 4, where the value never exceeded 16-bit limits.

Total loss: $370 million.

Numerical stability isn't just about correctness—it's about your code working in all conditions, not just the ones you tested.

---

## The Technique

### Algorithm 1: Kahan's Two-Pass Magnitude

We've seen this before, but let's understand why it works:

**The math:**

For a vector v = [v₀, v₁, ..., vₙ], the magnitude is:
```
||v|| = √(v₀² + v₁² + ... + vₙ²)
```

The stable reformulation:
```
M = max(|v₀|, |v₁|, ..., |vₙ|)
||v|| = M × √((v₀/M)² + (v₁/M)² + ... + (vₙ/M)²)
```

**Why it works:**

After dividing by M, all terms are ≤ 1. Squaring values ≤ 1 gives values ≤ 1. Summing N values ≤ 1 gives at most N. For a 512-dimensional vector, the maximum sum is 512—well within `Float` range.

```
Before scaling:            After scaling:
v[i] ∈ [-M, M]             v[i]/M ∈ [-1, 1]
(v[i])² ∈ [0, M²]          (v[i]/M)² ∈ [0, 1]
Σ(v[i]²) can overflow      Σ(v[i]/M)² ≤ n
```

**📍 See:** `Sources/VectorCore/Operations/Kernels/NormalizeKernels.swift:121-216`

### Algorithm 2: Compensated (Kahan) Summation

For long sums, error accumulates. Kahan summation tracks and corrects for it:

```swift
func kahanSum(_ values: [Float]) -> Float {
    var sum: Float = 0
    var compensation: Float = 0  // Running error correction

    for value in values {
        let corrected = value - compensation  // Add back what we lost last time
        let newSum = sum + corrected          // This addition loses precision
        compensation = (newSum - sum) - corrected  // Capture what was lost
        sum = newSum
    }

    return sum
}
```

**How it works:**

```
Standard sum:                  Kahan sum:
sum = 0                        sum = 0, c = 0
sum += 0.1  →  0.100000001    y = 0.1 - 0 = 0.1
sum += 0.1  →  0.200000003    t = 0.100000001 + 0.1 = 0.200000003
sum += 0.1  →  0.300000012    c = (0.200000003 - 0.100000001) - 0.1 = 0.000000002
...                            (error captured in c, corrected next iteration)

After 1M iterations:
Standard: 100958.34 (0.96% error)
Kahan:    100000.00 (negligible error)
```

VectorCore doesn't use full Kahan summation in kernels (it's slower), but uses:
- **Tree reduction** to minimize accumulated error
- **SIMD4 accumulators** which naturally batch operations

### Algorithm 3: Tree Reduction

Instead of summing left-to-right:
```
((((a + b) + c) + d) + e)
```

Sum in a tree:
```
    (a+b)     (c+d)
        \     /
         \   /
          \ /
        (ab+cd)
            \
             \
              \
             (abcd + e)
```

**Why it's better:**
- Fewer operations in the dependency chain
- Error from early additions doesn't compound as much
- Better for parallel execution

VectorCore applies this when combining accumulators:

```swift
// Instead of: acc0 + acc1 + acc2 + acc3
let combined = (acc0 + acc1) + (acc2 + acc3)  // Tree reduction
```

### Algorithm 4: Safe Division Guards

Division by zero or near-zero is a common source of problems:

```swift
// ❌ Dangerous
func normalize(_ v: [Float]) -> [Float] {
    let mag = magnitude(v)
    return v.map { $0 / mag }  // What if mag ≈ 0?
}

// ✅ Safe
func normalize(_ v: [Float]) -> Result<[Float], Error> {
    let mag = magnitude(v)
    guard mag > 0 else {
        return .failure(NormalizationError.zeroVector)
    }
    return .success(v.map { $0 / mag })
}
```

**📍 See:** `Sources/VectorCore/Protocols/VectorProtocol.swift:402-408`

```swift
/// Normalized (unit) vector
func normalized() -> Result<Self, VectorError> {
    let mag = magnitude
    guard mag > 0 else {
        return .failure(.invalidOperation("normalize", reason: "Cannot normalize zero vector"))
    }
    return .success(self / mag)
}
```

VectorCore uses `Result` type to make the failure explicit. Callers must handle the zero-vector case.

For hot paths where callers guarantee non-zero vectors:

```swift
/// Normalized (unit) vector without error checking.
/// - Precondition: `magnitude > 0` (asserted in debug builds only)
@inlinable
func normalizedUnchecked() -> Self {
    let mag = magnitude
    assert(mag > 0, "normalizedUnchecked() called on zero vector")
    return self / mag
}
```

### Algorithm 5: Range Clamping

After floating-point operations, values may drift slightly outside valid ranges:

```swift
func cosineSimilarity(_ a: Vector512Optimized, _ b: Vector512Optimized) -> Float {
    let dot = a.dotProduct(b)
    let magProduct = a.magnitude * b.magnitude

    guard magProduct > 0 else { return 0 }

    let similarity = dot / magProduct

    // Floating-point error might give 1.0000001 or -1.0000001
    // Clamp to valid [-1, 1] range
    return max(-1.0, min(1.0, similarity))
}
```

Without clamping, `acos(1.0000001)` returns `NaN` because the input is technically outside the domain.

---

## In VectorCore

### Complete Stable Magnitude Implementation

Let's trace through the full implementation:

**📍 See:** `Sources/VectorCore/Operations/Kernels/NormalizeKernels.swift`

```swift
static func magnitude(storage: ContiguousArray<SIMD4<Float>>, laneCount: Int) -> Float {

    // PHASE 1: Find maximum and detect NaN
    var maxVec = SIMD4<Float>(repeating: 0)
    var foundNaN = false

    for i in 0..<laneCount {
        let v = storage[i]

        // Check for NaN explicitly (SIMD comparisons with NaN are tricky)
        if v[0].isNaN || v[1].isNaN || v[2].isNaN || v[3].isNaN {
            foundNaN = true
        }

        // Track maximum absolute value
        let absVec = abs(v)
        maxVec = pointwiseMax(maxVec, absVec)
    }

    // Horizontal max across SIMD4 lanes
    let maxAbs = max(max(maxVec[0], maxVec[1]), max(maxVec[2], maxVec[3]))

    // EDGE CASE HANDLING
    if foundNaN { return Float.nan }     // Propagate NaN
    guard maxAbs > 0 else { return 0 }   // Zero vector
    guard maxAbs.isFinite else { return Float.infinity }  // Infinite component

    // PHASE 2: Scaled sum of squares
    let scale = 1.0 / maxAbs
    let simdScale = SIMD4<Float>(repeating: scale)

    // 4 accumulators for ILP
    var acc0 = SIMD4<Float>()
    var acc1 = SIMD4<Float>()
    var acc2 = SIMD4<Float>()
    var acc3 = SIMD4<Float>()

    for i in stride(from: 0, to: laneCount, by: 16) {
        // Scale and square in fused operations
        let s0 = storage[i+0] * simdScale; acc0 += s0 * s0
        let s1 = storage[i+1] * simdScale; acc1 += s1 * s1
        // ... (unrolled for all 16 elements per iteration)
    }

    // Tree reduction
    let sumSquares = ((acc0 + acc1) + (acc2 + acc3)).sum()

    // FINAL RESULT: Unscale
    return maxAbs * Foundation.sqrt(sumSquares)
}
```

**Stability features:**
1. NaN detection and propagation
2. Zero vector handling
3. Infinity handling
4. Kahan scaling for overflow prevention
5. 4-accumulator pattern for precision
6. Tree reduction for combining

---

## When to Use Stable vs. Fast Algorithms

| Scenario | Use Stable | Use Fast |
|----------|------------|----------|
| User-facing results | ✅ | ❌ |
| Inputs from untrusted sources | ✅ | ❌ |
| Values might be very large/small | ✅ | ❌ |
| Controlled test environment | ❌ | ✅ |
| Hot path with known-good inputs | ❌ | ✅ (with guards) |

VectorCore defaults to stable algorithms. For performance-critical paths with guaranteed input constraints, unchecked variants are available.

---

## Key Takeaways

1. **Kahan's two-pass algorithm prevents overflow** by scaling values before squaring.

2. **Tree reduction minimizes error accumulation** in sums and reductions.

3. **Edge cases need explicit handling.** Zero vectors, NaN, and infinity must be detected.

4. **Clamp outputs to valid ranges.** Floating-point drift can push values slightly out of bounds.

5. **Stable algorithms are ~20-30% slower.** Worth it for correctness; skip only when you can guarantee inputs.

6. **VectorCore provides both.** `normalized()` is safe; `normalizedUnchecked()` is fast for hot paths.

---

## Chapter Complete!

You now understand:
- How floats work (and fail)
- Specific failure modes in vector math
- Algorithms that stay correct under adversity

Next, we enter truly low-level territory—working directly with memory:

**[→ Chapter 4: Unsafe Swift](../04-Unsafe-Swift/README.md)**
