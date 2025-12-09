# SIMD in Swift

> **Reading time:** 12 minutes
> **Prerequisites:** [What Is SIMD?](./01-What-Is-SIMD.md)

---

## The Concept

Swift provides first-class SIMD support through the `simd` module. You get type-safe vector operations that compile down to efficient CPU instructions—no assembly required.

The key types:

| Type | Lanes | Total Size | Use Case |
|------|-------|------------|----------|
| `SIMD2<Float>` | 2 | 8 bytes | 2D graphics, complex numbers |
| `SIMD4<Float>` | 4 | 16 bytes | General purpose, matches NEON |
| `SIMD8<Float>` | 8 | 32 bytes | AVX on Intel |
| `SIMD16<Float>` | 16 | 64 bytes | AVX-512, or 4×SIMD4 on ARM |

VectorCore primarily uses `SIMD4<Float>` because:
- It's natively supported on all Apple platforms (NEON)
- It matches the register width (128-bit)
- It's the sweet spot for most operations

---

## Why It Matters

### Swift SIMD vs. Manual Assembly

In the old days, you'd write platform-specific assembly or intrinsics:

```c
// C with Intel intrinsics (don't do this in Swift!)
#include <immintrin.h>

float dot_product_avx(float* a, float* b, int n) {
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < n; i += 8) {
        __m256 va = _mm256_load_ps(a + i);
        __m256 vb = _mm256_load_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    // ... horizontal sum ...
}
```

With Swift SIMD, the same logic is type-safe and portable:

```swift
// Swift SIMD - clean, safe, fast
func dotProduct(_ a: [SIMD4<Float>], _ b: [SIMD4<Float>]) -> Float {
    var sum = SIMD4<Float>()
    for i in 0..<a.count {
        sum += a[i] * b[i]  // Fused multiply-add when available
    }
    return sum.sum()
}
```

The Swift compiler emits the same quality instructions, but you get:
- Type safety (can't mix SIMD4 with SIMD8 accidentally)
- Bounds checking in debug builds
- Readable, maintainable code

---

## The Technique

### Creating SIMD Values

```swift
import simd

// From literal values
let a = SIMD4<Float>(1.0, 2.0, 3.0, 4.0)

// Repeating value (broadcast)
let zeros = SIMD4<Float>(repeating: 0.0)
let ones = SIMD4<Float>(repeating: 1.0)

// From an array (must have exact count)
let array: [Float] = [1, 2, 3, 4]
let fromArray = SIMD4<Float>(array)  // Works!

// Zero-initialized
let zero = SIMD4<Float>()  // All zeros
```

### Accessing Lanes

```swift
let v = SIMD4<Float>(1.0, 2.0, 3.0, 4.0)

// Subscript access
print(v[0])  // 1.0
print(v[1])  // 2.0

// Named accessors (for 2D/3D/4D vectors)
print(v.x)   // 1.0 (same as v[0])
print(v.y)   // 2.0
print(v.z)   // 3.0
print(v.w)   // 4.0

// Swizzling (reordering lanes)
let swizzled = SIMD4<Float>(v.w, v.z, v.y, v.x)  // 4.0, 3.0, 2.0, 1.0
```

### Arithmetic Operations

All basic arithmetic works lane-wise:

```swift
let a = SIMD4<Float>(1, 2, 3, 4)
let b = SIMD4<Float>(5, 6, 7, 8)

// Lane-wise operations
let sum = a + b          // (6, 8, 10, 12)
let diff = a - b         // (-4, -4, -4, -4)
let product = a * b      // (5, 12, 21, 32)
let quotient = a / b     // (0.2, 0.33, 0.43, 0.5)

// Scalar operations broadcast
let scaled = a * 2.0     // (2, 4, 6, 8)
let shifted = a + 10.0   // (11, 12, 13, 14)

// Compound assignment
var c = a
c += b                   // c is now (6, 8, 10, 12)
c *= 2.0                 // c is now (12, 16, 20, 24)
```

### Reduction Operations

Getting scalar results from SIMD:

```swift
let v = SIMD4<Float>(1, 2, 3, 4)

// Horizontal sum
let sum = v.sum()           // 10.0

// Min/max
let minVal = v.min()        // 1.0
let maxVal = v.max()        // 4.0

// Reduce with custom operation
let product = v.indices.reduce(Float(1)) { $0 * v[$1] }  // 24.0
```

### Comparison Operations

Comparisons return SIMD masks:

```swift
let a = SIMD4<Float>(1, 2, 3, 4)
let b = SIMD4<Float>(2, 2, 2, 2)

// Lane-wise comparison
let lessThan = a .< b    // SIMDMask<SIMD4<Float>>: (true, false, false, false)
let equal = a .== b      // (false, true, false, false)
let greaterEq = a .>= b  // (false, true, true, true)

// Check if any/all lanes match
let anyLess = any(a .< b)     // true (lane 0 is less)
let allLess = all(a .< b)     // false

// Select based on mask
let mask = a .> b
let selected = mask.replacing(with: a, where: mask)  // Complex, see below
```

### Math Functions

The `simd` module provides vectorized math:

```swift
import simd

let v = SIMD4<Float>(0.0, 0.5, 1.0, 2.0)

// Transcendental functions (per-lane)
let sines = sin(v)       // sin of each lane
let cosines = cos(v)
let exps = exp(v)
let logs = log(v + 1)    // log(1), log(1.5), log(2), log(3)
let sqrts = sqrt(v)

// Clamping
let clamped = simd_clamp(v, SIMD4(repeating: 0.2), SIMD4(repeating: 1.5))
// Result: (0.2, 0.5, 1.0, 1.5)

// Absolute value
let abs_v = abs(SIMD4<Float>(-1, 2, -3, 4))  // (1, 2, 3, 4)

// Min/max between vectors
let a = SIMD4<Float>(1, 5, 3, 7)
let b = SIMD4<Float>(2, 4, 6, 8)
let mins = simd_min(a, b)  // (1, 4, 3, 7) - lane-wise minimum
let maxs = simd_max(a, b)  // (2, 5, 6, 8) - lane-wise maximum
// Or use pointwiseMin/pointwiseMax
let mins2 = pointwiseMin(a, b)
let maxs2 = pointwiseMax(a, b)
```

---

## Working with Arrays

Converting between arrays and SIMD:

### Loading from Arrays

```swift
let floats: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]

// Load 4 floats at a time
floats.withUnsafeBufferPointer { buffer in
    let simd0 = SIMD4<Float>(
        buffer[0], buffer[1], buffer[2], buffer[3]
    )
    let simd1 = SIMD4<Float>(
        buffer[4], buffer[5], buffer[6], buffer[7]
    )
}

// Or via pointer casting (more efficient)
floats.withUnsafeBufferPointer { buffer in
    let simdPtr = UnsafeRawPointer(buffer.baseAddress!)
        .bindMemory(to: SIMD4<Float>.self, capacity: 2)

    let simd0 = simdPtr[0]  // floats 0-3
    let simd1 = simdPtr[1]  // floats 4-7
}
```

### Storing to Arrays

```swift
var result = [Float](repeating: 0, count: 8)
let simd0 = SIMD4<Float>(1, 2, 3, 4)
let simd1 = SIMD4<Float>(5, 6, 7, 8)

result.withUnsafeMutableBufferPointer { buffer in
    let simdPtr = UnsafeMutableRawPointer(buffer.baseAddress!)
        .bindMemory(to: SIMD4<Float>.self, capacity: 2)

    simdPtr[0] = simd0
    simdPtr[1] = simd1
}
// result is now [1, 2, 3, 4, 5, 6, 7, 8]
```

---

## Common Patterns

### Pattern 1: Dot Product

```swift
func dotProduct(_ a: ContiguousArray<SIMD4<Float>>,
                _ b: ContiguousArray<SIMD4<Float>>) -> Float {
    var sum = SIMD4<Float>()
    for i in 0..<a.count {
        sum += a[i] * b[i]
    }
    return sum.sum()  // One horizontal operation at the end
}
```

### Pattern 2: Euclidean Distance Squared

```swift
func distanceSquared(_ a: ContiguousArray<SIMD4<Float>>,
                     _ b: ContiguousArray<SIMD4<Float>>) -> Float {
    var sum = SIMD4<Float>()
    for i in 0..<a.count {
        let diff = a[i] - b[i]
        sum += diff * diff
    }
    return sum.sum()
}
```

### Pattern 3: Element-wise Maximum

```swift
func elementwiseMax(_ a: ContiguousArray<SIMD4<Float>>,
                    _ b: ContiguousArray<SIMD4<Float>>) -> ContiguousArray<SIMD4<Float>> {
    var result = ContiguousArray<SIMD4<Float>>()
    result.reserveCapacity(a.count)
    for i in 0..<a.count {
        result.append(pointwiseMax(a[i], b[i]))
    }
    return result
}
```

---

## In VectorCore

VectorCore's optimized vectors use SIMD4 throughout:

**📍 See:** `Sources/VectorCore/Vectors/Vector512Optimized.swift:110-125`

```swift
/// Access individual elements
@inlinable
public subscript(index: Int) -> Scalar {
    get {
        precondition(index >= 0 && index < 512, "Index out of bounds")
        let vectorIndex = index >> 2   // Divide by 4 (which SIMD4)
        let scalarIndex = index & 3    // Modulo 4 (which lane)
        return storage[vectorIndex][scalarIndex]
    }
    set {
        precondition(index >= 0 && index < 512, "Index out of bounds")
        let vectorIndex = index >> 2
        let scalarIndex = index & 3
        storage[vectorIndex][scalarIndex] = newValue
    }
}
```

**Why bit operations?**
- `index >> 2` is faster than `index / 4` (single shift instruction)
- `index & 3` is faster than `index % 4` (single AND instruction)

These micro-optimizations matter in hot paths that run millions of times.

---

## Performance Tips

1. **Prefer vertical operations.** Keep data in SIMD format as long as possible; reduce only when needed.

2. **Avoid lane-by-lane access in loops.** Don't do `sum += v[0] + v[1] + v[2] + v[3]`; do `sum = v.sum()`.

3. **Match your SIMD width to the hardware.** SIMD4 is optimal for Apple Silicon; larger widths get split.

4. **Align your data.** SIMD loads are fastest when the address is aligned to the SIMD width.

5. **Let the compiler help.** `@inlinable` and `@inline(__always)` ensure SIMD operations don't get wrapped in function calls.

---

## Key Takeaways

1. **Swift's `simd` module is first-class.** Type-safe, expressive, and compiles to efficient machine code.

2. **SIMD4<Float> is 16 bytes.** Matches Apple Silicon's NEON registers exactly.

3. **All arithmetic is lane-wise by default.** `a * b` multiplies corresponding lanes.

4. **Horizontal operations exist but are slower.** `.sum()`, `.min()`, `.max()` reduce across lanes.

5. **Pointer casting enables efficient bulk operations.** Convert `[Float]` to `[SIMD4<Float>]` for processing.

---

## Next Up

Now you know how to use SIMD types. But how do we squeeze out maximum performance? The answer involves writing code that keeps the CPU's pipelines full:

**[→ Writing SIMD-Friendly Code](./03-Writing-SIMD-Friendly-Code.md)**
