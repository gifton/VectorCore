# Writing SIMD-Friendly Code

> **Reading time:** 15 minutes
> **Prerequisites:** [SIMD in Swift](./02-SIMD-In-Swift.md)

---

## The Concept

Using SIMD types is step one. Writing code that actually runs 4x faster is step two.

The gap between "uses SIMD" and "gets SIMD speedups" comes down to:

1. **Data dependencies:** Can the CPU work on multiple things at once?
2. **Memory access patterns:** Is data arriving fast enough?
3. **Pipeline utilization:** Are all execution units busy?

This guide shows you the patterns VectorCore uses to achieve near-theoretical speedups.

---

## Why It Matters

### The Dependency Problem

Consider this simple dot product:

```swift
func naiveDot(_ a: [SIMD4<Float>], _ b: [SIMD4<Float>]) -> Float {
    var acc = SIMD4<Float>()
    for i in 0..<a.count {
        acc += a[i] * b[i]  // Each iteration depends on the previous acc
    }
    return acc.sum()
}
```

Looks vectorized, right? But look at the dependency chain:

```
Iteration 1:  acc = acc + (a[0] * b[0])
                     ↑
Iteration 2:  acc = acc + (a[1] * b[1])
                     ↑
Iteration 3:  acc = acc + (a[2] * b[2])
              ...
```

Each iteration must wait for the previous `acc` value. The CPU has multiple execution units, but they're sitting idle waiting for the accumulator.

### The Solution: Multiple Accumulators

```swift
func betterDot(_ a: [SIMD4<Float>], _ b: [SIMD4<Float>]) -> Float {
    var acc0 = SIMD4<Float>()
    var acc1 = SIMD4<Float>()
    var acc2 = SIMD4<Float>()
    var acc3 = SIMD4<Float>()

    for i in stride(from: 0, to: a.count, by: 4) {
        acc0 += a[i+0] * b[i+0]  // Independent!
        acc1 += a[i+1] * b[i+1]  // No dependency on acc0
        acc2 += a[i+2] * b[i+2]  // No dependency on acc1
        acc3 += a[i+3] * b[i+3]  // No dependency on acc2
    }

    return ((acc0 + acc1) + (acc2 + acc3)).sum()
}
```

Now the dependency graph looks like:

```
acc0: ─────○─────○─────○─────○────→
acc1: ─────○─────○─────○─────○────→
acc2: ─────○─────○─────○─────○────→
acc3: ─────○─────○─────○─────○────→
                                    \
                                     └──→ combine at end
```

Four independent chains! The CPU can work on all four simultaneously.

---

## The Technique

### Pattern 1: Loop Unrolling with Multiple Accumulators

This is VectorCore's core pattern:

**📍 See:** `Sources/VectorCore/Operations/Kernels/DotKernels.swift:19-58`

```swift
@inline(__always)
private static func dot(
    storageA: ContiguousArray<SIMD4<Float>>,
    storageB: ContiguousArray<SIMD4<Float>>,
    laneCount: Int
) -> Float {
    // 4 independent accumulators
    var acc0 = SIMD4<Float>()
    var acc1 = SIMD4<Float>()
    var acc2 = SIMD4<Float>()
    var acc3 = SIMD4<Float>()

    // Process 16 SIMD4 values per iteration (64 floats)
    for i in stride(from: 0, to: laneCount, by: 16) {
        // Block 0
        acc0 += storageA[i+0] * storageB[i+0]
        acc1 += storageA[i+1] * storageB[i+1]
        acc2 += storageA[i+2] * storageB[i+2]
        acc3 += storageA[i+3] * storageB[i+3]

        // Block 1
        acc0 += storageA[i+4] * storageB[i+4]
        acc1 += storageA[i+5] * storageB[i+5]
        acc2 += storageA[i+6] * storageB[i+6]
        acc3 += storageA[i+7] * storageB[i+7]

        // Block 2
        acc0 += storageA[i+8] * storageB[i+8]
        acc1 += storageA[i+9] * storageB[i+9]
        acc2 += storageA[i+10] * storageB[i+10]
        acc3 += storageA[i+11] * storageB[i+11]

        // Block 3
        acc0 += storageA[i+12] * storageB[i+12]
        acc1 += storageA[i+13] * storageB[i+13]
        acc2 += storageA[i+14] * storageB[i+14]
        acc3 += storageA[i+15] * storageB[i+15]
    }

    // Combine at the end (one horizontal operation)
    let combined = (acc0 + acc1) + (acc2 + acc3)
    return combined.sum()
}
```

**Why this works:**

1. **4 accumulators:** Each can be updated independently
2. **Stride-16:** Each iteration processes 64 floats (16 SIMD4 values)
3. **Sequential access:** `i+0`, `i+1`, `i+2`... is cache-friendly
4. **One reduction:** `.sum()` only at the very end

### Visualizing the Pipeline

Modern CPUs are pipelined. Each instruction goes through stages:

```
Cycle:    1    2    3    4    5    6    7    8
        ┌────┬────┬────┬────┐
acc0:   │Load│Mul │Add │Done│
        └────┴────┴────┴────┘
             ┌────┬────┬────┬────┐
acc1:        │Load│Mul │Add │Done│
             └────┴────┴────┴────┘
                  ┌────┬────┬────┬────┐
acc2:             │Load│Mul │Add │Done│
                  └────┴────┴────┴────┘
                       ┌────┬────┬────┬────┐
acc3:                  │Load│Mul │Add │Done│
                       └────┴────┴────┴────┘
```

While `acc0`'s multiply is executing, `acc1`'s load can start. All pipeline stages stay busy.

### Pattern 2: Reduction Tree

When combining results, the order matters:

```swift
// ❌ Linear reduction (longer dependency chain)
let sum = acc0 + acc1 + acc2 + acc3
//        └──┬──┘
//           └────┬────┘
//                └─────┬─────┘
//                      result
// 3 sequential additions

// ✅ Tree reduction (shorter dependency chain)
let sum = (acc0 + acc1) + (acc2 + acc3)
//        └────┬────┘     └────┬────┘
//             └───────┬───────┘
//                   result
// 2 levels of addition (can parallelize first level)
```

### Pattern 3: Avoid Branches in Hot Loops

Branches are expensive because they can cause pipeline stalls:

```swift
// ❌ Branching per element
for i in 0..<a.count {
    if a[i] > 0 {
        result += a[i] * b[i]
    }
}

// ✅ Branchless with masking
for i in 0..<a.count {
    let mask = a[i] .> SIMD4<Float>(repeating: 0)
    let product = a[i] * b[i]
    // Use mask to conditionally add (implementation varies)
}

// ✅ Even better: handle edge cases outside the loop
precondition(a.count == b.count)  // Check once, not per iteration
for i in 0..<a.count {
    result += a[i] * b[i]
}
```

VectorCore uses `#if DEBUG` for assertions, removing them in release builds:

```swift
#if DEBUG
assert(storageA.count == laneCount, "Storage count mismatch")
#endif
```

### Pattern 4: @inline(__always) for Hot Paths

Function calls have overhead. For kernels called millions of times, force inlining:

```swift
@inline(__always)
@usableFromInline
static func dot512(_ a: Vector512Optimized, _ b: Vector512Optimized) -> Float {
    dot(storageA: a.storage, storageB: b.storage, laneCount: 128)
}
```

Without inlining:
```
Call site → Push arguments → Jump to function → Execute → Return
           ~~~~~~~~~~~overhead~~~~~~~~~~~
```

With `@inline(__always)`:
```
Call site: [function body inserted here, no call overhead]
```

### Pattern 5: Sequential Memory Access

CPUs predict memory access patterns. Sequential access triggers prefetching:

```swift
// ✅ Sequential access - CPU prefetches ahead
for i in 0..<n {
    sum += data[i]
}
// CPU sees: 0, 1, 2, 3... and prefetches 4, 5, 6, 7...

// ❌ Random access - CPU can't predict
for i in 0..<n {
    sum += data[indices[i]]  // Where does indices[i] point?
}
// CPU can't prefetch, cache misses everywhere
```

VectorCore's storage guarantees sequential access:

```swift
public var storage: ContiguousArray<SIMD4<Float>>
// Elements at indices 0, 1, 2, 3... are adjacent in memory
```

---

## In VectorCore

### The Complete DotKernels Pattern

Let's trace through VectorCore's dot product for a 512-dimensional vector:

```
Vector512Optimized.dotProduct(_:)
         │
         ▼
DotKernels.dot512(self, other)
         │
         ▼
DotKernels.dot(storageA, storageB, laneCount: 128)
         │
         ├── Initialize 4 accumulators
         │
         ├── Main loop: 8 iterations
         │   └── Each iteration: 16 SIMD4 multiply-adds
         │       └── Total: 128 SIMD4 operations = 512 floats
         │
         ├── Combine accumulators: tree reduction
         │
         └── Final .sum(): 1 horizontal operation
```

**Numbers:**
- 512 floats / 4 lanes = 128 SIMD4 operations
- 128 operations / 16 per iteration = 8 loop iterations
- 4 accumulators × 4 operations each = 16 operations per iteration ✓

### The NormalizeKernels Pattern

Normalization adds complexity with Kahan's two-pass algorithm:

**📍 See:** `Sources/VectorCore/Operations/Kernels/NormalizeKernels.swift`

```swift
// Phase 1: Find maximum (for numerical stability)
var maxVec = SIMD4<Float>(repeating: 0)
for i in 0..<laneCount {
    let absVec = abs(storage[i])
    maxVec = pointwiseMax(maxVec, absVec)
}
let maxAbs = max(max(maxVec[0], maxVec[1]), max(maxVec[2], maxVec[3]))

// Phase 2: Scale and sum squares (same 4-accumulator pattern)
let scale = 1.0 / maxAbs
var acc0 = SIMD4<Float>()
var acc1 = SIMD4<Float>()
var acc2 = SIMD4<Float>()
var acc3 = SIMD4<Float>()

for i in stride(from: 0, to: laneCount, by: 16) {
    let s0 = storage[i+0] * simdScale
    // ... same unrolling pattern ...
    acc0 += s0 * s0
    // ...
}
```

Same optimization patterns, applied to a more complex algorithm.

---

## Checklist: Is Your Code SIMD-Friendly?

| Question | If Yes | If No |
|----------|--------|-------|
| Is data contiguous in memory? | ✅ Good | Consider restructuring |
| Are operations independent between iterations? | ✅ Use multiple accumulators | Look for parallelism |
| Are you avoiding branches in loops? | ✅ Good | Use masks or move checks outside |
| Is the inner loop inlined? | ✅ Good | Add `@inline(__always)` |
| Are reductions at the end, not per-iteration? | ✅ Good | Delay until after loop |
| Are you processing enough data? | ✅ Good | SIMD overhead may not be worth it for small N |

---

## Key Takeaways

1. **Multiple accumulators break dependency chains.** 4 independent accumulators can keep 4 execution units busy.

2. **Unroll loops to amortize overhead.** Processing 16 SIMD4 values per iteration (64 floats) is more efficient than 1.

3. **Tree reduction beats linear reduction.** `(a+b) + (c+d)` has 2 levels; `a+b+c+d` has 3.

4. **`@inline(__always)` eliminates call overhead.** Critical for kernels called millions of times.

5. **Sequential access enables prefetching.** Contiguous memory access lets the CPU predict and prefetch.

6. **Debug assertions should be conditional.** Use `#if DEBUG` for checks that would slow down release builds.

---

## Chapter Complete!

You now understand how to:
- Use Swift's SIMD types effectively
- Write code that achieves near-theoretical speedups
- Recognize patterns that defeat SIMD optimization

Next, we'll explore a different kind of optimization—keeping your calculations **numerically correct**:

**[→ Chapter 3: Numerical Computing](../03-Numerical-Computing/README.md)**
