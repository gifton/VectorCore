# Reading a Real Kernel: DotKernels.swift

> **Reading time:** 20 minutes
> **Prerequisites:** Chapters 1-5 (or willingness to reference them)

---

## Introduction

You've learned memory fundamentals, SIMD patterns, numerical stability, unsafe Swift, and performance patterns. Now let's see how they all come together in real production code.

Open `Sources/VectorCore/Operations/Kernels/DotKernels.swift` in your editor and follow along.

**📍 Full path:** `Sources/VectorCore/Operations/Kernels/DotKernels.swift`

---

## The File Header (Lines 1-8)

```swift
//
//  DotKernels.swift
//  VectorCore
//
//  CPU-only, SIMD-optimized dot product kernels for optimized vector types.
//  Mirrors the 4-accumulator, stride-16 pattern used in optimized vectors,
//  exposed as standalone helpers for reuse by higher-level kernels.
//
```

**What you're seeing:**
- This is a kernel file—focused, optimized implementation
- "CPU-only" means no GPU/Metal in this file
- "4-accumulator, stride-16" is the key pattern we'll explore
- "Standalone helpers" means this code is shared by multiple vector operations

---

## The Imports (Lines 10-11)

```swift
import Foundation
import simd
```

**What you're seeing:**
- `Foundation` for basic types
- `simd` for `SIMD4<Float>` and related operations

💡 **Chapter 2 Connection:** The `simd` module provides the type-safe SIMD types we discussed.

---

## The Type Declaration (Lines 13-14)

```swift
@usableFromInline
internal enum DotKernels {
```

**What you're seeing:**
- `@usableFromInline` — Allows this type to be inlined into public functions in other modules
- `internal` — Not directly accessible to library users
- `enum` — A namespace with no instances (just static functions)

💡 **Chapter 5 Connection:** `@usableFromInline` is an optimization attribute that enables cross-module inlining while keeping the API internal.

---

## The Core Implementation (Lines 16-18)

```swift
    // Core implementation: 4 independent accumulators, stride 16 lanes.
    // Accepts storage buffers directly to avoid introducing new protocols.
    @inline(__always)
```

**What you're seeing:**
- The comment explains the optimization strategy upfront
- `@inline(__always)` forces the compiler to inline this function

💡 **Chapter 5 Connection:** Inlining eliminates function call overhead. For a kernel called millions of times, this matters.

---

## The Function Signature (Line 19)

```swift
    private static func dot(
        storageA: ContiguousArray<SIMD4<Float>>,
        storageB: ContiguousArray<SIMD4<Float>>,
        laneCount: Int
    ) -> Float {
```

*Note: In the actual source, this is a single line. Reformatted here for readability.*

**What you're seeing:**
- `private static` — Called only from within this enum
- Parameters take `ContiguousArray<SIMD4<Float>>` directly, not abstract types
- `laneCount` is the number of SIMD4 elements (not the number of floats)

💡 **Chapter 1 Connection:** `ContiguousArray` guarantees contiguous memory layout, essential for SIMD operations.

💡 **Why not generics?** The comment says "avoid introducing new protocols." Generic functions have indirect call overhead; this function is concrete and inlinable.

---

## The Debug Assertions (Lines 20-23)

```swift
        #if DEBUG
        assert(storageA.count == laneCount && storageB.count == laneCount, "Storage count mismatch")
        assert(laneCount % 16 == 0, "Lane count must be multiple of 16.")
        #endif
```

**What you're seeing:**
- Validation wrapped in `#if DEBUG`
- Checks that arrays are the expected size
- Checks that lane count is divisible by 16 (the unroll factor)

💡 **Chapter 5 Connection:** Debug-only assertions provide safety during development with zero overhead in release builds. This is the "zero-cost abstraction" pattern.

💡 **Why multiple of 16?** The main loop processes 16 SIMD4 elements per iteration. Non-multiples would require a cleanup loop.

---

## The Accumulators (Lines 25-28)

```swift
        var acc0 = SIMD4<Float>()
        var acc1 = SIMD4<Float>()
        var acc2 = SIMD4<Float>()
        var acc3 = SIMD4<Float>()
```

**What you're seeing:**
- Four independent accumulator variables
- Each initialized to zero

💡 **Chapter 2 Connection:** This is the "multiple accumulators" pattern for instruction-level parallelism (ILP). The CPU can compute all four additions simultaneously because they don't depend on each other.

```
Without multiple accumulators:
  acc = acc + (a[0] * b[0])  ← must wait for result
  acc = acc + (a[1] * b[1])  ← must wait for result
  acc = acc + (a[2] * b[2])  ← must wait for result
  ...
  (sequential dependency chain)

With 4 accumulators:
  acc0 = acc0 + (a[0] * b[0])  ← independent
  acc1 = acc1 + (a[1] * b[1])  ← independent
  acc2 = acc2 + (a[2] * b[2])  ← independent
  acc3 = acc3 + (a[3] * b[3])  ← independent
  (4 operations in flight simultaneously)
```

---

## The Main Loop (Lines 30-54)

```swift
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
```

**What you're seeing:**
- `stride(from: 0, to: laneCount, by: 16)` — Process 16 SIMD4 values per iteration
- Four "blocks" of 4 operations each
- Each block uses all 4 accumulators

Let's unpack the math:
- Each iteration processes 16 SIMD4 values
- Each SIMD4 has 4 floats
- So each iteration processes 64 floats
- For a 512-dimensional vector: 128 SIMD4 values / 16 per iteration = 8 iterations

💡 **Chapter 2 Connection:** This is the "loop unrolling" pattern. Instead of processing 1 element per iteration, we process 16. This:
- Reduces loop overhead (fewer branches)
- Keeps more data in flight (better pipelining)
- Enables better instruction scheduling

💡 **Chapter 5 Connection:** Sequential access (`i+0, i+1, i+2...`) enables hardware prefetching. The CPU predicts we'll need `i+16, i+17...` and loads them before we ask.

### Visualizing the Pipeline

```
Cycle:    1    2    3    4    5    6    7    8    9   10
        ┌────┬────┬────┬────┐
Block 0 │Load│Mul │Add │    │
        └────┴────┴────┴────┘
             ┌────┬────┬────┬────┐
Block 1      │Load│Mul │Add │    │
             └────┴────┴────┴────┘
                  ┌────┬────┬────┬────┐
Block 2           │Load│Mul │Add │    │
                  └────┴────┴────┴────┘
                       ┌────┬────┬────┬────┐
Block 3                │Load│Mul │Add │    │
                       └────┴────┴────┴────┘

Each block starts before the previous finishes.
All execution units stay busy.
```

---

## The Final Reduction (Lines 56-57)

```swift
        let combined = (acc0 + acc1) + (acc2 + acc3)
        return combined.sum()
```

**What you're seeing:**
- Tree reduction to combine accumulators
- Final horizontal sum

💡 **Chapter 3 Connection:** This is tree reduction. Compare:

```
Linear: acc0 + acc1 + acc2 + acc3
        └──┬──┘
           └────┬────┘
                └─────┬─────┘
                    result
        (3 sequential operations)

Tree: (acc0 + acc1) + (acc2 + acc3)
      └────┬────┘     └────┬────┘
           └──────┬───────┘
                result
        (2 levels, some parallelism)
```

💡 **Chapter 2 Connection:** `.sum()` is the horizontal operation we discussed. It's the only horizontal op in the entire kernel—we delay it until the very end.

---

## The Public Entry Points (Lines 60-80)

```swift
    // MARK: - Public per-dimension entry points

    @usableFromInline @inline(__always)
    static func dot384(_ a: Vector384Optimized, _ b: Vector384Optimized) -> Float {
        dot(storageA: a.storage, storageB: b.storage, laneCount: 96)
    }

    @usableFromInline @inline(__always)
    static func dot512(_ a: Vector512Optimized, _ b: Vector512Optimized) -> Float {
        dot(storageA: a.storage, storageB: b.storage, laneCount: 128)
    }

    @usableFromInline @inline(__always)
    static func dot768(_ a: Vector768Optimized, _ b: Vector768Optimized) -> Float {
        dot(storageA: a.storage, storageB: b.storage, laneCount: 192)
    }

    @usableFromInline @inline(__always)
    static func dot1536(_ a: Vector1536Optimized, _ b: Vector1536Optimized) -> Float {
        dot(storageA: a.storage, storageB: b.storage, laneCount: 384)
    }
```

**What you're seeing:**
- Concrete implementations for each vector dimension
- Each calls the core `dot` function with the appropriate lane count
- `@usableFromInline @inline(__always)` ensures these get inlined at call sites

💡 **Why separate functions per dimension?**
- Avoids generic dispatch overhead
- Lane count is a compile-time constant, enabling better optimization
- Type safety—can't accidentally dot a 512 with a 768

**Lane counts:**
- 384 dimensions / 4 = 96 SIMD4 values
- 512 dimensions / 4 = 128 SIMD4 values
- 768 dimensions / 4 = 192 SIMD4 values
- 1536 dimensions / 4 = 384 SIMD4 values

---

## Summary: The Complete Pattern

Let's trace a full dot product call:

```
User code:
  let similarity = v1.dotProduct(v2)
              │
              ▼
Vector512Optimized.dotProduct(_:)  [Vector512Optimized.swift:180-182]
  return DotKernels.dot512(self, other)
              │
              ▼
DotKernels.dot512(_:_:)  [DotKernels.swift:67-69]
  dot(storageA: a.storage, storageB: b.storage, laneCount: 128)
              │
              ▼
DotKernels.dot(storageA:storageB:laneCount:)  [DotKernels.swift:19-58]
  1. Initialize 4 accumulators
  2. Loop 8 times (128 / 16)
     - Each iteration: 16 SIMD4 multiply-adds
     - Each SIMD4 operation: 4 floats
     - Total per iteration: 64 floats
  3. Tree-reduce accumulators
  4. Horizontal sum
              │
              ▼
Result: single Float value
```

**By the numbers:**
- 512 floats processed
- 8 loop iterations
- 128 SIMD4 multiply-add operations
- ~100 nanoseconds total

---

## Techniques Checklist

Let's verify we saw every technique from the earlier chapters:

| Technique | Chapter | Where in DotKernels |
|-----------|---------|---------------------|
| Value types | 1 | Vectors passed by value |
| Contiguous storage | 1 | `ContiguousArray<SIMD4<Float>>` |
| SIMD operations | 2 | `storageA[i] * storageB[i]` |
| Multiple accumulators | 2 | `acc0, acc1, acc2, acc3` |
| Loop unrolling | 2 | `stride(..., by: 16)` |
| Tree reduction | 3 | `(acc0 + acc1) + (acc2 + acc3)` |
| Buffer access | 4 | Direct storage access |
| Debug-only checks | 5 | `#if DEBUG assert(...)` |
| Force inlining | 5 | `@inline(__always)` |
| Sequential access | 5 | `i+0, i+1, i+2...` |

---

## Challenge: Read NormalizeKernels.swift

Now try reading `NormalizeKernels.swift` on your own. Look for:

1. **Two-pass algorithm:** Find max, then compute
2. **NaN handling:** Explicit checks in each lane
3. **Infinity handling:** Edge case guards
4. **Same SIMD patterns:** 4 accumulators, stride-16

You have all the knowledge to understand it!

---

## Congratulations!

You've completed the VectorCore Learning Guide. You now understand:

- **Memory:** How data is stored, aligned, and accessed
- **SIMD:** Processing multiple values with single instructions
- **Numerical Computing:** Keeping floating-point math correct
- **Unsafe Swift:** Working with raw memory safely
- **Performance:** Measuring and optimizing effectively

These skills transfer far beyond VectorCore—to any high-performance Swift code you'll write.

---

## What's Next?

1. **Read more VectorCore code.** Try `EuclideanKernels.swift`, `BatchOperations.swift`

2. **Run the benchmarks.** See the performance characteristics firsthand:
   ```bash
   swift build -c release
   ./.build/release/vectorcore-bench
   ```

3. **Write your own kernel.** Pick an operation and implement it using these patterns

4. **Profile with Instruments.** See how the code behaves on real hardware

---

*Congratulations on completing the VectorCore Learning Guide!*

**[← Back to Welcome](../00-Welcome.md)**
