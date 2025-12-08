# Why Memory Alignment Matters

> **Reading time:** 10 minutes
> **Prerequisites:** [The Stack and Heap](./02-The-Stack-And-Heap.md)

---

## The Concept

When you allocate memory, the CPU doesn't fetch individual bytes—it fetches entire **cache lines** (64 bytes on modern processors). Think of it like buying eggs: you can't buy 3 eggs, you buy the whole carton.

**Memory alignment** means placing your data so it starts at addresses that are multiples of a specific number (usually 16, 32, or 64 bytes).

When data is properly aligned:
- SIMD operations work at full speed
- Memory accesses don't cross cache line boundaries
- The CPU's prefetcher can work effectively

When data is misaligned:
- SIMD operations may crash or run slowly
- Single accesses may require *two* cache line fetches
- Performance drops by 2-5x for memory-bound operations

---

## Why It Matters

### The Problem: Cache Line Splits

Imagine you have 16 floats (64 bytes) that you want to process together with SIMD:

```
                        Cache Line Boundary (address divisible by 64)
                               │
    ┌──────────────────────────┼──────────────────────┐
    │ float float float float  │ float float float... │
    │  0     1     2    3     [SPLIT]  5     6        │
    └──────────────────────────┼──────────────────────┘
                               │
           Cache Line A                Cache Line B

❌ Misaligned: Your 16 floats span TWO cache lines
```

When data crosses a cache line boundary:
1. CPU must fetch **two** cache lines instead of one
2. Memory bandwidth is doubled
3. Performance drops significantly

### The Solution: Aligned Allocation

```
    Cache Line Boundary
           │
    ┌──────┼─────────────────────────────────────────────────────────────┐
    │      │ float float float float float float float float ... (x16)  │
    │      │  0     1     2     3     4     5     6     7               │
    └──────┼─────────────────────────────────────────────────────────────┘
           │
           ↑
           Address is divisible by 64

✅ Aligned: All 16 floats fit in ONE cache line
```

### Real Performance Impact

Here's what misalignment costs in practice:

| Operation | Aligned | Misaligned | Slowdown |
|-----------|---------|------------|----------|
| Load 64 bytes | 1 fetch | 2 fetches | 2x |
| SIMD4 operation | ~1ns | ~3ns | 3x |
| 512-dim dot product | ~100ns | ~300-400ns | 3-4x |

For SIMD operations specifically, some older CPUs would **crash** on misaligned access. Modern CPUs handle it, but slowly.

---

## The Technique

### Swift Arrays Don't Guarantee Alignment

Standard Swift arrays are aligned to the element type, not to cache lines:

```swift
// ⚠️ Swift Arrays don't guarantee 64-byte alignment
var floats = [Float](repeating: 0, count: 16)

floats.withUnsafeBufferPointer { buffer in
    let address = Int(bitPattern: buffer.baseAddress!)
    print("Address: \(address)")
    print("Aligned to 64? \(address % 64 == 0)")  // Usually: false
    print("Aligned to 16? \(address % 16 == 0)")  // Maybe: true
    print("Aligned to 4?  \(address % 4 == 0)")   // Always: true (Float alignment)
}
```

Swift guarantees alignment to `MemoryLayout<Float>.alignment` (4 bytes), but not more.

### Requesting Specific Alignment with posix_memalign

The C function `posix_memalign` allocates memory at a specific alignment:

```swift
import Foundation

func allocateAligned(count: Int, alignment: Int) -> UnsafeMutablePointer<Float>? {
    var pointer: UnsafeMutableRawPointer?
    let byteCount = count * MemoryLayout<Float>.stride

    // posix_memalign: "Give me memory aligned to this boundary"
    let result = posix_memalign(&pointer, alignment, byteCount)

    guard result == 0, let ptr = pointer else { return nil }
    return ptr.assumingMemoryBound(to: Float.self)
}

// Usage
let aligned = allocateAligned(count: 16, alignment: 64)!
// address % 64 == 0, guaranteed
```

### ⚠️ Critical: Matching Allocator and Deallocator

This is a common source of bugs:

```swift
// Memory from posix_memalign MUST be freed with free()
// NOT with Swift's .deallocate()

let ptr = allocateAligned(count: 100, alignment: 64)!

// ❌ WRONG - causes heap corruption
ptr.deallocate()

// ✅ CORRECT
free(ptr)
```

**Why?** `posix_memalign` uses the C allocator's bookkeeping. Swift's `.deallocate()` uses Swift's allocator. They track allocations differently. Mixing them corrupts the heap—often silently, crashing much later in unrelated code.

💡 **Rule:** Always pair allocators with their matching deallocators.

| Allocator | Deallocator |
|-----------|-------------|
| `malloc()` / `posix_memalign()` | `free()` |
| `UnsafeMutablePointer.allocate()` | `.deallocate()` |
| `UnsafeMutableRawPointer.allocate()` | `.deallocate()` |

---

## Alignment Requirements by Platform

Different SIMD instruction sets have different requirements:

| Instruction Set | Platform | Minimum Alignment |
|-----------------|----------|-------------------|
| NEON | ARM (iPhone, M-series Mac) | 16 bytes |
| SSE | Intel x86 | 16 bytes |
| AVX | Intel x86 | 32 bytes |
| AVX-512 | Intel Xeon | 64 bytes |

VectorCore uses 64-byte alignment universally because:
1. It satisfies all SIMD requirements
2. It matches the cache line size
3. It prevents cache line splits for any operation

---

## Detecting Alignment Problems

### At Runtime

```swift
func checkAlignment<T>(_ pointer: UnsafePointer<T>, expected: Int) -> Bool {
    Int(bitPattern: pointer) % expected == 0
}

// Usage
myBuffer.withUnsafeBufferPointer { ptr in
    if !checkAlignment(ptr.baseAddress!, expected: 64) {
        print("Warning: Buffer is not cache-line aligned!")
    }
}
```

### With Sanitizers

Address Sanitizer can catch some alignment issues:

```bash
swift build -c debug -Xswiftc -sanitize=address
swift test -Xswiftc -sanitize=address
```

---

## In VectorCore

VectorCore wraps aligned allocation safely:

**📍 See:** `Sources/VectorCore/Storage/AlignedMemory.swift:43-72`

```swift
public enum AlignedMemory {
    /// Platform-specific optimal alignment for SIMD operations
    public static var optimalAlignment: Int {
        #if arch(arm64)
        return 64  // Apple Silicon: cache line size
        #elseif arch(x86_64)
        return 64  // Intel: works for AVX-512
        #else
        return 16  // Conservative fallback
        #endif
    }

    /// Allocate aligned memory for any type
    @inlinable
    public static func allocateAligned<T>(
        type: T.Type,
        count: Int,
        alignment: Int = optimalAlignment
    ) throws -> UnsafeMutablePointer<T> {
        precondition(alignment > 0 && (alignment & (alignment - 1)) == 0,
                     "Alignment must be a power of 2")
        precondition(alignment >= minimumAlignment,
                     "Alignment must be at least \(minimumAlignment) bytes")

        var rawPtr: UnsafeMutableRawPointer?
        let byteCount = count * MemoryLayout<T>.stride
        let result = posix_memalign(&rawPtr, alignment, byteCount)

        guard result == 0, let ptr = rawPtr else {
            throw VectorError.allocationFailed(size: byteCount, reason: "posix_memalign error \(result)")
        }

        return ptr.assumingMemoryBound(to: T.self)
    }

    /// Deallocate memory previously allocated via posix_memalign
    ///
    /// - Important: Memory allocated with `posix_memalign` MUST be freed with `free()`,
    ///   not with Swift's `.deallocate()`.
    @inlinable
    public static func deallocate<T>(_ ptr: UnsafeMutablePointer<T>) {
        free(UnsafeMutableRawPointer(ptr))
    }
}
```

**Why VectorCore wraps this:**

1. **Correct by construction:** You can't accidentally use `.deallocate()` if you go through `AlignedMemory.deallocate()`

2. **Platform-aware:** Automatically uses the right alignment for the current CPU

3. **Error handling:** Converts C-style error codes to Swift errors

Every optimized vector type uses properly aligned storage:

```swift
// Vector512Optimized uses ContiguousArray<SIMD4<Float>>
// SIMD4 is 16-byte aligned by Swift's guarantees
// ContiguousArray ensures contiguous layout
// Combined: efficient SIMD access without manual alignment
```

For more advanced use cases (batch operations, GPU transfers), VectorCore allocates explicitly aligned buffers using this API.

---

## Summary: The Memory Hierarchy

Putting together everything from this chapter:

```
┌───────────────────────────────────────────────────────────────────┐
│                         YOUR CODE                                 │
│                                                                   │
│   let v = Vector512Optimized(...)                                 │
│                                                                   │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│                        VALUE TYPE                                 │
│                                                                   │
│   Struct on stack → ContiguousArray (CoW) → SIMD4<Float>         │
│                                                                   │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│                    HEAP STORAGE                                   │
│                                                                   │
│   64-byte aligned │ SIMD4 │ SIMD4 │ SIMD4 │ ... │ contiguous     │
│                                                                   │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│                    CACHE LINE                                     │
│                                                                   │
│   64 bytes fetched at once │ prefetcher predicts next access      │
│                                                                   │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│                      CPU                                          │
│                                                                   │
│   SIMD unit processes 4 floats per instruction                    │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

1. **CPUs fetch cache lines (64 bytes), not individual bytes.** Misalignment forces extra fetches.

2. **Swift Arrays guarantee element alignment, not SIMD alignment.** Use explicit aligned allocation for high-performance code.

3. **`posix_memalign` + `free()` is the pattern.** Never mix C allocation with Swift deallocation.

4. **64-byte alignment covers all modern SIMD** and matches cache line size.

5. **VectorCore handles alignment for you** through `AlignedMemory` and `ContiguousArray<SIMD4<Float>>`.

---

## Chapter Complete!

You now understand the memory foundation that makes fast code possible:
- How Swift stores data (inline vs. indirection)
- Where allocations happen (stack vs. heap)
- Why alignment matters (cache lines and SIMD)

Next, we'll build on this foundation to explore **SIMD**—the technique that lets us process multiple values in a single instruction:

**[→ Chapter 2: SIMD Demystified](../02-SIMD-Demystified/README.md)**
