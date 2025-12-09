# How Swift Stores Data

> **Reading time:** 10 minutes
> **Prerequisites:** None—this is where we start

---

## The Concept

When you create a variable in Swift, the compiler needs to decide where to put it in memory. This decision affects everything: how fast your code runs, how much memory it uses, and even whether your code is thread-safe.

Swift has two fundamental storage strategies:

1. **Value Types** (`struct`, `enum`, tuple): Data is copied when assigned
2. **Reference Types** (`class`, closures): A pointer is copied; data is shared

This distinction isn't just about semantics—it's about where your bytes actually live.

---

## Why It Matters

### The Cost of Indirection

Consider two ways to store the same data:

```swift
// Value type: Data is inline
struct PointValue {
    var x: Float  // 4 bytes
    var y: Float  // 4 bytes
}                 // Total: 8 bytes, contiguous

// Reference type: Data is elsewhere
class PointReference {
    var x: Float  // 4 bytes
    var y: Float  // 4 bytes
}                 // Object: 8 bytes + 16 bytes overhead
                  // Plus: pointer (8 bytes) at use site
```

Memory layout comparison:

```
Value Type (inline):
┌─────────────────────────────────────┐
│  Stack or Container                 │
│  ┌─────────┬─────────┐              │
│  │  x: 1.0 │  y: 2.0 │  ← Data here │
│  └─────────┴─────────┘              │
└─────────────────────────────────────┘

Reference Type (indirection):
┌─────────────────────────────────────┐      ┌──────────────────────────┐
│  Stack or Container                 │      │  Heap                    │
│  ┌─────────────────────┐            │      │  ┌────────────────────┐  │
│  │  ptr: 0x7fff... ────────────────────────│──│ refCount │ x │ y │  │
│  └─────────────────────┘            │      │  └────────────────────┘  │
└─────────────────────────────────────┘      └──────────────────────────┘
          ↑                                             ↑
      8 bytes                                     24+ bytes
```

Every time you access the reference type's data, the CPU must:
1. Load the pointer
2. Follow it to the heap
3. Load the actual data

That extra hop isn't free—especially when you're doing it millions of times.

---

## The Technique

### Understanding Memory Layout

Swift provides tools to inspect exactly how your types are laid out:

```swift
import Foundation

struct Vector2D {
    var x: Float
    var y: Float
}

print("Size:", MemoryLayout<Vector2D>.size)       // 8 bytes
print("Stride:", MemoryLayout<Vector2D>.stride)   // 8 bytes
print("Alignment:", MemoryLayout<Vector2D>.alignment) // 4 bytes
```

**Key terms:**
- **Size**: How many bytes the data actually occupies
- **Stride**: How many bytes between consecutive elements in an array
- **Alignment**: Where in memory the type can start (must be divisible by this)

### The Padding Reality

Structs aren't always as compact as you'd expect:

```swift
struct Padded {
    var a: Bool   // 1 byte
    var b: Int64  // 8 bytes, needs 8-byte alignment
    var c: Bool   // 1 byte
}

print("Size:", MemoryLayout<Padded>.size)     // 24 bytes!
print("Stride:", MemoryLayout<Padded>.stride) // 24 bytes
```

Wait, 1 + 8 + 1 = 10 bytes. Where did 24 come from?

```
Memory layout of Padded:
┌────┬───────────────┬────────────────────────────────┬────┬───────────────┐
│ a  │    padding    │              b                 │ c  │    padding    │
│ 1B │      7B       │              8B                │ 1B │      7B       │
└────┴───────────────┴────────────────────────────────┴────┴───────────────┘
 0    1               8                                16   17              24
```

The `Int64` must start at an 8-byte boundary. The struct as a whole must have stride divisible by its alignment (8). So we get 14 bytes of padding!

### Fixing Padding with Better Ordering

```swift
struct Compact {
    var b: Int64  // 8 bytes, at offset 0 (aligned)
    var a: Bool   // 1 byte
    var c: Bool   // 1 byte
}                 // 2 bytes padding at end

print("Size:", MemoryLayout<Compact>.size)     // 10 bytes
print("Stride:", MemoryLayout<Compact>.stride) // 16 bytes (next multiple of 8)
```

```
Memory layout of Compact:
┌────────────────────────────────┬────┬────┬──────────────┐
│              b                 │ a  │ c  │   padding    │
│              8B                │ 1B │ 1B │      6B      │
└────────────────────────────────┴────┴────┴──────────────┘
 0                                8    9    10             16
```

Same data, 33% less memory.

💡 **Rule of thumb:** Order struct fields from largest alignment to smallest.

---

## Contiguous vs. Non-Contiguous Storage

For high-performance code, how elements are arranged matters enormously.

### Contiguous: Array, ContiguousArray

```swift
let array = [1.0, 2.0, 3.0, 4.0] as [Float]

// Memory layout:
// ┌──────┬──────┬──────┬──────┐
// │ 1.0  │ 2.0  │ 3.0  │ 4.0  │
// └──────┴──────┴──────┴──────┘
// Elements are adjacent in memory
```

This is cache-friendly: when the CPU loads `array[0]`, it also loads `array[1]`, `array[2]`, etc. into the cache line (we'll explore this in [Why Alignment Matters](./03-Why-Alignment-Matters.md)).

### Non-Contiguous: Array of Reference Types

```swift
class BoxedFloat {
    var value: Float
    init(_ v: Float) { value = v }
}

let array = [BoxedFloat(1.0), BoxedFloat(2.0), BoxedFloat(3.0)]

// Memory layout:
// ┌─────────┬─────────┬─────────┐
// │  ptr0   │  ptr1   │  ptr2   │  ← Array storage
// └────┬────┴────┬────┴────┬────┘
//      │         │         │
//      ▼         ▼         ▼
//   ┌──────┐  ┌──────┐  ┌──────┐
//   │ 1.0  │  │ 2.0  │  │ 3.0  │  ← Objects scattered on heap
//   └──────┘  └──────┘  └──────┘
```

To iterate through the values, the CPU must chase pointers all over the heap. This is called **pointer chasing** and it's slow.

---

## In VectorCore

VectorCore makes deliberate choices to keep data contiguous:

**📍 See:** `Sources/VectorCore/Vectors/Vector512Optimized.swift:25-29`

```swift
public struct Vector512Optimized: Sendable {
    public typealias Scalar = Float

    /// Internal storage as SIMD4 chunks for optimal performance
    public var storage: ContiguousArray<SIMD4<Float>>
```

**Why `ContiguousArray<SIMD4<Float>>`?**

1. **`struct`** — Value type, can live on stack, no heap allocation in many cases
2. **`ContiguousArray`** — Guaranteed contiguous, unlike `Array` which might bridge to NSArray
3. **`SIMD4<Float>`** — Four floats packed together, aligned for vector operations

Memory layout of a 512-dimensional vector:

```
┌──────────────────────────────────────────────────────────────────────┐
│                    ContiguousArray<SIMD4<Float>>                     │
├──────────────┬──────────────┬──────────────┬─────────────────────────┤
│ SIMD4[0]     │ SIMD4[1]     │ SIMD4[2]     │ ...         │ SIMD4[127]│
│ f0,f1,f2,f3  │ f4,f5,f6,f7  │ f8,f9,f10,f11│             │           │
├──────────────┴──────────────┴──────────────┴─────────────────────────┤
│ 128 × 16 bytes = 2048 bytes, contiguous, aligned                     │
└──────────────────────────────────────────────────────────────────────┘
```

All 512 floats are packed together, ready for SIMD operations.

---

## Key Takeaways

1. **Value types store data inline; reference types add indirection.** The extra pointer hop costs time, especially in loops.

2. **Struct field order affects memory usage.** Order fields from largest alignment to smallest to minimize padding.

3. **`MemoryLayout<T>` reveals the truth.** Use it to understand how your types are actually stored.

4. **Contiguous storage enables fast iteration.** `Array` and `ContiguousArray` keep elements adjacent; arrays of objects scatter data across the heap.

5. **VectorCore uses `ContiguousArray<SIMD4<Float>>`** for dense, aligned, cache-friendly storage.

---

## Next Up

Now that you understand how Swift organizes memory, let's explore *where* that memory comes from:

**[→ The Stack and Heap](./02-The-Stack-And-Heap.md)**
