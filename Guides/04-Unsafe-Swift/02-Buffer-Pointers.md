# Buffer Pointers

> **Reading time:** 10 minutes
> **Prerequisites:** [Pointer Primer](./01-Pointer-Primer.md)

---

## The Concept

A buffer pointer represents a contiguous region of memory containing multiple elements. It's a pointer plus a count:

```
UnsafeBufferPointer<Float>
┌─────────────────────────────────────────────────────────────────┐
│  baseAddress: 0x7fff5fbff900                                    │
│  count: 4                                                       │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────┬───────────┬───────────┬───────────┐
│   1.0     │   2.0     │   3.0     │   4.0     │
└───────────┴───────────┴───────────┴───────────┘
  offset 0    offset 1    offset 2    offset 3
```

Unlike raw pointers, buffer pointers:
- Know their size (`count` property)
- Conform to `Collection` (can use `for-in`, `map`, etc.)
- Can be sliced safely (with bounds knowledge)

---

## Why It Matters

Buffer pointers bridge safe Swift collections and raw memory access:

```swift
let array: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]

// Cross into unsafe territory
array.withUnsafeBufferPointer { buffer in
    // We know we have 8 elements
    print(buffer.count)  // 8

    // We can iterate
    for value in buffer {
        print(value)
    }

    // We can subscript (without bounds checking in release)
    print(buffer[0], buffer[7])

    // We can pass to SIMD operations
    processAsSimd(buffer)
}
```

---

## The Technique

### Creating Buffer Pointers

**From Arrays:**
```swift
var array = [1.0, 2.0, 3.0, 4.0] as [Float]

// Read-only
array.withUnsafeBufferPointer { buffer in
    let first = buffer[0]
}

// Read-write
array.withUnsafeMutableBufferPointer { buffer in
    buffer[0] = 42.0
}
```

**From ContiguousArray (guaranteed contiguous):**
```swift
var contiguous = ContiguousArray<Float>([1, 2, 3, 4])

contiguous.withUnsafeBufferPointer { buffer in
    // Guaranteed to be contiguous—no bridging to NSArray
}
```

**Allocating Manually:**
```swift
// Allocate 100 floats
let buffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: 100)
defer { buffer.deallocate() }  // Don't forget!

// Initialize
buffer.initialize(repeating: 0.0)

// Use
buffer[0] = 42.0

// Deinitialize before deallocating (for non-trivial types)
buffer.deinitialize(count: 100)
```

### Buffer Operations

**Iteration:**
```swift
buffer.withUnsafeBufferPointer { ptr in
    // For-in loop
    for value in ptr {
        print(value)
    }

    // With indices
    for i in ptr.indices {
        print(ptr[i])
    }

    // Functional style
    let sum = ptr.reduce(0, +)
}
```

**Slicing:**
```swift
let slice = buffer[4..<8]  // UnsafeBufferPointer<Float>.SubSequence
// Note: This is a view, not a copy

// To get a new buffer pointer from a slice:
let sliceBuffer = UnsafeBufferPointer(rebasing: slice)
```

**Copying Between Buffers:**
```swift
sourceBuffer.withUnsafeBufferPointer { src in
    destBuffer.withUnsafeMutableBufferPointer { dst in
        // Method 1: Element by element
        for i in 0..<src.count {
            dst[i] = src[i]
        }

        // Method 2: Memory copy (faster for large copies)
        dst.baseAddress!.update(from: src.baseAddress!, count: src.count)
    }
}
```

### Reinterpreting as Different Types

The powerful (and dangerous) capability:

```swift
let floats: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]

floats.withUnsafeBufferPointer { floatBuffer in
    // View as SIMD4<Float>
    let rawPtr = UnsafeRawPointer(floatBuffer.baseAddress!)
    let simd4Ptr = rawPtr.bindMemory(to: SIMD4<Float>.self, capacity: 2)
    let simd4Buffer = UnsafeBufferPointer(start: simd4Ptr, count: 2)

    print(simd4Buffer[0])  // SIMD4<Float>(1, 2, 3, 4)
    print(simd4Buffer[1])  // SIMD4<Float>(5, 6, 7, 8)
}
```

---

## In VectorCore

### Exposing Buffer Access

VectorCore's vector types expose buffer access for interoperability:

**📍 See:** `Sources/VectorCore/Protocols/VectorProtocol.swift:61-68`

```swift
/// Access storage for reading (enables SIMD optimizations)
func withUnsafeBufferPointer<R>(
    _ body: (UnsafeBufferPointer<Scalar>) throws -> R
) rethrows -> R

/// Access storage for writing (enables SIMD optimizations)
mutating func withUnsafeMutableBufferPointer<R>(
    _ body: (UnsafeMutableBufferPointer<Scalar>) throws -> R
) rethrows -> R
```

This allows:

```swift
let vector = try! Vector512Optimized([Float](repeating: 1, count: 512))

// Read access
vector.withUnsafeBufferPointer { buffer in
    // buffer is UnsafeBufferPointer<Float> with 512 elements
    let sum = buffer.reduce(0, +)
}

// Write access
var mutableVector = vector
mutableVector.withUnsafeMutableBufferPointer { buffer in
    for i in buffer.indices {
        buffer[i] *= 2
    }
}
```

### Converting toArray()

**📍 See:** `Sources/VectorCore/Vectors/Vector512Optimized.swift:134-145`

```swift
/// Convert to array
public func toArray() -> [Scalar] {
    var result = [Scalar]()
    result.reserveCapacity(512)

    storage.withUnsafeBufferPointer { buffer in
        let floatBuffer = UnsafeRawPointer(buffer.baseAddress!)
            .bindMemory(to: Float.self, capacity: 512)
        result.append(contentsOf: UnsafeBufferPointer(start: floatBuffer, count: 512))
    }

    return result
}
```

The `append(contentsOf:)` with a buffer pointer is highly optimized—it becomes a memory copy internally.

### Initializing from Array

**📍 See:** `Sources/VectorCore/Vectors/Vector512Optimized.swift:50-75`

```swift
@inlinable
public init(_ array: [Scalar]) throws {
    guard array.count == 512 else {
        throw VectorError.dimensionMismatch(expected: 512, actual: array.count)
    }

    storage = ContiguousArray<SIMD4<Float>>()
    storage.reserveCapacity(128)

    // Use bulk memory operations for efficient initialization
    array.withUnsafeBufferPointer { buffer in
        guard let baseAddress = buffer.baseAddress else {
            storage = ContiguousArray(repeating: SIMD4<Float>(), count: 128)
            return
        }

        // Cast the buffer to SIMD4 chunks for direct copying
        let simd4Buffer = UnsafeRawPointer(baseAddress).bindMemory(
            to: SIMD4<Float>.self,
            capacity: 128
        )

        // Bulk append using unsafe buffer
        storage.append(contentsOf: UnsafeBufferPointer(start: simd4Buffer, count: 128))
    }
}
```

**Why this is fast:**
1. `reserveCapacity(128)` — single allocation
2. `bindMemory` — reinterprets existing data, no copy
3. `append(contentsOf:)` — bulk copy from buffer

---

## Common Patterns

### Pattern 1: Process in Chunks

```swift
func processChunked(_ data: [Float], chunkSize: Int) {
    data.withUnsafeBufferPointer { buffer in
        var offset = 0
        while offset < buffer.count {
            let remaining = buffer.count - offset
            let size = min(chunkSize, remaining)

            let chunk = UnsafeBufferPointer(
                start: buffer.baseAddress! + offset,
                count: size
            )

            processChunk(chunk)
            offset += size
        }
    }
}
```

### Pattern 2: Parallel Read-Write

```swift
func transform(_ source: [Float], into dest: inout [Float]) {
    source.withUnsafeBufferPointer { src in
        dest.withUnsafeMutableBufferPointer { dst in
            for i in 0..<src.count {
                dst[i] = src[i] * 2
            }
        }
    }
}
```

### Pattern 3: Zero-Copy Conversion

```swift
extension Vector512Optimized {
    /// Create a view of existing data without copying
    static func withUnsafeView<R>(
        of buffer: UnsafeBufferPointer<Float>,
        _ body: (Vector512Optimized) throws -> R
    ) rethrows -> R {
        precondition(buffer.count == 512)
        // Create temporary vector from the buffer
        // In reality, VectorCore copies here, but this shows the pattern
        let vector = try! Vector512Optimized(Array(buffer))
        return try body(vector)
    }
}
```

---

## Key Takeaways

1. **Buffer pointers combine address + count.** They know their size, unlike raw pointers.

2. **They conform to Collection.** Use `for-in`, `reduce`, `map` as normal.

3. **`withUnsafeBufferPointer` is the safe entry point.** Ensures the underlying data stays valid.

4. **Type rebinding enables SIMD views.** View `[Float]` as `[SIMD4<Float>]` without copying.

5. **VectorCore uses buffers extensively.** Both for public API and internal implementation.

---

## Next Up

Now let's see how to work with C code, which is essential for system-level operations like aligned memory allocation:

**[→ C Interop Patterns](./03-C-Interop-Patterns.md)**
