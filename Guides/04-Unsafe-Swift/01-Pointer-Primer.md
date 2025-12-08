# Pointer Primer

> **Reading time:** 12 minutes
> **Prerequisites:** [Chapter 1: Memory Fundamentals](../01-Memory-Fundamentals/README.md)

---

## The Concept

A pointer is a variable that holds a memory address. In Swift, pointers are wrapped in types that express what you can do with them:

| Type | Meaning |
|------|---------|
| `UnsafePointer<T>` | Read-only pointer to a single T |
| `UnsafeMutablePointer<T>` | Read-write pointer to a single T |
| `UnsafeRawPointer` | Read-only pointer to raw bytes (no type) |
| `UnsafeMutableRawPointer` | Read-write pointer to raw bytes |
| `UnsafeBufferPointer<T>` | Read-only pointer to a contiguous region of T |
| `UnsafeMutableBufferPointer<T>` | Read-write pointer to a contiguous region |

The "unsafe" prefix means Swift won't check:
- Bounds (accessing memory outside the allocated region)
- Lifetime (using memory after it's deallocated)
- Type (interpreting bytes as the wrong type)

---

## Why It Matters

### Safe vs. Unsafe Access

```swift
// Safe: Swift checks bounds and manages memory
var array = [1.0, 2.0, 3.0, 4.0] as [Float]
print(array[0])  // Swift checks 0 < array.count

// Unsafe: Direct memory access, no checks
array.withUnsafeBufferPointer { ptr in
    print(ptr[0])  // No bounds check!
    print(ptr[100])  // Undefined behavior—might crash, might return garbage
}
```

The safe version has overhead: bounds checking, copy-on-write checks, potential memory allocation. For a million-element inner loop, that overhead matters.

---

## The Technique

### Getting Pointers from Arrays

Swift provides safe entry points into unsafe territory:

```swift
var array = [Float](repeating: 0, count: 100)

// Read-only access
array.withUnsafeBufferPointer { ptr in
    // ptr: UnsafeBufferPointer<Float>
    let first = ptr[0]
    let address = ptr.baseAddress  // UnsafePointer<Float>?
    let count = ptr.count          // 100
}

// Read-write access
array.withUnsafeMutableBufferPointer { ptr in
    // ptr: UnsafeMutableBufferPointer<Float>
    ptr[0] = 42.0  // Modifies the original array
}
```

**Key rule:** The pointer is only valid inside the closure. Never let it escape:

```swift
var savedPointer: UnsafeBufferPointer<Float>?

array.withUnsafeBufferPointer { ptr in
    savedPointer = ptr  // ❌ BAD: Pointer escapes the closure!
}

// Using savedPointer here is undefined behavior.
// The array might have moved, been deallocated, etc.
```

### Pointer Arithmetic

Pointers support arithmetic like in C:

```swift
array.withUnsafeBufferPointer { buffer in
    guard let base = buffer.baseAddress else { return }

    let first = base.pointee       // Same as base[0]
    let second = (base + 1).pointee  // Same as base[1]

    // Advance by n elements
    let fifth = base.advanced(by: 4).pointee  // Same as base[4]

    // Pointer difference
    let end = base.advanced(by: buffer.count)
    let distance = end - base  // 100
}
```

### Type Binding

Memory in Swift is "bound" to a type. You can rebind it to interpret the same bytes differently:

```swift
let floats: [Float] = [1.0, 2.0, 3.0, 4.0]  // 4 × 4 = 16 bytes

floats.withUnsafeBufferPointer { floatBuffer in
    // Get raw pointer (forget the Float type)
    let rawPtr = UnsafeRawPointer(floatBuffer.baseAddress!)

    // Rebind as SIMD4<Float> (same 16 bytes, different type)
    let simd4Ptr = rawPtr.bindMemory(to: SIMD4<Float>.self, capacity: 1)

    let simd4Value = simd4Ptr.pointee
    // simd4Value = SIMD4<Float>(1.0, 2.0, 3.0, 4.0)
}
```

**Warning:** Binding memory to an incompatible type is undefined behavior. `Float` → `SIMD4<Float>` works because SIMD4 is just 4 floats in a row. `Float` → `Int` would be wrong.

---

## Memory Lifetime

Pointers don't extend the lifetime of what they point to:

```swift
func dangerous() -> UnsafePointer<Float> {
    var value: Float = 42.0
    return withUnsafePointer(to: &value) { ptr in
        return ptr  // ❌ BAD: Returning pointer to stack variable
    }
    // 'value' is deallocated here
}

let ptr = dangerous()
print(ptr.pointee)  // 💥 Use-after-free!
```

**Rule:** The lifetime of the data must exceed the lifetime of the pointer.

### Fixing Lifetime Issues

```swift
// Option 1: Keep data alive in a broader scope
class DataHolder {
    var values: [Float] = [1, 2, 3, 4]

    func process() {
        values.withUnsafeBufferPointer { ptr in
            // ptr is valid because 'values' outlives this closure
            doWork(ptr)
        }
    }
}

// Option 2: Copy if you need to escape
func safe() -> [Float] {
    var value: Float = 42.0
    return [value]  // Copy the data, not the pointer
}
```

---

## In VectorCore

VectorCore uses pointer access throughout for performance:

**📍 See:** `Sources/VectorCore/Vectors/Vector512Optimized.swift:147-169`

```swift
/// Access storage for reading
@inlinable
public func withUnsafeBufferPointer<R>(
    _ body: (UnsafeBufferPointer<Scalar>) throws -> R
) rethrows -> R {
    try storage.withUnsafeBufferPointer { simd4Buffer in
        // Rebind SIMD4<Float> storage to plain Float
        let floatBuffer = UnsafeRawPointer(simd4Buffer.baseAddress!)
            .bindMemory(to: Float.self, capacity: 512)
        return try body(UnsafeBufferPointer(start: floatBuffer, count: 512))
    }
}

/// Access storage for writing
@inlinable
public mutating func withUnsafeMutableBufferPointer<R>(
    _ body: (UnsafeMutableBufferPointer<Scalar>) throws -> R
) rethrows -> R {
    try storage.withUnsafeMutableBufferPointer { simd4Buffer in
        let floatBuffer = UnsafeMutableRawPointer(simd4Buffer.baseAddress!)
            .bindMemory(to: Float.self, capacity: 512)
        return try body(UnsafeMutableBufferPointer(start: floatBuffer, count: 512))
    }
}
```

**What's happening:**

1. `storage` is `ContiguousArray<SIMD4<Float>>` — 128 elements × 16 bytes each
2. We get the raw pointer to the first SIMD4
3. We rebind it as 512 Floats (same bytes, different view)
4. The caller gets a flat buffer of floats

This lets users work with individual floats while VectorCore stores them as SIMD4 internally.

### Direct SIMD Access

For operations that need SIMD4 directly:

**📍 See:** `Sources/VectorCore/Operations/Kernels/DotKernels.swift:19`

```swift
private static func dot(
    storageA: ContiguousArray<SIMD4<Float>>,
    storageB: ContiguousArray<SIMD4<Float>>,
    laneCount: Int
) -> Float {
    // Direct access to SIMD4 elements—no rebinding needed
    for i in stride(from: 0, to: laneCount, by: 16) {
        acc0 += storageA[i+0] * storageB[i+0]
        // ...
    }
}
```

No pointer ceremony here—Swift's array subscripting is fast enough when bounds checking is disabled (release builds), and `@inline(__always)` ensures no function call overhead.

---

## Debugging Pointer Issues

### Address Sanitizer

Compile with Address Sanitizer to catch memory errors:

```bash
swift build -c debug -Xswiftc -sanitize=address
swift test -Xswiftc -sanitize=address
```

ASan catches:
- Use-after-free
- Out-of-bounds access
- Double-free
- Memory leaks

### Common Crash Signatures

| Crash | Likely Cause |
|-------|--------------|
| `EXC_BAD_ACCESS` | Dereferencing freed/invalid pointer |
| `SIGBUS` | Misaligned access |
| `SIGSEGV` | Accessing unmapped memory |
| Random corruption | Use-after-free with reused memory |

---

## Key Takeaways

1. **Pointers bypass Swift's safety checks.** No bounds checking, no lifetime management.

2. **Use `withUnsafe...` closures.** They ensure the data stays valid during access.

3. **Never escape pointers from closures.** The data may be deallocated after the closure returns.

4. **Type rebinding reinterprets bytes.** `Float` → `SIMD4<Float>` works; `Float` → `Int` doesn't.

5. **Address Sanitizer is your friend.** Use it during development to catch memory bugs.

---

## Next Up

Pointers access single values. For working with arrays, we need buffers:

**[→ Buffer Pointers](./02-Buffer-Pointers.md)**
