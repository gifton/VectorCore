# C Interop Patterns

> **Reading time:** 10 minutes
> **Prerequisites:** [Buffer Pointers](./02-Buffer-Pointers.md)

---

## The Concept

Swift can call C functions directly. This is essential for:

- System APIs (`posix_memalign`, `mmap`, `pthread`)
- High-performance libraries (Accelerate, Metal, vDSP)
- Legacy code integration

The key challenge: C and Swift have different memory models. You must bridge them correctly.

---

## Why It Matters

VectorCore uses C functions for aligned memory allocation because Swift's standard library doesn't provide this directly:

```swift
// Swift's built-in allocation (alignment not guaranteed)
let ptr = UnsafeMutablePointer<Float>.allocate(capacity: 100)

// C's aligned allocation (specific alignment guaranteed)
var ptr: UnsafeMutableRawPointer?
posix_memalign(&ptr, 64, 400)  // 64-byte aligned
```

The difference matters for SIMD operations that require aligned memory.

---

## The Technique

### Calling C Functions

C functions are available automatically when you import the right module:

```swift
import Foundation  // Brings in most POSIX functions

// posix_memalign is declared in C as:
// int posix_memalign(void **memptr, size_t alignment, size_t size);

var ptr: UnsafeMutableRawPointer?
let result = posix_memalign(&ptr, 64, 1024)

if result == 0, let validPtr = ptr {
    // Success: validPtr is 64-byte aligned, 1024 bytes
    defer { free(validPtr) }
    // ... use the memory ...
} else {
    // Handle error (result is errno value)
}
```

### The Allocator-Deallocator Rule

**Critical:** Memory must be freed by the matching deallocator:

| Allocator | Deallocator | Why |
|-----------|-------------|-----|
| `malloc()` | `free()` | C allocator pair |
| `posix_memalign()` | `free()` | Same C allocator |
| `UnsafeMutablePointer.allocate()` | `.deallocate()` | Swift allocator pair |
| `UnsafeMutableRawPointer.allocate()` | `.deallocate()` | Swift allocator pair |

**Never mix them:**

```swift
// ❌ WRONG: C allocation with Swift deallocation
var ptr: UnsafeMutableRawPointer?
posix_memalign(&ptr, 64, 1024)
ptr?.deallocate()  // 💥 Heap corruption!

// ✅ RIGHT: C allocation with C deallocation
var ptr: UnsafeMutableRawPointer?
posix_memalign(&ptr, 64, 1024)
free(ptr)  // ✓ Correct
```

**Why does mixing cause problems?**

C's `malloc`/`free` and Swift's allocate/deallocate use different bookkeeping. They track allocations in different data structures. If you `free()` memory that Swift allocated, `free()` looks in the wrong place, corrupts the heap, and your program crashes—usually much later, in unrelated code.

### Bridging Pointer Types

C pointers map to Swift unsafe pointers:

| C Type | Swift Type |
|--------|------------|
| `void*` | `UnsafeMutableRawPointer` |
| `const void*` | `UnsafeRawPointer` |
| `float*` | `UnsafeMutablePointer<Float>` |
| `const float*` | `UnsafePointer<Float>` |

Converting between them:

```swift
// Raw to typed
let rawPtr: UnsafeMutableRawPointer = ...
let floatPtr = rawPtr.assumingMemoryBound(to: Float.self)
// or
let floatPtr = rawPtr.bindMemory(to: Float.self, capacity: count)

// Typed to raw
let floatPtr: UnsafeMutablePointer<Float> = ...
let rawPtr = UnsafeMutableRawPointer(floatPtr)
```

**`assumingMemoryBound` vs `bindMemory`:**
- `assumingMemoryBound`: "I promise this memory is already bound to this type"
- `bindMemory`: "Bind this memory to this type now"

Use `assumingMemoryBound` for roundtripping (when you originally had typed pointers). Use `bindMemory` when interpreting raw bytes as a type.

---

## In VectorCore

### Aligned Allocation

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
        precondition(alignment >= 16,
                     "Alignment must be at least 16 bytes")

        var rawPtr: UnsafeMutableRawPointer?
        let byteCount = count * MemoryLayout<T>.stride
        let result = posix_memalign(&rawPtr, alignment, byteCount)

        guard result == 0, let ptr = rawPtr else {
            throw VectorError.allocationFailed(size: byteCount,
                                               reason: "posix_memalign error \(result)")
        }

        return ptr.assumingMemoryBound(to: T.self)
    }

    /// Deallocate memory previously allocated via posix_memalign
    @inlinable
    public static func deallocate<T>(_ ptr: UnsafeMutablePointer<T>) {
        free(UnsafeMutableRawPointer(ptr))
    }
}
```

**Key design decisions:**

1. **Generic over T:** Works for any type
2. **Platform-aware alignment:** Different defaults for ARM vs x86
3. **Throws on failure:** Converts C error codes to Swift errors
4. **Wrapper for deallocation:** Ensures `free()` is used, not `.deallocate()`

### Using AlignedMemory

```swift
// Allocate 512 floats, 64-byte aligned
let ptr = try AlignedMemory.allocateAligned(type: Float.self, count: 512)
defer { AlignedMemory.deallocate(ptr) }

// ptr is guaranteed to be on a 64-byte boundary
assert(Int(bitPattern: ptr) % 64 == 0)

// Use it
for i in 0..<512 {
    ptr[i] = Float(i)
}
```

### Checking Alignment

**📍 See:** `Sources/VectorCore/Storage/AlignedMemory.swift:29-38`

```swift
/// Check if a pointer is properly aligned
@inlinable
public static func isAligned<T>(_ pointer: UnsafePointer<T>, to alignment: Int) -> Bool {
    Int(bitPattern: pointer) % alignment == 0
}
```

Usage:
```swift
let ptr: UnsafePointer<Float> = ...
if AlignedMemory.isAligned(ptr, to: 64) {
    // Use fast SIMD path
} else {
    // Use slower unaligned path
}
```

---

## Working with C APIs

### Pattern 1: Inout Parameters

Many C functions use output parameters:

```swift
// C: int getvalue(int* output);
// Swift sees: getvalue(_: UnsafeMutablePointer<Int32>)

var output: Int32 = 0
let result = getvalue(&output)
// output now contains the value
```

### Pattern 2: Array Parameters

C arrays decay to pointers:

```swift
// C: void process(const float* data, size_t count);
// Swift sees: process(_: UnsafePointer<Float>, _: Int)

let data: [Float] = [1, 2, 3, 4]
data.withUnsafeBufferPointer { buffer in
    process(buffer.baseAddress!, buffer.count)
}
```

### Pattern 3: Callbacks

C function pointers become Swift closures (with restrictions):

```swift
// C: void forEach(void (*callback)(int value, void* context), void* context);

// In Swift:
typealias Callback = @convention(c) (Int32, UnsafeMutableRawPointer?) -> Void

let callback: Callback = { value, context in
    print("Got: \(value)")
}

forEach(callback, nil)
```

Note: The closure can't capture values (`@convention(c)` requires no captures). Use the `context` parameter for state.

---

## Common Pitfalls

### Pitfall 1: Wrong Deallocator
```swift
// ❌ Heap corruption
let ptr = try AlignedMemory.allocateAligned(type: Float.self, count: 100)
ptr.deallocate()  // Wrong! Uses Swift deallocator

// ✅ Correct
AlignedMemory.deallocate(ptr)
```

### Pitfall 2: Dangling Pointers to Stack Data
```swift
// ❌ Pointer outlives data
func bad() -> UnsafePointer<Float> {
    var value: Float = 42
    return withUnsafePointer(to: &value) { $0 }  // Returns pointer to stack!
}

// ✅ Copy the data if needed
func good() -> Float {
    var value: Float = 42
    return value  // Copy the value, not the pointer
}
```

### Pitfall 3: Type Punning Violations
```swift
// ❌ Wrong: Interpreting Float bits as Int
let float: Float = 1.0
let ptr = withUnsafePointer(to: float) { $0 }
let intPtr = UnsafeRawPointer(ptr).assumingMemoryBound(to: Int.self)
let int = intPtr.pointee  // Undefined behavior!

// ✅ Right: Use bitPattern initializer
let float: Float = 1.0
let bits = float.bitPattern  // UInt32 representation
```

---

## Key Takeaways

1. **C functions are directly callable from Swift.** Import the right module and go.

2. **Match allocators to deallocators.** `posix_memalign` → `free()`. Swift allocate → Swift deallocate.

3. **VectorCore wraps aligned allocation safely.** `AlignedMemory` hides the C details.

4. **Type bridging requires care.** Use `bindMemory` for new bindings, `assumingMemoryBound` for roundtrips.

5. **C callbacks can't capture values.** Use the context parameter for state.

---

## Chapter Complete!

You now understand:
- How Swift pointers work
- Buffer pointers for contiguous data
- C interoperability for system-level operations

Next, we'll put it all together with performance measurement and optimization:

**[→ Chapter 5: Performance Patterns](../05-Performance-Patterns/README.md)**
