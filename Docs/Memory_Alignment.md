# Memory Alignment in VectorCore

This document explains VectorCore's memory alignment strategy, why it matters for performance, and how to use aligned allocation correctly.

---

## Table of Contents

1. [Why Alignment Matters](#why-alignment-matters)
2. [VectorCore's Alignment Policy](#vectorcores-alignment-policy)
3. [Using AlignedMemory](#using-alignedmemory)
4. [Common Pitfalls](#common-pitfalls)
5. [Platform-Specific Details](#platform-specific-details)
6. [Debugging Alignment Issues](#debugging-alignment-issues)

---

## Why Alignment Matters

### SIMD Requirements

Modern CPUs have SIMD (Single Instruction, Multiple Data) instructions that operate on multiple values simultaneously. On Apple Silicon and Intel processors, these instructions have strict alignment requirements:

| Instruction Set | Data Type | Required Alignment |
|-----------------|-----------|-------------------|
| **NEON (ARM)** | `SIMD4<Float>` | 16 bytes |
| **AVX (Intel)** | `SIMD8<Float>` | 32 bytes |
| **AVX-512 (Intel)** | `SIMD16<Float>` | 64 bytes |

### Performance Impact

**Aligned access** (address is a multiple of alignment):
```assembly
; ARM NEON - aligned load (1 cycle)
LD1    {v0.4s}, [x0]  ; Load 4 floats (16 bytes) from address in x0
```

**Unaligned access** (address is NOT a multiple of alignment):
```assembly
; ARM NEON - unaligned load (multiple cycles + stalls)
LDR    w1, [x0]       ; Load byte-by-byte
LDR    w2, [x0, #4]
LDR    w3, [x0, #8]
LDR    w4, [x0, #12]
INS    v0.s[0], w1    ; Insert into SIMD register
; ... more instructions
```

**Performance penalty**: 2-5x slowdown for unaligned access

### Cache Line Optimization

Modern CPUs fetch memory in **cache lines** (typically 64 bytes). Aligning data to cache line boundaries:
- Prevents cache line splits (one value spans two cache lines)
- Improves prefetcher efficiency
- Reduces memory bandwidth usage

**Example of cache line split**:
```
Cache line 0: [... last 8 bytes | ← SIMD4<Float> (8 bytes) →]
Cache line 1: [← SIMD4<Float> (8 bytes) → | next values ...]
```
Result: Two cache line fetches for one SIMD load (2x memory bandwidth)

---

## VectorCore's Alignment Policy

### Default Alignment: 64 Bytes

VectorCore uses **64-byte alignment** for all SIMD storage:

```swift
public struct Vector512Optimized {
    // storage is 64-byte aligned
    public var storage: ContiguousArray<SIMD4<Float>>
}
```

**Why 64 bytes?**
- Matches cache line size on Apple Silicon and modern Intel
- Satisfies SIMD4 alignment (16 bytes) with margin
- Prevents cache line splits
- Future-proof for wider SIMD (AVX-512 requires 64-byte alignment)

### Allocation Methods

VectorCore provides centralized aligned allocation:

```swift
// Sources/VectorCore/Storage/AlignedMemory.swift

public enum AlignedMemory {
    /// Allocate aligned memory using posix_memalign
    public static func allocateAligned<T>(
        _ type: T.Type,
        count: Int,
        alignment: Int = 64
    ) throws -> UnsafeMutablePointer<T> {
        var ptr: UnsafeMutableRawPointer?
        let size = MemoryLayout<T>.stride * count
        let result = posix_memalign(&ptr, alignment, size)

        guard result == 0, let allocatedPtr = ptr else {
            throw VectorError.allocationFailed(size: size)
        }

        return allocatedPtr.bindMemory(to: T.self, capacity: count)
    }

    /// Deallocate memory allocated with allocateAligned
    public static func deallocate<T>(_ ptr: UnsafeMutablePointer<T>) {
        free(UnsafeMutableRawPointer(ptr))
    }

    /// Deallocate raw pointer
    public static func deallocate(_ ptr: UnsafeMutableRawPointer) {
        free(ptr)
    }
}
```

---

## Using AlignedMemory

### Automatic Alignment (Recommended)

Most users never need to think about alignment—VectorCore handles it automatically:

```swift
// ✅ Automatic 64-byte aligned allocation
let vector = Vector512Optimized(repeating: 1.0)
// storage is automatically aligned
```

### Manual Alignment (Advanced)

For custom data structures:

```swift
// Allocate aligned memory
let ptr = try AlignedMemory.allocateAligned(
    Float.self,
    count: 512,
    alignment: 64
)

// Use the memory
for i in 0..<512 {
    ptr[i] = Float(i)
}

// IMPORTANT: Use AlignedMemory.deallocate, NOT ptr.deallocate()
AlignedMemory.deallocate(ptr)
```

### Buffer Pool with Alignment

VectorCore's buffer pool maintains alignment:

```swift
actor SwiftBufferPool {
    func acquire(count: Int) -> UnsafeMutableBufferPointer<Float> {
        // Allocates 64-byte aligned buffer
        let ptr = try! AlignedMemory.allocateAligned(
            Float.self,
            count: count,
            alignment: 64
        )
        return UnsafeMutableBufferPointer(start: ptr, count: count)
    }

    func release(_ handle: BufferHandle) {
        // Correctly deallocates with free()
        AlignedMemory.deallocate(handle.pointer)
    }
}
```

---

## Common Pitfalls

### ❌ Pitfall 1: Using .deallocate() on posix_memalign Memory

**The Bug**:
```swift
// ❌ WRONG: Causes undefined behavior
let ptr = try AlignedMemory.allocateAligned(Float.self, count: 512)
// ... use ptr ...
ptr.deallocate()  // ❌ WRONG! posix_memalign memory MUST use free()
```

**Why it's wrong**:
- `posix_memalign` allocates memory using the C heap
- Swift's `.deallocate()` uses a different allocator (Swift runtime)
- Mismatched allocator/deallocator causes heap corruption

**The Fix**:
```swift
// ✅ CORRECT: Use AlignedMemory.deallocate
let ptr = try AlignedMemory.allocateAligned(Float.self, count: 512)
// ... use ptr ...
AlignedMemory.deallocate(ptr)  // ✅ Calls free() internally
```

**Impact**: This bug was fixed in VectorCore v0.1.0 (Phase 1). See git history for details.

---

### ❌ Pitfall 2: Assuming Swift Arrays are Aligned

**The Problem**:
```swift
// ❌ Swift Array is NOT guaranteed to be aligned
var array = [Float](repeating: 0, count: 512)
array.withUnsafeMutableBufferPointer { buffer in
    // buffer.baseAddress might not be 64-byte aligned!
    // SIMD operations may be slow or crash
}
```

**The Fix**:
```swift
// ✅ Use aligned allocation
let ptr = try AlignedMemory.allocateAligned(Float.self, count: 512, alignment: 64)
let buffer = UnsafeMutableBufferPointer(start: ptr, count: 512)
// buffer.baseAddress is guaranteed 64-byte aligned
// ... use buffer ...
AlignedMemory.deallocate(ptr)
```

---

### ❌ Pitfall 3: Misaligned Slicing

**The Problem**:
```swift
let vector = Vector512Optimized(...)
let storage = vector.storage  // Aligned

// ❌ Slicing breaks alignment
let slice = storage[10...]  // Offset is not 64-byte aligned!
slice.withUnsafeBufferPointer { buffer in
    // buffer is no longer aligned
}
```

**The Fix**: Avoid slicing SIMD storage. If you must, realign:
```swift
// Copy to new aligned storage
var alignedStorage = ContiguousArray<SIMD4<Float>>()
alignedStorage.reserveCapacity(slice.count)
alignedStorage.append(contentsOf: slice)
// alignedStorage is aligned again
```

---

## Platform-Specific Details

### Apple Silicon (ARM64)

**Hardware characteristics**:
- Cache line size: 64 bytes (M1/M2), 128 bytes (M3+)
- SIMD width: 128-bit (NEON)
- Required alignment: 16 bytes (SIMD4<Float>)
- Recommended alignment: 64 bytes (cache line)

**Performance notes**:
- Unaligned SIMD loads: ~2x slower
- Cache line split: ~1.5x slower
- `posix_memalign` overhead: ~50ns per allocation

**Optimal alignment strategy**:
```swift
// Use 64-byte alignment for M1/M2
#if arch(arm64)
let alignment = 64
#endif
```

### Intel (x86_64)

**Hardware characteristics**:
- Cache line size: 64 bytes
- SIMD width: 128-bit (SSE), 256-bit (AVX2), 512-bit (AVX-512)
- Required alignment: 16 bytes (SSE), 32 bytes (AVX2), 64 bytes (AVX-512)
- Recommended alignment: 64 bytes

**Performance notes**:
- Unaligned AVX loads: ~5x slower than aligned
- AVX-512 unaligned: May cause #GP fault (crash) on older CPUs
- `_mm_malloc` overhead: ~100ns per allocation

**Optimal alignment strategy**:
```swift
#if arch(x86_64)
let alignment = 64  // Safe for all Intel SIMD
#endif
```

---

## Debugging Alignment Issues

### Checking Alignment at Runtime

```swift
func isAligned(_ ptr: UnsafeRawPointer, alignment: Int) -> Bool {
    return Int(bitPattern: ptr) % alignment == 0
}

// Example usage
vector.storage.withUnsafeBufferPointer { buffer in
    let aligned = isAligned(buffer.baseAddress!, alignment: 64)
    print("Storage is \(aligned ? "aligned" : "NOT aligned")")
}
```

### Using AddressSanitizer (ASan)

ASan detects misaligned access and heap corruption:

```bash
# Build with ASan
swift build -c debug -Xswiftc -sanitize=address

# Run tests
swift test -Xswiftc -sanitize=address
```

**ASan will detect**:
- Heap corruption from mismatched allocator/deallocator
- Use-after-free from dangling pointers
- Buffer overflows

### Using Instruments

Profile memory access patterns:

```bash
# Record memory profile
xcrun xctrace record --template 'Memory' --launch .build/debug/YourApp

# Analyze cache misses
# Look for:
# - High L1 cache miss rate (>5%) → possible misalignment
# - High memory bandwidth usage → possible cache line splits
```

---

## Best Practices

### ✅ Do:
- Use VectorCore's built-in types (automatic alignment)
- Use `AlignedMemory.allocateAligned` for custom allocations
- Always use `AlignedMemory.deallocate` for cleanup
- Verify alignment in debug builds (`assert(isAligned(...))`)
- Use ASan/TSan in CI to catch bugs early

### ❌ Don't:
- Call `.deallocate()` on `posix_memalign` memory
- Assume Swift Arrays are aligned
- Slice SIMD storage without realigning
- Mix Swift and C allocators
- Ignore alignment requirements for performance-critical code

---

## API Reference

### AlignedMemory

```swift
public enum AlignedMemory {
    /// Allocate aligned memory
    public static func allocateAligned<T>(
        _ type: T.Type,
        count: Int,
        alignment: Int = 64
    ) throws -> UnsafeMutablePointer<T>

    /// Deallocate aligned memory (typed pointer)
    public static func deallocate<T>(_ ptr: UnsafeMutablePointer<T>)

    /// Deallocate aligned memory (raw pointer)
    public static func deallocate(_ ptr: UnsafeMutableRawPointer)
}
```

### VectorError

```swift
public enum VectorError: Error {
    case allocationFailed(size: Int)
    // ... other cases
}
```

---

## FAQ

**Q: Why not use Swift's `.aligned()` or `withAlignedBytes`?**
A: These APIs were introduced in Swift 5.7+ and are not yet widely adopted. VectorCore supports Swift 6.0+ but uses `posix_memalign` for compatibility and control.

**Q: Does `ContiguousArray` guarantee alignment?**
A: No. `ContiguousArray` only guarantees **element** alignment (16 bytes for SIMD4<Float>), not **storage** alignment (64 bytes for cache lines).

**Q: What happens if I use unaligned memory?**
A: On ARM (Apple Silicon): 2-5x slowdown. On Intel: 5-10x slowdown or crash (AVX-512).

**Q: Can I use `aligned_alloc` instead of `posix_memalign`?**
A: Yes, but `posix_memalign` is more portable (C99 vs C11) and has identical performance.

**Q: Why 64 bytes instead of 16 bytes?**
A: 16 bytes satisfies SIMD4 alignment, but 64 bytes prevents cache line splits and is future-proof for AVX-512.

---

## Further Reading

- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- [ARM NEON Programming Guide](https://developer.arm.com/architectures/instruction-sets/simd-isas/neon)
- [POSIX memalign documentation](https://man7.org/linux/man-pages/man3/posix_memalign.3.html)
- [Apple Silicon Performance Guide](https://developer.apple.com/documentation/apple-silicon/addressing-architectural-differences-in-your-macos-code)

---

**Document Version**: 1.0
**Last Updated**: October 2025
**Applies to**: VectorCore v0.1.0+
