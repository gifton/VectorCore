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

### Default Alignment: 64 Bytes (arm64 SIMD) — but not universal

VectorCore has **two distinct alignment constants** for two distinct purposes. Do not
conflate them:

| Constant | Value | Purpose |
|----------|-------|---------|
| `AlignedMemory.optimalAlignment` | **64** (arm64 & x86_64); 16 otherwise | Default alignment for `allocateAligned`; cache-line / SIMD storage |
| `AlignedMemory.minimumAlignment` | **16** | Floor enforced by a `precondition` (SIMD4<Float>) |
| `PlatformConfiguration.l1CacheLineSize` | **64** (arm64 & x86_64) | L1 cache line size |
| `PlatformConfiguration.optimalAlignment` | **128** on Apple Silicon (AMX); 64 on x86_64 | AMX / matrix-extension data alignment |
| `PlatformConfiguration.pageSize` | `getpagesize()` at runtime (16 KB Apple Silicon, 4 KB x86) | Page alignment for `bytesNoCopy` |

So "64 bytes" is the default for `AlignedMemory.allocateAligned` (good for SIMD storage and
the L1 cache line), but it is **not** a universal alignment for VectorCore. Apple Silicon's
AMX (Apple Matrix Extensions) path prefers **128-byte** alignment
(`PlatformConfiguration.optimalAlignment`), and the zero-copy GPU path needs **page**
alignment (see [The Page-Aligned Path](#the-page-aligned-path)). Always pick the alignment
that matches the consumer.

The optimized vector types back their storage with `ContiguousArray<SIMD4<Float>>`:

```swift
public struct Vector512Optimized {
    // SIMD4<Float> element region; 16-byte element alignment.
    public var storage: ContiguousArray<SIMD4<Float>>
}
```

**Why 64 bytes as the `allocateAligned` default?**
- Matches the L1 cache line size on Apple Silicon and modern Intel (`l1CacheLineSize == 64`)
- Satisfies SIMD4 alignment (16 bytes) with margin
- Prevents cache line splits
- Covers AVX-512 (which requires 64-byte alignment) on x86_64

### Allocation Methods

VectorCore provides centralized aligned allocation in
`Sources/VectorCore/Storage/AlignedMemory.swift`. These are the **real public signatures**
(the implementation uses `posix_memalign` and validates the alignment with `precondition`s):

```swift
public enum AlignedMemory {
    /// Platform-specific optimal alignment (64 on arm64 / x86_64, 16 otherwise).
    public static var optimalAlignment: Int { get }

    /// Minimum alignment enforced by allocateAligned (16 bytes; SIMD4<Float>).
    public static let minimumAlignment: Int

    /// Check whether a pointer is aligned to `alignment`.
    public static func isAligned<T>(_ pointer: UnsafePointer<T>, to alignment: Int) -> Bool
    public static func isAligned<T>(_ pointer: UnsafeMutablePointer<T>, to alignment: Int) -> Bool

    /// Allocate aligned memory for `Float` (the common case).
    /// - Precondition: `alignment` is a power of two and ≥ `minimumAlignment`.
    /// - Throws: `VectorError.allocationFailed(size:reason:)` if `posix_memalign` fails.
    public static func allocateAligned(
        count: Int,
        alignment: Int = optimalAlignment
    ) throws -> UnsafeMutablePointer<Float>

    /// Allocate aligned memory for any type `T`.
    /// - Precondition: `alignment` is a power of two and ≥ `minimumAlignment`.
    /// - Throws: `VectorError.allocationFailed(size:reason:)` if `posix_memalign` fails.
    public static func allocateAligned<T>(
        type: T.Type,
        count: Int,
        alignment: Int = optimalAlignment
    ) throws -> UnsafeMutablePointer<T>

    /// Deallocate (typed pointer). Calls `free()` internally — never `.deallocate()`.
    public static func deallocate<T>(_ ptr: UnsafeMutablePointer<T>)

    /// Deallocate (raw pointer). Calls `free()` internally.
    public static func deallocate(_ ptr: UnsafeMutableRawPointer)
}
```

> **API note.** There is **no** positional `allocateAligned(_ type:count:alignment:)`. Use
> either the `Float` overload `allocateAligned(count:)` or the labeled generic
> `allocateAligned(type:count:)`. The default `alignment` is `optimalAlignment` (64 on
> arm64), so you rarely pass it explicitly.

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

For custom data structures, use the `Float` overload (default alignment is
`optimalAlignment`, 64 on arm64):

```swift
// Allocate aligned Float memory (alignment defaults to optimalAlignment == 64 on arm64)
let ptr = try AlignedMemory.allocateAligned(count: 512)

// Use the memory
for i in 0..<512 {
    ptr[i] = Float(i)
}

// IMPORTANT: Use AlignedMemory.deallocate, NOT ptr.deallocate()
AlignedMemory.deallocate(ptr)
```

For a non-`Float` element type, use the labeled generic overload:

```swift
// Generic overload: pass the type with the `type:` label.
let simdPtr = try AlignedMemory.allocateAligned(type: SIMD4<Float>.self, count: 128)
defer { AlignedMemory.deallocate(simdPtr) }
```

### Buffer Pool with Alignment

VectorCore's buffer pool maintains alignment:

```swift
actor SwiftBufferPool {
    func acquire(count: Int) -> UnsafeMutableBufferPointer<Float> {
        // Allocates optimalAlignment (64-byte on arm64) aligned buffer
        let ptr = try! AlignedMemory.allocateAligned(count: count)
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
let ptr = try AlignedMemory.allocateAligned(count: 512)
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
let ptr = try AlignedMemory.allocateAligned(count: 512)
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
// ✅ Use aligned allocation (defaults to optimalAlignment == 64 on arm64)
let ptr = try AlignedMemory.allocateAligned(count: 512)
let buffer = UnsafeMutableBufferPointer(start: ptr, count: 512)
// buffer.baseAddress is guaranteed 64-byte aligned on arm64
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
- Cache line size: 64 bytes (`PlatformConfiguration.l1CacheLineSize`)
- Page size: `getpagesize()` — **16 KB** on Apple Silicon
- SIMD width: 128-bit (NEON)
- Required alignment: 16 bytes (SIMD4<Float>)
- Recommended SIMD/cache alignment: 64 bytes (`AlignedMemory.optimalAlignment`)
- AMX (matrix-extension) alignment: 128 bytes (`PlatformConfiguration.optimalAlignment`)

**Performance notes**:
- Unaligned SIMD loads: ~2x slower
- Cache line split: ~1.5x slower
- `posix_memalign` overhead: ~50ns per allocation

**Optimal alignment strategy** — use the constant for the consumer, not a hardcoded literal:
```swift
// SIMD / cache-line storage: 64 on arm64.
let simdAlignment = AlignedMemory.optimalAlignment        // 64 (arm64)

// AMX / matrix-extension data: 128 on Apple Silicon.
let amxAlignment = PlatformConfiguration.optimalAlignment // 128 (Apple Silicon)
```

### Intel (x86_64)

**Hardware characteristics**:
- Cache line size: 64 bytes (`PlatformConfiguration.l1CacheLineSize`)
- Page size: `getpagesize()` — **4 KB** on x86_64
- SIMD width: 128-bit (SSE), 256-bit (AVX2), 512-bit (AVX-512)
- Required alignment: 16 bytes (SSE), 32 bytes (AVX2), 64 bytes (AVX-512)
- Recommended alignment: 64 bytes (`AlignedMemory.optimalAlignment` == 64 on x86_64)

**Performance notes**:
- Unaligned AVX loads: ~5x slower than aligned
- AVX-512 unaligned: May cause #GP fault (crash) on older CPUs
- `_mm_malloc` overhead: ~100ns per allocation

**Optimal alignment strategy** — use the constant (64 on x86_64), don't hardcode:
```swift
let alignment = AlignedMemory.optimalAlignment  // 64 — safe for all Intel SIMD
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
    /// 64 on arm64 / x86_64, 16 otherwise.
    public static var optimalAlignment: Int { get }

    /// 16 (SIMD4<Float>); the floor enforced by allocateAligned.
    public static let minimumAlignment: Int

    public static func isAligned<T>(_ pointer: UnsafePointer<T>, to alignment: Int) -> Bool
    public static func isAligned<T>(_ pointer: UnsafeMutablePointer<T>, to alignment: Int) -> Bool

    /// Float overload. Default alignment = optimalAlignment.
    public static func allocateAligned(
        count: Int,
        alignment: Int = optimalAlignment
    ) throws -> UnsafeMutablePointer<Float>

    /// Generic overload (note the `type:` label). Default alignment = optimalAlignment.
    public static func allocateAligned<T>(
        type: T.Type,
        count: Int,
        alignment: Int = optimalAlignment
    ) throws -> UnsafeMutablePointer<T>

    /// Deallocate aligned memory (typed pointer); calls free().
    public static func deallocate<T>(_ ptr: UnsafeMutablePointer<T>)

    /// Deallocate aligned memory (raw pointer); calls free().
    public static func deallocate(_ ptr: UnsafeMutableRawPointer)
}
```

> The implementation `precondition`s that `alignment` is a power of two and
> `≥ minimumAlignment`, then calls `posix_memalign`.

### VectorError

`AlignedMemory` throws via the `allocationFailed(size:reason:)` factory (the
`reason` argument carries the `posix_memalign` error code):

```swift
extension VectorError {
    static func allocationFailed(
        size: Int,
        reason: String? = nil,
        // … source-location parameters with defaults …
    ) -> VectorError
}
```

### PlatformConfiguration (page / AMX constants)

```swift
// internal enum PlatformConfiguration
static var l1CacheLineSize: Int   // 64 (arm64 / x86_64)
static var optimalAlignment: Int  // 128 on Apple Silicon (AMX); 64 on x86_64
static var pageSize: Int           // getpagesize() — 16 KB Apple Silicon, 4 KB x86
static func roundUpToPage(_ byteCount: Int) -> Int  // the single page-rounding helper
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

**Q: Why is `AlignedMemory.optimalAlignment` 64 bytes instead of 16?**
A: 16 bytes satisfies SIMD4 alignment, but 64 bytes matches the L1 cache line
(`PlatformConfiguration.l1CacheLineSize`), prevents cache line splits, and covers AVX-512 on
x86_64. Note 64 is **not** universal: AMX data prefers 128 bytes
(`PlatformConfiguration.optimalAlignment` on Apple Silicon) and GPU `bytesNoCopy` import
needs page alignment (`PlatformConfiguration.pageSize`).

---

## Zero-Copy Buffer Contract (`UnifiedVectorBuffer`)

*(VectorCore v0.3.0+ — beta-evolution-4, DOCUMENT-4 S5)*

For cross-package, zero-copy hand-off (EmbedKit → VectorAccelerate → GPU), VectorCore exposes a
**CPU-only** memory contract. It imports no Metal or IOSurface; it only guarantees alignment and
contiguity. The GPU calls live in VectorAccelerate.

### The read contract

`UnifiedVectorBuffer` (`Sources/VectorCore/Storage/UnifiedVectorBuffer.swift`) is a contiguous,
alignment-guaranteed *read* view over a vector's `Float32` elements:

```swift
public protocol UnifiedVectorBuffer {
    var elementCount: Int { get }
    var alignment: Int { get }   // power of two ≥ MemoryLayout<Float>.alignment
    func withUnsafeContiguousBytes<R>(_ body: (UnsafeRawBufferPointer) throws -> R) rethrows -> R
}
```

- `elementCount` — logical element count
- `alignment` — guaranteed base alignment of the yielded pointer
- `withUnsafeContiguousBytes { (UnsafeRawBufferPointer) in … }` — scoped raw access to the logical
  bytes (`elementCount * MemoryLayout<Float>.stride`)

Conformers: `Vector384/512/768/1536Optimized` (alignment == `MemoryLayout<SIMD4<Float>>.alignment`,
i.e. **16**, their `ContiguousArray<SIMD4<Float>>` element region) and `DynamicVector`
(alignment == `MemoryLayout<Float>.alignment`, i.e. **4**, since storage may be inline). 16-byte
alignment is enough to **read**, but not enough for `makeBuffer(bytesNoCopy:)`.

### The page-aligned path (`makeBuffer(bytesNoCopy:)`)

`makeBuffer(bytesNoCopy:length:options:deallocator:)` requires, on macOS, that **both** the base
address and the length be multiples of the OS page size. The page size is **not** a fixed literal:
it is `PlatformConfiguration.pageSize` (`getpagesize()` at runtime — **16 KB** on Apple Silicon,
**4 KB** on x86_64). The single page-rounding helper is `PlatformConfiguration.roundUpToPage(_:)`,
the authoritative source of truth used by every page-aligned allocation in the package (so the
`length:` is computed identically everywhere).

There are two page-aligned producers. Both round their byte length up with `roundUpToPage(_:)`,
zero-fill the padding so the GPU never imports uninitialized memory, and back the allocation with
`posix_memalign` (freed with `free()` via `AlignedMemory.deallocate(_:)`).

#### Producer 1 — `PageAlignedBuffer` (single embedding)

`PageAlignedBuffer` (`public final class : UnifiedVectorBuffer`) is the right producer when you
have **one** vector to hand to the GPU (the canonical EmbedKit case: write a freshly produced
embedding into page-aligned memory once, then hand the pointer downstream). Real surface:

```swift
public final class PageAlignedBuffer: UnifiedVectorBuffer, @unchecked Sendable {
    public init(elementCount: Int)                  // zero-initialized, page-aligned
    public convenience init(copying source: [Float]) // allocate + copy once

    public let pageSize: Int                         // captured PlatformConfiguration.pageSize
    public let allocatedByteCount: Int               // logical bytes rounded up to a page multiple
    public private(set) var ownsAllocation: Bool

    public var baseAddress: UnsafeMutableRawPointer  // the bytesNoCopy: pointer (traps if consumed)
    public var alignment: Int { pageSize }
    public var elementCount: Int

    public func withUnsafeContiguousBytes<R>(_ body: (UnsafeRawBufferPointer) throws -> R) rethrows -> R
    public func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R

    /// Transfer ownership to a Metal bytesNoCopy deallocator.
    public func consumeAllocation() -> (baseAddress: UnsafeMutableRawPointer, allocatedByteCount: Int)
}
```

`baseAddress` and `allocatedByteCount` are exactly the `bytesNoCopy:` pointer and the `length:`.

#### Producer 2 — SoA page-aligned (the primary 0.3.0 batch source)

For a **batch** of candidates — the primary zero-copy source in 0.3.0 — build the
Structure-of-Arrays buffer page-aligned. The in-memory layout is **frozen** as of 0.3.0; see
[`Docs/SoA_Layout_Contract.md`](./SoA_Layout_Contract.md) for the authoritative contract (and
`SoA.layoutDescriptor` for the machine-readable form). Relevant surface:

```swift
// Build page-aligned (otherwise the default 16-byte allocation):
let soa = SoA<Vector768Optimized>.build(from: candidates, pageAligned: true)
// or:        SoA<Vector768Optimized>(vectors: candidates, pageAligned: true)

/// Page-aligned base + page-ROUNDED byte length (≥ logical), or nil if not page-aligned.
var pageAlignedBytes: (base: UnsafeRawPointer, byteCount: Int)? { get }

/// Transfer ownership of the page-aligned allocation (nil if not page-aligned).
func consumeAllocation() -> (base: UnsafeMutableRawPointer, byteCount: Int)?

/// Scoped read access to the logical bytes (count * lanes * 16).
func withUnsafeRawBuffer<R>(_ body: (UnsafeRawBufferPointer) throws -> R) rethrows -> R
```

`pageAlignedBytes.byteCount` is the **page-rounded** length (the `length:` for `bytesNoCopy`), not
the logical byte count. When deriving the candidate count `N` downstream, take it from
`SoALayout.count` (via `layoutDescriptor`) — never reverse-engineer `N` from the page-rounded
length.

### Free / lifetime contract — borrow vs. transfer

Both producers' page-aligned memory comes from `posix_memalign`, so it **must** be freed with
`free()` — exactly what `AlignedMemory.deallocate(base)` does, which is thread-safe and therefore
valid from a Metal `bytesNoCopy` deallocator running on an arbitrary thread. For a true zero-copy
hand-off the `MTLBuffer` outlives the Swift object; pick **one** of two modes and never mix them
(mixing double-frees):

- **BORROW** — hold a strong reference to the producer (`PageAlignedBuffer` or `SoA`) and read
  `baseAddress` / `pageAlignedBytes` **without** consuming. The producer still frees on `deinit`,
  so pass a **no-op** Metal deallocator. The producer's object lifetime is the **sole** validity
  guarantee: it MUST outlive the `MTLBuffer`.
- **TRANSFER** — call `consumeAllocation()`. The producer stops owning the allocation (no free on
  `deinit`; further access to `baseAddress` / `pageAlignedBytes` traps or returns `nil`), and the
  caller — typically the Metal `bytesNoCopy` deallocator — becomes responsible for freeing the
  returned base via `AlignedMemory.deallocate(_:)` (== `free`).

VectorCore performs none of the Metal calls; it only provides the page-aligned memory and the
ownership primitive.

---

## Further Reading

- [SoA Layout Contract](./SoA_Layout_Contract.md) — the frozen 0.3.0 SoA in-memory layout
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- [ARM NEON Programming Guide](https://developer.arm.com/architectures/instruction-sets/simd-isas/neon)
- [POSIX memalign documentation](https://man7.org/linux/man-pages/man3/posix_memalign.3.html)
- [Apple Silicon Performance Guide](https://developer.apple.com/documentation/apple-silicon/addressing-architectural-differences-in-your-macos-code)

---

**Document Version**: 0.3.0
**Last Updated**: 2026-06-07
**Applies to**: VectorCore v0.1.0+ (Zero-Copy Buffer Contract & page-aligned producers: v0.3.0+)
