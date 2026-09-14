# Memory, alignment, and ownership

This reference describes the public CPU-side memory interfaces in this
checkout. VectorCore does not perform GPU import or synchronization.
For the FP32 lane layout, see [SoA Layout Contract](SoA_Layout_Contract.md).

## Choose the contract you need

| Interface | Alignment and lifetime | Ownership |
|---|---|---|
| Vector `withUnsafeBufferPointer` APIs | Scoped borrow; follow the concrete type's contract | Vector retains storage |
| `UnifiedVectorBuffer.withUnsafeContiguousBytes` | Contiguous logical Float bytes, aligned to the value's reported `alignment`; pointer must not escape the closure | Conformer retains storage |
| `PageAlignedBuffer` | Page-aligned base and page-rounded allocation length; explicit owned-pointer access | Object owns allocation until consumed |
| `SoA<V>` | Lane-major SIMD4 storage; page alignment is opt-in | Object owns allocation until consumed |
| `AlignedMemory.allocateAligned` | Caller-selected alignment, subject to preconditions; uninitialized storage | Caller initializes and frees |
| `MemoryPool` | Reusable allocations; requested alignment and handle lifetime matter | Pool/handle lifecycle manages reuse |

The optimized 384/512/768/1536 vector types report
`MemoryLayout<SIMD4<Float>>.alignment` (16 bytes) through
`UnifiedVectorBuffer`; `DynamicVector` reports
`MemoryLayout<Float>.alignment` (4 bytes). Neither guarantees page alignment.
The protocol itself is read-only: it does not authorize shared mutation.
See [the protocol and conformances](../Sources/VectorCore/Storage/UnifiedVectorBuffer.swift).

## Logical bytes and allocation bytes

A `PageAlignedBuffer` with `n` Float elements has:

- `byteCount = n * MemoryLayout<Float>.stride`: the logical data size.
- `allocatedByteCount`: the logical size rounded up to an OS page multiple.
- `alignment == pageSize`: the guaranteed base alignment.

Construction initializes both logical data and padding to zero. Scoped reads
and mutable Float-buffer access expose only the logical elements. Padding is
not additional vector data.

```swift
import VectorCore

let buffer = PageAlignedBuffer(copying: [Float](repeating: 2, count: 512))
precondition(buffer.byteCount == 512 * MemoryLayout<Float>.stride)
precondition(buffer.allocatedByteCount >= buffer.byteCount)
precondition(buffer.allocatedByteCount % buffer.pageSize == 0)
buffer.withUnsafeMutableBufferPointer { values in
    values[0] = 3
}
let first: Float = buffer.withUnsafeContiguousBytes { bytes in
    bytes.load(as: Float.self)
}
precondition(first == 3)
print("Logical bytes: \(buffer.byteCount), allocated bytes: \(buffer.allocatedByteCount)")
```

This is a bounded example, not a size-validation API. Construction requires a
positive count; byte multiplication and page rounding must be representable.
`PageAlignedBuffer(copying:)` therefore also requires a nonempty array.
Allocation failure terminates via `fatalError`; this initializer is not
throwing. See [implementation](../Sources/VectorCore/Storage/UnifiedVectorBuffer.swift)
and [UnifiedVectorBufferTests](../Tests/ComprehensiveTests/UnifiedVectorBufferTests.swift).

## Borrowing versus transferring ownership

For an asynchronous consumer, choose one ownership model explicitly.

**Borrow an owned allocation.** Retain the owner until every consumer has
finished accessing the memory. Do not consume or mutate it incompatibly during
that interval. A strong reference alone is insufficient if another user can
call `consumeAllocation()`. Pointers obtained through scoped closures must
still remain inside those closures; use the allocation's explicit borrowing
interface when a persistent borrow is required.

**Transfer an allocation.** `PageAlignedBuffer.consumeAllocation()` returns
`(baseAddress, allocatedByteCount)`, marks ownership as transferred, and
prevents further buffer access through its checked accessors. The new owner
must eventually call `AlignedMemory.deallocate` on the returned allocation,
exactly once, after all uses finish. It must also clean up if a subsequent
consumer/import operation fails. Consuming twice violates a precondition.

Page-aligned `SoA.consumeAllocation()` instead returns an optional
`(base, byteCount)`; it returns `nil` for non-page-aligned, empty, or already
consumed storage. Treat a successful transfer as invalidating all CPU buffer
access through the original SoA, even where an accessor does not check this.
See [the SoA ownership contract](SoA_Layout_Contract.md#lifetime-and-transfer).

The consumer is responsible for checking its import API's alignment, length,
device, and storage-mode requirements. Meeting Core's allocation contract does
not by itself establish that a GPU import succeeds or that CPU/GPU accesses are
synchronized.

## Explicit aligned allocation

[`AlignedMemory`](../Sources/VectorCore/Storage/AlignedMemory.swift) uses
`posix_memalign`. Its default `optimalAlignment` is currently 64 bytes on
arm64/x86_64 and 16 bytes on other architectures; this is an allocation policy,
not the OS page size or a statement about a particular processor's cache line.

Supply a positive, representable element count. Alignment must be a positive
power of two, at least `minimumAlignment` (16), and sufficient for the
element type. The allocator checks the first two alignment constraints, but
does not provide comprehensive count/overflow/type-alignment validation.

```swift
import VectorCore

let count = 512
let pointer: UnsafeMutablePointer<Float> = try AlignedMemory.allocateAligned(
    count: count, alignment: 64)
pointer.initialize(repeating: 0, count: count)
defer {
    pointer.deinitialize(count: count)
    AlignedMemory.deallocate(pointer)
}
precondition(AlignedMemory.isAligned(pointer, to: 64))
pointer[0] = 1
print("First element: \(pointer[0])")
```

Initialize before reading, deinitialize initialized nontrivial elements when
required, and match the allocator with `AlignedMemory.deallocate` (which
uses `free`). Do not use Swift pointer `.deallocate()` on these allocations.
Allocation failure throws `VectorError.allocationFailed`; invalid alignment
preconditions can trap before allocation.

## Pools, copying, and concurrency

[`MemoryPool`](../Sources/VectorCore/Utilities/MemoryPool.swift) is a class,
not an actor. It synchronizes pool bookkeeping with an explicit lock. That
does not serialize reads or writes to a checked-out raw allocation. Do not
retain a pointer beyond its handle's return to the pool, and do not return a
buffer while another consumer still uses it. Pool allocations are not a
substitute for initializing typed elements or managing their destruction.
Relevant regression coverage is in
[MemoryPoolTests](../Tests/ComprehensiveTests/MemoryPoolTests.swift).

`PageAlignedBuffer`, `SoA`, and `MemoryPool` use `@unchecked Sendable`.
Their callers must preserve the ownership, access, and synchronization
invariants; the annotation is not a lock around arbitrary pointer use.

Ordinary vector copies may share copy-on-write storage. Mutation, conversion
to a different representation, candidate packing, and scratch/result creation
may allocate or copy. Generic
[`DimensionStorage`](../Sources/VectorCore/Storage/DimensionStorage.swift)
defaults to managed heap storage above 16 elements. Neither alignment nor
value semantics establishes allocation-free execution.

## Unsafe-call checklist

Before passing storage across a subsystem boundary, establish:

1. Every count, product, byte offset, and rounded size is valid and representable.
2. The initialized region covers all reads and the writable region covers all writes.
3. Base alignment and element binding match the consumer's requirements.
4. Borrowed pointers stay within scope; persistent allocations outlive every use.
5. Mutation respects aliasing, Swift exclusivity, and concurrent-access rules.
6. One owner releases the allocation using the matching allocator, including failure paths.

These are caller obligations, not a claim that all public unsafe APIs check
them. For behavior involving non-finite values and precision, see
[Numerical Behavior](Numerical_Behavior.md).
