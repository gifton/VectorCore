# SoA memory-layout contract

Status: frozen since VectorCore 0.3.0. This contract covers the public FP32
`SoA<Vector>` layout used by CPU kernels and downstream buffer consumers.
The element type, index formula, and lane stride will not change without a
major-version bump and advance notice to downstream consumers. This specific
layout promise is stronger than the package's general pre-1.0 API policy.

The programmatic descriptor is
[`SoALayout`](../Sources/VectorCore/Storage/SoALayout.swift), exposed by
`SoA.layoutDescriptor` and `SoALayout.forType(_:count:pageAligned:)`.
Regression coverage is in
[SoALayoutContractTests](../Tests/ComprehensiveTests/SoALayoutContractTests.swift).
Layout changes require coordinated updates to the contract, tests, and consumers.

## Layout

Storage is lane-major, then candidate index. Each element is a
`SIMD4<Float>`, containing four consecutive dimensions and occupying 16 bytes.

```text
elementIndex(lane, candidate) = lane * count + candidate
byteOffset(lane, candidate)  = elementIndex(lane, candidate) * 16
```

The candidate axis is never padded. For the built-in conformers every lane is
full: `lanes = dimension / 4`.

| Type | Dimensions | Lanes |
|---|---|---|
| `Vector384Optimized` | 384 | 96 |
| `Vector512Optimized` | 512 | 128 |
| `Vector768Optimized` | 768 | 192 |
| `Vector1536Optimized` | 1536 | 384 |

`DynamicVector` is not `SoACompatible`. The internal `SoAFP16` cache is
not covered by this public FP32 layout contract. Custom `SoACompatible`
conformers are responsible for consistent dimension, lane count, and storage.

## Descriptor

| Member | Meaning |
|---|---|
| `lanes` | SIMD4 lanes per vector |
| `count` | Number of candidates, N |
| `elementStrideBytes` (static) | 16 |
| `laneStrideBytes` | `count * 16` |
| `logicalByteCount` | `lanes * count * 16` |
| `allocatedByteCount` | Logical size for ordinary storage; page-rounded size for nonempty page-aligned storage |
| `elementCount` | `lanes * count` |
| `elementIndex(lane:candidate:)` | Bounds-checked index calculation |

`SoA.init`, `SoA.build`, and `SoALayout.forType` all default to
`pageAligned: false`. Use the same flag when predicting a descriptor and
building its storage.

```swift
import VectorCore

let vectors = (0..<5).map { Vector512Optimized(repeating: Float($0)) }
let soa = SoA(vectors: vectors, pageAligned: true)
let layout = soa.layoutDescriptor
let expected = SoALayout.forType(Vector512Optimized.self, count: 5, pageAligned: true)
precondition(layout == expected)
precondition(layout.lanes == 128 && layout.count == 5)
precondition(layout.laneStrideBytes == 80)
precondition(layout.logicalByteCount == 10_240)
precondition(layout.elementIndex(lane: 127, candidate: 4) == 639)
print("Logical bytes: \(layout.logicalByteCount), allocated bytes: \(layout.allocatedByteCount)")
```

Counts and byte-size calculations must be representable. Constructing a
descriptor does not allocate or validate an external buffer; the consumer
must ensure that its actual storage matches it.

## Logical bytes versus page rounding

For nonempty page-aligned storage, the allocation's base is page-aligned and
its byte length is rounded up to an OS page multiple. Construction zero-fills
the logical elements and trailing padding before storing the candidate data.
Rounding occurs at the end of the whole allocation, not between lanes or
candidates. See [SoA implementation](../Sources/VectorCore/Storage/SoA.swift)
and [SoAPageAlignTests](../Tests/ComprehensiveTests/SoAPageAlignTests.swift).

For 512 dimensions and N=5, logical bytes are 10,240. Allocation bytes are
16,384 with a 16 KiB page, or 12,288 with a 4 KiB page. Neither changes N=5
or the 80-byte lane stride. Never derive candidate count from the allocation
length; use the descriptor.

`pageAlignedBytes` exposes the allocation length, whereas
`withUnsafeRawBuffer` exposes only logical data. A consumer importing the
allocation must check its own API requirements and use the allocation length
where required; a kernel must restrict indexing to the logical region.
For ordinary storage `allocatedByteCount == logicalByteCount` and
`pageAlignedBytes == nil`. Empty SoA storage does not provide a page-aligned
allocation, even when the flag is set.

## Lifetime and transfer

While owned, a nonempty page-aligned SoA exposes
`pageAlignedBytes: (base: UnsafeRawPointer, byteCount: Int)?`.

There are two lifetime models:

- **Borrow:** retain the SoA until all consumers finish, with no ownership
  transfer or incompatible mutation during that interval. The owner remains
  responsible for freeing its memory. Strong retention alone is insufficient
  if another caller can consume the allocation.
- **Transfer:** call `consumeAllocation()` once. On success it returns
  `(base: UnsafeMutableRawPointer, byteCount: Int)`, and the SoA will no
  longer free that allocation. The recipient must release it exactly once,
  using `AlignedMemory.deallocate`, after all CPU/GPU uses finish. It must
  also free it if a later import fails.

`consumeAllocation()` returns `nil` for ordinary, empty, or already
consumed storage. After a successful transfer, do not access the original
SoA's buffer or previously borrowed pointers. Not all CPU accessors enforce
that rule at runtime. Pointer validity depends on both allocation lifetime
and ownership state, not just the lifetime of the Swift object.

`SoA` uses `@unchecked Sendable`; it does not synchronize ownership
transfer, reads, or writes on behalf of callers. A GPU consumer must establish
completion before reuse or release. VectorCore makes no Metal calls.
See [Memory Alignment](Memory_Alignment.md) for allocator and concurrency rules.

## Golden regression case

[SoALayoutContractTests](../Tests/ComprehensiveTests/SoALayoutContractTests.swift)
uses non-power-of-two candidate counts to exercise indexing without accidental
padding. For the five 512-dimensional constant vectors above, a zero query has
squared distances `512 * j * j` for candidate `j`. Descriptor and layout
tests protect the published formulas; they do not validate an external
shader or its synchronization.

The old `blockSize:` initializer argument was removed in 0.3.0. It did not
change this layout. Consumers must not reintroduce candidate-axis padding
based on that historical parameter.
