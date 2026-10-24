# O05 independent design review

**ACCEPT DESIGN.** No concrete blocker or hidden source consumer was found in the inspected scope. This approves the bounded design, not implementation, runtime lifetime proof or audit closure. No builds, source/brief edits, index changes or nested agents were performed.

## Reachability and removal scope

Read the O05 candidate and actual UnifiedStorage, DimensionStorage, three FP16 wrappers, direct FP16 kernel borrow routes, SoA protocol/layout and ownership documentation; compared historical OWN-15/16/17/24 qualifications. Searches of Core Sources/Tests/Benchmarks/Package for StorageSlice, MoveOnlyStorage, MOVE_ONLY_EXPERIMENTAL, .fp16Storage and .slice calls establish the stated bounded inventory:

- StorageSlice occurs only in its internal declaration and ManagedStorage/DimensionStorage slice-returning declarations/forwarding. It stores pointer/count without retaining an owner. The DimensionStorage inline path borrows a local copied value. Delete those dormant regions; preserve the public owned DynamicVector.slice and its existing test.
- MoveOnlyStorage occurs only inside UnifiedStorage's disabled experimental block, including its self-copy method. The package supplies no gate setting. Its consuming into returns storage still covered by unconditional deinit. Removing the complete block is a coherent disposition and requires no gate-enabled ownership-transfer fixture after the route is gone.
- Exactly three public fp16Storage getters return baseAddress from an already-ended ContiguousArray borrow. No executable Core getter caller was found. Similarly named local variables and storage-size counters are not property calls and must not be removed by a broad text match.
- Read-only searches of VectorAccelerate Sources/Tests/Package under VSK/future, and EmbedKit/VectorIndex Sources, found no reference to the removed symbols/getter calls. This is source inventory, not downstream compilation or runtime evidence. Recheck at the actual dispatch base because sibling development can continue independently.

The assigned UnifiedStorage and DimensionStorage deletions do not require changing ManagedStorage's COW implementation or O03's ownership repair. Waiting for O03 integration and applying deletion hunks to that accepted base is essential; do not restore an older full file. MixedPrecisionKernels ownership is likewise exclusive with Q03/Q04. No layout, numerical, conversion, diagnostic or scheduling change is required by this migration.

## Borrow contract and public migration

The proposed synchronous rethrows wrapper forwards the callback inside the immutable ContiguousArray owner's withUnsafeBufferPointer extent. It exposes exactly the existing fixed dimension of UInt16 bits, calls once, and propagates the result or identical error. Nested independent borrows and temporary-expression owners fit this contract without an owning pointer wrapper, asynchronous lifetime promise or new transfer operation.

Generic R deliberately permits callers to return unsafe pointers, so pointer nonescape remains an explicit unsafe caller obligation. The brief correctly prohibits escaping through return/capture/task/foreign callback/storage and does not claim compiler enforcement. The owned Array-returning migration is the appropriate retained/asynchronous data example. Removing the old getters is an authorized source break; keeping a deprecated escaping getter would leave the actual route unclosed and is not part of the chosen design.

Direct dot/Euclidean/cosine kernels already nest storage.withUnsafeBufferPointer callbacks (for example dotFP16_512/768/1536 and euclidean512/768/1536). They need no rewrite or forced forwarding through the new public API. SoAFP16.storage remains an owned ContiguousArray value and is not the escaped getter. Existing I01/S01/N01/Q01 and accepted Q03 semantics must remain intact.

## Verification and qualified claims

The paired compiler fixtures provide honest API migration RED/GREEN: old getters compile before removal and fail afterward; desired scoped methods fail before addition and compile afterward from non-@testable code. They do not constitute a reproduced UAF. All three widths need independent positive/negative checks so one missing declaration cannot be masked by another diagnostic. Normal/throwing/owned-copy public forms and exact diagnostics should be associated with frozen modules and commands.

Patterned UInt16 tests should compare bits directly, including NaN payloads and signed zero, rather than round-trip FP32 conversion. Temporary-owner/nested/throwing tests plus exact result/error identity exercise the callback extent; owned copies must survive dropping/reassigning the original. None should dereference a deliberately expired pointer. A returned raw pointer cannot be used as a negative compiler test because this explicitly unsafe generic-R signature permits it.

A narrow source-symbol absence check is suitable for the deleted internal families, and should match declarations/member accesses rather than ban every local identifier named fp16Storage. Public breaks require actual compiler fixtures. Existing direct-kernel tests with scalar references are preservation controls; no new precision or kernel-selection claim is needed. Focused ASan supplements the valid borrow/transfer controls and cannot prove the absence of all lifetime defects or validate an intentionally invalid custom conformer.

OWN-24 remains qualified documented context. The dimension8/lanes2 custom conformer with exactly two SIMD4 storage lanes is a valid external-style control for N0/1/5 and both allocation modes. Check lane-major values, logical length and page-rounded allocation length independently; an empty page-aligned case need not own a nonzero allocation. Preserve the frozen FP32 SoA contract and transfer identity/matching recipient deallocation tests. Caller synchronization and custom-conformer consistency remain duties, not newly checked constructor guarantees; no GPU import or concurrent transfer execution follows from these tests.

The adjacent AlignedMemory documentation correction is explicitly scoped to accepted O04 count/alignment validation and should cite those existing tests, while keeping persistent owned SoA/page allocations distinct from temporary FP16 borrows. No broader alignment, nonescape or safe-transfer promise is justified. Final disposition should say dormant routes removed and unsafe public borrowing migrated, preserving historical source-established versus runtime-reproduced distinctions.
