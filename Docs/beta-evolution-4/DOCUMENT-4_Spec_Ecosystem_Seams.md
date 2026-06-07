# DOCUMENT-4 — Design Spec: Ecosystem Seams

**Verdict:** Accept for `0.3.0`. **Highest leverage, lowest cost** of the entire suite.
**Scope:** Thin, additive API seams that the downstream packages have *explicitly requested* and
that `VectorIndex` is currently working around by bypassing Core. None of this is in the external
proposal; all of it is in the downstream repos' own request lists.

> **Why this matters more than the math.** `VectorIndex/IMPROVEMENTS.md` documents that it
> "intentionally bypasses most of VectorCore's type-safe abstractions for performance, using raw
> `[Float]` arrays and `UnsafePointer<Float>` throughout." Every seam below removes a reason to
> bypass Core. The win is ecosystem cohesion — the suite converging *onto* Core instead of
> routing *around* it.

## Implementation status (updated 2026-06-06, during execution)

Reviewing the live `0.2.2` source during implementation revealed that **S1 and S2 were already
shipped** (the BE3 / API-surface-cleaning cycle landed them) and **S3 is partially present**. The
spec below describes the intended surface; this table is the accurate state:

| Seam | Status | Notes |
|---|---|---|
| **S1** pointer Top-K | ✅ Already implemented | `TopKSelection.select(k:from:UnsafePointer<Float>,count:,ids:)` exists (`Operations/TopKSelection.swift`), returns `[Int32]` with `ids` remap |
| **S2** in-place normalize | ✅ Already implemented | `NormalizeKernels.normalizeUnchecked(_:dimension:)` (public, Kahan two-pass, inherits BE3 stability fixes) |
| **S3** tie-breaking | ✅ Implemented this cycle | `TieBreaker` enum (`smallerIndex` default / `insertionOrder` / `smallerValue`) threaded through array, pointer, generic, and optimized paths; array path unified onto `TopKBuffer` so boundary membership *and* result order are deterministic. 7 tests |
| **S4** provider protocols | ◻️ Open | `ComputeProvider`/`BufferProvider`/`AccelerationProvider` already exist; conformance contract + tests not written |
| **S5** zero-copy contract | ✅ Implemented this cycle | `UnifiedVectorBuffer` + `PageAlignedBuffer` (`Storage/UnifiedVectorBuffer.swift`): page-aligned/page-length, ownership transfer, optimized + dynamic conformances, 8 tests |

Remaining open work: S4's conformance contract + tests (medium). S1/S2 need only regression tests
to lock the behavior VectorIndex depends on.

---

## S1 — Pointer-level Top-K selection

**Requested:** `VectorIndex/IMPROVEMENTS.md` §9.5 (item 1), costing 15–20% per hot-path call to
array bridging today.

**Today:** `Operations/TopKSelection.swift` exposes `select(k:from: [Float])` and typed overloads
— all array-based. Wrapping a raw distance buffer in `[Float]`/`DynamicVector` to call them costs
two allocations.

**Add (additive overload):**
```swift
public extension TopKSelection {
    static func select(
        k: Int,
        from distances: UnsafePointer<Float>,
        count: Int,
        ids: UnsafePointer<Int32>? = nil,
        tieBreaker: TieBreaker = .smallerIndex      // see S3
    ) -> TopKResult
}
```
Reuses the existing `TopKBuffer` heap (`Operations/Kernels/TopKSelectionKernels.swift`) — only
the entry point changes. Zero allocation given caller-owned buffers.

---

## S2 — In-place raw-buffer normalize

**Requested:** `VectorIndex/IMPROVEMENTS.md` §9.5 (item 2). Core's existing
`normalizedUnchecked()` (`Vectors/Vector512Optimized.swift:231` et al.) returns a *new value* and
only accepts `VectorProtocol` types; VectorIndex must wrap raw buffers in `DynamicVector` (two
allocations) to use it.

**Add:**
```swift
public extension Operations {
    /// L2-normalize a raw buffer in place. Precondition-checked, allocation-free.
    static func normalizeUnchecked(
        _ buffer: UnsafeMutablePointer<Float>,
        dimension: Int
    )
}
```
Implementation reuses `NormalizeKernels` (`Operations/Kernels/NormalizeKernels.swift`) operating
directly on the pointer. Inherits the `0.2.2` BE3 normalization fixes (subnormal-reciprocal
guard, `Float.leastNormalMagnitude` floor) — which is precisely why this should live in Core once
rather than be re-derived in VectorIndex.

---

## S3 — Configurable tie-breaking

**Requested:** `VectorIndex/IMPROVEMENTS.md` §9.5 (item 3). VectorIndex needs deterministic
`smallerIndex` tie-breaking; Core's Top-K currently fixes a single policy.

**Add:**
```swift
public enum TieBreaker: Sendable {
    case insertionOrder   // current behavior (default preserved for source compat)
    case smallerIndex     // VectorIndex's required deterministic order
    case smallerValue
}
```
Threaded through `TopKSelection.select(...)` and `TopKBuffer.pushIfBetter`. Default stays
`insertionOrder` so existing callers are unaffected; VectorIndex opts into `smallerIndex`.

---

## S4 — Stabilize provider protocols (they already exist)

**Requested:** `VectorAccelerate/docs/ADD/VECTORCORE_ALIGNMENT_ROADMAP.md` — VA "implements 2 of
7+ available protocols" and wants to conform to three more.

**Key finding:** the protocols VA names — `ComputeProvider`, `BufferProvider`,
`AccelerationProvider` — **already exist in Core** (`Protocols/ComputeProvider.swift`,
`Protocols/BufferProvider.swift`, `Protocols/CoreProtocols.swift`). This is **not a "define new
protocols" task.** The seam work is:

1. **API-stability review** of those three protocols against VA's intended conformances
   (`ComputeEngine: ComputeProvider`, `BufferPool: BufferProvider`,
   `MetalContext: AccelerationProvider` per the roadmap), so VA can implement without Core
   churning the signatures underneath it.
2. **Fill capability gaps** the roadmap surfaces (e.g. `BufferProvider` must reconcile with VA's
   `BufferToken`/`BufferHandle` lifecycle; `AccelerationProvider` capability-query hooks).
3. **A conformance-contract doc + protocol-level tests** in Core so a downstream conformance is
   verifiably "correct," not best-effort.

**Explicitly not in scope:** Core implementing any of these with GPU code. Core owns the
*protocol*; VA owns the Metal-backed conformance. This is the boundary that keeps "zero GPU
dependencies" intact while still unblocking VA.

---

## S5 — Zero-copy buffer contract

**Requested (implicitly, by all three):** EmbedKit lists "zero-copy pipelines" as a P1
(`EmbedKit/dev_EKRefactor/Future_Improvements.md`); its current path copies
`MLMultiArray → [Float] → DynamicVector`. VectorAccelerate consumes via
`MTLStorageModeShared`/`makeBuffer(bytesNoCopy:)`. The original ecosystem note's "Finding 1"
(cross-package bridging penalty) is the same problem.

**Add — a contract, not a subsystem.** Core already guarantees aligned allocation
(`Storage/AlignedMemory.swift`, `posix_memalign`, 64-byte default; `Docs/Memory_Alignment.md`).
The seam is a documented, page-alignable raw view:
```swift
public protocol UnifiedVectorBuffer {
    /// Contiguous, alignment-guaranteed base for the vector's elements.
    func withUnsafeContiguousBytes<R>(_ body: (UnsafeRawBufferPointer) throws -> R) rethrows -> R
    var elementCount: Int { get }
    var alignment: Int { get }            // ≥ 64; page-aligned variant for bytesNoCopy
}
```
- Optimized vector types and a page-aligned allocator variant conform.
- **No Metal/IOSurface import in Core.** Core guarantees alignment + contiguity; `VectorAccelerate`
  calls `makeBuffer(bytesNoCopy:)` on the exposed pointer on *its* side; `EmbedKit` writes
  CoreML output into a Core-provided aligned buffer once and hands the pointer downstream.
- `makeBuffer(bytesNoCopy:)` requires **page-aligned base and length** — so the contract
  specifies an opt-in page-aligned (16 KB on Apple Silicon) allocation path beyond the default
  64-byte SIMD alignment. This is the concrete, CPU-only deliverable that satisfies the
  proposal's zero-copy "UnifiedVectorBuffer" idea without pulling GPU types into Core.

---

## Validation

1. **Allocation gates:** S1/S2 over caller-owned buffers assert `mallocCountTotal == 0`
   (harness decision per DOCUMENT-2 §6.3).
2. **Parity:** pointer-level `select` (S1) vs array `select` produce identical `TopKResult`;
   in-place normalize (S2) vs `normalizedUnchecked()` agree within `1e-6`.
3. **Tie-break determinism (S3):** seeded equal-distance inputs yield the policy-specified order.
4. **Protocol conformance tests (S4):** Core ships a test a downstream provider can run to prove
   correct conformance; validate against a mock provider in Core's own suite.
5. **Alignment contract (S5):** page-aligned path returns base & length both `% pageSize == 0`;
   round-trip a buffer through the contract and assert pointer identity (no copy).

---

## Sequencing note

S1–S3 are tiny and unblock VectorIndex immediately — ship first in `0.3.0`. S4 is a review +
hardening pass (no new types). S5 is the largest (new allocation path + protocol) but is the one
that pays off the whole-suite "Finding 1" bridging penalty, so it anchors the release.

| Seam | Requested by | Existing Core surface | New surface |
|---|---|---|---|
| S1 pointer Top-K | VectorIndex §9.5 | `TopKSelection.select`, `TopKBuffer` | 1 overload |
| S2 in-place normalize | VectorIndex §9.5 | `NormalizeKernels`, `normalizedUnchecked()` | 1 function |
| S3 tie-breaking | VectorIndex §9.5 | `TopKBuffer.pushIfBetter` | 1 enum + param |
| S4 provider protocols | VectorAccelerate roadmap | `ComputeProvider`, `BufferProvider`, `AccelerationProvider` (exist) | stability review + tests |
| S5 zero-copy contract | EmbedKit P1 / Finding 1 | `AlignedMemory`, `Docs/Memory_Alignment.md` | `UnifiedVectorBuffer` + page-aligned path |
