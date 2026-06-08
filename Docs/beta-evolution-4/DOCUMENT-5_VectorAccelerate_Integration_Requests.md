# DOCUMENT-5 — VectorAccelerate Integration Requests (gap analysis + plan)

**Source:** `future/VectorAccelerate/docs/VECTORCORE_INTEGRATION_REQUESTS.md` (filed vs VectorCore ≥ 0.2.2).
**Status:** Verified against the live source + the `v0.2.2` tag during the beta-evo-4 branch.
**Verdict:** None of R1/R2/R4 are supported yet; R3 rests on an incorrect premise. Adding the
P1 blockers (R1+R2) to the beta-evo-4 scope before merge, with R4 (P2) and R3 (a reply) tracked.

VectorAccelerate's `makeNoCopyBuffer(bytes:length:)` (page-aligned region → shared `MTLBuffer`,
zero-copy on UMA) is ready to consume aligned Core storage. The blocker is entirely on Core's side:
the high-value object to bridge — the `SoA` candidate database — is neither page-aligned (R1) nor
publicly addressable (R2).

---

## R1 — Page-align the SoA batch buffer (P1, blocks zero-copy batch search)

**Status: NOT supported.** `Sources/VectorCore/Storage/SoA.swift:67` allocates
`UnsafeMutablePointer<SIMD4<Float>>.allocate(capacity: bufferCapacity)` (≈16-byte alignment) and
frees via `.deallocate()` (`deinit`, ~L76–81). `makeBuffer(bytesNoCopy:)` needs page alignment
(16 KB) for both base and length, so it rejects this pointer → VA falls back to a full copy.

**Plan (opt-in, to avoid bloating every SoA to a 16 KB page):**
- Add a page-aligned allocation mode to `SoA` (e.g. `SoA.build(from:pageAligned: Bool = false)` or
  a dedicated initializer). When set, allocate the buffer via
  `AlignedMemory.allocateAligned(type: SIMD4<Float>.self, count: paddedCapacity, alignment: PlatformConfiguration.pageSize)`
  and **round the byte length up to a whole page** (as `PageAlignedBuffer` does) so the `bytesNoCopy`
  length requirement is met; zero-fill the padding lanes.
- `deinit` must free via the matching allocator: track an `ownsViaAlignedMemory` flag and call
  `AlignedMemory.deallocate(...)` (→ `free`) for the page-aligned path, `.deallocate()` otherwise.
  (Mismatched free is UB — the BE3 audit fixed exactly this class of bug.)
- Default stays 16-byte (no memory regression for the common CPU-only SoA).

## R2 — Publicly expose the SoA buffer pointer + byte length (P1)

**Status: NOT supported.** `SoA.swift:57` `buffer` is `@usableFromInline internal`; there is no
public raw accessor and `SoA` does not conform to `UnifiedVectorBuffer`.

**Plan:**
- Add a public, escaping accessor for the `bytesNoCopy` hand-off, gated on page alignment so VA
  only gets a pointer it can actually wrap:
  ```swift
  /// Page-aligned base + page-rounded byte length, or nil if this SoA was not built
  /// page-aligned. The SoA MUST outlive any MTLBuffer created from this pointer
  /// (hold a strong reference, or use a deallocator handshake).
  public var pageAlignedBytes: (base: UnsafeRawPointer, byteCount: Int)? { get }
  ```
- Also add a scoped `withUnsafeRawBuffer { (UnsafeRawPointer, Int) in … }` for read-only CPU use.
- Document the **lifetime contract** explicitly (the request calls this out): the MTLBuffer borrows
  Core-owned memory; Core frees it on `SoA` deinit, so the SoA must outlive the buffer.

## R3 — Confirm the release that ships page alignment (reply, not code)

**Status: premise incorrect — needs a correction sent to VA.** Verified at the `v0.2.2` tag and in
the working copy: page alignment did **not** land on `AlignedMemory` or `AlignedDynamicArrayStorage`
— both use `optimalAlignment` (64 bytes), not page size. `getpagesize()` lives only in
`PlatformConfiguration.pageSize`. The **only** page-aligned allocation in VectorCore is
`PageAlignedBuffer` (`Storage/UnifiedVectorBuffer.swift`), **added in this beta-evo-4 branch and not
yet released**.

**Reply to VA:** pin your floor to the release beta-evo-4 ships as (target **0.3.0**), not 0.2.2 —
and note that even 0.3.0 does not page-align `SoA` until R1 lands. The 64-byte `AlignedMemory` path
they referenced is not `bytesNoCopy`-eligible.

## R4 — `BatchKernelProvider` hook for transparent GPU dispatch (P2)

**Status: NOT supported.** No `BatchKernelProvider` exists; `Operations.findNearest`/batch paths
dispatch on concrete vector type, never downcasting the provider to a GPU kernel protocol;
`findNearestGPU` (`ExecutionOperations.swift:169`) is an internal stub that throws.

**Plan (matches VA's proposed shape):**
```swift
public protocol BatchKernelProvider: ComputeProvider {
    func batchDistance<V: VectorProtocol>(query: V, candidates: [V], metric: any DistanceMetric)
        async throws -> [Float] where V.Scalar == Float
    func findNearest<V: VectorProtocol>(query: V, candidates: [V], k: Int, metric: any DistanceMetric)
        async throws -> [(index: Int, distance: Float)] where V.Scalar == Float
}
```
- In `Operations.findNearest` (and the batch-distance path), downcast `computeProvider as? BatchKernelProvider`
  and delegate when present, else the existing CPU path. Composes with S4 (the conformance contract
  gets a `BatchKernelProvider` section + a mock test).
- Interaction with the new CPU GEMM routing (DOCUMENT-2): the GPU provider, when installed, takes
  precedence; CPU GEMM remains the default when none is installed. Document the precedence order.

---

## Recommendation for this branch (before merge)

| Req | Priority | Recommendation |
|---|---|---|
| R1 page-align SoA | P1 (blocks zero-copy batch) | **Build before merge** — on-theme with S5, completes the actual GPU-bridge object |
| R2 public SoA accessor | P1 (blocks zero-copy batch) | **Build before merge** — trivial once R1 exists |
| R3 version reply | — | **Reply now** (no code): correct the premise; floor = 0.3.0, gated on R1 |
| R4 BatchKernelProvider | P2 (transparent dispatch) | Build this branch *or* fast-follow; low risk, composes with S4 |

R1+R2 are the blocking pair and the natural completion of the S5 zero-copy contract (which today
covers vector types and `PageAlignedBuffer` but not `SoA`). R4 is independent and can land alongside
or immediately after.

## Status (implemented on feature/beta-evo-4)

- **R1 — page-align SoA:** ✅ `SoA.build(from:pageAligned:)` / `init(vectors:…:pageAligned:)` allocate
  via `AlignedMemory` (page-aligned base, page-rounded length, allocator-correct deinit). Opt-in;
  default stays 16-byte.
- **R2 — public accessor:** ✅ `SoA.pageAlignedBytes` (bytesNoCopy base + page-rounded length, lifetime
  contract documented) and `withUnsafeRawBuffer { (UnsafeRawBufferPointer) in … }`.
- **R4 — BatchKernelProvider:** ✅ protocol added (`Protocols/BatchKernelProvider.swift`);
  `Operations.findNearest` / `findNearestBatch` downcast the installed `computeProvider` and delegate
  (precedence over the CPU fast paths and the CPU GEMM routing).
- **R3 — version reply (sent):** page alignment was never on `AlignedMemory`/`AlignedDynamicArrayStorage`
  (those are 64-byte). Page-aligned storage (`PageAlignedBuffer`, and now opt-in `SoA`) ships in the
  **0.3.0** release that includes beta-evo-4 — **pin your floor to 0.3.0**, not 0.2.2.

### Post-review refinements (final review findings)

- **F2 — batched GPU dispatch:** `BatchKernelProvider` gained `findNearestBatch(queries:…)` with a
  default looping implementation, and `Operations.findNearestBatch` dispatches to it — so a Metal
  conformer can encode the whole query set in one kernel (the real GPU win) instead of per-query.
- **F3 — ownership handoff:** `SoA.consumeAllocation()` (mirrors `PageAlignedBuffer`) transfers the
  page-aligned buffer to a Metal `bytesNoCopy` deallocator without a double free; after consuming,
  the `SoA` no longer frees on deinit and `pageAlignedBytes` returns `nil`.
- **F1 — `batchDistance` doc:** clarified it's for direct consumer use (reranking), not routed by
  `Operations`' k-NN entry points.
