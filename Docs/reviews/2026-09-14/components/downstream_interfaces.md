# Component 11 — Public interfaces and downstream architecture

Review revision: 8b02e75bca7055ae11170873d15c06f1538709b7. Read-only review; no repository or downstream edits/builds.

## Reviewed scope and validation

Read entire Core `Protocols/ProviderProtocols.swift`, `BatchKernelProvider.swift`, `BufferProvider.swift`, `ComputeProvider.swift`, `DistanceMetric.swift`, `Integration/IndexableVector.swift`, `SearchResult.swift`, `VectorCollection.swift`, `VectorCore.swift`, `Version.swift`, `Package.swift`, `Platform/SIMDProvider.swift`; provider declarations in `Platform/ArraySIMDProvider.swift`; `Docs/Package_Boundaries.md`, `API_Overview_Map.md`, `Memory_Alignment.md`, `SoA_Layout_Contract.md`, `Numerical_Behavior.md`. Traced relevant portions of Operations dispatch/provider reads, NormalizeKernels, IntegrationProtocolsTests and public mixed-precision declarations.

Read actual VectorIndex `TypedOverloads.swift`, manifest/resolved file and HNSW inverse-norm cache implementation; traced query normalization flags in FlatIndex/FlatIndexOptimized/IVF. Read VectorAccelerate manifest/resolved, VectorCoreIntegration, StableNormalization, MetalComputeProvider's BatchKernelProvider bridge, cosine provider implementation, BufferPool's BufferProvider conformance, related CPU normalization callers and documentation. These are local source snapshots, not a release-wide compatibility build.

Compiled standalone public client `interfaces_probe.swift` against the orchestrator's existing Debug objects using `build_probe.py`; ran successfully after isolating the separately attempted zero-normalization assertion. Main output:

```text
chebyshev expected ID2 distance2; actual [2] [2.828427]
update rejected: VectorError.dimensionMismatch: Expected dimension 2, but got 1
original exists after failed update false
auto ID after explicit0 0 count 1 id0 [3.0, 4.0]
score -2.0 -1.0
score -1.0 inf
score -0.5 2.0
score 0.0 1.0
score 1.0 0.5
```

No timing benchmark, full test run, GPU execution or downstream build was performed by this component.

## Confirmed actionable Core findings

### IF1 — P2 correctness/API: Collection search silently executes a different metric

**Location:** `Sources/VectorCore/Integration/VectorCollection.swift:120-131` (especially :129-131).

`VectorCollection.search(query:k:metric:)` accepts `any DistanceMetric`, recognizes only Euclidean, cosine, Manhattan and negative dot product, and substitutes Euclidean for everything else. Built-in Chebyshev, Hamming and Minkowski as well as custom metrics therefore return wrong scores and potentially wrong neighbors. This is a reachable public default on SimpleVectorCollection and any consumer conformance inheriting it; the protocol offers no documented restriction to its four recognized metrics.

**Reproduction:** Query `[0,0]`, candidate `[2,2]`, Chebyshev returns `2.828427` rather than `2` in the executed public probe. A rank-reversal example derivable directly from the same implementation is candidates `[2.5,0]` and `[2,2]`: Chebyshev should choose the second (2 < 2.5), but Euclidean chooses the first (2.5 < sqrt(8)). The probe used `[3,0]` and thus demonstrated wrong score, not rank reversal.

**Remedy/test:** Constrain the existential to `any DistanceMetric<Float>` and invoke its actual witness (the downstream MetalComputeProvider already demonstrates this technique at :333-336). If retaining the unconstrained signature, reject unsupported scalar types explicitly instead of substituting a metric. Add a table-driven test spanning every built-in metric plus a custom metric deliberately giving opposite ordering from Euclidean. Existing IntegrationProtocolsTests exercise only Euclidean/cosine collection search.

**Confidence:** High; executed public-client result and direct source derivation.

### IF2 — P2 correctness: A rejected update permanently deletes the existing vector

**Location:** `Sources/VectorCore/Integration/VectorCollection.swift:239-244`.

Default `MutableVectorCollection.update` checks existence, removes the old value, then calls throwing `add`. A dimension mismatch or any other add failure leaves the collection missing its original value. Example exercised: dimension-2 collection, existing id7 `[1,2]`, update id7 to `[3]`; error is returned and `vector(for:7)` becomes nil. The advertised error condition is routine input validation, not deletion.

**Remedy/test:** Require implementers to implement replacement atomically (SimpleVectorCollection can validate and replace while holding its lock). The generic remove-then-add default cannot guarantee atomicity or successful rollback for arbitrary conformers. At minimum prevalidate shape before mutation, but that alone does not close other conformer-specific failure paths. Add mechanical failure-injection tests asserting original state survives each throwing replacement path; also define concurrent search/update snapshot semantics. No failed-update coverage was found in IntegrationProtocolsTests.

**Confidence:** High; executed public client.

### IF3 — P2 correctness: Explicit and generated IDs silently overwrite records

**Location:** `Sources/VectorCore/Integration/VectorCollection.swift:286-305`, protocol contract :213.

`add(id:vector:)` unconditionally assigns a dictionary entry, despite the protocol documenting that an existing ID throws. It also never advances or reserves `nextID`. Following an explicit `add(id:0, ...)`, `add(vector:...)` returns 0 and replaces that record. This is data loss on an ordinary combination of two public insertion APIs, independent of the update problem.

**Reproduction:** Executed explicit id0 `[1,2]`, generated add `[3,4]`: returned id0, count1, stored id0 `[3,4]`.

**Remedy/test:** Reject duplicate explicit IDs and generate an unused ID under the lock; define exhaustion behavior without integer overflow. Test explicit→generated, generated→explicit, sparse explicit IDs, remove/reinsert and duplicate insert. The reference implementation's small-dataset positioning reduces scale relevance, not correctness relevance; it should model the contract downstream authors are meant to follow.

**Confidence:** High; executed public client and explicit protocol mismatch.

### IF4 — P2 score semantics: computedScore is invalid for Core's negative-dot distances

**Location:** `Sources/VectorCore/Integration/SearchResult.swift:34-37`; producer `VectorCollection.swift:195-197,168`; negative-dot convention `Protocols/ProviderProtocols.swift:95` and `Operations/DistanceMetrics.swift`.

`computedScore` automatically applies `1/(1+distance)` although Core supports unbounded negative dot-product distances. Ordinary dot product 1 becomes distance -1 and score +Inf; dot product 2 becomes distance -2 and score -1. More similar matches can therefore acquire lower relevance than weaker matches across the pole, contradicting both the `(0,1]` documentation and higher-is-better relevance meaning. No manually forged invalid distance is needed: `searchDotProduct` constructs SearchResult without a score and reaches this fallback.

**Remedy/test:** Carry score kind/metric semantics or require an explicit transform for distance families that are not nonnegative. Avoid guessing a universal bounded relevance transform from an untyped Float. Add public search→computedScore tests with inner products 0.5,1,2 and ensure ordering, finiteness and documented range. Existing tests use only distances 0 and 1.

**Confidence:** High; executed transform results and direct producer trace. Actual downstream package source did not call computedScore in the reviewed searches; this remains an advertised public consumer trap rather than an observed downstream regression.

### IF5 — P2/P1 candidate, duplicate of orchestrator finding: normalization metadata is trusted far beyond the wrapper

**Core locations:** `Sources/VectorCore/Integration/IndexableVector.swift:107` public mutable `vector`, :203-208 mutation invalidation after a throwing closure; root already reproduced retained normalized=true/cachedMagnitude1 despite actual magnitudes2/3. Do not count this as another independent defect.

**Actual downstream consequence:**

- VectorIndex `TypedOverloads.swift:119-123,135-141` turns isNormalized/cachedMagnitude into a stored inverse norm. `HNSWIndex.swift:1097-1129` appends a supplied finite positive inverse norm directly when its incremental cache is live. Thus a stale hint can enter the ANN traversal's stored-vector normalization cache; fixing the wrapper later does not repair values already copied into an index.
- `TypedOverloads.swift:102,152,178,200` skips query normalization for hinted values. Flat/IVF/optimized-flat and HNSW therefore consume the metadata in real searches, not just documentation examples.
- VectorAccelerate `Integration/KernelDistanceProviders.swift:189-190,200-221` and :243-245/:273/:290 choose raw dot-product kernels when both arguments claim normalization, and transform to `1-dot`. A stale hinted `[2,0]` and valid hinted `[1,0]` consequently give -1 instead of cosine distance0 on that route. This consequence is source-derived; no GPU run was performed.
- `AcceleratedVectorIndex.swift:672-673,697-698` captures the metadata on insertion. Capture is confirmed; do not infer every captured field is consumed by every index route without an additional trace.

**Remedy/test priority:** Treat the wrapper as an invariant-bearing type: restrict or observe wrapped-value mutation and invalidate with `defer` or before entering the mutating closure. Test every public mutation surface including throwing buffers. Cross-package tests should compare hinted and unhinted vectors after mutation, using both query-side and indexed-side hints. Explicit `isNormalized:true` construction remains a caller assertion; do not conflate misuse of that initializer with the wrapper's failure to invalidate initially correct hints.

## Architecture priorities grounded in actual call sites

### A1 — Unify score, scalar and numerical contracts at the provider seam

`DistanceMetric<Scalar>` expresses scalar type; `BatchKernelProvider` and VectorCollection erase it to unconstrained `any DistanceMetric` even though their vectors/results are Float. VA currently recovers `any DistanceMetric<Float>` with a runtime cast. Prefer making the Float constraint explicit at the boundary.

The two metric APIs have materially different dot-product semantics today: Core VectorProtocol's `distance(to:metric:.dotProduct)` returns negative dot; VA's direct `MetalComputeProvider.distance(...metric: SupportedDistanceMetric)` returns raw dot at `MetalComputeProvider.swift:385`, as do its direct similarity-oriented paths. VA deliberately excludes dot product from `gpuMetric(for:)` and uses Core's actual metric witness for BatchKernelProvider fallback (:314-336), preserving the transparent Operations route. This is a useful existing correctness guard, but the shared enum/protocol does not itself express whether a returned Float is distance or similarity. Introduce an explicit score convention and a shared conformance suite, rather than relying on each consumer to remember sign changes.

Core BatchKernelProvider docs specify candidate order and ascending neighbor distances, but do not fully fix NaN membership/index-tie semantics to the new CPU Top-K contract. Operations explicitly lets installed providers supply their own ordering. This is documented variation, not a proved Core defect; decide whether a numerical foundation should promise a stronger common contract.

### A2 — Consolidate real execution APIs and stop presenting unused settings as active hooks

`Operations.bufferProvider` at `Operations.swift:42` has no source reads anywhere else in Sources/VectorCore. Binding a custom pool therefore cannot change Core's internal allocation behavior. VectorAccelerate does implement BufferProvider (`Core/BufferPool.swift:627-660`) but merely installing that pool into Operations is not an allocation integration. Mark the task-local as reserved/deprecate it or route specific documented scratch allocations through it; add a spy-provider call-count test if retained as active functionality.

`VectorCore.Configuration` exposes batchOperations and memoryAlignment, while `VectorCore.configuration` is `static let` and no source call site reads these settings. The actual BatchOperations mutable configuration is elsewhere. This public facade cannot tune operations as its "global configuration" wording implies. Consolidate configuration ownership instead of adding more parallel knobs. Execution agent owns the deeper ignored BatchOperations settings and concurrency defaults; do not duplicate those findings here.

`Array.findNearest` delegates to BatchOperations; `Operations.findNearest` uses its own validation/routing and honors BatchKernelProvider. VectorCollection search is a third independently implemented search loop. These separate routes create observable differences in overrides, metric support and error behavior. Keep deliberate low-level alternatives, but choose one canonical high-level implementation and make conveniences delegate to it. A matrix of each public entry point × installed provider × input shape × metric would prevent route drift.

ArraySIMDProvider is actively used by normalization/VectorMath/etc; SIMDProvider is a static pointer-level seam, and it is reasonable to keep these layers distinct. Do not report their mere coexistence as a defect. The problem is unclear effective scope: many kernels statically call concrete Swift/Accelerate implementations, and provider binding cannot imply package-wide backend substitution. API_Overview_Map now discloses this limitation.

### A3 — The advertised collection integration is not the interface VectorIndex actually implements

No `VectorCollection` conformance/use exists under the reviewed VectorIndex or VectorAccelerate Sources. VectorIndex's actual bridge is typed overloads accepting Core vectors and returning shared SearchResult types. Its indexes expose asynchronous throwing actor-oriented operations; Core's VectorCollection requires synchronous nonthrowing access/search and mutating CRUD. Comments saying VectorIndex "will implement" this are aspirational, not present architecture.

Either document VectorCollection as a small in-memory utility or design a consumer-driven async/throwing seam with coherent snapshots and explicit mutation guarantees. Avoid asking a production index to fit a protocol whose semantics already permit the removal-before-failure issue above. This does not justify moving ANN implementation into Core.

### A4 — Preserve ownership as a first-class contract

BufferHandle is a copied Sendable integer address plus ID/size; it does not retain its provider or serialize pointee access. VA's acquire docs explicitly require retaining the pool while the handle is active. Core's BufferProvider example `defer { await provider.release(handle) }` cannot compile because async work is not allowed in defer; provide an actual scoped helper supporting async body + cleanup, or a do/catch release example. This is a documentation/API ergonomics issue, not a demonstrated raw-pointer UAF under documented valid use.

The newer UnifiedVectorBuffer/SoA/PageAlignedBuffer docs correctly distinguish scoped borrow, persistent borrow and ownership transfer; they also correctly warn that retaining an owner does not prevent somebody else consuming its allocation. Frozen SoA layout + descriptor/tests is a strong actual cross-package interface. Make imported buffers' ownership enforceable where possible; do not mistake @unchecked Sendable for pointee synchronization.

### A5 — Public implementation surface creates maintenance obligations

API docs call mixed-precision caches/heuristics implementation details, but numerous concrete kernels and helper types are public (`MixedPrecisionKernels.SoAFP16`, `PlatformCapabilities`, conversion factories, etc.). Other agents assess their concrete pointer/numerical bugs. Mechanically inventory the emitted public symbol graph and classify supported low-level APIs versus accidental exposure before stabilization. Hiding unintended surface requires an explicit compatibility decision; prose alone cannot stop consumers compiling against public symbols. No package-level binary ABI stability is inferred.

## Normalization parity: verified real cross-package call impact

VA `Operations/StableNormalization.swift:56-62` explicitly documents two deliberate departures from Core: using the clamped denominator consistently for subnormal maxima and normalizing in the scaled domain when the true Float norm overflows. Its implementation computes scratch = v*scale, then divides scratch by scaledNorm (:103-123), rather than reconstructing/inverting the norm. Core NormalizeKernels still reconstructs with maxAbs and inverts the resulting magnitude (numerical agents own proof/reproduction).

These are active calls: VA `Operations/AccelerateFallback.swift:159`, `Operations/SIMDFallback.swift:260,265`, `Core/FallbackProvider.swift:195` invoke StableNormalization. Conversely VA `Integration/VectorCoreIntegration.swift:189` falls back to Core normalizedUnchecked when acceleration is unavailable. High-level accelerated normalization can consequently vary according to availability and path. This is more than a stale design comment, but no new Core numerical defect is independently counted here.

The unchecked API explicitly asserts nonzero magnitude in Debug (`VectorProtocol.swift:429`); the attempted zero `normalizedUncheckedWithHint` probe stopped on that assertion. Do not characterize zero input to this unchecked API as an additional valid-input failure. Errors from valid finite tiny/huge normalization and subsequent true hints should be handled with the numerical agents' evidence.

## Compatibility observations and downstream-owned issues

- VectorIndex manifest requires Core from0.3.2 and resolved pins0.3.2 at ad39ab6f4904838e300b836a42c7376d6e11da26. VA manifest requires from0.3.3 and resolved pins0.3.3 at fca4b602383589c46b627d8a0de2b6b2a68cd07d. Neither pinned release is this reviewed post-release main revision. Core Version.swift still identifies0.3.3. This is normal unreleased development, but a compatibility assertion must identify the exact revision; docs correctly disclose that memory-pool fixes are post-v0.3.3.
- Core/Index use Swift6.0 and macOS14+/iOS17+; current VA uses Swift6.2 and macOS26+/iOS26+/tvOS26+/visionOS26+, no watchOS. Shared protocol compatibility does not lower VA's deployment floor. VA's comments mentioning visionOS3/placeholder .v26 settings are stale relative to its actual manifest, a downstream documentation issue.
- VA MetalComputeProvider currently inherits Core's per-query BatchKernelProvider batch default (:352-353). Installing it genuinely reaches GPU-capable kernels, but does not currently imply one fused whole-query-batch dispatch. Already explicitly documented by Core; treat as a performance opportunity, not hidden failure.
- Source-derived downstream bug to track separately: VectorIndex `TypedOverloads.swift:107,183,205` derives the batch-wide queryIsNormalized flag from the first query only. Heterogeneous NormalizationHint batches are legal and can differ per element after mutation/auto-detection. A normalized first query causes later nonnormalized queries to skip normalization in Flat/IVF/optimized-flat. HNSW loops and checks each query independently at :160. This is owned by VectorIndex, not a Core fix, and was not executed here.

## Strengths, limitations and rejected suspicions

Strengths: real BatchKernelProvider delegation occurs before CPU routing; actual VA bridge preserves custom metric witness behavior instead of silently guessing; Swift scalar constraints and Sendable offer useful compile-time boundaries; explicit buffer ownership/layout docs and frozen SoA golden tests support concrete reuse; Core's current package-boundary docs carefully avoid claiming automatic GPU discovery, synchronized sibling releases or zero-copy end-to-end embeddings.

Not defects by themselves: manually passing false normalization assertions; unsafe pointer retention outside documented lifetime; lack of a full GPU implementation in CPU-only Core; separate static pointer and array provider layers; Core/index deployment floors differing from optional Metal4 consumers; documented matrix precision and external-provider ordering differences. SearchResult hashing omits score while equality includes it, which is legal (equal values still have equal hashes); do not report that as a Hashable bug. SearchResult Comparable has weak distance-only ordering across different IDs and NaNs, but this component prioritizes the concrete computedScore failure rather than turning tie-policy design into an additional unproven production bug.
