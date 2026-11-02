# Component 05 — nearest-neighbor search and Top-K

Revision reviewed: `8b02e75bca7055ae11170873d15c06f1538709b7`.

## Scope and evidence

Read all five assigned files: `Operations/{Operations,TopKSelection,MinHeap}.swift`, `Operations/Kernels/TopKSelectionKernels.swift`, `Protocols/DistanceMetric.swift`. Traced `CPUComputeProvider.parallelReduce`, `BatchKernelProvider` delegation, `ExecutionOperations` legacy heap callers, built-in distance definitions, matrix packing/preparation and `SearchResults(from:)`. Read current package-boundary/API/numerical docs and complete TopKSelectionTests, TopKTieBreakingTests, TopKNaNContractTests, TopKNaNWrapperTests, BatchKNNGEMMTests, BatchGEMMRoutingTests, BatchKernelProviderTests, OperationsValidationTests, OperationsBatchTests and ExecutionOperationsTests.

No repository edits, tests or shared builds. Standalone public-API probe linked stable Debug object files using the orchestrator's `build_probe.py`:

- `/private/tmp/vectorcore-review-2026-09-14/search_probe.swift`
- `/private/tmp/vectorcore-review-2026-09-14/search_probe.log`

Probe compiled and ran successfully. The findings below distinguish executed reproductions from source deductions. No performance measurements were taken.

## Verified actionable defects

### S1 — P1 correctness/API: `TopKSelection.nearest` silently ignores three built-in metrics and every custom metric

**Location:** `Sources/VectorCore/Operations/TopKSelection.swift:325–327`; signature at 310–315; batch propagation at 527–536.

The public method says “using any distance metric” and accepts an unconstrained `any DistanceMetric`. It recognizes only Euclidean, cosine, Manhattan and dot product; every other metric is replaced with a fresh `EuclideanDistance`. This includes Core's own `ChebyshevDistance`, `HammingDistance` and configurable `MinkowskiDistance`, in addition to downstream metrics. Both scores and neighbor membership can silently be wrong. `batchNearest` inherits the same defect.

**Executed reproduction:** query `[0,0]`; candidates `[[2.5,0],[2,2]]`; `ChebyshevDistance`; `k=1`. Direct metric scores are `[2.5,2]`, so index 1 is nearest. `TopKSelection.nearest` returns index 0. `Operations.findNearest` returns correct index 1 on the same inputs. This is not floating-point sensitivity; Euclidean intentionally orders these candidates differently.

**Remedy:** invoke the actual metric through a Float-constrained generic/existential (`M: DistanceMetric, M.Scalar == Float` or `any DistanceMetric<Float>` as appropriate to source compatibility). Do not solve this by extending the concrete allowlist: that would leave custom conformers and future built-ins broken. Audit both nearest and batchNearest signatures. Existing tests cover default Euclidean and cosine, so add a mechanically shared test over all built-ins plus a custom metric, comparing returned scores/membership to direct calls and precomputed Top-K. Include a nondefault Minkowski parameter. **Confidence: high, public reproduction.**

### S2 — P2 correctness: bounded heap array construction selects the wrong elements

**Location:** `Sources/VectorCore/Operations/MinHeap.swift:57–66`.

`MinHeap(from:capacity:compare:)` heapifies the entire array and discards the final leaves until capacity is met. A heap only orders each parent relative to its children; its prefix does not contain the best k members. It also preserves the extremal root that bounded streaming `insert` would evict.

**Executed reproduction:** `MinHeap(from: [1,2,3,4], capacity: 2, compare: >).sorted()` returns `[4,2]`. Constructing the same bounded heap empty and inserting the same values returns `[2,1]`, the correct two smallest. A bulk-initialized candidate heap therefore depends on construction style and can discard a true nearest candidate before later search begins. The analogous issue applies in the opposite comparator direction.

**Remedy:** initialize by bounded insertion or implement a selection algorithm before heapification; explicitly document that bounded capacity retains the side opposite the root's heap order. The adjacent insert comment at lines 80–81 says “comes before” although the branch retains elements for which `compare(root, element)` is true; fix that misleading explanation too. Add constructor-versus-streaming membership tests across comparator directions, permutations, ties and capacities 0/1/n/>n. No direct MinHeap tests were found in Tests. **Confidence: high, public reproduction.**

### S3 — P2 robustness: decoding bypasses `TopKResult`'s parallel-array invariant

**Location:** `Sources/VectorCore/Operations/TopKSelection.swift:651–654`, with subscript at 45–46 and direct-init precondition at 31–34.

The decoder accepts indices and distances independently without checking equal counts. The ordinary initializer enforces equal counts, and Collection iteration assumes it. A malformed serialized result can therefore be successfully decoded into an invalid public value.

**Executed reproduction:** JSON `{"indices":[0],"distances":[]}` decodes without error; `count == 1`, `distances.count == 0`, and `toTuples().count == 0`. Accessing the advertised valid collection position `result[0]` necessarily traps at the empty distances subscript (source derivation, crash not run). `SearchResults(from:)` at `Integration/SearchResult.swift:168` instead silently truncates through zip.

**Remedy:** decode into locals, check equal counts, and throw `DecodingError.dataCorrupted` before constructing the value. Add malformed shorter/longer arrays and valid empty cases; current `testTopKResult_Codable` only round-trips a valid object. Coordinate with the serialization review for the same invariant class in other types. **Confidence: high, public decode reproduction plus direct subscript derivation.**

### S4 — P2 robustness, source-proven: negative bounded array capacity never terminates

**Location:** `Sources/VectorCore/Operations/MinHeap.swift:63–65`; empty `removeLast` guard at 121.

`MinHeap<Int>(from: [], capacity: -1, compare: <)` enters `while count > capacity` with `0 > -1`; `removeLast` is a no-op on empty storage, so neither loop condition nor storage changes. Nonempty inputs eventually reach the same state. There is no precondition at this public initializer documenting/rejecting negative capacity. The other initializer happens to pass it into Array.reserveCapacity, yielding different invalid-input behavior.

**Remedy:** establish one explicit nonnegative-capacity precondition shared by both constructors (or a throwing validating API). Keep zero capacity valid if intended. Verify negative cases in bounded subprocesses so a regression cannot hang the suite. **Confidence: high direct derivation; no hang subprocess run.** This is lower priority than valid-input S1/S2/S3 and can be grouped with S2 as bounded-heap contract repair.

## Existing documented limitations and scale opportunities

### L1 — P2 architecture/performance opportunity: automatic batch search needs a bounded-memory and precision policy

**Location:** `Operations.swift:209–232`, matrix call 234–236.

The Q>=8, N>=256 optimized Euclidean/cosine route unconditionally materializes all Q*N Float scores, regardless of k, budget, or the selected CPU scheduler. It also repacks the candidate side on every call. `BatchOperations.enableMatrixRouting` does not govern this separate gate, as current docs explicitly explain. `CPUComputeProvider(mode: .sequential)` does not bypass it either. Thus the public convenience search API has no direct way to retain direct-distance behavior or bound scratch memory; callers must loop `findNearest`, use batches below eight, supply their own BatchKernelProvider, or assemble lower-level matrix/Top-K calls.

Source-derived scale: Q=10,000 and N=1,000,000 requires 40,000,000,000 bytes of score output alone, while k=10 only asks for 100,000 final neighbors. N=1,000,000 at d=512 also packs about 2.048 GB of candidate Float storage per matrix call, in addition to the existing candidates. No allocation benchmark or OOM run was performed. Documentation already discloses the full matrix, so this is an architectural scale limitation, not a newly discovered hidden allocation defect.

**Executed precision example of the documented limitation:** optimized 512 query has every coordinate 1000; candidate 0 is the same except its first coordinate is 1000.125; candidate 1 is exactly the query; remaining 254 candidates repeat candidate 0. One query finds exact index 1 at distance zero; eight repeated queries trigger GEMM and choose index 0 at distance zero. The same disagreement reproduced with offsets 0.25, 0.5 and 1.0. This illustrates an exact self-neighbor being lost rather than merely a tolerable score difference. Current numerical docs already say rankings can differ from cancellation; this should not be filed as a new cancellation discovery.

**Recommendation:** expose per-call strategy and scratch-memory budget, tile queries/candidates, stream tiles into global-index Top-K buffers, and permit prepared candidates for repeated search. For exact-neighbor workloads, allow direct differences or robust reranking of an adequately broad candidate set. Tiling bounds memory but does not itself fix GEMM cancellation. Record this deterministic fixture and test any precision mode rather than relying solely on random near-unit vectors. Benchmark proposed crossovers and memory behavior in Release before claiming performance improvements.

### L2 — P2 performance opportunity: array and generic small-k paths retain O(n) temporary score pairs

**Location:** `TopKSelection.swift:151–157`, generic select 284, nearest 343, `Operations.swift:127–137`.

Array select always materializes an `(Int, Float)` tuple for every candidate before choosing its heap path. On this 64-bit target each tuple is expected to stride 16 bytes, so the pair list alone is about 16 MB per million candidates, despite k=10. `nearestWithMetric` first adds a separate 4*n Float score array. Generic `Operations.findNearest` follows the same compute-all-then-select pattern. The pointer small-k overload already streams directly into O(k) heap storage. Correctness is unaffected; memory scaling is avoidable and especially relevant to generic/custom-metric searches.

**Recommendation:** stream array enumeration or metric evaluation straight into TopKBuffer for small k, retaining the common comparator. Keep full sort for the chosen large-k crossover. Compare allocation counts/peak memory as well as Release latency; the claimed large-k “better constants” at lines 581 etc. is an unmeasured claim in this review. Prefer an Int-indexed internal streaming primitive so reuse of the public pointer overload does not unintentionally impose its Int32 count limit on arrays.

### L3 — P3 performance opportunity: redundant validation and oversized partial heaps

Below GEMM, `findNearestBatch` validates all dimensions once then calls public `findNearest`, which rescans all candidate dimensions for every query. This adds Q*N dimension checks and traversals. A validated internal worker could remove repeated checks without weakening public validation. This is source-derived overhead, not a measured regression.

Optimized `findNearest` takes its fused heap path for every `k < n`, including k close to n. Each parallel chunk allocates a full k-entry vals/idxs pair even if it only sees roughly n/cores entries. Merge allocates a fresh output buffer, and `mergeBuffers` first allocates an empty output that `mergeTopK` immediately replaces. With C workers, peak partial heap storage is O(C*k), so near-full selection can consume several times the useful result storage. Tune large-k route and local capacities carefully; changing capacities also requires revisiting merge preconditions and the information retained by partial heaps. No claim of measured speedup.

## Legacy public heap consistency (secondary, not a reopened Top-K fix)

`KNearestHeap` remains public and uses bare `distance > distance` at `MinHeap.swift:215`. An executed `k=1` probe inserts `(0, NaN)` then `(1, 0)` and retains the NaN. It has no documented NaN-last/tie policy, so the recently added `TopKSelection` contract should not be asserted to cover it. Its only Core production caller is the internal `ExecutionOperations` path; public `Operations` and `TopKSelection` use the fixed TopKBuffer. Consider deprecating/consolidating this alternate public selector or documenting and testing its finite-input limits. Do not report this as a failure of TopKNaNContractTests or as an unfixed `Operations` regression.

## Strengths and rejected suspicions

- Shared `orderedBefore` correctly puts numeric scores (including infinities) ahead of NaNs in both directions. Exact signed-zero/two-NaN ties use original indices; `.smallerValue` is strict weak ordering. Current tests exercise comparator transitivity, literal expected membership, both directions, both crossover sides, mapped IDs and score bits.
- Heap root uses the inverse of best-first ordering, so admission and extraction agree. Index-based selection of disjoint top-k partials is order-independent: any candidate excluded from a partial already has k globally better candidates in that partial. The actual merge tests vary chunk and merge order. No new NaN/tie defect found in these public selectors.
- Public pointer select guards k/count before allocating, clamps k, checks Int32 representability, and only reads mapped IDs after choosing positions. Its count restriction is explicit in code and reasonable for Int32 output; caller buffer validity remains an unsafe-pointer precondition.
- Optimized wrappers are reached only when 0<k<n, preventing empty TopKBuffer root access. Public TopKSelection optimized methods guard/clamp similarly. Internal zero-capacity TopKBuffer insertion would trap, but no violating public route was found, so not filed.
- Squared-Euclidean membership may differ from sorting rounded square roots; this is explicitly documented, and both result extractors sort transformed values. Do not misreport as tie-order failure.
- Operations validates every candidate/query dimension before provider delegation or unsafe optimized paths. Custom metric handling in Operations retains the provided instance. Provider result validation remains the documented conformer's contract; sentinel tests intentionally verify passthrough.
- External providers precede CPU optimizations in actual source. However `BatchKernelProviderTests.findNearestBatchDelegates` uses only five queries, and the override test only four; neither crosses Q>=8. Their “precedence over GEMM” claim is not mechanically exercised. Raise those fixtures to >=8 to close the route test gap (P3 coverage).
- `DistanceMetric.batchDistance` is a protocol requirement, allowing witness dispatch. Neither nearest implementation calls it. `BatchKernelProvider.batchDistance` is separately documented for direct consumers; lack of routing to that method is not an API bug.

## Limitations

Debug standalone probes establish concrete behavior for this checkout/hardware, not general performance or cross-hardware GEMM bit patterns. Shared full Debug/Release suite validation belongs to the orchestrator. Direct matrix kernels, provider concurrency policy, scalar metric numerical edges, and aggregation behavior have separate component owners. No randomized property suite or throughput benchmark was added during this read-only review.
