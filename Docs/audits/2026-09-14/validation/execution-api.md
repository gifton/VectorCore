> Independent reviewer working notes. The [final claim register](../CLAIMS.md) controls priorities and dispositions. Historical temporary paths in replay commands are retained for provenance; portable replay guidance and preserved artifacts are in [the evidence index](../evidence/README.md).

# Execution, search, batch and downstream claim validation

Reviewed Core revision `8b02e75bca7055ae11170873d15c06f1538709b7`. Review only. Fresh Release probes and exact-source pool probe; commands and logs under `validate_execution_api/`. No downstream/GPU/TSAN run. Diagnoses are verified to the extent stated; improvements are unimplemented design proposals.

## Ledger

### API-01 — Two search facades replace unsupported metric witnesses with Euclidean

**confirmed-reproduced · P1 · Public TopKSelection.nearest/batchNearest and default collection search; Operations respects requested metric**

Original: A4, search_topk:S1, downstream_interfaces:IF1.

Sources: Sources/VectorCore/Operations/TopKSelection.swift:310; Sources/VectorCore/Integration/VectorCollection.swift:120.

**Evidence:** Fresh Release search_probe: direct Chebyshev [2.5,2], TopK index0, Operations index1. interfaces_probe proves wrong collection score 2.828427 instead of2. Inputs have equal shape, k1, finite ordinary values. Source default branch applies to Chebyshev/Hamming/Minkowski/custom witnesses.

**Qualification:** Collection probe itself shows wrong score, not rank reversal; source proves reversal for search_probe candidate set. Not a downstream VectorIndex implementation bug.

**Improvement:** Use actual Float metric witness; changing signatures requires compatibility decision. Expanding allowlist is incomplete.

**Acceptance:** Shared facade table for all built-ins, nondefault Minkowski, and custom reverse-order metric; score and member equality versus direct witness. Preserve provider route.

### API-02 — Batch configuration storage is disconnected from operation settings

**confirmed-reproduced · P2 · Public BatchOperations; matrix flags alone use live settings**

Original: A5, execution_concurrency:EC-3.

Sources: Sources/VectorCore/Operations/BatchOperations.swift:99; Sources/VectorCore/Operations/BatchOperations.swift:175; Sources/VectorCore/Operations/BatchOperations.swift:618.

**Evidence:** Fresh Release ExecutionPublicProbe stores defaultBatchSize2, processes5, observes [5]. Consumers read immutable defaultConfig for batch/parallel/chunk fields; pairwise routing reads live config.

**Qualification:** findNearest uses ParallelHeuristic, not configurable general parallelThreshold. Do not imply every knob governs every API.

**Improvement:** Define effective setting scope and take one validated snapshot at operation entry. Inspect newly activated arithmetic first.

**Acceptance:** Behavior tests per setting; reject zero/nonfinite/negative/huge oversubscription, invalid batch/chunk sizes before division, conversion, shift or addition.

### API-03 — Configuration read-modify-write loses concurrent updates

**confirmed-reproduced · P2 · Public updateConfiguration**

Original: A5, execution_concurrency:EC-4.

Sources: Sources/VectorCore/Operations/BatchOperations.swift:99; Sources/VectorCore/Configuration/ThreadSafeConfiguration.swift:9.

**Evidence:** Fresh Release 100 structured increments result29. Two separate actor awaits permit stale-copy overwrite, independent of exact scheduling.

**Qualification:** Actor individual access remains isolated; failure is transaction lost update, not unsynchronized stored memory.

**Improvement:** Run synchronous mutation closure inside one actor method; design Sendable boundary.

**Acceptance:** 100 concurrent increments and independent-field updates retain all changes; Swift6 consumer typecheck.

### API-04 — Several schedulers eagerly create all child tasks

**confirmed-reproduced · P2 · ComputeProvider defaults; public BatchOperations.process/map/filter; public Dim768 cosineSimilarityMatrix and TriangularMatrix.buildParallel; downstream VA inherited batch defaults**

Original: A6, execution_concurrency:EC-5, batch_matrix:opportunity5.

Sources: Sources/VectorCore/Protocols/ComputeProvider.swift:128; Sources/VectorCore/Operations/BatchOperations.swift:183; Sources/VectorCore/Operations/BatchOperations.swift:298; Sources/VectorCore/Operations/BatchOperations.swift:328; Sources/VectorCore/Operations/ExecutionOperations.swift:401; Sources/VectorCore/Operations/TransformOperations.swift:164; Sources/VectorCore/Utilities/TriangularMatrix.swift:250; Sources/VectorCore/Protocols/BatchKernelProvider.swift:50; ../future/VectorAccelerate/Sources/VectorAccelerate/Integration/MetalComputeProvider.swift:91.

**Evidence:** Fresh Release default conformer maxConcurrency1 reaches64 suspended active bodies. Add-all loops source-proven; Dim768 convenience reaches internal matrix above threshold. 10000*9999/2 =49995000 pair tasks, 4*10000^2=400000000 scalar-output bytes.

**Qualification:** Task count/resource exposure, not 64 simultaneous CPU threads or measured slowdown. CPUComputeProvider own bounded chunks/nesting fix is intact. Default reduction is coarsely chunked, not per-item.

**Improvement:** Bound workers/window and define nesting/cancellation/order. All public routes need explicit scheduler policy; actor reentrancy does not bound outstanding GPU work.

**Acceptance:** Instrument active bodies and enqueued tasks for default conformers, public batches, matrix builders, nested routes; verify bound and exact ordered coverage.

### API-05 — Cached raw actor pool allocations are not freed on owner destruction

**confirmed-reproduced · P2 · Public scoped SwiftBufferPool and TypedBufferPool; internal BufferPool same pattern**

Original: A7, execution_concurrency:EC-1.

Sources: Sources/VectorCore/Providers/SwiftBufferPool.swift:48; Sources/VectorCore/Operations/BufferPool.swift:174.

**Evidence:** Fresh optimized exact-source counted-deallocator probe: destroyed weak pool true; cachedBuffersFreed0. SwiftBufferPool stores raw addresses, empty deinit. TypedBufferPool/internal BufferPool similarly lack deinit; source only for latter two.

**Qualification:** Proof covers returned cached allocation without active handles. It does not establish active-handle teardown policy, generic element ownership, or leak in separately repaired MemoryPool.

**Improvement:** Introduce clear ownership and cached-allocation teardown for each pool; explicitly define active-handle lifetime policy.

**Acceptance:** Count allocate/free for empty, returned buffers, clear then destruction, destruction without clear for each pool; initialized generic-element lifetime tested separately.

### API-06 — SwiftBufferPool age cleanup does not enforce maxPoolSize

**confirmed-reproduced · P2 · Public SwiftBufferPool opt-in; Operations buffer binding currently unused internally**

Original: A7, execution_concurrency:EC-2.

Sources: Sources/VectorCore/Providers/SwiftBufferPool.swift:100; Sources/VectorCore/Providers/SwiftBufferPool.swift:164.

**Evidence:** Fresh optimized source probe: eight1024-byte returns retained/reused under1024-byte cap, zero frees before clear. Cleanup only evicts older-than60s entries; no idle timer. currentUsageBytes tracks active rounded bytes only.

**Qualification:** No universal RSS/time claim; indefinite idle retention is source-derived. maxBufferSize remains enforced on return.

**Improvement:** Track cached bytes and enforce budget independently of age; report active/cached separately.

**Acceptance:** Zero/exact/mixed bucket budgets; concurrent burst; no release leaves cache above cap. Verify live-handle and returned-memory accounting.

### API-07 — Nonpositive process batch sizes trap or discard work

**confirmed-reproduced · P2 · Public throwing BatchOperations.process explicit parameter**

Original: C15, execution_concurrency:EC-6, batch_matrix:6.

Sources: Sources/VectorCore/Operations/BatchOperations.swift:175; Sources/VectorCore/Operations/BatchOperations.swift:535.

**Evidence:** Fresh Release batchSize0 subprocess SIGTRAP; -1 returns[] without transform for one nonempty vector. No stated positivity precondition.

**Qualification:** Int positivity is a reasonable semantic expectation, but throwing API currently has neither clear contract nor validation; rank below ordinary positive-input bugs.

**Improvement:** Validate positive batch sizes before either route; clarify empty-input policy.

**Acceptance:** 0,-1,1,>count on N0,1,999,1000; exact-once ordered coverage when valid. Handle oversized positive endpoint addition too.

### API-08 — CPU chunk endpoints overflow before clamping valid short ranges

**confirmed-reproduced · P2 · Public explicitly parallel CPU helpers; default automatic only when gate reached; huge explicit minChunk option**

Original: C15, execution_concurrency:EC-7.

Sources: Sources/VectorCore/Providers/CPUComputeProvider.swift:128; Sources/VectorCore/Providers/CPUComputeProvider.swift:193; Sources/VectorCore/Providers/CPUComputeProvider.swift:234; Sources/VectorCore/Providers/CPUComputeProvider.swift:276; Sources/VectorCore/Providers/CPUComputeProvider.swift:317; Sources/VectorCore/Protocols/ComputeProvider.swift:178.

**Evidence:** ExtraProbe fresh Release all five async CPU helper variants SIGTRAP for (Int.max-2)..<Int.max; sequential control returns both indices. minChunk:Int.max with range1..<3 also SIGTRAP. Protocol fallback has overflow-prone hw*2/ceiling/end arithmetic, source only.

**Qualification:** No enormous allocation is required for CPU witnesses. Do not claim tiny near-max failure for default reduction without a case that actually overflows its smaller chunk.

**Improvement:** Use remaining-length partitioning; check arithmetic and config conversions as a shared class.

**Acceptance:** All five methods, zero/nonzero/near-max lower bounds, huge hints, exact index coverage/order; test protocol defaults separately. Empty and nonrepresentable Range.count cases need a defined policy.

### API-09 — Failed collection update deletes existing value

**confirmed-reproduced · P2 · Public MutableVectorCollection default; SimpleVectorCollection reference implementation**

Original: A8, downstream_interfaces:IF2.

Sources: Sources/VectorCore/Integration/VectorCollection.swift:239.

**Evidence:** Fresh Release dimension2 id7 updated with dimension1 throws then vector(for:7)==nil. Add rejection is routine documented failure.

**Qualification:** SimpleVectorCollection is a small/reference utility, not actual VectorIndex storage. Prechecking dimension alone does not guarantee rollback for arbitrary add failures.

**Improvement:** Require atomic replacement in implementations; validate and replace under SimpleVectorCollection lock.

**Acceptance:** Failure injection for all thrown replacement paths preserves old data; concurrent update/search snapshot policy defined.

### API-10 — Explicit/generated IDs overwrite records

**confirmed-reproduced · P2 · SimpleVectorCollection public insertion APIs**

Original: A8, downstream_interfaces:IF3.

Sources: Sources/VectorCore/Integration/VectorCollection.swift:213; Sources/VectorCore/Integration/VectorCollection.swift:286; Sources/VectorCore/Integration/VectorCollection.swift:298.

**Evidence:** Fresh Release explicit0 then generated add returns0,count1,replacement data. Dictionary overwrite also violates explicit duplicate-ID throws contract.

**Qualification:** Reference collection scope; integer exhaustion is design edge, not replayed production failure.

**Improvement:** Reject explicit duplicates and find unused generated IDs atomically; define exhaustion.

**Acceptance:** Explicit/generated permutations, sparse IDs, repeated explicit insertion, remove/reinsert, exhausted allocator strategy. Update must use replacement primitive rather than duplicate rejection.

### API-11 — computedScore pole breaks negative-dot relevance

**confirmed-reproduced · P2 · Public SearchResult fallback; actual collection searchDotProduct producer**

Original: A9, downstream_interfaces:IF4.

Sources: Sources/VectorCore/Integration/SearchResult.swift:34; Sources/VectorCore/Integration/VectorCollection.swift:195.

**Evidence:** ExtraProbe actual search query[1], candidates[.5],[1],[2] outputs distances[-2,-1,-.5], computedScores[-1,inf,2]. Finiteness/range/order promises fail with valid equal-shape ordinary vectors.

**Qualification:** No reviewed downstream computedScore callers; do not claim observed ANN/GPU regression. There is no universal metric-free transform for all Float scores.

**Improvement:** Carry metric/score semantics or require explicit transform for unsupported distance families.

**Acceptance:** Search-to-score monotonicity, finiteness, documented range across negative-dot pole and nonnegative families; preserve explicitly supplied score.

### API-12 — Advertised buffer/global configuration hooks have no active internal consumers

**confirmed-source · P3 · Public reserved-looking bindings and immutable facade**

Original: A10, downstream_interfaces:architecture A2.

Sources: Sources/VectorCore/Operations/Operations.swift:42; Sources/VectorCore/VectorCore.swift:53.

**Evidence:** rg over Sources finds bufferProvider only declaration; global VectorCore.configuration fields have no operation readers. Actual BatchOperations config is separate.

**Qualification:** Provider coexistence is not itself a defect. ArraySIMDProvider and pointer SIMDProvider have different legitimate scopes; API map already discloses static-kernel limitations. Extension-only vector dispatch is owned by vector reviewer.

**Improvement:** Document actual override map; deprecate/reserve unused hooks or connect named allocations with compatibility plan.

**Acceptance:** Spy provider verifies each claimed active hook; public-entry/provider matrix documents real dispatch.

### API-13 — Collection aggregation trusts first row before unsafe reads/forced factories

**confirmed-source · P1 · Public Operations.centroid/statistics; generic factory boundary**

Original: C2, batch_matrix:1.

Sources: Sources/VectorCore/Operations/Operations.swift:780; Sources/VectorCore/Operations/Operations.swift:794; Sources/VectorCore/Operations/Operations.swift:811; Sources/VectorCore/Operations/Operations.swift:854.

**Evidence:** Independently inspected pointer count=first dimension passed for each shorter toArray allocation; long rows truncated; empty fixed centroid try! create([]); statistics values[i] trusts first length. The root preserves first-pass aggregation evidence; this second-pass record is source-derived.

**Qualification:** No equal-dimension precondition appears at these aggregation entries. Do not transfer MatrixDistance unsafe contracts to them. Pointer overread is source proof; garbage returned by an unsanitized probe is not proof of exact read extent.

**Improvement:** Define empty policy, validate every row, and propagate factory errors. Include Operations.normalize try! generic factory in boundary inventory.

**Acceptance:** Short/long row at first/middle/last, empty dynamic/fixed arrays, failing custom factory; no unsafe operation before all shape checks.

### API-14 — Zero-query SoA cosine fills oversized output suffix

**confirmed-reproduced · P2 · Public Safe512 and dimension wrappers using shared internal routes**

Original: C10, batch_matrix:3.

Sources: Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift:930; Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift:1025; Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift:1126; Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift:1737.

**Evidence:** Fresh Release Safe512 N1,4,4097 output N+2 sentinel -77 becomes1 at both suffix positions. N4097 crosses >8MiB tile gate. Safe wrapper allows capacity>=N. Same generic implementation powers sibling widths.

**Qualification:** Writes remain inside provided capacity; this is logical-output overwrite, not out-of-allocation memory corruption. Nonzero branches write N only; no documented requirement to overwrite scratch suffix.

**Improvement:** Restrict zero fill to logical N.

**Acceptance:** Dimension512/768/1536, N0/1/3/4/5, tile255/256/257 and actual byte thresholds; oversized/offset outputs and zero/nonzero query sentinels.

### API-15 — Pairwise hard-coded zero diagonal fails built-in negative dot

**confirmed-reproduced · P2 · Public BatchOperations.pairwiseDistances direct routes**

Original: C11, batch_matrix:7.

Sources: Sources/VectorCore/Operations/BatchOperations.swift:550; Sources/VectorCore/Operations/BatchOperations.swift:577.

**Evidence:** Fresh Release singleton DynamicVector[2], DotProductDistance yields [[0]], direct scalar metric=-4. Source excludes all i==j; DotProductDistance is a supported built-in despite not a mathematical distance metric.

**Qualification:** Directed metric N2/N100 discrepancy reproduced but pairwise docs promise symmetric output: treat arbitrary asymmetric support as contract clarification, not required fix. VectorMath pairwise explicitly documents zero diagonal.

**Improvement:** Compute supplied metric diagonals or restrict shortcut to explicit zero-self contract; define symmetry requirements.

**Acceptance:** All built-ins direct-pair oracle around N99/100 and GEMM gates; include zero vectors and dot-product diagonal. Do not assume every cosine route treats zero self as0.

### API-16 — cosineSimilarityMatrix returns negative off-diagonal similarity and zero diagonal

**confirmed-reproduced · P2 · Public Array<Vector<Dim768>> convenience**

Original: C11, batch_matrix:7.

Sources: Sources/VectorCore/Operations/TransformOperations.swift:164; Sources/VectorCore/Operations/ExecutionOperations.swift:388.

**Evidence:** ExtraProbe two identical nonzero Dim768 vectors returns [[0,-1.0000001],[-1.0000001,0]], expected approximately1 throughout. Normalizes then passes negative dot to zero-diagonal matrix.

**Qualification:** Separate from directed-metric contract. Exact rounded -1.0000001 is numeric noise; sign/diagonal error is order-one.

**Improvement:** Compute cosine similarity with correct sign and self policy; unify matrix contract.

**Acceptance:** Identical, orthogonal, opposite nonzero vectors and defined zero behavior through serial/parallel matrix gates.

### API-17 — Bulk bounded heap membership differs from streaming; negative capacity loops

**confirmed-reproduced · P2 · Public MinHeap opt-in; no evidence high-level Operations uses bulk constructor**

Original: C12, search_topk:S2, search_topk:S4.

Sources: Sources/VectorCore/Operations/MinHeap.swift:57; Sources/VectorCore/Operations/MinHeap.swift:80; Sources/VectorCore/Operations/MinHeap.swift:121.

**Evidence:** Fresh Release > comparator bulk [4,2],stream[2,1]; opposite < comparator bulk[1,2],stream[3,4]. Negative empty capacity hangs for3s; 0>-1 and empty removeLast no-op prove nontermination.

**Qualification:** Implicit nonnegative capacity is reasonable but missing validation; separate P3 invalid-parameter subcase from valid bounded-selection P2. Parent/root ordering does not make heap prefix correct top-k.

**Improvement:** Select retained members before heapify or bounded insert; validate capacity; correct misleading insertion comment.

**Acceptance:** Bulk/stream membership for permutations, comparators, ties, capacity0/1/n/>n; bounded subprocess negative capacity rejection.

### API-18 — TopKResult decoder violates parallel-array invariant

**confirmed-reproduced · P2 · Public Codable from malformed external data**

Original: C12, search_topk:S3.

Sources: Sources/VectorCore/Operations/TopKSelection.swift:31; Sources/VectorCore/Operations/TopKSelection.swift:45; Sources/VectorCore/Operations/TopKSelection.swift:651; Sources/VectorCore/Integration/SearchResult.swift:168.

**Evidence:** Fresh Release JSON indices[0],distances[] decodes,count1,tuples0. Subsequent index0 traps by direct empty-array subscript proof; trap not executed.

**Qualification:** Malformed serialized input, not valid round-trip bug. Silent zip truncation is consequence of same invariant violation.

**Improvement:** Validate equal array counts before construction and throw DecodingError.

**Acceptance:** Both mismatch directions, empty valid data, normal round-trip and collection access; class-wide serialized invariant audit.

### API-19 — Factory batch dimension zero reaches remainder by zero

**confirmed-reproduced · P3 · Public throwing VectorTypeFactory.batch invalid dimension**

Original: C15, vectors:V6.

Sources: Sources/VectorCore/Factory/VectorTypeFactory.swift:260.

**Evidence:** ExtraProbe empty array,dimension0 SIGTRAP; first executable operation computes values.count%dimension. Public doc specifies multiple of dimension but no positive guard.

**Qualification:** A vector dimension normally implies positivity; classify invalid-parameter validation gap rather than ordinary valid-input failure. Not all factory methods trap at zero.

**Improvement:** Check positive dimension and document zero/negative policy before remainder/division.

**Acceptance:** Zero/negative dimension with empty/nonempty values, valid multiples and mismatches; ensure throws instead of trap where documented.

### API-20 — PCA oversampling Int.max overflows before bounded sketch clamp

**confirmed-reproduced · P3 · Public optional extreme PCAConfig; two valid one-dimensional samples**

Original: C15, linear_algebra:LA3.

Sources: Sources/VectorCore/LinearAlgebra/PCA.swift:144; Sources/VectorCore/LinearAlgebra/PCA.swift:179.

**Evidence:** ExtraProbe two1D rows,components1,oversamplingInt.max SIGTRAP at k+p. Current validation only rejects negative configuration; intended min limit is representable1.

**Qualification:** Extreme optional knob, not default PCA numeric failure or enormous input allocation.

**Improvement:** Clamp oversampling against remaining allowed sketch width before addition; define invalid config behavior.

**Acceptance:** 0,normal,Int.max oversampling on tiny data; correct exact fallback or documented error without overflow.

### API-21 — FP16 diagnostics enabled flag is read without lock

**confirmed-source · P2 · Concurrent diagnostics enable/disable with public mixed-precision conversions; opt-in recording**

Original: C18, reduced_precision:RP4, execution_concurrency:diagnostics.

Sources: Sources/VectorCore/Operations/Kernels/MixedPrecisionDiagnostics.swift:62; Sources/VectorCore/Operations/Kernels/MixedPrecisionDiagnostics.swift:103; Sources/VectorCore/Operations/Kernels/MixedPrecisionDiagnostics.swift:146; Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:820.

**Evidence:** Writers lock; recordBatchConversion guard reads isEnabled before lock; public FP16 conversion inits call recordConversionsIfEnabled, whose locked preliminary check can be followed by concurrent disable before unlocked guard. recordConversion has same pattern but no current callers found.

**Qualification:** Source-proven conflicting access, no TSAN report. Recorders normally unreachable while always disabled; default entire library not racy from this flag.

**Improvement:** Check flag and update under one lock or correctly specified atomic policy.

**Acceptance:** TSAN concurrent enable/disable plus all live conversion inits; stable state/stat count policy. Eliminate both duplicated unlocked guards.

### API-22 — Automatic batch GEMM requires full Q*N scores and can change near-neighbor ranking

**qualified · P2 opportunity · Operations optimized Euclidean/cosine Q>=8,N>=256 unless BatchKernelProvider installed**

Original: PERF2, search_topk:L1.

Sources: Sources/VectorCore/Operations/Operations.swift:209; Sources/VectorCore/Operations/Operations.swift:232; Docs/Numerical_Behavior.md:69.

**Evidence:** Fresh Release offset1000 plus delta .125/.25/.5/1: one-query exact self index1; eight-query gate index0 tie at0. Full output=4QN bytes: Q10000,N1000000 =>40000000000 bytes; candidate512 packing=2048000000 bytes.

**Qualification:** Already explicitly documented full allocation and cancellation/ranking limitation. Not new hidden bug or universally observed across hardware. Sequential provider and BatchOperations matrix setting do not bypass separate gate.

**Improvement:** Expose per-call strategy/scratch budget, tile or prepared views; robust reranking/direct precision option when needed. Tiling alone cannot restore precision; small rerank set may exclude true winner.

**Acceptance:** Memory budget invariants and deterministic exact-neighbor fixture; direct precision mode reproduces direct reference across gate; benchmark crossovers only after design.

### API-23 — Small-k array/generic selection retains O(N) temporary distances/pairs

**confirmed-source · P3 opportunity · Public TopKSelection arrays/nearest, generic Operations and BatchOperations; pointer small-k already streams**

Original: PERF5, search_topk:L2, batch_matrix:opportunity3.

Sources: Sources/VectorCore/Operations/TopKSelection.swift:151; Sources/VectorCore/Operations/TopKSelection.swift:284; Sources/VectorCore/Operations/TopKSelection.swift:343; Sources/VectorCore/Operations/Operations.swift:127; Sources/VectorCore/Operations/BatchOperations.swift:416; Sources/VectorCore/Operations/BatchOperations.swift:773.

**Evidence:** Source materializes complete distances and tuples before selection; ExtraProbe confirms tuple stride16 on target =>16MB per million pair entries plus4MB scores. Batch optimized helpers use candidates.chunked arrays for boundaries then index original candidates.

**Qualification:** No measured speedup/RSS; tuple metadata not full copied vector payload in every path. Avoid Int32 count restriction when adopting pointer implementation.

**Improvement:** Stream metric/array values into common Int-indexed top-k; use integer ranges instead of copied chunks.

**Acceptance:** Allocation/peak memory versus N at fixed k; exact membership/order/NaN/ties match shared selector; Release measurement before performance claims.

### API-24 — Redundant matrix reshape, validation and oversized partial heaps increase peak work

**confirmed-source · P3 opportunity · GEMM pairwise, fallback batch search, optimized near-full-k search**

Original: PERF6, search_topk:L3, batch_matrix:opportunity2.

Sources: Sources/VectorCore/Operations/BatchOperations.swift:269; Sources/VectorCore/Operations/Operations.swift:160; Sources/VectorCore/Operations/Operations.swift:194; Sources/VectorCore/Operations/Operations.swift:265; Sources/VectorCore/Operations/Operations.swift:294; Sources/VectorCore/Operations/Kernels/TopKSelectionKernels.swift:85.

**Evidence:** FlatN² scores remain while row arrays built: about8N² score bytes at end before lifetime optimization, plus packs. Batch validates whole input then each findNearest validates candidates. Each chunk allocates k-sized TopKBuffer arrays and merge allocates replacement output.

**Qualification:** Source logical live allocations/complexity, not measured peak RSS after optimizer. C*k is upper-model for C concurrent chunks, not all possible schedulers.

**Improvement:** Validated private worker, reusable flat output, per-chunk capacities and near-full-k strategy. Changes must preserve merge invariants.

**Acceptance:** Release allocation profiles; shared membership/order at k nearN and varied chunk sizes; public invalid inputs still validate exactly before unsafe routes.

### API-25 — MatrixDistance cannot consume downstream contiguous cache without packing

**confirmed-source · P3 opportunity · Public MatrixDistance and VectorIndex centroid batch scorer**

Original: PERF7, batch_matrix:opportunity1.

Sources: Sources/VectorCore/Operations/MatrixDistance.swift:1; ../VectorIndex/Sources/VectorIndex/Kernels/CentroidBatchScore.swift:10; Docs/Numerical_Behavior.md:71.

**Evidence:** Inspected source and actual consumer: centroid scorer uses direct cblas_sgemm with contiguous centroidsFlat and once-flattened queries; comment rejects mandatory copies. Prepared candidates still own pack and queries are repacked.

**Qualification:** Existing unchecked shape/preparation/BLAS contracts are documented; do not recast malformed MatrixDistance inputs as new memory-safety findings. No copy latency measured.

**Improvement:** Consumer-driven checked borrowed/owned matrix view with rows/columns/stride/lifetime and reusable output.

**Acceptance:** Real centroid consumer parity and removed candidate repacking/allocation; invalid stride/shape checks before BLAS; lifetime tests.

### API-26 — KNearestHeap retains NaN ahead of numeric candidate

**qualified · P3 · Public legacy helper; only internal ExecutionOperations production caller**

Original: search_topk:legacy public heap.

Sources: Sources/VectorCore/Operations/MinHeap.swift:215.

**Evidence:** Fresh Release k1 insert NaN then0 retainsNaN; bare distance comparator source explains admission.

**Qualification:** No documented NaN-last/index tie contract here; not regression of fixed Operations/TopKSelection or their tests.

**Improvement:** Document finite-only policy or consolidate/deprecate; adopt shared comparator only with contract decision.

**Acceptance:** NaN/tie/finite policy tests if retained; route inventory ensures high-level searches stay on intended selector.

### API-27 — Schedulers do not promise or enforce prompt cancellation inside CPU chunks

**qualified · P3 · Public schedulers with noncooperative synchronous body**

Original: execution_concurrency:other cancellation.

Sources: Sources/VectorCore/Providers/CPUComputeProvider.swift:135; Sources/VectorCore/Operations/BatchOperations.swift:183; Sources/VectorCore/Protocols/ComputeProvider.swift:128.

**Evidence:** Inspected loops lack Task.checkCancellation; throwing task-group sibling cancellation does not force synchronous user body termination.

**Qualification:** Contract gap/opportunity, not proof task cancellation is universally ignored; throwing/suspending user closures can cooperate.

**Improvement:** Specify cancellation at chunk/item boundaries and behavior of nonthrowing APIs.

**Acceptance:** Cancel before start/during long chunks; bounded new work and cleanup under chosen contract; user body obligations stated.

### API-28 — Public executeBatch autotuning claim contradicts single full-range invocation

**confirmed-source · P3 · Public CPUComputeProvider.executeBatch direct caller**

Original: execution_concurrency:executeBatch.

Sources: Sources/VectorCore/Providers/CPUComputeProvider.swift:335; Sources/VectorCore/Providers/CPUComputeProvider.swift:356; Sources/VectorCore/Providers/CPUComputeProvider.swift:362.

**Evidence:** Function calls kernel once0..<n unconditionally for n>0; private measureProbes unused; explicitThreshold/explicitMinChunk fields stored but not consumed for that promise.

**Qualification:** No measured runtime regression. Existing inline comment explicitly admits sequential implementation, contradicting public doc.

**Improvement:** Correct public promise or implement tested scheduling without unsafe shared-pointer writes.

**Acceptance:** Spy range kernel shows documented calls and settings; Swift6 pointer ownership/concurrency validation before enabling parallel path.

### API-29 — AutoTuning lock covers arbitrary calibrator and serializes unrelated keys

**confirmed-source · P2 opt-in · Public calibrator; live Operations generic/custom metric calibration unless disabled/cache already populated/cap exhausted**

Original: execution_concurrency:AutoTuning.

Sources: Sources/VectorCore/Providers/AutoTuning.swift:102; Sources/VectorCore/Providers/AutoTuning.swift:118; Sources/VectorCore/Operations/Operations.swift:414.

**Evidence:** NSLock acquired before calibrator(); get/set acquire same nonrecursive lock. Operations calibrator invokes arbitrary metric.distance under it. Reentrant get/set waits on current thread lock. Global20ms cumulative cap is checked before full callback, so a single long callback can exceed it.

**Qualification:** Source proof, no deadlock subprocess run. Not all Operations routes calibrate: fused top-k and installed providers may bypass; cached keys bypass callback. No duration benchmark.

**Improvement:** Calibrate outside lock with per-key in-flight protocol; define reentrancy and time-budget semantics.

**Acceptance:** Bounded subprocess custom metric querying tuner plus direct reentrant calibrator; unrelated-key concurrency; no duplicate publication/race; slow callback policy test.

### API-30 — ComputeDevice runtime metadata includes placeholders

**confirmed-source · P3 · Public read-only device queries, not actual automatic GPU discovery**

Original: execution_concurrency:ComputeDevice.

Sources: Sources/VectorCore/Execution/ComputeDevice.swift:58; Sources/VectorCore/Execution/ComputeDevice.swift:153; Sources/VectorCore/Execution/ComputeDevice.swift:231.

**Evidence:** GPU availability checks compile-time Metal presence, ignores index; neural checks processorCount>0; GPU/Neural capabilities hard-coded. mostCapable picks neural while createProvider returns nil for it.

**Qualification:** CPU-only Core boundary is documented; placeholders cannot be used as actual device validation. No unsupported hardware runtime probe.

**Improvement:** Mark estimates/unknown explicitly or inject real platform discovery downstream.

**Acceptance:** Invalid GPU indices, no-neural hardware and provider creation consistency under fake discovery backend.

### API-31 — Disabled logger eagerly constructs messages; timer uses wall clock

**confirmed-reproduced · P3 · Public Logger and PerformanceTimer**

Original: execution_concurrency:logger timer.

Sources: Sources/VectorCore/Logging/Logger.swift:140; Sources/VectorCore/Logging/Logger.swift:195; Sources/VectorCore/Logging/Logger.swift:246.

**Evidence:** ExtraProbe disables debug logging then side-effecting message is evaluated once. Source evaluates autoclosure before private filtering. Timer calls CFAbsoluteTimeGetCurrent at start/end.

**Qualification:** No timing overhead or clock-jump failure measured; private minimumLevel field locking is sound.

**Improvement:** Filter before evaluating or pass closure through; use monotonic duration clock for elapsed timing.

**Acceptance:** Disabled message spy remains unevaluated; enabled once; controllable timer clock verifies elapsed policy.

### API-32 — MemoryPool recomputes pooled bytes using flatMap arrays under lock

**confirmed-source · P3 opportunity · Repaired public MemoryPool on buffer return**

Original: execution_concurrency:MemoryPool contention.

Sources: Sources/VectorCore/Utilities/MemoryPool.swift:285; Sources/VectorCore/Utilities/MemoryPool.swift:309.

**Evidence:** Two flatMap calls and reduction occur inside stateLock.withLock. Byte accounting uses stride and checks available budget correctly.

**Qualification:** Performance hypothesis only; not regression of prior retention fix or evidence lock contention dominates.

**Improvement:** Measure representative return-heavy workloads before choosing maintained counter.

**Acceptance:** Release allocation/contention profile; if changed, retained-byte invariant across type/size buckets and teardown preserved.

### API-33 — Normalization hints reach real downstream query and cached inverse norm consumers

**confirmed-source · P1 dependency · VectorIndex typed APIs; VA cosine kernels; Core wrapper defect owned separately**

Original: A2, downstream_interfaces:IF5.

Sources: ../VectorIndex/Sources/VectorIndex/TypedOverloads.swift:102; ../VectorIndex/Sources/VectorIndex/TypedOverloads.swift:119; ../VectorIndex/Sources/VectorIndex/HNSWIndex.swift:1097; ../future/VectorAccelerate/Sources/VectorAccelerate/Integration/KernelDistanceProviders.swift:189; ../future/VectorAccelerate/Sources/VectorAccelerate/Integration/KernelDistanceProviders.swift:221.

**Evidence:** Fresh source trace: typed HNSW insert supplies1 or reciprocal hint; valid live cache appends supplied positive inverse norm unchanged. Query flags skip normalization. VA both-normalized path computes raw dot then1-dot. Correct hinted [2,0] stale after mutation against[1,0] would yield-1 instead of0 on this branch.

**Qualification:** No downstream build/GPU replay. VA AcceleratedVectorIndex captures hints but its doc explicitly says L2 kernel does not yet consume them, so captured metadata alone is not evidence of active L2 corruption. Cached Index damage is conditional on live cache/route.

**Improvement:** Fix wrapper invariant first, then test downstream insertion/query; assess rebuilding caches populated from stale values.

**Acceptance:** Mutate initially correct hints and compare hinted/unhinted insertion+queries, including HNSW live cache and VA cosine, throwing mutations.

### API-34 — VectorIndex batch query flag comes only from first query

**confirmed-source · P2 downstream · VectorIndex Flat/IVF/optimized flat typed batchSearch; HNSW handles each query**

Original: downstream_interfaces:compatibility heterogeneous hints.

Sources: ../VectorIndex/Sources/VectorIndex/TypedOverloads.swift:107; ../VectorIndex/Sources/VectorIndex/TypedOverloads.swift:183; ../VectorIndex/Sources/VectorIndex/TypedOverloads.swift:205; ../VectorIndex/Sources/VectorIndex/FlatIndex.swift:131; ../VectorIndex/Sources/VectorIndex/IVFIndex.swift:538.

**Evidence:** Legal heterogeneous NormalizationHint array: first true, second false; shared flag skips second query norm. Source consumer uses flag for all query rows. HNSW loops individual hints at160.

**Qualification:** Separate downstream defect even after Core A2 fix; no executed index integration test.

**Improvement:** Carry per-query flags or only enable batch optimization when all queries qualify.

**Acceptance:** Mixed normalized/raw hint batches in both orders match individual searches across Flat/IVF/optimized-flat.

### API-35 — Provider seam erases Float scalar/score conventions

**qualified · P3 · Public BatchKernelProvider/VectorCollection; VA direct enum distance versus Core distance**

Original: downstream_interfaces:architecture A1.

Sources: Sources/VectorCore/Protocols/BatchKernelProvider.swift:18; Sources/VectorCore/Integration/VectorCollection.swift:88; ../future/VectorAccelerate/Sources/VectorAccelerate/Integration/MetalComputeProvider.swift:314; ../future/VectorAccelerate/Sources/VectorAccelerate/Integration/MetalComputeProvider.swift:333; ../future/VectorAccelerate/Sources/VectorAccelerate/Integration/MetalComputeProvider.swift:385.

**Evidence:** VA direct enum dot returns raw dot; Core metric returns negative dot. VA bridge explicitly excludes dot GPU route and recovers actual Float witness; no silent bridge sign replacement. Core Operations allows external ordering by documented policy.

**Qualification:** Documented deliberate direct-API semantics and provider ordering variation, not proven transparent route bug. Float-constraining seam is design suggestion with compatibility cost.

**Improvement:** Explicit score-kind/scalar contract and provider conformance suite.

**Acceptance:** Direct-vs-bridge sign tests and NaN/tie policy declarations; custom metric witness remains authoritative.

### API-36 — VectorCollection is not actual VectorIndex conformance seam

**confirmed-source · P3 · Core small/reference collection API and local downstream snapshots**

Original: A10, downstream_interfaces:architecture A3.

Sources: Sources/VectorCore/Integration/VectorCollection.swift:13; ../VectorIndex/Sources/VectorIndex/TypedOverloads.swift:89.

**Evidence:** rg finds no VectorCollection use/conformance in either sibling Sources. Actual bridge is async typed index overloads sharing vectors/SearchResult.

**Qualification:** Future-tense will implement is aspiration, but example presenting HNSW conformance does not compile against this snapshot. No claim production ANN storage belongs in Core.

**Improvement:** Describe current small utility or design consumer-driven async throwing interface with snapshot/mutation contract.

**Acceptance:** Compile actual cross-package integration examples at pinned versions; route map reflects real conformance.

### API-37 — BufferProvider example puts await in defer and handles require explicit lifetime policy

**confirmed-source · P3 · Public raw BufferHandle and BufferProvider docs**

Original: downstream_interfaces:architecture A4.

Sources: Sources/VectorCore/Protocols/BufferProvider.swift:80; Sources/VectorCore/Protocols/BufferProvider.swift:97; ../future/VectorAccelerate/Sources/VectorAccelerate/Core/BufferPool.swift:627.

**Evidence:** Doc contains defer { await provider.release(handle) }; async await cannot appear in synchronous defer body. Handle is copied address/id/size, not an owning object or pointee synchronizer.

**Qualification:** No demonstrated valid-use UAF; VA requires keeping provider alive while handle active. Ownership/borrow docs elsewhere already state unsafe lifetime duties.

**Improvement:** Provide compiling do/catch async cleanup or scoped helper and document retention/aliasing requirements.

**Acceptance:** Swift6 doc consumer compile; success/throw/cancel each releases once, owner retained through operation. Active-handle policy coordinated with teardown.

### API-38 — Public low-level implementation types create support obligations

**qualified · P3 · Exported caches/kernels/providers/platform helpers**

Original: downstream_interfaces:architecture A5.

Sources: Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:1; Sources/VectorCore/Operations/Kernels/SoAFP16Cache.swift:1.

**Evidence:** Source contains public types/methods described elsewhere as implementation details; callers can compile against them.

**Qualification:** Not correctness bug by visibility alone and no binary ABI stability inferred. Complete symbol graph inventory not performed here.

**Improvement:** Inventory exported symbols and designate supported low-level APIs; intentional hiding needs compatibility decision.

**Acceptance:** Generated public symbol diff and downstream consumer compile tests for supported surface.

### API-39 — VA CPU fallbacks and Core unchecked normalization use different finite-range algorithms

**qualified · P2 dependency · Active VA fallback normalization and acceleratedNormalizeUnchecked fallback to Core**

Original: downstream_interfaces:normalization parity.

Sources: ../future/VectorAccelerate/Sources/VectorAccelerate/Operations/StableNormalization.swift:56; ../future/VectorAccelerate/Sources/VectorAccelerate/Operations/StableNormalization.swift:103; ../future/VectorAccelerate/Sources/VectorAccelerate/Operations/AccelerateFallback.swift:159; ../future/VectorAccelerate/Sources/VectorAccelerate/Operations/SIMDFallback.swift:260; ../future/VectorAccelerate/Sources/VectorAccelerate/Core/FallbackProvider.swift:195; ../future/VectorAccelerate/Sources/VectorAccelerate/Integration/VectorCoreIntegration.swift:189.

**Evidence:** Verified actual calls and VA scaled-domain divide versus Core magnitude reconstruction; numerical reviewer owns finite tiny/huge proof.

**Qualification:** Unchecked zero input violates explicit nonzero assertion and is not an additional valid-input defect. GPU parity not tested; avoid universal provider parity claim.

**Improvement:** Reuse explicit stable contract and shared fixtures after Core numeric correction.

**Acceptance:** Exponent-range direction/norm checks through actual Core/VA CPU paths, optional GPU separately; no false true normalization hints.

### API-40 — Local sibling dependency pins differ from reviewed Core main

**confirmed-source · P3 informational · Local source snapshots; no compatibility build**

Original: downstream_interfaces:compatibility snapshots.

Sources: ../VectorIndex/Package.swift:20; ../VectorIndex/Package.resolved:10; ../future/VectorAccelerate/Package.swift:97; ../future/VectorAccelerate/Package.resolved:20; Sources/VectorCore/Version.swift:1.

**Evidence:** Index HEAD6f114923d228739c207734685cfde18dd1c6cf60 pins Core0.3.2/ad39ab6f4904838e300b836a42c7376d6e11da26; VA HEAD3cd1442d4f4e092dda4f9bcd5e0878a9f8cad8bb pins0.3.3/fca4b602383589c46b627d8a0de2b6b2a68cd07d. Core reviewed8b02e75 still identifies0.3.3. Index/Core Swift6.0/macOS14+iOS17; VA Swift6.2/macOS26+iOS26/tvOS26/visionOS26, no watchOS.

**Qualification:** Normal unreleased development, not version compatibility defect. VA stale comments about unavailable v26 enums contradict manifest, downstream docs only.

**Improvement:** Record exact source/pin/toolchain matrix and correct stale platform comments.

**Acceptance:** Consumer builds against intended release/main separately; do not infer platform lowering from shared protocols.

### API-41 — Existing provider tests do not cross GEMM query threshold

**confirmed-source · P3 · BatchKernelProvider precedence tests**

Original: search_topk:provider precedence coverage.

Sources: Tests/ComprehensiveTests/BatchKernelProviderTests.swift:82; Tests/ComprehensiveTests/BatchKernelProviderTests.swift:108; Sources/VectorCore/Operations/Operations.swift:180.

**Evidence:** Tests use Q4/Q5 with N300; GEMM requiresQ>=8. Source delegation precedes GEMM, so current behavior is correct but named precedence claim lacks reaching fixture.

**Qualification:** Test gap, not provider dispatch bug.

**Improvement:** Raise/add fixture with Q>=8,N>=256 supported optimized metric while spy override installed.

**Acceptance:** Sentinel provider output at actual GEMM gate; default per-query and whole-batch override both exercised.

### API-42 — Dormant execution engines duplicate weak validation/reduction semantics

**qualified · P3 dormant · Internal ExecutionOperations/SyncBatchOperations; live transform/matrix conveniences separately identified**

Original: batch_matrix:opportunity4.

Sources: Sources/VectorCore/Operations/ExecutionOperations.swift:346; Sources/VectorCore/Operations/ExecutionOperations.swift:359; Sources/VectorCore/Operations/ExecutionOperations.swift:535; Sources/VectorCore/Operations/ExecutionOperations.swift:554; Sources/VectorCore/Operations/SyncBatchOperations.swift:341.

**Evidence:** Internal reduce injects initial per chunk and recursion; validators inspect first row Release or every100th; naive zero-diagonal/heap behavior duplicates public systems. No current public route for these reduce/search methods established.

**Qualification:** Do not label all ExecutionOperations dormant: public Array transforms/cosine matrix are live. Nonidentity initial conflicts with identity reduction law in existing ComputeProvider tests, so law/documentation first.

**Improvement:** Consolidate/remove unused engines; rederive live routing before activating any helper.

**Acceptance:** Mechanical caller map and property tests for any retained reachable engine; shape validation exhaustive before unsafe operations and reduction identity/associativity documented.

### API-43 — SoA/unrolled/prefetch throughput claims lack supplied benchmark/compiler evidence

**unverified · P3 · Public comments/helpers and batch kernel docs**

Original: batch_matrix:opportunity6, simd_kernels:performance claims.

Sources: Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift:115; Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift:459; Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift:1343; Sources/VectorCore/Optimization/OptimizationAttributes.swift:57.

**Evidence:** Inspected prefetch helper is unused load; Manhattan implementation uses original AoS/scalar abs lanes despite advertised vDSP optimization. No current benchmark evidence for stated1.5–2x or10–15% examined.

**Qualification:** Unsupported speedup claims, not demonstrated slowdown; optimizer may remove nominal prefetch, assembly not inspected.

**Improvement:** Remove/qualify performance assertions or tie to reproducible hardware/size/build benchmark; inspect assembly for real prefetch.

**Acceptance:** Release benchmark artifacts and emitted instruction check support each retained claim.

### API-44 — unrolledLoop arbitrary stride emits wrong indices; zero widths fail to progress

**confirmed-source · P2 exposed helper · Public helper opt-in; no internal/downstream production callers established**

Original: simd_kernels:3.

Sources: Sources/VectorCore/Optimization/OptimizationAttributes.swift:83; Sources/VectorCore/Optimization/OptimizationAttributes.swift:108.

**Evidence:** At stride1 body(i+1) always runs, duplicates and includes count; stride>4 emits max4 before advancing stride, skipping interior indices. Width0 loop never advances; negative widths have malformed progression/ranges.

**Qualification:** Original exact-source execution exists but was not independently replayed here; source one-step proof is sufficient. Zero/negative width subcase is lower-priority invalid-input validation; positive1/5/8 violates ordinary stride expectation.

**Improvement:** Implement full stride coverage or explicitly constrain supported positive widths; overflow-safe remaining-count conditions.

**Acceptance:** Emitted indices/flattened ranges exactly0..<count for widths1...17, tail sizes, zero/negative rejection and throwing body. Avoid executing huge count merely to test endpoint arithmetic.

### API-45 — VectorMath.nearestNeighbors fully sorts while claiming heap complexity

**confirmed-source · P3 opportunity · Public fixed Vector convenience**

Original: PERF5, simd_kernels:6.

Sources: Sources/VectorCore/Operations/VectorMath.swift:386; Sources/VectorCore/Operations/VectorMath.swift:402; Sources/VectorCore/Operations/VectorMath.swift:413.

**Evidence:** Source computes all scores, sorted comparator then prefix(k). Complexity O(N logN), memoryO(N); k0 still computes/sorts. Negative k hits Array.prefix precondition.

**Qualification:** Negative neighbor count has natural nonnegative precondition but missing explicit API documentation; not a new finite k wrong-ranking proof. Bare Float comparator has unspecified NaN/tie policy; do not transplant modern selector law without decision.

**Improvement:** Use shared selection or correct docs and k policy; coordinate NaN/tie behavior.

**Acceptance:** Metric spy k0 policy, k bounds, common order/member tests; Release allocation/time before performance claim.

### API-46 — Named optimized metric factory cannot create dotProduct

**confirmed-source · P3 · Public OptimizedDistanceMetrics.create(named:)**

Original: simd_kernels:factory correctness.

Sources: Sources/VectorCore/Operations/OptimizedDistanceMetrics.swift:368; Sources/VectorCore/Operations/OptimizedDistanceMetrics.swift:402.

**Evidence:** MetricType raw value dotProduct has uppercaseP; every input lowercased before rawValue construction; lowercasing cannot yield that enum case. Enum-based create works.

**Qualification:** One-step mapping proof, not runtime replay. Separate from search metric witness replacement.

**Improvement:** Normalize enum lookup consistently or explicit accepted-name mapping.

**Acceptance:** Case-insensitive roundtrip for every MetricType.allCases and unknown names returnnil.

### API-47 — Singleton full-matrix triangular import reads nonexistent element

**confirmed-reproduced · P2 · Public TriangularMatrix.fromFullMatrix**

Original: C15, batch_matrix:5.

Sources: Sources/VectorCore/Utilities/TriangularMatrix.swift:218.

**Evidence:** Fresh Release batch_probe singleton [[0]] SIGTRAP. Nonempty and square checks succeed, then initializer reads matrix[0][1]. TriangularMatrix size1 itself permits empty strict upper triangle.

**Qualification:** Singleton input satisfies stated square-matrix requirement. Diagonal/lower-triangle omission is deliberate and is not the defect.

**Improvement:** Use existing valid seed or explicit empty storage for singleton; preserve import contract.

**Acceptance:** Empty, singleton,2x2, ragged inputs and singleton roundtrip to zero-diagonal full matrix; no index trap.

### API-48 — CPU sequential reduce ignores initial but identity is the existing provider law

**qualified · P3 documentation · Public CPUComputeProvider.parallelReduce**

Original: execution_concurrency:strengths reduce identity.

Sources: Sources/VectorCore/Providers/CPUComputeProvider.swift:284; Sources/VectorCore/Protocols/ComputeProvider.swift:78.

**Evidence:** Sequential branch returns rangeWork(items); parallel combines partials into initial. With identity initial the difference is lawful; original conformance tests intentionally use identity.

**Qualification:** Not a new numeric defect from nonidentity input. Current protocol comments should state identity and associativity requirements; floating-point nonassociativity means bitwise mode equivalence is not promised.

**Improvement:** Promote existing reduction law to current public documentation and distinguish sequential-fold APIs.

**Acceptance:** Conformance tests for identity and chunk combination; documented limits for nonassociative floating-point reductions.

## Component subsection crosswalk and factual appendix

- execution_concurrency EC1–7: API05,06,02,03,04,07,08. Other opportunities: API27–32. Diagnostics: API21. Strengths/rejections: MemoryPool synchronous lock and stride accounting observed; CPU task-local suppression checked in all five helpers; TaskLocal bindings propagate to structured children. This does not cover unrelated or detached tasks. LogConfiguration lock and ThreadSafeConfiguration primitive isolation observed, separate from transaction/API03. Identity-initial reduction law is documentation work, not a new numeric defect. Generic unsafe pool initialization/lifetime and forged/reused handles remain caller obligations. Swift6 TypedBufferPool usability claim not independently recompiled here; no contrary evidence and no defect filed. Coverage/probe-artifact sections are historical provenance, not assertions of a fresh exhaustive review.
- search_topk S1–4: API01,17,18,17. L1–3: API22,23,24. Legacy heap: API26. Provider precedence test gap has its own row. Shared comparator/merge strengths supported by current source and root full suites; no renewed NaN/tie regression alleged. Unsafe pointer capacity/Int32 boundaries are explicit caller contracts; zero-capacity internal buffer has no invalid public route established. Squared-distance rounded-sqrt membership differences are documented. Scope/limitations retain original exclusions.
- downstream_interfaces IF1–5: API01,09,10,11,33. Architecture A1–A5: provider seam, unused hooks, collection seam, BufferProvider ownership documentation, public surface rows. Normalization parity and all compatibility bullets have dedicated rows; VA whole-batch dispatch inheritance mapped under API04. Existing SoA descriptor and ownership docs are supported interfaces, not issues by mere presence. Hash excluding score remains legal because equal values hash equally; Comparable NaN/tie policy remains unproven design issue. False user-authored normalization assertions and unsafe-pointer lifetime violations are excluded.
- batch_matrix findings1–7: aggregation API13; variance C8 and entropy C9 assigned numeric reviewer (same replay logs corroborate); suffix API14; singleton API47 with fresh batch_probe singleton SIGTRAP; nonpositive batch API07; metric pairwise/similarity API15–16. Opportunities1–6: packing API25, reshape API24, small-k API23, dormant engines, taskfanout API04, unsupported claims row. Strengths: matrix route only fixed optimized types; no new ragged DynamicVector GEMM route; range kernel lower-bound indexing considered correct under valid buffer contract. Documented MatrixDistance ragged/zero/overflow and numerical limits excluded. Scope/limits do not promise exhaustive finite-input correctness.
- SIMD delegated supplemental: loop widths, full-sort VectorMath convenience, named dotProduct factory have dedicated rows. PCA extreme oversampling mapped API20. UMAP dimensions/rate and malformed CSR validation belong root/numeric and are deliberately not adjudicated here. PERF4 FP16 range wrappers and PERF8 cache/UMAP/PCA memory models belong numeric/ownership; no search/batch-specific additional allegation under those IDs.

## Important corrections for consolidation

1. Zero-query suffix fill is within caller-provided capacity, not out-of-allocation memory corruption.
2. Directed pairwise custom metric is a symmetry-contract clarification; the built-in negative-dot diagonal and named similarity sign fail without that ambiguity.
3. Pool instrumentation proves returned raw cached allocations are not freed, not arbitrary active-handle or generic initialized-element lifetime policy.
4. Existing GEMM precision and full-matrix memory costs are disclosed limitations; add policy/budget as an architectural improvement. Tiling alone does not fix cancellation; limited reranking needs sufficient candidate retention.
5. CPU near-Int.max requires a tiny valid range and all five helpers now have fresh runtime evidence. PCA Int.max oversampling/factory zero/negative capacities are explicit lower-priority invalid/extreme configuration cases.
6. VA AcceleratedVectorIndex stores hints but explicitly does not use them in its current L2 kernel. Active downstream hint risk is independently traced to Index query/HNSW cache and VA cosine routes.
7. No proposed remedy has been implemented or demonstrated sufficient by these probes. Acceptance criteria are required to close each claim class.
