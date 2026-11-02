# Validated claim register

[Improvement guide](README.md) · [Original-ID crosswalk](CROSSWALK.md) · [JSON](claims.json)

This register is the root reviewer’s final adjudication. Records include related symptoms, contract decisions, context and investigations; the record count is not an independent bug count. “Confirmed—reproduced” applies to the specific observation in Evidence, not every inferred consequence or proposed remedy. All improvements are unimplemented.

## Coverage

| Verdict | Records |
|---|---|
| confirmed-reproduced | 57 |
| confirmed-source | 42 |
| qualified | 28 |
| rejected | 1 |
| unverified | 3 |

Categories: architecture: 21, code quality: 91, performance: 19.

## Index

| Record | Priority | Category | Verdict | Claim |
|---|---|---|---|---|
| [API-01](#api-01) | P1 | architecture | confirmed-reproduced | Two search facades replace unsupported metric witnesses with Euclidean |
| [API-02](#api-02) | P2 | architecture | confirmed-reproduced | Batch configuration storage is disconnected from operation settings |
| [API-03](#api-03) | P2 | architecture | confirmed-reproduced | Configuration read-modify-write loses concurrent updates |
| [API-04](#api-04) | P2 | architecture | confirmed-reproduced | Several schedulers eagerly create all child tasks |
| [API-05](#api-05) | P2 | architecture | confirmed-reproduced | Cached raw actor pool allocations are not freed on owner destruction |
| [API-06](#api-06) | P2 | architecture | confirmed-reproduced | SwiftBufferPool age cleanup does not enforce maxPoolSize |
| [API-07](#api-07) | P2 | code quality | confirmed-reproduced | Nonpositive process batch sizes trap or discard work |
| [API-08](#api-08) | P2 | code quality | confirmed-reproduced | CPU chunk endpoints overflow before clamping valid short ranges |
| [API-09](#api-09) | P2 | architecture | confirmed-reproduced | Failed collection update deletes existing value |
| [API-10](#api-10) | P2 | architecture | confirmed-reproduced | Explicit/generated IDs overwrite records |
| [API-11](#api-11) | P2 | architecture | confirmed-reproduced | computedScore pole breaks negative-dot relevance |
| [API-12](#api-12) | P3 | architecture | confirmed-source | Advertised buffer/global configuration hooks have no active internal consumers |
| [API-13](#api-13) | P1 | code quality | confirmed-source | Collection aggregation trusts first row before unsafe reads/forced factories |
| [API-14](#api-14) | P2 | code quality | confirmed-reproduced | Zero-query SoA cosine fills oversized output suffix |
| [API-15](#api-15) | P2 | code quality | confirmed-reproduced | Pairwise hard-coded zero diagonal fails built-in negative dot |
| [API-16](#api-16) | P2 | code quality | confirmed-reproduced | cosineSimilarityMatrix returns negative off-diagonal similarity and zero diagonal |
| [API-17](#api-17) | P2 | code quality | confirmed-reproduced | Bulk bounded heap membership differs from streaming; negative capacity loops |
| [API-18](#api-18) | P2 | code quality | confirmed-reproduced | TopKResult decoder violates parallel-array invariant |
| [API-19](#api-19) | P3 | code quality | confirmed-reproduced | Factory batch dimension zero reaches remainder by zero |
| [API-20](#api-20) | P3 | code quality | confirmed-reproduced | PCA oversampling Int.max overflows before bounded sketch clamp |
| [API-21](#api-21) | P2 | code quality | confirmed-source | FP16 diagnostics enabled flag is read without lock |
| [API-22](#api-22) | P2 | performance | qualified | Automatic batch GEMM requires full Q*N scores and can change near-neighbor ranking |
| [API-23](#api-23) | P3 | performance | confirmed-source | Small-k array/generic selection retains O(N) temporary distances/pairs |
| [API-24](#api-24) | P3 | performance | confirmed-source | Redundant matrix reshape, validation and oversized partial heaps increase peak work |
| [API-25](#api-25) | P3 | performance | confirmed-source | MatrixDistance cannot consume downstream contiguous cache without packing |
| [API-26](#api-26) | P3 | code quality | qualified | KNearestHeap retains NaN ahead of numeric candidate |
| [API-27](#api-27) | P3 | code quality | qualified | Schedulers do not promise or enforce prompt cancellation inside CPU chunks |
| [API-28](#api-28) | P3 | code quality | confirmed-source | Public executeBatch autotuning claim contradicts single full-range invocation |
| [API-29](#api-29) | P2 | code quality | confirmed-source | AutoTuning lock covers arbitrary calibrator and serializes unrelated keys |
| [API-30](#api-30) | P3 | code quality | confirmed-source | ComputeDevice runtime metadata includes placeholders |
| [API-31](#api-31) | P3 | code quality | confirmed-reproduced | Disabled logger eagerly constructs messages; timer uses wall clock |
| [API-32](#api-32) | P3 | performance | confirmed-source | MemoryPool recomputes pooled bytes using flatMap arrays under lock |
| [API-33](#api-33) | P1 | architecture | confirmed-source | Normalization hints reach real downstream query and cached inverse norm consumers |
| [API-34](#api-34) | P2 | code quality | confirmed-source | VectorIndex batch query flag comes only from first query |
| [API-35](#api-35) | P3 | code quality | qualified | Provider seam erases Float scalar/score conventions |
| [API-36](#api-36) | P3 | architecture | confirmed-source | VectorCollection is not actual VectorIndex conformance seam |
| [API-37](#api-37) | P3 | code quality | confirmed-source | BufferProvider example puts await in defer and handles require explicit lifetime policy |
| [API-38](#api-38) | P3 | code quality | qualified | Public low-level implementation types create support obligations |
| [API-39](#api-39) | P2 | code quality | qualified | VA CPU fallbacks and Core unchecked normalization use different finite-range algorithms |
| [API-40](#api-40) | P3 | code quality | confirmed-source | Local sibling dependency pins differ from reviewed Core main |
| [API-41](#api-41) | P3 | code quality | confirmed-source | Existing provider tests do not cross GEMM query threshold |
| [API-42](#api-42) | P3 | code quality | qualified | Dormant execution engines duplicate weak validation/reduction semantics |
| [API-43](#api-43) | P3 | performance | unverified | SoA/unrolled/prefetch throughput claims lack supplied benchmark/compiler evidence |
| [API-44](#api-44) | P2 | code quality | confirmed-source | unrolledLoop arbitrary stride emits wrong indices; zero widths fail to progress |
| [API-45](#api-45) | P3 | performance | confirmed-source | VectorMath.nearestNeighbors fully sorts while claiming heap complexity |
| [API-46](#api-46) | P3 | code quality | confirmed-source | Named optimized metric factory cannot create dotProduct |
| [API-47](#api-47) | P2 | code quality | confirmed-reproduced | Singleton full-matrix triangular import reads nonexistent element |
| [API-48](#api-48) | P3 | code quality | qualified | CPU sequential reduce ignores initial but identity is the existing provider law |
| [NUM-01](#num-01) | P2 | code quality | confirmed-reproduced | Scalar division computes an overflowing reciprocal even when each quotient is finite. |
| [NUM-02](#num-02) | P2 | code quality | confirmed-reproduced | Normalization reconstructs a mismatched scale or relies on an unrepresentable norm/reciprocal. |
| [NUM-03](#num-03) | P2 | code quality | confirmed-reproduced | QuantizationSchemes clamps its zero point so positive/negative-only and constant intervals cannot be represented. |
| [NUM-04](#num-04) | P2 | code quality | confirmed-source | Clamping integer conversions trap before the clamp for finite large intermediates. |
| [NUM-05](#num-05) | P2 | code quality | confirmed-reproduced | Quantized cosine overflows the product of finite squared norms. |
| [NUM-06](#num-06) | P2 | code quality | confirmed-reproduced | Batch magnitude variance uses cancellation-prone raw moments above N=1000. |
| [NUM-07](#num-07) | P2 | code quality | confirmed-reproduced | Entropy cutoff and Float sum violate finite scale invariance. |
| [NUM-08](#num-08) | P2 | code quality | confirmed-reproduced | Swift SVD accepts nonorthogonal columns as converged at harmless input scales. |
| [NUM-09](#num-09) | P2 | code quality | confirmed-reproduced | Swift QR and SVD final normalization lose representable finite vectors through Float squared norms. |
| [NUM-10](#num-10) | P3 | code quality | qualified | Rounding PCA mean before centering doubles variance in a valid offset fixture. |
| [NUM-11](#num-11) | P2 | code quality | confirmed-reproduced | UMAP output curve fit depends on units and silently returns its initial guess. |
| [NUM-12](#num-12) | P2 | code quality | confirmed-reproduced | Smooth-kNN bandwidth search starts at absolute1 and fails outside its fixed64-step range. |
| [NUM-13](#num-13) | P2 | code quality | confirmed-reproduced | UMAP accepts infinite learning rate/spread at an existing throwing validation boundary. |
| [NUM-14](#num-14) | P2 | code quality | confirmed-reproduced | Approximate equality accepts NaN differences as equal. |
| [NUM-15](#num-15) | P2 | code quality | qualified | safeLog bypasses options for NaN and positive infinity. |
| [NUM-16](#num-16) | P2 | code quality | confirmed-reproduced | Adaptive distance ignores threshold and candidate range. |
| [NUM-17](#num-17) | P2 | code quality | confirmed-reproduced | Quantized tuner cache excludes accuracyRequirement, making selection call-history dependent. |
| [NUM-18](#num-18) | P3 | code quality | confirmed-reproduced | Diagnostics misclassifies underflows, rounded boundary values, and observed extrema. |
| [NUM-19](#num-19) | P3 | code quality | confirmed-source | KernelAutoTuner accepts memory/latency constraints it never uses and persists profiles without provenance. |
| [NUM-20](#num-20) | P3 | code quality | qualified | Explicit selectMixedPrecisionStrategy maps full precision mode to nonzero tolerance. |
| [NUM-21](#num-21) | P3 | performance | confirmed-source | Mixed tuner calibrates all5 strategies synchronously on exact workload key and converts candidates inside each reduced execution. |
| [NUM-22](#num-22) | P2 | performance | confirmed-source | FP16 range wrappers map all candidates even for small or empty ranges. |
| [NUM-23](#num-23) | P3 | code quality | confirmed-source | Source numerical/performance assurances exceed tests or contradict implementation. |
| [NUM-24](#num-24) | P3 | code quality | confirmed-source | Immutable FP16 vector pool cannot refill storage for conversion of new values via a supported API. |
| [NUM-25](#num-25) | P3 | code quality | qualified | CSR initializer may dereference malformed offsets before throwing validation catches them. |
| [NUM-26](#num-26) | P3 | code quality | qualified | Accepted extreme integer configuration can trap before throwing or clamping. |
| [NUM-27](#num-27) | P3 | code quality | qualified | Exact brute-force kNN builder is exhaustive over computed Gram scores, not a reliable numerical distance oracle. |
| [NUM-28](#num-28) | P3 | code quality | qualified | UMAP initial coordinates are shape checked, not validated for finiteness or safe scale. |
| [NUM-29](#num-29) | P3 | performance | confirmed-source | UMAP and PCA expose material in-memory storage costs at large corpus/sample size. |
| [NUM-30](#num-30) | P3 | performance | confirmed-source | Graph sorts/epoch scans and PCA per-point transforms offer concrete allocation/work opportunities. |
| [NUM-31](#num-31) | P3 | code quality | qualified | Long synchronous manifold/factorization work lacks cancellation checkpoints. |
| [NUM-32](#num-32) | P3 | code quality | qualified | C INT8 accumulation can exceed signed32 range only beyond supported fixed-vector dimensions. |
| [NUM-33](#num-33) | P3 | code quality | qualified | C header range indexing wording and proposed unimplemented declarations need precise public status. |
| [NUM-34](#num-34) | P3 | performance | qualified | Allocation-heavy vector helpers, unequal SIMD reduction structure and platform heuristics warrant measurement, not speed claims. |
| [NUM-35](#num-35) | P3 | code quality | qualified | LAPACK dimension ceiling does not by itself establish extreme workspace arithmetic safety. |
| [OWN-01](#own-01) | P1 | architecture | confirmed-reproduced | Public DynamicVector and optimized FP32 storage mutations break shape invariants. |
| [OWN-02](#own-02) | P1 | architecture | confirmed-reproduced | INT8 raw constructors, mutable storage, and synthesized Codable permit malformed fixed storage. |
| [OWN-03](#own-03) | P1 | architecture | confirmed-reproduced | NormalizationHint retains metadata after direct wrapped mutation or a throwing mutable-buffer closure. |
| [OWN-04](#own-04) | P1 | architecture | confirmed-reproduced | SoAFP16Cache512 treats vector hash digests as exact candidate identity. |
| [OWN-05](#own-05) | P1 | code quality | confirmed-reproduced | Compact-tier triangular matrix copies share later mutations. |
| [OWN-06](#own-06) | P2 | code quality | confirmed-reproduced | Binary in-place DynamicVector routes omit dimension checks. |
| [OWN-07](#own-07) | P2 | performance | confirmed-reproduced | Optimized-vector hashes omit all but first and last eight scalar values. |
| [OWN-08](#own-08) | P2 | code quality | confirmed-reproduced | Pointer normalization documents16-byte alignment as preferred but SIMD loads require it. |
| [OWN-09](#own-09) | P2 | code quality | qualified | Vector384Optimized permanently rebinds typed-owner memory where other widths restore binding. |
| [OWN-10](#own-10) | P2 | code quality | confirmed-reproduced | Compact generic storage does not destroy manually initialized ManagedBuffer tail elements. |
| [OWN-11](#own-11) | P2 | code quality | confirmed-reproduced | alignedCopy has incompatible initialize versus update semantics across alignment branches. |
| [OWN-12](#own-12) | P2 | performance | confirmed-source | HybridStorage repeating construction/fill performs quadratic buffer traffic. |
| [OWN-13](#own-13) | P2 | performance | confirmed-reproduced | TriangularMatrix scalar writes copy complete standard/aligned allocations even when uniquely owned. |
| [OWN-14](#own-14) | P3 | code quality | confirmed-reproduced | Dormant TieredStorage bulk access loses standard writes and inline writes on thrown closure exit. |
| [OWN-15](#own-15) | P3 | code quality | confirmed-source | Internal StorageSlice returns scoped pointer without keeping its owner alive. |
| [OWN-16](#own-16) | P3 | code quality | confirmed-source | Experimental consuming into() returns memory still freed by deinit. |
| [OWN-17](#own-17) | P2 | code quality | qualified | Public FP16 pointer getters escape ContiguousArray borrow scopes. |
| [OWN-18](#own-18) | P3 | code quality | confirmed-source | SoAFP16Cache512 accepts nonpositive maxSize then evicts from an empty list. |
| [OWN-19](#own-19) | P2 | performance | confirmed-source | FP16 cache caps entry count while payload bytes scale with candidates per entry. |
| [OWN-20](#own-20) | P3 | code quality | confirmed-source | StaticDimension.Storage customization is unused by Vector<D> storage yet controls extension availability. |
| [OWN-21](#own-21) | P3 | code quality | confirmed-source | VectorTypeFactory optimized wording describes generic fixed-vector selection, not specialized concrete types. |
| [OWN-22](#own-22) | P3 | performance | confirmed-source | VectorCoreOperations creates redundant arrays and temporary results. |
| [OWN-23](#own-23) | P3 | code quality | qualified | Performance/robustness comments exceed evidence and overlapping storage layers obscure actual ownership. |
| [OWN-24](#own-24) | Context | code quality | qualified | SoA/page/raw ownership, size and custom-conformer obligations are already documented. |
| [OWN-25](#own-25) | P3 | code quality | unverified | AlignedDynamicArrayStorage fallback can weaken alignment while guaranteedAlignment reports the request. |
| [OWN-26](#own-26) | Context | code quality | rejected | Reject extending HybridStorage quadratic-copy behavior to default generic DimensionStorage without evidence. |
| [OWN-27](#own-27) | P3 | architecture | confirmed-source | Extension-only protocol methods select static defaults in generic code. |
| [SER-01](#ser-01) | P2 | code quality | confirmed-reproduced | A valid empty DynamicVector binary encoding cannot be decoded by its own type. |
| [SER-02](#ser-02) | P2 | code quality | confirmed-reproduced | Serialization tests include 15 comment-only bodies and mislabeled optimized-type coverage. |
| [SER-03](#ser-03) | P3 | performance | qualified | Binary conversion has scalar loops and temporary representations; a Release allocation speedup is not yet established. |
| [SER-04](#ser-04) | P3 | architecture | qualified | Binary and JSON representations have different type/precision contracts. |
| [SER-05](#ser-05) | P3 | architecture | confirmed-source | Vectors expose binary methods without adopting the public binary protocols. |
| [SER-06](#ser-06) | P3 | code quality | confirmed-reproduced | Header documentation describes VECT while the actual little-endian wire bytes spell TCEV. |
| [SER-07](#ser-07) | P3 | code quality | confirmed-reproduced | Decoder framing/CRC bounds and Data-slice behavior support the original limited positive assessment. |
| [SER-08](#ser-08) | P2 | code quality | confirmed-source | Decoded INT8 storage, quantization parameters and TopK result arrays require invariant validation. |
| [TOOL-01](#tool-01) | P1 | code quality | confirmed-reproduced | Strict benchmark correctness accepts nonfinite errors, mismatched output lengths and missing validation. |
| [TOOL-02](#tool-02) | P2 | code quality | confirmed-reproduced | Requested benchmark output can fail to write while the CLI exits successfully. |
| [TOOL-03](#tool-03) | P2 | code quality | confirmed-reproduced | Benchmark metadata claims Release/C/size settings that do not establish the binary executed. |
| [TOOL-04](#tool-04) | P2 | code quality | confirmed-source | Extended mixed-precision tests bypass the strict-performance opt-out. |
| [TOOL-05](#tool-05) | P2 | code quality | confirmed-reproduced | Selecting optimized-only dot cases runs zero cases and succeeds. |
| [TOOL-06](#tool-06) | P2 | code quality | confirmed-reproduced | Fused/pre-normalized cosine A/B pairs cannot join under their current grouping keys. |
| [TOOL-07](#tool-07) | P2 | performance | qualified | Several timings lack consistent result consumption and compare different setup/allocation boundaries. |
| [TOOL-08](#tool-08) | P3 | code quality | confirmed-reproduced | Memory-footprint output counts owning struct handles rather than heap payload. |
| [TOOL-09](#tool-09) | P2 | code quality | confirmed-source | Matrix benchmark ignores shared selection/timing controls and has only 17 distinct generated vectors. |
| [TOOL-10](#tool-10) | P2 | architecture | qualified | Current benchmark coverage does not establish downstream large-scale behavior. |
| [TOOL-11](#tool-11) | P2 | code quality | confirmed-source | CI coverage excludes important tooling and selected safety surfaces. |
| [TOOL-12](#tool-12) | P2 | code quality | confirmed-source | Baseline comparison ignores provenance and CSV metadata does not escape multiline/arbitrary values. |
| [TOOL-13](#tool-13) | P3 | architecture | unverified | Mobile availability of the separate benchmarking product remains unverified. |

## API-01

**Two search facades replace unsupported metric witnesses with Euclidean**

**P1 · architecture · confirmed-reproduced · finding**

**Exposure:** Public TopKSelection.nearest/batchNearest and default collection search; Operations respects requested metric

**Original references:** A4, search_topk:S1, downstream_interfaces:IF1

**Sources:** [Sources/VectorCore/Operations/TopKSelection.swift:310](../../../Sources/VectorCore/Operations/TopKSelection.swift:310); [Sources/VectorCore/Integration/VectorCollection.swift:120](../../../Sources/VectorCore/Integration/VectorCollection.swift:120)

**Evidence:** Fresh Release search_probe: direct Chebyshev [2.5,2], TopK index0, Operations index1. interfaces_probe proves wrong collection score 2.828427 instead of2. Inputs have equal shape, k1, finite ordinary values. Source default branch applies to Chebyshev/Hamming/Minkowski/custom witnesses.

**Correction / limit:** Collection probe itself shows wrong score, not rank reversal; source proves reversal for search_probe candidate set. Not a downstream VectorIndex implementation bug.

**Improvement:** Use actual Float metric witness; changing signatures requires compatibility decision. Expanding allowlist is incomplete.

**Acceptance:** Shared facade table for all built-ins, nondefault Minkowski, and custom reverse-order metric; score and member equality versus direct witness. Preserve provider route.

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-02

**Batch configuration storage is disconnected from operation settings**

**P2 · architecture · confirmed-reproduced · finding**

**Exposure:** Public BatchOperations; matrix flags alone use live settings

**Original references:** A5, execution_concurrency:EC-3

**Sources:** [Sources/VectorCore/Operations/BatchOperations.swift:99](../../../Sources/VectorCore/Operations/BatchOperations.swift:99); [Sources/VectorCore/Operations/BatchOperations.swift:175](../../../Sources/VectorCore/Operations/BatchOperations.swift:175); [Sources/VectorCore/Operations/BatchOperations.swift:618](../../../Sources/VectorCore/Operations/BatchOperations.swift:618)

**Evidence:** Fresh Release ExecutionPublicProbe stores defaultBatchSize2, processes5, observes [5]. Consumers read immutable defaultConfig for batch/parallel/chunk fields; pairwise routing reads live config.

**Correction / limit:** findNearest uses ParallelHeuristic, not configurable general parallelThreshold. Do not imply every knob governs every API.

**Improvement:** Define effective setting scope and take one validated snapshot at operation entry. Inspect newly activated arithmetic first.

**Acceptance:** Behavior tests per setting; reject zero/nonfinite/negative/huge oversubscription, invalid batch/chunk sizes before division, conversion, shift or addition.

**Dependencies:** API-03, API-07, API-08

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-03

**Configuration read-modify-write loses concurrent updates**

**P2 · architecture · confirmed-reproduced · finding**

**Exposure:** Public updateConfiguration

**Original references:** A5, execution_concurrency:EC-4

**Sources:** [Sources/VectorCore/Operations/BatchOperations.swift:99](../../../Sources/VectorCore/Operations/BatchOperations.swift:99); [Sources/VectorCore/Configuration/ThreadSafeConfiguration.swift:9](../../../Sources/VectorCore/Configuration/ThreadSafeConfiguration.swift:9)

**Evidence:** Fresh Release 100 structured increments result29. Two separate actor awaits permit stale-copy overwrite, independent of exact scheduling.

**Correction / limit:** Actor individual access remains isolated; failure is transaction lost update, not unsynchronized stored memory.

**Improvement:** Run synchronous mutation closure inside one actor method; design Sendable boundary.

**Acceptance:** 100 concurrent increments and independent-field updates retain all changes; Swift6 consumer typecheck.

**Dependencies:** API-02

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-04

**Several schedulers eagerly create all child tasks**

**P2 · architecture · confirmed-reproduced · finding**

**Exposure:** ComputeProvider defaults; public BatchOperations.process/map/filter; public Dim768 cosineSimilarityMatrix and TriangularMatrix.buildParallel; downstream VA inherited batch defaults

**Original references:** A6, execution_concurrency:EC-5, batch_matrix:opportunity5

**Sources:** [Sources/VectorCore/Protocols/ComputeProvider.swift:128](../../../Sources/VectorCore/Protocols/ComputeProvider.swift:128); [Sources/VectorCore/Operations/BatchOperations.swift:183](../../../Sources/VectorCore/Operations/BatchOperations.swift:183); [Sources/VectorCore/Operations/BatchOperations.swift:298](../../../Sources/VectorCore/Operations/BatchOperations.swift:298); [Sources/VectorCore/Operations/BatchOperations.swift:328](../../../Sources/VectorCore/Operations/BatchOperations.swift:328); [Sources/VectorCore/Operations/ExecutionOperations.swift:401](../../../Sources/VectorCore/Operations/ExecutionOperations.swift:401); [Sources/VectorCore/Operations/TransformOperations.swift:164](../../../Sources/VectorCore/Operations/TransformOperations.swift:164); [Sources/VectorCore/Utilities/TriangularMatrix.swift:250](../../../Sources/VectorCore/Utilities/TriangularMatrix.swift:250); [Sources/VectorCore/Protocols/BatchKernelProvider.swift:50](../../../Sources/VectorCore/Protocols/BatchKernelProvider.swift:50); [../future/VectorAccelerate/Sources/VectorAccelerate/Integration/MetalComputeProvider.swift:91](../../../../future/VectorAccelerate/Sources/VectorAccelerate/Integration/MetalComputeProvider.swift:91)

**Evidence:** Fresh Release default conformer maxConcurrency1 reaches64 suspended active bodies. Add-all loops source-proven; Dim768 convenience reaches internal matrix above threshold. 10000*9999/2 =49995000 pair tasks, 4*10000^2=400000000 scalar-output bytes.

**Correction / limit:** Task count/resource exposure, not 64 simultaneous CPU threads or measured slowdown. CPUComputeProvider own bounded chunks/nesting fix is intact. Default reduction is coarsely chunked, not per-item.

**Improvement:** Bound workers/window and define nesting/cancellation/order. All public routes need explicit scheduler policy; actor reentrancy does not bound outstanding GPU work.

**Acceptance:** Instrument active bodies and enqueued tasks for default conformers, public batches, matrix builders, nested routes; verify bound and exact ordered coverage.

**Dependencies:** API-08, API-24

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-05

**Cached raw actor pool allocations are not freed on owner destruction**

**P2 · architecture · confirmed-reproduced · finding**

**Exposure:** Public scoped SwiftBufferPool and TypedBufferPool; internal BufferPool same pattern

**Original references:** A7, execution_concurrency:EC-1

**Sources:** [Sources/VectorCore/Providers/SwiftBufferPool.swift:48](../../../Sources/VectorCore/Providers/SwiftBufferPool.swift:48); [Sources/VectorCore/Operations/BufferPool.swift:174](../../../Sources/VectorCore/Operations/BufferPool.swift:174)

**Evidence:** Fresh optimized exact-source counted-deallocator probe: destroyed weak pool true; cachedBuffersFreed0. SwiftBufferPool stores raw addresses, empty deinit. TypedBufferPool/internal BufferPool similarly lack deinit; source only for latter two.

**Correction / limit:** Proof covers returned cached allocation without active handles. It does not establish active-handle teardown policy, generic element ownership, or leak in separately repaired MemoryPool.

**Improvement:** Introduce clear ownership and cached-allocation teardown for each pool; explicitly define active-handle lifetime policy.

**Acceptance:** Count allocate/free for empty, returned buffers, clear then destruction, destruction without clear for each pool; initialized generic-element lifetime tested separately.

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-06

**SwiftBufferPool age cleanup does not enforce maxPoolSize**

**P2 · architecture · confirmed-reproduced · finding**

**Exposure:** Public SwiftBufferPool opt-in; Operations buffer binding currently unused internally

**Original references:** A7, execution_concurrency:EC-2

**Sources:** [Sources/VectorCore/Providers/SwiftBufferPool.swift:100](../../../Sources/VectorCore/Providers/SwiftBufferPool.swift:100); [Sources/VectorCore/Providers/SwiftBufferPool.swift:164](../../../Sources/VectorCore/Providers/SwiftBufferPool.swift:164)

**Evidence:** Fresh optimized source probe: eight1024-byte returns retained/reused under1024-byte cap, zero frees before clear. Cleanup only evicts older-than60s entries; no idle timer. currentUsageBytes tracks active rounded bytes only.

**Correction / limit:** No universal RSS/time claim; indefinite idle retention is source-derived. maxBufferSize remains enforced on return.

**Improvement:** Track cached bytes and enforce budget independently of age; report active/cached separately.

**Acceptance:** Zero/exact/mixed bucket budgets; concurrent burst; no release leaves cache above cap. Verify live-handle and returned-memory accounting.

**Dependencies:** API-05

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-07

**Nonpositive process batch sizes trap or discard work**

**P2 · code quality · confirmed-reproduced · finding**

**Exposure:** Public throwing BatchOperations.process explicit parameter

**Original references:** C15, execution_concurrency:EC-6, batch_matrix:6

**Sources:** [Sources/VectorCore/Operations/BatchOperations.swift:175](../../../Sources/VectorCore/Operations/BatchOperations.swift:175); [Sources/VectorCore/Operations/BatchOperations.swift:535](../../../Sources/VectorCore/Operations/BatchOperations.swift:535)

**Evidence:** Fresh Release batchSize0 subprocess SIGTRAP; -1 returns[] without transform for one nonempty vector. No stated positivity precondition.

**Correction / limit:** Int positivity is a reasonable semantic expectation, but throwing API currently has neither clear contract nor validation; rank below ordinary positive-input bugs.

**Improvement:** Validate positive batch sizes before either route; clarify empty-input policy.

**Acceptance:** 0,-1,1,>count on N0,1,999,1000; exact-once ordered coverage when valid. Handle oversized positive endpoint addition too.

**Dependencies:** API-02

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-08

**CPU chunk endpoints overflow before clamping valid short ranges**

**P2 · code quality · confirmed-reproduced · finding**

**Exposure:** Public explicitly parallel CPU helpers; default automatic only when gate reached; huge explicit minChunk option

**Original references:** C15, execution_concurrency:EC-7

**Sources:** [Sources/VectorCore/Providers/CPUComputeProvider.swift:128](../../../Sources/VectorCore/Providers/CPUComputeProvider.swift:128); [Sources/VectorCore/Providers/CPUComputeProvider.swift:193](../../../Sources/VectorCore/Providers/CPUComputeProvider.swift:193); [Sources/VectorCore/Providers/CPUComputeProvider.swift:234](../../../Sources/VectorCore/Providers/CPUComputeProvider.swift:234); [Sources/VectorCore/Providers/CPUComputeProvider.swift:276](../../../Sources/VectorCore/Providers/CPUComputeProvider.swift:276); [Sources/VectorCore/Providers/CPUComputeProvider.swift:317](../../../Sources/VectorCore/Providers/CPUComputeProvider.swift:317); [Sources/VectorCore/Protocols/ComputeProvider.swift:178](../../../Sources/VectorCore/Protocols/ComputeProvider.swift:178)

**Evidence:** ExtraProbe fresh Release all five async CPU helper variants SIGTRAP for (Int.max-2)..<Int.max; sequential control returns both indices. minChunk:Int.max with range1..<3 also SIGTRAP. Protocol fallback has overflow-prone hw*2/ceiling/end arithmetic, source only.

**Correction / limit:** No enormous allocation is required for CPU witnesses. Do not claim tiny near-max failure for default reduction without a case that actually overflows its smaller chunk.

**Improvement:** Use remaining-length partitioning; check arithmetic and config conversions as a shared class.

**Acceptance:** All five methods, zero/nonzero/near-max lower bounds, huge hints, exact index coverage/order; test protocol defaults separately. Empty and nonrepresentable Range.count cases need a defined policy.

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-09

**Failed collection update deletes existing value**

**P2 · architecture · confirmed-reproduced · finding**

**Exposure:** Public MutableVectorCollection default; SimpleVectorCollection reference implementation

**Original references:** A8, downstream_interfaces:IF2

**Sources:** [Sources/VectorCore/Integration/VectorCollection.swift:239](../../../Sources/VectorCore/Integration/VectorCollection.swift:239)

**Evidence:** Fresh Release dimension2 id7 updated with dimension1 throws then vector(for:7)==nil. Add rejection is routine documented failure.

**Correction / limit:** SimpleVectorCollection is a small/reference utility, not actual VectorIndex storage. Prechecking dimension alone does not guarantee rollback for arbitrary add failures.

**Improvement:** Require atomic replacement in implementations; validate and replace under SimpleVectorCollection lock.

**Acceptance:** Failure injection for all thrown replacement paths preserves old data; concurrent update/search snapshot policy defined.

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-10

**Explicit/generated IDs overwrite records**

**P2 · architecture · confirmed-reproduced · finding**

**Exposure:** SimpleVectorCollection public insertion APIs

**Original references:** A8, downstream_interfaces:IF3

**Sources:** [Sources/VectorCore/Integration/VectorCollection.swift:213](../../../Sources/VectorCore/Integration/VectorCollection.swift:213); [Sources/VectorCore/Integration/VectorCollection.swift:286](../../../Sources/VectorCore/Integration/VectorCollection.swift:286); [Sources/VectorCore/Integration/VectorCollection.swift:298](../../../Sources/VectorCore/Integration/VectorCollection.swift:298)

**Evidence:** Fresh Release explicit0 then generated add returns0,count1,replacement data. Dictionary overwrite also violates explicit duplicate-ID throws contract.

**Correction / limit:** Reference collection scope; integer exhaustion is design edge, not replayed production failure.

**Improvement:** Reject explicit duplicates and find unused generated IDs atomically; define exhaustion.

**Acceptance:** Explicit/generated permutations, sparse IDs, repeated explicit insertion, remove/reinsert, exhausted allocator strategy. Update must use replacement primitive rather than duplicate rejection.

**Dependencies:** API-09

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-11

**computedScore pole breaks negative-dot relevance**

**P2 · architecture · confirmed-reproduced · finding**

**Exposure:** Public SearchResult fallback; actual collection searchDotProduct producer

**Original references:** A9, downstream_interfaces:IF4

**Sources:** [Sources/VectorCore/Integration/SearchResult.swift:34](../../../Sources/VectorCore/Integration/SearchResult.swift:34); [Sources/VectorCore/Integration/VectorCollection.swift:195](../../../Sources/VectorCore/Integration/VectorCollection.swift:195)

**Evidence:** ExtraProbe actual search query[1], candidates[.5],[1],[2] outputs distances[-2,-1,-.5], computedScores[-1,inf,2]. Finiteness/range/order promises fail with valid equal-shape ordinary vectors.

**Correction / limit:** No reviewed downstream computedScore callers; do not claim observed ANN/GPU regression. There is no universal metric-free transform for all Float scores.

**Improvement:** Carry metric/score semantics or require explicit transform for unsupported distance families.

**Acceptance:** Search-to-score monotonicity, finiteness, documented range across negative-dot pole and nonnegative families; preserve explicitly supplied score.

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-12

**Advertised buffer/global configuration hooks have no active internal consumers**

**P3 · architecture · confirmed-source · opportunity**

**Exposure:** Public reserved-looking bindings and immutable facade

**Original references:** A10, downstream_interfaces:architecture A2

**Sources:** [Sources/VectorCore/Operations/Operations.swift:42](../../../Sources/VectorCore/Operations/Operations.swift:42); [Sources/VectorCore/VectorCore.swift:53](../../../Sources/VectorCore/VectorCore.swift:53)

**Evidence:** rg over Sources finds bufferProvider only declaration; global VectorCore.configuration fields have no operation readers. Actual BatchOperations config is separate.

**Correction / limit:** Provider coexistence is not itself a defect. ArraySIMDProvider and pointer SIMDProvider have different legitimate scopes; API map already discloses static-kernel limitations. Extension-only vector dispatch is owned by vector reviewer.

**Improvement:** Document actual override map; deprecate/reserve unused hooks or connect named allocations with compatibility plan.

**Acceptance:** Spy provider verifies each claimed active hook; public-entry/provider matrix documents real dispatch.

**Dependencies:** API-01, API-02

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-13

**Collection aggregation trusts first row before unsafe reads/forced factories**

**P1 · code quality · confirmed-source · finding**

**Exposure:** Public Operations.centroid/statistics; generic factory boundary

**Original references:** C2, batch_matrix:1

**Sources:** [Sources/VectorCore/Operations/Operations.swift:780](../../../Sources/VectorCore/Operations/Operations.swift:780); [Sources/VectorCore/Operations/Operations.swift:794](../../../Sources/VectorCore/Operations/Operations.swift:794); [Sources/VectorCore/Operations/Operations.swift:811](../../../Sources/VectorCore/Operations/Operations.swift:811); [Sources/VectorCore/Operations/Operations.swift:854](../../../Sources/VectorCore/Operations/Operations.swift:854)

**Evidence:** Independently inspected pointer count=first dimension passed for each shorter toArray allocation; long rows truncated; empty fixed centroid try! create([]); statistics values[i] trusts first length. This second-pass record is source-derived; earlier aggregation observations are retained in the initial review evidence.

**Correction / limit:** No equal-dimension precondition appears at these aggregation entries. Do not transfer MatrixDistance unsafe contracts to them. Pointer overread is source proof; garbage returned by an unsanitized probe is not proof of exact read extent.

**Improvement:** Define empty policy, validate every row, and propagate factory errors. Include Operations.normalize try! generic factory in boundary inventory.

**Acceptance:** Short/long row at first/middle/last, empty dynamic/fixed arrays, failing custom factory; no unsafe operation before all shape checks.

**Dependencies:** OWN-01

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-14

**Zero-query SoA cosine fills oversized output suffix**

**P2 · code quality · confirmed-reproduced · finding**

**Exposure:** Public Safe512 and dimension wrappers using shared internal routes

**Original references:** C10, batch_matrix:3

**Sources:** [Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift:930](../../../Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift:930); [Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift:1025](../../../Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift:1025); [Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift:1126](../../../Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift:1126); [Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift:1737](../../../Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift:1737)

**Evidence:** Fresh Release Safe512 N1,4,4097 output N+2 sentinel -77 becomes1 at both suffix positions. N4097 crosses >8MiB tile gate. Safe wrapper allows capacity>=N. Same generic implementation powers sibling widths.

**Correction / limit:** Writes remain inside provided capacity; this is logical-output overwrite, not out-of-allocation memory corruption. Nonzero branches write N only; no documented requirement to overwrite scratch suffix.

**Improvement:** Restrict zero fill to logical N.

**Acceptance:** Dimension512/768/1536, N0/1/3/4/5, tile255/256/257 and actual byte thresholds; oversized/offset outputs and zero/nonzero query sentinels.

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-15

**Pairwise hard-coded zero diagonal fails built-in negative dot**

**P2 · code quality · confirmed-reproduced · finding**

**Exposure:** Public BatchOperations.pairwiseDistances direct routes

**Original references:** C11, batch_matrix:7

**Sources:** [Sources/VectorCore/Operations/BatchOperations.swift:550](../../../Sources/VectorCore/Operations/BatchOperations.swift:550); [Sources/VectorCore/Operations/BatchOperations.swift:577](../../../Sources/VectorCore/Operations/BatchOperations.swift:577)

**Evidence:** Fresh Release singleton DynamicVector[2], DotProductDistance yields [[0]], direct scalar metric=-4. Source excludes all i==j; DotProductDistance is a supported built-in despite not a mathematical distance metric.

**Correction / limit:** Directed metric N2/N100 discrepancy reproduced but pairwise docs promise symmetric output: treat arbitrary asymmetric support as contract clarification, not required fix. VectorMath pairwise explicitly documents zero diagonal.

**Improvement:** Compute supplied metric diagonals or restrict shortcut to explicit zero-self contract; define symmetry requirements.

**Acceptance:** All built-ins direct-pair oracle around N99/100 and GEMM gates; include zero vectors and dot-product diagonal. Do not assume every cosine route treats zero self as0.

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-16

**cosineSimilarityMatrix returns negative off-diagonal similarity and zero diagonal**

**P2 · code quality · confirmed-reproduced · finding**

**Exposure:** Public Array<Vector<Dim768>> convenience

**Original references:** C11, batch_matrix:7

**Sources:** [Sources/VectorCore/Operations/TransformOperations.swift:164](../../../Sources/VectorCore/Operations/TransformOperations.swift:164); [Sources/VectorCore/Operations/ExecutionOperations.swift:388](../../../Sources/VectorCore/Operations/ExecutionOperations.swift:388)

**Evidence:** ExtraProbe two identical nonzero Dim768 vectors returns [[0,-1.0000001],[-1.0000001,0]], expected approximately1 throughout. Normalizes then passes negative dot to zero-diagonal matrix.

**Correction / limit:** Separate from directed-metric contract. Exact rounded -1.0000001 is numeric noise; sign/diagonal error is order-one.

**Improvement:** Compute cosine similarity with correct sign and self policy; unify matrix contract.

**Acceptance:** Identical, orthogonal, opposite nonzero vectors and defined zero behavior through serial/parallel matrix gates.

**Dependencies:** API-04, API-15

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-17

**Bulk bounded heap membership differs from streaming; negative capacity loops**

**P2 · code quality · confirmed-reproduced · finding**

**Exposure:** Public MinHeap opt-in; no evidence high-level Operations uses bulk constructor

**Original references:** C12, search_topk:S2, search_topk:S4

**Sources:** [Sources/VectorCore/Operations/MinHeap.swift:57](../../../Sources/VectorCore/Operations/MinHeap.swift:57); [Sources/VectorCore/Operations/MinHeap.swift:80](../../../Sources/VectorCore/Operations/MinHeap.swift:80); [Sources/VectorCore/Operations/MinHeap.swift:121](../../../Sources/VectorCore/Operations/MinHeap.swift:121)

**Evidence:** Fresh Release > comparator bulk [4,2],stream[2,1]; opposite < comparator bulk[1,2],stream[3,4]. Negative empty capacity hangs for3s; 0>-1 and empty removeLast no-op prove nontermination.

**Correction / limit:** Implicit nonnegative capacity is reasonable but missing validation; separate P3 invalid-parameter subcase from valid bounded-selection P2. Parent/root ordering does not make heap prefix correct top-k.

**Improvement:** Select retained members before heapify or bounded insert; validate capacity; correct misleading insertion comment.

**Acceptance:** Bulk/stream membership for permutations, comparators, ties, capacity0/1/n/>n; bounded subprocess negative capacity rejection.

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-18

**TopKResult decoder violates parallel-array invariant**

**P2 · code quality · confirmed-reproduced · finding**

**Exposure:** Public Codable from malformed external data

**Original references:** C12, search_topk:S3

**Sources:** [Sources/VectorCore/Operations/TopKSelection.swift:31](../../../Sources/VectorCore/Operations/TopKSelection.swift:31); [Sources/VectorCore/Operations/TopKSelection.swift:45](../../../Sources/VectorCore/Operations/TopKSelection.swift:45); [Sources/VectorCore/Operations/TopKSelection.swift:651](../../../Sources/VectorCore/Operations/TopKSelection.swift:651); [Sources/VectorCore/Integration/SearchResult.swift:168](../../../Sources/VectorCore/Integration/SearchResult.swift:168)

**Evidence:** Fresh Release JSON indices[0],distances[] decodes,count1,tuples0. Subsequent index0 traps by direct empty-array subscript proof; trap not executed.

**Correction / limit:** Malformed serialized input, not valid round-trip bug. Silent zip truncation is consequence of same invariant violation.

**Improvement:** Validate equal array counts before construction and throw DecodingError.

**Acceptance:** Both mismatch directions, empty valid data, normal round-trip and collection access; class-wide serialized invariant audit.

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-19

**Factory batch dimension zero reaches remainder by zero**

**P3 · code quality · confirmed-reproduced · finding**

**Exposure:** Public throwing VectorTypeFactory.batch invalid dimension

**Original references:** C15, vectors:V6

**Sources:** [Sources/VectorCore/Factory/VectorTypeFactory.swift:260](../../../Sources/VectorCore/Factory/VectorTypeFactory.swift:260)

**Evidence:** ExtraProbe empty array,dimension0 SIGTRAP; first executable operation computes values.count%dimension. Public doc specifies multiple of dimension but no positive guard.

**Correction / limit:** A vector dimension normally implies positivity; classify invalid-parameter validation gap rather than ordinary valid-input failure. Not all factory methods trap at zero.

**Improvement:** Check positive dimension and document zero/negative policy before remainder/division.

**Acceptance:** Zero/negative dimension with empty/nonempty values, valid multiples and mismatches; ensure throws instead of trap where documented.

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-20

**PCA oversampling Int.max overflows before bounded sketch clamp**

**P3 · code quality · confirmed-reproduced · finding**

**Exposure:** Public optional extreme PCAConfig; two valid one-dimensional samples

**Original references:** C15, linear_algebra:LA3

**Sources:** [Sources/VectorCore/LinearAlgebra/PCA.swift:144](../../../Sources/VectorCore/LinearAlgebra/PCA.swift:144); [Sources/VectorCore/LinearAlgebra/PCA.swift:179](../../../Sources/VectorCore/LinearAlgebra/PCA.swift:179)

**Evidence:** ExtraProbe two1D rows,components1,oversamplingInt.max SIGTRAP at k+p. Current validation only rejects negative configuration; intended min limit is representable1.

**Correction / limit:** Extreme optional knob, not default PCA numeric failure or enormous input allocation.

**Improvement:** Clamp oversampling against remaining allowed sketch width before addition; define invalid config behavior.

**Acceptance:** 0,normal,Int.max oversampling on tiny data; correct exact fallback or documented error without overflow.

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-21

**FP16 diagnostics enabled flag is read without lock**

**P2 · code quality · confirmed-source · finding**

**Exposure:** Concurrent diagnostics enable/disable with public mixed-precision conversions; opt-in recording

**Original references:** C18, reduced_precision:RP4, execution_concurrency:diagnostics

**Sources:** [Sources/VectorCore/Operations/Kernels/MixedPrecisionDiagnostics.swift:62](../../../Sources/VectorCore/Operations/Kernels/MixedPrecisionDiagnostics.swift:62); [Sources/VectorCore/Operations/Kernels/MixedPrecisionDiagnostics.swift:103](../../../Sources/VectorCore/Operations/Kernels/MixedPrecisionDiagnostics.swift:103); [Sources/VectorCore/Operations/Kernels/MixedPrecisionDiagnostics.swift:146](../../../Sources/VectorCore/Operations/Kernels/MixedPrecisionDiagnostics.swift:146); [Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:820](../../../Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:820)

**Evidence:** Writers lock; recordBatchConversion guard reads isEnabled before lock; public FP16 conversion inits call recordConversionsIfEnabled, whose locked preliminary check can be followed by concurrent disable before unlocked guard. recordConversion has same pattern but no current callers found.

**Correction / limit:** Source-proven conflicting access, no TSAN report. Recorders normally unreachable while always disabled; default entire library not racy from this flag.

**Improvement:** Check flag and update under one lock or correctly specified atomic policy.

**Acceptance:** TSAN concurrent enable/disable plus all live conversion inits; stable state/stat count policy. Eliminate both duplicated unlocked guards.

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-22

**Automatic batch GEMM requires full Q*N scores and can change near-neighbor ranking**

**P2 · performance · qualified · opportunity**

**Exposure:** Operations optimized Euclidean/cosine Q>=8,N>=256 unless BatchKernelProvider installed

**Original references:** PERF2, search_topk:L1

**Sources:** [Sources/VectorCore/Operations/Operations.swift:209](../../../Sources/VectorCore/Operations/Operations.swift:209); [Sources/VectorCore/Operations/Operations.swift:232](../../../Sources/VectorCore/Operations/Operations.swift:232); [Docs/Numerical_Behavior.md:69](../../../Docs/Numerical_Behavior.md:69)

**Evidence:** Fresh Release offset1000 plus delta .125/.25/.5/1: one-query exact self index1; eight-query gate index0 tie at0. Full output=4QN bytes: Q10000,N1000000 =>40000000000 bytes; candidate512 packing=2048000000 bytes.

**Correction / limit:** Already explicitly documented full allocation and cancellation/ranking limitation. Not new hidden bug or universally observed across hardware. Sequential provider and BatchOperations matrix setting do not bypass separate gate.

**Improvement:** Expose per-call strategy/scratch budget, tile or prepared views; robust reranking/direct precision option when needed. Tiling alone cannot restore precision; small rerank set may exclude true winner.

**Acceptance:** Memory budget invariants and deterministic exact-neighbor fixture; direct precision mode reproduces direct reference across gate; benchmark crossovers only after design.

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-23

**Small-k array/generic selection retains O(N) temporary distances/pairs**

**P3 · performance · confirmed-source · opportunity**

**Exposure:** Public TopKSelection arrays/nearest, generic Operations and BatchOperations; pointer small-k already streams

**Original references:** PERF5, search_topk:L2, batch_matrix:opportunity3

**Sources:** [Sources/VectorCore/Operations/TopKSelection.swift:151](../../../Sources/VectorCore/Operations/TopKSelection.swift:151); [Sources/VectorCore/Operations/TopKSelection.swift:284](../../../Sources/VectorCore/Operations/TopKSelection.swift:284); [Sources/VectorCore/Operations/TopKSelection.swift:343](../../../Sources/VectorCore/Operations/TopKSelection.swift:343); [Sources/VectorCore/Operations/Operations.swift:127](../../../Sources/VectorCore/Operations/Operations.swift:127); [Sources/VectorCore/Operations/BatchOperations.swift:416](../../../Sources/VectorCore/Operations/BatchOperations.swift:416); [Sources/VectorCore/Operations/BatchOperations.swift:773](../../../Sources/VectorCore/Operations/BatchOperations.swift:773)

**Evidence:** Source materializes complete distances and tuples before selection; ExtraProbe confirms tuple stride16 on target =>16MB per million pair entries plus4MB scores. Batch optimized helpers use candidates.chunked arrays for boundaries then index original candidates.

**Correction / limit:** No measured speedup/RSS; tuple metadata not full copied vector payload in every path. Avoid Int32 count restriction when adopting pointer implementation.

**Improvement:** Stream metric/array values into common Int-indexed top-k; use integer ranges instead of copied chunks.

**Acceptance:** Allocation/peak memory versus N at fixed k; exact membership/order/NaN/ties match shared selector; Release measurement before performance claims.

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-24

**Redundant matrix reshape, validation and oversized partial heaps increase peak work**

**P3 · performance · confirmed-source · opportunity**

**Exposure:** GEMM pairwise, fallback batch search, optimized near-full-k search

**Original references:** PERF6, search_topk:L3, batch_matrix:opportunity2

**Sources:** [Sources/VectorCore/Operations/BatchOperations.swift:269](../../../Sources/VectorCore/Operations/BatchOperations.swift:269); [Sources/VectorCore/Operations/Operations.swift:160](../../../Sources/VectorCore/Operations/Operations.swift:160); [Sources/VectorCore/Operations/Operations.swift:194](../../../Sources/VectorCore/Operations/Operations.swift:194); [Sources/VectorCore/Operations/Operations.swift:265](../../../Sources/VectorCore/Operations/Operations.swift:265); [Sources/VectorCore/Operations/Operations.swift:294](../../../Sources/VectorCore/Operations/Operations.swift:294); [Sources/VectorCore/Operations/Kernels/TopKSelectionKernels.swift:85](../../../Sources/VectorCore/Operations/Kernels/TopKSelectionKernels.swift:85)

**Evidence:** FlatN² scores remain while row arrays built: about8N² score bytes at end before lifetime optimization, plus packs. Batch validates whole input then each findNearest validates candidates. Each chunk allocates k-sized TopKBuffer arrays and merge allocates replacement output.

**Correction / limit:** Source logical live allocations/complexity, not measured peak RSS after optimizer. C*k is upper-model for C concurrent chunks, not all possible schedulers.

**Improvement:** Validated private worker, reusable flat output, per-chunk capacities and near-full-k strategy. Changes must preserve merge invariants.

**Acceptance:** Release allocation profiles; shared membership/order at k nearN and varied chunk sizes; public invalid inputs still validate exactly before unsafe routes.

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-25

**MatrixDistance cannot consume downstream contiguous cache without packing**

**P3 · performance · confirmed-source · opportunity**

**Exposure:** Public MatrixDistance and VectorIndex centroid batch scorer

**Original references:** PERF7, batch_matrix:opportunity1

**Sources:** [Sources/VectorCore/Operations/MatrixDistance.swift:1](../../../Sources/VectorCore/Operations/MatrixDistance.swift:1); [../VectorIndex/Sources/VectorIndex/Kernels/CentroidBatchScore.swift:10](../../../../VectorIndex/Sources/VectorIndex/Kernels/CentroidBatchScore.swift:10); [Docs/Numerical_Behavior.md:71](../../../Docs/Numerical_Behavior.md:71)

**Evidence:** Inspected source and actual consumer: centroid scorer uses direct cblas_sgemm with contiguous centroidsFlat and once-flattened queries; comment rejects mandatory copies. Prepared candidates still own pack and queries are repacked.

**Correction / limit:** Existing unchecked shape/preparation/BLAS contracts are documented; do not recast malformed MatrixDistance inputs as new memory-safety findings. No copy latency measured.

**Improvement:** Consumer-driven checked borrowed/owned matrix view with rows/columns/stride/lifetime and reusable output.

**Acceptance:** Real centroid consumer parity and removed candidate repacking/allocation; invalid stride/shape checks before BLAS; lifetime tests.

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-26

**KNearestHeap retains NaN ahead of numeric candidate**

**P3 · code quality · qualified · finding**

**Exposure:** Public legacy helper; only internal ExecutionOperations production caller

**Original references:** search_topk:legacy public heap

**Sources:** [Sources/VectorCore/Operations/MinHeap.swift:215](../../../Sources/VectorCore/Operations/MinHeap.swift:215)

**Evidence:** Fresh Release k1 insert NaN then0 retainsNaN; bare distance comparator source explains admission.

**Correction / limit:** No documented NaN-last/index tie contract here; not regression of fixed Operations/TopKSelection or their tests.

**Improvement:** Document finite-only policy or consolidate/deprecate; adopt shared comparator only with contract decision.

**Acceptance:** NaN/tie/finite policy tests if retained; route inventory ensures high-level searches stay on intended selector.

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-27

**Schedulers do not promise or enforce prompt cancellation inside CPU chunks**

**P3 · code quality · qualified · opportunity**

**Exposure:** Public schedulers with noncooperative synchronous body

**Original references:** execution_concurrency:other cancellation

**Sources:** [Sources/VectorCore/Providers/CPUComputeProvider.swift:135](../../../Sources/VectorCore/Providers/CPUComputeProvider.swift:135); [Sources/VectorCore/Operations/BatchOperations.swift:183](../../../Sources/VectorCore/Operations/BatchOperations.swift:183); [Sources/VectorCore/Protocols/ComputeProvider.swift:128](../../../Sources/VectorCore/Protocols/ComputeProvider.swift:128)

**Evidence:** Inspected loops lack Task.checkCancellation; throwing task-group sibling cancellation does not force synchronous user body termination.

**Correction / limit:** Contract gap/opportunity, not proof task cancellation is universally ignored; throwing/suspending user closures can cooperate.

**Improvement:** Specify cancellation at chunk/item boundaries and behavior of nonthrowing APIs.

**Acceptance:** Cancel before start/during long chunks; bounded new work and cleanup under chosen contract; user body obligations stated.

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-28

**Public executeBatch autotuning claim contradicts single full-range invocation**

**P3 · code quality · confirmed-source · opportunity**

**Exposure:** Public CPUComputeProvider.executeBatch direct caller

**Original references:** execution_concurrency:executeBatch

**Sources:** [Sources/VectorCore/Providers/CPUComputeProvider.swift:335](../../../Sources/VectorCore/Providers/CPUComputeProvider.swift:335); [Sources/VectorCore/Providers/CPUComputeProvider.swift:356](../../../Sources/VectorCore/Providers/CPUComputeProvider.swift:356); [Sources/VectorCore/Providers/CPUComputeProvider.swift:362](../../../Sources/VectorCore/Providers/CPUComputeProvider.swift:362)

**Evidence:** Function calls kernel once0..<n unconditionally for n>0; private measureProbes unused; explicitThreshold/explicitMinChunk fields stored but not consumed for that promise.

**Correction / limit:** No measured runtime regression. Existing inline comment explicitly admits sequential implementation, contradicting public doc.

**Improvement:** Correct public promise or implement tested scheduling without unsafe shared-pointer writes.

**Acceptance:** Spy range kernel shows documented calls and settings; Swift6 pointer ownership/concurrency validation before enabling parallel path.

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-29

**AutoTuning lock covers arbitrary calibrator and serializes unrelated keys**

**P2 · code quality · confirmed-source · finding**

**Exposure:** Public calibrator; live Operations generic/custom metric calibration unless disabled/cache already populated/cap exhausted

**Original references:** execution_concurrency:AutoTuning

**Sources:** [Sources/VectorCore/Providers/AutoTuning.swift:102](../../../Sources/VectorCore/Providers/AutoTuning.swift:102); [Sources/VectorCore/Providers/AutoTuning.swift:118](../../../Sources/VectorCore/Providers/AutoTuning.swift:118); [Sources/VectorCore/Operations/Operations.swift:414](../../../Sources/VectorCore/Operations/Operations.swift:414)

**Evidence:** NSLock acquired before calibrator(); get/set acquire same nonrecursive lock. Operations calibrator invokes arbitrary metric.distance under it. Reentrant get/set waits on current thread lock. Global20ms cumulative cap is checked before full callback, so a single long callback can exceed it.

**Correction / limit:** Source proof, no deadlock subprocess run. Not all Operations routes calibrate: fused top-k and installed providers may bypass; cached keys bypass callback. No duration benchmark.

**Improvement:** Calibrate outside lock with per-key in-flight protocol; define reentrancy and time-budget semantics.

**Acceptance:** Bounded subprocess custom metric querying tuner plus direct reentrant calibrator; unrelated-key concurrency; no duplicate publication/race; slow callback policy test.

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-30

**ComputeDevice runtime metadata includes placeholders**

**P3 · code quality · confirmed-source · finding**

**Exposure:** Public read-only device queries, not actual automatic GPU discovery

**Original references:** execution_concurrency:ComputeDevice

**Sources:** [Sources/VectorCore/Execution/ComputeDevice.swift:58](../../../Sources/VectorCore/Execution/ComputeDevice.swift:58); [Sources/VectorCore/Execution/ComputeDevice.swift:153](../../../Sources/VectorCore/Execution/ComputeDevice.swift:153); [Sources/VectorCore/Execution/ComputeDevice.swift:231](../../../Sources/VectorCore/Execution/ComputeDevice.swift:231)

**Evidence:** GPU availability checks compile-time Metal presence, ignores index; neural checks processorCount>0; GPU/Neural capabilities hard-coded. mostCapable picks neural while createProvider returns nil for it.

**Correction / limit:** CPU-only Core boundary is documented; placeholders cannot be used as actual device validation. No unsupported hardware runtime probe.

**Improvement:** Mark estimates/unknown explicitly or inject real platform discovery downstream.

**Acceptance:** Invalid GPU indices, no-neural hardware and provider creation consistency under fake discovery backend.

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-31

**Disabled logger eagerly constructs messages; timer uses wall clock**

**P3 · code quality · confirmed-reproduced · finding**

**Exposure:** Public Logger and PerformanceTimer

**Original references:** execution_concurrency:logger timer

**Sources:** [Sources/VectorCore/Logging/Logger.swift:128](../../../Sources/VectorCore/Logging/Logger.swift:128); [Sources/VectorCore/Logging/Logger.swift:179](../../../Sources/VectorCore/Logging/Logger.swift:179); [Sources/VectorCore/Logging/Logger.swift:246](../../../Sources/VectorCore/Logging/Logger.swift:246)

**Evidence:** ExtraProbe disables debug logging then side-effecting message is evaluated once. Source evaluates autoclosure before private filtering. Timer calls CFAbsoluteTimeGetCurrent at start/end.

**Correction / limit:** No timing overhead or clock-jump failure measured; private minimumLevel field locking is sound.

**Improvement:** Filter before evaluating or pass closure through; use monotonic duration clock for elapsed timing.

**Acceptance:** Disabled message spy remains unevaluated; enabled once; controllable timer clock verifies elapsed policy.

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-32

**MemoryPool recomputes pooled bytes using flatMap arrays under lock**

**P3 · performance · confirmed-source · opportunity**

**Exposure:** Repaired public MemoryPool on buffer return

**Original references:** execution_concurrency:MemoryPool contention

**Sources:** [Sources/VectorCore/Utilities/MemoryPool.swift:285](../../../Sources/VectorCore/Utilities/MemoryPool.swift:285); [Sources/VectorCore/Utilities/MemoryPool.swift:309](../../../Sources/VectorCore/Utilities/MemoryPool.swift:309)

**Evidence:** Two flatMap calls and reduction occur inside stateLock.withLock. Byte accounting uses stride and checks available budget correctly.

**Correction / limit:** Performance hypothesis only; not regression of prior retention fix or evidence lock contention dominates.

**Improvement:** Measure representative return-heavy workloads before choosing maintained counter.

**Acceptance:** Release allocation/contention profile; if changed, retained-byte invariant across type/size buckets and teardown preserved.

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-33

**Normalization hints reach real downstream query and cached inverse norm consumers**

**P1 · architecture · confirmed-source · finding**

**Exposure:** VectorIndex typed APIs; VA cosine kernels; Core wrapper defect owned separately

**Original references:** A2, downstream_interfaces:IF5

**Sources:** [../VectorIndex/Sources/VectorIndex/TypedOverloads.swift:102](../../../../VectorIndex/Sources/VectorIndex/TypedOverloads.swift:102); [../VectorIndex/Sources/VectorIndex/TypedOverloads.swift:119](../../../../VectorIndex/Sources/VectorIndex/TypedOverloads.swift:119); [../VectorIndex/Sources/VectorIndex/HNSWIndex.swift:1097](../../../../VectorIndex/Sources/VectorIndex/HNSWIndex.swift:1097); [../future/VectorAccelerate/Sources/VectorAccelerate/Integration/KernelDistanceProviders.swift:189](../../../../future/VectorAccelerate/Sources/VectorAccelerate/Integration/KernelDistanceProviders.swift:189); [../future/VectorAccelerate/Sources/VectorAccelerate/Integration/KernelDistanceProviders.swift:221](../../../../future/VectorAccelerate/Sources/VectorAccelerate/Integration/KernelDistanceProviders.swift:221)

**Evidence:** Fresh source trace: typed HNSW insert supplies1 or reciprocal hint; valid live cache appends supplied positive inverse norm unchanged. Query flags skip normalization. VA both-normalized path computes raw dot then1-dot. Correct hinted [2,0] stale after mutation against[1,0] would yield-1 instead of0 on this branch.

**Correction / limit:** No downstream build/GPU replay. VA AcceleratedVectorIndex captures hints but its doc explicitly says L2 kernel does not yet consume them, so captured metadata alone is not evidence of active L2 corruption. Cached Index damage is conditional on live cache/route.

**Improvement:** Fix wrapper invariant first, then test downstream insertion/query; assess rebuilding caches populated from stale values.

**Acceptance:** Mutate initially correct hints and compare hinted/unhinted insertion+queries, including HNSW live cache and VA cosine, throwing mutations.

**Dependencies:** OWN-03

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-34

**VectorIndex batch query flag comes only from first query**

**P2 · code quality · confirmed-source · finding**

**Exposure:** VectorIndex Flat/IVF/optimized flat typed batchSearch; HNSW handles each query

**Original references:** downstream_interfaces:compatibility heterogeneous hints

**Sources:** [../VectorIndex/Sources/VectorIndex/TypedOverloads.swift:107](../../../../VectorIndex/Sources/VectorIndex/TypedOverloads.swift:107); [../VectorIndex/Sources/VectorIndex/TypedOverloads.swift:183](../../../../VectorIndex/Sources/VectorIndex/TypedOverloads.swift:183); [../VectorIndex/Sources/VectorIndex/TypedOverloads.swift:205](../../../../VectorIndex/Sources/VectorIndex/TypedOverloads.swift:205); [../VectorIndex/Sources/VectorIndex/FlatIndex.swift:131](../../../../VectorIndex/Sources/VectorIndex/FlatIndex.swift:131); [../VectorIndex/Sources/VectorIndex/IVFIndex.swift:538](../../../../VectorIndex/Sources/VectorIndex/IVFIndex.swift:538)

**Evidence:** Legal heterogeneous NormalizationHint array: first true, second false; shared flag skips second query norm. Source consumer uses flag for all query rows. HNSW loops individual hints at160.

**Correction / limit:** Separate downstream defect even after Core A2 fix; no executed index integration test.

**Improvement:** Carry per-query flags or only enable batch optimization when all queries qualify.

**Acceptance:** Mixed normalized/raw hint batches in both orders match individual searches across Flat/IVF/optimized-flat.

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-35

**Provider seam erases Float scalar/score conventions**

**P3 · code quality · qualified · context**

**Exposure:** Public BatchKernelProvider/VectorCollection; VA direct enum distance versus Core distance

**Original references:** downstream_interfaces:architecture A1

**Sources:** [Sources/VectorCore/Protocols/BatchKernelProvider.swift:18](../../../Sources/VectorCore/Protocols/BatchKernelProvider.swift:18); [Sources/VectorCore/Integration/VectorCollection.swift:88](../../../Sources/VectorCore/Integration/VectorCollection.swift:88); [../future/VectorAccelerate/Sources/VectorAccelerate/Integration/MetalComputeProvider.swift:314](../../../../future/VectorAccelerate/Sources/VectorAccelerate/Integration/MetalComputeProvider.swift:314); [../future/VectorAccelerate/Sources/VectorAccelerate/Integration/MetalComputeProvider.swift:333](../../../../future/VectorAccelerate/Sources/VectorAccelerate/Integration/MetalComputeProvider.swift:333); [../future/VectorAccelerate/Sources/VectorAccelerate/Integration/MetalComputeProvider.swift:385](../../../../future/VectorAccelerate/Sources/VectorAccelerate/Integration/MetalComputeProvider.swift:385)

**Evidence:** VA direct enum dot returns raw dot; Core metric returns negative dot. VA bridge explicitly excludes dot GPU route and recovers actual Float witness; no silent bridge sign replacement. Core Operations allows external ordering by documented policy.

**Correction / limit:** Documented deliberate direct-API semantics and provider ordering variation, not proven transparent route bug. Float-constraining seam is design suggestion with compatibility cost.

**Improvement:** Explicit score-kind/scalar contract and provider conformance suite.

**Acceptance:** Direct-vs-bridge sign tests and NaN/tie policy declarations; custom metric witness remains authoritative.

**Dependencies:** API-01, API-11

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-36

**VectorCollection is not actual VectorIndex conformance seam**

**P3 · architecture · confirmed-source · opportunity**

**Exposure:** Core small/reference collection API and local downstream snapshots

**Original references:** A10, downstream_interfaces:architecture A3

**Sources:** [Sources/VectorCore/Integration/VectorCollection.swift:13](../../../Sources/VectorCore/Integration/VectorCollection.swift:13); [../VectorIndex/Sources/VectorIndex/TypedOverloads.swift:89](../../../../VectorIndex/Sources/VectorIndex/TypedOverloads.swift:89)

**Evidence:** rg finds no VectorCollection use/conformance in either sibling Sources. Actual bridge is async typed index overloads sharing vectors/SearchResult.

**Correction / limit:** Future-tense will implement is aspiration, but example presenting HNSW conformance does not compile against this snapshot. No claim production ANN storage belongs in Core.

**Improvement:** Describe current small utility or design consumer-driven async throwing interface with snapshot/mutation contract.

**Acceptance:** Compile actual cross-package integration examples at pinned versions; route map reflects real conformance.

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-37

**BufferProvider example puts await in defer and handles require explicit lifetime policy**

**P3 · code quality · confirmed-source · finding**

**Exposure:** Public raw BufferHandle and BufferProvider docs

**Original references:** downstream_interfaces:architecture A4

**Sources:** [Sources/VectorCore/Protocols/BufferProvider.swift:80](../../../Sources/VectorCore/Protocols/BufferProvider.swift:80); [Sources/VectorCore/Protocols/BufferProvider.swift:97](../../../Sources/VectorCore/Protocols/BufferProvider.swift:97); [../future/VectorAccelerate/Sources/VectorAccelerate/Core/BufferPool.swift:627](../../../../future/VectorAccelerate/Sources/VectorAccelerate/Core/BufferPool.swift:627)

**Evidence:** Doc contains defer { await provider.release(handle) }; async await cannot appear in synchronous defer body. Handle is copied address/id/size, not an owning object or pointee synchronizer.

**Correction / limit:** No demonstrated valid-use UAF; VA requires keeping provider alive while handle active. Ownership/borrow docs elsewhere already state unsafe lifetime duties.

**Improvement:** Provide compiling do/catch async cleanup or scoped helper and document retention/aliasing requirements.

**Acceptance:** Swift6 doc consumer compile; success/throw/cancel each releases once, owner retained through operation. Active-handle policy coordinated with teardown.

**Dependencies:** API-05

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-38

**Public low-level implementation types create support obligations**

**P3 · code quality · qualified · context**

**Exposure:** Exported caches/kernels/providers/platform helpers

**Original references:** downstream_interfaces:architecture A5

**Sources:** [Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:1](../../../Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:1); [Sources/VectorCore/Operations/Kernels/SoAFP16Cache.swift:1](../../../Sources/VectorCore/Operations/Kernels/SoAFP16Cache.swift:1)

**Evidence:** Source contains public types/methods described elsewhere as implementation details; callers can compile against them.

**Correction / limit:** Not correctness bug by visibility alone and no binary ABI stability inferred. Complete symbol graph inventory not performed here.

**Improvement:** Inventory exported symbols and designate supported low-level APIs; intentional hiding needs compatibility decision.

**Acceptance:** Generated public symbol diff and downstream consumer compile tests for supported surface.

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-39

**VA CPU fallbacks and Core unchecked normalization use different finite-range algorithms**

**P2 · code quality · qualified · opportunity**

**Exposure:** Active VA fallback normalization and acceleratedNormalizeUnchecked fallback to Core

**Original references:** downstream_interfaces:normalization parity

**Sources:** [../future/VectorAccelerate/Sources/VectorAccelerate/Operations/StableNormalization.swift:56](../../../../future/VectorAccelerate/Sources/VectorAccelerate/Operations/StableNormalization.swift:56); [../future/VectorAccelerate/Sources/VectorAccelerate/Operations/StableNormalization.swift:103](../../../../future/VectorAccelerate/Sources/VectorAccelerate/Operations/StableNormalization.swift:103); [../future/VectorAccelerate/Sources/VectorAccelerate/Operations/AccelerateFallback.swift:159](../../../../future/VectorAccelerate/Sources/VectorAccelerate/Operations/AccelerateFallback.swift:159); [../future/VectorAccelerate/Sources/VectorAccelerate/Operations/SIMDFallback.swift:260](../../../../future/VectorAccelerate/Sources/VectorAccelerate/Operations/SIMDFallback.swift:260); [../future/VectorAccelerate/Sources/VectorAccelerate/Core/FallbackProvider.swift:195](../../../../future/VectorAccelerate/Sources/VectorAccelerate/Core/FallbackProvider.swift:195); [../future/VectorAccelerate/Sources/VectorAccelerate/Integration/VectorCoreIntegration.swift:189](../../../../future/VectorAccelerate/Sources/VectorAccelerate/Integration/VectorCoreIntegration.swift:189)

**Evidence:** Verified actual calls and VA scaled-domain divide versus Core magnitude reconstruction; numerical reviewer owns finite tiny/huge proof.

**Correction / limit:** Unchecked zero input violates explicit nonzero assertion and is not an additional valid-input defect. GPU parity not tested; avoid universal provider parity claim.

**Improvement:** Reuse explicit stable contract and shared fixtures after Core numeric correction.

**Acceptance:** Exponent-range direction/norm checks through actual Core/VA CPU paths, optional GPU separately; no false true normalization hints.

**Dependencies:** NUM-02, OWN-03

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-40

**Local sibling dependency pins differ from reviewed Core main**

**P3 · code quality · confirmed-source · context**

**Exposure:** Local source snapshots; no compatibility build

**Original references:** downstream_interfaces:compatibility snapshots

**Sources:** [../VectorIndex/Package.swift:20](../../../../VectorIndex/Package.swift:20); [../VectorIndex/Package.resolved:10](../../../../VectorIndex/Package.resolved:10); [../future/VectorAccelerate/Package.swift:97](../../../../future/VectorAccelerate/Package.swift:97); [../future/VectorAccelerate/Package.resolved:20](../../../../future/VectorAccelerate/Package.resolved:20); [Sources/VectorCore/Version.swift:1](../../../Sources/VectorCore/Version.swift:1)

**Evidence:** Index HEAD6f114923d228739c207734685cfde18dd1c6cf60 pins Core0.3.2/ad39ab6f4904838e300b836a42c7376d6e11da26; VA HEAD3cd1442d4f4e092dda4f9bcd5e0878a9f8cad8bb pins0.3.3/fca4b602383589c46b627d8a0de2b6b2a68cd07d. Core reviewed8b02e75 still identifies0.3.3. Index/Core Swift6.0/macOS14+iOS17; VA Swift6.2/macOS26+iOS26/tvOS26/visionOS26, no watchOS.

**Correction / limit:** Normal unreleased development, not version compatibility defect. VA stale comments about unavailable v26 enums contradict manifest, downstream docs only.

**Improvement:** Record exact source/pin/toolchain matrix and correct stale platform comments.

**Acceptance:** Consumer builds against intended release/main separately; do not infer platform lowering from shared protocols.

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-41

**Existing provider tests do not cross GEMM query threshold**

**P3 · code quality · confirmed-source · finding**

**Exposure:** BatchKernelProvider precedence tests

**Original references:** search_topk:provider precedence coverage

**Sources:** [Tests/ComprehensiveTests/BatchKernelProviderTests.swift:82](../../../Tests/ComprehensiveTests/BatchKernelProviderTests.swift:82); [Tests/ComprehensiveTests/BatchKernelProviderTests.swift:108](../../../Tests/ComprehensiveTests/BatchKernelProviderTests.swift:108); [Sources/VectorCore/Operations/Operations.swift:180](../../../Sources/VectorCore/Operations/Operations.swift:180)

**Evidence:** Tests use Q4/Q5 with N300; GEMM requiresQ>=8. Source delegation precedes GEMM, so current behavior is correct but named precedence claim lacks reaching fixture.

**Correction / limit:** Test gap, not provider dispatch bug.

**Improvement:** Raise/add fixture with Q>=8,N>=256 supported optimized metric while spy override installed.

**Acceptance:** Sentinel provider output at actual GEMM gate; default per-query and whole-batch override both exercised.

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-42

**Dormant execution engines duplicate weak validation/reduction semantics**

**P3 · code quality · qualified · opportunity**

**Exposure:** Internal ExecutionOperations/SyncBatchOperations; live transform/matrix conveniences separately identified

**Original references:** batch_matrix:opportunity4

**Sources:** [Sources/VectorCore/Operations/ExecutionOperations.swift:346](../../../Sources/VectorCore/Operations/ExecutionOperations.swift:346); [Sources/VectorCore/Operations/ExecutionOperations.swift:359](../../../Sources/VectorCore/Operations/ExecutionOperations.swift:359); [Sources/VectorCore/Operations/ExecutionOperations.swift:535](../../../Sources/VectorCore/Operations/ExecutionOperations.swift:535); [Sources/VectorCore/Operations/ExecutionOperations.swift:554](../../../Sources/VectorCore/Operations/ExecutionOperations.swift:554); [Sources/VectorCore/Operations/SyncBatchOperations.swift:341](../../../Sources/VectorCore/Operations/SyncBatchOperations.swift:341)

**Evidence:** Internal reduce injects initial per chunk and recursion; validators inspect first row Release or every100th; naive zero-diagonal/heap behavior duplicates public systems. No current public route for these reduce/search methods established.

**Correction / limit:** Do not label all ExecutionOperations dormant: public Array transforms/cosine matrix are live. Nonidentity initial conflicts with identity reduction law in existing ComputeProvider tests, so law/documentation first.

**Improvement:** Consolidate/remove unused engines; rederive live routing before activating any helper.

**Acceptance:** Mechanical caller map and property tests for any retained reachable engine; shape validation exhaustive before unsafe operations and reduction identity/associativity documented.

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-43

**SoA/unrolled/prefetch throughput claims lack supplied benchmark/compiler evidence**

**P3 · performance · unverified · investigation**

**Exposure:** Public comments/helpers and batch kernel docs

**Original references:** batch_matrix:opportunity6, simd_kernels:performance claims

**Sources:** [Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift:115](../../../Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift:115); [Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift:459](../../../Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift:459); [Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift:1343](../../../Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift:1343); [Sources/VectorCore/Optimization/OptimizationAttributes.swift:57](../../../Sources/VectorCore/Optimization/OptimizationAttributes.swift:57)

**Evidence:** Inspected prefetch helper is unused load; Manhattan implementation uses original AoS/scalar abs lanes despite advertised vDSP optimization. No current benchmark evidence for stated1.5–2x or10–15% examined.

**Correction / limit:** Unsupported speedup claims, not demonstrated slowdown; optimizer may remove nominal prefetch, assembly not inspected.

**Improvement:** Remove/qualify performance assertions or tie to reproducible hardware/size/build benchmark; inspect assembly for real prefetch.

**Acceptance:** Release benchmark artifacts and emitted instruction check support each retained claim.

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-44

**unrolledLoop arbitrary stride emits wrong indices; zero widths fail to progress**

**P2 · code quality · confirmed-source · finding**

**Exposure:** Public helper opt-in; no internal/downstream production callers established

**Original references:** simd_kernels:3

**Sources:** [Sources/VectorCore/Optimization/OptimizationAttributes.swift:83](../../../Sources/VectorCore/Optimization/OptimizationAttributes.swift:83); [Sources/VectorCore/Optimization/OptimizationAttributes.swift:108](../../../Sources/VectorCore/Optimization/OptimizationAttributes.swift:108)

**Evidence:** At stride1 body(i+1) always runs, duplicates and includes count; stride>4 emits max4 before advancing stride, skipping interior indices. Width0 loop never advances; negative widths have malformed progression/ranges.

**Correction / limit:** Original exact-source execution exists but was not independently replayed here; source one-step proof is sufficient. Zero/negative width subcase is lower-priority invalid-input validation; positive1/5/8 violates ordinary stride expectation.

**Improvement:** Implement full stride coverage or explicitly constrain supported positive widths; overflow-safe remaining-count conditions.

**Acceptance:** Emitted indices/flattened ranges exactly0..<count for widths1...17, tail sizes, zero/negative rejection and throwing body. Avoid executing huge count merely to test endpoint arithmetic.

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-45

**VectorMath.nearestNeighbors fully sorts while claiming heap complexity**

**P3 · performance · confirmed-source · opportunity**

**Exposure:** Public fixed Vector convenience

**Original references:** PERF5, simd_kernels:6

**Sources:** [Sources/VectorCore/Operations/VectorMath.swift:386](../../../Sources/VectorCore/Operations/VectorMath.swift:386); [Sources/VectorCore/Operations/VectorMath.swift:402](../../../Sources/VectorCore/Operations/VectorMath.swift:402); [Sources/VectorCore/Operations/VectorMath.swift:413](../../../Sources/VectorCore/Operations/VectorMath.swift:413)

**Evidence:** Source computes all scores, sorted comparator then prefix(k). Complexity O(N logN), memoryO(N); k0 still computes/sorts. Negative k hits Array.prefix precondition.

**Correction / limit:** Negative neighbor count has natural nonnegative precondition but missing explicit API documentation; not a new finite k wrong-ranking proof. Bare Float comparator has unspecified NaN/tie policy; do not transplant modern selector law without decision.

**Improvement:** Use shared selection or correct docs and k policy; coordinate NaN/tie behavior.

**Acceptance:** Metric spy k0 policy, k bounds, common order/member tests; Release allocation/time before performance claim.

**Dependencies:** API-23

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-46

**Named optimized metric factory cannot create dotProduct**

**P3 · code quality · confirmed-source · finding**

**Exposure:** Public OptimizedDistanceMetrics.create(named:)

**Original references:** simd_kernels:factory correctness

**Sources:** [Sources/VectorCore/Operations/OptimizedDistanceMetrics.swift:368](../../../Sources/VectorCore/Operations/OptimizedDistanceMetrics.swift:368); [Sources/VectorCore/Operations/OptimizedDistanceMetrics.swift:402](../../../Sources/VectorCore/Operations/OptimizedDistanceMetrics.swift:402)

**Evidence:** MetricType raw value dotProduct has uppercaseP; every input lowercased before rawValue construction; lowercasing cannot yield that enum case. Enum-based create works.

**Correction / limit:** One-step mapping proof, not runtime replay. Separate from search metric witness replacement.

**Improvement:** Normalize enum lookup consistently or explicit accepted-name mapping.

**Acceptance:** Case-insensitive roundtrip for every MetricType.allCases and unknown names returnnil.

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-47

**Singleton full-matrix triangular import reads nonexistent element**

**P2 · code quality · confirmed-reproduced · finding**

**Exposure:** Public TriangularMatrix.fromFullMatrix

**Original references:** C15, batch_matrix:5

**Sources:** [Sources/VectorCore/Utilities/TriangularMatrix.swift:218](../../../Sources/VectorCore/Utilities/TriangularMatrix.swift:218)

**Evidence:** Fresh Release batch_probe singleton [[0]] SIGTRAP. Nonempty and square checks succeed, then initializer reads matrix[0][1]. TriangularMatrix size1 itself permits empty strict upper triangle.

**Correction / limit:** Singleton input satisfies stated square-matrix requirement. Diagonal/lower-triangle omission is deliberate and is not the defect.

**Improvement:** Use existing valid seed or explicit empty storage for singleton; preserve import contract.

**Acceptance:** Empty, singleton,2x2, ragged inputs and singleton roundtrip to zero-diagonal full matrix; no index trap.

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## API-48

**CPU sequential reduce ignores initial but identity is the existing provider law**

**P3 · code quality · qualified · context**

**Exposure:** Public CPUComputeProvider.parallelReduce

**Original references:** execution_concurrency:strengths reduce identity

**Sources:** [Sources/VectorCore/Providers/CPUComputeProvider.swift:284](../../../Sources/VectorCore/Providers/CPUComputeProvider.swift:284); [Sources/VectorCore/Protocols/ComputeProvider.swift:78](../../../Sources/VectorCore/Protocols/ComputeProvider.swift:78)

**Evidence:** Sequential branch returns rangeWork(items); parallel combines partials into initial. With identity initial the difference is lawful; original conformance tests intentionally use identity.

**Correction / limit:** Not a new numeric defect from nonidentity input. Current protocol comments should state identity and associativity requirements; floating-point nonassociativity means bitwise mode equivalence is not promised.

**Improvement:** Promote existing reduction law to current public documentation and distinguish sequential-fold APIs.

**Acceptance:** Conformance tests for identity and chunk combination; documented limits for nonassociative floating-point reductions.

**Evidence set:** [execution-api](evidence/validate_execution_api/) (see [replay/provenance index](evidence/README.md)).

## NUM-01

**Scalar division computes an overflowing reciprocal even when each quotient is finite.**

**P2 · code quality · confirmed-reproduced · finding**

**Exposure:** Public vector operators, public Swift array/static providers; Accelerate provider also fails on this host.

**Original references:** C3, vectors V2, simd_kernels finding 4

**Sources:** [Sources/VectorCore/Protocols/VectorProtocol.swift:264](../../../Sources/VectorCore/Protocols/VectorProtocol.swift:264); [Sources/VectorCore/Platform/ArraySIMDProvider.swift:181](../../../Sources/VectorCore/Platform/ArraySIMDProvider.swift:181); [Sources/VectorCore/Platform/ArraySIMDProvider.swift:639](../../../Sources/VectorCore/Platform/ArraySIMDProvider.swift:639); [Sources/VectorCore/Platform/SwiftFloatSIMDProvider.swift:358](../../../Sources/VectorCore/Platform/SwiftFloatSIMDProvider.swift:358); [Sources/VectorCore/Platform/SIMDProvider.swift:210](../../../Sources/VectorCore/Platform/SIMDProvider.swift:210); [Sources/VectorCore/Platform/AccelerateSIMDProvider.swift:95](../../../Sources/VectorCore/Platform/AccelerateSIMDProvider.swift:95)

**Evidence:** Fresh Release NumericProbe results.log: DynamicVector [s,0]/s and AccelerateFloatProvider.divideByScalar return [inf,nan] at s=1e-40. Independent DivisionProbe gives direct inPlaceDivide=[1,0], /= operator=[inf,nan], scalar s/s=1. Original Double probe source reviewed; Double reciprocal-overflow mechanism source-confirmed.

**Correction / limit:** Do not assume vDSP_vsdiv repairs the problem: it fails on this Apple host. Direct inPlaceDivide and /= are distinct observed routes.

**Improvement:** Use actual per-element division or a tested safe fallback when reciprocal cannot represent the operation; preserve signed-zero and nonfinite policies.

**Acceptance:** Provider/type/operator conformance matrix Float/Double, normal/subnormal divisors, zero numerators and tails; compare representable quotients to widened scalar reference; test /= separately from inPlaceDivide.

**Evidence set:** [numerics](evidence/validate_numerics/) (see [replay/provenance index](evidence/README.md)).

## NUM-02

**Normalization reconstructs a mismatched scale or relies on an unrepresentable norm/reciprocal.**

**P2 · code quality · confirmed-reproduced · finding**

**Exposure:** Public generic checked normalization and public pointer path; concrete optimized helpers have additional scale limits.

**Original references:** C3, vectors V2, simd_kernels finding 1

**Sources:** [Sources/VectorCore/Protocols/VectorProtocol.swift:402](../../../Sources/VectorCore/Protocols/VectorProtocol.swift:402); [Sources/VectorCore/Operations/Kernels/NormalizeKernels.swift:453](../../../Sources/VectorCore/Operations/Kernels/NormalizeKernels.swift:453); [Sources/VectorCore/Operations/Kernels/NormalizeKernels.swift:483](../../../Sources/VectorCore/Operations/Kernels/NormalizeKernels.swift:483); [Sources/VectorCore/Operations/Kernels/NormalizeKernels.swift:276](../../../Sources/VectorCore/Operations/Kernels/NormalizeKernels.swift:276)

**Evidence:** Fresh Release NumericProbe results.log: 1e-38 repeated over 1/4/5/17/512 has output Double norm 1.1754944..1.1754947. greatestFinite repeated over 4/5/17/512 becomes all zero (dimension1 approximately unit). Generic checked DynamicVector [1e-40,0] reports success with inf/nan. For clamped denominator den, current reconstructed mag=trueNorm*maxAbs/den, so output norm=den/maxAbs.

**Correction / limit:** Representability of the full norm or its reciprocal is unnecessary for a finite unit direction. Deep-subnormal unchanged behavior is currently an explicit implementation policy, not proof that normalization is mathematically impossible.

**Improvement:** Normalize in a scaled domain and state failure/unchanged policy; do not just replace maxAbs with den and retain reciprocal dependence. Preserve sign/direction while supporting finite norms outside Float range.

**Acceptance:** Across typed/generic/pointer paths and widths/tails, assert unit norm in Double, positive alignment with original direction, zero/nonfinite policy, and unchanged-on-failure if promised. Include least subnormal,1e-38,maxFinite.

**Dependencies:** NUM-01, OWN-08

**Evidence set:** [numerics](evidence/validate_numerics/) (see [replay/provenance index](evidence/README.md)).

## NUM-03

**QuantizationSchemes clamps its zero point so positive/negative-only and constant intervals cannot be represented.**

**P2 · code quality · confirmed-reproduced · finding**

**Exposure:** Public opt-in QuantizationSchemes calibration.

**Original references:** C6, reduced_precision finding 2

**Sources:** [Sources/VectorCore/Quantization/QuantizationSchemes.swift:96](../../../Sources/VectorCore/Quantization/QuantizationSchemes.swift:96); [Sources/VectorCore/Quantization/QuantizationSchemes.swift:104](../../../Sources/VectorCore/Quantization/QuantizationSchemes.swift:104); [Sources/VectorCore/Quantization/QuantizationSchemes.swift:130](../../../Sources/VectorCore/Quantization/QuantizationSchemes.swift:130)

**Evidence:** Fresh Release NumericProbe results.log: [1,2] reconstructs [1,1], [-10,-1] reconstructs [-9,-0.98823535], constant1.5 reconstructs ~3.57e-43 using the source documented decode (q-offset)*scale.

**Correction / limit:** Independent decode is legitimate: same formula appears in analyzeQuantizationError. LinearQuantizationParams is a separate public surface and its existing tests do not establish this one.

**Improvement:** Consolidate affine calibration, include zero before constraining zero point or retain adequate bias, and define constants. Merely widening zeroPoint to Int32 is not a universal finite-range solution.

**Acceptance:** Both calibration APIs across signed/one-sided/cross-zero/constant intervals must cover endpoints within a defined quantization step; scalar/SIMD/decoded construction agree.

**Dependencies:** NUM-04

**Evidence set:** [numerics](evidence/validate_numerics/) (see [replay/provenance index](evidence/README.md)).

## NUM-04

**Clamping integer conversions trap before the clamp for finite large intermediates.**

**P2 · code quality · confirmed-source · finding**

**Exposure:** Public scalar quantizer and asymmetric parameter construction.

**Original references:** C6, reduced_precision finding 3

**Sources:** [Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:46](../../../Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:46); [Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:54](../../../Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:54)

**Evidence:** Original executable crash sources inspected: calibrated [-1,1].quantize(1e20) evaluates Int(scaledValue) outside Int range; asymmetric constant1e20 evaluates Int(-128-1e20). Int8/Int32(clamping:) is outermost and never reached. No fresh crash run by this reviewer.

**Correction / limit:** Keep finite outliers distinct from NaN/Inf policy and from intentionally unsafe raw buffers. An Int32 clamp avoids traps but can still destroy representation for high offsets.

**Improvement:** Clamp/validate floating intermediates before integer conversion; define nonfinite and unrepresentable-parameter policy.

**Acceptance:** Mechanical Float-to-integer conversion matrix: finite extremes/constants/outliers, NaN/Inf, scalar versus SIMD, decoded params; no process termination for supported inputs and explicit error/fallback otherwise.

**Evidence set:** [numerics](evidence/validate_numerics/) (see [replay/provenance index](evidence/README.md)).

## NUM-05

**Quantized cosine overflows the product of finite squared norms.**

**P2 · code quality · confirmed-reproduced · finding**

**Exposure:** Public INT8 methods at widths512/768/1536, opt-in.

**Original references:** C7, reduced_precision finding 5

**Sources:** [Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:754](../../../Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:754); [Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:793](../../../Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:793); [Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:825](../../../Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:825)

**Evidence:** Fresh Release NumericProbe results.log: all three widths at repeating1e10 return self-distance1; repeating1 and1e-10 return0. Finite nonzero decoded vector has self cosine distance0 by definition.

**Correction / limit:** Not quantization noise. sqrt(qMagSq*cMagSq) overflows even with finite individual squared norms. Fixing only square-root order still leaves scale-squared/product underflow/overflow elsewhere.

**Improvement:** Cancel positive symmetric scales and evaluate bounded integer cosine with sufficiently wide arithmetic; keep asymmetric fallback and zero policy explicit.

**Acceptance:** Self/opposite/orthogonal scale-family matrix all widths, asymmetric and unequal-scale cases, zero/nonfinite policy; compare to Double decoded-vector reference.

**Evidence set:** [numerics](evidence/validate_numerics/) (see [replay/provenance index](evidence/README.md)).

## NUM-06

**Batch magnitude variance uses cancellation-prone raw moments above N=1000.**

**P2 · code quality · confirmed-reproduced · finding**

**Exposure:** Public BatchOperations.statistics; serial route is also inaccurate for large repeated values.

**Original references:** C8, batch_matrix finding 2

**Sources:** [Sources/VectorCore/Operations/BatchOperations.swift:357](../../../Sources/VectorCore/Operations/BatchOperations.swift:357); [Sources/VectorCore/Operations/BatchOperations.swift:367](../../../Sources/VectorCore/Operations/BatchOperations.swift:367); [Sources/VectorCore/Operations/BatchOperations.swift:392](../../../Sources/VectorCore/Operations/BatchOperations.swift:392); [Sources/VectorCore/Operations/BatchOperations.swift:604](../../../Sources/VectorCore/Operations/BatchOperations.swift:604)

**Evidence:** Fresh Release NumericProbe results.log: alternating10000/10001: std0.49999908 at999 and0 at1000/1024. Repeated Float(1e18): serial std6.25e12; parallel inf. For balanced samples independent population variance=1/4; for odd999, sqrt(499*500)/999.

**Correction / limit:** Statistics concerns magnitudes, not component covariance. At1e18 adding1 rounds away, so those inputs really are identical. Merely replacing negative variance clamp does not repair cancellation or squared-sum overflow.

**Improvement:** Use stable widened mean/M2 for both serial and parallel, with stable merges; define nonfinite magnitude behavior.

**Acceptance:** Both sides of999/1000 and multiple chunk counts, shuffled order, constants and large offsets; compare Double population standard deviation and assert finite results where reference is representable.

**Evidence set:** [numerics](evidence/validate_numerics/) (see [replay/provenance index](evidence/README.md)).

## NUM-07

**Entropy cutoff and Float sum violate finite scale invariance.**

**P2 · code quality · confirmed-reproduced · finding**

**Exposure:** Public Vector entropy facade, small and large routes.

**Original references:** C9, batch_matrix finding 4

**Sources:** [Sources/VectorCore/Operations/VectorEntropy.swift:42](../../../Sources/VectorCore/Operations/VectorEntropy.swift:42); [Sources/VectorCore/Operations/VectorEntropy.swift:81](../../../Sources/VectorCore/Operations/VectorEntropy.swift:81); [Sources/VectorCore/Operations/VectorMath.swift:488](../../../Sources/VectorCore/Operations/VectorMath.swift:488)

**Evidence:** Fresh Release NumericProbe results.log: uniform1e-10 returns0 at Dim4/128; unit inputs yield log4/log128; maxFinite yields0/NaN. Independent probabilities are1/d at every nonzero finite common scale.

**Correction / limit:** Existing tiny-probability cutoff beside a dominant1 does not justify the absolute absSum cutoff on uniform vectors.

**Improvement:** Scale by max absolute component or sum in Double; distinguish all-zero from small nonzero. Keep tiny normalized-probability policy separate.

**Acceptance:** Scale sweeps over finite exponent range and zero-padding across64; compare normalized Double entropy, include all-zero/nonfinite and existing tiny-probability tests.

**Evidence set:** [numerics](evidence/validate_numerics/) (see [replay/provenance index](evidence/README.md)).

## NUM-08

**Swift SVD accepts nonorthogonal columns as converged at harmless input scales.**

**P2 · code quality · confirmed-reproduced · finding**

**Exposure:** Public SwiftLinearAlgebraProvider; PCA when override selects it, default on unsupported LAPACK platforms.

**Original references:** C13, linear_algebra LA-1

**Sources:** [Sources/VectorCore/LinearAlgebra/SwiftLinearAlgebraProvider.swift:231](../../../Sources/VectorCore/LinearAlgebra/SwiftLinearAlgebraProvider.swift:231); [Sources/VectorCore/LinearAlgebra/SwiftLinearAlgebraProvider.swift:238](../../../Sources/VectorCore/LinearAlgebra/SwiftLinearAlgebraProvider.swift:238)

**Evidence:** Fresh Release NumericProbe results.log: s*[1,0,1,1] returns normalized singular spectrum[sqrt2,1] and U correlation0.7071 at1e-12/1e15; unit scale returns golden-ratio spectrum. Exact A-transpose-A eigenvalues=(3±sqrt5)/2.

**Correction / limit:** Full rank and condition number2.618: documented null-space-basis limitation is irrelevant. Reconstruction alone can pass with invalid SVD factors.

**Improvement:** Widen or scale correlation tests and rotations; repair final column norm too. A wider comparison alone does not close normalization or reflector range failures.

**Acceptance:** Scale-equivariant spectra, orthogonality and normalized residuals for full-rank rectangular matrices; output finiteness; isolate rank-deficiency exception.

**Dependencies:** NUM-09

**Evidence set:** [numerics](evidence/validate_numerics/) (see [replay/provenance index](evidence/README.md)).

## NUM-09

**Swift QR and SVD final normalization lose representable finite vectors through Float squared norms.**

**P2 · code quality · confirmed-reproduced · finding**

**Exposure:** Public Swift fallback and PCA using its QR.

**Original references:** C13, linear_algebra LA-2

**Sources:** [Sources/VectorCore/LinearAlgebra/SwiftLinearAlgebraProvider.swift:46](../../../Sources/VectorCore/LinearAlgebra/SwiftLinearAlgebraProvider.swift:46); [Sources/VectorCore/LinearAlgebra/SwiftLinearAlgebraProvider.swift:53](../../../Sources/VectorCore/LinearAlgebra/SwiftLinearAlgebraProvider.swift:53); [Sources/VectorCore/LinearAlgebra/SwiftLinearAlgebraProvider.swift:268](../../../Sources/VectorCore/LinearAlgebra/SwiftLinearAlgebraProvider.swift:268)

**Evidence:** Fresh Release NumericProbe results.log: QR [s,s] at1e-25 returns Q[1,0],R[s]; at1e20 Q NaN,R-inf. Fresh single-column SVD at1e-25 gives singular0 and tiny unnormalized U; at1e20 singularinf and zero U. Exact norm=sqrt2*abs(s) is representable.

**Correction / limit:** Widened norm accumulation immediately narrowed into unsafe reciprocal/alpha-beta arithmetic is insufficient for full exponent range.

**Improvement:** Scaled/wide Householder and normalization, explicit handling when outputs genuinely cannot be represented.

**Acceptance:** Shared exponent sweep QR/SVD/eigen with normalized residual, orthogonality, spectrum, finiteness; include single-column and near-extreme finite vectors.

**Evidence set:** [numerics](evidence/validate_numerics/) (see [replay/provenance index](evidence/README.md)).

## NUM-10

**Rounding PCA mean before centering doubles variance in a valid offset fixture.**

**P3 · code quality · qualified · opportunity**

**Exposure:** Default LAPACK-backed centered PCA, large-offset near-constant data.

**Original references:** README lower-priority PCA centering, linear_algebra LA-4

**Sources:** [Sources/VectorCore/LinearAlgebra/PCA.swift:160](../../../Sources/VectorCore/LinearAlgebra/PCA.swift:160); [Sources/VectorCore/LinearAlgebra/PCA.swift:257](../../../Sources/VectorCore/LinearAlgebra/PCA.swift:257); [Sources/VectorCore/LinearAlgebra/PCA.swift:291](../../../Sources/VectorCore/LinearAlgebra/PCA.swift:291)

**Evidence:** Fresh Release NumericProbe results.log: [0,2] produces variance~2, projections[-1,1]; translated exactly representable Float[16777216,16777218] produces mean16777216, variance4, projections[0,2]. Exact mean16777217 and sample variance2.

**Correction / limit:** Numerical-quality improvement under broad documented conditioning limitations, not universal relative-accuracy violation. Error term is n/(n-1)*deltaMean*deltaMean^T before further rounding.

**Improvement:** Retain Double center or Float anchor plus residual mean and apply same centering in fit and both transforms.

**Acceptance:** Translation/offset invariance with representable input/residual fixtures; compare Double covariance and projected sample mean; no fit/transform center mismatch.

**Evidence set:** [numerics](evidence/validate_numerics/) (see [replay/provenance index](evidence/README.md)).

## NUM-11

**UMAP output curve fit depends on units and silently returns its initial guess.**

**P2 · code quality · confirmed-reproduced · finding**

**Exposure:** Public valid minDist/spread configurations through internal fitter.

**Original references:** C14, umap finding 1

**Sources:** [Sources/VectorCore/ManifoldLearning/UMAP.swift:314](../../../Sources/VectorCore/ManifoldLearning/UMAP.swift:314); [Sources/VectorCore/ManifoldLearning/UMAP.swift:350](../../../Sources/VectorCore/ManifoldLearning/UMAP.swift:350); [Sources/VectorCore/ManifoldLearning/UMAP.swift:372](../../../Sources/VectorCore/ManifoldLearning/UMAP.swift:372)

**Evidence:** Fresh Release NumericProbe results.log: fixed minDist/spread0.1: kernel(spread)0.3880566 at1 and0.001; ~1 at1e-5 and1e-20 at1e10. Change variable x=spread*t gives same dimensionless fitting problem with A=a*spread^(2b).

**Correction / limit:** Claim concerns kernel calibration, not stochastic layout-scale invariance: learning rate, gradient clipping and init need not rescale the whole layout.

**Improvement:** Fit dimensionless parameters, detect failed fit, and handle coefficient representability via log/normalized evaluation where needed.

**Acceptance:** Assert b and dimensionless fitted curve/SSE invariant over scale families and check convergence; explicit failure outside supported range. Do not assert entire SGD layout is scale-equivariant.

**Evidence set:** [numerics](evidence/validate_numerics/) (see [replay/provenance index](evidence/README.md)).

## NUM-12

**Smooth-kNN bandwidth search starts at absolute1 and fails outside its fixed64-step range.**

**P2 · code quality · confirmed-reproduced · finding**

**Exposure:** Public graph-only UMAP input with finite nonnegative distances.

**Original references:** C14, umap finding 2

**Sources:** [Sources/VectorCore/ManifoldLearning/FuzzySimplicialSet.swift:65](../../../Sources/VectorCore/ManifoldLearning/FuzzySimplicialSet.swift:65); [Sources/VectorCore/ManifoldLearning/FuzzySimplicialSet.swift:87](../../../Sources/VectorCore/ManifoldLearning/FuzzySimplicialSet.swift:87)

**Evidence:** Fresh Release NumericProbe results.log: valid row distances[s,2s] produce nonnearest weight0.584969 at1, weight1 at1e-30, and lost pair at1e30. Equation1+exp(-s/sigma)=log2(3) gives sigma=-s/log(log2(3)-1), ~1.86495s.

**Correction / limit:** Use solvable rows for equation assertions; duplicates/tied minimum can make target unattainable and are already handled by floor. Float sigma can itself become unrepresentable at extreme inputs even after bracket fix.

**Improvement:** Scale row distances or starting bracket; verify residual/convergence and preserve explicit floor behavior.

**Acceptance:** Multiplicative-scale tests with unequal degrees/isolated/duplicate rows; assert weights/topology and equation residual where solvable, with supported-range policy for sigma overflow.

**Evidence set:** [numerics](evidence/validate_numerics/) (see [replay/provenance index](evidence/README.md)).

## NUM-13

**UMAP accepts infinite learning rate/spread at an existing throwing validation boundary.**

**P2 · code quality · confirmed-reproduced · finding**

**Exposure:** Public UMAPConfig used by both overloads.

**Original references:** C14, umap finding 3

**Sources:** [Sources/VectorCore/ManifoldLearning/UMAP.swift:102](../../../Sources/VectorCore/ManifoldLearning/UMAP.swift:102); [Sources/VectorCore/ManifoldLearning/UMAP.swift:113](../../../Sources/VectorCore/ManifoldLearning/UMAP.swift:113)

**Evidence:** Fresh Release NumericProbe results.log: valid3-point graph and learningRate infinity returns [nan,nan,nan,inf,inf,inf]. Comparisons admit positive infinite spread and minDist=spread=infinity by source.

**Correction / limit:** NaN generally fails current comparisons. Finiteness checks are necessary but cannot promise every huge finite learning rate yields a finite converged layout.

**Improvement:** Reject nonfinite tuning fields; document supported ranges or output validation for large finite options.

**Acceptance:** Table of NaN/±Inf per floating option across both overloads must throw; normal configs remain deterministic; include finite extremes to characterize policy.

**Evidence set:** [numerics](evidence/validate_numerics/) (see [replay/provenance index](evidence/README.md)).

## NUM-14

**Approximate equality accepts NaN differences as equal.**

**P2 · code quality · confirmed-reproduced · finding**

**Exposure:** Public VectorProtocol approximate equality; PrecisionValidator is internal dormant helper.

**Original references:** C16, vectors V7, reduced_precision internal validator

**Sources:** [Sources/VectorCore/Protocols/VectorProtocol.swift:550](../../../Sources/VectorCore/Protocols/VectorProtocol.swift:550); [Sources/VectorCore/Operations/Kernels/MixedPrecisionValidation.swift:262](../../../Sources/VectorCore/Operations/Kernels/MixedPrecisionValidation.swift:262); [Sources/VectorCore/Operations/Kernels/MixedPrecisionValidation.swift:278](../../../Sources/VectorCore/Operations/Kernels/MixedPrecisionValidation.swift:278); [Sources/VectorCore/Operations/Kernels/MixedPrecisionValidation.swift:292](../../../Sources/VectorCore/Operations/Kernels/MixedPrecisionValidation.swift:292)

**Evidence:** Fresh Release NumericProbe results.log: NaN versus42 returns true; internal512 NaN-versus-zero validator true. Same-sign infinity self equality true because differenceNaN. Only diff>tolerance rejects.

**Correction / limit:** Matching infinities could be an intentional equality policy; NaN-versus-finite acceptance cannot be justified by tolerance. Negative/NaN/infinite tolerances also need explicit policy.

**Improvement:** Define shared finite/nonfinite and tolerance rules, then fail closed on undefined differences.

**Acceptance:** Mechanical helpers matrix finite/NaN/±Inf/signedzero and invalid tolerances across dimensions; distances and vectors use consistent declared rules.

**Evidence set:** [numerics](evidence/validate_numerics/) (see [replay/provenance index](evidence/README.md)).

## NUM-15

**safeLog bypasses options for NaN and positive infinity.**

**P2 · code quality · qualified · finding**

**Exposure:** Public Vector safeLog; narrower method docs conflict with general options.

**Original references:** C16, simd_kernels finding 5

**Sources:** [Sources/VectorCore/Operations/NaNInfinityHandling.swift:30](../../../Sources/VectorCore/Operations/NaNInfinityHandling.swift:30); [Sources/VectorCore/Operations/NaNInfinityHandling.swift:383](../../../Sources/VectorCore/Operations/NaNInfinityHandling.swift:383)

**Evidence:** Fresh Release NumericProbe results.log: strict NaN/+Inf returns nonfinite vector; strict -Inf throws. value<=0 routes are the only ones consulting options.

**Correction / limit:** Behavior reproduced, but method-specific options documentation says non-positive values. Classify as inconsistent option semantics requiring resolution, rather than an unqualified promise of finite logarithm for every call.

**Improvement:** Apply selected nonfinite policy separately from log domain, or expose a distinctly named/domain-only policy and clarify docs.

**Acceptance:** All presets on negative/zero/NaN/±Inf/mixed vectors; if strict keeps general meaning, strict success must imply finite result.

**Evidence set:** [numerics](evidence/validate_numerics/) (see [replay/provenance index](evidence/README.md)).

## NUM-16

**Adaptive distance ignores threshold and candidate range.**

**P2 · code quality · confirmed-reproduced · finding**

**Exposure:** Explicit public adaptive API; not default Operations routing.

**Original references:** C19, reduced_precision finding 6

**Sources:** [Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:3388](../../../Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:3388); [Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:3403](../../../Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:3403)

**Evidence:** Fresh Release NumericProbe results.log: threshold0 on ones vs1.0001 returns0 versus direct0.0022631169. Zero-query vs70000 yieldsinf while reversed operands yield1583918.9. Source never reads threshold.

**Correction / limit:** Default docs disclaim universal FP16 score bounds, but ignored threshold is still ineffective API control. Representability checks on both operands alone do not enforce an arbitrary relative error bound.

**Improvement:** Define absolute/relative policy and use FP32 fallback whenever requested tolerance cannot be established; or remove misleading parameter and name heuristic.

**Acceptance:** Operand reversal, high candidate/low query, threshold sweeps and near ties; prove/test conservative selection bounds and compare with defined baseline.

**Evidence set:** [numerics](evidence/validate_numerics/) (see [replay/provenance index](evidence/README.md)).

## NUM-17

**Quantized tuner cache excludes accuracyRequirement, making selection call-history dependent.**

**P2 · code quality · confirmed-reproduced · finding**

**Exposure:** Public opt-in singleton tuner; no automatic production consumer established.

**Original references:** C19, reduced_precision finding 7

**Sources:** [Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:1389](../../../Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:1389); [Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:1401](../../../Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:1401)

**Evidence:** Fresh Release NumericProbe results.log: dot/512/2000 first accuracy0.02=>candidatesINT8, warm0.2=>candidatesINT8, fresh0.2=>fullFP32.

**Correction / limit:** Counterintuitive heuristic direction is not independently a bug until accuracyRequirement units/meaning are specified. Quantized tuner lock protects dictionary access; no race alleged here.

**Improvement:** Include every decision input in key; define accuracy semantics and validate its domain.

**Acceptance:** Change one input at a time, compare warm/fresh answers, exercise call permutations and invalid accuracy values.

**Evidence set:** [numerics](evidence/validate_numerics/) (see [replay/provenance index](evidence/README.md)).

## NUM-18

**Diagnostics misclassifies underflows, rounded boundary values, and observed extrema.**

**P3 · code quality · confirmed-reproduced · finding**

**Exposure:** Opt-in diagnostics reached by public FP16 construction.

**Original references:** reduced_precision additional diagnostics classification

**Sources:** [Sources/VectorCore/Operations/Kernels/MixedPrecisionDiagnostics.swift:26](../../../Sources/VectorCore/Operations/Kernels/MixedPrecisionDiagnostics.swift:26); [Sources/VectorCore/Operations/Kernels/MixedPrecisionDiagnostics.swift:132](../../../Sources/VectorCore/Operations/Kernels/MixedPrecisionDiagnostics.swift:132); [Sources/VectorCore/Operations/Kernels/MixedPrecisionDiagnostics.swift:173](../../../Sources/VectorCore/Operations/Kernels/MixedPrecisionDiagnostics.swift:173)

**Evidence:** Fresh Release NumericProbe results.log: 1e-12 decodes0 but reports512 subnormals/0 underflows;65505 decodes65504 but reports512 overflows; positive-only min remains0. Underflow branch is dominated by broader subnormal predicate.

**Correction / limit:** Swapping branches or using65504 as an overflow cutoff is insufficient; classification must follow actual rounded conversion.

**Improvement:** Classify input and converted bit pattern with explicit terminology; seed extrema from first finite value or empty sentinels.

**Acceptance:** Scalar/batch classification same on rounding ties, adjacent FP16 normal/subnormal/zero/overflow boundaries, NaNs/infinities and positive/negative-only batches.

**Dependencies:** API-21

**Evidence set:** [numerics](evidence/validate_numerics/) (see [replay/provenance index](evidence/README.md)).

## NUM-19

**KernelAutoTuner accepts memory/latency constraints it never uses and persists profiles without provenance.**

**P3 · code quality · confirmed-source · opportunity**

**Exposure:** Public opt-in tuner, not automatic Operations dispatch.

**Original references:** reduced_precision tuner architecture

**Sources:** [Sources/VectorCore/Operations/Kernels/KernelAutoTuner.swift:41](../../../Sources/VectorCore/Operations/Kernels/KernelAutoTuner.swift:41); [Sources/VectorCore/Operations/Kernels/KernelAutoTuner.swift:189](../../../Sources/VectorCore/Operations/Kernels/KernelAutoTuner.swift:189); [Sources/VectorCore/Operations/Kernels/KernelAutoTuner.swift:224](../../../Sources/VectorCore/Operations/Kernels/KernelAutoTuner.swift:224); [Sources/VectorCore/Operations/Kernels/KernelAutoTuner.swift:592](../../../Sources/VectorCore/Operations/Kernels/KernelAutoTuner.swift:592); [Sources/VectorCore/Operations/Kernels/KernelAutoTuner.swift:774](../../../Sources/VectorCore/Operations/Kernels/KernelAutoTuner.swift:774)

**Evidence:** Source search: memoryBudget/latencyTarget only stored in workload, key uses dimension/batchSize/precisionMode. Profile serialization contains profileCache only; override dictionary never serialized despite persistence comment. No revision/hardware expiry gate inspected in loadProfiles.

**Correction / limit:** Missing hardware identity is stale-profile risk, not reproduced bad dispatch; memoryFootprint is an estimate, not allocation measurement. Tests explicitly bypass profile selection under detected CI/XCTest, so existing passes do not cover ordinary persistence path.

**Improvement:** Make constraints operative or remove implied guarantees; version profile provenance and persist override if promised.

**Acceptance:** Temporary-path persistence round trip and non-CI isolated selection test; vary constraints with same shape, require feasible or explicit no-feasible response; invalidate incompatible profiles.

**Evidence set:** [numerics](evidence/validate_numerics/) (see [replay/provenance index](evidence/README.md)).

## NUM-20

**Explicit selectMixedPrecisionStrategy maps full precision mode to nonzero tolerance.**

**P3 · code quality · qualified · opportunity**

**Exposure:** Public opt-in low-level tuner method.

**Original references:** reduced_precision tuner architecture fp32Full

**Sources:** [Sources/VectorCore/Operations/Kernels/KernelAutoTuner.swift:793](../../../Sources/VectorCore/Operations/Kernels/KernelAutoTuner.swift:793); [Sources/VectorCore/Operations/Kernels/KernelAutoTuner.swift:808](../../../Sources/VectorCore/Operations/Kernels/KernelAutoTuner.swift:808)

**Evidence:** Source maps fp32Full to0.0001 then calls mixed tuner, which filters measured synthetic errors and can select reduced precision. Normal selectStrategy heuristic/profile availability respects fullFP32 mode.

**Correction / limit:** Do not claim normal fp32Full dispatch automatically quantizes; contradictory behavior is limited to explicit mixed-selection entry point, whose docs expect cpuMixedPrecision workload.

**Improvement:** Reject incompatible mode or return fullFP32 immediately; clarify method precondition.

**Acceptance:** Explicit fp32Full call always returns fullFP32 or a documented error; no synthetic benchmark can weaken exact precision policy.

**Dependencies:** NUM-19

**Evidence set:** [numerics](evidence/validate_numerics/) (see [replay/provenance index](evidence/README.md)).

## NUM-21

**Mixed tuner calibrates all5 strategies synchronously on exact workload key and converts candidates inside each reduced execution.**

**P3 · performance · confirmed-source · opportunity**

**Exposure:** First opt-in MixedPrecisionAutoTuner selection for a fresh dimension/count/accuracy key.

**Original references:** reduced_precision calibration cost

**Sources:** [Sources/VectorCore/Operations/Kernels/MixedPrecisionAutoTuner.swift:155](../../../Sources/VectorCore/Operations/Kernels/MixedPrecisionAutoTuner.swift:155); [Sources/VectorCore/Operations/Kernels/MixedPrecisionAutoTuner.swift:269](../../../Sources/VectorCore/Operations/Kernels/MixedPrecisionAutoTuner.swift:269); [Sources/VectorCore/Operations/Kernels/MixedPrecisionAutoTuner.swift:419](../../../Sources/VectorCore/Operations/Kernels/MixedPrecisionAutoTuner.swift:419); [Sources/VectorCore/Operations/Kernels/MixedPrecisionAutoTuner.swift:909](../../../Sources/VectorCore/Operations/Kernels/MixedPrecisionAutoTuner.swift:909)

**Evidence:** 5 strategies*(5 warmups+20 measurements)=125 timed/warmup executions, plus data generation and FP32 baseline; empty accepted set can add25 FP32 fallback executions. Calibration synchronous inside actor; executeStrategy constructs FP16 SoA per invocation.

**Correction / limit:** 125 is normal-path count, not unconditional exact total. Profiles use synthetic data and end-to-end conversion cost; they cannot establish arbitrary input accuracy or preconverted serving latency. No wall-clock regression measured.

**Improvement:** Bound/offline calibration, coarser workload families or explicit profiles; distinguish preconverted serving and conversion-inclusive costs.

**Acceptance:** Count calibration work and conversion boundaries mechanically; measure coordinated Release first-call latency/peak allocations on representative workloads; preserve error semantics.

**Evidence set:** [numerics](evidence/validate_numerics/) (see [replay/provenance index](evidence/README.md)).

## NUM-22

**FP16 range wrappers map all candidates even for small or empty ranges.**

**P2 · performance · confirmed-source · finding**

**Exposure:** Nine public range FP16/mixed wrappers.

**Original references:** PERF4, reduced_precision range wrapper performance

**Sources:** [Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:4666](../../../Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:4666); [Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:4736](../../../Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:4736); [Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:4802](../../../Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:4802)

**Evidence:** All nine wrappers construct candidatesFP16.map{$0.storage} before forwarding range; K chunks over N inputs entails O(KN) metadata work in source and O(N) temporary handles per concurrent call.

**Correction / limit:** No measured latency/allocation count; optimizer may influence realized overhead. Merely slicing without adjusting range/global output conventions can break indexing.

**Improvement:** Pass typed candidates directly or process correctly indexed slice; preserve compact output convention.

**Acceptance:** Empty/single range retained metadata/work independent of unrelated candidate count; all widths and nonzero range starts produce correct scores/sentinels.

**Evidence set:** [numerics](evidence/validate_numerics/) (see [replay/provenance index](evidence/README.md)).

## NUM-23

**Source numerical/performance assurances exceed tests or contradict implementation.**

**P3 · code quality · confirmed-source · finding**

**Exposure:** Public API comments and internal optimization documentation.

**Original references:** reduced_precision misleading numerical/performance claims, simd_kernels architecture unsupported claims, linear_algebra limitations 6, batch_matrix opportunity 6

**Sources:** [Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:68](../../../Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:68); [Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:4968](../../../Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:4968); [Sources/VectorCore/LinearAlgebra/SwiftLinearAlgebraProvider.swift:6](../../../Sources/VectorCore/LinearAlgebra/SwiftLinearAlgebraProvider.swift:6); [Sources/VectorCore/LinearAlgebra/PCA.swift:14](../../../Sources/VectorCore/LinearAlgebra/PCA.swift:14); [Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift:115](../../../Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift:115); [Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift:1343](../../../Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift:1343)

**Evidence:** FP16 can merge distinct1/1.0001 values (fresh probe); FP32 accumulation cannot restore lost ordering. Pure Swift robustness disproved NUM08/09. FP16 pool acquire/release explicitly lock; source speed percentages lack named exercising benchmark. PCA exact fallback contradicts no-full-factorization wording.

**Correction / limit:** Do not convert unsupported speed claims into claims of measured slowdown; claims of external GPU limitations remain unverified without GPU execution.

**Improvement:** Remove absolute ranking/robustness claims; tie speedups to hardware/revision/workload and named artifacts; state documented numerical limitations.

**Acceptance:** Mechanical claim inventory: each guarantee has nearby one-step proof, named test, or explicit unverified-contract wording; performance records carry reproducible provenance.

**Dependencies:** NUM-08, NUM-09, NUM-16

**Evidence set:** [numerics](evidence/validate_numerics/) (see [replay/provenance index](evidence/README.md)).

## NUM-24

**Immutable FP16 vector pool cannot refill storage for conversion of new values via a supported API.**

**P3 · code quality · confirmed-source · opportunity**

**Exposure:** Public opt-in pool.

**Original references:** reduced_precision FP16VectorPool reuse

**Sources:** [Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:319](../../../Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:319); [Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:4983](../../../Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:4983); [Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:5012](../../../Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:5012)

**Evidence:** Acquire returns immutable FP16 value; release appends value. Supported construction creates storage, no refill operation. Source comments claim allocation elimination and GC pressure reduction.

**Correction / limit:** It can reuse immutable values for repeated computation; claim is specifically unsupported reuse as mutable conversion destination. No allocation speed benefit measured.

**Improvement:** Clarify purpose or provide explicit mutable owned scratch with scoped fill semantics; avoid changing ownership casually.

**Acceptance:** A consumer converts distinct successive vectors while demonstrating actual allocation reuse and correct values; otherwise remove conversion-reuse claim.

**Dependencies:** OWN-17

**Evidence set:** [numerics](evidence/validate_numerics/) (see [replay/provenance index](evidence/README.md)).

## NUM-25

**CSR initializer may dereference malformed offsets before throwing validation catches them.**

**P3 · code quality · qualified · opportunity**

**Exposure:** Malformed public KNNGraph input outside documented structural caller obligation.

**Original references:** umap CSR boundary, README KNNGraph ambiguity

**Sources:** [Sources/VectorCore/ManifoldLearning/KNNGraph.swift:42](../../../Sources/VectorCore/ManifoldLearning/KNNGraph.swift:42); [Sources/VectorCore/ManifoldLearning/KNNGraph.swift:67](../../../Sources/VectorCore/ManifoldLearning/KNNGraph.swift:67); [Docs/Linear_Algebra_and_Projection.md:113](../../../Docs/Linear_Algebra_and_Projection.md:113)

**Evidence:** Inspected original probe [0,3,2], neighbors[1,1],distances[1,1],pointCount2. End/count checks pass; first row iterates index2 out of bounds before second row monotonicity check. Prior trap is not freshly rerun here.

**Correction / limit:** Not supported valid graph input; docs explicitly prohibit treating constructor as arbitrary parser, while source promises malformed structure throws. Frame as boundary/documentation inconsistency, not ordinary-use corruption.

**Improvement:** Validate all offsets monotone and bounded before dereferencing; synchronize source/downstream documentation.

**Acceptance:** Generated small malformed-offset matrix including valid final endpoint but interior overshoot/negative/decrease must throw under chosen checked contract; valid uneven/unsorted rows remain accepted.

**Evidence set:** [numerics](evidence/validate_numerics/) (see [replay/provenance index](evidence/README.md)).

## NUM-26

**Accepted extreme integer configuration can trap before throwing or clamping.**

**P3 · code quality · qualified · finding**

**Exposure:** Explicit oversized positive PCA/UMAP options on tiny ordinary data.

**Original references:** C15, linear_algebra LA-3, umap extreme integer options

**Sources:** [Sources/VectorCore/LinearAlgebra/PCA.swift:137](../../../Sources/VectorCore/LinearAlgebra/PCA.swift:137); [Sources/VectorCore/LinearAlgebra/PCA.swift:173](../../../Sources/VectorCore/LinearAlgebra/PCA.swift:173); [Sources/VectorCore/ManifoldLearning/UMAP.swift:215](../../../Sources/VectorCore/ManifoldLearning/UMAP.swift:215); [Sources/VectorCore/ManifoldLearning/UMAP.swift:225](../../../Sources/VectorCore/ManifoldLearning/UMAP.swift:225); [Sources/VectorCore/ManifoldLearning/UMAP.swift:450](../../../Sources/VectorCore/ManifoldLearning/UMAP.swift:450)

**Evidence:** PCA k+Int.max overflows before min at k1. UMAP3*Int.max overflows before allocation; negativeSampleRate Int.max is converted to Double rounding to2^63, and due calculation can convert that value back to Int. Original crash sources inspected; no fresh crashes required.

**Correction / limit:** Operationally impractical settings; not ordinary-sized failure or giant allocation test. General allocation-safety disclaimer does not explain away the easily checked preallocation addition, but priority is below valid ordinary-input defects. Original UMAP report line229/241 corrected to215/225 for n*dimensions.

**Improvement:** Clamp oversampling before addition or typed rejection; checked dimensions products; bound/check sample counts before floating-to-Int conversion.

**Acceptance:** Tiny fixtures atInt.max, safe bounds, zero/negative/normal values either throw documented error or produce bounded intended result without process termination.

**Evidence set:** [numerics](evidence/validate_numerics/) (see [replay/provenance index](evidence/README.md)).

## NUM-27

**Exact brute-force kNN builder is exhaustive over computed Gram scores, not a reliable numerical distance oracle.**

**P3 · code quality · qualified · context**

**Exposure:** Public reference KNNGraph.bruteForce.

**Original references:** umap reference builder Gram cancellation

**Sources:** [Sources/VectorCore/ManifoldLearning/KNNGraph.swift:135](../../../Sources/VectorCore/ManifoldLearning/KNNGraph.swift:135); [Sources/VectorCore/ManifoldLearning/KNNGraph.swift:166](../../../Sources/VectorCore/ManifoldLearning/KNNGraph.swift:166); [Docs/Numerical_Behavior.md:86](../../../Docs/Numerical_Behavior.md:86)

**Evidence:** Fresh Release NumericProbe results.log: [10000,10002,10001],k1=>neighbors[1,0,0], all distances0. Direct Float-exact differences give row0 nearest2 at distance1, row1 nearest2.

**Correction / limit:** Documented Gram cancellation elsewhere; do not present as novel UMAP algorithm defect. Broad numerical docs apply; exact label should not imply stable reference scores.

**Improvement:** Link conditioning caveat; if advertised as numerical oracle, use direct differences or sufficiently comprehensive refinement.

**Acceptance:** Near-offset fixtures compared with widened/direct reference; any refinement must not lose true nearest candidate before reranking.

**Dependencies:** API-22

**Evidence set:** [numerics](evidence/validate_numerics/) (see [replay/provenance index](evidence/README.md)).

## NUM-28

**UMAP initial coordinates are shape checked, not validated for finiteness or safe scale.**

**P3 · code quality · qualified · opportunity**

**Exposure:** Explicit caller-supplied initialization.

**Original references:** umap explicit initial-coordinate values

**Sources:** [Sources/VectorCore/ManifoldLearning/UMAP.swift:211](../../../Sources/VectorCore/ManifoldLearning/UMAP.swift:211); [Sources/VectorCore/ManifoldLearning/UMAP.swift:215](../../../Sources/VectorCore/ManifoldLearning/UMAP.swift:215); [Sources/VectorCore/ManifoldLearning/UMAP.swift:471](../../../Sources/VectorCore/ManifoldLearning/UMAP.swift:471)

**Evidence:** Source copies coordinates verbatim; NaN differences fail positive squared guard and remain NaN; Float squared differences at1e30 overflow. Original NaN/huge probes inspected, no new initialization run.

**Correction / limit:** No all-finite or conditioning guarantee; do not generalize to ordinary random/PCA initialization.

**Improvement:** Specify responsibility, reject nonfinite values in checked entry point, optionally normalize supported initialization.

**Acceptance:** NaN/±Inf coordinate tests and large finite scale fixtures with documented result/error policy; ordinary init deterministic.

**Evidence set:** [numerics](evidence/validate_numerics/) (see [replay/provenance index](evidence/README.md)).

## NUM-29

**UMAP and PCA expose material in-memory storage costs at large corpus/sample size.**

**P3 · performance · confirmed-source · opportunity**

**Exposure:** Public in-memory UMAP/PCA; default sizes/workloads determine exposure.

**Original references:** PERF8, umap graph memory, linear_algebra limitations 2

**Sources:** [Sources/VectorCore/ManifoldLearning/FuzzySimplicialSet.swift:105](../../../Sources/VectorCore/ManifoldLearning/FuzzySimplicialSet.swift:105); [Sources/VectorCore/ManifoldLearning/FuzzySimplicialSet.swift:126](../../../Sources/VectorCore/ManifoldLearning/FuzzySimplicialSet.swift:126); [Sources/VectorCore/ManifoldLearning/UMAP.swift:412](../../../Sources/VectorCore/ManifoldLearning/UMAP.swift:412); [Sources/VectorCore/ManifoldLearning/UMAP.swift:425](../../../Sources/VectorCore/ManifoldLearning/UMAP.swift:425); [Sources/VectorCore/LinearAlgebra/PCA.swift:148](../../../Sources/VectorCore/LinearAlgebra/PCA.swift:148); [Sources/VectorCore/LinearAlgebra/PCA.swift:179](../../../Sources/VectorCore/LinearAlgebra/PCA.swift:179)

**Evidence:** UMAP original edges12F bytes plus retained optimizer8R indices+32R four Double arrays=>12F+40R logical payload. AtF=R=150M =>7.8GB decimal, before CSR/input/coordinates. PCA500000*1536*4=3.072GB sample payload; exact SVD also returns discarded U.

**Correction / limit:** F/R depend on reciprocity/pruning; staging/sort slack and ARC lifetimes vary. This is a byte model, not measured RSS. In-memory design explicitly documented.

**Improvement:** Budget/report logical bytes, reduce duplicate graph arrays, consider right-factor-only exact PCA and bounded transform APIs; profile before promising savings.

**Acceptance:** Measured peak allocation/RSS alongside predicted payload for representative Release runs; verify identical results/order or explicitly documented numerical changes.

**Evidence set:** [numerics](evidence/validate_numerics/) (see [replay/provenance index](evidence/README.md)).

## NUM-30

**Graph sorts/epoch scans and PCA per-point transforms offer concrete allocation/work opportunities.**

**P3 · performance · confirmed-source · opportunity**

**Exposure:** Public synchronous algorithms; no throughput regression measured.

**Original references:** umap sorts/epoch scans, linear_algebra limitations 3

**Sources:** [Sources/VectorCore/ManifoldLearning/FuzzySimplicialSet.swift:123](../../../Sources/VectorCore/ManifoldLearning/FuzzySimplicialSet.swift:123); [Sources/VectorCore/ManifoldLearning/FuzzySimplicialSet.swift:146](../../../Sources/VectorCore/ManifoldLearning/FuzzySimplicialSet.swift:146); [Sources/VectorCore/ManifoldLearning/UMAP.swift:439](../../../Sources/VectorCore/ManifoldLearning/UMAP.swift:439); [Sources/VectorCore/LinearAlgebra/PCA.swift:257](../../../Sources/VectorCore/LinearAlgebra/PCA.swift:257); [Sources/VectorCore/LinearAlgebra/PCA.swift:291](../../../Sources/VectorCore/LinearAlgebra/PCA.swift:291)

**Evidence:** Canonical and fused sorts O(E logE+F logF);64-pass calibration O(64E), optimizer scansR every epoch. Scalar transforms allocate outputs and use Float accumulation; batch output builds per-row arrays.

**Correction / limit:** Not missing ANN/GPU functionality; serial deterministic and in-memory contract is explicit.

**Improvement:** Evaluate compact graphs/scheduled edges/flat output views from consumer needs; preserve ordering and seed behavior.

**Acceptance:** Coordinated Release time/allocation profiles and result/seed reproducibility checks for any schedule or format change.

**Evidence set:** [numerics](evidence/validate_numerics/) (see [replay/provenance index](evidence/README.md)).

## NUM-31

**Long synchronous manifold/factorization work lacks cancellation checkpoints.**

**P3 · code quality · qualified · opportunity**

**Exposure:** Callers using synchronous APIs, including inside async tasks.

**Original references:** umap no cancellation, linear_algebra limitations 4

**Sources:** [Sources/VectorCore/ManifoldLearning/UMAP.swift:439](../../../Sources/VectorCore/ManifoldLearning/UMAP.swift:439); [Sources/VectorCore/ManifoldLearning/FuzzySimplicialSet.swift:46](../../../Sources/VectorCore/ManifoldLearning/FuzzySimplicialSet.swift:46); [Sources/VectorCore/LinearAlgebra/PCA.swift:189](../../../Sources/VectorCore/LinearAlgebra/PCA.swift:189); [Sources/VectorCore/LinearAlgebra/SwiftLinearAlgebraProvider.swift:30](../../../Sources/VectorCore/LinearAlgebra/SwiftLinearAlgebraProvider.swift:30)

**Evidence:** Inspected loops have no cancellation hooks; synchronous actor/executor caller cannot interrupt their work via task cancellation alone.

**Correction / limit:** No broken cancellation guarantee is documented. Large positive powerIterations requests more work intentionally.

**Improvement:** Consider throwing cancellation/progress overload for long jobs, with bounded polling cost and deterministic unchanged no-cancel behavior.

**Acceptance:** Cancel at meaningful phases and bound latency/resources; legacy synchronous success output remains consistent.

**Evidence set:** [numerics](evidence/validate_numerics/) (see [replay/provenance index](evidence/README.md)).

## NUM-32

**C INT8 accumulation can exceed signed32 range only beyond supported fixed-vector dimensions.**

**P3 · code quality · qualified · context**

**Exposure:** Generic exported C lanes API; optional Swift C INT8 call sites use fixed512/768/1536.

**Original references:** simd_kernels C arithmetic policy, reduced_precision INT8 strengths

**Sources:** [Sources/VectorCoreC/common/vc_scalar.c:43](../../../Sources/VectorCoreC/common/vc_scalar.c:43); [Sources/VectorCoreC/arm64/vc_arm64.c:116](../../../Sources/VectorCoreC/arm64/vc_arm64.c:116); [Sources/VectorCoreC/x86_64/vc_x86.c:47](../../../Sources/VectorCoreC/x86_64/vc_x86.c:47); [Sources/VectorCoreC/common/vc_dispatch.c:78](../../../Sources/VectorCoreC/common/vc_dispatch.c:78)

**Evidence:** Worst signed Int8 product16384;131072 terms sum2147483648 outside Int32. Maximum1536 terms=25165824 fits. Scalar C signed+= overflows for generic large input; Swift fallback wrapping arithmetic explicit. Tail bounds inspected: complete blocks then i<lanes remainder.

**Correction / limit:** Do not allege active fixed-width overflow. C intrinsic lane behavior differs from scalar C signed-overflow language semantics; no Intel/NEON overflow execution or UBSan run.

**Improvement:** Specify representable-sum bound or widen/define accumulator for generic lanes before broadening routing.

**Acceptance:** All architectures/C gate configurations compare max-magnitude fixed widths and remainder counts to Int64 reference; arbitrary supported lanes must have explicit overflow behavior.

**Evidence set:** [numerics](evidence/validate_numerics/) (see [replay/provenance index](evidence/README.md)).

## NUM-33

**C header range indexing wording and proposed unimplemented declarations need precise public status.**

**P3 · code quality · qualified · context**

**Exposure:** Public C/header and Swift wrapper interface; no current Swift link failure established.

**Original references:** simd_kernels architecture C API hygiene

**Sources:** [Sources/VectorCoreC/include/VectorCoreC.h:1](../../../Sources/VectorCoreC/include/VectorCoreC.h:1); [Sources/VectorCoreC/common/vc_scalar.c:39](../../../Sources/VectorCoreC/common/vc_scalar.c:39); [Sources/VectorCore/Interop/CKernels.swift:1](../../../Sources/VectorCore/Interop/CKernels.swift:1)

**Evidence:** Inspected scalar/NEON/x86 range implementations output compact0..<end-start; Swift wrapper follows compact convention. Proposed symbols are marked proposed, and FP32 wrappers have no established production consumers.

**Correction / limit:** Header declaration of proposed symbol is API-hygiene risk, not proof a normal build fails.

**Improvement:** Align range docs and segregate proposal declarations; preserve compact convention.

**Acceptance:** C consumer symbol/link and nonzero-start sentinel tests; supported public declarations all link.

**Evidence set:** [numerics](evidence/validate_numerics/) (see [replay/provenance index](evidence/README.md)).

## NUM-34

**Allocation-heavy vector helpers, unequal SIMD reduction structure and platform heuristics warrant measurement, not speed claims.**

**P3 · performance · qualified · opportunity**

**Exposure:** Public VectorMath helpers; internal PlatformConfiguration and mostly unused optimization helpers.

**Original references:** simd_kernels architecture allocation/heuristics

**Sources:** [Sources/VectorCore/Operations/VectorMath.swift:1](../../../Sources/VectorCore/Operations/VectorMath.swift:1); [Sources/VectorCore/Platform/SwiftDoubleSIMDProvider.swift:1](../../../Sources/VectorCore/Platform/SwiftDoubleSIMDProvider.swift:1); [Sources/VectorCore/Platform/PlatformConfiguration.swift:1](../../../Sources/VectorCore/Platform/PlatformConfiguration.swift:1); [Sources/VectorCore/Optimization/OptimizationAttributes.swift:1](../../../Sources/VectorCore/Optimization/OptimizationAttributes.swift:1)

**Evidence:** Original locations inspected: L1/variance create full arrays; Double reduces SIMD chunks horizontally while Float retains lanes. Compile-time architecture branches assume hardware traits; named likely/prefetch helpers do not establish compiler behavior.

**Correction / limit:** No universal measured slowdown; no illegal instruction dispatch is proven from these width/cache hints.

**Improvement:** Profile allocations/generated code before fusion/tuning changes; replace hardware facts with explicit heuristics/runtime facts where needed.

**Acceptance:** Representative Release allocation/throughput and generated-code records, matched semantics, supported-platform conformance.

**Evidence set:** [numerics](evidence/validate_numerics/) (see [replay/provenance index](evidence/README.md)).

## NUM-35

**LAPACK dimension ceiling does not by itself establish extreme workspace arithmetic safety.**

**P3 · code quality · qualified · opportunity**

**Exposure:** Shapes far beyond documented small-panel operating envelope.

**Original references:** linear_algebra limitations 7

**Sources:** [Sources/VectorCoreC/common/vc_lapack.c:99](../../../Sources/VectorCoreC/common/vc_lapack.c:99); [Sources/VectorCore/LinearAlgebra/LinearAlgebraProvider.swift:1](../../../Sources/VectorCore/LinearAlgebra/LinearAlgebraProvider.swift:1)

**Evidence:** C8*minmn occurs before size_t cast; optimal workspace may exceed32-bit with otherwise valid dimensions. No practical-size failure reproduced.

**Correction / limit:** Do not elevate to verified practical memory-corruption defect. Two Int32 dimensions multiply safely in64-bit Swift Int; workspace arithmetic is distinct.

**Improvement:** Checked workspace/index sizes if arbitrary-shape support expands; retain small-panel caveat.

**Acceptance:** Boundary size arithmetic tested without giant allocations; supported envelope documented and clean error outside it.

**Evidence set:** [numerics](evidence/validate_numerics/) (see [replay/provenance index](evidence/README.md)).

## OWN-01

**Public DynamicVector and optimized FP32 storage mutations break shape invariants.**

**P1 · architecture · confirmed-reproduced · finding**

**Exposure:** public default representations

**Original references:** A1, V1

**Sources:** [Sources/VectorCore/Vectors/DynamicVector.swift:17](../../../Sources/VectorCore/Vectors/DynamicVector.swift:17); [Sources/VectorCore/Vectors/DynamicVector.swift:139](../../../Sources/VectorCore/Vectors/DynamicVector.swift:139); [Sources/VectorCore/Vectors/Vector384Optimized.swift:156](../../../Sources/VectorCore/Vectors/Vector384Optimized.swift:156); [Sources/VectorCore/Vectors/Vector512Optimized.swift:150](../../../Sources/VectorCore/Vectors/Vector512Optimized.swift:150); [Sources/VectorCore/Vectors/Vector768Optimized.swift:150](../../../Sources/VectorCore/Vectors/Vector768Optimized.swift:150); [Sources/VectorCore/Vectors/Vector1536Optimized.swift:150](../../../Sources/VectorCore/Vectors/Vector1536Optimized.swift:150); [Sources/VectorCore/Protocols/VectorProtocol.swift:38](../../../Sources/VectorCore/Protocols/VectorProtocol.swift:38)

**Evidence:** Fresh Ownership Release reports all four optimized views advertising 384/512/768/1536 values over four actual scalars. Original valid DynamicVector replacement probe is independently source-checked; its first-pass reproduction is preserved: dimension remains 2 with three storage elements; equality inspects dimension while hash traverses storage.

**Correction / limit:** Do not say every vector or every check fails: generic Vector<D> uses a count-constrained DimensionStorage; FP16 conversions explicitly check malformed FP32 storage. No out-of-bounds write was executed. Different hash values are an observed consequence, not a mathematical guarantee against incidental hash collisions.

**Improvement:** Enforce shape at owners: dynamic dimension derives from storage or a checked setter keeps it synchronized; fixed storage must preserve exact lane count through replacement and collection mutation. Reconsider protocol-wide settable storage.

**Acceptance:** Mechanical matrix for all families and wrappers: short/equal/long replacement and direct collection edits; dimension/count/buffer/serialization agree; equal values hash equally; invalid shapes are rejected before memory access. Test generic protocol mutation as well.

**Evidence set:** [ownership](evidence/validate_ownership/) (see [replay/provenance index](evidence/README.md)).

## OWN-02

**INT8 raw constructors, mutable storage, and synthesized Codable permit malformed fixed storage.**

**P1 · architecture · confirmed-reproduced · finding**

**Exposure:** public opt-in INT8 APIs

**Original references:** A1, RP additional: public invariant gap

**Sources:** [Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:140](../../../Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:140); [Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:164](../../../Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:164); [Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:177](../../../Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:177); [Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:190](../../../Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:190); [Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:461](../../../Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:461)

**Evidence:** Ownership Release accepts one SIMD lane for a reported 512-dimensional vector. ShapeRoutes Release decodes storage:[] for 512/768/1536. Fixed-lane toFP32 reads follow directly from source. All four optimized FP32 decoders reject empty arrays as controls.

**Correction / limit:** FP16 vectors are immutable and exact-length validated; they are not part of this shape defect. Invalid INT8 storage is observed; resulting unsafe reads are source-derived, not an executed memory fault. Synthesized decoding bypasses any future initializer-only fix.

**Improvement:** Validate raw init and custom decoding; restrict storage mutation to shape-preserving operations, validate parameters separately.

**Acceptance:** Each INT8 width rejects short/long raw storage and JSON storage; valid round trips preserve dimension, payload and parameters. Exercise post-construction collection mutation and all unsafe conversion/kernel entry points.

**Dependencies:** OWN-01

**Evidence set:** [ownership](evidence/validate_ownership/) (see [replay/provenance index](evidence/README.md)).

## OWN-03

**NormalizationHint retains metadata after direct wrapped mutation or a throwing mutable-buffer closure.**

**P1 · architecture · confirmed-reproduced · finding**

**Exposure:** public wrapper; inspected downstream consumers

**Original references:** A2, IF5

**Sources:** [Sources/VectorCore/Integration/IndexableVector.swift:107](../../../Sources/VectorCore/Integration/IndexableVector.swift:107); [Sources/VectorCore/Integration/IndexableVector.swift:203](../../../Sources/VectorCore/Integration/IndexableVector.swift:203); [../VectorIndex/Sources/VectorIndex/TypedOverloads.swift:102](../../../../VectorIndex/Sources/VectorIndex/TypedOverloads.swift:102); [../VectorIndex/Sources/VectorIndex/TypedOverloads.swift:119](../../../../VectorIndex/Sources/VectorIndex/TypedOverloads.swift:119); [../future/VectorAccelerate/Sources/VectorAccelerate/Integration/KernelDistanceProviders.swift:189](../../../../future/VectorAccelerate/Sources/VectorAccelerate/Integration/KernelDistanceProviders.swift:189)

**Evidence:** Fresh Release: initially valid [1,0] auto-hint changes to [2,0] through vector and reports normalized=true/magnitude1 while underlying magnitude2; buffer writes3 then throws and metadata remains1. Read downstream consumers using flags and cached inverse norms.

**Correction / limit:** The initial hint is true, so this is not caller supplying a false assertion. GPU/downstream end-to-end behavior was not executed; existing contaminated indexes needing rebuild is conditional, not established deployment state.

**Improvement:** Observe/restrict wrapped-value mutation and invalidate before mutable access or with defer. Explicitly define trust and value-semantics requirements for external conformers and decoded hints.

**Acceptance:** Test direct vector replacement/subscript/storage, wrapper storage, successful and throwing buffer mutation on every built-in representation; after changes metadata must be cleared or recomputed. Downstream insertion/query tests compare fresh normalized reference.

**Dependencies:** OWN-01

**Evidence set:** [ownership](evidence/validate_ownership/) (see [replay/provenance index](evidence/README.md)).

## OWN-04

**SoAFP16Cache512 treats vector hash digests as exact candidate identity.**

**P1 · architecture · confirmed-reproduced · finding**

**Exposure:** public opt-in cache; no default or sibling caller found

**Original references:** A3, RP1

**Sources:** [Sources/VectorCore/Operations/Kernels/SoAFP16Cache.swift:16](../../../Sources/VectorCore/Operations/Kernels/SoAFP16Cache.swift:16); [Sources/VectorCore/Operations/Kernels/SoAFP16Cache.swift:94](../../../Sources/VectorCore/Operations/Kernels/SoAFP16Cache.swift:94); [Sources/VectorCore/Operations/Kernels/SoAFP16Cache.swift:181](../../../Sources/VectorCore/Operations/Kernels/SoAFP16Cache.swift:181); [Sources/VectorCore/Vectors/Vector512Optimized.swift:334](../../../Sources/VectorCore/Vectors/Vector512Optimized.swift:334)

**Evidence:** Fresh ShapeRoutes Release caches zero then independently changes each coordinate: 496 of512 changed inputs falsely hit; first/last eight miss. The first-pass returned-value corruption fixture is preserved separately.

**Correction / limit:** Only the512 cache exists; 384/768/1536 key initializers are currently unused future surface. Full-coordinate hashing alone cannot turn a digest into an equality proof.

**Improvement:** Use immutable candidate snapshot identity/version or compare complete contents on digest hits; retain order and candidate count in identity.

**Acceptance:** For every coordinate, reorder, count change and retained snapshot mutation, getOrCreate must match uncached conversion. Inject deliberate hash collisions to prove equality checking; repeated identical content still hits.

**Dependencies:** OWN-07

**Evidence set:** [ownership](evidence/validate_ownership/) (see [replay/provenance index](evidence/README.md)).

## OWN-05

**Compact-tier triangular matrix copies share later mutations.**

**P1 · code quality · confirmed-reproduced · finding**

**Exposure:** public TriangularMatrix default sizes4...32; internal compact counts5...512

**Original references:** C1, MS-1

**Sources:** [Sources/VectorCore/Utilities/TieredStorage.swift:175](../../../Sources/VectorCore/Utilities/TieredStorage.swift:175); [Sources/VectorCore/Utilities/TieredStorage.swift:450](../../../Sources/VectorCore/Utilities/TieredStorage.swift:450); [Sources/VectorCore/Utilities/TriangularMatrix.swift:94](../../../Sources/VectorCore/Utilities/TriangularMatrix.swift:94); [Sources/VectorCore/Utilities/TriangularMatrix.swift:157](../../../Sources/VectorCore/Utilities/TriangularMatrix.swift:157)

**Evidence:** Fresh actual-library Release verifies all508 compact element counts5...512 alias after copy/subscript write; public matrix n4,16,32 originals change99, controls3,33,64,65 stay1. CompactBuffer reference is shared while uniqueness inspects its nested ManagedBuffer.

**Correction / limit:** Range is derived from m=n(n-1)/2; no claim for an arbitrary configured tier. Iterator snapshot and setAt exposure follow their same stored-copy/setter paths, not independent runtime reproduction.

**Improvement:** Check uniqueness of the actual stored shared reference, not an inner property. Coordinate compact destruction repair.

**Acceptance:** COW tests across tier counts0,1,4,5,512,513,2048,2049 and matrix n3,4,32,33,64,65 for subscript,setAt,bulk and preexisting iterators; original unchanged and first mutation detaches once.

**Dependencies:** OWN-10, OWN-11

**Evidence set:** [ownership](evidence/validate_ownership/) (see [replay/provenance index](evidence/README.md)).

## OWN-06

**Binary in-place DynamicVector routes omit dimension checks.**

**P2 · code quality · confirmed-reproduced · finding**

**Exposure:** public direct defaults and generic VectorProtocol operator dispatch

**Original references:** C5, V3

**Sources:** [Sources/VectorCore/Protocols/VectorProtocol.swift:181](../../../Sources/VectorCore/Protocols/VectorProtocol.swift:181); [Sources/VectorCore/Protocols/VectorProtocol.swift:194](../../../Sources/VectorCore/Protocols/VectorProtocol.swift:194); [Sources/VectorCore/Protocols/VectorProtocol.swift:230](../../../Sources/VectorCore/Protocols/VectorProtocol.swift:230); [Sources/VectorCore/Protocols/VectorProtocol.swift:241](../../../Sources/VectorCore/Protocols/VectorProtocol.swift:241)

**Evidence:** Fresh Release longer RHS yields add[5,7],subtract[-3,-3],Hadamard[4,10],generic +=[5,7],generic -=[-3,-3]. Source loops over lhs scalarCount with no check. Original Debug short-RHS fixture inspected; its original evidence is retained; no fresh short-buffer Release overread was executed.

**Correction / limit:** Do not extend unchecked dispatch claim to concrete VectorFactory +=, which checks through +. Short RHS optimized overread is static reasoning; no Release overread was intentionally executed.

**Improvement:** Shared dimension precondition before mutation on all three binary mutators; retain direct/generic consistency.

**Acceptance:** Equal/short/long RHS, empty combinations and direct/generic +=/-= plus all three methods. Invalid dimensions fail before any write; subprocess tests for trapping API or recoverable errors for checked API.

**Dependencies:** OWN-01

**Evidence set:** [ownership](evidence/validate_ownership/) (see [replay/provenance index](evidence/README.md)).

## OWN-07

**Optimized-vector hashes omit all but first and last eight scalar values.**

**P2 · performance · confirmed-reproduced · finding**

**Exposure:** all four public optimized vector types; no vector-keyed downstream table found

**Original references:** PERF3, V5

**Sources:** [Sources/VectorCore/Vectors/Vector384Optimized.swift:334](../../../Sources/VectorCore/Vectors/Vector384Optimized.swift:334); [Sources/VectorCore/Vectors/Vector512Optimized.swift:334](../../../Sources/VectorCore/Vectors/Vector512Optimized.swift:334); [Sources/VectorCore/Vectors/Vector768Optimized.swift:334](../../../Sources/VectorCore/Vectors/Vector768Optimized.swift:334); [Sources/VectorCore/Vectors/Vector1536Optimized.swift:334](../../../Sources/VectorCore/Vectors/Vector1536Optimized.swift:334)

**Evidence:** Fresh Release across all four widths: interior-distinct values have identical hashes; signed-zero equality and hash both agree. Source combines exactly four SIMD4 values, giving deterministic same hash for all interior-only variation.

**Correction / limit:** Legal Hashable collisions are a scaling issue, not equal-values-have-equal-hash violation. O(N^2*d) is worst-case equality work for sufficiently late differences, not measured universal cost. SIMD signed zero is not an additional bug.

**Improvement:** Hash all equality-significant coordinates or choose explicit identity keys for consumers; repair cache exact identity separately.

**Acceptance:** Large structured/sparse interior-varying corpus must not deterministically collapse; equal values including signed zeros retain equal hashes. Coordinate Release insertion/lookup scaling measurement before speed claims.

**Evidence set:** [ownership](evidence/validate_ownership/) (see [replay/provenance index](evidence/README.md)).

## OWN-08

**Pointer normalization documents16-byte alignment as preferred but SIMD loads require it.**

**P2 · code quality · confirmed-reproduced · finding**

**Exposure:** public NormalizeKernels.normalizeUnchecked pointer overload

**Original references:** C4, SIMD finding2

**Sources:** [Sources/VectorCore/Operations/Kernels/NormalizeKernels.swift:419](../../../Sources/VectorCore/Operations/Kernels/NormalizeKernels.swift:419); [Sources/VectorCore/Operations/Kernels/NormalizeKernels.swift:439](../../../Sources/VectorCore/Operations/Kernels/NormalizeKernels.swift:439)

**Evidence:** Actual-library fresh Debug offset+4 from64-byte allocation traps at dimensions4,5,16,17: load from misaligned raw pointer. Dimension3 succeeds. Same actual Release fixtures all succeed on this ARM host.

**Correction / limit:** Report Debug trap and source alignment precondition violation, not a reproduced Release failure. Dimension<4 does not enter SIMD loads. Caller meets documented Float capacity/nonzero magnitude obligations.

**Improvement:** Use loadUnaligned or scalar/gather loads, or explicitly enforce and document mandatory SIMD alignment. A documentation-only compatibility tightening must be deliberate.

**Acceptance:** Offsets0,4,8,12 and dimensions0...5,15...17 in Debug/Release either compute correct finite unit values under supported inputs or reject consistently at a documented boundary; retain numerical reviewer coverage.

**Evidence set:** [ownership](evidence/validate_ownership/) (see [replay/provenance index](evidence/README.md)).

## OWN-09

**Vector384Optimized permanently rebinds typed-owner memory where other widths restore binding.**

**P2 · code quality · qualified · finding**

**Exposure:** normal public construction/toArray/buffer access

**Original references:** C17, V4

**Sources:** [Sources/VectorCore/Vectors/Vector384Optimized.swift:69](../../../Sources/VectorCore/Vectors/Vector384Optimized.swift:69); [Sources/VectorCore/Vectors/Vector384Optimized.swift:143](../../../Sources/VectorCore/Vectors/Vector384Optimized.swift:143); [Sources/VectorCore/Vectors/Vector384Optimized.swift:156](../../../Sources/VectorCore/Vectors/Vector384Optimized.swift:156); [Sources/VectorCore/Vectors/Vector384Optimized.swift:168](../../../Sources/VectorCore/Vectors/Vector384Optimized.swift:168)

**Evidence:** Source comparison finds four permanent bindMemory sites versus scoped withMemoryRebound in512/768/1536. Swift UnsafeRawPointer migration guide and SE-0107 require later typed accesses to agree with current binding and describe restoring temporary bindings. No optimizer miscompile is reproduced.

**Correction / limit:** Keep as binding-lifetime maintenance/contract finding. Layout compatibility is not a blanket strict-aliasing exemption; conversely merely seeing different spellings does not prove an observed miscompile. Alignment of Float-array source must be audited separately across all widths.

**Improvement:** Use one reviewed conversion/borrow strategy across widths, restoring ownership binding; do not treat replacing the spelling alone as an alignment proof.

**Acceptance:** Mechanical audit of typed-owner rebinding, scoped nonescape and alignment obligations plus constructor/read/write/COW Release round trips for all widths. Document the permitted type/layout model.

**Evidence set:** [ownership](evidence/validate_ownership/) (see [replay/provenance index](evidence/README.md)).

## OWN-10

**Compact generic storage does not destroy manually initialized ManagedBuffer tail elements.**

**P2 · code quality · confirmed-reproduced · finding**

**Exposure:** public TriangularMatrix<T>, default compact sizes4...32

**Original references:** MS-4

**Sources:** [Sources/VectorCore/Utilities/TieredStorage.swift:380](../../../Sources/VectorCore/Utilities/TieredStorage.swift:380); [Sources/VectorCore/Utilities/TieredStorage.swift:406](../../../Sources/VectorCore/Utilities/TieredStorage.swift:406); [Sources/VectorCore/Utilities/TieredStorage.swift:418](../../../Sources/VectorCore/Utilities/TieredStorage.swift:418)

**Evidence:** Fresh Release weak sentinel retained after matrices n4/32 leave scope; controls n3/33/65 release. CompactBuffer has no tail deinit; constructor and copy initialize count tail elements.

**Correction / limit:** Reference-bearing arrays/strings follow missing generic destruction by source; only class sentinel was executed. A ManagedBuffer frees allocation but does not implicitly destroy manually initialized elements.

**Improvement:** Destroy exactly initialized tail count at actual owner lifetime end; fix ownership sharing before choosing where deinit belongs.

**Acceptance:** Deinit/weak sentinel tests for all tiers, compact copy/detach/replacement/destruction and thrown access; each logical initialized reference released exactly once, no early destruction.

**Dependencies:** OWN-05

**Evidence set:** [ownership](evidence/validate_ownership/) (see [replay/provenance index](evidence/README.md)).

## OWN-11

**alignedCopy has incompatible initialize versus update semantics across alignment branches.**

**P2 · code quality · confirmed-reproduced · finding**

**Exposure:** public generic utility; no production caller found

**Original references:** C17, MS-5

**Sources:** [Sources/VectorCore/Utilities/MemoryUtilities.swift:135](../../../Sources/VectorCore/Utilities/MemoryUtilities.swift:135)

**Evidence:** Original exact-source Token fixture inspected and replayed: old initialized destination leaks at offset0, releases at naturally aligned offset1. Code initialize(from:) vs destination[i]=source[i] admits no single destination-state contract for nontrivial T.

**Correction / limit:** This is an unsafe API contract gap, not proof of a fault under a clearly documented supported destination state (none is supplied). Uninitialized-destination crash was not executed. Alignment here means preferred64-byte alignment, not natural-type misalignment.

**Improvement:** Specify initialize or update semantics uniformly, preferably separate explicitly named APIs; separately document overlap.

**Acceptance:** Class/deinit and nontrivial-value tests at preferred and natural-only alignment, using only the chosen destination state. Identical lifetime behavior in both branches; overlap behavior explicitly tested if supported.

**Evidence set:** [ownership](evidence/validate_ownership/) (see [replay/provenance index](evidence/README.md)).

## OWN-12

**HybridStorage repeating construction/fill performs quadratic buffer traffic.**

**P2 · performance · confirmed-source · finding**

**Exposure:** DynamicVector repeating/fill/resize above16 Float elements; explicit hybrid heap storage

**Original references:** PERF1, MS-2

**Sources:** [Sources/VectorCore/Storage/HybridStorage.swift:125](../../../Sources/VectorCore/Storage/HybridStorage.swift:125); [Sources/VectorCore/Storage/HybridStorage.swift:173](../../../Sources/VectorCore/Storage/HybridStorage.swift:173); [Sources/VectorCore/Storage/HybridStorage.swift:224](../../../Sources/VectorCore/Storage/HybridStorage.swift:224); [Sources/VectorCore/Vectors/DynamicVector.swift:38](../../../Sources/VectorCore/Vectors/DynamicVector.swift:38); [Sources/VectorCore/Vectors/DynamicVector.swift:139](../../../Sources/VectorCore/Vectors/DynamicVector.swift:139)

**Evidence:** Removed only integer counters from original COW fixture and verified full current source bodies occur byte-for-byte for UnifiedStorage,HybridStorage,DimensionStorage,VectorStorage. Replayed -O counter results: repeating512 514 normal+512 copy allocations=1026. Actual linked Release specialized ensureUnique and setter each retain before checking uniqueness, then allocate/copy; actual fill loops these paths. Excerpts retained.

**Correction / limit:** Exact count1026 belongs to the instrumented fixture; actual Release instruction path independently proves duplicated ownership/copying and quadratic element traffic. Pointer addresses staying unchanged after writes do not refute copying: two replacement allocations can reuse prior address. SIMD stubs are not used to claim numerical results. Generic DimensionStorage default heap mutation does not inherit this finding.

**Improvement:** Remove redundant outer COW ownership, mutate via exclusive ownership and fill once through a buffer. Preserve value semantics. Audit explicit DimensionStorage(.hybrid) as a dependency, separately from default heap.

**Acceptance:** Optimized allocation-scaling checks at Float16/17 and representative384/512/768/1536; unique scalar writes/fill stay bounded, shared first mutation detaches once, subsequent unique writes do not copy. No wall-clock speedup asserted until coordinated measurement.

**Dependencies:** OWN-05, OWN-14

**Evidence set:** [ownership](evidence/validate_ownership/) (see [replay/provenance index](evidence/README.md)).

## OWN-13

**TriangularMatrix scalar writes copy complete standard/aligned allocations even when uniquely owned.**

**P2 · performance · confirmed-reproduced · finding**

**Exposure:** public default matrix n>=33; original emphasized aligned n>=65

**Original references:** PERF1, MS-3

**Sources:** [Sources/VectorCore/Utilities/TieredStorage.swift:160](../../../Sources/VectorCore/Utilities/TieredStorage.swift:160); [Sources/VectorCore/Utilities/TieredStorage.swift:181](../../../Sources/VectorCore/Utilities/TieredStorage.swift:181); [Sources/VectorCore/Utilities/TieredStorage.swift:581](../../../Sources/VectorCore/Utilities/TieredStorage.swift:581); [Sources/VectorCore/Utilities/TriangularMatrix.swift:225](../../../Sources/VectorCore/Utilities/TriangularMatrix.swift:225); [Sources/VectorCore/Utilities/TriangularMatrix.swift:258](../../../Sources/VectorCore/Utilities/TriangularMatrix.swift:258)

**Evidence:** Fresh actual Release:10 unique writes cause10 allocation-address changes at n33,64,65. Original exact-source aligned-copy counter replay gives10 copies at65. Standard setter extracts ContiguousArray and mutates while enum still owns it; aligned uniqueness method retains self. Full m-element copy per scalar gives O(m^2)=O(n^4) construction traffic.

**Correction / limit:** Extend original exposure: standard tier513...2048 also detaches on scalar mutation, so repairing only AlignedBuffer.isUniquelyReferenced is insufficient. Address-change evidence uses integer addresses outside borrow scopes, not retained storage copies.

**Improvement:** Fix compact/aligned ownership checks and standard enum extraction; add bulk construction after dormant bulk correctness is repaired.

**Acceptance:** Across all tier boundaries unique writes perform no full-buffer detaches; retained copy produces exactly one detach and remains unchanged. Construction traffic scales linearly with number of stored elements. Benchmark only after allocation and value-semantic checks.

**Dependencies:** OWN-05, OWN-14

**Evidence set:** [ownership](evidence/validate_ownership/) (see [replay/provenance index](evidence/README.md)).

## OWN-14

**Dormant TieredStorage bulk access loses standard writes and inline writes on thrown closure exit.**

**P3 · code quality · confirmed-reproduced · finding**

**Exposure:** internal dormant mutable-buffer route; no production caller found

**Original references:** MS-6 standard/throwing bulk

**Sources:** [Sources/VectorCore/Utilities/TieredStorage.swift:218](../../../Sources/VectorCore/Utilities/TieredStorage.swift:218); [Sources/VectorCore/Utilities/TieredStorage.swift:224](../../../Sources/VectorCore/Utilities/TieredStorage.swift:224); [Sources/VectorCore/Utilities/TieredStorage.swift:365](../../../Sources/VectorCore/Utilities/TieredStorage.swift:365)

**Evidence:** Fresh actual Release throwing closure writes19 then throws: counts1,4,513,2048 remain1; compact5/512 and aligned2049 retain19. Original ordinary-closure exact-source replay confirms standard513/2048 lose writes even without throw.

**Correction / limit:** No current public matrix method routes through this bulk method. It becomes a required dependency if the proposed matrix bulk-fill optimization is introduced.

**Improvement:** Persist modified standard payload with defer; make inline inner and outer copyback consistent on throws, or define a uniform rollback contract across every tier.

**Acceptance:** Normal and throwing closure tests at0,1,4,5,512,513,2048,2049 with retained original copies, consistent mutation policy and independent value semantics.

**Dependencies:** OWN-05

**Evidence set:** [ownership](evidence/validate_ownership/) (see [replay/provenance index](evidence/README.md)).

## OWN-15

**Internal StorageSlice returns scoped pointer without keeping its owner alive.**

**P3 · code quality · confirmed-source · finding**

**Exposure:** internal dormant APIs; no production call found

**Original references:** MS-6 StorageSlice

**Sources:** [Sources/VectorCore/Storage/UnifiedStorage.swift:280](../../../Sources/VectorCore/Storage/UnifiedStorage.swift:280); [Sources/VectorCore/Storage/DimensionStorage.swift:298](../../../Sources/VectorCore/Storage/DimensionStorage.swift:298); [Sources/VectorCore/Storage/UnifiedStorage.swift:259](../../../Sources/VectorCore/Storage/UnifiedStorage.swift:259)

**Evidence:** Slice only stores pointer/count returned from withUnsafeBufferPointer. Managed slice has no owner reference; inline DimensionStorage borrows a local tuple temporary. Current source call search finds declarations and forwarding only.

**Correction / limit:** No supported-use UAF executed. This is distinct from DynamicVector.slice, which returns an owned vector.

**Improvement:** Remove unused API or use scoped slice closures; an owning slice must hold storage and define snapshot/COW behavior, including inline storage promotion.

**Acceptance:** Compile-time nonescape design or scoped lifetime tests for inline/heap slices; no pointer survives its borrow. If retaining owner, prove owner retention and copy isolation.

**Evidence set:** [ownership](evidence/validate_ownership/) (see [replay/provenance index](evidence/README.md)).

## OWN-16

**Experimental consuming into() returns memory still freed by deinit.**

**P3 · code quality · confirmed-source · finding**

**Exposure:** internal compile gate MOVE_ONLY_EXPERIMENTAL absent from package settings

**Original references:** MS-6 MoveOnlyStorage

**Sources:** [Sources/VectorCore/Storage/UnifiedStorage.swift:10](../../../Sources/VectorCore/Storage/UnifiedStorage.swift:10); [Sources/VectorCore/Storage/UnifiedStorage.swift:36](../../../Sources/VectorCore/Storage/UnifiedStorage.swift:36); [Sources/VectorCore/Storage/UnifiedStorage.swift:66](../../../Sources/VectorCore/Storage/UnifiedStorage.swift:66)

**Evidence:** into returns the stored pointer, while unconditional deinit deallocates it; no transferred-state guard or deinit suppression is present.

**Correction / limit:** Not compiled in ordinary builds and not runtime-reproduced. Do not enable this gate as an optimization without rederiving all lifetime paths.

**Improvement:** Remove dormant experimental route or implement a real consuming transfer with exactly-once ownership handoff.

**Acceptance:** Gate-enabled separate fixture proves pointer survives consumed wrapper, is freed once by recipient, and nontransferred wrapper still frees correctly; audit every newly compiled branch.

**Evidence set:** [ownership](evidence/validate_ownership/) (see [replay/provenance index](evidence/README.md)).

## OWN-17

**Public FP16 pointer getters escape ContiguousArray borrow scopes.**

**P2 · code quality · qualified · finding**

**Exposure:** public opt-in fp16Storage getters; kernels use scoped borrows

**Original references:** RP additional: ownership contract gap

**Sources:** [Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:344](../../../Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:344); [Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:387](../../../Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:387); [Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:430](../../../Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:430)

**Evidence:** All three getters return baseAddress from withUnsafeBufferPointer. Returned UnsafePointer has no lifetime token; temporary owner expressions can end storage lifetime. Source traced; no UAF executed.

**Correction / limit:** The getter is explicitly unsafe; do not equate pointer return with a demonstrated supported-use corruption. Retaining the array owner alone does not change the documented nonescape condition of withUnsafeBufferPointer.

**Improvement:** Provide scoped FP16 storage access; deprecate or precisely document any persistent pointer API with owned allocation design.

**Acceptance:** All direct kernels and callers use scoped borrows or explicit owned allocation lifetime. Temporary-expression use cannot silently escape through new safe API. Test512/768/1536 content and owner lifetime.

**Evidence set:** [ownership](evidence/validate_ownership/) (see [replay/provenance index](evidence/README.md)).

## OWN-18

**SoAFP16Cache512 accepts nonpositive maxSize then evicts from an empty list.**

**P3 · code quality · confirmed-source · finding**

**Exposure:** public explicit cache configuration; default100 unaffected

**Original references:** RP additional: cache capacity

**Sources:** [Sources/VectorCore/Operations/Kernels/SoAFP16Cache.swift:85](../../../Sources/VectorCore/Operations/Kernels/SoAFP16Cache.swift:85); [Sources/VectorCore/Operations/Kernels/SoAFP16Cache.swift:133](../../../Sources/VectorCore/Operations/Kernels/SoAFP16Cache.swift:133)

**Evidence:** With size0 and empty cache, cache.count>=maxSize is true and accessOrder.removeFirst traps. Same proof holds negative capacity; original zero subprocess reproduced by root/reduced-precision reviewer.

**Correction / limit:** Lower priority configuration boundary. Choose whether zero intentionally disables caching before testing desired policy.

**Improvement:** Precondition positive capacity or implement zero retention policy; reject negative.

**Acceptance:** Fresh empty/nonempty stores at negative,zero,one capacities have documented behavior, repeated eviction and statistics remain consistent.

**Evidence set:** [ownership](evidence/validate_ownership/) (see [replay/provenance index](evidence/README.md)).

## OWN-19

**FP16 cache caps entry count while payload bytes scale with candidates per entry.**

**P2 · performance · confirmed-source · opportunity**

**Exposure:** public opt-in512 cache

**Original references:** PERF8 FP16 cache, RP additional: retention

**Sources:** [Sources/VectorCore/Operations/Kernels/SoAFP16Cache.swift:49](../../../Sources/VectorCore/Operations/Kernels/SoAFP16Cache.swift:49); [Sources/VectorCore/Operations/Kernels/SoAFP16Cache.swift:75](../../../Sources/VectorCore/Operations/Kernels/SoAFP16Cache.swift:75); [Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:468](../../../Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:468)

**Evidence:** 512*2*N=1024N logical payload bytes per entry;100 million-candidate entries yield102400000000 bytes before padding/keys. Comment1KB per100 vectors understates logical100KiB by100x. Key map is O(N); accessOrder firstIndex/remove O(entry count).

**Correction / limit:** Sizing derivation, not attempted allocation/RSS measurement; immutable COW cache values are not themselves an actor data race.

**Improvement:** Byte admission/eviction budget and byte statistics; correct size docs and account for metadata/padding.

**Acceptance:** Tests with small byte budget cap retained payload; oversized entries follow explicit policy; statistics and eviction accounting agree with logical/allocated sizes. No claim of measured RSS.

**Dependencies:** OWN-04

**Evidence set:** [ownership](evidence/validate_ownership/) (see [replay/provenance index](evidence/README.md)).

## OWN-20

**StaticDimension.Storage customization is unused by Vector<D> storage yet controls extension availability.**

**P3 · code quality · confirmed-source · opportunity**

**Exposure:** public custom dimensions

**Original references:** vectors architecture: StaticDimension.Storage

**Sources:** [Sources/VectorCore/Core/Dimension.swift:14](../../../Sources/VectorCore/Core/Dimension.swift:14); [Sources/VectorCore/Vectors/Vector.swift:15](../../../Sources/VectorCore/Vectors/Vector.swift:15); [Sources/VectorCore/Operations/VectorCoreOperations.swift:10](../../../Sources/VectorCore/Operations/VectorCoreOperations.swift:10)

**Evidence:** Vector<D>.Storage explicitly aliases DimensionStorage<D,Float>; D.Storage is independently constrained by operation extensions.

**Correction / limit:** Confusing customization/API gating, not demonstrated numerical failure.

**Improvement:** Document/remove dead customization or deliberately wire it through with compatibility review.

**Acceptance:** Custom dimension conformance fixtures demonstrate actual storage and intended available operations; no advertised customization point has no effect.

**Evidence set:** [ownership](evidence/validate_ownership/) (see [replay/provenance index](evidence/README.md)).

## OWN-21

**VectorTypeFactory optimized wording describes generic fixed-vector selection, not specialized concrete types.**

**P3 · code quality · confirmed-source · opportunity**

**Exposure:** public factory

**Original references:** vectors architecture: factory routing

**Sources:** [Sources/VectorCore/Factory/VectorTypeFactory.swift:28](../../../Sources/VectorCore/Factory/VectorTypeFactory.swift:28); [Sources/VectorCore/Factory/VectorTypeFactory.swift:51](../../../Sources/VectorCore/Factory/VectorTypeFactory.swift:51)

**Evidence:** Factory switch constructs Vector<Dim*> for supported sizes; existing tests intentionally require these concrete outputs.

**Correction / limit:** No measured performance regression. Changing the output type can alter casts, provider routing and generic overload behavior.

**Improvement:** Document actual outputs and benchmark target workloads before deciding strategy; preserve compatibility or explicitly version change.

**Acceptance:** Concrete-output and routing tests remain deliberate; any strategy change reruns all reachable kernels and consumer casts at every supported dimension.

**Evidence set:** [ownership](evidence/validate_ownership/) (see [replay/provenance index](evidence/README.md)).

## OWN-22

**VectorCoreOperations creates redundant arrays and temporary results.**

**P3 · performance · confirmed-source · opportunity**

**Exposure:** public min/max/clamp operations

**Original references:** vectors architecture: allocations

**Sources:** [Sources/VectorCore/Operations/VectorCoreOperations.swift:17](../../../Sources/VectorCore/Operations/VectorCoreOperations.swift:17); [Sources/VectorCore/Operations/VectorCoreOperations.swift:196](../../../Sources/VectorCore/Operations/VectorCoreOperations.swift:196); [Sources/VectorCore/Operations/VectorCoreOperations.swift:215](../../../Sources/VectorCore/Operations/VectorCoreOperations.swift:215); [Sources/VectorCore/Operations/VectorCoreOperations.swift:233](../../../Sources/VectorCore/Operations/VectorCoreOperations.swift:233)

**Evidence:** Source constructs input arrays, provider result and owned vector; Dynamic min/max/clamp initially allocate zero result then replace storage.

**Correction / limit:** Source-visible allocation opportunities; no measured regression or speedup. Constructor cost can additionally be affected by HybridStorage finding.

**Improvement:** Eliminate unnecessary initial result allocation, then consider scoped provider APIs with shape guarantees.

**Acceptance:** Allocation records for representative dimensions show fewer temporaries without changing provider call counts, shape checks or outputs.

**Dependencies:** OWN-12

**Evidence set:** [ownership](evidence/validate_ownership/) (see [replay/provenance index](evidence/README.md)).

## OWN-23

**Performance/robustness comments exceed evidence and overlapping storage layers obscure actual ownership.**

**P3 · code quality · qualified · opportunity**

**Exposure:** public docs and maintainers

**Original references:** vectors architecture: claims, memory strengths: ownership layers

**Sources:** [Sources/VectorCore/Vectors/Vector512Optimized.swift:15](../../../Sources/VectorCore/Vectors/Vector512Optimized.swift:15); [Sources/VectorCore/Vectors/Vector512Optimized.swift:180](../../../Sources/VectorCore/Vectors/Vector512Optimized.swift:180); [Sources/VectorCore/Vectors/Vector512Optimized.swift:213](../../../Sources/VectorCore/Vectors/Vector512Optimized.swift:213); [Sources/VectorCore/Storage/COWDynamicStorage.swift:1](../../../Sources/VectorCore/Storage/COWDynamicStorage.swift:1); [Sources/VectorCore/Vectors/DynamicVector.swift:15](../../../Sources/VectorCore/Vectors/DynamicVector.swift:15)

**Evidence:** Fixed nanosecond/speedup assertions have no nearby hardware/revision measurements; stable magnitude squared still cannot represent a mathematically overflowing square. DynamicVector uses HybridStorage, despite older COWDynamicStorage descriptions.

**Correction / limit:** General consolidation is an opportunity, not authorization for a wholesale rewrite or proof of universal slowness. Norm arithmetic validation belongs numerical reviewer.

**Improvement:** Tie retained claims to tests or dated measurements, correct backend docs, consolidate after ownership and compatibility contracts are pinned.

**Acceptance:** Mechanical claim inventory names evidence for each retained correctness/performance assertion; docs reflect actual call graph; no unsupported universal assertion remains.

**Dependencies:** OWN-12, OWN-13

**Evidence set:** [ownership](evidence/validate_ownership/) (see [replay/provenance index](evidence/README.md)).

## OWN-24

**SoA/page/raw ownership, size and custom-conformer obligations are already documented.**

**Context · code quality · qualified · context**

**Exposure:** unsafe borrowed/transfer/custom conformer surfaces

**Original references:** MS documented: SoA transfer, MS documented: unchecked Sendable, MS documented: arithmetic/alignment obligations, MS documented: custom conformers

**Sources:** [Docs/Memory_Alignment.md:63](../../../Docs/Memory_Alignment.md:63); [Docs/Memory_Alignment.md:94](../../../Docs/Memory_Alignment.md:94); [Docs/Memory_Alignment.md:132](../../../Docs/Memory_Alignment.md:132); [Docs/SoA_Layout_Contract.md:105](../../../Docs/SoA_Layout_Contract.md:105); [Sources/VectorCore/Storage/SoA.swift:1](../../../Sources/VectorCore/Storage/SoA.swift:1)

**Evidence:** Re-read ownership docs: consuming invalidates CPU access, retaining owner does not prohibit another consume, callers synchronize, counts/type alignment must be valid. Built-ins use full SIMD4 lanes; custom layout consistency belongs conformers.

**Correction / limit:** No additional supported-use corruption established from missing post-transfer checks, @unchecked Sendable, or ordinary checked arithmetic trapping on invalid/oversized shapes. Four built-in layouts do not prove arbitrary custom conformers.

**Improvement:** Preserve explicit contracts; optional checked wrappers and immutable ownership APIs can reduce misuse.

**Acceptance:** Transfer identity, logical vs allocated length and frozen layout tests remain; external conformance examples assert required lane/dimension/storage consistency. New checked wrappers validate before arithmetic.

**Evidence set:** [ownership](evidence/validate_ownership/) (see [replay/provenance index](evidence/README.md)).

## OWN-25

**AlignedDynamicArrayStorage fallback can weaken alignment while guaranteedAlignment reports the request.**

**P3 · code quality · unverified · investigation**

**Exposure:** public allocation-failure fallback

**Original references:** MS documented: fallback alignment

**Sources:** [Sources/VectorCore/Storage/AlignedDynamicArrayStorage.swift:85](../../../Sources/VectorCore/Storage/AlignedDynamicArrayStorage.swift:85); [Sources/VectorCore/Storage/AlignedDynamicArrayStorage.swift:180](../../../Sources/VectorCore/Storage/AlignedDynamicArrayStorage.swift:180)

**Evidence:** Catch/fallback allocates with MemoryLayout<Float>.alignment while property returns requested alignment. No realistic allocation failure was induced; natural-alignment request does not force an actually misaligned pointer.

**Correction / limit:** Keep a potential guarantee gap, not reproduced misalignment or corruption.

**Improvement:** Preserve requested alignment or report verified actual guarantee; define whether allocation failure should propagate.

**Acceptance:** Inject allocator failure and validate reported guarantee against returned pointer for several requested alignments; callers never rely on stronger alignment than provided.

**Evidence set:** [ownership](evidence/validate_ownership/) (see [replay/provenance index](evidence/README.md)).

## OWN-26

**Reject extending HybridStorage quadratic-copy behavior to default generic DimensionStorage without evidence.**

**Context · code quality · rejected · context**

**Exposure:** generic Vector<D> default storage; explicit .hybrid separately exposed

**Original references:** MS-2 rejected DimensionStorage extrapolation, vectors architecture: convenience factories

**Sources:** [Sources/VectorCore/Storage/DimensionStorage.swift:36](../../../Sources/VectorCore/Storage/DimensionStorage.swift:36); [Sources/VectorCore/Storage/DimensionStorage.swift:127](../../../Sources/VectorCore/Storage/DimensionStorage.swift:127); [Sources/VectorCore/Storage/DimensionStorage.swift:256](../../../Sources/VectorCore/Storage/DimensionStorage.swift:256)

**Evidence:** Replayed optimized source fixture shows DimensionStorage<D512>10 writes0 allocations/copies and buffer mutation0; uniqueness reporting getter nonetheless returns false for unique heap instance. Default mode uses managed heap above16.

**Correction / limit:** Reject blanket claim that linspace/range/generate/geometric/sparse necessarily become quadratic for every generic fixed vector. Explicit .hybrid mode and Double inline threshold deserve dedicated coverage.

**Improvement:** Keep truthful uniqueness reporting if public diagnostic is intended; include factories in class-wide optimized allocation tests rather than assume costs.

**Acceptance:** Default and explicit modes at Float16/17 and Double8/9/16/17; constructor,factory,scalar and buffer mutation counts measured in Release with retained-copy controls.

**Dependencies:** OWN-12

**Evidence set:** [ownership](evidence/validate_ownership/) (see [replay/provenance index](evidence/README.md)).

## OWN-27

**Extension-only protocol methods select static defaults in generic code.**

**P3 · architecture · confirmed-source · opportunity**

**Exposure:** public generic VectorProtocol consumers

**Original references:** vectors architecture: extension dispatch, A10 vector portion

**Sources:** [Sources/VectorCore/Protocols/VectorProtocol.swift:181](../../../Sources/VectorCore/Protocols/VectorProtocol.swift:181); [Sources/VectorCore/Protocols/VectorProtocol.swift:402](../../../Sources/VectorCore/Protocols/VectorProtocol.swift:402); [Sources/VectorCore/Protocols/VectorFactory.swift:1](../../../Sources/VectorCore/Protocols/VectorFactory.swift:1)

**Evidence:** Fresh ShapeRoutes generic +=/-= uses unchecked default; numerical reviewer owns normalized direct/generic discrepancy. Methods not protocol requirements cannot supply dynamic witnesses.

**Correction / limit:** This alone is not a measured performance defect and does not mean all extensions are wrong. Split narrow read-only protocols only where consumer needs justify change.

**Improvement:** Make intended override points requirements and explicitly document remaining static helpers; test route matrix before consolidation.

**Acceptance:** Concrete/generic/existential compatible calls obey same documented contract; spy/conformance fixtures identify selected customization points.

**Dependencies:** OWN-06

**Evidence set:** [ownership](evidence/validate_ownership/) (see [replay/provenance index](evidence/README.md)).

## SER-01

**A valid empty DynamicVector binary encoding cannot be decoded by its own type.**

**P2 · code quality · confirmed-reproduced · finding**

**Exposure:** Public binary round trip of default/empty dynamic values.

**Original references:** C20, serialization:P2 empty DynamicVector

**Sources:** [Sources/VectorCore/Vectors/DynamicVector.swift:49](../../../Sources/VectorCore/Vectors/DynamicVector.swift:49); [Sources/VectorCore/Vectors/VectorSerialization.swift:61](../../../Sources/VectorCore/Vectors/VectorSerialization.swift:61); [Sources/VectorCore/Utilities/BinaryFormat.swift:164](../../../Sources/VectorCore/Utilities/BinaryFormat.swift:164); [Tests/ComprehensiveTests/SerializationEdgeCaseTests.swift:22](../../../Tests/ComprehensiveTests/SerializationEdgeCaseTests.swift:22)

**Evidence:** Fresh Release SerializationProbe: dimension0 emits16 bytes, binary decode throws invalidDimension, JSON roundtrip restores0. Named empty test substitutes a one-element value.

**Correction / limit:** The 10001 cap is a distinct intentionally tested limitation. Do not remove decoder limits as an incidental repair.

**Improvement:** Choose and document empty binary semantics; accept a valid empty frame for DynamicVector or expose checked encoding rejection.

**Acceptance:** Real default/last-element-removal empty roundtrips or explicit checked rejection; fixed-size decoders must retain dimension checks; CRC/size failures remain rejected.

**Evidence set:** [tooling-serialization](evidence/root/) (see [replay/provenance index](evidence/README.md)).

## SER-02

**Serialization tests include 15 comment-only bodies and mislabeled optimized-type coverage.**

**P2 · code quality · confirmed-reproduced · finding**

**Exposure:** Confidence in passing tests.

**Original references:** serialization:P3 coverage, serialization:golden fixtures

**Sources:** [Tests/ComprehensiveTests/SerializationEdgeCaseTests.swift:611](../../../Tests/ComprehensiveTests/SerializationEdgeCaseTests.swift:611); [Tests/ComprehensiveTests/SerializationEdgeCaseTests.swift:641](../../../Tests/ComprehensiveTests/SerializationEdgeCaseTests.swift:641); [Tests/ComprehensiveTests/SerializationEdgeCaseTests.swift:707](../../../Tests/ComprehensiveTests/SerializationEdgeCaseTests.swift:707)

**Evidence:** Fresh mechanical count saved in placeholder-tests.json:15 empty bodies after comments removed. Three optimized-named cases instantiate generic Vector and encode extracted arrays. Fixture placeholders contain no independent bytes.

**Correction / limit:** This does not mean all optimized Codable code is untested elsewhere, or that Core should implement every feature mentioned by a placeholder.

**Improvement:** Replace misleading active placeholders with real tests or future-work notes; exercise actual concrete Codable and add independent golden byte fixtures.

**Acceptance:** Every active test named for a behavior must execute/assert that behavior. Include signed-zero/NaN payload binary bits, direct sliced Data, valid/invalid CRC and exact frame lengths.

**Dependencies:** SER-01

**Evidence set:** [tooling-serialization](evidence/root/) (see [replay/provenance index](evidence/README.md)).

## SER-03

**Binary conversion has scalar loops and temporary representations; a Release allocation speedup is not yet established.**

**P3 · performance · qualified · opportunity**

**Exposure:** Binary encode/decode.

**Original references:** serialization:scalar-at-a-time binary conversion

**Sources:** [Sources/VectorCore/Utilities/BinaryFormat.swift:109](../../../Sources/VectorCore/Utilities/BinaryFormat.swift:109); [Sources/VectorCore/Utilities/BinaryFormat.swift:204](../../../Sources/VectorCore/Utilities/BinaryFormat.swift:204); [Sources/VectorCore/Vectors/VectorSerialization.swift:71](../../../Sources/VectorCore/Vectors/VectorSerialization.swift:71)

**Evidence:** Source defines scratch-allocating generic unaligned-load extensions and per-scalar decode; encoding materializes toArray then appends scalars. Overload selection/optimization may eliminate the apparent scratch allocation for concrete UInt32.

**Correction / limit:** Retain only a profiling candidate. Neither one heap allocation per scalar in Release nor a throughput regression was demonstrated; do not blindly replace the implementation.

**Improvement:** Inspect concrete generated code and allocation traces; assess scoped/bulk paths preserving unaligned input, endian conversion and exact bit patterns.

**Acceptance:** Before/after Release measurements with allocation counts, CRC included, representative dimensions and unaligned slices; byte-for-byte equality against independent fixtures.

**Dependencies:** SER-02, TOOL-07

**Evidence set:** [tooling-serialization](evidence/root/) (see [replay/provenance index](evidence/README.md)).

## SER-04

**Binary and JSON representations have different type/precision contracts.**

**P3 · architecture · qualified · opportunity**

**Exposure:** Serialization of optimized FP32 and quantized vectors.

**Original references:** serialization:Representation and API clarity:INT8, serialization:JSON schemas

**Sources:** [Sources/VectorCore/Vectors/VectorSerialization.swift:61](../../../Sources/VectorCore/Vectors/VectorSerialization.swift:61); [Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:148](../../../Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:148); [Sources/VectorCore/Vectors/VectorSerialization.swift:12](../../../Sources/VectorCore/Vectors/VectorSerialization.swift:12)

**Evidence:** Generic binary encoder writes dequantized FP32 for INT8 (16+4N bytes); same-concrete binary decoders exist for generic/dynamic only. JSON uses distinct schemas: generic object, dynamic dimension+elements, optimized array, synthesized INT8 storage+parameters.

**Correction / limit:** No compressed-INT8 binary or universal JSON schema promise found. These are documentation/interoperability decisions, not demonstrated corrupt roundtrips.

**Improvement:** Document value interchange versus storage-preserving encoding, concrete decoder availability and schemas. Add compressed storage framing only if required by consumers.

**Acceptance:** Examples and tests must name concrete source/destination types and expected byte sizes/precision; existing wire format compatibility preserved unless explicitly versioned.

**Evidence set:** [tooling-serialization](evidence/root/) (see [replay/provenance index](evidence/README.md)).

## SER-05

**Vectors expose binary methods without adopting the public binary protocols.**

**P3 · architecture · confirmed-source · opportunity**

**Exposure:** Generic BinaryCodable integrations; no internal dependent caller found.

**Original references:** serialization:Representation and API clarity:protocols

**Sources:** [Sources/VectorCore/Serialization/BinaryProtocols.swift:9](../../../Sources/VectorCore/Serialization/BinaryProtocols.swift:9); [Sources/VectorCore/Vectors/VectorSerialization.swift:61](../../../Sources/VectorCore/Vectors/VectorSerialization.swift:61)

**Evidence:** Repository declaration/reference inventory finds only the documentation example of vector conformance. Method-name coincidence does not declare Swift protocol conformance.

**Correction / limit:** An integration opportunity; no existing downstream generic BinaryCodable compilation failure was claimed.

**Improvement:** Add appropriate conformances or document the protocols as consumer extension points.

**Acceptance:** External generic helper compile test for supported conformances; clearly define types that encode only versus fully roundtrip.

**Evidence set:** [tooling-serialization](evidence/root/) (see [replay/provenance index](evidence/README.md)).

## SER-06

**Header documentation describes VECT while the actual little-endian wire bytes spell TCEV.**

**P3 · code quality · confirmed-reproduced · finding**

**Exposure:** External binary readers/documentation.

**Original references:** serialization:Representation and API clarity:magic

**Sources:** [Sources/VectorCore/Serialization/BinaryProtocols.swift:46](../../../Sources/VectorCore/Serialization/BinaryProtocols.swift:46); [Sources/VectorCore/Utilities/BinaryFormat.swift:23](../../../Sources/VectorCore/Utilities/BinaryFormat.swift:23)

**Evidence:** Fresh binary hex starts54434556. Independent Python struct+zlib generation matches complete encoded header/payload/CRC for six bit-pattern fixtures.

**Correction / limit:** Existing encoder and decoder agree and tests expect actual bytes. This is a documentation defect; changing wire bytes would be a compatibility change.

**Improvement:** Document numeric magic separately from actual bytes and every field endian order.

**Acceptance:** Publish independently specified full golden frame; old frames remain decodable.

**Dependencies:** SER-02

**Evidence set:** [tooling-serialization](evidence/root/) (see [replay/provenance index](evidence/README.md)).

## SER-07

**Decoder framing/CRC bounds and Data-slice behavior support the original limited positive assessment.**

**P3 · code quality · confirmed-reproduced · context**

**Exposure:** Public binary decoder; context/guardrails, not an additional defect.

**Original references:** serialization:Documented limitations, serialization:Strengths

**Sources:** [Sources/VectorCore/Utilities/BinaryFormat.swift:135](../../../Sources/VectorCore/Utilities/BinaryFormat.swift:135); [Sources/VectorCore/Utilities/BinaryFormat.swift:186](../../../Sources/VectorCore/Utilities/BinaryFormat.swift:186); [Sources/VectorCore/Vectors/VectorSerialization.swift:82](../../../Sources/VectorCore/Vectors/VectorSerialization.swift:82)

**Evidence:** Source validates16-byte minimum, magic/version,1...10000 dimension and exact total length before payload allocation, then CRC. Fresh probe independently matches CRC/bit patterns and successfully decodes nonzero-startIndex3 Data slice. Dimension10001 encode/decode asymmetry reproduced and is explicitly expected by existing test.

**Correction / limit:** No public overflow exploit, compression implementation defect or direct-slice bug found. CRC is integrity checking, not authentication. Reserved flags are read but unused; one uncompressed frame is the current contract.

**Improvement:** Preserve checks while fixing empty values; document cap/flags/version/whole-record behavior. Define rejection/version policy before assigning flag semantics.

**Acceptance:** Class-wide malformed framing/CRC and cap boundary tests, plus independent bits/slice regression; do not count absent streaming/compression/migration features as implemented.

**Evidence set:** [tooling-serialization](evidence/root/) (see [replay/provenance index](evidence/README.md)).

## SER-08

**Decoded INT8 storage, quantization parameters and TopK result arrays require invariant validation.**

**P2 · code quality · confirmed-source · context**

**Exposure:** Synthesized/custom Codable construction boundaries.

**Original references:** serialization:Boundary observations

**Sources:** [Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:175](../../../Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:175); [Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:17](../../../Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:17); [Sources/VectorCore/Operations/TopKSelection.swift:651](../../../Sources/VectorCore/Operations/TopKSelection.swift:651)

**Evidence:** Source constructors/Codable can bypass length/derived-field relationships. Ownership and API validators reproduce empty INT8 decoded storage and mismatched TopK lengths; quantization numeric validator owns scale/zero-point semantics.

**Correction / limit:** Cross-reference OWN/NUM/API rows for severity and actual effects; this row is boundary coverage, not a separate issue count.

**Improvement:** Include decoded construction in the same invariant repair as public constructors/mutation.

**Acceptance:** Table-driven invalid shape/derived-parameter/result-pair decoding tests across every representation; valid existing data remains compatible.

**Evidence set:** [tooling-serialization](evidence/root/) (see [replay/provenance index](evidence/README.md)).

## TOOL-01

**Strict benchmark correctness accepts nonfinite errors, mismatched output lengths and missing validation.**

**P1 · code quality · confirmed-reproduced · finding**

**Exposure:** Benchmark CLI; undermines validation of later optimizations.

**Original references:** B1, PERF9

**Sources:** [Benchmarks/VectorCoreBench/Utils/Correctness.swift:15](../../../Benchmarks/VectorCoreBench/Utils/Correctness.swift:15); [Benchmarks/VectorCoreBench/Utils/Correctness.swift:21](../../../Benchmarks/VectorCoreBench/Utils/Correctness.swift:21); [Benchmarks/VectorCoreBench/AppMain.swift:577](../../../Benchmarks/VectorCoreBench/AppMain.swift:577); [Benchmarks/VectorCoreBench/Suites/BatchBench.swift:541](../../../Benchmarks/VectorCoreBench/Suites/BatchBench.swift:541)

**Evidence:** Fresh exact-source tooling-probe.log: scalar NaN yields NaN maximum error and violates=false; vector NaN yields zero maxima; empty output and extra output pass. Source: CLI skips nil correctness; dot/normalization and several optimized batch results omit it. Timed try? provider failures can become zero while reference validation reruns a different kernel.

**Correction / limit:** NaN JSON failure is separate from threshold acceptance. There was no injected faulty production kernel. This is a validation blocker, not a claim every benchmark value is wrong.

**Improvement:** Use explicit validation status, exact expected shape, documented nonfinite policy and case-level validation requirements. Validate the execution path actually timed; surface provider errors.

**Acceptance:** CLI-level table tests: finite matches/mismatches, NaN, infinities, both length mismatch directions, missing validation and provider failure. Each invalid case must fail strict mode and emit a parseable failure artifact.

**Evidence set:** [tooling-serialization](evidence/root/) (see [replay/provenance index](evidence/README.md)).

## TOOL-02

**Requested benchmark output can fail to write while the CLI exits successfully.**

**P2 · code quality · confirmed-reproduced · finding**

**Exposure:** Benchmark automation.

**Original references:** B1:artifact-write, PERF9

**Sources:** [Benchmarks/VectorCoreBench/AppMain.swift:485](../../../Benchmarks/VectorCoreBench/AppMain.swift:485)

**Evidence:** Fresh CLI failed-write.log: output destination is an existing directory, error printed, exit 0. Fresh helper confirms NaN-containing correctness output is rejected by default JSONEncoder.

**Correction / limit:** The CLI experiment used a filesystem error; NaN-to-CLI exit behavior follows the same catch/control flow.

**Improvement:** Treat requested artifact write failure as a failed run; serialize validation failure without nonfinite JSON numbers.

**Acceptance:** Unwritable/invalid destination and invalid numeric statistics must produce nonzero exit and explicit error; successful runs must leave decodable artifacts.

**Dependencies:** TOOL-01

**Evidence set:** [tooling-serialization](evidence/root/) (see [replay/provenance index](evidence/README.md)).

## TOOL-03

**Benchmark metadata claims Release/C/size settings that do not establish the binary executed.**

**P2 · code quality · confirmed-reproduced · finding**

**Exposure:** Debug and Release CLI metadata; provenance.

**Original references:** B2, PERF9

**Sources:** [Benchmarks/VectorCoreBench/AppMain.swift:94](../../../Benchmarks/VectorCoreBench/AppMain.swift:94); [Benchmarks/VectorCoreBench/AppMain.swift:474](../../../Benchmarks/VectorCoreBench/AppMain.swift:474); [Sources/VectorCore/Interop/CKernels.swift:17](../../../Sources/VectorCore/Interop/CKernels.swift:17); [Sources/VectorCoreBenchmarking/EnvCapture.swift:164](../../../Sources/VectorCoreBenchmarking/EnvCapture.swift:164)

**Evidence:** Fresh debug-meta.json records buildConfiguration=release and useCKernels/releaseSize=true. Source references show these CLI options set metadata only; C routing is compile-time. EnvCapture runs git in the working directory and swiftc from PATH.

**Correction / limit:** Requested flags are not proof a C kernel actually ran. No separately compiled C-on/C-off A/B performance measurement was made.

**Improvement:** Record effective compile configuration and executable/source identity separately from requests; use a build driver for build variants.

**Acceptance:** Build Debug/Release and C-on/C-off fixtures; metadata must identify actual settings, and observable route tests must verify enabled C dispatch. Reject incompatible runtime requests.

**Evidence set:** [tooling-serialization](evidence/root/) (see [replay/provenance index](evidence/README.md)).

## TOOL-04

**Extended mixed-precision tests bypass the strict-performance opt-out.**

**P2 · code quality · confirmed-source · finding**

**Exposure:** Manual benchmarks workflow with EXTENDED=1, STRICT_PERF=0.

**Original references:** B3, PERF9

**Sources:** [Tests/ComprehensiveTests/MixedPrecisionPerformanceBenchmark.swift:100](../../../Tests/ComprehensiveTests/MixedPrecisionPerformanceBenchmark.swift:100); [Tests/ComprehensiveTests/MixedPrecisionPerformanceBenchmark.swift:134](../../../Tests/ComprehensiveTests/MixedPrecisionPerformanceBenchmark.swift:134); [Tests/ComprehensiveTests/MixedPrecisionPerformanceBenchmark.swift:227](../../../Tests/ComprehensiveTests/MixedPrecisionPerformanceBenchmark.swift:227); [Tests/ComprehensiveTests/PerfGate.swift:9](../../../Tests/ComprehensiveTests/PerfGate.swift:9); [.github/workflows/benchmarks.yml:33](../../../.github/workflows/benchmarks.yml:33)

**Evidence:** Four speedup assertions are reachable under EXTENDED=1 and never consult strictPerfGatesEnabled. Ordinary baseline did not enable these tests.

**Correction / limit:** No current hosted failure or compiler elimination was observed; universal speedup/dimension-order expectations are unsupported by numerical correctness.

**Improvement:** Move timing assertions into controlled benchmark comparisons or consistently gate them. Consume measured outputs.

**Acceptance:** Mechanically enumerate performance assertions; EXTENDED=1/STRICT_PERF=0 must disable every wall-clock pass/fail gate while retaining numerical checks.

**Evidence set:** [tooling-serialization](evidence/root/) (see [replay/provenance index](evidence/README.md)).

## TOOL-05

**Selecting optimized-only dot cases runs zero cases and succeeds.**

**P2 · code quality · confirmed-reproduced · finding**

**Exposure:** Targeted benchmark selection.

**Original references:** B4, PERF9

**Sources:** [Benchmarks/VectorCoreBench/Suites/DotProductBench.swift:57](../../../Benchmarks/VectorCoreBench/Suites/DotProductBench.swift:57); [Benchmarks/VectorCoreBench/Suites/DotProductBench.swift:116](../../../Benchmarks/VectorCoreBench/Suites/DotProductBench.swift:116); [Benchmarks/VectorCoreBench/Suites/DotProductBench.swift:160](../../../Benchmarks/VectorCoreBench/Suites/DotProductBench.swift:160); [Scripts/compare_benchmarks.swift:23](../../../Scripts/compare_benchmarks.swift:23)

**Evidence:** Fresh filter.json: --suites dot --dims 512,768,1536 --filter *.optimized yields results=[] and exit 0. Each dimension returns before optimized registration when generic is excluded. Comparator defaults to allowing missing cases; baseline README omits --strict.

**Correction / limit:** This is case selection/coverage loss, not a zero-duration optimized speedup.

**Improvement:** Register/filter sibling cases independently, reject unintended empty runs and require matching case sets for regression comparisons.

**Acceptance:** Table-driven selection test for every registered case, exact names, excludes and variant filters; expected counts and missing-baseline policy must be asserted.

**Evidence set:** [tooling-serialization](evidence/root/) (see [replay/provenance index](evidence/README.md)).

## TOOL-06

**Fused/pre-normalized cosine A/B pairs cannot join under their current grouping keys.**

**P2 · code quality · confirmed-reproduced · finding**

**Exposure:** JSON and console A/B reporting.

**Original references:** B5, PERF9

**Sources:** [Sources/VectorCoreBenchmarking/ABComparisons.swift:62](../../../Sources/VectorCoreBenchmarking/ABComparisons.swift:62); [Benchmarks/VectorCoreBench/AppMain.swift:341](../../../Benchmarks/VectorCoreBench/AppMain.swift:341); [Sources/VectorCoreBenchmarking/Models.swift:308](../../../Sources/VectorCoreBenchmarking/Models.swift:308)

**Evidence:** Fresh two-case helper reports cosineABPairCount=0. Variant is part of key yet differs for fused and preNorm; console duplicates this disjoint-key requirement. deltaPercent comment gives a different denominator/sign formula from implementation.

**Correction / limit:** Euclidean pairing is a separate path and is not invalidated by this proof.

**Improvement:** Use a common comparison family key and one model for both output formats; document the chosen percentage convention.

**Acceptance:** Every supported A/B family must yield exactly one valid pair; mismatched dimensions/N/provider must not pair; check numeric delta against hand-calculated examples.

**Evidence set:** [tooling-serialization](evidence/root/) (see [replay/provenance index](evidence/README.md)).

## TOOL-07

**Several timings lack consistent result consumption and compare different setup/allocation boundaries.**

**P2 · performance · qualified · opportunity**

**Exposure:** Benchmark/test timing paths.

**Original references:** B6, PERF9

**Sources:** [Benchmarks/VectorCoreBench/Suites/BatchBench.swift:449](../../../Benchmarks/VectorCoreBench/Suites/BatchBench.swift:449); [Benchmarks/VectorCoreBench/Suites/BatchBench.swift:610](../../../Benchmarks/VectorCoreBench/Suites/BatchBench.swift:610); [Benchmarks/VectorCoreBench/Suites/BatchBench.swift:864](../../../Benchmarks/VectorCoreBench/Suites/BatchBench.swift:864); [Benchmarks/VectorCoreBench/Suites/BatchBench.swift:1016](../../../Benchmarks/VectorCoreBench/Suites/BatchBench.swift:1016); [Benchmarks/VectorCoreBench/Suites/MatrixDistanceBench.swift:67](../../../Benchmarks/VectorCoreBench/Suites/MatrixDistanceBench.swift:67)

**Evidence:** Source rechecked: some 1536D reductions discarded; neighboring widths use blackHole. SoA packing/FP16 conversion occurs before timing despite comments saying included. AoS may allocate chunk outputs during timing while SoA reuses output.

**Correction / limit:** Missing consumption is proved; dead-code elimination and its timing magnitude are unverified. Different timing scopes can be intentional but must be identified before attributing speedups.

**Improvement:** Define setup scope and work units; maintain separate prepacked-kernel and end-to-end cases; consume/check all output consistently.

**Acceptance:** Inspect optimized code and allocation traces for representative cases; record consumer overhead, warmups, raw samples, machine/toolchain and scope. Compare only equivalent workloads.

**Dependencies:** TOOL-01, TOOL-03

**Evidence set:** [tooling-serialization](evidence/root/) (see [replay/provenance index](evidence/README.md)).

## TOOL-08

**Memory-footprint output counts owning struct handles rather than heap payload.**

**P3 · code quality · confirmed-reproduced · finding**

**Exposure:** Mixed-precision footprint report and misleading timing-test names.

**Original references:** B7, PERF9

**Sources:** [Tests/ComprehensiveTests/MixedPrecisionPerformanceBenchmark.swift:302](../../../Tests/ComprehensiveTests/MixedPrecisionPerformanceBenchmark.swift:302); [Tests/MinimalTests/ComprehensiveBenchmarks.swift:24](../../../Tests/MinimalTests/ComprehensiveBenchmarks.swift:24); [Sources/VectorCore/Vectors/Vector512Optimized.swift:29](../../../Sources/VectorCore/Vectors/Vector512Optimized.swift:29); [Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:302](../../../Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:302)

**Evidence:** Fresh linked Release probe reports FP32 handle=16, FP16 handle=8. The corresponding 512D scalar payloads alone are 2048 and 1024 bytes. Source uses handle stride for a 100000-vector footprint extrapolation; MinimalTests footprint names time magnitudes.

**Correction / limit:** No allocated-byte or RSS measurement was made; payload arithmetic excludes headers, allocator overhead and COW sharing.

**Improvement:** Separate handle bytes, logical payload and measured retained memory; rename timing-only tests.

**Acceptance:** Check representation payload size; profile realistic independent/shared datasets and report method, peak/retained bytes and workload.

**Evidence set:** [tooling-serialization](evidence/root/) (see [replay/provenance index](evidence/README.md)).

## TOOL-09

**Matrix benchmark ignores shared selection/timing controls and has only 17 distinct generated vectors.**

**P2 · code quality · confirmed-source · opportunity**

**Exposure:** Opt-in matrix benchmark.

**Original references:** validation_benchmarks:Matrix suite ignores common harness controls

**Sources:** [Benchmarks/VectorCoreBench/Suites/MatrixDistanceBench.swift:18](../../../Benchmarks/VectorCoreBench/Suites/MatrixDistanceBench.swift:18); [Benchmarks/VectorCoreBench/Suites/MatrixDistanceBench.swift:45](../../../Benchmarks/VectorCoreBench/Suites/MatrixDistanceBench.swift:45); [Sources/VectorCoreBenchmarking/CaseParsing.swift:17](../../../Sources/VectorCoreBenchmarking/CaseParsing.swift:17)

**Evidence:** Hardcoded N=64/256/1024, custom timer, no seed/repeats/samples/batchNs consumption; filtering occurs after suite execution. Values depend on (coordinate+row)%17; slash-form names do not parse dim/N.

**Correction / limit:** No claim that 17 distinct vectors makes numerical computation incorrect; it limits representativeness and provenance.

**Improvement:** Adopt shared descriptors/harness and explicit packed versus one-shot cases.

**Acceptance:** Changing controls must change scheduled cases/samples; filters must prevent execution; names or structured fields must preserve dim/N; seeded fixtures must have documented diversity.

**Evidence set:** [tooling-serialization](evidence/root/) (see [replay/provenance index](evidence/README.md)).

## TOOL-10

**Current benchmark coverage does not establish downstream large-scale behavior.**

**P2 · architecture · qualified · opportunity**

**Exposure:** Scale-readiness validation gap; no universal throughput claim supported.

**Original references:** validation_benchmarks:Trustworthy coverage for downstream scale

**Sources:** [Benchmarks/VectorCoreBench/AppMain.swift:11](../../../Benchmarks/VectorCoreBench/AppMain.swift:11); [.github/workflows/ci.yml:145](../../../.github/workflows/ci.yml:145); [.github/workflows/benchmarks.yml:31](../../../.github/workflows/benchmarks.yml:31); [Tests/ComprehensiveTests/BatchGEMMRoutingTests.swift:1](../../../Tests/ComprehensiveTests/BatchGEMMRoutingTests.swift:1)

**Evidence:** Default/full suites omit matrix/memory; PR smoke uses three suites at 512D; SoA/FP16 opt-ins absent in workflows. Source registry lacks dedicated PCA/UMAP/quantization/streaming-search scale cases. Numerical parity tests alone do not prove a route executed.

**Correction / limit:** These are coverage opportunities, not proof of a current production outage or a demand to put all workloads into every PR job.

**Improvement:** Add representative downstream workload tiers with route sentinels, allocation/peak-memory measurements and bounded concurrency checks.

**Acceptance:** Publish exact hardware, inputs, routing, raw sample distribution and memory budget for chosen target scale; validate custom-provider precedence at every newly reachable boundary.

**Dependencies:** TOOL-01, TOOL-03, TOOL-07

**Evidence set:** [tooling-serialization](evidence/root/) (see [replay/provenance index](evidence/README.md)).

## TOOL-11

**CI coverage excludes important tooling and selected safety surfaces.**

**P2 · code quality · confirmed-source · finding**

**Exposure:** Test/lint/CI configuration.

**Original references:** validation_benchmarks:CI and style enforcement

**Sources:** [Package.swift:82](../../../Package.swift:82); [.swiftlint.yml:39](../../../.swiftlint.yml:39); [.github/workflows/ci.yml:90](../../../.github/workflows/ci.yml:90); [.github/workflows/ci.yml:157](../../../.github/workflows/ci.yml:157); [.github/workflows/sanitizers.yml:30](../../../.github/workflows/sanitizers.yml:30); [Tests/PerformanceTests/NSWOptimizationBenchmark.swift:11](../../../Tests/PerformanceTests/NSWOptimizationBenchmark.swift:11)

**Evidence:** No test target depends on VectorCoreBenchmarking. Root lint includes Sources/Tests; workflow invokes lint, not analyze. Required aggregate lists test/lint/repository-checks/consumer/platforms, not sanitizer/CodeQL/perf-smoke. NSW file outside registered targets references former graph APIs. New Python 25-test run and Ruby 9-file policy validation pass.

**Correction / limit:** No remote branch-protection settings inspected; independent checks may be required externally. Declared analyzer rules alone do not execute analysis. Do not count unregistered tests as coverage.

**Improvement:** Add focused tooling tests, audited sanitizer/filter coverage and benchmark lint scope. Archive/move obsolete NSW benchmark. Enforce claim-to-evidence discipline in review; avoid cosmetic blanket inlining/weak-self rules.

**Acceptance:** CI inventory must map each promised test/analysis to an executed target/command; mechanical policy tests must reject missing required jobs and test/route coverage gaps.

**Evidence set:** [tooling-serialization](evidence/root/) (see [replay/provenance index](evidence/README.md)).

## TOOL-12

**Baseline comparison ignores provenance and CSV metadata does not escape multiline/arbitrary values.**

**P2 · code quality · confirmed-source · opportunity**

**Exposure:** Benchmark artifact comparison/export.

**Original references:** validation_benchmarks:Metadata and baseline handling

**Sources:** [Scripts/compare_benchmarks.swift:46](../../../Scripts/compare_benchmarks.swift:46); [Benchmarks/VectorCoreBench/Formatters/CSVWriter.swift:11](../../../Benchmarks/VectorCoreBench/Formatters/CSVWriter.swift:11); [Benchmarks/baselines/arm64-os15.4.1-canonical.json:7](../../../Benchmarks/baselines/arm64-os15.4.1-canonical.json:7); [Benchmarks/baselines/arm64-os15.4.1-local.json:7](../../../Benchmarks/baselines/arm64-os15.4.1-local.json:7); [Sources/VectorCoreBenchmarking/ABComparisons.swift:21](../../../Sources/VectorCoreBenchmarking/ABComparisons.swift:21)

**Evidence:** Comparator decodes results only. Both checked-in baselines record 2025/0.1.0-alpha. CSV header fields interpolate labels/newlines unquoted, while data rows use quoting. A/B uses aggregate ns/unit; shared p90 is a percentile of per-sample averages.

**Correction / limit:** Historical baselines are useful records, not current guarantees. Aggregate means versus medians are choices needing explicit labeling, not inherently arithmetic errors.

**Improvement:** Validate hardware/build/workload compatibility before comparisons; quote all CSV fields; retain raw samples and label statistics precisely.

**Acceptance:** Incompatible metadata must be rejected or explicitly overridden; round-trip labels containing commas/quotes/newlines; independently verify chosen statistic and comparison formula.

**Dependencies:** TOOL-03, TOOL-06

**Evidence set:** [tooling-serialization](evidence/root/) (see [replay/provenance index](evidence/README.md)).

## TOOL-13

**Mobile availability of the separate benchmarking product remains unverified.**

**P3 · architecture · unverified · investigation**

**Exposure:** VectorCoreBenchmarking product only.

**Original references:** validation_benchmarks:Metadata and baseline handling:mobile

**Sources:** [Sources/VectorCoreBenchmarking/EnvCapture.swift:120](../../../Sources/VectorCoreBenchmarking/EnvCapture.swift:120); [Sources/VectorCoreBenchmarking/EnvCapture.swift:164](../../../Sources/VectorCoreBenchmarking/EnvCapture.swift:164); [Package.swift:20](../../../Package.swift:20); [.github/workflows/ci.yml:107](../../../.github/workflows/ci.yml:107)

**Evidence:** EnvCapture references Process/Host and sysctl; the Apple platform loop targets VectorCore. No mobile benchmarking-product build was performed.

**Correction / limit:** Do not report a confirmed mobile compile failure or extend this suspicion to the Core product.

**Improvement:** Explicitly scope tooling to supported hosts or separately compile advertised product/platform combinations.

**Acceptance:** Product support matrix backed by actual target builds, or clear host-only documentation.

**Evidence set:** [tooling-serialization](evidence/root/) (see [replay/provenance index](evidence/README.md)).
