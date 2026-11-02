# VectorCore foundation review — 2026-09-14

## Assessment

**Resolve the ownership, invariant, metric-dispatch and metadata defects before a broad production rollout.** The package has useful numerical primitives, clear CPU/GPU boundaries and substantial regression coverage. However, ordinary public calls can return incorrect neighbors, preserve stale normalization metadata, share mutations across value copies, or defeat the assumptions of unsafe kernels. Separate scale problems amplify copying, retained memory and task counts.

This is a review of **`8b02e75bca7055ae11170873d15c06f1538709b7`**, on `main`, identifying itself as 0.3.3. Twelve independent GPT-6 Astra reviewers at high reasoning effort reviewed the components below, three at a time. The orchestrator examined their evidence, reproduced selected findings against both Debug and Release builds, deduplicated overlapping findings, and made the priorities in this report. Component notes contain additional evidence and qualified observations; this report controls the final prioritization.

No implementation or existing test files were changed. Only review documents and reproduction evidence were added under this directory. Existing untracked `.antigravitycli/` and `GEMINI.md` were preserved.

### Priority and evidence conventions

- **P1:** fix before relying on the affected public path in production; silent corruption, broken ownership/invariants, or wrong algorithm selection.
- **P2:** actionable correctness, resource, API or performance problem with a specific trigger; prioritize by actual workload exposure.
- **P3 / opportunity:** lower-impact issue, documented limitation, dormant implementation, or optimization needing measurement.
- **Reproduced:** a small executable demonstrated behavior. **Source-derived:** the mechanism follows from inspected code; no runtime failure or speedup is implied.

The opt-in FP16 cache, tuners and pure-Swift factorization provider are **not** used by every default `Operations` call. Their failures must not be generalized to all Core workloads. No P0 universal outage or whole-library safety claim is made.

## Review coverage

| # | Component | Primary scope | Independent report |
|---|---|---|---|
| 1 | Vector representations and protocols | Generic/dynamic/optimized vectors, dimensions, factories, arithmetic and producer contracts | [Vectors](components/vectors.md) |
| 2 | Memory and storage | COW, aligned storage, tiered storage, SoA, page ownership and pointer bridge | [Memory/storage](components/memory_storage.md) |
| 3 | Execution and concurrency | Providers, pools, configuration, task scheduling, progress, logging and errors | [Execution/concurrency](components/execution_concurrency.md) |
| 4 | SIMD and C kernels | Float/Double providers, scalar/SIMD/C distance and normalization paths, platform dispatch | [SIMD/kernels](components/simd_kernels.md) |
| 5 | Search and Top-K | Nearest-neighbor routing, selectors, heaps, merges and ordering contracts | [Search/Top-K](components/search_topk.md) |
| 6 | Batch and matrix operations | Batch APIs, AoS/SoA kernels, matrix distances, aggregations and entropy | [Batch/matrix](components/batch_matrix.md) |
| 7 | Reduced precision | FP16/INT8 conversion and kernels, affine schemes, diagnostics, caches and tuners | [Reduced precision](components/reduced_precision.md) |
| 8 | Serialization | Binary framing, CRC, Codable, representation and round-trip contracts | [Serialization](components/serialization.md) |
| 9 | Linear algebra and PCA | LAPACK bridge, Swift fallback, factorizations, PCA fit/transform | [Linear algebra/PCA](components/linear_algebra.md) |
| 10 | Manifold learning | CSR graph, fuzzy weights, UMAP calibration and optimization | [UMAP](components/umap.md) |
| 11 | Downstream interfaces | Collections, hints, score semantics, provider seams and actual sibling call sites | [Interfaces](components/downstream_interfaces.md) |
| 12 | Validation and benchmarking | Benchmark library/executable, test assertions, CI, lint and scripts | [Validation/benchmarks](components/validation_benchmarks.md) |

Each report inventories its source coverage, relevant tests, reproductions, strengths and limits. Test files were reviewed according to the component; this is not a claim that every line of every test or historical design document received an independent audit. Downstream source review was limited to interfaces and affected call paths, not a full audit of both sibling libraries.

## 1. Architecture

### A1 — P1: Public storage mutation can invalidate every later shape check

[`DynamicVector.storage`](../../../Sources/VectorCore/Vectors/DynamicVector.swift:17) can be replaced while its stored dimension remains unchanged. The optimized vectors expose resizable `ContiguousArray` storage yet construct fixed-size Float buffer views, for example [`Vector512Optimized`](../../../Sources/VectorCore/Vectors/Vector512Optimized.swift:150).

**Reproduced in Debug and Release:** replacing a two-element DynamicVector's storage with three elements leaves it equal to the original but with a different hash. Replacing a 512-dimensional vector's storage with one SIMD4 leaves its public Float view advertising 512 elements over four actual scalars. A write beyond that allocation was not executed; subsequent fixed-count reads/writes follow directly from the exported view.

**Decision:** enforce shape at the owning type, not only constructors. Derive dynamic dimensions from storage, use shape-preserving storage for fixed vectors, and reconsider the unrestricted setter required by `VectorProtocol`. Include INT8 raw constructors and synthesized decoding in the same repair. A single invariant suite should cover every vector family, wrappers, direct/generic mutation, buffers, equality/hash and serialization. See V1 and reduced-precision invariant notes.

### A2 — P1: Normalization hints remain valid after the underlying vector changes

[`NormalizationHint.vector`](../../../Sources/VectorCore/Integration/IndexableVector.swift:107) has no mutation observer; [`withUnsafeMutableBufferPointer`](../../../Sources/VectorCore/Integration/IndexableVector.swift:203) invalidates only after successful return.

**Orchestrator reproduction in both builds:** a correct hint for `[1,0]`, then direct mutation to `[2,0]`, still reports normalized=true and magnitude=1. A buffer closure that writes `[3,0]` then throws likewise retains magnitude=1. This is not misuse of an initially false assertion.

**Downstream impact is concrete:** VectorIndex's `TypedOverloads.swift` consumes these flags for query normalization and stored inverse norms; HNSW can retain the supplied inverse norm. VectorAccelerate cosine providers use raw dot products when both hints claim normalization. Those call sites were inspected; no GPU run was performed.

**Decision:** invalidate all public mutation paths, including exceptional exits, before further downstream reliance. Use `defer` or invalidate before mutation, observe/restrict wrapped-value mutation, and test hints through actual index insertion and query paths. Existing indexed metadata may require rebuilding if it was populated from stale hints. See IF5.

### A3 — P1, opt-in: FP16 cache keys confuse partial hashes with exact identity

[`SoAFP16CacheKey`](../../../Sources/VectorCore/Operations/Kernels/SoAFP16Cache.swift:26) contains only vector hashes. [`Vector512Optimized.hash`](../../../Sources/VectorCore/Vectors/Vector512Optimized.swift:334) samples just the first and last eight scalars.

**Reproduced in both builds:** cache a zero vector, change scalar 100 to 1, and request the changed candidate set. The cache returns scalar 100 as zero. This is a deterministic collision on ordinary data, not a probabilistic hash concern.

**Decision:** use an immutable candidate identity/version or compare complete data on hash hits. Hashing every scalar alone still does not make hashes an equality proof. Add a test that changes each coordinate independently. This cache is an explicit opt-in surface; no default `Operations` or inspected sibling caller was found. See RP finding 1.

### A4 — P1: Two public search facades silently replace the requested metric

[`TopKSelection.nearest`](../../../Sources/VectorCore/Operations/TopKSelection.swift:325) and [`VectorCollection.search`](../../../Sources/VectorCore/Integration/VectorCollection.swift:129) recognize four concrete metrics and otherwise execute Euclidean distance. Built-in Chebyshev, Hamming, configurable Minkowski and custom metrics are affected.

**Reproduced in both builds:** query `[0,0]`, candidates `[[2.5,0],[2,2]]`, Chebyshev, k=1: TopKSelection chooses index 0, although the requested scores are 2.5 and 2. `Operations.findNearest` correctly chooses index 1. Collection search also reports a Euclidean score for Chebyshev.

**Decision:** dispatch the actual Float metric witness and consolidate facade behavior. Expanding the concrete allowlist does not repair custom metrics. Test every built-in plus a custom reverse-order metric through all public search facades. See S1/IF1.

### Other accepted architecture findings

| ID / priority | Finding and evidence | Decision |
|---|---|---|
| A5 — P2 | **Configuration is not an effective transaction or consistent control plane.** `BatchOperations` stores configured batch size/parallel settings but reads `defaultConfig` in their consumers. Configured size2 processed five items as one batch. Its separate get/update awaits lost increments: 100 updates produced 28 in one run. [Source](../../../Sources/VectorCore/Operations/BatchOperations.swift:99); EC3–4. | Snapshot live configuration per operation; apply mutations inside one actor method. Validate all values before making currently dormant settings effective. |
| A6 — P2 | **Several public schedulers enqueue all work at once.** ComputeProvider defaults ignore maxConcurrency; a conformer declaring 1 reached 64 active bodies. Public map/filter and legacy matrix paths also fan out eagerly; 10,000 matrix rows imply 49,995,000 pair tasks. VectorAccelerate inherits the defaults. [Source](../../../Sources/VectorCore/Protocols/ComputeProvider.swift:130); EC5. | Share bounded chunk/window scheduling, with explicit cancellation and nesting contracts. An actor is not a bound on outstanding suspended operations. |
| A7 — P2 | **Pool teardown and byte budgets are incorrect.** SwiftBufferPool's empty deinit leaves raw allocations retained; its budget cleanup removes only old entries, keeping a fresh 8 KB burst under a 1 KB cap. TypedBufferPool also lacks allocation teardown. [Source](../../../Sources/VectorCore/Providers/SwiftBufferPool.swift:48); EC1–2. | Make allocation ownership explicit, free on teardown, track cached bytes and evict to budget independently of age. Test all pool owners, not only the separately repaired MemoryPool. |
| A8 — P2 | **Collection mutation can lose records.** Default update removes the old value before throwing add; a rejected dimension update deletes id7. Explicit id0 followed by generated insertion reuses id0 and overwrites it. Both reproduced in both builds. [Update](../../../Sources/VectorCore/Integration/VectorCollection.swift:243), [insert](../../../Sources/VectorCore/Integration/VectorCollection.swift:286); IF2–3. | Require atomic replacement and preserve old state on failure; reject duplicates and generate unused IDs under the lock. This reference collection is not the actual VectorIndex storage engine. |
| A9 — P2 | **Scores lack a sufficient semantic contract.** `computedScore = 1/(1+distance)` maps valid negative-dot distances −1 to infinity and −2 to −1. [Source](../../../Sources/VectorCore/Integration/SearchResult.swift:34); IF4. | Carry score kind or require an explicit metric-specific transform; test monotonicity and finiteness from actual search results. |
| A10 — opportunity | **Some advertised hooks have no effect.** `Operations.bufferProvider` has no internal reads; VectorCore's separate global configuration is also unused. Extension-only protocol methods select different implementations in generic code. Multiple search/execution facades have drifted. IF architecture notes; V2/V3. | Publish one map of active override points, make intended customization points protocol requirements, and consolidate consumers before adding more configuration. |

**Boundary judgment:** retain Core's CPU numerical role, Core-owned CSR interchange and explicit SoA ownership contracts. Do not move ANN storage or GPU scheduling into Core to fix these problems. The actual VectorIndex integration is typed async index overloads; its source does not implement the synchronous `VectorCollection` protocol advertised as a future seam.

## 2. Code quality and numerical correctness

### C1 — P1: Compact triangular-matrix copies share mutations

[`TieredStorage.ensureUniqueTier`](../../../Sources/VectorCore/Utilities/TieredStorage.swift:175) tests the inner ManagedBuffer's ownership rather than the shared CompactBuffer owner. A copied matrix therefore mutates its original for default matrix sizes **4 through 32**.

**Optimized exact-source probe, corroborated against the complete Release library:** `let a = TriangularMatrix<Float>(size:4, defaultValue:1); var b=a; b[0,1]=99` leaves both values at 99. Controls across tier boundaries localize the defect. Compact storage also never destroys manually initialized generic tail elements; a weak sentinel remains retained after its matrix scope ends in both probes.

**Decision:** repair ownership at the stored-reference boundary and element destruction, with a mechanical test matrix across inline/compact/standard/aligned transitions, scalar/bulk/throwing mutation, copies and iterator snapshots. See MS1/MS4. This is a value-semantics defect, not merely a concurrency concern.

### C2 — P1: Aggregation APIs omit validation before unsafe reads or forced construction

[`Operations.centroid`](../../../Sources/VectorCore/Operations/Operations.swift:777) passes the first vector's dimension to every later raw-pointer addition. Short rows can be read beyond their logical allocation; long rows are truncated. Empty fixed-vector input hits `try!` with a zero-length array and terminates. [`statistics`](../../../Sources/VectorCore/Operations/Operations.swift:854) also trusts the first row's length.

**Orchestrator reproduction:** empty fixed centroid traps; a ragged centroid returns numbers from a mismatched shape; ragged statistics traps. No equal-shape precondition is documented at these public aggregation entry points.

**Decision:** define empty behavior and validate all rows before reading; propagate factory failures rather than forcing generic construction. Test every aggregation with short/long first, middle and last rows, and empty collections. Do not confuse this with MatrixDistance's separately documented unsafe caller obligations.

### Other accepted code-quality findings

| ID / priority | Trigger, failure and source | Recommended remedy / validation |
|---|---|---|
| C3 — P2 | **Finite division and normalization can produce Inf/NaN or a wrong unit norm.** `[s,0]/s`, s=least subnormal, becomes `[Inf,NaN]` through reciprocal multiplication. Checked generic normalization can report success with that output while a concrete optimized call rejects it. Pointer normalization of components1e−38 yields norm≈1.17549. [Division](../../../Sources/VectorCore/Protocols/VectorProtocol.swift:264); V2 and SIMD findings1/4. | Normalize in a scaled domain and avoid requiring a representable reciprocal/full norm. Test every vector/provider/direct/generic route over exponent ranges, with norm and direction assertions rather than only finiteness. |
| C4 — P2 | **Pointer normalization's alignment contract is inaccurate.** A naturally Float-aligned buffer offset by4 bytes traps at SIMD4 load although16-byte alignment is described as preferred. SIMD finding2. | Either use unaligned loads or explicitly enforce/document mandatory alignment at the boundary; test offsets and tails. |
| C5 — P2 | **In-place vector arithmetic omits dimension checks.** A longer RHS is silently truncated; a shorter RHS traps in Debug and can overread in optimized pointer loops. [Source](../../../Sources/VectorCore/Protocols/VectorProtocol.swift:181); V3. | One shared shape check across in-place add/subtract/multiply and generic operators, before any mutation. |
| C6 — P2 | **Affine calibration and saturation are inconsistent.** QuantizationSchemes `[1,2]` round-trips as `[1,1]`; constant1.5 nearly becomes zero. LinearQuantizationParams converts to Int before clamping and traps on finite1e20 outliers. [Calibration](../../../Sources/VectorCore/Quantization/QuantizationSchemes.swift:96), [conversion](../../../Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:54); RP2–3. | Consolidate duplicated calibration, define constants/nonfinite inputs, and bound the floating intermediate before integer conversion. Cover scalar/SIMD and decoded parameter construction. |
| C7 — P2 | **Quantized cosine loses scale invariance.** A vector repeating1e10 has self-distance1 because multiplying its finite squared norms overflows before sqrt. Repeated across512/768/1536. [Source](../../../Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:754); RP5. | Cancel symmetric scales or use range-safe norm arithmetic; test self/opposite vectors under scale sweeps across widths. |
| C8 — P2 | **Batch variance changes numerically at N=1000.** Alternating magnitudes10000/10001 produce std≈0.5 at999 but0 at1000/1024. Identical1e18 values produce infinite std. [Source](../../../Sources/VectorCore/Operations/BatchOperations.swift:367); batch finding2. | Stable Double mean/M2 accumulation and merge, with independent covariance tests around the routing boundary. |
| C9 — P2 | **Entropy discards finite scale information.** Uniform Dim4 at1e−9 returns0 rather than log4; finite maximum values return0 orNaN depending on dimension. [Source](../../../Sources/VectorCore/Operations/VectorEntropy.swift:42); batch finding4. | Scale before summation and distinguish zero vectors from small nonzero vectors. Test scale and dimension-route equivalence. |
| C10 — P2 | **Zero-query SoA cosine overwrites output suffixes.** Valid oversized Safe512 outputs lose both sentinel elements for N=1,4,4097; all three kernel routes fill the whole buffer. [Source](../../../Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift:930); batch finding3. | Write exactly the logical N outputs; sentinel tests across dimensions, offsets and route/tile boundaries. |
| C11 — P2 | **Pairwise and similarity helpers impose incorrect diagonal/sign semantics.** Negative-dot self-distance of `[2]` is−4, but pairwise returns0; cosineSimilarityMatrix returns negative similarities off-diagonal and0 on the diagonal. [Source](../../../Sources/VectorCore/Operations/TransformOperations.swift:164); batch finding7. | Compute metric-specific diagonals; correct similarity sign. State symmetry requirements and test every built-in metric against direct pair evaluation. |
| C12 — P2 | **Bulk heap construction and decoded results violate their own invariants.** Capacity2 bulk max-heap from [1,2,3,4] retains[4,2], while bounded insertion retains[2,1]. TopKResult JSON allows one index and zero distances. [Heap](../../../Sources/VectorCore/Operations/MinHeap.swift:57), [decoder](../../../Sources/VectorCore/Operations/TopKSelection.swift:651); S2–3. | Select bounded members before heapification, and validate parallel array lengths during decode. Compare bulk/streaming membership across permutations and capacities. |
| C13 — P2 | **Pure-Swift factorizations are not range-safe.** For scale×[[1,1],[0,1]], scales1e−12/1e15 return spectrum[1.414,1] instead of[1.618,0.618] after rescaling, with U columns correlated≈0.707. QR([1e−25,1e−25]) loses a row; QR at1e20 emitsNaN. [SVD](../../../Sources/VectorCore/LinearAlgebra/SwiftLinearAlgebraProvider.swift:238), [QR](../../../Sources/VectorCore/LinearAlgebra/SwiftLinearAlgebraProvider.swift:46); LA1–2. | Scaled/wider intermediate arithmetic; test spectra, orthogonality, finiteness and residuals. LAPACK handled these fixtures correctly; this finding concerns the public Swift fallback. |
| C14 — P2 | **UMAP calibration depends incorrectly on units.** spread1e−5/minDist1e−6 silently gives a=b=1; smooth-kNN distances scaled1e±30 change weights or remove neighbor pairs. Infinite learningRate passes validation and produces nonfinite coordinates. [Curve](../../../Sources/VectorCore/ManifoldLearning/UMAP.swift:350), [bandwidth](../../../Sources/VectorCore/ManifoldLearning/FuzzySimplicialSet.swift:65); UMAP1–3. | Fit/calibrate dimensionless quantities, verify convergence, and validate finite parameters. Test scaling against the local defining equations. |
| C15 — P2 | **Ordinary validation boundaries can trap or discard work.** Batch process size0 traps and−1 returns no results; factory dimension0 traps before throwing; 1×1 triangular import reads[0][1]. PCA's Int.max oversampling overflows before clamping even on two1D samples. EC6, V6, batch5, LA3. | Validate before stride/remainder/index/arithmetic; parameterize boundary tests across related APIs. Treat enormous optional settings below normal-input defects in rollout priority. |
| C16 — P2 | **Nonfinite checking helpers accept invalid outcomes.** Approximate equality accepts NaN vs42; safeLog bypasses NaN/+Inf policies. [Comparison](../../../Sources/VectorCore/Protocols/VectorProtocol.swift:550), [log](../../../Sources/VectorCore/Operations/NaNInfinityHandling.swift:383); V7/SIMD5. | Define nonfinite/tolerance policy and test all option presets. Internal PrecisionValidator has the same NaN-acceptance class. |
| C17 — P2 | **Unsafe lifetime semantics vary by implementation.** Vector384Optimized retains permanent bindMemory where siblings were repaired with scoped rebinding. Generic alignedCopy initializes one alignment branch but assigns the other, leaking an initialized reference destination in a probe. [384](../../../Sources/VectorCore/Vectors/Vector384Optimized.swift:143), [copy](../../../Sources/VectorCore/Utilities/MemoryUtilities.swift:135); V4/MS5. | Apply binding repair mechanically across variants; split/define initialization versus update semantics. Binding discrepancy is source-derived; no optimizer miscompile was reproduced. |
| C18 — P2 | **Diagnostics has an actual unsynchronized field access.** isEnabled reads occur outside the lock while enable/disable write under it. [Source](../../../Sources/VectorCore/Operations/Kernels/MixedPrecisionDiagnostics.swift:103); RP4. | Check and update under one lock, or use a correctly specified atomic fast path. Source-proven conflicting accesses; no TSan run claimed. |
| C19 — P2, opt-in | **Precision controls are not honored.** Adaptive distance ignores threshold and candidate range; zero query versus70000 candidates returnsInf instead of finite1,583,918.9. Quantized tuner omits accuracyRequirement from its key, so warm/fresh answers differ. [Adaptive](../../../Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:3388), [tuner](../../../Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:1389); RP6–7. | Define operative accuracy semantics, test both operands and every key input, and fall back when the requested policy cannot be met. |
| C20 — P2 | **Supported empty DynamicVector does not round-trip through binary encoding.** Encoder writes a 16-byte dimension-0 record; decoder rejects dimension 0. JSON supports the same value. [Source](../../../Sources/VectorCore/Utilities/BinaryFormat.swift:164); serialization finding 1. | Make empty encoding/decoding consistent or provide checked encoding with explicit domain. Replace the current “empty” test that actually uses dimension 1. Source finding corroborated by the orchestrator against Release. |

### Style and claim discipline

SwiftLint 0.65.1 completed with **293 warnings, zero errors**, over 204 files. The largest groups are 107 line-length warnings, 30 private-over-fileprivate, 20 trailing-comma, 16 for-where and 15 redundant-discardable-let warnings. These are secondary to correctness; reducing the warning count alone will not establish production quality.

The more consequential style problems are duplicated implementations, public methods whose parameters do not affect behavior, `try!` at generic boundaries, and claims that exceed evidence. Examples include “actor deinit automatically handles cleanup,” “proper backpressure” on an add-all task loop, “unconditional numerical robustness” beside range-unsafe factorizations, and fixed speedup claims without named benchmarks. Keep mathematical notation and justified kernel complexity; remove misleading assurances and tie retained claims to actual tests or dated measurements.

## 3. Performance and scale

### PERF1 — P2: Accidental COW turns ordinary writes into whole-buffer copies

Two independent implementation errors deserve priority over micro-optimization:

- [`HybridStorage`](../../../Sources/VectorCore/Storage/HybridStorage.swift:125) creates extra owners around mutation; fill performs this per scalar. An optimized instrumented source probe counted **1,026 buffer allocations** for a repeating 512-element storage construction. Construction has Θ(d²) element traffic rather than Θ(d). This reaches public DynamicVector repeating construction and resize.
- [`AlignedBuffer.isUniquelyReferenced`](../../../Sources/VectorCore/Utilities/TieredStorage.swift:584) creates another self reference, so large triangular matrices always clone on scalar writes. At size 65, ten unique writes caused ten 2080-element copies. Filling the triangle via scalar setters has **Θ(n⁴)** storage traffic instead of Θ(n²).

These are allocation-count/complexity findings from optimized probes, **not wall-clock speedup estimates**. Similar-looking DimensionStorage scalar mutation was tested and optimized away correctly; that suspicion was rejected. Fix the actual ownership representations, then assert bounded allocations for unique writes and exactly one detach after a retained copy. Before switching to bulk tier mutation, repair its dormant standard-tier lost-write bug; otherwise the performance fix would activate another defect. See MS2/MS3/MS6.

### PERF2 — Architecture opportunity: Bound search scratch memory and expose precision policy

[`Operations.findNearestBatch`](../../../Sources/VectorCore/Operations/Operations.swift:211) switches to full GEMM at Q≥8/N≥256 for supported optimized types, allocating4 × Q × N bytes of scores irrespective of k. At Q=10,000/N=1,000,000 this is **40 GB decimal for scores alone**, plus candidate/query packing and inputs, for only 100,000 final neighbors when k=10. This is source-derived sizing, not an attempted allocation.

This allocation and GEMM cancellation are already documented limitations. They still need a serving-scale design decision. A Release fixture with 512 coordinates near 1000 selects the exact self-neighbor for one query, but selects a nearby nonidentical neighbor when eight queries trigger GEMM. The positive direct squared difference disappears in the Gram subtraction.

**Decision:** add a per-call strategy and scratch budget; tile scores into a shared deterministic Top-K reduction, expose prepared candidates, and provide a direct-distance option for sensitive inputs. Tiling fixes peak storage, not cancellation. A narrow rerank after GEMM does not guarantee recovery if the true neighbor was already excluded. See search L1 and batch performance notes.

### Additional performance findings and opportunities

| ID | Finding | Assessment / next step |
|---|---|---|
| PERF3 — P2 | Sampled optimized-vector hashes deterministically collide for all interior-only variation. Set/Dictionary construction can degrade to quadratic equality scans. V5. | Hash equality-significant data or use explicit stable identities; benchmark a structured/sparse key corpus. No downstream vector-keyed hash table was identified in the inspected siblings. |
| PERF4 — P2 | FP16 range wrappers map the entire N-candidate collection for every requested chunk, adding O(KN) metadata work for K chunks. RP opportunities. | Process typed arrays directly or only the requested slice. Verify work/allocations for a tiny range stay independent of unrelated candidates. |
| PERF5 — opportunity | Array Top-K builds O(N) tuples even on small-k paths; VectorMath.nearestNeighbors fully sorts O(N log N) while claiming O(N log k). S/L2 and SIMD6. | Reuse streaming selection/comparators; preserve score/tie contracts. Measure both allocations and latency. |
| PERF6 — opportunity | Matrix pairwise reshaping temporarily holds flat and nested N² outputs; large-k parallel searches allocate k-sized heaps per chunk; fallback batch search revalidates every candidate per query. | Eliminate demonstrated redundant copies/validation, and choose small-k/large-k routes using representative Release measurements. |
| PERF7 — opportunity | MatrixDistance requires packing inputs already contiguous downstream. VectorIndex's CentroidBatchScore explicitly bypasses it for that reason. | Add a checked, owned/borrowed contiguous matrix view with dimensions/strides and reusable outputs, driven by a real consumer. |
| PERF8 — opportunity | FP16 cache limits entries rather than bytes; 100 entries each containing one million 512D candidates retain about 102.4 GB of payload. UMAP optimizer arrays can be multi-GB before CSR/input storage; PCA packs the full sample. | Publish byte models/budgets, reduce duplicate graph/index arrays, offer bounded transforms and cancellation-aware long-work APIs. These estimates are not measured RSS. |
| PERF9 — validation blocker | Benchmark comparisons and correctness checks can report success without measuring or validating the requested work. | See the validation report and the final verification section; repair benchmark evidence before choosing dispatch thresholds or claiming speedups. |

### Benchmark evidence that needs repair

I accept the following findings from the [independent tooling review](components/validation_benchmarks.md). These affect confidence in performance decisions; they do not prove a particular kernel is slow.

| Priority | Finding | Evidence and required change |
|---|---|---|
| P2 | **Correctness gates accept NaN and missing output.** | Current-source helper probes accept scalar NaN, vector NaNs and empty output against a nonempty reference. [Correctness.swift](../../../Benchmarks/VectorCoreBench/Utils/Correctness.swift:15) truncates to the shorter length and uses comparisons that fail to reject NaN. Separate validation failure from error statistics, require shape equality, and test the CLI exit/artifact boundary. Several suites supply no correctness result even in strict mode. |
| P2 | **Artifact metadata can describe a build that never ran.** | The Debug executable reports release; C-kernel, underscored and release-size CLI flags only alter metadata. [AppMain.swift](../../../Benchmarks/VectorCoreBench/AppMain.swift:467). Record effective compile-time configuration and executable identity; build variants through a driver. |
| P2 | **An optimized-only dot filter silently runs zero cases.** | Existing Release CLI with `--suites dot --dims 512,768,1536 --filter '*.optimized'` exits successfully with no results. [DotProductBench.swift](../../../Benchmarks/VectorCoreBench/Suites/DotProductBench.swift:57) exits the dimension routine when its generic sibling is excluded. Register/filter each case independently and reject unintended empty selections. |
| P2 | **Cosine A/B pairs cannot form.** | Group keys include the different fused/preNorm variants, then require both within one group. Current-source probe returns zero comparisons; the error is duplicated in JSON and console logic. [ABComparisons.swift](../../../Sources/VectorCoreBenchmarking/ABComparisons.swift:62). Share one comparison model and test every advertised pair. |
| P2 | **Extended performance assertions ignore the strict-performance opt-out.** | `EXTENDED=1, STRICT_PERF=0` in the benchmark workflow still enables speedup>1 and dimension-ratio assertions in [MixedPrecisionPerformanceBenchmark.swift](../../../Tests/ComprehensiveTests/MixedPrecisionPerformanceBenchmark.swift:100). This is source-proven gate reachability, not an observed hosted failure. Apply one assertion policy mechanically across tests. |
| P2 / measurement limit | **Timed work and setup boundaries are inconsistent.** | Some 1536D paths discard results while sibling cases consume them; SoA/FP16 setup occurs outside timing despite comments; compared routes allocate different temporary buffers. Generated-code elimination was not measured. Use explicit case descriptors, observable outputs, and separate prepacked-kernel versus end-to-end cases. |
| P3 | **Memory reports count handles rather than payloads.** | `MemoryLayout` of structs owning ContiguousArray omits heap data. A 512D vector has at least 2,048 FP32 or 1,024 FP16 payload bytes. Report logical payload, allocated bytes and measured RSS separately. |

The package has no dedicated test target for `VectorCoreBenchmarking`. Fifteen serialization tests have comment-only bodies; several optimized-serialization tests actually exercise generic vectors/raw arrays. The NSW benchmark is not registered in Package.swift. Convert relevant placeholders to actual contract tests or explicitly archive future work. The required lint selection excludes the benchmark executable, and the declared analyzer rules are not exercised by `swiftlint lint` alone.

Normal CI's artifact-only performance policy is appropriate for shared runners. Before adopting performance gates, fix provenance and correctness checks, compare matching case sets and environments, retain raw samples, and measure representative consumer workloads. The checked-in 2025/0.1.0-alpha baselines are historical records, not a current scale guarantee.

## Orchestrator disposition of lower-confidence and documented cases

- **Accepted, but lower priority:** PCA rounded centering changes sample variance from2 to4 for exactly representable1D inputs16777216/16777218. The behavior is reproduced in both builds. Treat this as a numerical-quality improvement for large-offset data, not a claim that ordinary normalized embeddings or LAPACK SVD are generally wrong. Fit and transform must share any improved center.
- **Accepted secondary boundary repairs:** negative bulk-heap capacity never terminates by direct loop analysis; CPU chunk endpoints overflow on a valid two-element range near Int.max; zero FP16-cache capacity traps; unrolledLoop mishandles strides other than its hard-coded4. These rank below default data/ownership defects. Details are in S4, EC7, RP notes and SIMD3.
- **Documented unsafe obligations:** MatrixDistance ragged/zero/oversized shapes, post-transfer SoA access, forged/used-after-release raw handles and caller-managed generic pool initialization were not promoted into new supported-use bugs.
- **Existing limitations:** GEMM cancellation/full matrices, Swift SVD's incomplete zero-singular-value basis, non-streaming PCA/UMAP, and binary decode's deliberate10,000-dimension ceiling were not counted as novel defects.
- **Dormant code:** escaping internal StorageSlice pointers, move-only experimental transfer, lost standard-tier bulk writes, internal reduction initial-value duplication and legacy heap NaN behavior are recorded for removal or repair before routing changes. They do not invalidate the recent public Top-K NaN contract.
- **KNNGraph ambiguity:** malformed CSR can trap despite throwing-validator wording, but docs explicitly require structurally valid buffers. Resolve the documentation/validation mismatch as boundary hardening; no broad security characterization is warranted here.
- **Downstream-owned follow-up:** VectorIndex batch typed overloads derive a batch-wide normalization flag from the first query. Heterogeneous hint batches can skip needed normalization. This was source-traced, not executed, and belongs in VectorIndex's issue list rather than a Core patch.

## Recommended remediation order

1. **Protect values and search meaning:** A1–A4, C1–C2. Close storage/hint/cache identity classes and metric dispatch before downstream reliance. Preserve old state on failed collection updates and eliminate insertion overwrites.
2. **Repair shared numerical contracts:** division/normalization, calibration/clamping, variance/entropy and metric diagonals. Test direct/generic/provider paths, all supported widths, exponent scales and route boundaries. Scope QR/SVD/UMAP work to applications using those APIs.
3. **Make resource controls effective:** pool ownership and byte caps, atomic/live configuration, bounded task creation, then accidental-copy removal and bounded search scratch. Validate dormant configuration and bulk paths before activating them.
4. **Repair the evidence pipeline:** benchmark correctness/selection/metadata, executable test bodies and class-wide coverage. Then measure representative Release workloads to choose thresholds and document performance.
5. **Consolidate and stabilize:** canonical high-level routes, explicit score kinds, effective provider hooks, supported public symbol inventory, and exact-revision downstream builds. Avoid a broad rewrite before these invariants are pinned by tests.

## Verification record

Host/toolchain: Apple arm64, macOS 26.5.2, Swift 6.3.3, Xcode 26.6. Test timings are execution records, not comparable performance benchmarks. Existing test tasks run concurrently and are unsuitable for kernel speedup claims.

| Check | Result |
|---|---|
| `swift test` | Exit 0; 1,144 Swift Testing tests in 168 suites passed; 91 XCTest cases, 27 skipped, 0 failures. Swift Testing phase 392.400 s. |
| `swift test -c release` | Exit 0; 1,144 Swift Testing tests in 168 suites passed; 91 XCTest cases, 27 skipped, 0 failures. Swift Testing phase 22.518 s. |
| Python CI-guard tests | 25 passed. |
| GitHub policy validator | 9 YAML/template files and workflow policies validated. |
| Documentation consumer smoke | 7 standalone programs built and ran successfully. |
| CI-pinned SwiftLint 0.65.1 | Exit 0; 293 warnings, 0 errors, 204 files. Initial sandbox cache-write error was resolved by rerunning with cache access. |
| Targeted evidence | Eight standalone clients were compiled against current Release objects and executed by the orchestrator; vector, hint, cache, metric, heap, collection, batch, quantization and factorization failures persisted. The final client additionally corroborated compact matrix aliasing, generic-element retention and empty binary round-trip failure. |

Eight reproduction sources and their [Release results](evidence/release-results.json) are preserved with the report; see [replay instructions](evidence/README.md). Additional evidence/logs remain locally under [/private/tmp/vectorcore-review-2026-09-14](/private/tmp/vectorcore-review-2026-09-14), including component probe sources, Debug/Release logs, lint JSON and consumer logs. Memory/pool instrumentation uses exact source with documented allocation counters or small dependency stubs; this distinction is preserved in the component reports. No binaries or full temporary build outputs were added to the repository.

**Limits:** no end-to-end sibling build against this local revision, GPU execution, sanitizer run, Intel/device runtime test, sustained corpus benchmark, allocator-failure campaign or peak-RSS profile was completed. CI configuration was inspected; its remote jobs were not rerun. No conclusion of universal correctness follows from a green suite. All proposed speedups require coordinated Release measurement after correctness repairs.

## What is already strong

- Current shared Top-K comparator/merge tests exercise NaNs, signed zeros, original-index ties and route boundaries; no new defect was found in that repaired ordering core.
- Existing MemoryPool saturation and typed-byte-retention fixes appear intact; the newly found pool issues concern other implementations.
- The LAPACK seam's layouts, lifetime scopes and ordinary QR/SVD/eigen invariants are sound in the reviewed paths, and LAPACK provided correct independent results for the failing Swift fixtures.
- Binary framing centralizes size/version/CRC checks before payload allocation; fixed FP32 Codable initializers validate counts.
- Frozen SoA layout, scoped buffer access, explicit ownership transfer and the graph interchange boundary are useful foundations worth preserving.

The main investment should be **fewer independent implementations of each contract, stronger invariant tests, and bounded resource behavior**. That addresses both correctness and scalable performance more directly than stylistic cleanup or additional optimization layers.
