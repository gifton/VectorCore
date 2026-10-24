# E02 caller and scheduling inventory

Read-only source inventory for API-04/27/28/42/48, not implementation acceptance. Read historical E02 WORK_ITEMS and CLAIMS entries, inspected the named provider/batch/execution/transform/matrix sources and actual Sources/Tests/Benchmarks references. Current snapshot includes accepted E03/N01/N02/R02; isolated N03/E04 are not treated as imported. No builds, production/test edits, removals or delegation. Architecture/removal decisions remain with root.

`E02-reachability-inventory.json` records HEAD, twelve relevant source hashes, 44 comment-filtered lexical symbol references and all 30 current source `group.addTask` sites. It includes declarations and is not a compiler call graph; the access/overload conclusions below come from manual inspection. Reproduce with `rg -n '(ExecutionOperations|SyncBatchOperations)\.' Sources Tests Benchmarks`, whole-word searches for each method name (including unqualified calls), `rg -n 'addTask|withTaskGroup|withThrowingTaskGroup' Sources/VectorCore`, and inspections of extension access modifiers. Comments/examples are excluded from reachability conclusions.

## API-42: live versus dormant engines

| Family | Actual incoming route | Current disposition |
|---|---|---|
| ExecutionOperations.map, Transform add/multiply | Public Array.mapElements/adding/multiplying in TransformOperations.swift:126/136/141; add/multiply call map unqualified | Live public conveniences, even where no direct repository test caller exists |
| ExecutionOperations.normalize | Public Array.normalized at TransformOperations.swift:131; also cosineSimilarityMatrix; N01 NormalizationRangeContractTests:157/308 exercise normal and factory-error routes | Live; preserve N01 behavior |
| ExecutionOperations.distanceMatrix | Public Array<Vector<Dim768>>.cosineSimilarityMatrix at TransformOperations.swift:152/154 | Live; internal declaration is not evidence of dormancy |
| ExecutionOperations.findNearest and sequential/parallelCPU/GPU helpers | Only internal findNearestBatch calls findNearest at ExecutionOperations.swift:209/225; internal nearest calls the helpers | Dormant internal component; no executable external incoming edge found |
| ExecutionOperations.findNearestBatch | No Sources/Tests/Benchmarks caller | Dormant root of the nearest component |
| ExecutionOperations.reduce/parallelReduce | reduce calls parallelReduce; parallelReduce recursively calls itself | Dormant component; no public Array reduction forwarding |
| ExecutionOperations.centroid | No executable caller | Dormant; tests named ExecutionOperationsTests call **Operations.centroid**, not this method |
| validateDimensions | Only dormant findNearest:47 | Dormant with nearest |
| validateNonEmpty | Only dormant findNearestBatch:193/194 and centroid:431 | Dormant with those roots |
| validateConsistentDimensions | findNearestBatch:195, centroid:432, **live distanceMatrix:381** | Must retain/repair for live matrix; cannot delete with dormant validation family |
| Transform combine | Only block-comment example calls at TransformOperations.swift:178/179 | Dormant; no public combining Array convenience found |
| CPUContext.withParallelExecution | No caller in Sources/Tests/Benchmarks | Dormant despite its add-all loop; optimalChunkSize has only this caller |
| CPUContext.execute | Live transform/map/normalize/matrix serial branches through ExecutionContext | Live; CPUContext type is internal but ExecutionContext protocol itself is public |

`SyncBatchOperations` is an internal namespace. Its Array extension at SyncBatchOperations.swift:704 is explicitly **internal**, unlike the public async Array<VectorType>.findNearest/pairwiseDistances conveniences in VectorCore.swift:92–107, which forward to **BatchOperations**. No executable caller of the internal Sync Array.findNearest/pairwiseDistances/centroid/batchStatistics methods was found in this checkout. Similar names in current tests bind to public Operations/BatchOperations or collection APIs.

Internal Sync dependency chains were checked, not just qualified references:

- findNearest reaches its own serial optimized distance helpers, selectTopK and heapSelect. No external incoming edge beyond the unused internal Array.findNearest forwarder.
- mean and updateCentroids invoke centroid unqualified; internal Array.centroid also forwards to it. None of these roots has an executable caller. weightedCentroid/sum/assignToCentroids/batchDistances/findWithinRadius/map/mapInPlace/filter/partition have no incoming executable usage found outside declarations or their own dormant component. The old weighted-centroid test mentions removal only in a comment.
- stratifiedSample calls randomSample unqualified; neither is reached from live repository code.
- findOutliers calls statistics unqualified. Array.batchStatistics forwards to statistics. In the separate **internal extension SyncBatchOperations** at NaNInfinityHandling.swift:415, finiteStatistics:446 calls filterFinite and then statistics unqualified. finiteStatistics itself has no caller. findNonFinite likewise has no caller found. These are not public statistics conveniences, but they are real dependency edges and must not be overlooked.

Root's candidate removal of dormant Execution nearest/batch-nearest/reduce/centroid and CPUContext.withParallelExecution, plus pruning Sync to statistics and Array.batchStatistics while leaving the NaNInfinityHandling extension unchanged, has no executable consumer outside removed code/comments in the inspected repository. This is a bounded caller finding, not removal authorization. Retained statistics is also pending N03 work; its dormant status does not justify deleting or invalidating that owned task. Preserve live validateConsistentDimensions, OperationContext/E04 ownership and public transform forwarding. No downstream runtime inference was made.

## API-04: actual task submission inventory

All listed task groups enqueue their selected work before draining; this alone does not mean unbounded task counts. Let N be item count, Q query count, C the CPU provider's positive processor count and T the Batch default targetChunks.

| Source/symbol, addTask lines | Submitted task count | Reachability / bound |
|---|---|---|
| ComputeProvider.parallelExecute:141 | N | Public default, unbounded by deviceInfo.maxThreads |
| ComputeProvider.parallelForEach:163, including minChunk forwarding | N | Public default; minChunk is validated but does not change task selection |
| ComputeProvider.parallelReduce:198, including minChunk forwarding | ceil(N/chunk), chunk=ceil(N/min(N,saturating 2*max(1,maxThreads))) | Already source-bounded by at most min(N,saturating 2*max(1,maxThreads)); not per-item |
| CPUComputeProvider parallelExecute:130; foreach:195/279; reduce:236/321 | ceil(N/max(256,ceil(N/min(N,C)),optional minChunk)) | Already at most C tasks per call. TaskLocal isInsideParallelRegion makes nested CPU calls sequential. This is not a bound for arbitrary other providers or independently created parent tasks |
| BatchOperations.process:199 | ceil(N/effectiveBatchSize) | Public; N>=default threshold selects parallel; batchSize1 creates N children |
| BatchOperations.map:308 and filter:338 | N | Public; threshold chooses sequential versus add-all |
| BatchOperations.statistics:375 | ceil(N/optimalChunkSize) | Public; current default hardware-scaled chunk heuristic already bounds normal task count; N03 is not imported |
| BatchOperations generic nearest:521 and six optimized nearest helpers:783/812/841/871/900/929 | ceil(N/optimalChunkSize) | Public findNearest route, heuristic selects parallel; already hardware-scaled coarse chunks |
| BatchOperations.computePairwiseParallel:581 | N | Public pairwise fallback, n>=100; GEMM may bypass for supported optimized metric/type/config |
| ExecutionOperations.map:276, Transform normalize:54 | N | Live public Array conveniences at N>=1000 |
| ExecutionOperations.distanceMatrix:404 | N(N-1)/2 | Live Dim768 cosine matrix when N*N>=1000 |
| TriangularMatrix.buildParallel:251 | N(N-1)/2 | Public entry point even though no direct repository caller found |
| ExecutionOperations.findNearestBatch:224 | Q, each potentially launches nearest chunk tasks | Dormant component; do not activate as a scheduler replacement |
| ExecutionOperations.findNearestParallelCPU:136 | ceil(N/max(preferredChunkSize,floor(N/maxThreadCount))) | Dormant; conditional hardware bound assumes valid positive context fields; unchecked arithmetic remains |
| ExecutionOperations.parallelReduce:345 | Coarse chunks, recursively repeated | Dormant; per-level count is hardware-related with positive context count, but arbitrary combine/order/initial semantics differ |
| ExecutionOperations.centroid:464 | ceil(N/max(1,floor(N/maxThreadCount))) | Dormant, assumes valid positive context count |
| Transform combine:106 | N | Dormant despite async task group |
| CPUContext.withParallelExecution:162 | N | Dormant; outer chunk loop does not bound children because inner loop creates one task per index |

Batch optimalChunkSize at BatchOperations.swift:626 uses fixed default oversubscription2 and minimum256, not the mutable actor configuration. For valid representable arithmetic, its nearest-power-of-two rounding is at least half the positive ideal floor chunk; a conservative count bound is <=4*T, with T=2*activeProcessorCount, rather than N children. This is a source-derived bound under those arithmetic assumptions, not a scheduling throughput claim. Its floating/integer/shift arithmetic and configuration consumption are E06-sensitive; do not casually connect the mutable settings during E02. All optimized nearest helper overloads use the same chunk routine. Their eager Array.chunked materialization is separate from child-count bounds.

BatchKernelProvider.findNearestBatch:47 adds no direct group. It delegates Q indices to `self.parallelExecute` and then calls self.findNearest; thus its bound depends on the actual provider witness. Public Operations.findNearestBatch reaches it for an installed BatchKernelProvider. With a default conformer it inherits the per-query add-all default; with CPUComputeProvider-style bounded overrides its scheduler may be bounded. No sibling execution was performed.

## Actual provider/caller evidence and execution laws

Public Operations uses computeProvider.parallelExecute for generic query work and parallelReduce for optimized TopK/range kernels. Benchmarks/VectorCoreBench/Suites/BatchBench.swift has actual range-reduction calls (for example lines321/330/362/376), so provider changes affect executable benchmarks, not merely docs. ProviderConformanceTests.swift:47–76 checks ordered exactly-once map/foreach and identity-based reduction; lines201 onward compare CPU modes with identity initial. ExecutionBoundaryTests.swift:36–49/204/206 covers default/CPU range and minChunk boundaries. BatchKernelProviderTests exercises actual Operations delegation. BatchGEMMRoutingTests and OperationsBatchTests reach pairwise routes; TopKNaNWrapperTests reaches generic/optimized nearest; ExecutionBoundaryTests:224/237/245 reaches process ordering and throwing bodies. Test names alone are not evidence of ExecutionOperations engine use.

No explicit Task.checkCancellation, Task.isCancelled or addTaskUnlessCancelled occurs in the inspected scheduling files. Structured throwing groups cancel siblings after errors, but user async bodies must cooperate, and synchronous chunks do not automatically stop. Nonthrowing Batch nearest/statistics/pairwise and Triangular build APIs need a stated completion/cancellation policy rather than silently returning incomplete values.

CPUComputeProvider.parallelReduce returns rangeWork(items) on its nonempty sequential branch and combines partials into initial on parallel completion. Default reduction also combines completion-order partials into initial. Existing tests deliberately require an identity initial; nonidentity input is not evidence of a newly introduced defect under that law. Associativity and floating-point ordering limits need explicit documentation; noncommutative combination also needs an ordering decision because present partial combination uses completion order. Dormant ExecutionOperations.reduce additionally injects initial into every chunk and recursively combines, so it is not a lawful generic sequential fold with arbitrary initial.

CPUComputeProvider.executeBatch:346 is public in a public extension but has no current executable Sources/Tests/Benchmarks caller. For nonempty candidates it calls the supplied kernel exactly once over 0..<N; for empty input it calls zero times. It ignores tuning/threshold/chunk promises and does not validate output capacity here. Its private measureProbes:364 has no caller. Do not infer active autotuning from their existence or activate unsafe shared output writes just to match the old documentation.

## Carry-forward hazards requiring root attention

1. **Live factory errors trap:** ExecutionOperations.map serial/parallel paths use `try! V.create` at268/286. Public Array.mapElements/adding/multiplying are async throws yet cannot propagate a custom conformer's legitimate factory rejection. N01 normalize already uses throwing factories. Root has been alerted; retaining/refactoring live map should preserve output order and propagate the original error.
2. **Live matrix weak validation:** validateConsistentDimensions checks only every100th row in Release; distanceMatrix invokes it before metric work. The currently exposed Dim768 convenience has a fixed shape, so this does not by itself reproduce dynamic-row unsafe access through that convenience. Nevertheless the retained generic helper must not become a newly reachable weak-validation route if architecture changes. Its N*N threshold multiplication is unchecked. Matrix diagonal remains zero and mirrored off-diagonal assumes symmetry; cosineSimilarityMatrix currently feeds negative dot after normalization. These existing mathematical policies belong to their appropriate contract repair, not an accidental scheduling rewrite.
3. **CPUContext loses task context:** execute:92–107 always reaches queue.async with current initializers, resuming via continuation. Cancellation/task-local checks inside that GCD block do not observe the original Swift task as if it were executing inline. Retaining this route while promising cooperative task-boundary cancellation requires explicit treatment. Priority execute additionally uses an unstructured Task. CPUContext's thread count is not validated; its dormant chunk helper divides by it.
4. **Do not activate dormant shape hazards:** nearest checks only the first row in Release; centroid then uses forced base addresses and first-row dimension in raw provider calls, including zero-dimensional inputs. These remain dormant under the caller map. Deletion or deliberate repair is preferable to routing live APIs through them without new exhaustive validation/ownership work.
5. **Distinct bounds:** child-task limits do not bound dense matrix output allocation or guarantee simultaneous CPU thread count, and a per-call worker limit does not bound arbitrary nesting across providers. Root must specify the chosen observable scope before implementation.

The candidate dormant-removal set is supported by this source inventory. No correctness closure, benchmark result, downstream execution or implementation approval is implied.
