# Component 03 — execution, concurrency, pools, configuration

Revision reviewed: `8b02e75bca7055ae11170873d15c06f1538709b7` (main). Review only; no repository edits or shared builds. Swift concurrency skill read and applied. Package uses Swift tools 6.0, enables StrictConcurrency/ExistentialAny, and sets no custom default actor isolation or NonisolatedNonsendingByDefault feature. Standalone consumer probes explicitly used `-swift-version 6`.

## Summary

Seven actionable candidate findings below, all P2. Strongest at deployment scale: actor-pool teardown leaks, unenforced pool byte limits, and unbounded task creation on public/downstream routes. Configuration defects are independently reproduced. CPU provider's new task-local nesting suppression and MemoryPool's synchronous lock bookkeeping address the older specific concurrency problems; I found no evidence those old fixes regressed.

## Findings

### EC-1 — P2 — Cached raw allocations leak when actor pools are destroyed

**Category:** resource correctness / architecture. **Location:** `Sources/VectorCore/Providers/SwiftBufferPool.swift:48-51`; repeated class at `Sources/VectorCore/Operations/BufferPool.swift:174-201` (TypedBufferPool), and the internal BufferPool has manual cleanup but no deinitializer.

**Trigger:** construct a scoped `SwiftBufferPool`, acquire and release an ordinary poolable buffer, then drop the pool without explicitly invoking `clear()`. The public initializer allows pools per pipeline/request. Normal release puts the raw pointer in `pools`; actor destruction destroys pointer values, not their allocations. The empty deinit comment incorrectly says actor deinit handles cleanup. Repeated scoped lifetimes leak every retained allocation. TypedBufferPool has the same missing owner teardown and has no public clear operation.

**Evidence / confidence:** verified, high. `ExecutionPoolProbe.swift` compiles the actual unchanged SwiftBufferPool/BufferProvider sources with an instrumented AlignedMemory.deallocate dependency (real allocations and frees). Output: `poolDestroyed=true cachedBuffersFreed=0` after releasing one 1024-byte allocation. Weak reference confirms actual actor destruction. The probe manually frees that leaked pointer afterward. This is independent of the fixed MemoryPool class.

**Remedy / tests:** make allocation ownership explicit and free all cached raw allocations during owner teardown. Test the class of owners with counted allocate/free boundaries: normal clear, normal destruction without clear, empty pool, and active-handle lifetime policy. Treat initialized generic elements separately: raw pool teardown should not silently invent typed initialization ownership.

### EC-2 — P2 — SwiftBufferPool does not enforce maxPoolSize

**Category:** resource correctness / scale. **Location:** `Sources/VectorCore/Providers/SwiftBufferPool.swift:100-115,164-191`.

**Trigger:** acquire several simultaneous buffers, then release them within one minute. Every returned buffer is stamped with the current time, appended, and cleanup only removes entries older than 60 seconds. No eviction is performed to satisfy the configured byte budget. No timer invokes cleanup after the burst ends. Thus an arbitrarily large fresh burst remains cached indefinitely until another release or explicit clear, even when maxPoolSize is zero. currentUsageBytes excludes this cached memory, so statistics after release can show zero while retained allocation bytes exceed the configured limit.

**Evidence / confidence:** verified, high. With maxPoolSize=1024 and maxBufferSize=1024, acquire eight 1024-byte buffers, release all, then reacquire eight: `cap=1024 returned=8192 reused=8 frees=0`. All eight survived a budget permitting one. Source demonstrates retention after inactivity; the probe does not wait a minute.

**Remedy / tests:** track cached bytes and evict/reject returns until within the byte budget, independent of age. Test a simultaneous burst, zero budget, exact limit, mixed size buckets, and reuse after eviction. Expose active and cached byte counts separately if statistics remain public. Apply the fixed MemoryPool retention test pattern to all pool implementations, not only MemoryPool.

### EC-3 — P2 — Public batch settings are stored but ignored

**Category:** public semantics / architecture. **Location:** `Sources/VectorCore/Operations/BatchOperations.swift:175-178,293-295,323-325,356-357,618-631`.

**Trigger:** call `await BatchOperations.updateConfiguration { $0.defaultBatchSize = 2 }`, then `process` without an explicit batch size. The operation reads immutable `defaultConfig`, not the actor-held configuration. Likewise parallelThreshold, oversubscription, and minimumChunkSize never affect the corresponding operation branches/chunk helper. Only the pairwise matrix-routing settings read the live configuration.

**Impact:** applications cannot constrain their processing batch size or tune the documented CPU scheduling controls. Transform closures observe different batches than requested; memory/scheduling tuning is ineffective.

**Evidence / confidence:** verified, high. Existing-library consumer probe configured batch=2 and processed 5 vectors with transform `{ [$0.count] }`: observed `[5]`, expected `[2,2,1]`. Source search proves defaultConfig is used for the other four scheduling controls. No `BatchOperations.process/map/filter` tests were found; BatchGEMMRoutingTests only exercises live matrix settings.

**Remedy / tests:** snapshot current configuration once at entry and pass that snapshot through internal helpers. Validate before making dormant settings effective: oversubscription zero/NaN/infinity/huge values, nonpositive chunk/batch sizes, and integer products/conversions otherwise arm division/conversion/stride traps. Add behavior tests for each setting, not just round-trip configuration storage.

### EC-4 — P2 — Concurrent configuration updates lose changes

**Category:** concurrency correctness. **Location:** `Sources/VectorCore/Operations/BatchOperations.swift:99-102`.

**Trigger:** concurrent callers update independent fields or increment a field. Each call separately awaits get, mutates a local copy, then awaits update. The actor protects each primitive but not the read-modify-write transaction. A later write can replace a configuration based on a stale snapshot and erase another caller's update, including matrix-routing policy changes.

**Evidence / confidence:** verified, high. Consumer probe sets parallelThreshold=0 and launches 100 structured tasks, each invoking updateConfiguration to increment it; result was 28 instead of 100. Exact lost count is schedule-dependent; the interleaving is directly derivable from the two awaits.

**Remedy / tests:** perform the synchronous mutation closure inside one actor-isolated method; use a Sendable closure at the public concurrency boundary. Test concurrent independent-field updates and increments with exact final state. A Swift 6 main-actor call with an ordinary mutation closure currently diagnoses sending a non-Sendable closure; the nonisolated consumer probe avoids that separate ergonomics issue.

### EC-5 — P2 — Default and legacy schedulers enqueue unbounded task counts

**Category:** architecture / performance at scale. **Primary locations:** `Sources/VectorCore/Protocols/ComputeProvider.swift:128-134,148-154`; `Sources/VectorCore/Operations/BatchOperations.swift:183-196,298-305,328-335`; `Sources/VectorCore/Operations/ExecutionOperations.swift:401-409`.

**Trigger / mechanism:** ComputeProvider defaults eagerly add one child per item before consuming any result. They do not use maxConcurrency. BatchOperations.map/filter add one task per vector; process adds one per batch and copies each batch while enqueueing (batchSize=1 is one task per vector). These bypass CPUComputeProvider's bounded chunk scheduler and nesting task-local marker. The internal distanceMatrix path queues n(n-1)/2 children before receiving results; it is publicly reachable through Array<Vector<Dim768>>.cosineSimilarityMatrix at TransformOperations.swift:164-169. At n=10,000 this is 49,995,000 children for a result whose scalar storage is about 400 MB.

**Downstream reachability:** `../future/VectorAccelerate/Sources/VectorAccelerate/Integration/MetalComputeProvider.swift:91-92` explicitly inherits the defaults. BatchKernelProvider.findNearestBatch (Core protocol file:50) dispatches through inherited parallelExecute, so a large query batch enqueues every query concurrently even if the provider reports a small maximum. Its actor can suspend at GPU awaits, so actor isolation is not a bound on outstanding operations.

**Evidence / confidence:** verified boundedness mismatch, high; resource impact is source-derived, not a timing claim. A consumer conformer with maxConcurrency=1 and maxThreads=1 inherited parallelForEach over 64 indices; actor instrumentation around a short suspension measured `observedPeak=64`. CPUComputeProvider itself now creates at most roughly the processor-count number of chunks, but these routes do not use it. `process`'s “proper backpressure” claim is unsupported by its add-all loop.

**Remedy / tests:** introduce a bounded worker/chunk scheduler or refill a window as tasks finish, retaining index order as needed. Route all public batch fronts through it or define their independent limits. Count active bodies with an actor and assert a bound; test nested public APIs and protocol-default conformers. A mechanical inventory/test for eager add-all patterns is warranted because the defect class repeats. Do not assert speedup without coordinated Release benchmarks.

### EC-6 — P2 — Invalid batchSize traps or silently discards all work

**Category:** validation / correctness. **Location:** `Sources/VectorCore/Operations/BatchOperations.swift:175,186-188,535-536`.

**Trigger:** process(nonemptyVectors,batchSize:0) or batchSize:-1. Zero reaches stride(by:0) and traps inside an async throwing API; a negative stride toward a positive upper bound performs no iterations and returns an empty array without invoking the transform.

**Evidence / confidence:** verified, high. Consumer probe with one DynamicVector and batchSize=-1 returns `[]`; subprocess batchSize=0 exits -5 with `Swift/Stride.swift:348: Fatal error: Stride size must not be zero`. No batch-size positivity precondition or throwing validation appears in the public docs/source.

**Remedy / tests:** validate strictly positive batch size before choosing serial/parallel paths and throw VectorError.invalidConfiguration/invalidData consistently. Cover empty/nonempty input with zero, negative, one, oversized valid batch size, and both serial/parallel branches. Combine validation with EC-3 before honoring configurable defaultBatchSize.

### EC-7 — P2 — Chunk endpoint arithmetic traps for a tiny valid range near Int.max

**Category:** integer correctness. **Location:** `Sources/VectorCore/Providers/CPUComputeProvider.swift:128` (also 193,234,276,317).

**Trigger:** `CPUComputeProvider.parallel.parallelExecute(items:(Int.max-2)..<Int.max) { $0 }`. This valid two-element Range has no allocation-size problem. The provider picks a minimum chunk of 256 and computes start+chunk before min(...,upperBound); that addition traps even though the true end is representable. The same endpoint expression is duplicated across all five chunked CPU helpers.

**Evidence / confidence:** verified, high. Existing-library consumer subprocess exits -5 for the two-element range; source directly identifies the overflowing operation. Sequential mode can iterate this range, so execution mode changes a valid request into a trap.

**Remedy / tests:** add `min(chunk, upperBound-start)` to start (or use a remaining-count partition helper), and avoid overflow-prone ceiling division. Parameterize all CPU helper variants over nonzero lower bounds, empty ranges, near-Int.max short ranges, and large minChunk hints; assert exact index coverage/order and no trap. The protocol default reduction has similar arithmetic and should use the shared helper.

## Strengths and rejected suspicions

- MemoryPool uses a synchronous NSLock around state mutations; no async work waits behind queued barriers. BufferHandle retains a weak pool and directly frees its aligned pointer if the pool is already gone. Byte retention accounting uses stride and type buckets. MemoryPoolTests and the standalone MemoryPoolRegression fixture explicitly cover saturation, late handle destruction, multi-type retained-byte caps, and cleanup. I did not rerun these shared tests (orchestrator owns builds).
- CPUComputeProvider checks its task-local nested-region marker in all five async helpers and binds the marker around structured task groups. Child tasks inherit that marker. This closes nested CPU-provider calls; it does not cover unrelated BatchOperations task groups, arbitrary downstream task groups, detached tasks, or BLAS threading. No broader guarantee is claimed here.
- Operations providers are TaskLocal values rather than mutable global provider references. Reads in structured child tasks preserve the scoped binding. Public generic ComputeProvider/BufferProvider values are Sendable and SwiftBufferPool mutation remains actor-isolated.
- LogConfiguration's mutable minimumLevel is consistently locked. ThreadSafeConfiguration isolates stored value access. Error types use value-state and Sendable fields; no concrete race was identified in these components.
- parallelReduce omits initial on CPU sequential paths, but the existing conformance suite deliberately requires an identity initial; the historical provider law also explains this. Treat moving that law into current protocol doc comments as documentation work, not a new numeric defect. Floating-point addition is not exactly associative, so bitwise mode-invariance is not implied by these tests.
- Generic MemoryPool buffers do not manage initialized element lifetimes. Current Memory_Alignment.md explicitly assigns initialization/destruction obligations to callers, so I did not relabel that as a new leak. Forged raw BufferHandles and post-release use similarly violate unsafe ownership expectations.
- A minimal Swift 6 external TypedBufferPool<Float> acquire/initialize/release consumer typechecks; do not report it as unusable merely because UnsafeMutableBufferPointer lacks Sendable.

## Other opportunities / limits

- Cancellation is cooperative and these schedulers generally do not check Task cancellation between iterations. Throwing child groups cancel siblings, but CPU-only work that does not check cancellation continues through its chunk. Add an explicit cancellation contract and tests before promising responsiveness; this was not elevated above verified defects.
- CPUComputeProvider.executeBatch's public “auto-tuned mode and chunking” comment contradicts its unconditional single full-range kernel call (344-356); explicitMinChunk/explicitThreshold and measureProbes are not used to implement that promise. This is misleading public surface/dead implementation, not a measured regression.
- AutoTuning holds a process-wide NSLock while calling its calibrator (102-118). Calls using unrelated keys serialize, and reentrant get/set/calibrateIfNeeded from the calibrator or custom metric can deadlock. Not runtime-probed; consider calibrating outside the lock with an in-flight key mechanism. The 20 ms cumulative cap is checked before a whole callback and does not bound one slow callback.
- ComputeDevice availability/capabilities include placeholders: gpu index is not validated, neural availability equates to processorCount>0, and hardware metadata is hard-coded. Do not use mostCapable/queryCapabilities as actual runtime device discovery. Core's CPU-only boundary is documented, but public metadata still merits cleanup.
- Logger's debug/info autoclosures are evaluated before private log checks minimumLevel, so disabled logging still constructs messages. PerformanceTimer uses wall-clock CFAbsoluteTime, not a monotonic clock. These are source-derived opportunities; no timing benchmark was performed.
- MemoryPool.returnBuffer builds flatMap arrays across pooled entries while holding its global lock; measure high-contention workloads before deciding whether a cached retained-byte counter is worthwhile.
- FP16 diagnostics/cache are owned by the reduced-precision reviewer; no overlapping review performed.
- No TSAN, GPU, sustained memory benchmark, or full Debug/Release suite was run by this component. Pool teardown probe substitutes only the deallocation dependency. Public behavior probes link existing Debug package objects; they do not claim fresh full-build verification.

## Coverage

Read completely:

- Providers/AutoTuning.swift, CPUComputeProvider.swift, SwiftBufferPool.swift
- Protocols/ComputeProvider.swift, BufferProvider.swift
- Operations/CPUContext.swift, ExecutionContext.swift, BufferPool.swift
- Execution/ComputeDevice.swift
- Configuration/ThreadSafeConfiguration.swift
- Utilities/MemoryPool.swift, ParallelHeuristic.swift, TimeExtensions.swift, ErrorHandlingUtilities.swift
- Progress/OperationProgress.swift
- Logging/Logger.swift
- Errors/VSKError.swift, VectorError.swift

Consumer tracing: Operations.swift provider bindings, findNearestBatch routing, reductions, computeDistances calibration/provider routing; BatchOperations.swift configuration, process/map/filter/statistics, chunking, pairwise execution, and specialized parallel helpers; ExecutionOperations.swift and TransformOperations.swift; Protocols/BatchKernelProvider.swift; downstream MetalComputeProvider inheritance and batch routes. Searched VectorIndex provider/pool use (its score pools are separate implementations).

Read relevant tests: ProviderConformanceTests.swift, MemoryPoolTests.swift, MemoryPoolRegression/{Probe,AllocationSupport}.swift, ExecutionOperationsTests.swift, BatchGEMMRoutingTests.swift. Source/tests searches found no scheduling cap/nesting tests, BatchOperations.process/map/filter tests, or SwiftBufferPool destruction/budget tests.

Read docs: Package_Boundaries.md, API_Overview_Map.md, Numerical_Behavior.md, Memory_Alignment.md relevant pool/unsafe contracts, historical DOCUMENT-4_S4_Provider_Conformance.md.

## Probe artifacts / exact results

- `ExecutionPoolProbe.swift`, binary `execution-pool-probe`: actual SwiftBufferPool+BufferProvider source with counting free boundary.
- `ExecutionPublicProbe.swift`, binary `execution-public-probe`, linker response `execution-objects.rsp`: public consumer linked against existing Debug objects; normal / overflow / zero-batch modes.
- `ExecutionTypedPoolCompileProbe.swift`: Swift 6 typecheck passed.

```text
cap=1024 returned=8192 reused=8 frees=0
poolDestroyed=true cachedBuffersFreed=0
configuredBatch=2 observedBatches=[5]
negativeBatchResults=[]
100ConcurrentConfigIncrements=28
providerMaxConcurrency=1 observedPeak=64
overflow exit -5
zero-batch exit -5 Swift/Stride.swift:348: Fatal error: Stride size must not be zero
```

Repository status remained only the user's untracked `.antigravitycli/` and `GEMINI.md`.
