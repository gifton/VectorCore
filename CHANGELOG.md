# Changelog

All notable changes to VectorCore will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The changes below are present after `v0.3.3`; they are not included in that
published tag. A dependency resolved to `0.3.3` does not receive them.

### Added

- **`VectorCoreCompiledFeatures` (V02).** `VectorCoreCompiledFeatures.useCKernels` reports whether the
  linked VectorCore was compiled with `VC_USE_C_KERNELS`, and `VectorCoreCompiledFeatures.configuration`
  reports `"debug"`, `"release"` or `"unknown"` from the `VC_SWIFTPM_DEBUG` and `VC_SWIFTPM_RELEASE`
  conditions that the `VectorCore` target now defines per build configuration. They are diagnostics for
  two compile conditions, not binary provenance. Three existing internal `Interop` declarations became
  `@usableFromInline` for existing inlinable INT8 callers (bodies, gates and public source access are
  unchanged); rebuild as for every binary built against 0.3.3 (see the binary layout entry on
  `scalarCount`).

- **`ComputeDevice.isSupportedByCore` and `ComputeDevice.coreSupportedDevices`.** They report
  VectorCore's own compute providers: `true` and `[.cpu]` for `.cpu`, `false` for `.neural` and every
  `.gpu(index:)`, always equal to `createProvider() != nil`. They say nothing about whether GPU or
  Neural Engine hardware is present, and nothing about other packages' providers (`supportListAndProviderAgree`).

- **`BatchSearchCPUStrategy` and the `cpuStrategy:` parameter of
  `Operations.findNearestBatch`.** The new final parameter (default
  `.perQuery`) selects the CPU route when no `BatchKernelProvider` is
  installed. `.gram(queryTileRows:candidateTileRows:maximumTemporaryPayloadBytes:)`
  is an explicit tiled Gram (GEMM) route for `EuclideanDistance` and
  `CosineDistance` on `Vector512Optimized`, `Vector768Optimized` and
  `Vector1536Optimized`, at any batch size: caller-chosen tile rows; a checked
  plan; a limit on its logical temporary payload (named element slots: the
  packs, the score tile, the norms and the heaps, which is not an allocator,
  live-memory or resident-size limit); errors before any work
  (`.invalidConfiguration`, `.unsupportedOperation`, `.resourceUnavailable`,
  with no fallback to `.perQuery`); and cancellation checks between tiles. Its
  documented worst-case error bound rests on an unverified assumption about
  Accelerate's inner products. Tests: `P02GramPlanTests`, `P02GramRouteTests`,
  `P02GramCancellationTests` and `P02GramAccuracyTests`. On one test host
  (Apple silicon, Release; P02 phase G allocation probe, not a test), in 76
  measured searches the allocations of the packs, score tile, norms and heaps
  matched the payload the plan reports, byte for byte; the array headers, the
  heap array, Accelerate's own allocations (at some tile shapes one per BLAS
  call) and the result came on top of that payload, as documented.

- **Opt-in `boundary` benchmark suite.** `vectorcore-bench --suites boundary`
  adds reused-output counterparts, setup-inclusive companions and consumer-only
  records for squared Euclidean distance at 512, 768 and 1536 dimensions. It is
  never part of the default suites, is not selected by `--ab-only`, and declares
  no A/B comparison; SoA-FP16 entries are reported as unavailable, never timed.
  See [Benchmark measurement boundaries](Docs/Benchmark_Measurement_Boundaries.md#the-opt-in-boundary-suite).

### Changed

- **Breaking: fixed-size vector storage must have the type's lane count (S01).** On the optimized FP32
  types and the INT8 types, a raw, replacement, nested or `inout` storage write with the wrong lane count
  now traps with a required-count diagnostic in Debug and Release builds. The public `storage` of
  `Vector384Optimized`, `Vector512Optimized`, `Vector768Optimized`, `Vector1536Optimized`, `Vector512INT8`,
  `Vector768INT8` and `Vector1536INT8` is now a computed property (a guarded setter and a `_modify` that
  checks the count) over private backing storage: valid source compiles unchanged, but the stored layout
  changes, so binaries must be rebuilt. The INT8 types gain explicit `Codable` (`init(from:)`,
  `encode(to:)`) that throws `DecodingError.dataCorrupted` for a malformed storage shape; the keys
  `storage` and `quantizationParams` and the wire representation are unchanged. `DynamicVector`'s
  dimension follows its current storage count, so replacing its storage can change its dimension. Custom
  conformers must still report a truthful `scalarCount` and buffer shape; that is not mechanically
  enforced.

- **Breaking: `Operations.centroid(of:)` and `Operations.statistics(for:)` now throw, and dimensions are
  checked before mutation and aggregation (S02).** Both scan every row's `scalarCount` before reading any
  payload. An empty `centroid(of:)` throws `.insufficientData`; an empty `statistics(for:)` keeps the
  all-zero summary. `Operations.centroid` and `Operations.normalize` propagate generic factory errors
  unchanged. The three binary mutator defaults reject unequal `scalarCount` before either borrow (a named
  `fatalError`). In the vector factory, a negative dimension for random, zeros, ones and pattern
  construction stops at a named boundary before any work (zero is allowed); single-vector construction
  throws `.invalidDimension` for a negative dimension before count matching (zero with empty data is
  allowed); batch and basis construction require a positive dimension, checked before any modulo or
  division and also for empty data (non-multiples keep `.invalidData`). `indexOutOfBounds` errors report
  `max_index = -1` for a nonpositive dimension and keep the original message, code and index. Migration:
  add `try`, handle `.insufficientData`, and validate dimensions first where a trap is not acceptable.

- **Decoding a `TopKResult` with mismatched counts throws (S03).** When the decoded `indices` and
  `distances` counts differ, decoding now throws `DecodingError.dataCorrupted` at the `distances` key,
  keeping the enclosing coding path; before, the value decoded and a mismatch could be silently
  truncated by `toTuples` and `SearchResults(from:)` or fail at an indexed access. Valid encodings,
  arbitrary labels and the represented values are unchanged.

- **Breaking: the Float search facades take `any DistanceMetric<Float>` and use the supplied metric
  (R01).** `TopKSelection.nearest(k:query:candidates:metric:)`,
  `TopKSelection.batchNearest(k:queries:candidates:metric:)` and `VectorCollection.search(query:k:metric:)`
  (requirement and default) take `any DistanceMetric<Float>` instead of `any DistanceMetric`;
  `OptimizedDistanceMetrics.create(_:)` and `MetricType.createMetric()` return `any DistanceMetric<Float>`,
  and `OptimizedDistanceMetrics.create(named:)` returns `(any DistanceMetric<Float>)?`. The single-query
  and collection facades open the supplied metric and use it, and the batch facade forwards it per query:
  no witness replacement or Euclidean fallback remains, so re-baseline results computed with
  non-Euclidean or custom metrics.
  `create(named:)` compares case-insensitive raw values, including `dotProduct`. Migration: narrow
  `any DistanceMetric` values to `any DistanceMetric<Float>` (or cast and handle a mismatch). A third-party
  `VectorCollection` must change an override `search(query:k:metric:)` that takes `any DistanceMetric`
  to take `any DistanceMetric<Float>`: otherwise the old method stays a mere overload and the protocol
  requirement silently selects the default implementation.

- **Two defaulted protocol requirements can bind an existing client method (R01, R05).**
  `DistanceMetric.relevanceScore(for:)` and `VectorCollection.search(query:k:metric:)`, which now takes
  `any DistanceMetric<Float>`, are requirements with default implementations, so conformers still
  compile. A conforming type that already declares a method with the same signature now implements that
  requirement: calls through the protocol (generic or existential) reach that method instead of the
  default. Rename such a method if it was not meant to implement the requirement. A comparison of the
  public symbol graphs of 0.3.3 and this release found no other requirement added to a public protocol
  that existed in 0.3.3.

- **Breaking: `MutableVectorCollection.update(id:vector:)` has no default implementation, and collection
  IDs are not reused (R02).** The requirement stays, but its protocol-extension default is removed: a
  conformer that inherited it must implement an update that validates and commits within its own
  synchronization or storage transaction (catch-and-reinsert is not a generic guarantee).
  `SimpleVectorCollection` has its own `update(id:vector:)`. Duplicate explicit IDs throw
  `invalidConfiguration`; generated IDs skip occupied IDs and never rewind after removal or `removeAll`;
  exhausting the generated-ID cursor throws `resourceExhausted` without mutation, and explicit insertion
  does not advance it. `SimpleVectorCollection`'s batch and search operations capture bindings under one
  lock and score outside it; the generic defaults do not promise atomic snapshots, and captured bindings
  do not make external reference aliases immutable.

- **Breaking: matrix diagonals are the metric's self score, and `cosineSimilarityMatrix` returns
  similarity (R03).** `BatchOperations.pairwiseDistances` evaluates every diagonal entry with the
  supplied metric, also on its GEMM wrapper, so the diagonal is the metric's self score (for example a
  negative dot-product self score, and 1 for a cosine zero vector) instead of 0. `cosineSimilarityMatrix`,
  the 768-dimensional array convenience, returns cosine similarity (the negated negative-dot score after
  normalization), with finite values clamped to [-1, 1] and similarity 0 for a zero vector; before, it
  returned the negative-dot score. The SoA cosine kernels write only `0..<soa.count` for a zero query
  and leave output capacity beyond the logical count unchanged. The separately documented `VectorMath`
  zero-diagonal helper is unchanged. Custom pairwise metrics must be symmetric; there is no
  directed-metric contract.

- **`KNearestHeap` and the bounded heaps share one ordering and reject negative sizes (R04).** The legacy
  `KNearestHeap` orders numeric distances before NaN, then lower distance, then the smaller index on ties
  (signed zeros and NaN pairs included); a negative `k` is rejected, `k` 0 stays empty, and `Int.max` with
  a small input reserves nothing eagerly. The heap initializers reject a negative capacity before any
  work, and capacity is a retention cap with no eager reservation. Bounded bulk membership matches
  streaming under a strict weak ordering (`<` keeps the largest, `>` keeps the smallest). Duplicate
  identical entries remain allowed and public signatures are unchanged; there is no stable order among
  comparator-equivalent generic records.

- **Execution domains are validated (E01, E02).** A nonpositive `batchSize` or `minChunk` that was
  ignored now throws `invalidConfiguration`; a range whose length cannot be represented throws
  `invalidRange`; an invalid loop domain stops with a precondition failure in Debug and in normal Release
  builds. The default providers require positive concurrency and chunk controls even for empty input
  (nonpositive limits throw `invalidConfiguration`; `invalidRange` keeps precedence).
  `CPUComputeProvider.executeBatch` rejects insufficient output capacity with a named guard before
  calling the kernel. Errors from a live map factory propagate instead of trapping at a `try!`, and matrix
  rows are shape-checked before any metric callback. Validate these arguments yourself where a rejection
  must be recoverable.

- **Throwing execution routes honor cancellation, and reductions combine in range order (E02).**
  Throwing routes check task cancellation before work, around each managed work unit and before success,
  and honor it before and after user callbacks, including an ordered combine. Default and CPU partition
  reductions use `initial` exactly once and combine the partition aggregates in range order; empty input
  invokes neither the range work nor the combine. Default providers use `maxConcurrency` instead of
  `deviceInfo.maxThreads`. A custom reduction must be associative with an identity `initial`
  (commutativity is not required), and floating-point bits that do not depend on the partition are not
  promised. Nonthrowing nearest, pairwise, statistics and triangular-matrix operations keep returning
  complete results when cancelled.

- **Breaking: `BatchOperations.updateConfiguration(_:)` is `async throws` and validates the whole update
  (E03, E06).** It changed from `(_ update: (inout Configuration) -> Void) async` to
  `(_ update: @Sendable (inout Configuration) throws -> Void) async throws`. The closure runs first and its
  error propagates unchanged; the complete candidate is then validated in field declaration order and
  published once. A validation error (`invalidConfiguration`) names the first invalid field, and any
  rejected update keeps the previous configuration. The four `Int` settings must be positive;
  `oversubscription` must be finite and positive, with a product with the processor count no greater than
  `Double(Int.max).nextDown`. Each of the six public async `BatchOperations` compute operations reads one
  configuration snapshot: `process` with a nil batch size uses the live `defaultBatchSize`, and `process`,
  `map`, `filter`, `statistics` and `findNearest` use the live `parallelThreshold`; `sample` stays
  independent of the configuration. The new `ThreadSafeConfiguration.modify(_:)` runs a `@Sendable`
  throwing body and publishes only after it succeeds; it rolls back the binding publication, not side
  effects through aliased references. Migration: call with `try await`, capture immutable `Sendable`
  snapshots in the closure, and handle validation errors; tests that change the global configuration
  need process isolation.

- **Breaking: `TypedBufferPool` has a byte cap and returns lease handles, and the actor pools own their
  allocations (E04).** `TypedBufferPool.init(maxBuffersPerSize:)` is removed: use
  `TypedBufferPool(maxCachedBytes:maxBuffersPerSize:)` (`maxCachedBytes` is required; `maxBuffersPerSize`
  still defaults to 10). `acquire(count:)` returns an immutable `TypedBufferPool<Element>.Handle`
  (`buffer`, `pointer`, `count`) instead of an `UnsafeMutableBufferPointer<Element>`, and `release(_:)`
  takes that handle; `withBuffer(count:_:)`, `cachedBytes`, `activeBytes` and `clear()` are new.
  `BufferProvider` gains `withBuffer(size:_:)`, an `async throws` helper with a `Sendable` body and result
  that awaits the release on both exits, and `SwiftBufferPool` gains `cachedBytes`. An invalid
  `SwiftBufferPool` size throws the new `BufferError.invalidRequest(size:)`, so an exhaustive `switch`
  over `BufferError` must handle that case. `TypedBufferPool` and `SwiftBufferPool` own both checked-out
  and cached allocations: each acquisition is a fresh lease; a return validates identity, count and
  address; cache admission counts physical payload bytes and never exceeds the byte or per-size cap;
  clearing releases cached storage while active leases stay valid. Generic storage is uninitialized
  when acquired: initialize and deinitialize exactly the elements you use before returning it. Handles
  are borrowed metadata: keep the pool alive until the release, do not use a pointer after it, and
  synchronize pointee access yourself. A scoped body keeps its lease until it finishes, even if it
  ignores cancellation. `MemoryPool.BufferHandle` now retains an allocation owner and a weak pool, and
  its `count` is stored instead of computed, so binaries must be rebuilt; MemoryPool handles may outlive
  their pool, an invalid MemoryPool acquisition returns `nil`, and a nonpositive, nonfinite or
  beyond-horizon timer interval disables automatic cleanup. `BufferConfiguration.init` now stops with a
  named `fatalError`, in every build, for a negative `maxPoolSize` or `maxBufferSize` or an `alignment`
  that is not a power of two of at least 16; 0.3.3 accepted them, so a stored configuration with
  `alignment: 8` aborts at construction (`MemoryPoolBoundaryTests.guardChildProbe`).

- **Breaking (source-compatible): AutoTuning calibration is reentrant and accepts throwing calibrators
  (E05).**
  `AutoTuning.calibrateIfNeeded(dim:kind:providerCores:calibrator:)` accepts a throwing calibrator and
  `rethrows`; existing nonthrowing callers compile unchanged but must be rebuilt. A concurrent or
  recursive calibration of the same key returns an unpublished provisional default while the active
  callback runs, so simultaneous first calls can return different values. A thrown error is rethrown
  unchanged, and a successful manual publication wins. Invalid dimensions or core counts bypass the
  callback and the cache lookup; an invalid probe or an unsafe intermediate publishes the default unless
  a manual value wins. The 20 ms admission cap for completed callbacks is soft, and a callback that
  never returns has no deadline.

- **Breaking (binary layout): `NormalizationHint` clears its hints before mutable access and when
  decoded (I01).**
  `NormalizationHint.vector` is now a computed property (`get`, `set` and `_modify` over stored backing)
  that clears the hints before any mutable access: the setter, `_modify` and mutable buffer forwarding,
  and conservatively a mutable borrow that writes nothing. The layout change requires a rebuild.
  Decoding a `NormalizationHint` decodes only the vector and clears the advisory hints (missing or
  mistyped advisory values are ignored), so a decoded value can differ from the encoded one in equality
  and hashing; reconstruct trusted hints explicitly after decoding. A custom conformer with external
  reference aliases needs stable caller snapshots.

- **Breaking: `SoAFP16Cache512` is a caller-owned cache with a byte budget (I02).**
  `SoAFP16Cache512.shared`, `init(maxSize:)` and the public `store(_:for:)` are removed. Own a cache
  created with `SoAFP16Cache512(maxSize:maxBytes:)` (`maxSize` defaults to 100; `maxBytes` is required),
  replace `store` with `getOrCreate(for:)`, and read the used bytes and the byte limit from
  `retentionStatistics()`. A key compares the full Float bit patterns of every ordered candidate
  coordinate (NaN payloads and signed zeros included). Retention is bounded by `maxSize` and `maxBytes`:
  zero in either disables retention while conversion continues, a negative limit stops at initialization
  with a named diagnostic, an entry larger than the budget is not inserted, and `clear()` resets the
  retained bytes and the hit and miss counts. Returned SoA storage stays valid after eviction. The byte
  count covers the allocated snapshot and padded SoA payload capacities, not metadata, transient storage
  or resident memory.

- **`TriangularMatrix` storage copies are isolated, and partial writes persist (O01, O02).** In the
  internal tiered storage behind `TriangularMatrix`, a copy of compact storage no longer shares its
  owner (uniqueness now checks the real owner), and a compact allocation destroys exactly its initialized
  elements. Standard and inline scalar and bulk mutations keep partial writes on normal and throwing
  exits (there is no rollback) and detach shared heap storage once. `TriangularMatrix.fromFullMatrix(_:)`
  of a 1 x 1 matrix now succeeds instead of trapping. No public signature changes.

- **Redundant storage detaches and unused result-array initializations are removed (O03).** Mutation of
  the hybrid and managed vector storage, and several `DynamicVector` paths, no longer make redundant
  copy-on-write detaches or allocate unused initial result arrays. No public API changes: value
  semantics, storage mode selection, partial writes on throwing exits and provider witness dispatch are
  preserved. The reduction is in requested allocation events and bytes, measured for one linked Release
  build; latency, live memory, resident size and throughput were not measured.

- **Breaking (source-compatible): aligned-memory helpers validate their inputs and leave initialization
  to the caller (O04).**
  `alignedCopy` always updates initialized destinations and releases the values it replaces; source and
  destination must be valid, naturally aligned, initialized and nonoverlapping, so a caller with a raw
  or uninitialized destination must call `initialize(from:count:)` itself. A negative count or a
  nonpositive alignment hint is rejected even at count zero, and the three alignment predicates return
  `false` for a nonpositive boundary. `AlignedMemory.allocateAligned<T>` binds new memory and returns
  uninitialized storage: an invalid count throws `invalidDimension` without allocating, the alignment must
  be a power of two of at least 16 that also satisfies `T`'s alignment, and a logical count of zero
  allocates one uninitialized element; release it through `AlignedMemory.deallocate` or `free`. The four
  `AlignedDynamicArrayStorage` initializers are now `convenience` initializers over one validated
  initializer with a checked fallback allocator; their signatures are unchanged, but clients that inline
  them must be rebuilt. Pointer normalization uses unaligned loads, and a negative dimension stops.

- **Breaking: the FP16 vector types' `fp16Storage` getters are replaced by a scoped accessor (O05).** The
  escaping `fp16Storage` getters (`UnsafePointer<UInt16>?`) of `MixedPrecisionKernels.Vector512FP16`,
  `Vector768FP16` and `Vector1536FP16` are removed. Use the synchronous `withUnsafeFP16BufferPointer(_:)`:
  the buffer and its base pointer must not escape the callback (no returns, captures, tasks or stored
  pointers), which the generic return type does not enforce. Copy into an owned `Array` inside the
  callback for retained or asynchronous use.

- **Normalization follows one range policy (N01).** Unchecked typed normalization stops invalid nonempty
  input with named fatal guards in Debug and normal Release builds (a pointer call with dimension zero
  still accesses nothing; a negative dimension stops). Checked normalization of empty, nonfinite or zero
  input returns `invalidOperation` and leaves the input unchanged, and finite extreme values can now be
  normalized even when the scalar norm does not fit in a Float. The permissive array, fast, tolerance and
  batch paths keep the bits of invalid input, which may stay unnormalized. A custom tolerance must be
  finite and nonnegative and is validated first; a large tolerance can return non-unit input.
  `safeNormalized` handles the original nonfinite values before normalizing (replaced infinities can
  normalize; NaN-only `replaceAll` gives zero). Automatic normalized hints are set only when the returned
  Float coordinates meet the postcondition, so a lossy writeback can leave them unset. Scalar and vector
  division uses native division, and a zero vector divisor is still rejected.

- **Breaking: approximate equality, precision validation and `safeLog` check their domains (N02).**
  Approximate equality and precision validation require a finite, nonnegative tolerance and finite
  values: matching infinities now compare `false`, and a NaN or infinite tolerance no longer passes.
  Scalar and batch distance-accuracy checks share one widened hybrid relative-error computation, and an
  invalid batch returns `false`, infinity and infinity. `safeLog` applies the original-input nonfinite
  handling before the nonpositive-domain policy: in strict mode NaN and +infinity fail with
  `NonFiniteError` before any domain error, and the default NaN replacement gives zero.

- **Batch magnitude statistics accumulate in Double and return NaN for a nonfinite magnitude (N03).**
  `BatchOperations.statistics` and its internal synchronous route share one accumulator that widens each
  delivered magnitude to Double and uses a centered population update and merge, so identical finite
  magnitudes give a deviation of exactly zero. Any nonfinite delivered magnitude, including the
  overflowing norm of a row whose coordinates are finite, yields the full count with NaN mean and
  deviation on every statistics route. Partial results merge in their original chunk order, so a fixed
  partition gives a fixed result; equal bits across partitions are not promised. The statistics describe
  the Float magnitudes that `magnitude` delivers and do not repair that computation.

- **Vector entropy is scale-invariant truncated entropy (N04).** Coordinates are widened to Double, their
  magnitudes are summed with compensation, and only the normalized-probability cutoff is applied, so
  uniform tiny or maximum-finite vectors return `ln D` instead of 0 or NaN. NaN or infinity at any index
  propagates on every route, and all-zero and spike inputs return exactly `+0` (the internal large-width
  route returned `-0` for a spike). This is truncated entropy, not exact Shannon entropy for tiny
  probability mass; a uniform vector with `D >= 2^23` returns `+0` by contract. The large-width route
  moved from the SIMD-provider array chain to a scalar two-pass Double loop; its speed was not measured.

- **Breaking: INT8 quantization calibrates around zero and validates its inputs (Q01).** Asymmetric
  calibration now includes zero, so a narrow one-sided interval trades resolution for a representable
  zero point (for example, [-100, -10] uses scale 100/255 instead of 90/255). The scale is the widened
  width over 255, so an interval [a, b] with 0 < a loses a factor of b / (b − a) in resolution, and
  every value decodes to one code once b − a is below b / 255: [100, 101] has a step of 101/255 and
  [1000, 1001] quantizes to a single value. Keep such data symmetric or offset it before quantizing.
  Newly calibrated
  parameters decode all 256 codes to finite values. Nonfinite input to the throwing array conversions
  throws `.invalidData` naming the exact invalid indices; the nonthrowing conversion, calibration and
  mutation paths stop with named guards. Invalid decoded parameter tuples are rejected, while valid
  legacy five-key tuples keep their full `Int32` zero points (their Float projections can overflow). A
  quantized mutable borrow keeps the raw bytes and parameters when its body throws; a normal return
  validates and then recalibrates, including a no-op. INT8 metrics evaluate the affine coordinates before
  the Float projection, and an empty calibration gives a canonical zero. Re-baseline stored INT8 codes,
  scales and INT8 distances.

- **Breaking: the kernel, mixed-precision and quantized auto-tuner APIs are reshaped (Q02).**
  `WorkloadCharacteristics` loses `memoryBudget`, `latencyTarget` and its initializer; use
  `init(dimension:batchSize:precisionMode:maximumEstimatedPayloadBytes:maximumProfiledMeanLatency:)`.
  `PerformanceProfile` loses `accuracy` and `memoryFootprint` and gains `estimatedPayloadBytes`,
  `syntheticAccuracyScore`, `measuredSamples`, `warmupIterations` and `provenance` (new meanings, not
  renames), and its `workload` is a `KernelProfileWorkload` (`dimension`, `batchSize`, `precisionMode`)
  instead of a `String`. `KernelAutoTuner.selectStrategy(for:)` and `setOverride(_:for:)` now throw;
  `profileWorkload(_:iterations:warmupIterations:)` is `async throws`, and unsupported profiling throws
  `invalidConfiguration` instead of returning an empty array; `selectMixedPrecisionStrategy(for:)`
  (`async`) is replaced by `selectMixedPrecisionStrategy(for:maximumSyntheticRelativeError:)` (`async
  throws`). Kernel profile persistence is opt-in (`KernelAutoTuner.init(persistenceURL:)`,
  `persistenceStatus`,
  `clearCache(removePersistedProfiles:)`) with typed workload and provenance records, and unsupported or
  infeasible selections report errors. `MixedPrecisionAutoTuner` selection is lookup-only:
  `selectOptimalStrategy(candidateCount:accuracyRequirement:dimension:)` and
  `setOverride(_:candidateCount:accuracyRequirement:dimension:)` are replaced by
  `selectStrategy(candidateCount:maximumSyntheticRelativeError:dimension:)` and
  `setOverride(_:candidateCount:maximumSyntheticRelativeError:dimension:)` (both now throw); call
  `calibrate(candidateCount:dimension:iterations:warmupIterations:)` (`async throws`) explicitly before
  expecting a measured selection, and `MixedPrecisionAutoTuner()` creates a separate tuner.
  `QuantizedAutoTuner.selectOptimalQuantization(for:dimension:candidateCount:accuracyRequirement:)` is
  replaced by `suggestQuantization(for:dimension:candidateCount:relativeErrorHint:)`, and
  `QuantizedAutoTuner()` is now constructible. The hint's meaning flipped from a required accuracy to
  a tolerated relative error while its default stayed `0.02`, and a reduced strategy needs a hint of at
  least `0.05` (`candidatesINT8`, more than 1,000 candidates) or `0.1` (`bothINT8`, more than 5,000), so
  the default call now always returns `fullFP32` where 0.3.3 could return `candidatesINT8`; pass the
  error you tolerate. `selectMixedPrecisionStrategy(for:maximumSyntheticRelativeError:)` applies its one
  default, `0.001`, to every precision mode, where 0.3.3 used a per-mode limit, so `.int8Acceptable`
  workloads are held to a ten times tighter bound unless the limit is passed. Adaptive FP16 acceptance
  checks both operands and the actual
  widened error against one defined FP32 result, and full mode keeps the `fullFP32` policy. Synthetic
  profiles are fixture measurements, not data-dependent accuracy or production performance guarantees.

- **FP16 conversion diagnostics are synchronized and classify rounded outputs (Q03).** Preflight is a
  hint, and a prepared batch may join a new reset generation, so not every physical conversion made while
  diagnostics are toggled is promised to be counted. Counter saturation and the signed-zero bounds for
  finite inputs follow the documented policy. Conversion arithmetic is unchanged.

- **The FP16 candidate-range wrappers validate the range before any work (Q04).** The nine public
  candidate-range wrappers check the lower and upper bounds before subtracting, then the output capacity;
  an invalid range or a short output stops with distinct named fatal diagnostics in Debug and Release
  builds, a valid empty range returns immediately, and the work covers only the requested candidate
  range. Initialization, capacity and aliasing of unsafe outputs remain caller obligations.

- **Breaking: `PCAModel.mean` is `[Double]`, and QR, SVD, eigen and PCA reject nonfinite data with typed
  errors (L01, L02).** `PCAModel.mean` changed from `[Float]` to `[Double]`, a compensated Double center;
  components, centered panels, projections and variance statistics stay `Float`. Reconstruct in Double,
  or round `mean` explicitly where a Float is needed. QR and SVD reject nonfinite input; eigen validates
  only the referenced lower triangle; both linear-algebra providers reject nonfinite or unrepresentable
  Float results with typed errors. Factorization arithmetic is widened to Double working storage, which
  needs more temporary memory. PCA reports nonfinite input as `invalidData`; internal invalid shapes and
  representability failures as `invalidDimension`; valid-shape wrong flat counts and ragged rows as
  `dimensionMismatch`; unrepresentable residuals or nonfinite published statistics as stage-specific
  `operationFailed`; a public `n < 2` stays `invalidOperation`. Every nonnegative oversampling value,
  `Int.max` included, is clamped to the remaining sketch width, and provider errors propagate unchanged.
  Finite input can still overflow intermediate Float panels. VectorCoreC: the LAPACK shim (`vc_lapack.h`)
  rejects invalid scalar, query and local-size domains before unsafe work, under documented caller
  pointer obligations. It adds `VC_LAPACK_INVALID_QUERY` (-1001) and `VC_LAPACK_WORKSPACE_OVERFLOW`
  (-1002), documents that `VC_LAPACK_UNAVAILABLE` (-1000) is also returned when a local workspace
  allocation fails, as in 0.3.3, reports
  scalar validation failures as negative `info` argument positions, and documents `vc_lapack_sorgqr`'s
  output as the m x n orthonormal Q with `0 <= k <= n <= m`; the function declarations are unchanged.

- **UMAP curve fitting and smooth-kNN calibration changed, and manifold entry points validate their
  configuration (L03).** Curve fitting is dimensionless and throws instead of publishing its initial
  guess; smooth-kNN calibration is bracketed; public entry points check finite floating configuration,
  output counts and bytes, and the scheduling domain before allocating; SGD rejects nonfinite
  intermediates. Public signatures are unchanged, but the curve fit and the smooth-kNN memberships are
  computed differently, so re-baseline UMAP outputs, and expect nonfinite configuration to be rejected
  with an error. Fewer than two points now throws `.invalidDimension` from `KNNGraph.bruteForce` and
  `Operations.umap` (0.3.3 threw `.invalidOperation`); `PCAModel.fit` keeps `.invalidOperation`. A
  runtime failure may leave an internal optimizer buffer partly updated.

- **Empty `DynamicVector` binary frames decode (B01).** A valid empty `DynamicVector` binary frame now
  decodes; fixed positive dimensions still reject an empty frame after the CRC check, and version-1 bytes
  and the validation order are unchanged. The built-in binary methods do not add `BinaryCodable`
  conformance; use explicit witnesses that you own. The decode limit stays 10000, and nonthrowing
  encoding can exceed it.

- **`Logger` evaluates admitted messages once, and `PerformanceTimer` uses a monotonic clock (D03).**
  `Logger` snapshots its minimum level under its lock and evaluates a message only when it is admitted,
  exactly once and outside the lock, across every wrapper and the `VectorError` overload, so do not rely
  on side effects inside gated message closures. `PerformanceTimer` measures cumulative elapsed time from
  one `ContinuousClock` instant and suppresses a negative, NaN or infinite threshold without formatting.
  Public signatures are unchanged; no logging cost or latency figure is claimed.

- **Breaking (benchmark support products): `RunFlags`, `CorrectnessOut` and the report initializers
  (V01, V02).** `RunFlags.useUnderscored`, `RunFlags.releaseSize` and the initializer that took them are
  removed; use `RunFlags.init(preferSoA:useMixedPrecision:abCompare:abOnly:useCKernels:requestedCKernels:)`
  and read `requestedCKernels`. `CorrectnessOut`'s numeric memberwise initializer is removed (decode it
  with `init(from:)`), and its `maxAbsError`, `maxRelError`, `meanAbsError` and `meanRelError` are
  `Double?`: an invalid comparison omits them, and the JSON holds no NaN or infinity. `CorrectnessStats`
  distinguishes `.valid` from `.invalid` with a reason (a length mismatch, an empty comparison and
  nonfinite input are invalid); `BenchResult.executionFailure` keeps provider error descriptions, and
  `BenchRun.failures` names failed cases, which are excluded from the timed `results`.
  `ABComparisons.buildComparisons(cases:)` now throws and wraps the new `ABComparisons.report(cases:)`.
  The memberwise initializers of `BenchResult`, `BenchCase`, `BenchMetadata`, `BenchRun` and
  `ABComparison`, and `EnvCapture.collectMetadata`, gained parameters, and their old forms are removed
  (see also the schema 3 entry). `vectorcore-bench`: invalid or unsupported options and empty
  selections fail explicitly; the effective compiled C settings must match an explicit request; missing
  validation and empty result sets fail strict mode; nonfinite, negative or malformed thresholds are
  rejected; output, encoding and execution errors fail regardless of strict mode. Reports carry fixture,
  boundary and configuration identity, and legacy or incompatible metadata needs an explicit override,
  which cannot waive report integrity.

- **Breaking: `StaticDimension` requires only `value`.** Source that names `D.Storage` or a
  `DimN.Storage` alias no longer compiles; other source compiles unchanged. The `Storage`
  associated-type requirement is removed, and every `Vector<D>` member that is not specific to `Dim2`,
  `Dim3` or `Dim4` is now available for every static dimension. The convenience initializers, the NaN and infinity handling
  members, `VectorCoreOperations` (and `Vector`'s conformance to it), the `VectorMath` members,
  normalization and entropy were available only when `D.Storage` conformed to
  `VectorStorageOperations`. Migration: a custom dimension declares only `static let value`; name a
  vector's storage as `Vector<D>.Storage` (always `DimensionStorage<D, Float>`); delete custom
  `typealias Storage` declarations, which no longer constrain anything. `DynamicDimension.Storage` stays.
  The formerly constrained public members keep their source spelling but get new mangled names, so a
  binary built against an earlier VectorCore must be rebuilt. Covered by
  `builtInWidthEquivalence`, `nonBuiltInWidths` and the external consumer
  `Scripts/ci/tests/D02/Reachability.swift`. A dimension's `value` must be `>= 0`; the results at
  widths 0 and 1 are listed member by member in `D02DegenerateWidthTests`.
- **`minElement()` and `maxElement()` state their width-0 precondition.** At width 0, reachable only
  through a custom dimension whose `value` is 0, both stop with "minElement requires a nonzero
  dimension" (or "maxElement …") instead of an index-out-of-range trap (`trappingOutcomes`).
- **Breaking: `isNormalized`, and four norms in code generic over the dimension, bind to `Vector`'s
  own members.** With the constraint gone, `Vector.isNormalized` (true when
  `abs(magnitudeSquared - 1) < 1e-6`, computed on every read) is `Vector`'s `IndexableVector` witness
  for every dimension, built-in ones included. Code generic over `IndexableVector` or over the
  dimension (`func f<D: StaticDimension>(_ v: Vector<D>)`) read the protocol default `false` before
  and now reads the check. In code generic over the dimension, `l1Norm`, `lInfinityNorm`,
  `manhattanDistance(to:)` and `chebyshevDistance(to:)` also bind to `Vector`'s members instead of the
  `VectorProtocol` extension: their results equal the concrete calls' bit for bit, and they read
  `Operations.simdProvider` (`l1Norm` and `manhattanDistance(to:)` above 32 elements), so an installed
  provider now applies to them (`genericOverDimensionEqualsConcrete`). `DynamicVector`, the
  `Vector*Optimized` types and `NormalizationHint` are unchanged (`indexableIsNormalizedWitness`).
- **`entropyFast` is deprecated, renamed `entropy`.** Both return the same value from the same
  computation; `entropy` now carries the contract documentation.
- **VectorCoreC: `vc_dot_int8` has a defined result for every input.** It returns the exact integer
  dot product reduced modulo 2^32 and read in the Int32 range, on the scalar, arm64 SDOT, arm64 NEON
  and x86_64 AVX2 paths. Before, the scalar loop and the final tail additions accumulated in signed
  `int32_t`, which overflowed (undefined behavior) when a partial sum left the Int32 range; block
  loops now test the remaining count, so the index cannot wrap. The Swift package build does not call
  it (`VC_USE_C_KERNELS` is not defined). Covered by `dispatcherFollowsTheModuloLaw` and
  `Scripts/ci/tests/D02CContract.py`; the x86_64 path was cross-compiled, not executed.
- **`PrecisionProfile.summary` reasoning text.** The `.fp16`, `.int8` and `.mixed` reasoning lines no
  longer state memory-saving percentages or a balance label; they give the scalar payload size instead
  (FP16 2 bytes, INT8 1 byte, FP32 4 bytes per scalar), which is not a process-memory figure. The FP16
  error line is unchanged and, like the rest of the analysis, not verified.

- **Breaking: `Operations.findNearestBatch` no longer switches to a GEMM route
  by itself.** Before, without a `BatchKernelProvider`, at least 8 queries and
  256 candidates with Euclidean or cosine distance on a 512, 768 or 1536
  optimized type took a GEMM route, a different arithmetic that
  `enableMatrixRouting` did not affect: on the offset-1000 rows, on one test
  host (Apple silicon, Accelerate), it returned a candidate at distance 0.0
  whose true distance is 0.125, recorded as the failure of
  `P02DefaultRouteRedTests.offset1000AnalyticOracle` before this change (the
  test now pins the new default). Now the default, `cpuStrategy: .perQuery`, runs `findNearest`'s
  routes for every query at every batch size, so each row has the indices and
  score bits that `findNearest` returns for its query
  (`P02DefaultRouteRedTests.defaultRouteEqualsSingleQuery`). For the inputs
  that took the GEMM route this changes scores and rankings, and the work and
  temporary storage: no queries × candidates score matrix is built, and each
  query runs its own single-query search. With the default
  `CPUComputeProvider.automatic` those searches run on one core unless the batch
  has at least 50,000 queries (the provider's `parallelizationThreshold`); the
  `.parallel` and `.performance` presets still submit a single worker for up to
  256 queries (the 256-item chunk minimum of `parallelExecute`), and the
  per-query scan inside each worker runs inline as a nested region. On one test
  host (P02 phase G
  allocation probe, not a test) the old route requested a buffer of Q × N
  Floats on every call at Q 8 with N 256 and at Q 64 with N 4096, and the new
  default requested none that large. The new default is slower in most measured
  cases: on that host (two one-host timing records, medians of 9 samples, not
  tests: the P02 phase G record of 2026-10-07, on a quiet machine, and a
  high-power re-run of 2026-10-16 with two repetitions, both outside this
  repository) every measured case at Q 64 with N 4096 took longer, for example
  about 57 ms instead of about 2.4 to 2.5 ms per call for Euclidean distance at
  1536, about 23 times as long (22.6-23.9× across the records), and about 21 ms
  instead of about 1.3 to 1.5 ms at 512, about 14 to 16 times as long
  (13.7-15.9×); Euclidean distance at Q 8 with N 256 took about 2.8 to 4.4 times
  as long (2.75-4.43×); cosine distance at Q 8 with N 256, and every measured
  case at Q 7 with N 256, took less time in every record (cosine at 1536 with
  Q 7 by 0.3% in one high-power repetition, effectively equal). To score a
  large batch with the GEMM route, opt in
  with `cpuStrategy: .gram(...)`. The default batch search can now be
  cancelled through the provider's and `findNearest`'s checks
  (`P02PerQueryRouteTests.preCancelledDefaultThrowsBeforeAnyMetricCall`); the
  GEMM route had no cancellation check. The GEMM route remains available as
  `cpuStrategy: .gram(...)`. A reference to the function by its full name
  `findNearestBatch(queries:in:k:metric:)` must add `cpuStrategy:`, and the
  error context captured in Debug builds names
  `findNearestBatch(queries:in:k:metric:cpuStrategy:)`.


- **Breaking: `BatchOperations.pairwiseDistances` no longer takes the GEMM route
  by default.** `BatchOperations.Configuration.enableMatrixRouting` now defaults
  to `false` (it was `true`): the default keeps every pairwise matrix on the
  per-pair kernels, whose off-diagonal entries equal the metric's own score bit
  for bit. The GEMM route is the arithmetic the `findNearestBatch` entry above
  describes, and it loses the 0.125 offset of the offset-1000 rows at N = 256;
  the default is pinned by
  `P02PairwiseRowTests.defaultConfigurationUsesPerPairKernelsAtAnEligibleN`.
  Set `enableMatrixRouting = true` through `updateConfiguration` to opt in. When
  routed, a Euclidean entry whose squared score is NaN now stays NaN (it was
  formatted to 0, so a row with one NaN coordinate became the nearest neighbour
  of every other row), as on the per-pair route, `.gram` and `.perQuery`
  (`P02PairwiseRowTests.nanRowsAtAnEligibleN`).
- **Breaking (binary layout): the optimized vector types no longer store
  `scalarCount`.** `Vector384Optimized`, `Vector512Optimized`,
  `Vector768Optimized` and `Vector1536Optimized` now compute `scalarCount`
  (the type's dimension) instead of storing an `Int` beside their storage, so
  each value holds only its storage reference: its `MemoryLayout` stride equals
  that of `ContiguousArray<SIMD4<Float>>`
  (`P02HandleLayoutTests.handleStrideIsTheStorageStride`): 8 bytes instead of
  16 on the 64-bit test host, where a fresh array of 1024 handles requested
  8,224 bytes instead of 16,416 (P02 phase G probe). Source that reads
  `scalarCount` compiles unchanged, but code built against an earlier build must
  be rebuilt (the package is not built with library evolution). `Mirror` and
  `dump` output no longer show a `scalarCount` child
  (`P02HandleLayoutTests.scalarCountIsTheDimensionAndNotAStoredChild`).
  Equality and hashing never read `scalarCount` (they use the storage and a
  literal dimension) and the JSON encoding writes the coordinates; the binary
  encoding does read `scalarCount` (its reserved size and header dimension),
  and its bytes are unchanged because the computed value equals the former
  stored value. All four keep their results (`OptimizedHashContractTests`;
  `P02HandleLayoutTests.jsonGoldenBytesAtEveryWidth` and
  `binaryGoldenFramesAtEveryWidth`).

- **`BatchOperations.pairwiseDistances` builds its matrix-route rows one query
  row at a time.** With `enableMatrixRouting`, at least `matrixRoutingMinN`
  vectors and Euclidean or cosine distance on a 512, 768 or 1536 optimized
  type, it now packs the vectors once and computes each row with
  `MatrixDistance`'s prepared `into:` entry directly into the returned row,
  formatting Euclidean distances in place; it no longer builds a separate
  N × N score buffer beside the result. Each row also builds its own one-query
  input (a D-float pack, and a one-Float norm for Euclidean), so the number of
  allocations grows with N; on one test host (Apple silicon, Release; P02
  phase G allocation probe, not a test) at N 256, 512 and 1024 and D 512, 768
  and 1536 the requests per call rose from about N to about 2N (cosine) or 4N
  (Euclidean), while the total requested bytes per call fell in all 18 cases,
  including D > N (for example 3.4 MB instead of 6.9 MB at N 256, D 1536).
  The process's resident high-water mark after one call fell in the six
  cases measured at N 1024 (for example 25.0 MB instead of 33.0 MB at D 512;
  process-wide, not the call's own peak). Each row is its own one-query
  `cblas_sgemm` instead of one N × N multiply, so off-diagonal values can
  differ in their last bits (`P02PairwiseRowTests.rowsEqualPreparedIntoPerRow`
  checks every row against the prepared entry). The diagonal (the metric's own
  self-distance), the NaN-to-zero formatting of Euclidean rows and the
  complete result under cancellation are kept
  (`P02PairwiseRowTests.nanRowsAtAnEligibleN`,
  `completeResultUnderCancellation`). The per-row calls cost time: in the P02
  phase G timing record of 2026-10-07 (a quiet host, median of 9 samples) and
  in a high-power re-run of 2026-10-16 (two repetitions, median of 9 samples;
  both outside this repository) every one of the 18 cases took longer per call,
  about 1.6 to 8.7 times as long (1.57-8.69× across the records), for example
  about 20.3 to 20.8 ms instead of about 2.8 ms at N 1024, D 512, Euclidean;
  these are one-host records, not a speed target.

- **Optimized routes test an array's element type without a conditional
  cast.** The 39 expressions such as `vectors as? [Vector512Optimized]` in
  `Operations`, `BatchOperations` and `MixedPrecisionKernels` now use an
  internal type-identity helper that returns the caller's array value
  (`P02ArrayIdentityTests`) instead of converting it element by element. For
  every vector type VectorCore defines, the same branch is taken as before, so
  results are unchanged (the route observations `P01BatchRouteTests` and
  `P01OperationsRouteTests` pass unchanged). A client type whose arrays cast
  element by element to an optimized array (for example a retroactive
  `VectorProtocol` conformance of `Optional<Vector512Optimized>`) is now
  scored by the per-candidate route (`metric.distance` for each candidate,
  into a full array of N scores); before, it took the Top-K heap route
  (`TopKSelectionKernels`) when k < N and a `BatchKernels` helper when
  k >= N. No test covers such a type. On one test
  host such a cast requested a copy of
  the array's handles in unspecialized code (P02 phase 1 probe). With the new
  form, the same host recorded one fewer request of 16 * N + 32 bytes per call
  on every measured route, no new request of 8 * N + 32 bytes, and the same
  result bits (54 cases, N 1024 to 16384: `Operations.findNearest` with k 3
  for Euclidean and cosine distance at three widths and dot product at 512,
  and k = N for cosine at 768 and 1536; `BatchOperations.findNearest` serial,
  parallel and minimum-chunk, Euclidean at 512, k 3 and k = N;
  `SoAFP16.init(vectors:)` at three widths; P02 phase G allocation probe, not
  a test). The pairwise route lost two to six such copies per call.
  `Scripts/ci/tests/test_no_array_conditional_casts.py` keeps these casts out of
  `Sources/VectorCore`.

- **Breaking (benchmark support products): benchmark report schema 3 and
  `JSONWriter.toCases` now throws.** `VectorCoreBenchmarking` reports gain a
  per-case `MeasurementBoundary` (timing scope, operation and work unit, input and
  output representation, inclusion flags, consumer, fixture identity), raw
  per-sample values, per-operation statistics and `unavailable` records;
  `BenchMetadata.schemaVersion` is 3 and `measurementBoundaryVersion` is
  `V04-boundaries-1`. The new `BenchMetadata`, `BenchResult`, `BenchCase`,
  `ABComparison` and `BenchRun` fields mean schema-2 reports no longer decode
  with the synthesized `Codable` conformances, and `JSONWriter.toCases(_:)`
  throws `BenchmarkReportError` for a result without a valid boundary and for
  duplicate case IDs (it was non-throwing). A/B pairs form only from declared `boundary.comparison`
  membership. Migration: attach a boundary to every `BenchResult`, call
  `try JSONWriter.toCases(...)`, and compare reports with
  `Scripts/compare_benchmarks.swift`, which treats a missing boundary in a
  pre-schema-3 report as waivable only with `--allow-incompatible` and a
  present-but-different boundary as never waivable.

- **Breaking: `VectorCoreBenchmarking.ProgressReporter` is now
  `BenchmarkProgressReporter`** (file renamed accordingly). The macOS 27 SDK
  adds `Foundation.ProgressReporter` (other OS 27 SDKs are unverified), which made the unqualified name
  ambiguous for any file importing both Foundation and `VectorCoreBenchmarking`
  and broke compilation under Xcode 27 / Swift 6.4. No deprecated alias is
  provided, because a declaration under the old name would still collide.
  Migration: replace `ProgressReporter` with `BenchmarkProgressReporter` where it
  refers to the VectorCoreBenchmarking type; the initializer and API are unchanged.
  `Scripts/ci/check_sdk_name_collisions.py` now reports public top-level names
  that collide with SDK modules.

- **Breaking: `SearchResult.computedScore` is replaced by `computedScore(using:)`,
  and search results now carry relevance scores.** `DistanceMetric` gains a
  defaulted `relevanceScore(for:)` requirement (default `nil`). The six
  inverse-distance builtins score finite nonnegative distances as `1 / (1 + d)`;
  `DotProductDistance` uses the logistic `1 / (1 + exp(d))`. Producer results from
  `SimpleVectorCollection` and the default `VectorCollection.search` now populate
  `score` for the selected results. Migration: use `.score`, or call
  `.computedScore(using: metric)` with the metric that produced the distance.
  Because `SearchResult` has synthesized `Equatable` over `score`,
  `producerResult == SearchResult(id:, distance:)` is now false for builtin
  metrics; `Hashable` still hashes only `id` and `distance`.

- **Breaking: `VectorMath.nearestNeighbors(query:in:k:using:)` now follows the
  shared Top-K selection policy.** Results are ordered by ascending numeric
  distance, NaN distances come after every numeric one, and exact ties
  (including `+0`/`-0` and two NaNs) keep the smaller original index, as in
  `TopKSelection.select(k:from:)`. Before, a plain `<` sort left NaN positions
  unspecified (distances `[NaN, 1, 0]` with `k = 3` returned indices
  `[0, 2, 1]`; now `[2, 1, 0]`). `k <= 0` and an empty candidate
  array now return `[]` without calling the metric; before, `k == 0` called it
  once per candidate and a negative `k` trapped after calling it once per
  candidate. For positive `k` the metric still runs exactly once per candidate
  in index order and the result still has `min(k, vectors.count)` entries.
  Selection keeps a bounded heap of `min(k, n)` entries when `min(k, n) < n / 10`
  and sorts all `n` pairs otherwise. Migration: callers that depended on NaN
  positions or on the negative-`k` trap must handle the new order and the empty
  result.

- **Breaking: generic small-`k` `Operations.findNearest` streams each candidate
  through `parallelReduce`, without AutoTuning calibration.** This route applies
  when no `BatchKernelProvider` is installed, `min(k, n) < n / 10` and no
  optimized route does: any metric on `DynamicVector`, `Vector<D>` or
  `Vector384Optimized`; any metric except Euclidean, cosine and dot product on
  `Vector512Optimized`; and any metric except Euclidean and cosine on
  `Vector768Optimized` and `Vector1536Optimized`. Before, the first search for a
  cold AutoTuning (dimension, kernel) key evaluated up to 256 candidates twice
  (calibration probes, then scoring) and published a tuning, unless AutoTuning
  was disabled (`VC_AUTOTUNE=0`), its calibration-time budget was spent or that
  key was being calibrated at the same moment; and a candidate count below the
  tuned break-even used a synchronous map that never called the installed
  `ComputeProvider` (observed on the unchanged base: 200 metric calls for 100
  candidates, no provider call). Now, on a successful return, the metric has run
  exactly once per candidate, in index order within each range the provider
  supplies to one `parallelReduce` call (the overload without a chunk hint), each
  range keeping a heap of at most `min(k, range count)` entries; this route
  calibrates and publishes no tuning. The provider's own serial, bounded and
  nested policies decide the partition, call order across ranges is
  unspecified, and a cancelled task now gets `CancellationError` from this route
  before its next metric call (the provider checks around each callback and the
  scan checks before each metric call); before, a run below the tuned
  break-even completed without checking (observed on the unchanged base with 64
  candidates). Large `k`, the 512/768/1536 optimized Euclidean routes, cosine
  and dot on `Vector512Optimized`, and cosine on `Vector768Optimized` and
  `Vector1536Optimized` (which still scores every candidate with the fused
  kernels, after any calibration) keep their routes and their cancellation
  behaviour: below the tuned break-even a cancelled large-`k` or cosine
  768/1536 search can still return a complete result.
  `findNearestBatch`'s per-query fallback now validates the candidate rows once
  per batch instead of once per query. Migration: metrics with side effects see
  `n` calls per successful search instead of up to `n + 256`; a custom
  `ComputeProvider` now receives `parallelReduce` for these searches and must
  honour its range and combine contract; do not rely on a small-`k` search to
  warm an AutoTuning key.

- **Optimized-vector hashing now covers every coordinate.** `hash(into:)` on
  `Vector384Optimized`, `Vector512Optimized`, `Vector768Optimized` and
  `Vector1536Optimized` feeds every SIMD4 storage block in order, then the
  dimension. Before, it fed only the first two and last two blocks plus the
  dimension, so vectors that differed only in the other coordinates always
  hashed equally and collided in a `Set` or `Dictionary`. Equality and storage
  are unchanged, and Float `Hashable` semantics apply, so `+0` and `-0` still
  hash equally. Each hash is now O(D) in the dimension and is not cached. Hash
  values remain process-seeded by Swift's `Hasher`: they can differ between
  processes and must not be persisted, and collisions remain legal. Covered by
  `OptimizedHashContractTests` and
  `Scripts/ci/tests/test_optimized_hash_coverage.py`. A single-host record, not
  a performance claim (Release, Swift 6.4, Mac15,9; medians of 1,024-hash
  invocations, before and after measured in one session): per hash, 44.9 → 744.3
  ns at 384 (16.6×), 45.4 → 949.8 ns at 512 (20.9×), 48.1 → 1,451.0 ns at 768
  (30.2×) and 45.2 → 2,880.5 ns at 1536 (63.7×). On the same host, `Dictionary`
  insert-plus-lookup over corpora that differ only in interior coordinates took
  longer at 64 vectors and less time at 512 and 1,024 vectors, at every width.

- **Benchmark auto-scaling calibration (`V04-calibration-2`).** Without
  `--repeats`, calibration doubles the invocation count until one calibration
  interval spans at least 10 clock ticks (cap 2^24 invocations, then an explicit
  error), and the loop that tops a sample up to `--min-time` never measures fewer
  than 10 ticks. Default cases whose first single-invocation interval is shorter
  than 10 ticks (about 420 ns at the cited Apple Silicon tick) therefore get
  different iteration counts than before. Which default cases fall below that
  interval was estimated once from a steady-state per-invocation proxy on one
  host, not from the first interval itself, and is not verified by an artifact
  in this repository. An explicit `--repeats N` whose samples span no clock tick
  now fails the case with an actionable message.
- **Benchmark consumers fold every produced output.** The `matrix` cases fold
  all N×N produced entries and `normalize.success` folds every coordinate of the
  unit vector (`u64-bitfold-sink-v1`), so their measured work changed; earlier
  `matrix` and `normalize.success` numbers are not an equivalent measurement.
- **arm64 batch FP16 encoding canonicalizes NaN.** On arm64,
  `MixedPrecisionKernels.platformOptimizedConvertBatch` now encodes every NaN in
  its SIMD4 chunks to the quiet NaN `sign | 0x7E00`, consistent with the scalar
  converter and the batch tail. Chunk lanes used to keep the high payload bits,
  so one buffer could mix NaN encodings. For example, the FP32 NaN `0x7FC12345`
  now encodes to `0x7E00` (it was `0x7E09`), and `0xFFFFE000` to `0xFE00` (it
  was `0xFFFF`). Non-NaN results are unchanged: all 2^32 FP32 inputs were
  compared before and after on one arm64 host (records outside this repository).
  Decoding still maps every FP16 NaN to `0x7FC00000`, and no other conversion
  path changed. The policy is pinned for the scalar, SIMD8, batch, vector, SoA
  and `platformOptimizedConvertBatch` conversion paths by
  [FP16ConverterExhaustiveTests](Tests/ComprehensiveTests/FP16ConverterExhaustiveTests.swift),
  which also checks all 65,536 decodes, all 63,490 non-NaN round trips and
  190,464 rounding midpoints and neighbours against integer references.

- **Euclidean and dot-product scores on the optimized types now use one
  arithmetic.** For `Vector512Optimized`, `Vector768Optimized` and
  `Vector1536Optimized`, the `EuclideanDistance` and `DotProductDistance`
  witnesses (so also `EuclideanDistance.batchDistance`, the default
  `DotProductDistance.batchDistance` and `MinkowskiDistance(p: 2)`) now use
  `EuclideanKernels`/`DotKernels` instead of `SIMDOperations.FloatProvider`
  (vDSP on Apple platforms). Other vector types keep the `FloatProvider` path.
  A direct call, generic or existential code and the search APIs (`Operations`,
  `BatchOperations`, `TopKSelection`, the default `VectorCollection.search`)
  now return the same bits for a candidate, except on the GEMM routes, which
  are a different arithmetic: Euclidean `Operations.findNearestBatch` with the
  explicit `cpuStrategy: .gram(...)` (see the Breaking entry on
  `findNearestBatch` above), and the off-diagonal entries of Euclidean
  `BatchOperations.pairwiseDistances` when `enableMatrixRouting` is set and the
  count reaches `matrixRoutingMinN`. Dot product has no GEMM route. An
  installed `BatchKernelProvider` still returns its own results from
  `Operations`. Covered outside the GEMM routes by
  `MetricWitnessEquivalenceTests` and `MetricSearchWitnessConsistencyTests`.
  The figures below come from T14's probe on one Apple-silicon host (Release
  builds before and after; records outside this repository) and are not
  pinned by a test.
  - **Unchanged:** a call with the concrete optimized types bound a removed
    overload that already used these kernels, and its Euclidean and
    dot-product bits did not change on any probe fixture (see Removed).
    `Operations.findNearest` and `BatchOperations.findNearest` already
    computed Euclidean scores this way: none of their Euclidean probe scores
    changed.
  - **Euclidean through the witness:** generic and existential calls,
    `TopKSelection`, `VectorCollection.search`, `Operations.distanceMatrix`
    and the per-pair `pairwiseDistances` route moved by up to 3 ulp in T14's
    witness-level probe (the old and new witness compared on the same probe
    pairs; these APIs score through the witness, so the figure carries over to
    them from the code, not from a probe of each API), and NaN payloads can
    differ.
  - **Dot-product search scores changed.** Only `Operations.findNearest` on
    `Vector512Optimized` with `k < n` already used `DotKernels`, and it is
    unchanged. The other dot-product search routes scored through the
    witness, so scores changed on `Vector768Optimized` and
    `Vector1536Optimized` at every `k`, and on `Vector512Optimized` at `k = n`
    and through `BatchOperations`. At `k = n`, `Operations.findNearest` changed
    33 of 48 probe scores at 512, 43 of 48 at 768 and 73 of 88 at 1536, and
    `BatchOperations.findNearest` 33 of 48 at 512, 43 of 48 at 768 and 73 of 88
    at 1536. A `Vector512Optimized` score from `Operations.findNearest` no
    longer depends on `k` (it used the kernel for `k < n` and the
    `FloatProvider` path for `k = n`).
  - **Not only the last bits:** where the products cancel, a dot product moved
    by up to 1.54 % (163,840 ulp): one `Vector512Optimized` pair went from
    -32.375 to -31.875, and the largest moves at 768 and 1536 were 0.33 % and
    0.41 %. P01's `Vector1536Optimized` fixture row moved by 2 ulp, from
    `0xc1ce604c` to `0xc1ce604e`.
  - **Overflow can change class:** a `Vector768Optimized` pair whose product
    sum overflows returned `-∞` and now returns NaN. As a dot-product distance
    `-∞` ranks first and NaN ranks last, so such a candidate moves from best to
    last in a search. Zero signs and NaN payloads can also differ.

- **Cosine scores on the optimized types now use one arithmetic, and no longer
  depend on a candidate's position or on k.** For `Vector512Optimized`,
  `Vector768Optimized` and `Vector1536Optimized`, the `CosineDistance` witness
  (so also the default `batchDistance`, generic and existential calls,
  `TopKSelection.nearest`, `VectorCollection.search`,
  `Operations.distanceMatrix` and the per-pair `pairwiseDistances` route) now
  uses `CosineKernels`' fused kernel instead of `SIMDOperations.FloatProvider`
  (vDSP on Apple platforms), and the cosine search kernels compute the same
  arithmetic: the pair loops of `BatchKernels.range_cosine_fused_*` and
  `TopKSelectionKernels.range_topk_cosine_fused_512` (the
  `Operations.findNearest` route for `Vector512Optimized` with `k < n`). A
  direct call, generic or existential code and the search APIs now return the
  same bits for a candidate, except on the GEMM routes, which are a different
  arithmetic: cosine `Operations.findNearestBatch` with the explicit
  `cpuStrategy: .gram(...)`, and the off-diagonal entries of cosine
  `BatchOperations.pairwiseDistances` when `enableMatrixRouting` is set and the
  count reaches `matrixRoutingMinN`. An installed `BatchKernelProvider` still
  returns its own results from `Operations`. Covered by
  `MetricWitnessEquivalenceTests`, `MetricSearchWitnessConsistencyTests`,
  `MetricCosinePositionTests` and `BatchKernelsRangeClosureTests`. Not covered
  by these tests: `VectorProtocol.cosineDistance(to:)` computes through
  `cosineSimilarity(to:)` and `dotProduct`, a separate arithmetic that is not
  asserted equal to the witness's on the optimized types (not verified by a
  test); D02 documents it as a separate, statically selected arithmetic and bounds it against the
  witness (`cosineDistanceStaysWithinTheDerivedBoundOfTheWitness`); it is not aligned. The figures
  below come from T14's probe on one Apple-silicon host (Release builds before
  and after; records outside this repository) and are not pinned by a test.
  - **Unchanged:** direct calls with the concrete optimized types (they bound
    the removed overloads, which used this kernel; see Removed),
    `TopKSelection.nearestCosineFused512`, the odd-tail scores of the
    `BatchKernels` range helpers, and every result on NaN, infinite and zero
    inputs (T15's policy).
  - **Through the witness:** 325 of 2,786 probe pairs changed at 512, 294 of
    2,786 at 768 and 344 of 2,866 at 1536, by at most 3.6e-7 outside the tiny
    pairs below.
  - **Search scores changed:** `Operations.findNearest` with `k < n` on 87 of
    1,200 / 97 of 1,200 / 129 of 1,241 probe rows (512 / 768 / 1536) and with
    `k = n` on 122 of 1,192 / 98 of 1,192 / 128 of 1,232, and
    `BatchOperations.findNearest` on 245 of 2,390 / 195 of 2,392 / 257 of 2,473,
    by at most 3.6e-7.
  - **Position and k:** a search score used to depend on whether the candidate
    was the odd last candidate of a scoring range, so on the number of
    candidates and on the chunking (the core count and the `BatchOperations`
    configuration), and `Operations.findNearest` on `Vector512Optimized` scored
    `k < n` and `k = n` differently (129 of 1,187 probe candidates). Both are
    gone. All eight `BatchKernels` cosine range helpers (fused and
    pre-normalized, 384 to 1536) now score a pair with the arithmetic of their
    single-candidate kernel, so their results no longer depend on a candidate's
    position in the range either.
  - **Tiny vectors:** on these three types the witness now treats a vector
    whose squared norm is at most `Float.leastNormalMagnitude` as a zero
    vector, as the search kernels already did, so a tiny vector against a
    normal one gives distance 1.0 instead of the computed value (1.049, 1.079
    and 0.977 on the probe's tiny pairs). Other vector types keep the witness's
    product guard (T15's per-family zero classes, pinned by
    `T15CosineExceptionalTableTests`).
  - **Not measured:** on an x86_64 target without FMA in its baseline (the
    `x86_64-apple-macosx14.0` triple), the fused kernel's multiply-adds compile
    to `_fmaf` library calls, 4 per SIMD4 multiply-add (assembly inspection;
    speed not measured). These paths now pay that cost: the cosine search pair
    loops (`BatchKernels.range_cosine_fused_*`), the rerouted witness on the
    three types (generic and existential `CosineDistance.distance`, the default
    `batchDistance`, `TopKSelection.nearest`, `distanceMatrix` and
    `pairwiseDistances` below matrix routing, previously vDSP) and
    `range_topk_cosine_fused_512`; direct calls with the concrete types already
    paid it. Follow-up: measure all of them, without raising the platform
    floor.

- **Breaking: cosine distance no longer turns NaN or infinite inputs into a
  perfect match.** Every cosine distance path (`CosineDistance` and its
  `Vector{512,768,1536}Optimized` overloads, the `BatchKernels` fused and
  pre-normalized range kernels, the `BatchKernels_SoA` cosine kernels, the FP16
  and mixed-precision cosine kernels, `VectorProtocol.cosineDistance(to:)` and
  `distance(to:metric: .cosine)`, and the `Operations`, `BatchOperations`,
  `TopKSelection`, `VectorMath` and `VectorCollection` searches built on them)
  now returns NaN when either vector has a NaN or infinite coordinate, including
  against a zero vector. Previously `min(1, NaN)` mapped a NaN similarity to
  distance 0, so such rows ranked nearest on the fused routes (and ±∞ rows on the
  generic routes); the `[-1, 1]` clamp now applies only to finite similarities,
  so the Top-K rule ranks these rows after every numeric distance. Apart from
  the finite changes listed in this entry, finite results were bit-identical in
  the T15 phase 2 probe comparison (1,954 finite rows against the phase 1
  baseline run; one host; records outside this repository:
  `T15/phase2/runs/phase2-compare.txt` and
  `T15/phase1/runs/attempt3-testable/run.stdout` in the stabilization records;
  not pinned by a test). `cosineSimilarity` on `VectorProtocol`, the optimized
  vectors and the array SIMD providers returns NaN for a non-finite magnitude
  (was 0), and the optimized vectors and array providers return 0 instead of NaN
  when two tiny nonzero magnitudes multiply to zero;
  `[Vector<Dim768>].cosineSimilarityMatrix()` returns NaN where it passed a
  non-finite score through. `VectorProtocol.cosineDistance(to:)` now clamps
  finite rounding overshoot to [0, 2] (it could return a small negative
  distance). Two zero vectors now have cosine distance 1 on every path (the fused
  kernels and the FP16 single-pair kernels returned 0). Each path keeps its own
  zero-vector threshold, unchanged, and the new value applies to whatever that
  path already treats as zero: on the fused kernels this includes vectors whose
  squared norms are at or below `Float.leastNormalMagnitude`. Vectors whose
  squared norm overflows Float (`‖v‖² ≥ Float.greatestFiniteMagnitude`) are
  outside the supported domain: the per-pair paths can return NaN (two such
  parallel vectors returned 0) or 1 where `MatrixDistance` returns the true
  distance, and the array SIMD providers' `cosineSimilarity`, whose Float sum of
  squares overflows there, returns NaN (was 0). The mixed-precision cosine
  kernels that pair an FP32 vector with an FP16 vector have a narrower domain:
  the product of the two squared norms must stay below
  `Float.greatestFiniteMagnitude`; past it they return 1 for finite inputs in
  any direction, as before. Migration: filter or validate non-finite embeddings
  before search if NaN distances are unwanted; code that relied on 0 or 1 for
  such rows must test `isNaN`. Pinned by
  [T15CosineExceptionalTableTests](Tests/ComprehensiveTests/T15CosineExceptionalTableTests.swift)
  (every registered implementation) and
  [T15CosineSelectionTests](Tests/ComprehensiveTests/T15CosineSelectionTests.swift)
  (every public cosine search route). `Scripts/ci/tests/test_cosine_registry.py`
  is a lexical guard that keeps the registry complete: it fails when a body it
  discovers (a name containing "cosine", a clamp idiom, a call of the shared
  helper, or a body in a Cosine-named type) is unregistered or a registry entry
  is stale. Its clamp rule is a body-level token check (an `isFinite` or `isNaN`
  token in the same body), not proof that the clamp is guarded; the table is the
  semantic check.

- **Breaking: `ChebyshevDistance` and `HammingDistance` return NaN for a NaN
  coordinate.** `ChebyshevDistance` dropped a NaN coordinate difference
  (`max(d, NaN) == d`) and `HammingDistance` counted a NaN coordinate as bit 0;
  both now return NaN when either vector has a NaN coordinate, and
  `ChebyshevDistance` also when one coordinate holds the same infinity in both
  vectors (their difference is NaN, as in Euclidean distance), so the Top-K rule
  ranks such rows last. This includes `ChebyshevDistance` calls on concrete
  `Vector512Optimized`, `Vector768Optimized` and `Vector1536Optimized` values,
  which bind the witness since its concrete overloads were removed (see
  Removed); those overloads reduced with `max`, which drops a NaN (T15 phase 1
  probe of the `Vector512Optimized` overload; records outside this repository).
  `lInfinityNorm` on
  `VectorProtocol` and on `Vector<D>` now returns NaN when any element is NaN,
  as `magnitude` and `l1Norm` already did (before, the `VectorProtocol` version
  dropped every NaN and the `Vector<D>` version kept one only at element 0), so
  `chebyshevDistance(to:)` on `VectorProtocol` and on `Vector<D>`, and
  `distance(to:metric: .chebyshev)`, return NaN in the same cases as
  `ChebyshevDistance`. Other results are unchanged: each change is a NaN check
  ahead of the unchanged reduction.
  Pinned by `T15MetricExceptionalTableTests` in
  [T15CosineExceptionalTableTests](Tests/ComprehensiveTests/T15CosineExceptionalTableTests.swift).

- **Breaking: min/max reductions, index reductions, element statistics,
  element-wise min/max and clip propagate NaN.** Swift's `max(a, b)` and
  `min(a, b)` keep a NaN only in their first argument, `Array.max()` keeps one
  only at index 0, `SIMD.max()` ignores one, and vDSP `maxv`/`minv` do not
  document NaN handling (on the test host they dropped a NaN in their scalar
  tail and for 1 to 3 elements), so these results depended on where the NaN
  was and on the provider. Now any NaN input gives NaN (its sign and payload
  are unspecified; test with `isNaN`):
  - `SwiftFloatSIMDProvider` and `SwiftDoubleSIMDProvider` `maximum`, `minimum`
    and `maximumMagnitude` (and `SIMDOperations` on non-Apple platforms, and
    the default and legacy `SwiftSIMDProvider` array `max`/`min`): NaN for any
    NaN element; before, only when element 0 was NaN. With `count == 0`,
    `maximum` returns -infinity and `minimum` +infinity (was 0), as the
    Accelerate providers do, and a zero `maximumMagnitude` is +0 (it could be
    -0).
  - `AccelerateFloatProvider` and `AccelerateDoubleProvider` `maximum` and
    `minimum` (`SIMDOperations.FloatProvider`/`DoubleProvider` on Apple
    platforms, `AccelerateArraySIMDProvider.max`/`min`): NaN for any NaN
    element; before, a NaN in the vDSP scalar tail or in 1 to 3 elements was
    dropped, and an all-NaN array of 1 to 3 elements gave -infinity or
    +infinity. `maximumMagnitude` already returned NaN on the test host; that is
    now guaranteed by an explicit check instead of vDSP behaviour.
  - `minIndex`/`maxIndex` of the array providers, and `Vector.minElement()` /
    `maxElement()`: the index of the first NaN, with value NaN; before, a NaN
    was skipped unless it was element 0.
  - `VectorProtocol.min` and `.max` (`Vector`, `DynamicVector`, the optimized
    vectors): NaN; before, never NaN, and an all-NaN vector gave +infinity /
    -infinity. Empty vectors still give nil.
  - `SIMDStorage.minMax()` on `SIMD4<Float>` storage: (NaN, NaN) for any NaN
    lane (before, never NaN). It now starts from +-infinity instead of
    +-`greatestFiniteMagnitude`, so all-+infinity data has min +infinity (was
    3.4e38), all--infinity data has max -infinity, and empty storage gives
    (+infinity, -infinity) (was (3.4e38, -3.4e38)).
  - `Operations.statistics(for:)`: a NaN component makes its column's `min`
    and `max` NaN, as its `mean` already was (before, never NaN); a NaN
    magnitude makes `magnitudes.min` and `.max` NaN (before, only when it was
    the first row's).
  - `QuantizationSchemes.analyzeQuantizationError`: `maxAbsoluteError` is NaN
    when an error is NaN, consistent with its `meanSquaredError` and
    `signalToNoiseRatio` (it was finite).
  - `PerformanceProfiler.recordAccuracyLoss` and
    `recordMemoryBandwidthUtilization`: a NaN keeps the tracked maximum NaN
    until `reset()` (it was ignored).
  - `elementWiseMin`/`elementWiseMax` of the array providers, `Vector.min(_:)`
    / `max(_:)` and `DynamicVector.min(_:)` / `max(_:)`: NaN where either
    operand is NaN (before, only the first operand's NaN was kept).
  - `clip` of `SwiftFloatSIMDProvider`, `SwiftDoubleSIMDProvider` and the
    default and legacy array providers, so also `Vector.clamped(to:)`,
    `clamp(to:)` and `DynamicVector.clamped(to:)` with the default provider: a
    NaN element stays NaN, as the documented formula
    `min(max(a[i], low), high)` and `vDSP_vclip` give (before, the SIMD chunks
    replaced it with the lower bound and the `Double` scalar tail with the
    upper bound).
  `Vector.softmax()` is unchanged (a NaN input already gave NaN in every
  lane), and so is the `lerp`/`smoothstep` parameter clamp (`t = NaN` returns
  the other endpoint). Other results are unchanged: each change selects NaN
  over the unchanged existing result, so outputs for NaN-free input keep their
  bits except for the three cases above (count 0 on the Swift providers, a -0
  magnitude, and `minMax()` on all-infinite or empty storage), which a
  before/after probe over 4,630 NaN-free arrays confirmed (one host; records
  outside this repository). The sign of a zero `maximum` or `minimum` remains
  unspecified. The Accelerate reductions now make a second pass over the data
  for the NaN check, and the Swift reductions track a NaN flag. Migration:
  filter NaNs before reducing if NaN-ignoring extrema are wanted, and test
  results with `isNaN`. Pinned by `T16ReductionExceptionalTableTests` and
  `T16ReductionProfilerTests` in
  [T16ReductionExceptionalTableTests](Tests/ComprehensiveTests/T16ReductionExceptionalTableTests.swift),
  which evaluate every implementation and provider with a NaN at every index
  of 18 lengths and check that the Swift and Accelerate providers agree.
  `Scripts/ci/tests/test_reduction_nan_guard.py` fails when a `Float`/`Double`
  `min`/`max`/vDSP fold in `Sources` is not registered with a reviewed NaN
  decision in `Tests/ComprehensiveTests/T16ReductionRegistry.swift`.

### Removed

- **Breaking: the `StaticDimension.Storage` requirement and the `Dim2` … `Dim3072` `Storage`
  aliases.** See Changed for the migration.
- **Breaking: `ComputeDevice` discovery members and `DeviceCapabilities`.** `isAvailable`,
  `isAccelerated`, `requiredAlignment`, `minimumEfficientSize`, `queryCapabilities()`,
  `availableDevices`, `mostCapable` and `DeviceCapabilities` (with its `Precision` enum) are removed.
  They were fixed constants and simplified checks, not hardware discovery or measured thresholds.
  `ComputeDevice` stays a descriptive label with `createProvider()`; use `isSupportedByCore` and
  `coreSupportedDevices` for VectorCore's own providers, and a backend's own discovery API for hardware.
- **Breaking (C): 18 proposed declarations in `VectorCoreC.h`.** `vc_l2sq_int8`,
  `vc_dequantize_fp32_from_int8`, `vc_l2sq_mixed_int8_fp32`, the four `vc_soa_*` kernels,
  `vc_range_l2sq_fp32_768`, `vc_range_l2sq_fp32_1536`, `vc_cosine_fused_pairwise_fp32`,
  `vc_heapifyDown_smallk`, `vc_topk_merge_smallk`, the three `vc_range_topk_*` kernels,
  `vc_magnitude_sq_fp32`, `vc_range_magnitude_sq_fp32` and `vc_scale_in_place_fp32` were declared but
  never defined, so a reference to one failed to link. The 13 implemented functions stay.
- **Breaking: `Operations.bufferProvider`, `VectorCore.Configuration` and `VectorCore.configuration`.**
  Nothing in VectorCore read the task-local `bufferProvider` (or its `$bufferProvider` binding), the
  `Configuration` struct (`batchOperations`, `memoryAlignment`, `init()`) or the `configuration`
  instance, so setting them changed no behavior. For batch settings use `BatchOperations.configuration()`
  and `BatchOperations.updateConfiguration`. For pooled buffers call a `BufferProvider`, such as a
  `SwiftBufferPool` or `SwiftBufferPool.shared`, directly (`acquire(size:)`, `release(_:)`,
  `withBuffer(size:_:)`). No global setting replaces `memoryAlignment`: a pool's alignment is its
  `BufferConfiguration.alignment`. The `BufferProvider` protocol and its implementations stay.
- **Breaking: the flag helpers `isSIMDEnabled`, `isReleaseMode`, `releaseOnly` and
  `performancePrecondition`.** They read compilation conditions that no package target defines, so in
  package builds the first two returned `false`, `releaseOnly` skipped its block and
  `performancePrecondition` checked its condition in every configuration. Nothing in the package used
  them apart from the test that pinned this, now the compile-time check
  `Scripts/ci/tests/D02/RoutedNegative.swift`. Use `#if` with your own compilation condition, or
  `precondition`. The other helpers in `OptimizationAttributes.swift` stay. The `#if PERFORMANCE_METRICS`
  call counters of `BatchKernels_SoA`, with their `reportMetrics()`, are deleted as well; no package
  target defines that condition, so default builds never compiled them.
- **Breaking: `SoA.memoryEfficiency`.** It returned fixed text comparing two equal byte counts; nothing
  in the package called it. `memoryFootprint` and `layoutDescriptor` stay.
- **Breaking: `BatchKernels_SoA.estimatePerformanceGain`.** It returned a fixed estimate that no
  measurement backed; nothing in the package called it.
- **The internal `SIMDMemoryUtilities` helpers.** They had no callers. They were internal, so no public
  API changes.

- **Breaking (source-compatible): 24 concrete metric overloads.** The
  `distance(_:_:)` and `batchDistance(query:candidates:)` overloads of
  `EuclideanDistance`, `DotProductDistance`, `ManhattanDistance` and
  `ChebyshevDistance` for `Vector512Optimized`, `Vector768Optimized` and
  `Vector1536Optimized` are removed from `OptimizedDistanceMetrics.swift`. Calls
  with those concrete types now bind the `DistanceMetric` witness and compile
  unchanged; the Euclidean and dot-product witnesses use the removed overloads'
  kernels (see Changed), and the Manhattan and Chebyshev witnesses returned the
  same bits as the overloads on every T14 probe fixture when the overloads were
  removed (T14 probe, one host, before T15; records outside this repository;
  not pinned by a test). T15 then changed `ChebyshevDistance` for a NaN
  coordinate and for the same infinity in both vectors (see Changed), so on
  those inputs a concrete Chebyshev call now returns NaN where the removed
  overload returned a number. Binaries built against the removed symbols must
  be rebuilt. `MetricWitnessEquivalenceTests` checks that
  concrete, generic and existential calls agree, and
  `Scripts/ci/tests/test_metric_overload_registry.py`
  requires a registry entry for any concrete `distance`/`batchDistance` on a
  `DistanceMetric` type.

- **Breaking (source-compatible): the 6 concrete `CosineDistance`
  overloads.** `distance(_:_:)` and `batchDistance(query:candidates:)` for
  `Vector512Optimized`, `Vector768Optimized` and `Vector1536Optimized` are
  removed from `OptimizedDistanceMetrics.swift`, which now holds only the
  metric factory. Calls with those concrete types bind the `CosineDistance`
  witness and compile unchanged; the witness uses the removed overloads'
  kernel (`CosineKernels`' fused kernel; see Changed), so their bits did not
  change on any T14 probe pair (one host; records outside this repository).
  Binaries built against the removed symbols must be rebuilt. No concrete
  `distance`/`batchDistance` overload remains on a `DistanceMetric` type; the
  registry guard (`Scripts/ci/tests/test_metric_overload_registry.py`) fails
  on a new one without a `MetricWitnessEquivalenceTests` entry.

- **The native `Float16` branch of the FP16 scalar converters, never enabled
  by default.** The internal `NATIVE_FLOAT16_SUPPORTED` branch, an unread
  private constant and an unused private bulk-conversion helper are deleted.
  No package target defines that condition, so default builds never compiled
  the branch, and this removal does not change what they compute. Builds that
  defined it explicitly (for example with `-Xswiftc -DNATIVE_FLOAT16_SUPPORTED`)
  now use the software converters and the NaN policy above. No public API
  changes.

- **Two unused internal quantization analyzers.** `QuantizationErrorAnalyzer`
  (with its `QuantizationError` result type) and `QuantizationQualityAnalyzer`
  had no callers and no tests, and their maximum-error folds dropped NaN. They
  were internal, so no public API changes.

### Fixed

- **Release builds check the output capacity of the FP16 SoA batch kernels and
  the row shapes of `MatrixDistance`.** The fifteen public
  `MixedPrecisionKernels.batchEuclidean{512,768,1536}`, `batchDotProductSoA` and
  `batchEuclideanBlocked{512,768,1536}` SoA kernels, the six public
  `batchDotFP16_{512,768,1536}` and `batchDotMixed{512,768,1536}` array kernels
  (through two shared internal kernels) and the nine internal SoA kernels behind
  the optimized batch routes checked the output count only under `#if DEBUG`, so
  a Release caller passing a buffer shorter than the candidate count was written
  past its end with no diagnostic. They now stop with a named `fatalError` before
  any write, in every build; the mixed-precision range core's own capacity and
  range preconditions are active in every build as well. The policy test
  `test_release_active_capacity_guards.py` fails on any new Debug-only capacity
  assert in `Sources/VectorCore`.
  `MatrixDistance.prepare` and the six matrix entries took the dimension from
  the first row and copied that many bytes from every row, reading past a
  shorter row; every packed row must now have that dimension, checked by a
  named `fatalError` in every build. `ReleaseSizeGuardTests.kernelTrapsBeforeWriting`
  and `matrixRowsOfUnequalLengthTrap` run each case in a child process on the
  Debug and Release rows and check that the output it was given is untouched.

- **Pool teardown leaks active leases instead of freeing them, and the
  `MemoryPool` cleanup timer no longer depends on the width of `Int`.**
  Destroying a `TypedBufferPool`, a `SwiftBufferPool` or the internal
  `BufferPool` while a lease was still checked out freed that lease, so a
  `Handle` or `BufferHandle` that outlived its pool pointed at freed memory and
  the first write through it was a silent use-after-free. The three pools now
  leak active leases on teardown and still free their cached allocations, as
  0.3.3 did, so an escaped handle stays valid
  (`BufferPoolOwnershipTests.internalAndTypedDestructionOwnCachedAndLeakActive`,
  `swiftPoolDestructionLeaksActiveLease`); the lease contract is unchanged:
  release every handle before dropping the pool. `MemoryPool` compared the
  cleanup interval against `Double(Int.max) / 1e9`, which is 2.147 s on a 32-bit
  `Int` (arm64_32 watchOS devices, inside the declared floor), so the default
  30 s interval silently disabled automatic cleanup there; the bound is now the
  64-bit nanosecond horizon on every platform
  (`MemoryPoolBoundaryTests.timerHorizonIsIndependentOfIntWidth`).

- **`ComputeProvider`'s chunk-hinted defaults call the hint-less requirements.**
  The default `parallelForEach(items:minChunk:)` and
  `parallelReduce(items:initial:minChunk:_:_:)` validate `minChunk` and then call
  the hint-less requirement, the direction 0.3.3 used, so a conformer that
  implements only the hint-less overloads (the ones the 0.3.3 guide named) is
  reached by every optimized search route and its `maxConcurrency` is not
  consulted by those defaults. During development the direction had been
  reversed, which bypassed such conformers and threw `invalidConfiguration` for
  a `maxConcurrency` of 0
  (`ComputeProviderForwardingTests.hintedDefaultsReachHintlessWitnesses`).

- **The never-run consumer fixtures are gone, and the must-not-compile contracts
  now run.** Forty-six fixture, probe and driver files under `Tests/*Consumers`,
  `Scripts/ci/tests` and `Scripts/ci/fixtures` were executed by nothing: not a
  SwiftPM test target, not collected by unittest, not called by any workflow.
  They are removed; what remains in those directories is read or executed by a
  live test or regenerates a committed table. The nine must-not-compile fixtures
  (`R01MetricScalarContract`, `R02CollectionUpdateConsumer`, the five D02
  negatives, `B01BareConformanceNegative`, `R05OldPropertyNegative`) and their
  positive controls are compiled by `consumer_smoke.py` in the consumer job, from
  the documentation examples' scratch package: a negative must fail with one
  error on each expected line and the named diagnostic, a control must build or
  run and print its marker. The table is checked without a compiler by
  `test_consumer_smoke.CompileContractTests.test_table_names_the_nine_negatives_and_their_controls`
  and the judgement by `ContractJudgementTests.test_negative_with_every_code_line_erroring`.
  The stale paragraph in
  `Docs/README.md` that called the Performance Guide and the how-to "retained
  unchanged" is rewritten, and the how-to's four throwing calls now say `try`.

- **`VectorCoreBenchmarking` builds on every declared Apple platform, and normalization builds on Intel
  macOS and Mac Catalyst (T07).** `Host.current()` and Foundation's `Process` are macOS-only, so off
  macOS `EnvCapture` now reports the device tag `"host-unavailable"`, subprocess observations as
  `.unavailable(reason)` and an OS string with an OS-family prefix; macOS metadata is unchanged byte for
  byte. The `Float16` identity check in `NormalizationCore` is guarded where the SDK marks `Float16`
  unavailable (x86_64 macOS and Mac Catalyst), with the same guard as `MixedPrecisionKernels`, fixing a
  build failure introduced by N01 (never released); arm64 behavior is unchanged. The off-macOS fallbacks
  are compile-verified only, and the simulator and Catalyst destinations are not supported tooling hosts.

- **`vectorcore-bench` correctness gate.** The generic `batch.euclidean2` cases
  no longer record correctness failures on correct output, so the guide's CI
  comparison runs without `--exclude`. The Euclidean and squared-Euclidean
  batch routes, their `boundary` companions and the `matrix` routes are held to
  thresholds derived from each route's worst-case rounding bound for the case's
  dimension and candidate count, replacing a fixed relative `1e-6` that the
  generic route's long FP32 sums legitimately exceed. A relative error now
  divides by `max(|ref|, 1)`, and the `matrix` routes, whose diagonal
  references are zero, are gated on absolute error. `noop.scalar-add` and the
  `mem.alloc`, `mem.pool` and `mem.fill` cases declare that they have no output
  to validate instead of recording "Missing validation"; `dot`, `normalize` and
  `mem.copy` still record it. Other validated routes keep the fixed
  thresholds. Regression coverage: `BenchmarkCorrectnessPolicyTests` in
  [BenchmarkContractTests](Tests/BenchmarkContractTests/BenchmarkContractTests.swift).

- **Benchmark metadata `binarySizes.vectorCoreLibraryBytes`.** A loaded image is
  now matched by file name (`libVectorCore.dylib`, or the binary inside
  `VectorCore.framework`), and the field is nil when VectorCore is linked
  statically. It previously took the first loaded image whose path contained
  "VectorCore", which under a VectorCore checkout was the benchmark executable,
  so it repeated `benchExecutableBytes`.

- **MemoryPool bookkeeping and lifetime.** Replace asynchronous queue updates
  and synchronous queue waits with synchronous `NSLock`-protected state
  changes, addressing cooperative-task starvation. Free allocations when their
  handles outlive the pool; enforce retained-memory limits and cleanup accounting
  in bytes. Regression coverage: [MemoryPoolTests](Tests/ComprehensiveTests/MemoryPoolTests.swift)
  and the [externally timed pool probe](Scripts/ci/check_memory_pool.py).
  This does not synchronize caller access to checked-out raw buffers.
- **FP16 portability on Intel macOS and Mac Catalyst.** Remove unconditional
  uses of native `Float16`, which these Intel SDK targets mark unavailable.
  The 512/768/1536 FP16 buffers retain portable `UInt16` storage and classify
  finite values from their exponent bits. Native conversion remains opt-in;
  `detectOverflow(value:)`, whose return type is `Float16?`, is excluded on
  those Intel targets. Use `canRepresent(_:)` or `validateBatch(values:)` for
  FP32 range checks there. Regression coverage:
  [MixedPrecisionRangeValidationTests](Tests/ComprehensiveTests/MixedPrecisionRangeValidationTests.swift).

- **FP16 kernels no longer heap-allocate on each 8-lane load.** A
  module-internal `UnsafeRawPointer.loadUnaligned(as:)` extension shadowed the
  standard library's allocation-free `loadUnaligned` inside VectorCore. Each
  `SIMD8<UInt16>` load in the FP16 range kernels
  (`range_{euclid2,cosine,dot}_mixed_*`), pair kernels (`dotFP16_*`,
  `dotMixed*`, `euclidean*`, `cosine*`), batch kernels (`batchDotFP16_*`,
  `batchDotMixed*`) and `toFP32()` allocated, copied and freed 16 bytes. The
  extension is removed, together with an unused `UnsafeRawBufferPointer`
  companion. The binary serialization reads already used the standard-library
  overload, and their allocation counts are unchanged (T12 emitted-code
  inspection and allocation probe, one host; records outside this repository).
  Kernel outputs are unchanged: T12's fixture probe found them byte-identical
  before and after (one host, records outside this repository). One-host
  Release measurement (arm64, macOS 27.0.1), allocation events per call over
  4,096 candidates:
  `range_euclid2_mixed_512` fell from 262,145 to 1 (by size, inferred to be the
  per-call candidate-handle array), `range_euclid2_mixed_1536` from 786,433 to
  1 and `batchDotFP16_512` from 524,289 to 1. Single-pair calls such as
  `dotFP16_512` fell from 128 to 0.
  `Scripts/ci/tests/test_no_stdlib_member_shadows.py` now rejects module
  extensions that redeclare curated standard-library raw-pointer or SIMD
  members, except two pinned `SIMD.sum()` declarations, allowlisted pending a
  routed semantic review (T14).

- **LAPACK workspace check on near-square inputs (introduced after `v0.3.3`,
  fixed before release).** `v0.3.3` is not affected: its shim sized each
  workspace from Accelerate's answer and rejected none. Earlier in this release
  cycle the shim began checking Accelerate's workspace answers and treated its
  reference bounds (4k² + 7k for the SVD) as minimums for them. With the
  default LAPACK provider, `LAPACKLinearAlgebraProvider.svdThin` and
  `PCAModel.fit` then threw `.operationFailed` ("sgesdd returned an invalid
  workspace query") on shapes where LAPACK's `sgesdd` does not QR-factor first
  (max(m, n) < ⌊11·min(m, n)/6⌋) and Accelerate's answer was below the bound:
  for example exact PCA (`PCAConfig(oversampling: Int.max)`) of 500 samples in
  dimension 384 or 768, and sketch-path fits whose ℓ×d panel has that shape. In
  T19's sweep that was every such shape with min(m, n) = 64, only
  max(m, n) ≤ 17 with min(m, n) = 16, and none with min(m, n) = 8. This entry
  fixes the check before release: the SVD, symmetric-eigen and QR bounds are
  now allocation floors only. The shim allocates the larger of Accelerate's
  answer (plus the existing headroom) and the bound, and still rejects
  non-finite, non-positive and unrepresentable answers. Regression coverage, `T19LAPACKWorkspaceQueryTests` against designed
  spectra: `svdThinSweepCrossesPathFive` and `exactPCASweepCrossesPathFive`
  (across the boundary, tall and wide), `reportedCellsSucceed`,
  `randomizedPCAWidePanelsCrossPathFive` (two sketch-path panels), and the
  near-boundary `symmetricEigenNearBoundaryShapes` and `qrThinNearBoundaryShapes`.

### Documentation

- Refresh the README and core references, distinguish current contracts from
  historical evidence, and compile/run the complete reference examples as
  downstream consumers. Tutorials remain pending a separate guide pass.
- Correct historical matrix allocation, accuracy, and routing claims below;
  clarify Swift SVD/PCA rank-deficiency limits and release availability. These
  are documentation corrections, not changes to the numerical algorithms.
- Reconcile performance and robustness claims with their evidence. Benchmark
  harness and model comments, selected library comments and
  `Docs/Performance_Guide.md` now keep a claim only when adjacent code or a named
  test backs it, or mark it as explicitly unverified; unsupported time figures
  and speed multipliers were removed. The claim corrections in library sources
  are comment-only. `Scripts/ci/check_claim_ledger.py` checks these statements
  against `Scripts/ci/claim_ledger.json` in the repository-policy unittest step;
  see [Maintaining](Docs/Maintaining.md#claim-ledger).
- `Docs/Performance_Guide.md` uses the real `vectorcore-bench` options
  (`--suites`, `--out`) and `Scripts/compare_benchmarks.swift` instead of a
  nonexistent Python script, and lists the `boundary` and `noop` suites.
  Its CI recipe now passes `--samples 5`: a single-sample report carries no
  median, and the comparator never substitutes another statistic, so
  `--metric median` on the recipe's former reports always exited 2.
- `Docs/Benchmark_Measurement_Boundaries.md` defines `unverified` per case ID,
  lists the measured `routeInternalAllocation` resolutions and their scope (28
  case IDs at 512 dimensions, one Release arm64 build, records outside the
  repository), and states that report metadata (`gitSHA`, `swiftVersion`,
  `runtimeCheckout`, `runtimeToolchain`) is a run-time observation, not the
  identity of the measured source or of the compiler that built the binary.
- Extend the claim ledger (T13). Qualitative performance and robustness
  labels, per-operation times, speed multipliers and the undefined copy-free
  phrase were removed from library comments, `README.md` and the maintained
  documents, or replaced by the mechanism, a named test or unverified wording;
  the changes in library sources are comment-only. New ledger rules reject
  these forms lexically, rule 6 also rejects spelled-out speed multipliers,
  and line-number citations are rejected in `README.md`, `CHANGELOG.md` and
  the maintained documents. `Operations.swift`, `Vector384Optimized`,
  `Vector768Optimized`, `Vector1536Optimized`, `README.md`,
  `Docs/API_Overview_Map.md`, `Docs/HowTo_NearestNeighbor.md` and this file's
  `[Unreleased]` section are ledger regions, and every `acceptedContract` must
  name an entry of the ledger's `contracts` table
  (`test_checked_in_ledger_matches_sources`). The `UnifiedVectorBuffer`
  header now says that only `PageAlignedBuffer` and a page-aligned `SoA` meet
  the `makeBuffer(bytesNoCopy:)` preconditions; the nearest-neighbor how-to
  and the API map state the current selection and metric-call rules; and the
  2026-09-05 verification record carries a dated note on what P01
  superseded.

See the [September 7 verification record](Docs/verification-public-readiness-2026-09-07.md)
for the post-`0.3.3` fixes and the configurations actually checked. Compile
coverage is not a claim of runtime coverage on every declared platform.

- **VectorCoreC contracts.** `VectorCoreC.h` now states that `vc_has_neon`/`vc_has_dotprod` report
  compile-target macros while `vc_has_avx2`/`vc_has_avx512f` query the executing x86_64 CPU; that
  `vc_range_l2sq_fp32_512` writes compactly to `out[0 ..< end - start]` and its caller obligations;
  and the `vc_dot_int8` law. The internal `vc_scalar_*`, `vc_arm64_*` and `vc_x86_*` symbols are
  linker-visible but not supported API.
- **Dispatch of protocol members.** `VectorProtocol` and `VectorFactory` document which members are
  dynamically dispatched requirements and which extension members are statically selected, checked
  for every member by `D02WitnessDispatchMatrixTests`. `VectorProtocol.cosineDistance(to:)` is
  documented as its own arithmetic, bounded against the `CosineDistance` witness by
  `cosineDistanceStaysWithinTheDerivedBoundOfTheWitness`; its arithmetic is unchanged.
- **`VectorTypeFactory.isSupported(dimension:)`** names the factory's actual results: `Vector<DimN>`
  for 128, 256, 384, 512, 768, 1536 and 3072, `DynamicVector` otherwise, never a `Vector*Optimized`
  type (`concreteResults`).
- **Provider hooks.** The `Operations` documentation lists every public entry that reads
  `computeProvider`, `simdProvider` or `linearAlgebraProvider`, with the calls a spy installed through
  that hook records at the real route (`D02ProviderHookRouteTests`). `test_d02_surface_guards.py` fails
  when a hook read appears in a function the table does not cover.
- **`VectorCore.createVector(dimension:data:)` and `hasOptimizedSupport(for:)`.** `createVector` states
  its preconditions (`dimension >= 0`, and `data`, when given, has `dimension` elements) and that a
  violation stops the process, with `VectorTypeFactory.vector(of:from:)` as the throwing alternative
  (`createVectorTrapsOnMalformedInput`). `hasOptimizedSupport(for:)` names the seven `Vector<DimN>`
  widths it reports and claims no speed difference.
- **`Logger` next to `os.Logger`.** `Logger` keeps its name. In a file that imports both `os` and
  VectorCore, `import struct VectorCore.Logger` makes `Logger` name VectorCore's type there, and Apple's
  stays `os.Logger`. Client code cannot spell it `VectorCore.Logger`: `VectorCore` resolves to the
  `VectorCore` enum. Without the scoped import a bare `Logger` is ambiguous under Swift 6.4 and accepted
  by Swift 6.0 (Xcode 16.2), so write the scoped import rather than relying on either. Checked by the D02
  compile contracts `Scripts/ci/tests/D02/LoggerQualification.swift` and `LoggerNegative.swift`, which the
  consumer job compiles with Xcode 16.2 and which were also run with Swift 6.4.
- **Products and hosts.** `Docs/Package_Boundaries.md` lists, for each product, the hosts that are
  supported, compiled in CI or not verified. `VectorCoreBenchmarking` and `vectorcore-bench` are macOS
  host tooling; `VectorCoreBenchmarking` compiles in CI for the simulator and Catalyst destinations, but
  those are not supported tooling hosts.
- **`VectorQualityMetrics.entropy`** is documented as a caller-supplied value that the type stores
  without computing it, not as Shannon entropy.
- **Measured workload envelope for PCA, kNN and UMAP (one host).** `Docs/Performance_Guide.md` gains a
  dated record of exact and randomized `PCAModel.fit` with `transform`, `KNNGraph.bruteForce` and the
  `Operations.umap` vector entry, for N from 50 to 100,000 and D of 384, 768 and 1536 on synthetic
  embedding-like data, measured on one Apple M3 Max host in High Power Mode. The full record is
  `Benchmarks/WorkloadEnvelope/RESULTS.md`. Inputs outside that grid are not measured: no size limit
  applies to them beyond the documented representability checks, which throw
  `VectorError.invalidDimension`, and their time and memory are not promised. Doc comments on
  `PCAModel.fit`, `Operations.pca`, `KNNGraph.bruteForce` and `Operations.umap` point to the record and
  state the work and storage their code implies. No code changed.

## [0.3.3] - 2026-09-06

### Fixed

- Top-K selection now ranks NaN scores after all numeric scores in both
  minimization and maximization. The default smaller-index policy resolves exact
  numeric ties (including signed zeros) and NaN ties consistently for membership
  and output ordering. NaNs remain in results when needed to satisfy the requested
  count. Pointer IDs remain output labels; ties use original input positions.
- CPU `Operations.findNearest` and `BatchOperations.findNearest` use the same
  selection contract, including parallel partial merges. Euclidean GEMM batch
  output preserves selected NaNs. Metric computation, squared-distance selection,
  and third-party provider delegation retain their existing behavior.

## [0.3.2] - 2026-07-24

Maintenance release: an Accelerate-backed `ArraySIMDProvider`, a fresh
sanitizer-verified baseline, and deterministic (non-flaky) test sweeps.

### Added

- **`AccelerateArraySIMDProvider`** — an `ArraySIMDProvider` for Apple
  platforms that routes every operation vDSP covers (arithmetic, reductions,
  dot/distance, element-wise combine) through Accelerate, and delegates the
  rest (the vForce-backed `abs`/`sqrt`/`log` and trivial index scans) to
  `DefaultArraySIMDProvider`. Parity-tested against `DefaultArraySIMDProvider`
  at lengths 1–513, including non-multiple-of-SIMD-width inputs. Installed the
  standard way:
  `Operations.$simdProvider.withValue(AccelerateArraySIMDProvider()) { ... }`.

### Fixed

- **`Operations` provider docs showed a non-compiling pattern.** The header
  example assigned providers directly (`Operations.simdProvider = ...`); the
  providers are `@TaskLocal`, so the docs now show scoped
  `$provider.withValue(_:operation:)` binding.
- **Two tests asserted debug-only `VectorError` behavior in all
  configurations.** Source-location context capture is `#if DEBUG`-gated in
  `VectorError`; the assertions now carry the same gate, fixing spurious
  failures in Release test runs.

### Tests

- **Wall-clock performance assertions are now opt-in.** 47 timing/speedup
  `#expect`s across the comprehensive suites only assert when
  `VECTORCORE_STRICT_PERF=1` is set (see `Tests/ComprehensiveTests/PerfGate.swift`);
  otherwise measurements are printed but never fail the suite. This makes
  Debug/Release/ASan/TSan sweeps deterministic — sanitizer slowdowns and CI
  load no longer produce noise failures. Accuracy and layout assertions are
  unaffected.

### Documentation

- `Docs/verification-baseline-0.3.1.md` — full verification baseline on
  `main` @ 0.3.1: Debug and Release suites, an ASan pass (0 reports,
  including the LAPACK shim and `PageAlignedBuffer` allocation-consumption
  paths), and a TSan pass over the batch/GEMM/SoA kernels (0 reports), with
  every failure classified (test bugs and timing noise; no code regressions).
- `Docs/gap-analysis-hn-semantic-search.md` (+ its execution plan) added;
  ecosystem strategy doc renumbered.

## [0.3.1] - 2026-06-11

The projection stack from the HN semantic-search gap analysis (the P0 band):
dense linear algebra, PCA, and UMAP land in Core, plus the Core-owned
kNN-graph interchange contract that VectorIndex populates at corpus scale.
Zero new dependencies; everything stochastic is seeded and deterministic.

### Added

- **`LinearAlgebraProvider` seam — thin QR / thin SVD / symmetric eigen.**
  Column-major `[Float]` factorizations behind a task-local provider
  (`Operations.$linearAlgebraProvider`): `LAPACKLinearAlgebraProvider` routes
  through Accelerate's modern LAPACK (`ACCELERATE_NEW_LAPACK` via `cSettings`,
  32-bit indices, a `VectorCoreC` shim with lwork negotiation), and
  `SwiftLinearAlgebraProvider` (Householder QR, cyclic Jacobi eigen, Hestenes
  SVD) keeps non-Apple platforms working. Parity-tested against each other.
- **`PCAModel` / `Operations.pca` — PCA via randomized SVD**
  (Halko–Martinsson–Tropp): Gaussian sketch plus QR-stabilized power
  iterations; fit on a sample, then stream the corpus through `transform` in
  caller-sized batches (one GEMM per batch). Seeded deterministic sketch,
  pinned sign convention (largest-|coordinate| positive), exact thin-SVD
  fallback when the sketch cannot be thinner than the data;
  `explainedVariance` / `explainedVarianceRatio` reported.
- **`Operations.umap` — UMAP layout.** Fuzzy simplicial set (smooth-kNN ρ/σ
  bisection against the log₂(k+1) target, probabilistic t-conorm union),
  output-kernel curve fit by Levenberg–Marquardt (reproduces umap-learn's
  a≈1.577, b≈0.895 defaults), and SGD with epoch-cadence edge sampling,
  negative sampling, ±4 gradient clipping, and linear annealing.
  Single-threaded and fully seeded: identical inputs give a bit-for-bit
  identical layout. PCA initialization by default (per the gap report's scale
  correction); random and caller-provided coordinates supported.
- **`KNNGraph` — the Core-owned CSR interchange contract.** A validated
  sparse k-nearest-neighbor graph (no self-loops, finite non-negative
  distances, monotone offsets, variable row degree) that graph producers —
  VectorIndex's ANN indexes at corpus scale — hand TO Core: data flows
  Index → Core, code never does. `KNNGraph.bruteForce` (blocked Gram-trick
  GEMM, O(n²·d)) is the exact in-Core builder for samples and tests.
- **`LAMatMul`** (internal) — column-major GEMM utility (`cblas_sgemm` on
  Apple platforms; a parity-tested pure-Swift kernel elsewhere) shared by
  PCA, UMAP, and the brute-force kNN builder.

### Fixed

- **`VectorCoreVersion` drift.** The constants still reported 0.2.2 in the
  0.3.0 release; they now match the package version again.

### Documentation

- `Docs/Package_Boundaries.md` records the ratified projection-stack
  placement: dense linear algebra / PCA / UMAP *math* in Core; the `KNNGraph`
  CSR container Core-owned as a data-interchange contract; graph
  *construction* and clustering remain VectorIndex territory.

## [0.3.0] - 2026-06-07

> **Documentation correction — 2026-09-14:** The original entry below
> overstated three contracts; it is retained as release history, not as an
> allocation or accuracy guarantee. `into:` reuses caller-owned output but
> still allocates intermediate storage, including packed queries and norms.
> GEMM distances have no universal `~1e-3` relative-error bound, especially
> near zero. `BatchOperations.updateConfiguration` controls the pairwise
> matrix gate, not the separate gate in `Operations.findNearestBatch`.
> These clarifications also apply to the `v0.3.0` implementation; they do not
> describe newly changed behavior. See [Numerical Behavior](Docs/Numerical_Behavior.md#matrix-distances-shape-memory-and-precision)
> and [Search and matrix routing](Docs/API_Overview_Map.md#search-and-matrix-routing).

Outcome of the "beta-evolution-4" (BE4) review: **hold the line, sharpen the
seams.** Rather than grow VectorCore into a database/index engine, this release
hardens the CPU-primitive foundation and the integration seams that downstream
packages (VectorIndex, VectorAccelerate, EmbedKit) were bypassing. Net new
surface: one CPU math primitive (GEMM batch distance), a zero-copy GPU-bridge
buffer contract, a transparent GPU-dispatch provider seam, and deterministic
Top-K — plus the removal of a misleading no-op parameter.

### Added

- **`MatrixDistance` — CPU GEMM batch distance.** `euclideanSquaredMatrix` and
  `cosineDistanceMatrix` compute a full query×candidate distance matrix via
  Accelerate `cblas_sgemm` (which routes to the AMX coprocessor on Apple
  Silicon), using the identity `‖x−Y‖² = ‖x‖² + ‖Y‖² − 2⟨x,Y⟩`. Each metric has
  an `into:` (zero-allocation), an allocating, and a `prepared:` overload;
  `prepare(_:normalized:)` packs a candidate set once (`PreparedCandidates`) for
  reuse across query batches. Generic over `UnifiedVectorBuffer`. Output is
  row-major (`out[i*n + j]`); the identity can round slightly negative when
  `q ≈ c`, so results are clamped (Euclidean at 0, cosine to `[0, 2]`) and agree
  with the per-pair kernels to within ~1e-3 relative, not bit-for-bit.
- **Zero-copy GPU-bridge buffer contract.** `UnifiedVectorBuffer` protocol
  (`elementCount`, `alignment`, `withUnsafeContiguousBytes`) + `PageAlignedBuffer`
  — a page-aligned, page-length-rounded `[Float]` slab valid for
  `MTLDevice.makeBuffer(bytesNoCopy:)`. The optimized vector types and
  `DynamicVector` conform. `PageAlignedBuffer.consumeAllocation()` transfers
  ownership to a Metal deallocator without a double free.
- **Frozen SoA memory-layout contract.** `SoALayout` descriptor +
  `SoA.layoutDescriptor` / `SoALayout.forType(_:count:pageAligned:)` publish the
  SoA layout — `lanes`, `count`, `laneStrideBytes`, `logicalByteCount`,
  `allocatedByteCount`, and the `lane*count+candidate` index formula — as a
  stable contract for downstream GPU kernels. Documented at
  `Docs/SoA_Layout_Contract.md`.
- **Opt-in page-aligned `SoA`.** `SoA.build(from:pageAligned:)` /
  `init(vectors:pageAligned:)` allocate a page-aligned, page-rounded buffer;
  `pageAlignedBytes` exposes the base + length for zero-copy GPU import,
  `consumeAllocation()` hands ownership to a Metal deallocator, and
  `withUnsafeRawBuffer` gives scoped read access. Default stays 16-byte aligned.
- **`BatchKernelProvider` — transparent GPU dispatch.** A `ComputeProvider`
  sub-protocol (`batchDistance` / `findNearest` / `findNearestBatch`). When an
  installed `computeProvider` conforms, `Operations.findNearest` and
  `findNearestBatch` delegate to it (GPU path), taking precedence over the CPU
  GEMM routing; otherwise the CPU path is used.
- **Deterministic Top-K tie-breaking.** `TieBreaker` (`.smallerIndex` (default),
  `.insertionOrder`, `.smallerValue`) makes `TopKSelection` results reproducible
  when distances tie.
- **GEMM routing for batch operations.** `BatchOperations.Configuration` gains
  `enableMatrixRouting` (default `true`) and `matrixRoutingMinN` (default `256`);
  `pairwiseDistances` and `Operations.findNearestBatch` route large batches
  through `MatrixDistance`. Tune via `await BatchOperations.updateConfiguration { … }`.
- **`PlatformConfiguration.roundUpToPage(_:)`** — the single page-rounding helper
  shared by `SoA` and `PageAlignedBuffer`.

### Removed

- **The no-op `blockSize:` parameter is removed from `SoA.init(vectors:…)` and
  `SoAFP16.init(vectors:…)`.** It was ignored (neither type pads the candidate
  axis) and implied a layout guarantee that does not exist. **Source-breaking**
  only for call sites that passed `blockSize:` explicitly — it had a default, so
  `SoA(vectors:)` / `SoAFP16(vectors:)` are unaffected.

### Documentation

- New `Docs/SoA_Layout_Contract.md` (the frozen SoA layout). Refreshed the
  Package Boundaries, Memory Alignment, Performance, API Overview, and roadmap
  docs for the 0.3.0 surface.

## [0.2.2] - 2026-06-06

> **Documentation correction — 2026-10-04:** The "Candidate-tiled SoA kernels"
> entry under Performance below is retained as release history, not as a
> performance claim. Its "~8 MB (≈ L2)" and "prefetcher-optimal" descriptions
> were not measured and are not supported by retained evidence: the threshold is
> a candidate-buffer size (more than 4096 candidates at 512 dimensions), not a
> measured cache size, and no prefetch behavior was measured.

Outcome of the "beta-evolution-3" (BE3) architectural audit: a sweep for
correctness, memory-safety, and numerical-rigor defects, plus targeted
performance work and the elimination of test flakiness. The full failing-test
suite is green (0 failures), ThreadSanitizer- and AddressSanitizer-clean, and
deterministic across runs.

### Changed

- **`LinearQuantizationParams.zeroPoint` widened from `Int8` to `Int32`.** The
  affine zero-point offset overflows a signed 8-bit field for value ranges that
  do not straddle zero, collapsing INT8-quantized signals; `Int32` fixes the
  correctness bug. **Source-breaking** for code that reads or constructs
  `zeroPoint` as `Int8` (the type is `public` and `Codable`). Numeric JSON
  encoding is unchanged.

### Added

- `BatchKernels_SoA.batchEuclideanSoA` (512/768/1536) — correctly-named
  replacement for `batchEuclideanSquaredSoA` (it returns the true Euclidean
  distance, not the squared value).

### Deprecated

- `BatchKernels_SoA.batchEuclideanSquaredSoA` — renamed to `batchEuclideanSoA`;
  a deprecated shim is retained for source compatibility and emits a rename
  warning.

### Fixed

Correctness & memory safety:
- **Heap-buffer-overflow** in `SwiftFloatSIMDProvider` SIMD8 reductions
  (`maximum`/`minimum`/`maximumMagnitude`) — an off-by-one read past the buffer
  for any element count that is a multiple of 8 (i.e. all standard dimensions),
  which was the root cause of intermittent `softmax` NaNs.
- **Heap out-of-bounds read** in `adaptiveEuclideanDistance` (a flat 512-element
  buffer was wrapped as a 2048-element SoA).
- **Strict-aliasing (TBAA) undefined behavior** in the optimized vector types —
  permanent `bindMemory(to:)` replaced with scoped `withMemoryRebound(to:)`.
- **Int16 overflow** in the INT8 quantized Euclidean kernel — squared component
  differences now accumulate without two's-complement wrapping.
- **Infinity overflow** in the cosine-distance denominator — computed as
  `sqrt(sumAA) * sqrt(sumBB)` instead of `sqrt(sumAA * sumBB)`.
- **Memory leak** on asynchronous `MemoryPool` buffer return — `posix_memalign`
  memory is now released via `free()` even if the pool deallocates first.
- **Subnormal reciprocal overflow / NaN poisoning** during normalization of
  all-subnormal vectors.
- **Epsilon truncation** that rejected valid dense micro-vectors in cosine
  distance — the zero-vector floor is now `Float.leastNormalMagnitude`.
- Subnormal FP32→FP16 rounding error; `analyzePrecision` precision selection;
  and `SoAFP16.init` silently producing empty containers for 768/1536-dim.

### Performance

- **Eliminated heap zero-fill churn** in `DefaultArraySIMDProvider`:
  arithmetic, element-wise, and transcendental operations now allocate their
  result via `Array(unsafeUninitializedCapacity:)` instead of paying for a
  whole-buffer `memset` that the SIMD/vForce primitive immediately overwrites.
  `Operations.centroid` accumulates in place (N allocations → 1).
- **Candidate-tiled SoA kernels** for large batches: `euclid2`, `dot`, and
  fused-cosine kernels switch to a lane-outer / candidate-inner traversal above
  an ~8 MB (≈ L2) candidate-buffer threshold, streaming one sequential column
  per lane (intended for the hardware prefetcher; not measured) instead of L
  interleaved strided streams.
  Register-blocked 4-way kernels are retained below the threshold. Results are
  **bitwise-identical** to the 4-way path.

### Internal / Tests

- Eliminated test flakiness: ~300 unseeded `random` draws in the numerical test
  suites and the source-side benchmark/auto-tuner data generators
  (`KernelAutoTuner`, `MixedPrecisionBenchmark`) are now seeded, making accuracy
  and timing measurements deterministic.
- Full failing-test triage; debug-invalid wall-clock performance assertions are
  gated behind `VECTORCORE_TEST_EXTENDED=1`.
- Added a SIMD-provider bounds-safety regression sweep (all ops × sizes 1–64),
  the BE3 architectural audit documents, and `.gitignore` entries for
  test-result artifacts.

## [0.2.1] - 2026-04-11

### Added

#### `NormalizationHint<V>` conforms to `IndexableVector`
- `NormalizationHint<V: IndexableVector>` now conforms to `VectorProtocol`, `IndexableVector`, `Codable`, `Hashable`, and `Collection`
- Enables passing hinted vectors directly to `<V: IndexableVector>` typed insert APIs in VectorAccelerate 0.4.3+ and VectorIndex 0.1.4+
- Downstream cosine distance fast paths (dot product when `isNormalized == true`) are now reachable from consumer code
- All `VectorProtocol` operations forward transparently to the wrapped vector
- Mutation through any `VectorProtocol` API automatically invalidates hints (`isNormalized` resets to `false`, `cachedMagnitude` clears to `nil`)
- Codable round-trip preserves vector data and hint metadata via keyed container
- `Equatable`/`Hashable` include hint state: same vector with different hints are distinct values
- Double-wrapping prevention: `hint.withNormalizationHint()` returns `self` instead of nesting

### Tests
- 13 new tests for `NormalizationHint` conformance: type conformance, storage forwarding, mutation invalidation, required initializers, equality, hashing, Codable round-trip (Vector512Optimized + DynamicVector), insert simulation, arithmetic, double-wrapping prevention, collection iteration

## [0.2.0] - 2026-03-28

> **Documentation correction — 2026-10-04:** Several entries below are retained
> as release history, not as performance claims. The "~1.5-2x throughput
> improvement", "Halves memory bandwidth", "~75ns dot product, ~90ns distance",
> "15-20% faster" and "3-4x faster" figures were not measured and are not
> supported by retained evidence. The "prefetch hints" are discarded pointee
> loads, not prefetch instructions. An `[Int32]` index result is half the size of
> `[Int]` on 64-bit platforms; any bandwidth effect was not measured.

### Added

#### DistanceMetric Batch Protocol Requirement
- `batchDistance(query:candidates:)` promoted from extension to formal protocol requirement on `DistanceMetric`
- Enables witness-table dispatch through `any DistanceMetric` existentials, unblocking polymorphic GPU metric overrides from VectorAccelerate
- Default extension implementation preserved — all existing conformances compile unchanged

#### 4-Way Euclidean SoA Register Blocking
- New `euclid2_blocked_4way` kernel in `BatchKernels_SoA` with 4 independent SIMD4 accumulators
- Prefetch hints for next-block data, matching existing `dot_blocked_4way` pattern
- `euclid2_512/768/1536` automatically select 4-way for candidate sets >= 4, 2-way fallback for smaller
- ~1.5-2x throughput improvement for large candidate batches via better ILP on Apple Silicon

#### Zero-Copy Pointer APIs
- **`TopKSelection.select(k:from:count:ids:)`**: Raw `UnsafePointer<Float>` API returning `[Int32]` indices for GPU/IOSurface interop. Halves memory bandwidth vs `[Int]` for Metal buffer uploads
- **`NormalizeKernels.normalizeUnchecked(_:dimension:)`**: In-place normalization via raw pointer for arbitrary dimensions. Kahan two-pass scaling with 4-way unrolled accumulators. `@inlinable` for cross-module optimization

#### C-Kernel Intrinsics (Experimental)
- **ARM64 NEON**: FP32 dot/L2sq with 4-accumulator `vfmaq_f32`; INT8 dot product with dual-path SDOT (`vdotq_s32`, A12+) and `vmull_s8`/`vpadalq_s16` fallback for older ARM64
- **x86_64 AVX2**: INT8 dot product via `cvtepi8_epi16` + `madd_epi16`, gated behind `__attribute__((target("avx2")))` (no `.unsafeFlags` needed in Package.swift)
- **Dispatch**: `vc_dot_int8` routes to arch-specific implementations with runtime CPU feature detection and scalar fallback
- **Swift wrappers**: `CKernels.dotInt8` + `withSIMD4Int8Pointer` bridge for `ContiguousArray<SIMD4<Int8>>` interop
- **QuantizedKernels integration**: `euclidean_generic_int8` and `accumulate_fused_generic` route through hardware-accelerated `dotInt8` when `VC_USE_C_KERNELS` is enabled
- Enable via: `swift build -Xswiftc -DVC_USE_C_KERNELS`

#### Nested Parallelism Prevention
- `@TaskLocal isInsideParallelRegion` guard on `CPUComputeProvider` prevents O(cores^2) thread explosion from nested `parallelExecute`/`parallelReduce`/`parallelForEach` calls
- All 5 parallel methods check the flag before spawning TaskGroups and propagate it to child tasks via Swift's TaskLocal inheritance
- Transparent to callers — inner operations silently fall back to sequential execution

#### Dim384 Optimized Vector Type
- **`Vector384Optimized`**: New SIMD-accelerated 384-dimensional vector type for MiniLM and Sentence-BERT embeddings
- Full kernel support: dot product, Euclidean distance, cosine distance, normalization
- SoA (Structure-of-Arrays) layout support via `SoA384` typealias
- `Dim384` dimension type with `VectorTypeFactory` integration
- Performance: ~75ns dot product, ~90ns distance computation

#### `normalizedUnchecked()` API
- New `normalizedUnchecked() -> Self` method on all vector types
- Bypasses zero-vector validation for maximum performance in hot paths
- Debug-only assertions catch misuse during development
- Optimized SIMD kernels for Vector384/512/768/1536Optimized
- Estimated 15-20% faster than `normalized()` in hot paths (not verified in this repository)

#### SIMD Manhattan Distance
- Generic `ManhattanDistance` now uses SIMD4 vectorization
- Processes 4 elements per iteration with scalar tail loop
- Works with `DynamicVector` and arbitrary dimensions
- Estimated 3-4x faster than previous scalar implementation (not verified in this repository)

#### Top-K Selection Public API
- **`TopKSelection`** enum for k-nearest neighbor selection
- `select(k:from:)` - O(n log k) heap-based selection from pre-computed distances
- `nearest(k:query:candidates:metric:)` - Generic k-NN with any distance metric
- Optimized variants: `nearestEuclidean384/512/768/1536()`, `nearestCosinePreNormalized512()`, `nearestCosineFused512()`, `nearestDotProduct512()`
- `batchNearest()` for multi-query batch operations
- **`TopKResult`** struct with Collection and Codable conformance
- Adaptive algorithm: heap for k < n/10, sort for larger k

#### VectorIndex Integration Protocols
- **`IndexableVector`** protocol extending `VectorProtocol` with optimization hints
  - `isNormalized: Bool` - indicates pre-normalized vectors
  - `cachedMagnitude: Float?` - cached magnitude value
  - `isApproximatelyNormalized` computed property
- **`NormalizationHint<V>`** wrapper for tracking normalization status
- **`SearchResult<ID>`** - Single search result with id, distance, score
- **`SearchResults<ID>`** - Collection with metadata (candidatesSearched, searchTimeNanos, isExhaustive)
- **`VectorCollection`** protocol for searchable vector collections
  - Default brute-force `search(query:k:metric:)` implementation
  - Convenience methods: `searchEuclidean()`, `searchCosine()`, `searchDotProduct()`
- **`MutableVectorCollection`** protocol with add/remove/update operations
- **`SimpleVectorCollection<V>`** - Reference implementation for testing

### Changed

- `VectorTypeFactory` now returns `Vector384Optimized` for dimension 384
- `optimalDimension(for:)` includes 384 in supported dimensions
- Updated documentation to reflect 384-dimension support

### Tests

- 27 new tests for Top-K Selection
- 24 new tests for Integration Protocols
- 9 new tests for `normalizedUnchecked()`
- 7 new tests for SIMD Manhattan Distance
- All 140 tests in relevant suites pass

---

## [0.1.4] - Previous Release

Initial pre-release version with:
- Vector512/768/1536Optimized types
- SIMD-accelerated kernels
- Distance metrics (Euclidean, Cosine, Manhattan, Chebyshev, Dot Product)
- Batch operations with async support
- Mixed-precision computation support
