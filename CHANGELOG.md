# Changelog

All notable changes to VectorCore will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
  per lane (prefetcher-optimal) instead of L interleaved strided streams.
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
- Estimated 15-20% faster than `normalized()` in hot paths

#### SIMD Manhattan Distance
- Generic `ManhattanDistance` now uses SIMD4 vectorization
- Processes 4 elements per iteration with scalar tail loop
- Works with `DynamicVector` and arbitrary dimensions
- Estimated 3-4x faster than previous scalar implementation

#### Top-K Selection Public API
- **`TopKSelection`** enum with efficient k-nearest neighbor selection
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
