# VectorCore Performance Guide

This guide covers choosing vector types, provider and batch configuration, matrix routing, memory use, and measuring
VectorCore with the `vectorcore-bench` executable. It makes no speed claim beyond the dated one-host record in
[Measured Workload Envelope (One Host)](#measured-workload-envelope-one-host), which cites
`Benchmarks/WorkloadEnvelope/RESULTS.md`: every other statement below about speed is marked as not verified in this
repository, and the stored baselines in `Benchmarks/baselines/` are historical, machine-specific records. Measure on
your own hardware and workload (see [Benchmarking](#benchmarking)).

---

## Table of Contents

1. [Choosing the Right Vector Type](#choosing-the-right-vector-type)
2. [Optimized Vectors](#optimized-vectors)
3. [Provider Configuration](#provider-configuration)
4. [Batch Operations](#batch-operations)
5. [GEMM / Matrix Batch-Distance](#gemm--matrix-batch-distance)
6. [Memory Management](#memory-management)
7. [Benchmarking](#benchmarking)
8. [Performance Pitfalls](#performance-pitfalls)
9. [Measured Workload Envelope (One Host)](#measured-workload-envelope-one-host)

---

## Choosing the Right Vector Type

| Dimension | Type | Storage |
|-----------|------|---------|
| 384, 512, 768 or 1536 | `Vector384Optimized`, `Vector512Optimized`, `Vector768Optimized`, `Vector1536Optimized` | `ContiguousArray<SIMD4<Float>>` with dedicated kernels |
| Known at compile time | `Vector<D>` | Generic storage |
| Known only at run time | `DynamicVector` | Dimension checked at run time |

Whether an optimized type is faster than `Vector<D>` or `DynamicVector` for your workload is not verified in this
repository. The `dot` and `distance` suites run a `generic` and an `optimized` variant of each case, so you can time
both on your hardware.

---

## Optimized Vectors

`Vector512Optimized` keeps its 512 lanes as 128 `SIMD4<Float>` elements; its `storage` setter traps on any other
count. `dotProduct` delegates to `DotKernels.dot512`, which keeps four `SIMD4<Float>` accumulators over 16 elements
per loop iteration and adds them as `(acc0 + acc1) + (acc2 + acc3)` before the horizontal sum. The 384, 768 and 1536
types use the same kernel over 96, 192 and 384 elements.

Whether this accumulator layout improves instruction-level parallelism or speed on a given processor is not verified
in this repository.

---

## Provider Configuration

Provider values on `Operations` are task-local bindings. The
[API overview](API_Overview_Map.md#provider-boundaries) lists each binding, the protocol it accepts and its default.
Bind one around the work that should observe it:

```swift
let results = try await Operations.$computeProvider.withValue(CPUComputeProvider.sequential) {
    try await Operations.findNearest(to: query, in: candidates, k: 1)
}
```

`CPUComputeProvider` presets (factory properties in `CPUComputeProvider.swift`):

- `.sequential`: sequential mode.
- `.parallel`: parallel mode.
- `.automatic`: automatic mode.
- `.performance`: parallel mode with `parallelizationThreshold: 100`.
- `.efficiency`: automatic mode with `parallelizationThreshold: 5000`.

The preset names describe configurations; their effect on speed or energy is not verified in this repository.

### `BatchKernelProvider`

`Operations.findNearest` and `Operations.findNearestBatch` delegate to an installed `BatchKernelProvider` before the
built-in CPU routes (`findNearestDelegates`, `findNearestBatchDelegates` and `findNearestBatchUsesOverride` in
`BatchKernelProviderTests`). VectorCore defines the protocol; an external provider,
such as the one in VectorAccelerate, owns its implementation, capability checks, allocation behavior and
synchronization. Whether a GPU provider is faster than the CPU routes is not verified in this repository.

```swift
let results = try await Operations.$computeProvider.withValue(yourGPUProvider) {
    try await Operations.findNearest(to: query, in: database, k: 100)
}
```

---

## Batch Operations

`BatchOperations.Configuration` has six fields: `parallelThreshold` (default `1000`), `oversubscription` (default
`2.0`), `minimumChunkSize` (default `256`), `defaultBatchSize` (default `1024`), `enableMatrixRouting` (default `false`)
and `matrixRoutingMinN` (default `256`). Update them with a throwing transaction; an invalid value is rejected and every
old field is kept (`batchRollbackPreservesEveryFieldAndErrorIdentity` in `ConfigurationTransactionTests`):

```swift
try await BatchOperations.updateConfiguration { config in
    config.parallelThreshold = 500
    config.minimumChunkSize = 128
}
```

`BatchOperations.findNearest`, `map`, `filter`, `process` and `statistics` take their parallel route when the input
count is at least `parallelThreshold` (the `linear` scenario of `configurationScenarioInBoundedChild` in
`BatchConfigurationContractTests`).

Whether the parallel route is faster at a given count is not verified in this repository.

---

## GEMM / Matrix Batch-Distance

`MatrixDistance` computes a query-by-candidate distance matrix. Euclidean output is the squared distance through
`‖x‖² + ‖y‖² − 2⟨x, y⟩`, with one `cblas_sgemm` call for the cross term; cosine output is `1 − cos`. Results are
row-major: `out[i * candidates.count + j]` is the distance from `queries[i]` to `candidates[j]`.

```swift
let flat: [Float] = MatrixDistance.euclideanSquaredMatrix(queries: queries, candidates: candidates)

var out = [Float](repeating: 0, count: queries.count * candidates.count)
MatrixDistance.euclideanSquaredMatrix(queries: queries, candidates: candidates, into: &out)

let prepared = MatrixDistance.prepare(candidates, normalized: false)   // true for cosine
MatrixDistance.euclideanSquaredMatrix(queries: queries, prepared: prepared, into: &out)
```

Which hardware unit executes `cblas_sgemm`, and whether the matrix route is faster than per-pair kernels at a given
size, are not verified in this repository.

- **Routing.** `BatchOperations.pairwiseDistances` takes the matrix route by its configured gate, which
  `BatchOperations.updateConfiguration` changes; `Operations.findNearestBatch` takes its tiled Gram route only for
  `cpuStrategy: .gram(...)`, never by size. The
  [search and matrix routing table](API_Overview_Map.md#search-and-matrix-routing) gives the current conditions.
- **Precision.** The matrix route and the per-pair kernels can return different distances and rankings, especially
  near zero, with no universal relative-error guarantee. See
  [Numerical Behavior](Numerical_Behavior.md#matrix-distances-shape-memory-and-precision).
- **Memory.** `into:` and prepared candidates reuse particular storage, with no allocation-free guarantee.

The `matrix` suite of `vectorcore-bench` times `MatrixDistance.euclideanSquaredMatrix` on N×N inputs, and the
`batch` suite times per-candidate batch routes. No A/B pair joins a `matrix` case and a `batch` case: pairs form only
from declared boundary membership, and the two families the `batch` suite declares pair `batch` cases with each other.
Time both suites on your hardware to choose `matrixRoutingMinN`.

---

## Memory Management

Allocation behavior depends on the type and the operation, and VectorCore makes no package-wide allocation-free
guarantee. [Memory Alignment](Memory_Alignment.md) gives the alignment, ownership and pool contracts.

`AlignedMemory.allocateAligned(count:alignment:)` forwards to the generic `allocateAligned(type:count:alignment:)`, whose
storage must be released with `AlignedMemory.deallocate`. `allocatorValuesAndCounts` in `UnsafeMemoryContractTests`
checks that generic overload at alignment 64 only; other alignments rest on the `posix_memalign` contract (not verified
in this repository).

Reuse one caller-owned output buffer across queries instead of building a result array for each query:

```swift
var distances = [Float](repeating: 0, count: candidates.count)
for query in queries {
    distances.withUnsafeMutableBufferPointer { buffer in
        for (i, candidate) in candidates.enumerated() {
            buffer[i] = query.euclideanDistance(to: candidate)
        }
    }
}
```

Whether the distance calls themselves allocate is not established here, and the speed effect of reusing the buffer
is not verified in this repository.

---

## Benchmarking

### Using vectorcore-bench

```bash
# Build the benchmark executable
swift build -c release --product vectorcore-bench

# The batch suite with the short profile (the default suites are dot, distance, normalize and batch)
.build/release/vectorcore-bench --suites batch --profile short

# One suite, one width, fixed repeats
.build/release/vectorcore-bench --suites distance --dims 512 --repeats 50

# Machine-readable output is written to --out
.build/release/vectorcore-bench --suites batch --profile short --format csv --out results.csv
```

Suites are selected with `--suites` (a comma-separated list); positional suite names are rejected. Without `--out`,
JSON and CSV reports go to a default path under `.bench/runs/`. Every case carries a measurement boundary that
records what one timed invocation does; see
[Benchmark measurement boundaries](Benchmark_Measurement_Boundaries.md).

### Benchmark Suites

| Suite | What it times | Default |
|-------|---------------|---------|
| `dot` | `dotProduct` on a vector pair, generic and optimized types | yes |
| `distance` | Seven distance metrics on a vector pair, generic and optimized types | yes |
| `normalize` | `normalized()` on a constant vector and on a zero vector (failure outcome) | yes |
| `batch` | Per-candidate batch distance routes, by provider (SoA and FP16 routes with `--prefer-soa` and `--use-mixed-precision`) | yes |
| `matrix` | `MatrixDistance.euclideanSquaredMatrix` on N×N inputs | no |
| `memory` | Aligned allocation, copy, fill and pool acquisition | no |
| `noop` | A fixed scalar loop (`noop.scalar-add`), a reference for harness behavior | no |
| `boundary` | Opt-in companions: reused-output, setup-inclusive and consumer-only records ([details](Benchmark_Measurement_Boundaries.md#the-opt-in-boundary-suite)) | no; never auto-paired |

```bash
.build/release/vectorcore-bench --suites matrix
.build/release/vectorcore-bench --suites boundary --dims 512 --batch-ns 257,4096 --repeats 3 --samples 9 --format json --out boundary.json
```

### Performance Regression Detection

```yaml
- name: Run performance benchmarks
  run: .build/release/vectorcore-bench --suites batch --profile short --samples 5 --format json --out bench.json

- name: Check for regressions
  run: swift Scripts/compare_benchmarks.swift --baseline baseline.json --current bench.json --metric median --per-unit --threshold 10.0
```

`Scripts/compare_benchmarks.swift` exits 1 when the chosen statistic of any matched case rises by more than
`--threshold` percent over the baseline, or when a case is missing from either report (unless `--allow-missing`). It
exits 2 when the comparison is invalid: either report records an execution or correctness failure in its `failures`
array (never waivable), incompatible run metadata (unless `--allow-incompatible`), a present but different boundary
(never waivable), zero matched cases, a missing requested statistic, or malformed input. A report has a median, p90 and
mean only when it was run with `--samples` of 2 or more, and the comparator never substitutes another statistic, so
the benchmark step above passes `--samples 5`. These exit codes are read from the script's `main()` and its report
reader. The benchmark-contracts CI job (`Scripts/ci/tests/test_benchmark_contracts.py`) runs the comparator only for
the baselines README invocations (exit 0 against the stored files; exit 2 without `--allow-incompatible` and for a
missing statistic) and for its build-configuration warnings; the other exit codes are not verified by a test. See
[Comparing reports across runs](Benchmark_Measurement_Boundaries.md#comparing-reports-across-runs) and
`Benchmarks/baselines/README.md`.

**Correctness gate.** A report records a correctness failure, which the comparison above rejects with exit 2, when a
case's comparison with its Double reference exceeds the case's threshold. The Euclidean and squared-Euclidean `batch`
routes, their `boundary` companions and the `matrix` routes are held to thresholds derived from each route's worst-case
rounding bound for the case's dimension and candidate count: `γ_k = k·u/(1 − k·u)` with `u = 2^-24` and `k` counted from
the route's summation (`CorrectnessPolicy` in `Benchmarks/VectorCoreBench/Utils/CorrectnessPolicy.swift`;
`testDerivedThresholdsFollowTheRoundingCounts` and `testGenericSquaredEuclideanBoundHoldsAndStillCatchesDroppedWork`
in `BenchmarkCorrectnessPolicyTests`). The `matrix` routes are gated on absolute error, because their diagonal
references are zero (`testMatrixZeroReferenceIsGatedOnAbsoluteError`), and a relative error never divides by a
reference smaller than 1 (`testRelativeErrorNeverDividesByReferenceBelowFloor`). The bounds for the `batch.euclidean`
`generic` routes (`vDSP_distancesq`) and the `matrix` routes (`vDSP_svesq`, `cblas_sgemm`) assume that Accelerate
evaluates each inner product with binary32 operations in some order, not with a fast matrix-multiplication algorithm;
that assumption is not verified. The thresholds are worst-case bounds, so a defect that moves a result by less than its
threshold, such as one small term dropped from one distance, passes them. The other validated routes keep fixed
relative thresholds (FP32 `1e-6`, FP16 `2e-3`, FP16 cosine `3e-3`), which are not derived from their summations and are
not verified to bound their rounding error.

`noop.scalar-add` and the `mem.alloc`, `mem.pool` and `mem.fill` cases declare that they have no VectorCore output to
validate, so they record no failure (`testValidationRequirementIsDeclaredPerCase`). The `dot`, `normalize` and
`mem.copy` cases have an output but no validation yet, so each records "Missing validation": a report that includes the
`dot` or `normalize` suite (both in the default suite list) or the `memory` suite cannot be compared. The T11 runs of the
`batch` command above on 2026-10-04 recorded no correctness failure, but on that loaded host the comparison still exited
1 from timing noise alone: four back-to-back pairs of runs of the same binary each found 2 to 7 `parallel` or
`automatic` cases more than 10% slower (an observation of one host, outside this repository; not verified by a test
here). Run regression comparisons on a quiet host.

---

## Performance Pitfalls

Each pitfall below is advice, not a measured result: its effect is not verified in this repository. Compare the two
forms on your workload with `vectorcore-bench` or Instruments.

### Pitfall 1: Generic vectors for the supported widths

```swift
let generic = try Vector<Dim512>(array)
let optimized = try Vector512Optimized(array)
```

`Vector512Optimized` uses the dedicated kernels described above; any speed difference is not verified in this
repository.

### Pitfall 2: Building arrays in hot loops

A `map` or other array-returning call inside a per-query loop builds a new array on every iteration. Reuse one output
buffer, as in [Memory Management](#memory-management); the speed effect is not verified in this repository.

### Pitfall 3: Expecting `findNearestBatch` to search differently

```swift
let results = try await Operations.findNearestBatch(queries: queries, in: database, k: 10)
```

`findNearestBatch` runs `findNearest`'s routes once per query by default and its tiled Gram route only for
`cpuStrategy: .gram(...)` (see the [routing table](API_Overview_Map.md#search-and-matrix-routing)); whether either is
faster than a loop over `findNearest` is not verified in this repository. Under the default
`CPUComputeProvider.automatic` the per-query searches run on one core unless the batch has at least 50,000 queries,
and the `.parallel` and `.performance` presets submit one worker for up to 256 queries, because `parallelExecute`
chunks by at least 256 items; the scan inside each worker runs inline as a nested region. To use more cores on a
smaller batch, split it across tasks yourself or install a provider with a smaller chunk minimum.

### Pitfall 4: Random access over large candidate sets

Visiting candidates in storage order rather than through random indices changes the memory access pattern; the
effect is not verified in this repository.

### Pitfall 5: Normalizing the same vector repeatedly

Normalize vectors once (for example with `Operations.normalize(_:)`) and reuse the results for cosine comparisons
instead of normalizing inside the comparison loop; the saving is not verified in this repository.

---

## Performance Checklist

- [ ] The vector type matches the dimension (see the table above).
- [ ] Hot loops reuse output buffers.
- [ ] Batch and matrix routing settings were chosen with `vectorcore-bench` on the target hardware.
- [ ] Cosine inputs are normalized once when the same vectors are compared repeatedly.
- [ ] Changes were measured in Release with `vectorcore-bench` and compared with `Scripts/compare_benchmarks.swift`.
- [ ] Hot spots were profiled with Instruments.

---

## Advanced: Custom Providers

A `BatchKernelProvider` conformer must honor the kernel contract:

- `batchDistance` returns one distance per candidate, in candidate order.
- `findNearest` returns up to `k` `(index, distance)` pairs sorted ascending by distance, indexing into `candidates`.
- `findNearestBatch` has a default that calls `findNearest` once per query; a conformer may override it with one
  batched call (`findNearestBatchUsesOverride`).

Whether a batched GPU dispatch is faster than per-query calls is not verified in this repository.

### Page-Aligned SoA Batches

`SoA.build(from:pageAligned: true)` allocates storage with a page-aligned base and a page-multiple length
(`pageAlignedExposesBytes` in `SoAPageAlignTests`). That such storage can back a `MTLDevice.makeBuffer(bytesNoCopy:)`
call is Metal's documented requirement, cited here and not verified in this repository; no test calls Metal.
[SoA_Layout_Contract.md](SoA_Layout_Contract.md) gives the byte-count contract that `SoALayout` and
`makeBuffer(bytesNoCopy:)` must agree on. Whether the no-copy path avoids a staging copy
or changes throughput on a given device is not verified in this repository.

```swift
let soa = SoA.build(from: candidates, pageAligned: true)
```

---

## Profiling with Instruments

```bash
# Release build with debug symbols
swift build -c release -Xswiftc -g

# Time Profiler and Allocations
xcrun xctrace record --template 'Time Profiler' --launch -- .build/release/YourApp
xcrun xctrace record --template 'Allocations' --launch -- .build/release/YourApp
```

Look for the functions that dominate the profile and for allocations inside hot loops. Platform differences between
Apple Silicon and Intel are not verified in this repository; measure on each target.

---

## Measured Workload Envelope (One Host)

This section is a dated one-host record of `PCAModel.fit` with batch `transform` (exact and randomized),
`KNNGraph.bruteForce` and the vector entry of `Operations.umap`, sized for text-chunk embedding corpora (N from 50 to
100,000; D of 384, 768 or 1536). The full record, with method, ranges, analytic payloads and reproducibility checks,
is `Benchmarks/WorkloadEnvelope/RESULTS.md` (2026-10-16). It reports what one host observed, not a promise for other
hardware, data or parameters.

**Not measured.** Inputs outside this grid are not measured: N above 100,000, D above 1536, other K, k, epochs or
configurations, other linear-algebra providers and other vector types. No size limit applies to them beyond the
documented representability checks (LP64 BLAS dimensions and Int element and byte counts, which throw
`VectorError.invalidDimension`; see the `- Throws:` lists of these APIs and [Linear Algebra and
Projection](Linear_Algebra_and_Projection.md)). Passing those checks does not promise that allocation succeeds, and
their time and memory are not promised. These calls are synchronous and cannot be cancelled.

- **Host.** Apple M3 Max (12 performance and 4 efficiency cores), 48 GiB memory, macOS 27.0.1; Xcode 27.0 and Apple
  Swift 6.4; Release build; default linear-algebra provider (LAPACK through Accelerate). High Power Mode (`pmset`
  `powermode 2`) on AC power, checked before and after every timing window; every window started with a 1-minute load
  average below 8. The system installer daemon (`installd`) was often among the busiest processes (up to 250% CPU),
  which can add noise to the times.
- **Data.** Seeded synthetic vectors: a mixture of 50 unit-norm Gaussian centers with Gaussian per-point noise, each
  vector L2-normalized and stored as `DynamicVector`. Real text-chunk embeddings can form a different neighbor graph,
  which changes UMAP layout time.
- **Calls.** PCA uses K = 50 (K = 49 at N = 50, because K must be below N when centered); exact is
  `PCAConfig(oversampling: Int.max)` and randomized is `PCAConfig()`. The PCA time covers `fit` and then `transform`
  of all N vectors, the two calls `Operations.pca` makes. kNN uses k = 15. UMAP uses two output dimensions and the
  default `UMAPConfig` (a brute-force 15-neighbor graph; 500 epochs up to N = 10,000 and 200 above) with PCA or random
  initialization.
- **Columns.** Time is the median wall time of the call over 5, 3 or 2 fresh processes (N up to 5,000, N = 25,000, N =
  100,000). Peak RSS is `ru_maxrss` read right after the call; it includes the process baseline (about 7 MiB) and the
  caller-owned input vectors. The sampled malloc peak is a lower bound polled every 1 ms from `malloc_zone_statistics`
  (`max_size_in_use` reads 0 on this host).

At N = 100,000 and D = 768 (`Benchmarks/WorkloadEnvelope/RESULTS.md`): exact PCA took 2.14 s with 1585 MiB peak RSS,
randomized PCA 0.76 s with 996 MiB, `KNNGraph.bruteForce` 44.4 s with 740 MiB, and UMAP 85.5 s with 1262 MiB (PCA
initialization) or 89.9 s with 958 MiB (random initialization).

kNN time grew faster than N² over the two largest measured intervals on this host
(`Benchmarks/WorkloadEnvelope/RESULTS.md`): from N = 5,000 to 25,000 (25 times the pairs) it rose 32 to 38 times, and
from 25,000 to 100,000 (16 times the pairs) 18 to 19 times; from 500 to 5,000 (100 times the pairs) it rose only 29 to
36 times, which is less than quadratic. Do not extrapolate beyond the measured points. At N = 100,000 about half of
the UMAP vector-entry time is the brute-force graph, judged from the separate kNN case; a graph supplied by an ANN
index skips that builder, but that route is not measured here.

| Algorithm | N | D | Median time (s) | Peak RSS (MiB) | Sampled malloc peak (MiB, lower bound) |
|---|---:|---:|---:|---:|---:|
| PCA exact | 50 | 384 | 0.001 | 7 | 0 |
| PCA exact | 50 | 768 | 0.001 | 8 | 0 |
| PCA exact | 50 | 1536 | 0.003 | 8 | 2 |
| PCA exact | 500 | 384 | 0.013 | 16 | 7 |
| PCA exact | 500 | 768 | 0.025 | 21 | 13 |
| PCA exact | 500 | 1536 | 0.041 | 29 | 20 |
| PCA exact | 5,000 | 384 | 0.057 | 52 | 42 |
| PCA exact | 5,000 | 768 | 0.150 | 98 | 88 |
| PCA exact | 5,000 | 1536 | 0.470 | 216 | 203 |
| PCA exact | 25,000 | 384 | 0.243 | 212 | 191 |
| PCA exact | 25,000 | 768 | 0.542 | 409 | 387 |
| PCA exact | 25,000 | 1536 | 1.393 | 821 | 799 |
| PCA exact | 100,000 | 384 | 0.954 | 822 | 753 |
| PCA exact | 100,000 | 768 | 2.140 | 1585 | 1500 |
| PCA exact | 100,000 | 1536 | 5.352 | 3080 | 3005 |
| PCA randomized | 50 | 384 | 0.001 | 7 | 0 |
| PCA randomized | 50 | 768 | 0.001 | 8 | 0 |
| PCA randomized | 50 | 1536 | 0.003 | 8 | 2 |
| PCA randomized | 500 | 384 | 0.004 | 11 | 3 |
| PCA randomized | 500 | 768 | 0.006 | 13 | 5 |
| PCA randomized | 500 | 1536 | 0.012 | 18 | 10 |
| PCA randomized | 5,000 | 384 | 0.024 | 39 | 27 |
| PCA randomized | 5,000 | 768 | 0.038 | 59 | 50 |
| PCA randomized | 5,000 | 1536 | 0.070 | 104 | 94 |
| PCA randomized | 25,000 | 384 | 0.119 | 142 | 125 |
| PCA randomized | 25,000 | 768 | 0.186 | 255 | 236 |
| PCA randomized | 25,000 | 1536 | 0.343 | 473 | 456 |
| PCA randomized | 100,000 | 384 | 0.480 | 547 | 503 |
| PCA randomized | 100,000 | 768 | 0.761 | 996 | 942 |
| PCA randomized | 100,000 | 1536 | 1.333 | 1873 | 1827 |
| kNN bruteForce | 50 | 384 | 0.000 | 7 | 0 |
| kNN bruteForce | 50 | 768 | 0.000 | 7 | 0 |
| kNN bruteForce | 50 | 1536 | 0.000 | 8 | 1 |
| kNN bruteForce | 500 | 384 | 0.002 | 9 | 3 |
| kNN bruteForce | 500 | 768 | 0.002 | 11 | 4 |
| kNN bruteForce | 500 | 1536 | 0.003 | 15 | 8 |
| kNN bruteForce | 5,000 | 384 | 0.061 | 29 | 21 |
| kNN bruteForce | 5,000 | 768 | 0.069 | 45 | 36 |
| kNN bruteForce | 5,000 | 1536 | 0.090 | 79 | 70 |
| kNN bruteForce | 25,000 | 384 | 2.282 | 115 | 105 |
| kNN bruteForce | 25,000 | 768 | 2.452 | 193 | 180 |
| kNN bruteForce | 25,000 | 1536 | 2.872 | 342 | 328 |
| kNN bruteForce | 100,000 | 384 | 42.1 | 434 | 416 |
| kNN bruteForce | 100,000 | 768 | 44.4 | 740 | 710 |
| kNN bruteForce | 100,000 | 1536 | 54.8 | 1321 | 1299 |
| UMAP, PCA init | 50 | 384 | 0.041 | 8 | 0 |
| UMAP, PCA init | 50 | 768 | 0.043 | 8 | 0 |
| UMAP, PCA init | 50 | 1536 | 0.040 | 8 | 1 |
| UMAP, PCA init | 500 | 384 | 0.361 | 11 | 2 |
| UMAP, PCA init | 500 | 768 | 0.363 | 14 | 5 |
| UMAP, PCA init | 500 | 1536 | 0.362 | 21 | 9 |
| UMAP, PCA init | 5,000 | 384 | 4.392 | 46 | 25 |
| UMAP, PCA init | 5,000 | 768 | 4.434 | 69 | 51 |
| UMAP, PCA init | 5,000 | 1536 | 4.470 | 117 | 90 |
| UMAP, PCA init | 25,000 | 384 | 11.5 | 205 | 119 |
| UMAP, PCA init | 25,000 | 768 | 12.0 | 317 | 229 |
| UMAP, PCA init | 25,000 | 1536 | 13.0 | 538 | 449 |
| UMAP, PCA init | 100,000 | 384 | 79.3 | 810 | 473 |
| UMAP, PCA init | 100,000 | 768 | 85.5 | 1262 | 912 |
| UMAP, PCA init | 100,000 | 1536 | 94.5 | 2133 | 1791 |
| UMAP, random init | 5,000 | 768 | 4.397 | 54 | 36 |
| UMAP, random init | 100,000 | 768 | 89.9 | 958 | 710 |

---

**Last Updated**: 2026-10-16 (adds the measured workload envelope; benchmark CLI and suites as of the unreleased
changes after v0.3.3)
