# Nearest Neighbor Recipes (How‑to)

This guide shows practical patterns for k‑nearest neighbor (kNN) search with VectorCore using the public APIs in the repository:
- `Operations` (async, provider‑aware)
- `DistanceMetrics` (Euclidean, Cosine, Dot, etc.)
- Vectors: `Vector<DimN>` and optimized `Vector{512,768,1536}Optimized`

It covers single and batch queries, metric selection, normalization, performance tips, and troubleshooting.

Note: Code below uses Swift 6 concurrency (`async/await`) and VectorCore’s public types.

---

## Quick kNN (single query)

Option A — Optimized vector type (dedicated search routes for 512/768/1536)

```swift
import VectorCore

// Build optimized 512‑dimensional vectors
let query = try Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
let database: [Vector512Optimized] = (0..<10_000).map { _ in
    try! Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
}

// k‑NN with default metric (Euclidean)
let k = 10
let results = try await Operations.findNearest(to: query, in: database, k: k)

// Each result has the index into `database` and a distance (Float)
for r in results {
    print("idx=\(r.index) dist=\(r.distance)")
}
```

Option B — Compile‑time dimensioned vectors

```swift
import VectorCore

// Fixed dimension with compile‑time safety
let query = try Vector<Dim512>((0..<512).map { _ in Float.random(in: -1...1) })
let database: [Vector<Dim512>] = (0..<10_000).map { _ in
    try! Vector<Dim512>((0..<512).map { _ in Float.random(in: -1...1) })
}

let results = try await Operations.findNearest(to: query, in: database, k: 10)
```

Notes
- For 512/768/1536, the `Vector{512,768,1536}Optimized` types select dedicated routes in `Operations.findNearest` (see its route list): per-range Top-K heaps for Euclidean on all three and for cosine and dot product on 512; fused scoring of every candidate for cosine on 768 and 1536 (`P01OperationsSelectionTests.cosine768And1536StayOnTheFusedFullScoreRoute`).
- Distances are metric distances. For `DotProductDistance`, the distance is the negative dot product (more on this below).

---

## Batch kNN (multiple queries)

```swift
import VectorCore

let queries: [Vector512Optimized] = (0..<128).map { _ in
    try! Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
}
let database: [Vector512Optimized] = (0..<50_000).map { _ in
    try! Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
}

let k = 20
let resultsPerQuery: [[NearestNeighborResult]] = try await Operations.findNearestBatch(
    queries: queries,
    in: database,
    k: k,
    metric: EuclideanDistance()
)

// Interpret Top‑K per query
for (qi, knn) in resultsPerQuery.enumerated() {
    print("query #\(qi) → best idx=\(knn.first?.index ?? -1), dist=\(knn.first?.distance ?? .nan)")
}
```

Interpretation
- Each `NearestNeighborResult` contains `index` (into the `database` array) and `distance` (metric‑specific).
- Results are sorted ascending; computed NaN distances follow every numeric distance, and exact ties (including +0/−0 and NaN pairs) keep the smaller original index (`genericOperationsSelectsNumericScoresBeforeNaNs`, `optimizedParallelFiniteTiesUseOriginalIndices` in `TopKNaNWrapperTests`). A query returns `min(k, database.count)` results. An installed `BatchKernelProvider` supplies its own ordering.
- If using `DotProductDistance`, the “distance” is negative similarity; smaller (more negative) means larger dot product.

Metric calls and cost
- `Operations.findNearest` checks `k > 0`, a nonempty database, the query dimension and every row before any provider or metric call, and throws otherwise (`P01OperationsSelectionTests.lateMalformedRowsAreRejectedBeforeAnyWork`).
- On its generic small-`k` route (`k' = min(k, n) < n / 10`, and no dedicated optimized route applies) the metric runs exactly once per candidate on a successful return, each range heap holds at most `min(k', range size)` entries, there is no AutoTuning calibration, and the order of calls across ranges is unspecified (`P01OperationsRedTests.operationsColdKeySmallKEvaluatesEachCandidateOnce`).
- Its full-score route (every case without a dedicated route or the small-`k` stream) keeps all `n` scores, and the first use of a (dimension, kernel) key may call the metric on up to 256 candidates to calibrate (`P01OperationsRedTests.operationsColdKeyLargeKKeepsCalibrationControl`).
- `VectorMath.nearestNeighbors` is synchronous: `k <= 0` or empty input returns `[]` without calling the metric (`VectorMathNearestContractTests.zeroKEvaluatesNoMetric`); otherwise it makes exactly `n` metric calls in index order (`VectorMathNearestContractTests.positiveKCallsEachCandidateOnceInIndexOrder`), with O(k') working arrays for small `k` and O(n) arrays plus an O(n log n) sort otherwise.
- `findNearestBatch` runs `findNearest`'s routes for each query by default (`cpuStrategy: .perQuery`), so each row equals `findNearest` for its query (`P02DefaultRouteRedTests.defaultRouteEqualsSingleQuery`). Its explicit `.gram(...)` route scores one query tile against one candidate tile at a time, so each score block holds at most `queryTileRows × candidateTileRows` entries (each clamped to the query or candidate count); one block is the whole queries × candidates matrix when the tile rows cover both sets, for example with `Int.max` (`P02GramRouteTests.scratchMatchesThePlan`). It rejects, before any work, a plan whose logical temporary payload exceeds the caller's limit (`P02GramRouteTests.limitBoundaryAndClampedK`), and it still returns `queries.count` rows of `min(k, n)` entries.

---

## Choosing Metrics and Normalization

Common choices
- EuclideanDistance (L2): the default metric; it does not require normalized inputs.
- CosineDistance (1 − cosine similarity): normalize vectors first if you expect cosine geometry.
- ManhattanDistance (L1), ChebyshevDistance (L∞): alternatives for specific needs.
- DotProductDistance: returns negative dot product as a distance — useful for maximum dot similarity.

Switching metrics

```swift
// Cosine distance (normalize recommended)
let cosine = CosineDistance()
let knnCos = try await Operations.findNearest(to: query, in: database, k: 10, metric: cosine)

// Manhattan distance
let manhattan = ManhattanDistance()
let knnL1 = try await Operations.findNearest(to: query, in: database, k: 10, metric: manhattan)

// Dot product distance (negative dot)
let dot = DotProductDistance()
let knnDot = try await Operations.findNearest(to: query, in: database, k: 10, metric: dot)
```

Normalization patterns

```swift
// Generic vectors (VectorProtocol) return a Result for normalized()
let maybeNormalized = try? query.normalized().get()
let normalizedQuery = maybeNormalized ?? query

// Optimized vectors have the same API pattern
let normalizedOpt = try? (try Vector512Optimized.randomUnit()).normalized().get()
```

Cosine edge cases
- Default `CosineDistance` returns 1.0 if either vector is zero magnitude and neither has a NaN or infinite coordinate (includes “both zero” case) — i.e., maximum distance.
- The optimized fused and FP16 cosine kernels also return 1.0 when both vectors are zero.
- A NaN or infinite coordinate in either vector gives a NaN cosine distance on every path, so search results rank that row after every numeric distance (T15CosineSelectionTests).
- Recommendation: If using cosine, normalize vectors; guard zero vectors where possible.

---

## Performance Tips

Prefer optimized vectors when possible
- `Vector512Optimized`, `Vector768Optimized`, `Vector1536Optimized` use `SIMD4<Float>` storage and unrolled loops.
- `Operations.findNearest` selects these routes for them (see its documentation):
  - Euclidean: per-range Top-K heaps on 512/768/1536, selected on squared distances.
  - Cosine: a fused per-range Top-K route on 512; on 768/1536 the fused range helpers score every candidate before selection.
  - Dot product: a per-range Top-K route on 512 (negated similarities returned).

Batch sizes and auto‑parallelization
- `Operations.findNearest` schedules through `Operations.computeProvider`; the default `CPUComputeProvider.automatic` runs a scan as parallel ranges only when the candidate count is at least its `parallelizationThreshold` (50,000 by default).
- `BatchOperations` linear operations, including `findNearest`, take their parallel route when the input count is at least `Configuration.parallelThreshold` (1,000 by default); `pairwiseDistances` uses bounded row workers at N ≥ 100, independent of that setting.

GEMM auto‑routing (0.3.0)
- For large inputs, `BatchOperations.pairwiseDistances` automatically routes through a CPU GEMM path (`MatrixDistance` via Accelerate `cblas_sgemm`; which hardware unit Accelerate uses is not specified by this repository) once the input clears its configured gate. `Operations.findNearestBatch` takes a GEMM route only when called with `cpuStrategy: .gram(...)`.
- `BatchOperations.pairwiseDistances` is controlled by `BatchOperations.Configuration.enableMatrixRouting` (default `true`) and `matrixRoutingMinN` (default `256`); its diagonal is re‑scored with the metric's own `distance`.
- `Operations.findNearestBatch` reads neither setting and has no size gate: its default never takes the GEMM route, and the snippet below does not affect it (`P02GramRouteTests.matrixRoutingFlagDoesNotAffectBatchSearch`).
- The GEMM path is a different arithmetic from the per‑pair kernels and the `DistanceMetric` witness, so its scores are not guaranteed to be bit‑identical to them, and near zero they can differ by more than any fixed relative tolerance (one host's T14 probe, not pinned by a test: GEMM returned 0 where the per‑pair Euclidean distance was about 0.02). See [Numerical Behavior](Numerical_Behavior.md#matrix-distances-shape-memory-and-precision).
- Tune or disable it:

```swift
await BatchOperations.updateConfiguration { config in
    config.enableMatrixRouting = false
}
```

- Reuse recipe: for a fixed candidate database queried by many query batches, pack the database once and reuse it across calls:

```swift
// normalized: true for cosine geometry
let prepared = MatrixDistance.prepare(candidates, normalized: false)
MatrixDistance.euclideanSquaredMatrix(queries: queries, prepared: prepared, into: &out)
```

Provider overrides (advanced)

```swift
// Force array-SIMD provider for this scope
await Operations.$simdProvider.withValue(DefaultArraySIMDProvider()) {
    let _ = try await Operations.findNearest(to: query, in: database, k: 10)
}

// Hint parallel CPU execution
await Operations.$computeProvider.withValue(CPUComputeProvider.parallel) {
    let _ = try await Operations.distanceMatrix(between: [query], and: database)
}
```

Scaling up to the GPU (0.3.0)
- Installing a `BatchKernelProvider` (e.g. VectorAccelerate's Metal provider) makes `Operations.findNearest` and `findNearestBatch` delegate to it before any built-in CPU route (and before `findNearestBatch` reads its `cpuStrategy`), with the original `k` and metric; its results, ordering and errors are returned unchanged, and where the work runs is the provider's choice (`P01OperationsRouteTests.providerPrecedence`).

```swift
await Operations.$computeProvider.withValue(yourGPUProvider) {
    let _ = try await Operations.findNearestBatch(queries: queries, in: database, k: 10)
}
```

---

## Troubleshooting

Dimension mismatch
- All vectors in a set must share the same dimension; queries must match database dimension.
- Violations throw `VectorError.dimensionMismatch` from `Operations`.

Zero vectors and cosine
- `CosineDistance` returns 1.0 if either vector has zero magnitude and neither has a NaN or infinite coordinate (NaN otherwise).
- To avoid surprises, normalize inputs or filter out zero vectors before cosine distance.

Dot product distances
- `DotProductDistance` reports “distance = −dot”; distances are negative, so the “closest” item has the most negative (largest magnitude) dot.
- If you need similarity scores, compute the dot directly or negate the distance returned.

Reproducible ties (0.3.0)
- `TopKSelection` takes a `TieBreaker` (default `.smallerIndex`; `.insertionOrder` is equivalent for array and pointer scans), and with either, equal-distance results are reproducible run-to-run; `.smallerValue` leaves identities and order within equal-value or NaN groups unspecified (`TopKNaNContractTests.explicitTiePoliciesRespectTheirScope`). `Operations.findNearest`, `findNearestBatch`, `BatchOperations.findNearest` and `VectorMath.nearestNeighbors` always break exact ties by the smaller original index.

Mixed vector types in an array
- `Operations.findNearest` takes `[V]` for one concrete `V`; the dedicated routes above are selected from that type (`[Vector512Optimized]`, not `[Vector<Dim512>]`).

---

## Full Example (Putting It Together)

```swift
import VectorCore

// 1) Prepare database with optimized vectors (selects the dedicated 512 routes)
let database: [Vector512Optimized] = (0..<100_000).map { _ in
    try! Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
}

// 2) Single query, cosine knn with normalization
var q = try Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
if case .success(let nq) = q.normalized() { q = nq }
let knn = try await Operations.findNearest(to: q, in: database, k: 10, metric: CosineDistance())

// 3) Batch queries, Euclidean knn
let queries: [Vector512Optimized] = (0..<256).map { _ in
    try! Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
}
let batchKnn = try await Operations.findNearestBatch(queries: queries, in: database, k: 20)

// 4) Interpret Top‑K
for (qi, results) in batchKnn.enumerated() {
    let best = results.first
    print("q#\(qi) → best idx=\(best?.index ?? -1), dist=\(best?.distance ?? .nan)")
}
```

---

## Summary
- Use `Operations.findNearest` for single queries, `findNearestBatch` for multiple queries.
- Prefer optimized vectors (512/768/1536) where applicable.
- Choose metrics based on geometry; normalize for cosine.
- Parallel scheduling follows the provider and thresholds above; you can override providers in advanced scenarios.
- Handle dimension mismatches and zero vectors proactively.
