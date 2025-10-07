# Nearest Neighbor Recipes (How‑to)

This guide shows practical patterns for k‑nearest neighbor (kNN) search with VectorCore using the public APIs in the repository:
- `Operations` (async, provider‑aware)
- `DistanceMetrics` (Euclidean, Cosine, Dot, etc.)
- Vectors: `Vector<DimN>` and optimized `Vector{512,768,1536}Optimized`

It covers single and batch queries, metric selection, normalization, performance tips, and troubleshooting.

Note: Code below uses Swift 6 concurrency (`async/await`) and VectorCore’s public types.

---

## Quick kNN (single query)

Option A — Optimized vector type (fastest for 512/768/1536)

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
- For 512/768/1536, prefer `Vector{512,768,1536}Optimized` to unlock fused kernels and top‑K selection fast paths.
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
- Distances are sorted ascending. If using `DotProductDistance`, the “distance” is negative similarity; smaller (more negative) means larger dot product.

---

## Choosing Metrics and Normalization

Common choices
- EuclideanDistance (L2): default; works well without normalization.
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
- Default `CosineDistance` returns 1.0 if either vector is zero magnitude (includes “both zero” case) — i.e., maximum distance.
- Optimized fused cosine kernels may return 0.0 when both vectors are zero (don’t penalize identical zero inputs).
- Recommendation: If using cosine, normalize vectors; guard zero vectors where possible.

---

## Performance Tips

Prefer optimized vectors when possible
- `Vector512Optimized`, `Vector768Optimized`, `Vector1536Optimized` use `SIMD4<Float>` storage and unrolled loops.
- Internal fused kernels and Top‑K selection paths are enabled:
  - Euclidean: 512/768/1536 optimized top‑K.
  - Cosine: strongest fused Top‑K at 512; fused distance kernels exist for 768/1536 too.

Batch sizes and auto‑parallelization
- `Operations` uses provider heuristics; `BatchOperations` auto‑parallelizes for large inputs.
- As a rule of thumb:
  - Batch operations: arrays ≥ ~1,000 elements tend to parallelize.
  - Pairwise (O(n²)) operations parallelize at smaller thresholds (≈100) due to quadratic work.

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

---

## Troubleshooting

Dimension mismatch
- All vectors in a set must share the same dimension; queries must match database dimension.
- Violations throw `VectorError.dimensionMismatch` from `Operations`.

Zero vectors and cosine
- `CosineDistance` returns 1.0 if either vector has zero magnitude.
- To avoid surprises, normalize inputs or filter out zero vectors before cosine distance.

Dot product distances
- `DotProductDistance` reports “distance = −dot”; distances are negative, so the “closest” item has the most negative (largest magnitude) dot.
- If you need similarity scores, compute the dot directly or negate the distance returned.

Mixed vector types in an array
- Use a single vector type per array (`[Vector<Dim512>]` or `[Vector512Optimized]`), not mixed, to prevent type/dispatch overhead and ensure kernel fast paths are available.

---

## Full Example (Putting It Together)

```swift
import VectorCore

// 1) Prepare database with optimized vectors (best performance)
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
- Expect auto‑parallelization on large inputs; you can override providers in advanced scenarios.
- Handle dimension mismatches and zero vectors proactively.
