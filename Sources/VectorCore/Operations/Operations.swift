// VectorCore: Modern Operations API
//
// Clean API using provider abstraction - no legacy code
//

import Foundation

/// Vector operations
///
/// Operations provides vector computations with pluggable providers for compute
/// and SIMD operations. The default SIMD provider is `SwiftSIMDProvider` and the
/// default compute provider is `CPUComputeProvider.automatic` (the two bindings below).
///
/// ## Usage
/// ```swift
/// // Default: Pure Swift implementation
/// let results = try await Operations.findNearest(to: query, in: database, k: 10)
///
/// // Opt into Accelerate-backed SIMD for a scoped region. The providers are
/// // `@TaskLocal`, so they are installed via `$provider.withValue(_:operation:)`
/// // rather than assigned directly:
/// try await Operations.$simdProvider.withValue(AccelerateArraySIMDProvider()) {
///     let normalized = try await Operations.normalize(vectors)
///     // ...further Operations calls here also see the Accelerate provider...
/// }
/// ```
///
/// Downstream packages (e.g. VectorAccelerate) can swap the compute strategy the
/// same way — bind a concrete `ComputeProvider` for the scope, e.g.
/// `Operations.$computeProvider.withValue(someMetalComputeProvider) { ... }`.
///
/// ## Provider hooks
///
/// The public entries that read a task-local provider hook, directly or through another listed entry, and the
/// exact, ordered provider calls a spy installed through that hook's `$hook.withValue` records at the real route for
/// the inputs of `D02ProviderHookRouteTests` (`computeProviderRoutes`, `simdProviderRoutes` and
/// `linearAlgebraProviderRoute`, one `expectRow` per row). An empty call list means no provider call. Parenthesized
/// text is a note, not a call. `linearAlgebraProvider` is declared with the `LinearAlgebraProvider` protocol.
///
/// | Public entry | Hook | Recorded provider calls |
/// |---|---|---|
/// | `Operations.findNearest`, an installed `BatchKernelProvider` | `computeProvider` | `findNearest` (before the CPU routes) |
/// | `Operations.findNearest`, generic route | `computeProvider` | `parallelReduce` |
/// | `Operations.findNearest`, `Vector512Optimized` Top-K route | `computeProvider` | `parallelReduce` |
/// | `Operations.findNearestBatch`, an installed `BatchKernelProvider` | `computeProvider` | `findNearestBatch` (before `cpuStrategy` is read) |
/// | `Operations.findNearestBatch`, default `.perQuery` route, one query | `computeProvider` | `parallelExecute`, `parallelReduce` (one `parallelReduce` per query) |
/// | `Operations.distanceMatrix` | `computeProvider` | `parallelExecute` |
/// | `Operations.normalize` | `computeProvider` | `parallelExecute` |
/// | `Operations.normalize` | `simdProvider` | `normalize` (one per vector) |
/// | `Operations.centroid` | `simdProvider` | `multiply` |
/// | `Operations.statistics`, two vectors | `simdProvider` | `magnitude`, `magnitude` (one per vector) |
/// | `Operations.pca` | `linearAlgebraProvider` | `svdThin` (through `PCAModel.fit`, exact path) |
/// | `Operations.umap`, default `.pca` initialization | `linearAlgebraProvider` | `svdThin` (through `PCAModel.fit`) |
/// | `Vector` `.*` | `simdProvider` | `elementWiseMultiply` |
/// | `Vector` `./` | `simdProvider` | `elementWiseDivide` |
/// | `Vector` `mean` | `simdProvider` | `mean` |
/// | `Vector` `sum` | `simdProvider` | `sum` |
/// | `Vector` `softmax()` | `simdProvider` | `max`, `divide` |
/// | `Vector` `clamped(to:)` | `simdProvider` | `clip` |
/// | `Vector` `l1Norm`, more than 32 elements | `simdProvider` | `abs` |
/// | `Vector` `l1Norm`, 32 elements or fewer | `simdProvider` | none (scalar loop) |
/// | `Vector` `lInfinityNorm` | `simdProvider` | `abs` |
/// | `Vector` `lInfinityNorm`, an input holding a NaN | `simdProvider` | none (returns NaN before any provider call) |
/// | `Vector` `manhattanDistance(to:)`, more than 32 elements | `simdProvider` | `subtract`, `abs` |
/// | `Vector` `manhattanDistance(to:)`, 32 elements or fewer | `simdProvider` | none (scalar loop) |
/// | `Vector` `chebyshevDistance(to:)` | `simdProvider` | `abs` (through `lInfinityNorm`) |
/// | `Vector` `min(_:)` | `simdProvider` | `elementWiseMin` |
/// | `Vector` `max(_:)` | `simdProvider` | `elementWiseMax` |
/// | `Vector` `minElement()` | `simdProvider` | `minIndex` |
/// | `Vector` `maxElement()` | `simdProvider` | `maxIndex` |
/// | `Vector` `clamp(to:)` | `simdProvider` | `clip` |
/// | `Vector` `absoluteValue()` | `simdProvider` | `abs` |
/// | `Vector` `squareRoot()` | `simdProvider` | `sqrt` |
/// | `DynamicVector` `min(_:)` | `simdProvider` | `elementWiseMin` |
/// | `DynamicVector` `max(_:)` | `simdProvider` | `elementWiseMax` |
/// | `DynamicVector` `clamped(to:)` | `simdProvider` | `clip` |
/// | `PCAModel.fit`, exact path | `linearAlgebraProvider` | `svdThin` |
/// | `PCAModel.fit`, sketch path | `linearAlgebraProvider` | `qrThin`, `qrThin`, `qrThin`, `qrThin`, `qrThin`, `svdThin` (2 × `powerIterations` + 1 `qrThin`; default `powerIterations` 2) |
///
/// For custom static dimensions (widths 7 and 100) `D02DimensionProviderDispatchTests` sees provider calls for
/// `min(_:)`, `max(_:)`, `minElement()`, `maxElement()`, `clamped(to:)`, `clamp(to:)`, `absoluteValue()`,
/// `squareRoot()` and `lInfinityNorm`, and at width 100 for `l1Norm` and `manhattanDistance(to:)`. The static
/// `SIMDOperations.FloatProvider` kernels, the direct `BatchKernels` entries and the BLAS-backed `MatrixDistance`
/// entries read none of these hooks (`D02ProviderHookRouteTests.separateBoundariesReadNoHook`).
/// `test_d02_surface_guards.py` fails when a hook read sits in a function this table does not cover
/// (`test_hook_reads_have_table_rows`), and when a row differs from its `expectRow` in `D02ProviderHookRouteTests`
/// (`test_hook_table_rows_equal_the_route_tests`).
public enum Operations {

    // MARK: - Provider Configuration

    /// The compute provider for execution strategy
    @TaskLocal public static var computeProvider: any ComputeProvider = CPUComputeProvider.automatic

    /// The SIMD provider for the array operations (default `SwiftSIMDProvider`)
    @TaskLocal public static var simdProvider: any ArraySIMDProvider = SwiftSIMDProvider()

    // MARK: - Nearest Neighbor Search

    /// Find k nearest neighbors to a query vector
    ///
    /// CPU selection ranks computed NaN scores after numeric scores and resolves
    /// exact score ties (including signed zeros and NaN pairs) by original input index.
    /// NaNs fill any remaining slots up to `min(k, vectors.count)`.
    /// Optimized Euclidean membership uses squared scores; equal rounded square
    /// roots do not introduce ties at the selection boundary.
    /// See `TopKNaNWrapperTests` (`genericOperationsSelectsNumericScoresBeforeNaNs`,
    /// `optimizedParallelFiniteTiesUseOriginalIndices`, `optimizedParallelEuclideanRetainsNaNsAfterNumericCandidates`)
    /// for generic and optimized CPU coverage.
    /// An installed `BatchKernelProvider` supplies its own result ordering.
    ///
    /// Validation runs in this order before any provider or metric work: `k > 0`, a nonempty `vectors`,
    /// the query against `vectors[0]`'s dimension, then every candidate row
    /// (`P01OperationsSelectionTests.lateMalformedRowsAreRejectedBeforeAnyWork`).
    ///
    /// Routes, in order (`n = vectors.count`, `k' = min(k, n)`):
    /// 1. An installed `BatchKernelProvider` receives the original `k` and `metric`; its results and errors
    ///    are returned unchanged (`BatchKernelProviderTests`, `P01OperationsRouteTests.providerPrecedence`).
    /// 2. When `k < n`: Euclidean on `Vector512Optimized`, `Vector768Optimized` and `Vector1536Optimized`
    ///    (selected on squared distances, square roots returned), fused cosine on `Vector512Optimized`, and dot
    ///    product on `Vector512Optimized` (negated similarities returned). Each provider range keeps its own
    ///    heap of capacity `min(k, range.count)`; an empty range keeps an empty heap and scores nothing.
    /// 3. When `k' < n / 10` (`TopKSelection.usesSmallKStrategy`) and the case is not Euclidean or cosine on a
    ///    512/768/1536 optimized type (after step 2 that leaves only cosine on 768 and 1536): one
    ///    `computeProvider.parallelReduce` call. Each supplied range is scanned in index order into its own
    ///    heap of capacity `min(k', range.count)`. On a successful return `metric` has been called exactly once
    ///    per candidate (a cancelled search stops earlier; see below); there is no AutoTuning calibration, and
    ///    the order of calls across ranges is unspecified.
    ///    `P01OperationsRedTests.operationsColdKeySmallKEvaluatesEachCandidateOnce` observes `n` calls, provider
    ///    entry and no AutoTuning publication on a cold key; `P01OperationsRouteTests.customAndParameterizedWitnesses`
    ///    observes the per-range call order.
    /// 4. Every other case first scores all `n` candidates into an `n`-element array, then selects with
    ///    `TopKSelection.select`. When the metric maps to an AutoTuning kernel key (every metric except
    ///    `ManhattanDistance`), the first use of its (dimension, kernel) key runs a calibration that calls
    ///    `metric` on up to the first 256 candidates before scoring
    ///    (`P01OperationsRedTests.operationsColdKeyLargeKKeepsCalibrationControl`), unless AutoTuning is
    ///    disabled (`VC_AUTOTUNE=0`), its cumulative calibration-time budget is spent, or the same key is being
    ///    calibrated at that moment; then a default tuning is used without calling `metric`
    ///    (`AutoTuning.calibrateIfNeeded`; `AutoTuningConcurrencyTests.isolatedContractMatrix`, scenarios
    ///    `disabled`, `cycle` and `time-slow`). Cosine on
    ///    `Vector768Optimized` and `Vector1536Optimized` takes this route for every `k`, scored by the fused
    ///    `BatchKernels` range helpers, so it keeps the `n`-score array even when `k` is small
    ///    (`P01OperationsSelectionTests.cosine768And1536StayOnTheFusedFullScoreRoute`).
    ///
    /// Partial heaps from steps 2 and 3 combine in place when the left heap already has the full capacity `k'`,
    /// otherwise into a new heap of capacity `min(k', a.k + b.k)`. With a provider whose ranges are disjoint,
    /// the per-range capacities sum to at most `n`; this bound holds for custom providers too, while the
    /// library's worker-count resource model covers only the built-in bounded providers. Capacities, empty
    /// ranges and that sum are observed by `P01OperationsRouteTests.extremePartitions`.
    ///
    /// Cancellation differs by route (cases observed by `P01OperationsRouteTests.providerPrecedence` and
    /// `P01OperationsRouteTests.serialParallelNestedAndCancelled`):
    /// 1. Provider-defined: Operations adds no check, so a provider that ignores cancellation returns its reply.
    /// 2. Only the provider's checks around each range and combine callback (the `ComputeProvider` contract):
    ///    the built-in CPU providers throw `CancellationError`.
    /// 3. The provider's checks, plus a check before each metric call, so a cancelled task throws
    ///    `CancellationError` before its next metric call even under a provider that performs no checks.
    /// 4. Only the provider's checks on its parallel branch (`parallelExecute`, or the hinted `parallelReduce`
    ///    of the fused helpers). That parallel-branch part is an unverified contract taken from
    ///    `ComputeProvider`: no test cancels a search on that branch. Calibration and the serial branch below the
    ///    tuned break-even contain no check, so a cancelled task can return a complete result: generic large `k`
    ///    over 100 candidates, and cosine on `Vector768Optimized` with `k = 3` over 64 candidates.
    ///
    /// - Parameters:
    ///   - query: Query vector
    ///   - vectors: Search space
    ///   - k: Number of neighbors to find
    ///   - metric: Distance metric (default: Euclidean)
    /// - Returns: Nearest neighbors sorted by distance
    public static func findNearest<V: VectorProtocol, M: DistanceMetric>(
        to query: V,
        in vectors: [V],
        k: Int = 1,
        metric: M = EuclideanDistance()
    ) async throws -> [NearestNeighborResult] where V.Scalar == Float, M.Scalar == Float {
        // Validate inputs
        guard k > 0 else {
            throw VectorError.invalidDimension(k, reason: "k must be positive")
        }
        guard !vectors.isEmpty else {
            throw VectorError.invalidDimension(0, reason: "Vector collection cannot be empty")
        }
        let expectedDim = vectors[0].scalarCount
        guard query.scalarCount == expectedDim else {
            throw VectorError.dimensionMismatch(expected: expectedDim, actual: query.scalarCount)
        }
        // Ensure all candidate vectors have consistent dimension
        for v in vectors where v.scalarCount != expectedDim {
            throw VectorError.dimensionMismatch(expected: expectedDim, actual: v.scalarCount)
        }

        return try await validatedNearest(to: query, in: vectors, k: k, metric: metric)
    }

    /// The single-query search after validation: steps 1-4 of `findNearest`.
    ///
    /// Callers must already have checked everything `findNearest` validates: `k > 0`, a nonempty `vectors`,
    /// and one dimension shared by `query` and every row. Its only callers are `findNearest` (after its own
    /// checks) and the `.perQuery` route of `findNearestBatch`, its default CPU strategy (after the batch
    /// checks), so the batch validates the candidate rows once rather than once per query
    /// (`P01OperationsRedTests.batchFallbackCandidateValidationDoesNotRepeatPerQuery`).
    private static func validatedNearest<V: VectorProtocol, M: DistanceMetric>(
        to query: V,
        in vectors: [V],
        k: Int,
        metric: M
    ) async throws -> [NearestNeighborResult] where V.Scalar == Float, M.Scalar == Float {
        // Route 1: an installed BatchKernelProvider receives the call, with the original `k` and
        // `metric`, before the CPU routes below (`P01OperationsRouteTests.providerPrecedence`).
        if let gpu = computeProvider as? any BatchKernelProvider {
            let pairs = try await gpu.findNearest(query: query, candidates: vectors, k: k, metric: metric)
            return pairs.map { NearestNeighborResult(index: $0.index, distance: $0.distance) }
        }

        // Route 2: per-range Top-K heaps for Vector*Optimized with the metrics below (see `findNearest`)
        if k < vectors.count { // only worth it when K < N
            // Euclidean: membership and order use squared distances; square roots are returned
            if metric is EuclideanDistance {
                if let q = query as? Vector512Optimized, let c = exactArray(vectors, as: Vector512Optimized.self) {
                    let results = try await topk_euclid_optimized(query: q, candidates: c, k: k, dim: 512)
                    return results
                }
                if let q = query as? Vector768Optimized, let c = exactArray(vectors, as: Vector768Optimized.self) {
                    let results = try await topk_euclid_optimized(query: q, candidates: c, k: k, dim: 768)
                    return results
                }
                if let q = query as? Vector1536Optimized, let c = exactArray(vectors, as: Vector1536Optimized.self) {
                    let results = try await topk_euclid_optimized(query: q, candidates: c, k: k, dim: 1536)
                    return results
                }
            }
            // Cosine fused Top‑K (512 supported)
            if metric is CosineDistance {
                if let q = query as? Vector512Optimized, let c = exactArray(vectors, as: Vector512Optimized.self) {
                    let results = try await topk_cosine_fused_512(query: q, candidates: c, k: k)
                    return results
                }
            }
            // Dot product Top‑K (512 supported) — return negative dot as distance
            if metric is DotProductDistance {
                if let q = query as? Vector512Optimized, let c = exactArray(vectors, as: Vector512Optimized.self) {
                    let results = try await topk_dot_512(query: q, candidates: c, k: k)
                    return results
                }
            }
        }

        // Generic small-k: stream the candidates into per-range heaps, scoring each at most once (route 3 of
        // `findNearest`). `actualK` is at most `vectors.count`, so no heap is sized from an unbounded `k`
        // (`P01OperationsSelectionTests.unboundedKSizesNothing` runs `k = Int.max`).
        let actualK = min(k, vectors.count)
        if TopKSelection.usesSmallKStrategy(actualK: actualK, count: vectors.count),
           !scoresWithBatchKernels(query: query, metric: metric) {
            return try await streamNearest(query: query, candidates: vectors, k: actualK, metric: metric)
        }

        let distances = try await computeDistances(
            from: query,
            to: vectors,
            metric: metric
        )

        let selection = TopKSelection.select(k: k, from: distances)
        return selection.toTuples().map {
            NearestNeighborResult(index: $0.index, distance: $0.distance)
        }
    }

    /// True when `computeDistances` takes its optimized branch for this metric: Euclidean or cosine with a query
    /// that casts to `Vector512Optimized`, `Vector768Optimized` or `Vector1536Optimized`. It repeats the type tests
    /// of `computeDistances` (`variant == .optimized`, then the metric and query casts). For the vector types
    /// VectorCore defines, that is exactly when `computeDistances` scores the case with a `BatchKernels` range
    /// helper: `V` is one concrete type, so the query cast succeeds exactly when `V` is that type, and then the
    /// array's type-identity test (`exactArray`) succeeds too. A client type whose values cast to an optimized
    /// vector while `V` is not that type (for example a retroactive `VectorProtocol` conformance of
    /// `Optional<Vector512Optimized>`) also makes this true, but `exactArray` returns nil there, so
    /// `computeDistances` scores it with its per-candidate fallback (`metric.distance` for each candidate) and
    /// small k still keeps the full-score route; no test covers such a type (P02 final review C-M1). These cases
    /// keep the full-score route.
    /// `P01OperationsSelectionTests.routeMatrixMatchesComputeDistancesBatchKernelBranches` observes the agreement
    /// for every built-in metric and a custom one on the three optimized types and two generic types; for the
    /// direct-route cases step 2 of `findNearest` runs first, so this predicate's value there is not observable.
    private static func scoresWithBatchKernels<V: VectorProtocol, M: DistanceMetric>(query: V, metric: M) -> Bool {
        guard metric is EuclideanDistance || metric is CosineDistance else { return false }
        return query is Vector512Optimized || query is Vector768Optimized || query is Vector1536Optimized
    }

    /// Route 3 of `findNearest`: one `computeProvider.parallelReduce` (the overload without a chunk hint, so the
    /// provider's own policy partitions the work) over `0..<candidates.count`.
    ///
    /// Each supplied range scans its candidates in increasing index order, checking cancellation before each
    /// `metric` call and pushing every score into a heap of capacity `min(k, range.count)`; an empty range
    /// returns an empty heap without calling `metric`. Partials combine with `combinePartialHeaps`.
    ///
    /// Precondition: `0 < k < candidates.count / 10` (the caller's `usesSmallKStrategy` test), so `initial`
    /// is bounded by the candidate count.
    private static func streamNearest<V: VectorProtocol, M: DistanceMetric>(
        query: V,
        candidates: [V],
        k: Int,
        metric: M
    ) async throws -> [NearestNeighborResult] where V.Scalar == Float, M.Scalar == Float {
        let initial = TopKBuffer(k: k, isMinHeap: false)
        let buf = try await computeProvider.parallelReduce(items: 0..<candidates.count, initial: initial, { range in
            var local = TopKBuffer(k: min(k, range.count), isMinHeap: false)
            for i in range {
                try Task.checkCancellation()
                local.pushIfBetter(val: metric.distance(query, candidates[i]), idx: i)
            }
            return local
        }, { a, b in combinePartialHeaps(a, b, k: k) })
        return toResults(buf)
    }

    /// Find k nearest neighbors for multiple queries
    ///
    /// Validation (`k > 0`, nonempty queries and vectors, every query row, every candidate row, then the
    /// query dimension against the candidate dimension) runs once for the whole batch, before any provider,
    /// `cpuStrategy` or metric work (`P01OperationsSelectionTests.lateMalformedRowsAreRejectedBeforeAnyWork`,
    /// `P02GramRouteTests.batchValidationPrecedesStrategyChecks`).
    ///
    /// Routes, in order:
    /// 1. An installed `BatchKernelProvider`: one `findNearestBatch` call with the original whole batch, `k` and
    ///    `metric`, before `cpuStrategy` is validated or planned; its results and errors are returned unchanged
    ///    (`BatchKernelProviderTests.findNearestBatchUsesOverride`). `cpuStrategy`, including its payload limit,
    ///    describes only the CPU route below: it neither constrains nor bypasses the provider
    ///    (`P02BatchProviderTests.providerPrecedesInvalidAndInfeasibleCPUOptions`).
    /// 2. Otherwise `cpuStrategy` selects the CPU route (`BatchSearchCPUStrategy`):
    ///    - `.perQuery`, the default: `computeProvider.parallelExecute` over the query indices runs the single-query
    ///      routes of `findNearest` for each query without repeating the candidate validation
    ///      (`P01OperationsRedTests.batchFallbackCandidateValidationDoesNotRepeatPerQuery`). Each row has the indices
    ///      and score bits that `findNearest` returns for its query under the same `k`, metric and provider, with that
    ///      route's ordering and cancellation behavior (`P02DefaultRouteRedTests.defaultRouteEqualsSingleQuery`,
    ///      `P02BatchProviderTests.sequentialProviderMatchesSingleQuery`). Those routes are not promised to equal
    ///      `metric.distance` bit for bit or exact real-number distances, and the order in which a custom metric's
    ///      `distance` is called, across queries and candidate ranges, is unspecified.
    ///    - `.gram(queryTileRows:candidateTileRows:maximumTemporaryPayloadBytes:)`: the serial tiled Gram route, only
    ///      for `EuclideanDistance` or `CosineDistance` on `Vector512Optimized`, `Vector768Optimized` or
    ///      `Vector1536Optimized`. It schedules nothing on `computeProvider` (`P02GramRouteTests.serialAndUnscheduled`).
    ///
    /// `.gram` errors, all thrown before any cancellation check, packing, BLAS call, heap or result allocation, in this
    /// order when several apply (`P02GramRouteTests.validationPrecedence`): `.invalidConfiguration` naming the first
    /// field out of its domain (tile rows must be positive, the limit nonnegative); `.unsupportedOperation` for any
    /// other metric or vector type; `.resourceUnavailable` for a plan whose counts do not fit `Int` or the BLAS `Int32`
    /// arguments, then for a logical temporary payload above `maximumTemporaryPayloadBytes`. None of them falls back
    /// to `.perQuery`.
    ///
    /// `.gram` rows: each holds `min(k, vectors.count)` entries in ascending score order, computed NaNs after numeric
    /// scores and exact score ties by the smaller original index. Euclidean membership and order use the squared
    /// scores, then each squared score `s` returns `s` if it is NaN, its square root if it is positive and 0 otherwise,
    /// so two entries whose roots are equal keep the order of their squared scores
    /// (`P02GramRouteTests.selectionMatchesSortedTileScores`). The result always holds `queries.count` rows of
    /// `min(k, vectors.count)` entries, outside the payload limit: `k >= vectors.count` returns every candidate for
    /// every query. Gram scores are a different arithmetic from `findNearest`; see `BatchSearchCPUStrategy.gram` for
    /// the derived error bound and what it does not promise.
    ///
    /// Cancellation:
    /// - `.perQuery`: each query runs `findNearest`'s routes, with the cancellation behavior `findNearest` documents
    ///   for them, under `computeProvider.parallelExecute`.
    /// - `.gram`: `Task.checkCancellation()` runs after the plan and limit succeed and before any scratch or result
    ///   allocation, at each query-tile and candidate-tile boundary, after each tile's scores are computed, before each
    ///   result row is allocated and filled, and before the result is returned; a cancelled search throws
    ///   `CancellationError` without a partial result (`P02GramCancellationTests.preCancelledValidPlanDoesNoWork`,
    ///   `P02GramCancellationTests.cancellationAtEveryEventStopsAtTheNextCheckpoint`). Packing, normalization, a BLAS call
    ///   or the fill of one row is not interrupted once started, so the work between two checks depends on the tile
    ///   rows and `k`; there is no latency or memory-release deadline. A cancellation that arrives after the last
    ///   check does not affect the returned result.
    ///
    /// `BatchOperations.Configuration` (`enableMatrixRouting`, `matrixRoutingMinN`) does not affect this function: it
    /// reads no `BatchOperations` setting, and the Gram route runs only when `cpuStrategy` is `.gram`
    /// (`P02GramRouteTests.matrixRoutingFlagDoesNotAffectBatchSearch`; lexical guard
    /// `Scripts/ci/tests/test_gram_route_lexical.py`).
    ///
    /// - Parameters:
    ///   - queries: Query vectors
    ///   - vectors: Search space
    ///   - k: Number of neighbors per query
    ///   - metric: Distance metric
    ///   - cpuStrategy: How the CPU route searches the batch when no `BatchKernelProvider` is installed
    /// - Returns: Results for each query
    public static func findNearestBatch<V: VectorProtocol, M: DistanceMetric>(
        queries: [V],
        in vectors: [V],
        k: Int = 1,
        metric: M = EuclideanDistance(),
        cpuStrategy: BatchSearchCPUStrategy = .perQuery
    ) async throws -> [[NearestNeighborResult]] where V.Scalar == Float, M.Scalar == Float {
        // Validate inputs
        guard k > 0 else {
            throw VectorError.invalidDimension(k, reason: "k must be positive")
        }
        guard !queries.isEmpty else {
            throw VectorError.invalidDimension(0, reason: "queries cannot be empty")
        }
        guard !vectors.isEmpty else {
            throw VectorError.invalidDimension(0, reason: "vectors cannot be empty")
        }
        let qDim = queries[0].scalarCount
        let vDim = vectors[0].scalarCount
        // All queries same dim
        for q in queries where q.scalarCount != qDim {
            throw VectorError.dimensionMismatch(expected: qDim, actual: q.scalarCount)
        }
        // All vectors same dim
        for v in vectors where v.scalarCount != vDim {
            throw VectorError.dimensionMismatch(expected: vDim, actual: v.scalarCount)
        }
        // Queries dim must match vectors dim
        guard qDim == vDim else {
            throw VectorError.dimensionMismatch(expected: vDim, actual: qDim)
        }
        // Route 1: an installed BatchKernelProvider receives the whole batch (a conformer may encode all queries in
        // one kernel; the default loops findNearest), before `cpuStrategy` is read
        // (`BatchKernelProviderTests.findNearestBatchUsesOverride`).
        if let gpu = computeProvider as? any BatchKernelProvider {
            let batched = try await gpu.findNearestBatch(queries: queries, candidates: vectors, k: k, metric: metric)
            return batched.map { $0.map { NearestNeighborResult(index: $0.index, distance: $0.distance) } }
        }
        switch cpuStrategy {
        case .perQuery:
            // Every query and candidate row is validated above, so each query skips straight to the routes.
            return try await computeProvider.parallelExecute(items: 0..<queries.count) { i in
                try await validatedNearest(
                    to: queries[i],
                    in: vectors,
                    k: k,
                    metric: metric
                )
            }
        case let .gram(queryTileRows, candidateTileRows, maximumTemporaryPayloadBytes):
            return try gramBatch(queries: queries, candidates: vectors, k: k, metric: metric,
                                 queryTileRows: queryTileRows, candidateTileRows: candidateTileRows,
                                 maximumTemporaryPayloadBytes: maximumTemporaryPayloadBytes)
        }
    }

    // MARK: - Top‑K helpers for the optimized types

    /// Ordered combine for the per-range heaps of `findNearest` steps 2 and 3, where `k` is the global
    /// capacity (`k' = min(k, n)` of `findNearest`) and every partial has capacity at most `k`.
    ///
    /// When `a` already has capacity `k`, `b`'s entries are merged into a copy of `a` (`mergeIn`). Otherwise
    /// both merge into a new heap of capacity `min(k, a.k + b.k)`, enough to keep the exact best of two disjoint
    /// partials. Either way the result keeps the best `min(k, a.size + b.size)` entries of the two, so the
    /// combine is associative over disjoint ranges, as `parallelReduce` requires. Direction and tie policy
    /// come from the partials, and `mergeIn` traps if they differ (`TopKBufferMergeInTrapTests`). Whether a
    /// merge copies arrays depends on copy-on-write sharing; no allocation count is claimed. The capacities
    /// chosen here are observed by `P01OperationsRouteTests.extremePartitions`.
    private static func combinePartialHeaps(_ a: TopKBuffer, _ b: TopKBuffer, k: Int) -> TopKBuffer {
        if a.k == k {
            var merged = a
            merged.mergeIn(from: b)
            return merged
        }
        // a.k < k and b.k <= k, so `k - b.k` cannot overflow and the sum is formed only when it is at most k.
        let capacity = a.k > k - b.k ? k : a.k + b.k
        var merged = TopKBuffer(k: capacity, isMinHeap: a.isMinHeap, tieBreaker: a.tieBreaker)
        merged.mergeIn(from: a)
        merged.mergeIn(from: b)
        return merged
    }

    @inline(__always)
    private static func toResults(_ buf: TopKBuffer, sqrtValues: Bool = false, negateValues: Bool = false) -> [NearestNeighborResult] {
        let count = min(buf.size, buf.k)
        var pairs: [(Int, Float)] = []
        pairs.reserveCapacity(count)
        for i in 0..<count {
            var v = buf.vals[i]
            if sqrtValues { v = sqrt(v) }
            if negateValues { v = -v }
            pairs.append((buf.idxs[i], v))
        }
        return pairs.sorted { TopKSelection.orderedAscending($0, $1, buf.tieBreaker) }
            .map { NearestNeighborResult(index: $0.0, distance: $0.1) }
    }

    // Euclidean Top‑K using squared distance kernels for selection, sqrt at the end for distances
    private static func topk_euclid_optimized(
        query: Vector512Optimized,
        candidates: [Vector512Optimized],
        k: Int,
        dim: Int
    ) async throws -> [NearestNeighborResult] {
        let n = candidates.count
        let initial = TopKBuffer(k: k, isMinHeap: false)
        let buf = try await computeProvider.parallelReduce(items: 0..<n, initial: initial, { range in
            let capacity = min(k, range.count)
            var local = TopKBuffer(k: capacity, isMinHeap: false)
            guard capacity > 0 else { return local }
            TopKSelectionKernels.range_topk_euclid2_512(query: query, candidates: candidates, range: range, k: capacity, localTopK: &local)
            return local
        }, { a, b in combinePartialHeaps(a, b, k: k) })
        return toResults(buf, sqrtValues: true)
    }

    private static func topk_euclid_optimized(
        query: Vector768Optimized,
        candidates: [Vector768Optimized],
        k: Int,
        dim: Int
    ) async throws -> [NearestNeighborResult] {
        let n = candidates.count
        let initial = TopKBuffer(k: k, isMinHeap: false)
        let buf = try await computeProvider.parallelReduce(items: 0..<n, initial: initial, { range in
            let capacity = min(k, range.count)
            var local = TopKBuffer(k: capacity, isMinHeap: false)
            guard capacity > 0 else { return local }
            TopKSelectionKernels.range_topk_euclid2_768(query: query, candidates: candidates, range: range, k: capacity, localTopK: &local)
            return local
        }, { a, b in combinePartialHeaps(a, b, k: k) })
        return toResults(buf, sqrtValues: true)
    }

    private static func topk_euclid_optimized(
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized],
        k: Int,
        dim: Int
    ) async throws -> [NearestNeighborResult] {
        let n = candidates.count
        let initial = TopKBuffer(k: k, isMinHeap: false)
        let buf = try await computeProvider.parallelReduce(items: 0..<n, initial: initial, { range in
            let capacity = min(k, range.count)
            var local = TopKBuffer(k: capacity, isMinHeap: false)
            guard capacity > 0 else { return local }
            TopKSelectionKernels.range_topk_euclid2_1536(query: query, candidates: candidates, range: range, k: capacity, localTopK: &local)
            return local
        }, { a, b in combinePartialHeaps(a, b, k: k) })
        return toResults(buf, sqrtValues: true)
    }

    private static func topk_cosine_fused_512(
        query: Vector512Optimized,
        candidates: [Vector512Optimized],
        k: Int
    ) async throws -> [NearestNeighborResult] {
        let n = candidates.count
        let initial = TopKBuffer(k: k, isMinHeap: false)
        let buf = try await computeProvider.parallelReduce(items: 0..<n, initial: initial, { range in
            let capacity = min(k, range.count)
            var local = TopKBuffer(k: capacity, isMinHeap: false)
            guard capacity > 0 else { return local }
            TopKSelectionKernels.range_topk_cosine_fused_512(query: query, candidates: candidates, range: range, k: capacity,
                                                            localTopK: &local)
            return local
        }, { a, b in combinePartialHeaps(a, b, k: k) })
        return toResults(buf)
    }

    private static func topk_dot_512(
        query: Vector512Optimized,
        candidates: [Vector512Optimized],
        k: Int
    ) async throws -> [NearestNeighborResult] {
        let n = candidates.count
        let initial = TopKBuffer(k: k, isMinHeap: true)
        let buf = try await computeProvider.parallelReduce(items: 0..<n, initial: initial, { range in
            let capacity = min(k, range.count)
            var local = TopKBuffer(k: capacity, isMinHeap: true)
            guard capacity > 0 else { return local }
            TopKSelectionKernels.range_topk_dot_512(query: query, candidates: candidates, range: range, k: capacity, localTopK: &local)
            return local
        }, { a, b in combinePartialHeaps(a, b, k: k) })
        // Convert similarity to distance via negation
        return toResults(buf, negateValues: true)
    }

    // MARK: - Distance Computation

    /// Compute distances from query to all vectors
    private static func computeDistances<V: VectorProtocol, M: DistanceMetric>(
        from query: V,
        to vectors: [V],
        metric: M
    ) async throws -> [Float] where V.Scalar == Float, M.Scalar == Float {
        let dim = query.scalarCount
        let items = vectors.count

        // Infer variant (optimized vs generic) based on runtime type
        let variant: ParallelHeuristic.Variant = {
            if query is Vector512Optimized || query is Vector768Optimized || query is Vector1536Optimized {
                return .optimized
            }
            return .generic
        }()

        // Map metric type to metric class
        let metricClass: ParallelHeuristic.MetricClass = {
            switch metric {
            case is EuclideanDistance: return .euclideanLike
            case is DotProductDistance: return .dot
            case is CosineDistance: return .cosine
            case is ManhattanDistance: return .manhattan
            default: return .euclideanLike
            }
        }()

        // Decide parallelism using AutoTuning if available; otherwise fallback heuristic
        var tunedMinChunk: Int?
        var parallel: Bool
        if let kind = {
            switch metricClass {
            case .euclideanLike:
                // We are using euclidean (sqrt) at this level
                return KernelKind.euclid
            case .cosine:
                return KernelKind.cosineFused
            case .dot:
                return KernelKind.dot
            case .manhattan:
                return nil
            }
        }() {
            let cores = ProcessInfo.processInfo.activeProcessorCount
            // Calibrator: measure seq cost over a tiny sample; estimate parallel overhead and effective factor
            let tuning = AutoTuning.calibrateIfNeeded(dim: dim, kind: kind, providerCores: cores) {
                // Fix: Clamp M to actual array size to prevent out-of-bounds access when items < 32
                let M = min(max(32, min(items, 256)), items)
                var idx = 0
                let clock = ContinuousClock()
                let start = clock.now
                if M > 0 {
                    for _ in 0..<M {
                        let v = vectors[idx]
                        _ = metric.distance(query, v)
                        idx &+= 1
                    }
                }
                let elapsed = start.duration(to: clock.now).nanoseconds
                let a = (M > 0) ? (elapsed / Double(M)) : 1_000.0
                let Tp = ParallelHeuristic.parallelOverheadNs(items: M)
                let peff = Double(min(cores, 8)) * 0.7
                return CalibrationProbes(nsPerCandidateSeq: a, parallelOverheadNs: Tp, effectiveParallelFactor: peff)
            }
            tunedMinChunk = tuning.minChunk
            parallel = items >= tuning.breakEvenN
        } else {
            parallel = ParallelHeuristic.shouldParallelize(dim: dim, items: items, variant: variant, metric: metricClass)
        }

        // Euclidean and cosine on the optimized types: score with the `BatchKernels` range helpers
        if variant == .optimized {
            // Euclidean distance (sqrt)
            if metric is EuclideanDistance {
                if let q = query as? Vector512Optimized, let c = exactArray(vectors, as: Vector512Optimized.self) {
                    return try await batchKernelDistances_euclid(query: q, candidates: c, dim: 512, parallel: parallel, tunedMinChunk: tunedMinChunk)
                }
                if let q = query as? Vector768Optimized, let c = exactArray(vectors, as: Vector768Optimized.self) {
                    return try await batchKernelDistances_euclid(query: q, candidates: c, dim: 768, parallel: parallel, tunedMinChunk: tunedMinChunk)
                }
                if let q = query as? Vector1536Optimized, let c = exactArray(vectors, as: Vector1536Optimized.self) {
                    return try await batchKernelDistances_euclid(query: q, candidates: c, dim: 1536, parallel: parallel, tunedMinChunk: tunedMinChunk)
                }
            }
            // Cosine distance (fused one-pass)
            if metric is CosineDistance {
                if let q = query as? Vector512Optimized, let c = exactArray(vectors, as: Vector512Optimized.self) {
                    return try await batchKernelDistances_cosineFused(query: q, candidates: c, dim: 512, parallel: parallel, tunedMinChunk: tunedMinChunk)
                }
                if let q = query as? Vector768Optimized, let c = exactArray(vectors, as: Vector768Optimized.self) {
                    return try await batchKernelDistances_cosineFused(query: q, candidates: c, dim: 768, parallel: parallel, tunedMinChunk: tunedMinChunk)
                }
                if let q = query as? Vector1536Optimized, let c = exactArray(vectors, as: Vector1536Optimized.self) {
                    return try await batchKernelDistances_cosineFused(query: q, candidates: c, dim: 1536, parallel: parallel, tunedMinChunk: tunedMinChunk)
                }
            }
        }

        // Fallback: existing per-candidate path
        if parallel {
            return try await computeProvider.parallelExecute(items: 0..<vectors.count) { i in
                metric.distance(query, vectors[i])
            }
        } else {
            return vectors.map { metric.distance(query, $0) }
        }
    }

    // MARK: - Batch-kernel helpers for the optimized types

    @usableFromInline
    struct _Chunk: Sendable { let start: Int; let values: [Float] }

    @inline(__always)
    private static func minChunk(forDim dim: Int) -> Int {
        switch dim {
        case 1536: return 512
        case 768: return 256
        default: return 256
        }
    }

    // Euclidean (sqrt) batch via BatchKernels
    private static func batchKernelDistances_euclid(
        query: Vector512Optimized,
        candidates: [Vector512Optimized],
        dim: Int,
        parallel: Bool,
        tunedMinChunk: Int? = nil
    ) async throws -> [Float] {
        let n = candidates.count
        if !parallel {
            var out = [Float](repeating: 0, count: n)
            out.withUnsafeMutableBufferPointer { buf in
                var start = 0
                let step = tunedMinChunk ?? minChunk(forDim: dim)
                while start < n {
                    let end = min(start + step, n)
                    let sub = UnsafeMutableBufferPointer<Float>(start: buf.baseAddress!.advanced(by: start), count: end - start)
                    BatchKernels.range_euclid_512(query: query, candidates: candidates, range: start..<end, out: sub)
                    start = end
                }
            }
            return out
        } else {
            let chunks: [_Chunk] = try await computeProvider.parallelReduce(
                items: 0..<n,
                initial: [],
                minChunk: tunedMinChunk ?? minChunk(forDim: dim),
                { range in
                    var tmp = [Float](repeating: 0, count: range.count)
                    tmp.withUnsafeMutableBufferPointer { sub in
                        BatchKernels.range_euclid_512(query: query, candidates: candidates, range: range, out: sub)
                    }
                    return [_Chunk(start: range.lowerBound, values: tmp)]
                },
                { $0 + $1 }
            )
            var out = [Float](repeating: 0, count: n)
            for ch in chunks { out.replaceSubrange(ch.start..<(ch.start + ch.values.count), with: ch.values) }
            return out
        }
    }

    private static func batchKernelDistances_euclid(
        query: Vector768Optimized,
        candidates: [Vector768Optimized],
        dim: Int,
        parallel: Bool,
        tunedMinChunk: Int? = nil
    ) async throws -> [Float] {
        let n = candidates.count
        if !parallel {
            var out = [Float](repeating: 0, count: n)
            out.withUnsafeMutableBufferPointer { buf in
                var start = 0
                let step = tunedMinChunk ?? minChunk(forDim: dim)
                while start < n {
                    let end = min(start + step, n)
                    let sub = UnsafeMutableBufferPointer<Float>(start: buf.baseAddress!.advanced(by: start), count: end - start)
                    BatchKernels.range_euclid_768(query: query, candidates: candidates, range: start..<end, out: sub)
                    start = end
                }
            }
            return out
        } else {
            let chunks: [_Chunk] = try await computeProvider.parallelReduce(
                items: 0..<n,
                initial: [],
                minChunk: tunedMinChunk ?? minChunk(forDim: dim),
                { range in
                    var tmp = [Float](repeating: 0, count: range.count)
                    tmp.withUnsafeMutableBufferPointer { sub in
                        BatchKernels.range_euclid_768(query: query, candidates: candidates, range: range, out: sub)
                    }
                    return [_Chunk(start: range.lowerBound, values: tmp)]
                },
                { $0 + $1 }
            )
            var out = [Float](repeating: 0, count: n)
            for ch in chunks { out.replaceSubrange(ch.start..<(ch.start + ch.values.count), with: ch.values) }
            return out
        }
    }

    private static func batchKernelDistances_euclid(
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized],
        dim: Int,
        parallel: Bool,
        tunedMinChunk: Int? = nil
    ) async throws -> [Float] {
        let n = candidates.count
        if !parallel {
            var out = [Float](repeating: 0, count: n)
            out.withUnsafeMutableBufferPointer { buf in
                var start = 0
                let step = tunedMinChunk ?? minChunk(forDim: dim)
                while start < n {
                    let end = min(start + step, n)
                    let sub = UnsafeMutableBufferPointer<Float>(start: buf.baseAddress!.advanced(by: start), count: end - start)
                    BatchKernels.range_euclid_1536(query: query, candidates: candidates, range: start..<end, out: sub)
                    start = end
                }
            }
            return out
        } else {
            let chunks: [_Chunk] = try await computeProvider.parallelReduce(
                items: 0..<n,
                initial: [],
                minChunk: tunedMinChunk ?? minChunk(forDim: dim),
                { range in
                    var tmp = [Float](repeating: 0, count: range.count)
                    tmp.withUnsafeMutableBufferPointer { sub in
                        BatchKernels.range_euclid_1536(query: query, candidates: candidates, range: range, out: sub)
                    }
                    return [_Chunk(start: range.lowerBound, values: tmp)]
                },
                { $0 + $1 }
            )
            var out = [Float](repeating: 0, count: n)
            for ch in chunks { out.replaceSubrange(ch.start..<(ch.start + ch.values.count), with: ch.values) }
            return out
        }
    }

    // Cosine (fused) batch via BatchKernels
    private static func batchKernelDistances_cosineFused(
        query: Vector512Optimized,
        candidates: [Vector512Optimized],
        dim: Int,
        parallel: Bool,
        tunedMinChunk: Int? = nil
    ) async throws -> [Float] {
        let n = candidates.count
        if !parallel {
            var out = [Float](repeating: 0, count: n)
            out.withUnsafeMutableBufferPointer { buf in
                var start = 0
                let step = tunedMinChunk ?? minChunk(forDim: dim)
                while start < n {
                    let end = min(start + step, n)
                    let sub = UnsafeMutableBufferPointer<Float>(start: buf.baseAddress!.advanced(by: start), count: end - start)
                    BatchKernels.range_cosine_fused_512(query: query, candidates: candidates, range: start..<end, out: sub)
                    start = end
                }
            }
            return out
        } else {
            let chunks: [_Chunk] = try await computeProvider.parallelReduce(
                items: 0..<n,
                initial: [],
                minChunk: tunedMinChunk ?? minChunk(forDim: dim),
                { range in
                    var tmp = [Float](repeating: 0, count: range.count)
                    tmp.withUnsafeMutableBufferPointer { sub in
                        BatchKernels.range_cosine_fused_512(query: query, candidates: candidates, range: range, out: sub)
                    }
                    return [_Chunk(start: range.lowerBound, values: tmp)]
                },
                { $0 + $1 }
            )
            var out = [Float](repeating: 0, count: n)
            for ch in chunks { out.replaceSubrange(ch.start..<(ch.start + ch.values.count), with: ch.values) }
            return out
        }
    }

    private static func batchKernelDistances_cosineFused(
        query: Vector768Optimized,
        candidates: [Vector768Optimized],
        dim: Int,
        parallel: Bool,
        tunedMinChunk: Int? = nil
    ) async throws -> [Float] {
        let n = candidates.count
        if !parallel {
            var out = [Float](repeating: 0, count: n)
            out.withUnsafeMutableBufferPointer { buf in
                var start = 0
                let step = tunedMinChunk ?? minChunk(forDim: dim)
                while start < n {
                    let end = min(start + step, n)
                    let sub = UnsafeMutableBufferPointer<Float>(start: buf.baseAddress!.advanced(by: start), count: end - start)
                    BatchKernels.range_cosine_fused_768(query: query, candidates: candidates, range: start..<end, out: sub)
                    start = end
                }
            }
            return out
        } else {
            let chunks: [_Chunk] = try await computeProvider.parallelReduce(
                items: 0..<n,
                initial: [],
                minChunk: tunedMinChunk ?? minChunk(forDim: dim),
                { range in
                    var tmp = [Float](repeating: 0, count: range.count)
                    tmp.withUnsafeMutableBufferPointer { sub in
                        BatchKernels.range_cosine_fused_768(query: query, candidates: candidates, range: range, out: sub)
                    }
                    return [_Chunk(start: range.lowerBound, values: tmp)]
                },
                { $0 + $1 }
            )
            var out = [Float](repeating: 0, count: n)
            for ch in chunks { out.replaceSubrange(ch.start..<(ch.start + ch.values.count), with: ch.values) }
            return out
        }
    }

    private static func batchKernelDistances_cosineFused(
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized],
        dim: Int,
        parallel: Bool,
        tunedMinChunk: Int? = nil
    ) async throws -> [Float] {
        let n = candidates.count
        if !parallel {
            var out = [Float](repeating: 0, count: n)
            out.withUnsafeMutableBufferPointer { buf in
                var start = 0
                let step = tunedMinChunk ?? minChunk(forDim: dim)
                while start < n {
                    let end = min(start + step, n)
                    let sub = UnsafeMutableBufferPointer<Float>(start: buf.baseAddress!.advanced(by: start), count: end - start)
                    BatchKernels.range_cosine_fused_1536(query: query, candidates: candidates, range: start..<end, out: sub)
                    start = end
                }
            }
            return out
        } else {
            let chunks: [_Chunk] = try await computeProvider.parallelReduce(
                items: 0..<n,
                initial: [],
                minChunk: tunedMinChunk ?? minChunk(forDim: dim),
                { range in
                    var tmp = [Float](repeating: 0, count: range.count)
                    tmp.withUnsafeMutableBufferPointer { sub in
                        BatchKernels.range_cosine_fused_1536(query: query, candidates: candidates, range: range, out: sub)
                    }
                    return [_Chunk(start: range.lowerBound, values: tmp)]
                },
                { $0 + $1 }
            )
            var out = [Float](repeating: 0, count: n)
            for ch in chunks { out.replaceSubrange(ch.start..<(ch.start + ch.values.count), with: ch.values) }
            return out
        }
    }

    /// Compute pairwise distance matrix
    ///
    /// - Parameters:
    ///   - setA: First vector set
    ///   - setB: Second vector set
    ///   - metric: Distance metric
    /// - Returns: Distance matrix [setA.count x setB.count]
    public static func distanceMatrix<V: VectorProtocol, M: DistanceMetric>(
        between setA: [V],
        and setB: [V],
        metric: M = EuclideanDistance()
    ) async throws -> [[Float]] where V.Scalar == Float, M.Scalar == Float {
        // If either set is empty, return an empty matrix with appropriate shape
        if setA.isEmpty || setB.isEmpty { return Array(repeating: [], count: setA.count) }
        let aDim = setA[0].scalarCount
        let bDim = setB[0].scalarCount
        // Validate dimensions within each set
        for v in setA where v.scalarCount != aDim {
            throw VectorError.dimensionMismatch(expected: aDim, actual: v.scalarCount)
        }
        for v in setB where v.scalarCount != bDim {
            throw VectorError.dimensionMismatch(expected: bDim, actual: v.scalarCount)
        }
        // Cross-set dimension match
        guard aDim == bDim else {
            throw VectorError.dimensionMismatch(expected: bDim, actual: aDim)
        }
        return try await computeProvider.parallelExecute(items: 0..<setA.count) { i in
            setB.map { metric.distance(setA[i], $0) }
        }
    }

    // MARK: - Vector Operations

    /// Compute the component-wise mean: cᵢ = Σⱼvⱼᵢ / N.
    ///
    /// Uses O(N(D + 1)) work and O(D) auxiliary scalar storage with Float accumulation.
    /// - Parameter vectors: Nonempty collection with equal scalar counts. Zero-dimensional rows are valid.
    /// - Returns: Centroid vector.
    /// - Throws: `.insufficientData` for empty input, `.dimensionMismatch` for inconsistent
    ///   dimensions or array counts, or the unchanged error from `V.create(from:)`.
    public static func centroid<V: VectorProtocol & VectorFactory>(
        of vectors: [V]
    ) throws -> V where V.Scalar == Float {
        guard let first = vectors.first else {
            throw VectorError(.insufficientData, message: "Centroid requires at least one vector")
        }
        let dimension = first.scalarCount
        for vector in vectors where vector.scalarCount != dimension {
            throw VectorError.dimensionMismatch(expected: dimension, actual: vector.scalarCount)
        }
        var sum = [Float](repeating: 0, count: dimension)
        try sum.withUnsafeMutableBufferPointer { sumBuffer in
            for vector in vectors {
                let values = vector.toArray()
                guard values.count == dimension else {
                    throw VectorError.dimensionMismatch(expected: dimension, actual: values.count)
                }
                // Empty arrays need no scalar pointers or provider work.
                guard dimension > 0 else { continue }
                values.withUnsafeBufferPointer { valuesBuffer in
                    SwiftFloatSIMDProvider.add(
                        sumBuffer.baseAddress!,
                        valuesBuffer.baseAddress!,
                        result: sumBuffer.baseAddress!,
                        count: dimension
                    )
                }
            }
        }
        if dimension == 0 { return try V.create(from: []) }
        let scale = 1.0 / Float(vectors.count)
        let result = simdProvider.multiply(sum, by: scale)
        return try V.create(from: result)
    }

    /// Normalize vectors to unit length
    ///
    /// - Parameter vectors: Vectors to normalize
    /// Each row is normalized independently; dimensions may differ. Empty input returns empty.
    /// - Returns: Normalized vectors
    /// - Throws: Provider failures or the unchanged error from `V.create(from:)`.
    public static func normalize<V: VectorProtocol & VectorFactory>(
        _ vectors: [V]
    ) async throws -> [V] where V.Scalar == Float {
        try await computeProvider.parallelExecute(items: 0..<vectors.count) { i in
            let values = vectors[i].toArray()
            let normalized = simdProvider.normalize(values)
            return try V.create(from: normalized)
        }
    }

    // MARK: - Statistics

    /// Compute statistics for vector collection
    ///
    /// - Parameter vectors: Vector collection
    /// Empty input has count/dimensions zero; zero-dimensional rows retain their count
    /// with empty component arrays and zero magnitude summaries.
    /// Uses Float accumulation; numerical range behavior is unchanged.
    /// A NaN component makes that column's `min` and `max` NaN (as its `mean`), and a NaN magnitude makes
    /// `magnitudes.min` and `magnitudes.max` NaN (T16 reduction NaN contract; T16ReductionExceptionalTableTests).
    /// - Returns: Statistical summary
    /// - Throws: `.dimensionMismatch` if scalar counts or materialized array counts differ.
    public static func statistics<V: VectorProtocol>(
        for vectors: [V]
    ) throws -> VectorStatistics where V.Scalar == Float {
        guard !vectors.isEmpty else {
            return VectorStatistics(
                count: 0,
                dimensions: 0,
                mean: [],
                min: [],
                max: [],
                magnitudes: VectorStatistics.MagnitudeStats(min: 0, max: 0, mean: 0)
            )
        }

        let dimension = vectors.first!.scalarCount
        for vector in vectors where vector.scalarCount != dimension {
            throw VectorError.dimensionMismatch(expected: dimension, actual: vector.scalarCount)
        }
        var mins = [Float](repeating: .infinity, count: dimension)
        var maxs = [Float](repeating: -.infinity, count: dimension)
        var sums = [Float](repeating: 0, count: dimension)
        var magnitudes: [Float] = []

        for vector in vectors {
            let values = vector.toArray()
            guard values.count == dimension else {
                throw VectorError.dimensionMismatch(expected: dimension, actual: values.count)
            }

            // Update component-wise stats
            for i in 0..<dimension {
                mins[i] = ReductionNaN.propagatingMin(mins[i], values[i])
                maxs[i] = ReductionNaN.propagatingMax(maxs[i], values[i])
                sums[i] += values[i]
            }

            // Track magnitude
            magnitudes.append(dimension == 0 ? 0 : simdProvider.magnitude(values))
        }

        let count = Float(vectors.count)
        let mean = sums.map { $0 / count }

        // `Array.min()`/`max()` keep a NaN only at index 0; a NaN magnitude makes both extremes NaN (T16).
        let magnitudeHasNaN = magnitudes.contains { $0.isNaN }
        let magMin = magnitudeHasNaN ? Float.nan : (magnitudes.min() ?? 0)
        let magMax = magnitudeHasNaN ? Float.nan : (magnitudes.max() ?? 0)
        let magMean = magnitudes.reduce(0, +) / count

        return VectorStatistics(
            count: vectors.count,
            dimensions: dimension,
            mean: mean,
            min: mins,
            max: maxs,
            magnitudes: VectorStatistics.MagnitudeStats(min: magMin, max: magMax, mean: magMean)
        )
    }
}

// MARK: - Result Types

/// Result of nearest neighbor search
public struct NearestNeighborResult: Sendable, Equatable {
    public let index: Int
    public let distance: Float

    public init(index: Int, distance: Float) {
        self.index = index
        self.distance = distance
    }
}

/// Statistical summary of vector collection
public struct VectorStatistics: Sendable {
    public let count: Int
    public let dimensions: Int
    public let mean: [Float]
    public let min: [Float]
    public let max: [Float]
    public let magnitudes: MagnitudeStats

    public struct MagnitudeStats: Sendable {
        public let min: Float
        public let max: Float
        public let mean: Float
    }
}
