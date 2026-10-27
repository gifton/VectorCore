// VectorCore: Batch Operations
//
// Modern async-first batch processing for vector operations
//

import Foundation

/// Contract: async batch operations use one live configuration snapshot per invocation.
/// Contract: threshold settings select bounded library workers; nested callbacks
/// execute inline. Configuration changes affect later invocations only.
/// `sample(_:k:)` remains synchronous and configuration-independent.
///
/// Linear process/map/filter/statistics/nearest operations use `parallelThreshold`.
/// Pairwise distances retain a separate N100 row gate and their two GEMM settings.
/// Minimum chunk size and oversubscription control parallel nearest/statistics
/// logical partitions, not the hardware-capped worker count.
///
/// ```swift
/// try await BatchOperations.updateConfiguration { configuration in
///     configuration.parallelThreshold = 500
///     configuration.minimumChunkSize = 128
/// }
/// ```
public enum BatchOperations {

    /// Configuration for batch processing behavior
    public struct Configuration: Sendable {
        /// Positive minimum input count for process/map/filter/statistics/nearest parallel routes.
        public var parallelThreshold: Int = 1000

        /// Positive finite target logical chunks per active core for parallel nearest/statistics.
        /// Does not increase the worker limit; publication checks representability.
        public var oversubscription: Double = 2.0

        /// Positive minimum logical chunk size for parallel nearest/statistics, clamped to input count.
        public var minimumChunkSize: Int = 256

        /// Positive default callback batch size for process when its argument is nil.
        public var defaultBatchSize: Int = 1024

        /// Route large pairwise distance matrices through GEMM (`cblas_sgemm`)
        /// for supported optimized vector types and Euclidean/cosine metrics; off
        /// by default. The GEMM identity ‖q‖² + ‖c‖² − 2⟨q, c⟩ is a different
        /// arithmetic from the per-pair kernels with no universal relative-error
        /// bound, and it loses the offset of near-duplicate rows at large
        /// magnitude (the offset-1000 rows, as for `findNearestBatch`), so the
        /// default keeps `pairwiseDistances` on its per-pair path at every size
        /// (`P02PairwiseRowTests.defaultConfigurationUsesPerPairKernelsAtAnEligibleN`).
        /// Setting this true routes matrices with at least `matrixRoutingMinN`
        /// rows. It does not affect `Operations.findNearestBatch`, which selects
        /// Gram search only through its `cpuStrategy` parameter
        /// (`P02GramRouteTests.matrixRoutingFlagDoesNotAffectBatchSearch`).
        public var enableMatrixRouting: Bool = false

        /// Positive minimum N per side for eligible pairwise GEMM routing.
        /// This is a routing policy, not a universal performance crossover.
        public var matrixRoutingMinN: Int = 256

        public init() {}
    }

    /// Publication is serialized by the actor's synchronous candidate transaction.
    private static let _configuration = ThreadSafeConfiguration(Configuration())

    /// Get configuration
    public static func configuration() async -> Configuration {
        await _configuration.get()
    }

    /// Atomically updates the configuration, publishing all value fields on success.
    /// Contract: the user closure runs first; its error propagates unchanged.
    /// Complete-candidate validation then rejects the first invalid field in declaration
    /// order with invalidConfiguration. Either failure preserves all six old fields.
    /// See `ConfigurationTransactionTests.batchRollbackPreservesEveryFieldAndErrorIdentity`.
    /// - Parameter update: Synchronous mutation with immutable Sendable captures.
    ///   Capture a value snapshot instead of mutable non-Sendable state.
    public static func updateConfiguration(
        _ update: @Sendable (inout Configuration) throws -> Void
    ) async throws {
        try await _configuration.modify { candidate in
            try update(&candidate)
            try validate(candidate)
        }
    }

    private static func validate(_ candidate: Configuration) throws {
        func reject(_ field: String) -> VectorError {
            VectorError(.invalidConfiguration, message: "Invalid batch configuration field: \(field)")
        }
        guard candidate.parallelThreshold > 0 else { throw reject("parallelThreshold") }
        let factor = candidate.oversubscription
        let target = Double(max(1, ProcessInfo.processInfo.processorCount)) * factor
        guard factor.isFinite, factor > 0, target.isFinite,
              target <= Double(Int.max).nextDown else { throw reject("oversubscription") }
        guard candidate.minimumChunkSize > 0 else { throw reject("minimumChunkSize") }
        guard candidate.defaultBatchSize > 0 else { throw reject("defaultBatchSize") }
        guard candidate.matrixRoutingMinN > 0 else { throw reject("matrixRoutingMinN") }
    }

    /// Observation-only logical phases; no scheduling decision depends on the observer.
    private struct LogicalObservation: Sendable {
        let observer: (@Sendable (BoundedExecution.Event) -> Void)?
        let id: UUID?
        let phase: String
        let cores: Int
        init(phase: String, cores: Int) {
            self.observer = BoundedExecution.observation
            self.id = observer == nil ? nil : UUID()
            self.phase = phase
            self.cores = cores
            emit(.begin)
        }
        func emit(_ kind: BoundedExecution.Event.Kind, index: Int = -1) {
            if let id { observer?(BoundedExecution.Event(id: id, phase: phase, kind: kind, worker: index, limit: cores)) }
        }
        func range(_ range: Range<Int>, index: Int) { emit(.logicalRange(range), index: index) }
        func finish() { emit(.end) }
    }

    // MARK: - Core Operations

    /// Find k nearest neighbors with automatic parallelization
    ///
    /// Contract: the entry snapshot selects the parallel route at parallelThreshold.
    /// Parallel logical partitions use its minimumChunkSize and oversubscription.
    ///
    /// Computed NaN distances rank after numeric distances. Exact ties, including
    /// signed zeros and NaN pairs, use original input indices. Results contain
    /// `min(k, vectors.count)` entries for positive k, retaining NaNs when needed.
    /// See `TopKNaNWrapperTests.batchSerialHeapAndSortKeepNaNsLast` and
    /// `TopKNaNWrapperTests.batchParallelHeapAndSortPreserveGlobalIndices`.
    ///
    /// Selection, with `n = vectors.count` and `k' = min(k, n)`, is decided once at entry by
    /// `TopKSelection.usesSmallKStrategy(actualK: k', count: n)`, that is `k' < n / 10`
    /// (`P01BatchRouteTests.strategyNeighbors`):
    /// - Small k: each candidate is scored once and pushed into a bounded heap; only the retained entries are
    ///   sorted. Generic metric calls are counted by `P01BatchRouteTests.genericPartitions`; the optimized index
    ///   stream by `P01BatchTilingParityTests`. A serial run uses one heap of capacity `k'`. A parallel run uses
    ///   one heap per worker partition, of capacity `min(k', candidates assigned to that worker)`, and merges them
    ///   in partition order. The six optimized routes (Euclidean or cosine on `Vector512Optimized`,
    ///   `Vector768Optimized` or `Vector1536Optimized`) score with the same `BatchKernels` range helpers as the
    ///   large-k path, on tiles of at most 256 candidates that restart at each original range's lower bound. Their
    ///   complete per-candidate scores are bit-equal to the untiled helpers on the fixtures of
    ///   `P01BatchTilingParityTests`. Work: `n` scores and `n` heap pushes, plus at most `2·(P - 1)·k'` pushes to
    ///   merge `P` partial heaps (`combineInOrder`), each push O(log k'), then a sort of `k'` entries.
    /// - Otherwise: all `n` scores and `n` (index, distance) pairs are built and the pairs are sorted, as before
    ///   this split: O(n log n) comparisons.
    ///
    /// Batch Euclidean ranks the computed root distance it returns, so two distinct squared distances whose
    /// roots round to the same value tie and keep the smaller index (`P01BatchRouteTests.rootTieAdmission`).
    /// `Operations.findNearest`'s direct Euclidean routes rank squared distances instead (`range_topk_euclid2_*`):
    /// `P01DirectEuclideanAdmissionTests.squaredAdmissionAtRootTiedBoundaries` tells the two admissions apart on
    /// root-tied pairs at the k boundary, and `P01OperationsRouteTests.extremePartitions` checks those routes against
    /// a squared-score oracle on fixtures without such pairs.
    ///
    /// Temporary storage model, excluding the inputs and the result: small k keeps at most `C` partial heaps and
    /// `C` score tiles, where `C` is the active processor count captured at entry, so O(C·k' + C·256 + k');
    /// large k keeps O(n) scores and pairs. This is a model, not a measured bound, and no allocation-free claim is
    /// made. Partial heap and tile capacities are observed by the six `P01BatchRouteTests.optimized{Euclid,Cosine}`
    /// `{512,768,1536}` tests and by `P01BatchRouteTests.genericPartitions`. Outside this model, the six optimized
    /// routes take the candidate array as `[Vector{D}Optimized]` with `exactArray(vectors, as:)` (`smallKRoute`, and
    /// the large-k helpers), a type-identity test that returns the caller's array value, sharing its buffer
    /// (`P02ArrayIdentityTests.sharesTheSourceBufferOnSuccess`). Until P02 phase E this was the conditional cast
    /// `vectors as? [Vector{D}Optimized]`, for which one Release measurement on one host recorded a 16·n + 32 B
    /// request per small-k call, attributed by layout and size, not by allocation site. With `exactArray`, P02 phase
    /// G's allocation probe (one host, not a test) recorded one fewer such request per call, no new 8·n + 32 B
    /// request and the same result bits on the routes it measured here: Euclidean on `Vector512Optimized`, serial,
    /// parallel and minimum-chunk, k 3 and k = n. Other widths and metrics of these routes were not measured.
    ///
    /// - Parameters:
    ///   - query: The query vector
    ///   - vectors: Array of vectors to search
    ///   - k: Number of nearest neighbors to find
    ///   - metric: Distance metric to use (default: Euclidean)
    /// - Returns: Array of (index, distance) tuples sorted by distance
    /// Cancellation does not truncate this nonthrowing result (ExecutionSchedulingContractTests,
    /// `P01BatchRouteTests.cancelledAndNested`).
    public static func findNearest<V: VectorProtocol & VectorProtocol & Sendable, M: DistanceMetric>(
        to query: V,
        in vectors: [V],
        k: Int,
        metric: M = EuclideanDistance()
    ) async -> [(index: Int, distance: Float)] where V.Scalar == M.Scalar, M.Scalar == Float {
        let config = await _configuration.get()
        let coreCount = BoundedExecution.hardwareLimit
        guard k > 0, !vectors.isEmpty else { return [] }
        let parallel = vectors.count >= config.parallelThreshold
        let actualK = min(k, vectors.count)

        if TopKSelection.usesSmallKStrategy(actualK: actualK, count: vectors.count) {
            let route = smallKRoute(query: query, vectors: vectors, metric: metric)
            if parallel {
                return await streamNearestParallel(route: route, count: vectors.count, k: actualK,
                    chunkSize: logicalChunkSize(count: vectors.count, cores: coreCount, configuration: config), cores: coreCount)
            }
            return BoundedExecution.$isInsideRegion.withValue(true) { streamNearestSerial(route: route, count: vectors.count, k: actualK) }
        }

        if parallel {
            return await findNearestParallel(to: query, in: vectors, k: k, metric: metric,
                chunkSize: logicalChunkSize(count: vectors.count, cores: coreCount, configuration: config), cores: coreCount)
        } else {
            return BoundedExecution.$isInsideRegion.withValue(true) { findNearestSerial(to: query, in: vectors, k: k, metric: metric) }
        }
    }

    /// Process successive contiguous batches
    ///
    /// Automatically parallelizes for large datasets. Batch results are flattened in
    /// input batch order (ExecutionBoundaryTests.testDangerousBoundariesInChildren).
    ///
    /// - Parameters:
    ///   - vectors: Array of vectors to process
    ///   - batchSize: Positive size of each batch; nil uses the entry snapshot's live defaultBatchSize.
    ///   - transform: Transformation to apply to each batch
    /// - Returns: Concatenated results maintaining batch order; empty input invokes no transform.
    /// - Throws: `.invalidConfiguration` for nonpositive batchSize, even for empty input,
    ///   or the original transform error.
    public static func process<V: VectorProtocol & Sendable, R: Sendable>(
        _ vectors: [V],
        batchSize: Int? = nil,
        transform: @Sendable @escaping ([V]) throws -> [R]
    ) async throws -> [R] {
        let config = await _configuration.get()
        let coreCount = BoundedExecution.hardwareLimit
        let effectiveBatchSize = batchSize ?? config.defaultBatchSize
        guard effectiveBatchSize > 0 else {
            throw VectorError(.invalidConfiguration, message: "batchSize must be positive")
        }
        let batches = ComputeRangeBoundary.ceilingDivide(vectors.count, by: effectiveBatchSize)
        let results = try await BoundedExecution.map(0..<batches, limit: coreCount,
            parallel: vectors.count >= config.parallelThreshold, phase: "batch.process") { index in
            let start = index * effectiveBatchSize
            let end = start + min(effectiveBatchSize, vectors.count - start)
            return try transform(Array(vectors[start..<end]))
        }
        let flattened = results.flatMap { $0 }
        try Task.checkCancellation()
        return flattened
    }

    /// Complete-result operation even when cancelled; workers are bounded per phase.
    /// ExecutionSchedulingContractTests covers its scheduling policy.
    /// Compute the pairwise distance matrix (entry `[i][j]` is the distance between `vectors[i]` and `vectors[j]`)
    ///
    /// Contract: the non-GEMM path uses bounded row workers at N >= 100,
    /// independently of the linear-operation parallelThreshold setting.
    ///
    /// - Parameters:
    ///   - vectors: Vectors to compute distances for
    ///   - metric: Distance metric to use
    /// - Returns: The N×N distance matrix, with the supplied metric's self-distance on each diagonal. It is symmetric
    ///   up to rounding, not bitwise: the GEMM route computes each row with its own one-query SGEMM, so `[i][j]` and
    ///   `[j][i]` come from different calls, and for Euclidean distance their last bits can differ (on one test
    ///   host, for example, 221 of the 32,385 pairs at width 512 and N 255; P02 phase D-F review A, observation O-1;
    ///   no test asserts either way). The serial route (fewer than 100 vectors, when the GEMM route does not apply)
    ///   computes the upper triangle and mirrors it (`computePairwiseSerial`).
    /// - Precondition: Custom metrics must be symmetric.
    public static func pairwiseDistances<V: VectorProtocol & VectorProtocol & Sendable, M: DistanceMetric>(
        _ vectors: [V],
        metric: M = EuclideanDistance()
    ) async -> [[Float]] where V.Scalar == M.Scalar, M.Scalar == Float {
        let config = await _configuration.get()
        let coreCount = BoundedExecution.hardwareLimit
        let n = vectors.count
        guard n > 0 else { return [] }

        // Concrete metric/type eligibility remains separate from the configured gate.
        if config.enableMatrixRouting, n >= config.matrixRoutingMinN,
           var routed = gemmPairwise(vectors, metric: metric, cores: coreCount) {
            // Preserve direct self-score semantics despite GEMM rounding differences.
            BoundedExecution.$isInsideRegion.withValue(true) {
                for i in 0..<n { routed[i][i] = metric.distance(vectors[i], vectors[i]) }
            }
            return routed
        }

        // Serial for small matrices
        if n < 100 {
            return BoundedExecution.$isInsideRegion.withValue(true) { computePairwiseSerial(vectors, metric: metric) }
        }

        // Parallel block-based computation
        return await computePairwiseParallel(vectors, metric: metric, cores: coreCount)
    }

    /// GEMM pairwise dispatch: returns the `[[Float]]` distance matrix when `vectors`
    /// is a known optimized type and `metric` is Euclidean or Cosine; otherwise `nil`
    /// (caller falls back to the per-pair path).
    private static func gemmPairwise<V: VectorProtocol, M: DistanceMetric>(
        _ vectors: [V], metric: M, cores: Int
    ) -> [[Float]]? {
        let isEuclid = metric is EuclideanDistance
        let isCosine = metric is CosineDistance
        guard isEuclid || isCosine else { return nil }

        guard exactArray(vectors, as: Vector512Optimized.self) != nil
                || exactArray(vectors, as: Vector768Optimized.self) != nil
                || exactArray(vectors, as: Vector1536Optimized.self) != nil else { return nil }
        let observation = LogicalObservation(phase: "batch.pairwise.gemm", cores: cores)
        defer { observation.finish() }

        if let vs = exactArray(vectors, as: Vector512Optimized.self) { return gemmPairwiseTyped(vs, euclid: isEuclid) }
        if let vs = exactArray(vectors, as: Vector768Optimized.self) { return gemmPairwiseTyped(vs, euclid: isEuclid) }
        if let vs = exactArray(vectors, as: Vector1536Optimized.self) { return gemmPairwiseTyped(vs, euclid: isEuclid) }
        return nil
    }

    /// Compute the pairwise matrix row by row through `MatrixDistance`'s prepared/into path (P02 RD10): the candidates
    /// are packed (and, for cosine, normalized) once with `prepare`, then each query row is computed straight into its
    /// final N-element row, and Euclidean rows are formatted in place (a NaN squared score stays NaN, as on the per-pair
    /// route and in `GramBatchSearch`; otherwise `v > 0 ? sqrt(v) : 0`; `P02PairwiseRowTests.nanRowsAtAnEligibleN`). No
    /// flat N·N score buffer is built. Each row is its own one-query SGEMM, so off-diagonal bits can differ from a
    /// single N×N multiply (`P02PairwiseRowTests.rowsEqualPreparedIntoPerRow`).
    private static func gemmPairwiseTyped<O: UnifiedVectorBuffer>(
        _ vs: [O], euclid: Bool
    ) -> [[Float]] {
        let n = vs.count
        let prepared = MatrixDistance.prepare(vs, normalized: !euclid)
        var rows = [[Float]]()
        rows.reserveCapacity(n)
        for i in 0..<n {
            // swiftlang/swift#92562 (P02 R-LS4): the row comes from [Float](repeating:count:), never an array literal.
            var row = [Float](repeating: 0, count: n)
            if euclid {
                MatrixDistance.euclideanSquaredMatrix(queries: [vs[i]], prepared: prepared, into: &row)
                for j in 0..<n { let v = row[j]; row[j] = v.isNaN ? v : (v > 0 ? sqrt(v) : 0) }
            } else {
                MatrixDistance.cosineDistanceMatrix(queries: [vs[i]], prepared: prepared, into: &row)
            }
            rows.append(row)
        }
        return rows
    }

    // MARK: - Convenience Operations

    /// Transform vectors with automatic parallelization
    public static func map<V: VectorProtocol & Sendable, U: VectorProtocol & Sendable>(
        _ vectors: [V],
        transform: @Sendable @escaping (V) throws -> U
    ) async throws -> [U] {
        let config = await _configuration.get()
        let coreCount = BoundedExecution.hardwareLimit
        return try await BoundedExecution.map(0..<vectors.count, limit: coreCount,
            parallel: vectors.count >= config.parallelThreshold, phase: "batch.map") { index in
            try transform(vectors[index])
        }
    }

    /// Filter vectors with parallel processing
    public static func filter<V: VectorProtocol & Sendable>(
        _ vectors: [V],
        predicate: @Sendable @escaping (V) throws -> Bool
    ) async throws -> [V] {
        let config = await _configuration.get()
        let coreCount = BoundedExecution.hardwareLimit
        let retained = try await BoundedExecution.map(0..<vectors.count, limit: coreCount,
            parallel: vectors.count >= config.parallelThreshold, phase: "batch.filter") { index in
            try predicate(vectors[index]) ? vectors[index] : nil
        }
        let result = retained.compactMap { $0 }
        try Task.checkCancellation()
        return result
    }

    /// Complete-result operation even when cancelled (ExecutionSchedulingContractTests).
    /// Population statistics of the Float magnitudes delivered by each vector.
    /// Each magnitude is evaluated once. Empty input returns positive zeros;
    /// any nonfinite magnitude produces two NaNs while retaining the full count.
    /// Reduction uses centered Double statistics, with chunk-index merge order.
    public static func statistics<V: VectorProtocol & Sendable>(
        for vectors: [V]
    ) async -> BatchStatistics where V.Scalar == Float {
        let config = await _configuration.get()
        let coreCount = BoundedExecution.hardwareLimit
        guard !vectors.isEmpty else {
            return BatchStatistics(count: 0, meanMagnitude: 0, stdMagnitude: 0)
        }
        if vectors.count < config.parallelThreshold {
            return BoundedExecution.$isInsideRegion.withValue(true) { statisticsSerial(for: vectors) }
        }
        return await statisticsParallel(for: vectors, chunkSize: logicalChunkSize(count: vectors.count, cores: coreCount, configuration: config), cores: coreCount)
    }

    /// Statistics-only partition control. Merge order is deterministic for a given
    /// input and positive chunk size; different partitions may round differently.
    internal static func statisticsParallel<V: VectorProtocol & Sendable>(
        for vectors: [V], chunkSize: Int
    ) async -> BatchStatistics where V.Scalar == Float {
        await statisticsParallel(for: vectors, chunkSize: chunkSize, cores: BoundedExecution.hardwareLimit)
    }

    private static func statisticsParallel<V: VectorProtocol & Sendable>(
        for vectors: [V], chunkSize: Int, cores coreCount: Int
    ) async -> BatchStatistics where V.Scalar == Float {
        precondition(chunkSize > 0, "Statistics chunk size must be positive")
        let count = ComputeRangeBoundary.ceilingDivide(vectors.count, by: chunkSize)
        let observation = LogicalObservation(phase: "batch.statistics.logical", cores: coreCount)
        defer { observation.finish() }
        let partials = await BoundedExecution.completeMap(0..<count, limit: coreCount,
            phase: "batch.statistics") { index in
            let start = index * chunkSize
            let end = start + min(chunkSize, vectors.count - start)
            observation.range(start..<end, index: index)
            var partial = MagnitudeStatisticsAccumulator()
            for vector in vectors[start..<end] { partial.add(vector.magnitude) }
            return partial
        }
        // Retain N03 logical chunk boundaries and ascending chunk-index merge order.
        var total = MagnitudeStatisticsAccumulator()
        for partial in partials { total.merge(partial) }
        return total.statistics
    }

    /// Synchronous random sampling, independent of batch configuration.
    public static func sample<V>(_ vectors: [V], k: Int) -> [V] {
        guard k > 0 && k <= vectors.count else {
            return k <= 0 ? [] : vectors
        }

        var indices = Array(0..<vectors.count)
        indices.shuffle()

        return indices.prefix(k).map { vectors[$0] }
    }

    // MARK: - Private Helpers

    private static func findNearestSerial<V: VectorProtocol, M: DistanceMetric>(
        to query: V,
        in vectors: [V],
        k: Int,
        metric: M
    ) -> [(index: Int, distance: Float)] where V.Scalar == M.Scalar, M.Scalar == Float {
        // Optimized fast path for Vector*Optimized with Euclidean/Cosine using BatchKernels
        if let e = metric as? EuclideanDistance {
            _ = e // silence unused
            if let q = query as? Vector512Optimized, let c = exactArray(vectors, as: Vector512Optimized.self) {
                let dists = computeDistances_euclid_optimized_serial(query: q, candidates: c, dim: 512)
                let pairs = dists.enumerated().map { (index: $0.offset, distance: $0.element) }
                return heapSelect(pairs, k: k)
            }
            if let q = query as? Vector768Optimized, let c = exactArray(vectors, as: Vector768Optimized.self) {
                let dists = computeDistances_euclid_optimized_serial(query: q, candidates: c, dim: 768)
                let pairs = dists.enumerated().map { (index: $0.offset, distance: $0.element) }
                return heapSelect(pairs, k: k)
            }
            if let q = query as? Vector1536Optimized, let c = exactArray(vectors, as: Vector1536Optimized.self) {
                let dists = computeDistances_euclid_optimized_serial(query: q, candidates: c, dim: 1536)
                let pairs = dists.enumerated().map { (index: $0.offset, distance: $0.element) }
                return heapSelect(pairs, k: k)
            }
        } else if let cMetric = metric as? CosineDistance {
            _ = cMetric
            if let q = query as? Vector512Optimized, let c = exactArray(vectors, as: Vector512Optimized.self) {
                let dists = computeDistances_cosine_fused_serial(query: q, candidates: c, dim: 512)
                let pairs = dists.enumerated().map { (index: $0.offset, distance: $0.element) }
                return heapSelect(pairs, k: k)
            }
            if let q = query as? Vector768Optimized, let c = exactArray(vectors, as: Vector768Optimized.self) {
                let dists = computeDistances_cosine_fused_serial(query: q, candidates: c, dim: 768)
                let pairs = dists.enumerated().map { (index: $0.offset, distance: $0.element) }
                return heapSelect(pairs, k: k)
            }
            if let q = query as? Vector1536Optimized, let c = exactArray(vectors, as: Vector1536Optimized.self) {
                let dists = computeDistances_cosine_fused_serial(query: q, candidates: c, dim: 1536)
                let pairs = dists.enumerated().map { (index: $0.offset, distance: $0.element) }
                return heapSelect(pairs, k: k)
            }
        }

        // Generic path
        let distances = vectors.enumerated().map { index, vector in
            (index: index, distance: metric.distance(query, vector))
        }

        return heapSelect(distances, k: k)
    }

    private static func findNearestParallel<V: VectorProtocol & Sendable, M: DistanceMetric>(
        to query: V,
        in vectors: [V],
        k: Int,
        metric: M, chunkSize: Int, cores coreCount: Int
    ) async -> [(index: Int, distance: Float)] where V.Scalar == M.Scalar, M.Scalar == Float {
        // Optimized fast path: compute distances via BatchKernels in parallel chunks
        if let _ = metric as? EuclideanDistance {
            if let q = query as? Vector512Optimized, let all = exactArray(vectors, as: Vector512Optimized.self) {
                let dists = await computeDistances_euclid_optimized_parallel(query: q, candidates: all, dim: 512, chunkSize: chunkSize, cores: coreCount)
                let pairs = dists.enumerated().map { (index: $0.offset, distance: $0.element) }
                return heapSelect(pairs, k: k)
            }
            if let q = query as? Vector768Optimized, let all = exactArray(vectors, as: Vector768Optimized.self) {
                let dists = await computeDistances_euclid_optimized_parallel(query: q, candidates: all, dim: 768, chunkSize: chunkSize, cores: coreCount)
                let pairs = dists.enumerated().map { (index: $0.offset, distance: $0.element) }
                return heapSelect(pairs, k: k)
            }
            if let q = query as? Vector1536Optimized, let all = exactArray(vectors, as: Vector1536Optimized.self) {
                let dists = await computeDistances_euclid_optimized_parallel(query: q, candidates: all, dim: 1536, chunkSize: chunkSize, cores: coreCount)
                let pairs = dists.enumerated().map { (index: $0.offset, distance: $0.element) }
                return heapSelect(pairs, k: k)
            }
        } else if let _ = metric as? CosineDistance {
            if let q = query as? Vector512Optimized, let all = exactArray(vectors, as: Vector512Optimized.self) {
                let dists = await computeDistances_cosine_fused_parallel(query: q, candidates: all, dim: 512, chunkSize: chunkSize, cores: coreCount)
                let pairs = dists.enumerated().map { (index: $0.offset, distance: $0.element) }
                return heapSelect(pairs, k: k)
            }
            if let q = query as? Vector768Optimized, let all = exactArray(vectors, as: Vector768Optimized.self) {
                let dists = await computeDistances_cosine_fused_parallel(query: q, candidates: all, dim: 768, chunkSize: chunkSize, cores: coreCount)
                let pairs = dists.enumerated().map { (index: $0.offset, distance: $0.element) }
                return heapSelect(pairs, k: k)
            }
            if let q = query as? Vector1536Optimized, let all = exactArray(vectors, as: Vector1536Optimized.self) {
                let dists = await computeDistances_cosine_fused_parallel(query: q, candidates: all, dim: 1536, chunkSize: chunkSize, cores: coreCount)
                let pairs = dists.enumerated().map { (index: $0.offset, distance: $0.element) }
                return heapSelect(pairs, k: k)
            }
        }

        let count = ComputeRangeBoundary.ceilingDivide(vectors.count, by: chunkSize)
        let observation = LogicalObservation(phase: "batch.nearest.logical", cores: coreCount)
        defer { observation.finish() }
        let parts = await BoundedExecution.completeMap(0..<count, limit: coreCount,
            phase: "batch.nearest") { index in
            let start = index * chunkSize
            let end = start + min(chunkSize, vectors.count - start)
            observation.range(start..<end, index: index)
            return (start..<end).map { (index: $0, distance: metric.distance(query, vectors[$0])) }
        }
        let distances = parts.flatMap { $0 }
        return heapSelect(distances, k: k)
    }

    private static func computePairwiseSerial<V: VectorProtocol, M: DistanceMetric>(
        _ vectors: [V],
        metric: M
    ) -> [[Float]] where V.Scalar == M.Scalar, M.Scalar == Float {
        let n = vectors.count
        var distances = Array(repeating: Array(repeating: Float(0), count: n), count: n)

        for i in 0..<n {
            distances[i][i] = metric.distance(vectors[i], vectors[i])
            for j in i+1..<n {
                let dist = metric.distance(vectors[i], vectors[j])
                distances[i][j] = dist
                distances[j][i] = dist
            }
        }

        return distances
    }

    private static func computePairwiseParallel<V: VectorProtocol & Sendable, M: DistanceMetric>(
        _ vectors: [V],
        metric: M, cores coreCount: Int
    ) async -> [[Float]] where V.Scalar == M.Scalar, M.Scalar == Float {
        let n = vectors.count
        return await BoundedExecution.completeMap(0..<n, limit: coreCount,
            phase: "batch.pairwise") { i in
            (0..<n).map { j in metric.distance(vectors[i], vectors[j]) }
        }
    }

    private static func statisticsSerial<V: VectorProtocol>(
        for vectors: [V]
    ) -> BatchStatistics where V.Scalar == Float {
        var accumulator = MagnitudeStatisticsAccumulator()
        for vector in vectors { accumulator.add(vector.magnitude) }
        return accumulator.statistics
    }

    /// Pure partition rule for validated settings and positive cores. Boundary coverage:
    /// BatchConfigurationContractTests.partitionArithmeticBoundaries.
    internal static func logicalChunkSize(count: Int, cores: Int, configuration: Configuration) -> Int {
        guard count > 0 else { return 0 }
        let target = Double(cores) * configuration.oversubscription
        let chunks = target >= Double(count) ? count : max(1, Int(target))
        let ideal = count / chunks + (count % chunks == 0 ? 0 : 1)
        return min(count, max(configuration.minimumChunkSize, ideal))
    }

    // MARK: - Small-k streaming selection

    /// Maximum candidates per `BatchKernels` score tile on the small-k optimized routes. Relative to
    /// `range.lowerBound`, every `BatchKernels` range helper scores candidates in pairs (`while i + 1 < end`),
    /// scores an odd last candidate with a single-vector kernel, and `applySqrt` takes roots in groups of four
    /// (`while i + 3 < count`). 256 is a multiple of 4, so a tile that starts at an original range's lower bound
    /// keeps each candidate's pair position, tail status and root group. Bit parity against the untiled helpers
    /// and the tile boundaries are checked by `P01BatchTilingParityTests.tiledScoresMatchUntiledHelpers`.
    internal static let smallKScoreTile = 256

    /// How the small-k route scores candidates.
    internal enum SmallKScorer: Sendable {
        /// One metric evaluation for candidate `i`, in the caller's generic context.
        case metric(@Sendable (Int) -> Float)
        /// A `BatchKernels` range helper. It writes `out[i - range.lowerBound]` for each `i` in `range`, the store
        /// form of every `BatchKernels.range_*` helper (the Euclidean root helpers' `applySqrt` then reads and
        /// writes `out[0..<range.count]`).
        case kernel(@Sendable (Range<Int>, UnsafeMutableBufferPointer<Float>) -> Void)
    }

    /// One small-k scoring route: the worker phase of the matching parallel route (`batch.nearest`,
    /// `batch.euclid{D}` or `batch.cosine_fused{D}`), the serial step of the matching serial route
    /// (`minChunk(forDim:)` on optimized routes, `Int.max` otherwise) and the scorer.
    internal struct SmallKRoute: Sendable {
        let phase: String
        let serialStep: Int
        let scorer: SmallKScorer
    }

    /// The small-k route for this call. The six optimized cases use the range helper that the large-k path's
    /// helpers use for the same metric and type; every other case calls `metric.distance(query, vectors[i])`.
    internal static func smallKRoute<V: VectorProtocol & Sendable, M: DistanceMetric>(
        query: V, vectors: [V], metric: M
    ) -> SmallKRoute where V.Scalar == M.Scalar, M.Scalar == Float {
        func tiled(_ phase: String, dim: Int,
                   _ kernel: @escaping @Sendable (Range<Int>, UnsafeMutableBufferPointer<Float>) -> Void) -> SmallKRoute {
            SmallKRoute(phase: phase, serialStep: minChunk(forDim: dim), scorer: .kernel(kernel))
        }
        if metric is EuclideanDistance {
            if let q = query as? Vector512Optimized, let c = exactArray(vectors, as: Vector512Optimized.self) {
                return tiled("batch.euclid512", dim: 512) { BatchKernels.range_euclid_512(query: q, candidates: c, range: $0, out: $1) }
            }
            if let q = query as? Vector768Optimized, let c = exactArray(vectors, as: Vector768Optimized.self) {
                return tiled("batch.euclid768", dim: 768) { BatchKernels.range_euclid_768(query: q, candidates: c, range: $0, out: $1) }
            }
            if let q = query as? Vector1536Optimized, let c = exactArray(vectors, as: Vector1536Optimized.self) {
                return tiled("batch.euclid1536", dim: 1536) { BatchKernels.range_euclid_1536(query: q, candidates: c, range: $0, out: $1) }
            }
        } else if metric is CosineDistance {
            if let q = query as? Vector512Optimized, let c = exactArray(vectors, as: Vector512Optimized.self) {
                return tiled("batch.cosine_fused512", dim: 512) {
                    BatchKernels.range_cosine_fused_512(query: q, candidates: c, range: $0, out: $1)
                }
            }
            if let q = query as? Vector768Optimized, let c = exactArray(vectors, as: Vector768Optimized.self) {
                return tiled("batch.cosine_fused768", dim: 768) {
                    BatchKernels.range_cosine_fused_768(query: q, candidates: c, range: $0, out: $1)
                }
            }
            if let q = query as? Vector1536Optimized, let c = exactArray(vectors, as: Vector1536Optimized.self) {
                return tiled("batch.cosine_fused1536", dim: 1536) {
                    BatchKernels.range_cosine_fused_1536(query: q, candidates: c, range: $0, out: $1)
                }
            }
        }
        return SmallKRoute(phase: "batch.nearest", serialStep: Int.max,
                           scorer: .metric { metric.distance(query, vectors[$0]) })
    }

    /// Pure partition of `count` candidates into logical chunks of `chunkSize` (clamped to `count`), and of the
    /// chunks into worker partitions by `BoundedExecution.completeMap`'s own grouping of `chunkCount` items under
    /// `limit: cores`: `width = ceil(L / min(L, C))` chunks per partition, `partitionCount = ceil(L / width)`, and
    /// partition `p` holds chunks `[p·width, min((p+1)·width, L))`. So `partitionCount <= min(L, C)`, and
    /// `completeMap(0..<partitionCount, limit: cores)` submits one task per partition when it spawns (outside a
    /// library region); nested calls run the partitions inline without tasks
    /// (`P01BatchRouteTests.cancelledAndNested`), and the serial plan never reaches `completeMap`. Chunk `c` holds
    /// candidates `[c·chunkSize, min((c+1)·chunkSize, count))`, as in the large-k parallel helpers. Nothing is
    /// stored per chunk.
    /// Arithmetic and grouping are checked by `P01BatchTilingParityTests.planMirrorsCompleteMap` and
    /// `P01BatchTilingParityTests.planArithmeticAtRepresentableLimits`.
    internal struct SmallKPlan: Equatable, Sendable {
        let count: Int
        let chunkSize: Int
        let chunkCount: Int
        let width: Int
        let partitionCount: Int

        init(count: Int, chunkSize: Int, cores: Int) {
            precondition(count > 0 && chunkSize > 0 && cores > 0, "Small-k plans need positive sizes")
            self.count = count
            self.chunkSize = min(chunkSize, count)
            chunkCount = ComputeRangeBoundary.ceilingDivide(count, by: self.chunkSize)
            width = ComputeRangeBoundary.ceilingDivide(chunkCount, by: min(chunkCount, cores))
            partitionCount = ComputeRangeBoundary.ceilingDivide(chunkCount, by: width)
        }

        /// The serial plan: the route's serial steps from 0, all in one partition.
        static func serial(route: SmallKRoute, count: Int) -> SmallKPlan {
            SmallKPlan(count: count, chunkSize: route.serialStep, cores: 1)
        }

        func candidates(ofChunk chunk: Int) -> Range<Int> {
            let start = chunk * chunkSize
            return start..<(start + min(chunkSize, count - start))
        }

        func chunks(ofPartition partition: Int) -> Range<Int> {
            let first = partition * width
            return first..<(first + min(width, chunkCount - first))
        }

        func candidates(ofPartition partition: Int) -> Range<Int> {
            let chunks = chunks(ofPartition: partition)
            return candidates(ofChunk: chunks.lowerBound).lowerBound..<candidates(ofChunk: chunks.upperBound - 1).upperBound
        }
    }

    /// Feeds the score of every candidate in the original range `range` to `sink` with its global index, in
    /// increasing index order. Kernel routes score the tiles `[t, min(t + smallKScoreTile, range.upperBound))` for
    /// `t = range.lowerBound, range.lowerBound + 256, ...` into the first slots of `scratch`, so a tile starts at the
    /// range's own lower bound and never crosses its upper bound. Metric routes call the metric once per candidate
    /// and ignore `scratch`. Observed by `P01BatchTilingParityTests.scoreRangeTilesAndMetricOrder` and by the tile
    /// log of `P01BatchTilingParityTests.tiledScoresMatchUntiledHelpers`.
    /// - Precondition: on kernel routes, `scratch.count >= min(smallKScoreTile, range.count)`.
    internal static func scoreRange(_ range: Range<Int>, route: SmallKRoute, scratch: UnsafeMutableBufferPointer<Float>,
                                    sink: (Float, Int) -> Void) {
        switch route.scorer {
        case .metric(let distance):
            for index in range { sink(distance(index), index) }
        case .kernel(let kernel):
            precondition(scratch.count >= min(smallKScoreTile, range.count), "Score tile scratch too small")
            var start = range.lowerBound
            while start < range.upperBound {
                let length = min(smallKScoreTile, range.upperBound - start)
                let tile = UnsafeMutableBufferPointer(rebasing: scratch[0..<length])
                kernel(start..<(start + length), tile)
                for offset in 0..<length { sink(tile[offset], start + offset) }
                start += length
            }
        }
    }

    /// Scores partition `partition` of `plan` chunk by chunk, in increasing chunk order: `observe` receives each
    /// chunk's candidate range and chunk index, then `scoreRange` scores that chunk from its own lower bound.
    /// A fresh scratch of `min(smallKScoreTile, plan.chunkSize, assigned candidates)` floats is used on kernel
    /// routes, none on metric routes. Returns the scratch capacity.
    @discardableResult
    internal static func scorePartition(_ partition: Int, plan: SmallKPlan, route: SmallKRoute,
                                        observe: (Range<Int>, Int) -> Void, sink: (Float, Int) -> Void) -> Int {
        var capacity = 0
        if case .kernel = route.scorer {
            capacity = min(smallKScoreTile, plan.chunkSize, plan.candidates(ofPartition: partition).count)
        }
        var scratch = [Float](repeating: 0, count: capacity)
        scratch.withUnsafeMutableBufferPointer { buffer in
            for chunk in plan.chunks(ofPartition: partition) {
                let range = plan.candidates(ofChunk: chunk)
                observe(range, chunk)
                scoreRange(range, route: route, scratch: buffer, sink: sink)
            }
        }
        return capacity
    }

    /// Test seam (B10): every (global index, score) that the small-k route feeds its heaps under `plan`, in
    /// partition order, produced by the same `scorePartition` calls as the production heaps. Runs inline.
    internal static func smallKScores(route: SmallKRoute, plan: SmallKPlan) -> [(index: Int, score: Float)] {
        var scores: [(index: Int, score: Float)] = []
        scores.reserveCapacity(plan.count)
        for partition in 0..<plan.partitionCount {
            scorePartition(partition, plan: plan, route: route, observe: { _, _ in },
                           sink: { scores.append((index: $1, score: $0)) })
        }
        return scores
    }

    /// Test seam (B16): records of the small-k route's partial heaps and final combines. Nil in ordinary
    /// execution; read once per call at the route's start.
    internal enum SmallKTraceEvent: Sendable, Equatable {
        case partial(partition: Int, candidates: Range<Int>, heapCapacity: Int, scratchCapacity: Int, retained: Int)
        case combine(left: Int, right: Int, result: Int)
    }

    @TaskLocal internal static var smallKTrace: (@Sendable (SmallKTraceEvent) -> Void)?

    /// Observers of one small-k call: the per-chunk logical ranges (parallel runs only) and the trace seam.
    private struct SmallKObservers: Sendable {
        let logical: LogicalObservation?
        let trace: (@Sendable (SmallKTraceEvent) -> Void)?
    }

    /// One worker's heap: capacity `min(k, candidates assigned to partition)`, fed every score of the partition.
    private static func partialHeap(_ partition: Int, plan: SmallKPlan, route: SmallKRoute, k: Int,
                                    observers: SmallKObservers) -> TopKBuffer {
        let candidates = plan.candidates(ofPartition: partition)
        var heap = TopKBuffer(k: min(k, candidates.count), isMinHeap: false, tieBreaker: .smallerIndex)
        let scratchCapacity = scorePartition(partition, plan: plan, route: route,
                                             observe: { observers.logical?.range($0, index: $1) },
                                             sink: { heap.pushIfBetter(val: $0, idx: $1) })
        observers.trace?(.partial(partition: partition, candidates: candidates, heapCapacity: heap.k,
                                  scratchCapacity: scratchCapacity, retained: heap.size))
        return heap
    }

    /// Serial small k: one heap of capacity `k` over the route's serial steps. No tasks and no observations,
    /// as on the serial large-k route.
    private static func streamNearestSerial(route: SmallKRoute, count: Int, k: Int) -> [(index: Int, distance: Float)] {
        let heap = partialHeap(0, plan: .serial(route: route, count: count), route: route, k: k,
                               observers: SmallKObservers(logical: nil, trace: smallKTrace))
        return ascendingEntries(heap)
    }

    /// Parallel small k: the E06 logical chunks of `chunkSize`, grouped into `SmallKPlan` partitions, one
    /// `completeMap` item and one heap per partition, each logical chunk observed once. `completeMap` has no
    /// cancellation check, and nested calls run its items inline without submitting tasks
    /// (`P01BatchRouteTests.cancelledAndNested`).
    private static func streamNearestParallel(route: SmallKRoute, count: Int, k: Int, chunkSize: Int,
                                              cores: Int) async -> [(index: Int, distance: Float)] {
        let plan = SmallKPlan(count: count, chunkSize: chunkSize, cores: cores)
        let observation = LogicalObservation(phase: route.phase + ".logical", cores: cores)
        defer { observation.finish() }
        let observers = SmallKObservers(logical: observation, trace: smallKTrace)
        let partials = await BoundedExecution.completeMap(0..<plan.partitionCount, limit: cores,
            phase: route.phase) { partition in
            partialHeap(partition, plan: plan, route: route, k: k, observers: observers)
        }
        return ascendingEntries(combineInOrder(partials, k: k, trace: observers.trace))
    }

    /// Merges the partial heaps in ascending partition order, mirroring `Operations.findNearest`'s combine rule:
    /// when the accumulated heap already has capacity `k` the next partial is merged into it in place, otherwise
    /// both merge into a new heap of capacity `min(k, left + right)`. Every partial has capacity at most `k`.
    private static func combineInOrder(_ partials: [TopKBuffer], k: Int,
                                       trace: (@Sendable (SmallKTraceEvent) -> Void)?) -> TopKBuffer {
        var total = partials[0]
        for partial in partials.dropFirst() {
            let left = total.k
            if total.k == k {
                total.mergeIn(from: partial)
            } else {
                // total.k < k and partial.k <= k, so `k - partial.k` cannot overflow.
                let capacity = total.k > k - partial.k ? k : total.k + partial.k
                var merged = TopKBuffer(k: capacity, isMinHeap: false, tieBreaker: .smallerIndex)
                merged.mergeIn(from: total)
                merged.mergeIn(from: partial)
                total = merged
            }
            trace?(.combine(left: left, right: partial.k, result: total.k))
        }
        return total
    }

    /// The retained entries, best first: numeric distances ascending, then NaNs, exact ties by smaller index.
    private static func ascendingEntries(_ heap: TopKBuffer) -> [(index: Int, distance: Float)] {
        var entries: [(index: Int, distance: Float)] = []
        entries.reserveCapacity(heap.size)
        for slot in 0..<heap.size {
            entries.append((index: heap.idxs[slot], distance: heap.vals[slot]))
        }
        entries.sort { TopKSelection.orderedAscending(($0.index, $0.distance), ($1.index, $1.distance), .smallerIndex) }
        return entries
    }

    // MARK: - Optimized distance computation helpers (BatchKernels)

    @inline(__always)
    private static func minChunk(forDim dim: Int) -> Int {
        switch dim { case 1536: return 512; case 768: return 256; default: return 256 }
    }

    // Serial Euclidean
    private static func computeDistances_euclid_optimized_serial(
        query: Vector512Optimized,
        candidates: [Vector512Optimized],
        dim: Int
    ) -> [Float] {
        let n = candidates.count
        var out = [Float](repeating: 0, count: n)
        out.withUnsafeMutableBufferPointer { buf in
            var start = 0
            let step = minChunk(forDim: dim)
            while start < n {
                let end = min(start + step, n)
                let sub = UnsafeMutableBufferPointer<Float>(start: buf.baseAddress!.advanced(by: start), count: end - start)
                BatchKernels.range_euclid_512(query: query, candidates: candidates, range: start..<end, out: sub)
                start = end
            }
        }
        return out
    }

    private static func computeDistances_euclid_optimized_serial(
        query: Vector768Optimized,
        candidates: [Vector768Optimized],
        dim: Int
    ) -> [Float] {
        let n = candidates.count
        var out = [Float](repeating: 0, count: n)
        out.withUnsafeMutableBufferPointer { buf in
            var start = 0
            let step = minChunk(forDim: dim)
            while start < n {
                let end = min(start + step, n)
                let sub = UnsafeMutableBufferPointer<Float>(start: buf.baseAddress!.advanced(by: start), count: end - start)
                BatchKernels.range_euclid_768(query: query, candidates: candidates, range: start..<end, out: sub)
                start = end
            }
        }
        return out
    }

    private static func computeDistances_euclid_optimized_serial(
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized],
        dim: Int
    ) -> [Float] {
        let n = candidates.count
        var out = [Float](repeating: 0, count: n)
        out.withUnsafeMutableBufferPointer { buf in
            var start = 0
            let step = minChunk(forDim: dim)
            while start < n {
                let end = min(start + step, n)
                let sub = UnsafeMutableBufferPointer<Float>(start: buf.baseAddress!.advanced(by: start), count: end - start)
                BatchKernels.range_euclid_1536(query: query, candidates: candidates, range: start..<end, out: sub)
                start = end
            }
        }
        return out
    }

    // Serial Cosine fused
    private static func computeDistances_cosine_fused_serial(
        query: Vector512Optimized,
        candidates: [Vector512Optimized],
        dim: Int
    ) -> [Float] {
        let n = candidates.count
        var out = [Float](repeating: 0, count: n)
        out.withUnsafeMutableBufferPointer { buf in
            var start = 0
            let step = minChunk(forDim: dim)
            while start < n {
                let end = min(start + step, n)
                let sub = UnsafeMutableBufferPointer<Float>(start: buf.baseAddress!.advanced(by: start), count: end - start)
                BatchKernels.range_cosine_fused_512(query: query, candidates: candidates, range: start..<end, out: sub)
                start = end
            }
        }
        return out
    }

    private static func computeDistances_cosine_fused_serial(
        query: Vector768Optimized,
        candidates: [Vector768Optimized],
        dim: Int
    ) -> [Float] {
        let n = candidates.count
        var out = [Float](repeating: 0, count: n)
        out.withUnsafeMutableBufferPointer { buf in
            var start = 0
            let step = minChunk(forDim: dim)
            while start < n {
                let end = min(start + step, n)
                let sub = UnsafeMutableBufferPointer<Float>(start: buf.baseAddress!.advanced(by: start), count: end - start)
                BatchKernels.range_cosine_fused_768(query: query, candidates: candidates, range: start..<end, out: sub)
                start = end
            }
        }
        return out
    }

    private static func computeDistances_cosine_fused_serial(
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized],
        dim: Int
    ) -> [Float] {
        let n = candidates.count
        var out = [Float](repeating: 0, count: n)
        out.withUnsafeMutableBufferPointer { buf in
            var start = 0
            let step = minChunk(forDim: dim)
            while start < n {
                let end = min(start + step, n)
                let sub = UnsafeMutableBufferPointer<Float>(start: buf.baseAddress!.advanced(by: start), count: end - start)
                BatchKernels.range_cosine_fused_1536(query: query, candidates: candidates, range: start..<end, out: sub)
                start = end
            }
        }
        return out
    }

    // Parallel Euclidean
    // Internal direct coverage remains in ExecutionSchedulingContractTests.
    internal static func computeDistances_euclid_optimized_parallel(
        query: Vector512Optimized,
        candidates: [Vector512Optimized],
        dim: Int, chunkSize requestedChunkSize: Int? = nil, cores coreCount: Int = BoundedExecution.hardwareLimit
    ) async -> [Float] {
        let n = candidates.count
        guard n > 0 else { return [] }
        let chunkSize = requestedChunkSize ?? logicalChunkSize(count: n, cores: coreCount, configuration: Configuration())
        let count = ComputeRangeBoundary.ceilingDivide(n, by: chunkSize)
        let observation = LogicalObservation(phase: "batch.euclid512.logical", cores: coreCount)
        defer { observation.finish() }
        let parts = await BoundedExecution.completeMap(0..<count, limit: coreCount,
            phase: "batch.euclid512") { index in
            let start = index * chunkSize
            let end = start + min(chunkSize, n - start)
            observation.range(start..<end, index: index)
            var values = [Float](repeating: 0, count: end - start)
            values.withUnsafeMutableBufferPointer { out in
                BatchKernels.range_euclid_512(query: query, candidates: candidates, range: start..<end, out: out)
            }
            return values
        }
        return parts.flatMap { $0 }
    }

    // Internal direct coverage remains in ExecutionSchedulingContractTests.
    internal static func computeDistances_euclid_optimized_parallel(
        query: Vector768Optimized,
        candidates: [Vector768Optimized],
        dim: Int, chunkSize requestedChunkSize: Int? = nil, cores coreCount: Int = BoundedExecution.hardwareLimit
    ) async -> [Float] {
        let n = candidates.count
        guard n > 0 else { return [] }
        let chunkSize = requestedChunkSize ?? logicalChunkSize(count: n, cores: coreCount, configuration: Configuration())
        let count = ComputeRangeBoundary.ceilingDivide(n, by: chunkSize)
        let observation = LogicalObservation(phase: "batch.euclid768.logical", cores: coreCount)
        defer { observation.finish() }
        let parts = await BoundedExecution.completeMap(0..<count, limit: coreCount,
            phase: "batch.euclid768") { index in
            let start = index * chunkSize
            let end = start + min(chunkSize, n - start)
            observation.range(start..<end, index: index)
            var values = [Float](repeating: 0, count: end - start)
            values.withUnsafeMutableBufferPointer { out in
                BatchKernels.range_euclid_768(query: query, candidates: candidates, range: start..<end, out: out)
            }
            return values
        }
        return parts.flatMap { $0 }
    }

    // Internal direct coverage remains in ExecutionSchedulingContractTests.
    internal static func computeDistances_euclid_optimized_parallel(
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized],
        dim: Int, chunkSize requestedChunkSize: Int? = nil, cores coreCount: Int = BoundedExecution.hardwareLimit
    ) async -> [Float] {
        let n = candidates.count
        guard n > 0 else { return [] }
        let chunkSize = requestedChunkSize ?? logicalChunkSize(count: n, cores: coreCount, configuration: Configuration())
        let count = ComputeRangeBoundary.ceilingDivide(n, by: chunkSize)
        let observation = LogicalObservation(phase: "batch.euclid1536.logical", cores: coreCount)
        defer { observation.finish() }
        let parts = await BoundedExecution.completeMap(0..<count, limit: coreCount,
            phase: "batch.euclid1536") { index in
            let start = index * chunkSize
            let end = start + min(chunkSize, n - start)
            observation.range(start..<end, index: index)
            var values = [Float](repeating: 0, count: end - start)
            values.withUnsafeMutableBufferPointer { out in
                BatchKernels.range_euclid_1536(query: query, candidates: candidates, range: start..<end, out: out)
            }
            return values
        }
        return parts.flatMap { $0 }
    }

    // Parallel Cosine fused
    // Internal direct coverage remains in ExecutionSchedulingContractTests.
    internal static func computeDistances_cosine_fused_parallel(
        query: Vector512Optimized,
        candidates: [Vector512Optimized],
        dim: Int, chunkSize requestedChunkSize: Int? = nil, cores coreCount: Int = BoundedExecution.hardwareLimit
    ) async -> [Float] {
        let n = candidates.count
        guard n > 0 else { return [] }
        let chunkSize = requestedChunkSize ?? logicalChunkSize(count: n, cores: coreCount, configuration: Configuration())
        let count = ComputeRangeBoundary.ceilingDivide(n, by: chunkSize)
        let observation = LogicalObservation(phase: "batch.cosine_fused512.logical", cores: coreCount)
        defer { observation.finish() }
        let parts = await BoundedExecution.completeMap(0..<count, limit: coreCount,
            phase: "batch.cosine_fused512") { index in
            let start = index * chunkSize
            let end = start + min(chunkSize, n - start)
            observation.range(start..<end, index: index)
            var values = [Float](repeating: 0, count: end - start)
            values.withUnsafeMutableBufferPointer { out in
                BatchKernels.range_cosine_fused_512(query: query, candidates: candidates, range: start..<end, out: out)
            }
            return values
        }
        return parts.flatMap { $0 }
    }

    // Internal direct coverage remains in ExecutionSchedulingContractTests.
    internal static func computeDistances_cosine_fused_parallel(
        query: Vector768Optimized,
        candidates: [Vector768Optimized],
        dim: Int, chunkSize requestedChunkSize: Int? = nil, cores coreCount: Int = BoundedExecution.hardwareLimit
    ) async -> [Float] {
        let n = candidates.count
        guard n > 0 else { return [] }
        let chunkSize = requestedChunkSize ?? logicalChunkSize(count: n, cores: coreCount, configuration: Configuration())
        let count = ComputeRangeBoundary.ceilingDivide(n, by: chunkSize)
        let observation = LogicalObservation(phase: "batch.cosine_fused768.logical", cores: coreCount)
        defer { observation.finish() }
        let parts = await BoundedExecution.completeMap(0..<count, limit: coreCount,
            phase: "batch.cosine_fused768") { index in
            let start = index * chunkSize
            let end = start + min(chunkSize, n - start)
            observation.range(start..<end, index: index)
            var values = [Float](repeating: 0, count: end - start)
            values.withUnsafeMutableBufferPointer { out in
                BatchKernels.range_cosine_fused_768(query: query, candidates: candidates, range: start..<end, out: out)
            }
            return values
        }
        return parts.flatMap { $0 }
    }

    // Internal direct coverage remains in ExecutionSchedulingContractTests.
    internal static func computeDistances_cosine_fused_parallel(
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized],
        dim: Int, chunkSize requestedChunkSize: Int? = nil, cores coreCount: Int = BoundedExecution.hardwareLimit
    ) async -> [Float] {
        let n = candidates.count
        guard n > 0 else { return [] }
        let chunkSize = requestedChunkSize ?? logicalChunkSize(count: n, cores: coreCount, configuration: Configuration())
        let count = ComputeRangeBoundary.ceilingDivide(n, by: chunkSize)
        let observation = LogicalObservation(phase: "batch.cosine_fused1536.logical", cores: coreCount)
        defer { observation.finish() }
        let parts = await BoundedExecution.completeMap(0..<count, limit: coreCount,
            phase: "batch.cosine_fused1536") { index in
            let start = index * chunkSize
            let end = start + min(chunkSize, n - start)
            observation.range(start..<end, index: index)
            var values = [Float](repeating: 0, count: end - start)
            values.withUnsafeMutableBufferPointer { out in
                BatchKernels.range_cosine_fused_1536(query: query, candidates: candidates, range: start..<end, out: out)
            }
            return values
        }
        return parts.flatMap { $0 }
    }

    /// Sorts all `n` pairs and keeps the best `k`; it builds no heap (the name predates the entry split).
    /// Large-k selection only: `findNearest` routes every `TopKSelection.usesSmallKStrategy` case to the
    /// streaming routes first, and every caller here passes all `n` pairs with the caller's `k`, so the former
    /// small-k branch could not run and was removed.
    private static func heapSelect(
        _ elements: [(index: Int, distance: Float)],
        k: Int
    ) -> [(index: Int, distance: Float)] {
        let actualK = min(k, elements.count)
        guard actualK > 0 else { return [] }
        return TopKSelection.sortSelectLargeK(elements, k: actualK, tieBreaker: .smallerIndex)
    }
}

/// Population statistics of delivered Float vector magnitudes, not components.
/// Empty input yields positive zeros. Any nonfinite magnitude yields two NaNs.
public struct BatchStatistics {
    /// Total number of vectors, including any with nonfinite magnitudes.
    public let count: Int
    /// Arithmetic mean, accumulated in Double and projected once to Float.
    public let meanMagnitude: Float
    /// Population deviation sqrt(M2 / count), projected once to Float.
    /// Singleton and identical finite measurements have exact zero deviation.
    public let stdMagnitude: Float
}
