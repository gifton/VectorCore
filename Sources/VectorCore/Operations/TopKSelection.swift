//
//  TopKSelection.swift
//  VectorCore
//
//  Public API for Top-K selection in k-nearest neighbor search.
//  Provides both distance-based and pre-computed selection algorithms.
//

import Foundation
import simd

// MARK: - Result Types

/// Result of a top-k selection operation.
///
/// Selection APIs return best-first scores: ascending distances, or descending
/// similarities for `nearestDotProduct512`. NaNs follow all numeric scores.
public struct TopKResult: Sendable, Equatable {
    /// Original candidate indices in best-first order.
    public let indices: [Int]

    /// Scores corresponding to each index (similarities for `nearestDotProduct512`).
    public let distances: [Float]

    /// Number of results (may be less than k if fewer candidates available)
    public var count: Int { indices.count }

    /// Whether the result is empty
    public var isEmpty: Bool { indices.isEmpty }

    /// Initialize with parallel arrays
    public init(indices: [Int], distances: [Float]) {
        precondition(indices.count == distances.count, "Indices and distances must have same count")
        self.indices = indices
        self.distances = distances
    }

    /// Initialize empty result
    public init() {
        self.indices = []
        self.distances = []
    }

    /// Access result at index as tuple
    public subscript(index: Int) -> (index: Int, distance: Float) {
        (indices[index], distances[index])
    }

    /// Convert to array of tuples
    public func toTuples() -> [(index: Int, distance: Float)] {
        zip(indices, distances).map { ($0, $1) }
    }
}

// MARK: - Tie-Breaking Policy

/// Policy for resolving ties when two candidates compare equal on distance/similarity.
///
/// Top-K results are ordered best-first; when two candidates are equal on value,
/// the policy decides both *which* candidates survive at the k boundary and their
/// relative order in the result.
///
/// - Note: For the array/pointer selection APIs a candidate's index equals its
///   position in the input scan, so `.smallerIndex` and `.insertionOrder` produce
///   identical output. Ties use exact Float equality (including signed zeros),
///   or two NaNs. `.smallerValue` applies no index tie-break; identities and order
///   within equal-value or NaN groups are unspecified.
public enum TieBreaker: Sendable {
    /// Prefer the candidate with the smaller index — fully deterministic. Default.
    case smallerIndex
    /// Prefer original input order among ties, equivalent to `.smallerIndex`
    /// for public array/pointer scans. This does not mean parallel merge order.
    case insertionOrder
    /// Order by value only; equal numeric scores and two NaNs are equivalent.
    case smallerValue
}

// MARK: - Top-K Selection API

/// Top-K selection algorithms for k-nearest neighbor search.
///
/// Selects the k smallest distances from a set of candidates.
/// Uses heap-based algorithms for O(n log k) complexity when k << n.
///
/// Selection contract: numeric scores (including infinities) precede NaNs in
/// both directions. Exact ties, including +0/-0 and two NaNs, use `TieBreaker`.
/// For positive k, select min(k, n) candidates, retaining NaNs when needed to
/// fill the count. Precomputed selection preserves original scores and zero signs.
/// `TopKNaNContractTests` exercises membership, output order, and comparator laws.
///
/// Optimized Euclidean paths select squared scores and then return square roots.
/// Distinct squared scores that round to equal roots need not tie at admission.
/// Ordering agrees across implementations only for identical computed scores;
/// this contract does not change metric rounding or non-finite metric policies.
///
/// ## Usage Examples
///
/// ### Select from pre-computed distances:
/// ```swift
/// let distances: [Float] = computeDistances(query, candidates)
/// let result = TopKSelection.select(k: 10, from: distances)
/// print("Nearest indices: \(result.indices)")
/// ```
///
/// ### Find k-nearest with distance computation:
/// ```swift
/// let result = TopKSelection.nearest(
///     k: 10,
///     query: queryVector,
///     candidates: vectorDatabase,
///     metric: EuclideanDistance()
/// )
/// ```
///
/// ### Optimized path for specific vector types:
/// ```swift
/// let result = TopKSelection.nearestEuclidean512(
///     k: 10,
///     query: query512,
///     candidates: candidates512
/// )
/// ```
public enum TopKSelection {

    // MARK: - Selection from Pre-computed Distances

    /// Select the k smallest values from a distance array.
    ///
    /// NaNs rank last and remain candidates. Exact ties use the original input
    /// position by default. Nonpositive k or empty input returns an empty result.
    ///
    /// - Parameters:
    ///   - k: Number of nearest neighbors to select
    ///   - distances: Array of pre-computed distances
    /// - Returns: TopKResult with indices and distances of k smallest
    /// - Complexity: With n = `distances.count` and k' = min(k, n): O(n log k')
    ///   time and O(k') working arrays when k' < n / 10; otherwise O(n log n)
    ///   time and O(n) working arrays.
    ///
    /// ## Algorithm Selection
    /// - k' < n / 10 (integer division): streams each value into a bounded
    ///   max-heap of k' entries, then sorts only those entries.
    /// - Otherwise: sorts all n (index, value) pairs and keeps the first k'.
    public static func select(
        k: Int,
        from distances: [Float],
        tieBreaker: TieBreaker = .smallerIndex
    ) -> TopKResult {
        let selected = selectByScore(count: distances.count, k: k, tieBreaker: tieBreaker) { distances[$0] }
        return TopKResult(
            indices: selected.map { $0.index },
            distances: selected.map { $0.distance }
        )
    }

    // MARK: - Pointer API

    /// Select the k smallest values from a raw distance buffer.
    ///
    /// For distances held in raw memory (for example a buffer shared with the GPU). Reads
    /// `distances` (and `ids`) through the pointers without first copying them into an array;
    /// the small-k path keeps a heap of k entries, the large-k path allocates a `count`-element
    /// index array, and the results are new `[Int32]` and `[Float]` arrays.
    ///
    /// NaNs rank last; positive k returns min(k, count) entries. Ties use exact
    /// equality (+0 and -0 tie), with original buffer positions as the default
    /// tie-break. IDs are output labels mapped after selection, even when
    /// duplicate or nonmonotonic; they never affect admission or ordering.
    /// Original values, including signed zero, are copied into the result.
    ///
    /// - Parameters:
    ///   - k: Number of nearest neighbors to select
    ///   - distances: Pointer to pre-computed distance values
    ///   - count: Number of elements in the distances buffer
    ///   - ids: Optional pointer to Int32 identifiers. If nil, indices 0..<count are used.
    /// - Returns: Tuple of (indices, distances) sorted ascending by distance
    /// - Complexity: O(n log k) for small k (heap, O(k) memory), O(n log n) for large k (sort, O(n) memory)
    public static func select(
        k: Int,
        from distances: UnsafePointer<Float>,
        count: Int,
        ids: UnsafePointer<Int32>? = nil,
        tieBreaker: TieBreaker = .smallerIndex
    ) -> (indices: [Int32], distances: [Float]) {
        guard k > 0, count > 0 else {
            return (indices: [], distances: [])
        }
        precondition(count <= Int(Int32.max), "Count exceeds Int32 range")

        let actualK = min(k, count)

        if actualK < count / 10 {
            return heapSelectPointer(distances: distances, count: count, k: actualK, ids: ids, tieBreaker: tieBreaker)
        } else {
            return sortSelectPointer(distances: distances, count: count, k: actualK, ids: ids, tieBreaker: tieBreaker)
        }
    }

    /// Heap-based selection from raw pointer — O(n log k)
    private static func heapSelectPointer(
        distances: UnsafePointer<Float>,
        count: Int,
        k: Int,
        ids: UnsafePointer<Int32>?,
        tieBreaker: TieBreaker
    ) -> (indices: [Int32], distances: [Float]) {
        var buffer = TopKBuffer(k: k, isMinHeap: false, tieBreaker: tieBreaker)

        for i in 0..<count {
            buffer.pushIfBetter(val: distances[i], idx: i)
        }

        // Extract and sort ascending by distance (ties resolved per policy)
        var pairs: [(Int, Float)] = []
        pairs.reserveCapacity(buffer.size)
        for i in 0..<buffer.size {
            pairs.append((buffer.idxs[i], buffer.vals[i]))
        }
        pairs.sort { orderedAscending(($0.0, $0.1), ($1.0, $1.1), tieBreaker) }

        if let ids = ids {
            return (
                indices: pairs.map { ids[$0.0] },
                distances: pairs.map { $0.1 }
            )
        } else {
            return (
                indices: pairs.map { Int32($0.0) },
                distances: pairs.map { $0.1 }
            )
        }
    }

    /// Sort-based selection from raw pointer — large k path
    private static func sortSelectPointer(
        distances: UnsafePointer<Float>,
        count: Int,
        k: Int,
        ids: UnsafePointer<Int32>?,
        tieBreaker: TieBreaker
    ) -> (indices: [Int32], distances: [Float]) {
        var indexArray = Array(0..<count)
        indexArray.sort { orderedAscending(($0, distances[$0]), ($1, distances[$1]), tieBreaker) }

        let selected = indexArray.prefix(k)
        if let ids = ids {
            return (
                indices: selected.map { ids[$0] },
                distances: selected.map { distances[$0] }
            )
        } else {
            return (
                indices: selected.map { Int32($0) },
                distances: selected.map { distances[$0] }
            )
        }
    }

    // MARK: - Generic Selection

    /// Select the k elements with the smallest `distance` values.
    ///
    /// Same order, NaN, count and strategy rules as `select(k:from:tieBreaker:)`.
    /// An element's position is its index, so `.smallerIndex` and `.insertionOrder`
    /// break exact ties by position; `.smallerValue` leaves tie identity and order
    /// unspecified (see `TieBreaker`). For positive k and nonempty `elements`,
    /// `distance` runs exactly once per element in increasing position order (NaN
    /// and losing values included) and never during the final extraction or sort;
    /// on the small-k path each value is offered to the bounded heap as it is
    /// computed. Nonpositive k or empty input returns `[]` without calling it
    /// (`TopKSelectorOracleTests`).
    ///
    /// - Parameters:
    ///   - k: Number of elements to select
    ///   - elements: Array of elements to select from
    ///   - distance: Closure to extract distance value from element
    /// - Returns: Array of min(k, elements.count) elements, best first
    public static func select<T>(
        k: Int,
        from elements: [T],
        distance: (T) -> Float,
        tieBreaker: TieBreaker = .smallerIndex
    ) -> [T] {
        let selected = selectByScore(count: elements.count, k: k, tieBreaker: tieBreaker) { distance(elements[$0]) }
        return selected.map { elements[$0.index] }
    }

    // MARK: - Generic K-Nearest with Distance Computation

    /// Find k nearest neighbors using any distance metric.
    ///
    /// The metric's own `distance` witness scores every candidate (no built-in
    /// metric is substituted). For positive k and nonempty `candidates` it runs
    /// exactly once per candidate in increasing index order and never during the
    /// final extraction or sort; on the small-k path each distance is offered to
    /// the bounded heap as it is computed. Nonpositive k or empty input returns an
    /// empty result without calling it.
    /// Order, ties (smaller index), NaN and count rules are those of
    /// `select(k:from:tieBreaker:)` (`TopKSelectorOracleTests`).
    ///
    /// - Parameters:
    ///   - k: Number of nearest neighbors to find
    ///   - query: The query vector
    ///   - candidates: Array of candidate vectors to search
    ///   - metric: Distance metric to use (default: Euclidean)
    /// - Returns: TopKResult with indices and distances of k nearest
    /// - Complexity: n metric evaluations, plus (k' = min(k, n)) O(n log k') heap
    ///   work when k' < n / 10, otherwise an O(n log n) sort of n pairs.
    public static func nearest<V: VectorProtocol>(
        k: Int,
        query: V,
        candidates: [V],
        metric: any DistanceMetric<Float> = EuclideanDistance()
    ) -> TopKResult where V.Scalar == Float {
        nearestWithMetric(k: k, query: query, candidates: candidates, metric: metric)
    }

    /// Internal generic implementation with concrete metric type
    private static func nearestWithMetric<V: VectorProtocol, M: DistanceMetric>(
        k: Int,
        query: V,
        candidates: [V],
        metric: M
    ) -> TopKResult where V.Scalar == Float, M.Scalar == Float {
        let selected = selectByScore(count: candidates.count, k: k, tieBreaker: .smallerIndex) {
            metric.distance(query, candidates[$0])
        }
        return TopKResult(
            indices: selected.map { $0.index },
            distances: selected.map { $0.distance }
        )
    }

    // MARK: - Optimized Paths for Specific Vector Types

    /// Optimized k-nearest for Vector384Optimized with Euclidean distance.
    ///
    /// Scores each candidate with `EuclideanKernels.squared384` and keeps the best `min(k, n)` in a
    /// bounded heap (`TopKBuffer`).
    /// - Complexity: O(n * 384/4 + n log k) ≈ O(96n + n log k)
    @inlinable
    public static func nearestEuclidean384(
        k: Int,
        query: Vector384Optimized,
        candidates: [Vector384Optimized]
    ) -> TopKResult {
        guard k > 0 && !candidates.isEmpty else { return TopKResult() }

        let actualK = min(k, candidates.count)
        var buffer = TopKBuffer(k: actualK, isMinHeap: false)

        for i in 0..<candidates.count {
            let dist = EuclideanKernels.squared384(query, candidates[i])
            buffer.pushIfBetter(val: dist, idx: i)
        }

        return extractSortedResult(from: buffer, sqrt: true)
    }

    /// Optimized k-nearest for Vector512Optimized with Euclidean distance.
    @inlinable
    public static func nearestEuclidean512(
        k: Int,
        query: Vector512Optimized,
        candidates: [Vector512Optimized]
    ) -> TopKResult {
        guard k > 0 && !candidates.isEmpty else { return TopKResult() }

        let actualK = min(k, candidates.count)
        var buffer = TopKBuffer(k: actualK, isMinHeap: false)

        for i in 0..<candidates.count {
            let dist = EuclideanKernels.squared512(query, candidates[i])
            buffer.pushIfBetter(val: dist, idx: i)
        }

        return extractSortedResult(from: buffer, sqrt: true)
    }

    /// Optimized k-nearest for Vector768Optimized with Euclidean distance.
    @inlinable
    public static func nearestEuclidean768(
        k: Int,
        query: Vector768Optimized,
        candidates: [Vector768Optimized]
    ) -> TopKResult {
        guard k > 0 && !candidates.isEmpty else { return TopKResult() }

        let actualK = min(k, candidates.count)
        var buffer = TopKBuffer(k: actualK, isMinHeap: false)

        for i in 0..<candidates.count {
            let dist = EuclideanKernels.squared768(query, candidates[i])
            buffer.pushIfBetter(val: dist, idx: i)
        }

        return extractSortedResult(from: buffer, sqrt: true)
    }

    /// Optimized k-nearest for Vector1536Optimized with Euclidean distance.
    @inlinable
    public static func nearestEuclidean1536(
        k: Int,
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized]
    ) -> TopKResult {
        guard k > 0 && !candidates.isEmpty else { return TopKResult() }

        let actualK = min(k, candidates.count)
        var buffer = TopKBuffer(k: actualK, isMinHeap: false)

        for i in 0..<candidates.count {
            let dist = EuclideanKernels.squared1536(query, candidates[i])
            buffer.pushIfBetter(val: dist, idx: i)
        }

        return extractSortedResult(from: buffer, sqrt: true)
    }

    /// Optimized k-nearest for Vector512Optimized with Cosine distance (pre-normalized vectors).
    ///
    /// Assumes both query and candidates are unit vectors (magnitude = 1).
    /// Uses dot product directly: cosine_distance = 1 - dot(a, b)
    @inlinable
    public static func nearestCosinePreNormalized512(
        k: Int,
        query: Vector512Optimized,
        candidates: [Vector512Optimized]
    ) -> TopKResult {
        guard k > 0 && !candidates.isEmpty else { return TopKResult() }

        let actualK = min(k, candidates.count)
        var buffer = TopKBuffer(k: actualK, isMinHeap: false)

        for i in 0..<candidates.count {
            let dist = CosineKernels.distance512_preNormalized(query, candidates[i])
            buffer.pushIfBetter(val: dist, idx: i)
        }

        return extractSortedResult(from: buffer, sqrt: false)
    }

    /// Optimized k-nearest for Vector512Optimized with Cosine distance (fused computation).
    ///
    /// Computes cosine distance in a single pass, handling non-normalized vectors.
    @inlinable
    public static func nearestCosineFused512(
        k: Int,
        query: Vector512Optimized,
        candidates: [Vector512Optimized]
    ) -> TopKResult {
        guard k > 0 && !candidates.isEmpty else { return TopKResult() }

        let actualK = min(k, candidates.count)
        var buffer = TopKBuffer(k: actualK, isMinHeap: false)

        for i in 0..<candidates.count {
            let dist = CosineKernels.distance512_fused(query, candidates[i])
            buffer.pushIfBetter(val: dist, idx: i)
        }

        return extractSortedResult(from: buffer, sqrt: false)
    }

    /// Optimized k-nearest for dot product similarity (maximize).
    ///
    /// Returns results sorted by similarity (highest numeric first, NaNs last).
    /// Exact ties, including two NaNs, prefer smaller original indices.
    /// Use this when you want to maximize dot product rather than minimize distance.
    @inlinable
    public static func nearestDotProduct512(
        k: Int,
        query: Vector512Optimized,
        candidates: [Vector512Optimized]
    ) -> TopKResult {
        guard k > 0 && !candidates.isEmpty else { return TopKResult() }

        let actualK = min(k, candidates.count)
        var buffer = TopKBuffer(k: actualK, isMinHeap: true)  // Min-heap for maximization

        for i in 0..<candidates.count {
            let sim = DotKernels.dot512(query, candidates[i])
            buffer.pushIfBetter(val: sim, idx: i)
        }

        // For dot product, higher is better, so sort descending
        return extractSortedResultDescending(from: buffer)
    }

    // MARK: - Batch Top-K (Multiple Queries)

    /// Find k-nearest for multiple queries in batch.
    ///
    /// - Parameters:
    ///   - k: Number of nearest neighbors per query
    ///   - queries: Array of query vectors
    ///   - candidates: Array of candidate vectors to search
    ///   - metric: Distance metric to use
    /// - Returns: Array of TopKResult, one per query
    public static func batchNearest<V: VectorProtocol>(
        k: Int,
        queries: [V],
        candidates: [V],
        metric: any DistanceMetric<Float> = EuclideanDistance()
    ) -> [TopKResult] where V.Scalar == Float {
        queries.map { query in
            nearest(k: k, query: query, candidates: candidates, metric: metric)
        }
    }

    // MARK: - Private Helpers

    /// Best-first candidate order. With unique indices, index policies give a
    /// strict total order; `.smallerValue` gives a strict weak order.
    /// Exercised by `TopKNaNContractTests.comparatorLaws` and literal fixtures.
    @usableFromInline @inline(__always)
    internal static func orderedBefore(
        _ a: (Int, Float), _ b: (Int, Float),
        descending: Bool, tieBreaker: TieBreaker
    ) -> Bool {
        let aNaN: Bool = a.1.isNaN
        let bNaN: Bool = b.1.isNaN
        if aNaN != bNaN { return !aNaN }
        if !aNaN && a.1 != b.1 {
            return descending ? a.1 > b.1 : a.1 < b.1
        }
        switch tieBreaker {
        case .smallerIndex, .insertionOrder: return a.0 < b.0
        case .smallerValue: return false
        }
    }

    /// Ascending-distance specialization of the shared candidate order.
    @usableFromInline @inline(__always)
    internal static func orderedAscending(
        _ a: (Int, Float), _ b: (Int, Float), _ tieBreaker: TieBreaker
    ) -> Bool {
        orderedBefore(a, b, descending: false, tieBreaker: tieBreaker)
    }

    /// Small-k strategy predicate for the Int-indexed selector: stream into a bounded heap when
    /// `actualK < count / 10` (integer division), otherwise sort all pairs. It keeps the pre-P01 array and
    /// generic predicate as a compatibility choice, not a measured crossover. Boundary values are pinned by
    /// `TopKSelectorOracleTests.strategyPredicateBoundaries`.
    @inline(__always)
    internal static func usesSmallKStrategy(actualK: Int, count: Int) -> Bool {
        actualK < count / 10
    }

    /// Common Int-indexed selection of the `k` best candidates `0..<count` by ascending score.
    ///
    /// Callers: `select(k:from:tieBreaker:)`, `select(k:from:distance:tieBreaker:)`,
    /// `nearest(k:query:candidates:metric:)` (so `batchNearest`) and `VectorMath.nearestNeighbors`.
    ///
    /// Callback law: when `k > 0` and `count > 0`, `score(i)` is called exactly once for each `i` in
    /// `0..<count`, in increasing `i`, NaN and losing scores included; no call happens during extraction or
    /// sorting. Otherwise it returns `[]` without calling `score`.
    ///
    /// Result law: `min(k, count)` entries, best first: numeric scores ascending, then NaNs; exact ties
    /// (including +0/-0 and two NaNs) by `tieBreaker` through `orderedAscending`. Each entry carries the
    /// original index and the value `score` returned for it, unmodified (zero sign and NaN payload included).
    ///
    /// Strategy (`usesSmallKStrategy`), with positive `k` clamped to `count` before any allocation:
    /// - small k: each score is pushed once into a `TopKBuffer` of capacity `actualK`; only the retained
    ///   entries are extracted and sorted. No `count`-sized score or pair array is built.
    /// - otherwise: `count` (index, score) pairs are built in index order and sorted by `sortSelectLargeK`.
    ///
    /// Indices stay `Int`: there is no `Int32` conversion and no count ceiling.
    ///
    /// Tests: `TopKSelectorOracleTests` observes the callback law (number and order of calls, none for
    /// `k <= 0` or empty input), the result law (order, ties per policy, NaN placement, every k class,
    /// original indices and score bits) and the strategy predicate's boundaries; `VectorMathNearestContractTests`
    /// observes the callback and result laws through `VectorMath`. Two properties follow from this body
    /// alone and are not observed by any test: no `count`-sized array is built on the small-k branch, and
    /// there is no Int32 count ceiling.
    ///
    /// - Complexity: `count` score calls; small k: O(count log actualK) heap work and O(actualK) working
    ///   arrays; otherwise O(count log count) and O(count) working arrays.
    internal static func selectByScore(
        count: Int,
        k: Int,
        tieBreaker: TieBreaker,
        score: (Int) -> Float
    ) -> [(index: Int, distance: Float)] {
        guard k > 0, count > 0 else { return [] }
        let actualK = min(k, count)

        guard usesSmallKStrategy(actualK: actualK, count: count) else {
            var pairs: [(index: Int, distance: Float)] = []
            pairs.reserveCapacity(count)
            for index in 0..<count {
                pairs.append((index: index, distance: score(index)))
            }
            return sortSelectLargeK(pairs, k: actualK, tieBreaker: tieBreaker)
        }

        var buffer = TopKBuffer(k: actualK, isMinHeap: false, tieBreaker: tieBreaker)
        for index in 0..<count {
            buffer.pushIfBetter(val: score(index), idx: index)
        }
        var retained: [(index: Int, distance: Float)] = []
        retained.reserveCapacity(buffer.size)
        for slot in 0..<buffer.size {
            retained.append((index: buffer.idxs[slot], distance: buffer.vals[slot]))
        }
        retained.sort { orderedAscending(($0.index, $0.distance), ($1.index, $1.distance), tieBreaker) }
        return retained
    }

    /// Sort-based selection for large k (better constants), using `tieBreaker`.
    @usableFromInline
    internal static func sortSelectLargeK(
        _ elements: [(index: Int, distance: Float)],
        k: Int,
        tieBreaker: TieBreaker = .smallerIndex
    ) -> [(index: Int, distance: Float)] {
        Array(elements.sorted {
            orderedAscending(($0.index, $0.distance), ($1.index, $1.distance), tieBreaker)
        }.prefix(k))
    }

    /// Extract sorted result from TopKBuffer (for distance minimization)
    @usableFromInline
    internal static func extractSortedResult(from buffer: TopKBuffer, sqrt applySqrt: Bool) -> TopKResult {
        // Build array of (index, distance) from buffer
        var pairs: [(Int, Float)] = []
        pairs.reserveCapacity(buffer.size)

        for i in 0..<buffer.size {
            let dist = applySqrt ? Foundation.sqrt(buffer.vals[i]) : buffer.vals[i]
            pairs.append((buffer.idxs[i], dist))
        }

        // Output ties use transformed values; membership was selected on raw scores.
        pairs.sort { orderedAscending(($0.0, $0.1), ($1.0, $1.1), buffer.tieBreaker) }

        return TopKResult(
            indices: pairs.map { $0.0 },
            distances: pairs.map { $0.1 }
        )
    }

    /// Extract sorted result from TopKBuffer for similarity maximization (descending)
    @usableFromInline
    internal static func extractSortedResultDescending(from buffer: TopKBuffer) -> TopKResult {
        var pairs: [(Int, Float)] = []
        pairs.reserveCapacity(buffer.size)

        for i in 0..<buffer.size {
            pairs.append((buffer.idxs[i], buffer.vals[i]))
        }

        // Sort by similarity descending (highest first), ties resolved per policy.
        pairs.sort { orderedBefore($0, $1, descending: true, tieBreaker: buffer.tieBreaker) }

        return TopKResult(
            indices: pairs.map { $0.0 },
            distances: pairs.map { $0.1 }
        )
    }
}

// MARK: - TopKResult Collection Conformance

extension TopKResult: Collection {
    public typealias Index = Int
    public typealias Element = (index: Int, distance: Float)

    public var startIndex: Int { 0 }
    public var endIndex: Int { count }

    public func index(after i: Int) -> Int { i + 1 }
}

// MARK: - TopKResult Codable Conformance

extension TopKResult: Codable {
    enum CodingKeys: String, CodingKey {
        case indices
        case distances
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let indices = try container.decode([Int].self, forKey: .indices)
        let distances = try container.decode([Float].self, forKey: .distances)
        guard indices.count == distances.count else {
            throw DecodingError.dataCorruptedError(
                forKey: .distances, in: container,
                debugDescription: "Indices count \(indices.count) does not match distances count \(distances.count)"
            )
        }
        self.indices = indices
        self.distances = distances
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(indices, forKey: .indices)
        try container.encode(distances, forKey: .distances)
    }
}
