//
//  TopKSelection.swift
//  VectorCore
//
//  Public API for efficient Top-K selection in k-nearest neighbor search.
//  Provides both distance-based and pre-computed selection algorithms.
//

import Foundation
import simd

// MARK: - Result Types

/// Result of a top-k selection operation.
///
/// Contains the indices and distances of the k nearest vectors,
/// sorted by distance in ascending order (nearest first).
public struct TopKResult: Sendable, Equatable {
    /// Indices of the k nearest vectors (sorted by distance, ascending)
    public let indices: [Int]

    /// Distances corresponding to each index
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

// MARK: - Top-K Selection API

/// Top-K selection algorithms for k-nearest neighbor search.
///
/// Provides efficient selection of the k smallest distances from a set of candidates.
/// Uses heap-based algorithms for O(n log k) complexity when k << n.
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
    /// - Parameters:
    ///   - k: Number of nearest neighbors to select
    ///   - distances: Array of pre-computed distances
    /// - Returns: TopKResult with indices and distances of k smallest
    /// - Complexity: O(n log k) for small k, O(n log n) for large k
    ///
    /// ## Algorithm Selection
    /// - For k < n/10: Uses max-heap to maintain k smallest (O(n log k))
    /// - For k >= n/10: Uses partial sort (O(n log n) but better constants)
    public static func select(k: Int, from distances: [Float]) -> TopKResult {
        guard k > 0 else { return TopKResult() }
        guard !distances.isEmpty else { return TopKResult() }

        let actualK = min(k, distances.count)

        // Create index-distance pairs
        let pairs = distances.enumerated().map { (index: $0.offset, distance: $0.element) }

        // Use adaptive algorithm based on k/n ratio
        let selected: [(index: Int, distance: Float)]
        if actualK < distances.count / 10 {
            selected = heapSelectSmallK(pairs, k: actualK)
        } else {
            selected = sortSelectLargeK(pairs, k: actualK)
        }

        return TopKResult(
            indices: selected.map { $0.index },
            distances: selected.map { $0.distance }
        )
    }

    /// Select the k smallest values with custom comparison.
    ///
    /// - Parameters:
    ///   - k: Number of elements to select
    ///   - elements: Array of elements to select from
    ///   - distance: Closure to extract distance value from element
    /// - Returns: Array of k elements with smallest distances
    public static func select<T>(
        k: Int,
        from elements: [T],
        distance: (T) -> Float
    ) -> [T] {
        guard k > 0 && !elements.isEmpty else { return [] }

        let actualK = min(k, elements.count)
        let indexed = elements.enumerated().map { (index: $0.offset, distance: distance($0.element)) }

        let selected: [(index: Int, distance: Float)]
        if actualK < elements.count / 10 {
            selected = heapSelectSmallK(indexed, k: actualK)
        } else {
            selected = sortSelectLargeK(indexed, k: actualK)
        }

        return selected.map { elements[$0.index] }
    }

    // MARK: - Generic K-Nearest with Distance Computation

    /// Find k nearest neighbors using any distance metric.
    ///
    /// - Parameters:
    ///   - k: Number of nearest neighbors to find
    ///   - query: The query vector
    ///   - candidates: Array of candidate vectors to search
    ///   - metric: Distance metric to use (default: Euclidean)
    /// - Returns: TopKResult with indices and distances of k nearest
    /// - Complexity: O(n * d + n log k) where d is vector dimension
    public static func nearest<V: VectorProtocol>(
        k: Int,
        query: V,
        candidates: [V],
        metric: any DistanceMetric = EuclideanDistance()
    ) -> TopKResult where V.Scalar == Float {
        // Use specific metric implementations to avoid existential limitations
        if metric is EuclideanDistance {
            return nearestWithMetric(k: k, query: query, candidates: candidates, metric: EuclideanDistance())
        } else if metric is CosineDistance {
            return nearestWithMetric(k: k, query: query, candidates: candidates, metric: CosineDistance())
        } else if metric is ManhattanDistance {
            return nearestWithMetric(k: k, query: query, candidates: candidates, metric: ManhattanDistance())
        } else if metric is DotProductDistance {
            return nearestWithMetric(k: k, query: query, candidates: candidates, metric: DotProductDistance())
        } else {
            // Fallback to Euclidean
            return nearestWithMetric(k: k, query: query, candidates: candidates, metric: EuclideanDistance())
        }
    }

    /// Internal generic implementation with concrete metric type
    private static func nearestWithMetric<V: VectorProtocol, M: DistanceMetric>(
        k: Int,
        query: V,
        candidates: [V],
        metric: M
    ) -> TopKResult where V.Scalar == Float, M.Scalar == Float {
        guard k > 0 && !candidates.isEmpty else { return TopKResult() }

        let actualK = min(k, candidates.count)

        // Compute all distances
        let distances = candidates.map { metric.distance(query, $0) }

        return select(k: actualK, from: distances)
    }

    // MARK: - Optimized Paths for Specific Vector Types

    /// Optimized k-nearest for Vector384Optimized with Euclidean distance.
    ///
    /// Uses SIMD-accelerated distance computation and efficient heap selection.
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
    /// Returns results sorted by similarity (highest first).
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
        metric: any DistanceMetric = EuclideanDistance()
    ) -> [TopKResult] where V.Scalar == Float {
        queries.map { query in
            nearest(k: k, query: query, candidates: candidates, metric: metric)
        }
    }

    // MARK: - Private Helpers

    /// Heap-based selection for small k (O(n log k))
    @usableFromInline
    internal static func heapSelectSmallK(
        _ elements: [(index: Int, distance: Float)],
        k: Int
    ) -> [(index: Int, distance: Float)] {
        var heap = KNearestHeap(k: k)

        for element in elements {
            heap.insert(index: element.index, distance: element.distance)
        }

        return heap.getSorted()
    }

    /// Sort-based selection for large k (better constants)
    @usableFromInline
    internal static func sortSelectLargeK(
        _ elements: [(index: Int, distance: Float)],
        k: Int
    ) -> [(index: Int, distance: Float)] {
        Array(elements.sorted { $0.distance < $1.distance }.prefix(k))
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

        // Sort by distance ascending
        pairs.sort { $0.1 < $1.1 }

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

        // Sort by similarity descending (highest first)
        pairs.sort { $0.1 > $1.1 }

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
        indices = try container.decode([Int].self, forKey: .indices)
        distances = try container.decode([Float].self, forKey: .distances)
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(indices, forKey: .indices)
        try container.encode(distances, forKey: .distances)
    }
}
