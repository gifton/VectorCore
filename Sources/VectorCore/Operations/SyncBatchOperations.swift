// VectorCore: Synchronous Batch Operations
//
// High-performance synchronous batch processing for vector operations
//

import Foundation

/// Synchronous batch processing utilities for vector operations
///
/// SyncBatchOperations provides high-performance synchronous operations optimized
/// for immediate execution without async overhead. These are ideal for:
/// - Small to medium datasets where async overhead isn't justified
/// - Integration with synchronous codebases
/// - Performance-critical paths where predictable timing is essential
///
/// ## Performance Characteristics
///
/// - Uses vDSP and SIMD operations for maximum performance
/// - Optimized memory access patterns for cache efficiency
/// - Zero allocation for most operations
/// - Predictable execution time without scheduling overhead
///
/// ## Example Usage
///
/// ```swift
/// // Find nearest neighbors synchronously
/// let neighbors = SyncBatchOperations.findNearest(
///     to: queryVector,
///     in: vectorDatabase,
///     k: 10
/// )
///
/// // Batch normalize vectors
/// let normalized = SyncBatchOperations.map(vectors) { $0.normalized() }
///
/// // Compute centroid
/// let centroid = SyncBatchOperations.centroid(of: vectors)
/// ```
///
/// - Note: This API is internal. Use `BatchOperations` (async) or `Operations` for public APIs.
internal enum SyncBatchOperations {

    // MARK: - Nearest Neighbor Search

    /// Find k nearest neighbors synchronously
    ///
    /// Uses optimized heap selection for O(n log k) complexity.
    ///
    /// - Parameters:
    ///   - query: The query vector
    ///   - vectors: Array of vectors to search
    ///   - k: Number of nearest neighbors to find
    ///   - metric: Distance metric to use (default: Euclidean)
    /// - Returns: Array of (index, distance) tuples sorted by distance
    internal static func findNearest<V: VectorProtocol, M: DistanceMetric>(
        to query: V,
        in vectors: [V],
        k: Int,
        metric: M = EuclideanDistance()
    ) -> [(index: Int, distance: Float)] where M.Scalar == Float, V.Scalar == Float {
        guard k > 0 else { return [] }
        guard !vectors.isEmpty else { return [] }

        // Optimized fast path for Vector*Optimized + Euclidean/Cosine using BatchKernels
        if let _ = metric as? EuclideanDistance {
            if let q = query as? Vector512Optimized, let c = vectors as? [Vector512Optimized] {
                let dists = computeDistances_euclid_optimized_serial(query: q, candidates: c, dim: 512)
                return selectTopK(from: dists, k: k)
            }
            if let q = query as? Vector768Optimized, let c = vectors as? [Vector768Optimized] {
                let dists = computeDistances_euclid_optimized_serial(query: q, candidates: c, dim: 768)
                return selectTopK(from: dists, k: k)
            }
            if let q = query as? Vector1536Optimized, let c = vectors as? [Vector1536Optimized] {
                let dists = computeDistances_euclid_optimized_serial(query: q, candidates: c, dim: 1536)
                return selectTopK(from: dists, k: k)
            }
        } else if let _ = metric as? CosineDistance {
            if let q = query as? Vector512Optimized, let c = vectors as? [Vector512Optimized] {
                let dists = computeDistances_cosine_fused_serial(query: q, candidates: c, dim: 512)
                return selectTopK(from: dists, k: k)
            }
            if let q = query as? Vector768Optimized, let c = vectors as? [Vector768Optimized] {
                let dists = computeDistances_cosine_fused_serial(query: q, candidates: c, dim: 768)
                return selectTopK(from: dists, k: k)
            }
            if let q = query as? Vector1536Optimized, let c = vectors as? [Vector1536Optimized] {
                let dists = computeDistances_cosine_fused_serial(query: q, candidates: c, dim: 1536)
                return selectTopK(from: dists, k: k)
            }
        }

        // For small k, use heap selection (generic path)
        if k < vectors.count / 10 {
            return heapSelect(query: query, vectors: vectors, k: k, metric: metric)
        }

        // For larger k, compute all distances and partial sort (generic path)
        let distances = vectors.enumerated().map { index, vector in
            (index: index, distance: metric.distance(query, vector))
        }
        return Array(distances.sorted { $0.distance < $1.distance }.prefix(k))
    }

    /// Find vectors within a distance threshold
    ///
    /// - Parameters:
    ///   - query: The query vector
    ///   - vectors: Array of vectors to search
    ///   - radius: Maximum distance threshold
    ///   - metric: Distance metric to use
    /// - Returns: Array of (index, distance) tuples within radius
    internal static func findWithinRadius<V: VectorProtocol, M: DistanceMetric>(
        of query: V,
        in vectors: [V],
        radius: Float,
        metric: M = EuclideanDistance()
    ) -> [(index: Int, distance: Float)] where M.Scalar == Float, V.Scalar == Float {
        vectors.enumerated().compactMap { index, vector in
            let distance = metric.distance(query, vector)
            return distance <= radius ? (index: index, distance: distance) : nil
        }
    }

    // MARK: - Batch Transformations

    /// Transform vectors using a mapping function
    ///
    /// - Parameters:
    ///   - vectors: Input vectors
    ///   - transform: Transformation function
    /// - Returns: Transformed vectors
    internal static func map<V: VectorProtocol, U: VectorProtocol>(
        _ vectors: [V],
        transform: (V) throws -> U
    ) rethrows -> [U] {
        try vectors.map(transform)
    }

    /// Transform vectors in-place
    ///
    /// - Parameters:
    ///   - vectors: Vectors to transform (modified in-place)
    ///   - transform: In-place transformation function
    internal static func mapInPlace<V: VectorProtocol>(
        _ vectors: inout [V],
        transform: (inout V) throws -> Void
    ) rethrows {
        for index in vectors.indices {
            try transform(&vectors[index])
        }
    }

    /// Filter vectors based on a predicate
    ///
    /// - Parameters:
    ///   - vectors: Input vectors
    ///   - predicate: Filter predicate
    /// - Returns: Filtered vectors
    internal static func filter<V: VectorProtocol>(
        _ vectors: [V],
        predicate: (V) throws -> Bool
    ) rethrows -> [V] {
        try vectors.filter(predicate)
    }

    /// Partition vectors based on a predicate
    ///
    /// - Parameters:
    ///   - vectors: Input vectors
    ///   - predicate: Partitioning predicate
    /// - Returns: Tuple of (matching, non-matching) vectors
    internal static func partition<V: VectorProtocol>(
        _ vectors: [V],
        by predicate: (V) throws -> Bool
    ) rethrows -> (matching: [V], nonMatching: [V]) {
        var matching: [V] = []
        var nonMatching: [V] = []

        matching.reserveCapacity(vectors.count / 2)
        nonMatching.reserveCapacity(vectors.count / 2)

        for vector in vectors {
            if try predicate(vector) {
                matching.append(vector)
            } else {
                nonMatching.append(vector)
            }
        }

        return (matching, nonMatching)
    }

    // MARK: - Aggregation Operations

    /// Compute the centroid (mean) of vectors
    ///
    /// - Parameter vectors: Input vectors
    /// - Returns: Centroid vector, or nil if input is empty
    internal static func centroid<D: StaticDimension>(
        of vectors: [Vector<D>]
    ) -> Vector<D>? {
        guard !vectors.isEmpty else { return nil }

        // For single vector, return copy
        if vectors.count == 1 {
            return vectors[0]
        }

        // Initialize accumulator with first vector
        var sum = vectors[0]

        // Add remaining vectors
        for i in 1..<vectors.count {
            sum += vectors[i]
        }

        // Divide by count
        return sum / Float(vectors.count)
    }

    /// Compute weighted centroid
    ///
    /// - Parameters:
    ///   - vectors: Input vectors
    ///   - weights: Weights for each vector
    /// - Returns: Weighted centroid, or nil if input is empty
    internal static func weightedCentroid<D: StaticDimension>(
        of vectors: [Vector<D>],
        weights: [Float]
    ) throws -> Vector<D>? where D.Storage: VectorStorageOperations {
        guard !vectors.isEmpty else { return nil }
        guard vectors.count == weights.count else {
            throw VectorError.dimensionMismatch(expected: vectors.count, actual: weights.count)
        }

        var totalWeight: Float = 0
        var sum = Vector<D>.zeros()

        for i in 0..<vectors.count {
            sum += (vectors[i] * weights[i])
            totalWeight += weights[i]
        }

        guard totalWeight > 0 else { return nil }

        return sum / totalWeight
    }

    /// Compute element-wise sum of vectors
    ///
    /// - Parameter vectors: Input vectors
    /// - Returns: Sum vector, or nil if input is empty
    internal static func sum<D: StaticDimension>(
        _ vectors: [Vector<D>]
    ) -> Vector<D>? where D.Storage: VectorStorageOperations {
        guard !vectors.isEmpty else { return nil }

        var result = vectors[0]
        for i in 1..<vectors.count {
            result += vectors[i]
        }

        return result
    }

    /// Compute element-wise mean of vectors
    ///
    /// - Parameter vectors: Input vectors
    /// - Returns: Mean vector, or nil if input is empty
    internal static func mean<D: StaticDimension>(
        _ vectors: [Vector<D>]
    ) -> Vector<D>? where D.Storage: VectorStorageOperations {
        centroid(of: vectors)
    }

    // MARK: - Statistical Operations

    /// Compute batch statistics
    ///
    /// - Parameter vectors: Input vectors
    /// - Returns: Statistics including count, mean magnitude, and std deviation
    internal static func statistics<V: VectorProtocol>(
        for vectors: [V]
    ) -> BatchStatistics where V.Scalar == Float {
        guard !vectors.isEmpty else {
            return BatchStatistics(count: 0, meanMagnitude: 0, stdMagnitude: 0)
        }

        let magnitudes = vectors.map { $0.magnitude }
        let count = Float(vectors.count)

        // Compute mean
        let meanMag = magnitudes.reduce(0, +) / count

        // Compute variance
        let variance = magnitudes.reduce(0) { sum, mag in
            let diff = mag - meanMag
            return sum + diff * diff
        } / count

        return BatchStatistics(
            count: vectors.count,
            meanMagnitude: meanMag,
            stdMagnitude: sqrt(variance)
        )
    }

    /// Find vectors that are outliers based on magnitude
    ///
    /// - Parameters:
    ///   - vectors: Input vectors
    ///   - zscore: Z-score threshold for outlier detection (default: 3)
    /// - Returns: Indices of outlier vectors
    internal static func findOutliers<V: VectorProtocol>(
        in vectors: [V],
        zscoreThreshold: Float = 3
    ) -> [Int] where V.Scalar == Float {
        let stats = statistics(for: vectors)
        guard stats.stdMagnitude > 0 else { return [] }

        return vectors.enumerated().compactMap { index, vector in
            let zscore = abs(vector.magnitude - stats.meanMagnitude) / stats.stdMagnitude
            return zscore > zscoreThreshold ? index : nil
        }
    }

    // MARK: - Distance Matrix Operations

    /// Compute pairwise distances between all vectors
    ///
    /// - Parameters:
    ///   - vectors: Input vectors
    ///   - metric: Distance metric to use
    /// - Returns: Symmetric distance matrix
    internal static func pairwiseDistances<V: VectorProtocol, M: DistanceMetric>(
        _ vectors: [V],
        metric: M = EuclideanDistance()
    ) -> [[Float]] where M.Scalar == Float, V.Scalar == Float {
        let n = vectors.count
        var distances = Array(repeating: Array(repeating: Float(0), count: n), count: n)

        // Only compute upper triangle (matrix is symmetric)
        for i in 0..<n {
            for j in (i+1)..<n {
                let dist = metric.distance(vectors[i], vectors[j])
                distances[i][j] = dist
                distances[j][i] = dist
            }
        }

        return distances
    }

    /// Compute distances from multiple queries to multiple candidates
    ///
    /// - Parameters:
    ///   - queries: Query vectors
    ///   - candidates: Candidate vectors
    ///   - metric: Distance metric to use
    /// - Returns: Distance matrix [queries x candidates]
    internal static func batchDistances<V: VectorProtocol, M: DistanceMetric>(
        from queries: [V],
        to candidates: [V],
        metric: M = EuclideanDistance()
    ) -> [[Float]] where V.Scalar == Float, M.Scalar == Float {
        queries.map { query in
            candidates.map { candidate in
                metric.distance(query, candidate)
            }
        }
    }

    // MARK: - Clustering Support

    /// Assign vectors to nearest centroids
    ///
    /// - Parameters:
    ///   - vectors: Vectors to assign
    ///   - centroids: Cluster centroids
    ///   - metric: Distance metric to use
    /// - Returns: Array of centroid indices for each vector
    internal static func assignToCentroids<V: VectorProtocol, M: DistanceMetric>(
        _ vectors: [V],
        centroids: [V],
        metric: M = EuclideanDistance()
    ) -> [Int] where V.Scalar == Float, M.Scalar == Float {
        vectors.map { vector in
            var minDistance = Float.infinity
            var minIndex = 0

            for (index, centroid) in centroids.enumerated() {
                let distance = metric.distance(vector, centroid)
                if distance < minDistance {
                    minDistance = distance
                    minIndex = index
                }
            }

            return minIndex
        }
    }

    /// Update centroids based on assigned vectors
    ///
    /// - Parameters:
    ///   - vectors: All vectors
    ///   - assignments: Cluster assignment for each vector
    ///   - k: Number of clusters
    /// - Returns: Updated centroids
    internal static func updateCentroids<D: StaticDimension>(
        vectors: [Vector<D>],
        assignments: [Int],
        k: Int
    ) -> [Vector<D>] where D.Storage: VectorStorageOperations {
        var clusterVectors = Array(repeating: [Vector<D>](), count: k)

        // Group vectors by cluster
        for (vector, cluster) in zip(vectors, assignments) {
            clusterVectors[cluster].append(vector)
        }

        // Compute centroid for each cluster
        return clusterVectors.compactMap { cluster in
            centroid(of: cluster)
        }
    }

    // MARK: - Sampling Operations

    /// Random sampling without replacement
    ///
    /// - Parameters:
    ///   - vectors: Input vectors
    ///   - k: Number of samples
    /// - Returns: Random sample of k vectors
    internal static func randomSample<V>(
        from vectors: [V],
        k: Int
    ) -> [V] {
        guard k > 0 && k <= vectors.count else {
            return k <= 0 ? [] : vectors
        }

        // For small samples, use reservoir sampling
        if k < vectors.count / 4 {
            var sample = Array(vectors.prefix(k))

            for i in k..<vectors.count {
                let j = Int.random(in: 0...i)
                if j < k {
                    sample[j] = vectors[i]
                }
            }

            return sample
        }

        // For larger samples, shuffle indices
        var indices = Array(0..<vectors.count)
        indices.shuffle()

        return indices.prefix(k).map { vectors[$0] }
    }

    /// Stratified sampling based on magnitude
    ///
    /// - Parameters:
    ///   - vectors: Input vectors
    ///   - k: Total number of samples
    ///   - strata: Number of magnitude-based strata
    /// - Returns: Stratified sample
    internal static func stratifiedSample<V: VectorProtocol>(
        from vectors: [V],
        k: Int,
        strata: Int = 5
    ) -> [V] {
        guard k > 0 && k <= vectors.count else {
            return k <= 0 ? [] : vectors
        }

        // Sort by magnitude
        let sorted = vectors.sorted { $0.magnitude < $1.magnitude }

        // Sample from each stratum
        let samplesPerStratum = k / strata
        let remainder = k % strata
        var sample: [V] = []

        for s in 0..<strata {
            let stratumStart = (vectors.count * s) / strata
            let stratumEnd = (vectors.count * (s + 1)) / strata
            let stratumSize = stratumEnd - stratumStart

            let samplesToTake = s < remainder ? samplesPerStratum + 1 : samplesPerStratum
            let actualSamples = min(samplesToTake, stratumSize)

            if actualSamples > 0 {
                let stratumVectors = Array(sorted[stratumStart..<stratumEnd])
                sample.append(contentsOf: randomSample(from: stratumVectors, k: actualSamples))
            }
        }

        return sample
    }

    // MARK: - Private Helpers

    @inline(__always)
    private static func minChunk(forDim dim: Int) -> Int {
        switch dim { case 1536: return 512; case 768: return 256; default: return 256 }
    }

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

    @inline(__always)
    private static func selectTopK(from distances: [Float], k: Int) -> [(index: Int, distance: Float)] {
        let n = distances.count
        let kClamped = min(k, n)
        // Small k: max-heap on precomputed distances
        if kClamped < n / 10 {
            var heap = [(index: Int, distance: Float)]()
            heap.reserveCapacity(kClamped)
            for (i, d) in distances.enumerated() {
                if heap.count < kClamped {
                    heap.append((i, d))
                    if heap.count == kClamped { heap.sort { $0.distance > $1.distance } }
                } else if d < heap[0].distance {
                    heap[0] = (i, d)
                    // Simple bubble-down restore
                    var idx = 0
                    while idx < kClamped - 1 && heap[idx].distance < heap[idx + 1].distance {
                        heap.swapAt(idx, idx + 1)
                        idx += 1
                    }
                }
            }
            return heap.sorted { $0.distance < $1.distance }
        } else {
            // Large k: sort pairs
            let pairs = distances.enumerated().map { (index: $0.offset, distance: $0.element) }
            return Array(pairs.sorted { $0.distance < $1.distance }.prefix(kClamped))
        }
    }

    private static func heapSelect<V: VectorProtocol, M: DistanceMetric>(
        query: V,
        vectors: [V],
        k: Int,
        metric: M
    ) -> [(index: Int, distance: Float)] where M.Scalar == Float, V.Scalar == Float {
        var heap = [(index: Int, distance: Float)]()
        heap.reserveCapacity(k)

        for (index, vector) in vectors.enumerated() {
            let distance = metric.distance(query, vector)

            if heap.count < k {
                heap.append((index: index, distance: distance))
                if heap.count == k {
                    // Heapify (max heap - largest at top)
                    heap.sort { $0.distance > $1.distance }
                }
            } else if distance < heap[0].distance {
                // Replace top element and restore heap
                heap[0] = (index: index, distance: distance)
                // Simple bubble down
                var i = 0
                while i < k - 1 {
                    if heap[i].distance < heap[i + 1].distance {
                        heap.swapAt(i, i + 1)
                        i += 1
                    } else {
                        break
                    }
                }
            }
        }

        // Sort in ascending order before returning
        return heap.sorted { $0.distance < $1.distance }
    }
}

// MARK: - Batch Processing Extensions

internal extension Array where Element: VectorProtocol {
    /// Find k nearest neighbors to a query
    func findNearest<M: DistanceMetric>(
        to query: Element,
        k: Int,
        metric: M = EuclideanDistance()
    ) -> [(index: Int, distance: Float)] where M.Scalar == Float, Element.Scalar == Float {
        SyncBatchOperations.findNearest(to: query, in: self, k: k, metric: metric)
    }

    /// Compute pairwise distances
    func pairwiseDistances<M: DistanceMetric>(
        metric: M = EuclideanDistance()
    ) -> [[Float]] where M.Scalar == Float, Element.Scalar == Float {
        SyncBatchOperations.pairwiseDistances(self, metric: metric)
    }

    /// Get batch statistics
    func batchStatistics() -> BatchStatistics where Element.Scalar == Float {
        SyncBatchOperations.statistics(for: self)
    }
}

// Extension specifically for Vector<D> types to support centroid
internal extension Array {
    /// Compute centroid of vectors
    func centroid<D: StaticDimension>() -> Vector<D>? where Element == Vector<D>, D.Storage: VectorStorageOperations {
        SyncBatchOperations.centroid(of: self)
    }
}
