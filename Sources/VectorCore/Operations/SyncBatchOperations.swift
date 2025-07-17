// VectorCore: Synchronous Batch Operations
//
// High-performance synchronous batch processing for vector operations
//

import Foundation
import Accelerate

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
public enum SyncBatchOperations {
    
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
    public static func findNearest<V: ExtendedVectorProtocol>(
        to query: V,
        in vectors: [V],
        k: Int,
        metric: any DistanceMetric = EuclideanDistance()
    ) -> [(index: Int, distance: Float)] {
        guard k > 0 else { return [] }
        guard !vectors.isEmpty else { return [] }
        
        // For small k, use heap selection
        if k < vectors.count / 10 {
            return heapSelect(query: query, vectors: vectors, k: k, metric: metric)
        }
        
        // For larger k, compute all distances and partial sort
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
    public static func findWithinRadius<V: ExtendedVectorProtocol>(
        of query: V,
        in vectors: [V],
        radius: Float,
        metric: any DistanceMetric = EuclideanDistance()
    ) -> [(index: Int, distance: Float)] {
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
    public static func map<V: BaseVectorProtocol, U: BaseVectorProtocol>(
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
    public static func mapInPlace<V: BaseVectorProtocol>(
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
    public static func filter<V: BaseVectorProtocol>(
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
    public static func partition<V: BaseVectorProtocol>(
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
    public static func centroid<D: Dimension>(
        of vectors: [Vector<D>]
    ) -> Vector<D>? where D.Storage: VectorStorageOperations {
        guard !vectors.isEmpty else { return nil }
        
        // For single vector, return copy
        if vectors.count == 1 {
            return vectors[0]
        }
        
        // Initialize accumulator with first vector
        var sum = vectors[0]
        
        // Add remaining vectors
        for i in 1..<vectors.count {
            sum = sum + vectors[i]
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
    public static func weightedCentroid<D: Dimension>(
        of vectors: [Vector<D>],
        weights: [Float]
    ) -> Vector<D>? where D.Storage: VectorStorageOperations {
        guard !vectors.isEmpty else { return nil }
        guard vectors.count == weights.count else {
            fatalError("Vectors and weights must have the same count")
        }
        
        var totalWeight: Float = 0
        var sum = Vector<D>.zeros()
        
        for i in 0..<vectors.count {
            sum = sum + (vectors[i] * weights[i])
            totalWeight += weights[i]
        }
        
        guard totalWeight > 0 else { return nil }
        
        return sum / totalWeight
    }
    
    /// Compute element-wise sum of vectors
    ///
    /// - Parameter vectors: Input vectors
    /// - Returns: Sum vector, or nil if input is empty
    public static func sum<D: Dimension>(
        _ vectors: [Vector<D>]
    ) -> Vector<D>? where D.Storage: VectorStorageOperations {
        guard !vectors.isEmpty else { return nil }
        
        var result = vectors[0]
        for i in 1..<vectors.count {
            result = result + vectors[i]
        }
        
        return result
    }
    
    /// Compute element-wise mean of vectors
    ///
    /// - Parameter vectors: Input vectors
    /// - Returns: Mean vector, or nil if input is empty
    public static func mean<D: Dimension>(
        _ vectors: [Vector<D>]
    ) -> Vector<D>? where D.Storage: VectorStorageOperations {
        centroid(of: vectors)
    }
    
    // MARK: - Statistical Operations
    
    /// Compute batch statistics
    ///
    /// - Parameter vectors: Input vectors
    /// - Returns: Statistics including count, mean magnitude, and std deviation
    public static func statistics<V: ExtendedVectorProtocol>(
        for vectors: [V]
    ) -> BatchStatistics {
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
    public static func findOutliers<V: ExtendedVectorProtocol>(
        in vectors: [V],
        zscoreThreshold: Float = 3
    ) -> [Int] {
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
    public static func pairwiseDistances<V: ExtendedVectorProtocol>(
        _ vectors: [V],
        metric: any DistanceMetric = EuclideanDistance()
    ) -> [[Float]] {
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
    public static func batchDistances<V: ExtendedVectorProtocol>(
        from queries: [V],
        to candidates: [V],
        metric: any DistanceMetric = EuclideanDistance()
    ) -> [[Float]] {
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
    public static func assignToCentroids<V: ExtendedVectorProtocol>(
        _ vectors: [V],
        centroids: [V],
        metric: any DistanceMetric = EuclideanDistance()
    ) -> [Int] {
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
    public static func updateCentroids<D: Dimension>(
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
    public static func randomSample<V>(
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
    public static func stratifiedSample<V: ExtendedVectorProtocol>(
        from vectors: [V],
        k: Int,
        strata: Int = 5
    ) -> [V] {
        guard k > 0 && k <= vectors.count else {
            return k <= 0 ? [] : vectors
        }
        
        // Sort by magnitude
        let sorted = vectors.enumerated().sorted { $0.element.magnitude < $1.element.magnitude }
        
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
                let stratumVectors = sorted[stratumStart..<stratumEnd].map { $0.element }
                sample.append(contentsOf: randomSample(from: Array(stratumVectors), k: actualSamples))
            }
        }
        
        return sample
    }
    
    // MARK: - Private Helpers
    
    private static func heapSelect<V: ExtendedVectorProtocol>(
        query: V,
        vectors: [V],
        k: Int,
        metric: any DistanceMetric
    ) -> [(index: Int, distance: Float)] {
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

public extension Array where Element: ExtendedVectorProtocol {
    /// Find k nearest neighbors to a query
    func findNearest(
        to query: Element,
        k: Int,
        metric: any DistanceMetric = EuclideanDistance()
    ) -> [(index: Int, distance: Float)] {
        SyncBatchOperations.findNearest(to: query, in: self, k: k, metric: metric)
    }
    
    /// Compute pairwise distances
    func pairwiseDistances(
        metric: any DistanceMetric = EuclideanDistance()
    ) -> [[Float]] {
        SyncBatchOperations.pairwiseDistances(self, metric: metric)
    }
    
    /// Get batch statistics
    var batchStatistics: BatchStatistics {
        SyncBatchOperations.statistics(for: self)
    }
}

// Extension specifically for Vector<D> types to support centroid
public extension Array {
    /// Compute centroid of vectors
    func centroid<D: Dimension>() -> Vector<D>? where Element == Vector<D>, D.Storage: VectorStorageOperations {
        SyncBatchOperations.centroid(of: self)
    }
}