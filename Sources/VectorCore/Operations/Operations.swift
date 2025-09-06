// VectorCore: Modern Operations API
//
// Clean API using provider abstraction - no legacy code
//

import Foundation

/// High-performance vector operations
///
/// Operations provides optimized vector computations with pluggable providers
/// for compute and SIMD operations. Default providers are pure Swift for
/// cross-platform compatibility.
///
/// ## Usage
/// ```swift
/// // Default: Pure Swift implementation
/// let results = try await Operations.findNearest(to: query, in: database, k: 10)
/// 
/// // With custom providers (e.g., from VectorAccelerate)
/// Operations.computeProvider = MetalComputeProvider()
/// Operations.simdProvider = AccelerateSIMDProvider()
/// ```
public enum Operations {
    
    // MARK: - Provider Configuration
    
    /// The compute provider for execution strategy
    @TaskLocal public static var computeProvider: any ComputeProvider = CPUComputeProvider.automatic
    
    /// The SIMD provider for vectorized operations
    @TaskLocal public static var simdProvider: any ArraySIMDProvider = SwiftSIMDProvider()
    
    /// The buffer provider for memory management
    @TaskLocal public static var bufferProvider: any BufferProvider = SwiftBufferPool.shared
    
    // MARK: - Nearest Neighbor Search
    
    /// Find k nearest neighbors to a query vector
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
        precondition(k > 0, "k must be positive")
        precondition(!vectors.isEmpty, "Vector collection cannot be empty")
        
        let distances = try await computeDistances(
            from: query,
            to: vectors,
            metric: metric
        )
        
        // Find k smallest distances
        let results = distances
            .enumerated()
            .map { NearestNeighborResult(index: $0.offset, distance: $0.element) }
            .sorted { $0.distance < $1.distance }
            .prefix(min(k, vectors.count))
        
        return Array(results)
    }
    
    /// Find k nearest neighbors for multiple queries
    ///
    /// - Parameters:
    ///   - queries: Query vectors
    ///   - vectors: Search space
    ///   - k: Number of neighbors per query
    ///   - metric: Distance metric
    /// - Returns: Results for each query
    public static func findNearestBatch<V: VectorProtocol, M: DistanceMetric>(
        queries: [V],
        in vectors: [V],
        k: Int = 1,
        metric: M = EuclideanDistance()
    ) async throws -> [[NearestNeighborResult]] where V.Scalar == Float, M.Scalar == Float {
        try await computeProvider.parallelExecute(items: 0..<queries.count) { i in
            try await findNearest(
                to: queries[i],
                in: vectors,
                k: k,
                metric: metric
            )
        }
    }
    
    // MARK: - Distance Computation
    
    /// Compute distances from query to all vectors
    private static func computeDistances<V: VectorProtocol, M: DistanceMetric>(
        from query: V,
        to vectors: [V],
        metric: M
    ) async throws -> [Float] where V.Scalar == Float, M.Scalar == Float {
        // For large datasets, parallelize
        if vectors.count > 1000 {
            return try await computeProvider.parallelExecute(items: 0..<vectors.count) { i in
                metric.distance(query, vectors[i])
            }
        } else {
            // Sequential for small datasets
            return vectors.map { metric.distance(query, $0) }
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
        try await computeProvider.parallelExecute(items: 0..<setA.count) { i in
            setB.map { metric.distance(setA[i], $0) }
        }
    }
    
    // MARK: - Vector Operations
    
    /// Compute centroid of vectors
    ///
    /// - Parameter vectors: Collection of vectors
    /// - Returns: Centroid vector
    public static func centroid<V: VectorProtocol>(
        of vectors: [V]
    ) -> V where V.Scalar == Float, V: ExpressibleByArrayLiteral, V.ArrayLiteralElement == Float {
        guard !vectors.isEmpty else {
            let dimension = vectors.first?.scalarCount ?? 0
            return try! V( [Float](repeating: 0, count: dimension))
        }
        
        let dimension = vectors.first!.scalarCount
        var sum = [Float](repeating: 0, count: dimension)
        
        // Sum all vectors
        for vector in vectors {
            let values = vector.toArray()
            sum = simdProvider.add(sum, values)
        }
        
        // Divide by count
        let scale = 1.0 / Float(vectors.count)
        let result = simdProvider.multiply(sum, by: scale)
        
        return try! V( result)
    }
    
    /// Normalize vectors to unit length
    ///
    /// - Parameter vectors: Vectors to normalize
    /// - Returns: Normalized vectors
    public static func normalize<V: VectorProtocol>(
        _ vectors: [V]
    ) async throws -> [V] where V.Scalar == Float {
        try await computeProvider.parallelExecute(items: 0..<vectors.count) { i in
            let values = vectors[i].toArray()
            let normalized = simdProvider.normalize(values)
            return try! V( normalized)
        }
    }
    
    // MARK: - Statistics
    
    /// Compute statistics for vector collection
    ///
    /// - Parameter vectors: Vector collection
    /// - Returns: Statistical summary
    public static func statistics<V: VectorProtocol>(
        for vectors: [V]
    ) -> VectorStatistics where V.Scalar == Float {
        guard !vectors.isEmpty else {
            return VectorStatistics(
                count: 0,
                dimensions: vectors.first?.scalarCount ?? 0,
                mean: [],
                min: [],
                max: [],
                magnitudes: VectorStatistics.MagnitudeStats(min: 0, max: 0, mean: 0)
            )
        }
        
        let dimension = vectors.first!.scalarCount
        var mins = [Float](repeating: .infinity, count: dimension)
        var maxs = [Float](repeating: -.infinity, count: dimension)
        var sums = [Float](repeating: 0, count: dimension)
        var magnitudes: [Float] = []
        
        for vector in vectors {
            let values = vector.toArray()
            
            // Update component-wise stats
            for i in 0..<dimension {
                mins[i] = min(mins[i], values[i])
                maxs[i] = max(maxs[i], values[i])
                sums[i] += values[i]
            }
            
            // Track magnitude
            magnitudes.append(simdProvider.magnitude(values))
        }
        
        let count = Float(vectors.count)
        let mean = sums.map { $0 / count }
        
        let magMin = magnitudes.min() ?? 0
        let magMax = magnitudes.max() ?? 0
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