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

    /// Threshold (in bytes) above which distance matrix row allocations use the buffer pool
    @usableFromInline internal static let largeRowThresholdBytes: Int = 64 * 1024
    
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
        guard k > 0 else { throw VectorError.invalidDimension(k, reason: "k must be positive") }
        guard !vectors.isEmpty else { throw VectorError.invalidDimension(0, reason: "Vector collection cannot be empty") }
        // Validate dimensions match between query and candidates
        if let first = vectors.first, first.scalarCount != query.scalarCount {
            throw VectorError.dimensionMismatch(expected: query.scalarCount, actual: first.scalarCount)
        }
        
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
        guard !queries.isEmpty else { throw VectorError.invalidDimension(0, reason: "queries cannot be empty") }
        guard !vectors.isEmpty else { throw VectorError.invalidDimension(0, reason: "vectors cannot be empty") }
        // Validate consistent query dimensions and match with vectors
        let expectedDim = queries.first!.scalarCount
        for q in queries {
            if q.scalarCount != expectedDim {
                throw VectorError.dimensionMismatch(expected: expectedDim, actual: q.scalarCount)
            }
        }
        if let firstVec = vectors.first, firstVec.scalarCount != expectedDim {
            throw VectorError.dimensionMismatch(expected: expectedDim, actual: firstVec.scalarCount)
        }
        return try await computeProvider.parallelExecute(items: 0..<queries.count) { i in
            try await findNearest(
                to: queries[i],
                in: vectors,
                k: k,
                metric: metric
            )
        }
    }

    // MARK: - Transformations

    /// Map operation over elements of vectors with automatic parallelization
    public static func map<V: VectorProtocol & VectorFactory>(
        _ vectors: [V],
        transform: @Sendable @escaping (Float) -> Float
    ) async throws -> [V] where V.Scalar == Float {
        guard !vectors.isEmpty else { return [] }
        if vectors.count > 1000 {
            return try await computeProvider.parallelExecute(items: 0..<vectors.count) { i in
                let v = vectors[i]
                let result: [Float] = .init(unsafeUninitializedCapacity: v.scalarCount) { buffer, initializedCount in
                    v.withUnsafeBufferPointer { src in
                        for j in 0..<src.count { buffer[j] = transform(src[j]) }
                    }
                    initializedCount = v.scalarCount
                }
                return try V.create(from: result)
            }
        } else {
            return try vectors.map { v in
                let result: [Float] = .init(unsafeUninitializedCapacity: v.scalarCount) { buffer, initializedCount in
                    v.withUnsafeBufferPointer { src in
                        for j in 0..<src.count { buffer[j] = transform(src[j]) }
                    }
                    initializedCount = v.scalarCount
                }
                return try V.create(from: result)
            }
        }
    }

    /// Add scalar to all elements of each vector
    public static func add<V: VectorProtocol & VectorFactory>(
        _ vectors: [V],
        _ scalar: V.Scalar
    ) async throws -> [V] where V.Scalar == Float {
        try await map(vectors, transform: { $0 + scalar })
    }

    /// Multiply all elements of each vector by a scalar
    public static func multiply<V: VectorProtocol & VectorFactory>(
        _ vectors: [V],
        by scalar: V.Scalar
    ) async throws -> [V] where V.Scalar == Float {
        try await map(vectors, transform: { $0 * scalar })
    }

    /// Normalize vectors to unit length (zero vector stays zero)
    public static func normalize<V: VectorProtocol & VectorFactory>(
        _ vectors: [V]
    ) async throws -> [V] where V.Scalar == Float {
        guard !vectors.isEmpty else { return [] }
        if vectors.count > 1000 {
            return try await computeProvider.parallelExecute(items: 0..<vectors.count) { i in
                let v = vectors[i]
                let mag = v.magnitude
                if mag > Float.ulpOfOne {
                    let inv = 1 / mag
                    // Fast scalar multiply with unsafe-uninitialized array
                    let result: [Float] = .init(unsafeUninitializedCapacity: v.scalarCount) { buffer, initializedCount in
                        v.withUnsafeBufferPointer { src in
                            for j in 0..<src.count { buffer[j] = src[j] * inv }
                        }
                        initializedCount = v.scalarCount
                    }
                    return try V.create(from: result)
                } else {
                    return try V.create(from: Array(repeating: 0, count: v.scalarCount))
                }
            }
        } else {
            return try vectors.map { v in
                let mag = v.magnitude
                if mag > Float.ulpOfOne {
                    let inv = 1 / mag
                    let result: [Float] = .init(unsafeUninitializedCapacity: v.scalarCount) { buffer, initializedCount in
                        v.withUnsafeBufferPointer { src in
                            for j in 0..<src.count { buffer[j] = src[j] * inv }
                        }
                        initializedCount = v.scalarCount
                    }
                    return try V.create(from: result)
                } else {
                    return try V.create(from: Array(repeating: 0, count: v.scalarCount))
                }
            }
        }
    }

    /// Combine two vector arrays element-wise
    public static func combine<V: VectorProtocol & VectorFactory>(
        _ vectors1: [V],
        _ vectors2: [V],
        _ operation: @Sendable @escaping (Float, Float) -> Float
    ) async throws -> [V] where V.Scalar == Float {
        guard vectors1.count == vectors2.count else {
            throw VectorError.invalidDimension(vectors2.count, reason: "Vector arrays must have same count: \(vectors1.count) != \(vectors2.count)")
        }
        if let first1 = vectors1.first, let first2 = vectors2.first, first1.scalarCount != first2.scalarCount {
            throw VectorError.dimensionMismatch(expected: first1.scalarCount, actual: first2.scalarCount)
        }
        guard !vectors1.isEmpty else { return [] }
        if vectors1.count > 1000 {
            return try await computeProvider.parallelExecute(items: 0..<vectors1.count) { i in
                try V.createByCombining(vectors1[i], vectors2[i], operation)
            }
        } else {
            return try zip(vectors1, vectors2).map { v1, v2 in
                try V.createByCombining(v1, v2, operation)
            }
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
            let m = setB.count
            let rowBytes = m * MemoryLayout<Float>.stride
            // Use buffer pool for large rows to reduce allocator pressure; otherwise use uninitialized array
            if rowBytes >= largeRowThresholdBytes {
                let handle = try await bufferProvider.acquire(size: rowBytes)
                // Ensure release after array construction
                defer { Task { await bufferProvider.release(handle) } }
                let fptr = handle.pointer.bindMemory(to: Float.self, capacity: m)
                for j in 0..<m {
                    fptr[j] = metric.distance(setA[i], setB[j])
                }
                let row = Array(UnsafeBufferPointer(start: fptr, count: m))
                return row
            } else {
                return [Float](unsafeUninitializedCapacity: m) { buffer, initializedCount in
                    for j in 0..<m {
                        buffer[j] = metric.distance(setA[i], setB[j])
                    }
                    initializedCount = m
                }
            }
        }
    }

    /// Compute pairwise distance matrix for a single set
    public static func distanceMatrix<V: VectorProtocol, M: DistanceMetric>(
        _ vectors: [V],
        metric: M = EuclideanDistance()
    ) async throws -> [[Float]] where V.Scalar == Float, M.Scalar == Float {
        try await distanceMatrix(between: vectors, and: vectors, metric: metric)
    }
    
    // MARK: - Vector Operations
    
    /// Compute centroid of vectors
    ///
    /// - Parameter vectors: Collection of vectors
    /// - Returns: Centroid vector
    public static func centroid<V: VectorProtocol & VectorFactory>(
        of vectors: [V]
    ) -> V where V.Scalar == Float {
        guard !vectors.isEmpty else {
            let dimension = vectors.first?.scalarCount ?? 0
            return try! V.create(from: [Float](repeating: 0, count: dimension))
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
        
        return try! V.create(from: result)
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
