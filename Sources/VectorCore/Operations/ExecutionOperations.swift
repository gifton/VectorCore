//
//  ExecutionOperations.swift
//  VectorCore
//
//

import Foundation

// MARK: - Operations Namespace

/// High-performance vector operations with ExecutionContext support
public enum ExecutionOperations {
    // Thresholds based on empirical testing
    internal static let parallelThreshold = 1000
    internal static let vectorizedThreshold = 64
    
    // Thread-local storage for temporary buffers
    @TaskLocal static var temporaryBuffers: BufferPool?
}

// MARK: - K-Nearest Neighbors

extension ExecutionOperations {
    /// Find k nearest neighbors to a query vector
    /// - Parameters:
    ///   - query: Query vector
    ///   - vectors: Search space
    ///   - k: Number of neighbors to find
    ///   - metric: Distance metric (default: Euclidean)
    ///   - context: Execution context (default: automatic CPU)
    /// - Returns: Array of (index, distance) pairs sorted by distance
    public static func findNearest<V: VectorProtocol, M: DistanceMetric>(
        to query: V,
        in vectors: [V],
        k: Int = 10,
        metric: M = EuclideanDistance(),
        context: any ExecutionContext = CPUContext.automatic
    ) async throws -> [(index: Int, distance: Float)] where M.Scalar == Float, V.Scalar == Float {
        
        let count = vectors.count
        guard count > 0 else { return [] }
        guard k > 0 else { return [] }
        
        // Validate dimensions
        try validateDimensions(query, vectors)
        
        // Choose execution strategy based on size
        if count < parallelThreshold {
            return try await context.execute {
                findNearestSequential(
                    query: query,
                    vectors: vectors,
                    k: k,
                    metric: metric
                )
            }
        }
        
        // Parallel execution for large datasets
        switch context.device {
        case .cpu:
            return try await findNearestParallelCPU(
                query: query,
                vectors: vectors,
                k: k,
                metric: metric,
                context: context
            )
            
        case .gpu:
            return try await findNearestGPU(
                query: query,
                vectors: vectors,
                k: k,
                metric: metric,
                context: context
            )
            
        case .neural:
            throw VectorError.unsupportedDevice("Neural Engine not yet supported")
        }
    }
    
    // Sequential implementation
    @usableFromInline
    internal static func findNearestSequential<V: VectorProtocol, M: DistanceMetric>(
        query: V,
        vectors: [V],
        k: Int,
        metric: M
    ) -> [(index: Int, distance: Float)] where M.Scalar == Float, V.Scalar == Float {
        
        // Use a min-heap for efficient k-nearest tracking
        var heap = KNearestHeap(k: k)
        
        // For small k, use heap; for large k, sort all
        if k < vectors.count / 10 {
            // Heap-based selection
            for (index, vector) in vectors.enumerated() {
                let distance = metric.distance(query, vector)
                heap.insert(index: index, distance: distance)
            }
            return heap.getSorted()
        } else {
            // Sort-based selection
            let distances = vectors.enumerated().map { index, vector in
                (index, metric.distance(query, vector))
            }
            return Array(distances.sorted(by: { $0.1 < $1.1 }).prefix(k))
        }
    }
    
    // Parallel CPU implementation
    private static func findNearestParallelCPU<V: VectorProtocol, M: DistanceMetric>(
        query: V,
        vectors: [V],
        k: Int,
        metric: M,
        context: any ExecutionContext
    ) async throws -> [(index: Int, distance: Float)] where M.Scalar == Float, V.Scalar == Float {
        
        let count = vectors.count
        let chunkSize = max(context.preferredChunkSize, count / context.maxThreadCount)
        let chunkCount = (count + chunkSize - 1) / chunkSize
        
        // Process chunks in parallel
        let chunkResults = try await withThrowingTaskGroup(
            of: [(index: Int, distance: Float)].self
        ) { group in
            for chunkIndex in 0..<chunkCount {
                let start = chunkIndex * chunkSize
                let end = min(start + chunkSize, count)
                
                group.addTask {
                    let chunkVectors = Array(vectors[start..<end])
                    let chunkK = min(k, chunkVectors.count)
                    
                    var results = findNearestSequential(
                        query: query,
                        vectors: chunkVectors,
                        k: chunkK,
                        metric: metric
                    )
                    
                    // Adjust indices to global range
                    for i in results.indices {
                        results[i].index += start
                    }
                    
                    return results
                }
            }
            
            // Collect all chunk results
            var allResults: [(index: Int, distance: Float)] = []
            for try await chunkResult in group {
                allResults.append(contentsOf: chunkResult)
            }
            return allResults
        }
        
        // Merge chunk results
        return Array(chunkResults.sorted(by: { $0.distance < $1.distance }).prefix(k))
    }
    
    // GPU implementation placeholder
    private static func findNearestGPU<V: VectorProtocol, M: DistanceMetric>(
        query: V,
        vectors: [V],
        k: Int,
        metric: M,
        context: any ExecutionContext
    ) async throws -> [(index: Int, distance: Float)] where M.Scalar == Float, V.Scalar == Float {
        throw VectorError.unsupportedDevice("GPU execution not yet implemented")
    }
}

// MARK: - Batch Operations

extension ExecutionOperations {
    /// Process multiple queries efficiently
    public static func findNearestBatch<V: VectorProtocol, M: DistanceMetric>(
        queries: [V],
        in vectors: [V],
        k: Int = 10,
        metric: M = EuclideanDistance(),
        context: any ExecutionContext = CPUContext.automatic
    ) async throws -> [[(index: Int, distance: Float)]] where M.Scalar == Float, V.Scalar == Float {
        
        // Validate inputs
        try validateNonEmpty(queries, operation: "Batch query")
        try validateNonEmpty(vectors, operation: "Batch search")
        try validateConsistentDimensions(queries)
        
        // Validate queries match vectors dimension
        if let firstQuery = queries.first, let firstVector = vectors.first {
            guard firstQuery.scalarCount == firstVector.scalarCount else {
                throw VectorError.dimensionMismatch(
                    expected: firstVector.scalarCount,
                    actual: firstQuery.scalarCount
                )
            }
        }
        
        // For single query, use non-batch version
        if queries.count == 1 {
            let result = try await findNearest(
                to: queries[0],
                in: vectors,
                k: k,
                metric: metric,
                context: context
            )
            return [result]
        }
        
        // Process queries in parallel
        return try await withThrowingTaskGroup(
            of: (Int, [(index: Int, distance: Float)]).self
        ) { group in
            for (queryIndex, query) in queries.enumerated() {
                group.addTask {
                    let result = try await findNearest(
                        to: query,
                        in: vectors,
                        k: k,
                        metric: metric,
                        context: context
                    )
                    return (queryIndex, result)
                }
            }
            
            // Collect results in order
            var results = Array(repeating: [(index: Int, distance: Float)](), count: queries.count)
            for try await (index, result) in group {
                results[index] = result
            }
            return results
        }
    }
}

// MARK: - Vector Transformations

extension ExecutionOperations {
    /// Map operation with automatic parallelization
    public static func map<V: VectorProtocol & VectorFactory>(
        _ vectors: [V],
        transform: @Sendable @escaping (Float) -> Float,
        context: any ExecutionContext = CPUContext.automatic
    ) async throws -> [V] where V.Scalar == Float {
        
        if vectors.count < parallelThreshold {
            // Sequential for small arrays
            return try await context.execute {
                vectors.map { vector in
                    var result = Array(repeating: Float(0), count: vector.scalarCount)
                    vector.withUnsafeBufferPointer { buffer in
                        for i in 0..<buffer.count {
                            result[i] = transform(buffer[i])
                        }
                    }
                    
                    // Use factory protocol to create appropriate vector type
                    return try! V.create(from: result)
                }
            }
        }
        
        // Parallel for large arrays
        return try await withThrowingTaskGroup(of: (Int, V).self) { group in
            for (index, vector) in vectors.enumerated() {
                group.addTask {
                    var result = Array(repeating: Float(0), count: vector.scalarCount)
                    vector.withUnsafeBufferPointer { buffer in
                        for i in 0..<buffer.count {
                            result[i] = transform(buffer[i])
                        }
                    }
                    
                    // Use factory protocol to create appropriate vector type
                    let newVector = try! V.create(from: result)
                    return (index, newVector)
                }
            }
            
            var results = Array<V?>(repeating: nil, count: vectors.count)
            for try await (index, result) in group {
                results[index] = result
            }
            return results.compactMap { $0 }
        }
    }
    
    /// Reduce operation
    public static func reduce<V: VectorProtocol>(
        _ vectors: [V],
        _ initialResult: V,
        _ nextPartialResult: @Sendable @escaping (V, V) -> V,
        context: any ExecutionContext = CPUContext.automatic
    ) async throws -> V {
        
        guard !vectors.isEmpty else { return initialResult }
        
        if vectors.count < parallelThreshold {
            // Sequential reduction
            return try await context.execute {
                vectors.reduce(initialResult, nextPartialResult)
            }
        }
        
        // Parallel reduction using divide-and-conquer
        return try await parallelReduce(
            vectors,
            initialResult,
            nextPartialResult,
            context: context
        )
    }
    
    private static func parallelReduce<V: VectorProtocol>(
        _ vectors: [V],
        _ initialResult: V,
        _ combine: @Sendable @escaping (V, V) -> V,
        context: any ExecutionContext
    ) async throws -> V {
        
        let count = vectors.count
        guard count > 1 else {
            return count == 1 ? combine(initialResult, vectors[0]) : initialResult
        }
        
        // Divide into chunks
        let chunkSize = max(2, count / context.maxThreadCount)
        
        // Reduce chunks in parallel
        let chunkResults = try await withThrowingTaskGroup(of: V.self) { group in
            for chunkStart in stride(from: 0, to: count, by: chunkSize) {
                let chunkEnd = min(chunkStart + chunkSize, count)
                let chunk = Array(vectors[chunkStart..<chunkEnd])
                
                group.addTask {
                    chunk.reduce(initialResult, combine)
                }
            }
            
            var results: [V] = []
            for try await result in group {
                results.append(result)
            }
            return results
        }
        
        // Recursively reduce chunk results
        if chunkResults.count > 1 {
            return try await parallelReduce(
                chunkResults,
                initialResult,
                combine,
                context: context
            )
        } else {
            return chunkResults[0]
        }
    }
}

// MARK: - Distance Matrix

extension ExecutionOperations {
    /// Compute pairwise distance matrix
    public static func distanceMatrix<V: VectorProtocol, M: DistanceMetric>(
        _ vectors: [V],
        metric: M = EuclideanDistance(),
        context: any ExecutionContext = CPUContext.automatic
    ) async throws -> [[Float]] where M.Scalar == Float, V.Scalar == Float {
        
        try validateConsistentDimensions(vectors)
        
        let n = vectors.count
        
        if n * n < parallelThreshold {
            // Sequential for small matrices
            return try await context.execute {
                var matrix = Array(repeating: Array(repeating: Float(0), count: n), count: n)
                for i in 0..<n {
                    for j in (i+1)..<n {
                        let distance = metric.distance(vectors[i], vectors[j])
                        matrix[i][j] = distance
                        matrix[j][i] = distance  // Symmetric
                    }
                }
                return matrix
            }
        }
        
        // Parallel computation
        return try await withThrowingTaskGroup(of: (Int, Int, Float).self) { group in
            for i in 0..<n {
                for j in (i+1)..<n {
                    group.addTask {
                        let distance = metric.distance(vectors[i], vectors[j])
                        return (i, j, distance)
                    }
                }
            }
            
            var matrix = Array(repeating: Array(repeating: Float(0), count: n), count: n)
            for try await (i, j, distance) in group {
                matrix[i][j] = distance
                matrix[j][i] = distance  // Symmetric
            }
            
            return matrix
        }
    }
}

// MARK: - Vector Statistics

extension ExecutionOperations {
    /// Compute centroid of vectors
    public static func centroid<V: VectorProtocol>(
        of vectors: [V],
        context: any ExecutionContext = CPUContext.automatic
    ) async throws -> DynamicVector where V.Scalar == Float {
        
        try validateNonEmpty(vectors, operation: "Centroid computation")
        try validateConsistentDimensions(vectors)
        
        let dimension = vectors.first!.scalarCount
        var sum = Array(repeating: Float(0), count: dimension)
        
        if vectors.count < parallelThreshold {
            // Sequential summation
            sum = try await context.execute {
                var localSum = Array(repeating: Float(0), count: dimension)
                for vector in vectors {
                    vector.withUnsafeBufferPointer { buffer in
                        localSum.withUnsafeMutableBufferPointer { resultBuffer in
                            SIMDOperations.FloatProvider.add(
                                resultBuffer.baseAddress!,
                                buffer.baseAddress!,
                                result: resultBuffer.baseAddress!,
                                count: dimension
                            )
                        }
                    }
                }
                return localSum
            }
        } else {
            // Parallel summation
            let partialSums = try await withThrowingTaskGroup(of: [Float].self) { group in
                let chunkSize = max(1, vectors.count / context.maxThreadCount)
                
                for chunkStart in stride(from: 0, to: vectors.count, by: chunkSize) {
                    let chunkEnd = min(chunkStart + chunkSize, vectors.count)
                    let chunk = Array(vectors[chunkStart..<chunkEnd])
                    
                    group.addTask {
                        var chunkSum = Array(repeating: Float(0), count: dimension)
                        for vector in chunk {
                            vector.withUnsafeBufferPointer { buffer in
                                chunkSum.withUnsafeMutableBufferPointer { chunkSumBuffer in
                                    SIMDOperations.FloatProvider.add(
                                        chunkSumBuffer.baseAddress!,
                                        buffer.baseAddress!,
                                        result: chunkSumBuffer.baseAddress!,
                                        count: dimension
                                    )
                                }
                            }
                        }
                        return chunkSum
                    }
                }
                
                var results: [[Float]] = []
                for try await partial in group {
                    results.append(partial)
                }
                return results
            }
            
            // Combine partial sums
            for partial in partialSums {
                sum.withUnsafeMutableBufferPointer { sumBuffer in
                    partial.withUnsafeBufferPointer { partialBuffer in
                        SIMDOperations.FloatProvider.add(
                            sumBuffer.baseAddress!,
                            partialBuffer.baseAddress!,
                            result: sumBuffer.baseAddress!,
                            count: dimension
                        )
                    }
                }
            }
        }
        
        // Divide by count
        let scale = 1.0 / Float(vectors.count)
        sum.withUnsafeMutableBufferPointer { sumBuffer in
            SIMDOperations.FloatProvider.multiplyScalar(
                sumBuffer.baseAddress!,
                scalar: scale,
                result: sumBuffer.baseAddress!,
                count: dimension
            )
        }
        
        return DynamicVector(sum)
    }
}

// MARK: - Validation Helpers

extension ExecutionOperations {
    /// Validate dimension consistency between query and vectors
    @inlinable
    internal static func validateDimensions<V: VectorProtocol>(_ query: V, _ vectors: [V]) throws {
        guard let first = vectors.first else { return }
        let expectedDim = query.scalarCount
        
        guard first.scalarCount == expectedDim else {
            throw VectorError.dimensionMismatch(
                expected: expectedDim,
                actual: first.scalarCount
            )
        }
        
        // Check all vectors if in debug mode
        #if DEBUG
        for (_, vector) in vectors.enumerated() {
            guard vector.scalarCount == expectedDim else {
                throw VectorError.dimensionMismatch(
                    expected: expectedDim,
                    actual: vector.scalarCount
                )
            }
        }
        #endif
    }
    
    /// Validate all vectors have consistent dimensions
    @inlinable
    internal static func validateConsistentDimensions<V: VectorProtocol>(_ vectors: [V]) throws {
        guard let first = vectors.first else { return }
        let expectedDim = first.scalarCount
        
        // In release mode, sample validation for performance
        #if !DEBUG
        // Check a sample of vectors in release mode (every 100th vector)
        for (index, vector) in vectors.enumerated() where index % 100 == 0 {
            guard vector.scalarCount == expectedDim else {
                throw VectorError.dimensionMismatch(
                    expected: expectedDim,
                    actual: vector.scalarCount
                )
            }
        }
        #else
        // Check all vectors in debug mode
        for (_, vector) in vectors.enumerated().dropFirst() {
            guard vector.scalarCount == expectedDim else {
                throw VectorError.dimensionMismatch(
                    expected: expectedDim,
                    actual: vector.scalarCount
                )
            }
        }
        #endif
    }
    
    /// Validate non-empty collection
    @inlinable
    internal static func validateNonEmpty<V: VectorProtocol>(_ vectors: [V], operation: String) throws {
        guard !vectors.isEmpty else {
            throw VectorError.invalidDimension(0, reason: "\(operation) requires non-empty vector collection")
        }
    }
}

// MARK: - Performance Context

/// Context for managing operation performance
public struct OperationContext: Sendable {
    public let bufferPool: BufferPool
    public let executionContext: any ExecutionContext
    
    public init(
        bufferPool: BufferPool = globalBufferPool,
        executionContext: any ExecutionContext = CPUContext.automatic
    ) {
        self.bufferPool = bufferPool
        self.executionContext = executionContext
    }
    
    /// Use temporary buffer for an operation
    public func withTemporaryBuffer<T>(
        count: Int,
        _ body: (UnsafeMutableBufferPointer<Float>) async throws -> T
    ) async rethrows -> T where T: Sendable {
        let sendableBuffer = await bufferPool.acquire(count: count)
        defer {
            Task { @Sendable in
                await bufferPool.release(sendableBuffer)
            }
        }
        return try await body(sendableBuffer.buffer)
    }
}
