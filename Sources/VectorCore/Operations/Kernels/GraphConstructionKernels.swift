//
//  GraphConstructionKernelImpl.swift
//  VectorCore
//
//  Kernel #16: Graph Construction Kernels
//  k-NN Graphs, Range Graphs, NSW, and Graph Generators
//

import Foundation
import Accelerate
import simd

// MARK: - Graph Construction Error Types

/// Errors that can occur during graph construction operations
public enum GraphConstructionError: Error, CustomStringConvertible {
    case invalidParameters(String)
    case dimensionMismatch(expected: Int, got: Int)
    case insufficientData(String)
    case invalidGraphStructure(String)
    case numericalInstability(String)

    public var description: String {
        switch self {
        case .invalidParameters(let message):
            return "Invalid parameters: \(message)"
        case .dimensionMismatch(let expected, let got):
            return "Dimension mismatch: expected \(expected), got \(got)"
        case .insufficientData(let message):
            return "Insufficient data: \(message)"
        case .invalidGraphStructure(let message):
            return "Invalid graph structure: \(message)"
        case .numericalInstability(let message):
            return "Numerical instability: \(message)"
        }
    }
}

// MARK: - Vector Type System Integration

/// Base protocol for graph-compatible vectors
public protocol GraphVector {
    var dimensions: Int { get }
    var magnitude: Float { get }

    func dotProduct(_ other: Self) -> Float
    func subtract(_ other: Self) -> Self
    func elementWiseMultiply(_ other: Self) -> Self
    func sum() -> Float
    func toFloatArray() -> [Float]

    subscript(index: Int) -> Float { get }
}

// MARK: - Extensions for existing VectorCore types

extension Vector512Optimized: GraphVector {
    public var dimensions: Int { scalarCount }

    public func subtract(_ other: Vector512Optimized) -> Vector512Optimized {
        return self - other
    }

    public func elementWiseMultiply(_ other: Vector512Optimized) -> Vector512Optimized {
        return self .* other
    }

    public func sum() -> Float {
        let total = storage.reduce(SIMD4<Float>(), +)
        return total.x + total.y + total.z + total.w
    }

    public func toFloatArray() -> [Float] {
        return toArray()
    }
}

extension Vector768Optimized: GraphVector {
    public var dimensions: Int { scalarCount }

    public func subtract(_ other: Vector768Optimized) -> Vector768Optimized {
        return self - other
    }

    public func elementWiseMultiply(_ other: Vector768Optimized) -> Vector768Optimized {
        return self .* other
    }

    public func sum() -> Float {
        let total = storage.reduce(SIMD4<Float>(), +)
        return total.x + total.y + total.z + total.w
    }

    public func toFloatArray() -> [Float] {
        return toArray()
    }
}

extension Vector1536Optimized: GraphVector {
    public var dimensions: Int { scalarCount }

    public func subtract(_ other: Vector1536Optimized) -> Vector1536Optimized {
        return self - other
    }

    public func elementWiseMultiply(_ other: Vector1536Optimized) -> Vector1536Optimized {
        return self .* other
    }

    public func sum() -> Float {
        let total = storage.reduce(SIMD4<Float>(), +)
        return total.x + total.y + total.z + total.w
    }

    public func toFloatArray() -> [Float] {
        return toArray()
    }
}

// MARK: - Generic Vector Support using ContiguousArray

extension ContiguousArray: GraphVector where Element == Float {
    public var dimensions: Int { count }

    public var magnitude: Float {
        var result: Float = 0
        self.withUnsafeBufferPointer { ptr in
            vDSP_svesq(ptr.baseAddress!, 1, &result, vDSP_Length(count))
        }
        return sqrt(result)
    }

    public func dotProduct(_ other: ContiguousArray<Float>) -> Float {
        precondition(dimensions == other.dimensions, "Vector dimensions must match")
        var result: Float = 0
        self.withUnsafeBufferPointer { aPtr in
            other.withUnsafeBufferPointer { bPtr in
                vDSP_dotpr(aPtr.baseAddress!, 1, bPtr.baseAddress!, 1, &result, vDSP_Length(count))
            }
        }
        return result
    }

    public func subtract(_ other: ContiguousArray<Float>) -> ContiguousArray<Float> {
        precondition(dimensions == other.dimensions, "Vector dimensions must match")
        var result = ContiguousArray<Float>(repeating: 0, count: count)
        self.withUnsafeBufferPointer { aPtr in
            other.withUnsafeBufferPointer { bPtr in
                result.withUnsafeMutableBufferPointer { resPtr in
                    vDSP_vsub(bPtr.baseAddress!, 1, aPtr.baseAddress!, 1, resPtr.baseAddress!, 1, vDSP_Length(count))
                }
            }
        }
        return result
    }

    public func elementWiseMultiply(_ other: ContiguousArray<Float>) -> ContiguousArray<Float> {
        precondition(dimensions == other.dimensions, "Vector dimensions must match")
        var result = ContiguousArray<Float>(repeating: 0, count: count)
        self.withUnsafeBufferPointer { aPtr in
            other.withUnsafeBufferPointer { bPtr in
                result.withUnsafeMutableBufferPointer { resPtr in
                    vDSP_vmul(aPtr.baseAddress!, 1, bPtr.baseAddress!, 1, resPtr.baseAddress!, 1, vDSP_Length(count))
                }
            }
        }
        return result
    }

    public func sum() -> Float {
        var result: Float = 0
        self.withUnsafeBufferPointer { ptr in
            vDSP_sve(ptr.baseAddress!, 1, &result, vDSP_Length(count))
        }
        return result
    }

    public func abs() -> ContiguousArray<Float> {
        var result = ContiguousArray<Float>(repeating: 0, count: count)
        self.withUnsafeBufferPointer { srcPtr in
            result.withUnsafeMutableBufferPointer { dstPtr in
                vDSP_vabs(srcPtr.baseAddress!, 1, dstPtr.baseAddress!, 1, vDSP_Length(count))
            }
        }
        return result
    }

    public func toFloatArray() -> [Float] {
        return Array(self)
    }

    // MARK: - Operator Overloads for Convenience

    /// Element-wise multiplication operator
    public static func * (lhs: ContiguousArray<Float>, rhs: ContiguousArray<Float>) -> ContiguousArray<Float> {
        precondition(lhs.count == rhs.count, "Vector dimensions must match")
        var result = ContiguousArray<Float>(repeating: 0, count: lhs.count)
        lhs.withUnsafeBufferPointer { lhsPtr in
            rhs.withUnsafeBufferPointer { rhsPtr in
                result.withUnsafeMutableBufferPointer { resPtr in
                    vDSP_vmul(lhsPtr.baseAddress!, 1, rhsPtr.baseAddress!, 1, resPtr.baseAddress!, 1, vDSP_Length(lhs.count))
                }
            }
        }
        return result
    }

    /// Element-wise addition operator
    public static func + (lhs: ContiguousArray<Float>, rhs: ContiguousArray<Float>) -> ContiguousArray<Float> {
        precondition(lhs.count == rhs.count, "Vector dimensions must match")
        var result = ContiguousArray<Float>(repeating: 0, count: lhs.count)
        lhs.withUnsafeBufferPointer { lhsPtr in
            rhs.withUnsafeBufferPointer { rhsPtr in
                result.withUnsafeMutableBufferPointer { resPtr in
                    vDSP_vadd(lhsPtr.baseAddress!, 1, rhsPtr.baseAddress!, 1, resPtr.baseAddress!, 1, vDSP_Length(lhs.count))
                }
            }
        }
        return result
    }

    /// Element-wise subtraction operator
    public static func - (lhs: ContiguousArray<Float>, rhs: ContiguousArray<Float>) -> ContiguousArray<Float> {
        return lhs.subtract(rhs)
    }
}

// MARK: - SparseMatrix Extension for Graph Construction

// SparseMatrix extension removed - using existing SparseMatrix from GraphPrimitivesKernels

// MARK: - Graph Construction Kernels

public struct GraphConstructionKernels {}

// MARK: - Helper Types for NSW

private struct NSWSearchResult: Comparable {
    let id: Int32
    let distance: Float

    static func < (lhs: NSWSearchResult, rhs: NSWSearchResult) -> Bool {
        return lhs.distance < rhs.distance
    }
}

extension GraphConstructionKernels {

    // MARK: - COO to CSR Conversion

    public static func cooToCSR(
        edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)>,
        rows: Int,
        cols: Int
    ) -> SparseMatrix {
        var sortedEdges = edges
        sortedEdges.sort { lhs, rhs in
            if lhs.row != rhs.row {
                return lhs.row < rhs.row
            }
            return lhs.col < rhs.col
        }

        var rowPointers = ContiguousArray<UInt32>(repeating: 0, count: rows + 1)
        var columnIndices = ContiguousArray<UInt32>()
        var values = ContiguousArray<Float>()

        let hasInputValues = edges.contains(where: { $0.value != nil })

        var currentRow: Int = 0
        var nnzCount = 0

        for edge in sortedEdges {
            let r = Int(edge.row)
            let c = edge.col

            if r >= rows || Int(c) >= cols { continue }

            let v = edge.value ?? 1.0

            while currentRow < r {
                rowPointers[currentRow + 1] = UInt32(nnzCount)
                currentRow += 1
            }

            if nnzCount > 0 && columnIndices[nnzCount - 1] == c && currentRow == r {
                if hasInputValues {
                    values[nnzCount - 1] = v
                }
            } else {
                columnIndices.append(c)
                if hasInputValues {
                    values.append(v)
                }
                nnzCount += 1
            }
        }

        while currentRow < rows {
            rowPointers[currentRow + 1] = UInt32(nnzCount)
            currentRow += 1
        }

        // Create the sparse matrix using the CSR data
        return try! SparseMatrix(
            rows: rows,
            cols: cols,
            rowPointers: rowPointers,
            columnIndices: columnIndices,
            values: values.isEmpty ? nil : values
        )
    }

    // MARK: - Distance Metrics

    public enum DistanceMetric: Sendable {
        case euclidean
        case cosine
        case manhattan
        case custom(@Sendable (ContiguousArray<Float>, ContiguousArray<Float>) -> Float)
    }

    @inlinable
    public static func computeDistance<Vector: GraphVector>(
        _ a: Vector,
        _ b: Vector,
        metric: DistanceMetric
    ) -> Float {
        precondition(a.dimensions == b.dimensions, "Vector dimensions must match")

        switch metric {
        case .euclidean:
            let diff = a.subtract(b)
            return diff.magnitude

        case .cosine:
            let dotProduct = a.dotProduct(b)
            let magnitudeA = a.magnitude
            let magnitudeB = b.magnitude

            guard magnitudeA > Float.ulpOfOne && magnitudeB > Float.ulpOfOne else {
                // Return maximum distance (2.0) for zero vectors
                return 2.0
            }

            let similarity = dotProduct / (magnitudeA * magnitudeB)
            return 1.0 - max(-1.0, min(1.0, similarity))

        case .manhattan:
            let diff = a.subtract(b)
            var absSum: Float = 0

            // Check each component for infinity/NaN
            for i in 0..<a.dimensions {
                let component = abs(diff[i])
                guard component.isFinite else {
                    return Float.infinity
                }
                absSum += component
            }

            // Check final sum for overflow
            guard absSum.isFinite else {
                return Float.infinity
            }

            return absSum

        case .custom(let f):
            let aArray = ContiguousArray(a.toFloatArray())
            let bArray = ContiguousArray(b.toFloatArray())
            return f(aArray, bArray)
        }
    }

    // MARK: - Sorted Array Helpers for Performance

    /// Helper functions for maintaining sorted ContiguousArray (replaces Set for better cache locality)
    @usableFromInline
    internal struct SortedArray<T: Comparable> {
        @usableFromInline
        var elements: ContiguousArray<T>

        @inlinable
        init(capacity: Int = 0) {
            self.elements = ContiguousArray<T>()
            if capacity > 0 {
                self.elements.reserveCapacity(capacity)
            }
        }

        @inlinable
        var count: Int { elements.count }

        @inlinable
        var isEmpty: Bool { elements.isEmpty }

        /// Binary search to find insertion point
        @inlinable
        func insertionIndex(for element: T) -> Int {
            var left = 0
            var right = elements.count

            while left < right {
                let mid = (left + right) / 2
                if elements[mid] < element {
                    left = mid + 1
                } else {
                    right = mid
                }
            }
            return left
        }

        /// Insert element maintaining sorted order
        @inlinable
        mutating func insert(_ element: T) {
            let index = insertionIndex(for: element)
            if index < elements.count && elements[index] == element {
                return  // Already exists
            }
            elements.insert(element, at: index)
        }

        /// Check if element exists (binary search)
        @inlinable
        func contains(_ element: T) -> Bool {
            let index = insertionIndex(for: element)
            return index < elements.count && elements[index] == element
        }

        /// Remove element if it exists
        @inlinable
        mutating func remove(_ element: T) {
            let index = insertionIndex(for: element)
            if index < elements.count && elements[index] == element {
                elements.remove(at: index)
            }
        }
    }

    // MARK: - Batch Distance Computations

    /// Computes distances from a single vector to multiple vectors efficiently using vDSP
    @inlinable
    public static func computeDistancesBatch(
        from source: ContiguousArray<Float>,
        to targets: [ContiguousArray<Float>],
        metric: DistanceMetric
    ) -> ContiguousArray<Float> {
        guard !targets.isEmpty else { return ContiguousArray<Float>() }

        // Ensure all vectors have the same dimension
        let dim = source.count
        for target in targets {
            precondition(target.count == dim, "All vectors must have the same dimension")
        }

        var distances = ContiguousArray<Float>(repeating: 0, count: targets.count)

        switch metric {
        case .euclidean:
            computeBatchEuclideanDistances(from: source, to: targets, result: &distances)

        case .cosine:
            computeBatchCosineDistances(from: source, to: targets, result: &distances)

        case .manhattan:
            computeBatchManhattanDistances(from: source, to: targets, result: &distances)

        case .custom(let f):
            // Fall back to sequential computation for custom metrics
            for (i, target) in targets.enumerated() {
                distances[i] = f(source, target)
            }
        }

        return distances
    }

    /// Batch computation of Euclidean distances using vDSP
    @inlinable
    static func computeBatchEuclideanDistances(
        from source: ContiguousArray<Float>,
        to targets: [ContiguousArray<Float>],
        result: inout ContiguousArray<Float>
    ) {
        let dim = source.count
        var tempDiff = ContiguousArray<Float>(repeating: 0, count: dim)

        source.withUnsafeBufferPointer { srcPtr in
            tempDiff.withUnsafeMutableBufferPointer { diffPtr in
                for (i, target) in targets.enumerated() {
                    target.withUnsafeBufferPointer { tgtPtr in
                        // Compute difference: diff = target - source
                        vDSP_vsub(srcPtr.baseAddress!, 1,
                                  tgtPtr.baseAddress!, 1,
                                  diffPtr.baseAddress!, 1,
                                  vDSP_Length(dim))

                        // Compute squared L2 norm
                        var squaredDist: Float = 0
                        vDSP_svesq(diffPtr.baseAddress!, 1,
                                   &squaredDist,
                                   vDSP_Length(dim))

                        result[i] = sqrt(squaredDist)
                    }
                }
            }
        }
    }

    /// Batch computation of Cosine distances using vDSP
    @inlinable
    static func computeBatchCosineDistances(
        from source: ContiguousArray<Float>,
        to targets: [ContiguousArray<Float>],
        result: inout ContiguousArray<Float>
    ) {
        let dim = source.count

        // Pre-compute source magnitude
        var sourceMagSq: Float = 0
        source.withUnsafeBufferPointer { srcPtr in
            vDSP_svesq(srcPtr.baseAddress!, 1, &sourceMagSq, vDSP_Length(dim))
        }
        let sourceMag = sqrt(sourceMagSq)

        // Early exit if source is zero vector
        if sourceMag < Float.ulpOfOne {
            for i in 0..<targets.count {
                result[i] = 1.0 // Max cosine distance
            }
            return
        }

        source.withUnsafeBufferPointer { srcPtr in
            for (i, target) in targets.enumerated() {
                target.withUnsafeBufferPointer { tgtPtr in
                    // Compute dot product
                    var dotProduct: Float = 0
                    vDSP_dotpr(srcPtr.baseAddress!, 1,
                               tgtPtr.baseAddress!, 1,
                               &dotProduct,
                               vDSP_Length(dim))

                    // Compute target magnitude
                    var targetMagSq: Float = 0
                    vDSP_svesq(tgtPtr.baseAddress!, 1, &targetMagSq, vDSP_Length(dim))
                    let targetMag = sqrt(targetMagSq)

                    // Compute cosine distance
                    if targetMag < Float.ulpOfOne {
                        result[i] = 1.0 // Max distance for zero vector
                    } else {
                        let similarity = dotProduct / (sourceMag * targetMag)
                        result[i] = 1.0 - max(-1.0, min(1.0, similarity))
                    }
                }
            }
        }
    }

    /// Batch computation of Manhattan distances using vDSP
    @inlinable
    static func computeBatchManhattanDistances(
        from source: ContiguousArray<Float>,
        to targets: [ContiguousArray<Float>],
        result: inout ContiguousArray<Float>
    ) {
        let dim = source.count
        var tempDiff = ContiguousArray<Float>(repeating: 0, count: dim)

        source.withUnsafeBufferPointer { srcPtr in
            tempDiff.withUnsafeMutableBufferPointer { diffPtr in
                for (i, target) in targets.enumerated() {
                    target.withUnsafeBufferPointer { tgtPtr in
                        // Compute difference: diff = target - source
                        vDSP_vsub(srcPtr.baseAddress!, 1,
                                  tgtPtr.baseAddress!, 1,
                                  diffPtr.baseAddress!, 1,
                                  vDSP_Length(dim))

                        // Compute L1 norm (sum of absolute values)
                        var l1Norm: Float = 0
                        vDSP_svemg(diffPtr.baseAddress!, 1,
                                   &l1Norm,
                                   vDSP_Length(dim))

                        result[i] = l1Norm
                    }
                }
            }
        }
    }

    /// Generic batch distance computation for GraphVector types
    public static func computeDistancesBatch<Vector: GraphVector>(
        from source: Vector,
        to targets: ContiguousArray<Vector>,
        metric: DistanceMetric
    ) -> ContiguousArray<Float> {
        // Convert source to float array once
        let sourceArray = ContiguousArray(source.toFloatArray())

        // Convert all targets to float arrays
        let targetArrays = targets.map { ContiguousArray($0.toFloatArray()) }

        // Use the optimized float array version
        return computeDistancesBatch(from: sourceArray, to: targetArrays, metric: metric)
    }

    // MARK: - k-NN Graph Construction

    public struct KNNGraphOptions: Sendable {
        public let k: Int
        public let metric: DistanceMetric
        public let approximate: Bool
        public let symmetric: Bool
        public let includeSelfLoops: Bool
        public let parallel: Bool

        public init(
            k: Int,
            metric: DistanceMetric = .euclidean,
            approximate: Bool = false,
            symmetric: Bool = true,
            includeSelfLoops: Bool = false,
            parallel: Bool = true
        ) {
            self.k = k
            self.metric = metric
            self.approximate = approximate
            self.symmetric = symmetric
            self.includeSelfLoops = includeSelfLoops
            self.parallel = parallel
        }
    }

    public static func buildKNNGraph<Vector: GraphVector & Sendable>(
        vectors: ContiguousArray<Vector>,
        options: KNNGraphOptions
    ) async -> SparseMatrix {
        let n = vectors.count
        guard n > 0 else { return try! SparseMatrix(rows: 0, cols: 0, edges: []) }

        if options.approximate && n > 10000 {
            return await buildApproximateKNNGraph(vectors: vectors, options: options)
        } else if options.parallel && n > 1000 {
            return await buildParallelKNNGraph(vectors: vectors, options: options)
        } else {
            return buildExactKNNGraph(vectors: vectors, options: options)
        }
    }

    // MARK: - Exact k-NN Graph

    private static func buildExactKNNGraph<Vector: GraphVector>(
        vectors: ContiguousArray<Vector>,
        options: KNNGraphOptions
    ) -> SparseMatrix {
        let n = vectors.count
        let maxK = options.includeSelfLoops ? n : max(0, n - 1)
        let k = min(options.k, maxK)

        if k <= 0 { return try! SparseMatrix(rows: n, cols: n, edges: []) }

        var edges = ContiguousArray<(row: UInt32, col: UInt32, value: Float?)>()
        edges.reserveCapacity(n * k * (options.symmetric ? 2 : 1))

        for i in 0..<n {
            // Use batch distance computation for better performance
            let batchDistances = computeDistancesBatch(
                from: vectors[i],
                to: vectors,
                metric: options.metric
            )

            // Create indexed distances array
            var distances = ContiguousArray<(index: Int, distance: Float)>()
            distances.reserveCapacity(n)

            for j in 0..<n {
                if i == j && !options.includeSelfLoops {
                    continue
                }
                distances.append((j, batchDistances[j]))
            }

            // Sort by distance and select k nearest
            distances.sort { $0.distance < $1.distance }

            for (j, dist) in distances.prefix(k) {
                edges.append((UInt32(i), UInt32(j), dist))

                if options.symmetric {
                    edges.append((UInt32(j), UInt32(i), dist))
                }
            }
        }

        return GraphConstructionKernels.cooToCSR(edges: edges, rows: n, cols: n)
    }

    // MARK: - Parallel k-NN Construction

    private static func buildParallelKNNGraph<Vector: GraphVector & Sendable>(
        vectors: ContiguousArray<Vector>,
        options: KNNGraphOptions
    ) async -> SparseMatrix {
        let n = vectors.count
        let maxK = options.includeSelfLoops ? n : max(0, n - 1)
        let k = min(options.k, maxK)

        if k <= 0 { return try! SparseMatrix(rows: n, cols: n, edges: []) }

        // Use 2x oversubscription for better load balancing
        let numProcessors = ProcessInfo.processInfo.activeProcessorCount
        let numTasks = numProcessors * 2
        let chunkSize = (n + numTasks - 1) / numTasks

        // Use TaskGroup for structured concurrency
        let allEdges = await withTaskGroup(
            of: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)>.self,
            returning: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)>.self
        ) { group in
            // Schedule tasks
            for taskId in 0..<numTasks {
                group.addTask {
                    let start = taskId * chunkSize
                    let end = min((taskId + 1) * chunkSize, n)
                    guard start < n else {
                        return ContiguousArray()
                    }

                    var localEdges = ContiguousArray<(row: UInt32, col: UInt32, value: Float?)>()
                    localEdges.reserveCapacity((end - start) * k * (options.symmetric ? 2 : 1))

                    for i in start..<end {
                        // Check for cancellation at chunk boundaries
                        if Task.isCancelled {
                            return localEdges
                        }

                        // Yield periodically for fairness
                        if (i - start) % 2000 == 0 {
                            await Task.yield()
                        }

                        // Use batch distance computation for this row
                        let batchDistances = computeDistancesBatch(
                            from: vectors[i],
                            to: vectors,
                            metric: options.metric
                        )

                        // Create indexed distances array
                        var distances = ContiguousArray<(index: Int, distance: Float)>()
                        distances.reserveCapacity(n)

                        for j in 0..<n {
                            if i == j && !options.includeSelfLoops {
                                continue
                            }
                            distances.append((j, batchDistances[j]))
                        }

                        // Sort and select k nearest
                        distances.sort { $0.distance < $1.distance }

                        for (j, dist) in distances.prefix(k) {
                            localEdges.append((UInt32(i), UInt32(j), dist))
                            if options.symmetric {
                                localEdges.append((UInt32(j), UInt32(i), dist))
                            }
                        }
                    }

                    return localEdges
                }
            }

            // Collect results from all tasks
            var combined = ContiguousArray<(row: UInt32, col: UInt32, value: Float?)>()
            for await edges in group {
                combined.append(contentsOf: edges)
            }
            return combined
        }

        return GraphConstructionKernels.cooToCSR(edges: allEdges, rows: n, cols: n)
    }

    // MARK: - Approximate k-NN using LSH

    private static func buildApproximateKNNGraph<Vector: GraphVector & Sendable>(
        vectors: ContiguousArray<Vector>,
        options: KNNGraphOptions
    ) async -> SparseMatrix {
        let n = vectors.count
        let k = min(options.k, options.includeSelfLoops ? n : max(0, n - 1))
        if k <= 0 { return try! SparseMatrix(rows: n, cols: n, edges: []) }

        let numTables = 8
        let numHashes = 12
        let bucketWidth: Float = 4.0

        // Build LSH index (mutable phase)
        var lshIndex = EuclideanLSHIndex(
            numTables: numTables,
            numHashes: numHashes,
            dimension: vectors[0].dimensions,
            bucketWidth: bucketWidth
        )

        for (i, vector) in vectors.enumerated() {
            lshIndex.insert(vector: vector, id: i)
        }

        // After population, index is read-only and thread-safe
        let immutableIndex = lshIndex

        // Use 2x oversubscription for better load balancing
        let numProcessors = ProcessInfo.processInfo.activeProcessorCount
        let numTasks = numProcessors * 2
        let chunkSize = (n + numTasks - 1) / numTasks

        // Use TaskGroup for structured concurrency
        let allEdges = await withTaskGroup(
            of: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)>.self,
            returning: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)>.self
        ) { group in
            // Schedule tasks
            for taskId in 0..<numTasks {
                group.addTask {
                    let start = taskId * chunkSize
                    let end = min((taskId + 1) * chunkSize, n)
                    guard start < n else {
                        return ContiguousArray()
                    }

                    var localEdges = ContiguousArray<(row: UInt32, col: UInt32, value: Float?)>()

                    for i in start..<end {
                        // Check for cancellation at chunk boundaries
                        if Task.isCancelled {
                            return localEdges
                        }

                        // Yield periodically for fairness
                        if (i - start) % 2000 == 0 {
                            await Task.yield()
                        }

                        // Query LSH index (thread-safe read-only operation)
                        let candidates = immutableIndex.query(vector: vectors[i], candidateMultiplier: 3)

                        var distances = ContiguousArray<(index: Int, distance: Float)>()
                        for j in candidates {
                            if i == j && !options.includeSelfLoops {
                                continue
                            }

                            let dist = computeDistance(vectors[i], vectors[j], metric: options.metric)
                            distances.append((j, dist))
                        }

                        distances.sort { $0.distance < $1.distance }

                        for (j, dist) in distances.prefix(k) {
                            localEdges.append((UInt32(i), UInt32(j), dist))
                            if options.symmetric {
                                localEdges.append((UInt32(j), UInt32(i), dist))
                            }
                        }
                    }

                    return localEdges
                }
            }

            // Collect results from all tasks
            var combined = ContiguousArray<(row: UInt32, col: UInt32, value: Float?)>()
            for await edges in group {
                combined.append(contentsOf: edges)
            }
            return combined
        }

        return GraphConstructionKernels.cooToCSR(edges: allEdges, rows: n, cols: n)
    }

    // MARK: - Range-based Graph Construction

    public struct RangeGraphOptions: Sendable {
        public let radius: Float
        public let adaptiveRadius: Bool
        public let minDegree: Int?
        public let maxDegree: Int?
        public let metric: DistanceMetric

        public init(
            radius: Float,
            adaptiveRadius: Bool = false,
            minDegree: Int? = nil,
            maxDegree: Int? = nil,
            metric: DistanceMetric = .euclidean
        ) {
            self.radius = radius
            self.adaptiveRadius = adaptiveRadius
            self.minDegree = minDegree
            self.maxDegree = maxDegree
            self.metric = metric
        }
    }

    public static func buildRangeGraph<Vector: GraphVector>(
        vectors: ContiguousArray<Vector>,
        options: RangeGraphOptions
    ) -> SparseMatrix {
        let n = vectors.count
        var edges = ContiguousArray<(row: UInt32, col: UInt32, value: Float?)>()

        for i in 0..<n {
            var radius = options.radius

            if options.adaptiveRadius {
                radius = estimateLocalRadius(
                    vectors: vectors,
                    index: i,
                    targetNeighbors: options.minDegree ?? 5,
                    metric: options.metric
                )
            }

            var neighbors = ContiguousArray<(index: Int, distance: Float)>()

            for j in 0..<n {
                if i == j { continue }

                let dist = computeDistance(vectors[i], vectors[j], metric: options.metric)

                if dist <= radius {
                    neighbors.append((j, dist))
                }
            }

            if let maxDeg = options.maxDegree, neighbors.count > maxDeg {
                neighbors.sort { $0.distance < $1.distance }
                neighbors = ContiguousArray(neighbors.prefix(maxDeg))
            }

            for (j, dist) in neighbors {
                edges.append((UInt32(i), UInt32(j), dist))
            }
        }

        return GraphConstructionKernels.cooToCSR(edges: edges, rows: n, cols: n)
    }

    // MARK: - Navigable Small World (NSW) Construction

    public struct NSWOptions: Sendable {
        public let M: Int
        public let efConstruction: Int
        public let metric: DistanceMetric
        public let heuristic: Bool

        public init(
            M: Int = 16,
            efConstruction: Int = 200,
            metric: DistanceMetric = .euclidean,
            heuristic: Bool = true
        ) {
            self.M = M
            self.efConstruction = efConstruction
            self.metric = metric
            self.heuristic = heuristic
        }
    }

    public struct NSWGraph {
        public let graph: SparseMatrix
        public let entryPoint: Int32
    }


    public static func buildNSWGraph<Vector: GraphVector>(
        vectors: ContiguousArray<Vector>,
        options: NSWOptions = NSWOptions()
    ) -> NSWGraph {
        let n = vectors.count
        guard n > 0 else {
            return NSWGraph(graph: try! SparseMatrix(rows: 0, cols: 0, edges: []), entryPoint: -1)
        }

        var adjacencyList = Array(repeating: SortedArray<Int32>(capacity: options.M * 2), count: n)
        var entryPoint: Int32 = 0

        for i in 1..<n {
            let newNode = Int32(i)
            let newVector = vectors[i]

            let candidates = searchNSW(
                query: newVector,
                vectors: vectors,
                adjacencyList: adjacencyList,
                entryPoint: entryPoint,
                ef: options.efConstruction,
                metric: options.metric
            )

            let neighbors = selectNeighbors(
                query: newVector,
                candidates: candidates,
                M: options.M,
                vectors: vectors,
                metric: options.metric,
                useHeuristic: options.heuristic
            )

            for neighborId in neighbors {
                adjacencyList[i].insert(neighborId)
                adjacencyList[Int(neighborId)].insert(newNode)

                if adjacencyList[Int(neighborId)].count > options.M {
                    pruneConnections(
                        nodeId: neighborId,
                        adjacencyList: &adjacencyList,
                        vectors: vectors,
                        M: options.M,
                        metric: options.metric,
                        useHeuristic: options.heuristic
                    )
                }
            }

            // CRITICAL FIX: Update entry point if new node is better
            if i > 0 {
                let distToEntry = computeDistance(
                    vectors[i],
                    vectors[Int(entryPoint)],
                    metric: options.metric
                )
                let entryToFirst = computeDistance(
                    vectors[Int(entryPoint)],
                    vectors[0],
                    metric: options.metric
                )

                if distToEntry < entryToFirst {
                    entryPoint = newNode
                }
            }
        }

        // Pre-allocate edges with accurate capacity
        var totalEdges = 0
        for adj in adjacencyList {
            totalEdges += adj.count
        }

        var edges = ContiguousArray<(row: UInt32, col: UInt32, value: Float?)>()
        edges.reserveCapacity(totalEdges)

        for (i, neighbors) in adjacencyList.enumerated() {
            for j in neighbors.elements {
                let dist = computeDistance(vectors[i], vectors[Int(j)], metric: options.metric)
                edges.append((UInt32(i), UInt32(j), dist))
            }
        }

        let graph = GraphConstructionKernels.cooToCSR(edges: edges, rows: n, cols: n)
        return NSWGraph(graph: graph, entryPoint: entryPoint)
    }

    // MARK: - Graph Generators

    public static func generateRandomGraph(
        n: Int,
        p: Float,
        directed: Bool = false,
        weighted: Bool = false
    ) -> SparseMatrix {
        // Pre-allocate edges with accurate capacity
        let totalEdges = directed ? n * (n - 1) : n * (n - 1) / 2
        var edges = ContiguousArray<(row: UInt32, col: UInt32, value: Float?)>()
        edges.reserveCapacity(totalEdges)

        for i in 0..<n {
            let jStart = directed ? 0 : i + 1
            for j in jStart..<n {
                if i != j && Float.random(in: 0...1) < p {
                    let weight = weighted ? Float.random(in: 0.1...1.0) : 1.0
                    edges.append((UInt32(i), UInt32(j), weight))

                    if !directed {
                        edges.append((UInt32(j), UInt32(i), weight))
                    }
                }
            }
        }

        return GraphConstructionKernels.cooToCSR(edges: edges, rows: n, cols: n)
    }

    public static func generateSmallWorldGraph(
        n: Int,
        k: Int,
        p: Float
    ) throws -> SparseMatrix {
        guard k % 2 == 0 && k < n && k > 0 else {
            throw GraphConstructionError.invalidParameters(
                "Watts-Strogatz model requires N > K > 0 and K must be even. Got n=\(n), k=\(k)"
            )
        }

        guard p >= 0 && p <= 1 else {
            throw GraphConstructionError.invalidParameters(
                "Rewiring probability p must be between 0 and 1. Got p=\(p)"
            )
        }

        var adj = Array(repeating: Set<Int>(), count: n)

        let halfK = k / 2
        for i in 0..<n {
            for j in 1...halfK {
                let neighbor = (i + j) % n
                adj[i].insert(neighbor)
                adj[neighbor].insert(i)
            }
        }

        for i in 0..<n {
            for j in 1...halfK {
                let originalTarget = (i + j) % n

                if Float.random(in: 0...1) < p {
                    var newTarget: Int
                    repeat {
                        newTarget = Int.random(in: 0..<n)
                    } while newTarget == i || adj[i].contains(newTarget)

                    adj[i].remove(originalTarget)
                    adj[originalTarget].remove(i)
                    adj[i].insert(newTarget)
                    adj[newTarget].insert(i)
                }
            }
        }

        // Pre-allocate edges with accurate capacity
        let totalEdges = adj.reduce(0) { $0 + $1.count }
        var edges = ContiguousArray<(row: UInt32, col: UInt32, value: Float?)>()
        edges.reserveCapacity(totalEdges)

        for (i, neighbors) in adj.enumerated() {
            for j in neighbors {
                edges.append((UInt32(i), UInt32(j), 1.0))
            }
        }

        return GraphConstructionKernels.cooToCSR(edges: edges, rows: n, cols: n)
    }

    public static func generateScaleFreeGraph(
        n: Int,
        m: Int
    ) throws -> SparseMatrix {
        guard n > m && m >= 1 else {
            throw GraphConstructionError.invalidParameters(
                "Barabási–Albert model requires N > M >= 1. Got n=\(n), m=\(m)"
            )
        }

        var adj = Array(repeating: Set<Int>(), count: n)
        var degreeList = ContiguousArray<Int>()

        let m0 = m + 1
        if n < m0 { return try! SparseMatrix(rows: n, cols: n, edges: []) }

        for i in 0..<m0 {
            for j in (i+1)..<m0 {
                adj[i].insert(j)
                adj[j].insert(i)
                degreeList.append(i)
                degreeList.append(j)
            }
        }

        for i in m0..<n {
            var targets = Set<Int>()

            while targets.count < m {
                guard let target = degreeList.randomElement() else { break }
                targets.insert(target)
            }

            for target in targets {
                adj[i].insert(target)
                adj[target].insert(i)
                degreeList.append(i)
                degreeList.append(target)
            }
        }

        // Pre-allocate edges with accurate capacity
        let totalEdges = adj.reduce(0) { $0 + $1.count }
        var edges = ContiguousArray<(row: UInt32, col: UInt32, value: Float?)>()
        edges.reserveCapacity(totalEdges)

        for (i, neighbors) in adj.enumerated() {
            for j in neighbors {
                edges.append((UInt32(i), UInt32(j), 1.0))
            }
        }

        return GraphConstructionKernels.cooToCSR(edges: edges, rows: n, cols: n)
    }

    // MARK: - Helper Functions

    private static func estimateLocalRadius<Vector: GraphVector>(
        vectors: ContiguousArray<Vector>,
        index: Int,
        targetNeighbors: Int,
        metric: DistanceMetric
    ) -> Float {
        let k = min(targetNeighbors, vectors.count - 1)
        if k <= 0 { return 0.0 }

        var distances = ContiguousArray<Float>()
        distances.reserveCapacity(vectors.count - 1)

        for j in 0..<vectors.count {
            if j != index {
                let dist = computeDistance(vectors[index], vectors[j], metric: metric)
                distances.append(dist)
            }
        }

        distances.sort()
        return distances[k - 1] * 1.2
    }

    private static func searchNSW<Vector: GraphVector>(
        query: Vector,
        vectors: ContiguousArray<Vector>,
        adjacencyList: [SortedArray<Int32>],
        entryPoint: Int32,
        ef: Int,
        metric: DistanceMetric
    ) -> [NSWSearchResult] {
        var visited = SortedArray<Int32>(capacity: ef * 2)
        var candidates = GraphPriorityQueue<NSWSearchResult>(sort: <)
        var results = GraphPriorityQueue<NSWSearchResult>(sort: >)

        let initialDist = computeDistance(query, vectors[Int(entryPoint)], metric: metric)
        let entry = NSWSearchResult(id: entryPoint, distance: initialDist)

        candidates.enqueue(entry)
        results.enqueue(entry)
        visited.insert(entryPoint)

        while !candidates.isEmpty {
            let c = candidates.dequeue()!

            if let f = results.peek() {
                if c.distance > f.distance && results.count >= ef {
                    break
                }
            }

            for neighborId in adjacencyList[Int(c.id)].elements {
                if !visited.contains(neighborId) {
                    visited.insert(neighborId)

                    let dist = computeDistance(query, vectors[Int(neighborId)], metric: metric)
                    let neighborResult = NSWSearchResult(id: neighborId, distance: dist)

                    if results.count < ef || dist < results.peek()!.distance {
                        candidates.enqueue(neighborResult)
                        results.enqueue(neighborResult)

                        if results.count > ef {
                            _ = results.dequeue()
                        }
                    }
                }
            }
        }

        return results.elements.sorted()
    }

    private static func selectNeighbors<Vector: GraphVector>(
        query: Vector,
        candidates: [NSWSearchResult],
        M: Int,
        vectors: ContiguousArray<Vector>,
        metric: DistanceMetric,
        useHeuristic: Bool
    ) -> [Int32] {
        if candidates.count <= M || !useHeuristic {
            return Array(candidates.sorted().prefix(M).map { $0.id })
        }

        var selected = [Int32]()
        var workingSet = candidates.sorted()

        while !workingSet.isEmpty && selected.count < M {
            let closest = workingSet.removeFirst()
            var isDiverse = true

            for selectedId in selected {
                let distToSelected = computeDistance(
                    vectors[Int(closest.id)],
                    vectors[Int(selectedId)],
                    metric: metric
                )

                if distToSelected < closest.distance {
                    isDiverse = false
                    break
                }
            }

            if isDiverse {
                selected.append(closest.id)
            }
        }

        return selected
    }

    private static func pruneConnections<Vector: GraphVector>(
        nodeId: Int32,
        adjacencyList: inout [SortedArray<Int32>],
        vectors: ContiguousArray<Vector>,
        M: Int,
        metric: DistanceMetric,
        useHeuristic: Bool
    ) {
        let connections = adjacencyList[Int(nodeId)].elements
        let nodeVector = vectors[Int(nodeId)]

        let connectionDistances = connections.map { connId -> NSWSearchResult in
            let dist = computeDistance(nodeVector, vectors[Int(connId)], metric: metric)
            return NSWSearchResult(id: connId, distance: dist)
        }

        let prunedConnections = selectNeighbors(
            query: nodeVector,
            candidates: connectionDistances,
            M: M,
            vectors: vectors,
            metric: metric,
            useHeuristic: useHeuristic
        )

        // Clear and rebuild the sorted array with pruned connections
        adjacencyList[Int(nodeId)] = SortedArray<Int32>(capacity: M)
        for conn in prunedConnections {
            adjacencyList[Int(nodeId)].insert(conn)
        }
    }
}

// MARK: - Helper Data Structures

private struct GraphPriorityQueue<T> {
    var elements: [T] = []
    private let sort: (T, T) -> Bool

    init(sort: @escaping (T, T) -> Bool) {
        self.sort = sort
    }

    var isEmpty: Bool { return elements.isEmpty }
    var count: Int { return elements.count }
    func peek() -> T? { return elements.first }

    mutating func enqueue(_ element: T) {
        elements.append(element)
        siftUp(elements.count - 1)
    }

    mutating func dequeue() -> T? {
        guard !isEmpty else { return nil }
        if count == 1 {
            return elements.removeLast()
        } else {
            let value = elements[0]
            elements[0] = elements.removeLast()
            siftDown(0)
            return value
        }
    }

    private mutating func siftUp(_ index: Int) {
        var child = index
        var parent = (child - 1) / 2
        while child > 0 && sort(elements[child], elements[parent]) {
            elements.swapAt(child, parent)
            child = parent
            parent = (child - 1) / 2
        }
    }

    private mutating func siftDown(_ index: Int) {
        var parent = index
        while true {
            let left = 2 * parent + 1
            let right = 2 * parent + 2
            var candidate = parent
            if left < count && sort(elements[left], elements[candidate]) {
                candidate = left
            }
            if right < count && sort(elements[right], elements[candidate]) {
                candidate = right
            }
            if candidate == parent { return }
            elements.swapAt(parent, candidate)
            parent = candidate
        }
    }
}

// MARK: - Euclidean LSH Index

private struct EuclideanLSHIndex {
    let numTables: Int
    let numHashes: Int
    let dimension: Int
    let bucketWidth: Float

    private var projections: [[(vector: ContiguousArray<Float>, shift: Float)]]
    private var tables: [Dictionary<Int, [Int]>]

    init(numTables: Int, numHashes: Int, dimension: Int, bucketWidth: Float) {
        self.numTables = numTables
        self.numHashes = numHashes
        self.dimension = dimension
        self.bucketWidth = bucketWidth
        self.tables = Array(repeating: [:], count: numTables)
        self.projections = []

        for _ in 0..<numTables {
            var tableProjections: [(vector: ContiguousArray<Float>, shift: Float)] = []
            for _ in 0..<numHashes {
                let vector = (0..<dimension).map { _ in Float.randomGaussian() }
                let shift = Float.random(in: 0...bucketWidth)
                tableProjections.append((ContiguousArray(vector), shift))
            }
            projections.append(tableProjections)
        }
    }

    private func computeHashes<Vector: GraphVector>(vector: Vector, tableIndex: Int) -> [Int] {
        let tableProjections = projections[tableIndex]
        var hashes: [Int] = []
        let vectorArray = vector.toFloatArray()

        for (projVector, shift) in tableProjections {
            var dotProduct: Float = 0
            for i in 0..<dimension {
                dotProduct += vectorArray[i] * projVector[i]
            }

            let hashValue = floor((dotProduct + shift) / bucketWidth)
            hashes.append(Int(hashValue))
        }
        return hashes
    }

    private func getBucketKey(hashes: [Int]) -> Int {
        // FNV-1a hash for deterministic results
        var hash: UInt32 = 2166136261
        for value in hashes {
            hash ^= UInt32(bitPattern: Int32(value))
            hash = hash &* 16777619
        }
        return Int(hash)
    }

    mutating func insert<Vector: GraphVector>(vector: Vector, id: Int) {
        for i in 0..<numTables {
            let hashes = computeHashes(vector: vector, tableIndex: i)
            let bucketKey = getBucketKey(hashes: hashes)

            tables[i][bucketKey, default: []].append(id)
        }
    }

    func query<Vector: GraphVector>(vector: Vector, candidateMultiplier: Int = 3) -> Set<Int> {
        var candidates = Set<Int>()

        for i in 0..<numTables {
            let hashes = computeHashes(vector: vector, tableIndex: i)
            let bucketKey = getBucketKey(hashes: hashes)

            if let bucket = tables[i][bucketKey] {
                candidates.formUnion(bucket)
            }

            // Early termination if we have enough candidates
            if candidates.count >= candidateMultiplier * 100 {
                break
            }
        }

        return candidates
    }
}

// MARK: - Helper Extensions

extension Float {
    fileprivate static func randomGaussian() -> Float {
        let u1 = Float.random(in: Float.leastNonzeroMagnitude...1)
        let u2 = Float.random(in: 0...1)
        let r = sqrt(-2.0 * log(u1))
        let theta = 2.0 * Float.pi * u2
        return r * cos(theta)
    }
}

// SIMD4 sum extension is already provided by VectorCore
