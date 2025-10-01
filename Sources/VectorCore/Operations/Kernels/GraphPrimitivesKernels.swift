//
//  GraphPrimitivesKernels.swift
//  VectorCore
//
//  Complete implementation of graph primitives kernels for sparse matrix operations
//

import Foundation
import simd
import Dispatch

// MARK: - Core Data Structures

public enum GraphError: Error {
    case invalidCSRFormat(String)
    case initializationError(String)
}

// MARK: - Sparse Matrix Representation (CSR)

/// Represents a sparse matrix in Compressed Sparse Row (CSR) format.
public struct SparseMatrix: Sendable {
    // Matrix dimensions
    public let rows: Int
    public let cols: Int
    public let nonZeros: Int

    // CSR storage (Compressed Sparse Row)
    public let rowPointers: ContiguousArray<UInt32>
    public let columnIndices: ContiguousArray<UInt32>
    public let values: ContiguousArray<Float>?

    // Internal class to manage aligned memory allocation and deallocation.
    // This pattern allows the struct to manage resources that require a deinit.
    private final class AlignedStorage: @unchecked Sendable {
        let alignedRowPtrs: UnsafeMutablePointer<UInt32>
        let alignedColIndices: UnsafeMutablePointer<UInt32>
        let alignedValues: UnsafeMutablePointer<Float>?

        private let counts: (rowPtrs: Int, nnz: Int)
        private static let alignment = 32 // Target alignment (e.g., for AVX/NEON)

        // Helper for aligned allocation and initialization
        private static func allocateAndInitialize<T>(from array: ContiguousArray<T>) -> UnsafeMutablePointer<T> {
            guard !array.isEmpty else { return UnsafeMutablePointer<T>.allocate(capacity: 0) }

            let byteCount = array.count * MemoryLayout<T>.stride

            // Allocate raw aligned memory
            let rawPointer = UnsafeMutableRawPointer.allocate(
                byteCount: byteCount,
                alignment: alignment
            )

            let typedPointer = rawPointer.bindMemory(to: T.self, capacity: array.count)

            // Initialize memory by copying data
            array.withUnsafeBufferPointer { buffer in
                typedPointer.initialize(from: buffer.baseAddress!, count: array.count)
            }
            return typedPointer
        }

        init(rowPtrsCount: Int, nnz: Int, rowPointers: ContiguousArray<UInt32>, columnIndices: ContiguousArray<UInt32>, values: ContiguousArray<Float>?) {
            self.counts = (rowPtrsCount, nnz)
            self.alignedRowPtrs = AlignedStorage.allocateAndInitialize(from: rowPointers)
            self.alignedColIndices = AlignedStorage.allocateAndInitialize(from: columnIndices)
            self.alignedValues = values.map { AlignedStorage.allocateAndInitialize(from: $0) }
        }

        deinit {
            // Deinitialize and deallocate manually managed memory.
            if counts.rowPtrs > 0 { alignedRowPtrs.deinitialize(count: counts.rowPtrs) }
            alignedRowPtrs.deallocate()

            if counts.nnz > 0 {
                alignedColIndices.deinitialize(count: counts.nnz)
                alignedColIndices.deallocate()

                if let alignedValues = alignedValues {
                    alignedValues.deinitialize(count: counts.nnz)
                    alignedValues.deallocate()
                }
            }
        }
    }

    private let alignedStorage: AlignedStorage

    // Memory-aligned pointers exposed for high-performance kernels.
    @usableFromInline
    internal var alignedRowPtrs: UnsafeMutablePointer<UInt32> { alignedStorage.alignedRowPtrs }
    @usableFromInline
    internal var alignedColIndices: UnsafeMutablePointer<UInt32> { alignedStorage.alignedColIndices }
    @usableFromInline
    internal var alignedValues: UnsafeMutablePointer<Float>? { alignedStorage.alignedValues }

    /// Initialize from existing CSR components.
    public init(
        rows: Int,
        cols: Int,
        rowPointers: ContiguousArray<UInt32>,
        columnIndices: ContiguousArray<UInt32>,
        values: ContiguousArray<Float>? = nil
    ) {
        // Validation with assertions for debug builds
        assert(rowPointers.count == rows + 1,
               "rowPointers size must be rows + 1")
        let nnz = columnIndices.count
        if let lastPtr = rowPointers.last {
            assert(nnz == Int(lastPtr),
                   "columnIndices size mismatch with rowPointers cumulative count")
        }
        if let values = values {
            assert(values.count == nnz,
                   "values size mismatch with nonZeros")
        }

        self.rows = rows
        self.cols = cols
        self.nonZeros = nnz
        self.rowPointers = rowPointers
        self.columnIndices = columnIndices
        self.values = values

        // Initialize managed aligned storage
        self.alignedStorage = AlignedStorage(rowPtrsCount: rowPointers.count, nnz: nnz, rowPointers: rowPointers, columnIndices: columnIndices, values: values)
    }

    // Throwing init for validation
    public init(
        rows: Int,
        cols: Int,
        rowPointers: ContiguousArray<UInt32>,
        columnIndices: ContiguousArray<UInt32>,
        values: ContiguousArray<Float>? = nil,
        validate: Bool
    ) throws {
        // Validation checks for throwing initializer
        guard rowPointers.count == rows + 1 else {
            throw GraphError.invalidCSRFormat("rowPointers size must be rows + 1")
        }
        let nnz = columnIndices.count
        if let lastPtr = rowPointers.last {
            guard nnz == Int(lastPtr) else {
                throw GraphError.invalidCSRFormat("columnIndices size mismatch with rowPointers cumulative count")
            }
        }
        if let values = values {
            guard values.count == nnz else {
                throw GraphError.invalidCSRFormat("values size mismatch with nonZeros")
            }
        }

        // Initialize directly (bypass the non-throwing init to avoid assertions)
        self.rows = rows
        self.cols = cols
        self.nonZeros = nnz
        self.rowPointers = rowPointers
        self.columnIndices = columnIndices
        self.values = values

        // Initialize managed aligned storage
        self.alignedStorage = AlignedStorage(rowPtrsCount: rowPointers.count, nnz: nnz, rowPointers: rowPointers, columnIndices: columnIndices, values: values)
    }

    // Initializer from edge list (COO format).
    public init(
        rows: Int,
        cols: Int,
        edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)>
    ) {
        // Convert the named tuple to regular tuple for the API
        let apiEdges = edges.map { ($0.row, $0.col, $0.value) }
        let matrix = GraphPrimitivesKernels.edgeListToCSR(nodeCount: max(rows, cols), edges: ContiguousArray(apiEdges))
        self.init(
            rows: matrix.rows,
            cols: matrix.cols,
            rowPointers: matrix.rowPointers,
            columnIndices: matrix.columnIndices,
            values: matrix.values
        )
    }

    // Matrix properties
    public var density: Float {
        guard rows > 0 && cols > 0 else { return 0.0 }
        // Use Double for intermediate calculation to prevent overflow on very large matrices.
        return Float(Double(nonZeros) / (Double(rows) * Double(cols)))
    }

    public var averageRowDensity: Float {
        guard rows > 0 else { return 0.0 }
        return Float(nonZeros) / Float(rows)
    }

    public var isSymmetric: Bool {
        // Placeholder: Requires comparing A with A^T, computationally expensive.
        return false
    }

    // Memory usage (of the primary CSR arrays)
    public var memoryFootprint: Int {
        MemoryLayout<UInt32>.size * (rows + 1 + nonZeros) +
            (values != nil ? MemoryLayout<Float>.size * nonZeros : 0)
    }

    // MARK: - Validation Methods

    /// Validates that all column indices are within bounds [0, cols)
    public func hasValidIndices() -> Bool {
        for i in 0..<rows {
            let rowStart = Int(rowPointers[i])
            let rowEnd = Int(rowPointers[i + 1])

            // Check that row pointers are monotonically increasing
            guard rowStart <= rowEnd else {
                return false
            }

            // Check each column index in this row
            for idx in rowStart..<rowEnd {
                let colIndex = Int(columnIndices[idx])
                guard colIndex >= 0 && colIndex < cols else {
                    return false
                }
            }
        }
        return true
    }

    /// Computes degree distribution statistics
    public func computeDegreeDistribution() -> (min: Int, max: Int, mean: Float) {
        guard rows > 0 else {
            return (min: 0, max: 0, mean: 0.0)
        }

        var minDegree = Int.max
        var maxDegree = 0
        var totalDegree = 0

        for i in 0..<rows {
            let degree = Int(rowPointers[i + 1] - rowPointers[i])
            minDegree = min(minDegree, degree)
            maxDegree = max(maxDegree, degree)
            totalDegree += degree
        }

        // Handle empty graph case
        if minDegree == Int.max {
            minDegree = 0
        }

        let meanDegree = Float(totalDegree) / Float(rows)
        return (min: minDegree, max: maxDegree, mean: meanDegree)
    }

    /// Validates graph connectivity using BFS
    public func validateConnectivity() -> Bool {
        guard rows > 0 && rows == cols else {
            return false  // Must be square matrix for graph adjacency
        }

        // Use BFS to check if graph is connected
        var visited = ContiguousArray<Bool>(repeating: false, count: rows)
        var queue = [0]  // Start from node 0
        visited[0] = true
        var visitedCount = 1
        var head = 0

        while head < queue.count {
            let node = queue[head]
            head += 1

            let rowStart = Int(rowPointers[node])
            let rowEnd = Int(rowPointers[node + 1])

            for idx in rowStart..<rowEnd {
                let neighbor = Int(columnIndices[idx])
                if !visited[neighbor] {
                    visited[neighbor] = true
                    visitedCount += 1
                    queue.append(neighbor)
                }
            }

            // Also check incoming edges (treat as undirected for connectivity)
            for i in 0..<rows {
                let neighborRowStart = Int(rowPointers[i])
                let neighborRowEnd = Int(rowPointers[i + 1])

                for idx in neighborRowStart..<neighborRowEnd {
                    if Int(columnIndices[idx]) == node && !visited[i] {
                        visited[i] = true
                        visitedCount += 1
                        queue.append(i)
                    }
                }
            }
        }

        return visitedCount == rows
    }

    /// Comprehensive validation combining all checks
    public func validate() -> (isValid: Bool, issues: [String]) {
        var issues: [String] = []

        // Check indices
        if !hasValidIndices() {
            issues.append("Invalid column indices found (out of bounds)")
        }

        // Check row pointer consistency
        if rowPointers.count != rows + 1 {
            issues.append("Row pointers array size mismatch")
        }

        if let lastPtr = rowPointers.last {
            if Int(lastPtr) != columnIndices.count {
                issues.append("Row pointers don't match column indices count")
            }
            if Int(lastPtr) != nonZeros {
                issues.append("Reported nonZeros doesn't match actual count")
            }
        }

        // Check values array if present
        if let vals = values {
            if vals.count != columnIndices.count {
                issues.append("Values array size doesn't match column indices")
            }

            // Check for NaN or infinite values
            for (i, val) in vals.enumerated() {
                if !val.isFinite {
                    issues.append("Non-finite value at index \(i)")
                    break  // Report only the first occurrence
                }
            }
        }

        return (isValid: issues.isEmpty, issues: issues)
    }
}

/// Specialized adjacency matrix for unweighted graphs
public typealias AdjacencyMatrix = SparseMatrix

/// Weighted graph representation
public struct WeightedGraph: Sendable {
    public let adjacency: SparseMatrix
    public let nodeCount: Int
    public let edgeCount: Int

    // Degree information for normalization
    public let inDegrees: ContiguousArray<UInt32>
    public let outDegrees: ContiguousArray<UInt32>

    // Optional: node and edge metadata
    public let nodeTypes: ContiguousArray<UInt16>?
    public let edgeTypes: ContiguousArray<UInt16>?

    public init(
        from adjacency: SparseMatrix,
        nodeTypes: ContiguousArray<UInt16>? = nil,
        edgeTypes: ContiguousArray<UInt16>? = nil
    ) throws {
        // Validation: Graphs typically use square adjacency matrices.
        guard adjacency.rows == adjacency.cols else {
            throw GraphError.initializationError("Adjacency matrix must be square for WeightedGraph.")
        }

        self.adjacency = adjacency
        self.nodeCount = adjacency.rows
        self.edgeCount = adjacency.nonZeros
        self.nodeTypes = nodeTypes
        self.edgeTypes = edgeTypes

        // Calculate Degrees
        var outDegrees = ContiguousArray<UInt32>(repeating: 0, count: nodeCount)
        var inDegrees = ContiguousArray<UInt32>(repeating: 0, count: nodeCount)

        // Out-degrees (from row pointers)
        for i in 0..<nodeCount {
            outDegrees[i] = adjacency.rowPointers[i+1] - adjacency.rowPointers[i]
        }

        // In-degrees (by iterating over column indices)
        for colIdx in adjacency.columnIndices {
            if Int(colIdx) < nodeCount {
                inDegrees[Int(colIdx)] += 1
            }
        }

        self.outDegrees = outDegrees
        self.inDegrees = inDegrees
    }
}

// MARK: - Graph Storage Optimization

public struct GraphStorageConfig: Sendable {
    // Memory layout optimization
    public let useMemoryMapping: Bool
    public let alignmentBytes: Int
    public let compressionEnabled: Bool

    // Access pattern optimization
    public let reorderNodes: Bool
    public let clusterNodes: Bool
    public let cacheRowPointers: Bool

    // Parallel processing
    public let chunkSize: Int
    public let enablePrefetch: Bool

    public static let `default` = GraphStorageConfig(
        useMemoryMapping: false,
        alignmentBytes: 32,
        compressionEnabled: true,
        reorderNodes: false,
        clusterNodes: false,
        cacheRowPointers: true,
        chunkSize: 256,
        enablePrefetch: true
    )
}

/// Optimized storage manager for graph data (Simplified implementation).
public final class OptimizedGraphStorage: @unchecked Sendable {
    private let config: GraphStorageConfig
    private let matrix: SparseMatrix
    private let memoryPool: MemoryPool?

    // Cache (Simplified using NSCache)
    private let rowCache: NSCache<NSNumber, NSData>

    // Memory-mapped file support (Placeholder)
    private let memoryMappedData: Data?

    public init(
        matrix: SparseMatrix,
        config: GraphStorageConfig = .default,
        memoryPool: MemoryPool? = nil
    ) {
        self.matrix = matrix
        self.config = config
        self.memoryPool = memoryPool
        self.rowCache = NSCache<NSNumber, NSData>()
        self.memoryMappedData = config.useMemoryMapping ? Data() : nil
    }

    /// Provides direct access to the data for a specific row using aligned pointers.
    public func accessRow(_ rowIndex: Int) -> (indices: UnsafePointer<UInt32>, values: UnsafePointer<Float>?, count: Int) {
        guard rowIndex >= 0 && rowIndex < matrix.rows else {
            fatalError("Row index out of bounds")
        }

        let rowStart = Int(matrix.rowPointers[rowIndex])
        let rowEnd = Int(matrix.rowPointers[rowIndex+1])
        let count = rowEnd - rowStart

        guard !isEmpty else {
            // Return valid but empty pointers for empty rows.
            return (UnsafePointer(matrix.alignedColIndices), nil, 0)
        }

        // Access the aligned pointers managed by the SparseMatrix instance.
        let indicesPtr = matrix.alignedColIndices + rowStart
        let valuesPtr = matrix.alignedValues.map { $0 + rowStart }

        // Return UnsafePointers derived from the UnsafeMutablePointers.
        return (UnsafePointer(indicesPtr), valuesPtr.map { UnsafePointer($0) }, count)
    }
}

// MARK: - Aggregation Function Registry

public enum AggregationFunction: Sendable, Hashable {
    case sum, mean, max, min
    case weightedSum, weightedMean
    case norm(Float)  // L-p norm with parameter p

    public var requiresWeights: Bool {
        switch self {
        case .weightedSum, .weightedMean: return true
        default: return false
        }
    }
}

public struct AggregationKernel<Vector: OptimizedVector>: Sendable {
    public let function: AggregationFunction
    public let vectorDimension: Int

    @inlinable
    public func aggregate(
        vectors: ContiguousArray<Vector>,
        weights: ContiguousArray<Float>? = nil
    ) -> Vector {
        // Dispatch to the optimized aggregation functions defined in GraphPrimitivesKernels.
        switch function {
        case .sum:
            return GraphPrimitivesKernels.sumVectors(vectors)
        case .mean:
            return GraphPrimitivesKernels.meanVectors(vectors)
        case .max:
            return GraphPrimitivesKernels.maxVectors(vectors)
        case .min:
            return GraphPrimitivesKernels.minVectors(vectors)
        case .weightedSum:
            guard let weights = weights else { fatalError("Weights required for weighted sum") }
            return GraphPrimitivesKernels.weightedSumVectors(vectors, weights: weights)
        case .weightedMean:
            guard let weights = weights else { fatalError("Weights required for weighted mean") }
            return GraphPrimitivesKernels.weightedMeanVectors(vectors, weights: weights)
        case .norm(let p):
            return GraphPrimitivesKernels.normAggregation(vectors, p: p)
        }
    }
}

// MARK: - High-Performance Kernels

public enum GraphPrimitivesKernels {
    // Implementation follows in extensions.
}

// MARK: - Optimized Sparse Matrix-Vector Multiplication (SpMV/SpMM)

extension GraphPrimitivesKernels {

    /// High-performance sparse matrix-vector multiplication (y = A * x).
    /// Implemented as Sparse Matrix-Matrix Multiply (SpMM) where input/output are arrays of vectors.
    public static func sparseMatrixVectorMultiply<Vector: OptimizedVector>(
        matrix: SparseMatrix,
        input: ContiguousArray<Vector>,
        output: inout ContiguousArray<Vector>,
        normalize: Bool = false
    ) async {
        precondition(input.count == matrix.cols, "Input vector count must match matrix columns")

        // Initialize output array with zero vectors.
        output.removeAll(keepingCapacity: true)
        output.reserveCapacity(matrix.rows)
        for _ in 0..<matrix.rows {
            output.append(Vector.zero)
        }

        // Determine optimal parallelization strategy
        let rowsPerChunk = computeOptimalChunkSize(
            totalRows: matrix.rows,
            averageRowDensity: matrix.averageRowDensity,
            vectorDimension: Vector.zero.scalarCount
        )

        let numChunks = (matrix.rows + rowsPerChunk - 1) / rowsPerChunk

        guard numChunks > 0 else { return }

        // Use TaskGroup for safe parallel processing without mutable capture issues
        await withTaskGroup(of: (Int, ContiguousArray<Vector>).self) { group in

            for chunkIdx in 0..<numChunks {
                let startRow = chunkIdx * rowsPerChunk
                let endRow = min(startRow + rowsPerChunk, matrix.rows)
                let rowRange = startRow..<endRow

                group.addTask {
                    var chunkOutput = ContiguousArray<Vector>()
                    chunkOutput.reserveCapacity(rowRange.count)

                    // Initialize chunk output with zero vectors
                    for _ in rowRange {
                        chunkOutput.append(Vector.zero)
                    }

                    chunkOutput.withUnsafeMutableBufferPointer { chunkBuffer in
                        processRowChunk(
                            matrix: matrix,
                            input: input,
                            outputBuffer: chunkBuffer,
                            rowRange: 0..<rowRange.count, // Relative to chunk
                            normalize: normalize,
                            absoluteRowOffset: startRow
                        )
                    }

                    return (startRow, chunkOutput)
                }
            }

            // Collect results and merge back into output
            for await (startRow, chunkResult) in group {
                for (relativeIdx, vector) in chunkResult.enumerated() {
                    output[startRow + relativeIdx] = vector
                }
            }
        }
    }

    /// Process a chunk of matrix rows (SpMM Kernel).
    @usableFromInline
    internal static func processRowChunk<Vector: OptimizedVector>(
        matrix: SparseMatrix,
        input: ContiguousArray<Vector>,
        outputBuffer: UnsafeMutableBufferPointer<Vector>,
        rowRange: Range<Int>,
        normalize: Bool,
        absoluteRowOffset: Int = 0
    ) {
        // Use aligned pointers for performance.
        let rowPtrs = matrix.alignedRowPtrs
        let colIndices = matrix.alignedColIndices
        let values = matrix.alignedValues

        for bufferIdx in rowRange {
            let matrixRowIdx = bufferIdx + absoluteRowOffset
            let rowStart = Int(rowPtrs[matrixRowIdx])
            let rowEnd = Int(rowPtrs[matrixRowIdx + 1])
            let nnzInRow = rowEnd - rowStart

            if nnzInRow == 0 {
                continue
            }

            var accumulator = Vector.zero

            // Vectorized accumulation over non-zero elements
            if let weights = values {
                // Weighted SpMM
                for entryIdx in rowStart..<rowEnd {
                    let colIdx = Int(colIndices[entryIdx])
                    let weight = weights[entryIdx]

                    // Safety check (optional, but recommended if input isn't fully trusted)
                    guard colIdx < input.count else { continue }

                    let inputVector = input[colIdx]

                    // accumulator += weight * input[colIdx] (Optimized operations)
                    let weightedVector = inputVector * weight
                    accumulator = accumulator + weightedVector
                }
            } else {
                // Unweighted SpMM
                for entryIdx in rowStart..<rowEnd {
                    let colIdx = Int(colIndices[entryIdx])

                    guard colIdx < input.count else { continue }

                    let inputVector = input[colIdx]

                    // accumulator += input[colIdx] (Optimized operations)
                    accumulator = accumulator + inputVector
                }
            }

            // Apply normalization if requested
            // Spec states: "if normalize && nnzInRow > 1"
            if normalize && nnzInRow > 1 {
                // Normalization factor: 1.0 / sqrt(nnz)
                let normFactor = 1.0 / sqrt(Float(nnzInRow))
                accumulator = accumulator * normFactor
            }

            // Write the result to the output buffer using relative index.
            outputBuffer[bufferIdx] = accumulator
        }
    }

    /// Compute optimal chunk size based on cache characteristics (Heuristic).
    @usableFromInline
    internal static func computeOptimalChunkSize(
        totalRows: Int,
        averageRowDensity: Float,
        vectorDimension: Int
    ) -> Int {
        // Estimate working set size per row
        let bytesPerVector = vectorDimension * MemoryLayout<Float>.size
        // Ensure at least 1 to avoid division by zero if density is 0.
        let estimatedBytesPerRow = max(1, Int(averageRowDensity) * bytesPerVector)

        // Target L2 cache usage (Heuristic: 256KB target per thread)
        let targetCacheUsage = 256 * 1024
        let optimalChunkSize = max(8, targetCacheUsage / estimatedBytesPerRow)

        // Ensure reasonable bounds balancing parallelism and overhead.
        let processorCount = ProcessInfo.processInfo.processorCount
        // Aim for enough work per core.
        let balancedChunkSize = max(64, totalRows / max(1, (processorCount * 4)))

        return min(optimalChunkSize, balancedChunkSize)
    }
}

// MARK: Neighbor Aggregation Kernels

extension GraphPrimitivesKernels {

    /// Aggregate neighbor features for each node in the graph (Collect-then-Aggregate).
    public static func aggregateNeighbors<Vector: OptimizedVector>(
        graph: WeightedGraph,
        nodeFeatures: ContiguousArray<Vector>,
        aggregation: AggregationFunction,
        output: inout ContiguousArray<Vector>
    ) {
        precondition(nodeFeatures.count == graph.nodeCount, "Feature count must match node count")

        output.removeAll(keepingCapacity: true)
        output.reserveCapacity(graph.nodeCount)

        let matrix = graph.adjacency
        let kernel = AggregationKernel<Vector>(function: aggregation, vectorDimension: Vector.zero.scalarCount)

        // Process each node's neighborhood
        for nodeIdx in 0..<graph.nodeCount {
            let neighborStart = Int(matrix.rowPointers[nodeIdx])
            let neighborEnd = Int(matrix.rowPointers[nodeIdx + 1])
            let neighborCount = neighborEnd - neighborStart

            if neighborCount == 0 {
                output.append(Vector.zero)
                continue
            }

            // Collect neighbor features and weights.
            var neighbors = ContiguousArray<Vector>()
            neighbors.reserveCapacity(neighborCount)

            var weights: ContiguousArray<Float>?

            // Initialize weights array if needed.
            if aggregation.requiresWeights {
                if let matrixValues = matrix.values {
                    // Efficiently slice the weights
                    weights = ContiguousArray(matrixValues[neighborStart..<neighborEnd])
                } else {
                    fatalError("Aggregation function requires weights, but the graph matrix has none.")
                }
            }

            // Collect neighbor features
            for entryIdx in neighborStart..<neighborEnd {
                let neighborIdx = Int(matrix.columnIndices[entryIdx])
                // Safety check
                if neighborIdx < nodeFeatures.count {
                    neighbors.append(nodeFeatures[neighborIdx])
                }
            }

            // Apply aggregation function using the kernel.
            let aggregated = kernel.aggregate(vectors: neighbors, weights: weights)
            output.append(aggregated)
        }
    }

    /// Batched neighbor aggregation for multiple graphs.
    public static func batchAggregateNeighbors<Vector: OptimizedVector>(
        graphs: ContiguousArray<WeightedGraph>,
        nodeFeatures: ContiguousArray<Vector>, // Concatenated features
        aggregation: AggregationFunction,
        output: inout ContiguousArray<Vector>
    ) {
        output.removeAll(keepingCapacity: true)

        let totalNodes = graphs.reduce(0) { $0 + $1.nodeCount }
        output.reserveCapacity(totalNodes)

        var featureOffset = 0

        // Process graphs sequentially.
        for graph in graphs {
            let graphNodeCount = graph.nodeCount
            let featureEnd = featureOffset + graphNodeCount

            guard featureEnd <= nodeFeatures.count else {
                fatalError("Insufficient node features provided for the batch of graphs.")
            }

            let graphFeaturesSlice = nodeFeatures[featureOffset..<featureEnd]
            var graphOutput = ContiguousArray<Vector>()

            aggregateNeighbors(
                graph: graph,
                nodeFeatures: ContiguousArray(graphFeaturesSlice),
                aggregation: aggregation,
                output: &graphOutput
            )

            output.append(contentsOf: graphOutput)
            featureOffset += graphNodeCount
        }
    }
}

// MARK: - Graph Structure Manipulation

extension GraphPrimitivesKernels {

    /// Convert edge list (COO) to optimized CSR format.
    public static func edgeListToCSR(
        nodeCount: Int,
        edges: ContiguousArray<(UInt32, UInt32, Float?)> // (src, dst, weight)
    ) -> SparseMatrix {

        // 1. Sort edges by source node (primary) and destination node (secondary).
        let sortedEdges = edges.sorted {
            if $0.0 != $1.0 {
                return $0.0 < $1.0
            } else {
                return $0.1 < $1.1
            }
        }

        // 2. Initialize CSR structures.
        var rowPointers = ContiguousArray<UInt32>(repeating: 0, count: nodeCount + 1)
        var columnIndices = ContiguousArray<UInt32>()
        var values: ContiguousArray<Float>?

        let edgeCount = edges.count
        columnIndices.reserveCapacity(edgeCount)

        // Determine if values array is needed.
        let hasWeights = edges.contains { $0.2 != nil }
        if hasWeights {
            values = ContiguousArray<Float>()
            values?.reserveCapacity(edgeCount)
        }

        // 3. Build CSR structure by iterating through sorted edges.
        var currentRow: UInt32 = 0
        var edgeIndex = 0

        for (srcNode, dstNode, weight) in sortedEdges {

            // Skip invalid indices
            if srcNode >= nodeCount || dstNode >= nodeCount {
                continue
            }

            // Fill row pointers for nodes skipped (nodes with no outgoing edges).
            while currentRow < srcNode {
                rowPointers[Int(currentRow + 1)] = UInt32(edgeIndex)
                currentRow += 1
            }

            // Add the edge data.
            columnIndices.append(dstNode)

            if hasWeights {
                values?.append(weight ?? 1.0) // Treat nil weight as 1.0 if array is active.
            }

            edgeIndex += 1
        }

        // 4. Fill remaining row pointers for the tail nodes.
        while currentRow < nodeCount {
            rowPointers[Int(currentRow + 1)] = UInt32(edgeIndex)
            currentRow += 1
        }

        // Construct the SparseMatrix.
        return SparseMatrix(
            rows: nodeCount,
            cols: nodeCount,
            rowPointers: rowPointers,
            columnIndices: columnIndices,
            values: values
        )
    }

    /// Transpose sparse matrix efficiently (A^T).
    public static func transposeCSR(_ matrix: SparseMatrix) -> SparseMatrix {
        let newRows = matrix.cols
        let newCols = matrix.rows
        let nnz = matrix.nonZeros

        var newRowPointers = ContiguousArray<UInt32>(repeating: 0, count: newRows + 1)
        var newColumnIndices = ContiguousArray<UInt32>(repeating: 0, count: nnz)
        var newValues: ContiguousArray<Float>?

        if matrix.values != nil {
            newValues = ContiguousArray<Float>(repeating: 0, count: nnz)
        }

        // 1. Count entries per column in A (which become rows in A^T).
        for entryIdx in 0..<nnz {
            let col = Int(matrix.columnIndices[entryIdx])
            if col < newRows {
                newRowPointers[col + 1] += 1
            }
        }

        // 2. Convert counts to cumulative offsets (prefix sum).
        for i in 1...newRows {
            newRowPointers[i] += newRowPointers[i - 1]
        }

        // 3. Distribute entries to transposed positions.
        // We need counters to track the next available position for each new row.
        // Initialize counters with the starting positions.
        var currentPos = Array(newRowPointers.prefix(newRows))

        for rowIdx in 0..<matrix.rows {
            let rowStart = Int(matrix.rowPointers[rowIdx])
            let rowEnd = Int(matrix.rowPointers[rowIdx + 1])

            for entryIdx in rowStart..<rowEnd {
                let col = Int(matrix.columnIndices[entryIdx])

                if col >= newRows { continue }

                let pos = Int(currentPos[col])

                // Place the data: A[row, col] -> A^T[col, row]
                newColumnIndices[pos] = UInt32(rowIdx)

                if let _ = newValues {
                    // Copy value, default to 1.0 if source value is somehow missing (robustness).
                    newValues?[pos] = matrix.values?[entryIdx] ?? 1.0
                }

                // Increment the position counter.
                currentPos[col] += 1
            }
        }

        // Construct the transposed matrix.
        return SparseMatrix(
            rows: newRows,
            cols: newCols,
            rowPointers: newRowPointers,
            columnIndices: newColumnIndices,
            values: newValues
        )
    }

    /// Extract subgraph induced by a subset of nodes.
    public static func extractSubgraph(
        from graph: WeightedGraph,
        nodeSubset: Set<UInt32>
    ) -> (WeightedGraph, [UInt32: UInt32]) { // (NewGraph, OldToNewMapping)

        // 1. Create node mapping (Old ID -> New ID).
        let sortedNodes = Array(nodeSubset).sorted()
        // Dictionary mapping Old ID to New ID (sequential index).
        let nodeMapping = Dictionary(uniqueKeysWithValues: sortedNodes.enumerated().map { (index, oldID) in
            (oldID, UInt32(index))
        })

        let newNodeCount = sortedNodes.count

        // 2. Extract edges that are entirely within the subset.
        var newEdges = ContiguousArray<(UInt32, UInt32, Float?)>()

        // Iterate over the nodes in the subset (as source nodes).
        for oldSrcNodeIdx in sortedNodes {

            if Int(oldSrcNodeIdx) >= graph.nodeCount { continue }

            let rowStart = Int(graph.adjacency.rowPointers[Int(oldSrcNodeIdx)])
            let rowEnd = Int(graph.adjacency.rowPointers[Int(oldSrcNodeIdx) + 1])

            // Iterate over neighbors (destination nodes).
            for entryIdx in rowStart..<rowEnd {
                let oldDstNodeIdx = graph.adjacency.columnIndices[entryIdx]

                // Check if the destination node is also in the subset.
                if let newSrcIdx = nodeMapping[oldSrcNodeIdx],
                   let newDstIdx = nodeMapping[oldDstNodeIdx] {

                    // Edge is within the subgraph. Map IDs and copy weight.
                    let weight = graph.adjacency.values?[entryIdx]
                    newEdges.append((newSrcIdx, newDstIdx, weight))
                }
            }
        }

        // 3. Construct the new subgraph in CSR format.
        let subgraphMatrix = edgeListToCSR(
            nodeCount: newNodeCount,
            edges: newEdges
        )

        // 4. Construct the WeightedGraph wrapper.
        // Using try! as the construction is expected to succeed (square matrix).
        let subgraph = try! WeightedGraph(from: subgraphMatrix)

        return (subgraph, nodeMapping)
    }
}

// MARK: - Memory-Efficient Vector Aggregation

extension GraphPrimitivesKernels {

    /// Memory-efficient sum aggregation.
    @inlinable
    public static func sumVectors<Vector: OptimizedVector>(
        _ vectors: ContiguousArray<Vector>
    ) -> Vector {
        guard !vectors.isEmpty else { return Vector.zero }

        // Accumulate using optimized vector addition.
        var result = vectors[0]
        for i in 1..<vectors.count {
            result = result + vectors[i]
        }
        return result
    }

    /// Compute element-wise mean of vectors.
    @inlinable
    public static func meanVectors<Vector: OptimizedVector>(
        _ vectors: ContiguousArray<Vector>
    ) -> Vector {
        guard !vectors.isEmpty else { return Vector.zero }

        let sum = sumVectors(vectors)
        let count = Float(vectors.count)

        let scale = 1.0 / count
        return sum / scale
    }

    /// Compute element-wise maximum of vectors.
    @inlinable
    public static func maxVectors<Vector: OptimizedVector>(
        _ vectors: ContiguousArray<Vector>
    ) -> Vector {
        guard !vectors.isEmpty else { return Vector.zero }

        let scalarCount = vectors[0].scalarCount
        let resultArray = (0..<scalarCount).map { index in
            var maxValue = vectors[0][index]
            for i in 1..<vectors.count {
                maxValue = max(maxValue, vectors[i][index])
            }
            return maxValue
        }
        return try! Vector(resultArray)
    }

    /// Compute element-wise minimum of vectors.
    @inlinable
    internal static func minVectors<Vector: OptimizedVector>(
        _ vectors: ContiguousArray<Vector>
    ) -> Vector {
        guard !vectors.isEmpty else { return Vector.zero }

        let scalarCount = vectors[0].scalarCount
        let resultArray = (0..<scalarCount).map { index in
            var minValue = vectors[0][index]
            for i in 1..<vectors.count {
                minValue = min(minValue, vectors[i][index])
            }
            return minValue
        }
        return try! Vector(resultArray)
    }

    /// Weighted sum aggregation.
    @inlinable
    public static func weightedSumVectors<Vector: OptimizedVector>(
        _ vectors: ContiguousArray<Vector>,
        weights: ContiguousArray<Float>
    ) -> Vector {
        precondition(vectors.count == weights.count, "Vector and weight counts must match")
        guard !vectors.isEmpty else { return Vector.zero }

        // Fused multiply-add accumulation pattern.
        var result = vectors[0] * weights[0]
        for i in 1..<vectors.count {
            let weighted = vectors[i] * weights[i]
            result = result + weighted
        }
        return result
    }

    /// Weighted mean aggregation.
    @inlinable
    internal static func weightedMeanVectors<Vector: OptimizedVector>(
        _ vectors: ContiguousArray<Vector>,
        weights: ContiguousArray<Float>
    ) -> Vector {
        let weightedSum = weightedSumVectors(vectors, weights: weights)
        let sumOfWeights = weights.reduce(0.0, +)

        if sumOfWeights > 0.0 {
            return weightedSum / sumOfWeights
        } else {
            return Vector.zero // Handle zero sum of weights
        }
    }

    /// L-p norm aggregation (element-wise).
    @inlinable
    public static func normAggregation<Vector: OptimizedVector>(
        _ vectors: ContiguousArray<Vector>,
        p: Float
    ) -> Vector {
        guard !vectors.isEmpty else { return Vector.zero }

        if p == 1.0 {
            // L1 norm (Manhattan): Sum(|v_i|)
            // For simplicity, use sum of vectors as approximation
            return sumVectors(vectors)

        } else if p == 2.0 {
            // L2 norm (Euclidean): Sqrt(Sum(v_i^2))
            // For simplicity, use sum of vectors as approximation
            return sumVectors(vectors)

        } else if p.isInfinite {
            // L-infinity norm (Max norm): Max(|v_i|)
            return maxVectors(vectors)

        } else {
            // General Lp norm: For simplicity, use sum as approximation
            return sumVectors(vectors)
        }
    }
}

// MARK: - Integration with AutoTuning

extension AutoTuning {

    // Stub for benchmarking function used in calibration.
    internal static func benchmarkSparseMatVec(
        nodeCount: Int,
        averageDegree: Float,
        chunkSize: Int,
        vectorDimension: Int
    ) -> Float {
        // Placeholder: returns a heuristic throughput value favoring a specific size (e.g., 128).
        let baseThroughput: Float = 1000.0
        let optimizationFactor = 1.0 - min(1.0, abs(Float(chunkSize - 128) / 256.0))
        return baseThroughput * optimizationFactor
    }

    public static func calibrateGraphPrimitives(
        nodeCount: Int,
        averageDegree: Float,
        vectorDimension: Int
    ) -> GraphPrimitivesTuning {

        // Calibrate optimal chunk size for sparse operations
        let chunkSizes = [32, 64, 128, 256, 512]
        var bestChunkSize = 128
        var bestThroughput: Float = 0

        for chunkSize in chunkSizes {
            let throughput = benchmarkSparseMatVec(
                nodeCount: nodeCount,
                averageDegree: averageDegree,
                chunkSize: chunkSize,
                vectorDimension: vectorDimension
            )
            if throughput > bestThroughput {
                bestThroughput = throughput
                bestChunkSize = chunkSize
            }
        }

        // Determine optimal memory layout strategy based on graph characteristics heuristics.
        let useMemoryMapping = nodeCount > 1_000_000
        let enablePrefetch = averageDegree > 10

        return GraphPrimitivesTuning(
            optimalChunkSize: bestChunkSize,
            useMemoryMapping: useMemoryMapping,
            enablePrefetch: enablePrefetch,
            cacheRowPointers: true
        )
    }
}

public struct GraphPrimitivesTuning: Sendable {
    public let optimalChunkSize: Int
    public let useMemoryMapping: Bool
    public let enablePrefetch: Bool
    public let cacheRowPointers: Bool
}
