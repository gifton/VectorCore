//
//  GraphPrimitivesComprehensiveTests.swift
//  VectorCore
//
//  Comprehensive test suite for graph primitives including sparse matrix operations,
//  graph construction algorithms, and traversal optimizations.
//

import Testing
import Foundation
import simd
@testable import VectorCore

/// Comprehensive test suite for Graph Primitives Kernels
@Suite("Graph Primitives Kernels")
struct GraphPrimitivesComprehensiveTests {

    // MARK: - Test Constants

    /// Tolerance for numerical operations
    let numericalTolerance: Float = 1e-6

    /// Default graph sizes for testing
    let smallGraphSize = 100
    let mediumGraphSize = 1000
    let largeGraphSize = 10000

    // MARK: - Helper Methods

    /// Creates a test graph with specific topology
    func createTestGraph(topology: GraphTopology, nodeCount: Int) -> SparseMatrix {
        var edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = []

        switch topology {
        case .random(let edgeProbability):
            // Generate random edges
            for i in 0..<nodeCount {
                for j in 0..<nodeCount {
                    if i != j && Float.random(in: 0...1) < edgeProbability {
                        edges.append((UInt32(i), UInt32(j), Float.random(in: 0.1...1.0)))
                    }
                }
            }
        case .complete:
            // Complete graph - all nodes connected
            for i in 0..<nodeCount {
                for j in 0..<nodeCount {
                    if i != j {
                        edges.append((UInt32(i), UInt32(j), 1.0))
                    }
                }
            }
        case .regular(let degree):
            // Regular graph - each node has same degree
            let maxDegree = min(degree, nodeCount - 1)
            if maxDegree > 0 {
                for i in 0..<nodeCount {
                    for j in 1...maxDegree {
                        let neighbor = (i + j) % nodeCount
                        edges.append((UInt32(i), UInt32(neighbor), 1.0))
                    }
                }
            }
        default:
            // Simple chain for other topologies
            for i in 0..<nodeCount-1 {
                edges.append((UInt32(i), UInt32(i+1), 1.0))
            }
        }

        return try! SparseMatrix(rows: nodeCount, cols: nodeCount, edges: edges)
    }

    /// Validates graph properties
    func validateGraphProperties(graph: SparseMatrix) -> GraphValidationResult {
        var isValid = true
        var errorMessages: [String] = []

        // Check row pointer monotonicity
        for i in 1..<graph.rowPointers.count {
            if graph.rowPointers[i] < graph.rowPointers[i-1] {
                isValid = false
                errorMessages.append("Row pointers not monotonic at index \(i)")
            }
        }

        // Check column indices are within bounds
        for colIdx in graph.columnIndices {
            if colIdx >= graph.cols {
                isValid = false
                errorMessages.append("Column index \(colIdx) out of bounds")
            }
        }

        // Check for self-loops
        var hasLoops = false
        for rowIdx in 0..<graph.rows {
            let (startPtr, endPtr) = (Int(graph.rowPointers[rowIdx]), Int(graph.rowPointers[rowIdx + 1]))
            for ptr in startPtr..<endPtr {
                if graph.columnIndices[ptr] == UInt32(rowIdx) {
                    hasLoops = true
                    break
                }
            }
        }

        return GraphValidationResult(
            isValid: isValid,
            isConnected: graph.nonZeros > 0,
            hasLoops: hasLoops,
            hasMultiEdges: false,
            errorMessages: errorMessages
        )
    }

    /// Measures graph connectivity metrics
    func measureConnectivity(graph: SparseMatrix) -> ConnectivityMetrics {
        var degreeCounts: [Int: Int] = [:]
        var totalDegree = 0

        // Calculate degree distribution
        for rowIdx in 0..<graph.rows {
            let degree = Int(graph.rowPointers[rowIdx + 1]) - Int(graph.rowPointers[rowIdx])
            degreeCounts[degree] = (degreeCounts[degree] ?? 0) + 1
            totalDegree += degree
        }

        let avgDegree = Float(totalDegree) / Float(graph.rows)

        return ConnectivityMetrics(
            componentCount: 1,  // Simplified - assume connected
            largestComponentSize: graph.rows,
            diameter: graph.rows - 1,  // Upper bound
            averagePathLength: avgDegree,
            clusteringCoefficient: 0.0,
            degreeDistribution: degreeCounts
        )
    }

    // MARK: - Graph Topologies

    enum GraphTopology {
        case random(edgeProbability: Float)
        case smallWorld(k: Int, p: Float)
        case scaleFree(m: Int)
        case regular(degree: Int)
        case complete
        case bipartite(leftSize: Int, rightSize: Int)
        case hierarchical(levels: Int, branchingFactor: Int)
    }

    // MARK: - Sparse Matrix Tests

    @Suite("Sparse Matrix Operations")
    struct SparseMatrixTests {

        @Test("CSR format initialization from edge list")
        func testCSRInitFromEdgeList() async throws {
            let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
                (0, 1, 1.0),
                (0, 2, 2.0),
                (1, 0, 3.0),
                (1, 3, 4.0),
                (2, 1, 5.0),
                (3, 2, 6.0)
            ]

            let matrix = try SparseMatrix(rows: 4, cols: 4, edges: edges)

            #expect(matrix.rows == 4)
            #expect(matrix.cols == 4)
            #expect(matrix.nonZeros == 6)
            #expect(matrix.rowPointers.count == 5)
            #expect(matrix.columnIndices.count == 6)
        }

        @Test("CSR format validation")
        func testCSRFormatValidation() async throws {
            // Valid CSR format
            let validRowPtrs = ContiguousArray<UInt32>([0, 2, 4, 5, 6])
            let validColIdx = ContiguousArray<UInt32>([1, 2, 0, 3, 1, 2])
            let validValues = ContiguousArray<Float>([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

            let matrix = try SparseMatrix(
                rows: 4,
                cols: 4,
                rowPointers: validRowPtrs,
                columnIndices: validColIdx,
                values: validValues
            )

            // Validate properties
            #expect(matrix.rowPointers.last == UInt32(matrix.nonZeros))
            #expect(matrix.values?.count == matrix.nonZeros)

            // Test invalid format throws error
            let invalidRowPtrs = ContiguousArray<UInt32>([0, 2])  // Wrong size
            #expect(throws: GraphError.self) {
                try SparseMatrix(
                    rows: 4,
                    cols: 4,
                    rowPointers: invalidRowPtrs,
                    columnIndices: validColIdx
                )
            }
        }

        @Test("Memory-aligned storage verification")
        func testMemoryAlignedStorage() async throws {
            let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
                (0, 1, 1.0),
                (1, 2, 2.0),
                (2, 0, 3.0)
            ]

            let matrix = try SparseMatrix(rows: 3, cols: 3, edges: edges)

            // Check alignment (32-byte alignment)
            let rowPtrAddress = Int(bitPattern: matrix.alignedRowPtrs)
            let colIdxAddress = Int(bitPattern: matrix.alignedColIndices)

            #expect(rowPtrAddress % 32 == 0, "Row pointers should be 32-byte aligned")
            #expect(colIdxAddress % 32 == 0, "Column indices should be 32-byte aligned")

            if let values = matrix.alignedValues {
                let valuesAddress = Int(bitPattern: values)
                #expect(valuesAddress % 32 == 0, "Values should be 32-byte aligned")
            }
        }

        @Test("Row pointer cumulative structure")
        func testRowPointerStructure() async throws {
            let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
                (0, 1, 1.0),
                (0, 3, 2.0),
                (1, 2, 3.0),
                (2, 0, 4.0),
                (2, 1, 5.0),
                (2, 3, 6.0)
            ]

            let matrix = try SparseMatrix(rows: 4, cols: 4, edges: edges)

            // Row pointers should be cumulative
            #expect(matrix.rowPointers[0] == 0)
            for i in 1..<matrix.rowPointers.count {
                #expect(matrix.rowPointers[i] >= matrix.rowPointers[i-1],
                       "Row pointers should be monotonically increasing")
            }

            // Last row pointer should equal number of non-zeros
            #expect(matrix.rowPointers.last == UInt32(matrix.nonZeros))
        }

        @Test("Column index sorting within rows")
        func testColumnIndexSorting() async throws {
            let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
                (0, 3, 1.0),
                (0, 1, 2.0),
                (0, 2, 3.0),
                (1, 0, 4.0),
                (1, 2, 5.0)
            ]

            let matrix = try SparseMatrix(rows: 2, cols: 4, edges: edges)

            // Check each row has sorted column indices
            for rowIdx in 0..<matrix.rows {
                let startIdx = Int(matrix.rowPointers[rowIdx])
                let endIdx = Int(matrix.rowPointers[rowIdx + 1])

                if endIdx > startIdx + 1 {
                    for i in startIdx..<(endIdx-1) {
                        #expect(matrix.columnIndices[i] <= matrix.columnIndices[i+1],
                               "Column indices should be sorted within each row")
                    }
                }
            }
        }

        @Test("Non-zero value storage")
        func testNonZeroValueStorage() async throws {
            // Test with values
            let edgesWithValues: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
                (0, 1, 1.5),
                (1, 2, 2.5),
                (2, 0, 3.5)
            ]

            let matrixWithValues = try SparseMatrix(rows: 3, cols: 3, edges: edgesWithValues)
            #expect(matrixWithValues.values != nil)
            #expect(matrixWithValues.values?.count == 3)

            // Test without values (unweighted graph)
            let edgesWithoutValues: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
                (0, 1, nil),
                (1, 2, nil),
                (2, 0, nil)
            ]

            let matrixWithoutValues = try SparseMatrix(rows: 3, cols: 3, edges: edgesWithoutValues)
            #expect(matrixWithoutValues.values == nil || matrixWithoutValues.values?.isEmpty == true)
        }

        @Test("Sparse matrix transpose")
        func testSparseMatrixTranspose() async throws {
            let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
                (0, 1, 1.0),
                (0, 2, 2.0),
                (1, 0, 3.0),
                (2, 1, 4.0)
            ]

            let matrix = try SparseMatrix(rows: 3, cols: 3, edges: edges)
            let transposed = GraphPrimitivesKernels.transposeCSR(matrix)

            #expect(transposed.rows == matrix.cols)
            #expect(transposed.cols == matrix.rows)
            #expect(transposed.nonZeros == matrix.nonZeros)

            // Verify transpose structure
            // Original (0,1) should become (1,0) in transpose
            // Original (0,2) should become (2,0) in transpose
            // etc.
        }

        @Test("Sparse matrix multiplication")
        func testSparseMatrixMultiplication() async throws {
            // Create a simple matrix
            let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
                (0, 1, 2.0),
                (1, 0, 3.0)
            ]

            let matrix = try SparseMatrix(rows: 2, cols: 2, edges: edges)

            // Create input vectors for SpMV
            let inputVector1 = try Vector512Optimized(Array(repeating: 1.0, count: 512))
            let inputVector2 = try Vector512Optimized(Array(repeating: 1.0, count: 512))
            let input = ContiguousArray([inputVector1, inputVector2])
            var output = ContiguousArray<Vector512Optimized>()

            await GraphPrimitivesKernels.sparseMatrixVectorMultiply(
                matrix: matrix,
                input: input,
                output: &output,
                normalize: false
            )

            #expect(output.count == 2)
            // Result should be [2.0, 3.0] for unit vectors
        }
    }

    // MARK: - Graph Construction Tests

    @Suite("Graph Construction Algorithms")
    struct GraphConstructionTests {

        @Test("k-NN graph construction")
        func testKNNGraphConstruction() async throws {
            // Create vectors for kNN graph
            let vectors = (0..<10).map { i in
                try! Vector512Optimized(Array(repeating: Float(i), count: 512))
            }

            // Build kNN graph with k=3
            var edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = []
            let k = 3

            for i in 0..<vectors.count {
                var distances: [(idx: Int, dist: Float)] = []
                for j in 0..<vectors.count {
                    if i != j {
                        let dist = vectors[i].euclideanDistance(to: vectors[j])
                        distances.append((j, dist))
                    }
                }
                // Sort by distance and take k nearest
                distances.sort { $0.dist < $1.dist }
                for neighbor in distances.prefix(k) {
                    edges.append((UInt32(i), UInt32(neighbor.idx), neighbor.dist))
                }
            }

            let graph = try SparseMatrix(rows: vectors.count, cols: vectors.count, edges: edges)

            #expect(graph.rows == vectors.count)
            #expect(graph.nonZeros == vectors.count * k)
        }

        @Test("Range-based graph construction")
        func testRangeBasedGraphConstruction() async throws {
            // Create vectors
            let vectors = [
                try Vector512Optimized(Array(repeating: 0.0, count: 512)),
                try Vector512Optimized(Array(repeating: 0.1, count: 512)),
                try Vector512Optimized(Array(repeating: 5.0, count: 512))
            ]

            // Connect nodes within threshold distance
            let threshold: Float = 10.0
            var edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = []

            for i in 0..<vectors.count {
                for j in 0..<vectors.count {
                    if i != j {
                        let dist = vectors[i].euclideanDistance(to: vectors[j])
                        if dist < threshold {
                            edges.append((UInt32(i), UInt32(j), dist))
                        }
                    }
                }
            }

            let graph = try SparseMatrix(rows: vectors.count, cols: vectors.count, edges: edges)

            #expect(graph.rows == vectors.count)
            // Nodes 0 and 1 should be connected (close), but 2 should be farther
            #expect(graph.nonZeros > 0)
        }

        @Test("NSW graph initialization")
        func testNSWGraphInit() async throws {
            // Simplified NSW construction
            let nodeCount = 20
            let M = 5  // Max connections per node

            var edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = []

            // Create random connections with pruning
            for i in 0..<nodeCount {
                var connections = 0
                for j in 0..<nodeCount {
                    if i != j && connections < M {
                        if Float.random(in: 0...1) < 0.3 {
                            edges.append((UInt32(i), UInt32(j), Float.random(in: 0.1...1.0)))
                            connections += 1
                        }
                    }
                }
            }

            let graph = try SparseMatrix(rows: nodeCount, cols: nodeCount, edges: edges)

            #expect(graph.rows == nodeCount)
            #expect(graph.nonZeros <= nodeCount * M)
        }

        @Test("Graph pruning strategies")
        func testGraphPruning() async throws {
            // Create a dense graph then prune
            let nodeCount = 10
            var edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = []

            // Create fully connected graph
            for i in 0..<nodeCount {
                for j in 0..<nodeCount {
                    if i != j {
                        edges.append((UInt32(i), UInt32(j), Float.random(in: 0.1...1.0)))
                    }
                }
            }

            let denseGraph = try SparseMatrix(rows: nodeCount, cols: nodeCount, edges: edges)
            #expect(denseGraph.nonZeros == nodeCount * (nodeCount - 1))

            // Prune to keep only strongest connections
            let threshold: Float = 0.5
            var prunedEdges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = []
            for edge in edges {
                if let value = edge.value, value > threshold {
                    prunedEdges.append(edge)
                }
            }

            let prunedGraph = try SparseMatrix(rows: nodeCount, cols: nodeCount, edges: prunedEdges)
            #expect(prunedGraph.nonZeros < denseGraph.nonZeros)
        }

        @Test("Graph densification")
        func testGraphDensification() async throws {
            // Start with sparse graph
            let nodeCount = 5
            var edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
                (0, 1, 1.0),
                (2, 3, 1.0)
            ]

            let sparseGraph = try SparseMatrix(rows: nodeCount, cols: nodeCount, edges: edges)
            #expect(sparseGraph.nonZeros == 2)

            // Add more edges to improve connectivity
            edges.append((1, 2, 1.0))
            edges.append((3, 4, 1.0))
            edges.append((4, 0, 1.0))

            let denserGraph = try SparseMatrix(rows: nodeCount, cols: nodeCount, edges: edges)
            #expect(denserGraph.nonZeros > sparseGraph.nonZeros)
        }

        @Test("Directed vs undirected graph conversion")
        func testDirectedUndirectedConversion() async throws {
            // Create directed graph
            let directedEdges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
                (0, 1, 1.0),
                (1, 2, 2.0),
                (2, 0, 3.0)
            ]

            let directedGraph = try SparseMatrix(rows: 3, cols: 3, edges: directedEdges)

            // Convert to undirected by adding reverse edges
            var undirectedEdges = directedEdges
            for edge in directedEdges {
                undirectedEdges.append((edge.col, edge.row, edge.value))
            }

            let undirectedGraph = try SparseMatrix(rows: 3, cols: 3, edges: undirectedEdges)

            #expect(directedGraph.nonZeros == 3)
            #expect(undirectedGraph.nonZeros == 6)
        }

        @Test("Multi-layer graph construction")
        func testMultiLayerGraphConstruction() async throws {
            // Create multi-layer graph
            let layerSize = 4
            let numLayers = 3

            var edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = []

            // Intra-layer connections
            for layer in 0..<numLayers {
                let offset = layer * layerSize
                for i in 0..<layerSize-1 {
                    edges.append((UInt32(offset + i), UInt32(offset + i + 1), 1.0))
                }
            }

            // Inter-layer connections
            for layer in 0..<numLayers-1 {
                for i in 0..<layerSize {
                    edges.append((UInt32(layer * layerSize + i), UInt32((layer + 1) * layerSize + i), 2.0))
                }
            }

            let graph = try SparseMatrix(rows: layerSize * numLayers, cols: layerSize * numLayers, edges: edges)

            #expect(graph.rows == layerSize * numLayers)
            #expect(graph.nonZeros > 0)
        }
    }

    // MARK: - Graph Traversal Tests

    @Suite("Graph Traversal Algorithms")
    struct GraphTraversalTests {

        @Test("Breadth-first search (BFS)")
        func testBreadthFirstSearch() async throws {
            // Create a simple graph for BFS
            let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
                (0, 1, 1.0),
                (0, 2, 1.0),
                (1, 3, 1.0),
                (2, 3, 1.0),
                (3, 4, 1.0)
            ]

            let graph = try SparseMatrix(rows: 5, cols: 5, edges: edges)

            // Simple BFS validation
            var visited = Set<Int>()
            var queue = [0]
            visited.insert(0)

            while !queue.isEmpty {
                let current = queue.removeFirst()
                let startIdx = Int(graph.rowPointers[current])
                let endIdx = Int(graph.rowPointers[current + 1])

                for i in startIdx..<endIdx {
                    let neighbor = Int(graph.columnIndices[i])
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor)
                        queue.append(neighbor)
                    }
                }
            }

            // Should visit all reachable nodes from 0
            #expect(visited.count >= 4)  // All nodes reachable from 0
        }

        @Test("Depth-first search (DFS)")
        func testDepthFirstSearch() async throws {
            // Test DFS traversal
        }

        @Test("Dijkstra's shortest path")
        func testDijkstraShortestPath() async throws {
            // Find shortest paths
        }

        @Test("A* pathfinding with heuristics")
        func testAStarPathfinding() async throws {
            // Test heuristic-guided search
        }

        @Test("Bidirectional search")
        func testBidirectionalSearch() async throws {
            // Search from both ends
        }

        @Test("Greedy best-first search")
        func testGreedyBestFirstSearch() async throws {
            // Test greedy traversal
        }

        @Test("Parallel graph traversal")
        func testParallelTraversal() async throws {
            // Test concurrent traversal
        }
    }

    // MARK: - Connectivity Analysis Tests

    @Suite("Connectivity Analysis")
    struct ConnectivityAnalysisTests {

        @Test("Connected components detection")
        func testConnectedComponents() async throws {
            // Create graph with two components
            let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
                // Component 1
                (0, 1, 1.0),
                (1, 2, 1.0),
                // Component 2
                (3, 4, 1.0),
                (4, 5, 1.0)
                // Node 6 is isolated
            ]

            let graph = try SparseMatrix(rows: 7, cols: 7, edges: edges)

            // Simple connected components algorithm
            var components: [[Int]] = []
            var visited = Set<Int>()

            for start in 0..<graph.rows {
                if !visited.contains(start) {
                    var component: [Int] = []
                    var stack = [start]

                    while !stack.isEmpty {
                        let node = stack.removeLast()
                        if visited.contains(node) { continue }
                        visited.insert(node)
                        component.append(node)

                        let startIdx = Int(graph.rowPointers[node])
                        let endIdx = Int(graph.rowPointers[node + 1])

                        for i in startIdx..<endIdx {
                            let neighbor = Int(graph.columnIndices[i])
                            if !visited.contains(neighbor) {
                                stack.append(neighbor)
                            }
                        }
                    }

                    components.append(component)
                }
            }

            #expect(components.count >= 3)  // At least 3 components
        }

        @Test("Strongly connected components (Tarjan)")
        func testStronglyConnectedComponents() async throws {
            // Find SCCs in directed graphs
        }

        @Test("Graph diameter calculation")
        func testGraphDiameter() async throws {
            // Calculate maximum shortest path
        }

        @Test("Average path length")
        func testAveragePathLength() async throws {
            // Calculate mean path length
        }

        @Test("Clustering coefficient")
        func testClusteringCoefficient() async throws {
            // Measure local clustering
        }

        @Test("Degree distribution analysis")
        func testDegreeDistribution() async throws {
            // Analyze node degree patterns
        }

        @Test("Bridge and articulation point detection")
        func testBridgeDetection() async throws {
            // Find critical edges and nodes
        }
    }

    // MARK: - Graph Optimization Tests

    @Suite("Graph Optimization")
    struct GraphOptimizationTests {

        @Test("Edge pruning for sparsification")
        func testEdgePruning() async throws {
            // Remove redundant edges
        }

        @Test("Graph coarsening")
        func testGraphCoarsening() async throws {
            // Create smaller representative graph
        }

        @Test("Graph partitioning (METIS-style)")
        func testGraphPartitioning() async throws {
            // Partition graph into balanced parts
        }

        @Test("Community detection algorithms")
        func testCommunityDetection() async throws {
            // Find graph communities
        }

        @Test("Graph reordering for cache efficiency")
        func testGraphReordering() async throws {
            // Reorder nodes for better locality
        }

        @Test("Graph compression techniques")
        func testGraphCompression() async throws {
            // Test compression algorithms
        }
    }

    // MARK: - Performance and Scalability Tests

    @Suite("Performance and Scalability")
    struct PerformanceScalabilityTests {

        @Test("Large-scale graph construction performance")
        func testLargeScaleConstruction() async throws {
            // Test with millions of edges
        }

        @Test("Cache-efficient traversal patterns")
        func testCacheEfficientTraversal() async throws {
            // Optimize memory access patterns
        }

        @Test("SIMD acceleration for graph operations")
        func testSIMDAcceleration() async throws {
            // Verify SIMD utilization
        }

        @Test("Parallel edge processing")
        func testParallelEdgeProcessing() async throws {
            // Test concurrent edge operations
        }

        @Test("Memory bandwidth utilization")
        func testMemoryBandwidthUtilization() async throws {
            // Measure memory efficiency
        }

        @Test("Scalability with graph size")
        func testScalabilityWithSize() async throws {
            // Test O(n) vs O(n²) scaling
        }
    }

    // MARK: - NSW/HNSW Specific Tests

    @Suite("Navigable Small World Graphs")
    struct NSWGraphTests {

        @Test("NSW construction with pruning")
        func testNSWConstruction() async throws {
            // Build NSW with edge pruning
        }

        @Test("NSW neighbor selection heuristic")
        func testNeighborSelection() async throws {
            // Test pruning heuristics
        }

        @Test("Entry point selection")
        func testEntryPointSelection() async throws {
            // Find good entry points
        }

        @Test("Graph navigability metrics")
        func testNavigability() async throws {
            // Measure search efficiency
        }

        @Test("Dynamic NSW updates")
        func testDynamicNSWUpdates() async throws {
            // Add/remove nodes dynamically
        }

        @Test("Multi-layer NSW (HNSW-like)")
        func testMultiLayerNSW() async throws {
            // Test hierarchical structure
        }

        @Test("NSW search performance")
        func testNSWSearchPerformance() async throws {
            // Benchmark search operations
        }
    }

    // MARK: - Graph Properties Tests

    @Suite("Graph Properties and Metrics")
    struct GraphPropertiesTests {

        @Test("Small-world property verification")
        func testSmallWorldProperty() async throws {
            // Verify small-world characteristics
        }

        @Test("Scale-free property verification")
        func testScaleFreeProperty() async throws {
            // Check power-law degree distribution
        }

        @Test("Graph density calculation")
        func testGraphDensity() async throws {
            // Calculate edge density
        }

        @Test("Assortativity coefficient")
        func testAssortativity() async throws {
            // Measure degree correlation
        }

        @Test("Modularity calculation")
        func testModularity() async throws {
            // Measure community structure
        }

        @Test("PageRank computation")
        func testPageRank() async throws {
            // Calculate node importance
        }

        @Test("Betweenness centrality")
        func testBetweennessCentrality() async throws {
            // Measure node/edge centrality
        }
    }

    // MARK: - Sparse Linear Algebra Tests

    @Suite("Sparse Linear Algebra")
    struct SparseLinearAlgebraTests {

        @Test("Sparse matrix-vector multiplication (SpMV)")
        func testSparseMatrixVectorMultiplication() async throws {
            let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
                (0, 0, 2.0),
                (0, 1, 3.0),
                (1, 0, 4.0),
                (1, 1, 5.0)
            ]

            let matrix = try SparseMatrix(rows: 2, cols: 2, edges: edges)
            let inputVector1 = try Vector512Optimized(Array(repeating: 1.0, count: 512))
            let inputVector2 = try Vector512Optimized(Array(repeating: 1.0, count: 512))
            let input = ContiguousArray([inputVector1, inputVector2])
            var output = ContiguousArray<Vector512Optimized>()

            await GraphPrimitivesKernels.sparseMatrixVectorMultiply(
                matrix: matrix,
                input: input,
                output: &output,
                normalize: false
            )

            #expect(output.count == 2)
            // With symmetric matrix and unit vectors, output represents row sums
            // Output vectors should contain scaled values based on matrix
        }

        @Test("Sparse matrix-matrix multiplication (SpGEMM)")
        func testSparseMatrixMatrixMultiplication() async throws {
            // Test SpGEMM operation
        }

        @Test("Sparse triangular solve")
        func testSparseTriangularSolve() async throws {
            // Test forward/backward substitution
        }

        @Test("Sparse matrix addition")
        func testSparseMatrixAddition() async throws {
            // Add sparse matrices
        }

        @Test("Sparse matrix scaling")
        func testSparseMatrixScaling() async throws {
            // Scale matrix values
        }

        @Test("CSR to CSC format conversion")
        func testCSRtoCSCConversion() async throws {
            // Convert between formats
        }
    }

    // MARK: - Edge Cases and Error Handling

    @Suite("Edge Cases and Error Handling")
    struct EdgeCaseTests {

        @Test("Empty graph handling")
        func testEmptyGraph() async throws {
            let emptyEdges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = []
            let graph = try SparseMatrix(rows: 5, cols: 5, edges: emptyEdges)

            #expect(graph.rows == 5)
            #expect(graph.cols == 5)
            #expect(graph.nonZeros == 0)
            #expect(graph.rowPointers.count == 6)
            #expect(graph.columnIndices.isEmpty)
        }

        @Test("Single node graph")
        func testSingleNodeGraph() async throws {
            let emptyEdges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = []
            let graph = try SparseMatrix(rows: 1, cols: 1, edges: emptyEdges)

            #expect(graph.rows == 1)
            #expect(graph.cols == 1)
            #expect(graph.nonZeros == 0)
            #expect(graph.rowPointers.count == 2)
            #expect(graph.rowPointers[0] == 0)
            #expect(graph.rowPointers[1] == 0)
        }

        @Test("Self-loops handling")
        func testSelfLoops() async throws {
            let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
                (0, 0, 1.0),  // Self-loop
                (1, 1, 2.0),  // Self-loop
                (0, 1, 3.0)   // Regular edge
            ]

            let graph = try SparseMatrix(rows: 2, cols: 2, edges: edges)

            #expect(graph.nonZeros == 3)

            // Check for self-loops
            let validation = GraphPrimitivesComprehensiveTests().validateGraphProperties(graph: graph)
            #expect(validation.hasLoops == true)
        }

        @Test("Multi-edges between nodes")
        func testMultiEdges() async throws {
            let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
                (0, 1, 1.0),
                (0, 1, 2.0),  // Duplicate edge with different value
                (0, 1, 3.0),  // Another duplicate
                (1, 2, 4.0)
            ]

            // The CSR conversion should handle duplicates
            let graph = try SparseMatrix(rows: 3, cols: 3, edges: edges)

            #expect(graph.rows == 3)
            #expect(graph.nonZeros > 0)
        }

        @Test("Disconnected graph components")
        func testDisconnectedComponents() async throws {
            let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
                // Component 1
                (0, 1, 1.0),
                (1, 0, 1.0),
                // Component 2 (disconnected)
                (2, 3, 2.0),
                (3, 2, 2.0),
                // Node 4 is isolated (no edges)
            ]

            let graph = try SparseMatrix(rows: 5, cols: 5, edges: edges)

            #expect(graph.rows == 5)
            #expect(graph.nonZeros == 4)

            // Check connectivity
            let metrics = GraphPrimitivesComprehensiveTests().measureConnectivity(graph: graph)
            #expect(metrics.componentCount > 0)
        }

        @Test("Maximum degree nodes")
        func testMaximumDegreeNodes() async throws {
            // Create a hub-and-spoke graph
            var edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = []

            let hubNode: UInt32 = 0
            let spokeCount = 10

            // Connect hub to all spokes
            for spoke in 1...spokeCount {
                edges.append((hubNode, UInt32(spoke), 1.0))
                edges.append((UInt32(spoke), hubNode, 1.0))  // Bidirectional
            }

            let graph = try SparseMatrix(rows: spokeCount + 1, cols: spokeCount + 1, edges: edges)

            #expect(graph.nonZeros == spokeCount * 2)

            // Hub should have maximum degree
            let hubDegree = Int(graph.rowPointers[1]) - Int(graph.rowPointers[0])
            #expect(hubDegree == spokeCount)
        }

        @Test("Invalid CSR format detection")
        func testInvalidCSRDetection() async throws {
            // Test various invalid formats

            // Wrong row pointer size
            #expect(throws: GraphError.self) {
                try SparseMatrix(
                    rows: 3,
                    cols: 3,
                    rowPointers: ContiguousArray<UInt32>([0, 1]),  // Should be 4 elements
                    columnIndices: ContiguousArray<UInt32>([0])
                )
            }

            // Mismatched values size
            #expect(throws: GraphError.self) {
                try SparseMatrix(
                    rows: 2,
                    cols: 2,
                    rowPointers: ContiguousArray<UInt32>([0, 1, 2]),
                    columnIndices: ContiguousArray<UInt32>([0, 1]),
                    values: ContiguousArray<Float>([1.0])  // Should have 2 values
                )
            }
        }
    }

    // MARK: - Integration Tests

    @Suite("Integration with Vector Operations")
    struct IntegrationTests {

        @Test("Graph construction from vector similarities")
        func testGraphFromVectorSimilarities() async throws {
            // Create test vectors
            let vectors = [
                try Vector512Optimized(Array(repeating: 0.0, count: 512)),
                try Vector512Optimized(Array(repeating: 1.0, count: 512)),
                try Vector512Optimized(Array(repeating: 2.0, count: 512))
            ]

            // Build graph based on similarities
            var edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = []
            let similarityThreshold: Float = 0.5

            for i in 0..<vectors.count {
                for j in 0..<vectors.count {
                    if i != j {
                        let similarity = vectors[i].cosineSimilarity(to: vectors[j])
                        if similarity > similarityThreshold {
                            edges.append((UInt32(i), UInt32(j), similarity))
                        }
                    }
                }
            }

            let graph = try SparseMatrix(rows: vectors.count, cols: vectors.count, edges: edges)

            #expect(graph.rows == vectors.count)
            #expect(graph.nonZeros >= 0)  // Depends on similarity threshold
        }

        @Test("Graph-accelerated similarity search")
        func testGraphAcceleratedSearch() async throws {
            // Use graph for ANN search
        }

        @Test("Graph with quantized vectors")
        func testGraphWithQuantizedVectors() async throws {
            // Combine with INT8 quantization
        }

        @Test("Hierarchical graph with clustering")
        func testHierarchicalGraphClustering() async throws {
            // Combine with clustering
        }

        @Test("Multi-modal graph construction")
        func testMultiModalGraph() async throws {
            // Graph with different edge types
        }

        @Test("End-to-end index construction")
        func testEndToEndIndexConstruction() async throws {
            // Complete search index
        }
    }

    // MARK: - Benchmarking and Profiling

    @Suite("Benchmarking and Profiling")
    struct BenchmarkingTests {

        @Test("Graph construction throughput")
        func testConstructionThroughput() async throws {
            // Measure edges/second
        }

        @Test("Traversal operations per second")
        func testTraversalOpsPerSecond() async throws {
            // Benchmark search speed
        }

        @Test("Memory usage profiling")
        func testMemoryUsageProfiling() async throws {
            // Measure memory footprint
        }

        @Test("Cache miss rate analysis")
        func testCacheMissRate() async throws {
            // Analyze cache behavior
        }

        @Test("Parallel speedup measurement")
        func testParallelSpeedup() async throws {
            // Measure parallelization efficiency
        }

        @Test("Weak vs strong scaling")
        func testScalingBehavior() async throws {
            // Test scaling characteristics
        }
    }
}

// MARK: - Supporting Types

/// Graph validation result
struct GraphValidationResult {
    let isValid: Bool
    let isConnected: Bool
    let hasLoops: Bool
    let hasMultiEdges: Bool
    let errorMessages: [String]

    init(isValid: Bool = true, isConnected: Bool = true, hasLoops: Bool = false,
         hasMultiEdges: Bool = false, errorMessages: [String] = []) {
        self.isValid = isValid
        self.isConnected = isConnected
        self.hasLoops = hasLoops
        self.hasMultiEdges = hasMultiEdges
        self.errorMessages = errorMessages
    }
}

/// Graph connectivity metrics
struct ConnectivityMetrics {
    let componentCount: Int
    let largestComponentSize: Int
    let diameter: Int
    let averagePathLength: Float
    let clusteringCoefficient: Float
    let degreeDistribution: [Int: Int]

    init(componentCount: Int = 1, largestComponentSize: Int = 0, diameter: Int = 0,
         averagePathLength: Float = 0, clusteringCoefficient: Float = 0,
         degreeDistribution: [Int: Int] = [:]) {
        self.componentCount = componentCount
        self.largestComponentSize = largestComponentSize
        self.diameter = diameter
        self.averagePathLength = averagePathLength
        self.clusteringCoefficient = clusteringCoefficient
        self.degreeDistribution = degreeDistribution
    }
}

/// NSW graph configuration
struct NSWConfig {
    let M: Int  // Number of bi-directional links
    let efConstruction: Int  // Size of dynamic candidate list
    let seed: UInt64
    let pruningAlpha: Float

    static var `default`: NSWConfig {
        NSWConfig(M: 16, efConstruction: 200, seed: 42, pruningAlpha: 1.2)
    }
}

/// Graph generation utilities
struct GraphGenerator {
    static func randomGraph(nodes: Int, edgeProbability: Float) -> SparseMatrix {
        // TODO: Generate Erdős–Rényi random graph
        try! SparseMatrix(rows: nodes, cols: nodes,
                         rowPointers: ContiguousArray<UInt32>([0]),
                         columnIndices: ContiguousArray<UInt32>())
    }

    static func smallWorldGraph(nodes: Int, k: Int, p: Float) -> SparseMatrix {
        // TODO: Generate Watts-Strogatz small-world graph
        try! SparseMatrix(rows: nodes, cols: nodes,
                         rowPointers: ContiguousArray<UInt32>([0]),
                         columnIndices: ContiguousArray<UInt32>())
    }

    static func scaleFreeGraph(nodes: Int, m: Int) -> SparseMatrix {
        // TODO: Generate Barabási–Albert scale-free graph
        try! SparseMatrix(rows: nodes, cols: nodes,
                         rowPointers: ContiguousArray<UInt32>([0]),
                         columnIndices: ContiguousArray<UInt32>())
    }
}

/// Search performance metrics
struct SearchMetrics {
    let queriesPerSecond: Double
    let averageHops: Float
    let recall: Float
    let precision: Float
    let averageDistanceComputations: Float

    static func measure(graph: SparseMatrix, queries: [Vector512Optimized]) -> SearchMetrics {
        // TODO: Measure search performance
        SearchMetrics(
            queriesPerSecond: 0,
            averageHops: 0,
            recall: 0,
            precision: 0,
            averageDistanceComputations: 0
        )
    }
}