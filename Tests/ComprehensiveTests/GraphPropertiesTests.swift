//
//  GraphPropertiesTests.swift
//  VectorCore
//
//  Tests for Graph Properties and Algorithms Kernels
//

import XCTest
@testable import VectorCore

final class GraphPropertiesTests: XCTestCase {

    // MARK: - Test Helpers

    func createSimpleGraph() -> SparseMatrix {
        // Create a simple 5-node graph
        // 0 -> 1, 2
        // 1 -> 2, 3
        // 2 -> 3, 4
        // 3 -> 4
        // 4 -> 0
        let rows = 5
        let cols = 5
        let rowPointers = ContiguousArray<UInt32>([0, 2, 4, 6, 7, 8])
        let columnIndices = ContiguousArray<UInt32>([1, 2, 2, 3, 3, 4, 4, 0])
        let values: ContiguousArray<Float>? = nil

        return SparseMatrix(
            rows: rows,
            cols: cols,
            rowPointers: rowPointers,
            columnIndices: columnIndices,
            values: values
        )
    }

    func createCompleteGraph(n: Int) -> SparseMatrix {
        // Create a complete graph (all nodes connected)
        var rowPointers = ContiguousArray<UInt32>(repeating: 0, count: n + 1)
        var columnIndices = ContiguousArray<UInt32>()

        for i in 0..<n {
            rowPointers[i] = UInt32(columnIndices.count)
            for j in 0..<n {
                if i != j {
                    columnIndices.append(UInt32(j))
                }
            }
        }
        rowPointers[n] = UInt32(columnIndices.count)

        return SparseMatrix(
            rows: n,
            cols: n,
            rowPointers: rowPointers,
            columnIndices: columnIndices,
            values: nil
        )
    }

    func createBipartiteGraph() -> SparseMatrix {
        // Create a bipartite graph K(3,3)
        // Nodes 0,1,2 connect to nodes 3,4,5
        let rows = 6
        let cols = 6
        let rowPointers = ContiguousArray<UInt32>([0, 3, 6, 9, 12, 15, 18])
        let columnIndices = ContiguousArray<UInt32>([
            3, 4, 5,  // Node 0 connects to 3,4,5
            3, 4, 5,  // Node 1 connects to 3,4,5
            3, 4, 5,  // Node 2 connects to 3,4,5
            0, 1, 2,  // Node 3 connects to 0,1,2
            0, 1, 2,  // Node 4 connects to 0,1,2
            0, 1, 2   // Node 5 connects to 0,1,2
        ])
        let values: ContiguousArray<Float>? = nil

        return SparseMatrix(
            rows: rows,
            cols: cols,
            rowPointers: rowPointers,
            columnIndices: columnIndices,
            values: values
        )
    }

    // NOTE: BFS, DFS, and Connected Components tests removed as these are already
    // tested in GraphTraversalTests.swift with the existing implementations

    // MARK: - Graph Properties Tests

    func testGraphPropertiesBasic() async {
        let graph = createSimpleGraph()
        let properties = await GraphPrimitivesKernels.computeGraphProperties(
            matrix: graph,
            options: GraphPrimitivesKernels.GraphPropertiesOptions(directed: true)
        )

        XCTAssertGreaterThan(properties.diameter, 0, "Diameter should be positive")
        XCTAssertGreaterThan(properties.radius, 0, "Radius should be positive")
        XCTAssertGreaterThan(properties.averagePathLength, 0, "Average path length should be positive")
        XCTAssertGreaterThan(properties.density, 0, "Density should be positive")
        XCTAssertTrue(properties.isCyclic, "Graph should have cycles")
    }

    func testDegreeDistribution() async {
        let graph = createCompleteGraph(n: 5)
        let properties = await GraphPrimitivesKernels.computeGraphProperties(
            matrix: graph,
            options: GraphPrimitivesKernels.GraphPropertiesOptions(directed: false)
        )

        let degDist = properties.degreeDistribution
        XCTAssertEqual(degDist.degrees.count, 5)
        XCTAssertEqual(degDist.max, 4, "In K5, each node has degree 4")
        XCTAssertEqual(degDist.min, 4, "In K5, each node has degree 4")
        XCTAssertEqual(degDist.mean, 4.0, accuracy: 0.01, "Mean degree should be 4")
    }

    func testBipartiteness() async {
        let bipartiteGraph = createBipartiteGraph()
        let properties = await GraphPrimitivesKernels.computeGraphProperties(
            matrix: bipartiteGraph,
            options: GraphPrimitivesKernels.GraphPropertiesOptions(directed: false)
        )

        XCTAssertTrue(properties.isBipartite, "K(3,3) should be bipartite")

        // Test non-bipartite graph (triangle)
        let triangleGraph = createCompleteGraph(n: 3)
        let triangleProperties = await GraphPrimitivesKernels.computeGraphProperties(
            matrix: triangleGraph,
            options: GraphPrimitivesKernels.GraphPropertiesOptions(directed: false)
        )
        XCTAssertFalse(triangleProperties.isBipartite, "Triangle should not be bipartite")
    }

    func testAssortativity() async {
        let graph = createCompleteGraph(n: 5)
        let properties = await GraphPrimitivesKernels.computeGraphProperties(
            matrix: graph,
            options: GraphPrimitivesKernels.GraphPropertiesOptions(directed: false)
        )

        // Complete graph has neutral assortativity (all nodes have same degree)
        XCTAssertTrue(abs(properties.assortativity - 1.0) < 0.1 || abs(properties.assortativity) < 0.1,
                      "Complete graph should have neutral or perfect assortativity")
    }

    // MARK: - K-Core Decomposition Tests

    func testKCoreDecomposition() {
        let graph = createSimpleGraph()
        let kcore = GraphPrimitivesKernels.kCoreDecomposition(matrix: graph)

        XCTAssertEqual(kcore.coreNumbers.count, 5)
        XCTAssertGreaterThanOrEqual(kcore.maxCore, 0, "Max core should be non-negative")

        // Verify core distribution
        var totalNodes = 0
        for (_, count) in kcore.coreDistribution {
            totalNodes += count
        }
        XCTAssertEqual(totalNodes, 5, "Core distribution should account for all nodes")
    }

    func testKCoreCompleteGraph() {
        let graph = createCompleteGraph(n: 6)
        let kcore = GraphPrimitivesKernels.kCoreDecomposition(matrix: graph)

        // In a complete graph K6, all nodes should be in the 5-core
        XCTAssertEqual(kcore.maxCore, 5, "K6 should have max core of 5")
        for coreNum in kcore.coreNumbers {
            XCTAssertEqual(coreNum, 5, "All nodes in K6 should be in 5-core")
        }
    }

    // MARK: - Motif Detection Tests

    func testTriangleCounting() async {
        // Create a graph with known triangles
        let rows = 4
        let cols = 4
        // Complete graph K4 has 4 triangles
        let rowPointers = ContiguousArray<UInt32>([0, 3, 5, 7, 9])
        let columnIndices = ContiguousArray<UInt32>([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2])
        let values: ContiguousArray<Float>? = nil

        let graph = SparseMatrix(
            rows: rows,
            cols: cols,
            rowPointers: rowPointers,
            columnIndices: columnIndices,
            values: values
        )

        let motifs = await GraphPrimitivesKernels.detectMotifs(
            matrix: graph,
            options: GraphPrimitivesKernels.MotifDetectionOptions(motifSizes: [3])
        )

        if let triangleCount = motifs.motifCounts[.triangle] {
            XCTAssertGreaterThan(triangleCount, 0, "K4 should have triangles")
        } else {
            XCTFail("Triangle count not found")
        }
    }

    func test4NodeMotifs() async {
        let graph = createCompleteGraph(n: 5)
        let motifs = await GraphPrimitivesKernels.detectMotifs(
            matrix: graph,
            options: GraphPrimitivesKernels.MotifDetectionOptions(motifSizes: [4])
        )

        // K5 should have many 4-node motifs
        if let squareCount = motifs.motifCounts[.square] {
            XCTAssertGreaterThan(squareCount, 0, "K5 should have squares")
        }

        if let starCount = motifs.motifCounts[.star(3)] {
            XCTAssertGreaterThan(starCount, 0, "K5 should have 3-stars")
        }
    }

    // MARK: - Graph Coloring Tests

    func testGreedyColoring() {
        let graph = createSimpleGraph()
        let result = GraphPrimitivesKernels.graphColoring(matrix: graph, algorithm: .greedy)

        XCTAssertEqual(result.colors.count, 5)
        XCTAssertGreaterThan(result.chromaticNumber, 0, "Should use at least one color")
        XCTAssertFalse(result.isOptimal, "Greedy coloring is not guaranteed optimal")

        // Verify proper coloring (no adjacent nodes have same color)
        for i in 0..<graph.rows {
            let rowStart = Int(graph.rowPointers[i])
            let rowEnd = Int(graph.rowPointers[i + 1])

            for idx in rowStart..<rowEnd {
                let j = Int(graph.columnIndices[idx])
                XCTAssertNotEqual(result.colors[i], result.colors[j],
                                  "Adjacent nodes should have different colors")
            }
        }
    }

    func testWelshPowellColoring() {
        let graph = createCompleteGraph(n: 4)
        let result = GraphPrimitivesKernels.graphColoring(matrix: graph, algorithm: .welshPowell)

        // K4 needs exactly 4 colors
        XCTAssertEqual(result.chromaticNumber, 4, "K4 should need 4 colors")
    }

    func testDSATURColoring() {
        let graph = createBipartiteGraph()
        let result = GraphPrimitivesKernels.graphColoring(matrix: graph, algorithm: .dsatur)

        // Bipartite graph needs at most 2 colors
        XCTAssertLessThanOrEqual(result.chromaticNumber, 2, "Bipartite graph needs at most 2 colors")
    }

    // MARK: - Performance Tests

    func testGraphPropertiesPerformance() async {
        let graph = createCompleteGraph(n: 50)

        // Note: XCTest measure blocks don't directly support async
        // This will measure the synchronous overhead, not the full async work
        for _ in 0..<10 {
            _ = await GraphPrimitivesKernels.computeGraphProperties(
                matrix: graph,
                options: GraphPrimitivesKernels.GraphPropertiesOptions(directed: false)
            )
        }
    }

    func testKCorePerformance() {
        let graph = createCompleteGraph(n: 100)

        measure {
            _ = GraphPrimitivesKernels.kCoreDecomposition(matrix: graph)
        }
    }

    func testMotifDetectionPerformance() async {
        let graph = createCompleteGraph(n: 20)

        // Note: XCTest measure blocks don't directly support async
        // This will run the test multiple times for basic performance validation
        for _ in 0..<10 {
            _ = await GraphPrimitivesKernels.detectMotifs(
                matrix: graph,
                options: GraphPrimitivesKernels.MotifDetectionOptions(motifSizes: [3, 4])
            )
        }
    }
}
