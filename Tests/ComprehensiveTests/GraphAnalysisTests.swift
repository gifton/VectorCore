//
//  GraphAnalysisTests.swift
//  VectorCore
//
//  Tests for Graph Analysis Kernels - Part 1
//

import XCTest
@testable import VectorCore

final class GraphAnalysisTests: XCTestCase {

    // MARK: - Test Helpers

    func createTestGraph() -> SparseMatrix {
        // Create a small test graph:
        // 0 -> 1, 2
        // 1 -> 2
        // 2 -> 0, 3
        // 3 -> 2
        let rows = 4
        let cols = 4
        let rowPointers = ContiguousArray<UInt32>([0, 2, 3, 5, 6])
        let columnIndices = ContiguousArray<UInt32>([1, 2, 2, 0, 3, 2])
        let values: ContiguousArray<Float>? = nil // unweighted

        return SparseMatrix(
            rows: rows,
            cols: cols,
            rowPointers: rowPointers,
            columnIndices: columnIndices,
            values: values
        )
    }

    func createWeightedGraph() -> SparseMatrix {
        // Same structure but with weights
        let rows = 4
        let cols = 4
        let rowPointers = ContiguousArray<UInt32>([0, 2, 3, 5, 6])
        let columnIndices = ContiguousArray<UInt32>([1, 2, 2, 0, 3, 2])
        let values = ContiguousArray<Float>([0.5, 0.5, 1.0, 0.3, 0.7, 1.0])

        return SparseMatrix(
            rows: rows,
            cols: cols,
            rowPointers: rowPointers,
            columnIndices: columnIndices,
            values: values
        )
    }

    func createLineGraph(n: Int) -> SparseMatrix {
        // Create a line graph: 0 -> 1 -> 2 -> ... -> n-1
        var rowPointers = ContiguousArray<UInt32>(repeating: 0, count: n + 1)
        var columnIndices = ContiguousArray<UInt32>()

        for i in 0..<n {
            rowPointers[i] = UInt32(columnIndices.count)
            if i > 0 {
                columnIndices.append(UInt32(i - 1))
            }
            if i < n - 1 {
                columnIndices.append(UInt32(i + 1))
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

    // MARK: - PageRank Tests

    func testPageRankBasic() {
        let graph = createTestGraph()
        let options = GraphPrimitivesKernels.PageRankOptions(
            dampingFactor: 0.85,
            tolerance: 1e-6,
            maxIterations: 100
        )

        let result = GraphPrimitivesKernels.pageRank(matrix: graph, options: options)

        XCTAssertEqual(result.scores.count, 4)
        XCTAssertTrue(result.converged, "PageRank should converge")

        // Check that scores sum to approximately 1
        let sum = result.scores.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 0.01, "PageRank scores should sum to 1")

        // Node 2 should have highest PageRank (most incoming links)
        let maxScore = result.scores.max()!
        let maxIndex = result.scores.firstIndex(of: maxScore)!
        XCTAssertEqual(maxIndex, 2, "Node 2 should have highest PageRank")
    }

    func testPageRankPersonalized() {
        let graph = createTestGraph()
        let personalization: [Int32: Float] = [0: 1.0] // Personalized to node 0

        let options = GraphPrimitivesKernels.PageRankOptions(
            dampingFactor: 0.85,
            tolerance: 1e-6,
            maxIterations: 100,
            personalized: personalization
        )

        let result = GraphPrimitivesKernels.pageRank(matrix: graph, options: options)

        XCTAssertTrue(result.converged)
        // Node 0 and its neighbors should have higher scores
        XCTAssertGreaterThan(result.scores[0], 0.2, "Node 0 should have high score in personalized PageRank")
    }

    func testPageRankDanglingNodes() {
        // Create graph with dangling node (no outgoing edges)
        let rows = 3
        let cols = 3
        let rowPointers = ContiguousArray<UInt32>([0, 1, 2, 2]) // Node 2 has no outgoing edges
        let columnIndices = ContiguousArray<UInt32>([1, 2])
        let values: ContiguousArray<Float>? = nil

        let graph = SparseMatrix(
            rows: rows,
            cols: cols,
            rowPointers: rowPointers,
            columnIndices: columnIndices,
            values: values
        )

        let result = GraphPrimitivesKernels.pageRank(matrix: graph)

        XCTAssertTrue(result.converged)
        let sum = result.scores.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 0.01, "PageRank should handle dangling nodes correctly")
    }

    // MARK: - Betweenness Centrality Tests

    func testBetweennessCentralityBasic() {
        let graph = createTestGraph()
        let options = GraphPrimitivesKernels.BetweennessCentralityOptions(
            normalized: true,
            weighted: false,
            parallel: false
        )

        let centrality = GraphPrimitivesKernels.betweennessCentrality(matrix: graph, options: options)

        XCTAssertEqual(centrality.count, 4)
        // All values should be non-negative
        for value in centrality {
            XCTAssertGreaterThanOrEqual(value, 0)
        }
    }

    func testBetweennessCentralityWeighted() {
        let graph = createWeightedGraph()
        let options = GraphPrimitivesKernels.BetweennessCentralityOptions(
            normalized: true,
            weighted: true,
            parallel: false
        )

        let centrality = GraphPrimitivesKernels.betweennessCentrality(matrix: graph, options: options)

        XCTAssertEqual(centrality.count, 4)
        // Check that at least one node has non-zero centrality
        let maxCentrality = centrality.max()!
        XCTAssertGreaterThan(maxCentrality, 0, "At least one node should have non-zero centrality")
    }

    func testBetweennessCentralityApproximate() {
        let graph = createLineGraph(n: 10)
        let options = GraphPrimitivesKernels.BetweennessCentralityOptions(
            normalized: true,
            weighted: false,
            approximate: true,
            sampleSize: 5
        )

        let centrality = GraphPrimitivesKernels.betweennessCentrality(matrix: graph, options: options)

        XCTAssertEqual(centrality.count, 10)
        // Middle nodes should have higher centrality in a line graph
        let middleIndex = 4
        XCTAssertGreaterThan(centrality[middleIndex], centrality[0], "Middle nodes should have higher centrality")
        XCTAssertGreaterThan(centrality[middleIndex], centrality[9], "Middle nodes should have higher centrality")
    }

    func testBetweennessCentralityParallel() {
        let graph = createLineGraph(n: 20)
        let serialOptions = GraphPrimitivesKernels.BetweennessCentralityOptions(
            normalized: true,
            parallel: false
        )
        let parallelOptions = GraphPrimitivesKernels.BetweennessCentralityOptions(
            normalized: true,
            parallel: true
        )

        let serialCentrality = GraphPrimitivesKernels.betweennessCentrality(matrix: graph, options: serialOptions)
        let parallelCentrality = GraphPrimitivesKernels.betweennessCentrality(matrix: graph, options: parallelOptions)

        // Results should be approximately the same
        for i in 0..<serialCentrality.count {
            XCTAssertEqual(serialCentrality[i], parallelCentrality[i], accuracy: 1e-5,
                          "Serial and parallel results should match")
        }
    }

    // MARK: - Eigenvector Centrality Tests

    func testEigenvectorCentrality() {
        let graph = createTestGraph()
        let options = GraphPrimitivesKernels.EigenvectorCentralityOptions(
            tolerance: 1e-6,
            maxIterations: 100
        )

        let centrality = GraphPrimitivesKernels.eigenvectorCentrality(matrix: graph, options: options)

        XCTAssertEqual(centrality.count, 4)

        // Check normalization (L2 norm should be 1)
        let norm = centrality.map { $0 * $0 }.reduce(0, +)
        XCTAssertEqual(sqrt(norm), 1.0, accuracy: 0.01, "Eigenvector should be normalized")
    }

    func testEigenvectorCentralityWithStartVector() {
        let graph = createTestGraph()
        let startVector = ContiguousArray<Float>([0.5, 0.5, 0.5, 0.5])
        let options = GraphPrimitivesKernels.EigenvectorCentralityOptions(
            tolerance: 1e-6,
            maxIterations: 100,
            startVector: startVector
        )

        let centrality = GraphPrimitivesKernels.eigenvectorCentrality(matrix: graph, options: options)

        XCTAssertEqual(centrality.count, 4)
        // Should still converge to same eigenvector regardless of start
        let norm = centrality.map { $0 * $0 }.reduce(0, +)
        XCTAssertEqual(sqrt(norm), 1.0, accuracy: 0.01)
    }

    // MARK: - Community Detection Tests

    func testCommunityDetection() {
        // Create a graph with clear community structure
        // Two cliques connected by a single edge
        let rows = 6
        let cols = 6
        let rowPointers = ContiguousArray<UInt32>([0, 2, 4, 7, 9, 11, 13])
        let columnIndices = ContiguousArray<UInt32>([
            1, 2,       // Node 0 -> 1, 2
            0, 2,       // Node 1 -> 0, 2
            0, 1, 3,    // Node 2 -> 0, 1, 3 (bridge)
            2, 4,       // Node 3 -> 2, 4 (bridge)
            3, 5,       // Node 4 -> 3, 5
            3, 4        // Node 5 -> 3, 4
        ])
        let values: ContiguousArray<Float>? = nil

        let graph = SparseMatrix(
            rows: rows,
            cols: cols,
            rowPointers: rowPointers,
            columnIndices: columnIndices,
            values: values
        )

        let options = GraphPrimitivesKernels.CommunityDetectionOptions(
            resolution: 1.0,
            randomSeed: 42,
            maxIterations: 10
        )

        let result = GraphPrimitivesKernels.detectCommunities(matrix: graph, options: options)

        XCTAssertEqual(result.communities.count, 6)
        // Should detect 2 communities
        XCTAssertLessThanOrEqual(result.numCommunities, 3, "Should detect approximately 2-3 communities")
        XCTAssertGreaterThan(result.modularity, 0, "Modularity should be positive")
    }

    func testCommunityDetectionWithResolution() {
        let graph = createTestGraph()

        let lowResOptions = GraphPrimitivesKernels.CommunityDetectionOptions(
            resolution: 0.5,
            randomSeed: 42
        )
        let highResOptions = GraphPrimitivesKernels.CommunityDetectionOptions(
            resolution: 2.0,
            randomSeed: 42
        )

        let lowResResult = GraphPrimitivesKernels.detectCommunities(matrix: graph, options: lowResOptions)
        let highResResult = GraphPrimitivesKernels.detectCommunities(matrix: graph, options: highResOptions)

        // Higher resolution should generally lead to more communities
        XCTAssertLessThanOrEqual(lowResResult.numCommunities, highResResult.numCommunities,
                                 "Higher resolution should find more communities")
    }

    // MARK: - Label Propagation Tests

    func testLabelPropagation() {
        let graph = createTestGraph()
        let labels = GraphPrimitivesKernels.labelPropagation(matrix: graph, maxIterations: 10)

        XCTAssertEqual(labels.count, 4)
        // Should converge to some partitioning
        let uniqueLabels = Set(labels)
        XCTAssertGreaterThan(uniqueLabels.count, 0)
        XCTAssertLessThanOrEqual(uniqueLabels.count, 4)
    }

    func testLabelPropagationConvergence() {
        // Line graph should converge to single community
        let graph = createLineGraph(n: 5)
        let labels = GraphPrimitivesKernels.labelPropagation(matrix: graph, maxIterations: 100)

        XCTAssertEqual(labels.count, 5)
        // Connected graph often converges to single community
        let uniqueLabels = Set(labels)
        XCTAssertLessThanOrEqual(uniqueLabels.count, 3, "Line graph should have few communities")
    }

    // MARK: - Clustering Coefficient Tests

    func testClusteringCoefficientTriangle() {
        // Create a triangle graph (fully connected 3 nodes)
        let rows = 3
        let cols = 3
        let rowPointers = ContiguousArray<UInt32>([0, 2, 4, 6])
        let columnIndices = ContiguousArray<UInt32>([1, 2, 0, 2, 0, 1])
        let values: ContiguousArray<Float>? = nil

        let graph = SparseMatrix(
            rows: rows,
            cols: cols,
            rowPointers: rowPointers,
            columnIndices: columnIndices,
            values: values
        )

        let localCoeff = GraphPrimitivesKernels.clusteringCoefficient(matrix: graph, local: true)
        let globalCoeff = GraphPrimitivesKernels.clusteringCoefficient(matrix: graph, local: false)

        // Triangle has perfect clustering
        XCTAssertEqual(localCoeff, 1.0, accuracy: 0.01, "Triangle should have clustering coefficient of 1")
        XCTAssertGreaterThan(globalCoeff, 0, "Global coefficient should be positive")
    }

    func testClusteringCoefficientLine() {
        // Line graph has no triangles
        let graph = createLineGraph(n: 5)
        let localCoeff = GraphPrimitivesKernels.clusteringCoefficient(matrix: graph, local: true)

        XCTAssertEqual(localCoeff, 0.0, accuracy: 0.01, "Line graph should have clustering coefficient of 0")
    }

    // MARK: - Performance Tests

    func testPageRankPerformance() {
        measure {
            let graph = createLineGraph(n: 100)
            _ = GraphPrimitivesKernels.pageRank(matrix: graph)
        }
    }

    func testBetweennessCentralityPerformance() {
        let graph = createLineGraph(n: 50)
        let options = GraphPrimitivesKernels.BetweennessCentralityOptions(
            normalized: true,
            approximate: true,
            sampleSize: 10
        )

        measure {
            _ = GraphPrimitivesKernels.betweennessCentrality(matrix: graph, options: options)
        }
    }

    func testCommunityDetectionPerformance() {
        let graph = createLineGraph(n: 100)
        let options = GraphPrimitivesKernels.CommunityDetectionOptions(
            randomSeed: 42,
            maxIterations: 5
        )

        measure {
            _ = GraphPrimitivesKernels.detectCommunities(matrix: graph, options: options)
        }
    }
}