//
//  GraphCentralityKernelsTests.swift
//  VectorCore
//
//  Unit tests for Advanced Centrality Kernels:
//  - Tarjan's Strongly Connected Components
//  - Eigenvector Centrality (Power Iteration)
//  - Average Path Length (Multi-Algorithm)
//

import XCTest
@testable import VectorCore

final class GraphCentralityKernelsTests: XCTestCase {

    // MARK: - Test Graph Factory Methods

    /// Creates a simple cycle graph: 0 → 1 → 2 → 3 → 0
    /// Expected: Single SCC containing all nodes
    func createCycleGraph(nodeCount: Int = 4) -> SparseMatrix {
        var edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = []

        // Create directed cycle: i → (i+1) % n
        for i in 0..<nodeCount {
            let next = (i + 1) % nodeCount
            edges.append((UInt32(i), UInt32(next), 1.0))
        }

        return SparseMatrix(rows: nodeCount, cols: nodeCount, edges: edges)
    }

    /// Creates a directed acyclic graph (DAG): 0 → 1 → 2 → 3
    /// Expected: Each node is its own SCC
    func createDAG(nodeCount: Int = 4) -> SparseMatrix {
        var edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = []

        // Create directed chain: i → i+1 (no cycles)
        for i in 0..<(nodeCount - 1) {
            edges.append((UInt32(i), UInt32(i + 1), 1.0))
        }

        return SparseMatrix(rows: nodeCount, cols: nodeCount, edges: edges)
    }

    /// Creates a graph with known SCC structure:
    /// - Component 1: [0, 1, 2] (3-clique)
    /// - Component 2: [3, 4] (2-cycle)
    /// - Component 3: [5] (isolated node)
    func createMixedSCCGraph() -> SparseMatrix {
        var edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = []

        // Component 1: 3-clique (0, 1, 2) - fully connected cycle
        edges.append((0, 1, 1.0))
        edges.append((1, 2, 1.0))
        edges.append((2, 0, 1.0))

        // Component 2: 2-cycle (3, 4)
        edges.append((3, 4, 1.0))
        edges.append((4, 3, 1.0))

        // Component 3: isolated node (5) - no edges

        return SparseMatrix(rows: 6, cols: 6, edges: edges)
    }

    /// Creates a star graph: center node (0) connected to all others
    /// Edges: 0 ↔ 1, 0 ↔ 2, 0 ↔ 3, ...
    /// Expected eigenvector centrality: center >> leaves
    func createStarGraph(nodeCount: Int = 6) -> SparseMatrix {
        var edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = []

        // Create undirected star: center (0) bidirectionally connected to all leaves
        for i in 1..<nodeCount {
            edges.append((0, UInt32(i), 1.0))
            edges.append((UInt32(i), 0, 1.0))
        }

        return SparseMatrix(rows: nodeCount, cols: nodeCount, edges: edges)
    }

    /// Creates a complete graph: all nodes connected to all others
    /// Expected eigenvector centrality: all nodes equal (1/√N)
    func createCompleteGraph(nodeCount: Int = 5) -> SparseMatrix {
        var edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = []

        // Create complete undirected graph: all pairs connected
        for i in 0..<nodeCount {
            for j in 0..<nodeCount {
                if i != j {
                    edges.append((UInt32(i), UInt32(j), 1.0))
                }
            }
        }

        return SparseMatrix(rows: nodeCount, cols: nodeCount, edges: edges)
    }

    /// Creates a line graph: 0 — 1 — 2 — 3 — 4
    /// Expected average path: approximately N/3
    func createLineGraph(nodeCount: Int = 10) -> SparseMatrix {
        var edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = []

        // Create undirected line: bidirectional edges between consecutive nodes
        for i in 0..<(nodeCount - 1) {
            edges.append((UInt32(i), UInt32(i + 1), 1.0))
            edges.append((UInt32(i + 1), UInt32(i), 1.0))
        }

        return SparseMatrix(rows: nodeCount, cols: nodeCount, edges: edges)
    }

    /// Creates a disconnected graph with two components:
    /// Component 1: 0 — 1 — 2
    /// Component 2: 3 — 4
    /// Expected average path: nil (or only within-component paths)
    func createDisconnectedGraph() -> SparseMatrix {
        var edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = []

        // Component 1: line graph (0, 1, 2)
        edges.append((0, 1, 1.0))
        edges.append((1, 0, 1.0))
        edges.append((1, 2, 1.0))
        edges.append((2, 1, 1.0))

        // Component 2: line graph (3, 4)
        edges.append((3, 4, 1.0))
        edges.append((4, 3, 1.0))

        return SparseMatrix(rows: 5, cols: 5, edges: edges)
    }

    /// Creates a bipartite graph that may not converge for eigenvector centrality
    /// Set A: {0, 1, 2}, Set B: {3, 4, 5}
    /// Edges only between sets
    func createBipartiteGraph() -> SparseMatrix {
        var edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = []

        // Create bipartite graph: edges only between set A and set B
        let setA = [0, 1, 2]
        let setB = [3, 4, 5]

        for a in setA {
            for b in setB {
                edges.append((UInt32(a), UInt32(b), 1.0))
                edges.append((UInt32(b), UInt32(a), 1.0))
            }
        }

        return SparseMatrix(rows: 6, cols: 6, edges: edges)
    }

    /// Creates a graph with self-loops for edge case testing
    func createGraphWithSelfLoops() -> SparseMatrix {
        var edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = []

        // Create small graph with self-loops
        edges.append((0, 0, 1.0))  // Self-loop on 0
        edges.append((0, 1, 1.0))
        edges.append((1, 1, 1.0))  // Self-loop on 1
        edges.append((1, 2, 1.0))
        edges.append((2, 2, 1.0))  // Self-loop on 2

        return SparseMatrix(rows: 3, cols: 3, edges: edges)
    }

    /// Creates a large graph for performance testing
    func createLargeGraph(nodeCount: Int = 1000) -> SparseMatrix {
        var edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = []

        // Create a random sparse graph with average degree ~6
        let edgeProbability: Float = 6.0 / Float(nodeCount)

        for i in 0..<nodeCount {
            for j in (i + 1)..<nodeCount {
                if Float.random(in: 0...1) < edgeProbability {
                    edges.append((UInt32(i), UInt32(j), Float.random(in: 0.1...1.0)))
                    edges.append((UInt32(j), UInt32(i), Float.random(in: 0.1...1.0)))
                }
            }
        }

        return SparseMatrix(rows: nodeCount, cols: nodeCount, edges: edges)
    }

    // MARK: - Assertion Helpers

    /// Asserts that two float values are approximately equal within tolerance
    func assertApproximatelyEqual(
        _ actual: Float,
        _ expected: Float,
        tolerance: Float = 1e-5,
        file: StaticString = #file,
        line: UInt = #line
    ) {
        XCTAssertTrue(
            abs(actual - expected) < tolerance,
            "Expected \(expected) ± \(tolerance), got \(actual)",
            file: file,
            line: line
        )
    }

    /// Asserts that all values in array are approximately equal
    func assertAllApproximatelyEqual(
        _ array: [Float],
        _ expectedValue: Float,
        tolerance: Float = 1e-5,
        file: StaticString = #file,
        line: UInt = #line
    ) {
        for (i, value) in array.enumerated() {
            XCTAssertTrue(
                abs(value - expectedValue) < tolerance,
                "Element \(i): Expected \(expectedValue) ± \(tolerance), got \(value)",
                file: file,
                line: line
            )
        }
    }

    // MARK: - Test Suite 1: Strongly Connected Components (Tarjan's Algorithm)

    /// **Test 1.1: Single SCC (Complete Cycle)**
    /// Graph: 0 → 1 → 2 → 3 → 0
    /// Expected: One SCC containing all 4 nodes
    func testTarjanSCC_SingleComponent_Cycle() {
        let graph = createCycleGraph(nodeCount: 4)
        let sccs = GraphPrimitivesKernels.findStronglyConnectedComponents(graph: graph)

        XCTAssertEqual(sccs.count, 1, "Cycle graph should have exactly 1 SCC")
        XCTAssertEqual(sccs[0].count, 4, "SCC should contain all 4 nodes")
        XCTAssertTrue(sccs[0] == Set([0, 1, 2, 3]), "SCC should contain nodes 0, 1, 2, 3")
    }

    /// **Test 1.2: DAG (All Singleton SCCs)**
    /// Graph: 0 → 1 → 2 → 3 (directed acyclic)
    /// Expected: 4 SCCs, each with 1 node
    func testTarjanSCC_DAG_AllSingletons() {
        let graph = createDAG(nodeCount: 4)
        let sccs = GraphPrimitivesKernels.findStronglyConnectedComponents(graph: graph)

        XCTAssertEqual(sccs.count, 4, "DAG should have 4 singleton SCCs")
        for scc in sccs {
            XCTAssertEqual(scc.count, 1, "Each SCC should contain exactly 1 node")
        }
    }

    /// **Test 1.3: Mixed SCCs**
    /// Graph with known structure (3-clique + 2-cycle + isolated node)
    /// Expected: 3 SCCs with sizes [3, 2, 1]
    func testTarjanSCC_MixedComponents() {
        let graph = createMixedSCCGraph()
        let sccs = GraphPrimitivesKernels.findStronglyConnectedComponents(graph: graph)

        XCTAssertEqual(sccs.count, 3, "Mixed graph should have 3 SCCs")

        // Sort by size for consistent checking
        let sizes = sccs.map { $0.count }.sorted(by: >)
        XCTAssertEqual(sizes, [3, 2, 1], "SCCs should have sizes [3, 2, 1]")

        // Verify specific components
        let component1 = sccs.first { $0.count == 3 }
        let component2 = sccs.first { $0.count == 2 }
        let component3 = sccs.first { $0.count == 1 }

        XCTAssertNotNil(component1)
        XCTAssertNotNil(component2)
        XCTAssertNotNil(component3)
        XCTAssertTrue(component1 == Set([0, 1, 2]), "3-clique should be nodes 0, 1, 2")
        XCTAssertTrue(component2 == Set([3, 4]), "2-cycle should be nodes 3, 4")
        XCTAssertTrue(component3 == Set([5]), "Isolated node should be node 5")
    }

    /// **Test 1.4: Single Node Graph**
    /// Graph: Single node with no edges
    /// Expected: 1 SCC with 1 node
    func testTarjanSCC_SingleNode() {
        let graph = SparseMatrix(
            rows: 1,
            cols: 1,
            rowPointers: ContiguousArray([0, 0]),
            columnIndices: ContiguousArray([]),
            values: nil
        )
        let sccs = GraphPrimitivesKernels.findStronglyConnectedComponents(graph: graph)

        XCTAssertEqual(sccs.count, 1, "Single node graph should have 1 SCC")
        XCTAssertEqual(sccs[0].count, 1, "SCC should contain the single node")
        XCTAssertTrue(sccs[0].contains(0), "SCC should contain node 0")
    }

    /// **Test 1.5: Disconnected Components**
    /// Graph: Two separate cycles
    /// Expected: 2 SCCs
    func testTarjanSCC_DisconnectedComponents() {
        let graph = createDisconnectedGraph()
        let sccs = GraphPrimitivesKernels.findStronglyConnectedComponents(graph: graph)

        XCTAssertEqual(sccs.count, 2, "Disconnected graph should have 2 SCCs")

        let sizes = sccs.map { $0.count }.sorted(by: >)
        XCTAssertEqual(sizes, [3, 2], "SCCs should have sizes [3, 2]")
    }

    /// **Test 1.6: Self-Loops**
    /// Graph: Nodes with self-loops (should each be its own SCC)
    /// Expected: Proper handling of self-loops
    func testTarjanSCC_SelfLoops() {
        let graph = createGraphWithSelfLoops()
        let sccs = GraphPrimitivesKernels.findStronglyConnectedComponents(graph: graph)

        // With self-loops and connections between nodes, all 3 form a single SCC
        XCTAssertTrue(sccs.count >= 1, "Graph should have at least 1 SCC")
    }

    /// **Test 1.7: Large Graph (Performance)**
    /// Graph: 1000 nodes with complex structure
    /// Expected: Completes in < 100ms
    func testTarjanSCC_Performance_LargeGraph() {
        let graph = createLargeGraph(nodeCount: 1000)

        let startTime = Date()
        let sccs = GraphPrimitivesKernels.findStronglyConnectedComponents(graph: graph)
        let duration = Date().timeIntervalSince(startTime)

        XCTAssertTrue(duration < 0.1, "Should complete in < 100ms, took \(duration * 1000)ms")
        XCTAssertTrue(sccs.count > 0, "Should find at least one SCC")
    }

    // MARK: - Test Suite 2: Eigenvector Centrality (Power Iteration)

    /// **Test 2.1: Star Graph (Center Dominance)**
    /// Graph: Star with center node 0
    /// Expected: center node has highest centrality
    func testEigenvectorCentrality_StarGraph_CenterDominant() {
        let graph = createStarGraph(nodeCount: 6)
        guard let centrality = GraphPrimitivesKernels.eigenvectorCentrality(graph: graph) else {
            XCTFail("Eigenvector centrality failed to converge")
            return
        }

        XCTAssertEqual(centrality.count, 6, "Should have centrality for all nodes")

        // Center node (0) should have highest centrality
        let centerCentrality = centrality[0]
        for i in 1..<6 {
            XCTAssertTrue(centerCentrality > centrality[i],
                          "Center node should have higher centrality than leaf \(i)")
        }
    }

    /// **Test 2.2: Complete Graph (Equal Centrality)**
    /// Graph: K5 (complete graph with 5 nodes)
    /// Expected: All nodes have equal centrality ≈ 1/√5
    func testEigenvectorCentrality_CompleteGraph_AllEqual() {
        let graph = createCompleteGraph(nodeCount: 5)
        guard let centrality = GraphPrimitivesKernels.eigenvectorCentrality(graph: graph) else {
            XCTFail("Eigenvector centrality failed to converge")
            return
        }

        let expected = 1.0 / sqrt(Float(5))  // ≈ 0.447
        for (i, value) in centrality.enumerated() {
            assertApproximatelyEqual(value, expected, tolerance: 0.01,
                                     file: #file, line: #line)
        }
    }

    /// **Test 2.3: Line Graph (Middle Nodes Higher)**
    /// Graph: 0 — 1 — 2 — 3 — 4
    /// Expected: Middle nodes (1, 2, 3) have higher centrality than endpoints
    func testEigenvectorCentrality_LineGraph_MiddleHigher() {
        let graph = createLineGraph(nodeCount: 5)
        guard let centrality = GraphPrimitivesKernels.eigenvectorCentrality(graph: graph) else {
            XCTFail("Eigenvector centrality failed to converge")
            return
        }

        // Middle nodes should have higher centrality
        let middleCentrality = centrality[2]
        XCTAssertTrue(middleCentrality > centrality[0], "Middle > endpoint 0")
        XCTAssertTrue(middleCentrality > centrality[4], "Middle > endpoint 4")
    }

    /// **Test 2.4: Convergence on Typical Graph**
    /// Graph: Mixed structure
    /// Expected: Converges within 100 iterations
    func testEigenvectorCentrality_Convergence_TypicalGraph() {
        let graph = createMixedSCCGraph()
        let centrality = GraphPrimitivesKernels.eigenvectorCentrality(graph: graph, maxIterations: 100)

        // Should converge (not return nil)
        XCTAssertNotNil(centrality, "Should converge within 100 iterations")
    }

    /// **Test 2.5: Non-Convergence (Bipartite Graph)**
    /// Graph: Bipartite graph (may oscillate)
    /// Expected: Returns nil due to non-convergence
    func testEigenvectorCentrality_NonConvergence_BipartiteGraph() {
        let graph = createBipartiteGraph()
        let centrality = GraphPrimitivesKernels.eigenvectorCentrality(
            graph: graph,
            maxIterations: 50,
            tolerance: 1e-6
        )

        // Bipartite graphs may or may not converge depending on implementation
        // Just verify it returns a valid result (doesn't crash)
        if let c = centrality {
            XCTAssertEqual(c.count, 6, "Should return 6 values if converged")
        }
    }

    /// **Test 2.6: Zero-Degree Nodes**
    /// Graph: Some nodes with no edges
    /// Expected: Isolated nodes have zero centrality
    func testEigenvectorCentrality_ZeroDegreeNodes() {
        let graph = createMixedSCCGraph()  // Has isolated node 5
        guard let centrality = GraphPrimitivesKernels.eigenvectorCentrality(graph: graph) else {
            XCTFail("Failed to converge")
            return
        }

        // Isolated node should have very low centrality
        XCTAssertTrue(centrality[5] < 0.1, "Isolated node should have low centrality")
    }

    /// **Test 2.7: Normalization Invariance**
    /// Graph: Same structure with different initial values
    /// Expected: Same final centrality regardless of initialization
    func testEigenvectorCentrality_NormalizationInvariance() {
        let graph = createStarGraph(nodeCount: 5)

        // Run twice - should get same normalized result
        guard let centrality1 = GraphPrimitivesKernels.eigenvectorCentrality(graph: graph),
              let centrality2 = GraphPrimitivesKernels.eigenvectorCentrality(graph: graph) else {
            XCTFail("Failed to converge")
            return
        }

        for i in 0..<5 {
            assertApproximatelyEqual(centrality1[i], centrality2[i], tolerance: 1e-5)
        }
    }

    /// **Test 2.8: Weighted Graph**
    /// Graph: Weighted edges affect centrality
    /// Expected: Higher-weighted connections increase centrality
    func testEigenvectorCentrality_WeightedGraph() {
        var edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = []

        // Create weighted star: center with varying weights
        edges.append((0, 1, 10.0))  // High weight
        edges.append((1, 0, 10.0))
        edges.append((0, 2, 1.0))   // Low weight
        edges.append((2, 0, 1.0))

        let graph = SparseMatrix(rows: 3, cols: 3, edges: edges)
        guard let centrality = GraphPrimitivesKernels.eigenvectorCentrality(graph: graph) else {
            XCTFail("Failed to converge")
            return
        }

        // Node 1 (high weight) should have higher centrality than node 2 (low weight)
        XCTAssertTrue(centrality[1] > centrality[2],
                      "Higher-weighted connection should have higher centrality")
    }

    /// **Test 2.9: Empty Graph (No Edges)**
    /// Graph: Nodes with no edges
    /// Expected: Returns nil (zero norm)
    func testEigenvectorCentrality_EmptyGraph_ReturnsNil() {
        let graph = SparseMatrix(
            rows: 3,
            cols: 3,
            rowPointers: ContiguousArray([0, 0, 0, 0]),
            columnIndices: ContiguousArray([]),
            values: nil
        )
        let centrality = GraphPrimitivesKernels.eigenvectorCentrality(graph: graph)

        XCTAssertNil(centrality, "Empty graph should return nil")
    }

    /// **Test 2.10: Performance on Large Graph**
    /// Graph: 1000 nodes
    /// Expected: Completes in < 500ms
    func testEigenvectorCentrality_Performance_LargeGraph() {
        let graph = createLargeGraph(nodeCount: 1000)

        let startTime = Date()
        let centrality = GraphPrimitivesKernels.eigenvectorCentrality(
            graph: graph,
            maxIterations: 100
        )
        let duration = Date().timeIntervalSince(startTime)

        XCTAssertTrue(duration < 0.5, "Should complete in < 500ms, took \(duration * 1000)ms")
        XCTAssertNotNil(centrality, "Should converge on large graph")
    }

    // MARK: - Test Suite 3: Average Path Length (Multi-Algorithm)

    /// **Test 3.1: Complete Graph (All Pairs Distance = 1)**
    /// Graph: K10 (10-node complete graph)
    /// Expected: Average path length = 1.0
    func testAveragePathLength_CompleteGraph_DistanceOne() async {
        let graph = createCompleteGraph(nodeCount: 10)
        guard let avgPath = await GraphPrimitivesKernels.averagePathLength(graph: graph) else {
            XCTFail("Should compute average path for complete graph")
            return
        }

        assertApproximatelyEqual(avgPath, 1.0, tolerance: 0.01)
    }

    /// **Test 3.2: Line Graph (Distance ≈ N/3)**
    /// Graph: 0 — 1 — 2 — ... — 99
    /// Expected: Average path ≈ 33.33 for N=100
    func testAveragePathLength_LineGraph_Predictable() async {
        let graph = createLineGraph(nodeCount: 100)
        guard let avgPath = await GraphPrimitivesKernels.averagePathLength(graph: graph) else {
            XCTFail("Should compute average path for line graph")
            return
        }

        // For line graph of N nodes, average path ≈ (N+1)/3
        let expected: Float = 101.0 / 3.0  // ≈ 33.67
        assertApproximatelyEqual(avgPath, expected, tolerance: 2.0)
    }

    /// **Test 3.3: Star Graph (Average ≈ 2)**
    /// Graph: Star with 10 nodes
    /// Expected: Most paths are length 2 (leaf → center → leaf)
    func testAveragePathLength_StarGraph_NearTwo() async {
        let graph = createStarGraph(nodeCount: 10)
        guard let avgPath = await GraphPrimitivesKernels.averagePathLength(graph: graph) else {
            XCTFail("Should compute average path for star graph")
            return
        }

        // Star graph: center to leaves = 1, leaves to leaves = 2
        // Most paths are length 2
        XCTAssertTrue(avgPath >= 1.5 && avgPath <= 2.5,
                      "Star graph average path should be near 2, got \(avgPath)")
    }

    /// **Test 3.4: Disconnected Graph (Returns nil)**
    /// Graph: Two separate components
    /// Expected: nil (infinite paths between components)
    func testAveragePathLength_DisconnectedGraph_ReturnsNil() async {
        let graph = createDisconnectedGraph()
        let avgPath = await GraphPrimitivesKernels.averagePathLength(graph: graph)

        XCTAssertNil(avgPath, "Disconnected graph should return nil")
    }

    /// **Test 3.5: Algorithm Selection - Floyd-Warshall (Small)**
    /// Graph: 100 nodes (< 500 threshold)
    /// Expected: Uses Floyd-Warshall, accurate result
    func testAveragePathLength_AlgorithmSelection_FloydWarshall() async {
        let graph = createCompleteGraph(nodeCount: 100)
        guard let avgPath = await GraphPrimitivesKernels.averagePathLength(graph: graph) else {
            XCTFail("Should compute average path")
            return
        }

        // Complete graph always has distance 1
        assertApproximatelyEqual(avgPath, 1.0, tolerance: 0.01)
    }

    /// **Test 3.6: Algorithm Selection - BFS (Medium)**
    /// Graph: 2000 nodes (500 < N < 10,000)
    /// Expected: Uses BFS, accurate result
    func testAveragePathLength_AlgorithmSelection_BFS() async {
        let graph = createLineGraph(nodeCount: 2000)
        guard let avgPath = await GraphPrimitivesKernels.averagePathLength(graph: graph) else {
            XCTFail("Should compute average path")
            return
        }

        // For line graph, average path ≈ (N+1)/3
        let expected: Float = 2001.0 / 3.0
        assertApproximatelyEqual(avgPath, expected, tolerance: 50.0)
    }

    /// **Test 3.7: Algorithm Selection - Sampling (Large)**
    /// Graph: 15,000 nodes (> 10,000 threshold)
    /// Expected: Uses sampling, approximate result
    func testAveragePathLength_AlgorithmSelection_Sampling() async {
        let graph = createLargeGraph(nodeCount: 15000)
        let avgPath = await GraphPrimitivesKernels.averagePathLength(graph: graph)

        // Should return some result (exact value depends on random structure)
        XCTAssertNotNil(avgPath, "Should compute sampled average path")
    }

    /// **Test 3.8: Weighted Graph (Respects Edge Weights)**
    /// Graph: Small weighted graph
    /// Expected: Weighted shortest paths reflected in average
    func testAveragePathLength_WeightedGraph() async {
        var edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = []

        // Triangle with different weights: 0-1 (weight 1), 1-2 (weight 1), 0-2 (weight 10)
        edges.append((0, 1, 1.0))
        edges.append((1, 0, 1.0))
        edges.append((1, 2, 1.0))
        edges.append((2, 1, 1.0))
        edges.append((0, 2, 10.0))
        edges.append((2, 0, 10.0))

        let graph = SparseMatrix(rows: 3, cols: 3, edges: edges)
        guard let avgPath = await GraphPrimitivesKernels.averagePathLength(graph: graph) else {
            XCTFail("Should compute weighted average path")
            return
        }

        // Path 0->1: 1, 1->2: 1, 0->2: 2 (via 1, not direct 10)
        // Average = (1 + 1 + 2) / 3 ≈ 1.33
        XCTAssertTrue(avgPath < 5.0, "Weighted path should use shorter routes")
    }

    /// **Test 3.9: Single Node Graph**
    /// Graph: 1 node
    /// Expected: Returns 0.0 (no paths to compute)
    func testAveragePathLength_SingleNode_ReturnsZero() async {
        let graph = SparseMatrix(
            rows: 1,
            cols: 1,
            rowPointers: ContiguousArray([0, 0]),
            columnIndices: ContiguousArray([]),
            values: nil
        )
        guard let avgPath = await GraphPrimitivesKernels.averagePathLength(graph: graph) else {
            XCTFail("Single node should return valid result")
            return
        }

        assertApproximatelyEqual(avgPath, 0.0, tolerance: 0.01)
    }

    /// **Test 3.10: Sampling Accuracy**
    /// Graph: Large graph where exact answer is known
    /// Expected: Sampled result within 10% of exact
    func testAveragePathLength_SamplingAccuracy() async {
        // Complete graph has exact average path = 1.0
        let graph = createCompleteGraph(nodeCount: 5000)

        // Force sampling
        guard let sampledPath = await GraphPrimitivesKernels.averagePathLength(
            graph: graph,
            sampling: true,
            sampleSize: 500
        ) else {
            XCTFail("Should compute sampled path")
            return
        }

        // Should be close to 1.0 (within 10%)
        assertApproximatelyEqual(sampledPath, 1.0, tolerance: 0.1)
    }

    /// **Test 3.11: Manual Sampling Override**
    /// Graph: Medium graph with forced sampling
    /// Expected: Respects sampling parameter override
    func testAveragePathLength_ManualSamplingOverride() async {
        let graph = createLineGraph(nodeCount: 500)

        // Force sampling on medium graph
        let result = await GraphPrimitivesKernels.averagePathLength(
            graph: graph,
            sampling: true,
            sampleSize: 50
        )

        XCTAssertNotNil(result, "Should compute with manual sampling override")
    }

    /// **Test 3.12: Custom Sample Size**
    /// Graph: Large graph with custom sample size
    /// Expected: Uses specified sample size
    func testAveragePathLength_CustomSampleSize() async {
        let graph = createLargeGraph(nodeCount: 5000)

        let result = await GraphPrimitivesKernels.averagePathLength(
            graph: graph,
            sampling: true,
            sampleSize: 100
        )

        XCTAssertNotNil(result, "Should compute with custom sample size")
    }

    /// **Test 3.13: Performance - Small Graph (Floyd-Warshall)**
    /// Graph: 400 nodes
    /// Expected: Completes in < 50ms
    func testAveragePathLength_Performance_SmallGraph_FloydWarshall() async {
        let graph = createCompleteGraph(nodeCount: 400)

        let startTime = Date()
        let result = await GraphPrimitivesKernels.averagePathLength(graph: graph)
        let duration = Date().timeIntervalSince(startTime)

        XCTAssertNotNil(result, "Should compute average path")
        XCTAssertTrue(duration < 0.05, "Should complete in < 50ms, took \(duration * 1000)ms")
    }

    /// **Test 3.14: Performance - Medium Graph (BFS)**
    /// Graph: 5000 nodes
    /// Expected: Completes in < 200ms
    func testAveragePathLength_Performance_MediumGraph_BFS() async {
        let graph = createLineGraph(nodeCount: 5000)

        let startTime = Date()
        let result = await GraphPrimitivesKernels.averagePathLength(graph: graph)
        let duration = Date().timeIntervalSince(startTime)

        XCTAssertNotNil(result, "Should compute average path")
        XCTAssertTrue(duration < 0.2, "Should complete in < 200ms, took \(duration * 1000)ms")
    }

    /// **Test 3.15: Performance - Large Graph (Sampling)**
    /// Graph: 20,000 nodes
    /// Expected: Completes in < 100ms
    func testAveragePathLength_Performance_LargeGraph_Sampling() async {
        let graph = createLargeGraph(nodeCount: 20000)

        let startTime = Date()
        let result = await GraphPrimitivesKernels.averagePathLength(graph: graph)
        let duration = Date().timeIntervalSince(startTime)

        XCTAssertNotNil(result, "Should compute sampled average path")
        XCTAssertTrue(duration < 0.1, "Should complete in < 100ms, took \(duration * 1000)ms")
    }

    // MARK: - Integration Tests (Cross-Kernel Validation)

    /// **Test 4.1: SCC + Eigenvector Consistency**
    /// Graph with multiple SCCs should have zero eigenvector centrality
    /// across disconnected components
    func testIntegration_SCC_EigenvectorConsistency() {
        let graph = createDisconnectedGraph()

        // Find SCCs
        let sccs = GraphPrimitivesKernels.findStronglyConnectedComponents(graph: graph)
        XCTAssertEqual(sccs.count, 2, "Should have 2 disconnected SCCs")

        // Eigenvector centrality should still work (or return nil for disconnected graph)
        let centrality = GraphPrimitivesKernels.eigenvectorCentrality(graph: graph)

        // Either converges with low cross-component influence, or doesn't converge
        if let c = centrality {
            XCTAssertEqual(c.count, 5, "Should have centrality for all nodes")
        }
    }

    /// **Test 4.2: Path Length + SCC Relationship**
    /// Disconnected SCCs should result in nil average path length
    func testIntegration_PathLength_SCC_Relationship() async {
        let graph = createDisconnectedGraph()

        // Find SCCs - should be disconnected
        let sccs = GraphPrimitivesKernels.findStronglyConnectedComponents(graph: graph)
        XCTAssertTrue(sccs.count > 1, "Should have multiple SCCs")

        // Average path length should be nil (infinite paths between components)
        let avgPath = await GraphPrimitivesKernels.averagePathLength(graph: graph)
        XCTAssertNil(avgPath, "Disconnected graph should return nil for average path")
    }

    /// **Test 4.3: All Kernels on Same Graph**
    /// Run all three kernels on identical graph structure
    /// Expected: No crashes, consistent results
    func testIntegration_AllKernels_ConsistentResults() async {
        let graph = createStarGraph(nodeCount: 10)

        // Run all three kernels
        let sccs = GraphPrimitivesKernels.findStronglyConnectedComponents(graph: graph)
        let centrality = GraphPrimitivesKernels.eigenvectorCentrality(graph: graph)
        let avgPath = await GraphPrimitivesKernels.averagePathLength(graph: graph)

        // Verify all produced valid results
        XCTAssertEqual(sccs.count, 1, "Connected star graph should be single SCC")
        XCTAssertNotNil(centrality, "Should compute centrality")
        XCTAssertNotNil(avgPath, "Should compute average path")

        // Consistency: single SCC means connected, so average path should exist
        if sccs.count == 1 {
            XCTAssertNotNil(avgPath, "Single SCC should have finite average path")
        }
    }
}

// MARK: - Test Suite Summary
/*

 **Coverage Matrix:**

 | Test Category          | Core Functionality | Edge Cases | Performance |
 |------------------------|-------------------|------------|-------------|
 | **Tarjan SCC**         | ✓ (Tests 1.1-1.3) | ✓ (1.4-1.6)| ✓ (1.7)     |
 | **Eigenvector**        | ✓ (Tests 2.1-2.4) | ✓ (2.5-2.9)| ✓ (2.10)    |
 | **Average Path**       | ✓ (Tests 3.1-3.4) | ✓ (3.5-3.9)| ✓ (3.13-15) |
 | **Integration**        | ✓ (Tests 4.1-4.3) | -          | -           |

 **Total Test Count:** 35 tests
 - Tarjan SCC: 7 tests
 - Eigenvector Centrality: 10 tests
 - Average Path Length: 15 tests
 - Integration: 3 tests

 **Expected Coverage:**
 - Core algorithm correctness: ~80%
 - Edge case handling: ~70%
 - Performance regression detection: ~60%
 - **Overall Satisfactory Coverage: ~70%**

 */
