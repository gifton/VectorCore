//
//  GraphTraversalTests.swift
//  Comprehensive test suite for graph traversal kernels
//

import XCTest
@testable import VectorCore

// MARK: - Test Suite

final class GraphTraversalTests: XCTestCase {

    // MARK: - Test Helpers

    /// Creates a simple 6-node test graph
    private func createSimpleGraph() -> SparseMatrix {
        let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
            // Row 0
            (0, 1, 1.0), (0, 3, 1.0),
            // Row 1
            (1, 0, 1.0), (1, 2, 1.0), (1, 4, 1.0),
            // Row 2
            (2, 1, 1.0), (2, 5, 1.0),
            // Row 3
            (3, 0, 1.0), (3, 4, 1.0),
            // Row 4
            (4, 1, 1.0), (4, 3, 1.0), (4, 5, 1.0),
            // Row 5
            (5, 2, 1.0), (5, 4, 1.0)
        ]

        return try! SparseMatrix(rows: 6, cols: 6, edges: edges)
    }

    /// Creates a weighted graph for shortest path testing
    private func createWeightedGraph() -> SparseMatrix {
        let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
            (0, 1, 4.0), (0, 3, 2.0),
            (1, 0, 4.0), (1, 2, 5.0), (1, 4, 1.0),
            (2, 1, 5.0), (2, 5, 3.0),
            (3, 0, 2.0), (3, 4, 8.0),
            (4, 1, 1.0), (4, 3, 8.0), (4, 5, 2.0),
            (5, 2, 3.0), (5, 4, 2.0)
        ]

        return try! SparseMatrix(rows: 6, cols: 6, edges: edges)
    }

    /// Creates a directed graph with cycles for SCC testing
    private func createDirectedGraphWithSCCs() -> SparseMatrix {
        let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
            (0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0),  // SCC 1: 3-node cycle
            (1, 3, 1.0),                             // Bridge to SCC 2
            (3, 4, 1.0), (4, 3, 1.0),               // SCC 2: 2-node cycle
            (5, 6, 1.0), (6, 5, 1.0),               // SCC 3: isolated cycle
            (7, 8, 1.0)                              // SCC 4 & 5: chain
        ]

        return try! SparseMatrix(rows: 9, cols: 9, edges: edges)
    }

    /// Creates a disconnected graph for component testing
    private func createDisconnectedGraph() -> SparseMatrix {
        let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
            // Component 1
            (0, 1, 1.0), (1, 0, 1.0), (1, 2, 1.0), (2, 1, 1.0),
            // Component 2
            (3, 4, 1.0), (4, 3, 1.0), (4, 5, 1.0), (5, 4, 1.0),
            // Component 3 (single node: 6)
            // Component 4
            (7, 8, 1.0), (8, 7, 1.0), (8, 9, 1.0), (9, 8, 1.0)
        ]

        return try! SparseMatrix(rows: 10, cols: 10, edges: edges)
    }

    /// Creates a grid graph for pathfinding tests
    private func createGridGraph(size: Int) -> SparseMatrix {
        var edges = ContiguousArray<(row: UInt32, col: UInt32, value: Float?)>()

        for row in 0..<size {
            for col in 0..<size {
                let node = row * size + col
                // Right edge
                if col < size - 1 {
                    edges.append((UInt32(node), UInt32(node + 1), 1.0))
                    edges.append((UInt32(node + 1), UInt32(node), 1.0))
                }
                // Down edge
                if row < size - 1 {
                    edges.append((UInt32(node), UInt32(node + size), 1.0))
                    edges.append((UInt32(node + size), UInt32(node), 1.0))
                }
            }
        }

        return try! SparseMatrix(rows: size * size, cols: size * size, edges: edges)
    }

    // MARK: - Part 1: Basic Traversal Tests

    func testBreadthFirstSearch() {
        let graph = createSimpleGraph()

        let result = GraphPrimitivesKernels.breadthFirstSearch(
            matrix: graph,
            source: 0,
            options: .init(parallel: false)
        )

        // Verify distances
        XCTAssertEqual(result.distances[0], 0)
        XCTAssertEqual(result.distances[1], 1)
        XCTAssertEqual(result.distances[2], 2)
        XCTAssertEqual(result.distances[3], 1)
        XCTAssertEqual(result.distances[4], 2)
        XCTAssertEqual(result.distances[5], 3)

        // Verify levels
        XCTAssertEqual(result.levels.count, 4)
        XCTAssertEqual(Set(result.levels[0]), Set([0]))
        XCTAssertEqual(Set(result.levels[1]), Set([1, 3]))
        XCTAssertEqual(Set(result.levels[2]), Set([2, 4]))
        XCTAssertEqual(Set(result.levels[3]), Set([5]))

        // Verify visit order starts with source
        XCTAssertEqual(result.visitOrder.first, 0)
    }

    func testParallelBFS() {
        let graph = createSimpleGraph()

        let serialResult = GraphPrimitivesKernels.breadthFirstSearch(
            matrix: graph,
            source: 0,
            options: .init(parallel: false)
        )

        let parallelResult = GraphPrimitivesKernels.breadthFirstSearch(
            matrix: graph,
            source: 0,
            options: .init(parallel: true)
        )

        // Results should be identical
        XCTAssertEqual(serialResult.distances, parallelResult.distances)
        XCTAssertEqual(serialResult.levels, parallelResult.levels)
    }

    func testDepthFirstSearch() {
        let graph = createSimpleGraph()

        let result = GraphPrimitivesKernels.depthFirstSearch(
            matrix: graph,
            source: 0,
            options: .init(visitAll: false, detectCycles: true)
        )

        // Verify all nodes are visited
        XCTAssertEqual(Set(result.visitOrder), Set([0, 1, 2, 3, 4, 5] as [Int32]))

        // Verify timestamps are consistent
        for i in 0..<6 {
            XCTAssertLessThan(result.discoveryTime[i], result.finishTime[i])
        }

        // Source should be discovered first
        XCTAssertEqual(result.discoveryTime[0], 0)
    }

    func testBidirectionalBFS() {
        let graph = createSimpleGraph()

        // Test path from 0 to 5
        let (path, distance) = GraphPrimitivesKernels.bidirectionalBFS(
            matrix: graph,
            source: 0,
            target: 5
        )

        XCTAssertNotNil(path)
        XCTAssertEqual(distance, 3)

        if let path = path {
            XCTAssertEqual(path.first, 0)
            XCTAssertEqual(path.last, 5)
            XCTAssertEqual(path.count - 1, Int(distance)) // Path length = distance
        }

        // Test same source and target
        let (samePath, sameDistance) = GraphPrimitivesKernels.bidirectionalBFS(
            matrix: graph,
            source: 2,
            target: 2
        )

        XCTAssertEqual(samePath, [2])
        XCTAssertEqual(sameDistance, 0)
    }

    func testDirectionOptimizingBFS() {
        let graph = createGridGraph(size: 10)

        let result = GraphPrimitivesKernels.breadthFirstSearch(
            matrix: graph,
            source: 0,
            options: .init(directionOptimizing: true)
        )

        // Verify corner distances in grid
        XCTAssertEqual(result.distances[0], 0)     // Top-left
        XCTAssertEqual(result.distances[9], 9)     // Top-right
        XCTAssertEqual(result.distances[90], 9)    // Bottom-left
        XCTAssertEqual(result.distances[99], 18)   // Bottom-right

        // Check statistics if available
        if let stats = result.statistics {
            XCTAssertGreaterThan(stats.nodesVisited, 0)
            XCTAssertGreaterThan(stats.edgesExplored, 0)
        }
    }

    // MARK: - Part 2: Shortest Path Tests

    func testDijkstraShortestPath() {
        let graph = createWeightedGraph()

        let result = GraphPrimitivesKernels.dijkstraShortestPath(
            matrix: graph,
            options: .init(source: 0, target: 5)
        )

        // Verify shortest distances
        XCTAssertEqual(result.distances[0], 0)
        XCTAssertEqual(result.distances[1], 4)
        XCTAssertEqual(result.distances[2], 9)
        XCTAssertEqual(result.distances[3], 2)
        XCTAssertEqual(result.distances[4], 5)
        XCTAssertEqual(result.distances[5], 7)

        // Reconstruct and verify path to node 5
        var path: [Int32] = []
        var current: Int32 = 5
        while current != -1 {
            path.append(current)
            current = result.parents[Int(current)]
        }
        path.reverse()

        XCTAssertEqual(path, [0, 1, 4, 5])
    }

    func testDijkstraWithMaxDistance() {
        let graph = createWeightedGraph()

        let result = GraphPrimitivesKernels.dijkstraShortestPath(
            matrix: graph,
            options: .init(source: 0, maxDistance: 5.0)
        )

        // Nodes within distance 5
        XCTAssertEqual(result.distances[0], 0)
        XCTAssertEqual(result.distances[1], 4)
        XCTAssertEqual(result.distances[3], 2)
        XCTAssertEqual(result.distances[4], 5)

        // Nodes beyond distance 5 might not be fully explored
        // but this depends on implementation details
    }

    func testAStarPathfinding() {
        let gridSize = 5
        let graph = createGridGraph(size: gridSize)

        // Manhattan distance heuristic
        let heuristic: GraphPrimitivesKernels.HeuristicFunction = { from, to in
            let fx = from % Int32(gridSize)
            let fy = from / Int32(gridSize)
            let tx = to % Int32(gridSize)
            let ty = to / Int32(gridSize)
            return Float(abs(fx - tx) + abs(fy - ty))
        }

        let result = GraphPrimitivesKernels.aStarPathfinding(
            matrix: graph,
            source: 0,
            target: 24,
            options: .init(heuristic: heuristic, admissible: true)
        )

        XCTAssertNotNil(result.path)
        XCTAssertEqual(result.distance, 8) // Manhattan distance in grid

        if let path = result.path {
            XCTAssertEqual(path.first, 0)
            XCTAssertEqual(path.last, 24)
            XCTAssertEqual(path.count - 1, 8) // Optimal path length
        }

        // A* should explore fewer nodes than uninformed search
        XCTAssertLessThan(result.nodesExpanded, 25) // Less than all nodes
    }

    func testAStarWithBeamSearch() {
        let graph = createGridGraph(size: 10)

        let heuristic: GraphPrimitivesKernels.HeuristicFunction = { from, to in
            let fx = from % 10
            let fy = from / 10
            let tx = to % 10
            let ty = to / 10
            return Float(abs(fx - tx) + abs(fy - ty))
        }

        let result = GraphPrimitivesKernels.aStarPathfinding(
            matrix: graph,
            source: 0,
            target: 99,
            options: .init(heuristic: heuristic, beamWidth: 10)
        )

        // Beam search may not find optimal path but should find a path
        XCTAssertNotNil(result.path)
        if result.path != nil {
            XCTAssertLessThanOrEqual(result.distance, 30) // Some reasonable bound
        }
    }

    func testGreedyBestFirstSearch() {
        let gridSize = 5
        let graph = createGridGraph(size: gridSize)

        let heuristic: GraphPrimitivesKernels.HeuristicFunction = { from, to in
            let fx = from % Int32(gridSize)
            let fy = from / Int32(gridSize)
            let tx = to % Int32(gridSize)
            let ty = to / Int32(gridSize)
            return Float(abs(fx - tx) + abs(fy - ty))
        }

        let (path, nodesExpanded) = GraphPrimitivesKernels.greedyBestFirstSearch(
            matrix: graph,
            source: 0,
            target: 24,
            heuristic: heuristic
        )

        XCTAssertNotNil(path)
        if let path = path {
            XCTAssertEqual(path.first, 0)
            XCTAssertEqual(path.last, 24)
            // Greedy may not find optimal path
            XCTAssertGreaterThanOrEqual(path.count - 1, 8)
        }

        // Greedy typically expands fewer nodes than A*
        XCTAssertLessThan(nodesExpanded, 25)
    }

    // MARK: - Component Tests

    func testConnectedComponents() {
        let graph = createDisconnectedGraph()

        let result = GraphPrimitivesKernels.connectedComponents(
            matrix: graph,
            directed: false
        )

        XCTAssertEqual(result.numberOfComponents, 4)

        // Verify component assignments
        XCTAssertEqual(result.componentIds[0], result.componentIds[1])
        XCTAssertEqual(result.componentIds[1], result.componentIds[2])

        XCTAssertEqual(result.componentIds[3], result.componentIds[4])
        XCTAssertEqual(result.componentIds[4], result.componentIds[5])

        XCTAssertNotEqual(result.componentIds[6], result.componentIds[0])
        XCTAssertNotEqual(result.componentIds[6], result.componentIds[3])

        XCTAssertEqual(result.componentIds[7], result.componentIds[8])
        XCTAssertEqual(result.componentIds[8], result.componentIds[9])

        // Verify component sizes
        let sizes = Array(result.componentSizes.values).sorted()
        XCTAssertEqual(sizes, [1, 3, 3, 3]) // One isolated node, three 3-node components
    }

    func testStronglyConnectedComponents() {
        let graph = createDirectedGraphWithSCCs()

        let result = GraphPrimitivesKernels.stronglyConnectedComponents(
            matrix: graph
        )

        // Should have 5 SCCs: one 3-cycle, one 2-cycle, one 2-cycle, and two single nodes
        XCTAssertEqual(result.numberOfComponents, 5)

        // Nodes 0, 1, 2 should be in same SCC
        XCTAssertEqual(result.componentIds[0], result.componentIds[1])
        XCTAssertEqual(result.componentIds[1], result.componentIds[2])

        // Nodes 3, 4 should be in same SCC
        XCTAssertEqual(result.componentIds[3], result.componentIds[4])

        // Nodes 5, 6 should be in same SCC
        XCTAssertEqual(result.componentIds[5], result.componentIds[6])

        // Nodes 7, 8 should be in different SCCs (directed edge, no cycle)
        XCTAssertNotEqual(result.componentIds[7], result.componentIds[8])
    }

    // MARK: - Edge Cases

    func testSingleNodeGraph() {
        let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = []
        let graph = try! SparseMatrix(rows: 1, cols: 1, edges: edges)

        // BFS on single node
        let bfsResult = GraphPrimitivesKernels.breadthFirstSearch(
            matrix: graph,
            source: 0
        )
        XCTAssertEqual(bfsResult.distances[0], 0)
        XCTAssertEqual(bfsResult.levels.count, 1)

        // Dijkstra on single node
        let dijkstraResult = GraphPrimitivesKernels.dijkstraShortestPath(
            matrix: graph,
            options: .init(source: 0)
        )
        XCTAssertEqual(dijkstraResult.distances[0], 0)

        // Components of single node
        let componentResult = GraphPrimitivesKernels.connectedComponents(
            matrix: graph
        )
        XCTAssertEqual(componentResult.numberOfComponents, 1)
    }

    func testEmptyGraph() {
        let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = []
        let graph = try! SparseMatrix(rows: 5, cols: 5, edges: edges)

        // BFS on disconnected nodes
        let bfsResult = GraphPrimitivesKernels.breadthFirstSearch(
            matrix: graph,
            source: 0
        )
        XCTAssertEqual(bfsResult.distances[0], 0)
        for i in 1..<5 {
            XCTAssertEqual(bfsResult.distances[i], -1) // Unreachable
        }

        // Components should all be separate
        let componentResult = GraphPrimitivesKernels.connectedComponents(
            matrix: graph
        )
        XCTAssertEqual(componentResult.numberOfComponents, 5)
    }

    func testUnreachableTarget() {
        let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
            (0, 1, 1.0), (1, 0, 1.0),  // Component 1
            (2, 3, 1.0), (3, 2, 1.0)   // Component 2
        ]
        let graph = try! SparseMatrix(rows: 4, cols: 4, edges: edges)

        // Bidirectional BFS with unreachable target
        let (path, distance) = GraphPrimitivesKernels.bidirectionalBFS(
            matrix: graph,
            source: 0,
            target: 2
        )

        XCTAssertNil(path)
        XCTAssertEqual(distance, -1)

        // A* with unreachable target
        let astarResult = GraphPrimitivesKernels.aStarPathfinding(
            matrix: graph,
            source: 0,
            target: 2,
            options: .init(heuristic: { _, _ in 1.0 })
        )

        XCTAssertNil(astarResult.path)
        XCTAssertEqual(astarResult.distance, .infinity)
    }

    // MARK: - Performance Tests

    func testLargeGraphPerformance() {
        self.measure {
            let graph = createGridGraph(size: 100) // 10,000 nodes

            _ = GraphPrimitivesKernels.breadthFirstSearch(
                matrix: graph,
                source: 0,
                options: .init(parallel: true)
            )
        }
    }

    func testDijkstraPerformance() {
        let graph = createGridGraph(size: 50) // 2,500 nodes

        self.measure {
            _ = GraphPrimitivesKernels.dijkstraShortestPath(
                matrix: graph,
                options: .init(source: 0, target: 2499, parallel: false)
            )
        }
    }

    func testParallelDijkstraPerformance() {
        let graph = createGridGraph(size: 100) // 10,000 nodes

        self.measure {
            _ = GraphPrimitivesKernels.dijkstraShortestPath(
                matrix: graph,
                options: .init(source: 0, target: 9999, parallel: true)
            )
        }
    }

    // MARK: - Correctness Validation

    func testBFSParentPointers() {
        let graph = createSimpleGraph()

        let result = GraphPrimitivesKernels.breadthFirstSearch(
            matrix: graph,
            source: 0
        )

        // Verify parent pointers form valid paths
        for target in 1..<6 {
            var path: [Int32] = []
            var current = Int32(target)
            var visited = Set<Int32>()

            while current != -1 && current != 0 {
                XCTAssertFalse(visited.contains(current), "Cycle detected in parent pointers")
                visited.insert(current)
                path.append(current)
                current = result.parents[Int(current)]
            }

            if current == 0 {
                path.append(0)
                path.reverse()

                // Path length should match distance
                XCTAssertEqual(path.count - 1, Int(result.distances[target]))
            }
        }
    }

    func testDijkstraOptimality() {
        let graph = createWeightedGraph()

        let result = GraphPrimitivesKernels.dijkstraShortestPath(
            matrix: graph,
            options: .init(source: 0)
        )

        // Manually verify some known shortest paths
        // Path 0->1: direct edge weight 4
        XCTAssertEqual(result.distances[1], 4)

        // Path 0->4: via 1 (0->1->4 = 4+1 = 5) is better than 0->3->4 (2+8=10)
        XCTAssertEqual(result.distances[4], 5)

        // Path 0->5: via 1,4 (0->1->4->5 = 4+1+2 = 7)
        XCTAssertEqual(result.distances[5], 7)
    }

    func testAStarAdmissibility() {
        let graph = createGridGraph(size: 10)

        // Admissible heuristic (Manhattan distance)
        let admissibleHeuristic: GraphPrimitivesKernels.HeuristicFunction = { from, to in
            let fx = from % 10
            let fy = from / 10
            let tx = to % 10
            let ty = to / 10
            return Float(abs(fx - tx) + abs(fy - ty))
        }

        let astarResult = GraphPrimitivesKernels.aStarPathfinding(
            matrix: graph,
            source: 0,
            target: 99,
            options: .init(heuristic: admissibleHeuristic, admissible: true)
        )

        // Compare with Dijkstra (should give same optimal distance)
        let dijkstraResult = GraphPrimitivesKernels.dijkstraShortestPath(
            matrix: graph,
            options: .init(source: 0, target: 99)
        )

        XCTAssertEqual(astarResult.distance, dijkstraResult.distances[99])
    }
}

// MARK: - Test Registration

extension GraphTraversalTests {
    static let allTests = [
        // Part 1 Tests
        ("testBreadthFirstSearch", testBreadthFirstSearch),
        ("testParallelBFS", testParallelBFS),
        ("testDepthFirstSearch", testDepthFirstSearch),
        ("testBidirectionalBFS", testBidirectionalBFS),
        ("testDirectionOptimizingBFS", testDirectionOptimizingBFS),

        // Part 2 Tests
        ("testDijkstraShortestPath", testDijkstraShortestPath),
        ("testDijkstraWithMaxDistance", testDijkstraWithMaxDistance),
        ("testAStarPathfinding", testAStarPathfinding),
        ("testAStarWithBeamSearch", testAStarWithBeamSearch),
        ("testGreedyBestFirstSearch", testGreedyBestFirstSearch),
        ("testConnectedComponents", testConnectedComponents),
        ("testStronglyConnectedComponents", testStronglyConnectedComponents),

        // Edge Cases
        ("testSingleNodeGraph", testSingleNodeGraph),
        ("testEmptyGraph", testEmptyGraph),
        ("testUnreachableTarget", testUnreachableTarget),

        // Performance
        ("testLargeGraphPerformance", testLargeGraphPerformance),
        ("testDijkstraPerformance", testDijkstraPerformance),
        ("testParallelDijkstraPerformance", testParallelDijkstraPerformance),

        // Correctness
        ("testBFSParentPointers", testBFSParentPointers),
        ("testDijkstraOptimality", testDijkstraOptimality),
        ("testAStarAdmissibility", testAStarAdmissibility)
    ]
}