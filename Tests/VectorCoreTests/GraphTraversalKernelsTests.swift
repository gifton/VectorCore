//
//  GraphTraversalKernelsTests.swift
//  VectorCore
//
//  Unit tests for graph traversal kernels (BFS, DFS, Bidirectional)
//

import Testing
import Foundation
@testable import VectorCore

@Suite("Graph Traversal Kernels Tests")
struct GraphTraversalKernelsTests {

    // MARK: - Test Helpers

    /// Create a simple test graph (undirected)
    /// Graph structure:
    ///     0 -- 1 -- 2
    ///     |    |    |
    ///     3 -- 4 -- 5
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

    /// Create a disconnected graph for component testing
    private func createDisconnectedGraph() -> SparseMatrix {
        let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
            // Component 1: 0-1-2
            (0, 1, 1.0), (1, 0, 1.0),
            (1, 2, 1.0), (2, 1, 1.0),
            // Component 2: 3-4
            (3, 4, 1.0), (4, 3, 1.0),
            // Isolated node: 5
        ]

        return try! SparseMatrix(rows: 6, cols: 6, edges: edges)
    }

    /// Create a directed graph with cycles
    private func createDirectedCyclicGraph() -> SparseMatrix {
        let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 0, 1.0), // Cycle: 0->1->2->0
            (2, 3, 1.0),
            (3, 4, 1.0),
            (4, 3, 1.0)  // Cycle: 3->4->3
        ]

        return try! SparseMatrix(rows: 5, cols: 5, edges: edges)
    }

    // MARK: - BFS Tests

    @Test("BFS: Basic traversal from source")
    func testBFSBasicTraversal() async throws {
        let graph = createSimpleGraph()

        let result = GraphPrimitivesKernels.breadthFirstSearch(
            matrix: graph,
            source: 0,
            options: .init(parallel: false)
        )

        // Check distances from node 0
        #expect(result.distances[0] == 0)  // Source
        #expect(result.distances[1] == 1)  // Direct neighbor
        #expect(result.distances[3] == 1)  // Direct neighbor
        #expect(result.distances[2] == 2)  // Two hops
        #expect(result.distances[4] == 2)  // Two hops
        #expect(result.distances[5] == 3)  // Three hops

        // Check all nodes were visited
        #expect(result.visitOrder.count == 6)

        // Check levels
        #expect(result.levels.count == 4)  // 0, {1,3}, {2,4}, {5}
        #expect(result.levels[0] == [0])
        #expect(Set(result.levels[1]) == Set([1, 3]))
        #expect(Set(result.levels[2]) == Set([2, 4]))
        #expect(result.levels[3] == [5])
    }

    @Test("BFS: Parallel execution")
    func testBFSParallel() async throws {
        let graph = createSimpleGraph()

        let parallelResult = GraphPrimitivesKernels.breadthFirstSearch(
            matrix: graph,
            source: 0,
            options: .init(parallel: true)
        )

        let serialResult = GraphPrimitivesKernels.breadthFirstSearch(
            matrix: graph,
            source: 0,
            options: .init(parallel: false)
        )

        // Results should be identical
        #expect(parallelResult.distances == serialResult.distances)
        #expect(parallelResult.parents == serialResult.parents)
        #expect(parallelResult.levels.count == serialResult.levels.count)
    }

    @Test("BFS: Max distance cutoff")
    func testBFSMaxDistance() async throws {
        let graph = createSimpleGraph()

        let result = GraphPrimitivesKernels.breadthFirstSearch(
            matrix: graph,
            source: 0,
            options: .init(maxDistance: 1)
        )

        // Only nodes at distance 0 and 1 should be visited
        #expect(result.distances[0] == 0)
        #expect(result.distances[1] == 1)
        #expect(result.distances[3] == 1)
        #expect(result.distances[2] == -1)  // Not visited
        #expect(result.distances[4] == -1)  // Not visited
        #expect(result.distances[5] == -1)  // Not visited

        #expect(result.levels.count == 2)  // Only 2 levels
    }

    @Test("BFS: Early termination")
    func testBFSEarlyTermination() async throws {
        let graph = createSimpleGraph()

        let result = GraphPrimitivesKernels.breadthFirstSearch(
            matrix: graph,
            source: 0,
            options: .init(earlyTermination: { node in node == 4 })
        )

        // Should stop when node 4 is found
        #expect(result.distances[4] == 2)
        // Node 5 might not be visited (depends on traversal order)
        // But nodes closer should be visited
        #expect(result.distances[0] == 0)
        #expect(result.distances[1] == 1)
    }

    @Test("BFS: Disconnected graph")
    func testBFSDisconnected() async throws {
        let graph = createDisconnectedGraph()

        let result = GraphPrimitivesKernels.breadthFirstSearch(
            matrix: graph,
            source: 0
        )

        // Component 1 reachable
        #expect(result.distances[0] == 0)
        #expect(result.distances[1] == 1)
        #expect(result.distances[2] == 2)

        // Component 2 and isolated node unreachable
        #expect(result.distances[3] == -1)
        #expect(result.distances[4] == -1)
        #expect(result.distances[5] == -1)
    }

    // MARK: - DFS Tests

    @Test("DFS: Basic traversal")
    func testDFSBasicTraversal() async throws {
        let graph = createSimpleGraph()

        let result = GraphPrimitivesKernels.depthFirstSearch(
            matrix: graph,
            source: 0,
            options: .init(visitAll: false)
        )

        // Check all reachable nodes were visited
        #expect(result.visitOrder.contains(0))
        #expect(result.visitOrder.contains(1))
        #expect(result.visitOrder.contains(2))
        #expect(result.visitOrder.contains(3))
        #expect(result.visitOrder.contains(4))
        #expect(result.visitOrder.contains(5))

        // Check discovery times are set
        #expect(result.discoveryTime[0] >= 0)
        #expect(result.finishTime[0] >= result.discoveryTime[0])

        // Parent of source should be -1
        #expect(result.parents[0] == -1)
    }

    @Test("DFS: Visit all components")
    func testDFSVisitAll() async throws {
        let graph = createDisconnectedGraph()

        let result = GraphPrimitivesKernels.depthFirstSearch(
            matrix: graph,
            source: 0,
            options: .init(visitAll: true)
        )

        // All nodes should be visited
        #expect(result.visitOrder.count == 6)
        #expect(Set(result.visitOrder) == Set(0..<6))

        // Check timestamps are valid
        for i in 0..<6 {
            #expect(result.discoveryTime[i] >= 0)
            #expect(result.finishTime[i] > result.discoveryTime[i])
        }
    }

    @Test("DFS: Cycle detection")
    func testDFSCycleDetection() async throws {
        let graph = createDirectedCyclicGraph()

        let result = GraphPrimitivesKernels.depthFirstSearch(
            matrix: graph,
            source: 0,
            options: .init(visitAll: true, detectCycles: true, classifyEdges: true)
        )

        // Should detect back edges (cycles)
        #expect(!result.backEdges.isEmpty)

        // The graph has cycles, so there should be back edges
        let backEdgeSet = Set(result.backEdges.map { "\($0.0)->\($0.1)" })
        #expect(backEdgeSet.contains("2->0") || backEdgeSet.contains("4->3"))
    }

    // MARK: - Bidirectional BFS Tests

    @Test("Bidirectional BFS: Path finding")
    func testBidirectionalBFS() async throws {
        let graph = createSimpleGraph()

        let (path, distance) = GraphPrimitivesKernels.bidirectionalBFS(
            matrix: graph,
            source: 0,
            target: 5
        )

        #expect(path != nil)
        #expect(distance == 3)  // Shortest path from 0 to 5 is 3

        if let path = path {
            #expect(path.first == 0)
            #expect(path.last == 5)
            #expect(path.count == 4)  // 0 -> (1 or 3) -> 4 -> 5
        }
    }

    @Test("Bidirectional BFS: Same source and target")
    func testBidirectionalBFSSameNode() async throws {
        let graph = createSimpleGraph()

        let (path, distance) = GraphPrimitivesKernels.bidirectionalBFS(
            matrix: graph,
            source: 2,
            target: 2
        )

        #expect(path == [2])
        #expect(distance == 0)
    }

    @Test("Bidirectional BFS: Unreachable target")
    func testBidirectionalBFSUnreachable() async throws {
        let graph = createDisconnectedGraph()

        let (path, distance) = GraphPrimitivesKernels.bidirectionalBFS(
            matrix: graph,
            source: 0,
            target: 4
        )

        #expect(path == nil)
        #expect(distance == -1)
    }

    // MARK: - Path Reconstruction Tests

    @Test("Path reconstruction from parent pointers")
    func testPathReconstruction() async throws {
        let graph = createSimpleGraph()

        let result = GraphPrimitivesKernels.breadthFirstSearch(
            matrix: graph,
            source: 0
        )

        // Reconstruct path from 0 to 5
        let path = GraphPrimitivesKernels.reconstructPath(
            parents: result.parents,
            source: 0,
            target: 5
        )

        #expect(path != nil)
        if let path = path {
            #expect(path.first == 0)
            #expect(path.last == 5)
            #expect(path.count == 4)  // Shortest path length + 1
        }
    }

    // MARK: - Performance Tests

    @Test("BFS: Direction-optimizing performance")
    func testDirectionOptimizingBFS() async throws {
        // Create a larger graph for performance testing
        var edges = ContiguousArray<(row: UInt32, col: UInt32, value: Float?)>()
        let n = 100

        // Create a grid-like graph
        for i in 0..<n {
            for j in 0..<n {
                let node = i * n + j
                // Connect to right neighbor
                if j < n - 1 {
                    let right = node + 1
                    edges.append((UInt32(node), UInt32(right), 1.0))
                    edges.append((UInt32(right), UInt32(node), 1.0))
                }
                // Connect to bottom neighbor
                if i < n - 1 {
                    let bottom = node + n
                    edges.append((UInt32(node), UInt32(bottom), 1.0))
                    edges.append((UInt32(bottom), UInt32(node), 1.0))
                }
            }
        }

        let graph = try! SparseMatrix(rows: n * n, cols: n * n, edges: edges)

        let result = GraphPrimitivesKernels.breadthFirstSearch(
            matrix: graph,
            source: 0,
            options: .init(directionOptimizing: true)
        )

        // Check corner node is reachable
        let cornerNode = Int32((n - 1) * n + (n - 1))
        #expect(result.distances[Int(cornerNode)] == Int32(2 * (n - 1)))

        // Check statistics if available
        if let stats = result.statistics {
            print("Direction-optimizing BFS stats:")
            print("  Nodes visited: \(stats.nodesVisited)")
            print("  Edges explored: \(stats.edgesExplored)")
            print("  Time: \(stats.elapsedTime)s")
        }
    }
}