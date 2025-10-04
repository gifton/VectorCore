//
//  GraphTraversalKernels.swift
//  VectorCore
//
//  High-performance graph traversal algorithms (BFS, DFS, Bidirectional Search)
//  Optimized for modern CPUs with SIMD, parallel execution, and cache-aware design
//

import Foundation
import Dispatch
import simd
import os

// MARK: - Graph Traversal Kernels

extension GraphPrimitivesKernels {

    // MARK: - BFS Types

    public struct BFSResult: Sendable {
        /// Distance from source to each node (-1 if unreachable)
        public let distances: ContiguousArray<Int32>
        /// Parent in BFS tree (-1 if none)
        public let parents: ContiguousArray<Int32>
        /// Order of visitation
        public let visitOrder: ContiguousArray<Int32>
        /// Nodes at each BFS level
        public let levels: [[Int32]]
        /// Statistics for performance monitoring
        public let statistics: TraversalStatistics?
    }

    public struct BFSOptions: Sendable {
        /// Enable parallel execution for large graphs
        public let parallel: Bool
        /// Maximum distance to explore
        public let maxDistance: Int32?
        /// Early termination predicate
        public let earlyTermination: (@Sendable (Int32) -> Bool)?
        /// Enable direction-optimizing BFS
        public let directionOptimizing: Bool
        /// Threshold for switching to bottom-up (fraction of unvisited)
        public let bottomUpThreshold: Float
        /// Enable prefetching
        public let prefetching: Bool

        public init(
            parallel: Bool = true,
            maxDistance: Int32? = nil,
            earlyTermination: (@Sendable (Int32) -> Bool)? = nil,
            directionOptimizing: Bool = true,
            bottomUpThreshold: Float = 0.1,
            prefetching: Bool = true
        ) {
            self.parallel = parallel
            self.maxDistance = maxDistance
            self.earlyTermination = earlyTermination
            self.directionOptimizing = directionOptimizing
            self.bottomUpThreshold = bottomUpThreshold
            self.prefetching = prefetching
        }
    }

    // MARK: - DFS Types

    public struct DFSResult: Sendable {
        /// Pre-order traversal
        public let visitOrder: ContiguousArray<Int32>
        /// Post-order traversal
        public let finishOrder: ContiguousArray<Int32>
        /// DFS tree parents
        public let parents: ContiguousArray<Int32>
        /// Discovery timestamps
        public let discoveryTime: ContiguousArray<Int32>
        /// Finish timestamps
        public let finishTime: ContiguousArray<Int32>
        /// Back edges (for cycle detection)
        public let backEdges: [(Int32, Int32)]
        /// Cross edges
        public let crossEdges: [(Int32, Int32)]
        /// Statistics
        public let statistics: TraversalStatistics?
    }

    public struct DFSOptions: Sendable {
        /// Visit all components
        public let visitAll: Bool
        /// Detect cycles via back edges
        public let detectCycles: Bool
        /// Track edge classifications
        public let classifyEdges: Bool
        /// Enable prefetching
        public let prefetching: Bool

        public init(
            visitAll: Bool = false,
            detectCycles: Bool = false,
            classifyEdges: Bool = false,
            prefetching: Bool = true
        ) {
            self.visitAll = visitAll
            self.detectCycles = detectCycles
            self.classifyEdges = classifyEdges
            self.prefetching = prefetching
        }
    }

    /// Mutable state for DFS traversal (reduces parameter count in helper functions)
    private struct DFSState {
        var visited: ContiguousArray<Bool>
        var visitOrder: ContiguousArray<Int32>
        var finishOrder: ContiguousArray<Int32>
        var parents: ContiguousArray<Int32>
        var discoveryTime: ContiguousArray<Int32>
        var finishTime: ContiguousArray<Int32>
        var time: Int32
        var backEdges: [(Int32, Int32)]
        var crossEdges: [(Int32, Int32)]
    }

    // MARK: - Performance Statistics

    public struct TraversalStatistics: Sendable {
        public let nodesVisited: Int
        public let edgesExplored: Int
        public let elapsedTime: TimeInterval
        public let parallelSpeedup: Float?
        public let cacheHits: Int?
    }

    // MARK: - Main BFS Implementation

    public static func breadthFirstSearch(
        matrix: SparseMatrix,
        source: Int32,
        options: BFSOptions = BFSOptions()
    ) async -> BFSResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        let nodeCount = Int32(matrix.rows)

        // Validate source
        guard source >= 0 && source < nodeCount else {
            return BFSResult(
                distances: ContiguousArray(repeating: -1, count: Int(nodeCount)),
                parents: ContiguousArray(repeating: -1, count: Int(nodeCount)),
                visitOrder: [],
                levels: [],
                statistics: nil
            )
        }

        var distances = ContiguousArray<Int32>(repeating: -1, count: Int(nodeCount))
        var parents = ContiguousArray<Int32>(repeating: -1, count: Int(nodeCount))
        var visitOrder = ContiguousArray<Int32>()
        var levels: [[Int32]] = []

        // Initialize source
        distances[Int(source)] = 0
        visitOrder.reserveCapacity(Int(nodeCount))

        // Choose implementation based on graph size and options
        let result: BFSResult

        if options.directionOptimizing && nodeCount > 1000 {
            result = directionOptimizingBFS(
                matrix: matrix,
                source: source,
                distances: &distances,
                parents: &parents,
                visitOrder: &visitOrder,
                levels: &levels,
                options: options
            )
        } else if options.parallel && nodeCount > 1000 {
            result = await parallelBFS(
                matrix: matrix,
                source: source,
                distances: &distances,
                parents: &parents,
                visitOrder: &visitOrder,
                levels: &levels,
                options: options
            )
        } else {
            result = serialBFS(
                matrix: matrix,
                source: source,
                distances: &distances,
                parents: &parents,
                visitOrder: &visitOrder,
                levels: &levels,
                options: options
            )
        }

        // Compute statistics
        let elapsedTime = CFAbsoluteTimeGetCurrent() - startTime
        let statistics = TraversalStatistics(
            nodesVisited: visitOrder.count,
            edgesExplored: computeEdgesExplored(matrix: matrix, visited: distances),
            elapsedTime: elapsedTime,
            parallelSpeedup: nil,
            cacheHits: nil
        )

        return BFSResult(
            distances: result.distances,
            parents: result.parents,
            visitOrder: result.visitOrder,
            levels: result.levels,
            statistics: statistics
        )
    }

    // MARK: - Serial BFS (Optimized)

    private static func serialBFS(
        matrix: SparseMatrix,
        source: Int32,
        distances: inout ContiguousArray<Int32>,
        parents: inout ContiguousArray<Int32>,
        visitOrder: inout ContiguousArray<Int32>,
        levels: inout [[Int32]],
        options: BFSOptions
    ) -> BFSResult {
        var currentLevel = ContiguousArray<Int32>([source])
        var nextLevel = ContiguousArray<Int32>()
        var distance: Int32 = 0

        visitOrder.append(source)

        while !currentLevel.isEmpty {
            levels.append(Array(currentLevel))
            nextLevel.removeAll(keepingCapacity: true)

            // Check max distance
            if let maxDist = options.maxDistance, distance >= maxDist {
                break
            }

            // Process current level
            for node in currentLevel {
                // Early termination check
                if let shouldTerminate = options.earlyTermination, shouldTerminate(node) {
                    return BFSResult(
                        distances: distances,
                        parents: parents,
                        visitOrder: visitOrder,
                        levels: levels,
                        statistics: nil
                    )
                }

                // Prefetch next node's adjacency list
                if options.prefetching && !currentLevel.isEmpty {
                    if let idx = currentLevel.firstIndex(of: node), idx + 1 < currentLevel.count {
                        prefetchAdjacencyList(matrix: matrix, node: currentLevel[idx + 1])
                    }
                }

                // Process neighbors
                let rowStart = Int(matrix.rowPointers[Int(node)])
                let rowEnd = Int(matrix.rowPointers[Int(node) + 1])

                for idx in rowStart..<rowEnd {
                    let neighbor = Int32(matrix.columnIndices[idx])

                    if distances[Int(neighbor)] == -1 {
                        distances[Int(neighbor)] = distance + 1
                        parents[Int(neighbor)] = node
                        visitOrder.append(neighbor)
                        nextLevel.append(neighbor)
                    }
                }
            }

            swap(&currentLevel, &nextLevel)
            distance += 1
        }

        return BFSResult(
            distances: distances,
            parents: parents,
            visitOrder: visitOrder,
            levels: levels,
            statistics: nil
        )
    }

    // MARK: - Parallel BFS (Modern Implementation)

    // Thread-safe state container for parallel BFS
    private final class ParallelBFSState: @unchecked Sendable {
        var visited: ContiguousArray<Bool>
        var distances: ContiguousArray<Int32>
        var parents: ContiguousArray<Int32>
        var visitOrder: ContiguousArray<Int32>
        var nextFrontier: ContiguousArray<Int32>

        private var visitedLock = os_unfair_lock()
        private var frontierLock = os_unfair_lock()

        init(nodeCount: Int, source: Int32) {
            self.visited = ContiguousArray<Bool>(repeating: false, count: nodeCount)
            self.distances = ContiguousArray<Int32>(repeating: -1, count: nodeCount)
            self.parents = ContiguousArray<Int32>(repeating: -1, count: nodeCount)
            self.visitOrder = ContiguousArray<Int32>()
            self.nextFrontier = ContiguousArray<Int32>()

            // Initialize source
            visited[Int(source)] = true
            distances[Int(source)] = 0
            visitOrder.append(source)
        }

        func tryVisitNode(_ neighbor: Int32, from parent: Int32, at distance: Int32) -> Bool {
            os_unfair_lock_lock(&visitedLock)
            defer { os_unfair_lock_unlock(&visitedLock) }

            if !visited[Int(neighbor)] {
                visited[Int(neighbor)] = true
                distances[Int(neighbor)] = distance
                parents[Int(neighbor)] = parent
                return true
            }
            return false
        }

        func addToNextFrontier(_ nodes: ContiguousArray<Int32>) {
            guard !nodes.isEmpty else { return }

            os_unfair_lock_lock(&frontierLock)
            defer { os_unfair_lock_unlock(&frontierLock) }

            nextFrontier.append(contentsOf: nodes)
            visitOrder.append(contentsOf: nodes)
        }
    }

    private static func parallelBFS(
        matrix: SparseMatrix,
        source: Int32,
        distances: inout ContiguousArray<Int32>,
        parents: inout ContiguousArray<Int32>,
        visitOrder: inout ContiguousArray<Int32>,
        levels: inout [[Int32]],
        options: BFSOptions
    ) async -> BFSResult {
        let nodeCount = Int32(matrix.rows)
        let numTasks = ProcessInfo.processInfo.activeProcessorCount * 2  // 2x oversubscription

        // Create thread-safe state
        let state = ParallelBFSState(nodeCount: Int(nodeCount), source: source)

        var currentFrontier = ContiguousArray<Int32>([source])
        var currentDistance: Int32 = 0

        while !currentFrontier.isEmpty {
            levels.append(Array(currentFrontier))

            // Check max distance
            if let maxDist = options.maxDistance, currentDistance >= maxDist {
                break
            }

            // Reset next frontier
            state.nextFrontier.removeAll(keepingCapacity: true)

            // Create immutable copy for concurrent access
            let frontierCopy = currentFrontier
            let frontierCount = frontierCopy.count
            let nextDistance = currentDistance + 1

            // Work-stealing approach for better load balancing
            let chunkSize = max(1, (frontierCount + numTasks - 1) / numTasks)

            await withTaskGroup(of: Void.self) { group in
                for taskId in 0..<numTasks {
                    group.addTask {
                        let startIdx = taskId * chunkSize
                        let endIdx = min(startIdx + chunkSize, frontierCount)

                        guard startIdx < endIdx else { return }

                        var localFrontier = ContiguousArray<Int32>()
                        localFrontier.reserveCapacity(endIdx - startIdx)

                        for i in startIdx..<endIdx {
                            if Task.isCancelled { return }

                            if (i - startIdx) % 100 == 0 {
                                await Task.yield()
                            }

                            let node = frontierCopy[i]

                            // Prefetch for cache optimization
                            if options.prefetching && i + 1 < endIdx {
                                prefetchAdjacencyList(matrix: matrix, node: frontierCopy[i + 1])
                            }

                            let rowStart = Int(matrix.rowPointers[Int(node)])
                            let rowEnd = Int(matrix.rowPointers[Int(node) + 1])

                            for idx in rowStart..<rowEnd {
                                let neighbor = Int32(matrix.columnIndices[idx])

                                if state.tryVisitNode(neighbor, from: node, at: nextDistance) {
                                    localFrontier.append(neighbor)
                                }
                            }
                        }

                        // Merge local results
                        state.addToNextFrontier(localFrontier)
                    }
                }

                await group.waitForAll()
            }

            currentFrontier = state.nextFrontier
            currentDistance = nextDistance
        }

        // Copy results back
        distances = state.distances
        parents = state.parents
        visitOrder = state.visitOrder

        return BFSResult(
            distances: distances,
            parents: parents,
            visitOrder: visitOrder,
            levels: levels,
            statistics: nil
        )
    }

    // MARK: - Direction-Optimizing BFS

    private static func directionOptimizingBFS(
        matrix: SparseMatrix,
        source: Int32,
        distances: inout ContiguousArray<Int32>,
        parents: inout ContiguousArray<Int32>,
        visitOrder: inout ContiguousArray<Int32>,
        levels: inout [[Int32]],
        options: BFSOptions
    ) -> BFSResult {
        let nodeCount = Int32(matrix.rows)
        var visited = ContiguousArray<Bool>(repeating: false, count: Int(nodeCount))
        visited[Int(source)] = true

        var currentFrontier = ContiguousArray<Int32>([source])
        var unvisitedCount = Int(nodeCount) - 1
        var distance: Int32 = 0

        visitOrder.append(source)

        while !currentFrontier.isEmpty {
            levels.append(Array(currentFrontier))

            // Check max distance
            if let maxDist = options.maxDistance, distance >= maxDist {
                break
            }

            // Decide direction: top-down vs bottom-up
            let frontierSize = currentFrontier.count
            let useBottomUp = Float(frontierSize) > Float(unvisitedCount) * options.bottomUpThreshold

            var nextFrontier: ContiguousArray<Int32>

            if useBottomUp {
                nextFrontier = bottomUpStep(
                    matrix: matrix,
                    currentFrontier: currentFrontier,
                    visited: &visited,
                    distances: &distances,
                    parents: &parents,
                    distance: distance + 1,
                    prefetching: options.prefetching
                )
            } else {
                nextFrontier = topDownStep(
                    matrix: matrix,
                    currentFrontier: currentFrontier,
                    visited: &visited,
                    distances: &distances,
                    parents: &parents,
                    distance: distance + 1,
                    prefetching: options.prefetching
                )
            }

            visitOrder.append(contentsOf: nextFrontier)
            unvisitedCount -= nextFrontier.count
            currentFrontier = nextFrontier
            distance += 1
        }

        return BFSResult(
            distances: distances,
            parents: parents,
            visitOrder: visitOrder,
            levels: levels,
            statistics: nil
        )
    }

    // MARK: - Top-down BFS Step

    private static func topDownStep(
        matrix: SparseMatrix,
        currentFrontier: ContiguousArray<Int32>,
        visited: inout ContiguousArray<Bool>,
        distances: inout ContiguousArray<Int32>,
        parents: inout ContiguousArray<Int32>,
        distance: Int32,
        prefetching: Bool
    ) -> ContiguousArray<Int32> {
        var nextFrontier = ContiguousArray<Int32>()

        for node in currentFrontier {
            if prefetching && !currentFrontier.isEmpty {
                let nextIdx = currentFrontier.firstIndex(of: node)! + 1
                if nextIdx < currentFrontier.count {
                    prefetchAdjacencyList(matrix: matrix, node: currentFrontier[nextIdx])
                }
            }

            let rowStart = Int(matrix.rowPointers[Int(node)])
            let rowEnd = Int(matrix.rowPointers[Int(node) + 1])

            // SIMD-friendly inner loop
            for idx in rowStart..<rowEnd {
                let neighbor = Int32(matrix.columnIndices[idx])

                if !visited[Int(neighbor)] {
                    visited[Int(neighbor)] = true
                    distances[Int(neighbor)] = distance
                    parents[Int(neighbor)] = node
                    nextFrontier.append(neighbor)
                }
            }
        }

        return nextFrontier
    }

    // MARK: - Bottom-up BFS Step

    private static func bottomUpStep(
        matrix: SparseMatrix,
        currentFrontier: ContiguousArray<Int32>,
        visited: inout ContiguousArray<Bool>,
        distances: inout ContiguousArray<Int32>,
        parents: inout ContiguousArray<Int32>,
        distance: Int32,
        prefetching: Bool
    ) -> ContiguousArray<Int32> {
        var nextFrontier = ContiguousArray<Int32>()
        let frontierSet = Set(currentFrontier)

        // Check all unvisited nodes
        for v in 0..<Int32(matrix.rows) {
            if visited[Int(v)] { continue }

            if prefetching && v + 1 < Int32(matrix.rows) {
                prefetchAdjacencyList(matrix: matrix, node: v + 1)
            }

            // Check if v has a neighbor in current frontier
            let rowStart = Int(matrix.rowPointers[Int(v)])
            let rowEnd = Int(matrix.rowPointers[Int(v) + 1])

            for idx in rowStart..<rowEnd {
                let neighbor = Int32(matrix.columnIndices[idx])

                if frontierSet.contains(neighbor) {
                    visited[Int(v)] = true
                    distances[Int(v)] = distance
                    parents[Int(v)] = neighbor
                    nextFrontier.append(v)
                    break
                }
            }
        }

        return nextFrontier
    }

    // MARK: - Main DFS Implementation

    public static func depthFirstSearch(
        matrix: SparseMatrix,
        source: Int32,
        options: DFSOptions = DFSOptions()
    ) -> DFSResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        let nodeCount = Int32(matrix.rows)

        // Validate source
        guard source >= 0 && source < nodeCount else {
            return DFSResult(
                visitOrder: [],
                finishOrder: [],
                parents: ContiguousArray(repeating: -1, count: Int(nodeCount)),
                discoveryTime: ContiguousArray(repeating: -1, count: Int(nodeCount)),
                finishTime: ContiguousArray(repeating: -1, count: Int(nodeCount)),
                backEdges: [],
                crossEdges: [],
                statistics: nil
            )
        }

        // Create DFS state
        var state = DFSState(
            visited: ContiguousArray<Bool>(repeating: false, count: Int(nodeCount)),
            visitOrder: ContiguousArray<Int32>(),
            finishOrder: ContiguousArray<Int32>(),
            parents: ContiguousArray<Int32>(repeating: -1, count: Int(nodeCount)),
            discoveryTime: ContiguousArray<Int32>(repeating: -1, count: Int(nodeCount)),
            finishTime: ContiguousArray<Int32>(repeating: -1, count: Int(nodeCount)),
            time: 0,
            backEdges: [],
            crossEdges: []
        )

        // Visit source component
        dfsVisit(
            matrix: matrix,
            startNode: source,
            state: &state,
            options: options
        )

        // Visit remaining components if requested
        if options.visitAll {
            for node in 0..<nodeCount {
                if !state.visited[Int(node)] {
                    dfsVisit(
                        matrix: matrix,
                        startNode: node,
                        state: &state,
                        options: options
                    )
                }
            }
        }

        let elapsedTime = CFAbsoluteTimeGetCurrent() - startTime
        let statistics = TraversalStatistics(
            nodesVisited: state.visitOrder.count,
            edgesExplored: computeEdgesExplored(matrix: matrix, visited: state.discoveryTime),
            elapsedTime: elapsedTime,
            parallelSpeedup: nil,
            cacheHits: nil
        )

        return DFSResult(
            visitOrder: state.visitOrder,
            finishOrder: state.finishOrder,
            parents: state.parents,
            discoveryTime: state.discoveryTime,
            finishTime: state.finishTime,
            backEdges: state.backEdges,
            crossEdges: state.crossEdges,
            statistics: statistics
        )
    }

    // MARK: - DFS Visit (Iterative, Optimized)

    private static func dfsVisit(
        matrix: SparseMatrix,
        startNode: Int32,
        state: inout DFSState,
        options: DFSOptions
    ) {
        // Stack frame for iterative DFS
        struct Frame {
            let node: Int32
            let phase: Phase
            let neighborStart: Int
            let neighborEnd: Int
            var neighborIdx: Int

            enum Phase {
                case discovery
                case exploration
                case finishing
            }
        }

        var stack = [Frame(
            node: startNode,
            phase: .discovery,
            neighborStart: 0,
            neighborEnd: 0,
            neighborIdx: 0
        )]

        while !stack.isEmpty {
            var frame = stack.removeLast()

            switch frame.phase {
            case .discovery:
                // First visit to node
                if state.visited[Int(frame.node)] {
                    continue
                }

                state.visited[Int(frame.node)] = true
                state.discoveryTime[Int(frame.node)] = state.time
                state.time += 1
                state.visitOrder.append(frame.node)

                // Setup neighbor exploration
                let rowStart = Int(matrix.rowPointers[Int(frame.node)])
                let rowEnd = Int(matrix.rowPointers[Int(frame.node) + 1])

                // Push finishing frame
                stack.append(Frame(
                    node: frame.node,
                    phase: .finishing,
                    neighborStart: 0,
                    neighborEnd: 0,
                    neighborIdx: 0
                ))

                // Push exploration frame if there are neighbors
                if rowStart < rowEnd {
                    stack.append(Frame(
                        node: frame.node,
                        phase: .exploration,
                        neighborStart: rowStart,
                        neighborEnd: rowEnd,
                        neighborIdx: rowStart
                    ))
                }

            case .exploration:
                // Explore neighbors
                if options.prefetching && frame.neighborIdx + 1 < frame.neighborEnd {
                    let nextNeighbor = Int32(matrix.columnIndices[frame.neighborIdx + 1])
                    prefetchAdjacencyList(matrix: matrix, node: nextNeighbor)
                }

                while frame.neighborIdx < frame.neighborEnd {
                    let neighbor = Int32(matrix.columnIndices[frame.neighborIdx])
                    frame.neighborIdx += 1

                    if !state.visited[Int(neighbor)] {
                        state.parents[Int(neighbor)] = frame.node

                        // Continue exploring current node after visiting neighbor
                        if frame.neighborIdx < frame.neighborEnd {
                            stack.append(frame)
                        }

                        // Visit neighbor
                        stack.append(Frame(
                            node: neighbor,
                            phase: .discovery,
                            neighborStart: 0,
                            neighborEnd: 0,
                            neighborIdx: 0
                        ))
                        break
                    } else if options.classifyEdges || options.detectCycles {
                        // Classify edge
                        if state.discoveryTime[Int(neighbor)] < state.discoveryTime[Int(frame.node)] {
                            if state.finishTime[Int(neighbor)] == -1 {
                                // Back edge (to ancestor)
                                state.backEdges.append((frame.node, neighbor))
                            } else {
                                // Cross edge
                                state.crossEdges.append((frame.node, neighbor))
                            }
                        }
                    }
                }

            case .finishing:
                // Post-order processing
                state.finishTime[Int(frame.node)] = state.time
                state.time += 1
                state.finishOrder.append(frame.node)
            }
        }
    }

    // MARK: - Bidirectional BFS

    // swiftlint:disable:next cyclomatic_complexity
    // Justification: Bidirectional BFS inherently requires high complexity:
    // 1. Simultaneous frontier expansion from source and target
    // 2. Intersection detection between forward/backward searches
    // 3. Path reconstruction from meeting point
    // 4. Directed vs undirected graph handling (reverse matrix)
    // This is a classic graph algorithm that cannot be meaningfully decomposed.
    public static func bidirectionalBFS(
        matrix: SparseMatrix,
        reverseMatrix: SparseMatrix? = nil,  // For directed graphs
        source: Int32,
        target: Int32
    ) -> (path: [Int32]?, distance: Int32) {
        if source == target {
            return ([source], 0)
        }

        let nodeCount = Int32(matrix.rows)

        // Validate inputs
        guard source >= 0 && source < nodeCount && target >= 0 && target < nodeCount else {
            return (nil, -1)
        }

        var forwardVisited = ContiguousArray<Bool>(repeating: false, count: Int(nodeCount))
        var backwardVisited = ContiguousArray<Bool>(repeating: false, count: Int(nodeCount))
        var forwardParents = ContiguousArray<Int32>(repeating: -1, count: Int(nodeCount))
        var backwardParents = ContiguousArray<Int32>(repeating: -1, count: Int(nodeCount))
        var forwardDistance = ContiguousArray<Int32>(repeating: -1, count: Int(nodeCount))
        var backwardDistance = ContiguousArray<Int32>(repeating: -1, count: Int(nodeCount))

        forwardVisited[Int(source)] = true
        forwardDistance[Int(source)] = 0
        backwardVisited[Int(target)] = true
        backwardDistance[Int(target)] = 0

        var forwardFrontier = ContiguousArray<Int32>([source])
        var backwardFrontier = ContiguousArray<Int32>([target])
        var meetingNode: Int32 = -1
        var totalPathDistance: Int32 = -1

        // Use reverse matrix for backward search if provided (directed graphs)
        let backwardMatrix = reverseMatrix ?? matrix

        while !forwardFrontier.isEmpty && !backwardFrontier.isEmpty {
            // Expand smaller frontier for efficiency
            if forwardFrontier.count <= backwardFrontier.count {
                // Expand forward frontier
                var nextFrontier = ContiguousArray<Int32>()

                for node in forwardFrontier {
                    let rowStart = Int(matrix.rowPointers[Int(node)])
                    let rowEnd = Int(matrix.rowPointers[Int(node) + 1])

                    for idx in rowStart..<rowEnd {
                        let neighbor = Int32(matrix.columnIndices[idx])

                        if backwardVisited[Int(neighbor)] {
                            // Path found! The searches meet at 'neighbor'
                            // The node 'node' from forward search connects to 'neighbor'
                            meetingNode = neighbor
                            forwardParents[Int(neighbor)] = node
                            // Total distance = distance to node + 1 (edge) + backward distance from neighbor
                            totalPathDistance = forwardDistance[Int(node)] + 1 + backwardDistance[Int(neighbor)]
                            break
                        }

                        if !forwardVisited[Int(neighbor)] {
                            forwardVisited[Int(neighbor)] = true
                            forwardDistance[Int(neighbor)] = forwardDistance[Int(node)] + 1
                            forwardParents[Int(neighbor)] = node
                            nextFrontier.append(neighbor)
                        }
                    }

                    if meetingNode != -1 { break }
                }

                if meetingNode != -1 { break }
                forwardFrontier = nextFrontier
            } else {
                // Expand backward frontier
                var nextFrontier = ContiguousArray<Int32>()

                for node in backwardFrontier {
                    let rowStart = Int(backwardMatrix.rowPointers[Int(node)])
                    let rowEnd = Int(backwardMatrix.rowPointers[Int(node) + 1])

                    for idx in rowStart..<rowEnd {
                        let neighbor = Int32(backwardMatrix.columnIndices[idx])

                        if forwardVisited[Int(neighbor)] {
                            // Path found! The searches meet at 'neighbor'
                            // The node 'node' from backward search connects to 'neighbor'
                            meetingNode = neighbor
                            backwardParents[Int(neighbor)] = node
                            // Total distance = forward distance to neighbor + 1 (edge) + backward distance from node
                            totalPathDistance = forwardDistance[Int(neighbor)] + 1 + backwardDistance[Int(node)]
                            break
                        }

                        if !backwardVisited[Int(neighbor)] {
                            backwardVisited[Int(neighbor)] = true
                            backwardDistance[Int(neighbor)] = backwardDistance[Int(node)] + 1
                            backwardParents[Int(neighbor)] = node
                            nextFrontier.append(neighbor)
                        }
                    }

                    if meetingNode != -1 { break }
                }

                if meetingNode != -1 { break }
                backwardFrontier = nextFrontier
            }
        }

        // Reconstruct path if found
        guard meetingNode != -1 else {
            return (nil, -1)
        }

        // Build path from source to meeting point
        var forwardPath = [Int32]()
        var current = meetingNode
        while current != -1 && current != source {
            forwardPath.append(current)
            current = forwardParents[Int(current)]
        }
        if current == source {
            forwardPath.append(source)
        }
        forwardPath.reverse()

        // Build path from meeting point to target (excluding meeting point since it's already in forward path)
        var backwardPath = [Int32]()
        current = backwardParents[Int(meetingNode)]
        while current != -1 && current != target {
            backwardPath.append(current)
            current = backwardParents[Int(current)]
        }
        if current == target {
            backwardPath.append(target)
        }

        // Combine paths (forward path already includes meeting point)
        var completePath = forwardPath
        completePath.append(contentsOf: backwardPath)

        return (completePath, totalPathDistance)
    }

    // MARK: - Helper Functions

    /// Prefetch adjacency list data for cache optimization
    @inline(__always)
    private static func prefetchAdjacencyList(matrix: SparseMatrix, node: Int32) {
        let rowStart = Int(matrix.rowPointers[Int(node)])
        if rowStart < matrix.columnIndices.count {
            matrix.columnIndices.withUnsafeBufferPointer { ptr in
                // Prefetch for read with high temporal locality
                #if arch(x86_64)
                _mm_prefetch(ptr.baseAddress?.advanced(by: rowStart), Int32(_MM_HINT_T0))
                #elseif arch(arm64)
                // ARM64 prefetch instruction via inline assembly would go here
                // For now, just touch the memory to trigger hardware prefetch
                _ = ptr[rowStart]
                #endif
            }
        }
    }

    /// Compute number of edges explored during traversal
    private static func computeEdgesExplored(
        matrix: SparseMatrix,
        visited: ContiguousArray<Int32>
    ) -> Int {
        var edgeCount = 0
        for i in 0..<matrix.rows {
            if visited[i] != -1 {  // Node was visited
                let rowStart = Int(matrix.rowPointers[i])
                let rowEnd = Int(matrix.rowPointers[i + 1])
                edgeCount += rowEnd - rowStart
            }
        }
        return edgeCount
    }

    // MARK: - Path Reconstruction

    /// Reconstruct path from source to target using parent pointers
    public static func reconstructPath(
        parents: ContiguousArray<Int32>,
        source: Int32,
        target: Int32
    ) -> [Int32]? {
        guard parents[Int(target)] != -1 || source == target else {
            return nil  // No path exists
        }

        var path = [Int32]()
        var current = target

        while current != -1 && current != source {
            path.append(current)
            current = parents[Int(current)]

            // Detect cycles (safety check)
            if path.count > parents.count {
                return nil
            }
        }

        if current == source {
            path.append(source)
            path.reverse()
            return path
        }

        return nil
    }

    // MARK: - Part 2: Shortest Path Algorithms & Components

    // MARK: - Dijkstra Types

    public struct DijkstraResult: Sendable {
        public let distances: ContiguousArray<Float>
        public let parents: ContiguousArray<Int32>
        public let visitOrder: ContiguousArray<Int32>
    }

    public struct DijkstraOptions: Sendable {
        public let source: Int32
        public let target: Int32?
        public let maxDistance: Float?
        public let parallel: Bool

        public init(
            source: Int32,
            target: Int32? = nil,
            maxDistance: Float? = nil,
            parallel: Bool = false
        ) {
            self.source = source
            self.target = target
            self.maxDistance = maxDistance
            self.parallel = parallel
        }
    }

    // MARK: - A* Types

    public typealias HeuristicFunction = @Sendable (_ from: Int32, _ to: Int32) -> Float

    public struct AStarResult: Sendable {
        public let path: [Int32]?
        public let distance: Float
        public let nodesExpanded: Int
        public let fScores: ContiguousArray<Float>
    }

    public struct AStarOptions: Sendable {
        public let heuristic: HeuristicFunction
        public let admissible: Bool
        public let consistentHeuristic: Bool
        public let beamWidth: Int?

        public init(
            heuristic: @escaping HeuristicFunction,
            admissible: Bool = true,
            consistentHeuristic: Bool = false,
            beamWidth: Int? = nil
        ) {
            self.heuristic = heuristic
            self.admissible = admissible
            self.consistentHeuristic = consistentHeuristic
            self.beamWidth = beamWidth
        }
    }

    // MARK: - Connected Components Types

    public struct ConnectedComponentsResult: Sendable {
        public let componentIds: ContiguousArray<Int32>
        public let componentSizes: [Int32: Int]
        public let numberOfComponents: Int
        public let largestComponent: Int32
    }

    // MARK: - Main Dijkstra Implementation

    public static func dijkstraShortestPath(
        matrix: SparseMatrix,
        options: DijkstraOptions
    ) async -> DijkstraResult {
        let nodeCount = matrix.rows
        var distances = ContiguousArray<Float>(repeating: .infinity, count: nodeCount)
        var parents = ContiguousArray<Int32>(repeating: -1, count: nodeCount)
        var visitOrder = ContiguousArray<Int32>()

        distances[Int(options.source)] = 0

        if options.parallel && nodeCount > 10000 {
            return await parallelDijkstra(
                matrix: matrix,
                distances: &distances,
                parents: &parents,
                visitOrder: &visitOrder,
                options: options
            )
        } else {
            return serialDijkstra(
                matrix: matrix,
                distances: &distances,
                parents: &parents,
                visitOrder: &visitOrder,
                options: options
            )
        }
    }

    // MARK: - Serial Dijkstra with Binary Heap

    private static func serialDijkstra(
        matrix: SparseMatrix,
        distances: inout ContiguousArray<Float>,
        parents: inout ContiguousArray<Int32>,
        visitOrder: inout ContiguousArray<Int32>,
        options: DijkstraOptions
    ) -> DijkstraResult {
        struct HeapNode: Comparable {
            let node: Int32
            let distance: Float
            static func < (lhs: HeapNode, rhs: HeapNode) -> Bool {
                lhs.distance < rhs.distance
            }
        }

        var heap = Heap<HeapNode>()
        heap.insert(HeapNode(node: options.source, distance: 0))

        while let current = heap.extractMin() {
            let u = current.node

            // If we've found a better path already, skip.
            if current.distance > distances[Int(u)] {
                continue
            }

            visitOrder.append(u)

            if let target = options.target, u == target { break }
            if let maxDist = options.maxDistance, current.distance > maxDist { break }

            let rowStart = Int(matrix.rowPointers[Int(u)])
            let rowEnd = Int(matrix.rowPointers[Int(u) + 1])

            for idx in rowStart..<rowEnd {
                let v = matrix.columnIndices[idx]
                let weight = matrix.values?[idx] ?? 1.0
                let altDistance = distances[Int(u)] + weight

                if altDistance < distances[Int(v)] {
                    distances[Int(v)] = altDistance
                    parents[Int(v)] = u
                    heap.insert(HeapNode(node: Int32(v), distance: altDistance))
                }
            }
        }
        return DijkstraResult(distances: distances, parents: parents, visitOrder: visitOrder)
    }

    // MARK: - Parallel Delta-Stepping Dijkstra

    private static func parallelDijkstra(
        matrix: SparseMatrix,
        distances: inout ContiguousArray<Float>,
        parents: inout ContiguousArray<Int32>,
        visitOrder: inout ContiguousArray<Int32>,
        options: DijkstraOptions
    ) async -> DijkstraResult {
        let nodeCount = matrix.rows
        let delta = estimateDelta(matrix: matrix)
        let numBuckets = 128

        // Thread-safe state
        let state = ParallelDijkstraState(
            nodeCount: nodeCount,
            numBuckets: numBuckets,
            delta: delta
        )

        // Initialize distances and add source to first bucket
        state.setDistance(node: Int(options.source), distance: 0)
        state.addToBucket(node: options.source, distance: 0)

        for bucketIdx in 0..<numBuckets {
            while !state.isBucketEmpty(bucketIdx) {
                let nodes = state.extractBucket(bucketIdx)
                guard !nodes.isEmpty else { continue }

                // Process nodes in parallel with TaskGroup
                let numTasks = min(ProcessInfo.processInfo.activeProcessorCount * 2, nodes.count)
                let chunkSize = max(1, (nodes.count + numTasks - 1) / numTasks)

                await withTaskGroup(of: Void.self) { group in
                    for taskId in 0..<numTasks {
                        group.addTask {
                            let start = taskId * chunkSize
                            let end = min((taskId + 1) * chunkSize, nodes.count)
                            guard start < end else { return }

                            for i in start..<end {
                                if Task.isCancelled { return }

                                if (i - start) % 50 == 0 {
                                    await Task.yield()
                                }

                                let u = nodes[i]
                                let uDist = state.getDistance(node: Int(u))

                                // Skip if we found a better path
                                if uDist > Float(bucketIdx + 1) * delta { continue }

                                let rowStart = Int(matrix.rowPointers[Int(u)])
                                let rowEnd = Int(matrix.rowPointers[Int(u) + 1])

                                for idx in rowStart..<rowEnd {
                                    let v = matrix.columnIndices[idx]
                                    let weight = matrix.values?[idx] ?? 1.0
                                    let newDist = uDist + weight

                                    // Try to update distance
                                    if state.tryUpdateDistance(node: Int(v), newDistance: newDist, parent: u) {
                                        state.addToBucket(node: Int32(v), distance: newDist)
                                    }
                                }
                            }
                        }
                    }
                    await group.waitForAll()
                }
            }
        }

        // Copy results back
        distances = state.getDistances()
        parents = state.getParents()

        return DijkstraResult(distances: distances, parents: parents, visitOrder: visitOrder)
    }

    // Thread-safe state for parallel Dijkstra
    private final class ParallelDijkstraState: @unchecked Sendable {
        private let distances: UnsafeMutablePointer<Float>
        private let parents: UnsafeMutablePointer<Int32>
        private var buckets: [Set<Int32>]
        private var bucketLocks: [os_unfair_lock_s]
        private var distanceLock: os_unfair_lock_s
        private let nodeCount: Int
        private let numBuckets: Int
        private let delta: Float

        init(nodeCount: Int, numBuckets: Int, delta: Float) {
            self.nodeCount = nodeCount
            self.numBuckets = numBuckets
            self.delta = delta

            self.distances = UnsafeMutablePointer<Float>.allocate(capacity: nodeCount)
            self.parents = UnsafeMutablePointer<Int32>.allocate(capacity: nodeCount)
            self.buckets = Array(repeating: Set<Int32>(), count: numBuckets)
            self.bucketLocks = Array(repeating: os_unfair_lock_s(), count: numBuckets)
            self.distanceLock = os_unfair_lock_s()

            // Initialize distances to infinity
            for i in 0..<nodeCount {
                distances[i] = .infinity
                parents[i] = -1
            }
        }

        deinit {
            distances.deallocate()
            parents.deallocate()
        }

        func setDistance(node: Int, distance: Float) {
            distances[node] = distance
        }

        func getDistance(node: Int) -> Float {
            return distances[node]
        }

        func tryUpdateDistance(node: Int, newDistance: Float, parent: Int32) -> Bool {
            var updated = false
            withUnsafeMutablePointer(to: &distanceLock) { lock in
                os_unfair_lock_lock(lock)
                if newDistance < distances[node] {
                    distances[node] = newDistance
                    parents[node] = parent
                    updated = true
                }
                os_unfair_lock_unlock(lock)
            }
            return updated
        }

        func addToBucket(node: Int32, distance: Float) {
            let bucketIdx = min(Int(distance / delta), numBuckets - 1)
            withUnsafeMutablePointer(to: &bucketLocks[bucketIdx]) { lock in
                os_unfair_lock_lock(lock)
                buckets[bucketIdx].insert(node)
                os_unfair_lock_unlock(lock)
            }
        }

        func isBucketEmpty(_ idx: Int) -> Bool {
            var isEmpty = false
            withUnsafeMutablePointer(to: &bucketLocks[idx]) { lock in
                os_unfair_lock_lock(lock)
                isEmpty = buckets[idx].isEmpty
                os_unfair_lock_unlock(lock)
            }
            return isEmpty
        }

        func extractBucket(_ idx: Int) -> [Int32] {
            var nodes: [Int32] = []
            withUnsafeMutablePointer(to: &bucketLocks[idx]) { lock in
                os_unfair_lock_lock(lock)
                nodes = Array(buckets[idx])
                buckets[idx].removeAll()
                os_unfair_lock_unlock(lock)
            }
            return nodes
        }

        func getDistances() -> ContiguousArray<Float> {
            return ContiguousArray(UnsafeBufferPointer(start: distances, count: nodeCount))
        }

        func getParents() -> ContiguousArray<Int32> {
            return ContiguousArray(UnsafeBufferPointer(start: parents, count: nodeCount))
        }
    }

    // MARK: - A* Pathfinding

    public static func aStarPathfinding(
        matrix: SparseMatrix,
        source: Int32,
        target: Int32,
        options: AStarOptions
    ) -> AStarResult {
        struct AStarNode: Comparable {
            let node: Int32
            let fScore: Float
            static func < (lhs: AStarNode, rhs: AStarNode) -> Bool {
                lhs.fScore < rhs.fScore
            }
        }

        let nodeCount = matrix.rows
        var gScores = ContiguousArray<Float>(repeating: .infinity, count: nodeCount)
        var fScores = ContiguousArray<Float>(repeating: .infinity, count: nodeCount)
        var parents = ContiguousArray<Int32>(repeating: -1, count: nodeCount)
        var closedSet = ContiguousArray<Bool>(repeating: false, count: nodeCount)
        var nodesExpanded = 0

        gScores[Int(source)] = 0
        fScores[Int(source)] = options.heuristic(source, target)

        var openSet = Heap<AStarNode>()
        openSet.insert(AStarNode(node: source, fScore: fScores[Int(source)]))

        while let current = openSet.extractMin() {
            let u = current.node

            if u == target {
                var path: [Int32] = []
                var curr = target
                while curr != -1 {
                    path.append(curr)
                    curr = parents[Int(curr)]
                }
                path.reverse()
                return AStarResult(
                    path: path,
                    distance: gScores[Int(target)],
                    nodesExpanded: nodesExpanded,
                    fScores: fScores
                )
            }

            // Skip if already processed
            if closedSet[Int(u)] {
                continue
            }
            closedSet[Int(u)] = true
            nodesExpanded += 1

            // Beam search memory management
            if let beamWidth = options.beamWidth, openSet.count > beamWidth {
                openSet.trimToSize(beamWidth)
            }

            let rowStart = Int(matrix.rowPointers[Int(u)])
            let rowEnd = Int(matrix.rowPointers[Int(u) + 1])

            for idx in rowStart..<rowEnd {
                let v = matrix.columnIndices[idx]

                if closedSet[Int(v)] {
                    continue
                }

                let weight = matrix.values?[idx] ?? 1.0
                let tentativeGScore = gScores[Int(u)] + weight

                if tentativeGScore < gScores[Int(v)] {
                    parents[Int(v)] = u
                    gScores[Int(v)] = tentativeGScore
                    fScores[Int(v)] = tentativeGScore + options.heuristic(Int32(v), target)
                    openSet.insert(AStarNode(node: Int32(v), fScore: fScores[Int(v)]))
                }
            }
        }

        return AStarResult(path: nil, distance: .infinity, nodesExpanded: nodesExpanded, fScores: fScores)
    }

    // MARK: - Connected Components

    public static func connectedComponents(
        matrix: SparseMatrix,
        directed: Bool = false
    ) -> ConnectedComponentsResult {
        let nodeCount = matrix.rows
        var componentIds = ContiguousArray<Int32>(repeating: -1, count: nodeCount)
        var componentId: Int32 = 0

        for startNode in 0..<nodeCount {
            if componentIds[startNode] == -1 {
                // BFS to mark component
                var queue = ContiguousArray<Int32>()
                queue.append(Int32(startNode))
                componentIds[startNode] = componentId
                var head = 0

                while head < queue.count {
                    let u = queue[head]
                    head += 1

                    let rowStart = Int(matrix.rowPointers[Int(u)])
                    let rowEnd = Int(matrix.rowPointers[Int(u) + 1])

                    for idx in rowStart..<rowEnd {
                        let v = matrix.columnIndices[idx]
                        if componentIds[Int(v)] == -1 {
                            componentIds[Int(v)] = componentId
                            queue.append(Int32(v))
                        }
                    }

                    // For undirected graphs, check reverse edges
                    if !directed {
                        for row in 0..<nodeCount {
                            let rowStart = Int(matrix.rowPointers[row])
                            let rowEnd = Int(matrix.rowPointers[row + 1])

                            for idx in rowStart..<rowEnd {
                                if matrix.columnIndices[idx] == u && componentIds[row] == -1 {
                                    componentIds[row] = componentId
                                    queue.append(Int32(row))
                                }
                            }
                        }
                    }
                }
                componentId += 1
            }
        }

        var componentSizes: [Int32: Int] = [:]
        for id in componentIds {
            componentSizes[id, default: 0] += 1
        }

        let largestComponent = componentSizes.max { $0.value < $1.value }?.key ?? -1

        return ConnectedComponentsResult(
            componentIds: componentIds,
            componentSizes: componentSizes,
            numberOfComponents: Int(componentId),
            largestComponent: largestComponent
        )
    }

    // MARK: - Strongly Connected Components (Tarjan)

    public static func stronglyConnectedComponents(
        matrix: SparseMatrix
    ) -> ConnectedComponentsResult {
        let nodeCount = matrix.rows
        var index: Int = 0
        var stack: [Int32] = []
        var indices = ContiguousArray<Int>(repeating: -1, count: nodeCount)
        var lowlinks = ContiguousArray<Int>(repeating: -1, count: nodeCount)
        var onStack = ContiguousArray<Bool>(repeating: false, count: nodeCount)
        var componentIds = ContiguousArray<Int32>(repeating: -1, count: nodeCount)
        var componentId: Int32 = 0

        func strongConnect(_ v: Int32) {
            indices[Int(v)] = index
            lowlinks[Int(v)] = index
            index += 1
            stack.append(v)
            onStack[Int(v)] = true

            let rowStart = Int(matrix.rowPointers[Int(v)])
            let rowEnd = Int(matrix.rowPointers[Int(v) + 1])
            for idx in rowStart..<rowEnd {
                let w = matrix.columnIndices[idx]
                if indices[Int(w)] == -1 {
                    strongConnect(Int32(w))
                    lowlinks[Int(v)] = min(lowlinks[Int(v)], lowlinks[Int(w)])
                } else if onStack[Int(w)] {
                    lowlinks[Int(v)] = min(lowlinks[Int(v)], indices[Int(w)])
                }
            }

            if lowlinks[Int(v)] == indices[Int(v)] {
                while let w = stack.popLast() {
                    onStack[Int(w)] = false
                    componentIds[Int(w)] = componentId
                    if w == v { break }
                }
                componentId += 1
            }
        }

        for v in 0..<Int32(nodeCount) {
            if indices[Int(v)] == -1 {
                strongConnect(v)
            }
        }

        var componentSizes: [Int32: Int] = [:]
        for id in componentIds {
            if id != -1 { componentSizes[id, default: 0] += 1 }
        }

        let largestComponent = componentSizes.max { $0.value < $1.value }?.key ?? -1

        return ConnectedComponentsResult(
            componentIds: componentIds,
            componentSizes: componentSizes,
            numberOfComponents: Int(componentId),
            largestComponent: largestComponent
        )
    }

    // MARK: - Greedy Best-First Search

    public static func greedyBestFirstSearch(
        matrix: SparseMatrix,
        source: Int32,
        target: Int32,
        heuristic: @escaping HeuristicFunction
    ) -> (path: [Int32]?, nodesExpanded: Int) {
        struct GreedyNode: Comparable {
            let node: Int32
            let priority: Float
            static func < (lhs: GreedyNode, rhs: GreedyNode) -> Bool {
                lhs.priority < rhs.priority
            }
        }

        let nodeCount = matrix.rows
        var parents = ContiguousArray<Int32>(repeating: -1, count: nodeCount)
        var visited = ContiguousArray<Bool>(repeating: false, count: nodeCount)
        var nodesExpanded = 0

        var openSet = Heap<GreedyNode>()
        openSet.insert(GreedyNode(node: source, priority: heuristic(source, target)))

        while let current = openSet.extractMin() {
            let u = current.node

            if u == target {
                var path: [Int32] = []
                var curr = target
                while curr != -1 {
                    path.append(curr)
                    curr = parents[Int(curr)]
                }
                path.reverse()
                return (path, nodesExpanded)
            }

            if visited[Int(u)] {
                continue
            }
            visited[Int(u)] = true
            nodesExpanded += 1

            let rowStart = Int(matrix.rowPointers[Int(u)])
            let rowEnd = Int(matrix.rowPointers[Int(u) + 1])
            for idx in rowStart..<rowEnd {
                let v = matrix.columnIndices[idx]
                if !visited[Int(v)] {
                    parents[Int(v)] = u
                    openSet.insert(GreedyNode(node: Int32(v), priority: heuristic(Int32(v), target)))
                }
            }
        }

        return (nil, nodesExpanded)
    }

    // MARK: - Helper Functions

    private static func estimateDelta(matrix: SparseMatrix) -> Float {
        guard let values = matrix.values, !values.isEmpty else { return 1.0 }
        let sampleCount = min(1000, values.count)
        let sample = Array(values.prefix(sampleCount)).sorted()
        // Use 10th percentile as delta
        let percentileIdx = max(0, sampleCount / 10)
        return sample[percentileIdx]
    }
}

// MARK: - Binary Heap Implementation

private struct Heap<T: Comparable> {
    private var elements: [T] = []

    var count: Int { elements.count }
    var isEmpty: Bool { elements.isEmpty }

    mutating func insert(_ element: T) {
        elements.append(element)
        siftUp(from: elements.count - 1)
    }

    mutating func extractMin() -> T? {
        guard !isEmpty else { return nil }
        if count == 1 {
            return elements.removeLast()
        }
        let min = elements[0]
        elements[0] = elements.removeLast()
        siftDown(from: 0)
        return min
    }

    mutating func trimToSize(_ size: Int) {
        if count > size {
            elements.sort()
            elements = Array(elements.prefix(size))
            // Re-heapify
            for i in stride(from: (count / 2) - 1, through: 0, by: -1) {
                siftDown(from: i)
            }
        }
    }

    private mutating func siftUp(from index: Int) {
        var child = index
        var parent = (child - 1) / 2
        while child > 0 && elements[child] < elements[parent] {
            elements.swapAt(child, parent)
            child = parent
            parent = (child - 1) / 2
        }
    }

    private mutating func siftDown(from index: Int) {
        var parent = index
        while true {
            let left = 2 * parent + 1
            let right = 2 * parent + 2
            var candidate = parent
            if left < count && elements[left] < elements[candidate] {
                candidate = left
            }
            if right < count && elements[right] < elements[candidate] {
                candidate = right
            }
            if candidate == parent {
                return
            }
            elements.swapAt(parent, candidate)
            parent = candidate
        }
    }
}

// MARK: - x86 Intrinsics Support

#if arch(x86_64)
import Darwin

// Prefetch hints for x86
private let _MM_HINT_T0: Int32 = 3  // Prefetch to all cache levels
private let _MM_HINT_T1: Int32 = 2  // Prefetch to L2 and L3
private let _MM_HINT_T2: Int32 = 1  // Prefetch to L3 only
private let _MM_HINT_NTA: Int32 = 0 // Non-temporal, bypass cache

@_silgen_name("_mm_prefetch")
private func _mm_prefetch(_ p: UnsafeRawPointer?, _ hint: Int32)
#endif
