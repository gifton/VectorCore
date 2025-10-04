//
//  GraphAlgorithmsKernels.swift
//  VectorCore
//
//  Advanced graph algorithms including motif detection, graph coloring, and structural analysis
//

import Foundation

extension GraphPrimitivesKernels {

    // MARK: - Motif Detection

    public struct MotifResult: Sendable {
        public let motifCounts: [MotifType: Int]
        public let executionTime: TimeInterval?
    }

    public enum MotifType: Hashable, Sendable {
        case triangle
        case square
        case star(Int)  // star(k) means k-star
    }

    public struct MotifDetectionOptions: Sendable {
        public let motifSizes: [Int]
        public let parallel: Bool
        public let countInducedOnly: Bool  // If true, count only induced subgraphs

        public init(
            motifSizes: [Int] = [3, 4],
            parallel: Bool = true,
            countInducedOnly: Bool = false
        ) {
            self.motifSizes = motifSizes
            self.parallel = parallel
            self.countInducedOnly = countInducedOnly
        }
    }

    /// Detects and counts graph motifs (small subgraph patterns)
    public static func detectMotifs(
        matrix: SparseMatrix,
        options: MotifDetectionOptions = MotifDetectionOptions()
    ) async -> MotifResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        var motifCounts: [MotifType: Int] = [:]

        // Count 3-node motifs
        if options.motifSizes.contains(3) {
            let triangleCount = await countTriangles(matrix: matrix, parallel: options.parallel)
            motifCounts[.triangle] = triangleCount
        }

        // Count 4-node motifs
        if options.motifSizes.contains(4) {
            let (squares, threeStars) = count4NodeMotifs(matrix: matrix, parallel: options.parallel)
            motifCounts[.square] = squares
            motifCounts[.star(3)] = threeStars
        }

        let executionTime = CFAbsoluteTimeGetCurrent() - startTime

        return MotifResult(
            motifCounts: motifCounts,
            executionTime: executionTime
        )
    }

    /// Efficient triangle counting using node iterator with intersection
    // swiftlint:disable:next cyclomatic_complexity
    // Justification: Triangle counting requires complex logic for:
    // 1. Parallel vs sequential execution paths
    // 2. Task group coordination and buffer management
    // 3. Neighbor set intersection algorithms
    // 4. Three-way edge existence checks (i-j, j-k, i-k)
    // This is a fundamental graph mining operation with unavoidable branching.
    private static func countTriangles(matrix: SparseMatrix, parallel: Bool) async -> Int {
        let n = matrix.rows

        if parallel && n > 100 {
            // Parallel triangle counting with Sendable buffer wrapper
            let bufferWrapper = SendableMutableBufferWrapper<Int>(capacity: n)
            bufferWrapper.bufferPointer.initialize(repeating: 0)

            let numTasks = min(ProcessInfo.processInfo.activeProcessorCount * 2, n)
            let chunkSize = (n + numTasks - 1) / numTasks

            await withTaskGroup(of: Void.self) { group in
                for taskId in 0..<numTasks {
                    group.addTask {
                        let start = taskId * chunkSize
                        let end = min((taskId + 1) * chunkSize, n)
                        guard start < end else { return }

                        for i in start..<end {
                            if Task.isCancelled { return }

                            if (i - start) % 200 == 0 {
                                await Task.yield()
                            }

                            var localCount = 0
                            let rowStartI = Int(matrix.rowPointers[i])
                            let rowEndI = Int(matrix.rowPointers[i + 1])

                            for idx1 in rowStartI..<rowEndI {
                                let j = Int(matrix.columnIndices[idx1])
                                if j <= i { continue }  // Avoid counting same triangle multiple times

                                let rowStartJ = Int(matrix.rowPointers[j])
                                let rowEndJ = Int(matrix.rowPointers[j + 1])

                                // Two-pointer intersection for common neighbors
                                var ptrI = rowStartI
                                var ptrJ = rowStartJ

                                while ptrI < rowEndI && ptrJ < rowEndJ {
                                    let kI = Int(matrix.columnIndices[ptrI])
                                    let kJ = Int(matrix.columnIndices[ptrJ])

                                    if kI == kJ {
                                        if kI > j {  // Ensure i < j < k to count each triangle once
                                            localCount += 1
                                        }
                                        ptrI += 1
                                        ptrJ += 1
                                    } else if kI < kJ {
                                        ptrI += 1
                                    } else {
                                        ptrJ += 1
                                    }
                                }
                            }
                            bufferWrapper.bufferPointer[i] = localCount
                        }
                    }
                }

                await group.waitForAll()
            }

            return bufferWrapper.bufferPointer.reduce(0, +)
        } else {
            // Serial triangle counting
            var triangleCount = 0

            for i in 0..<n {
                let rowStartI = Int(matrix.rowPointers[i])
                let rowEndI = Int(matrix.rowPointers[i + 1])

                for idx1 in rowStartI..<rowEndI {
                    let j = Int(matrix.columnIndices[idx1])
                    if j <= i { continue }

                    let rowStartJ = Int(matrix.rowPointers[j])
                    let rowEndJ = Int(matrix.rowPointers[j + 1])

                    // Two-pointer intersection
                    var ptrI = rowStartI
                    var ptrJ = rowStartJ

                    while ptrI < rowEndI && ptrJ < rowEndJ {
                        let kI = Int(matrix.columnIndices[ptrI])
                        let kJ = Int(matrix.columnIndices[ptrJ])

                        if kI == kJ {
                            if kI > j {
                                triangleCount += 1
                            }
                            ptrI += 1
                            ptrJ += 1
                        } else if kI < kJ {
                            ptrI += 1
                        } else {
                            ptrJ += 1
                        }
                    }
                }
            }

            return triangleCount
        }
    }

    /// Count 4-node motifs (squares and 3-stars)
    private static func count4NodeMotifs(matrix: SparseMatrix, parallel: Bool) -> (squares: Int, threeStars: Int) {
        let n = matrix.rows

        // Count 3-stars (simple degree-based calculation)
        var threeStars = 0
        for i in 0..<n {
            let degree = Int(matrix.rowPointers[i + 1] - matrix.rowPointers[i])
            if degree >= 3 {
                // Choose 3 from degree: C(degree, 3)
                threeStars += degree * (degree - 1) * (degree - 2) / 6
            }
        }

        // Count squares (4-cycles) using path-of-length-2 counting
        var squares = 0

        for i in 0..<n {
            var path2Counts: [Int: Int] = [:]

            let rowStartI = Int(matrix.rowPointers[i])
            let rowEndI = Int(matrix.rowPointers[i + 1])

            // Find all paths of length 2 from i
            for idx1 in rowStartI..<rowEndI {
                let j = Int(matrix.columnIndices[idx1])

                let rowStartJ = Int(matrix.rowPointers[j])
                let rowEndJ = Int(matrix.rowPointers[j + 1])

                for idx2 in rowStartJ..<rowEndJ {
                    let k = Int(matrix.columnIndices[idx2])

                    if k > i && k != i {  // Avoid self-loops and ensure ordering
                        path2Counts[k, default: 0] += 1
                    }
                }
            }

            // Count squares: if there are C paths from i to k, they form C(C-1)/2 squares
            for (_, count) in path2Counts {
                if count >= 2 {
                    squares += count * (count - 1) / 2
                }
            }
        }

        return (squares, threeStars)
    }

    // MARK: - Graph Coloring

    public struct GraphColoringResult: Sendable {
        public let colors: ContiguousArray<Int>
        public let chromaticNumber: Int
        public let isOptimal: Bool
        public let executionTime: TimeInterval?
    }

    public enum ColoringAlgorithm: Sendable {
        case greedy
        case welshPowell
        case dsatur
    }

    /// Performs graph coloring using the specified algorithm
    public static func graphColoring(
        matrix: SparseMatrix,
        algorithm: ColoringAlgorithm = .welshPowell
    ) -> GraphColoringResult {
        let startTime = CFAbsoluteTimeGetCurrent()

        let result: (colors: ContiguousArray<Int>, chromaticNumber: Int)

        switch algorithm {
        case .greedy:
            result = greedyColoring(matrix: matrix, order: nil)
        case .welshPowell:
            result = welshPowellColoring(matrix: matrix)
        case .dsatur:
            result = dsaturColoring(matrix: matrix)
        }

        let executionTime = CFAbsoluteTimeGetCurrent() - startTime

        return GraphColoringResult(
            colors: result.colors,
            chromaticNumber: result.chromaticNumber,
            isOptimal: false,  // These are all heuristic algorithms
            executionTime: executionTime
        )
    }

    /// Greedy coloring with optional node ordering
    private static func greedyColoring(
        matrix: SparseMatrix,
        order: [Int]?
    ) -> (colors: ContiguousArray<Int>, chromaticNumber: Int) {
        let n = matrix.rows
        var colors = ContiguousArray<Int>(repeating: -1, count: n)
        let nodeOrder = order ?? Array(0..<n)
        var maxColor = -1

        for i in nodeOrder {
            // Find colors used by neighbors
            var usedColors = Set<Int>()

            let rowStart = Int(matrix.rowPointers[i])
            let rowEnd = Int(matrix.rowPointers[i + 1])

            for idx in rowStart..<rowEnd {
                let neighbor = Int(matrix.columnIndices[idx])
                if colors[neighbor] != -1 {
                    usedColors.insert(colors[neighbor])
                }
            }

            // Find first available color
            var color = 0
            while usedColors.contains(color) {
                color += 1
            }

            colors[i] = color
            maxColor = max(maxColor, color)
        }

        return (colors, maxColor + 1)
    }

    /// Welsh-Powell coloring (largest degree first)
    private static func welshPowellColoring(matrix: SparseMatrix) -> (colors: ContiguousArray<Int>, chromaticNumber: Int) {
        let n = matrix.rows

        // Calculate degrees and sort nodes
        var nodesByDegree: [(node: Int, degree: Int)] = []
        for i in 0..<n {
            let degree = Int(matrix.rowPointers[i + 1] - matrix.rowPointers[i])
            nodesByDegree.append((i, degree))
        }

        // Sort by degree (descending)
        nodesByDegree.sort { $0.degree > $1.degree }
        let order = nodesByDegree.map { $0.node }

        return greedyColoring(matrix: matrix, order: order)
    }

    /// DSATUR coloring (degree of saturation)
    private static func dsaturColoring(matrix: SparseMatrix) -> (colors: ContiguousArray<Int>, chromaticNumber: Int) {
        let n = matrix.rows
        var colors = ContiguousArray<Int>(repeating: -1, count: n)
        var saturation = ContiguousArray<Int>(repeating: 0, count: n)
        var adjColors = Array(repeating: Set<Int>(), count: n)

        // Calculate degrees for tie-breaking
        var degrees = ContiguousArray<Int>(repeating: 0, count: n)
        for i in 0..<n {
            degrees[i] = Int(matrix.rowPointers[i + 1] - matrix.rowPointers[i])
        }

        // Find next node to color (max saturation, tie-break by degree)
        func findNextNode() -> Int? {
            var maxSat = -1
            var nextNode: Int?
            var maxDeg = -1

            for i in 0..<n {
                if colors[i] == -1 {
                    let sat = saturation[i]
                    let deg = degrees[i]

                    if sat > maxSat || (sat == maxSat && deg > maxDeg) {
                        maxSat = sat
                        maxDeg = deg
                        nextNode = i
                    }
                }
            }
            return nextNode
        }

        var maxColor = -1

        // Color nodes one by one
        while let node = findNextNode() {
            // Find first available color
            let usedColors = adjColors[node]
            var color = 0
            while usedColors.contains(color) {
                color += 1
            }

            colors[node] = color
            maxColor = max(maxColor, color)

            // Update saturation of neighbors
            let rowStart = Int(matrix.rowPointers[node])
            let rowEnd = Int(matrix.rowPointers[node + 1])

            for idx in rowStart..<rowEnd {
                let neighbor = Int(matrix.columnIndices[idx])
                if colors[neighbor] == -1 {
                    if !adjColors[neighbor].contains(color) {
                        adjColors[neighbor].insert(color)
                        saturation[neighbor] += 1
                    }
                }
            }
        }

        return (colors, maxColor + 1)
    }

    // MARK: - Structural Analysis

    /// Check if a graph is bipartite (2-colorable)
    public static func isBipartite(matrix: SparseMatrix) -> Bool {
        let n = matrix.rows
        var colors = ContiguousArray<Int>(repeating: -1, count: n)

        for start in 0..<n {
            if colors[start] == -1 {
                // BFS coloring
                var queue = [start]
                colors[start] = 0
                var head = 0

                while head < queue.count {
                    let u = queue[head]
                    head += 1
                    let currentColor = colors[u]

                    let rowStart = Int(matrix.rowPointers[u])
                    let rowEnd = Int(matrix.rowPointers[u + 1])

                    for idx in rowStart..<rowEnd {
                        let v = Int(matrix.columnIndices[idx])

                        if colors[v] == -1 {
                            colors[v] = 1 - currentColor
                            queue.append(v)
                        } else if colors[v] == currentColor {
                            return false  // Odd cycle found
                        }
                    }
                }
            }
        }
        return true
    }

    /// Check if a graph contains cycles
    public static func hasCycle(matrix: SparseMatrix, directed: Bool) -> Bool {
        if directed {
            // Use DFS for directed cycle detection
            // Start from node 0 and let visitAll handle the rest
            let dfsResult = GraphPrimitivesKernels.depthFirstSearch(
                matrix: matrix,
                source: 0,
                options: GraphPrimitivesKernels.DFSOptions(visitAll: true, detectCycles: true)
            )
            // Check for back edges which indicate cycles
            return !dfsResult.backEdges.isEmpty
        } else {
            // Undirected cycle detection using DFS with parent tracking
            let n = matrix.rows
            var visited = ContiguousArray<Bool>(repeating: false, count: n)

            func dfsCheck(u: Int, parent: Int) -> Bool {
                visited[u] = true

                let rowStart = Int(matrix.rowPointers[u])
                let rowEnd = Int(matrix.rowPointers[u + 1])

                for idx in rowStart..<rowEnd {
                    let v = Int(matrix.columnIndices[idx])

                    if v == parent { continue }  // Skip parent edge

                    if visited[v] {
                        return true  // Found cycle
                    }

                    if dfsCheck(u: v, parent: u) {
                        return true
                    }
                }
                return false
            }

            // Check each component
            for i in 0..<n {
                if !visited[i] {
                    if dfsCheck(u: i, parent: -1) {
                        return true
                    }
                }
            }
            return false
        }
    }

    /// Find the girth (length of shortest cycle) in a graph
    public static func findGirth(matrix: SparseMatrix) -> Int? {
        let n = matrix.rows
        var minCycleLength = Int.max

        // BFS from each vertex to find shortest cycle containing it
        for start in 0..<n {
            var distances = ContiguousArray<Int>(repeating: -1, count: n)
            var queue = [start]
            distances[start] = 0
            var head = 0

            while head < queue.count {
                let u = queue[head]
                head += 1

                let rowStart = Int(matrix.rowPointers[u])
                let rowEnd = Int(matrix.rowPointers[u + 1])

                for idx in rowStart..<rowEnd {
                    let v = Int(matrix.columnIndices[idx])

                    if distances[v] == -1 {
                        distances[v] = distances[u] + 1
                        queue.append(v)
                    } else if distances[v] >= distances[u] {
                        // Found a cycle
                        let cycleLength = distances[u] + distances[v] + 1
                        minCycleLength = min(minCycleLength, cycleLength)
                    }
                }
            }
        }

        return minCycleLength == Int.max ? nil : minCycleLength
    }
}
