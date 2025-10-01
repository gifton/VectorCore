//
//  GraphPropertiesKernels.swift
//  VectorCore
//
//  Graph property analysis including metrics, degree distribution, and k-core decomposition
//

import Foundation
import Accelerate

extension GraphPrimitivesKernels {

    // MARK: - Graph Properties

    public struct GraphPropertiesResult: Sendable {
        public let diameter: Int
        public let radius: Int
        public let averagePathLength: Float
        public let density: Float
        public let assortativity: Float
        public let degreeDistribution: DegreeDistribution
        public let isConnected: Bool
        public let isDirected: Bool
        public let isCyclic: Bool
        public let isBipartite: Bool
        public let executionTime: TimeInterval?
    }

    public struct DegreeDistribution: Sendable {
        public let degrees: ContiguousArray<Int>
        public let histogram: [Int: Int]
        public let mean: Float
        public let variance: Float
        public let max: Int
        public let min: Int
        public let powerLawExponent: Float?
    }

    public struct GraphPropertiesOptions: Sendable {
        public let directed: Bool
        public let parallel: Bool
        public let sampleForDistances: Bool  // Use sampling for diameter/radius computation

        public init(
            directed: Bool = false,
            parallel: Bool = true,
            sampleForDistances: Bool = true
        ) {
            self.directed = directed
            self.parallel = parallel
            self.sampleForDistances = sampleForDistances
        }
    }

    /// Computes comprehensive graph properties including structural metrics and statistics
    public static func computeGraphProperties(
        matrix: SparseMatrix,
        options: GraphPropertiesOptions = GraphPropertiesOptions()
    ) async -> GraphPropertiesResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        let n = matrix.rows

        guard n > 0 else {
            return GraphPropertiesResult(
                diameter: 0, radius: 0, averagePathLength: 0, density: 0, assortativity: 0,
                degreeDistribution: DegreeDistribution(degrees: [], histogram: [:], mean: 0, variance: 0, max: 0, min: 0, powerLawExponent: nil),
                isConnected: false, isDirected: options.directed, isCyclic: false, isBipartite: true,
                executionTime: 0
            )
        }

        // 1. Degree Distribution
        let degreeDistribution = computeDegreeDistribution(matrix: matrix, directed: options.directed)

        // 2. Graph Distances (using existing BFS from GraphTraversalKernels)
        let (diameter, radius, avgPathLength) = await computeGraphDistances(
            matrix: matrix,
            parallel: options.parallel,
            useSampling: options.sampleForDistances
        )

        // 3. Density
        let density = computeGraphDensity(matrix: matrix, directed: options.directed)

        // 4. Assortativity
        let assortativity = computeAssortativity(matrix: matrix, degrees: degreeDistribution.degrees)

        // 5. Connectivity (using existing functions from GraphPrimitivesKernels)
        let componentsResult = GraphPrimitivesKernels.connectedComponents(matrix: matrix, directed: options.directed)
        let isConnected = componentsResult.numberOfComponents == 1

        // 6. Structural properties
        let isCyclic = detectCycles(matrix: matrix, directed: options.directed)
        let isBipartite = checkBipartiteness(matrix: matrix)

        let executionTime = CFAbsoluteTimeGetCurrent() - startTime

        return GraphPropertiesResult(
            diameter: diameter,
            radius: radius,
            averagePathLength: avgPathLength,
            density: density,
            assortativity: assortativity,
            degreeDistribution: degreeDistribution,
            isConnected: isConnected,
            isDirected: options.directed,
            isCyclic: isCyclic,
            isBipartite: isBipartite,
            executionTime: executionTime
        )
    }

    // MARK: - Private Helper Functions

    private static func computeDegreeDistribution(
        matrix: SparseMatrix,
        directed: Bool
    ) -> DegreeDistribution {
        let n = matrix.rows
        var degrees = ContiguousArray<Int>(repeating: 0, count: n)

        // Calculate out-degrees
        for i in 0..<n {
            degrees[i] = Int(matrix.rowPointers[i + 1] - matrix.rowPointers[i])
        }

        // Create histogram
        var histogram: [Int: Int] = [:]
        for degree in degrees {
            histogram[degree, default: 0] += 1
        }

        // Calculate statistics using vDSP
        let mean: Float
        let variance: Float

        if n > 0 {
            let degreesFloat = ContiguousArray(degrees.map { Float($0) })

            var meanValue: Float = 0
            degreesFloat.withUnsafeBufferPointer { ptr in
                vDSP_meanv(ptr.baseAddress!, 1, &meanValue, vDSP_Length(n))
            }
            mean = meanValue

            var varianceValue: Float = 0
            degreesFloat.withUnsafeBufferPointer { ptr in
                vDSP_measqv(ptr.baseAddress!, 1, &varianceValue, vDSP_Length(n))
            }
            variance = varianceValue - mean * mean
        } else {
            mean = 0
            variance = 0
        }

        let maxDeg = degrees.max() ?? 0
        let minDeg = degrees.min() ?? 0

        // Estimate power law exponent
        let powerLawExponent = estimatePowerLawExponent(histogram: histogram)

        return DegreeDistribution(
            degrees: degrees,
            histogram: histogram,
            mean: mean,
            variance: variance,
            max: maxDeg,
            min: minDeg,
            powerLawExponent: powerLawExponent
        )
    }

    private static func computeGraphDistances(
        matrix: SparseMatrix,
        parallel: Bool,
        useSampling: Bool
    ) async -> (diameter: Int, radius: Int, avgPathLength: Float) {
        let n = matrix.rows

        // Determine sampling strategy
        let complexity = Double(n) * Double(matrix.nonZeros)
        let shouldSample = useSampling && complexity > 5e7

        let sources: [Int]
        if shouldSample {
            let sampleSize = min(max(500, Int(log(Double(n)) * 100.0)), n)
            sources = Array((0..<n).shuffled().prefix(sampleSize))
        } else {
            sources = Array(0..<n)
        }

        var diameter = 0
        var radius = Int.max
        var totalPathSum: Double = 0
        var totalPathCount = 0

        if parallel && sources.count > 10 {
            // Parallel computation with TaskGroup using return values pattern
            let numTasks = min(ProcessInfo.processInfo.activeProcessorCount * 2, sources.count)
            let chunkSize = (sources.count + numTasks - 1) / numTasks

            let results = await withTaskGroup(
                of: (maxEccentricity: Int, minEccentricity: Int, pathSum: Double, pathCount: Int).self,
                returning: [(Int, Int, Double, Int)].self
            ) { group in
                for taskId in 0..<numTasks {
                    group.addTask {
                        let start = taskId * chunkSize
                        let end = min((taskId + 1) * chunkSize, sources.count)
                        guard start < end else { return (0, Int.max, 0, 0) }

                        var taskMaxEccentricity = 0
                        var taskMinEccentricity = Int.max
                        var taskPathSum: Double = 0
                        var taskPathCount = 0

                        for idx in start..<end {
                            if Task.isCancelled { return (taskMaxEccentricity, taskMinEccentricity, taskPathSum, taskPathCount) }

                            if (idx - start) % 50 == 0 {
                                await Task.yield()
                            }

                            let source = sources[idx]
                            let bfsResult = await GraphPrimitivesKernels.breadthFirstSearch(
                                matrix: matrix,
                                source: Int32(source)
                            )

                            var localEccentricity = 0
                            var localSum: Double = 0
                            var localCount = 0

                            for dist in bfsResult.distances {
                                if dist > 0 {
                                    localEccentricity = max(localEccentricity, Int(dist))
                                    localSum += Double(dist)
                                    localCount += 1
                                }
                            }

                            if localEccentricity > 0 {
                                taskMaxEccentricity = max(taskMaxEccentricity, localEccentricity)
                                taskMinEccentricity = min(taskMinEccentricity, localEccentricity)
                            }
                            taskPathSum += localSum
                            taskPathCount += localCount
                        }

                        return (taskMaxEccentricity, taskMinEccentricity, taskPathSum, taskPathCount)
                    }
                }

                var collected: [(Int, Int, Double, Int)] = []
                for await result in group {
                    collected.append(result)
                }
                return collected
            }

            // Reduction phase - combine results from all tasks
            for (maxEcc, minEcc, pathSum, pathCount) in results {
                if maxEcc > 0 {
                    diameter = max(diameter, maxEcc)
                    radius = min(radius, minEcc)
                }
                totalPathSum += pathSum
                totalPathCount += pathCount
            }
        } else {
            // Serial computation
            for source in sources {
                let bfsResult = await GraphPrimitivesKernels.breadthFirstSearch(
                    matrix: matrix,
                    source: Int32(source)
                )

                var eccentricity = 0
                for dist in bfsResult.distances {
                    if dist > 0 {
                        eccentricity = max(eccentricity, Int(dist))
                        totalPathSum += Double(dist)
                        totalPathCount += 1
                    }
                }

                if eccentricity > 0 {
                    diameter = max(diameter, eccentricity)
                    radius = min(radius, eccentricity)
                }
            }
        }

        let avgPathLength = totalPathCount > 0 ? Float(totalPathSum / Double(totalPathCount)) : 0
        return (diameter, radius == Int.max ? 0 : radius, avgPathLength)
    }

    private static func computeGraphDensity(matrix: SparseMatrix, directed: Bool) -> Float {
        let n = Float(matrix.rows)
        let e = Float(matrix.nonZeros)

        if n <= 1 { return 0.0 }

        let potentialEdges = n * (n - 1)
        return directed ? e / potentialEdges : e / potentialEdges
    }

    private static func computeAssortativity(
        matrix: SparseMatrix,
        degrees: ContiguousArray<Int>
    ) -> Float {
        let M = Double(matrix.nonZeros)
        if M == 0 { return 0.0 }

        var sumProduct: Double = 0.0
        var sumDegrees: Double = 0.0
        var sumSquares: Double = 0.0

        for i in 0..<matrix.rows {
            let rowStart = Int(matrix.rowPointers[i])
            let rowEnd = Int(matrix.rowPointers[i + 1])
            let di = Double(degrees[i])

            for idx in rowStart..<rowEnd {
                let j = Int(matrix.columnIndices[idx])
                let dj = Double(degrees[j])

                sumProduct += di * dj
                sumDegrees += di + dj
                sumSquares += di * di + dj * dj
            }
        }

        let meanTermBase = sumDegrees / (2.0 * M)
        let meanTermSq = meanTermBase * meanTermBase

        let numerator = (sumProduct / M) - meanTermSq
        let denominator = (sumSquares / (2.0 * M)) - meanTermSq

        if abs(denominator) < 1e-9 {
            return abs(numerator) < 1e-9 ? 1.0 : 0.0
        }

        return Float(numerator / denominator)
    }

    private static func detectCycles(matrix: SparseMatrix, directed: Bool) -> Bool {
        if directed {
            // Use DFS from GraphPrimitivesKernels for cycle detection
            // Start from node 0 and let visitAll handle the rest
            let dfsResult = GraphPrimitivesKernels.depthFirstSearch(
                matrix: matrix,
                source: 0,
                options: GraphPrimitivesKernels.DFSOptions(visitAll: true, detectCycles: true)
            )
            // Check for back edges which indicate cycles
            return !dfsResult.backEdges.isEmpty
        } else {
            // Undirected cycle detection
            let n = matrix.rows
            var visited = ContiguousArray<Bool>(repeating: false, count: n)

            func dfsUndirected(u: Int, parent: Int) -> Bool {
                visited[u] = true

                let rowStart = Int(matrix.rowPointers[u])
                let rowEnd = Int(matrix.rowPointers[u + 1])

                for idx in rowStart..<rowEnd {
                    let v = Int(matrix.columnIndices[idx])

                    if v == parent { continue }

                    if visited[v] || dfsUndirected(u: v, parent: u) {
                        return true
                    }
                }
                return false
            }

            for i in 0..<n {
                if !visited[i] && dfsUndirected(u: i, parent: -1) {
                    return true
                }
            }
            return false
        }
    }

    private static func checkBipartiteness(matrix: SparseMatrix) -> Bool {
        let n = matrix.rows
        var colors = ContiguousArray<Int>(repeating: -1, count: n)

        for start in 0..<n {
            if colors[start] == -1 {
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
                            return false
                        }
                    }
                }
            }
        }
        return true
    }

    private static func estimatePowerLawExponent(histogram: [Int: Int]) -> Float? {
        guard let kMin = histogram.keys.filter({ $0 > 0 }).min(), kMin > 0 else { return nil }

        var sumLog: Double = 0.0
        var count = 0
        let kMinFloat = Double(kMin)

        for (degree, freq) in histogram where degree >= kMin {
            let ratio = Double(degree) / kMinFloat
            sumLog += Double(freq) * log(ratio)
            count += freq
        }

        if count > 1 && sumLog > 0 {
            let alpha = 1.0 + Double(count) / sumLog
            return Float(alpha)
        }

        return nil
    }

    // MARK: - K-Core Decomposition

    public struct KCoreDecomposition: Sendable {
        public let coreNumbers: ContiguousArray<Int>
        public let maxCore: Int
        public let coreDistribution: [Int: Int]
        public let executionTime: TimeInterval?
    }

    /// Computes k-core decomposition using the Matula-Beck algorithm in O(V+E) time
    public static func kCoreDecomposition(matrix: SparseMatrix) -> KCoreDecomposition {
        let startTime = CFAbsoluteTimeGetCurrent()
        let n = matrix.rows

        // Initialize degrees
        var degrees = ContiguousArray<Int>(repeating: 0, count: n)
        var maxDegree = 0

        for i in 0..<n {
            let degree = Int(matrix.rowPointers[i + 1] - matrix.rowPointers[i])
            degrees[i] = degree
            maxDegree = max(maxDegree, degree)
        }

        // Bin sort initialization
        var bin = ContiguousArray<Int>(repeating: 0, count: maxDegree + 1)
        for degree in degrees {
            bin[degree] += 1
        }

        // Calculate starting positions
        var start = 0
        for i in 0...maxDegree {
            let count = bin[i]
            bin[i] = start
            start += count
        }

        // Place nodes into sorted array
        var vert = ContiguousArray<Int>(repeating: 0, count: n)
        var pos = ContiguousArray<Int>(repeating: 0, count: n)

        for i in 0..<n {
            let degree = degrees[i]
            let position = bin[degree]
            pos[i] = position
            vert[position] = i
            bin[degree] += 1
        }

        // Restore bin starting positions
        for i in stride(from: maxDegree, to: 0, by: -1) {
            bin[i] = bin[i-1]
        }
        bin[0] = 0

        // Peeling process
        for i in 0..<n {
            let v = vert[i]
            let degreeV = degrees[v]

            let rowStart = Int(matrix.rowPointers[v])
            let rowEnd = Int(matrix.rowPointers[v + 1])

            for idx in rowStart..<rowEnd {
                let u = Int(matrix.columnIndices[idx])

                if degrees[u] > degreeV {
                    let degreeU = degrees[u]
                    let posU = pos[u]

                    let posW = bin[degreeU]
                    let w = vert[posW]

                    if u != w {
                        pos[u] = posW
                        pos[w] = posU
                        vert[posU] = w
                        vert[posW] = u
                    }

                    bin[degreeU] += 1
                    degrees[u] -= 1
                }
            }
        }

        // Compute distribution
        var coreDistribution: [Int: Int] = [:]
        for core in degrees {
            coreDistribution[core, default: 0] += 1
        }

        let executionTime = CFAbsoluteTimeGetCurrent() - startTime

        return KCoreDecomposition(
            coreNumbers: degrees,
            maxCore: degrees.max() ?? 0,
            coreDistribution: coreDistribution,
            executionTime: executionTime
        )
    }
}
