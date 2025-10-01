//
//  GraphAnalysisKernelsImpl.swift
//  VectorCore
//
//  Implementation of Graph Analysis Kernels - Part 1
//  PageRank, Centrality Measures, and Community Detection
//

import Foundation
import Accelerate

// MARK: - Graph Analysis Kernels

extension GraphPrimitivesKernels {

    // MARK: - PageRank Types

    public struct PageRankOptions: Sendable {
        public let dampingFactor: Float
        public let tolerance: Float
        public let maxIterations: Int
        public let personalized: [Int32: Float]?
        public let parallel: Bool

        public init(
            dampingFactor: Float = 0.85,
            tolerance: Float = 1e-6,
            maxIterations: Int = 100,
            personalized: [Int32: Float]? = nil,
            parallel: Bool = true
        ) {
            self.dampingFactor = dampingFactor
            self.tolerance = tolerance
            self.maxIterations = maxIterations
            self.personalized = personalized
            self.parallel = parallel
        }
    }

    public struct PageRankResult {
        public let scores: ContiguousArray<Float>
        public let iterations: Int
        public let converged: Bool
        public let residual: Float
    }

    // MARK: - PageRank Implementation

    public static func pageRank(
        matrix: SparseMatrix,
        options: PageRankOptions = PageRankOptions()
    ) async -> PageRankResult {
        let n = matrix.rows
        let damping = options.dampingFactor
        let tolerance = options.tolerance

        // Initialize PageRank scores uniformly
        var scores = ContiguousArray<Float>(repeating: 1.0 / Float(n), count: n)
        var newScores = ContiguousArray<Float>(repeating: 0.0, count: n)

        // Compute out-degree for each node
        let outDegrees = computeOutDegrees(matrix: matrix)

        // Personalization vector (uniform or custom)
        let personalization: ContiguousArray<Float>
        if let personal = options.personalized {
            var tempPersonalization = ContiguousArray<Float>(repeating: 0.0, count: n)
            var sum: Float = 0.0
            for (node, weight) in personal {
                if Int(node) < n && Int(node) >= 0 {
                    tempPersonalization[Int(node)] = weight
                    sum += weight
                }
            }
            // Normalize
            if sum > 0 {
                var invSum = 1.0 / sum
                tempPersonalization.withUnsafeMutableBufferPointer { ptr in
                    vDSP_vsmul(ptr.baseAddress!, 1, &invSum, ptr.baseAddress!, 1, vDSP_Length(n))
                }
            } else {
                tempPersonalization = ContiguousArray<Float>(repeating: 1.0 / Float(n), count: n)
            }
            personalization = tempPersonalization
        } else {
            personalization = ContiguousArray<Float>(repeating: 1.0 / Float(n), count: n)
        }

        var iteration = 0
        var converged = false
        var residual: Float = .infinity

        while iteration < options.maxIterations && !converged {
            if options.parallel && n > 1000 {
                await parallelPageRankIteration(
                    matrix: matrix,
                    scores: scores,
                    newScores: &newScores,
                    outDegrees: outDegrees,
                    damping: damping,
                    personalization: personalization
                )
            } else {
                serialPageRankIteration(
                    matrix: matrix,
                    scores: scores,
                    newScores: &newScores,
                    outDegrees: outDegrees,
                    damping: damping,
                    personalization: personalization
                )
            }

            // Check convergence
            residual = computeL1Norm(scores, newScores)
            converged = residual < tolerance

            // Swap buffers
            swap(&scores, &newScores)
            iteration += 1
        }

        return PageRankResult(
            scores: scores,
            iterations: iteration,
            converged: converged,
            residual: residual
        )
    }

    // MARK: - Serial PageRank Iteration

    private static func serialPageRankIteration(
        matrix: SparseMatrix,
        scores: ContiguousArray<Float>,
        newScores: inout ContiguousArray<Float>,
        outDegrees: ContiguousArray<Int>,
        damping: Float,
        personalization: ContiguousArray<Float>
    ) {
        let n = matrix.rows

        // Reset new scores
        for i in 0..<n {
            newScores[i] = (1.0 - damping) * personalization[i]
        }

        // Distribute PageRank scores
        for i in 0..<n {
            if outDegrees[i] > 0 {
                let contribution = damping * scores[i] / Float(outDegrees[i])

                let rowStart = Int(matrix.rowPointers[i])
                let rowEnd = Int(matrix.rowPointers[i + 1])

                for idx in rowStart..<rowEnd {
                    let j = Int(matrix.columnIndices[idx])
                    newScores[j] += contribution
                }
            } else {
                // Handle dangling nodes (distribute uniformly)
                let contribution = damping * scores[i] / Float(n)
                for j in 0..<n {
                    newScores[j] += contribution
                }
            }
        }
    }

    // MARK: - Parallel PageRank

    private static func parallelPageRankIteration(
        matrix: SparseMatrix,
        scores: ContiguousArray<Float>,
        newScores: inout ContiguousArray<Float>,
        outDegrees: ContiguousArray<Int>,
        damping: Float,
        personalization: ContiguousArray<Float>
    ) async {
        let n = matrix.rows

        // Initialize new scores
        for i in 0..<n {
            newScores[i] = (1.0 - damping) * personalization[i]
        }

        // Use Sendable buffer wrapper for thread-safe concurrent access
        let bufferWrapper = SendableBufferWrapper<Float>(capacity: n)
        newScores.withUnsafeBufferPointer { ptr in
            bufferWrapper.pointer.initialize(from: ptr.baseAddress!, count: n)
        }

        defer {
            // Copy back to newScores
            for i in 0..<n {
                newScores[i] = bufferWrapper.pointer[i]
            }
        }

        // Use striped locking for thread-safe updates
        let stripeCount = 128
        let locks = (0..<stripeCount).map { _ in NSLock() }

        // Use 2x oversubscription for better load balancing
        let numProcessors = ProcessInfo.processInfo.activeProcessorCount
        let numTasks = numProcessors * 2
        let chunkSize = (n + numTasks - 1) / numTasks

        await withTaskGroup(of: Void.self) { group in
            for taskId in 0..<numTasks {
                group.addTask {
                    let start = taskId * chunkSize
                    let end = min((taskId + 1) * chunkSize, n)
                    guard start < n else { return }

                    for i in start..<end {
                        // Check for cancellation at chunk boundaries
                        if Task.isCancelled {
                            return
                        }

                        // Yield periodically for fairness
                        if (i - start) % 500 == 0 {
                            await Task.yield()
                        }

                        // Perform locked updates in a non-async context
                        let performUpdate: () -> Void = {
                            if outDegrees[i] > 0 {
                                let contribution = damping * scores[i] / Float(outDegrees[i])

                                let rowStart = Int(matrix.rowPointers[i])
                                let rowEnd = Int(matrix.rowPointers[i + 1])

                                for idx in rowStart..<rowEnd {
                                    let j = Int(matrix.columnIndices[idx])

                                    let lock = locks[j % stripeCount]
                                    lock.lock()
                                    bufferWrapper.pointer[j] += contribution
                                    lock.unlock()
                                }
                            } else {
                                // Handle dangling nodes
                                let contribution = damping * scores[i] / Float(n)
                                for j in 0..<n {
                                    let lock = locks[j % stripeCount]
                                    lock.lock()
                                    bufferWrapper.pointer[j] += contribution
                                    lock.unlock()
                                }
                            }
                        }
                        performUpdate()
                    }
                }
            }

            // Wait for all tasks to complete
            await group.waitForAll()
        }
    }

    // MARK: - Betweenness Centrality

    public struct BetweennessCentralityOptions: Sendable {
        public let normalized: Bool
        public let weighted: Bool
        public let approximate: Bool
        public let sampleSize: Int?
        public let parallel: Bool

        public init(
            normalized: Bool = true,
            weighted: Bool = false,
            approximate: Bool = false,
            sampleSize: Int? = nil,
            parallel: Bool = true
        ) {
            self.normalized = normalized
            self.weighted = weighted
            self.approximate = approximate
            self.sampleSize = sampleSize
            self.parallel = parallel
        }
    }

    public static func betweennessCentrality(
        matrix: SparseMatrix,
        options: BetweennessCentralityOptions = BetweennessCentralityOptions()
    ) async -> ContiguousArray<Float> {
        let n = matrix.rows
        var centrality = ContiguousArray<Float>(repeating: 0.0, count: n)

        // Choose source nodes for computation
        let sources: [Int32]
        if options.approximate, let sampleSize = options.sampleSize {
            // Random sampling for approximation
            sources = (0..<n).shuffled().prefix(min(sampleSize, n)).map { Int32($0) }
        } else {
            sources = (0..<n).map { Int32($0) }
        }

        if options.parallel && sources.count > 10 {
            // Use 2x oversubscription for better load balancing
            let numProcessors = ProcessInfo.processInfo.activeProcessorCount
            let numTasks = min(numProcessors * 2, sources.count)
            let chunkSize = (sources.count + numTasks - 1) / numTasks

            // Use TaskGroup for structured concurrency
            let allLocalCentrality = await withTaskGroup(
                of: ContiguousArray<Float>.self,
                returning: [ContiguousArray<Float>].self
            ) { group in
                // Schedule tasks
                for taskId in 0..<numTasks {
                    group.addTask {
                        let start = taskId * chunkSize
                        let end = min((taskId + 1) * chunkSize, sources.count)
                        guard start < end else {
                            return ContiguousArray<Float>(repeating: 0.0, count: n)
                        }

                        var localCentrality = ContiguousArray<Float>(repeating: 0.0, count: n)

                        for idx in start..<end {
                            // Check for cancellation
                            if Task.isCancelled {
                                return localCentrality
                            }

                            // Yield periodically for fairness
                            if (idx - start) % 100 == 0 {
                                await Task.yield()
                            }

                            let source = sources[idx]
                            let contribution = computeBetweennessContribution(
                                matrix: matrix,
                                source: source,
                                weighted: options.weighted
                            )

                            // Accumulate locally using vDSP_vadd
                            localCentrality.withUnsafeMutableBufferPointer { destPtr in
                                contribution.withUnsafeBufferPointer { srcPtr in
                                    vDSP_vadd(destPtr.baseAddress!, 1, srcPtr.baseAddress!, 1, destPtr.baseAddress!, 1, vDSP_Length(n))
                                }
                            }
                        }

                        return localCentrality
                    }
                }

                // Collect results from all tasks
                var results: [ContiguousArray<Float>] = []
                for await localCentrality in group {
                    results.append(localCentrality)
                }
                return results
            }

            // Reduction phase
            for localCentrality in allLocalCentrality {
                centrality.withUnsafeMutableBufferPointer { destPtr in
                    localCentrality.withUnsafeBufferPointer { srcPtr in
                        vDSP_vadd(destPtr.baseAddress!, 1, srcPtr.baseAddress!, 1, destPtr.baseAddress!, 1, vDSP_Length(n))
                    }
                }
            }
        } else {
            // Serial computation
            for source in sources {
                let contribution = computeBetweennessContribution(
                    matrix: matrix,
                    source: source,
                    weighted: options.weighted
                )

                centrality.withUnsafeMutableBufferPointer { destPtr in
                    contribution.withUnsafeBufferPointer { srcPtr in
                        vDSP_vadd(destPtr.baseAddress!, 1, srcPtr.baseAddress!, 1, destPtr.baseAddress!, 1, vDSP_Length(n))
                    }
                }
            }
        }

        // Normalization
        if options.normalized && n > 2 {
            var scale: Float = 1.0

            if options.approximate && options.sampleSize != nil {
                scale = Float(n - 1) / Float(sources.count)
            } else {
                scale = 1.0 / Float((n - 1) * (n - 2))
            }

            centrality.withUnsafeMutableBufferPointer { ptr in
                vDSP_vsmul(ptr.baseAddress!, 1, &scale, ptr.baseAddress!, 1, vDSP_Length(n))
            }
        }

        return centrality
    }

    // MARK: - Betweenness Contribution (Brandes Algorithm)

    private static func computeBetweennessContribution(
        matrix: SparseMatrix,
        source: Int32,
        weighted: Bool
    ) -> ContiguousArray<Float> {
        let n = matrix.rows
        var stack: [Int32] = []
        var paths = ContiguousArray<[Int32]>(repeating: [], count: n)
        var sigma = ContiguousArray<Float>(repeating: 0.0, count: n)
        var distance = ContiguousArray<Float>(repeating: .infinity, count: n)
        var delta = ContiguousArray<Float>(repeating: 0.0, count: n)

        // Initialize source
        sigma[Int(source)] = 1.0
        distance[Int(source)] = 0.0

        // BFS/Dijkstra phase
        if !weighted {
            // BFS for unweighted graphs
            var queue = [source]
            var head = 0

            while head < queue.count {
                let v = queue[head]
                head += 1
                stack.append(v)

                let rowStart = Int(matrix.rowPointers[Int(v)])
                let rowEnd = Int(matrix.rowPointers[Int(v) + 1])

                for idx in rowStart..<rowEnd {
                    let w = Int32(matrix.columnIndices[idx])

                    // Path discovery
                    if distance[Int(w)] == .infinity {
                        distance[Int(w)] = distance[Int(v)] + 1.0
                        queue.append(w)
                    }

                    // Path counting
                    if distance[Int(w)] == distance[Int(v)] + 1.0 {
                        sigma[Int(w)] += sigma[Int(v)]
                        paths[Int(w)].append(v)
                    }
                }
            }
        } else {
            // Dijkstra for weighted graphs
            var heap = BinaryHeap<(node: Int32, dist: Float)> { $0.dist < $1.dist }
            heap.insert((source, 0.0))

            while let current = heap.extractMin() {
                let v = current.node

                if current.dist > distance[Int(v)] { continue }

                stack.append(v)

                let rowStart = Int(matrix.rowPointers[Int(v)])
                let rowEnd = Int(matrix.rowPointers[Int(v) + 1])

                for idx in rowStart..<rowEnd {
                    let w = Int32(matrix.columnIndices[idx])
                    let weight = matrix.values?[idx] ?? 1.0
                    let newDist = distance[Int(v)] + weight

                    if newDist < distance[Int(w)] {
                        distance[Int(w)] = newDist
                        heap.insert((w, newDist))
                        sigma[Int(w)] = sigma[Int(v)]
                        paths[Int(w)] = [v]
                    } else if abs(newDist - distance[Int(w)]) < 1e-9 {
                        sigma[Int(w)] += sigma[Int(v)]
                        paths[Int(w)].append(v)
                    }
                }
            }
        }

        // Accumulation phase
        while !stack.isEmpty {
            let w = stack.removeLast()
            for v in paths[Int(w)] {
                delta[Int(v)] += (sigma[Int(v)] / sigma[Int(w)]) * (1.0 + delta[Int(w)])
            }
        }

        // Remove source contribution
        delta[Int(source)] = 0

        return delta
    }

    // MARK: - Eigenvector Centrality

    public struct EigenvectorCentralityOptions: Sendable {
        public let tolerance: Float
        public let maxIterations: Int
        public let startVector: ContiguousArray<Float>?

        public init(
            tolerance: Float = 1e-6,
            maxIterations: Int = 100,
            startVector: ContiguousArray<Float>? = nil
        ) {
            self.tolerance = tolerance
            self.maxIterations = maxIterations
            self.startVector = startVector
        }
    }

    public static func eigenvectorCentrality(
        matrix: SparseMatrix,
        options: EigenvectorCentralityOptions = EigenvectorCentralityOptions()
    ) -> ContiguousArray<Float> {
        let n = matrix.rows

        // Initialize eigenvector
        var eigenvector: ContiguousArray<Float>
        if let start = options.startVector, start.count == n {
            eigenvector = start
            normalizeL2(&eigenvector)
        } else {
            eigenvector = ContiguousArray<Float>(repeating: 1.0 / sqrt(Float(n)), count: n)
        }

        var newEigenvector = ContiguousArray<Float>(repeating: 0.0, count: n)

        var iteration = 0
        var converged = false

        while iteration < options.maxIterations && !converged {
            // Matrix-vector multiplication
            for i in 0..<n {
                var sum: Float = 0.0
                let rowStart = Int(matrix.rowPointers[i])
                let rowEnd = Int(matrix.rowPointers[i + 1])

                for idx in rowStart..<rowEnd {
                    let j = Int(matrix.columnIndices[idx])
                    let weight = matrix.values?[idx] ?? 1.0
                    sum += weight * eigenvector[j]
                }

                newEigenvector[i] = sum
            }

            // Normalize
            let norm = normalizeL2(&newEigenvector)

            if norm == 0 {
                break
            }

            // Check convergence
            let diff = computeL2Norm(eigenvector, newEigenvector)
            converged = diff < options.tolerance

            swap(&eigenvector, &newEigenvector)
            iteration += 1
        }

        return eigenvector
    }

    // MARK: - Community Detection (Louvain Algorithm)

    public struct CommunityDetectionOptions: Sendable {
        public let resolution: Float
        public let randomSeed: Int?
        public let maxIterations: Int
        public let minModularityGain: Float

        public init(
            resolution: Float = 1.0,
            randomSeed: Int? = nil,
            maxIterations: Int = 10,
            minModularityGain: Float = 1e-7
        ) {
            self.resolution = resolution
            self.randomSeed = randomSeed
            self.maxIterations = maxIterations
            self.minModularityGain = minModularityGain
        }
    }

    public struct CommunityDetectionResult {
        public let communities: ContiguousArray<Int32>
        public let modularity: Float
        public let numCommunities: Int
        public let hierarchy: [[Int32]]?
    }

    public static func detectCommunities(
        matrix: SparseMatrix,
        options: CommunityDetectionOptions = CommunityDetectionOptions()
    ) -> CommunityDetectionResult {
        let n = matrix.rows
        var communities = ContiguousArray<Int32>((0..<n).map { Int32($0) })
        var improved = true
        var level = 0
        var hierarchicalLevels: [[Int32]] = []

        // Compute total edge weight
        let totalWeight = computeTotalWeight(matrix: matrix)

        while improved && level < options.maxIterations {
            improved = false
            let currentModularity = computeModularity(
                matrix: matrix,
                communities: communities,
                totalWeight: totalWeight,
                resolution: options.resolution
            )

            // Phase 1: Local optimization
            for _ in 0..<n {
                // Random order traversal
                let nodeOrder: [Int]
                if let seed = options.randomSeed {
                    var rng = SeededRandomNumberGenerator(seed: seed)
                    nodeOrder = (0..<n).shuffled(using: &rng)
                } else {
                    nodeOrder = Array(0..<n)
                }

                for i in nodeOrder {
                    let node = Int32(i)
                    let currentCommunity = communities[i]

                    // Find best community for this node
                    let neighbors = getNeighborCommunities(
                        matrix: matrix,
                        node: node,
                        communities: communities
                    )

                    var bestCommunity = currentCommunity
                    var bestGain: Float = 0.0

                    for (community, _) in neighbors {
                        if community != currentCommunity {
                            // Calculate modularity gain
                            let gain = calculateModularityGain(
                                matrix: matrix,
                                node: node,
                                targetCommunity: community,
                                communities: communities,
                                totalWeight: totalWeight,
                                resolution: options.resolution
                            )

                            if gain > bestGain {
                                bestGain = gain
                                bestCommunity = community
                            }
                        }
                    }

                    // Move node if beneficial
                    if bestGain > options.minModularityGain {
                        communities[i] = bestCommunity
                        improved = true
                    }
                }

                if !improved { break }
            }

            // Save hierarchical level
            hierarchicalLevels.append(Array(communities))

            // Phase 2: Community aggregation
            if improved {
                let (aggregatedMatrix, newCommunities) = aggregateCommunities(
                    matrix: matrix,
                    communities: communities
                )

                // Continue with aggregated graph
                if aggregatedMatrix.rows < matrix.rows {
                    // Map communities to new level
                    communities = newCommunities
                    level += 1
                } else {
                    break
                }
            }
        }

        // Compute final statistics
        let finalModularity = computeModularity(
            matrix: matrix,
            communities: communities,
            totalWeight: totalWeight,
            resolution: options.resolution
        )

        let uniqueCommunities = Set(communities)

        return CommunityDetectionResult(
            communities: communities,
            modularity: finalModularity,
            numCommunities: uniqueCommunities.count,
            hierarchy: hierarchicalLevels
        )
    }

    // MARK: - Label Propagation

    public static func labelPropagation(
        matrix: SparseMatrix,
        maxIterations: Int = 100
    ) -> ContiguousArray<Int32> {
        let n = matrix.rows
        var labels = ContiguousArray<Int32>((0..<n).map { Int32($0) })
        var updated = true
        var iteration = 0

        while updated && iteration < maxIterations {
            updated = false
            let nodeOrder = (0..<n).shuffled()

            for i in nodeOrder {
                // Count neighbor labels
                var labelCounts: [Int32: Float] = [:]

                let rowStart = Int(matrix.rowPointers[i])
                let rowEnd = Int(matrix.rowPointers[i + 1])

                for idx in rowStart..<rowEnd {
                    let neighbor = Int32(matrix.columnIndices[idx])
                    let weight = matrix.values?[idx] ?? 1.0
                    let neighborLabel = labels[Int(neighbor)]

                    labelCounts[neighborLabel, default: 0.0] += weight
                }

                // Select most frequent label
                if let maxLabel = labelCounts.max(by: { $0.value < $1.value }) {
                    if maxLabel.key != labels[i] {
                        labels[i] = maxLabel.key
                        updated = true
                    }
                }
            }

            iteration += 1
        }

        return labels
    }

    // MARK: - Clustering Coefficient

    public static func clusteringCoefficient(
        matrix: SparseMatrix,
        local: Bool = false
    ) -> Float {
        let n = matrix.rows
        var coefficients = ContiguousArray<Float>(repeating: 0.0, count: n)

        for i in 0..<n {
            // Get neighbors
            let rowStart = Int(matrix.rowPointers[i])
            let rowEnd = Int(matrix.rowPointers[i + 1])
            let degree = rowEnd - rowStart

            if degree < 2 {
                coefficients[i] = 0.0
                continue
            }

            // Count triangles
            var triangles = 0
            for idx1 in rowStart..<rowEnd {
                let neighbor1 = Int(matrix.columnIndices[idx1])

                for idx2 in (idx1 + 1)..<rowEnd {
                    let neighbor2 = Int(matrix.columnIndices[idx2])

                    // Check if neighbor1 and neighbor2 are connected
                    if areConnected(
                        matrix: matrix,
                        node1: Int32(neighbor1),
                        node2: Int32(neighbor2)
                    ) {
                        triangles += 1
                    }
                }
            }

            // Local clustering coefficient
            let possibleTriangles = degree * (degree - 1) / 2
            coefficients[i] = Float(triangles) / Float(possibleTriangles)
        }

        if local {
            // Return average local clustering coefficient
            return coefficients.reduce(0, +) / Float(n)
        } else {
            // Return global clustering coefficient
            let totalTriangles = coefficients.reduce(0, +)
            let totalPossible = Float(n * (n - 1) * (n - 2) / 6)
            return totalTriangles / totalPossible
        }
    }

    // MARK: - Helper Functions

    private static func computeOutDegrees(matrix: SparseMatrix) -> ContiguousArray<Int> {
        var outDegrees = ContiguousArray<Int>(repeating: 0, count: matrix.rows)
        for i in 0..<matrix.rows {
            outDegrees[i] = Int(matrix.rowPointers[i + 1] - matrix.rowPointers[i])
        }
        return outDegrees
    }

    private static func computeL1Norm(
        _ a: ContiguousArray<Float>,
        _ b: ContiguousArray<Float>
    ) -> Float {
        var sum: Float = 0
        for i in 0..<a.count {
            sum += abs(a[i] - b[i])
        }
        return sum
    }

    private static func computeL2Norm(
        _ a: ContiguousArray<Float>,
        _ b: ContiguousArray<Float>
    ) -> Float {
        var sum: Float = 0
        for i in 0..<a.count {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        return sqrt(sum)
    }

    @discardableResult
    private static func normalizeL2(_ vector: inout ContiguousArray<Float>) -> Float {
        let count = vector.count
        var norm: Float = 0
        vector.withUnsafeBufferPointer { ptr in
            norm = cblas_snrm2(Int32(count), ptr.baseAddress!, 1)
        }

        if norm > 0 {
            let scale = 1.0 / norm
            vector.withUnsafeMutableBufferPointer { ptr in
                vDSP_vsmul(ptr.baseAddress!, 1, [scale], ptr.baseAddress!, 1, vDSP_Length(count))
            }
        }
        return norm
    }

    private static func computeTotalWeight(matrix: SparseMatrix) -> Float {
        if let values = matrix.values {
            return values.reduce(0, +)
        } else {
            return Float(matrix.nonZeros)
        }
    }

    private static func computeModularity(
        matrix: SparseMatrix,
        communities: ContiguousArray<Int32>,
        totalWeight: Float,
        resolution: Float
    ) -> Float {
        // Simplified modularity computation for demonstration
        // Full implementation would calculate:
        // Q = (1/2m) * Σ[Aij - γ*ki*kj/2m] * δ(ci, cj)
        var modularity: Float = 0.0

        // Group nodes by community
        var communityNodes: [Int32: [Int]] = [:]
        for i in 0..<communities.count {
            communityNodes[communities[i], default: []].append(i)
        }

        // Calculate modularity for each community
        for (_, nodes) in communityNodes {
            var internalWeight: Float = 0.0
            var totalDegree: Float = 0.0

            for node in nodes {
                let rowStart = Int(matrix.rowPointers[node])
                let rowEnd = Int(matrix.rowPointers[node + 1])

                for idx in rowStart..<rowEnd {
                    let neighbor = Int(matrix.columnIndices[idx])
                    let weight = matrix.values?[idx] ?? 1.0

                    totalDegree += weight

                    if communities[neighbor] == communities[node] {
                        internalWeight += weight
                    }
                }
            }

            let term1 = internalWeight / totalWeight
            let term2 = resolution * (totalDegree / totalWeight) * (totalDegree / totalWeight)
            modularity += term1 - term2
        }

        return modularity
    }

    private static func getNeighborCommunities(
        matrix: SparseMatrix,
        node: Int32,
        communities: ContiguousArray<Int32>
    ) -> [(Int32, Float)] {
        var communityWeights: [Int32: Float] = [:]

        let rowStart = Int(matrix.rowPointers[Int(node)])
        let rowEnd = Int(matrix.rowPointers[Int(node) + 1])

        for idx in rowStart..<rowEnd {
            let neighbor = Int(matrix.columnIndices[idx])
            let weight = matrix.values?[idx] ?? 1.0
            let community = communities[neighbor]

            communityWeights[community, default: 0.0] += weight
        }

        return Array(communityWeights)
    }

    private static func calculateModularityGain(
        matrix: SparseMatrix,
        node: Int32,
        targetCommunity: Int32,
        communities: ContiguousArray<Int32>,
        totalWeight: Float,
        resolution: Float
    ) -> Float {
        // Calculate the modularity gain from moving node to targetCommunity
        // ΔQ = [Sum_in - Sum_tot * ki / 2m] * (1 / m)

        var sumIn: Float = 0.0
        var sumTot: Float = 0.0
        var ki: Float = 0.0

        let rowStart = Int(matrix.rowPointers[Int(node)])
        let rowEnd = Int(matrix.rowPointers[Int(node) + 1])

        for idx in rowStart..<rowEnd {
            let neighbor = Int(matrix.columnIndices[idx])
            let weight = matrix.values?[idx] ?? 1.0

            ki += weight

            if communities[neighbor] == targetCommunity {
                sumIn += weight
            }
            if communities[neighbor] == targetCommunity || communities[neighbor] == communities[Int(node)] {
                sumTot += weight
            }
        }

        if totalWeight == 0 { return 0.0 }

        let gain = (sumIn - resolution * sumTot * ki / totalWeight) / totalWeight
        return gain
    }

    private static func aggregateCommunities(
        matrix: SparseMatrix,
        communities: ContiguousArray<Int32>
    ) -> (SparseMatrix, ContiguousArray<Int32>) {
        // Create aggregated graph where each community becomes a node
        // This is a simplified version - full implementation would properly aggregate edges

        // Renumber communities to be contiguous
        var communityMap: [Int32: Int32] = [:]
        var nextId: Int32 = 0
        var newCommunities = ContiguousArray<Int32>(repeating: 0, count: communities.count)

        for i in 0..<communities.count {
            let comm = communities[i]
            if communityMap[comm] == nil {
                communityMap[comm] = nextId
                nextId += 1
            }
            newCommunities[i] = communityMap[comm]!
        }

        // For demonstration, return original matrix with renumbered communities
        // Full implementation would create a new aggregated matrix
        return (matrix, newCommunities)
    }

    private static func areConnected(
        matrix: SparseMatrix,
        node1: Int32,
        node2: Int32
    ) -> Bool {
        let rowStart = Int(matrix.rowPointers[Int(node1)])
        let rowEnd = Int(matrix.rowPointers[Int(node1) + 1])

        for idx in rowStart..<rowEnd {
            if matrix.columnIndices[idx] == UInt32(node2) {
                return true
            }
        }
        return false
    }
}

// MARK: - Helper Data Structures

// Binary Heap for Dijkstra's algorithm
private struct BinaryHeap<T> {
    private var elements: [T] = []
    private let compare: (T, T) -> Bool

    init(compare: @escaping (T, T) -> Bool) {
        self.compare = compare
    }

    var isEmpty: Bool { elements.isEmpty }

    mutating func insert(_ element: T) {
        elements.append(element)
        siftUp(elements.count - 1)
    }

    mutating func extractMin() -> T? {
        guard !isEmpty else { return nil }
        if elements.count == 1 {
            return elements.removeLast()
        }
        let min = elements[0]
        elements[0] = elements.removeLast()
        siftDown(0)
        return min
    }

    private mutating func siftUp(_ index: Int) {
        var child = index
        var parent = (child - 1) / 2

        while child > 0 && compare(elements[child], elements[parent]) {
            elements.swapAt(child, parent)
            child = parent
            parent = (child - 1) / 2
        }
    }

    private mutating func siftDown(_ index: Int) {
        var parent = index
        while true {
            let left = 2 * parent + 1
            let right = 2 * parent + 2
            var candidate = parent

            if left < elements.count && compare(elements[left], elements[candidate]) {
                candidate = left
            }
            if right < elements.count && compare(elements[right], elements[candidate]) {
                candidate = right
            }
            if candidate == parent { break }

            elements.swapAt(parent, candidate)
            parent = candidate
        }
    }
}

// Seeded random number generator for reproducible results
private struct SeededRandomNumberGenerator: RandomNumberGenerator {
    private var state: UInt64

    init(seed: Int) {
        self.state = UInt64(seed)
    }

    mutating func next() -> UInt64 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return state
    }
}

// MARK: - Advanced Centrality Measures

extension GraphPrimitivesKernels {

    // MARK: - 1. Strongly Connected Components (Tarjan's Algorithm)

    /// Internal state management for Tarjan's algorithm.
    /// Optimized using arrays for O(1) access, assuming contiguous node IDs 0..N-1.
    private struct TarjanState {
        var index: Int = 0                    // Global DFS counter
        var stack: [Int] = []                 // DFS stack
        var onStack: [Bool]                   // O(1) membership check
        var indices: [Int]                    // node -> discovery time (-1 if unvisited)
        var lowLinks: [Int]                   // node -> lowest reachable index
        var sccs: [Set<Int>] = []             // Result accumulator

        init(nodeCount: Int) {
            self.indices = Array(repeating: -1, count: nodeCount)
            self.lowLinks = Array(repeating: -1, count: nodeCount)
            self.onStack = Array(repeating: false, count: nodeCount)
            self.stack.reserveCapacity(nodeCount)
        }
    }

    /// Find strongly connected components using Tarjan's algorithm.
    ///
    /// A strongly connected component (SCC) is a maximal set of vertices where
    /// every vertex is reachable from every other vertex in the set.
    ///
    /// # Algorithm
    /// Tarjan's single-pass DFS with discovery times and low-link values.
    /// Iterative implementation to avoid stack overflow on deep graphs.
    ///
    /// # Complexity
    /// - Time: O(V + E)
    /// - Space: O(V)
    ///
    /// # Parameters
    /// - `graph`: Directed graph in CSR format
    ///
    /// # Returns
    /// Array of SCCs, each SCC is a `Set<Int>` of node IDs
    ///
    /// # Example
    /// ```swift
    /// let graph = SparseMatrix(...)
    /// let sccs = GraphPrimitivesKernels.findStronglyConnectedComponents(graph: graph)
    /// print("Found \(sccs.count) strongly connected components")
    /// ```
    public static func findStronglyConnectedComponents(
        graph: SparseMatrix
    ) -> [Set<Int>] {
        let nodeCount = graph.rows
        var state = TarjanState(nodeCount: nodeCount)

        // Iterate over all nodes to handle disconnected components
        for v in 0..<nodeCount {
            if state.indices[v] == -1 {
                tarjanDFSIterative(v, graph: graph, state: &state)
            }
        }

        return state.sccs
    }

    /// Iterative DFS for Tarjan's algorithm (stack-overflow safe).
    ///
    /// Uses explicit call stack to simulate recursion. Each frame tracks:
    /// - Current node
    /// - Neighbor iteration index
    /// - Whether we're returning from a child call
    private static func tarjanDFSIterative(_ start: Int, graph: SparseMatrix, state: inout TarjanState) {
        // Call stack: (node, neighborIndex, isReturning)
        var callStack: [(node: Int, neighborIdx: Int, returning: Bool)] = [(start, 0, false)]

        while !callStack.isEmpty {
            var frame = callStack.removeLast()
            let v = frame.node

            if !frame.returning {
                // First visit to node v
                if state.indices[v] != -1 {
                    continue  // Already processed
                }

                // Initialize discovery time and low-link
                state.indices[v] = state.index
                state.lowLinks[v] = state.index
                state.index += 1
                state.stack.append(v)
                state.onStack[v] = true

                // Mark frame for post-processing after exploring children
                callStack.append((v, 0, true))
            }

            // Process neighbors using CSR format directly
            let start = Int(graph.rowPointers[v])
            let end = Int(graph.rowPointers[v + 1])
            var childProcessed = false

            for idx in frame.neighborIdx..<(end - start) {
                let w = Int(graph.columnIndices[start + idx])

                if state.indices[w] == -1 {
                    // Tree edge: unvisited neighbor
                    // Push continuation frame and child frame
                    callStack.append((v, idx + 1, true))  // Resume from next neighbor
                    callStack.append((w, 0, false))        // Visit child
                    childProcessed = true
                    break
                } else if state.onStack[w] {
                    // Back edge: update low-link
                    state.lowLinks[v] = min(state.lowLinks[v], state.indices[w])
                }
                // Forward/cross edges: ignore
            }

            // Post-processing after all children explored
            if frame.returning && !childProcessed {
                // Check if v is SCC root
                if state.lowLinks[v] == state.indices[v] {
                    var scc = Set<Int>()
                    // Pop stack until v to extract SCC
                    while let node = state.stack.popLast() {
                        state.onStack[node] = false
                        scc.insert(node)
                        if node == v { break }
                    }
                    state.sccs.append(scc)
                }

                // Update parent's low-link (if parent exists on call stack)
                if let parentFrame = callStack.last(where: { !$0.returning }) {
                    let parent = parentFrame.node
                    state.lowLinks[parent] = min(state.lowLinks[parent], state.lowLinks[v])
                }
            }
        }
    }

    // MARK: - 2. Eigenvector Centrality (Power Iteration)

    /// Compute eigenvector centrality using power iteration.
    ///
    /// Eigenvector centrality assigns scores based on the principle that connections
    /// to high-scoring nodes contribute more than connections to low-scoring nodes.
    ///
    /// # Mathematical Foundation
    /// Solves the eigenvector equation: `A·x = λ·x`
    /// - A: Adjacency matrix
    /// - x: Centrality vector (principal eigenvector)
    /// - λ: Largest eigenvalue
    ///
    /// Computed via power iteration: `x_{k+1} = A·x_k / ||A·x_k||₂`
    ///
    /// # Complexity
    /// - Time: O(iterations × E)
    /// - Space: O(V)
    ///
    /// # Parameters
    /// - `graph`: Undirected or directed graph (if directed, uses incoming edges)
    /// - `maxIterations`: Maximum iterations (default: 100)
    /// - `tolerance`: Convergence threshold for L1 norm change (default: 1e-6)
    ///
    /// # Returns
    /// Array of centrality scores indexed by node ID, or `nil` if doesn't converge
    ///
    /// # Convergence Notes
    /// - May not converge for bipartite graphs (oscillation)
    /// - Disconnected graphs: some components may have zero centrality
    /// - Zero-degree nodes remain zero throughout
    ///
    /// # Example
    /// ```swift
    /// if let centrality = GraphPrimitivesKernels.eigenvectorCentrality(graph: graph) {
    ///     let topNode = centrality.enumerated().max { $0.element < $1.element }
    ///     print("Most central node: \(topNode?.offset ?? -1)")
    /// }
    /// ```
    public static func eigenvectorCentrality(
        graph: SparseMatrix,
        maxIterations: Int = 100,
        tolerance: Float = 1e-6
    ) -> [Float]? {
        let N = graph.rows
        guard N > 0 else { return [] }

        // Initialize centrality vector uniformly: x = [1/√N, ..., 1/√N]
        let initialValue = 1.0 / sqrt(Float(N))
        var x = [Float](repeating: initialValue, count: N)
        var y = [Float](repeating: 0.0, count: N)

        // Power iteration loop
        for _ in 0..<maxIterations {
            // Matrix-Vector Multiply: y = A·x (using CSR format)
            graph.rowPointers.withUnsafeBufferPointer { rowPtrs in
                graph.columnIndices.withUnsafeBufferPointer { colIndices in
                    for i in 0..<N {
                        var sum: Float = 0.0
                        let start = Int(rowPtrs[i])
                        let end = Int(rowPtrs[i+1])

                        if let weights = graph.values {
                            // Weighted graph
                            weights.withUnsafeBufferPointer { valPtr in
                                for j in start..<end {
                                    let col = Int(colIndices[j])
                                    sum += valPtr[j] * x[col]
                                }
                            }
                        } else {
                            // Unweighted graph
                            for j in start..<end {
                                let col = Int(colIndices[j])
                                sum += x[col]
                            }
                        }
                        y[i] = sum
                    }
                }
            }

            // Compute L2 norm: ||y||₂
            let norm: Float
            #if canImport(Accelerate)
            // SIMD-optimized sum of squares
            var sumSq: Float = 0
            vDSP_svesq(y, 1, &sumSq, vDSP_Length(N))
            norm = sqrt(sumSq)
            #else
            norm = sqrt(y.reduce(0) { $0 + $1 * $1 })
            #endif

            // Handle degenerate case: zero vector (graph with no edges)
            if norm == 0 || !norm.isFinite {
                return nil  // Non-convergent or degenerate graph
            }

            // Normalize: x_new = y / ||y||₂ and compute convergence metric
            var l1Change: Float = 0.0
            for i in 0..<N {
                let normalizedValue = y[i] / norm
                l1Change += abs(normalizedValue - x[i])
                x[i] = normalizedValue
            }

            // Check convergence: ||x_new - x||₁ < tolerance
            if l1Change < tolerance {
                return x
            }
        }

        // Did not converge within maxIterations
        return nil
    }

    // MARK: - 3. Average Path Length

    /// Compute average shortest path length for the graph.
    ///
    /// Calculates the mean of all-pairs shortest paths. Useful for measuring
    /// graph diameter and connectivity. Infinite paths (disconnected nodes) are excluded.
    ///
    /// # Algorithm Selection
    /// Automatically chooses the most efficient algorithm based on graph size:
    /// - **Small graphs (N < 500)**: Floyd-Warshall O(V³)
    /// - **Medium graphs**: BFS from each node O(V(V+E))
    /// - **Large graphs (N > 10,000)**: Monte Carlo sampling O(S(V+E))
    ///
    /// # Complexity
    /// - Time: O(V³) or O(V(V+E)) or O(S(V+E)) depending on algorithm
    /// - Space: O(V²) for Floyd-Warshall, O(V) otherwise
    ///
    /// # Parameters
    /// - `graph`: Undirected or directed graph
    /// - `sampling`: Override auto-selection (default: auto based on size)
    /// - `sampleSize`: Number of source nodes to sample (default: min(1000, N))
    ///
    /// # Returns
    /// Average path length, or `nil` if graph is completely disconnected
    ///
    /// # Example
    /// ```swift
    /// if let avgPath = await GraphPrimitivesKernels.averagePathLength(graph: graph) {
    ///     print("Average shortest path: \(avgPath)")
    /// }
    /// ```
    public static func averagePathLength(
        graph: SparseMatrix,
        sampling: Bool? = nil,
        sampleSize: Int? = nil
    ) async -> Float? {
        let N = graph.rows
        guard N > 1 else { return 0.0 }

        // Algorithm selection logic
        let useSampling = sampling ?? (N > 10_000)
        let actualSampleSize = sampleSize ?? min(1000, N)

        if useSampling && N > actualSampleSize {
            return await averagePathLengthSampled(graph, sampleSize: actualSampleSize)
        } else if N < 500 {
            // Floyd-Warshall for small graphs (works for both weighted and unweighted)
            return averagePathLengthFloydWarshall(graph)
        } else {
            // BFS for medium/large graphs
            return await averagePathLengthBFS(graph)
        }
    }

    // MARK: - Floyd-Warshall Implementation (Small Graphs)

    /// Floyd-Warshall all-pairs shortest path algorithm.
    ///
    /// Complexity: O(V³) time, O(V²) space
    /// Suitable for small dense graphs (N < 500)
    private static func averagePathLengthFloydWarshall(_ graph: SparseMatrix) -> Float? {
        let N = graph.rows

        // Safety check: Floyd-Warshall is O(V³) and should only be used for small graphs
        guard N < 500 else {
            assertionFailure("Floyd-Warshall called with N=\(N) (should be < 500). Use BFS or sampling instead.")
            return nil
        }

        // Initialize distance matrix with infinity
        var dist = [[Float]](
            repeating: [Float](repeating: Float.infinity, count: N),
            count: N
        )

        // Set diagonal to 0 (distance to self)
        for i in 0..<N {
            dist[i][i] = 0
        }

        // Set direct edges from adjacency matrix
        graph.rowPointers.withUnsafeBufferPointer { rowPtrs in
            graph.columnIndices.withUnsafeBufferPointer { colIndices in
                if let weights = graph.values {
                    // Weighted graph
                    weights.withUnsafeBufferPointer { valPtr in
                        for i in 0..<N {
                            let start = Int(rowPtrs[i])
                            let end = Int(rowPtrs[i+1])
                            for j in start..<end {
                                let col = Int(colIndices[j])
                                // Handle multi-edges: take minimum weight
                                dist[i][col] = min(dist[i][col], valPtr[j])
                            }
                        }
                    }
                } else {
                    // Unweighted graph (distance = 1)
                    for i in 0..<N {
                        let start = Int(rowPtrs[i])
                        let end = Int(rowPtrs[i+1])
                        for j in start..<end {
                            let col = Int(colIndices[j])
                            dist[i][col] = 1.0
                        }
                    }
                }
            }
        }

        // Floyd-Warshall: k is intermediate vertex
        for k in 0..<N {
            for i in 0..<N {
                for j in 0..<N {
                    let throughK = dist[i][k] + dist[k][j]
                    if throughK < dist[i][j] {
                        dist[i][j] = throughK
                    }
                }
            }
        }

        // Compute average of finite paths (excluding self-distances)
        var totalDistance: Double = 0.0
        var pathCount: Int = 0

        for i in 0..<N {
            for j in 0..<N {
                if i != j && dist[i][j].isFinite {
                    totalDistance += Double(dist[i][j])
                    pathCount += 1
                }
            }
        }

        return pathCount > 0 ? Float(totalDistance / Double(pathCount)) : nil
    }

    // MARK: - BFS-Based Implementations (Medium/Large Graphs)

    /// Helper for BFS-based average path length calculation.
    ///
    /// Integrates with VectorCore's async BFS API.
    /// Accumulates distances from specified source nodes to all reachable targets.
    private static func calculateAverageFromBFS(
        graph: SparseMatrix,
        sources: ArraySlice<Int>
    ) async -> Float? {
        var totalLength: Int64 = 0
        var pathCount: Int64 = 0

        // Run BFS from each source node
        for startNode in sources {
            // Use VectorCore's async BFS with default options
            let result = await GraphPrimitivesKernels.breadthFirstSearch(
                matrix: graph,
                source: Int32(startNode),
                options: GraphPrimitivesKernels.BFSOptions()
            )

            // Extract distances from BFSResult
            // distances[i] = hop count from source to node i (-1 if unreachable)
            for (nodeId, distance) in result.distances.enumerated() {
                if nodeId != startNode && distance != -1 {
                    totalLength += Int64(distance)
                    pathCount += 1
                }
            }
        }

        return pathCount > 0 ? Float(totalLength) / Float(pathCount) : nil
    }

    /// BFS from all nodes (exact average for medium graphs).
    ///
    /// Complexity: O(V(V+E))
    private static func averagePathLengthBFS(_ graph: SparseMatrix) async -> Float? {
        let sources = (0..<graph.rows)
        return await calculateAverageFromBFS(graph: graph, sources: ArraySlice(sources))
    }

    /// Sampled BFS (approximate average for very large graphs).
    ///
    /// Uses random sampling of source nodes to estimate average path length.
    /// Trades accuracy for performance on graphs with N > 10,000.
    ///
    /// Complexity: O(S(V+E)) where S = sample size
    private static func averagePathLengthSampled(
        _ graph: SparseMatrix,
        sampleSize: Int
    ) async -> Float? {
        let N = graph.rows

        // Random sample of source nodes
        let allNodes = Array(0..<N)
        let sampledNodes = allNodes.shuffled().prefix(sampleSize)

        return await calculateAverageFromBFS(graph: graph, sources: sampledNodes)
    }
}
