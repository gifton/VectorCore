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

    public struct PageRankOptions {
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
    ) -> PageRankResult {
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
                parallelPageRankIteration(
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
    ) {
        let n = matrix.rows

        // Initialize new scores
        for i in 0..<n {
            newScores[i] = (1.0 - damping) * personalization[i]
        }

        // Use atomic buffer for thread-safe updates
        let atomicBuffer = UnsafeMutablePointer<Float>.allocate(capacity: n)
        newScores.withUnsafeBufferPointer { ptr in
            atomicBuffer.initialize(from: ptr.baseAddress!, count: n)
        }

        defer {
            // Copy back and cleanup
            for i in 0..<n {
                newScores[i] = atomicBuffer[i]
            }
            atomicBuffer.deinitialize(count: n)
            atomicBuffer.deallocate()
        }

        // Use striped locking for thread-safe updates
        let stripeCount = 128
        let locks = (0..<stripeCount).map { _ in NSLock() }

        DispatchQueue.concurrentPerform(iterations: n) { i in
            if outDegrees[i] > 0 {
                let contribution = damping * scores[i] / Float(outDegrees[i])

                let rowStart = Int(matrix.rowPointers[i])
                let rowEnd = Int(matrix.rowPointers[i + 1])

                for idx in rowStart..<rowEnd {
                    let j = Int(matrix.columnIndices[idx])

                    let lock = locks[j % stripeCount]
                    lock.lock()
                    atomicBuffer[j] += contribution
                    lock.unlock()
                }
            } else {
                // Handle dangling nodes
                let contribution = damping * scores[i] / Float(n)
                for j in 0..<n {
                    let lock = locks[j % stripeCount]
                    lock.lock()
                    atomicBuffer[j] += contribution
                    lock.unlock()
                }
            }
        }
    }

    // MARK: - Betweenness Centrality

    public struct BetweennessCentralityOptions {
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
    ) -> ContiguousArray<Float> {
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
            // Parallel computation with thread-local storage
            let numThreads = min(ProcessInfo.processInfo.activeProcessorCount, sources.count)
            let chunkSize = (sources.count + numThreads - 1) / numThreads

            var threadLocalCentrality = Array(repeating: ContiguousArray<Float>(repeating: 0.0, count: n), count: numThreads)

            DispatchQueue.concurrentPerform(iterations: numThreads) { t in
                let start = t * chunkSize
                let end = min((t + 1) * chunkSize, sources.count)

                var localCentrality = ContiguousArray<Float>(repeating: 0.0, count: n)

                for idx in start..<end {
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

                threadLocalCentrality[t] = localCentrality
            }

            // Reduction phase
            for localCentrality in threadLocalCentrality {
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

    public struct EigenvectorCentralityOptions {
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

    public struct CommunityDetectionOptions {
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