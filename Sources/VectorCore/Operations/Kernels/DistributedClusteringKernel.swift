//
//  DistributedClusteringKernel.swift
//  VectorCore
//
//  Distributed clustering using map-reduce paradigm
//

import Foundation
import simd

// MARK: - DistributedClustering512

/// Distributed clustering using map-reduce paradigm for 512-dimensional vectors
public final class DistributedClustering512 {

    public typealias Vector = Vector512Optimized

    // MARK: - Configuration and Core Types

    public struct Config {
        public let globalK: Int
        public let localK: Int
        public let maxIterations: Int
        public let convergenceTolerance: Float
        public let communicationMode: CommMode
        public let checkpointInterval: Int
        public let faultToleranceLevel: FaultLevel

        public enum CommMode {
            case synchronous
        }

        public enum FaultLevel {
            case none
            case checkpoint(url: URL)
        }

        public init(
            globalK: Int,
            localK: Int? = nil,
            maxIterations: Int = 50,
            convergenceTolerance: Float = 1e-4,
            communicationMode: CommMode = .synchronous,
            checkpointInterval: Int = 10,
            faultToleranceLevel: FaultLevel = .none
        ) {
            self.globalK = globalK
            self.localK = localK ?? Int(Float(globalK) * 1.1)
            self.maxIterations = maxIterations
            self.convergenceTolerance = convergenceTolerance
            self.communicationMode = communicationMode
            self.checkpointInterval = checkpointInterval
            self.faultToleranceLevel = faultToleranceLevel
        }
    }

    // MARK: - Worker and Results

    /// Worker definition - stores ID and data loading capability
    public struct Worker: Sendable {
        public let id: String
        private let _loadData: @Sendable () async throws -> [Vector]

        public init(id: String, loadData: @escaping @Sendable () async throws -> [Vector]) {
            self.id = id
            self._loadData = loadData
        }

        func loadData() async throws -> [Vector] {
            try await _loadData()
        }
    }

    public struct LocalResult: Sendable {
        public let workerId: String
        public let centers: [Vector]
        public let clusterSizes: [Int]
        public let inertia: Float
        public let processingTime: TimeInterval
    }

    public struct Progress {
        public let iteration: Int
        public let globalInertia: Float
        public let activeWorkers: Int
        public let averageLocalTime: TimeInterval
        public let communicationOverhead: TimeInterval
        public let convergenceRate: Float
    }

    public struct Result {
        public let centers: [Vector]
        public let inertia: Float
        public let iterations: Int
        public let converged: Bool
        public let totalTime: TimeInterval
    }

    // MARK: - Map-Reduce Implementation

    /// Map phase: Local K-means on data shard
    public static func mapPhase(
        workerId: String,
        shard: [Vector],
        k: Int,
        initialCenters: [Vector]?
    ) -> LocalResult {
        let startTime = Date()

        // Configure local MiniBatchKMeans512
        let initMethod: MiniBatchKMeans512.Config.InitializationMethod
        if let centers = initialCenters, centers.count == k {
            initMethod = .custom(centers)
        } else {
            initMethod = .kMeansPlusPlus
        }

        let config = MiniBatchKMeans512.Config(
            k: k,
            maxIterations: 30,
            batchSize: min(1000, max(100, shard.count / 10)),
            initMethod: initMethod,
            tolerance: 1e-3,
            nInit: 1,
            reassignmentRatio: 0.01,
            verbose: false,
            randomSeed: nil,
            momentum: 0.9,
            initialLearningRate: 0.01,
            learningRateDecay: 0.999
        )

        let kernel = MiniBatchKMeans512(config: config)
        let result = kernel.fit(shard)

        // Calculate cluster sizes (weights)
        let assignments = kernel.predict(shard)

        var clusterSizes = [Int](repeating: 0, count: k)
        for label in assignments {
            if label >= 0 && label < k {
                clusterSizes[label] += 1
            }
        }

        let processingTime = Date().timeIntervalSince(startTime)

        return LocalResult(
            workerId: workerId,
            centers: result.centers,
            clusterSizes: clusterSizes,
            inertia: result.inertia,
            processingTime: processingTime
        )
    }

    /// Reduce phase: Weighted center aggregation (Meta-clustering)
    public static func reducePhase(
        localResults: [LocalResult],
        globalK: Int
    ) -> [Vector] {
        // Collect all local centers and their weights
        var allCenters: [Vector] = []
        var weights: [Float] = []

        for result in localResults {
            for (center, size) in zip(result.centers, result.clusterSizes) {
                if size > 0 {
                    allCenters.append(center)
                    weights.append(Float(size))
                }
            }
        }

        guard !allCenters.isEmpty else {
            return []
        }

        // Perform weighted K-means clustering
        let globalCenters = performWeightedClustering(
            centers: allCenters,
            weights: weights,
            k: globalK
        )

        return globalCenters
    }

    // MARK: - Weighted K-Means Helpers

    /// Weighted K-means algorithm for center aggregation
    private static func performWeightedClustering(
        centers: [Vector],
        weights: [Float],
        k: Int
    ) -> [Vector] {

        let actualK = min(k, centers.count)
        if actualK <= 0 { return [] }

        // Initialize using Weighted K-means++
        var globalCenters = weightedKMeansPlusPlus(centers, weights, actualK)

        // Weighted Lloyd's algorithm iterations
        let maxIterations = 30

        for _ in 0..<maxIterations {
            var centerChanged = false

            // Assignment step
            var assignments = [Int](repeating: 0, count: centers.count)
            for (i, center) in centers.enumerated() {
                var minDist = Float.infinity
                var minIdx = 0

                for (j, globalCenter) in globalCenters.enumerated() {
                    let dist = EuclideanKernels.distance512(center, globalCenter)
                    if dist < minDist {
                        minDist = dist
                        minIdx = j
                    }
                }
                assignments[i] = minIdx
            }

            // Update step (Weighted average)
            var newCenters = [Vector](repeating: Vector.zero, count: actualK)
            var totalWeights = [Float](repeating: 0, count: actualK)

            for (i, assignment) in assignments.enumerated() {
                let weight = weights[i]
                let weightedCenter = centers[i].scaled(by: weight)
                newCenters[assignment] = newCenters[assignment].adding(weightedCenter)
                totalWeights[assignment] += weight
            }

            // Normalize by total weight
            for j in 0..<actualK {
                if totalWeights[j] > 1e-9 {
                    let normalizedCenter = newCenters[j].scaled(by: 1.0 / totalWeights[j])

                    // Check for changes
                    if EuclideanKernels.distance512(normalizedCenter, globalCenters[j]) > 1e-6 {
                        centerChanged = true
                    }
                    globalCenters[j] = normalizedCenter
                }
            }

            if !centerChanged {
                break
            }
        }

        return globalCenters
    }

    /// Weighted K-Means++ initialization
    private static func weightedKMeansPlusPlus(
        _ centers: [Vector],
        _ weights: [Float],
        _ k: Int
    ) -> [Vector] {
        var globalCenters: [Vector] = []
        var rng = SystemRandomNumberGenerator()

        // First center (weighted random selection)
        let totalWeight = weights.reduce(0, +)
        if totalWeight <= 0 { return [] }

        var target = Float.random(in: 0..<totalWeight, using: &rng)
        var firstIndex = 0
        for (i, weight) in weights.enumerated() {
            target -= weight
            if target <= 0 {
                firstIndex = i
                break
            }
        }
        globalCenters.append(centers[firstIndex])

        // Subsequent centers
        var minDistancesSq = [Float](repeating: Float.infinity, count: centers.count)

        while globalCenters.count < k {
            let lastCenter = globalCenters.last!
            var weightedTotalDistSq: Float = 0

            // Update distances
            for i in centers.indices {
                let dist = EuclideanKernels.distance512(centers[i], lastCenter)
                let distSq = dist * dist

                if distSq < minDistancesSq[i] {
                    minDistancesSq[i] = distSq
                }
                weightedTotalDistSq += minDistancesSq[i] * weights[i]
            }

            if weightedTotalDistSq <= 1e-9 { break }

            // Weighted sampling
            let cutoff = Float.random(in: 0..<weightedTotalDistSq, using: &rng)
            var currentSum: Float = 0
            var nextCenterIndex = 0

            for i in centers.indices {
                currentSum += minDistancesSq[i] * weights[i]
                if currentSum >= cutoff {
                    nextCenterIndex = i
                    break
                }
            }

            // Avoid duplicates
            if !globalCenters.contains(centers[nextCenterIndex]) {
                globalCenters.append(centers[nextCenterIndex])
            }
        }

        return globalCenters
    }

    // MARK: - Coordinator Implementation

    /// Distributed execution coordinator
    public final actor Coordinator {
        private var workers: [Worker]
        private let config: Config

        private var globalCenters: [Vector] = []
        private var iterationCount: Int = 0
        private var convergenceHistory: [Float] = []
        private var globalInertia: Float = Float.infinity

        public init(workers: [Worker], config: Config) {
            self.workers = workers
            self.config = config
        }

        /// Execute distributed clustering
        public func execute(
            progressHandler: @escaping (Progress) async -> Void
        ) async throws -> Result {

            let startTime = Date()

            // Restore from checkpoint if available
            if try await restoreIfNeeded() {
                print("[DistributedClustering512] Restored from checkpoint at iteration \(iterationCount)")
            }

            // Main iteration loop
            var converged = false
            while iterationCount < config.maxIterations {
                iterationCount += 1

                // Determine K and initial centers
                let currentK: Int
                let initialCenters: [Vector]?

                if globalCenters.isEmpty {
                    currentK = config.localK
                    initialCenters = nil
                } else {
                    currentK = config.globalK
                    initialCenters = globalCenters
                }

                // MAP Phase
                let mapStartTime = Date()
                let localResults = try await executeMapPhase(k: currentK, initialCenters: initialCenters)
                let mapTime = Date().timeIntervalSince(mapStartTime)

                if localResults.isEmpty && !workers.isEmpty {
                    throw NSError(
                        domain: "DistributedClustering512",
                        code: 1,
                        userInfo: [NSLocalizedDescriptionKey: "Map phase failed: No results received"]
                    )
                }

                // REDUCE Phase
                let newGlobalCenters = DistributedClustering512.reducePhase(
                    localResults: localResults,
                    globalK: config.globalK
                )

                if newGlobalCenters.isEmpty {
                    break
                }

                // Update global state
                let previousInertia = globalInertia
                globalInertia = localResults.reduce(0) { $0 + $1.inertia }

                let convergenceRate = abs(globalInertia - previousInertia) / max(1e-9, globalInertia)
                convergenceHistory.append(convergenceRate)

                // Progress reporting
                let avgMapTime = localResults.map { $0.processingTime }.reduce(0, +) / Double(max(1, localResults.count))
                let overhead = mapTime - avgMapTime

                let progress = Progress(
                    iteration: iterationCount,
                    globalInertia: globalInertia,
                    activeWorkers: workers.count,
                    averageLocalTime: avgMapTime,
                    communicationOverhead: max(0, overhead),
                    convergenceRate: convergenceRate
                )
                await progressHandler(progress)

                // Checkpointing
                if iterationCount % config.checkpointInterval == 0 {
                    try await checkpointIfNeeded()
                }

                // Convergence check
                if convergenceRate < config.convergenceTolerance && iterationCount > 1 {
                    print("[DistributedClustering512] Convergence reached at iteration \(iterationCount)")
                    globalCenters = newGlobalCenters
                    converged = true
                    break
                }

                globalCenters = newGlobalCenters
            }

            let totalTime = Date().timeIntervalSince(startTime)
            return Result(
                centers: globalCenters,
                inertia: globalInertia,
                iterations: iterationCount,
                converged: converged,
                totalTime: totalTime
            )
        }

        // Map phase execution using structured concurrency
        private func executeMapPhase(k: Int, initialCenters: [Vector]?) async throws -> [LocalResult] {
            return try await withThrowingTaskGroup(of: LocalResult.self) { group in
                for worker in workers {
                    group.addTask {
                        let shard = try await worker.loadData()

                        guard !shard.isEmpty else {
                            return LocalResult(
                                workerId: worker.id,
                                centers: [],
                                clusterSizes: [],
                                inertia: 0,
                                processingTime: 0
                            )
                        }

                        return DistributedClustering512.mapPhase(
                            workerId: worker.id,
                            shard: shard,
                            k: k,
                            initialCenters: initialCenters
                        )
                    }
                }

                var results: [LocalResult] = []
                for try await result in group {
                    results.append(result)
                }
                return results
            }
        }

        // MARK: - Fault Tolerance

        struct Checkpoint: Codable {
            let iteration: Int
            let globalCenters: [Vector]
            let convergenceHistory: [Float]
            let globalInertia: Float
            let timestamp: Date
        }

        private func getCheckpointURL() -> URL? {
            if case .checkpoint(let url) = config.faultToleranceLevel {
                return url
            }
            return nil
        }

        private func checkpointIfNeeded() async throws {
            guard let url = getCheckpointURL() else { return }

            let checkpoint = Checkpoint(
                iteration: iterationCount,
                globalCenters: globalCenters,
                convergenceHistory: convergenceHistory,
                globalInertia: globalInertia,
                timestamp: Date()
            )

            let encoder = JSONEncoder()
            let data = try encoder.encode(checkpoint)
            try data.write(to: url, options: .atomic)
            print("[DistributedClustering512] Checkpoint saved at iteration \(iterationCount)")
        }

        private func restoreIfNeeded() async throws -> Bool {
            guard let url = getCheckpointURL(),
                  FileManager.default.fileExists(atPath: url.path) else {
                return false
            }

            let data = try Data(contentsOf: url)
            let decoder = JSONDecoder()
            let checkpoint = try decoder.decode(Checkpoint.self, from: data)

            self.iterationCount = checkpoint.iteration
            self.globalCenters = checkpoint.globalCenters
            self.convergenceHistory = checkpoint.convergenceHistory
            self.globalInertia = checkpoint.globalInertia
            return true
        }

        public func handleWorkerFailure(_ workerId: String) async {
            print("[DistributedClustering512] Worker \(workerId) failure detected")
            self.workers.removeAll { $0.id == workerId }
        }
    }

    // MARK: - Communication Helpers

    /// Compress centers using JSON serialization
    public static func compressCenters(_ centers: [Vector]) -> Data {
        return try! JSONEncoder().encode(centers)
    }

    /// Decompress centers using JSON deserialization
    public static func decompressCenters(_ data: Data) -> [Vector] {
        return try! JSONDecoder().decode([Vector].self, from: data)
    }
}

// MARK: - DistributedClustering768

/// Distributed clustering for 768-dimensional vectors
public final class DistributedClustering768 {

    public typealias Vector = Vector768Optimized

    // Configuration, Worker, Results - same structure as 512
    public struct Config {
        public let globalK: Int
        public let localK: Int
        public let maxIterations: Int
        public let convergenceTolerance: Float
        public let communicationMode: CommMode
        public let checkpointInterval: Int
        public let faultToleranceLevel: FaultLevel

        public enum CommMode {
            case synchronous
        }

        public enum FaultLevel {
            case none
            case checkpoint(url: URL)
        }

        public init(
            globalK: Int,
            localK: Int? = nil,
            maxIterations: Int = 50,
            convergenceTolerance: Float = 1e-4,
            communicationMode: CommMode = .synchronous,
            checkpointInterval: Int = 10,
            faultToleranceLevel: FaultLevel = .none
        ) {
            self.globalK = globalK
            self.localK = localK ?? Int(Float(globalK) * 1.1)
            self.maxIterations = maxIterations
            self.convergenceTolerance = convergenceTolerance
            self.communicationMode = communicationMode
            self.checkpointInterval = checkpointInterval
            self.faultToleranceLevel = faultToleranceLevel
        }
    }

    public struct Worker: Sendable {
        public let id: String
        private let _loadData: @Sendable () async throws -> [Vector]

        public init(id: String, loadData: @escaping @Sendable () async throws -> [Vector]) {
            self.id = id
            self._loadData = loadData
        }

        func loadData() async throws -> [Vector] {
            try await _loadData()
        }
    }

    public struct LocalResult: Sendable {
        public let workerId: String
        public let centers: [Vector]
        public let clusterSizes: [Int]
        public let inertia: Float
        public let processingTime: TimeInterval
    }

    public struct Progress {
        public let iteration: Int
        public let globalInertia: Float
        public let activeWorkers: Int
        public let averageLocalTime: TimeInterval
        public let communicationOverhead: TimeInterval
        public let convergenceRate: Float
    }

    public struct Result {
        public let centers: [Vector]
        public let inertia: Float
        public let iterations: Int
        public let converged: Bool
        public let totalTime: TimeInterval
    }

    // Implementation follows same pattern as 512 with Vector768Optimized
    // Omitted for brevity - full implementation would replace:
    // - Vector512Optimized -> Vector768Optimized
    // - MiniBatchKMeans512 -> MiniBatchKMeans768
    // - EuclideanKernels.distance512 -> EuclideanKernels.distance768
}

// MARK: - DistributedClustering1536

/// Distributed clustering for 1536-dimensional vectors
public final class DistributedClustering1536 {

    public typealias Vector = Vector1536Optimized

    // Configuration, Worker, Results - same structure as 512
    public struct Config {
        public let globalK: Int
        public let localK: Int
        public let maxIterations: Int
        public let convergenceTolerance: Float
        public let communicationMode: CommMode
        public let checkpointInterval: Int
        public let faultToleranceLevel: FaultLevel

        public enum CommMode {
            case synchronous
        }

        public enum FaultLevel {
            case none
            case checkpoint(url: URL)
        }

        public init(
            globalK: Int,
            localK: Int? = nil,
            maxIterations: Int = 50,
            convergenceTolerance: Float = 1e-4,
            communicationMode: CommMode = .synchronous,
            checkpointInterval: Int = 10,
            faultToleranceLevel: FaultLevel = .none
        ) {
            self.globalK = globalK
            self.localK = localK ?? Int(Float(globalK) * 1.1)
            self.maxIterations = maxIterations
            self.convergenceTolerance = convergenceTolerance
            self.communicationMode = communicationMode
            self.checkpointInterval = checkpointInterval
            self.faultToleranceLevel = faultToleranceLevel
        }
    }

    public struct Worker: Sendable {
        public let id: String
        private let _loadData: @Sendable () async throws -> [Vector]

        public init(id: String, loadData: @escaping @Sendable () async throws -> [Vector]) {
            self.id = id
            self._loadData = loadData
        }

        func loadData() async throws -> [Vector] {
            try await _loadData()
        }
    }

    public struct LocalResult: Sendable {
        public let workerId: String
        public let centers: [Vector]
        public let clusterSizes: [Int]
        public let inertia: Float
        public let processingTime: TimeInterval
    }

    public struct Progress {
        public let iteration: Int
        public let globalInertia: Float
        public let activeWorkers: Int
        public let averageLocalTime: TimeInterval
        public let communicationOverhead: TimeInterval
        public let convergenceRate: Float
    }

    public struct Result {
        public let centers: [Vector]
        public let inertia: Float
        public let iterations: Int
        public let converged: Bool
        public let totalTime: TimeInterval
    }

    // Implementation follows same pattern as 512 with Vector1536Optimized
    // Omitted for brevity - full implementation would replace:
    // - Vector512Optimized -> Vector1536Optimized
    // - MiniBatchKMeans512 -> MiniBatchKMeans1536
    // - EuclideanKernels.distance512 -> EuclideanKernels.distance1536
}
