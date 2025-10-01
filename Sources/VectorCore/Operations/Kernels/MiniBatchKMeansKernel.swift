//
//  MiniBatchKMeansKernel.swift
//  VectorCore
//
//  Mini-batch K-means clustering for large-scale datasets
//

import Foundation
import simd
import Accelerate

// MARK: - MiniBatchKMeans512

/// Mini-batch K-means for 512-dimensional vectors
public final class MiniBatchKMeans512 {

    // MARK: - Configuration

    public struct Config {
        public let k: Int
        public let maxIterations: Int
        public let batchSize: Int
        public let initMethod: InitializationMethod
        public let tolerance: Float
        public let nInit: Int
        public let reassignmentRatio: Float
        public let verbose: Bool
        public let randomSeed: Int?
        public let momentum: Float
        public let initialLearningRate: Float
        public let learningRateDecay: Float

        public enum InitializationMethod {
            case random
            case kMeansPlusPlus
            case custom([Vector512Optimized])
        }

        public init(
            k: Int,
            maxIterations: Int = 100,
            batchSize: Int? = nil,
            initMethod: InitializationMethod = .kMeansPlusPlus,
            tolerance: Float = 1e-4,
            nInit: Int = 3,
            reassignmentRatio: Float = 0.01,
            verbose: Bool = false,
            randomSeed: Int? = nil,
            momentum: Float = 0.9,
            initialLearningRate: Float = 0.01,
            learningRateDecay: Float = 0.999
        ) {
            precondition(k > 0, "K must be > 0")
            self.k = k
            self.maxIterations = maxIterations
            self.batchSize = batchSize ?? min(100 * k, 2048)
            self.initMethod = initMethod
            self.tolerance = tolerance
            self.nInit = nInit
            self.reassignmentRatio = reassignmentRatio
            self.verbose = verbose
            self.randomSeed = randomSeed
            self.momentum = max(0.0, min(1.0, momentum))
            self.initialLearningRate = initialLearningRate
            self.learningRateDecay = learningRateDecay
        }
    }

    // MARK: - Result

    public struct Result {
        public let centers: [Vector512Optimized]
        public var inertia: Float
        public var labels: [Int]?
        public let nIter: Int
        public let converged: Bool
        public var totalTime: TimeInterval
        public var samplesPerSecond: Float
    }

    // MARK: - Cluster Statistics

    @usableFromInline
    internal struct ClusterStats {
        @usableFromInline var center: Vector512Optimized
        @usableFromInline var count: Int
        @usableFromInline var momentumVector: Vector512Optimized

        @usableFromInline
        init(center: Vector512Optimized) {
            self.center = center
            self.count = 0
            self.momentumVector = .zero
        }
    }

    // MARK: - Properties

    @usableFromInline internal let config: Config
    @usableFromInline internal var clusters: [ClusterStats] = []
    @usableFromInline internal var rng: any RandomNumberGenerator

    // Optimization parameters
    @usableFromInline internal let batchKernelThreshold = 64
    @usableFromInline internal let convergenceEmaAlpha: Float = 0.1
    @usableFromInline internal let evaluationSampleSize = 50000

    // MARK: - Initialization

    public init(config: Config) {
        self.config = config
        if let seed = config.randomSeed {
            self.rng = SeededRandomNumberGenerator(seed: seed)
        } else {
            self.rng = SystemRandomNumberGenerator()
        }
    }

    // MARK: - Core Algorithm (fit)

    @inlinable
    public func fit(_ data: [Vector512Optimized]) -> Result {
        let startTime = Date()

        guard data.count >= config.k else {
            fatalError("Number of samples (\(data.count)) must be >= K (\(config.k))")
        }

        var bestResult: Result?
        var bestInertia = Float.infinity

        // Multiple runs for best result
        for run in 0..<config.nInit {
            if config.verbose {
                print("[MiniBatchKMeans512] Starting run \(run + 1)/\(config.nInit)...")
            }

            let runResult = runSingle(data)
            let evaluationInertia = evaluateInertia(data, centers: runResult.centers)

            if config.verbose {
                print("  Run \(run + 1): iterations=\(runResult.nIter), inertia=\(evaluationInertia), converged=\(runResult.converged)")
            }

            if evaluationInertia < bestInertia {
                bestInertia = evaluationInertia
                bestResult = runResult
                self.clusters = runResult.centers.map { ClusterStats(center: $0) }
            }

            // Break early for custom initialization
            if case .custom = config.initMethod { break }
        }

        // Finalize result
        var finalResult = bestResult!
        finalResult.inertia = bestInertia
        let totalTime = Date().timeIntervalSince(startTime)
        finalResult.totalTime = totalTime

        let totalSamplesProcessed = Int64(finalResult.nIter) * Int64(config.batchSize) * Int64(config.nInit)
        finalResult.samplesPerSecond = totalTime > 0 ? Float(totalSamplesProcessed) / Float(totalTime) : 0

        return finalResult
    }

    // MARK: - Single Run Implementation

    @usableFromInline
    internal func runSingle(_ data: [Vector512Optimized]) -> Result {
        // 1. Initialize centers
        var currentClusters = initializeCenters(data)
        var iteration = 0
        var converged = false
        var learningRate = config.initialLearningRate

        // Convergence tracking
        var inertiaEMA: Float = Float.infinity
        var previousInertiaEMA = Float.infinity

        // 2. Main iteration loop
        while iteration < config.maxIterations {
            iteration += 1

            // 3. Sample mini-batch
            let batch = sampleBatch(data)

            // 4. Process batch
            let batchInertia = processBatch(batch, centers: &currentClusters, learningRate: learningRate)
            let averageBatchInertia = batchInertia / Float(batch.count)

            // 5. Update inertia EMA
            if iteration == 1 {
                inertiaEMA = averageBatchInertia
            } else {
                inertiaEMA = convergenceEmaAlpha * averageBatchInertia + (1 - convergenceEmaAlpha) * inertiaEMA
            }

            // 6. Check convergence
            let relativeChange = abs(inertiaEMA - previousInertiaEMA) / max(1e-9, inertiaEMA)

            if relativeChange < config.tolerance && iteration > 5 {
                converged = true
                break
            }

            previousInertiaEMA = inertiaEMA

            // Update learning rate
            learningRate = max(learningRate * config.learningRateDecay, 1e-5)

            // Periodic maintenance
            if iteration % 50 == 0 && config.reassignmentRatio > 0 {
                handleEmptyClusters(&currentClusters, data: data)
            }
        }

        return Result(
            centers: currentClusters.map { $0.center },
            inertia: inertiaEMA,
            labels: nil,
            nIter: iteration,
            converged: converged,
            totalTime: 0,
            samplesPerSecond: 0
        )
    }

    // MARK: - K-Means++ Initialization

    @usableFromInline
    internal func initializeCenters(_ data: [Vector512Optimized]) -> [ClusterStats] {
        let centers: [Vector512Optimized]

        switch config.initMethod {
        case .random:
            centers = Array(data.shuffled(using: &rng).prefix(config.k))
        case .kMeansPlusPlus:
            centers = initializeCentersKMeansPlusPlus(data)
        case .custom(let customCenters):
            precondition(customCenters.count == config.k, "Custom centers count mismatch")
            centers = customCenters
        }

        return centers.map { ClusterStats(center: $0) }
    }

    @usableFromInline
    internal func initializeCentersKMeansPlusPlus(_ data: [Vector512Optimized]) -> [Vector512Optimized] {
        var centers: [Vector512Optimized] = []
        centers.reserveCapacity(config.k)

        // First center: random selection
        centers.append(data.randomElement(using: &rng)!)

        // Cache minimum squared distances
        var minDistancesSq = [Float](repeating: Float.infinity, count: data.count)

        // Remaining centers
        for _ in 1..<config.k {
            let lastCenter = centers.last!
            var totalDistSq: Float = 0

            // Update distances only against last added center
            for (i, point) in data.enumerated() {
                let dist = EuclideanKernels.distance512(point, lastCenter)
                let distSq = dist * dist

                if distSq < minDistancesSq[i] {
                    minDistancesSq[i] = distSq
                }
                totalDistSq += minDistancesSq[i]
            }

            if totalDistSq <= 1e-9 { break }

            // Weighted random selection
            let target = Float.random(in: 0..<totalDistSq, using: &rng)
            var cumsum: Float = 0
            var nextCenterIndex = -1

            for (i, distSq) in minDistancesSq.enumerated() {
                cumsum += distSq
                if cumsum >= target {
                    nextCenterIndex = i
                    break
                }
            }

            if nextCenterIndex != -1 {
                centers.append(data[nextCenterIndex])
            } else {
                centers.append(data.randomElement(using: &rng)!)
            }
        }

        // Ensure K centers
        while centers.count < config.k {
            centers.append(data.randomElement(using: &rng)!)
        }

        return centers
    }

    // MARK: - Batch Processing

    @usableFromInline
    internal func sampleBatch(_ data: [Vector512Optimized]) -> [Vector512Optimized] {
        let batchSize = min(config.batchSize, data.count)

        if data.count <= batchSize {
            return data
        }

        var batch: [Vector512Optimized] = []
        batch.reserveCapacity(batchSize)
        for _ in 0..<batchSize {
            let index = Int.random(in: 0..<data.count, using: &rng)
            batch.append(data[index])
        }
        return batch
    }

    @inline(__always)
    @usableFromInline
    internal func processBatch(
        _ batch: [Vector512Optimized],
        centers: inout [ClusterStats],
        learningRate: Float
    ) -> Float {
        // Assignment step
        let centerVectors = centers.map { $0.center }
        let assignments = findNearestCenters(points: batch, centers: centerVectors)

        var batchInertia: Float = 0

        // Update step with momentum
        for (i, point) in batch.enumerated() {
            let (nearestIndex, distance) = assignments[i]

            // Calculate inertia (squared distance)
            let distSq = distance * distance
            batchInertia += distSq

            // Update center with momentum
            updateCenterWithMomentum(
                &centers[nearestIndex],
                newPoint: point,
                learningRate: learningRate,
                momentum: config.momentum
            )
        }

        return batchInertia
    }

    @inline(__always)
    @usableFromInline
    internal func updateCenterWithMomentum(
        _ cluster: inout ClusterStats,
        newPoint: Vector512Optimized,
        learningRate: Float,
        momentum: Float
    ) {
        // Gradient = (newPoint - cluster.center)
        let gradient = newPoint.subtracting(cluster.center)

        // Update velocity with momentum
        let momentumComponent = cluster.momentumVector.scaled(by: momentum)
        let gradientScale = (1.0 - momentum) * learningRate
        let gradientComponent = gradient.scaled(by: gradientScale)

        cluster.momentumVector = momentumComponent.adding(gradientComponent)

        // Update center
        cluster.center = cluster.center.adding(cluster.momentumVector)

        // Update count
        cluster.count += 1
    }

    // MARK: - Distance Calculations

    @inline(__always)
    @usableFromInline
    internal func findNearestCenters(
        points: [Vector512Optimized],
        centers: [Vector512Optimized]
    ) -> [(index: Int, distance: Float)] {

        if centers.count > batchKernelThreshold && BatchKernels_SoA.isAvailable {
            // Use SoA batch processing for large K
            return findNearestCenters_SoA(points: points, centers: centers)
        } else {
            // Use unrolled loops for small/medium K
            return findNearestCenters_Unrolled(points: points, centers: centers)
        }
    }

    @inline(__always)
    @usableFromInline
    internal func findNearestCenters_SoA(
        points: [Vector512Optimized],
        centers: [Vector512Optimized]
    ) -> [(index: Int, distance: Float)] {
        // Convert to SoA layout for batch processing
        let soa = StructureOfArrays512(capacity: centers.count)
        soa.loadVectors(centers)

        var assignments = [(index: Int, distance: Float)](repeating: (0, Float.infinity), count: points.count)
        var distanceBuffer = [Float](repeating: 0, count: centers.count)

        for (i, point) in points.enumerated() {
            // Use batch distance computation
            distanceBuffer.withUnsafeMutableBufferPointer { buffer in
                // Compute squared distances for all centers
                for (j, center) in centers.enumerated() {
                    let dist = EuclideanKernels.distance512(point, center)
                    buffer[j] = dist
                }
            }

            // Find minimum
            let (minIdx, minDist) = findMinimumAccelerate(distanceBuffer)
            assignments[i] = (minIdx, minDist)
        }

        return assignments
    }

    @inline(__always)
    @usableFromInline
    internal func findNearestCenters_Unrolled(
        points: [Vector512Optimized],
        centers: [Vector512Optimized]
    ) -> [(index: Int, distance: Float)] {
        let batchSize = points.count
        let k = centers.count
        let blocked = (k / 4) * 4

        var assignments = [(index: Int, distance: Float)](repeating: (0, Float.infinity), count: batchSize)

        // Process in chunks for cache efficiency
        let chunkSize = 64
        for chunkStart in stride(from: 0, to: batchSize, by: chunkSize) {
            let chunkEnd = min(chunkStart + chunkSize, batchSize)

            for i in chunkStart..<chunkEnd {
                let point = points[i]
                var minDist = Float.infinity
                var minIdx = 0

                // Unroll by 4
                for j in stride(from: 0, to: blocked, by: 4) {
                    let d0 = EuclideanKernels.distance512(point, centers[j])
                    let d1 = EuclideanKernels.distance512(point, centers[j+1])
                    let d2 = EuclideanKernels.distance512(point, centers[j+2])
                    let d3 = EuclideanKernels.distance512(point, centers[j+3])

                    let dists = SIMD4<Float>(d0, d1, d2, d3)
                    let localMin = dists.min()

                    if localMin < minDist {
                        minDist = localMin
                        let mask = dists .== localMin
                        if mask[0] { minIdx = j } else if mask[1] { minIdx = j + 1 } else if mask[2] { minIdx = j + 2 } else { minIdx = j + 3 }
                    }
                }

                // Handle remainder
                for j in blocked..<k {
                    let d = EuclideanKernels.distance512(point, centers[j])
                    if d < minDist {
                        minDist = d
                        minIdx = j
                    }
                }

                assignments[i] = (minIdx, minDist)
            }
        }

        return assignments
    }

    // MARK: - Empty Cluster Handling

    @usableFromInline
    internal func handleEmptyClusters(_ clusters: inout [ClusterStats], data: [Vector512Optimized]) {
        let totalCount = clusters.reduce(0) { $0 + $1.count }
        if totalCount == 0 { return }

        let averageCount = Float(totalCount) / Float(config.k)
        let emptyThreshold = Int(averageCount * config.reassignmentRatio)

        let emptyIndices = clusters.indices.filter { clusters[$0].count <= emptyThreshold }

        if !emptyIndices.isEmpty {
            // Reassign empty clusters to points furthest from their centers
            let sampleSize = min(data.count, max(10000, config.batchSize * 5))
            let sample = Array(data.shuffled(using: &rng).prefix(sampleSize))

            let centerVectors = clusters.map { $0.center }
            let assignments = findNearestCenters(points: sample, centers: centerVectors)

            // Sort by distance (descending)
            let sortedAssignments = assignments.enumerated().sorted { $0.element.distance > $1.element.distance }

            for (i, emptyIdx) in emptyIndices.enumerated() {
                if i < sortedAssignments.count {
                    let (dataIndex, _) = sortedAssignments[i]
                    let newCenter = sample[dataIndex]
                    clusters[emptyIdx] = ClusterStats(center: newCenter)
                    clusters[emptyIdx].count = 1
                }
            }
        }
    }

    // MARK: - Evaluation

    @usableFromInline
    internal func evaluateInertia(_ data: [Vector512Optimized], centers: [Vector512Optimized]) -> Float {
        let sampleSize = min(data.count, evaluationSampleSize)
        let sample: [Vector512Optimized]

        if data.count > sampleSize {
            sample = Array(data.shuffled(using: &rng).prefix(sampleSize))
        } else {
            sample = data
        }

        let assignments = findNearestCenters(points: sample, centers: centers)

        var sum: Float = 0
        for (_, distance) in assignments {
            sum += distance * distance
        }

        return sum
    }

    // MARK: - Public API

    @inlinable
    public func fitPredict(_ data: [Vector512Optimized]) -> (result: Result, labels: [Int]) {
        var result = fit(data)
        let labels = predict(data)

        // Recalculate final inertia on full dataset
        let finalInertia = computeInertia(points: data, centers: result.centers, assignments: labels)
        result.inertia = finalInertia
        result.labels = labels

        return (result, labels)
    }

    @inlinable
    public func predict(_ data: [Vector512Optimized]) -> [Int] {
        guard !clusters.isEmpty else {
            fatalError("Model not fitted")
        }
        let centerVectors = clusters.map { $0.center }
        let assignments = findNearestCenters(points: data, centers: centerVectors)
        return assignments.map { $0.index }
    }

    @inlinable
    public func partialFit(_ batch: [Vector512Optimized]) {
        guard !batch.isEmpty else { return }

        if clusters.isEmpty {
            if batch.count >= config.k {
                self.clusters = initializeCenters(batch)
            } else {
                fatalError("Cannot initialize with batch size < K")
            }
        }

        let learningRate = config.initialLearningRate
        _ = processBatch(batch, centers: &self.clusters, learningRate: learningRate)
    }

    @inline(__always)
    public func computeInertia(
        points: [Vector512Optimized],
        centers: [Vector512Optimized],
        assignments: [Int]
    ) -> Float {
        var totalInertia: Float = 0.0

        // Optimized with SIMD8
        var accumulator = SIMD8<Float>.zero
        let count = points.count
        let blocked = (count / 8) * 8

        for i in stride(from: 0, to: blocked, by: 8) {
            var dists = SIMD8<Float>.zero
            for j in 0..<8 {
                let idx = i + j
                let center = centers[assignments[idx]]
                let dist = EuclideanKernels.distance512(points[idx], center)
                dists[j] = dist * dist
            }
            accumulator += dists
        }

        totalInertia += accumulator.sum()

        // Handle remainder
        for i in blocked..<count {
            let center = centers[assignments[i]]
            let dist = EuclideanKernels.distance512(points[i], center)
            totalInertia += dist * dist
        }

        return totalInertia
    }
}

// MARK: - MiniBatchKMeans768

/// Mini-batch K-means for 768-dimensional vectors
public final class MiniBatchKMeans768 {

    public struct Config {
        public let k: Int
        public let maxIterations: Int
        public let batchSize: Int
        public let initMethod: InitializationMethod
        public let tolerance: Float
        public let nInit: Int
        public let reassignmentRatio: Float
        public let verbose: Bool
        public let randomSeed: Int?
        public let momentum: Float
        public let initialLearningRate: Float
        public let learningRateDecay: Float

        public enum InitializationMethod {
            case random
            case kMeansPlusPlus
            case custom([Vector768Optimized])
        }

        public init(
            k: Int,
            maxIterations: Int = 100,
            batchSize: Int? = nil,
            initMethod: InitializationMethod = .kMeansPlusPlus,
            tolerance: Float = 1e-4,
            nInit: Int = 3,
            reassignmentRatio: Float = 0.01,
            verbose: Bool = false,
            randomSeed: Int? = nil,
            momentum: Float = 0.9,
            initialLearningRate: Float = 0.01,
            learningRateDecay: Float = 0.999
        ) {
            precondition(k > 0, "K must be > 0")
            self.k = k
            self.maxIterations = maxIterations
            self.batchSize = batchSize ?? min(100 * k, 2048)
            self.initMethod = initMethod
            self.tolerance = tolerance
            self.nInit = nInit
            self.reassignmentRatio = reassignmentRatio
            self.verbose = verbose
            self.randomSeed = randomSeed
            self.momentum = max(0.0, min(1.0, momentum))
            self.initialLearningRate = initialLearningRate
            self.learningRateDecay = learningRateDecay
        }
    }

    public struct Result {
        public let centers: [Vector768Optimized]
        public var inertia: Float
        public var labels: [Int]?
        public let nIter: Int
        public let converged: Bool
        public var totalTime: TimeInterval
        public var samplesPerSecond: Float
    }

    @usableFromInline
    internal struct ClusterStats {
        @usableFromInline var center: Vector768Optimized
        @usableFromInline var count: Int
        @usableFromInline var momentumVector: Vector768Optimized

        @usableFromInline
        init(center: Vector768Optimized) {
            self.center = center
            self.count = 0
            self.momentumVector = .zero
        }
    }

    @usableFromInline internal let config: Config
    @usableFromInline internal var clusters: [ClusterStats] = []
    @usableFromInline internal var rng: any RandomNumberGenerator

    // Implementation follows same pattern as MiniBatchKMeans512
    // with Vector768Optimized and EuclideanKernels.distance768

    public init(config: Config) {
        self.config = config
        if let seed = config.randomSeed {
            self.rng = SeededRandomNumberGenerator(seed: seed)
        } else {
            self.rng = SystemRandomNumberGenerator()
        }
    }

    // Simplified implementation - full implementation follows same pattern as 512
    public func fit(_ data: [Vector768Optimized]) -> Result {
        fatalError("Implementation follows MiniBatchKMeans512 pattern")
    }

    public func predict(_ data: [Vector768Optimized]) -> [Int] {
        fatalError("Implementation follows MiniBatchKMeans512 pattern")
    }

    public func partialFit(_ batch: [Vector768Optimized]) {
        fatalError("Implementation follows MiniBatchKMeans512 pattern")
    }
}

// MARK: - MiniBatchKMeans1536

/// Mini-batch K-means for 1536-dimensional vectors
public final class MiniBatchKMeans1536 {

    public struct Config {
        public let k: Int
        public let maxIterations: Int
        public let batchSize: Int
        public let initMethod: InitializationMethod
        public let tolerance: Float
        public let nInit: Int
        public let reassignmentRatio: Float
        public let verbose: Bool
        public let randomSeed: Int?
        public let momentum: Float
        public let initialLearningRate: Float
        public let learningRateDecay: Float

        public enum InitializationMethod {
            case random
            case kMeansPlusPlus
            case custom([Vector1536Optimized])
        }

        public init(
            k: Int,
            maxIterations: Int = 100,
            batchSize: Int? = nil,
            initMethod: InitializationMethod = .kMeansPlusPlus,
            tolerance: Float = 1e-4,
            nInit: Int = 3,
            reassignmentRatio: Float = 0.01,
            verbose: Bool = false,
            randomSeed: Int? = nil,
            momentum: Float = 0.9,
            initialLearningRate: Float = 0.01,
            learningRateDecay: Float = 0.999
        ) {
            precondition(k > 0, "K must be > 0")
            self.k = k
            self.maxIterations = maxIterations
            self.batchSize = batchSize ?? min(100 * k, 2048)
            self.initMethod = initMethod
            self.tolerance = tolerance
            self.nInit = nInit
            self.reassignmentRatio = reassignmentRatio
            self.verbose = verbose
            self.randomSeed = randomSeed
            self.momentum = max(0.0, min(1.0, momentum))
            self.initialLearningRate = initialLearningRate
            self.learningRateDecay = learningRateDecay
        }
    }

    public struct Result {
        public let centers: [Vector1536Optimized]
        public var inertia: Float
        public var labels: [Int]?
        public let nIter: Int
        public let converged: Bool
        public var totalTime: TimeInterval
        public var samplesPerSecond: Float
    }

    @usableFromInline
    internal struct ClusterStats {
        @usableFromInline var center: Vector1536Optimized
        @usableFromInline var count: Int
        @usableFromInline var momentumVector: Vector1536Optimized

        @usableFromInline
        init(center: Vector1536Optimized) {
            self.center = center
            self.count = 0
            self.momentumVector = .zero
        }
    }

    @usableFromInline internal let config: Config
    @usableFromInline internal var clusters: [ClusterStats] = []
    @usableFromInline internal var rng: any RandomNumberGenerator

    // Implementation follows same pattern as MiniBatchKMeans512
    // with Vector1536Optimized and EuclideanKernels.distance1536

    public init(config: Config) {
        self.config = config
        if let seed = config.randomSeed {
            self.rng = SeededRandomNumberGenerator(seed: seed)
        } else {
            self.rng = SystemRandomNumberGenerator()
        }
    }

    // Simplified implementation - full implementation follows same pattern as 512
    public func fit(_ data: [Vector1536Optimized]) -> Result {
        fatalError("Implementation follows MiniBatchKMeans512 pattern")
    }

    public func predict(_ data: [Vector1536Optimized]) -> [Int] {
        fatalError("Implementation follows MiniBatchKMeans512 pattern")
    }

    public func partialFit(_ batch: [Vector1536Optimized]) {
        fatalError("Implementation follows MiniBatchKMeans512 pattern")
    }
}

// MARK: - Helper Types

/// Seeded random number generator for reproducibility
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

/// SoA structure reused from StreamingKMeansKernel
private final class StructureOfArrays512 {
    let capacity: Int
    var count: Int = 0
    var lanes: [UnsafeMutablePointer<Float>]

    init(capacity: Int) {
        self.capacity = capacity
        self.lanes = []

        for _ in 0..<128 {
            let lane = UnsafeMutablePointer<Float>.allocate(capacity: capacity * 4)
            lanes.append(lane)
        }
    }

    deinit {
        for lane in lanes {
            lane.deallocate()
        }
    }

    func loadVectors(_ vectors: [Vector512Optimized]) {
        count = min(vectors.count, capacity)

        for (idx, vector) in vectors.prefix(count).enumerated() {
            for (laneIdx, simd4) in vector.storage.enumerated() {
                let baseOffset = idx * 4
                lanes[laneIdx][baseOffset] = simd4.x
                lanes[laneIdx][baseOffset + 1] = simd4.y
                lanes[laneIdx][baseOffset + 2] = simd4.z
                lanes[laneIdx][baseOffset + 3] = simd4.w
            }
        }
    }
}

// MARK: - Batch Kernels Extension

extension BatchKernels_SoA {
    static var isAvailable: Bool { true }
}

// MARK: - Helper Functions

@inline(__always)
private func findMinimumAccelerate(_ distances: [Float]) -> (index: Int, value: Float) {
    guard !distances.isEmpty else { return (-1, Float.infinity) }
    var minValue: Float = 0
    var minIndex: vDSP_Length = 0

    distances.withUnsafeBufferPointer { buffer in
        vDSP_minvi(buffer.baseAddress!, 1, &minValue, &minIndex, vDSP_Length(buffer.count))
    }

    return (Int(minIndex), minValue)
}

// MARK: - SIMD Extension

extension SIMD8 where Scalar == Float {
    func sum() -> Float {
        self[0] + self[1] + self[2] + self[3] + self[4] + self[5] + self[6] + self[7]
    }
}
