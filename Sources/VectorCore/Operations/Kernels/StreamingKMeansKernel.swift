//
//  StreamingKMeansKernel.swift
//  VectorCore
//
//  Streaming K-means clustering with bounded memory for large datasets
//

import Foundation
import simd
import Accelerate

// MARK: - StreamingKMeans512

/// Streaming K-means for 512-dimensional vectors with bounded memory
/// Memory complexity: O(K*D + B*D) where K=clusters, D=dimensions, B=batch size
@available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
public actor StreamingKMeans512 {
    /// Configuration for streaming K-means
    public struct Config {
        let k: Int                           // Number of clusters
        let batchSize: Int                   // Streaming batch size
        let learningRate: Float              // Initial learning rate
        let decayFactor: Float               // Learning rate decay per batch
        let convergenceWindow: Int           // Window for convergence check
        let convergenceTolerance: Float      // Relative change threshold
        let maxIterations: Int?              // Optional iteration limit
        let initMethod: InitMethod           // Initialization strategy
        let verbose: Bool                    // Progress reporting

        public enum InitMethod {
            case random(seed: Int?)
            case plusPlus(samples: Int)  // K-means++ with sample size
            case custom([Vector512Optimized])
        }

        public init(
            k: Int,
            batchSize: Int = 1000,
            learningRate: Float = 0.1,
            decayFactor: Float = 0.995,
            convergenceWindow: Int = 10,
            convergenceTolerance: Float = 1e-4,
            maxIterations: Int? = nil,
            initMethod: InitMethod = .plusPlus(samples: 10000),
            verbose: Bool = false
        ) {
            self.k = k
            self.batchSize = batchSize
            self.learningRate = learningRate
            self.decayFactor = decayFactor
            self.convergenceWindow = convergenceWindow
            self.convergenceTolerance = convergenceTolerance
            self.maxIterations = maxIterations
            self.initMethod = initMethod
            self.verbose = verbose
        }
    }

    /// Clustering result
    public struct Result {
        public let centers: [Vector512Optimized]
        public let clusterSizes: [Int]
        public let inertia: Float
        public let converged: Bool
        public let iterations: Int
        public let totalSamples: Int
    }

    // Private state
    private let config: Config
    private var centers: [Vector512Optimized]
    private var clusterSizes: [Int]
    private var clusterSums: [Vector512Optimized]
    private var currentLearningRate: Float
    private var batchesProcessed: Int = 0
    private var totalSamples: Int = 0

    // Convergence monitoring using circular buffer
    private var convergenceHistory: CircularBuffer<Float>
    private var lastInertia: Float = .infinity

    // Pre-allocated buffers for performance
    private let distanceBuffer: BufferWrapper<Float>
    private var soaCache: StructureOfArrays512?

    public init(config: Config) {
        self.config = config
        self.centers = []
        self.clusterSizes = Array(repeating: 0, count: config.k)
        self.clusterSums = Array(repeating: .zero, count: config.k)
        self.currentLearningRate = config.learningRate
        self.convergenceHistory = CircularBuffer<Float>(capacity: config.convergenceWindow)

        // Pre-allocate distance buffer
        self.distanceBuffer = BufferWrapper<Float>(capacity: config.k)
    }

    // MARK: - Public API

    /// Initialize centers from initial data batch
    public func initialize(from data: [Vector512Optimized]) async {
        switch config.initMethod {
        case .random(let seed):
            initializeRandom(from: data, seed: seed)
        case .plusPlus(let samples):
            initializeKMeansPlusPlus(from: data, samples: samples)
        case .custom(let customCenters):
            centers = customCenters
        }

        if config.verbose {
            print("[StreamingKMeans512] Initialized \(config.k) centers")
        }
    }

    /// Process a streaming batch
    public func processBatch(_ batch: [Vector512Optimized]) async -> Float {
        guard !batch.isEmpty else { return lastInertia }

        let startTime = CFAbsoluteTimeGetCurrent()
        var batchInertia: Float = 0

        // Reuse SoA cache for large K
        if config.k > 64 && soaCache == nil {
            soaCache = StructureOfArrays512(capacity: config.k)
        }

        // Convert centers to SoA for batch processing if beneficial
        if let soa = soaCache, config.k > 64 {
            soa.loadVectors(centers)

            // Process batch using SoA distances
            for point in batch {
                let (nearest, distance) = findNearestCenterSoA(point, soa: soa)
                updateCluster(nearest, with: point)
                batchInertia += distance * distance
            }
        } else {
            // Direct processing for small K
            for point in batch {
                let (nearest, distance) = findNearestCenter(point)
                updateCluster(nearest, with: point)
                batchInertia += distance * distance
            }
        }

        // Update learning rate
        currentLearningRate *= config.decayFactor
        batchesProcessed += 1
        totalSamples += batch.count

        // Track convergence
        let avgInertia = batchInertia / Float(batch.count)
        convergenceHistory.append(avgInertia)
        lastInertia = avgInertia

        if config.verbose {
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            let throughput = Float(batch.count) / Float(elapsed)
            print("[StreamingKMeans512] Batch \(batchesProcessed): inertia=\(avgInertia), rate=\(currentLearningRate), throughput=\(throughput) vec/s")
        }

        return avgInertia
    }

    /// Check convergence based on inertia history
    public func hasConverged() async -> Bool {
        guard convergenceHistory.isFull else { return false }

        if let maxIter = config.maxIterations, batchesProcessed >= maxIter {
            return true
        }

        let values = convergenceHistory.values
        let mean = values.reduce(0, +) / Float(values.count)
        let variance = values.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(values.count)
        let relativeChange = sqrt(variance) / (abs(mean) + 1e-10)

        return relativeChange < config.convergenceTolerance
    }

    /// Get current clustering result
    public func getResult() async -> Result {
        Result(
            centers: centers,
            clusterSizes: clusterSizes,
            inertia: lastInertia,
            converged: await hasConverged(),
            iterations: batchesProcessed,
            totalSamples: totalSamples
        )
    }

    /// Predict cluster assignments for new data
    public func predict(_ data: [Vector512Optimized]) async -> [Int] {
        var assignments = [Int]()
        assignments.reserveCapacity(data.count)

        for point in data {
            let (nearest, _) = findNearestCenter(point)
            assignments.append(nearest)
        }

        return assignments
    }

    // MARK: - Private Implementation

    private func initializeRandom(from data: [Vector512Optimized], seed: Int?) {
        if let seed = seed {
            var rng = ArithmeticRandomNumberGenerator(seed: seed)
            let indices = (0..<data.count).shuffled(using: &rng).prefix(config.k)
            centers = indices.map { data[$0] }
        } else {
            var rng = SystemRandomNumberGenerator()
            let indices = (0..<data.count).shuffled(using: &rng).prefix(config.k)
            centers = indices.map { data[$0] }
        }
    }

    private func initializeKMeansPlusPlus(from data: [Vector512Optimized], samples: Int) {
        guard !data.isEmpty else { return }

        let sampleSize = min(samples, data.count)
        let sampledData = data.count > samples ? Array(data.shuffled().prefix(sampleSize)) : data

        // First center: random
        centers = [sampledData.randomElement()!]

        // Remaining centers: weighted by squared distance
        for _ in 1..<config.k {
            var weights = [Float]()
            weights.reserveCapacity(sampledData.count)

            for point in sampledData {
                let minDist = centers.map { distance512(point, $0) }.min()!
                weights.append(minDist * minDist)
            }

            // Weighted random selection
            let totalWeight = weights.reduce(0, +)
            var target = Float.random(in: 0..<totalWeight)

            for (idx, weight) in weights.enumerated() {
                target -= weight
                if target <= 0 {
                    centers.append(sampledData[idx])
                    break
                }
            }
        }
    }

    @inline(__always)
    private func findNearestCenter(_ point: Vector512Optimized) -> (index: Int, distance: Float) {
        var minDist = Float.infinity
        var minIdx = 0

        // Unroll by 4 for better ILP
        let blocked = (config.k / 4) * 4

        if blocked >= 4 {
            var dists = SIMD4<Float>(repeating: .infinity)
            var indices = SIMD4<Int32>(0, 1, 2, 3)

            for i in stride(from: 0, to: blocked, by: 4) {
                let d0 = distance512(point, centers[i])
                let d1 = distance512(point, centers[i+1])
                let d2 = distance512(point, centers[i+2])
                let d3 = distance512(point, centers[i+3])

                let newDists = SIMD4<Float>(d0, d1, d2, d3)
                let mask = newDists .< dists
                dists.replace(with: newDists, where: mask)
                indices.replace(with: SIMD4<Int32>(Int32(i), Int32(i+1), Int32(i+2), Int32(i+3)), where: mask)
            }

            // Find minimum from SIMD4
            for lane in 0..<4 {
                if dists[lane] < minDist {
                    minDist = dists[lane]
                    minIdx = Int(indices[lane])
                }
            }
        }

        // Handle remainder
        for i in blocked..<config.k {
            let d = distance512(point, centers[i])
            if d < minDist {
                minDist = d
                minIdx = i
            }
        }

        return (minIdx, minDist)
    }

    @inline(__always)
    private func findNearestCenterSoA(_ point: Vector512Optimized, soa: StructureOfArrays512) -> (index: Int, distance: Float) {
        // Compute distances directly since we don't have batchSquaredDistance512 yet
        // This can be optimized later when the batch squared distance kernel is added
        var minDist = Float.infinity
        var minIdx = 0

        for i in 0..<config.k {
            let d = EuclideanKernels.distance512(point, centers[i])
            if d < minDist {
                minDist = d
                minIdx = i
            }
        }

        return (minIdx, minDist)
    }

    @inline(__always)
    private func updateCluster(_ clusterIdx: Int, with point: Vector512Optimized) {
        // Exponentially weighted moving average update
        let alpha = currentLearningRate / Float(clusterSizes[clusterIdx] + 1)

        // Update cluster center: c_new = (1-α)*c_old + α*point
        centers[clusterIdx] = centers[clusterIdx].scaled(by: 1 - alpha)
            .adding(point.scaled(by: alpha))

        clusterSizes[clusterIdx] += 1
    }

    // Optimized distance computation using existing kernel
    @inline(__always)
    private func distance512(_ a: Vector512Optimized, _ b: Vector512Optimized) -> Float {
        return EuclideanKernels.distance512(a, b)
    }
}

// MARK: - Helper Types

/// Thread-safe wrapper for UnsafeMutablePointer to work with actors
private final class BufferWrapper<T> {
    private let buffer: UnsafeMutablePointer<T>
    let capacity: Int

    init(capacity: Int) {
        self.capacity = capacity
        self.buffer = UnsafeMutablePointer<T>.allocate(capacity: capacity)
    }

    deinit {
        buffer.deallocate()
    }

    nonisolated var pointer: UnsafeMutablePointer<T> {
        buffer
    }

    subscript(index: Int) -> T {
        get { buffer[index] }
        set { buffer[index] = newValue }
    }
}

// Mark as Sendable since we only use it from within actors
extension BufferWrapper: @unchecked Sendable {}

/// Circular buffer for convergence monitoring
private struct CircularBuffer<T> {
    private var buffer: [T?]
    private var writeIndex = 0
    private var count = 0
    private let capacity: Int

    init(capacity: Int) {
        self.capacity = capacity
        self.buffer = Array(repeating: nil, count: capacity)
    }

    mutating func append(_ value: T) {
        buffer[writeIndex] = value
        writeIndex = (writeIndex + 1) % capacity
        count = min(count + 1, capacity)
    }

    var isFull: Bool { count == capacity }

    var values: [T] {
        if count < capacity {
            return buffer.prefix(count).compactMap { $0 }
        } else {
            return buffer.compactMap { $0 }
        }
    }
}

/// Simple arithmetic RNG for reproducible initialization
private struct ArithmeticRandomNumberGenerator: RandomNumberGenerator {
    private var state: UInt64

    init(seed: Int) {
        self.state = UInt64(seed)
    }

    mutating func next() -> UInt64 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return state
    }
}

/// Structure-of-Arrays storage for batch operations
private final class StructureOfArrays512 {
    let capacity: Int
    var count: Int = 0

    // Transposed storage for each SIMD lane
    var lanes: [UnsafeMutablePointer<Float>]

    init(capacity: Int) {
        self.capacity = capacity
        self.lanes = []

        // Allocate 128 lanes (for 512 dimensions with SIMD4)
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

// MARK: - StreamingKMeans768

/// Streaming K-means for 768-dimensional vectors
@available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
public actor StreamingKMeans768 {
    // Implementation follows the same pattern as StreamingKMeans512
    // with adjustments for 768 dimensions (192 SIMD4 chunks)

    public struct Config {
        let k: Int
        let batchSize: Int
        let learningRate: Float
        let decayFactor: Float
        let convergenceWindow: Int
        let convergenceTolerance: Float
        let maxIterations: Int?
        let initMethod: InitMethod
        let verbose: Bool

        public enum InitMethod {
            case random(seed: Int?)
            case plusPlus(samples: Int)
            case custom([Vector768Optimized])
        }

        public init(
            k: Int,
            batchSize: Int = 1000,
            learningRate: Float = 0.1,
            decayFactor: Float = 0.995,
            convergenceWindow: Int = 10,
            convergenceTolerance: Float = 1e-4,
            maxIterations: Int? = nil,
            initMethod: InitMethod = .plusPlus(samples: 10000),
            verbose: Bool = false
        ) {
            self.k = k
            self.batchSize = batchSize
            self.learningRate = learningRate
            self.decayFactor = decayFactor
            self.convergenceWindow = convergenceWindow
            self.convergenceTolerance = convergenceTolerance
            self.maxIterations = maxIterations
            self.initMethod = initMethod
            self.verbose = verbose
        }
    }

    public struct Result {
        public let centers: [Vector768Optimized]
        public let clusterSizes: [Int]
        public let inertia: Float
        public let converged: Bool
        public let iterations: Int
        public let totalSamples: Int
    }

    private let config: Config
    private var centers: [Vector768Optimized]
    private var clusterSizes: [Int]
    private var clusterSums: [Vector768Optimized]
    private var currentLearningRate: Float
    private var batchesProcessed: Int = 0
    private var totalSamples: Int = 0
    private var convergenceHistory: CircularBuffer<Float>
    private var lastInertia: Float = .infinity
    private let distanceBuffer: BufferWrapper<Float>

    public init(config: Config) {
        self.config = config
        self.centers = []
        self.clusterSizes = Array(repeating: 0, count: config.k)
        self.clusterSums = Array(repeating: .zero, count: config.k)
        self.currentLearningRate = config.learningRate
        self.convergenceHistory = CircularBuffer<Float>(capacity: config.convergenceWindow)
        self.distanceBuffer = BufferWrapper<Float>(capacity: config.k)
    }

    public func initialize(from data: [Vector768Optimized]) async {
        switch config.initMethod {
        case .random(let seed):
            if let seed = seed {
                var rng = ArithmeticRandomNumberGenerator(seed: seed)
                let indices = (0..<data.count).shuffled(using: &rng).prefix(config.k)
                centers = indices.map { data[$0] }
            } else {
                var rng = SystemRandomNumberGenerator()
                let indices = (0..<data.count).shuffled(using: &rng).prefix(config.k)
                centers = indices.map { data[$0] }
            }
        case .plusPlus(let samples):
            initializeKMeansPlusPlus(from: data, samples: samples)
        case .custom(let customCenters):
            centers = customCenters
        }
    }

    public func processBatch(_ batch: [Vector768Optimized]) async -> Float {
        guard !batch.isEmpty else { return lastInertia }

        var batchInertia: Float = 0

        for point in batch {
            let (nearest, distance) = findNearestCenter(point)
            updateCluster(nearest, with: point)
            batchInertia += distance * distance
        }

        currentLearningRate *= config.decayFactor
        batchesProcessed += 1
        totalSamples += batch.count

        let avgInertia = batchInertia / Float(batch.count)
        convergenceHistory.append(avgInertia)
        lastInertia = avgInertia

        return avgInertia
    }

    public func hasConverged() async -> Bool {
        guard convergenceHistory.isFull else { return false }

        if let maxIter = config.maxIterations, batchesProcessed >= maxIter {
            return true
        }

        let values = convergenceHistory.values
        let mean = values.reduce(0, +) / Float(values.count)
        let variance = values.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(values.count)
        let relativeChange = sqrt(variance) / (abs(mean) + 1e-10)

        return relativeChange < config.convergenceTolerance
    }

    public func getResult() async -> Result {
        Result(
            centers: centers,
            clusterSizes: clusterSizes,
            inertia: lastInertia,
            converged: await hasConverged(),
            iterations: batchesProcessed,
            totalSamples: totalSamples
        )
    }

    public func predict(_ data: [Vector768Optimized]) async -> [Int] {
        data.map { point in
            findNearestCenter(point).index
        }
    }

    private func initializeKMeansPlusPlus(from data: [Vector768Optimized], samples: Int) {
        guard !data.isEmpty else { return }

        let sampleSize = min(samples, data.count)
        let sampledData = data.count > samples ? Array(data.shuffled().prefix(sampleSize)) : data

        centers = [sampledData.randomElement()!]

        for _ in 1..<config.k {
            var weights = [Float]()
            weights.reserveCapacity(sampledData.count)

            for point in sampledData {
                let minDist = centers.map { distance768(point, $0) }.min()!
                weights.append(minDist * minDist)
            }

            let totalWeight = weights.reduce(0, +)
            var target = Float.random(in: 0..<totalWeight)

            for (idx, weight) in weights.enumerated() {
                target -= weight
                if target <= 0 {
                    centers.append(sampledData[idx])
                    break
                }
            }
        }
    }

    @inline(__always)
    private func findNearestCenter(_ point: Vector768Optimized) -> (index: Int, distance: Float) {
        var minDist = Float.infinity
        var minIdx = 0

        for (idx, center) in centers.enumerated() {
            let d = distance768(point, center)
            if d < minDist {
                minDist = d
                minIdx = idx
            }
        }

        return (minIdx, minDist)
    }

    @inline(__always)
    private func updateCluster(_ clusterIdx: Int, with point: Vector768Optimized) {
        let alpha = currentLearningRate / Float(clusterSizes[clusterIdx] + 1)
        centers[clusterIdx] = centers[clusterIdx].scaled(by: 1 - alpha)
            .adding(point.scaled(by: alpha))
        clusterSizes[clusterIdx] += 1
    }

    @inline(__always)
    private func distance768(_ a: Vector768Optimized, _ b: Vector768Optimized) -> Float {
        return EuclideanKernels.distance768(a, b)
    }
}

// MARK: - StreamingKMeans1536

/// Streaming K-means for 1536-dimensional vectors
@available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
public actor StreamingKMeans1536 {
    // Implementation follows the same pattern as StreamingKMeans512
    // with adjustments for 1536 dimensions (384 SIMD4 chunks)

    public struct Config {
        let k: Int
        let batchSize: Int
        let learningRate: Float
        let decayFactor: Float
        let convergenceWindow: Int
        let convergenceTolerance: Float
        let maxIterations: Int?
        let initMethod: InitMethod
        let verbose: Bool

        public enum InitMethod {
            case random(seed: Int?)
            case plusPlus(samples: Int)
            case custom([Vector1536Optimized])
        }

        public init(
            k: Int,
            batchSize: Int = 1000,
            learningRate: Float = 0.1,
            decayFactor: Float = 0.995,
            convergenceWindow: Int = 10,
            convergenceTolerance: Float = 1e-4,
            maxIterations: Int? = nil,
            initMethod: InitMethod = .plusPlus(samples: 10000),
            verbose: Bool = false
        ) {
            self.k = k
            self.batchSize = batchSize
            self.learningRate = learningRate
            self.decayFactor = decayFactor
            self.convergenceWindow = convergenceWindow
            self.convergenceTolerance = convergenceTolerance
            self.maxIterations = maxIterations
            self.initMethod = initMethod
            self.verbose = verbose
        }
    }

    public struct Result {
        public let centers: [Vector1536Optimized]
        public let clusterSizes: [Int]
        public let inertia: Float
        public let converged: Bool
        public let iterations: Int
        public let totalSamples: Int
    }

    private let config: Config
    private var centers: [Vector1536Optimized]
    private var clusterSizes: [Int]
    private var clusterSums: [Vector1536Optimized]
    private var currentLearningRate: Float
    private var batchesProcessed: Int = 0
    private var totalSamples: Int = 0
    private var convergenceHistory: CircularBuffer<Float>
    private var lastInertia: Float = .infinity
    private let distanceBuffer: BufferWrapper<Float>

    public init(config: Config) {
        self.config = config
        self.centers = []
        self.clusterSizes = Array(repeating: 0, count: config.k)
        self.clusterSums = Array(repeating: .zero, count: config.k)
        self.currentLearningRate = config.learningRate
        self.convergenceHistory = CircularBuffer<Float>(capacity: config.convergenceWindow)
        self.distanceBuffer = BufferWrapper<Float>(capacity: config.k)
    }

    public func initialize(from data: [Vector1536Optimized]) async {
        switch config.initMethod {
        case .random(let seed):
            if let seed = seed {
                var rng = ArithmeticRandomNumberGenerator(seed: seed)
                let indices = (0..<data.count).shuffled(using: &rng).prefix(config.k)
                centers = indices.map { data[$0] }
            } else {
                var rng = SystemRandomNumberGenerator()
                let indices = (0..<data.count).shuffled(using: &rng).prefix(config.k)
                centers = indices.map { data[$0] }
            }
        case .plusPlus(let samples):
            initializeKMeansPlusPlus(from: data, samples: samples)
        case .custom(let customCenters):
            centers = customCenters
        }
    }

    public func processBatch(_ batch: [Vector1536Optimized]) async -> Float {
        guard !batch.isEmpty else { return lastInertia }

        var batchInertia: Float = 0

        for point in batch {
            let (nearest, distance) = findNearestCenter(point)
            updateCluster(nearest, with: point)
            batchInertia += distance * distance
        }

        currentLearningRate *= config.decayFactor
        batchesProcessed += 1
        totalSamples += batch.count

        let avgInertia = batchInertia / Float(batch.count)
        convergenceHistory.append(avgInertia)
        lastInertia = avgInertia

        return avgInertia
    }

    public func hasConverged() async -> Bool {
        guard convergenceHistory.isFull else { return false }

        if let maxIter = config.maxIterations, batchesProcessed >= maxIter {
            return true
        }

        let values = convergenceHistory.values
        let mean = values.reduce(0, +) / Float(values.count)
        let variance = values.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(values.count)
        let relativeChange = sqrt(variance) / (abs(mean) + 1e-10)

        return relativeChange < config.convergenceTolerance
    }

    public func getResult() async -> Result {
        Result(
            centers: centers,
            clusterSizes: clusterSizes,
            inertia: lastInertia,
            converged: await hasConverged(),
            iterations: batchesProcessed,
            totalSamples: totalSamples
        )
    }

    public func predict(_ data: [Vector1536Optimized]) async -> [Int] {
        data.map { point in
            findNearestCenter(point).index
        }
    }

    private func initializeKMeansPlusPlus(from data: [Vector1536Optimized], samples: Int) {
        guard !data.isEmpty else { return }

        let sampleSize = min(samples, data.count)
        let sampledData = data.count > samples ? Array(data.shuffled().prefix(sampleSize)) : data

        centers = [sampledData.randomElement()!]

        for _ in 1..<config.k {
            var weights = [Float]()
            weights.reserveCapacity(sampledData.count)

            for point in sampledData {
                let minDist = centers.map { distance1536(point, $0) }.min()!
                weights.append(minDist * minDist)
            }

            let totalWeight = weights.reduce(0, +)
            var target = Float.random(in: 0..<totalWeight)

            for (idx, weight) in weights.enumerated() {
                target -= weight
                if target <= 0 {
                    centers.append(sampledData[idx])
                    break
                }
            }
        }
    }

    @inline(__always)
    private func findNearestCenter(_ point: Vector1536Optimized) -> (index: Int, distance: Float) {
        var minDist = Float.infinity
        var minIdx = 0

        for (idx, center) in centers.enumerated() {
            let d = distance1536(point, center)
            if d < minDist {
                minDist = d
                minIdx = idx
            }
        }

        return (minIdx, minDist)
    }

    @inline(__always)
    private func updateCluster(_ clusterIdx: Int, with point: Vector1536Optimized) {
        let alpha = currentLearningRate / Float(clusterSizes[clusterIdx] + 1)
        centers[clusterIdx] = centers[clusterIdx].scaled(by: 1 - alpha)
            .adding(point.scaled(by: alpha))
        clusterSizes[clusterIdx] += 1
    }

    @inline(__always)
    private func distance1536(_ a: Vector1536Optimized, _ b: Vector1536Optimized) -> Float {
        return EuclideanKernels.distance1536(a, b)
    }
}
