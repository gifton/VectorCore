//
//  ClusteringTestUtilities.swift
//  VectorCore
//
//  Test utilities for generating synthetic clusterable data and evaluating clustering quality.
//  Implements Kernel Spec #30 with bug fixes and VectorCore integration.
//

import Foundation
import simd
@testable import VectorCore

#if canImport(Accelerate)
import Accelerate
#endif

// Platform-specific imports for timing and randomization
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

// =============================================================================
// MARK: - Cross-Platform Utilities (Timing and Randomization)
// =============================================================================

/// Provides cross-platform access to high-resolution timing and reproducible randomization
fileprivate enum PlatformUtils {

    // MARK: Randomization (Wrappers for srand48/drand48 or fallbacks)

    static func seed(seed: UInt64) {
        #if canImport(Darwin) || canImport(Glibc)
        srand48(Int(truncatingIfNeeded: seed))
        #else
        srand(UInt32(truncatingIfNeeded: seed))
        #endif
    }

    /// Returns a Double in [0.0, 1.0) (or [0.0, 1.0] for fallback platforms)
    static func randomDouble() -> Double {
        #if canImport(Darwin) || canImport(Glibc)
        return drand48()
        #else
        return Double(rand()) / Double(RAND_MAX)
        #endif
    }

    // MARK: Timing (High-resolution monotonic time)

    #if canImport(Darwin)
    static func monotonicTime() -> UInt64 {
        return mach_absolute_time()
    }

    static func timeToSeconds(_ time: UInt64) -> TimeInterval {
        var timebase = mach_timebase_info_data_t()
        guard mach_timebase_info(&timebase) == KERN_SUCCESS else { return 0 }

        // Use Double for calculation to prevent overflow
        let numer = Double(timebase.numer)
        let denom = Double(timebase.denom)
        guard denom != 0 else { return 0 }

        let nanos = Double(time) * numer / denom
        return TimeInterval(nanos) / 1_000_000_000.0
    }

    #elseif canImport(Glibc)
    static func monotonicTime() -> UInt64 {
        var ts = timespec()
        clock_gettime(CLOCK_MONOTONIC, &ts)
        return UInt64(ts.tv_sec) * 1_000_000_000 + UInt64(ts.tv_nsec)
    }

    static func timeToSeconds(_ time: UInt64) -> TimeInterval {
        return TimeInterval(time) / 1_000_000_000.0
    }

    #else
    // Fallback (NOT monotonic, but better than nothing)
    static func monotonicTime() -> UInt64 {
        return UInt64(Date().timeIntervalSince1970 * 1_000_000_000)
    }

    static func timeToSeconds(_ time: UInt64) -> TimeInterval {
        return TimeInterval(time) / 1_000_000_000.0
    }
    #endif
}

// =============================================================================
// MARK: - Extensions for Random Number Generation
// =============================================================================

extension Float {
    /// Generate random number from standard normal distribution N(0, 1) using Box-Muller transform
    fileprivate static func randomGaussian() -> Float {
        // Ensure u1 is non-zero to avoid log(0) = -inf
        var u1 = Float(PlatformUtils.randomDouble())
        while u1 == 0 {
            u1 = Float(PlatformUtils.randomDouble())
        }
        let u2 = Float(PlatformUtils.randomDouble())

        let r = sqrt(-2.0 * log(u1))
        let theta = 2.0 * Float.pi * u2

        return r * cos(theta)
    }
}

// =============================================================================
// MARK: - 1. Synthetic Data Generation
// =============================================================================

/// Utilities for generating synthetic clusterable test data
public enum SyntheticDataGenerator {

    // MARK: 1.1 Gaussian Mixture Generation (512-dimensional)

    /// Generate 512-dimensional vectors from Gaussian mixture model (GMM)
    ///
    /// Creates K clusters with specified separation and variance.
    /// Each cluster is a multivariate Gaussian with random center.
    ///
    /// - Parameters:
    ///   - numClusters: Number of clusters (K)
    ///   - vectorsPerCluster: Vectors per cluster (N/K)
    ///   - separation: Distance between cluster centers (larger = easier to cluster)
    ///   - variance: Within-cluster variance (σ²)
    ///   - seed: Random seed for reproducibility
    /// - Returns: (vectors, labels) where labels[i] = cluster ID for vectors[i]
    public static func generateGaussianMixture512(
        numClusters: Int,
        vectorsPerCluster: Int,
        separation: Float = 3.0,
        variance: Float = 1.0,
        seed: UInt64? = nil
    ) -> (vectors: [Vector512Optimized], labels: [Int]) {

        precondition(numClusters > 0 && vectorsPerCluster > 0)
        if let seed = seed { PlatformUtils.seed(seed: seed) }

        // 1. Generate cluster centers
        var clusterCenters: [[Float]] = []
        for _ in 0..<numClusters {
            var center = [Float](repeating: 0, count: 512)
            for d in 0..<512 {
                center[d] = Float.randomGaussian() * separation
            }
            clusterCenters.append(center)
        }

        // 2. Generate vectors for each cluster
        var vectors: [Vector512Optimized] = []
        var labels: [Int] = []
        let totalVectors = numClusters * vectorsPerCluster
        vectors.reserveCapacity(totalVectors)
        labels.reserveCapacity(totalVectors)

        let stdDev = sqrt(variance)

        for clusterID in 0..<numClusters {
            let center = clusterCenters[clusterID]

            for _ in 0..<vectorsPerCluster {
                var vectorData = [Float](repeating: 0, count: 512)
                for d in 0..<512 {
                    vectorData[d] = center[d] + Float.randomGaussian() * stdDev
                }

                let vector = try! Vector512Optimized(vectorData)
                vectors.append(vector)
                labels.append(clusterID)
            }
        }

        // 3. Shuffle to remove order bias
        return shuffle(vectors: vectors, labels: labels)
    }

    // MARK: 1.2 Gaussian Mixture Generation (768-dimensional)

    public static func generateGaussianMixture768(
        numClusters: Int,
        vectorsPerCluster: Int,
        separation: Float = 3.0,
        variance: Float = 1.0,
        seed: UInt64? = nil
    ) -> (vectors: [Vector768Optimized], labels: [Int]) {

        precondition(numClusters > 0 && vectorsPerCluster > 0)
        if let seed = seed { PlatformUtils.seed(seed: seed) }

        var clusterCenters: [[Float]] = []
        for _ in 0..<numClusters {
            var center = [Float](repeating: 0, count: 768)
            for d in 0..<768 {
                center[d] = Float.randomGaussian() * separation
            }
            clusterCenters.append(center)
        }

        var vectors: [Vector768Optimized] = []
        var labels: [Int] = []
        vectors.reserveCapacity(numClusters * vectorsPerCluster)
        labels.reserveCapacity(numClusters * vectorsPerCluster)

        let stdDev = sqrt(variance)

        for clusterID in 0..<numClusters {
            let center = clusterCenters[clusterID]
            for _ in 0..<vectorsPerCluster {
                var vectorData = [Float](repeating: 0, count: 768)
                for d in 0..<768 {
                    vectorData[d] = center[d] + Float.randomGaussian() * stdDev
                }
                let vector = try! Vector768Optimized(vectorData)
                vectors.append(vector)
                labels.append(clusterID)
            }
        }

        return shuffle(vectors: vectors, labels: labels)
    }

    // MARK: 1.3 Gaussian Mixture Generation (1536-dimensional)

    public static func generateGaussianMixture1536(
        numClusters: Int,
        vectorsPerCluster: Int,
        separation: Float = 3.0,
        variance: Float = 1.0,
        seed: UInt64? = nil
    ) -> (vectors: [Vector1536Optimized], labels: [Int]) {

        precondition(numClusters > 0 && vectorsPerCluster > 0)
        if let seed = seed { PlatformUtils.seed(seed: seed) }

        var clusterCenters: [[Float]] = []
        for _ in 0..<numClusters {
            var center = [Float](repeating: 0, count: 1536)
            for d in 0..<1536 {
                center[d] = Float.randomGaussian() * separation
            }
            clusterCenters.append(center)
        }

        var vectors: [Vector1536Optimized] = []
        var labels: [Int] = []
        vectors.reserveCapacity(numClusters * vectorsPerCluster)
        labels.reserveCapacity(numClusters * vectorsPerCluster)

        let stdDev = sqrt(variance)

        for clusterID in 0..<numClusters {
            let center = clusterCenters[clusterID]
            for _ in 0..<vectorsPerCluster {
                var vectorData = [Float](repeating: 0, count: 1536)
                for d in 0..<1536 {
                    vectorData[d] = center[d] + Float.randomGaussian() * stdDev
                }
                let vector = try! Vector1536Optimized(vectorData)
                vectors.append(vector)
                labels.append(clusterID)
            }
        }

        return shuffle(vectors: vectors, labels: labels)
    }

    // MARK: 1.4 Concentric Circles (512-dimensional)

    /// Generate concentric circular clusters (2D embedded in 512-D space)
    public static func generateConcentricCircles512(
        numRings: Int,
        pointsPerRing: Int,
        radiusMultiplier: Float = 2.0,
        noise: Float = 0.1,
        seed: UInt64? = nil
    ) -> (vectors: [Vector512Optimized], labels: [Int]) {

        if let seed = seed { PlatformUtils.seed(seed: seed) }

        // Generate random projection matrix (2D → 512D)
        var projectionMatrix: [[Float]] = []
        for _ in 0..<512 {
            let row = [Float.randomGaussian(), Float.randomGaussian()]
            projectionMatrix.append(row)
        }

        var vectors: [Vector512Optimized] = []
        var labels: [Int] = []
        vectors.reserveCapacity(numRings * pointsPerRing)
        labels.reserveCapacity(numRings * pointsPerRing)

        for ring in 0..<numRings {
            let radius = Float(ring + 1) * radiusMultiplier

            for point in 0..<pointsPerRing {
                let angle = 2.0 * Float.pi * Float(point) / Float(pointsPerRing)
                let x2d = radius * cos(angle) + Float.randomGaussian() * noise
                let y2d = radius * sin(angle) + Float.randomGaussian() * noise

                var vectorData = [Float](repeating: 0, count: 512)
                for d in 0..<512 {
                    vectorData[d] = projectionMatrix[d][0] * x2d + projectionMatrix[d][1] * y2d
                }

                let vector = try! Vector512Optimized(vectorData)
                vectors.append(vector)
                labels.append(ring)
            }
        }

        return shuffle(vectors: vectors, labels: labels)
    }

    // MARK: 1.5 Random Vectors (No Structure)

    public enum Distribution {
        case gaussian
        case uniform
    }

    /// Generate random 512-dimensional vectors with no structure (negative control)
    public static func generateRandomVectors512(
        count: Int,
        distribution: Distribution = .gaussian,
        seed: UInt64? = nil
    ) -> [Vector512Optimized] {

        if let seed = seed { PlatformUtils.seed(seed: seed) }

        var vectors: [Vector512Optimized] = []
        vectors.reserveCapacity(count)

        for _ in 0..<count {
            var vectorData = [Float](repeating: 0, count: 512)

            switch distribution {
            case .gaussian:
                for d in 0..<512 {
                    vectorData[d] = Float.randomGaussian()
                }
            case .uniform:
                for d in 0..<512 {
                    vectorData[d] = Float(PlatformUtils.randomDouble() * 2.0 - 1.0)
                }
            }

            let vector = try! Vector512Optimized(vectorData)
            vectors.append(vector)
        }

        return vectors
    }

    /// Generate random 768-dimensional vectors
    public static func generateRandomVectors768(
        count: Int,
        distribution: Distribution = .gaussian,
        seed: UInt64? = nil
    ) -> [Vector768Optimized] {

        if let seed = seed { PlatformUtils.seed(seed: seed) }

        var vectors: [Vector768Optimized] = []
        vectors.reserveCapacity(count)

        for _ in 0..<count {
            var vectorData = [Float](repeating: 0, count: 768)
            switch distribution {
            case .gaussian:
                for d in 0..<768 { vectorData[d] = Float.randomGaussian() }
            case .uniform:
                for d in 0..<768 { vectorData[d] = Float(PlatformUtils.randomDouble() * 2.0 - 1.0) }
            }
            let vector = try! Vector768Optimized(vectorData)
            vectors.append(vector)
        }

        return vectors
    }

    /// Generate random 1536-dimensional vectors
    public static func generateRandomVectors1536(
        count: Int,
        distribution: Distribution = .gaussian,
        seed: UInt64? = nil
    ) -> [Vector1536Optimized] {

        if let seed = seed { PlatformUtils.seed(seed: seed) }

        var vectors: [Vector1536Optimized] = []
        vectors.reserveCapacity(count)

        for _ in 0..<count {
            var vectorData = [Float](repeating: 0, count: 1536)
            switch distribution {
            case .gaussian:
                for d in 0..<1536 { vectorData[d] = Float.randomGaussian() }
            case .uniform:
                for d in 0..<1536 { vectorData[d] = Float(PlatformUtils.randomDouble() * 2.0 - 1.0) }
            }
            let vector = try! Vector1536Optimized(vectorData)
            vectors.append(vector)
        }

        return vectors
    }

    // MARK: - Helper Functions

    /// Shuffles vectors and labels consistently using Fisher-Yates algorithm
    ///
    /// **BUG FIX #1**: Uses min() to prevent index overflow when randomDouble() returns exactly 1.0
    private static func shuffle<T>(
        vectors: [T],
        labels: [Int]
    ) -> (vectors: [T], labels: [Int]) {
        var indices = Array(0..<vectors.count)

        // Fisher-Yates shuffle with overflow protection
        for i in stride(from: indices.count - 1, through: 1, by: -1) {
            // BUG FIX: Clamp j to valid range [0, i]
            let j = min(Int(PlatformUtils.randomDouble() * Double(i + 1)), i)
            indices.swapAt(i, j)
        }

        let shuffledVectors = indices.map { vectors[$0] }
        let shuffledLabels = indices.map { labels[$0] }

        return (shuffledVectors, shuffledLabels)
    }
}

// =============================================================================
// MARK: - 2. Clustering Quality Metrics
// =============================================================================

/// Clustering evaluation metrics
public enum ClusteringMetrics {

    public enum DistanceMetric {
        case euclidean
        case cosine
        case manhattan
    }

    // MARK: 2.1 Silhouette Score (512-dimensional)

    /// Compute Silhouette score for clustering (O(N² × D))
    public static func silhouetteScore512(
        vectors: [Vector512Optimized],
        labels: [Int],
        metric: DistanceMetric = .euclidean
    ) -> Float {
        let N = vectors.count
        guard N > 1 else { return 0 }

        let uniqueLabels = Set(labels)
        let numClusters = uniqueLabels.count
        guard numClusters > 1 else { return 0 }

        var silhouetteScores = [Float](repeating: 0.0, count: N)

        // Group indices by cluster for optimized access
        let clusterIndices = Dictionary(grouping: vectors.indices) { labels[$0] }

        for i in 0..<N {
            let clusterI = labels[i]

            guard let sameClusterMembers = clusterIndices[clusterI], sameClusterMembers.count > 1 else {
                continue
            }

            var sumA: Float = 0
            for j in sameClusterMembers where j != i {
                sumA += distance512(vectors[i], vectors[j], metric: metric)
            }
            let a = sumA / Float(sameClusterMembers.count - 1)

            var minAvgDistB: Float = .infinity

            for otherClusterID in uniqueLabels where otherClusterID != clusterI {
                guard let otherClusterMembers = clusterIndices[otherClusterID] else { continue }

                var sumB: Float = 0
                for j in otherClusterMembers {
                    sumB += distance512(vectors[i], vectors[j], metric: metric)
                }

                let avgDist = sumB / Float(otherClusterMembers.count)
                minAvgDistB = min(minAvgDistB, avgDist)
            }

            let b = minAvgDistB
            let denominator = max(a, b)

            if denominator > 1e-9 && denominator.isFinite {
                 silhouetteScores[i] = (b - a) / denominator
            }
        }

        return silhouetteScores.reduce(0, +) / Float(N)
    }

    // MARK: 2.2 Davies-Bouldin Index (512-dimensional)

    /// Davies-Bouldin Index (lower is better)
    public static func daviesBouldinIndex512(
        vectors: [Vector512Optimized],
        labels: [Int]
    ) -> Float {
        let clusters = Dictionary(grouping: vectors.indices) { labels[$0] }
        let K = clusters.count
        guard K > 1 else { return 0 }

        // Compute centroids
        var centroids: [Int: Vector512Optimized] = [:]
        for (clusterID, indices) in clusters {
            let clusterVectors = indices.map { vectors[$0] }
            centroids[clusterID] = computeCentroid512(clusterVectors)
        }

        // Compute within-cluster scatter
        var scatter: [Int: Float] = [:]
        for (clusterID, indices) in clusters {
            guard let centroid = centroids[clusterID], !indices.isEmpty else { continue }

            let sumDist = indices.map {
                EuclideanKernels.distance512(vectors[$0], centroid)
            }.reduce(0, +)

            scatter[clusterID] = sumDist / Float(indices.count)
        }

        // Compute DB index
        var dbSum: Float = 0
        let clusterIDs = Array(clusters.keys)

        for i in 0..<K {
            let clusterI = clusterIDs[i]
            var maxRatio: Float = -.infinity

            for j in 0..<K where i != j {
                let clusterJ = clusterIDs[j]

                guard let scatterI = scatter[clusterI], let scatterJ = scatter[clusterJ] else { continue }

                let centroidDist = EuclideanKernels.distance512(
                    centroids[clusterI]!,
                    centroids[clusterJ]!
                )

                // Handle identical centroids
                if centroidDist < 1e-9 {
                    if (scatterI + scatterJ) > 1e-9 {
                        maxRatio = .infinity
                    }
                } else {
                    let ratio = (scatterI + scatterJ) / centroidDist
                    maxRatio = max(maxRatio, ratio)
                }
            }

            if maxRatio.isFinite && maxRatio != -.infinity {
                dbSum += maxRatio
            } else if maxRatio == .infinity {
                return .infinity
            }
        }

        return dbSum / Float(K)
    }

    // MARK: 2.3 Adjusted Rand Index

    /// Adjusted Rand Index (ARI) for comparing clusterings
    ///
    /// **BUG FIX #2**: Converts to Double BEFORE multiplication to prevent Int64 overflow
    public static func adjustedRandIndex(
        labels1: [Int],
        labels2: [Int]
    ) -> Float {
        precondition(labels1.count == labels2.count)
        let N = labels1.count
        guard N > 1 else { return 1.0 }

        // Build contingency table
        struct Pair: Hashable {
            let l1: Int
            let l2: Int
        }

        var contingency = [Pair: Int]()
        for i in 0..<N {
            let pair = Pair(l1: labels1[i], l2: labels2[i])
            contingency[pair, default: 0] += 1
        }

        // Compute marginals
        var a = [Int: Int]()
        var b = [Int: Int]()

        for (pair, count) in contingency {
            a[pair.l1, default: 0] += count
            b[pair.l2, default: 0] += count
        }

        func choose2(_ n: Int) -> Int64 {
            let n64 = Int64(n)
            return n64 * (n64 - 1) / 2
        }

        // Compute ARI components
        let sumNij2 = contingency.values.reduce(0) { $0 + choose2($1) }
        let sumAi2 = a.values.reduce(0) { $0 + choose2($1) }
        let sumBj2 = b.values.reduce(0) { $0 + choose2($1) }
        let nChoose2Total = choose2(N)

        // BUG FIX #2: Convert to Double BEFORE multiplication to prevent overflow
        let expectedIndex = (Double(sumAi2) * Double(sumBj2)) / Double(nChoose2Total)
        let maxIndex = Double(sumAi2 + sumBj2) / 2.0

        let numerator = Double(sumNij2) - expectedIndex
        let denominator = maxIndex - expectedIndex

        if abs(denominator) < 1e-9 {
            return abs(numerator) < 1e-9 ? 1.0 : 0.0
        }

        let ari = Float(numerator / denominator)
        return ari.isFinite ? ari : 0.0
    }

    // MARK: - Helper Functions

    private static func distance512(
        _ a: Vector512Optimized,
        _ b: Vector512Optimized,
        metric: DistanceMetric
    ) -> Float {
        switch metric {
        case .euclidean:
            return EuclideanKernels.distance512(a, b)
        case .cosine:
            return CosineKernels.distance512_fused(a, b)
        case .manhattan:
            return ManhattanKernels.distance512(a, b)
        }
    }

    private static func computeCentroid512(_ vectors: [Vector512Optimized]) -> Vector512Optimized {
        guard !vectors.isEmpty else {
            return try! Vector512Optimized([Float](repeating: 0, count: 512))
        }

        var sum = vectors[0]
        for i in 1..<vectors.count {
            sum = sum + vectors[i]
        }

        return sum.scaled(by: 1.0 / Float(vectors.count))
    }
}

// =============================================================================
// MARK: - 3. Performance Measurement Framework
// =============================================================================

/// Performance benchmarking utilities
public struct PerformanceMeasurement {

    /// Benchmark result statistics
    public struct Result {
        public let meanTime: TimeInterval
        public let medianTime: TimeInterval
        public let stdDev: TimeInterval
        public let minTime: TimeInterval
        public let maxTime: TimeInterval
        public let iterations: Int
        public let throughput: Double

        public var summary: String {
            let throughputFormatted = throughput >= 1000 ?
                String(format: "%.2f K ops/sec", throughput / 1000) :
                String(format: "%.2f ops/sec", throughput)

            return """
            Performance Results (\(iterations) iterations):
              Mean:   \(String(format: "%.4f ms", meanTime * 1000))
              Median: \(String(format: "%.4f ms", medianTime * 1000))
              StdDev: \(String(format: "%.4f ms", stdDev * 1000))
              Range:  [\(String(format: "%.4f", minTime * 1000)) - \(String(format: "%.4f", maxTime * 1000))] ms
              Throughput: \(throughputFormatted)
            """
        }
    }

    /// Measure performance of an operation using high-resolution timing
    ///
    /// **BUG FIX #3**: Uses sample variance (n-1) for unbiased standard deviation estimate
    public static func measureClusteringPerformance(
        iterations: Int = 100,
        warmupIterations: Int = 10,
        operation: () -> Void
    ) -> Result {

        guard iterations > 0 else {
            fatalError("Iterations must be greater than 0.")
        }

        // Warmup phase (JIT compilation, cache loading)
        for _ in 0..<warmupIterations {
            operation()
        }

        // Benchmark phase
        var times: [TimeInterval] = []
        times.reserveCapacity(iterations)

        for _ in 0..<iterations {
            let start = PlatformUtils.monotonicTime()
            operation()
            let end = PlatformUtils.monotonicTime()
            times.append(PlatformUtils.timeToSeconds(end - start))
        }

        // Compute statistics
        let sorted = times.sorted()
        let mean = times.reduce(0, +) / Double(iterations)

        let median: TimeInterval
        if iterations % 2 == 0 {
            median = (sorted[iterations / 2 - 1] + sorted[iterations / 2]) / 2.0
        } else {
            median = sorted[iterations / 2]
        }

        // BUG FIX #3: Use sample variance (n-1) for unbiased estimator
        let variance = times.map { pow($0 - mean, 2) }.reduce(0, +) / Double(iterations - 1)
        let stdDev = sqrt(variance)

        let minTime = sorted.first!
        let maxTime = sorted.last!
        let throughput = mean > 0 ? 1.0 / mean : 0.0

        return Result(
            meanTime: mean,
            medianTime: median,
            stdDev: stdDev,
            minTime: minTime,
            maxTime: maxTime,
            iterations: iterations,
            throughput: throughput
        )
    }
}

// =============================================================================
// MARK: - 4. Algorithm Comparison Tools
// =============================================================================

/// Compare multiple clustering algorithms
public struct ClusteringComparison {

    public struct ComparisonResult {
        public let algorithmName: String
        public let quality: QualityMetrics
        public let performance: PerformanceMeasurement.Result

        public struct QualityMetrics {
            public let silhouetteScore: Float
            public let daviesBouldinIndex: Float
            public let ari: Float
        }
    }

    /// Compare clustering algorithms on the same dataset (512-dimensional)
    ///
    /// Note: For non-deterministic algorithms, quality metrics reflect the last iteration.
    /// Consider running quality evaluation separately for better statistical rigor.
    public static func compareAlgorithms512(
        vectors: [Vector512Optimized],
        groundTruthLabels: [Int],
        algorithms: [(name: String, cluster: ([Vector512Optimized]) -> [Int])]
    ) -> [ComparisonResult] {

        var results: [ComparisonResult] = []

        for (name, clusterFunc) in algorithms {

            var capturedLabels: [Int] = []

            let perfResult = PerformanceMeasurement.measureClusteringPerformance(
                iterations: 10,
                warmupIterations: 2
            ) {
                capturedLabels = clusterFunc(vectors)
            }

            let predictedLabels = capturedLabels

            guard predictedLabels.count == vectors.count else {
                print("Warning: Algorithm '\(name)' produced invalid label count (\(predictedLabels.count)/\(vectors.count)).")
                continue
            }

            // Compute quality metrics
            let silhouette = ClusteringMetrics.silhouetteScore512(
                vectors: vectors,
                labels: predictedLabels
            )
            let daviesBouldin = ClusteringMetrics.daviesBouldinIndex512(
                vectors: vectors,
                labels: predictedLabels
            )
            let ari = ClusteringMetrics.adjustedRandIndex(
                labels1: groundTruthLabels,
                labels2: predictedLabels
            )

            let quality = ComparisonResult.QualityMetrics(
                silhouetteScore: silhouette,
                daviesBouldinIndex: daviesBouldin,
                ari: ari
            )

            results.append(ComparisonResult(
                algorithmName: name,
                quality: quality,
                performance: perfResult
            ))
        }

        return results
    }
}
