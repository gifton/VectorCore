//
//  MixedPrecisionAutoTuner.swift
//  VectorCore
//
//  Auto-tuning framework for mixed precision kernel selection
//  Implements Phase 2 of kernel-specs/11-mixed-precision-kernels-part2.md
//

import Foundation
import simd

#if canImport(Darwin)
import Darwin
#endif

// MARK: - Precision Strategy

/// Mixed precision execution strategies for batch distance computations
public enum MixedPrecisionStrategy: String, Codable, Sendable, CaseIterable {
    case fullFP32           // Baseline: Full FP32 precision (no conversion)
    case queryFP16Standard  // FP16 query × FP16 SoA, standard kernel
    case queryFP16Blocked   // FP16 query × FP16 SoA, register-blocked kernel (8 candidates)
    case queryFP32Standard  // FP32 query × FP16 SoA, standard kernel (hybrid precision)
    case queryFP32Blocked   // FP32 query × FP16 SoA, register-blocked kernel (hybrid, production)

    /// Human-readable description
    public var description: String {
        switch self {
        case .fullFP32:
            return "Full FP32 (baseline)"
        case .queryFP16Standard:
            return "FP16×FP16 standard"
        case .queryFP16Blocked:
            return "FP16×FP16 blocked (8-way)"
        case .queryFP32Standard:
            return "FP32×FP16 hybrid standard"
        case .queryFP32Blocked:
            return "FP32×FP16 hybrid blocked (8-way)"
        }
    }
}

// MARK: - Performance Metrics

/// Performance and accuracy metrics for a specific strategy
internal struct StrategyMetrics: Codable, Sendable {
    public let strategy: MixedPrecisionStrategy
    public let meanLatency: TimeInterval      // Average execution time
    public let medianLatency: TimeInterval    // Median execution time
    public let throughput: Double             // Operations per second
    public let maxRelativeError: Float        // Maximum relative error vs FP32
    public let meanRelativeError: Float       // Average relative error vs FP32
    public let candidateCount: Int            // Benchmark candidate count
    public let timestamp: Date

    /// Combined quality score: higher is better
    /// Balances performance (throughput) with accuracy (1 - error)
    public var qualityScore: Double {
        guard throughput > 0, maxRelativeError.isFinite else {
            return -Double.infinity
        }

        // Logarithmic throughput score (1-10 range for typical values)
        let throughputScore = log10(max(throughput, 1.0))

        // Accuracy penalty: exponential for errors > 0.1%
        let errorPenalty = Double(maxRelativeError) > 0.001 ?
            -pow(Double(maxRelativeError) * 1000, 2) : 0

        return throughputScore + errorPenalty
    }
}

// MARK: - High-Resolution Timing Utilities

fileprivate struct HighResolutionTimer {

    #if canImport(Darwin)
    private static let timebaseInfo: mach_timebase_info_data_t = {
        var info = mach_timebase_info_data_t()
        mach_timebase_info(&info)
        return info
    }()

    static func now() -> UInt64 {
        return mach_absolute_time()
    }

    static func toSeconds(_ time: UInt64) -> TimeInterval {
        let nanos = Double(time) * Double(timebaseInfo.numer) / Double(timebaseInfo.denom)
        return TimeInterval(nanos) / 1_000_000_000.0
    }
    #else
    static func now() -> UInt64 {
        return UInt64(Date().timeIntervalSince1970 * 1_000_000_000)
    }

    static func toSeconds(_ time: UInt64) -> TimeInterval {
        return TimeInterval(time) / 1_000_000_000.0
    }
    #endif
}

// MARK: - Mixed Precision Auto-Tuner

/// Auto-tuning framework for mixed precision kernel selection
///
/// Automatically selects optimal mixed precision strategy based on:
/// - Workload characteristics (candidate count, accuracy requirements)
/// - Real performance measurements (latency, throughput)
/// - Accuracy validation against FP32 baseline
///
/// ## Usage Example
/// ```swift
/// let tuner = MixedPrecisionAutoTuner.shared
///
/// let strategy = await tuner.selectOptimalStrategy(
///     candidateCount: 1000,
///     accuracyRequirement: 0.001  // 0.1% max error
/// )
///
/// print("Selected: \(strategy.description)")
/// ```
public actor MixedPrecisionAutoTuner {

    // MARK: - Singleton

    public static let shared = MixedPrecisionAutoTuner()

    // MARK: - State

    private var metricsCache: [String: StrategyMetrics] = [:]
    private var strategyOverrides: [String: MixedPrecisionStrategy] = [:]

    // MARK: - Initialization

    private init() {}

    // MARK: - Strategy Selection

    /// Select optimal mixed precision strategy for given workload
    ///
    /// Algorithm:
    /// 1. Check for manual override
    /// 2. Look up cached metrics for this workload
    /// 3. If no cache, run calibration benchmarks
    /// 4. Return best strategy meeting accuracy requirements
    ///
    /// - Parameters:
    ///   - candidateCount: Number of candidate vectors to process
    ///   - accuracyRequirement: Maximum acceptable relative error (default 0.1%)
    ///   - dimension: Vector dimension (512, 768, or 1536)
    /// - Returns: Optimal precision strategy
    public func selectOptimalStrategy(
        candidateCount: Int,
        accuracyRequirement: Float = 0.001,
        dimension: Int = 512
    ) -> MixedPrecisionStrategy {

        let cacheKey = makeCacheKey(
            candidateCount: candidateCount,
            accuracyRequirement: accuracyRequirement,
            dimension: dimension
        )

        // 1. Check override
        if let override = strategyOverrides[cacheKey] {
            return override
        }

        // 2. Check cached metrics
        if let metrics = metricsCache[cacheKey] {
            return metrics.strategy
        }

        // 3. Calibrate and cache
        let metrics = calibrateStrategies(
            candidateCount: candidateCount,
            accuracyRequirement: accuracyRequirement,
            dimension: dimension
        )

        let bestMetrics = metrics.max(by: { $0.qualityScore < $1.qualityScore })!
        metricsCache[cacheKey] = bestMetrics

        return bestMetrics.strategy
    }

    /// Set manual override for specific workload characteristics
    public func setOverride(
        _ strategy: MixedPrecisionStrategy,
        candidateCount: Int,
        accuracyRequirement: Float = 0.001,
        dimension: Int = 512
    ) {
        let cacheKey = makeCacheKey(
            candidateCount: candidateCount,
            accuracyRequirement: accuracyRequirement,
            dimension: dimension
        )
        strategyOverrides[cacheKey] = strategy
    }

    /// Clear all cached metrics and overrides
    public func clearCache() {
        metricsCache.removeAll()
        strategyOverrides.removeAll()
    }

    // MARK: - Calibration

    /// Calibrate all strategies and return metrics
    ///
    /// This runs real benchmarks with synthetic test data and measures:
    /// - Performance (latency, throughput)
    /// - Accuracy (relative error vs FP32 baseline)
    ///
    /// Routes to dimension-specific calibration functions based on dimension parameter.
    ///
    /// - Parameters:
    ///   - candidateCount: Number of candidate vectors to benchmark
    ///   - accuracyRequirement: Maximum acceptable relative error
    ///   - dimension: Vector dimension (512, 768, or 1536)
    /// - Returns: Array of metrics for each strategy
    private func calibrateStrategies(
        candidateCount: Int,
        accuracyRequirement: Float,
        dimension: Int
    ) -> [StrategyMetrics] {

        // Route to dimension-specific calibration
        switch dimension {
        case 512:
            return calibrateStrategies512(
                candidateCount: candidateCount,
                accuracyRequirement: accuracyRequirement
            )
        case 768:
            return calibrateStrategies768(
                candidateCount: candidateCount,
                accuracyRequirement: accuracyRequirement
            )
        case 1536:
            return calibrateStrategies1536(
                candidateCount: candidateCount,
                accuracyRequirement: accuracyRequirement
            )
        default:
            fatalError("Unsupported dimension: \(dimension). Supported dimensions: 512, 768, 1536")
        }
    }

    /// Calibrate all strategies for 512D vectors
    ///
    /// This runs real benchmarks with synthetic 512D test data and measures:
    /// - Performance (latency, throughput)
    /// - Accuracy (relative error vs FP32 baseline)
    ///
    /// - Parameters:
    ///   - candidateCount: Number of candidate vectors to benchmark
    ///   - accuracyRequirement: Maximum acceptable relative error
    /// - Returns: Array of metrics for each strategy
    private func calibrateStrategies512(
        candidateCount: Int,
        accuracyRequirement: Float
    ) -> [StrategyMetrics] {

        // Generate test vectors (reproducible with seed)
        let testVectors = generateTestVectors512(count: candidateCount + 1, seed: 42)
        let query = testVectors[0]
        let candidates = Array(testVectors[1...])

        // Compute FP32 baseline for accuracy validation
        let baselineDistances = computeFP32Baseline(query: query, candidates: candidates)

        var allMetrics: [StrategyMetrics] = []

        for strategy in MixedPrecisionStrategy.allCases {
            let metrics = benchmarkStrategy(
                strategy,
                query: query,
                candidates: candidates,
                baseline: baselineDistances,
                accuracyRequirement: accuracyRequirement
            )

            // Only include strategies that meet accuracy requirement
            if metrics.maxRelativeError <= accuracyRequirement {
                allMetrics.append(metrics)
            }
        }

        // If no strategy meets requirement, fallback to fullFP32
        if allMetrics.isEmpty {
            let fp32Metrics = benchmarkStrategy(
                .fullFP32,
                query: query,
                candidates: candidates,
                baseline: baselineDistances,
                accuracyRequirement: accuracyRequirement
            )
            allMetrics.append(fp32Metrics)
        }

        return allMetrics
    }

    /// Calibrate all strategies for 768D vectors
    ///
    /// This runs real benchmarks with synthetic 768D test data and measures:
    /// - Performance (latency, throughput)
    /// - Accuracy (relative error vs FP32 baseline)
    ///
    /// - Parameters:
    ///   - candidateCount: Number of candidate vectors to benchmark
    ///   - accuracyRequirement: Maximum acceptable relative error
    /// - Returns: Array of metrics for each strategy
    private func calibrateStrategies768(
        candidateCount: Int,
        accuracyRequirement: Float
    ) -> [StrategyMetrics] {

        // Generate test vectors (reproducible with seed)
        let testVectors = generateTestVectors768(count: candidateCount + 1, seed: 42)
        let query = testVectors[0]
        let candidates = Array(testVectors[1...])

        // Compute FP32 baseline for accuracy validation
        let baselineDistances = computeFP32Baseline768(query: query, candidates: candidates)

        var allMetrics: [StrategyMetrics] = []

        for strategy in MixedPrecisionStrategy.allCases {
            let metrics = benchmarkStrategy768(
                strategy,
                query: query,
                candidates: candidates,
                baseline: baselineDistances,
                accuracyRequirement: accuracyRequirement
            )

            // Only include strategies that meet accuracy requirement
            if metrics.maxRelativeError <= accuracyRequirement {
                allMetrics.append(metrics)
            }
        }

        // If no strategy meets requirement, fallback to fullFP32
        if allMetrics.isEmpty {
            let fp32Metrics = benchmarkStrategy768(
                .fullFP32,
                query: query,
                candidates: candidates,
                baseline: baselineDistances,
                accuracyRequirement: accuracyRequirement
            )
            allMetrics.append(fp32Metrics)
        }

        return allMetrics
    }

    /// Calibrate all strategies for 1536D vectors
    ///
    /// This runs real benchmarks with synthetic 1536D test data and measures:
    /// - Performance (latency, throughput)
    /// - Accuracy (relative error vs FP32 baseline)
    ///
    /// - Parameters:
    ///   - candidateCount: Number of candidate vectors to benchmark
    ///   - accuracyRequirement: Maximum acceptable relative error
    /// - Returns: Array of metrics for each strategy
    private func calibrateStrategies1536(
        candidateCount: Int,
        accuracyRequirement: Float
    ) -> [StrategyMetrics] {

        // Generate test vectors (reproducible with seed)
        let testVectors = generateTestVectors1536(count: candidateCount + 1, seed: 42)
        let query = testVectors[0]
        let candidates = Array(testVectors[1...])

        // Compute FP32 baseline for accuracy validation
        let baselineDistances = computeFP32Baseline1536(query: query, candidates: candidates)

        var allMetrics: [StrategyMetrics] = []

        for strategy in MixedPrecisionStrategy.allCases {
            let metrics = benchmarkStrategy1536(
                strategy,
                query: query,
                candidates: candidates,
                baseline: baselineDistances,
                accuracyRequirement: accuracyRequirement
            )

            // Only include strategies that meet accuracy requirement
            if metrics.maxRelativeError <= accuracyRequirement {
                allMetrics.append(metrics)
            }
        }

        // If no strategy meets requirement, fallback to fullFP32
        if allMetrics.isEmpty {
            let fp32Metrics = benchmarkStrategy1536(
                .fullFP32,
                query: query,
                candidates: candidates,
                baseline: baselineDistances,
                accuracyRequirement: accuracyRequirement
            )
            allMetrics.append(fp32Metrics)
        }

        return allMetrics
    }

    /// Benchmark a specific strategy
    private func benchmarkStrategy(
        _ strategy: MixedPrecisionStrategy,
        query: Vector512Optimized,
        candidates: [Vector512Optimized],
        baseline: [Float],
        accuracyRequirement: Float
    ) -> StrategyMetrics {

        let iterations = 20
        let warmupIterations = 5
        var times: [TimeInterval] = []
        times.reserveCapacity(iterations)

        var results = [Float](repeating: 0, count: candidates.count)

        // Warmup
        for _ in 0..<warmupIterations {
            executeStrategy(strategy, query: query, candidates: candidates, results: &results)
        }

        // Benchmark
        for _ in 0..<iterations {
            let start = HighResolutionTimer.now()
            executeStrategy(strategy, query: query, candidates: candidates, results: &results)
            let end = HighResolutionTimer.now()
            times.append(HighResolutionTimer.toSeconds(end - start))
        }

        // Compute statistics
        let sorted = times.sorted()
        let mean = times.reduce(0, +) / Double(iterations)
        let median = sorted[iterations / 2]
        let throughput = mean > 0 ? 1.0 / mean : 0

        // Compute accuracy metrics
        var maxError: Float = 0
        var sumError: Float = 0

        for i in 0..<candidates.count {
            let error = abs(results[i] - baseline[i]) / max(baseline[i], 1e-6)
            maxError = max(maxError, error)
            sumError += error
        }

        let meanError = sumError / Float(candidates.count)

        return StrategyMetrics(
            strategy: strategy,
            meanLatency: mean,
            medianLatency: median,
            throughput: throughput,
            maxRelativeError: maxError,
            meanRelativeError: meanError,
            candidateCount: candidates.count,
            timestamp: Date()
        )
    }

    /// Benchmark a specific strategy (768D)
    private func benchmarkStrategy768(
        _ strategy: MixedPrecisionStrategy,
        query: Vector768Optimized,
        candidates: [Vector768Optimized],
        baseline: [Float],
        accuracyRequirement: Float
    ) -> StrategyMetrics {

        let iterations = 20
        let warmupIterations = 5
        var times: [TimeInterval] = []
        times.reserveCapacity(iterations)

        var results = [Float](repeating: 0, count: candidates.count)

        // Warmup
        for _ in 0..<warmupIterations {
            executeStrategy768(strategy, query: query, candidates: candidates, results: &results)
        }

        // Benchmark
        for _ in 0..<iterations {
            let start = HighResolutionTimer.now()
            executeStrategy768(strategy, query: query, candidates: candidates, results: &results)
            let end = HighResolutionTimer.now()
            times.append(HighResolutionTimer.toSeconds(end - start))
        }

        // Compute statistics
        let sorted = times.sorted()
        let mean = times.reduce(0, +) / Double(iterations)
        let median = sorted[iterations / 2]
        let throughput = mean > 0 ? 1.0 / mean : 0

        // Compute accuracy metrics
        var maxError: Float = 0
        var sumError: Float = 0

        for i in 0..<candidates.count {
            let error = abs(results[i] - baseline[i]) / max(baseline[i], 1e-6)
            maxError = max(maxError, error)
            sumError += error
        }

        let meanError = sumError / Float(candidates.count)

        return StrategyMetrics(
            strategy: strategy,
            meanLatency: mean,
            medianLatency: median,
            throughput: throughput,
            maxRelativeError: maxError,
            meanRelativeError: meanError,
            candidateCount: candidates.count,
            timestamp: Date()
        )
    }

    /// Benchmark a specific strategy (1536D)
    private func benchmarkStrategy1536(
        _ strategy: MixedPrecisionStrategy,
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized],
        baseline: [Float],
        accuracyRequirement: Float
    ) -> StrategyMetrics {

        let iterations = 20
        let warmupIterations = 5
        var times: [TimeInterval] = []
        times.reserveCapacity(iterations)

        var results = [Float](repeating: 0, count: candidates.count)

        // Warmup
        for _ in 0..<warmupIterations {
            executeStrategy1536(strategy, query: query, candidates: candidates, results: &results)
        }

        // Benchmark
        for _ in 0..<iterations {
            let start = HighResolutionTimer.now()
            executeStrategy1536(strategy, query: query, candidates: candidates, results: &results)
            let end = HighResolutionTimer.now()
            times.append(HighResolutionTimer.toSeconds(end - start))
        }

        // Compute statistics
        let sorted = times.sorted()
        let mean = times.reduce(0, +) / Double(iterations)
        let median = sorted[iterations / 2]
        let throughput = mean > 0 ? 1.0 / mean : 0

        // Compute accuracy metrics
        var maxError: Float = 0
        var sumError: Float = 0

        for i in 0..<candidates.count {
            let error = abs(results[i] - baseline[i]) / max(baseline[i], 1e-6)
            maxError = max(maxError, error)
            sumError += error
        }

        let meanError = sumError / Float(candidates.count)

        return StrategyMetrics(
            strategy: strategy,
            meanLatency: mean,
            medianLatency: median,
            throughput: throughput,
            maxRelativeError: maxError,
            meanRelativeError: meanError,
            candidateCount: candidates.count,
            timestamp: Date()
        )
    }

    /// Execute a specific strategy
    private func executeStrategy(
        _ strategy: MixedPrecisionStrategy,
        query: Vector512Optimized,
        candidates: [Vector512Optimized],
        results: inout [Float]
    ) {
        switch strategy {
        case .fullFP32:
            results.withUnsafeMutableBufferPointer { buffer in
                BatchKernels.range_euclid2_512(
                    query: query,
                    candidates: candidates,
                    range: 0..<candidates.count,
                    out: buffer
                )
            }
            // Take sqrt to get distances (range_euclid2_512 returns squared distances)
            for i in 0..<results.count {
                results[i] = sqrt(results[i])
            }

        case .queryFP16Standard:
            let queryFP16 = MixedPrecisionKernels.Vector512FP16(from: query)
            let candidatesSoA = MixedPrecisionKernels.createSoA512FP16(from: candidates)
            results.withUnsafeMutableBufferPointer { buffer in
                MixedPrecisionKernels.batchEuclidean512(
                    query: queryFP16,
                    candidates: candidatesSoA,
                    results: buffer
                )
            }

        case .queryFP16Blocked:
            let queryFP16 = MixedPrecisionKernels.Vector512FP16(from: query)
            let candidatesSoA = MixedPrecisionKernels.createSoA512FP16(from: candidates)
            results.withUnsafeMutableBufferPointer { buffer in
                MixedPrecisionKernels.batchEuclideanBlocked512(
                    query: queryFP16,
                    candidates: candidatesSoA,
                    results: buffer
                )
            }

        case .queryFP32Standard:
            let candidatesSoA = MixedPrecisionKernels.createSoA512FP16(from: candidates)
            results.withUnsafeMutableBufferPointer { buffer in
                MixedPrecisionKernels.batchEuclidean512(
                    query: query,
                    candidates: candidatesSoA,
                    results: buffer
                )
            }

        case .queryFP32Blocked:
            let candidatesSoA = MixedPrecisionKernels.createSoA512FP16(from: candidates)
            results.withUnsafeMutableBufferPointer { buffer in
                MixedPrecisionKernels.batchEuclideanBlocked512(
                    query: query,
                    candidates: candidatesSoA,
                    results: buffer
                )
            }
        }
    }

    /// Execute a specific strategy (768D)
    private func executeStrategy768(
        _ strategy: MixedPrecisionStrategy,
        query: Vector768Optimized,
        candidates: [Vector768Optimized],
        results: inout [Float]
    ) {
        switch strategy {
        case .fullFP32:
            results.withUnsafeMutableBufferPointer { buffer in
                BatchKernels.range_euclid2_768(
                    query: query,
                    candidates: candidates,
                    range: 0..<candidates.count,
                    out: buffer
                )
            }
            // Take sqrt to get distances (range_euclid2_768 returns squared distances)
            for i in 0..<results.count {
                results[i] = sqrt(results[i])
            }

        case .queryFP16Standard:
            let queryFP16 = MixedPrecisionKernels.Vector768FP16(from: query)
            let candidatesSoA = MixedPrecisionKernels.createSoA768FP16(from: candidates)
            results.withUnsafeMutableBufferPointer { buffer in
                MixedPrecisionKernels.batchEuclidean768(
                    query: queryFP16,
                    candidates: candidatesSoA,
                    results: buffer
                )
            }

        case .queryFP16Blocked:
            let queryFP16 = MixedPrecisionKernels.Vector768FP16(from: query)
            let candidatesSoA = MixedPrecisionKernels.createSoA768FP16(from: candidates)
            results.withUnsafeMutableBufferPointer { buffer in
                MixedPrecisionKernels.batchEuclideanBlocked768(
                    query: queryFP16,
                    candidates: candidatesSoA,
                    results: buffer
                )
            }

        case .queryFP32Standard:
            let candidatesSoA = MixedPrecisionKernels.createSoA768FP16(from: candidates)
            results.withUnsafeMutableBufferPointer { buffer in
                MixedPrecisionKernels.batchEuclidean768(
                    query: query,
                    candidates: candidatesSoA,
                    results: buffer
                )
            }

        case .queryFP32Blocked:
            let candidatesSoA = MixedPrecisionKernels.createSoA768FP16(from: candidates)
            results.withUnsafeMutableBufferPointer { buffer in
                MixedPrecisionKernels.batchEuclideanBlocked768(
                    query: query,
                    candidates: candidatesSoA,
                    results: buffer
                )
            }
        }
    }

    /// Execute a specific strategy (1536D)
    private func executeStrategy1536(
        _ strategy: MixedPrecisionStrategy,
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized],
        results: inout [Float]
    ) {
        switch strategy {
        case .fullFP32:
            results.withUnsafeMutableBufferPointer { buffer in
                BatchKernels.range_euclid2_1536(
                    query: query,
                    candidates: candidates,
                    range: 0..<candidates.count,
                    out: buffer
                )
            }
            // Take sqrt to get distances (range_euclid2_1536 returns squared distances)
            for i in 0..<results.count {
                results[i] = sqrt(results[i])
            }

        case .queryFP16Standard:
            let queryFP16 = MixedPrecisionKernels.Vector1536FP16(from: query)
            let candidatesSoA = MixedPrecisionKernels.createSoA1536FP16(from: candidates)
            results.withUnsafeMutableBufferPointer { buffer in
                MixedPrecisionKernels.batchEuclidean1536(
                    query: queryFP16,
                    candidates: candidatesSoA,
                    results: buffer
                )
            }

        case .queryFP16Blocked:
            let queryFP16 = MixedPrecisionKernels.Vector1536FP16(from: query)
            let candidatesSoA = MixedPrecisionKernels.createSoA1536FP16(from: candidates)
            results.withUnsafeMutableBufferPointer { buffer in
                MixedPrecisionKernels.batchEuclideanBlocked1536(
                    query: queryFP16,
                    candidates: candidatesSoA,
                    results: buffer
                )
            }

        case .queryFP32Standard:
            let candidatesSoA = MixedPrecisionKernels.createSoA1536FP16(from: candidates)
            results.withUnsafeMutableBufferPointer { buffer in
                MixedPrecisionKernels.batchEuclidean1536(
                    query: query,
                    candidates: candidatesSoA,
                    results: buffer
                )
            }

        case .queryFP32Blocked:
            let candidatesSoA = MixedPrecisionKernels.createSoA1536FP16(from: candidates)
            results.withUnsafeMutableBufferPointer { buffer in
                MixedPrecisionKernels.batchEuclideanBlocked1536(
                    query: query,
                    candidates: candidatesSoA,
                    results: buffer
                )
            }
        }
    }

    /// Compute FP32 baseline distances
    private func computeFP32Baseline(
        query: Vector512Optimized,
        candidates: [Vector512Optimized]
    ) -> [Float] {
        var baseline = [Float](repeating: 0, count: candidates.count)

        baseline.withUnsafeMutableBufferPointer { buffer in
            BatchKernels.range_euclid2_512(
                query: query,
                candidates: candidates,
                range: 0..<candidates.count,
                out: buffer
            )
        }

        // Take sqrt to get distances
        return baseline.map { sqrt($0) }
    }

    /// Compute FP32 baseline distances (768D)
    private func computeFP32Baseline768(
        query: Vector768Optimized,
        candidates: [Vector768Optimized]
    ) -> [Float] {
        var baseline = [Float](repeating: 0, count: candidates.count)

        baseline.withUnsafeMutableBufferPointer { buffer in
            BatchKernels.range_euclid2_768(
                query: query,
                candidates: candidates,
                range: 0..<candidates.count,
                out: buffer
            )
        }

        // Take sqrt to get distances
        return baseline.map { sqrt($0) }
    }

    /// Compute FP32 baseline distances (1536D)
    private func computeFP32Baseline1536(
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized]
    ) -> [Float] {
        var baseline = [Float](repeating: 0, count: candidates.count)

        baseline.withUnsafeMutableBufferPointer { buffer in
            BatchKernels.range_euclid2_1536(
                query: query,
                candidates: candidates,
                range: 0..<candidates.count,
                out: buffer
            )
        }

        // Take sqrt to get distances
        return baseline.map { sqrt($0) }
    }

    // MARK: - Test Data Generation

    /// Generate synthetic test vectors with reproducible randomness
    private func generateTestVectors512(count: Int, seed: UInt32) -> [Vector512Optimized] {
        var rng = SeededRNG(seed: seed)
        var vectors: [Vector512Optimized] = []
        vectors.reserveCapacity(count)

        for _ in 0..<count {
            var data = [Float](repeating: 0, count: 512)
            for i in 0..<512 {
                data[i] = rng.nextGaussian()
            }
            vectors.append(try! Vector512Optimized(data))
        }

        return vectors
    }

    /// Generate synthetic test vectors with reproducible randomness (768D)
    private func generateTestVectors768(count: Int, seed: UInt32) -> [Vector768Optimized] {
        var rng = SeededRNG(seed: seed)
        var vectors: [Vector768Optimized] = []
        vectors.reserveCapacity(count)

        for _ in 0..<count {
            var data = [Float](repeating: 0, count: 768)
            for i in 0..<768 {
                data[i] = rng.nextGaussian()
            }
            vectors.append(try! Vector768Optimized(data))
        }

        return vectors
    }

    /// Generate synthetic test vectors with reproducible randomness (1536D)
    private func generateTestVectors1536(count: Int, seed: UInt32) -> [Vector1536Optimized] {
        var rng = SeededRNG(seed: seed)
        var vectors: [Vector1536Optimized] = []
        vectors.reserveCapacity(count)

        for _ in 0..<count {
            var data = [Float](repeating: 0, count: 1536)
            for i in 0..<1536 {
                data[i] = rng.nextGaussian()
            }
            vectors.append(try! Vector1536Optimized(data))
        }

        return vectors
    }

    // MARK: - Utilities

    private func makeCacheKey(
        candidateCount: Int,
        accuracyRequirement: Float,
        dimension: Int
    ) -> String {
        return "\(dimension)_\(candidateCount)_\(accuracyRequirement)"
    }
}

// MARK: - Seeded Random Number Generator

/// Simple seeded RNG for reproducible test data generation
private struct SeededRNG {
    private var state: UInt64

    init(seed: UInt32) {
        self.state = UInt64(seed)
    }

    mutating func next() -> Float {
        // Linear congruential generator (simple but reproducible)
        state = state &* 1664525 &+ 1013904223
        return Float(state & 0xFFFFFF) / Float(0xFFFFFF)
    }

    /// Box-Muller transform for Gaussian random numbers
    mutating func nextGaussian() -> Float {
        var u1 = next()
        while u1 == 0 {
            u1 = next()
        }
        let u2 = next()

        let r = sqrt(-2.0 * log(u1))
        let theta = 2.0 * Float.pi * u2

        return r * cos(theta)
    }
}
