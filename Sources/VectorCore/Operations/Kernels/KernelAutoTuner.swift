//
//  KernelAutoTuner.swift
//  VectorCore
//
//  Kernel auto-tuning framework for optimal strategy selection
//  Based on kernel-specs/28-kernel-auto-tuning-framework.md
//

import Foundation
import simd

#if canImport(Darwin)
import Darwin
#endif

// MARK: - 1. Kernel Strategy Enum

/// Available kernel execution strategies
///
/// Strategies represent distinct implementation approaches, not configuration
/// parameters. Each strategy maps to a specific kernel implementation with
/// different performance characteristics.
public enum KernelStrategy: String, Codable, Sendable {
    // CPU Strategies
    case cpuAoS             // Array-of-Structures (simple loops, minimal overhead)
    case cpuSoA             // Structure-of-Arrays (auto-selects 2-way/4-way blocking)
    case cpuMixedPrecision  // FP16 storage, FP32 compute (2x memory reduction)
    case cpuQuantized       // INT8 quantized (4x memory reduction)

    // Adaptive
    case auto               // Auto-select based on workload (delegates to heuristic)
}

// MARK: - 2. Workload Characteristics

/// Characteristics of a computation workload
public struct WorkloadCharacteristics: Sendable {
    public let dimension: Int           // Vector dimension (512, 768, 1536)
    public let batchSize: Int           // Number of candidates
    public let precisionMode: PrecisionMode
    public let memoryBudget: Int?       // Optional memory constraint (bytes)
    public let latencyTarget: TimeInterval?  // Optional latency requirement

    public enum PrecisionMode: String, Codable, Sendable {
        case fp32Full           // Full FP32 precision required
        case fp16Acceptable     // FP16 error acceptable (< 0.1%)
        case int8Acceptable     // INT8 error acceptable (< 1%)
    }

    public init(
        dimension: Int,
        batchSize: Int,
        precisionMode: PrecisionMode = .fp16Acceptable,
        memoryBudget: Int? = nil,
        latencyTarget: TimeInterval? = nil
    ) {
        self.dimension = dimension
        self.batchSize = batchSize
        self.precisionMode = precisionMode
        self.memoryBudget = memoryBudget
        self.latencyTarget = latencyTarget
    }
}

// MARK: - 3. Performance Profile

/// Performance profile for a specific kernel + workload combination
public struct PerformanceProfile: Codable, Sendable {
    public let strategy: KernelStrategy
    public let workload: String         // Hash of workload characteristics
    public let throughput: Double       // Operations/sec
    public let latencyMean: TimeInterval
    public let latencyP50: TimeInterval
    public let latencyP95: TimeInterval
    public let latencyP99: TimeInterval
    public let memoryFootprint: Int     // Bytes
    public let accuracy: Float          // Relative to FP32 baseline
    public let timestamp: Date

    /// Quality score: balances throughput, latency, memory, accuracy
    public var score: Double {
        guard throughput > 0, latencyMean > 0, memoryFootprint > 0 else {
            return -Double.infinity
        }

        // Clamp inputs to prevent extreme scores from measurement noise
        let clampedThroughput = min(throughput, 1e9)  // 1B ops/sec max
        let clampedLatency = max(latencyMean * 1000, 1e-3)  // 1μs min
        let clampedMemory = max(Double(memoryFootprint) / 1_000_000, 0.001)  // 1KB min

        let throughputScore = log10(clampedThroughput)
        let latencyScore = -log10(clampedLatency)  // Lower is better (ms)
        let memoryScore = -log10(clampedMemory)  // Lower is better (MB)
        let accuracyScore = Double(accuracy) * 10  // 0-10 scale

        return throughputScore + latencyScore + memoryScore + accuracyScore
    }
}

// MARK: - 4. Auto-Tuner Core

/// Kernel auto-tuning framework
///
/// Automatically selects optimal kernel execution strategy based on workload
/// characteristics, with optional profiling and persistent caching.
///
/// ## Usage Example
/// ```swift
/// let workload = WorkloadCharacteristics(
///     dimension: 512,
///     batchSize: 1000,
///     precisionMode: .fp16Acceptable
/// )
///
/// // Automatic selection
/// let strategy = await KernelAutoTuner.shared.selectStrategy(for: workload)
///
/// // Profile all strategies
/// let profiles = await KernelAutoTuner.shared.profileWorkload(workload)
/// ```
public actor KernelAutoTuner {

    // MARK: - Singleton

    public static let shared = KernelAutoTuner()

    // MARK: - State

    private var profileCache: [String: [PerformanceProfile]] = [:]
    private var strategyOverrides: [String: KernelStrategy] = [:]
    private let persistenceURL: URL?

    // MARK: - Initialization

    private init(persistenceURL: URL? = nil) {
        // Disable persistence during unit tests or when explicitly requested
        let env = ProcessInfo.processInfo.environment
        let disableCache = env["VECTORCORE_DISABLE_TUNER_CACHE"] == "1" || env["XCTestConfigurationFilePath"] != nil

        self.persistenceURL = disableCache ? nil : (persistenceURL ?? Self.defaultPersistenceURL())
        // Note: Cannot call loadProfiles() synchronously in actor init
        // Profiles will be loaded lazily on first access
    }

    private var profilesLoaded = false

    private func ensureProfilesLoaded() {
        guard !profilesLoaded else { return }
        profilesLoaded = true
        loadProfiles()
    }

    private static func defaultPersistenceURL() -> URL? {
        do {
            let appSupportURL = try FileManager.default
                .url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)

            return appSupportURL
                .appendingPathComponent("VectorCore")
                .appendingPathComponent("kernel_profiles.json")
        } catch {
            print("KernelAutoTuner: Error determining default persistence URL: \(error)")
            return nil
        }
    }

    // MARK: - Strategy Selection

    /// Select optimal kernel strategy for given workload
    ///
    /// Algorithm:
    /// 1. Check for manual override
    /// 2. Look up cached profile for this workload
    /// 3. If no profile, use heuristic-based selection
    /// 4. Return selected strategy
    public func selectStrategy(
        for workload: WorkloadCharacteristics
    ) -> KernelStrategy {
        ensureProfilesLoaded()

        let workloadKey = hashWorkload(workload)

        // When tuner cache is disabled (e.g., during tests), skip profile-based selection
        // but still honor manual overrides.
        let env = ProcessInfo.processInfo.environment
        let cacheDisabled = env["VECTORCORE_DISABLE_TUNER_CACHE"] == "1" || env["XCTestConfigurationFilePath"] != nil

        // 1. Check override
        if let override = strategyOverrides[workloadKey] {
            return override
        }

        if !cacheDisabled {
            // 2. Check cached profiles
            if let profiles = profileCache[workloadKey],
               let best = profiles.max(by: { $0.score < $1.score }) {
                return best.strategy
            }
        }

        // 3. Fallback: heuristic-based selection
        return selectStrategyHeuristic(for: workload)
    }

    /// Set manual override for a specific workload (persists across restarts)
    public func setOverride(_ strategy: KernelStrategy, for workload: WorkloadCharacteristics) {
        let workloadKey = hashWorkload(workload)
        strategyOverrides[workloadKey] = strategy
    }

    /// Remove manual override for a specific workload
    public func removeOverride(for workload: WorkloadCharacteristics) {
        let workloadKey = hashWorkload(workload)
        strategyOverrides.removeValue(forKey: workloadKey)
    }

    /// Heuristic-based strategy selection (no profiling data)
    ///
    /// Selection algorithm based on batch size and precision requirements:
    /// - Small batches (N < 100): cpuAoS (minimal overhead)
    /// - Medium batches (100 ≤ N < 1000): cpuSoA with appropriate precision
    /// - Large batches (N ≥ 1000): cpuSoA or reduced-precision variants
    ///
    /// Precision modes:
    /// - fp32Full: Always uses FP32 (cpuAoS or cpuSoA)
    /// - fp16Acceptable: Uses cpuMixedPrecision (2x memory savings)
    /// - int8Acceptable: Uses cpuQuantized (4x memory savings)
    private func selectStrategyHeuristic(
        for workload: WorkloadCharacteristics
    ) -> KernelStrategy {
        // Small batches (N < 100): Simple AoS is fastest
        // Overhead of SoA transposition not worthwhile for small N
        if workload.batchSize < 100 {
            if workload.precisionMode == .fp32Full {
                return .cpuAoS
            } else {
                // Even for small batches, mixed precision provides 2x memory savings
                return .cpuMixedPrecision
            }
        }

        // Medium batches (100 ≤ N < 1000): SoA starts to pay off
        if workload.batchSize < 1000 {
            switch workload.precisionMode {
            case .fp32Full:
                return .cpuSoA  // SoA with auto-selected 2-way/4-way blocking
            case .fp16Acceptable:
                return .cpuMixedPrecision
            case .int8Acceptable:
                return .cpuQuantized
            }
        }

        // Large batches (N ≥ 1000): SoA provides best cache locality
        // BatchKernels_SoA internally selects 4-way blocking for N ≥ 4
        switch workload.precisionMode {
        case .fp32Full:
            return .cpuSoA
        case .fp16Acceptable:
            return .cpuMixedPrecision
        case .int8Acceptable:
            return .cpuQuantized
        }
    }

    // MARK: - Profiling

    /// Benchmark all available strategies for this workload
    ///
    /// Runs each strategy multiple times, collects statistics, caches results.
    /// Uses the same test data for all strategies to ensure fair comparison.
    public func profileWorkload(
        _ workload: WorkloadCharacteristics,
        iterations: Int = 100,
        warmupIterations: Int = 10
    ) async -> [PerformanceProfile] {

        var profiles: [PerformanceProfile] = []
        let strategies = availableStrategies(for: workload)

        // Generate test data ONCE for all strategies (ensures fair comparison)
        guard let testData = generateTestData(for: workload) else {
            print("KernelAutoTuner: Failed to generate test data for dimension \(workload.dimension)")
            return []
        }

        for strategy in strategies {
            if let profile = await benchmarkStrategy(
                strategy,
                workload: workload,
                testData: testData,
                iterations: iterations,
                warmupIterations: warmupIterations
            ) {
                profiles.append(profile)
            }
        }

        // Cache results
        let workloadKey = hashWorkload(workload)
        if !profiles.isEmpty {
            profileCache[workloadKey] = profiles
            saveProfiles()
        }

        return profiles.sorted { $0.score > $1.score }
    }

    /// Benchmark a specific strategy
    private func benchmarkStrategy(
        _ strategy: KernelStrategy,
        workload: WorkloadCharacteristics,
        testData: TestData,
        iterations: Int,
        warmupIterations: Int
    ) async -> PerformanceProfile? {

        guard iterations > 0 else { return nil }

        // Warmup phase
        for _ in 0..<warmupIterations {
            _ = try? executeKernel(strategy, testData: testData)
        }

        // Benchmark phase
        var latencies: [TimeInterval] = []
        latencies.reserveCapacity(iterations)

        for _ in 0..<iterations {
            #if canImport(Darwin)
            let start = mach_absolute_time()
            guard (try? executeKernel(strategy, testData: testData)) != nil else {
                return nil  // Strategy failed
            }
            let end = mach_absolute_time()
            latencies.append(machTimeToSeconds(end - start))
            #else
            // Fallback for non-Darwin platforms
            if #available(macOS 13.0, iOS 16.0, tvOS 16.0, watchOS 9.0, *) {
                let clock = ContinuousClock()
                do {
                    let duration = try clock.measure {
                        _ = try executeKernel(strategy, testData: testData)
                    }
                    latencies.append(duration.timeInterval)
                } catch {
                    return nil
                }
            } else {
                return nil  // Cannot profile on older platforms
            }
            #endif
        }

        // Compute statistics
        latencies.sort()
        let mean = latencies.reduce(0, +) / Double(iterations)

        // Safe percentile calculation (nearest-rank method)
        func percentile(_ p: Double) -> TimeInterval {
            let index = Int(Double(iterations) * p)
            return latencies[max(0, min(iterations - 1, index))]
        }

        let p50 = percentile(0.50)
        let p95 = percentile(0.95)
        let p99 = percentile(0.99)

        let throughput = mean > 0 ? Double(workload.batchSize) / mean : 0

        // Measure memory footprint
        let memoryFootprint = estimateMemoryFootprint(strategy, workload: workload)

        // Measure accuracy (compare to FP32 baseline)
        let accuracy = measureAccuracy(strategy, testData: testData)

        return PerformanceProfile(
            strategy: strategy,
            workload: hashWorkload(workload),
            throughput: throughput,
            latencyMean: mean,
            latencyP50: p50,
            latencyP95: p95,
            latencyP99: p99,
            memoryFootprint: memoryFootprint,
            accuracy: accuracy,
            timestamp: Date()
        )
    }

    // MARK: - Kernel Execution

    /// Execute kernel with selected strategy (internal testing)
    private nonisolated func executeKernel(
        _ strategy: KernelStrategy,
        testData: TestData
    ) throws -> [Float] {
        switch testData {
        case .dim512(let data):
            return try executeKernel512(strategy, data: data)
        case .dim768(let data):
            return try executeKernel768(strategy, data: data)
        case .dim1536(let data):
            return try executeKernel1536(strategy, data: data)
        }
    }

    private nonisolated func executeKernel512(
        _ strategy: KernelStrategy,
        data: TestData512
    ) throws -> [Float] {
        switch strategy {
        case .cpuAoS:
            // Simple loop over candidates - minimal overhead
            return data.candidates.map { candidate in
                DotKernels.dot512(data.query512, candidate)
            }

        case .cpuSoA:
            // Structure-of-Arrays layout with automatic 2-way/4-way blocking
            // BatchKernels_SoA.batchDotProduct512 internally selects:
            //   - 2-way blocking for N < 4
            //   - 4-way blocking for N >= 4
            return BatchKernels_SoA.batchDotProduct512(
                query: data.query512,
                candidates: data.candidates
            )

        case .cpuMixedPrecision:
            // FP16 storage with FP32 computation (2x memory reduction)
            return data.candidatesFP16.map { candidate in
                MixedPrecisionKernels.dotMixed512(
                    query: data.query512,
                    candidate: candidate
                )
            }

        case .cpuQuantized:
            // INT8 quantized storage (4x memory reduction)
            return data.candidatesINT8.map { candidate in
                QuantizedKernels.dotProduct512(
                    query: data.queryINT8,
                    candidate: candidate
                )
            }

        case .auto:
            throw VectorError(
                kind: .invalidOperation,
                context: ErrorContext(additionalInfo: [
                    "reason": "Cannot execute .auto strategy directly",
                    "solution": "Use selectStrategy() first"
                ])
            )
        }
    }

    private nonisolated func executeKernel768(
        _ strategy: KernelStrategy,
        data: TestData768
    ) throws -> [Float] {
        switch strategy {
        case .cpuAoS:
            return data.candidates.map { candidate in
                DotKernels.dot768(data.query768, candidate)
            }

        case .cpuSoA:
            return BatchKernels_SoA.batchDotProduct768(
                query: data.query768,
                candidates: data.candidates
            )

        case .cpuMixedPrecision:
            return data.candidatesFP16.map { candidate in
                MixedPrecisionKernels.dotMixed768(
                    query: data.query768,
                    candidate: candidate
                )
            }

        case .cpuQuantized:
            return data.candidatesINT8.map { candidate in
                QuantizedKernels.dotProduct768(
                    query: data.queryINT8,
                    candidate: candidate
                )
            }

        case .auto:
            throw VectorError(
                kind: .invalidOperation,
                context: ErrorContext(additionalInfo: ["strategy": "\(strategy)"])
            )
        }
    }

    private nonisolated func executeKernel1536(
        _ strategy: KernelStrategy,
        data: TestData1536
    ) throws -> [Float] {
        switch strategy {
        case .cpuAoS:
            return data.candidates.map { candidate in
                DotKernels.dot1536(data.query1536, candidate)
            }

        case .cpuSoA:
            return BatchKernels_SoA.batchDotProduct1536(
                query: data.query1536,
                candidates: data.candidates
            )

        case .cpuMixedPrecision:
            return data.candidatesFP16.map { candidate in
                MixedPrecisionKernels.dotMixed1536(
                    query: data.query1536,
                    candidate: candidate
                )
            }

        case .cpuQuantized:
            return data.candidatesINT8.map { candidate in
                QuantizedKernels.dotProduct1536(
                    query: data.queryINT8,
                    candidate: candidate
                )
            }

        case .auto:
            throw VectorError(
                kind: .invalidOperation,
                context: ErrorContext(additionalInfo: ["strategy": "\(strategy)"])
            )
        }
    }

    // MARK: - Helper Types

    /// Dimension-specific test data (ensures type safety)
    private enum TestData: Sendable {
        case dim512(TestData512)
        case dim768(TestData768)
        case dim1536(TestData1536)
    }

    private struct TestData512: Sendable {
        let query512: Vector512Optimized
        let queryINT8: Vector512INT8
        let candidates: [Vector512Optimized]
        let candidatesFP16: [MixedPrecisionKernels.Vector512FP16]
        let candidatesINT8: [Vector512INT8]
    }

    private struct TestData768: Sendable {
        let query768: Vector768Optimized
        let queryINT8: Vector768INT8
        let candidates: [Vector768Optimized]
        let candidatesFP16: [MixedPrecisionKernels.Vector768FP16]
        let candidatesINT8: [Vector768INT8]
    }

    private struct TestData1536: Sendable {
        let query1536: Vector1536Optimized
        let queryINT8: Vector1536INT8
        let candidates: [Vector1536Optimized]
        let candidatesFP16: [MixedPrecisionKernels.Vector1536FP16]
        let candidatesINT8: [Vector1536INT8]
    }

    // MARK: - Helper Functions

    private func hashWorkload(_ workload: WorkloadCharacteristics) -> String {
        "\(workload.dimension)_\(workload.batchSize)_\(workload.precisionMode.rawValue)"
    }

    private func availableStrategies(
        for workload: WorkloadCharacteristics
    ) -> [KernelStrategy] {
        // Always profile FP32 strategies
        var strategies: [KernelStrategy] = [.cpuAoS, .cpuSoA]

        // Add reduced-precision strategies if acceptable
        if workload.precisionMode != .fp32Full {
            strategies.append(.cpuMixedPrecision)
        }

        if workload.precisionMode == .int8Acceptable {
            strategies.append(.cpuQuantized)
        }

        return strategies
    }

    private func generateTestData(for workload: WorkloadCharacteristics) -> TestData? {
        switch workload.dimension {
        case 512:
            let query = Vector512Optimized(generator: { _ in Float.random(in: -1.0...1.0) })
            let candidates = (0..<workload.batchSize).map { _ in
                Vector512Optimized(generator: { _ in Float.random(in: -1.0...1.0) })
            }

            let queryINT8 = Vector512INT8(from: query, params: nil)
            let candidatesFP16 = candidates.map { MixedPrecisionKernels.Vector512FP16(from: $0) }
            let candidatesINT8 = candidates.map { Vector512INT8(from: $0, params: nil) }

            return .dim512(TestData512(
                query512: query,
                queryINT8: queryINT8,
                candidates: candidates,
                candidatesFP16: candidatesFP16,
                candidatesINT8: candidatesINT8
            ))

        case 768:
            let query = Vector768Optimized(generator: { _ in Float.random(in: -1.0...1.0) })
            let candidates = (0..<workload.batchSize).map { _ in
                Vector768Optimized(generator: { _ in Float.random(in: -1.0...1.0) })
            }

            let queryINT8 = Vector768INT8(from: query, params: nil)
            let candidatesFP16 = candidates.map { MixedPrecisionKernels.Vector768FP16(from: $0) }
            let candidatesINT8 = candidates.map { Vector768INT8(from: $0, params: nil) }

            return .dim768(TestData768(
                query768: query,
                queryINT8: queryINT8,
                candidates: candidates,
                candidatesFP16: candidatesFP16,
                candidatesINT8: candidatesINT8
            ))

        case 1536:
            let query = Vector1536Optimized(generator: { _ in Float.random(in: -1.0...1.0) })
            let candidates = (0..<workload.batchSize).map { _ in
                Vector1536Optimized(generator: { _ in Float.random(in: -1.0...1.0) })
            }

            let queryINT8 = Vector1536INT8(from: query, params: nil)
            let candidatesFP16 = candidates.map { MixedPrecisionKernels.Vector1536FP16(from: $0) }
            let candidatesINT8 = candidates.map { Vector1536INT8(from: $0, params: nil) }

            return .dim1536(TestData1536(
                query1536: query,
                queryINT8: queryINT8,
                candidates: candidates,
                candidatesFP16: candidatesFP16,
                candidatesINT8: candidatesINT8
            ))

        default:
            return nil  // Unsupported dimension
        }
    }

    private func estimateMemoryFootprint(
        _ strategy: KernelStrategy,
        workload: WorkloadCharacteristics
    ) -> Int {
        let dim = workload.dimension
        let n = workload.batchSize

        switch strategy {
        case .cpuAoS, .cpuSoA:
            // FP32: 4 bytes/element
            // Query + N candidates
            return (n + 1) * dim * 4

        case .cpuMixedPrecision:
            // Query: FP32 (4 bytes/element)
            // Candidates: FP16 (2 bytes/element)
            return dim * 4 + n * dim * 2

        case .cpuQuantized:
            // INT8: 1 byte/element
            // Note: This is simplified - actual storage includes quantization params
            return (n + 1) * dim * 1

        case .auto:
            return 0
        }
    }

    private nonisolated func measureAccuracy(
        _ strategy: KernelStrategy,
        testData: TestData
    ) -> Float {
        // Compute FP32 baseline (using cpuAoS as reference)
        guard let baseline = try? executeKernel(.cpuAoS, testData: testData) else {
            return 0.0
        }

        // Compute with target strategy
        guard let results = try? executeKernel(strategy, testData: testData) else {
            return 0.0
        }

        guard results.count == baseline.count, !baseline.isEmpty else {
            return 0.0
        }

        // Compute Mean Relative Error
        var totalError: Double = 0
        for (result, truth) in zip(results, baseline) {
            let denominator = max(abs(Double(truth)), 1e-6)
            let relativeError = abs(Double(result - truth)) / denominator
            totalError += relativeError
        }

        let meanRelativeError = Float(totalError / Double(results.count))
        return 1.0 - min(meanRelativeError, 1.0)  // 1.0 = perfect, 0.0 = terrible
    }

    #if canImport(Darwin)
    private nonisolated func machTimeToSeconds(_ time: UInt64) -> TimeInterval {
        var timebase = mach_timebase_info_data_t()

        guard mach_timebase_info(&timebase) == KERN_SUCCESS else {
            return 0
        }

        let numer = Double(timebase.numer)
        let denom = Double(timebase.denom)

        guard denom != 0 else { return 0 }

        let nanos = Double(time) * numer / denom
        return TimeInterval(nanos) / 1_000_000_000.0
    }
    #endif

    // MARK: - Persistence

    private func loadProfiles() {
        guard let url = persistenceURL,
              FileManager.default.fileExists(atPath: url.path) else {
            return
        }

        do {
            let data = try Data(contentsOf: url)
            profileCache = try JSONDecoder().decode([String: [PerformanceProfile]].self, from: data)
        } catch {
            print("KernelAutoTuner: Failed to load profiles from \(url): \(error)")
        }
    }

    private func saveProfiles() {
        guard let url = persistenceURL else { return }

        do {
            let data = try JSONEncoder().encode(profileCache)

            let directory = url.deletingLastPathComponent()
            try FileManager.default.createDirectory(
                at: directory,
                withIntermediateDirectories: true
            )

            try data.write(to: url, options: .atomic)
        } catch {
            print("KernelAutoTuner: Failed to save profiles to \(url): \(error)")
        }
    }

    // MARK: - Mixed Precision Integration

    /// Select optimal mixed precision strategy for given workload
    ///
    /// When the high-level strategy is .cpuMixedPrecision, this delegates to
    /// MixedPrecisionAutoTuner for fine-grained variant selection (standard vs blocked,
    /// FP16 query vs FP32 query, etc.).
    ///
    /// - Parameters:
    ///   - workload: Workload characteristics
    /// - Returns: Specific mixed precision strategy
    public func selectMixedPrecisionStrategy(
        for workload: WorkloadCharacteristics
    ) async -> MixedPrecisionStrategy {
        // Map PrecisionMode to accuracy requirement
        let accuracyRequirement: Float = switch workload.precisionMode {
        case .fp32Full:
            0.0001  // Very strict: 0.01% error
        case .fp16Acceptable:
            0.001   // Standard: 0.1% error
        case .int8Acceptable:
            0.01    // Relaxed: 1% error
        }

        return await MixedPrecisionAutoTuner.shared.selectOptimalStrategy(
            candidateCount: workload.batchSize,
            accuracyRequirement: accuracyRequirement,
            dimension: workload.dimension
        )
    }
}

// MARK: - Platform Compatibility Extensions

#if !canImport(Darwin)
@available(macOS 13.0, iOS 16.0, tvOS 16.0, watchOS 9.0, *)
extension Duration {
    var timeInterval: TimeInterval {
        let (sec, attosec) = self.components
        return Double(sec) + Double(attosec) * 1e-18
    }
}
#endif
