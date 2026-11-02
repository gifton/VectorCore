import Foundation
import simd

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

/// Synthetic conversion-inclusive Euclidean benchmark results. Latencies are
/// seconds per complete invocation; throughput is invocations per second. Errors
/// use a 1e-6 denominator floor on this dataset only, not an arbitrary-data bound.
public struct StrategyMetrics: Codable, Sendable {
    public let strategy: MixedPrecisionStrategy
    public let meanLatency: TimeInterval
    public let medianLatency: TimeInterval
    public let throughput: Double
    public let maxRelativeError: Double
    public let meanRelativeError: Double
    public let candidateCount: Int
    public let dimension: Int
    public let measuredSamples: Int
    public let warmupIterations: Int
    public let algorithmVersion: Int
    public let datasetVersion: String
    public let seed: UInt32
    public let conversionBoundary: String
    public let timestamp: Date

    internal var isValid: Bool {
        meanLatency.isFinite && meanLatency > 0 && medianLatency.isFinite && medianLatency > 0 &&
        throughput.isFinite && throughput > 0 && maxRelativeError.isFinite && maxRelativeError >= 0 &&
        meanRelativeError.isFinite && meanRelativeError >= 0 && meanRelativeError <= maxRelativeError &&
        candidateCount > 0 && [512, 768, 1536].contains(dimension) && measuredSamples > 0 && warmupIterations >= 0 &&
        algorithmVersion == 1 && datasetVersion == "gaussian-lcg-boxmuller-v1" && seed == 42 &&
        conversionBoundary == "fp32-input-conversion-per-reduced-euclidean-invocation-v1" &&
        timestamp.timeIntervalSince1970.isFinite
    }
    public var qualityScore: Double {
        guard isValid else { return -.infinity }
        let penalty = maxRelativeError > 0.001 ? pow(maxRelativeError * 1000, 2) : 0
        let score = log10(max(throughput, 1)) - penalty
        return score.isFinite ? score : -.infinity
    }
}

/// Selection only consults explicit synthetic profiles; a miss returns fullFP32.
/// Calibration is opt-in, conversion-inclusive work and has no hard time deadline.
/// PrecisionSelectionContractTests exercises selection, work and publication policy.
public actor MixedPrecisionAutoTuner {
    public static let shared = MixedPrecisionAutoTuner()
    private struct Family: Hashable { let dimension: Int; let candidateCount: Int }
    private struct Request: Hashable { let family: Family; let limitBits: UInt32 }
    private var profiles: [Family: [StrategyMetrics]] = [:]
    private var overrides: [Request: MixedPrecisionStrategy] = [:]
    private let hooks: TuningTestHooks?
    internal private(set) var workCounts: TuningWorkCounts?

    public init() { hooks = nil; workCounts = nil }
    internal init(testing hooks: TuningTestHooks) { self.hooks = hooks; workCounts = TuningWorkCounts() }

    private func checkpoint(_ event: TuningCheckpoint) throws {
        workCounts?.record(event)
        try hooks?.observer?(event)
        try Task.checkCancellation()
    }
    private func request(count: Int, dimension: Int, limit: Float) throws -> Request {
        try PrecisionTuningSupport.validateShape(count: count, dimension: dimension)
        try PrecisionTuningSupport.validateLimit(limit, field: "maximumSyntheticRelativeError")
        return Request(family: Family(dimension: dimension, candidateCount: count), limitBits: limit == 0 ? 0 : limit.bitPattern)
    }
    private func qualifying(_ family: Family, limit: Float) -> [StrategyMetrics] {
        (profiles[family] ?? []).filter {
            $0.dimension == family.dimension && $0.candidateCount == family.candidateCount &&
            $0.isValid && $0.qualityScore.isFinite && $0.maxRelativeError <= Double(limit)
        }
    }
    /// Performs no generation, conversion or kernel invocation. Error filters are
    /// re-applied on every lookup. Zero count/limit dominates overrides and profiles.
    public func selectStrategy(candidateCount: Int, maximumSyntheticRelativeError: Float = 0.001,
                               dimension: Int = 512) throws -> MixedPrecisionStrategy {
        let key = try request(count: candidateCount, dimension: dimension, limit: maximumSyntheticRelativeError)
        guard candidateCount > 0 && maximumSyntheticRelativeError > 0 else { return .fullFP32 }
        let available = qualifying(key.family, limit: maximumSyntheticRelativeError)
        if let override = overrides[key] {
            guard override == .fullFP32 || available.contains(where: { $0.strategy == override }) else {
                throw PrecisionTuningSupport.invalid("override", "no compatible exact-workload synthetic profile")
            }
            return override
        }
        return available.sorted {
            if $0.qualityScore != $1.qualityScore { return $0.qualityScore > $1.qualityScore }
            return MixedPrecisionStrategy.allCases.firstIndex(of: $0.strategy)! < MixedPrecisionStrategy.allCases.firstIndex(of: $1.strategy)!
        }.first?.strategy ?? .fullFP32
    }
    /// Reduced overrides require an existing compatible exact-count profile. This
    /// is a synthetic-dataset filter and does not certify arbitrary-data accuracy.
    public func setOverride(_ strategy: MixedPrecisionStrategy, candidateCount: Int,
                            maximumSyntheticRelativeError: Float = 0.001, dimension: Int = 512) throws {
        let key = try request(count: candidateCount, dimension: dimension, limit: maximumSyntheticRelativeError)
        guard strategy == .fullFP32 || (candidateCount > 0 && maximumSyntheticRelativeError > 0 &&
            qualifying(key.family, limit: maximumSyntheticRelativeError).contains(where: { $0.strategy == strategy })) else {
            throw PrecisionTuningSupport.invalid("override", "reduced override requires a qualifying exact-workload profile")
        }
        overrides[key] = strategy
    }
    public func clearCache() { profiles.removeAll(); overrides.removeAll() }
    internal func setProfilesForTesting(_ metrics: [StrategyMetrics], candidateCount: Int, dimension: Int) {
        profiles[Family(dimension: dimension, candidateCount: candidateCount)] = metrics
    }
    internal func profilesForTesting(candidateCount: Int, dimension: Int) -> [StrategyMetrics] {
        profiles[Family(dimension: dimension, candidateCount: candidateCount)] ?? []
    }

    /// Computes one FP32 reference and executes each of five strategies exactly
    /// warmupIterations + iterations times. Cancellation before the final commit
    /// preserves profiles/overrides; cancellation after commit does not roll it back.
    /// Synchronous kernels themselves are not interruptible.
    public func calibrate(candidateCount: Int, dimension: Int = 512, iterations: Int = 20,
                          warmupIterations: Int = 5) async throws -> [StrategyMetrics] {
        try PrecisionTuningSupport.validateCalibration(count: candidateCount, dimension: dimension,
            iterations: iterations, warmups: warmupIterations, strategies: 5)
        try Task.checkCancellation()
        var staged: [StrategyMetrics] = []
        switch dimension {
        case 512:
            let vectors = try generateTestVectors512(count: candidateCount + 1, seed: 42)
            let query = vectors[0], candidates = Array(vectors.dropFirst())
            try Task.checkCancellation()
            let baseline = computeFP32Baseline(query: query, candidates: candidates)
            try checkpoint(.baseline)
            for strategy in MixedPrecisionStrategy.allCases {
                if let metric = try benchmark(strategy, dimension: dimension, count: candidateCount,
                    baseline: baseline, iterations: iterations, warmups: warmupIterations, execute: { results in
                        try executeStrategy(strategy, query: query, candidates: candidates, results: &results)
                    }) { staged.append(metric) }
            }
        case 768:
            let vectors = try generateTestVectors768(count: candidateCount + 1, seed: 42)
            let query = vectors[0], candidates = Array(vectors.dropFirst())
            try Task.checkCancellation()
            let baseline = computeFP32Baseline768(query: query, candidates: candidates)
            try checkpoint(.baseline)
            for strategy in MixedPrecisionStrategy.allCases {
                if let metric = try benchmark(strategy, dimension: dimension, count: candidateCount,
                    baseline: baseline, iterations: iterations, warmups: warmupIterations, execute: { results in
                        try executeStrategy768(strategy, query: query, candidates: candidates, results: &results)
                    }) { staged.append(metric) }
            }
        case 1536:
            let vectors = try generateTestVectors1536(count: candidateCount + 1, seed: 42)
            let query = vectors[0], candidates = Array(vectors.dropFirst())
            try Task.checkCancellation()
            let baseline = computeFP32Baseline1536(query: query, candidates: candidates)
            try checkpoint(.baseline)
            for strategy in MixedPrecisionStrategy.allCases {
                if let metric = try benchmark(strategy, dimension: dimension, count: candidateCount,
                    baseline: baseline, iterations: iterations, warmups: warmupIterations, execute: { results in
                        try executeStrategy1536(strategy, query: query, candidates: candidates, results: &results)
                    }) { staged.append(metric) }
            }
        default: throw PrecisionTuningSupport.invalid("dimension", "unsupported dimension")
        }
        guard !staged.isEmpty else { throw PrecisionTuningSupport.failed("calibration produced no valid synthetic profile") }
        try checkpoint(.willPublish)
        profiles[Family(dimension: dimension, candidateCount: candidateCount)] = staged
        // A test observer may request cancellation here; publication already committed.
        try hooks?.observer?(.didPublish)
        return staged
    }

    private func benchmark(_ strategy: MixedPrecisionStrategy, dimension: Int, count: Int,
                           baseline: [Float], iterations: Int, warmups: Int,
                           execute: (inout [Float]) throws -> Void) throws -> StrategyMetrics? {
        var results = Array(repeating: Float(0), count: count)
        for _ in 0..<warmups {
            try checkpoint(.willExecute(strategy.rawValue, true))
            try execute(&results)
            try checkpoint(.didExecute(strategy.rawValue, true))
        }
        var times: [Double] = []
        times.reserveCapacity(iterations)
        let clock = ContinuousClock()
        for iteration in 0..<iterations {
            try checkpoint(.willExecute(strategy.rawValue, false))
            let start = clock.now
            try execute(&results)
            let elapsed = PrecisionTuningSupport.seconds(start.duration(to: clock.now))
            try checkpoint(.didExecute(strategy.rawValue, false))
            times.append(hooks?.latency?(strategy.rawValue, iteration, elapsed) ?? elapsed)
        }
        try checkpoint(.accuracy)
        guard times.allSatisfy({ $0.isFinite && $0 > 0 }),
              let error = PrecisionTuningSupport.errors(results, reference: baseline) else { return nil }
        let mean = times.reduce(0, +) / Double(iterations)
        let metric = StrategyMetrics(strategy: strategy, meanLatency: mean,
            medianLatency: PrecisionTuningSupport.percentile(times.sorted(), 0.5), throughput: 1 / mean,
            maxRelativeError: error.maximum, meanRelativeError: error.mean, candidateCount: count, dimension: dimension,
            measuredSamples: iterations, warmupIterations: warmups, algorithmVersion: 1,
            datasetVersion: "gaussian-lcg-boxmuller-v1", seed: 42,
            conversionBoundary: "fp32-input-conversion-per-reduced-euclidean-invocation-v1", timestamp: Date())
        return metric.isValid && metric.qualityScore.isFinite ? metric : nil
    }

    /// Execute a specific strategy
    private func executeStrategy(
        _ strategy: MixedPrecisionStrategy,
        query: Vector512Optimized,
        candidates: [Vector512Optimized],
        results: inout [Float]
    ) throws {
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
            try checkpoint(.convertedFP16(1))
            let candidatesSoA = MixedPrecisionKernels.createSoA512FP16(from: candidates)
            try checkpoint(.convertedFP16(candidates.count))
            results.withUnsafeMutableBufferPointer { buffer in
                MixedPrecisionKernels.batchEuclidean512(
                    query: queryFP16,
                    candidates: candidatesSoA,
                    results: buffer
                )
            }

        case .queryFP16Blocked:
            let queryFP16 = MixedPrecisionKernels.Vector512FP16(from: query)
            try checkpoint(.convertedFP16(1))
            let candidatesSoA = MixedPrecisionKernels.createSoA512FP16(from: candidates)
            try checkpoint(.convertedFP16(candidates.count))
            results.withUnsafeMutableBufferPointer { buffer in
                MixedPrecisionKernels.batchEuclideanBlocked512(
                    query: queryFP16,
                    candidates: candidatesSoA,
                    results: buffer
                )
            }

        case .queryFP32Standard:
            let candidatesSoA = MixedPrecisionKernels.createSoA512FP16(from: candidates)
            try checkpoint(.convertedFP16(candidates.count))
            results.withUnsafeMutableBufferPointer { buffer in
                MixedPrecisionKernels.batchEuclidean512(
                    query: query,
                    candidates: candidatesSoA,
                    results: buffer
                )
            }

        case .queryFP32Blocked:
            let candidatesSoA = MixedPrecisionKernels.createSoA512FP16(from: candidates)
            try checkpoint(.convertedFP16(candidates.count))
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
    ) throws {
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
            try checkpoint(.convertedFP16(1))
            let candidatesSoA = MixedPrecisionKernels.createSoA768FP16(from: candidates)
            try checkpoint(.convertedFP16(candidates.count))
            results.withUnsafeMutableBufferPointer { buffer in
                MixedPrecisionKernels.batchEuclidean768(
                    query: queryFP16,
                    candidates: candidatesSoA,
                    results: buffer
                )
            }

        case .queryFP16Blocked:
            let queryFP16 = MixedPrecisionKernels.Vector768FP16(from: query)
            try checkpoint(.convertedFP16(1))
            let candidatesSoA = MixedPrecisionKernels.createSoA768FP16(from: candidates)
            try checkpoint(.convertedFP16(candidates.count))
            results.withUnsafeMutableBufferPointer { buffer in
                MixedPrecisionKernels.batchEuclideanBlocked768(
                    query: queryFP16,
                    candidates: candidatesSoA,
                    results: buffer
                )
            }

        case .queryFP32Standard:
            let candidatesSoA = MixedPrecisionKernels.createSoA768FP16(from: candidates)
            try checkpoint(.convertedFP16(candidates.count))
            results.withUnsafeMutableBufferPointer { buffer in
                MixedPrecisionKernels.batchEuclidean768(
                    query: query,
                    candidates: candidatesSoA,
                    results: buffer
                )
            }

        case .queryFP32Blocked:
            let candidatesSoA = MixedPrecisionKernels.createSoA768FP16(from: candidates)
            try checkpoint(.convertedFP16(candidates.count))
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
    ) throws {
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
            try checkpoint(.convertedFP16(1))
            let candidatesSoA = MixedPrecisionKernels.createSoA1536FP16(from: candidates)
            try checkpoint(.convertedFP16(candidates.count))
            results.withUnsafeMutableBufferPointer { buffer in
                MixedPrecisionKernels.batchEuclidean1536(
                    query: queryFP16,
                    candidates: candidatesSoA,
                    results: buffer
                )
            }

        case .queryFP16Blocked:
            let queryFP16 = MixedPrecisionKernels.Vector1536FP16(from: query)
            try checkpoint(.convertedFP16(1))
            let candidatesSoA = MixedPrecisionKernels.createSoA1536FP16(from: candidates)
            try checkpoint(.convertedFP16(candidates.count))
            results.withUnsafeMutableBufferPointer { buffer in
                MixedPrecisionKernels.batchEuclideanBlocked1536(
                    query: queryFP16,
                    candidates: candidatesSoA,
                    results: buffer
                )
            }

        case .queryFP32Standard:
            let candidatesSoA = MixedPrecisionKernels.createSoA1536FP16(from: candidates)
            try checkpoint(.convertedFP16(candidates.count))
            results.withUnsafeMutableBufferPointer { buffer in
                MixedPrecisionKernels.batchEuclidean1536(
                    query: query,
                    candidates: candidatesSoA,
                    results: buffer
                )
            }

        case .queryFP32Blocked:
            let candidatesSoA = MixedPrecisionKernels.createSoA1536FP16(from: candidates)
            try checkpoint(.convertedFP16(candidates.count))
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

    private func generateTestVectors512(count: Int, seed: UInt32) throws -> [Vector512Optimized] {
        var rng = SeededRNG(seed: seed)
        var vectors: [Vector512Optimized] = []
        vectors.reserveCapacity(count)
        for _ in 0..<count {
            try Task.checkCancellation()
            var values = Array(repeating: Float(0), count: 512)
            for i in values.indices { values[i] = rng.nextGaussian() }
            vectors.append(try Vector512Optimized(values))
            try checkpoint(.generatedVector)
        }
        return vectors
    }
    private func generateTestVectors768(count: Int, seed: UInt32) throws -> [Vector768Optimized] {
        var rng = SeededRNG(seed: seed)
        var vectors: [Vector768Optimized] = []
        vectors.reserveCapacity(count)
        for _ in 0..<count {
            try Task.checkCancellation()
            var values = Array(repeating: Float(0), count: 768)
            for i in values.indices { values[i] = rng.nextGaussian() }
            vectors.append(try Vector768Optimized(values))
            try checkpoint(.generatedVector)
        }
        return vectors
    }
    private func generateTestVectors1536(count: Int, seed: UInt32) throws -> [Vector1536Optimized] {
        var rng = SeededRNG(seed: seed)
        var vectors: [Vector1536Optimized] = []
        vectors.reserveCapacity(count)
        for _ in 0..<count {
            try Task.checkCancellation()
            var values = Array(repeating: Float(0), count: 1536)
            for i in values.indices { values[i] = rng.nextGaussian() }
            vectors.append(try Vector1536Optimized(values))
            try checkpoint(.generatedVector)
        }
        return vectors
    }
}

/// Deterministic synthetic values, without a deterministic timing guarantee.
struct SeededRNG {
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
