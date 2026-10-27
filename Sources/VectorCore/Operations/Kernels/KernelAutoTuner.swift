import Foundation
import simd
#if canImport(Darwin)
import Darwin
#endif

public enum KernelStrategy: String, Codable, Sendable, CaseIterable {
    case cpuAoS, cpuSoA, cpuMixedPrecision, cpuQuantized, auto
}

/// Resource fields filter a scalar payload estimate and a synthetic DOT mean;
/// neither is a live-memory cap or execution deadline.
public struct WorkloadCharacteristics: Sendable {
    public let dimension: Int
    public let batchSize: Int
    public let precisionMode: PrecisionMode
    public let maximumEstimatedPayloadBytes: Int?
    public let maximumProfiledMeanLatency: TimeInterval?
    public enum PrecisionMode: String, Codable, Sendable {
        case fp32Full, fp16Acceptable, int8Acceptable
    }
    public init(dimension: Int, batchSize: Int, precisionMode: PrecisionMode = .fp16Acceptable,
                maximumEstimatedPayloadBytes: Int? = nil, maximumProfiledMeanLatency: TimeInterval? = nil) {
        self.dimension = dimension; self.batchSize = batchSize; self.precisionMode = precisionMode
        self.maximumEstimatedPayloadBytes = maximumEstimatedPayloadBytes
        self.maximumProfiledMeanLatency = maximumProfiledMeanLatency
    }
}

/// Exact measurement family. Decision filters are re-applied at every lookup.
public struct KernelProfileWorkload: Codable, Sendable, Hashable {
    public let dimension: Int
    public let batchSize: Int
    public let precisionMode: WorkloadCharacteristics.PrecisionMode
    public init(dimension: Int, batchSize: Int, precisionMode: WorkloadCharacteristics.PrecisionMode) {
        self.dimension = dimension; self.batchSize = batchSize; self.precisionMode = precisionMode
    }
}

public struct KernelProfileProvenance: Codable, Sendable, Equatable {
    public let algorithmVersion: Int
    public let operationBoundary: String
    public let datasetVersion: String
    public let datasetSeeds: [UInt32]
    public let architecture: String
    public let hardwareModel: String?
    public let processorCount: Int
    public let operatingSystem: String

    internal static func current() -> Self {
        #if arch(arm64)
        let architecture = "arm64"
        #elseif arch(x86_64)
        let architecture = "x86_64"
        #else
        let architecture = "other"
        #endif
        var model: String?
        #if canImport(Darwin)
        var length = 0
        if sysctlbyname("hw.model", nil, &length, nil, 0) == 0 && length > 1 {
            var bytes = Array(repeating: CChar(0), count: length)
            if sysctlbyname("hw.model", &bytes, &length, nil, 0) == 0 {
                let value = String(decoding: bytes.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
                if !value.isEmpty { model = value }
            }
        }
        #endif
        return Self(algorithmVersion: 1,
            operationBoundary: "synthetic-dot-preconverted-reduced-soa-convenience-packing-v1",
            datasetVersion: "uniform-lcg-v1", datasetSeeds: [0x5EED_0512, 0x5EED_0768, 0x5EED_1536],
            architecture: architecture, hardwareModel: model, processorCount: ProcessInfo.processInfo.processorCount,
            operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString)
    }
}

/// DOT benchmark metrics. Latencies are seconds per invocation; throughput is
/// candidates per second. The synthetic mean-error score does not bound real data.
public struct PerformanceProfile: Codable, Sendable {
    public let strategy: KernelStrategy
    public let workload: KernelProfileWorkload
    public let throughput: Double
    public let latencyMean: TimeInterval
    public let latencyP50: TimeInterval
    public let latencyP95: TimeInterval
    public let latencyP99: TimeInterval
    /// Scalar query/candidate payload only; excludes headers, output, padding,
    /// quantization parameters and simultaneously live calibration representations.
    public let estimatedPayloadBytes: Int
    public let syntheticAccuracyScore: Double
    public let measuredSamples: Int
    public let warmupIterations: Int
    public let provenance: KernelProfileProvenance
    public let timestamp: Date

    internal var hasValidMetrics: Bool {
        strategy != .auto && throughput.isFinite && throughput > 0 && latencyMean.isFinite && latencyMean > 0 &&
        latencyP50.isFinite && latencyP50 > 0 && latencyP95.isFinite && latencyP95 >= latencyP50 &&
        latencyP99.isFinite && latencyP99 >= latencyP95 && estimatedPayloadBytes > 0 &&
        syntheticAccuracyScore.isFinite && (0...1).contains(syntheticAccuracyScore) &&
        measuredSamples > 0 && warmupIterations >= 0 && timestamp.timeIntervalSince1970.isFinite
    }
    public var score: Double {
        guard hasValidMetrics else { return -.infinity }
        let score = log10(min(throughput, 1e9)) - log10(max(latencyMean * 1000, 1e-3)) -
            log10(max(Double(estimatedPayloadBytes) / 1e6, 0.001)) + syntheticAccuracyScore * 10
        return score.isFinite ? score : -.infinity
    }
}

/// Explicit synthetic profiling and representation-compatible selection.
/// Default/shared instances are memory-only. A URL opts into a single-owner file;
/// multiple instances writing that URL are not a coordinated multi-writer store.
/// PrecisionSelectionContractTests exercises filters and publication semantics.
public actor KernelAutoTuner {
    public static let shared = KernelAutoTuner()
    public enum PersistenceStatus: Sendable { case memoryOnly, enabled, disabledUnavailableHardware }
    public let persistenceStatus: PersistenceStatus
    private struct Request: Hashable {
        let family: KernelProfileWorkload
        let payload: Int?
        let latencyBits: UInt64?
    }
    private struct Envelope: Codable {
        let schemaVersion: Int
        let provenance: KernelProfileProvenance
        let profiles: [PerformanceProfile]
    }
    private var profileCache: [KernelProfileWorkload: [PerformanceProfile]] = [:]
    private var strategyOverrides: [Request: KernelStrategy] = [:]
    private var profilesLoaded = false
    private let persistenceURL: URL?
    private let provenance: KernelProfileProvenance
    private let hooks: TuningTestHooks?
    internal private(set) var workCounts: TuningWorkCounts?

    /// The supplied file's parent must exist. Without a stable hardware model,
    /// disk read/write reuse is disabled and persistenceStatus says so explicitly.
    public init(persistenceURL: URL? = nil) {
        let current = KernelProfileProvenance.current()
        provenance = current; hooks = nil; workCounts = nil
        self.persistenceURL = current.hardwareModel == nil ? nil : persistenceURL
        persistenceStatus = persistenceURL == nil ? .memoryOnly : (current.hardwareModel == nil ? .disabledUnavailableHardware : .enabled)
    }
    internal init(persistenceURL: URL? = nil, provenance: KernelProfileProvenance,
                  testing hooks: TuningTestHooks = TuningTestHooks()) {
        self.provenance = provenance; self.hooks = hooks; workCounts = TuningWorkCounts()
        self.persistenceURL = provenance.hardwareModel == nil ? nil : persistenceURL
        persistenceStatus = persistenceURL == nil ? .memoryOnly : (provenance.hardwareModel == nil ? .disabledUnavailableHardware : .enabled)
    }
    private func checkpoint(_ event: TuningCheckpoint) throws {
        workCounts?.record(event)
        try hooks?.observer?(event)
        try Task.checkCancellation()
    }
    private func family(_ workload: WorkloadCharacteristics) -> KernelProfileWorkload {
        KernelProfileWorkload(dimension: workload.dimension, batchSize: workload.batchSize, precisionMode: workload.precisionMode)
    }
    private func key(_ workload: WorkloadCharacteristics) -> Request {
        Request(family: family(workload), payload: workload.maximumEstimatedPayloadBytes,
                latencyBits: workload.maximumProfiledMeanLatency.map { $0 == 0 ? 0 : $0.bitPattern })
    }
    private func availableStrategies(for workload: WorkloadCharacteristics) -> [KernelStrategy] {
        switch workload.precisionMode {
        case .fp32Full: return [.cpuAoS, .cpuSoA]
        case .fp16Acceptable: return [.cpuAoS, .cpuSoA, .cpuMixedPrecision]
        case .int8Acceptable: return [.cpuAoS, .cpuSoA, .cpuMixedPrecision, .cpuQuantized]
        }
    }
    private func validate(_ workload: WorkloadCharacteristics) throws {
        try PrecisionTuningSupport.validateShape(count: workload.batchSize, dimension: workload.dimension)
        if let bytes = workload.maximumEstimatedPayloadBytes, bytes < 0 {
            throw PrecisionTuningSupport.invalid("maximumEstimatedPayloadBytes", "must be nonnegative")
        }
        if let latency = workload.maximumProfiledMeanLatency, !latency.isFinite || latency <= 0 {
            throw PrecisionTuningSupport.invalid("maximumProfiledMeanLatency", "must be finite and positive")
        }
        for strategy in availableStrategies(for: workload) { _ = try estimate(strategy, workload: workload) }
    }
    private func estimate(_ strategy: KernelStrategy, workload: WorkloadCharacteristics) throws -> Int {
        let n = workload.batchSize, d = workload.dimension
        switch strategy {
        case .cpuAoS, .cpuSoA:
            return try PrecisionTuningSupport.multiply(PrecisionTuningSupport.multiply(PrecisionTuningSupport.add(n, 1, field: "batchSize"), d, field: "batchSize"), 4, field: "batchSize")
        case .cpuMixedPrecision:
            return try PrecisionTuningSupport.add(PrecisionTuningSupport.multiply(d, 4, field: "dimension"), PrecisionTuningSupport.multiply(PrecisionTuningSupport.multiply(n,
                d, field: "batchSize"), 2, field: "batchSize"), field: "batchSize")
        case .cpuQuantized:
            return try PrecisionTuningSupport.multiply(PrecisionTuningSupport.add(n, 1, field: "batchSize"), d, field: "batchSize")
        case .auto: throw PrecisionTuningSupport.invalid("strategy", "auto cannot be a profile or override")
        }
    }
    private func valid(_ profile: PerformanceProfile, for workload: WorkloadCharacteristics) -> Bool {
        guard profile.workload == family(workload), profile.provenance == provenance,
              profile.hasValidMetrics, profile.score.isFinite, workload.batchSize > 0,
              availableStrategies(for: workload).contains(profile.strategy),
              let expected = try? estimate(profile.strategy, workload: workload) else { return false }
        return expected == profile.estimatedPayloadBytes
    }
    private func feasible(_ strategy: KernelStrategy, workload: WorkloadCharacteristics) -> Bool {
        guard availableStrategies(for: workload).contains(strategy), let bytes = try? estimate(strategy, workload: workload),
              workload.maximumEstimatedPayloadBytes.map({ bytes <= $0 }) ?? true else { return false }
        guard let limit = workload.maximumProfiledMeanLatency else { return true }
        return (profileCache[family(workload)] ?? []).contains {
            $0.strategy == strategy && valid($0, for: workload) && $0.latencyMean <= limit
        }
    }
    private func noFeasible() -> VectorError {
        VectorError(kind: .resourceUnavailable, context: ErrorContext(additionalInfo: ["reason": "noFeasibleStrategy"]))
    }
    /// Applies precision and both filters to overrides, profiles, and heuristics.
    /// A latency filter requires a matching measured profile. Heuristic fallback
    /// prefers the legacy choice, then AoS, SoA, FP16, INT8 in that fixed order.
    public func selectStrategy(for workload: WorkloadCharacteristics) throws -> KernelStrategy {
        try validate(workload)
        try ensureProfilesLoaded()
        if let override = strategyOverrides[key(workload)] {
            guard availableStrategies(for: workload).contains(override) else {
                throw PrecisionTuningSupport.invalid("override", "incompatible precision representation")
            }
            guard feasible(override, workload: workload) else { throw noFeasible() }
            return override
        }
        let profiles = (profileCache[family(workload)] ?? []).filter { profile in
            valid(profile, for: workload) && feasible(profile.strategy, workload: workload) &&
            (workload.maximumProfiledMeanLatency.map { limit in profile.latencyMean <= limit } ?? true)
        }
        if let best = sorted(profiles).first { return best.strategy }
        guard workload.maximumProfiledMeanLatency == nil else { throw noFeasible() }
        let preferred: KernelStrategy
        if workload.batchSize == 0 { preferred = .cpuAoS }
        else if workload.precisionMode == .fp32Full { preferred = workload.batchSize < 100 ? .cpuAoS : .cpuSoA }
        else if workload.precisionMode == .fp16Acceptable || workload.batchSize < 100 { preferred = .cpuMixedPrecision }
        else { preferred = .cpuQuantized }
        for strategy in [preferred] + availableStrategies(for: workload) {
            if feasible(strategy, workload: workload) { return strategy }
        }
        throw noFeasible()
    }
    /// Overrides are process-local. Rejected setters preserve all prior state.
    public func setOverride(_ strategy: KernelStrategy, for workload: WorkloadCharacteristics) throws {
        try validate(workload)
        guard availableStrategies(for: workload).contains(strategy) else {
            throw PrecisionTuningSupport.invalid("strategy", "incompatible precision representation or auto override")
        }
        try ensureProfilesLoaded()
        guard feasible(strategy, workload: workload) else { throw noFeasible() }
        strategyOverrides[key(workload)] = strategy
    }
    public func removeOverride(for workload: WorkloadCharacteristics) { strategyOverrides.removeValue(forKey: key(workload)) }
    private func sorted(_ profiles: [PerformanceProfile]) -> [PerformanceProfile] {
        profiles.sorted {
            if $0.score != $1.score { return $0.score > $1.score }
            return KernelStrategy.allCases.firstIndex(of: $0.strategy)! < KernelStrategy.allCases.firstIndex(of: $1.strategy)!
        }
    }
    internal func setProfilesForTesting(_ profiles: [PerformanceProfile], for workload: WorkloadCharacteristics) {
        profileCache[family(workload)] = profiles; profilesLoaded = true
    }
    internal func profilesForTesting(for workload: WorkloadCharacteristics) -> [PerformanceProfile] { profileCache[family(workload)] ?? [] }

    /// Explicit DOT calibration. Reduced candidates are prepared outside timing;
    /// the SoA convenience kernel still packs per invocation. One FP32 AoS baseline
    /// is computed before timing. Synchronous kernels are not interruptible.
    /// Pre-commit cancellation/failure preserves profiles and overrides. Successful
    /// profiles publish to memory before saving; a save error retains that result.
    public func profileWorkload(_ workload: WorkloadCharacteristics, iterations: Int = 100,
                                warmupIterations: Int = 10) async throws -> [PerformanceProfile] {
        try validate(workload)
        let strategies = availableStrategies(for: workload)
        try PrecisionTuningSupport.validateCalibration(count: workload.batchSize, dimension: workload.dimension,
            iterations: iterations, warmups: warmupIterations, strategies: strategies.count)
        try Task.checkCancellation()
        try ensureProfilesLoaded()
        let data = try generateTestData(for: workload)
        try Task.checkCancellation()
        let baseline = try executeKernel(.cpuAoS, testData: data)
        try checkpoint(.baseline)
        var staged: [PerformanceProfile] = []
        for strategy in strategies {
            var result: [Float] = []
            for _ in 0..<warmupIterations {
                try checkpoint(.willExecute(strategy.rawValue, true))
                result = try executeKernel(strategy, testData: data)
                try checkpoint(.didExecute(strategy.rawValue, true))
            }
            var times: [Double] = []
            times.reserveCapacity(iterations)
            let clock = ContinuousClock()
            for iteration in 0..<iterations {
                try checkpoint(.willExecute(strategy.rawValue, false))
                let start = clock.now
                result = try executeKernel(strategy, testData: data)
                let elapsed = PrecisionTuningSupport.seconds(start.duration(to: clock.now))
                try checkpoint(.didExecute(strategy.rawValue, false))
                times.append(hooks?.latency?(strategy.rawValue, iteration, elapsed) ?? elapsed)
            }
            try checkpoint(.accuracy)
            guard times.allSatisfy({ $0.isFinite && $0 > 0 }),
                  let error = PrecisionTuningSupport.errors(result, reference: baseline) else { continue }
            let mean = times.reduce(0, +) / Double(iterations), samples = times.sorted()
            let profile = PerformanceProfile(strategy: strategy, workload: family(workload),
                throughput: Double(workload.batchSize) / mean, latencyMean: mean,
                latencyP50: PrecisionTuningSupport.percentile(samples, 0.5),
                latencyP95: PrecisionTuningSupport.percentile(samples, 0.95),
                latencyP99: PrecisionTuningSupport.percentile(samples, 0.99),
                estimatedPayloadBytes: try estimate(strategy, workload: workload),
                syntheticAccuracyScore: 1 - min(error.mean, 1), measuredSamples: iterations,
                warmupIterations: warmupIterations, provenance: provenance, timestamp: Date())
            if valid(profile, for: workload) { staged.append(profile) }
        }
        guard !staged.isEmpty else { throw PrecisionTuningSupport.failed("calibration produced no valid profile") }
        try checkpoint(.willPublish)
        profileCache[family(workload)] = staged
        try hooks?.observer?(.didPublish)
        try saveProfiles()
        return sorted(staged)
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
            // One dot product per candidate (an array map)
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
            // FP16 storage with FP32 computation (half the FP32 scalar payload; payload model only, not a measured memory saving)
            return data.candidatesFP16.map { candidate in
                MixedPrecisionKernels.dotMixed512(
                    query: data.query512,
                    candidate: candidate
                )
            }

        case .cpuQuantized:
            guard let queryINT8 = data.queryINT8 else {
                throw PrecisionTuningSupport.failed("missing INT8 calibration representation")
            }
            // INT8 quantized storage (a quarter of the FP32 scalar payload; payload model only, not a measured memory saving)
            return data.candidatesINT8.map { candidate in
                QuantizedKernels.dotProduct512(
                    query: queryINT8,
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
            guard let queryINT8 = data.queryINT8 else {
                throw PrecisionTuningSupport.failed("missing INT8 calibration representation")
            }
            return data.candidatesINT8.map { candidate in
                QuantizedKernels.dotProduct768(
                    query: queryINT8,
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
            guard let queryINT8 = data.queryINT8 else {
                throw PrecisionTuningSupport.failed("missing INT8 calibration representation")
            }
            return data.candidatesINT8.map { candidate in
                QuantizedKernels.dotProduct1536(
                    query: queryINT8,
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

    private enum TestData: Sendable {
        case dim512(TestData512), dim768(TestData768), dim1536(TestData1536)
    }
    private struct TestData512: Sendable {
        let query512: Vector512Optimized
        let queryINT8: Vector512INT8?
        let candidates: [Vector512Optimized]
        let candidatesFP16: [MixedPrecisionKernels.Vector512FP16]
        let candidatesINT8: [Vector512INT8]
    }
    private struct TestData768: Sendable {
        let query768: Vector768Optimized
        let queryINT8: Vector768INT8?
        let candidates: [Vector768Optimized]
        let candidatesFP16: [MixedPrecisionKernels.Vector768FP16]
        let candidatesINT8: [Vector768INT8]
    }
    private struct TestData1536: Sendable {
        let query1536: Vector1536Optimized
        let queryINT8: Vector1536INT8?
        let candidates: [Vector1536Optimized]
        let candidatesFP16: [MixedPrecisionKernels.Vector1536FP16]
        let candidatesINT8: [Vector1536INT8]
    }
    private func generateTestData(for workload: WorkloadCharacteristics) throws -> TestData {
        switch workload.dimension {
        case 512:
            var rng = SeededRNG(seed: 0x5EED_0512)
            try Task.checkCancellation()
            let query = Vector512Optimized(generator: { _ in rng.next() * 2 - 1 })
            try checkpoint(.generatedVector)
            var candidates: [Vector512Optimized] = []
            for _ in 0..<workload.batchSize {
                try Task.checkCancellation()
                candidates.append(Vector512Optimized(generator: { _ in rng.next() * 2 - 1 }))
                try checkpoint(.generatedVector)
            }
            var fp16: [MixedPrecisionKernels.Vector512FP16] = []
            if workload.precisionMode != .fp32Full {
                for candidate in candidates {
                    try Task.checkCancellation()
                    fp16.append(MixedPrecisionKernels.Vector512FP16(from: candidate))
                    try checkpoint(.convertedFP16(1))
                }
            }
            var int8: [Vector512INT8] = []
            var queryINT8: Vector512INT8?
            if workload.precisionMode == .int8Acceptable {
                try Task.checkCancellation()
                queryINT8 = Vector512INT8(from: query, params: nil)
                try checkpoint(.convertedINT8(1))
                for candidate in candidates {
                    try Task.checkCancellation()
                    int8.append(Vector512INT8(from: candidate, params: nil))
                    try checkpoint(.convertedINT8(1))
                }
            }
            return .dim512(TestData512(query512: query, queryINT8: queryINT8,
                candidates: candidates, candidatesFP16: fp16, candidatesINT8: int8))
        case 768:
            var rng = SeededRNG(seed: 0x5EED_0768)
            try Task.checkCancellation()
            let query = Vector768Optimized(generator: { _ in rng.next() * 2 - 1 })
            try checkpoint(.generatedVector)
            var candidates: [Vector768Optimized] = []
            for _ in 0..<workload.batchSize {
                try Task.checkCancellation()
                candidates.append(Vector768Optimized(generator: { _ in rng.next() * 2 - 1 }))
                try checkpoint(.generatedVector)
            }
            var fp16: [MixedPrecisionKernels.Vector768FP16] = []
            if workload.precisionMode != .fp32Full {
                for candidate in candidates {
                    try Task.checkCancellation()
                    fp16.append(MixedPrecisionKernels.Vector768FP16(from: candidate))
                    try checkpoint(.convertedFP16(1))
                }
            }
            var int8: [Vector768INT8] = []
            var queryINT8: Vector768INT8?
            if workload.precisionMode == .int8Acceptable {
                try Task.checkCancellation()
                queryINT8 = Vector768INT8(from: query, params: nil)
                try checkpoint(.convertedINT8(1))
                for candidate in candidates {
                    try Task.checkCancellation()
                    int8.append(Vector768INT8(from: candidate, params: nil))
                    try checkpoint(.convertedINT8(1))
                }
            }
            return .dim768(TestData768(query768: query, queryINT8: queryINT8,
                candidates: candidates, candidatesFP16: fp16, candidatesINT8: int8))
        case 1536:
            var rng = SeededRNG(seed: 0x5EED_1536)
            try Task.checkCancellation()
            let query = Vector1536Optimized(generator: { _ in rng.next() * 2 - 1 })
            try checkpoint(.generatedVector)
            var candidates: [Vector1536Optimized] = []
            for _ in 0..<workload.batchSize {
                try Task.checkCancellation()
                candidates.append(Vector1536Optimized(generator: { _ in rng.next() * 2 - 1 }))
                try checkpoint(.generatedVector)
            }
            var fp16: [MixedPrecisionKernels.Vector1536FP16] = []
            if workload.precisionMode != .fp32Full {
                for candidate in candidates {
                    try Task.checkCancellation()
                    fp16.append(MixedPrecisionKernels.Vector1536FP16(from: candidate))
                    try checkpoint(.convertedFP16(1))
                }
            }
            var int8: [Vector1536INT8] = []
            var queryINT8: Vector1536INT8?
            if workload.precisionMode == .int8Acceptable {
                try Task.checkCancellation()
                queryINT8 = Vector1536INT8(from: query, params: nil)
                try checkpoint(.convertedINT8(1))
                for candidate in candidates {
                    try Task.checkCancellation()
                    int8.append(Vector1536INT8(from: candidate, params: nil))
                    try checkpoint(.convertedINT8(1))
                }
            }
            return .dim1536(TestData1536(query1536: query, queryINT8: queryINT8,
                candidates: candidates, candidatesFP16: fp16, candidatesINT8: int8))
        default: throw PrecisionTuningSupport.invalid("dimension", "unsupported dimension")
        }
    }

    private func ensureProfilesLoaded() throws {
        guard !profilesLoaded else { return }
        guard let url = persistenceURL else { profilesLoaded = true; return }
        let data: Data
        do { data = try Data(contentsOf: url) }
        catch let error as NSError where error.domain == NSCocoaErrorDomain && error.code == NSFileReadNoSuchFileError {
            profilesLoaded = true; return
        } catch { throw PrecisionTuningSupport.failed("profile read failed: \(error)") }
        guard let envelope = try? JSONDecoder().decode(Envelope.self, from: data),
              envelope.schemaVersion == 1, envelope.provenance == provenance else {
            profilesLoaded = true; return
        }
        var loaded: [KernelProfileWorkload: [PerformanceProfile]] = [:]
        for profile in envelope.profiles {
            let w = WorkloadCharacteristics(dimension: profile.workload.dimension, batchSize: profile.workload.batchSize,
                precisionMode: profile.workload.precisionMode)
            guard (try? validate(w)) != nil, valid(profile, for: w) else { continue }
            loaded[profile.workload, default: []].append(profile)
        }
        profileCache = loaded
        profilesLoaded = true
    }
    private func saveProfiles() throws {
        guard let url = persistenceURL else { return }
        do {
            let values = profileCache.values.flatMap { $0 }
            let data = try JSONEncoder().encode(Envelope(schemaVersion: 1, provenance: provenance, profiles: values))
            try data.write(to: url, options: .atomic)
        } catch {
            throw PrecisionTuningSupport.failed("in-memory calibration published; profile persistence failed: \(error)")
        }
    }
    /// A normal reset is memory-only and prevents reloading this instance. With
    /// true, remove only the configured file first; deletion failure preserves all
    /// in-memory state. Directories and unrelated files are never removed.
    public func clearCache(removePersistedProfiles: Bool = false) throws {
        if removePersistedProfiles, let url = persistenceURL {
            do {
                let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
                guard attributes[.type] as? FileAttributeType != .typeDirectory else {
                    throw PrecisionTuningSupport.failed("profile deletion refused: configured URL is a directory")
                }
                try FileManager.default.removeItem(at: url)
            } catch let error as NSError where error.domain == NSCocoaErrorDomain && error.code == NSFileReadNoSuchFileError {
                // A missing configured file needs no deletion.
            } catch {
                throw PrecisionTuningSupport.failed("profile deletion failed; memory retained: \(error)")
            }
        }
        profileCache.removeAll(); strategyOverrides.removeAll(); profilesLoaded = true
    }

    /// Validates both inputs, rejects resource filters (DOT profiles cannot certify
    /// this Euclidean path), then honors full mode without consulting either tuner.
    /// Reduced modes delegate lookup only; no calibration occurs here. The one default
    /// limit, 0.001, applies to every precision mode (0.3.3 used a per-mode limit, ten
    /// times looser for `.int8Acceptable`); pass the limit a workload tolerates.
    public func selectMixedPrecisionStrategy(for workload: WorkloadCharacteristics,
        maximumSyntheticRelativeError: Float = 0.001) async throws -> MixedPrecisionStrategy {
        try validate(workload)
        try PrecisionTuningSupport.validateLimit(maximumSyntheticRelativeError, field: "maximumSyntheticRelativeError")
        guard workload.maximumEstimatedPayloadBytes == nil && workload.maximumProfiledMeanLatency == nil else {
            throw VectorError(kind: .unsupportedOperation, context: ErrorContext(additionalInfo: ["reason": "DOT resource profiles cannot certify mixed Euclidean selection"]))
        }
        guard workload.precisionMode != .fp32Full else { return .fullFP32 }
        return try await MixedPrecisionAutoTuner.shared.selectStrategy(candidateCount: workload.batchSize,
            maximumSyntheticRelativeError: maximumSyntheticRelativeError, dimension: workload.dimension)
    }
}
