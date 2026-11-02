import Foundation

// MARK: - Base Models (no dependencies on other benchmark models)

/// Binary size information for benchmark metadata: file sizes observed at run time, not a binary identity.
public struct BinarySizes: Codable, Sendable {
    /// Size of the file at `CommandLine.arguments[0]`.
    public let benchExecutableBytes: UInt64
    /// Size of a loaded VectorCore dynamic library or framework binary, matched by file name
    /// (`EnvCapture.isVectorCoreLibraryImage`, `testVectorCoreLibraryImageMatchesFileNameOnly`); nil when no such image is
    /// loaded, which is the case when VectorCore is linked statically.
    public let vectorCoreLibraryBytes: UInt64?

    public init(benchExecutableBytes: UInt64, vectorCoreLibraryBytes: UInt64?) {
        self.benchExecutableBytes = benchExecutableBytes
        self.vectorCoreLibraryBytes = vectorCoreLibraryBytes
    }
}

/// Feature flags controlling benchmark behavior
public struct RunFlags: Codable, Sendable {
    public let preferSoA: Bool
    public let useMixedPrecision: Bool
    public let abCompare: Bool
    public let abOnly: Bool
    public let useCKernels: Bool
    public let requestedCKernels: Bool?

    public init(preferSoA: Bool, useMixedPrecision: Bool, abCompare: Bool, abOnly: Bool,
                useCKernels: Bool, requestedCKernels: Bool? = nil) {
        self.preferSoA = preferSoA
        self.useMixedPrecision = useMixedPrecision
        self.abCompare = abCompare
        self.abOnly = abOnly
        self.useCKernels = useCKernels
        self.requestedCKernels = requestedCKernels
    }
}

/// Correctness error thresholds for gating benchmark results
public struct RunThresholds: Codable, Sendable {
    public let maxRelError: Double?
    public let maxAbsError: Double?

    public init(maxRelError: Double?, maxAbsError: Double?) {
        self.maxRelError = maxRelError
        self.maxAbsError = maxAbsError
    }
}

/// Filter configuration for selective benchmark execution
public struct RunFilters: Codable, Sendable {
    public let mode: String // glob|regex
    public let include: [String]?
    public let exclude: [String]?

    public init(mode: String, include: [String]?, exclude: [String]?) {
        self.mode = mode
        self.include = include
        self.exclude = exclude
    }
}

/// Parsed parameters extracted from benchmark case names
public struct CaseParams: Codable, Sendable {
    public var kind: String? = nil         // dot, dist, normalize, batch
    public var metric: String? = nil       // euclidean, cosine, manhattan, dot; or mem op kind
    public var dim: Int? = nil
    public var variant: String? = nil      // generic|optimized
    public var provider: String? = nil     // sequential|parallel|automatic
    public var n: Int? = nil               // candidate count for batch
    public var status: String? = nil       // success|zeroFail for normalize

    public init(kind: String? = nil, metric: String? = nil, dim: Int? = nil,
                variant: String? = nil, provider: String? = nil, n: Int? = nil, status: String? = nil) {
        self.kind = kind
        self.metric = metric
        self.dim = dim
        self.variant = variant
        self.provider = provider
        self.n = n
        self.status = status
    }
}

/// Correctness validation statistics comparing against reference implementation
public struct CorrectnessStats: Sendable {
    public enum Status: String, Codable, Sendable { case valid, invalid }
    public let status: Status
    public let reason: String?
    public let samples: Int
    public let maxAbsError: Double
    public let meanAbsError: Double
    public let maxRelError: Double
    public let meanRelError: Double

    public init(samples: Int, maxAbsError: Double, meanAbsError: Double,
                maxRelError: Double, meanRelError: Double) {
        let valid = samples > 0 && [maxAbsError, meanAbsError, maxRelError, meanRelError].allSatisfy { $0.isFinite && $0 >= 0 }
        self.status = valid ? .valid : .invalid
        self.reason = valid ? nil : "Unavailable or nonfinite comparison statistics"
        self.samples = valid ? samples : 0
        self.maxAbsError = maxAbsError
        self.meanAbsError = meanAbsError
        self.maxRelError = maxRelError
        self.meanRelError = meanRelError
    }

    public init(invalidReason: String) {
        status = .invalid
        reason = invalidReason
        samples = 0
        maxAbsError = .nan
        meanAbsError = .nan
        maxRelError = .nan
        meanRelError = .nan
    }
}

public struct CorrectnessOut: Codable, Sendable {
    public let status: CorrectnessStats.Status
    public let reason: String?
    public let samples: Int
    public let maxAbsError: Double?
    public let meanAbsError: Double?
    public let maxRelError: Double?
    public let meanRelError: Double?

    public init(from stats: CorrectnessStats) {
        status = stats.status
        reason = stats.reason
        samples = stats.samples
        maxAbsError = stats.status == .valid ? stats.maxAbsError : nil
        meanAbsError = stats.status == .valid ? stats.meanAbsError : nil
        maxRelError = stats.status == .valid ? stats.maxRelError : nil
        meanRelError = stats.status == .valid ? stats.meanRelError : nil
    }
}

// MARK: - Composite Models (depend on base models)

/// Execution metadata derived from case name parsing
public struct ExecutionInfo: Codable, Sendable {
    public let kind: String?
    public let metric: String?
    public let dim: Int?
    public let variant: String?
    public let provider: String?
    public let n: Int?
    // Derived feature flags from the case name
    public let isOptimized: Bool?
    public let isFused: Bool?
    public let isPreNormalized: Bool?
    public let usesSoA: Bool?
    public let usesMixedPrecision: Bool?
    public let usesSquaredDistance: Bool?

    public init(kind: String?, metric: String?, dim: Int?, variant: String?, provider: String?, n: Int?,
                isOptimized: Bool?, isFused: Bool?, isPreNormalized: Bool?,
                usesSoA: Bool?, usesMixedPrecision: Bool?, usesSquaredDistance: Bool?) {
        self.kind = kind
        self.metric = metric
        self.dim = dim
        self.variant = variant
        self.provider = provider
        self.n = n
        self.isOptimized = isOptimized
        self.isFused = isFused
        self.isPreNormalized = isPreNormalized
        self.usesSoA = usesSoA
        self.usesMixedPrecision = usesMixedPrecision
        self.usesSquaredDistance = usesSquaredDistance
    }
}

/// Complete metadata about the benchmark run environment and configuration
public struct BenchMetadata: Codable, Sendable {
    public let schemaVersion: Int
    public let fixtureVersion: String
    public let measurementBoundaryVersion: String
    public let coreConfiguration: String
    public let benchmarkConfiguration: String
    public let selectedCaseIDs: [String]
    public let batchNs: [Int]
    public let profile: String
    public let mode: String?
    public let samples: Int
    public let measurementMode: String
    public let repeats: Int?
    public let minTimeSeconds: Double
    /// `git rev-parse HEAD` run at benchmark time in the benchmark process's working directory. It gives no guarantee
    /// about the measured source: the binary may have been built from another checkout or from uncommitted changes, and
    /// a dirty checkout is not detected. `Scripts/compare_benchmarks.swift` prints it as an observation and never
    /// compares it (read from that script's `main()`; not verified by a test here).
    public let runtimeCheckout: RuntimeObservation
    /// `swiftc -version` of the `swiftc` that `/usr/bin/env` resolves through PATH at benchmark time. It gives no
    /// guarantee about the compiler that built this binary. `Scripts/compare_benchmarks.swift` lists
    /// `runtimeToolchain.value` among its required-equal metadata keys (`mismatches`; not verified by a test here), so the
    /// observation gates a comparison without identifying the build.
    public let runtimeToolchain: RuntimeObservation
    /// Named statistic of the per-operation fields (`*NsPerOperation`): one sample's average nanoseconds per
    /// declared work unit (candidate, produced entry, element or invocation); see `boundary.workUnit`.
    public let statisticUnit: String
    /// Statistic of the V02 per-invocation fields (`meanNsPerOp`/`medianNsPerOp`/`p90NsPerOp`).
    public let invocationStatisticUnit: String
    /// The auto-scaling calibration rule, cap and clock tick that produced the iteration counts, or `not used (fixed repeats)`.
    /// Construction: `unknown` is only the initializer default. Decoding: the key is required, so a schema-3 report written before
    /// this field existed fails to decode with `keyNotFound` (schema 3 is unreleased); it does not decode as `unknown`.
    public let harnessCalibration: String
    /// Selected cases whose route is unavailable. They are never timed and never counted as tested
    /// (`testUnavailableEntriesAreRecordedNeverInvoked`).
    public let unavailableCaseIDs: [String]
    /// `ProcessInfo.thermalState` observations (or explicit unavailable reasons).
    public let thermalStateStart: RuntimeObservation
    public let thermalStateEnd: RuntimeObservation
    public let lowPowerMode: RuntimeObservation

    public static let currentSchemaVersion = 3
    public static let currentBoundaryVersion = "V04-boundaries-1"
    public static let currentStatisticUnit = "per-sample-average-ns-per-operation"
    public static let currentInvocationStatisticUnit = "per-sample-average-ns-per-invocation"

    public let package: String
    public let packageVersion: String
    /// The value of `runtimeCheckout`, kept for V02 compatibility. It gives no guarantee about the measured source.
    /// Default report file names include it (`defaultRunPath` in `AppMain.swift`); `Scripts/compare_benchmarks.swift`
    /// prints it and never compares it (not verified by a test here).
    public let gitSHA: String?
    public let date: String
    public let os: String
    public let arch: String
    public let cpuCores: Int
    public let deviceModel: String?
    /// The value of `runtimeToolchain`, kept for V02 compatibility. It gives no guarantee about the compiler that built
    /// this binary; `Scripts/compare_benchmarks.swift` does not read this field (not verified by a test here).
    public let swiftVersion: String?
    public let buildConfiguration: String
    public let deviceTag: String
    /// File sizes observed at run time (see `BinarySizes`); not a binary identity.
    public let binarySizes: BinarySizes
    public let suites: [String]
    public let dims: [Int]
    public let runSeed: UInt64
    public let runLabel: String?
    public let flags: RunFlags
    public let thresholds: RunThresholds?
    public let filters: RunFilters?

    public init(package: String, packageVersion: String, gitSHA: String?, date: String,
                os: String, arch: String, cpuCores: Int, deviceModel: String?, swiftVersion: String?,
                buildConfiguration: String, deviceTag: String, binarySizes: BinarySizes,
                suites: [String], dims: [Int], runSeed: UInt64, runLabel: String?,
                flags: RunFlags, thresholds: RunThresholds?, filters: RunFilters?,
                coreConfiguration: String = "unknown", benchmarkConfiguration: String = "unknown",
                selectedCaseIDs: [String] = [], samples: Int = 1, repeats: Int? = nil, minTimeSeconds: Double = 0,
                batchNs: [Int] = [], profile: String = "custom", mode: String? = nil,
                runtimeCheckout: RuntimeObservation = .unavailable("Not observed"),
                runtimeToolchain: RuntimeObservation = .unavailable("Not observed"),
                unavailableCaseIDs: [String] = [], thermalStateStart: RuntimeObservation = .unavailable("Not observed"),
                thermalStateEnd: RuntimeObservation = .unavailable("Not observed"),
                lowPowerMode: RuntimeObservation = .unavailable("Not observed"),
                harnessCalibration: String = "unknown") {
        self.harnessCalibration = harnessCalibration
        self.schemaVersion = BenchMetadata.currentSchemaVersion
        self.fixtureVersion = "V02-inputs-1"
        self.measurementBoundaryVersion = BenchMetadata.currentBoundaryVersion
        self.statisticUnit = BenchMetadata.currentStatisticUnit
        self.invocationStatisticUnit = BenchMetadata.currentInvocationStatisticUnit
        self.unavailableCaseIDs = unavailableCaseIDs
        self.thermalStateStart = thermalStateStart; self.thermalStateEnd = thermalStateEnd; self.lowPowerMode = lowPowerMode
        self.coreConfiguration = coreConfiguration; self.benchmarkConfiguration = benchmarkConfiguration
        self.batchNs = batchNs; self.profile = profile; self.mode = mode
        self.selectedCaseIDs = selectedCaseIDs; self.samples = samples
        self.measurementMode = repeats == nil ? "estimated-duration" : "fixed-repeats"
        self.repeats = repeats; self.minTimeSeconds = minTimeSeconds
        self.runtimeCheckout = runtimeCheckout; self.runtimeToolchain = runtimeToolchain
        self.package = package
        self.packageVersion = packageVersion
        self.gitSHA = gitSHA
        self.date = date
        self.os = os
        self.arch = arch
        self.cpuCores = cpuCores
        self.deviceModel = deviceModel
        self.swiftVersion = swiftVersion
        self.buildConfiguration = buildConfiguration
        self.deviceTag = deviceTag
        self.binarySizes = binarySizes
        self.suites = suites
        self.dims = dims
        self.runSeed = runSeed
        self.runLabel = runLabel
        self.flags = flags
        self.thresholds = thresholds
        self.filters = filters
    }
}

/// Raw timing result from harness measurement
public struct BenchResult: Sendable {
    public let executionFailure: String?
    public let name: String
    public let iterations: Int
    public let totalNanoseconds: UInt64
    public let unitCount: Int // work units per iteration (e.g., candidates per batch). Default 1.
    // Stats (present when samples > 1)
    public let samples: Int
    public let meanNsPerOp: Double?
    public let medianNsPerOp: Double?
    public let p90NsPerOp: Double?
    public let stddevNsPerOp: Double?
    public let rsdPercent: Double?
    public let correctness: CorrectnessStats?
    /// Per-case measurement boundary. Attached by the executing catalog case (`CaseExecution`), not re-derived from the
    /// result name (`testRepresentativeCaseBoundaryAllFamilies`).
    public let boundary: MeasurementBoundary?
    /// Raw per-sample average ns per invocation (one element per sample).
    public let sampleNsPerInvocation: [Double]
    /// Invocations that produced each raw sample.
    public let sampleIterations: [Int]
    public let warmupIterations: Int?

    public init(name: String, iterations: Int, totalNanoseconds: UInt64, unitCount: Int = 1,
                samples: Int = 1,
                meanNsPerOp: Double? = nil, medianNsPerOp: Double? = nil, p90NsPerOp: Double? = nil,
                stddevNsPerOp: Double? = nil, rsdPercent: Double? = nil,
                correctness: CorrectnessStats? = nil, executionFailure: String? = nil,
                boundary: MeasurementBoundary? = nil, sampleNsPerInvocation: [Double] = [],
                sampleIterations: [Int] = [], warmupIterations: Int? = nil) {
        self.boundary = boundary
        self.sampleNsPerInvocation = sampleNsPerInvocation
        self.sampleIterations = sampleIterations
        self.warmupIterations = warmupIterations
        self.executionFailure = executionFailure
        self.name = name
        self.iterations = iterations
        self.totalNanoseconds = totalNanoseconds
        self.unitCount = unitCount
        self.samples = samples
        self.meanNsPerOp = meanNsPerOp
        self.medianNsPerOp = medianNsPerOp
        self.p90NsPerOp = p90NsPerOp
        self.stddevNsPerOp = stddevNsPerOp
        self.rsdPercent = rsdPercent
        self.correctness = correctness
    }
}

// MARK: - Top-Level Models (complete benchmark results)

/// Individual benchmark case with computed metrics
public struct BenchCase: Codable, Sendable {
    public let name: String
    public let params: CaseParams
    public let execution: ExecutionInfo
    public let iterations: Int
    public let totalNS: UInt64
    public let nsPerOp: Double
    public let unitCount: Int
    public let nsPerUnit: Double
    public let throughputPerSec: Double
    public let gflops: Double?
    public let suspicious: Bool
    public let samples: Int?
    public let meanNsPerOp: Double?
    public let medianNsPerOp: Double?
    public let p90NsPerOp: Double?
    public let stddevNsPerOp: Double?
    public let rsdPercent: Double?
    public let correctness: CorrectnessOut?
    /// Authoritative measurement identity (a report without it is rejected: `testReportRequiresBoundaryIdentity`).
    /// `params` and `execution` are name-derived display hints only.
    public let boundary: MeasurementBoundary?
    public let sampleNsPerInvocation: [Double]?
    public let sampleIterations: [Int]?
    public let warmupIterations: Int?
    /// Per-sample average ns per declared work unit (`sampleNsPerInvocation[i] / boundary.unitsPerInvocation`),
    /// and its median, mean, nearest-rank p90, sample standard deviation and relative standard deviation
    /// (`testPerOperationStatisticsFollowDeclaredWorkUnit`).
    public let sampleNsPerOperation: [Double]?
    public let medianNsPerOperation: Double?
    public let meanNsPerOperation: Double?
    public let p90NsPerOperation: Double?
    public let stddevNsPerOperation: Double?
    public let rsdPercentPerOperation: Double?

    public init(name: String, params: CaseParams, execution: ExecutionInfo,
                iterations: Int, totalNS: UInt64, nsPerOp: Double, unitCount: Int,
                nsPerUnit: Double, throughputPerSec: Double, gflops: Double?, suspicious: Bool,
                samples: Int?, meanNsPerOp: Double?, medianNsPerOp: Double?, p90NsPerOp: Double?,
                stddevNsPerOp: Double?, rsdPercent: Double?, correctness: CorrectnessOut?,
                boundary: MeasurementBoundary? = nil, sampleNsPerInvocation: [Double]? = nil,
                sampleIterations: [Int]? = nil, warmupIterations: Int? = nil,
                perOperation: PerOperationStatistics? = nil) {
        self.sampleNsPerOperation = perOperation?.samples
        self.medianNsPerOperation = perOperation?.median
        self.meanNsPerOperation = perOperation?.mean
        self.p90NsPerOperation = perOperation?.p90
        self.stddevNsPerOperation = perOperation?.stddev
        self.rsdPercentPerOperation = perOperation?.rsdPercent
        self.boundary = boundary
        self.sampleNsPerInvocation = sampleNsPerInvocation
        self.sampleIterations = sampleIterations
        self.warmupIterations = warmupIterations
        self.name = name
        self.params = params
        self.execution = execution
        self.iterations = iterations
        self.totalNS = totalNS
        self.nsPerOp = nsPerOp
        self.unitCount = unitCount
        self.nsPerUnit = nsPerUnit
        self.throughputPerSec = throughputPerSec
        self.gflops = gflops
        self.suspicious = suspicious
        self.samples = samples
        self.meanNsPerOp = meanNsPerOp
        self.medianNsPerOp = medianNsPerOp
        self.p90NsPerOp = p90NsPerOp
        self.stddevNsPerOp = stddevNsPerOp
        self.rsdPercent = rsdPercent
        self.correctness = correctness
    }
}

/// A/B comparison between two declared benchmark variants.
///
/// Pairs form only from catalog-declared `boundary.comparison` membership and only when every
/// boundary field outside the family's allowed differences is equal (`testFamilyAllowedDifferenceMutationTable`,
/// `testCrossScopeNeverPairsAndUndeclaredNamesDoNotPair`). The comparison is a
/// declared-purpose comparison, not an end-to-end strategy-speed claim.
public struct ABComparison: Codable, Sendable {
    public let leftPreparation: String
    public let rightPreparation: String

    public let kind: String // "batch"
    public let comparison: String // "euclidean2_vs_euclidean" | "cosine_prenorm_vs_fused"
    public let family: String
    public let purpose: String
    public let allowedDifferences: [String]
    public let leftBoundary: MeasurementBoundary?
    public let rightBoundary: MeasurementBoundary?
    public let dim: Int
    public let n: Int
    public let variant: String?
    public let provider: String?
    public let leftName: String
    public let rightName: String
    public let leftNsPerUnit: Double
    public let rightNsPerUnit: Double
    public let deltaPercent: Double // (right - left) / right * 100

    public init(kind: String, comparison: String, dim: Int, n: Int,
                variant: String?, provider: String?,
                leftName: String, rightName: String,
                leftNsPerUnit: Double, rightNsPerUnit: Double, deltaPercent: Double,
                leftBoundary: MeasurementBoundary? = nil, rightBoundary: MeasurementBoundary? = nil) {
        self.kind = kind
        self.comparison = comparison
        self.family = comparison
        self.purpose = ComparisonFamilies.purpose(family: comparison)
        self.allowedDifferences = (ComparisonFamilies.allowedDifferences(family: comparison) ?? []).sorted()
        self.leftBoundary = leftBoundary
        self.rightBoundary = rightBoundary
        // Derived from the declared boundaries, never from the family name.
        self.leftPreparation = leftBoundary?.preparationDescription ?? "boundary unavailable"
        self.rightPreparation = rightBoundary?.preparationDescription ?? "boundary unavailable"
        self.dim = dim
        self.n = n
        self.variant = variant
        self.provider = provider
        self.leftName = leftName
        self.rightName = rightName
        self.leftNsPerUnit = leftNsPerUnit
        self.rightNsPerUnit = rightNsPerUnit
        self.deltaPercent = deltaPercent
    }
}

/// Complete benchmark run with metadata, results, and optional A/B comparisons
public struct BenchmarkFailure: Codable, Sendable {
    public let name: String
    public let reason: String
    /// Present for case failures and correctness violations; absent for run-level failures. That is how `AppMain` constructs
    /// failures; it is not verified by a test.
    public let boundary: MeasurementBoundary?
    public init(name: String, reason: String, boundary: MeasurementBoundary? = nil) {
        self.name = name; self.reason = reason; self.boundary = boundary
    }
}

public struct BenchRun: Codable, Sendable {
    public let failures: [BenchmarkFailure]
    public let metadata: BenchMetadata
    public let results: [BenchCase]
    public let abComparisons: [ABComparison]?
    public let abMissingSides: [String]
    public let unavailable: [UnavailableCase]

    public init(metadata: BenchMetadata, results: [BenchCase], abComparisons: [ABComparison]?,
                failures: [BenchmarkFailure] = [], abMissingSides: [String] = [], unavailable: [UnavailableCase] = []) {
        self.unavailable = unavailable
        self.abMissingSides = abMissingSides
        self.failures = failures
        self.metadata = metadata
        self.results = results
        self.abComparisons = abComparisons
    }
}

/// Runtime discovery, explicitly distinct from retained compiler/object provenance.
public struct RuntimeObservation: Codable, Sendable {
    public let value: String?
    public let unavailableReason: String?
    public init(value: String) { self.value = value; unavailableReason = nil }
    public static func unavailable(_ reason: String) -> RuntimeObservation { RuntimeObservation(reason: reason) }
    private init(reason: String) { value = nil; unavailableReason = reason }
}

/// Statistics over per-sample average ns per declared work unit. The values are derived from the raw
/// per-sample averages, never from separately measured preparation or overhead (nothing is subtracted).
public struct PerOperationStatistics: Sendable {
    public let samples: [Double]
    public let median: Double
    public let mean: Double
    public let p90: Double
    public let stddev: Double
    public let rsdPercent: Double

    /// - Returns: nil when there are no samples or the unit count is not positive.
    public init?(sampleNsPerInvocation: [Double], unitsPerInvocation: Int) {
        guard !sampleNsPerInvocation.isEmpty, unitsPerInvocation > 0 else { return nil }
        let units = Double(unitsPerInvocation)
        let values = sampleNsPerInvocation.map { $0 / units }
        let count = Double(values.count)
        let mean = values.reduce(0, +) / count
        let sorted = values.sorted()
        let mid = sorted.count / 2
        let median = sorted.count % 2 == 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]
        let rank = Int((0.90 * count).rounded(.up))
        let p90 = sorted[min(max(rank - 1, 0), sorted.count - 1)]
        let variance = values.count > 1 ? values.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / (count - 1) : 0
        let stddev = variance.squareRoot()
        self.samples = values
        self.median = median
        self.mean = mean
        self.p90 = p90
        self.stddev = stddev
        self.rsdPercent = mean != 0 ? stddev / mean * 100 : 0
    }
}
