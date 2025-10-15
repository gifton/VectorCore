import Foundation

// MARK: - Base Models (no dependencies on other benchmark models)

/// Binary size information for benchmark metadata
public struct BinarySizes: Codable, Sendable {
    public let benchExecutableBytes: UInt64
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
    public let useUnderscored: Bool
    public let useCKernels: Bool
    public let releaseSize: Bool

    public init(preferSoA: Bool, useMixedPrecision: Bool, abCompare: Bool, abOnly: Bool,
                useUnderscored: Bool, useCKernels: Bool, releaseSize: Bool) {
        self.preferSoA = preferSoA
        self.useMixedPrecision = useMixedPrecision
        self.abCompare = abCompare
        self.abOnly = abOnly
        self.useUnderscored = useUnderscored
        self.useCKernels = useCKernels
        self.releaseSize = releaseSize
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
    public let samples: Int
    public let maxAbsError: Double
    public let meanAbsError: Double
    public let maxRelError: Double
    public let meanRelError: Double

    public init(samples: Int, maxAbsError: Double, meanAbsError: Double,
                maxRelError: Double, meanRelError: Double) {
        self.samples = samples
        self.maxAbsError = maxAbsError
        self.meanAbsError = meanAbsError
        self.maxRelError = maxRelError
        self.meanRelError = meanRelError
    }
}

/// Codable version of CorrectnessStats for serialization
public struct CorrectnessOut: Codable, Sendable {
    public let samples: Int
    public let maxAbsError: Double
    public let meanAbsError: Double
    public let maxRelError: Double
    public let meanRelError: Double

    public init(samples: Int, maxAbsError: Double, meanAbsError: Double,
                maxRelError: Double, meanRelError: Double) {
        self.samples = samples
        self.maxAbsError = maxAbsError
        self.meanAbsError = meanAbsError
        self.maxRelError = maxRelError
        self.meanRelError = meanRelError
    }

    public init(from stats: CorrectnessStats) {
        self.samples = stats.samples
        self.maxAbsError = stats.maxAbsError
        self.meanAbsError = stats.meanAbsError
        self.maxRelError = stats.maxRelError
        self.meanRelError = stats.meanRelError
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
    public let package: String
    public let packageVersion: String
    public let gitSHA: String?
    public let date: String
    public let os: String
    public let arch: String
    public let cpuCores: Int
    public let deviceModel: String?
    public let swiftVersion: String?
    public let buildConfiguration: String
    public let deviceTag: String
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
                flags: RunFlags, thresholds: RunThresholds?, filters: RunFilters?) {
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

    public init(name: String, iterations: Int, totalNanoseconds: UInt64, unitCount: Int = 1,
                samples: Int = 1,
                meanNsPerOp: Double? = nil, medianNsPerOp: Double? = nil, p90NsPerOp: Double? = nil,
                stddevNsPerOp: Double? = nil, rsdPercent: Double? = nil,
                correctness: CorrectnessStats? = nil) {
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

    public init(name: String, params: CaseParams, execution: ExecutionInfo,
                iterations: Int, totalNS: UInt64, nsPerOp: Double, unitCount: Int,
                nsPerUnit: Double, throughputPerSec: Double, gflops: Double?, suspicious: Bool,
                samples: Int?, meanNsPerOp: Double?, medianNsPerOp: Double?, p90NsPerOp: Double?,
                stddevNsPerOp: Double?, rsdPercent: Double?, correctness: CorrectnessOut?) {
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

/// A/B comparison between two benchmark variants
public struct ABComparison: Codable, Sendable {
    public let kind: String // "batch"
    public let comparison: String // "euclidean2_vs_euclidean" | "cosine_prenorm_vs_fused"
    public let dim: Int
    public let n: Int
    public let variant: String?
    public let provider: String?
    public let leftName: String
    public let rightName: String
    public let leftNsPerUnit: Double
    public let rightNsPerUnit: Double
    public let deltaPercent: Double // (left - right) / left * 100

    public init(kind: String, comparison: String, dim: Int, n: Int,
                variant: String?, provider: String?,
                leftName: String, rightName: String,
                leftNsPerUnit: Double, rightNsPerUnit: Double, deltaPercent: Double) {
        self.kind = kind
        self.comparison = comparison
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
public struct BenchRun: Codable, Sendable {
    public let metadata: BenchMetadata
    public let results: [BenchCase]
    public let abComparisons: [ABComparison]?

    public init(metadata: BenchMetadata, results: [BenchCase], abComparisons: [ABComparison]?) {
        self.metadata = metadata
        self.results = results
        self.abComparisons = abComparisons
    }
}
