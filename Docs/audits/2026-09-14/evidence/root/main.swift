
// SOURCE: Sources/VectorCoreBenchmarking/Models.swift
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


// SOURCE: Sources/VectorCoreBenchmarking/CaseParsing.swift
import Foundation

/// Utilities for parsing structured information from benchmark case names
///
/// Case naming conventions:
/// - dot: `dot.<dim>.<variant>`
/// - dist: `dist.<metric>.<dim>.<variant>`
/// - normalize: `normalize.<status>.<dim>.<variant>`
/// - batch: `batch.<metric>.<dim>.N<count>.<variant>[.<provider>]`
/// - memory: `mem.<operation>.<variant>.<dim>`
public enum CaseParsing {
    /// Parse structured parameters from a benchmark case name
    public static func parse(name: String) -> CaseParams {
        var p = CaseParams()
        let parts = name.split(separator: ".").map(String.init)
        guard !parts.isEmpty else { return p }
        p.kind = parts.first

        switch p.kind {
        case "dot":
            if parts.count >= 3 {
                p.dim = Int(parts[1])
                p.variant = parts[2]
            }
        case "dist":
            if parts.count >= 4 {
                p.metric = parts[1]
                p.dim = Int(parts[2])
                p.variant = parts[3]
            }
        case "normalize":
            if parts.count >= 4 {
                p.status = parts[1]
                p.dim = Int(parts[2])
                p.variant = parts[3]
            }
        case "batch":
            // canonical: batch.<metric>.<dim>.N<count>.<variant>[.<mode>]
            if parts.count >= 5 {
                p.metric = parts[1]
                p.dim = Int(parts[2])
                if parts[3].hasPrefix("N") { p.n = Int(parts[3].dropFirst()) }
                p.variant = parts[4]
                if parts.count >= 6 { p.provider = parts[5] }
            }
        case "mem", "memory":
            // e.g., mem.alloc.aligned.<dim> or mem.copy.<dim>
            if parts.count >= 3 {
                p.metric = parts[1]
                // last component is often dim
                if let last = parts.last, let d = Int(last) { p.dim = d }
                // variant is any middle components beyond metric and before dim
                if parts.count > 3 {
                    let mid = parts[2..<(parts.count - 1)]
                    if !mid.isEmpty { p.variant = mid.joined(separator: "-") }
                }
            }
        default:
            break
        }
        return p
    }
}


// SOURCE: Sources/VectorCoreBenchmarking/JSONWriter.swift
import Foundation

/// Utilities for converting raw benchmark results to structured cases and writing JSON output
public enum JSONWriter {
    /// Convert raw harness results to structured benchmark cases with derived metrics
    ///
    /// - Parameter results: Array of raw BenchResult from harness
    /// - Returns: Array of BenchCase with computed metrics and parsed parameters
    public static func toCases(_ results: [BenchResult]) -> [BenchCase] {
        results.map { r in
            let nsPerOp = Double(r.totalNanoseconds) / Double(max(r.iterations, 1))
            let unitCount = max(r.unitCount, 1)
            let nsPerUnit = nsPerOp / Double(unitCount)
            let throughput = (Double(r.iterations * unitCount) / (Double(r.totalNanoseconds) / 1_000_000_000.0))
            let params = CaseParsing.parse(name: r.name)
            let exec = deriveExecution(from: params)
            // Derive GFLOP/s for dot and dist.dot
            let gflops: Double?
            if params.kind == "dot", let dim = params.dim {
                gflops = (2.0 * Double(dim)) / nsPerOp
            } else if params.kind == "dist", params.metric == "dot", let dim = params.dim {
                gflops = (2.0 * Double(dim)) / nsPerOp
            } else {
                gflops = nil
            }
            return BenchCase(
                name: r.name,
                params: params,
                execution: exec,
                iterations: r.iterations,
                totalNS: r.totalNanoseconds,
                nsPerOp: nsPerOp,
                unitCount: unitCount,
                nsPerUnit: nsPerUnit,
                throughputPerSec: throughput,
                gflops: gflops,
                suspicious: nsPerOp < 5.0,
                samples: r.samples > 1 ? r.samples : nil,
                meanNsPerOp: r.meanNsPerOp,
                medianNsPerOp: r.medianNsPerOp,
                p90NsPerOp: r.p90NsPerOp,
                stddevNsPerOp: r.stddevNsPerOp,
                rsdPercent: r.rsdPercent,
                correctness: r.correctness.map { CorrectnessOut(from: $0) }
            )
        }
    }

    /// Derive execution information from parsed case parameters
    ///
    /// Extracts feature flags like isOptimized, isFused, usesSoA from variant strings
    private static func deriveExecution(from p: CaseParams) -> ExecutionInfo {
        let metric = (p.metric ?? "").lowercased()
        let isOptimized: Bool? = p.variant.map { !$0.lowercased().contains("generic") }
        let isFused: Bool? = p.variant.map { $0.lowercased().contains("fused") }
        let isPreNorm: Bool? = p.variant.map { $0.lowercased().contains("prenorm") }
        let usesSoA: Bool? = p.variant.map { $0.lowercased().contains("soa") }
        let usesFP16: Bool? = p.variant.map { $0.lowercased().contains("fp16") }
        let usesSquared: Bool? = p.kind == "batch" ? (metric == "euclidean2") : nil
        return ExecutionInfo(
            kind: p.kind,
            metric: p.metric,
            dim: p.dim,
            variant: p.variant,
            provider: p.provider,
            n: p.n,
            isOptimized: isOptimized,
            isFused: isFused,
            isPreNormalized: isPreNorm,
            usesSoA: usesSoA,
            usesMixedPrecision: usesFP16,
            usesSquaredDistance: usesSquared
        )
    }

    /// Write complete benchmark run to JSON file
    ///
    /// Automatically creates parent directories if they don't exist
    ///
    /// - Parameters:
    ///   - run: Complete benchmark run with metadata and results
    ///   - path: Output file path
    /// - Throws: File I/O errors or encoding errors
    public static func write(run: BenchRun, to path: String) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(run)
        let url = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url)
    }
}


// SOURCE: Sources/VectorCoreBenchmarking/ABComparisons.swift
import Foundation

/// Utilities for building A/B performance comparisons between benchmark variants
///
/// Generates structured comparisons for:
/// - Euclidean distance: euclidean2 (squared) vs euclidean (sqrt)
/// - Cosine similarity: preNorm vs fused variants
public enum ABComparisons {
    /// Build all A/B comparisons from benchmark cases
    ///
    /// Groups cases by dimension, N, variant, and provider, then compares:
    /// 1. batch euclidean2 vs euclidean (squared distance optimization)
    /// 2. batch cosine preNorm vs fused (normalization strategy)
    ///
    /// - Parameter cases: Array of benchmark cases to analyze
    /// - Returns: Sorted array of A/B comparisons
    public static func buildComparisons(cases: [BenchCase]) -> [ABComparison] {
        var out: [ABComparison] = []

        // Helper to compute ns/unit from case
        func nsPerUnit(_ c: BenchCase) -> Double { c.nsPerUnit }

        // batch euclidean2 vs euclidean
        struct Key: Hashable { let dim: Int; let n: Int; let variant: String?; let provider: String? }
        var byKey: [Key: [String: BenchCase]] = [:]
        for c in cases {
            let p = c.params
            guard p.kind == "batch", let dim = p.dim, let n = p.n else { continue }
            let key = Key(dim: dim, n: n, variant: p.variant, provider: p.provider)
            if let m = p.metric { byKey[key, default: [:]][m] = c }
        }
        for (key, dict) in byKey {
            if let rEuclid2 = dict["euclidean2"], let rEuclid = dict["euclidean"] {
                // Match console summary semantics: label "euclidean2 vs euclidean",
                // value order = euclid2 (left) vs euclid (right),
                // delta = (euclid - euclid2)/euclid * 100
                let left = nsPerUnit(rEuclid2)
                let right = nsPerUnit(rEuclid)
                let delta = (right - left) / right * 100.0
                out.append(ABComparison(
                    kind: "batch",
                    comparison: "euclidean2_vs_euclidean",
                    dim: key.dim,
                    n: key.n,
                    variant: key.variant,
                    provider: key.provider,
                    leftName: rEuclid2.name,
                    rightName: rEuclid.name,
                    leftNsPerUnit: left,
                    rightNsPerUnit: right,
                    deltaPercent: delta
                ))
            }
        }

        // batch cosine preNorm vs fused (optimized)
        var cosByKey: [Key: [String: BenchCase]] = [:]
        for c in cases {
            let p = c.params
            guard p.kind == "batch", p.metric == "cosine", let dim = p.dim, let n = p.n else { continue }
            let v = (p.variant ?? "").lowercased()
            let key = Key(dim: dim, n: n, variant: p.variant, provider: p.provider)
            if v.contains("fused") { cosByKey[key, default: [:]]["fused"] = c }
            else if v.contains("prenorm") { cosByKey[key, default: [:]]["prenorm"] = c }
        }
        for (key, dict) in cosByKey {
            if let fused = dict["fused"], let prenorm = dict["prenorm"] {
                // Match console: label "preNorm vs fused" but we encode
                // comparison key as "cosine_prenorm_vs_fused".
                // value order = preNorm (left) vs fused (right),
                // delta = (fused - preNorm)/fused * 100
                let left = nsPerUnit(prenorm)
                let right = nsPerUnit(fused)
                let delta = (right - left) / right * 100.0
                out.append(ABComparison(
                    kind: "batch",
                    comparison: "cosine_prenorm_vs_fused",
                    dim: key.dim,
                    n: key.n,
                    variant: key.variant,
                    provider: key.provider,
                    leftName: prenorm.name,
                    rightName: fused.name,
                    leftNsPerUnit: left,
                    rightNsPerUnit: right,
                    deltaPercent: delta
                ))
            }
        }

        return out.sorted(by: { a, b in
            if a.comparison != b.comparison { return a.comparison < b.comparison }
            if a.dim != b.dim { return a.dim < b.dim }
            if a.n != b.n { return a.n < b.n }
            return (a.variant ?? "").lexicographicallyPrecedes(b.variant ?? "")
        })
    }
}


// SOURCE: Benchmarks/VectorCoreBench/Utils/Correctness.swift
import Foundation

// CorrectnessStats moved to VectorCoreBenchmarking library

enum Correctness {
    @inline(__always)
    static func relErr(absError: Double, ref: Double, epsilon: Double = 1e-12) -> Double {
        absError / max(abs(ref), epsilon)
    }

    static func scalar(current: Float, reference: Double) -> CorrectnessStats {
        let curr = Double(current)
        let absErr = abs(curr - reference)
        let rel = relErr(absError: absErr, ref: reference)
        return CorrectnessStats(samples: 1, maxAbsError: absErr, meanAbsError: absErr, maxRelError: rel, meanRelError: rel)
    }

    static func vector(currents: [Float], references: [Double]) -> CorrectnessStats {
        let n = min(currents.count, references.count)
        guard n > 0 else { return CorrectnessStats(samples: 0, maxAbsError: 0, meanAbsError: 0, maxRelError: 0, meanRelError: 0) }
        var maxAbs = 0.0
        var sumAbs = 0.0
        var maxRel = 0.0
        var sumRel = 0.0
        for i in 0..<n {
            let c = Double(currents[i])
            let r = references[i]
            let ae = abs(c - r)
            let re = relErr(absError: ae, ref: r)
            if ae > maxAbs { maxAbs = ae }
            if re > maxRel { maxRel = re }
            sumAbs += ae
            sumRel += re
        }
        return CorrectnessStats(samples: n, maxAbsError: maxAbs, meanAbsError: sumAbs / Double(n), maxRelError: maxRel, meanRelError: sumRel / Double(n))
    }
}

// MARK: - Double-precision reference implementations

enum DoubleRef {
    static func dot(_ a: UnsafeBufferPointer<Float>, _ b: UnsafeBufferPointer<Float>) -> Double {
        let n = min(a.count, b.count)
        var s = 0.0
        for i in 0..<n { s += Double(a[i]) * Double(b[i]) }
        return s
    }

    static func euclid2(_ a: UnsafeBufferPointer<Float>, _ b: UnsafeBufferPointer<Float>) -> Double {
        let n = min(a.count, b.count)
        var s = 0.0
        for i in 0..<n { let d = Double(a[i]) - Double(b[i]); s += d * d }
        return s
    }

    static func euclid(_ a: UnsafeBufferPointer<Float>, _ b: UnsafeBufferPointer<Float>) -> Double {
        sqrt(euclid2(a, b))
    }

    static func manhattan(_ a: UnsafeBufferPointer<Float>, _ b: UnsafeBufferPointer<Float>) -> Double {
        let n = min(a.count, b.count)
        var s = 0.0
        for i in 0..<n { s += abs(Double(a[i]) - Double(b[i])) }
        return s
    }

    static func chebyshev(_ a: UnsafeBufferPointer<Float>, _ b: UnsafeBufferPointer<Float>) -> Double {
        let n = min(a.count, b.count)
        var m = 0.0
        for i in 0..<n { let d = abs(Double(a[i]) - Double(b[i])); if d > m { m = d } }
        return m
    }

    static func hamming(_ a: UnsafeBufferPointer<Float>, _ b: UnsafeBufferPointer<Float>, threshold: Double = 0.5) -> Double {
        let n = min(a.count, b.count)
        var count = 0.0
        for i in 0..<n {
            let abit = Double(a[i]) > threshold
            let bbit = Double(b[i]) > threshold
            if abit != bbit { count += 1.0 }
        }
        return count
    }

    static func minkowski(_ a: UnsafeBufferPointer<Float>, _ b: UnsafeBufferPointer<Float>, p: Double) -> Double {
        if p == 1 { return manhattan(a, b) }
        if p == 2 { return euclid(a, b) }
        if p.isInfinite { return chebyshev(a, b) }
        let n = min(a.count, b.count)
        var s = 0.0
        for i in 0..<n { s += pow(abs(Double(a[i]) - Double(b[i])), p) }
        return pow(s, 1.0 / p)
    }

    static func cosineDist(_ a: UnsafeBufferPointer<Float>, _ b: UnsafeBufferPointer<Float>) -> Double {
        let n = min(a.count, b.count)
        var dot = 0.0, sa = 0.0, sb = 0.0
        for i in 0..<n { let x = Double(a[i]); let y = Double(b[i]); dot += x*y; sa += x*x; sb += y*y }
        let mag = sqrt(sa * sb)
        if mag <= Double.ulpOfOne { return 1.0 }
        let cosSim = max(-1.0, min(1.0, dot / mag))
        return 1.0 - cosSim
    }
}



let observations: [(String, CorrectnessStats)] = [
 ("nanScalar", Correctness.scalar(current: .nan, reference: 1)),
 ("nanVector", Correctness.vector(currents: [.nan], references: [1])),
 ("emptyVsNonempty", Correctness.vector(currents: [], references: [1])),
 ("longerOutput", Correctness.vector(currents: [1,999], references: [1])),
 ("equalInfinities", Correctness.scalar(current: .infinity, reference: .infinity))]
for (label,s) in observations { print(label,"samples",s.samples,"maxAbs",s.maxAbsError,"maxRel",s.maxRelError,"violates",s.maxRelError > 1e-6) }
let cases = JSONWriter.toCases([
 BenchResult(name:"batch.cosine.512.N100.optimized-fused.sequential",iterations:1,totalNanoseconds:100),
 BenchResult(name:"batch.cosine.512.N100.optimized-preNorm.sequential",iterations:1,totalNanoseconds:50)])
print("cosineABPairCount",ABComparisons.buildComparisons(cases:cases).count)
let invalid = JSONWriter.toCases([BenchResult(name:"probe",iterations:1,totalNanoseconds:1,correctness: Correctness.scalar(current: .nan, reference: 1))])
do { _ = try JSONEncoder().encode(invalid); print("nanJSON encoded") } catch { print("nanJSON encoding rejected") }
