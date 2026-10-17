import Foundation
import VectorCore
import VectorCoreBenchmarking

struct BenchmarkConfigurationError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

/// One immutable execution configuration; CLIOptions is only the input builder.
struct ValidatedOptions: Sendable {
    let suites: [String]
    let minTimeSeconds: Double
    let repeats: Int?
    let samples: Int
    let format: String
    let outputPath: String?
    let profile: String
    let mode: String?
    let runLabel: String?
    let filters: [String]
    let excludes: [String]
    let filterMode: String
    let dims: [Int]
    let batchNs: [Int]
    let runSeed: UInt64
    let preferSoA: Bool?
    let useMixedPrecision: Bool?
    let abCompare: Bool?
    let useCKernels: Bool?
    let abOnly: Bool?
    let maxRelError: Double?
    let maxAbsError: Double?
    let strictCorrectness: Bool
    let progressFormat: String
    let caseFilter: CaseFilter

    init(_ input: CLIOptions, allowedSuites: Set<String> = ["dot", "distance", "normalize", "batch", "memory", "matrix", "noop",
        "boundary"]) throws {
        if let error = input.configurationError { throw BenchmarkConfigurationError(error) }
        func require(_ condition: Bool, _ message: String) throws {
            guard condition else { throw BenchmarkConfigurationError(message) }
        }
        func unique<T: Hashable>(_ values: [T]) -> [T] {
            var seen = Set<T>(); return values.filter { seen.insert($0).inserted }
        }
        try require(!input.suites.isEmpty && input.suites.allSatisfy(allowedSuites.contains), "Unknown or empty suite list")
        try require(["short", "full"].contains(input.profile), "Unknown profile")
        try require(input.mode == nil || ["quick", "full", "smoke"].contains(input.mode!), "Unknown mode")
        try require(["pretty", "json", "csv"].contains(input.format), "Unknown format")
        try require(["glob", "regex"].contains(input.filterMode), "Unknown filter mode")
        try require(["none", "text", "json"].contains(input.progressFormat), "Unknown progress mode")
        try require(input.samples > 0 && (input.repeats == nil || input.repeats! > 0), "Samples and repeats must be positive")
        _ = try BenchmarkArithmetic.nanoseconds(input.minTimeSeconds)
        _ = try BenchmarkArithmetic.product(input.samples, MemoryLayout<Double>.stride)
        if let repeats = input.repeats { _ = try BenchmarkArithmetic.product(repeats, input.samples) }
        try require(!input.dims.isEmpty && input.dims.allSatisfy { $0 > 0 }, "Dimensions must be positive and nonempty")
        try require(!input.batchNs.isEmpty && input.batchNs.allSatisfy { $0 > 0 }, "Batch counts must be positive and nonempty")
        for threshold in [input.maxRelError, input.maxAbsError].compactMap({ $0 }) {
            try require(threshold.isFinite && threshold >= 0, "Correctness thresholds must be finite and nonnegative")
        }
        try require(input.useUnderscored == nil, "Unsupported setting: use-underscored")
        try require(input.releaseSize == nil, "Unsupported setting: release-size")
        try require(VectorCoreCompiledFeatures.configuration == BenchmarkCompiledFeatures.configuration, "Inconsistent compiled configurations")
        if let requested = input.useCKernels {
            try require(requested == VectorCoreCompiledFeatures.useCKernels, "Requested use-ckernels does not match compiled VC_USE_C_KERNELS")
        }
        let selectedSuites = input.abOnly == true ? ["batch"] : unique(input.suites)
        for suite in selectedSuites {
            if ["dot", "distance", "normalize", "batch", "matrix", "boundary"].contains(suite) {
                try require(input.dims.allSatisfy { [512, 768, 1536].contains($0) }, "Unsupported dimension for \(suite)")
            }
            for dim in input.dims {
                if suite == "memory" { _ = try BenchmarkArithmetic.product(try BenchmarkArithmetic.product(dim, 1024), MemoryLayout<Float>.stride) }
                if suite == "batch" || suite == "matrix" || suite == "boundary" {
                    for n in input.batchNs {
                        if suite == "matrix" || suite == "boundary" { try require(Int32(exactly: n) != nil && Int32(exactly: dim) != nil,
                            "Matrix N and dimension must fit CBLAS Int32") }
                        _ = try BenchmarkArithmetic.product(try BenchmarkArithmetic.product(n, dim), MemoryLayout<Float>.stride)
                        if suite == "matrix"
                            || suite == "boundary" { _ = try BenchmarkArithmetic.product(try BenchmarkArithmetic.product(n, n),
                                MemoryLayout<Float>.stride) }
                    }
                }
            }
        }
        suites = selectedSuites
        minTimeSeconds = input.minTimeSeconds
        repeats = input.repeats
        samples = input.samples
        format = input.format
        outputPath = input.outputPath
        profile = input.profile
        mode = input.mode
        runLabel = input.runLabel
        filters = input.filters
        excludes = input.excludes
        filterMode = input.filterMode
        runSeed = input.runSeed
        useCKernels = input.useCKernels
        maxRelError = input.maxRelError
        maxAbsError = input.maxAbsError
        strictCorrectness = input.strictCorrectness
        progressFormat = input.progressFormat

        dims = unique(input.dims); batchNs = unique(input.batchNs)
        preferSoA = input.preferSoA ?? false; useMixedPrecision = input.useMixedPrecision ?? false
        abOnly = input.abOnly ?? false; abCompare = input.abOnly == true || (input.abCompare ?? true)
        caseFilter = try CaseFilter(include: filters, exclude: excludes, mode: filterMode)
    }
}

enum BenchmarkCompiledFeatures {
    #if VC_SWIFTPM_DEBUG
    static let configuration = "debug"
    #elseif VC_SWIFTPM_RELEASE
    static let configuration = "release"
    #else
    static let configuration = "unknown"
    #endif
}

func parseCLI(
    arguments: [String] = Array(CommandLine.arguments.dropFirst()),
    environment: [String: String] = ProcessInfo.processInfo.environment
) -> CLIOptions {
    var result = CLIOptions()
    var iterator = arguments.makeIterator()
    func value(_ key: String) throws -> String {
        guard let next = iterator.next(), !next.hasPrefix("--") else {
            throw BenchmarkConfigurationError("Missing value for \(key)")
        }
        return next
    }
    do {
        for key in ["VC_USE_UNDERSCORED", "VC_RELEASE_SIZE"] where environment[key] != nil {
            throw BenchmarkConfigurationError("Unsupported setting: \(key)")
        }
        while let key = iterator.next() {
            if try CLIParsing.flag(key, result: &result) { continue }
            if CLIParsing.selectionKeys.contains(key) {
                try CLIParsing.selection(key, text: value(key), result: &result)
            } else if CLIParsing.measurementKeys.contains(key) {
                try CLIParsing.measurement(key, text: value(key), result: &result)
            } else {
                throw BenchmarkConfigurationError("Unknown argument: \(key)")
            }
        }
        result = applyMode(applyProfile(result))
        try CLIParsing.environment(environment, result: &result)
    } catch { result.configurationError = String(describing: error) }
    return result
}

private enum CLIParsing {
    static let selectionKeys: Set<String> = ["--suites", "--dims", "--batch-ns", "--format", "--out", "--profile", "--mode", "--run-label", "--filter", "--exclude",
        "--filter-mode", "--progress-format"]
    static let measurementKeys: Set<String> = ["--repeats", "--samples", "--min-time", "--run-seed", "--max-rel-error", "--max-abs-error"]

    static func list(_ text: String) throws -> [String] {
        let values = text.components(separatedBy: ",")
        guard values.allSatisfy({ !$0.isEmpty }) else { throw BenchmarkConfigurationError("Malformed list: \(text)") }
        return values
    }

    static func integer(_ text: String) throws -> Int {
        guard !text.isEmpty, text.allSatisfy({ $0.isASCII && $0.isNumber }), let value = Int(text), value > 0 else { throw BenchmarkConfigurationError("Invalid positive integer: \(text)") }
        return value
    }

    static func number(_ text: String) throws -> Double {
        guard let value = Double(text), value.isFinite, value >= 0 else { throw BenchmarkConfigurationError("Invalid nonnegative number: \(text)") }
        return value
    }

    static func seed(_ text: String) throws -> UInt64 {
        let hex = text.hasPrefix("0x") || text.hasPrefix("0X")
        let digits = hex ? String(text.dropFirst(2)) : text
        let valid = hex ? "0123456789abcdefABCDEF" : "0123456789"
        guard !digits.isEmpty, digits.allSatisfy(valid.contains), let value = UInt64(digits, radix: hex ? 16 : 10) else { throw BenchmarkConfigurationError("Invalid run seed: \(text)") }
        return value
    }

    static func boolean(_ key: String, environment: [String: String]) throws -> Bool? {
        guard let value = environment[key] else { return nil }
        guard value == "0" || value == "1" else { throw BenchmarkConfigurationError("Invalid boolean for \(key); expected 0 or 1") }
        return value == "1"
    }

    static func selection(_ key: String, text: String, result: inout CLIOptions) throws {
        switch key {
        case "--suites": result.suites = try list(text)
        case "--dims": result.dims = try list(text).map(integer); result._userSetDims = true
        case "--batch-ns": result.batchNs = try list(text).map(integer); result._userSetBatchNs = true
        case "--format": result.format = try text
        case "--out": result.outputPath = try text
        case "--profile": result.profile = try text
        case "--mode": result.mode = try text
        case "--run-label": result.runLabel = try text
        case "--filter": result.filters = try list(text)
        case "--exclude": result.excludes = try list(text)
        case "--filter-mode": result.filterMode = try text
        case "--progress-format": result.progressFormat = try text
        default: throw BenchmarkConfigurationError("Unknown argument: \(key)")
        }
    }

    static func measurement(_ key: String, text: String, result: inout CLIOptions) throws {
        switch key {
        case "--repeats": result.repeats = try integer(text)
        case "--samples": result.samples = try integer(text)
        case "--min-time": result.minTimeSeconds = try number(text); result._userSetMinTime = true
        case "--run-seed": result.runSeed = try seed(text); result._userSetRunSeed = true
        case "--max-rel-error": result.maxRelError = try number(text)
        case "--max-abs-error": result.maxAbsError = try number(text)
        default: throw BenchmarkConfigurationError("Unknown argument: \(key)")
        }
    }

    static func flag(_ key: String, result: inout CLIOptions) throws -> Bool {
        switch key {
        case "--help", "-h": result.help = true
        case "--strict-correctness": result.strictCorrectness = true
        case "--prefer-soa": result.preferSoA = true
        case "--no-prefer-soa": result.preferSoA = false
        case "--use-mixed-precision": result.useMixedPrecision = true
        case "--no-use-mixed-precision": result.useMixedPrecision = false
        case "--ab": result.abCompare = true
        case "--no-ab": result.abCompare = false
        case "--ab-only": result.abOnly = true
        case "--no-ab-only": result.abOnly = false
        case "--use-ckernels": result.useCKernels = true
        case "--no-use-ckernels": result.useCKernels = false
        case "--use-underscored", "--no-use-underscored", "--release-size", "--no-release-size": throw BenchmarkConfigurationError("Unsupported setting: \(key)")
        default: return false
        }
        return true
    }

    static func environment(_ environment: [String: String], result: inout CLIOptions) throws {
        if !result._userSetRunSeed, let text = environment["VC_RUN_SEED"] { result.runSeed = try seed(text) }
        if result.runLabel == nil { result.runLabel = environment["VC_RUN_LABEL"] }
        // Read only fallbacks actually selected by precedence.
        if result.preferSoA == nil { result.preferSoA = try boolean("VC_SOA", environment: environment) ?? false }
        if result.useMixedPrecision == nil { result.useMixedPrecision = try boolean("VC_MIXED_PRECISION", environment: environment) ?? false }
        if result.abCompare == nil { result.abCompare = try boolean("VC_BATCH_AB", environment: environment) ?? true }
        if result.abOnly == nil { result.abOnly = try boolean("VC_AB_ONLY", environment: environment) ?? false }
        if result.useCKernels == nil { result.useCKernels = try boolean("VC_USE_CKERNELS", environment: environment) }
        if result.maxRelError == nil, let text = environment["VC_MAX_REL_ERROR"] { result.maxRelError = try number(text) }
        if result.maxAbsError == nil, let text = environment["VC_MAX_ABS_ERROR"] { result.maxAbsError = try number(text) }
    }
}
