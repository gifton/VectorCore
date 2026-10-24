import Foundation

/// Internal immutable hooks, absent from ordinary public instances. Observers run
/// synchronously on the owning tuner actor; they never publish cache state.
internal struct TuningTestHooks: Sendable {
    let observer: (@Sendable (TuningCheckpoint) throws -> Void)?
    let latency: (@Sendable (String, Int, Double) -> Double)?
    init(observer: (@Sendable (TuningCheckpoint) throws -> Void)? = nil,
         latency: (@Sendable (String, Int, Double) -> Double)? = nil) {
        self.observer = observer
        self.latency = latency
    }
}

internal enum TuningCheckpoint: Sendable, Equatable {
    case generatedVector
    case convertedFP16(Int)
    case convertedINT8(Int)
    case baseline
    case willExecute(String, Bool)
    case didExecute(String, Bool)
    case accuracy
    case willPublish
    case didPublish
}

internal struct TuningWorkCounts: Sendable, Equatable {
    var generatedVectors = 0
    var convertedFP16Vectors = 0
    var convertedINT8Vectors = 0
    var baselineExecutions = 0
    var warmupExecutions: [String: Int] = [:]
    var measuredExecutions: [String: Int] = [:]
    mutating func record(_ event: TuningCheckpoint) {
        switch event {
        case .generatedVector: generatedVectors += 1
        case .convertedFP16(let n): convertedFP16Vectors += n
        case .convertedINT8(let n): convertedINT8Vectors += n
        case .baseline: baselineExecutions += 1
        case .didExecute(let strategy, let warmup):
            if warmup { warmupExecutions[strategy, default: 0] += 1 }
            else { measuredExecutions[strategy, default: 0] += 1 }
        default: break
        }
    }
}

internal enum PrecisionTuningSupport {
    static func invalid(_ field: String, _ reason: String) -> VectorError {
        VectorError(kind: .invalidConfiguration, context: ErrorContext(additionalInfo: ["field": field, "reason": reason]))
    }
    static func failed(_ reason: String) -> VectorError {
        VectorError(kind: .operationFailed, context: ErrorContext(additionalInfo: ["reason": reason]))
    }
    static func add(_ a: Int, _ b: Int, field: String) throws -> Int {
        let (value, overflow) = a.addingReportingOverflow(b)
        guard !overflow else { throw invalid(field, "checked arithmetic overflow") }
        return value
    }
    static func multiply(_ a: Int, _ b: Int, field: String) throws -> Int {
        let (value, overflow) = a.multipliedReportingOverflow(by: b)
        guard !overflow else { throw invalid(field, "checked arithmetic overflow") }
        return value
    }
    static func validateShape(count: Int, dimension: Int) throws {
        guard [512, 768, 1536].contains(dimension) else { throw invalid("dimension", "supported dimensions are 512, 768, 1536") }
        guard count >= 0 else { throw invalid("candidateCount", "must be nonnegative") }
    }
    static func validateCalibration(count: Int, dimension: Int, iterations: Int, warmups: Int, strategies: Int) throws {
        try validateShape(count: count, dimension: dimension)
        guard count > 0 else { throw invalid("candidateCount", "calibration requires a nonempty workload") }
        guard iterations > 0 else { throw invalid("iterations", "must be positive") }
        guard warmups >= 0 else { throw invalid("warmupIterations", "must be nonnegative") }
        let vectors = try add(count, 1, field: "candidateCount")
        _ = try multiply(try multiply(vectors, dimension, field: "candidateCount"), 4, field: "candidateCount")
        let perStrategy = try add(iterations, warmups, field: "iterations")
        _ = try add(try multiply(perStrategy, strategies, field: "iterations"), 1, field: "iterations")
        _ = try multiply(iterations, MemoryLayout<Double>.stride, field: "iterations")
    }
    static func validateLimit(_ value: Float, field: String) throws {
        guard value.isFinite && value >= 0 else { throw invalid(field, "must be finite and nonnegative") }
    }
    static func seconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) * 1e-18
    }
    static func errors(_ results: [Float], reference: [Float]) -> (maximum: Double, mean: Double)? {
        guard !reference.isEmpty && results.count == reference.count else { return nil }
        var maximum = 0.0, sum = 0.0
        for (result, baseline) in zip(results, reference) {
            guard result.isFinite && baseline.isFinite else { return nil }
            let error = abs(Double(result) - Double(baseline)) / max(abs(Double(baseline)), 1e-6)
            guard error.isFinite && error >= 0 else { return nil }
            maximum = max(maximum, error)
            sum += error
        }
        let mean = sum / Double(reference.count)
        guard mean.isFinite else { return nil }
        return (maximum, mean)
    }
    static func percentile(_ sorted: [Double], _ fraction: Double) -> Double {
        precondition(!sorted.isEmpty && fraction > 0 && fraction <= 1)
        let index = Int(ceil(fraction * Double(sorted.count))) - 1
        return sorted[max(0, min(sorted.count - 1, index))]
    }
}
