import Foundation
import VectorCoreBenchmarking

/// Bounded aggregate evidence from actual provider returns.
/// All mutable state is accessed under `lock`; the reference is immutable.
final class ProviderObservations: @unchecked Sendable {
    private let reference: Double
    private let lock = NSLock()
    private var count = 0
    private var maxAbs = 0.0
    private var maxRel = 0.0
    private var sumAbs = 0.0
    private var sumRel = 0.0
    private var invalidReason: String?

    init(reference: Double) { self.reference = reference }

    func observe(_ value: Float) {
        let comparison = Correctness.scalar(current: value, reference: reference)
        lock.withLock {
            guard invalidReason == nil else { return }
            guard comparison.status == .valid else {
                invalidReason = "Nonfinite provider observation or reference statistic"
                return
            }
            let (nextCount, overflow) = count.addingReportingOverflow(1)
            guard !overflow else { invalidReason = "Provider observation count overflow"; return }
            count = nextCount
            maxAbs = max(maxAbs, comparison.maxAbsError)
            maxRel = max(maxRel, comparison.maxRelError)
            sumAbs += comparison.meanAbsError
            sumRel += comparison.meanRelError
            if !sumAbs.isFinite || !sumRel.isFinite {
                invalidReason = "Nonfinite provider error totals"
            }
        }
    }

    var statistics: CorrectnessStats {
        lock.withLock {
            if let invalidReason { return CorrectnessStats(invalidReason: invalidReason) }
            guard count > 0 else { return CorrectnessStats(invalidReason: "No provider observations") }
            return CorrectnessStats(samples: count, maxAbsError: maxAbs, meanAbsError: sumAbs / Double(count),
                                    maxRelError: maxRel, meanRelError: sumRel / Double(count))
        }
    }

    func combining(_ distances: CorrectnessStats) -> CorrectnessStats {
        let actual = statistics
        guard actual.status == .valid else { return actual }
        guard distances.status == .valid else { return distances }
        let (count, overflow) = actual.samples.addingReportingOverflow(distances.samples)
        guard !overflow else { return CorrectnessStats(invalidReason: "Combined comparison count overflow") }
        let actualWeight = Double(actual.samples) / Double(count)
        let distanceWeight = Double(distances.samples) / Double(count)
        return CorrectnessStats(samples: count,
                                maxAbsError: max(actual.maxAbsError, distances.maxAbsError),
                                meanAbsError: actual.meanAbsError * actualWeight + distances.meanAbsError * distanceWeight,
                                maxRelError: max(actual.maxRelError, distances.maxRelError),
                                meanRelError: actual.meanRelError * actualWeight + distances.meanRelError * distanceWeight)
    }
}

extension BenchResult {
    func withCorrectness(_ correctness: CorrectnessStats) -> BenchResult {
        BenchResult(name: name, iterations: iterations, totalNanoseconds: totalNanoseconds, unitCount: unitCount,
                    samples: samples, meanNsPerOp: meanNsPerOp, medianNsPerOp: medianNsPerOp,
                    p90NsPerOp: p90NsPerOp, stddevNsPerOp: stddevNsPerOp, rsdPercent: rsdPercent,
                    correctness: correctness, executionFailure: executionFailure,
                    boundary: boundary, sampleNsPerInvocation: sampleNsPerInvocation, sampleIterations: sampleIterations,
                    warmupIterations: warmupIterations)
    }
}
