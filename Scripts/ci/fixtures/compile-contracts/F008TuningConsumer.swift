import Foundation
import VectorCore

@main struct TuningConsumer {
    static func routes<V: OptimizedVector>(_ type: V.Type, width: Int) async throws {
        var queryValues = Array(repeating: Float(0), count: width)
        queryValues[0] = 1
        let query = try V(queryValues)
        let candidates = try (0..<5).map { index -> V in
            var values = queryValues
            values[0] = Float(index + 1)
            values[1] = Float(index % 3)
            return try V(values)
        }
        for threshold in [0, Int.max] {
            for chunk in [1, Int.max] {
                for mode: CPUComputeProvider.Mode in [.sequential, .parallel] {
                    for kind: KernelKind in [.euclid, .cosineFused] {
                        AutoTuning.set(dim: width, kind: kind,
                                       tuning: Tuning(breakEvenN: threshold, minChunk: chunk, modeBias: .neutral))
                        let results = try await Operations.$computeProvider.withValue(CPUComputeProvider(mode: mode)) {
                            if kind == .euclid {
                                return try await Operations.findNearest(to: query, in: candidates, k: candidates.count, metric: EuclideanDistance())
                            }
                            return try await Operations.findNearest(to: query, in: candidates, k: candidates.count, metric: CosineDistance())
                        }
                        precondition(Set(results.map(\.index)) == Set(0..<5))
                        for result in results {
                            let a = Double(result.index + 1), b = Double(result.index % 3)
                            let expected = kind == .euclid ? ((a - 1) * (a - 1) + b * b).squareRoot()
                                : 1 - a / (a * a + b * b).squareRoot()
                            precondition(abs(Double(result.distance) - expected) < 1e-5)
                        }
                    }
                }
            }
        }
    }

    static func main() async throws {
        let manual = Tuning(breakEvenN: 17, minChunk: 3, modeBias: .favorParallel)
        AutoTuning.set(dim: 999_001, kind: .dot, tuning: manual)
        let saved = AutoTuning.get(dim: 999_001, kind: .dot)
        precondition(saved?.breakEvenN == 17 && saved?.minChunk == 3 && saved?.modeBias == .favorParallel)
        var called = false
        let calibrated = AutoTuning.calibrateIfNeeded(dim: 999_002, kind: .dot, providerCores: 2) {
            called = true
            return CalibrationProbes(nsPerCandidateSeq: 100, parallelOverheadNs: 1000, effectiveParallelFactor: 2)
        }
        precondition(called && calibrated.breakEvenN >= 0 && calibrated.minChunk > 0)
        let fallback = AutoTuning.calibrateIfNeeded(dim: 999_003, kind: .dot, providerCores: 2) {
            CalibrationProbes(nsPerCandidateSeq: .nan, parallelOverheadNs: -1, effectiveParallelFactor: 0)
        }
        precondition(fallback.breakEvenN == 1000 && fallback.minChunk == 512)
        try await routes(Vector512Optimized.self, width: 512)
        try await routes(Vector768Optimized.self, width: 768)
        try await routes(Vector1536Optimized.self, width: 1536)
        print("F008_PUBLIC_TUNING_OK")
    }
}
