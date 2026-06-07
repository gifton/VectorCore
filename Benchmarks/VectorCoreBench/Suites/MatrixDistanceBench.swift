import Foundation
import VectorCore
import VectorCoreBenchmarking

/// GEMM batch-distance throughput (beta-evolution-4, DOCUMENT-2).
///
/// Measures `MatrixDistance.euclideanSquaredMatrix` (cblas_sgemm / AMX) across
/// N × dim to calibrate the pairwise routing crossover (`matrixRoutingMinN`).
/// Reports ns per candidate-pair (`unitCount = N*N`); compare against the per-pair
/// ns from the `distance`/`batch` suites to locate the breakeven.
///
/// Opt-in: `vectorcore-bench --suites matrix`.
struct MatrixDistanceBench: BenchmarkSuite {
    static let name = "matrix"

    static func run(options: CLIOptions, progress: ProgressReporter) async -> [BenchResult] {
        var results: [BenchResult] = []
        let candidateNs = [64, 256, 1024]   // straddles the default crossover (256)
        for dim in options.dims {
            for n in candidateNs {
                switch dim {
                case 512:
                    results.append(bench((0..<n).map { try! Vector512Optimized(gen(dim, $0)) }, dim: dim, n: n, minTime: options.minTimeSeconds))
                case 768:
                    results.append(bench((0..<n).map { try! Vector768Optimized(gen(dim, $0)) }, dim: dim, n: n, minTime: options.minTimeSeconds))
                case 1536:
                    results.append(bench((0..<n).map { try! Vector1536Optimized(gen(dim, $0)) }, dim: dim, n: n, minTime: options.minTimeSeconds))
                default:
                    fputs("[matrix] unsupported dimension: \(dim)\n", stderr)
                }
            }
        }
        return results
    }

    /// Deterministic, non-trivial vector contents.
    private static func gen(_ dim: Int, _ i: Int) -> [Float] {
        (0..<dim).map { Float((($0 &+ i) % 17) - 8) }
    }

    private static func bench<V: UnifiedVectorBuffer>(
        _ vs: [V], dim: Int, n: Int, minTime: Double
    ) -> BenchResult {
        var out = [Float](repeating: 0, count: n * n)
        func once() {
            MatrixDistance.euclideanSquaredMatrix(queries: vs, candidates: vs, into: &out)
        }

        // Warm up (pipeline + first-touch the output pages).
        once(); once()

        let clock = ContinuousClock()
        let budget = Duration.seconds(max(0.1, minTime))
        var elapsed = Duration.zero
        var iters = 0
        while elapsed < budget {
            elapsed += clock.measure { once() }
            iters += 1
        }

        // Prevent dead-code elimination of the timed work.
        if out[0].isNaN { fputs("", stderr) }

        let c = elapsed.components
        let totalNs = UInt64(c.seconds) &* 1_000_000_000 &+ UInt64(c.attoseconds / 1_000_000_000)
        return BenchResult(
            name: "matrix/euclid/dim\(dim)/N\(n)",
            iterations: iters,
            totalNanoseconds: totalNs,
            unitCount: n * n
        )
    }
}
