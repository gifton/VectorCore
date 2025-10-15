import Foundation
import VectorCore
import VectorCoreBenchmarking

struct NoopBench: BenchmarkSuite {
    static let name = "noop"

    static func run(options: CLIOptions, progress: ProgressReporter) async -> [BenchResult] {
        let label = "noop.scalar-add"
        guard Filters.shouldRun(name: label, options: options) else { return [] }

        let caseStart = Date()
        progress.caseStarted(suite: Self.name, name: label, index: 0, total: 1)

        Harness.warmup {
            var acc: Float = 0
            for i in 0..<256 { acc += Float(i) * 0.5 }
            blackHole(acc)
        }

        let res = Harness.measure(name: label, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
            var acc: Float = 0
            for i in 0..<256 { acc += Float(i) * 0.5 }
            blackHole(acc)
        }

        let caseDuration = Date().timeIntervalSince(caseStart) * 1000.0
        progress.caseCompleted(suite: Self.name, name: label, index: 0, total: 1, durationMs: caseDuration)

        return [res]
    }
}
