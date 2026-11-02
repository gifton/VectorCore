import Foundation
import VectorCore
import VectorCoreBenchmarking

struct NoopBench: BenchmarkSuite {
    static let name = "noop"
    static func catalog(options: ValidatedOptions, progress: BenchmarkProgressReporter) -> [BenchmarkCase] {
        [BenchmarkCase("noop.scalar-add", boundary: Boundaries.record("noop.scalar-add", options)) {
            let body = {
                var acc: Float = 0
                for i in 0..<256 { acc += Float(i) * 0.5 }
                blackHole(acc)
            }
            try Harness.warmup(body)
            return [try Harness.measure(name: "noop.scalar-add", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples, body)]
        }]
    }
}
