import Foundation
import VectorCore
import VectorCoreBenchmarking

struct NormalizationBench: BenchmarkSuite {
    static let name = "normalize"

    static func run(options: CLIOptions, progress: ProgressReporter) async -> [BenchResult] {
        var results: [BenchResult] = []
        for dim in options.dims {
            switch dim {
            case 512: results += bench512(options)
            case 768: results += bench768(options)
            case 1536: results += bench1536(options)
            default:
                fputs("[normalize] unsupported dimension: \(dim)\n", stderr)
            }
        }
        return results
    }

    private static func bench512(_ options: CLIOptions) -> [BenchResult] {
        // Success inputs
        let g = Vector<Dim512>(repeating: 2.0)
        let o = Vector512Optimized(repeating: 2.0)
        // Failure inputs (zero vectors)
        let gz = Vector<Dim512>()
        let oz = Vector512Optimized()

        var out: [BenchResult] = []
        // Success (generic)
        do {
            let name = "normalize.success.512.generic"
            if Filters.shouldRun(name: name, options: options) {
                Harness.warmup { blackHole(try? g.normalized().get()) }
                let r = Harness.measure(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
                    withExtendedLifetime(g) { if let u = try? g.normalized().get() { blackHole(u[0]) } else { blackHole(Float.zero) } }
                }
                out.append(r)
            }
        }
        // Success (optimized)
        do {
            let name = "normalize.success.512.optimized"
            if Filters.shouldRun(name: name, options: options) {
                Harness.warmup { blackHole(try? o.normalized().get()) }
                let r = Harness.measure(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
                    withExtendedLifetime(o) { if let u = try? o.normalized().get() { blackHole(u[0]) } else { blackHole(Float.zero) } }
                }
                out.append(r)
            }
        }
        // Failure (generic)
        do {
            let name = "normalize.zeroFail.512.generic"
            if Filters.shouldRun(name: name, options: options) {
                Harness.warmup { blackHole(try? gz.normalized().get()) }
                let r = Harness.measure(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
                    withExtendedLifetime(gz) { if let u = try? gz.normalized().get() { blackHole(u[0]) } else { blackHole(Float.zero) } }
                }
                out.append(r)
            }
        }
        // Failure (optimized)
        do {
            let name = "normalize.zeroFail.512.optimized"
            if Filters.shouldRun(name: name, options: options) {
                Harness.warmup { blackHole(try? oz.normalized().get()) }
                let r = Harness.measure(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
                    withExtendedLifetime(oz) { if let u = try? oz.normalized().get() { blackHole(u[0]) } else { blackHole(Float.zero) } }
                }
                out.append(r)
            }
        }
        return out
    }

    private static func bench768(_ options: CLIOptions) -> [BenchResult] {
        let g = Vector<Dim768>(repeating: 2.0)
        let o = Vector768Optimized(repeating: 2.0)
        let gz = Vector<Dim768>()
        let oz = Vector768Optimized()

        var out: [BenchResult] = []
        do { let name = "normalize.success.768.generic"; if Filters.shouldRun(name: name, options: options) { Harness.warmup { blackHole(try? g.normalized().get()) }; let r = Harness.measure(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime(g) { if let u = try? g.normalized().get() { blackHole(u[0]) } else { blackHole(Float.zero) } } }; out.append(r) } }
        do { let name = "normalize.success.768.optimized"; if Filters.shouldRun(name: name, options: options) { Harness.warmup { blackHole(try? o.normalized().get()) }; let r = Harness.measure(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime(o) { if let u = try? o.normalized().get() { blackHole(u[0]) } else { blackHole(Float.zero) } } }; out.append(r) } }
        do { let name = "normalize.zeroFail.768.generic"; if Filters.shouldRun(name: name, options: options) { Harness.warmup { blackHole(try? gz.normalized().get()) }; let r = Harness.measure(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime(gz) { if let u = try? gz.normalized().get() { blackHole(u[0]) } else { blackHole(Float.zero) } } }; out.append(r) } }
        do { let name = "normalize.zeroFail.768.optimized"; if Filters.shouldRun(name: name, options: options) { Harness.warmup { blackHole(try? oz.normalized().get()) }; let r = Harness.measure(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime(oz) { if let u = try? oz.normalized().get() { blackHole(u[0]) } else { blackHole(Float.zero) } } }; out.append(r) } }
        return out
    }

    private static func bench1536(_ options: CLIOptions) -> [BenchResult] {
        let g = Vector<Dim1536>(repeating: 2.0)
        let o = Vector1536Optimized(repeating: 2.0)
        let gz = Vector<Dim1536>()
        let oz = Vector1536Optimized()

        var out: [BenchResult] = []
        do { let name = "normalize.success.1536.generic"; if Filters.shouldRun(name: name, options: options) { Harness.warmup { blackHole(try? g.normalized().get()) }; let r = Harness.measure(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime(g) { if let u = try? g.normalized().get() { blackHole(u[0]) } else { blackHole(Float.zero) } } }; out.append(r) } }
        do { let name = "normalize.success.1536.optimized"; if Filters.shouldRun(name: name, options: options) { Harness.warmup { blackHole(try? o.normalized().get()) }; let r = Harness.measure(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime(o) { if let u = try? o.normalized().get() { blackHole(u[0]) } else { blackHole(Float.zero) } } }; out.append(r) } }
        do { let name = "normalize.zeroFail.1536.generic"; if Filters.shouldRun(name: name, options: options) { Harness.warmup { blackHole(try? gz.normalized().get()) }; let r = Harness.measure(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime(gz) { if let u = try? gz.normalized().get() { blackHole(u[0]) } else { blackHole(Float.zero) } } }; out.append(r) } }
        do { let name = "normalize.zeroFail.1536.optimized"; if Filters.shouldRun(name: name, options: options) { Harness.warmup { blackHole(try? oz.normalized().get()) }; let r = Harness.measure(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime(oz) { if let u = try? oz.normalized().get() { blackHole(u[0]) } else { blackHole(Float.zero) } } }; out.append(r) } }
        return out
    }
}
