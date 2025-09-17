import Foundation
import VectorCore

struct NormalizationBench: BenchmarkSuite {
    static let name = "normalize"

    static func run(options: CLIOptions) async -> [BenchResult] {
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

        // Success (generic)
        Harness.warmup { blackHole(try? g.normalized().get()) }
        let r1 = Harness.measure(name: "normalize.success.512.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
            withExtendedLifetime(g) {
                if let u = try? g.normalized().get() { blackHole(u[0]) } else { blackHole(Float.zero) }
            }
        }
        // Success (optimized)
        Harness.warmup { blackHole(try? o.normalized().get()) }
        let r2 = Harness.measure(name: "normalize.success.512.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
            withExtendedLifetime(o) {
                if let u = try? o.normalized().get() { blackHole(u[0]) } else { blackHole(Float.zero) }
            }
        }
        // Failure (generic)
        Harness.warmup { blackHole(try? gz.normalized().get()) }
        let r3 = Harness.measure(name: "normalize.zeroFail.512.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
            withExtendedLifetime(gz) {
                if let u = try? gz.normalized().get() { blackHole(u[0]) } else { blackHole(Float.zero) }
            }
        }
        // Failure (optimized)
        Harness.warmup { blackHole(try? oz.normalized().get()) }
        let r4 = Harness.measure(name: "normalize.zeroFail.512.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
            withExtendedLifetime(oz) {
                if let u = try? oz.normalized().get() { blackHole(u[0]) } else { blackHole(Float.zero) }
            }
        }
        return [r1, r2, r3, r4]
    }

    private static func bench768(_ options: CLIOptions) -> [BenchResult] {
        let g = Vector<Dim768>(repeating: 2.0)
        let o = Vector768Optimized(repeating: 2.0)
        let gz = Vector<Dim768>()
        let oz = Vector768Optimized()

        Harness.warmup { blackHole(try? g.normalized().get()) }
        let r1 = Harness.measure(name: "normalize.success.768.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
            withExtendedLifetime(g) {
                if let u = try? g.normalized().get() { blackHole(u[0]) } else { blackHole(Float.zero) }
            }
        }
        Harness.warmup { blackHole(try? o.normalized().get()) }
        let r2 = Harness.measure(name: "normalize.success.768.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
            withExtendedLifetime(o) {
                if let u = try? o.normalized().get() { blackHole(u[0]) } else { blackHole(Float.zero) }
            }
        }
        Harness.warmup { blackHole(try? gz.normalized().get()) }
        let r3 = Harness.measure(name: "normalize.zeroFail.768.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
            withExtendedLifetime(gz) {
                if let u = try? gz.normalized().get() { blackHole(u[0]) } else { blackHole(Float.zero) }
            }
        }
        Harness.warmup { blackHole(try? oz.normalized().get()) }
        let r4 = Harness.measure(name: "normalize.zeroFail.768.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
            withExtendedLifetime(oz) {
                if let u = try? oz.normalized().get() { blackHole(u[0]) } else { blackHole(Float.zero) }
            }
        }
        return [r1, r2, r3, r4]
    }

    private static func bench1536(_ options: CLIOptions) -> [BenchResult] {
        let g = Vector<Dim1536>(repeating: 2.0)
        let o = Vector1536Optimized(repeating: 2.0)
        let gz = Vector<Dim1536>()
        let oz = Vector1536Optimized()

        Harness.warmup { blackHole(try? g.normalized().get()) }
        let r1 = Harness.measure(name: "normalize.success.1536.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
            withExtendedLifetime(g) {
                if let u = try? g.normalized().get() { blackHole(u[0]) } else { blackHole(Float.zero) }
            }
        }
        Harness.warmup { blackHole(try? o.normalized().get()) }
        let r2 = Harness.measure(name: "normalize.success.1536.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
            withExtendedLifetime(o) {
                if let u = try? o.normalized().get() { blackHole(u[0]) } else { blackHole(Float.zero) }
            }
        }
        Harness.warmup { blackHole(try? gz.normalized().get()) }
        let r3 = Harness.measure(name: "normalize.zeroFail.1536.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
            withExtendedLifetime(gz) {
                if let u = try? gz.normalized().get() { blackHole(u[0]) } else { blackHole(Float.zero) }
            }
        }
        Harness.warmup { blackHole(try? oz.normalized().get()) }
        let r4 = Harness.measure(name: "normalize.zeroFail.1536.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
            withExtendedLifetime(oz) {
                if let u = try? oz.normalized().get() { blackHole(u[0]) } else { blackHole(Float.zero) }
            }
        }
        return [r1, r2, r3, r4]
    }
}
