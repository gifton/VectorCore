import Foundation
import VectorCore

struct DotProductBench: BenchmarkSuite {
    static let name = "dot"

    static func run(options: CLIOptions) async -> [BenchResult] {
        var results: [BenchResult] = []

        for dim in options.dims {
            switch dim {
            case 512:
                results += bench512(options)
            case 768:
                results += bench768(options)
            case 1536:
                results += bench1536(options)
            default:
                fputs("[dot] unsupported dimension: \(dim)\n", stderr)
            }
        }

        return results
    }

    // MARK: - Per-dimension benches

    private static func bench512(_ options: CLIOptions) -> [BenchResult] {
        // Pre-generate inputs (deterministic)
        let a = InputFactory.randomArray(count: 512, seed: 515_201)
        let b = InputFactory.randomArray(count: 512, seed: 515_202)

        // Generic
        let g1 = try! Vector<Dim512>(a)
        let g2 = try! Vector<Dim512>(b)

        Harness.warmup {
            withExtendedLifetime((g1,g2)) {
                let v = g1.dotProduct(g2)
                blackHole(v)
            }
        }
        let rg = Harness.measure(name: "dot.512.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
            withExtendedLifetime((g1,g2)) {
                let v = g1.dotProduct(g2)
                blackHole(v)
            }
        }

        // Optimized
        let o1 = try! Vector512Optimized(a)
        let o2 = try! Vector512Optimized(b)

        Harness.warmup {
            withExtendedLifetime((o1,o2)) {
                let v = o1.dotProduct(o2)
                blackHole(v)
            }
        }
        let ro = Harness.measure(name: "dot.512.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
            withExtendedLifetime((o1,o2)) {
                let v = o1.dotProduct(o2)
                blackHole(v)
            }
        }

        return [rg, ro]
    }

    private static func bench768(_ options: CLIOptions) -> [BenchResult] {
        let a = InputFactory.randomArray(count: 768, seed: 768_001)
        let b = InputFactory.randomArray(count: 768, seed: 768_002)

        let g1 = try! Vector<Dim768>(a)
        let g2 = try! Vector<Dim768>(b)
        Harness.warmup {
            withExtendedLifetime((g1,g2)) { let v = g1.dotProduct(g2); blackHole(v) }
        }
        let rg = Harness.measure(name: "dot.768.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
            withExtendedLifetime((g1,g2)) { let v = g1.dotProduct(g2); blackHole(v) }
        }

        let o1 = try! Vector768Optimized(a)
        let o2 = try! Vector768Optimized(b)
        Harness.warmup { withExtendedLifetime((o1,o2)) { let v = o1.dotProduct(o2); blackHole(v) } }
        let ro = Harness.measure(name: "dot.768.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
            withExtendedLifetime((o1,o2)) { let v = o1.dotProduct(o2); blackHole(v) }
        }

        return [rg, ro]
    }

    private static func bench1536(_ options: CLIOptions) -> [BenchResult] {
        let a = InputFactory.randomArray(count: 1536, seed: 1_536_001)
        let b = InputFactory.randomArray(count: 1536, seed: 1_536_002)

        let g1 = try! Vector<Dim1536>(a)
        let g2 = try! Vector<Dim1536>(b)
        Harness.warmup { withExtendedLifetime((g1,g2)) { let v = g1.dotProduct(g2); blackHole(v) } }
        let rg = Harness.measure(name: "dot.1536.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
            withExtendedLifetime((g1,g2)) { let v = g1.dotProduct(g2); blackHole(v) }
        }

        let o1 = try! Vector1536Optimized(a)
        let o2 = try! Vector1536Optimized(b)
        Harness.warmup { withExtendedLifetime((o1,o2)) { let v = o1.dotProduct(o2); blackHole(v) } }
        let ro = Harness.measure(name: "dot.1536.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
            withExtendedLifetime((o1,o2)) { let v = o1.dotProduct(o2); blackHole(v) }
        }

        return [rg, ro]
    }
}
