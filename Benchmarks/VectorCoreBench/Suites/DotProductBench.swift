import Foundation
import VectorCore
import VectorCoreBenchmarking

struct DotProductBench: BenchmarkSuite {
    static let name = "dot"

    static func run(options: CLIOptions, progress: ProgressReporter) async -> [BenchResult] {
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
        // Pre-generate inputs (deterministic via centralized seeds)
        let seeds = InputSeeds.dot(dim: 512, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 512, seed: seeds.a)
        let b = InputFactory.randomArray(count: 512, seed: seeds.b)

        // Generic
        let g1 = try! Vector<Dim512>(a)
        let g2 = try! Vector<Dim512>(b)
        let nameG = "dot.512.generic"
        guard Filters.shouldRun(name: nameG, options: options) else { return [] }

        Harness.warmup {
            withExtendedLifetime((g1,g2)) {
                let v = g1.dotProduct(g2)
                blackHole(v)
            }
        }
        let rg = Harness.measure(name: nameG, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
            withExtendedLifetime((g1,g2)) {
                let v = g1.dotProduct(g2)
                blackHole(v)
            }
        }

        // Optimized
        let o1 = try! Vector512Optimized(a)
        let o2 = try! Vector512Optimized(b)
        let nameO = "dot.512.optimized"
        guard Filters.shouldRun(name: nameO, options: options) else { return [rg] }

        Harness.warmup {
            withExtendedLifetime((o1,o2)) {
                let v = o1.dotProduct(o2)
                blackHole(v)
            }
        }
        let ro = Harness.measure(name: nameO, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
            withExtendedLifetime((o1,o2)) {
                let v = o1.dotProduct(o2)
                blackHole(v)
            }
        }

        return [rg, ro]
    }

    private static func bench768(_ options: CLIOptions) -> [BenchResult] {
        let seeds = InputSeeds.dot(dim: 768, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 768, seed: seeds.a)
        let b = InputFactory.randomArray(count: 768, seed: seeds.b)

        let g1 = try! Vector<Dim768>(a)
        let g2 = try! Vector<Dim768>(b)
        let nameG768 = "dot.768.generic"
        guard Filters.shouldRun(name: nameG768, options: options) else { return [] }
        Harness.warmup {
            withExtendedLifetime((g1,g2)) { let v = g1.dotProduct(g2); blackHole(v) }
        }
        let rg = Harness.measure(name: nameG768, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
            withExtendedLifetime((g1,g2)) { let v = g1.dotProduct(g2); blackHole(v) }
        }

        let o1 = try! Vector768Optimized(a)
        let o2 = try! Vector768Optimized(b)
        let nameO768 = "dot.768.optimized"
        if !Filters.shouldRun(name: nameO768, options: options) { return [rg] }
        Harness.warmup { withExtendedLifetime((o1,o2)) { let v = o1.dotProduct(o2); blackHole(v) } }
        let ro = Harness.measure(name: nameO768, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
            withExtendedLifetime((o1,o2)) { let v = o1.dotProduct(o2); blackHole(v) }
        }

        return [rg, ro]
    }

    private static func bench1536(_ options: CLIOptions) -> [BenchResult] {
        let seeds = InputSeeds.dot(dim: 1536, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 1536, seed: seeds.a)
        let b = InputFactory.randomArray(count: 1536, seed: seeds.b)

        let g1 = try! Vector<Dim1536>(a)
        let g2 = try! Vector<Dim1536>(b)
        let nameG1536 = "dot.1536.generic"
        guard Filters.shouldRun(name: nameG1536, options: options) else { return [] }
        Harness.warmup { withExtendedLifetime((g1,g2)) { let v = g1.dotProduct(g2); blackHole(v) } }
        let rg = Harness.measure(name: nameG1536, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
            withExtendedLifetime((g1,g2)) { let v = g1.dotProduct(g2); blackHole(v) }
        }

        let o1 = try! Vector1536Optimized(a)
        let o2 = try! Vector1536Optimized(b)
        let nameO1536 = "dot.1536.optimized"
        if !Filters.shouldRun(name: nameO1536, options: options) { return [rg] }
        Harness.warmup { withExtendedLifetime((o1,o2)) { let v = o1.dotProduct(o2); blackHole(v) } }
        let ro = Harness.measure(name: nameO1536, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
            withExtendedLifetime((o1,o2)) { let v = o1.dotProduct(o2); blackHole(v) }
        }

        return [rg, ro]
    }
}
