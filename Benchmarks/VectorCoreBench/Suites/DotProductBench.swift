import Foundation
import VectorCore
import VectorCoreBenchmarking

struct DotProductBench: BenchmarkSuite {
    static let name = "dot"
    static func catalog(options: ValidatedOptions, progress: BenchmarkProgressReporter) -> [BenchmarkCase] {
        var cases: [BenchmarkCase] = []
        if options.dims.contains(512) {
            cases.append(BenchmarkCase("dot.512.generic", boundary: Boundaries.record("dot.512.generic", options)) {
                let seeds = InputSeeds.dot(dim: 512, runSeed: options.runSeed)
                let a = try Vector<Dim512>(InputFactory.randomArray(count: 512, seed: seeds.a))
                let b = try Vector<Dim512>(InputFactory.randomArray(count: 512, seed: seeds.b))
                try Harness.warmup { withExtendedLifetime((a, b)) { blackHole(a.dotProduct(b)) } }
                return [try Harness.measure(name: "dot.512.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                    samples: options.samples) { withExtendedLifetime((a, b)) { blackHole(a.dotProduct(b)) } }]
            })
        }
        if options.dims.contains(512) {
            cases.append(BenchmarkCase("dot.512.optimized", boundary: Boundaries.record("dot.512.optimized", options)) {
                let seeds = InputSeeds.dot(dim: 512, runSeed: options.runSeed)
                let a = try Vector512Optimized(InputFactory.randomArray(count: 512, seed: seeds.a))
                let b = try Vector512Optimized(InputFactory.randomArray(count: 512, seed: seeds.b))
                try Harness.warmup { withExtendedLifetime((a, b)) { blackHole(a.dotProduct(b)) } }
                return [try Harness.measure(name: "dot.512.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                    samples: options.samples) { withExtendedLifetime((a, b)) { blackHole(a.dotProduct(b)) } }]
            })
        }
        if options.dims.contains(768) {
            cases.append(BenchmarkCase("dot.768.generic", boundary: Boundaries.record("dot.768.generic", options)) {
                let seeds = InputSeeds.dot(dim: 768, runSeed: options.runSeed)
                let a = try Vector<Dim768>(InputFactory.randomArray(count: 768, seed: seeds.a))
                let b = try Vector<Dim768>(InputFactory.randomArray(count: 768, seed: seeds.b))
                try Harness.warmup { withExtendedLifetime((a, b)) { blackHole(a.dotProduct(b)) } }
                return [try Harness.measure(name: "dot.768.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                    samples: options.samples) { withExtendedLifetime((a, b)) { blackHole(a.dotProduct(b)) } }]
            })
        }
        if options.dims.contains(768) {
            cases.append(BenchmarkCase("dot.768.optimized", boundary: Boundaries.record("dot.768.optimized", options)) {
                let seeds = InputSeeds.dot(dim: 768, runSeed: options.runSeed)
                let a = try Vector768Optimized(InputFactory.randomArray(count: 768, seed: seeds.a))
                let b = try Vector768Optimized(InputFactory.randomArray(count: 768, seed: seeds.b))
                try Harness.warmup { withExtendedLifetime((a, b)) { blackHole(a.dotProduct(b)) } }
                return [try Harness.measure(name: "dot.768.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                    samples: options.samples) { withExtendedLifetime((a, b)) { blackHole(a.dotProduct(b)) } }]
            })
        }
        if options.dims.contains(1536) {
            cases.append(BenchmarkCase("dot.1536.generic", boundary: Boundaries.record("dot.1536.generic", options)) {
                let seeds = InputSeeds.dot(dim: 1536, runSeed: options.runSeed)
                let a = try Vector<Dim1536>(InputFactory.randomArray(count: 1536, seed: seeds.a))
                let b = try Vector<Dim1536>(InputFactory.randomArray(count: 1536, seed: seeds.b))
                try Harness.warmup { withExtendedLifetime((a, b)) { blackHole(a.dotProduct(b)) } }
                return [try Harness.measure(name: "dot.1536.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                    samples: options.samples) { withExtendedLifetime((a, b)) { blackHole(a.dotProduct(b)) } }]
            })
        }
        if options.dims.contains(1536) {
            cases.append(BenchmarkCase("dot.1536.optimized", boundary: Boundaries.record("dot.1536.optimized", options)) {
                let seeds = InputSeeds.dot(dim: 1536, runSeed: options.runSeed)
                let a = try Vector1536Optimized(InputFactory.randomArray(count: 1536, seed: seeds.a))
                let b = try Vector1536Optimized(InputFactory.randomArray(count: 1536, seed: seeds.b))
                try Harness.warmup { withExtendedLifetime((a, b)) { blackHole(a.dotProduct(b)) } }
                return [try Harness.measure(name: "dot.1536.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                    samples: options.samples) { withExtendedLifetime((a, b)) { blackHole(a.dotProduct(b)) } }]
            })
        }
        return cases
    }
}
