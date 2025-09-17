import Foundation
import VectorCore

struct DistanceBench: BenchmarkSuite {
    static let name = "distance"

    static func run(options: CLIOptions) async -> [BenchResult] {
        var results: [BenchResult] = []
        for dim in options.dims {
            switch dim {
            case 512: results += bench512(options)
            case 768: results += bench768(options)
            case 1536: results += bench1536(options)
            default:
                fputs("[distance] unsupported dimension: \(dim)\n", stderr)
            }
        }
        return results
    }

    private static func bench512(_ options: CLIOptions) -> [BenchResult] {
        let a = InputFactory.randomArray(count: 512, seed: 615_201)
        let b = InputFactory.randomArray(count: 512, seed: 615_202)

        let g1 = try! Vector<Dim512>(a)
        let g2 = try! Vector<Dim512>(b)
        let o1 = try! Vector512Optimized(a)
        let o2 = try! Vector512Optimized(b)

        let e = EuclideanDistance()
        let c = CosineDistance()
        let m = ManhattanDistance()
        let d = DotProductDistance()

        // Generic
        Harness.warmup { withExtendedLifetime((g1,g2)) { let v = e.distance(g1, g2); blackHole(v) } }
        let r1 = Harness.measure(name: "dist.euclidean.512.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
            withExtendedLifetime((g1,g2)) { let v = e.distance(g1, g2); blackHole(v) }
        }

        Harness.warmup { withExtendedLifetime((g1,g2)) { let v = c.distance(g1, g2); blackHole(v) } }
        let r2 = Harness.measure(name: "dist.cosine.512.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
            withExtendedLifetime((g1,g2)) { let v = c.distance(g1, g2); blackHole(v) }
        }

        Harness.warmup { withExtendedLifetime((g1,g2)) { let v = m.distance(g1, g2); blackHole(v) } }
        let r3 = Harness.measure(name: "dist.manhattan.512.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
            withExtendedLifetime((g1,g2)) { let v = m.distance(g1, g2); blackHole(v) }
        }

        Harness.warmup { withExtendedLifetime((g1,g2)) { let v = d.distance(g1, g2); blackHole(v) } }
        let r4 = Harness.measure(name: "dist.dot.512.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
            withExtendedLifetime((g1,g2)) { let v = d.distance(g1, g2); blackHole(v) }
        }

        // Optimized
        Harness.warmup { withExtendedLifetime((o1,o2)) { let v = e.distance(o1, o2); blackHole(v) } }
        let r5 = Harness.measure(name: "dist.euclidean.512.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((o1,o2)) { let v = e.distance(o1, o2); blackHole(v) } }

        Harness.warmup { withExtendedLifetime((o1,o2)) { let v = c.distance(o1, o2); blackHole(v) } }
        let r6 = Harness.measure(name: "dist.cosine.512.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((o1,o2)) { let v = c.distance(o1, o2); blackHole(v) } }

        Harness.warmup { withExtendedLifetime((o1,o2)) { let v = m.distance(o1, o2); blackHole(v) } }
        let r7 = Harness.measure(name: "dist.manhattan.512.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((o1,o2)) { let v = m.distance(o1, o2); blackHole(v) } }

        Harness.warmup { withExtendedLifetime((o1,o2)) { let v = d.distance(o1, o2); blackHole(v) } }
        let r8 = Harness.measure(name: "dist.dot.512.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((o1,o2)) { let v = d.distance(o1, o2); blackHole(v) } }

        return [r1, r2, r3, r4, r5, r6, r7, r8]
    }

    private static func bench768(_ options: CLIOptions) -> [BenchResult] {
        let a = InputFactory.randomArray(count: 768, seed: 768_101)
        let b = InputFactory.randomArray(count: 768, seed: 768_102)

        let g1 = try! Vector<Dim768>(a)
        let g2 = try! Vector<Dim768>(b)
        let o1 = try! Vector768Optimized(a)
        let o2 = try! Vector768Optimized(b)

        let e = EuclideanDistance()
        let c = CosineDistance()
        let m = ManhattanDistance()
        let d = DotProductDistance()

        Harness.warmup { blackHole(e.distance(g1, g2)) }
        let r1 = Harness.measure(name: "dist.euclidean.768.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((g1,g2)) { let v = e.distance(g1, g2); blackHole(v) } }
        Harness.warmup { blackHole(c.distance(g1, g2)) }
        let r2 = Harness.measure(name: "dist.cosine.768.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((g1,g2)) { let v = c.distance(g1, g2); blackHole(v) } }
        Harness.warmup { blackHole(m.distance(g1, g2)) }
        let r3 = Harness.measure(name: "dist.manhattan.768.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((g1,g2)) { let v = m.distance(g1, g2); blackHole(v) } }
        Harness.warmup { blackHole(d.distance(g1, g2)) }
        let r4 = Harness.measure(name: "dist.dot.768.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((g1,g2)) { let v = d.distance(g1, g2); blackHole(v) } }

        Harness.warmup { blackHole(e.distance(o1, o2)) }
        let r5 = Harness.measure(name: "dist.euclidean.768.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((o1,o2)) { let v = e.distance(o1, o2); blackHole(v) } }
        Harness.warmup { blackHole(c.distance(o1, o2)) }
        let r6 = Harness.measure(name: "dist.cosine.768.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((o1,o2)) { let v = c.distance(o1, o2); blackHole(v) } }
        Harness.warmup { blackHole(m.distance(o1, o2)) }
        let r7 = Harness.measure(name: "dist.manhattan.768.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((o1,o2)) { let v = m.distance(o1, o2); blackHole(v) } }
        Harness.warmup { blackHole(d.distance(o1, o2)) }
        let r8 = Harness.measure(name: "dist.dot.768.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((o1,o2)) { let v = d.distance(o1, o2); blackHole(v) } }

        return [r1, r2, r3, r4, r5, r6, r7, r8]
    }

    private static func bench1536(_ options: CLIOptions) -> [BenchResult] {
        let a = InputFactory.randomArray(count: 1536, seed: 1_536_101)
        let b = InputFactory.randomArray(count: 1536, seed: 1_536_102)

        let g1 = try! Vector<Dim1536>(a)
        let g2 = try! Vector<Dim1536>(b)
        let o1 = try! Vector1536Optimized(a)
        let o2 = try! Vector1536Optimized(b)

        let e = EuclideanDistance()
        let c = CosineDistance()
        let m = ManhattanDistance()
        let d = DotProductDistance()

        Harness.warmup { blackHole(e.distance(g1, g2)) }
        let r1 = Harness.measure(name: "dist.euclidean.1536.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((g1,g2)) { let v = e.distance(g1, g2); blackHole(v) } }
        Harness.warmup { blackHole(c.distance(g1, g2)) }
        let r2 = Harness.measure(name: "dist.cosine.1536.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((g1,g2)) { let v = c.distance(g1, g2); blackHole(v) } }
        Harness.warmup { blackHole(m.distance(g1, g2)) }
        let r3 = Harness.measure(name: "dist.manhattan.1536.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((g1,g2)) { let v = m.distance(g1, g2); blackHole(v) } }
        Harness.warmup { blackHole(d.distance(g1, g2)) }
        let r4 = Harness.measure(name: "dist.dot.1536.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((g1,g2)) { let v = d.distance(g1, g2); blackHole(v) } }

        Harness.warmup { blackHole(e.distance(o1, o2)) }
        let r5 = Harness.measure(name: "dist.euclidean.1536.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((o1,o2)) { let v = e.distance(o1, o2); blackHole(v) } }
        Harness.warmup { blackHole(c.distance(o1, o2)) }
        let r6 = Harness.measure(name: "dist.cosine.1536.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((o1,o2)) { let v = c.distance(o1, o2); blackHole(v) } }
        Harness.warmup { blackHole(m.distance(o1, o2)) }
        let r7 = Harness.measure(name: "dist.manhattan.1536.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((o1,o2)) { let v = m.distance(o1, o2); blackHole(v) } }
        Harness.warmup { blackHole(d.distance(o1, o2)) }
        let r8 = Harness.measure(name: "dist.dot.1536.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((o1,o2)) { let v = d.distance(o1, o2); blackHole(v) } }

        return [r1, r2, r3, r4, r5, r6, r7, r8]
    }
}
