import Foundation
import VectorCore
import VectorCoreBenchmarking

struct DistanceBench: BenchmarkSuite {
    static let name = "distance"

    static func run(options: CLIOptions, progress: ProgressReporter) async -> [BenchResult] {
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
        // Early gating: skip entire 512D set if no filters match any case name
        let _names512: [String] = [
            "dist.euclidean.512.generic","dist.cosine.512.generic","dist.manhattan.512.generic","dist.dot.512.generic","dist.chebyshev.512.generic","dist.hamming.512.generic","dist.minkowski.512.generic",
            "dist.euclidean.512.optimized","dist.cosine.512.optimized","dist.manhattan.512.optimized","dist.dot.512.optimized","dist.chebyshev.512.optimized","dist.hamming.512.optimized","dist.minkowski.512.optimized"
        ]
        if !options.filters.isEmpty || !options.excludes.isEmpty {
            let any = _names512.contains { Filters.shouldRun(name: $0, options: options) }
            if !any { return [] }
        }
        let seeds = InputSeeds.distance(dim: 512, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 512, seed: seeds.a)
        let b = InputFactory.randomArray(count: 512, seed: seeds.b)

        let g1 = try! Vector<Dim512>(a)
        let g2 = try! Vector<Dim512>(b)
        let o1 = try! Vector512Optimized(a)
        let o2 = try! Vector512Optimized(b)

        let e = EuclideanDistance()
        let c = CosineDistance()
        let m = ManhattanDistance()
        let d = DotProductDistance()
        let ch = ChebyshevDistance()
        let h = HammingDistance()
        let mk = MinkowskiDistance(p: 3.0)

        var out: [BenchResult] = []
        if Filters.shouldRun(name: "dist.euclidean.512.generic", options: options) {
            Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(e.distance(g1, g2)) } }
            var r = Harness.measure(name: "dist.euclidean.512.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(e.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in Correctness.scalar(current: e.distance(g1, g2), reference: DoubleRef.euclid(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out.append(r)
        }

        if Filters.shouldRun(name: "dist.cosine.512.generic", options: options) {
            Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(c.distance(g1, g2)) } }
            var r = Harness.measure(name: "dist.cosine.512.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(c.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in Correctness.scalar(current: c.distance(g1, g2), reference: DoubleRef.cosineDist(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out.append(r)
        }

        if Filters.shouldRun(name: "dist.manhattan.512.generic", options: options) {
            Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(m.distance(g1, g2)) } }
            var r = Harness.measure(name: "dist.manhattan.512.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(m.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in Correctness.scalar(current: m.distance(g1, g2), reference: DoubleRef.manhattan(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out.append(r)
        }

        if Filters.shouldRun(name: "dist.dot.512.generic", options: options) {
            Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(d.distance(g1, g2)) } }
            var r = Harness.measure(name: "dist.dot.512.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(d.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in let ref = -DoubleRef.dot(ab, bb); return Correctness.scalar(current: d.distance(g1, g2), reference: ref) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out.append(r)
        }

        if Filters.shouldRun(name: "dist.chebyshev.512.generic", options: options) {
            Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(ch.distance(g1, g2)) } }
            var r = Harness.measure(name: "dist.chebyshev.512.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(ch.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in Correctness.scalar(current: ch.distance(g1, g2), reference: DoubleRef.chebyshev(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out.append(r)
        }

        if Filters.shouldRun(name: "dist.hamming.512.generic", options: options) {
            Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(h.distance(g1, g2)) } }
            var r = Harness.measure(name: "dist.hamming.512.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(h.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in Correctness.scalar(current: h.distance(g1, g2), reference: DoubleRef.hamming(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out.append(r)
        }

        if Filters.shouldRun(name: "dist.minkowski.512.generic", options: options) {
            Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(mk.distance(g1, g2)) } }
            var r = Harness.measure(name: "dist.minkowski.512.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(mk.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in Correctness.scalar(current: mk.distance(g1, g2), reference: DoubleRef.minkowski(ab, bb, p: 3.0)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out.append(r)
        }

        // Optimized
        if Filters.shouldRun(name: "dist.euclidean.512.optimized", options: options) {
            Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(e.distance(o1, o2)) } }
            var r = Harness.measure(name: "dist.euclidean.512.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(e.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in Correctness.scalar(current: e.distance(o1, o2), reference: DoubleRef.euclid(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out.append(r)
        }

        if Filters.shouldRun(name: "dist.cosine.512.optimized", options: options) {
            Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(c.distance(o1, o2)) } }
            var r = Harness.measure(name: "dist.cosine.512.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(c.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in Correctness.scalar(current: c.distance(o1, o2), reference: DoubleRef.cosineDist(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out.append(r)
        }

        if Filters.shouldRun(name: "dist.manhattan.512.optimized", options: options) {
            Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(m.distance(o1, o2)) } }
            var r = Harness.measure(name: "dist.manhattan.512.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(m.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in Correctness.scalar(current: m.distance(o1, o2), reference: DoubleRef.manhattan(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out.append(r)
        }

        if Filters.shouldRun(name: "dist.dot.512.optimized", options: options) {
            Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(d.distance(o1, o2)) } }
            var r = Harness.measure(name: "dist.dot.512.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(d.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in let ref = -DoubleRef.dot(ab, bb); return Correctness.scalar(current: d.distance(o1, o2), reference: ref) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out.append(r)
        }

        if Filters.shouldRun(name: "dist.chebyshev.512.optimized", options: options) {
            Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(ch.distance(o1, o2)) } }
            var r = Harness.measure(name: "dist.chebyshev.512.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(ch.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in Correctness.scalar(current: ch.distance(o1, o2), reference: DoubleRef.chebyshev(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out.append(r)
        }

        if Filters.shouldRun(name: "dist.hamming.512.optimized", options: options) {
            Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(h.distance(o1, o2)) } }
            var r = Harness.measure(name: "dist.hamming.512.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(h.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in Correctness.scalar(current: h.distance(o1, o2), reference: DoubleRef.hamming(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out.append(r)
        }

        if Filters.shouldRun(name: "dist.minkowski.512.optimized", options: options) {
            Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(mk.distance(o1, o2)) } }
            var r = Harness.measure(name: "dist.minkowski.512.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(mk.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in Correctness.scalar(current: mk.distance(o1, o2), reference: DoubleRef.minkowski(ab, bb, p: 3.0)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out.append(r)
        }

        return out
    }

    private static func bench768(_ options: CLIOptions) -> [BenchResult] {
        // Early gating: skip entire 768D set if no filters match any case name
        let _names768: [String] = [
            "dist.euclidean.768.generic","dist.cosine.768.generic","dist.manhattan.768.generic","dist.dot.768.generic","dist.chebyshev.768.generic","dist.hamming.768.generic","dist.minkowski.768.generic",
            "dist.euclidean.768.optimized","dist.cosine.768.optimized","dist.manhattan.768.optimized","dist.dot.768.optimized","dist.chebyshev.768.optimized","dist.hamming.768.optimized","dist.minkowski.768.optimized"
        ]
        if !options.filters.isEmpty || !options.excludes.isEmpty {
            let any = _names768.contains { Filters.shouldRun(name: $0, options: options) }
            if !any { return [] }
        }
        let seeds = InputSeeds.distance(dim: 768, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 768, seed: seeds.a)
        let b = InputFactory.randomArray(count: 768, seed: seeds.b)

        let g1 = try! Vector<Dim768>(a)
        let g2 = try! Vector<Dim768>(b)
        let o1 = try! Vector768Optimized(a)
        let o2 = try! Vector768Optimized(b)

        let e = EuclideanDistance()
        let c = CosineDistance()
        let m = ManhattanDistance()
        let d = DotProductDistance()
        let ch = ChebyshevDistance()
        let h = HammingDistance()
        let mk = MinkowskiDistance(p: 3.0)

        var out768: [BenchResult] = []
        // Generic with gating
        if Filters.shouldRun(name: "dist.euclidean.768.generic", options: options) {
            Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(e.distance(g1, g2)) } }
            var r = Harness.measure(name: "dist.euclidean.768.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(e.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in Correctness.scalar(current: e.distance(g1, g2), reference: DoubleRef.euclid(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out768.append(r)
        }
        if Filters.shouldRun(name: "dist.cosine.768.generic", options: options) {
            Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(c.distance(g1, g2)) } }
            var r = Harness.measure(name: "dist.cosine.768.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(c.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in Correctness.scalar(current: c.distance(g1, g2), reference: DoubleRef.cosineDist(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out768.append(r)
        }
        if Filters.shouldRun(name: "dist.manhattan.768.generic", options: options) {
            Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(m.distance(g1, g2)) } }
            var r = Harness.measure(name: "dist.manhattan.768.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(m.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in Correctness.scalar(current: m.distance(g1, g2), reference: DoubleRef.manhattan(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out768.append(r)
        }
        if Filters.shouldRun(name: "dist.dot.768.generic", options: options) {
            Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(d.distance(g1, g2)) } }
            var r = Harness.measure(name: "dist.dot.768.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(d.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in let ref = -DoubleRef.dot(ab, bb); return Correctness.scalar(current: d.distance(g1, g2), reference: ref) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out768.append(r)
        }
        if Filters.shouldRun(name: "dist.chebyshev.768.generic", options: options) {
            Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(ch.distance(g1, g2)) } }
            var r = Harness.measure(name: "dist.chebyshev.768.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(ch.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in Correctness.scalar(current: ch.distance(g1, g2), reference: DoubleRef.chebyshev(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out768.append(r)
        }
        if Filters.shouldRun(name: "dist.hamming.768.generic", options: options) {
            Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(h.distance(g1, g2)) } }
            var r = Harness.measure(name: "dist.hamming.768.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(h.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in Correctness.scalar(current: h.distance(g1, g2), reference: DoubleRef.hamming(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out768.append(r)
        }
        if Filters.shouldRun(name: "dist.minkowski.768.generic", options: options) {
            Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(mk.distance(g1, g2)) } }
            var r = Harness.measure(name: "dist.minkowski.768.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(mk.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in Correctness.scalar(current: mk.distance(g1, g2), reference: DoubleRef.minkowski(ab, bb, p: 3.0)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out768.append(r)
        }

        // Optimized with gating
        if Filters.shouldRun(name: "dist.euclidean.768.optimized", options: options) {
            Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(e.distance(o1, o2)) } }
            var r = Harness.measure(name: "dist.euclidean.768.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(e.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in Correctness.scalar(current: e.distance(o1, o2), reference: DoubleRef.euclid(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out768.append(r)
        }
        if Filters.shouldRun(name: "dist.cosine.768.optimized", options: options) {
            Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(c.distance(o1, o2)) } }
            var r = Harness.measure(name: "dist.cosine.768.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(c.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in Correctness.scalar(current: c.distance(o1, o2), reference: DoubleRef.cosineDist(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out768.append(r)
        }
        if Filters.shouldRun(name: "dist.manhattan.768.optimized", options: options) {
            Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(m.distance(o1, o2)) } }
            var r = Harness.measure(name: "dist.manhattan.768.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(m.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in Correctness.scalar(current: m.distance(o1, o2), reference: DoubleRef.manhattan(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out768.append(r)
        }
        if Filters.shouldRun(name: "dist.dot.768.optimized", options: options) {
            Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(d.distance(o1, o2)) } }
            var r = Harness.measure(name: "dist.dot.768.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(d.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in let ref = -DoubleRef.dot(ab, bb); return Correctness.scalar(current: d.distance(o1, o2), reference: ref) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out768.append(r)
        }
        if Filters.shouldRun(name: "dist.chebyshev.768.optimized", options: options) {
            Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(ch.distance(o1, o2)) } }
            var r = Harness.measure(name: "dist.chebyshev.768.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(ch.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in Correctness.scalar(current: ch.distance(o1, o2), reference: DoubleRef.chebyshev(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out768.append(r)
        }
        if Filters.shouldRun(name: "dist.hamming.768.optimized", options: options) {
            Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(h.distance(o1, o2)) } }
            var r = Harness.measure(name: "dist.hamming.768.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(h.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in Correctness.scalar(current: h.distance(o1, o2), reference: DoubleRef.hamming(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out768.append(r)
        }
        if Filters.shouldRun(name: "dist.minkowski.768.optimized", options: options) {
            Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(mk.distance(o1, o2)) } }
            var r = Harness.measure(name: "dist.minkowski.768.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(mk.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in Correctness.scalar(current: mk.distance(o1, o2), reference: DoubleRef.minkowski(ab, bb, p: 3.0)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out768.append(r)
        }

        return out768
    }

    private static func bench1536(_ options: CLIOptions) -> [BenchResult] {
        // Early gating: skip entire 1536D set if no filters match any case name
        let _names: [String] = [
            "dist.euclidean.1536.generic","dist.cosine.1536.generic","dist.manhattan.1536.generic","dist.dot.1536.generic","dist.chebyshev.1536.generic","dist.hamming.1536.generic","dist.minkowski.1536.generic",
            "dist.euclidean.1536.optimized","dist.cosine.1536.optimized","dist.manhattan.1536.optimized","dist.dot.1536.optimized","dist.chebyshev.1536.optimized","dist.hamming.1536.optimized","dist.minkowski.1536.optimized"
        ]
        if !options.filters.isEmpty || !options.excludes.isEmpty {
            let any = _names.contains { Filters.shouldRun(name: $0, options: options) }
            if !any { return [] }
        }
        let seeds = InputSeeds.distance(dim: 1536, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 1536, seed: seeds.a)
        let b = InputFactory.randomArray(count: 1536, seed: seeds.b)

        let g1 = try! Vector<Dim1536>(a)
        let g2 = try! Vector<Dim1536>(b)
        let o1 = try! Vector1536Optimized(a)
        let o2 = try! Vector1536Optimized(b)

        let e = EuclideanDistance()
        let c = CosineDistance()
        let m = ManhattanDistance()
        let d = DotProductDistance()
        let ch = ChebyshevDistance()
        let h = HammingDistance()
        let mk = MinkowskiDistance(p: 3.0)

        var out1536: [BenchResult] = []

        // Generic with gating
        if Filters.shouldRun(name: "dist.euclidean.1536.generic", options: options) {
            Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(e.distance(g1, g2)) } }
            var r = Harness.measure(name: "dist.euclidean.1536.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(e.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in Correctness.scalar(current: e.distance(g1, g2), reference: DoubleRef.euclid(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out1536.append(r)
        }
        if Filters.shouldRun(name: "dist.cosine.1536.generic", options: options) {
            Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(c.distance(g1, g2)) } }
            var r = Harness.measure(name: "dist.cosine.1536.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(c.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in Correctness.scalar(current: c.distance(g1, g2), reference: DoubleRef.cosineDist(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out1536.append(r)
        }
        if Filters.shouldRun(name: "dist.manhattan.1536.generic", options: options) {
            Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(m.distance(g1, g2)) } }
            var r = Harness.measure(name: "dist.manhattan.1536.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(m.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in Correctness.scalar(current: m.distance(g1, g2), reference: DoubleRef.manhattan(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out1536.append(r)
        }
        if Filters.shouldRun(name: "dist.dot.1536.generic", options: options) {
            Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(d.distance(g1, g2)) } }
            var r = Harness.measure(name: "dist.dot.1536.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(d.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in let ref = -DoubleRef.dot(ab, bb); return Correctness.scalar(current: d.distance(g1, g2), reference: ref) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out1536.append(r)
        }
        if Filters.shouldRun(name: "dist.chebyshev.1536.generic", options: options) {
            Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(ch.distance(g1, g2)) } }
            var r = Harness.measure(name: "dist.chebyshev.1536.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(ch.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in Correctness.scalar(current: ch.distance(g1, g2), reference: DoubleRef.chebyshev(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out1536.append(r)
        }
        if Filters.shouldRun(name: "dist.hamming.1536.generic", options: options) {
            Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(h.distance(g1, g2)) } }
            var r = Harness.measure(name: "dist.hamming.1536.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(h.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in Correctness.scalar(current: h.distance(g1, g2), reference: DoubleRef.hamming(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out1536.append(r)
        }
        if Filters.shouldRun(name: "dist.minkowski.1536.generic", options: options) {
            Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(mk.distance(g1, g2)) } }
            var r = Harness.measure(name: "dist.minkowski.1536.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(mk.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in Correctness.scalar(current: mk.distance(g1, g2), reference: DoubleRef.minkowski(ab, bb, p: 3.0)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out1536.append(r)
        }

        // Optimized with gating
        if Filters.shouldRun(name: "dist.euclidean.1536.optimized", options: options) {
            Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(e.distance(o1, o2)) } }
            var r = Harness.measure(name: "dist.euclidean.1536.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(e.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in Correctness.scalar(current: e.distance(o1, o2), reference: DoubleRef.euclid(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out1536.append(r)
        }
        if Filters.shouldRun(name: "dist.cosine.1536.optimized", options: options) {
            Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(c.distance(o1, o2)) } }
            var r = Harness.measure(name: "dist.cosine.1536.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(c.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in Correctness.scalar(current: c.distance(o1, o2), reference: DoubleRef.cosineDist(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out1536.append(r)
        }
        if Filters.shouldRun(name: "dist.manhattan.1536.optimized", options: options) {
            Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(m.distance(o1, o2)) } }
            var r = Harness.measure(name: "dist.manhattan.1536.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(m.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in Correctness.scalar(current: m.distance(o1, o2), reference: DoubleRef.manhattan(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out1536.append(r)
        }
        if Filters.shouldRun(name: "dist.dot.1536.optimized", options: options) {
            Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(d.distance(o1, o2)) } }
            var r = Harness.measure(name: "dist.dot.1536.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(d.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in let ref = -DoubleRef.dot(ab, bb); return Correctness.scalar(current: d.distance(o1, o2), reference: ref) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out1536.append(r)
        }
        if Filters.shouldRun(name: "dist.chebyshev.1536.optimized", options: options) {
            Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(ch.distance(o1, o2)) } }
            var r = Harness.measure(name: "dist.chebyshev.1536.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(ch.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in Correctness.scalar(current: ch.distance(o1, o2), reference: DoubleRef.chebyshev(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out1536.append(r)
        }
        if Filters.shouldRun(name: "dist.hamming.1536.optimized", options: options) {
            Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(h.distance(o1, o2)) } }
            var r = Harness.measure(name: "dist.hamming.1536.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(h.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in Correctness.scalar(current: h.distance(o1, o2), reference: DoubleRef.hamming(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out1536.append(r)
        }
        if Filters.shouldRun(name: "dist.minkowski.1536.optimized", options: options) {
            Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(mk.distance(o1, o2)) } }
            var r = Harness.measure(name: "dist.minkowski.1536.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(mk.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in Correctness.scalar(current: mk.distance(o1, o2), reference: DoubleRef.minkowski(ab, bb, p: 3.0)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples, meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            out1536.append(r)
        }

        return out1536
    }
}
