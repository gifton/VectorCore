import Foundation
import VectorCore
import VectorCoreBenchmarking

struct DistanceBench: BenchmarkSuite {
    static let name = "distance"
    static func catalog(options: ValidatedOptions, progress: BenchmarkProgressReporter) -> [BenchmarkCase] {
        catalog512(options: options) + catalog768(options: options) + catalog1536(options: options)
    }

    private static func catalog512(options: ValidatedOptions) -> [BenchmarkCase] {
        var cases: [BenchmarkCase] = []
        if options.dims.contains(512) {
            cases.append(BenchmarkCase("dist.euclidean.512.generic", boundary: Boundaries.record("dist.euclidean.512.generic", options)) {
        let seeds = InputSeeds.distance(dim: 512, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 512, seed: seeds.a)
        let b = InputFactory.randomArray(count: 512, seed: seeds.b)

        let g1 = try Vector<Dim512>(a)
        let g2 = try Vector<Dim512>(b)

        let e = EuclideanDistance()

            try Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(e.distance(g1, g2)) } }
            var r = try Harness.measure(name: "dist.euclidean.512.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(e.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in Correctness.scalar(current: e.distance(g1, g2), reference: DoubleRef.euclid(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(512) {
            cases.append(BenchmarkCase("dist.cosine.512.generic", boundary: Boundaries.record("dist.cosine.512.generic", options)) {
        let seeds = InputSeeds.distance(dim: 512, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 512, seed: seeds.a)
        let b = InputFactory.randomArray(count: 512, seed: seeds.b)

        let g1 = try Vector<Dim512>(a)
        let g2 = try Vector<Dim512>(b)

        let c = CosineDistance()

            try Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(c.distance(g1, g2)) } }
            var r = try Harness.measure(name: "dist.cosine.512.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(c.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in Correctness.scalar(current: c.distance(g1, g2), reference: DoubleRef.cosineDist(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(512) {
            cases.append(BenchmarkCase("dist.manhattan.512.generic", boundary: Boundaries.record("dist.manhattan.512.generic", options)) {
        let seeds = InputSeeds.distance(dim: 512, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 512, seed: seeds.a)
        let b = InputFactory.randomArray(count: 512, seed: seeds.b)

        let g1 = try Vector<Dim512>(a)
        let g2 = try Vector<Dim512>(b)

        let m = ManhattanDistance()

            try Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(m.distance(g1, g2)) } }
            var r = try Harness.measure(name: "dist.manhattan.512.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(m.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in Correctness.scalar(current: m.distance(g1, g2), reference: DoubleRef.manhattan(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(512) {
            cases.append(BenchmarkCase("dist.dot.512.generic", boundary: Boundaries.record("dist.dot.512.generic", options)) {
        let seeds = InputSeeds.distance(dim: 512, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 512, seed: seeds.a)
        let b = InputFactory.randomArray(count: 512, seed: seeds.b)

        let g1 = try Vector<Dim512>(a)
        let g2 = try Vector<Dim512>(b)

        let d = DotProductDistance()

            try Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(d.distance(g1, g2)) } }
            var r = try Harness.measure(name: "dist.dot.512.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(d.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in let ref = -DoubleRef.dot(ab,
                bb); return Correctness.scalar(current: d.distance(g1, g2), reference: ref) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(512) {
            cases.append(BenchmarkCase("dist.chebyshev.512.generic", boundary: Boundaries.record("dist.chebyshev.512.generic", options)) {
        let seeds = InputSeeds.distance(dim: 512, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 512, seed: seeds.a)
        let b = InputFactory.randomArray(count: 512, seed: seeds.b)

        let g1 = try Vector<Dim512>(a)
        let g2 = try Vector<Dim512>(b)

        let ch = ChebyshevDistance()

            try Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(ch.distance(g1, g2)) } }
            var r = try Harness.measure(name: "dist.chebyshev.512.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(ch.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in Correctness.scalar(current: ch.distance(g1, g2), reference: DoubleRef.chebyshev(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(512) {
            cases.append(BenchmarkCase("dist.hamming.512.generic", boundary: Boundaries.record("dist.hamming.512.generic", options)) {
        let seeds = InputSeeds.distance(dim: 512, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 512, seed: seeds.a)
        let b = InputFactory.randomArray(count: 512, seed: seeds.b)

        let g1 = try Vector<Dim512>(a)
        let g2 = try Vector<Dim512>(b)

        let h = HammingDistance()

            try Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(h.distance(g1, g2)) } }
            var r = try Harness.measure(name: "dist.hamming.512.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(h.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in Correctness.scalar(current: h.distance(g1, g2), reference: DoubleRef.hamming(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(512) {
            cases.append(BenchmarkCase("dist.minkowski.512.generic", boundary: Boundaries.record("dist.minkowski.512.generic", options)) {
        let seeds = InputSeeds.distance(dim: 512, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 512, seed: seeds.a)
        let b = InputFactory.randomArray(count: 512, seed: seeds.b)

        let g1 = try Vector<Dim512>(a)
        let g2 = try Vector<Dim512>(b)

        let mk = MinkowskiDistance(p: 3.0)

            try Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(mk.distance(g1, g2)) } }
            var r = try Harness.measure(name: "dist.minkowski.512.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(mk.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in Correctness.scalar(current: mk.distance(g1, g2), reference: DoubleRef.minkowski(ab, bb, p: 3.0)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(512) {
            cases.append(BenchmarkCase("dist.euclidean.512.optimized", boundary: Boundaries.record("dist.euclidean.512.optimized",
                options)) {
        let seeds = InputSeeds.distance(dim: 512, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 512, seed: seeds.a)
        let b = InputFactory.randomArray(count: 512, seed: seeds.b)

        let o1 = try Vector512Optimized(a)
        let o2 = try Vector512Optimized(b)

        let e = EuclideanDistance()

            try Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(e.distance(o1, o2)) } }
            var r = try Harness.measure(name: "dist.euclidean.512.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(e.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in Correctness.scalar(current: e.distance(o1, o2), reference: DoubleRef.euclid(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(512) {
            cases.append(BenchmarkCase("dist.cosine.512.optimized", boundary: Boundaries.record("dist.cosine.512.optimized", options)) {
        let seeds = InputSeeds.distance(dim: 512, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 512, seed: seeds.a)
        let b = InputFactory.randomArray(count: 512, seed: seeds.b)

        let o1 = try Vector512Optimized(a)
        let o2 = try Vector512Optimized(b)

        let c = CosineDistance()

            try Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(c.distance(o1, o2)) } }
            var r = try Harness.measure(name: "dist.cosine.512.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(c.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in Correctness.scalar(current: c.distance(o1, o2), reference: DoubleRef.cosineDist(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(512) {
            cases.append(BenchmarkCase("dist.manhattan.512.optimized", boundary: Boundaries.record("dist.manhattan.512.optimized",
                options)) {
        let seeds = InputSeeds.distance(dim: 512, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 512, seed: seeds.a)
        let b = InputFactory.randomArray(count: 512, seed: seeds.b)

        let o1 = try Vector512Optimized(a)
        let o2 = try Vector512Optimized(b)

        let m = ManhattanDistance()

            try Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(m.distance(o1, o2)) } }
            var r = try Harness.measure(name: "dist.manhattan.512.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(m.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in Correctness.scalar(current: m.distance(o1, o2), reference: DoubleRef.manhattan(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(512) {
            cases.append(BenchmarkCase("dist.dot.512.optimized", boundary: Boundaries.record("dist.dot.512.optimized", options)) {
        let seeds = InputSeeds.distance(dim: 512, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 512, seed: seeds.a)
        let b = InputFactory.randomArray(count: 512, seed: seeds.b)

        let o1 = try Vector512Optimized(a)
        let o2 = try Vector512Optimized(b)

        let d = DotProductDistance()

            try Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(d.distance(o1, o2)) } }
            var r = try Harness.measure(name: "dist.dot.512.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(d.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in let ref = -DoubleRef.dot(ab,
                bb); return Correctness.scalar(current: d.distance(o1, o2), reference: ref) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(512) {
            cases.append(BenchmarkCase("dist.chebyshev.512.optimized", boundary: Boundaries.record("dist.chebyshev.512.optimized",
                options)) {
        let seeds = InputSeeds.distance(dim: 512, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 512, seed: seeds.a)
        let b = InputFactory.randomArray(count: 512, seed: seeds.b)

        let o1 = try Vector512Optimized(a)
        let o2 = try Vector512Optimized(b)

        let ch = ChebyshevDistance()

            try Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(ch.distance(o1, o2)) } }
            var r = try Harness.measure(name: "dist.chebyshev.512.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(ch.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in Correctness.scalar(current: ch.distance(o1, o2), reference: DoubleRef.chebyshev(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(512) {
            cases.append(BenchmarkCase("dist.hamming.512.optimized", boundary: Boundaries.record("dist.hamming.512.optimized", options)) {
        let seeds = InputSeeds.distance(dim: 512, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 512, seed: seeds.a)
        let b = InputFactory.randomArray(count: 512, seed: seeds.b)

        let o1 = try Vector512Optimized(a)
        let o2 = try Vector512Optimized(b)

        let h = HammingDistance()

            try Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(h.distance(o1, o2)) } }
            var r = try Harness.measure(name: "dist.hamming.512.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(h.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in Correctness.scalar(current: h.distance(o1, o2), reference: DoubleRef.hamming(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(512) {
            cases.append(BenchmarkCase("dist.minkowski.512.optimized", boundary: Boundaries.record("dist.minkowski.512.optimized",
                options)) {
        let seeds = InputSeeds.distance(dim: 512, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 512, seed: seeds.a)
        let b = InputFactory.randomArray(count: 512, seed: seeds.b)

        let o1 = try Vector512Optimized(a)
        let o2 = try Vector512Optimized(b)

        let mk = MinkowskiDistance(p: 3.0)

            try Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(mk.distance(o1, o2)) } }
            var r = try Harness.measure(name: "dist.minkowski.512.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(mk.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in Correctness.scalar(current: mk.distance(o1, o2), reference: DoubleRef.minkowski(ab, bb, p: 3.0)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        return cases
    }

    private static func catalog768(options: ValidatedOptions) -> [BenchmarkCase] {
        var cases: [BenchmarkCase] = []
        if options.dims.contains(768) {
            cases.append(BenchmarkCase("dist.euclidean.768.generic", boundary: Boundaries.record("dist.euclidean.768.generic", options)) {
        let seeds = InputSeeds.distance(dim: 768, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 768, seed: seeds.a)
        let b = InputFactory.randomArray(count: 768, seed: seeds.b)

        let g1 = try Vector<Dim768>(a)
        let g2 = try Vector<Dim768>(b)

        let e = EuclideanDistance()

            try Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(e.distance(g1, g2)) } }
            var r = try Harness.measure(name: "dist.euclidean.768.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(e.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in Correctness.scalar(current: e.distance(g1, g2), reference: DoubleRef.euclid(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(768) {
            cases.append(BenchmarkCase("dist.cosine.768.generic", boundary: Boundaries.record("dist.cosine.768.generic", options)) {
        let seeds = InputSeeds.distance(dim: 768, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 768, seed: seeds.a)
        let b = InputFactory.randomArray(count: 768, seed: seeds.b)

        let g1 = try Vector<Dim768>(a)
        let g2 = try Vector<Dim768>(b)

        let c = CosineDistance()

            try Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(c.distance(g1, g2)) } }
            var r = try Harness.measure(name: "dist.cosine.768.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(c.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in Correctness.scalar(current: c.distance(g1, g2), reference: DoubleRef.cosineDist(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(768) {
            cases.append(BenchmarkCase("dist.manhattan.768.generic", boundary: Boundaries.record("dist.manhattan.768.generic", options)) {
        let seeds = InputSeeds.distance(dim: 768, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 768, seed: seeds.a)
        let b = InputFactory.randomArray(count: 768, seed: seeds.b)

        let g1 = try Vector<Dim768>(a)
        let g2 = try Vector<Dim768>(b)

        let m = ManhattanDistance()

            try Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(m.distance(g1, g2)) } }
            var r = try Harness.measure(name: "dist.manhattan.768.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(m.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in Correctness.scalar(current: m.distance(g1, g2), reference: DoubleRef.manhattan(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(768) {
            cases.append(BenchmarkCase("dist.dot.768.generic", boundary: Boundaries.record("dist.dot.768.generic", options)) {
        let seeds = InputSeeds.distance(dim: 768, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 768, seed: seeds.a)
        let b = InputFactory.randomArray(count: 768, seed: seeds.b)

        let g1 = try Vector<Dim768>(a)
        let g2 = try Vector<Dim768>(b)

        let d = DotProductDistance()

            try Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(d.distance(g1, g2)) } }
            var r = try Harness.measure(name: "dist.dot.768.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(d.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in let ref = -DoubleRef.dot(ab,
                bb); return Correctness.scalar(current: d.distance(g1, g2), reference: ref) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(768) {
            cases.append(BenchmarkCase("dist.chebyshev.768.generic", boundary: Boundaries.record("dist.chebyshev.768.generic", options)) {
        let seeds = InputSeeds.distance(dim: 768, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 768, seed: seeds.a)
        let b = InputFactory.randomArray(count: 768, seed: seeds.b)

        let g1 = try Vector<Dim768>(a)
        let g2 = try Vector<Dim768>(b)

        let ch = ChebyshevDistance()

            try Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(ch.distance(g1, g2)) } }
            var r = try Harness.measure(name: "dist.chebyshev.768.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(ch.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in Correctness.scalar(current: ch.distance(g1, g2), reference: DoubleRef.chebyshev(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(768) {
            cases.append(BenchmarkCase("dist.hamming.768.generic", boundary: Boundaries.record("dist.hamming.768.generic", options)) {
        let seeds = InputSeeds.distance(dim: 768, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 768, seed: seeds.a)
        let b = InputFactory.randomArray(count: 768, seed: seeds.b)

        let g1 = try Vector<Dim768>(a)
        let g2 = try Vector<Dim768>(b)

        let h = HammingDistance()

            try Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(h.distance(g1, g2)) } }
            var r = try Harness.measure(name: "dist.hamming.768.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(h.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in Correctness.scalar(current: h.distance(g1, g2), reference: DoubleRef.hamming(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(768) {
            cases.append(BenchmarkCase("dist.minkowski.768.generic", boundary: Boundaries.record("dist.minkowski.768.generic", options)) {
        let seeds = InputSeeds.distance(dim: 768, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 768, seed: seeds.a)
        let b = InputFactory.randomArray(count: 768, seed: seeds.b)

        let g1 = try Vector<Dim768>(a)
        let g2 = try Vector<Dim768>(b)

        let mk = MinkowskiDistance(p: 3.0)

            try Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(mk.distance(g1, g2)) } }
            var r = try Harness.measure(name: "dist.minkowski.768.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(mk.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in Correctness.scalar(current: mk.distance(g1, g2), reference: DoubleRef.minkowski(ab, bb, p: 3.0)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(768) {
            cases.append(BenchmarkCase("dist.euclidean.768.optimized", boundary: Boundaries.record("dist.euclidean.768.optimized",
                options)) {
        let seeds = InputSeeds.distance(dim: 768, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 768, seed: seeds.a)
        let b = InputFactory.randomArray(count: 768, seed: seeds.b)

        let o1 = try Vector768Optimized(a)
        let o2 = try Vector768Optimized(b)

        let e = EuclideanDistance()

            try Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(e.distance(o1, o2)) } }
            var r = try Harness.measure(name: "dist.euclidean.768.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(e.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in Correctness.scalar(current: e.distance(o1, o2), reference: DoubleRef.euclid(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(768) {
            cases.append(BenchmarkCase("dist.cosine.768.optimized", boundary: Boundaries.record("dist.cosine.768.optimized", options)) {
        let seeds = InputSeeds.distance(dim: 768, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 768, seed: seeds.a)
        let b = InputFactory.randomArray(count: 768, seed: seeds.b)

        let o1 = try Vector768Optimized(a)
        let o2 = try Vector768Optimized(b)

        let c = CosineDistance()

            try Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(c.distance(o1, o2)) } }
            var r = try Harness.measure(name: "dist.cosine.768.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(c.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in Correctness.scalar(current: c.distance(o1, o2), reference: DoubleRef.cosineDist(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(768) {
            cases.append(BenchmarkCase("dist.manhattan.768.optimized", boundary: Boundaries.record("dist.manhattan.768.optimized",
                options)) {
        let seeds = InputSeeds.distance(dim: 768, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 768, seed: seeds.a)
        let b = InputFactory.randomArray(count: 768, seed: seeds.b)

        let o1 = try Vector768Optimized(a)
        let o2 = try Vector768Optimized(b)

        let m = ManhattanDistance()

            try Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(m.distance(o1, o2)) } }
            var r = try Harness.measure(name: "dist.manhattan.768.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(m.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in Correctness.scalar(current: m.distance(o1, o2), reference: DoubleRef.manhattan(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(768) {
            cases.append(BenchmarkCase("dist.dot.768.optimized", boundary: Boundaries.record("dist.dot.768.optimized", options)) {
        let seeds = InputSeeds.distance(dim: 768, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 768, seed: seeds.a)
        let b = InputFactory.randomArray(count: 768, seed: seeds.b)

        let o1 = try Vector768Optimized(a)
        let o2 = try Vector768Optimized(b)

        let d = DotProductDistance()

            try Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(d.distance(o1, o2)) } }
            var r = try Harness.measure(name: "dist.dot.768.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(d.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in let ref = -DoubleRef.dot(ab,
                bb); return Correctness.scalar(current: d.distance(o1, o2), reference: ref) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(768) {
            cases.append(BenchmarkCase("dist.chebyshev.768.optimized", boundary: Boundaries.record("dist.chebyshev.768.optimized",
                options)) {
        let seeds = InputSeeds.distance(dim: 768, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 768, seed: seeds.a)
        let b = InputFactory.randomArray(count: 768, seed: seeds.b)

        let o1 = try Vector768Optimized(a)
        let o2 = try Vector768Optimized(b)

        let ch = ChebyshevDistance()

            try Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(ch.distance(o1, o2)) } }
            var r = try Harness.measure(name: "dist.chebyshev.768.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(ch.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in Correctness.scalar(current: ch.distance(o1, o2), reference: DoubleRef.chebyshev(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(768) {
            cases.append(BenchmarkCase("dist.hamming.768.optimized", boundary: Boundaries.record("dist.hamming.768.optimized", options)) {
        let seeds = InputSeeds.distance(dim: 768, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 768, seed: seeds.a)
        let b = InputFactory.randomArray(count: 768, seed: seeds.b)

        let o1 = try Vector768Optimized(a)
        let o2 = try Vector768Optimized(b)

        let h = HammingDistance()

            try Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(h.distance(o1, o2)) } }
            var r = try Harness.measure(name: "dist.hamming.768.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(h.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in Correctness.scalar(current: h.distance(o1, o2), reference: DoubleRef.hamming(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(768) {
            cases.append(BenchmarkCase("dist.minkowski.768.optimized", boundary: Boundaries.record("dist.minkowski.768.optimized",
                options)) {
        let seeds = InputSeeds.distance(dim: 768, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 768, seed: seeds.a)
        let b = InputFactory.randomArray(count: 768, seed: seeds.b)

        let o1 = try Vector768Optimized(a)
        let o2 = try Vector768Optimized(b)

        let mk = MinkowskiDistance(p: 3.0)

            try Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(mk.distance(o1, o2)) } }
            var r = try Harness.measure(name: "dist.minkowski.768.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(mk.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in Correctness.scalar(current: mk.distance(o1, o2), reference: DoubleRef.minkowski(ab, bb, p: 3.0)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        return cases
    }

    private static func catalog1536(options: ValidatedOptions) -> [BenchmarkCase] {
        var cases: [BenchmarkCase] = []
        if options.dims.contains(1536) {
            cases.append(BenchmarkCase("dist.euclidean.1536.generic", boundary: Boundaries.record("dist.euclidean.1536.generic", options)) {
        let seeds = InputSeeds.distance(dim: 1536, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 1536, seed: seeds.a)
        let b = InputFactory.randomArray(count: 1536, seed: seeds.b)

        let g1 = try Vector<Dim1536>(a)
        let g2 = try Vector<Dim1536>(b)

        let e = EuclideanDistance()

            try Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(e.distance(g1, g2)) } }
            var r = try Harness.measure(name: "dist.euclidean.1536.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(e.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in Correctness.scalar(current: e.distance(g1, g2), reference: DoubleRef.euclid(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(1536) {
            cases.append(BenchmarkCase("dist.cosine.1536.generic", boundary: Boundaries.record("dist.cosine.1536.generic", options)) {
        let seeds = InputSeeds.distance(dim: 1536, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 1536, seed: seeds.a)
        let b = InputFactory.randomArray(count: 1536, seed: seeds.b)

        let g1 = try Vector<Dim1536>(a)
        let g2 = try Vector<Dim1536>(b)

        let c = CosineDistance()

            try Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(c.distance(g1, g2)) } }
            var r = try Harness.measure(name: "dist.cosine.1536.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(c.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in Correctness.scalar(current: c.distance(g1, g2), reference: DoubleRef.cosineDist(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(1536) {
            cases.append(BenchmarkCase("dist.manhattan.1536.generic", boundary: Boundaries.record("dist.manhattan.1536.generic", options)) {
        let seeds = InputSeeds.distance(dim: 1536, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 1536, seed: seeds.a)
        let b = InputFactory.randomArray(count: 1536, seed: seeds.b)

        let g1 = try Vector<Dim1536>(a)
        let g2 = try Vector<Dim1536>(b)

        let m = ManhattanDistance()

            try Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(m.distance(g1, g2)) } }
            var r = try Harness.measure(name: "dist.manhattan.1536.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(m.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in Correctness.scalar(current: m.distance(g1, g2), reference: DoubleRef.manhattan(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(1536) {
            cases.append(BenchmarkCase("dist.dot.1536.generic", boundary: Boundaries.record("dist.dot.1536.generic", options)) {
        let seeds = InputSeeds.distance(dim: 1536, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 1536, seed: seeds.a)
        let b = InputFactory.randomArray(count: 1536, seed: seeds.b)

        let g1 = try Vector<Dim1536>(a)
        let g2 = try Vector<Dim1536>(b)

        let d = DotProductDistance()

            try Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(d.distance(g1, g2)) } }
            var r = try Harness.measure(name: "dist.dot.1536.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(d.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in let ref = -DoubleRef.dot(ab,
                bb); return Correctness.scalar(current: d.distance(g1, g2), reference: ref) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(1536) {
            cases.append(BenchmarkCase("dist.chebyshev.1536.generic", boundary: Boundaries.record("dist.chebyshev.1536.generic", options)) {
        let seeds = InputSeeds.distance(dim: 1536, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 1536, seed: seeds.a)
        let b = InputFactory.randomArray(count: 1536, seed: seeds.b)

        let g1 = try Vector<Dim1536>(a)
        let g2 = try Vector<Dim1536>(b)

        let ch = ChebyshevDistance()

            try Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(ch.distance(g1, g2)) } }
            var r = try Harness.measure(name: "dist.chebyshev.1536.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(ch.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in Correctness.scalar(current: ch.distance(g1, g2), reference: DoubleRef.chebyshev(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(1536) {
            cases.append(BenchmarkCase("dist.hamming.1536.generic", boundary: Boundaries.record("dist.hamming.1536.generic", options)) {
        let seeds = InputSeeds.distance(dim: 1536, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 1536, seed: seeds.a)
        let b = InputFactory.randomArray(count: 1536, seed: seeds.b)

        let g1 = try Vector<Dim1536>(a)
        let g2 = try Vector<Dim1536>(b)

        let h = HammingDistance()

            try Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(h.distance(g1, g2)) } }
            var r = try Harness.measure(name: "dist.hamming.1536.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(h.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in Correctness.scalar(current: h.distance(g1, g2), reference: DoubleRef.hamming(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(1536) {
            cases.append(BenchmarkCase("dist.minkowski.1536.generic", boundary: Boundaries.record("dist.minkowski.1536.generic", options)) {
        let seeds = InputSeeds.distance(dim: 1536, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 1536, seed: seeds.a)
        let b = InputFactory.randomArray(count: 1536, seed: seeds.b)

        let g1 = try Vector<Dim1536>(a)
        let g2 = try Vector<Dim1536>(b)

        let mk = MinkowskiDistance(p: 3.0)

            try Harness.warmup { withExtendedLifetime((g1,g2)) { blackHole(mk.distance(g1, g2)) } }
            var r = try Harness.measure(name: "dist.minkowski.1536.generic", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((g1,g2)) { blackHole(mk.distance(g1, g2)) } }
            let corr = g1.withUnsafeBufferPointer { ab in g2.withUnsafeBufferPointer { bb in Correctness.scalar(current: mk.distance(g1, g2), reference: DoubleRef.minkowski(ab, bb, p: 3.0)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(1536) {
            cases.append(BenchmarkCase("dist.euclidean.1536.optimized", boundary: Boundaries.record("dist.euclidean.1536.optimized",
                options)) {
        let seeds = InputSeeds.distance(dim: 1536, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 1536, seed: seeds.a)
        let b = InputFactory.randomArray(count: 1536, seed: seeds.b)

        let o1 = try Vector1536Optimized(a)
        let o2 = try Vector1536Optimized(b)

        let e = EuclideanDistance()

            try Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(e.distance(o1, o2)) } }
            var r = try Harness.measure(name: "dist.euclidean.1536.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(e.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in Correctness.scalar(current: e.distance(o1, o2), reference: DoubleRef.euclid(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(1536) {
            cases.append(BenchmarkCase("dist.cosine.1536.optimized", boundary: Boundaries.record("dist.cosine.1536.optimized", options)) {
        let seeds = InputSeeds.distance(dim: 1536, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 1536, seed: seeds.a)
        let b = InputFactory.randomArray(count: 1536, seed: seeds.b)

        let o1 = try Vector1536Optimized(a)
        let o2 = try Vector1536Optimized(b)

        let c = CosineDistance()

            try Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(c.distance(o1, o2)) } }
            var r = try Harness.measure(name: "dist.cosine.1536.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(c.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in Correctness.scalar(current: c.distance(o1, o2), reference: DoubleRef.cosineDist(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(1536) {
            cases.append(BenchmarkCase("dist.manhattan.1536.optimized", boundary: Boundaries.record("dist.manhattan.1536.optimized",
                options)) {
        let seeds = InputSeeds.distance(dim: 1536, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 1536, seed: seeds.a)
        let b = InputFactory.randomArray(count: 1536, seed: seeds.b)

        let o1 = try Vector1536Optimized(a)
        let o2 = try Vector1536Optimized(b)

        let m = ManhattanDistance()

            try Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(m.distance(o1, o2)) } }
            var r = try Harness.measure(name: "dist.manhattan.1536.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(m.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in Correctness.scalar(current: m.distance(o1, o2), reference: DoubleRef.manhattan(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(1536) {
            cases.append(BenchmarkCase("dist.dot.1536.optimized", boundary: Boundaries.record("dist.dot.1536.optimized", options)) {
        let seeds = InputSeeds.distance(dim: 1536, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 1536, seed: seeds.a)
        let b = InputFactory.randomArray(count: 1536, seed: seeds.b)

        let o1 = try Vector1536Optimized(a)
        let o2 = try Vector1536Optimized(b)

        let d = DotProductDistance()

            try Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(d.distance(o1, o2)) } }
            var r = try Harness.measure(name: "dist.dot.1536.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(d.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in let ref = -DoubleRef.dot(ab,
                bb); return Correctness.scalar(current: d.distance(o1, o2), reference: ref) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(1536) {
            cases.append(BenchmarkCase("dist.chebyshev.1536.optimized", boundary: Boundaries.record("dist.chebyshev.1536.optimized",
                options)) {
        let seeds = InputSeeds.distance(dim: 1536, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 1536, seed: seeds.a)
        let b = InputFactory.randomArray(count: 1536, seed: seeds.b)

        let o1 = try Vector1536Optimized(a)
        let o2 = try Vector1536Optimized(b)

        let ch = ChebyshevDistance()

            try Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(ch.distance(o1, o2)) } }
            var r = try Harness.measure(name: "dist.chebyshev.1536.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(ch.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in Correctness.scalar(current: ch.distance(o1, o2), reference: DoubleRef.chebyshev(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(1536) {
            cases.append(BenchmarkCase("dist.hamming.1536.optimized", boundary: Boundaries.record("dist.hamming.1536.optimized", options)) {
        let seeds = InputSeeds.distance(dim: 1536, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 1536, seed: seeds.a)
        let b = InputFactory.randomArray(count: 1536, seed: seeds.b)

        let o1 = try Vector1536Optimized(a)
        let o2 = try Vector1536Optimized(b)

        let h = HammingDistance()

            try Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(h.distance(o1, o2)) } }
            var r = try Harness.measure(name: "dist.hamming.1536.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(h.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in Correctness.scalar(current: h.distance(o1, o2), reference: DoubleRef.hamming(ab, bb)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        if options.dims.contains(1536) {
            cases.append(BenchmarkCase("dist.minkowski.1536.optimized", boundary: Boundaries.record("dist.minkowski.1536.optimized",
                options)) {
        let seeds = InputSeeds.distance(dim: 1536, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 1536, seed: seeds.a)
        let b = InputFactory.randomArray(count: 1536, seed: seeds.b)

        let o1 = try Vector1536Optimized(a)
        let o2 = try Vector1536Optimized(b)

        let mk = MinkowskiDistance(p: 3.0)

            try Harness.warmup { withExtendedLifetime((o1,o2)) { blackHole(mk.distance(o1, o2)) } }
            var r = try Harness.measure(name: "dist.minkowski.1536.optimized", minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                samples: options.samples) { withExtendedLifetime((o1,o2)) { blackHole(mk.distance(o1, o2)) } }
            let corr = o1.withUnsafeBufferPointer { ab in o2.withUnsafeBufferPointer { bb in Correctness.scalar(current: mk.distance(o1, o2), reference: DoubleRef.minkowski(ab, bb, p: 3.0)) } }
            r = BenchResult(name: r.name, iterations: r.iterations, totalNanoseconds: r.totalNanoseconds, unitCount: r.unitCount, samples: r.samples,
                meanNsPerOp: r.meanNsPerOp, medianNsPerOp: r.medianNsPerOp, p90NsPerOp: r.p90NsPerOp, stddevNsPerOp: r.stddevNsPerOp, rsdPercent: r.rsdPercent, correctness: corr)
            return [r]
                    })
        }
        return cases
    }
}
