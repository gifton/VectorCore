import Foundation
import VectorCore
import VectorCoreBenchmarking

struct NormalizationBench: BenchmarkSuite {
    static let name = "normalize"
    // Each case defines one `body` that both the warmup and the measurement run, so they cannot diverge.
    // success: every coordinate of the produced unit vector is folded (u64-bitfold-sink-v1).
    // zeroFail: the failure outcome is the result; the sink sees the failure branch (error-outcome-sink-v1).
    static func catalog(options: ValidatedOptions, progress: BenchmarkProgressReporter) -> [BenchmarkCase] {
        catalog512(options: options) + catalog768(options: options) + catalog1536(options: options)
    }

    private static func catalog512(options: ValidatedOptions) -> [BenchmarkCase] {
        guard options.dims.contains(512) else { return [] }
        var cases: [BenchmarkCase] = []
        cases.append(BenchmarkCase("normalize.success.512.generic",
                                   boundary: Boundaries.record("normalize.success.512.generic", options)) {
            let vector = Vector<Dim512>(repeating: 2.0)
            let body = {
                withExtendedLifetime(vector) {
                    if let unit = try? vector.normalized().get() {
                        blackHole(unit.withUnsafeBufferPointer { OutputFold.fold($0) })
                    } else {
                        blackHole(UInt64.zero)
                    }
                }
            }
            try Harness.warmup(body)
            return [try Harness.measure(name: "normalize.success.512.generic", minTimeSeconds: options.minTimeSeconds,
                                        repeats: options.repeats, samples: options.samples, body)]
        })
        cases.append(BenchmarkCase("normalize.success.512.optimized",
                                   boundary: Boundaries.record("normalize.success.512.optimized", options)) {
            let vector = Vector512Optimized(repeating: 2.0)
            let body = {
                withExtendedLifetime(vector) {
                    if let unit = try? vector.normalized().get() {
                        blackHole(unit.withUnsafeBufferPointer { OutputFold.fold($0) })
                    } else {
                        blackHole(UInt64.zero)
                    }
                }
            }
            try Harness.warmup(body)
            return [try Harness.measure(name: "normalize.success.512.optimized", minTimeSeconds: options.minTimeSeconds,
                                        repeats: options.repeats, samples: options.samples, body)]
        })
        cases.append(BenchmarkCase("normalize.zeroFail.512.generic",
                                   boundary: Boundaries.record("normalize.zeroFail.512.generic", options)) {
            let vector = Vector<Dim512>()
            let body = {
                withExtendedLifetime(vector) {
                    if let unit = try? vector.normalized().get() {
                        blackHole(unit[0])
                    } else {
                        blackHole(Float.zero)
                    }
                }
            }
            try Harness.warmup(body)
            return [try Harness.measure(name: "normalize.zeroFail.512.generic", minTimeSeconds: options.minTimeSeconds,
                                        repeats: options.repeats, samples: options.samples, body)]
        })
        cases.append(BenchmarkCase("normalize.zeroFail.512.optimized",
                                   boundary: Boundaries.record("normalize.zeroFail.512.optimized", options)) {
            let vector = Vector512Optimized()
            let body = {
                withExtendedLifetime(vector) {
                    if let unit = try? vector.normalized().get() {
                        blackHole(unit[0])
                    } else {
                        blackHole(Float.zero)
                    }
                }
            }
            try Harness.warmup(body)
            return [try Harness.measure(name: "normalize.zeroFail.512.optimized", minTimeSeconds: options.minTimeSeconds,
                                        repeats: options.repeats, samples: options.samples, body)]
        })
        return cases
    }

    private static func catalog768(options: ValidatedOptions) -> [BenchmarkCase] {
        guard options.dims.contains(768) else { return [] }
        var cases: [BenchmarkCase] = []
        cases.append(BenchmarkCase("normalize.success.768.generic",
                                   boundary: Boundaries.record("normalize.success.768.generic", options)) {
            let vector = Vector<Dim768>(repeating: 2.0)
            let body = {
                withExtendedLifetime(vector) {
                    if let unit = try? vector.normalized().get() {
                        blackHole(unit.withUnsafeBufferPointer { OutputFold.fold($0) })
                    } else {
                        blackHole(UInt64.zero)
                    }
                }
            }
            try Harness.warmup(body)
            return [try Harness.measure(name: "normalize.success.768.generic", minTimeSeconds: options.minTimeSeconds,
                                        repeats: options.repeats, samples: options.samples, body)]
        })
        cases.append(BenchmarkCase("normalize.success.768.optimized",
                                   boundary: Boundaries.record("normalize.success.768.optimized", options)) {
            let vector = Vector768Optimized(repeating: 2.0)
            let body = {
                withExtendedLifetime(vector) {
                    if let unit = try? vector.normalized().get() {
                        blackHole(unit.withUnsafeBufferPointer { OutputFold.fold($0) })
                    } else {
                        blackHole(UInt64.zero)
                    }
                }
            }
            try Harness.warmup(body)
            return [try Harness.measure(name: "normalize.success.768.optimized", minTimeSeconds: options.minTimeSeconds,
                                        repeats: options.repeats, samples: options.samples, body)]
        })
        cases.append(BenchmarkCase("normalize.zeroFail.768.generic",
                                   boundary: Boundaries.record("normalize.zeroFail.768.generic", options)) {
            let vector = Vector<Dim768>()
            let body = {
                withExtendedLifetime(vector) {
                    if let unit = try? vector.normalized().get() {
                        blackHole(unit[0])
                    } else {
                        blackHole(Float.zero)
                    }
                }
            }
            try Harness.warmup(body)
            return [try Harness.measure(name: "normalize.zeroFail.768.generic", minTimeSeconds: options.minTimeSeconds,
                                        repeats: options.repeats, samples: options.samples, body)]
        })
        cases.append(BenchmarkCase("normalize.zeroFail.768.optimized",
                                   boundary: Boundaries.record("normalize.zeroFail.768.optimized", options)) {
            let vector = Vector768Optimized()
            let body = {
                withExtendedLifetime(vector) {
                    if let unit = try? vector.normalized().get() {
                        blackHole(unit[0])
                    } else {
                        blackHole(Float.zero)
                    }
                }
            }
            try Harness.warmup(body)
            return [try Harness.measure(name: "normalize.zeroFail.768.optimized", minTimeSeconds: options.minTimeSeconds,
                                        repeats: options.repeats, samples: options.samples, body)]
        })
        return cases
    }

    private static func catalog1536(options: ValidatedOptions) -> [BenchmarkCase] {
        guard options.dims.contains(1536) else { return [] }
        var cases: [BenchmarkCase] = []
        cases.append(BenchmarkCase("normalize.success.1536.generic",
                                   boundary: Boundaries.record("normalize.success.1536.generic", options)) {
            let vector = Vector<Dim1536>(repeating: 2.0)
            let body = {
                withExtendedLifetime(vector) {
                    if let unit = try? vector.normalized().get() {
                        blackHole(unit.withUnsafeBufferPointer { OutputFold.fold($0) })
                    } else {
                        blackHole(UInt64.zero)
                    }
                }
            }
            try Harness.warmup(body)
            return [try Harness.measure(name: "normalize.success.1536.generic", minTimeSeconds: options.minTimeSeconds,
                                        repeats: options.repeats, samples: options.samples, body)]
        })
        cases.append(BenchmarkCase("normalize.success.1536.optimized",
                                   boundary: Boundaries.record("normalize.success.1536.optimized", options)) {
            let vector = Vector1536Optimized(repeating: 2.0)
            let body = {
                withExtendedLifetime(vector) {
                    if let unit = try? vector.normalized().get() {
                        blackHole(unit.withUnsafeBufferPointer { OutputFold.fold($0) })
                    } else {
                        blackHole(UInt64.zero)
                    }
                }
            }
            try Harness.warmup(body)
            return [try Harness.measure(name: "normalize.success.1536.optimized", minTimeSeconds: options.minTimeSeconds,
                                        repeats: options.repeats, samples: options.samples, body)]
        })
        cases.append(BenchmarkCase("normalize.zeroFail.1536.generic",
                                   boundary: Boundaries.record("normalize.zeroFail.1536.generic", options)) {
            let vector = Vector<Dim1536>()
            let body = {
                withExtendedLifetime(vector) {
                    if let unit = try? vector.normalized().get() {
                        blackHole(unit[0])
                    } else {
                        blackHole(Float.zero)
                    }
                }
            }
            try Harness.warmup(body)
            return [try Harness.measure(name: "normalize.zeroFail.1536.generic", minTimeSeconds: options.minTimeSeconds,
                                        repeats: options.repeats, samples: options.samples, body)]
        })
        cases.append(BenchmarkCase("normalize.zeroFail.1536.optimized",
                                   boundary: Boundaries.record("normalize.zeroFail.1536.optimized", options)) {
            let vector = Vector1536Optimized()
            let body = {
                withExtendedLifetime(vector) {
                    if let unit = try? vector.normalized().get() {
                        blackHole(unit[0])
                    } else {
                        blackHole(Float.zero)
                    }
                }
            }
            try Harness.warmup(body)
            return [try Harness.measure(name: "normalize.zeroFail.1536.optimized", minTimeSeconds: options.minTimeSeconds,
                                        repeats: options.repeats, samples: options.samples, body)]
        })
        return cases
    }

}
