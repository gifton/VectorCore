import Foundation
import VectorCoreBenchmarking

// BenchResult moved to VectorCoreBenchmarking library

// MARK: - Stats helpers

private enum Stats {
    static func compute(_ values: [Double]) -> (mean: Double, median: Double, p90: Double, stddev: Double, rsdPct: Double) {
        let n = Double(values.count)
        guard values.count > 0 else { return (0,0,0,0,0) }
        let mean = values.reduce(0, +) / n
        let sorted = values.sorted()
        let mid = values.count / 2
        let median: Double
        if values.count % 2 == 0 {
            median = (sorted[mid - 1] + sorted[mid]) / 2
        } else {
            median = sorted[mid]
        }
        // p90: nearest-rank method
        let rank = Int(ceil(0.90 * n))
        let idx = min(max(rank - 1, 0), sorted.count - 1)
        let p90 = sorted[idx]
        // sample stddev
        var variance: Double = 0
        if values.count > 1 {
            variance = values.map { let d = $0 - mean; return d * d }.reduce(0, +) / (n - 1)
        }
        let stddev = sqrt(variance)
        let rsd = mean != 0 ? (stddev / mean) * 100.0 : 0
        return (mean, median, p90, stddev, rsd)
    }
}

// MARK: - Harness configuration

public enum HarnessConfig {
    /// Default warmup duration used when warmup is enabled.
    public static let defaultWarmupNs: UInt64 = 200_000_000 // 200ms
    /// Minimum floor for target measurement time to avoid extremely short runs.
    public static let minMeasurementNsFloor: UInt64 = 100_000 // 100µs
    /// Fraction of target time used for initial iteration estimate during calibration.
    public static let calibrationTargetFractionNumerator: UInt64 = 8
    public static let calibrationTargetFractionDenominator: UInt64 = 10
}

public struct Harness {
    public static func warmup(_ body: () -> Void, minWarmupNs: UInt64 = HarnessConfig.defaultWarmupNs) {
        let start = Clock.now()
        var iters = 0
        while Clock.now() - start < minWarmupNs {
            body(); iters += 1
        }
        if iters == 0 { body() }
    }

    /// Measure a synchronous benchmark body to a minimum wall-time or fixed repeat count.
    /// - Parameters:
    ///   - warmupNs: Optional warmup duration before calibration/measurement. Default 0 (disabled) to avoid double warmup in existing suites.
    public static func measure(name: String, minTimeSeconds: Double, repeats: Int?, unitCount: Int = 1, samples: Int = 1, warmupNs: UInt64 = 0, _ body: () -> Void) -> BenchResult {

        // Optional built-in warmup (kept off by default for back-compat with explicit warmups in suites)
        if warmupNs > 0 { warmup(body, minWarmupNs: warmupNs) }

        // Single-sample fast path if samples == 1
        if samples <= 1 {
            if let fixed = repeats, fixed > 0 {
                let start = Clock.now()
                for _ in 0..<fixed { body() }
                let end = Clock.now()
                return BenchResult(name: name, iterations: fixed, totalNanoseconds: end - start, unitCount: unitCount, samples: 1)
            }

            // Auto-scale iterations until min time reached
            var iterations = 1
            // Calibrate a single run cost
            var start = Clock.now(); body(); var end = Clock.now()
            let single = max(end - start, 1)

            // Estimate iterations needed to hit ~80% of minNs
            let minNs = UInt64(minTimeSeconds * 1_000_000_000)
            let target = max(minNs, HarnessConfig.minMeasurementNsFloor)
            let est = max(UInt64(1), (target * HarnessConfig.calibrationTargetFractionNumerator) / (single * HarnessConfig.calibrationTargetFractionDenominator))
            iterations = Int(est)

            start = Clock.now()
            for _ in 0..<iterations { body() }
            end = Clock.now()
            var elapsed = end - start

            // If still short, loop more proportionally
            if elapsed < minNs {
                let remain = minNs - elapsed
                let more = Int((UInt64(iterations) * remain) / max(elapsed, 1))
                if more > 0 {
                    start = Clock.now()
                    for _ in 0..<more { body() }
                    end = Clock.now()
                    elapsed += end - start
                    iterations += more
                }
            }

            return BenchResult(name: name, iterations: iterations, totalNanoseconds: elapsed, unitCount: unitCount, samples: 1)
        }

        // Multi-sample aggregation: run N samples and compute stats on nsPerOp per sample
        var totalNS: UInt64 = 0
        var totalIters: Int = 0
        var perSampleNsPerOp: [Double] = []
        perSampleNsPerOp.reserveCapacity(samples)
        for _ in 0..<samples {
            let r = measure(name: name, minTimeSeconds: minTimeSeconds, repeats: repeats, unitCount: unitCount, samples: 1, body)
            totalNS &+= r.totalNanoseconds
            totalIters &+= r.iterations
            let nsPerOp = Double(r.totalNanoseconds) / Double(max(r.iterations, 1))
            perSampleNsPerOp.append(nsPerOp)
        }
        let (mean, median, p90, stddev, rsd) = Stats.compute(perSampleNsPerOp)
        return BenchResult(name: name, iterations: totalIters, totalNanoseconds: totalNS, unitCount: unitCount, samples: samples,
                           meanNsPerOp: mean, medianNsPerOp: median, p90NsPerOp: p90, stddevNsPerOp: stddev, rsdPercent: rsd)
    }

    // MARK: - Async measurement

    public static func warmupAsync(_ body: @Sendable () async -> Void, minWarmupNs: UInt64 = HarnessConfig.defaultWarmupNs) async {
        let start = Clock.now()
        var iters = 0
        while Clock.now() - start < minWarmupNs {
            await body(); iters += 1
        }
        if iters == 0 { await body() }
    }

    /// Measure an async benchmark body with the same timing semantics as the sync variant.
    /// - Parameters:
    ///   - warmupNs: Optional warmup duration before calibration/measurement. Default 0 (disabled).
    public static func measureAsync(name: String, minTimeSeconds: Double, repeats: Int?, unitCount: Int = 1, samples: Int = 1, warmupNs: UInt64 = 0, _ body: @Sendable () async -> Void) async -> BenchResult {

        if warmupNs > 0 { await warmupAsync(body, minWarmupNs: warmupNs) }

        // Single-sample fast path if samples == 1
        if samples <= 1 {
            if let fixed = repeats, fixed > 0 {
                let start = Clock.now()
                for _ in 0..<fixed { await body() }
                let end = Clock.now()
                return BenchResult(name: name, iterations: fixed, totalNanoseconds: end - start, unitCount: unitCount, samples: 1)
            }

            // Auto-scale iterations until min time reached
            var iterations = 1
            // Calibrate a single run cost
            var start = Clock.now(); await body(); var end = Clock.now()
            let single = max(end - start, 1)

            let minNs = UInt64(minTimeSeconds * 1_000_000_000)
            let target = max(minNs, HarnessConfig.minMeasurementNsFloor)
            let est = max(UInt64(1), (target * HarnessConfig.calibrationTargetFractionNumerator) / (single * HarnessConfig.calibrationTargetFractionDenominator))
            iterations = Int(est)

            start = Clock.now()
            for _ in 0..<iterations { await body() }
            end = Clock.now()
            var elapsed = end - start

            if elapsed < minNs {
                let remain = minNs - elapsed
                let more = Int((UInt64(iterations) * remain) / max(elapsed, 1))
                if more > 0 {
                    start = Clock.now()
                    for _ in 0..<more { await body() }
                    end = Clock.now()
                    elapsed += end - start
                    iterations += more
                }
            }

            return BenchResult(name: name, iterations: iterations, totalNanoseconds: elapsed, unitCount: unitCount, samples: 1)
        }

        // Multi-sample aggregation
        var totalNS: UInt64 = 0
        var totalIters: Int = 0
        var perSampleNsPerOp: [Double] = []
        perSampleNsPerOp.reserveCapacity(samples)
        for _ in 0..<samples {
            let r = await measureAsync(name: name, minTimeSeconds: minTimeSeconds, repeats: repeats, unitCount: unitCount, samples: 1, body)
            totalNS &+= r.totalNanoseconds
            totalIters &+= r.iterations
            let nsPerOp = Double(r.totalNanoseconds) / Double(max(r.iterations, 1))
            perSampleNsPerOp.append(nsPerOp)
        }
        let (mean, median, p90, stddev, rsd) = Stats.compute(perSampleNsPerOp)
        return BenchResult(name: name, iterations: totalIters, totalNanoseconds: totalNS, unitCount: unitCount, samples: samples,
                           meanNsPerOp: mean, medianNsPerOp: median, p90NsPerOp: p90, stddevNsPerOp: stddev, rsdPercent: rsd)
    }
}
