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
    /// Calibration invocation counts double until one calibration interval spans at least this many clock ticks, so a body
    /// shorter than one tick is timed over enough invocations for the quantization error (under one tick per interval) to stay
    /// below 1 / minCalibrationTicks of the measured interval (`testSubTickBodyAutoScalesToAnAccurateEstimate`).
    public static let minCalibrationTicks: UInt64 = 10
    /// Hard cap on the invocations of a single calibration interval (2^24). Reaching it is an explicit, actionable error.
    public static let maxCalibrationInvocations = 1 << 24
    /// Recorded in every auto-scaled report (`metadata.harnessCalibration`): the rule that produced the iteration counts, including
    /// both parts of the algorithm: calibration grows geometrically to at least `minCalibrationTicks` ticks, and the loop that tops a
    /// sample up to `--min-time` runs at least that many ticks' worth of invocations at the per-invocation rate observed in the main
    /// timed loop (`Calibration.extensionFloor`).
    ///
    /// A body whose first single-invocation interval is at least `minCalibrationTicks` ticks (about 420 ns at the 42 ns tick of the
    /// cited 125/3 Apple Silicon timebase) calibrates on one invocation exactly as before
    /// (`testSlowBodyCalibratesOnOneInvocationExactlyAsBefore`). Existing default cases whose first single-invocation interval is
    /// shorter now calibrate on more than one invocation and get different iteration counts than under the previous rule.
    ///
    /// A steady-state proxy (time per invocation, not the first interval), measured once on one Apple Silicon host and not verified
    /// by any artifact in this repository, put SOME default cases in these families under about 420 ns: the three optimized dot
    /// cases, 29 of the 42 dist cases, mem alloc and pool.acquire, and noop. The other dot, dist and mem cases, and normalize, matrix
    /// and batch at small sizes, were above it.
    ///
    /// Bodies of 1 to 9 ticks used to be calibrated from a one-tick-quantized reading (inaccurate but accepted); only bodies shorter
    /// than one tick could fail. No version bump: the pre-floor variant was never published.
    public static let calibrationVersion = "V04-calibration-2: geometric growth to >= \(minCalibrationTicks) clock ticks, cap "
        + "\(maxCalibrationInvocations) invocations; extension top-up floored at >= \(minCalibrationTicks) ticks at the observed rate"
}

/// Calibration rule shared by the synchronous and asynchronous harness (see `HarnessConfig.calibrationVersion`).
enum Calibration {
    /// Per-invocation nanoseconds (at least 1) once the interval spans `minCalibrationTicks` ticks; nil to keep growing.
    static func estimate(interval: UInt64, invocations: Int, tick: UInt64, name: String = "body") throws -> UInt64? {
        let (needed, overflow) = tick.multipliedReportingOverflow(by: HarnessConfig.minCalibrationTicks)
        guard !overflow, interval >= needed else { return nil }
        // The later target arithmetic multiplies this estimate by 10; an interval that large is a clock jump, not a measurement.
        guard interval <= UInt64.max / 16 else {
            throw BenchmarkConfigurationError(
                "Unusable benchmark clock interval: \(name) measured \(interval) ns over \(invocations) invocations, which is not a "
                    + "plausible "
                + "duration (the clock jumped); rerun, or pass a fixed --repeats.")
        }
        let count = UInt64(invocations)
        return max(1, interval / count + (interval % count == 0 ? 0 : 1))
    }

    /// Invocations that span at least `minCalibrationTicks` ticks at the observed rate `elapsed / iterations`. If the extension runs
    /// at that rate its interval spans at least `minCalibrationTicks` ticks, so it is not a sub-tick (possibly zero) interval; a
    /// body that speeds up by about `minCalibrationTicks` times or more in the extension is not covered. Pinned with a constant-rate
    /// virtual clock by `testExtensionLoopNeverMeasuresASubTickInterval`.
    static func extensionFloor(iterations: Int, elapsed: UInt64, tick: UInt64) -> Int {
        guard elapsed > 0 else { return iterations }
        let (needed, overflow) = tick.multipliedReportingOverflow(by: HarnessConfig.minCalibrationTicks)
        let (scaled, scaledOverflow) = needed.multipliedReportingOverflow(by: UInt64(iterations))
        guard !overflow, !scaledOverflow else { return iterations }
        return Int(clamping: scaled / elapsed + 1)
    }

    /// The next (doubled) invocation count, or an explicit error once the cap was reached without a measurable interval.
    static func grow(_ invocations: Int, interval: UInt64, tick: UInt64, name: String) throws -> Int {
        guard invocations < HarnessConfig.maxCalibrationInvocations else {
            throw BenchmarkConfigurationError(
                "Unusable benchmark clock interval: \(name) ran \(invocations) invocations in \(interval) ns, still below "
                + "\(HarnessConfig.minCalibrationTicks) clock ticks (\(tick) ns each). The clock does not advance over this body or the "
                + "body does no observable work; give it measurable work or pass a fixed --repeats that spans more than one tick.")
        }
        return invocations * 2
    }

    /// A fixed `--repeats` count is honored exactly; a zero or backwards interval is an actionable failure, never a silent value
    /// (`testExplicitRepeatsOneIsExactlyOneInvocationWithAnActionableMessage` covers the exact count and the zero interval).
    static func fixedInterval(_ start: UInt64, _ end: UInt64, repeats: Int, tick: UInt64) throws -> UInt64 {
        guard end >= start else { throw BenchmarkConfigurationError("Unusable benchmark clock interval: the clock went backwards") }
        guard end > start else {
            throw BenchmarkConfigurationError(
                "Unusable benchmark clock interval: \(repeats) fixed invocation(s) completed within one clock tick (about \(tick) ns). "
                + "--repeats N runs exactly N invocations per sample, so raise --repeats or omit it to auto-scale.")
        }
        return end - start
    }
}

/// Measurement parameters shared by the synchronous and asynchronous cores.
private struct MeasureSpec {
    let name: String
    let minTimeSeconds: Double
    let repeats: Int?
    let unitCount: Int
    let samples: Int
    let warmupNs: UInt64
    let tickNanoseconds: UInt64

    /// One sample, no built-in warmup: the unit the multi-sample loop repeats.
    var single: MeasureSpec {
        MeasureSpec(name: name, minTimeSeconds: minTimeSeconds, repeats: repeats, unitCount: unitCount, samples: 1, warmupNs: 0,
                    tickNanoseconds: tickNanoseconds)
    }
}

struct Harness {
    static func warmup(_ body: () throws -> Void, minWarmupNs: UInt64 = HarnessConfig.defaultWarmupNs, now: () -> UInt64 = Clock.now) throws {
        let recorder = CaseRecorder.current, signals = PhaseSignals(fallbackName: "")
        signals.begin(.warmup)
        defer { signals.end(.warmup) }
        let start = now()
        var iters = 0
        while try BenchmarkArithmetic.duration(start, now(), requirePositive: false) < minWarmupNs {
            try body(); iters = try BenchmarkArithmetic.sum(iters, 1)
        }
        if iters == 0 { try body(); iters = 1 }
        recorder?.recordWarmup(iterations: iters)
    }

    /// Measure a synchronous body with estimated duration or exact fixed repeat count.
    /// - Parameters:
    ///   - warmupNs: Optional warmup duration before calibration/measurement. Default 0 (disabled) to avoid double warmup in existing suites.
    static func measure(name: String, minTimeSeconds: Double, repeats: Int?, unitCount: Int = 1, samples: Int = 1,
                        warmupNs: UInt64 = 0, calibrationTickNanoseconds: UInt64 = ClockResolution.tickNanoseconds,
                        now: () -> UInt64 = Clock.now, _ body: () throws -> Void) throws -> BenchResult {
        let spec = MeasureSpec(name: name, minTimeSeconds: minTimeSeconds, repeats: repeats, unitCount: unitCount, samples: samples,
            warmupNs: warmupNs, tickNanoseconds: calibrationTickNanoseconds)
        let result = try measureCore(spec, now: now, body)
        // Everything the case does after its measurement returns is the validation window.
        CaseRecorder.current?.recordSamples(nsPerInvocation: result.sampleNsPerInvocation, iterations: result.sampleIterations)
        PhaseSignals(fallbackName: name).begin(.validation)
        return result
    }

    private static func measureCore(_ spec: MeasureSpec, now: () -> UInt64, _ body: () throws -> Void) throws -> BenchResult {
        let (name, minTimeSeconds, repeats, unitCount, samples, warmupNs) =
            (spec.name, spec.minTimeSeconds, spec.repeats, spec.unitCount, spec.samples, spec.warmupNs)
        let tick = spec.tickNanoseconds
        let signals = PhaseSignals(fallbackName: name)
        guard samples > 0, unitCount > 0, repeats == nil || repeats! > 0 else { throw BenchmarkConfigurationError("Invalid measurement counts") }
        _ = try BenchmarkArithmetic.nanoseconds(minTimeSeconds)
        _ = try BenchmarkArithmetic.product(samples, MemoryLayout<Double>.stride)
        if let repeats { _ = try BenchmarkArithmetic.product(repeats, samples) }

        // Optional built-in warmup (kept off by default for back-compat with explicit warmups in suites)
        if warmupNs > 0 { try warmup(body, minWarmupNs: warmupNs, now: now) }

        // Single-sample fast path if samples == 1
        if samples == 1 {
            if let fixed = repeats, fixed > 0 {
                signals.begin(.timed)
                let start = now()
                for _ in 0..<fixed { try body() }
                let end = now()
                signals.end(.timed)
                let total = try Calibration.fixedInterval(start, end, repeats: fixed, tick: tick)
                return BenchResult(name: name, iterations: fixed, totalNanoseconds: total, unitCount: unitCount, samples: 1,
                                   sampleNsPerInvocation: [Double(total) / Double(fixed)], sampleIterations: [fixed])
            }

            // Auto-scale iterations until min time reached
            var iterations = 1
            // Calibrate the cost of one invocation: double the invocation count until the interval spans enough clock ticks.
            var start: UInt64 = 0, end: UInt64 = 0
            var single: UInt64 = 0
            var calibrationInvocations = 1
            while true {
                signals.begin(.timed)
                start = now(); for _ in 0..<calibrationInvocations { try body() }; end = now()
                signals.end(.timed)
                let interval = try BenchmarkArithmetic.duration(start, end, requirePositive: false)
                if let estimate = try Calibration.estimate(interval: interval, invocations: calibrationInvocations,
                                                           tick: tick, name: name) { single = estimate; break }
                calibrationInvocations = try Calibration.grow(calibrationInvocations, interval: interval, tick: tick, name: name)
            }

            // Estimate iterations needed to hit ~80% of the target, max(minNs, minMeasurementNsFloor)
            let minNs = try BenchmarkArithmetic.nanoseconds(minTimeSeconds)
            let target = max(minNs, HarnessConfig.minMeasurementNsFloor)
            let est = max(UInt64(1), (try BenchmarkArithmetic.product(target, HarnessConfig.calibrationTargetFractionNumerator)) / (try BenchmarkArithmetic.product(single,
                HarnessConfig.calibrationTargetFractionDenominator)))
            iterations = try BenchmarkArithmetic.count(est)

            signals.begin(.timed)
            start = now()
            for _ in 0..<iterations { try body() }
            end = now()
            signals.end(.timed)
            var elapsed = try BenchmarkArithmetic.duration(start, end)

            // If still short, loop more proportionally
            if elapsed < minNs {
                let remain = minNs - elapsed
                let proportional = try BenchmarkArithmetic.count(BenchmarkArithmetic.product(UInt64(iterations), remain) / elapsed)
                let more = proportional > 0 ? max(proportional, Calibration.extensionFloor(iterations: iterations, elapsed: elapsed,
                    tick: tick)) : 0
                if more > 0 {
                    signals.begin(.timed)
                    start = now()
                    for _ in 0..<more { try body() }
                    end = now()
                    signals.end(.timed)
                    elapsed = try BenchmarkArithmetic.sum(elapsed, BenchmarkArithmetic.duration(start, end))
                    iterations = try BenchmarkArithmetic.sum(iterations, more)
                }
            }

            return BenchResult(name: name, iterations: iterations, totalNanoseconds: elapsed, unitCount: unitCount, samples: 1,
                               sampleNsPerInvocation: [Double(elapsed) / Double(max(iterations, 1))], sampleIterations: [iterations])
        }

        // Multi-sample aggregation: run N samples and compute stats on nsPerOp per sample
        var totalNS: UInt64 = 0
        var totalIters: Int = 0
        var perSampleNsPerOp: [Double] = []
        var perSampleIterations: [Int] = []
        perSampleNsPerOp.reserveCapacity(samples)
        for _ in 0..<samples {
            let r = try measureCore(spec.single, now: now, body)
            totalNS = try BenchmarkArithmetic.sum(totalNS, r.totalNanoseconds)
            totalIters = try BenchmarkArithmetic.sum(totalIters, r.iterations)
            let nsPerOp = Double(r.totalNanoseconds) / Double(max(r.iterations, 1))
            perSampleNsPerOp.append(nsPerOp)
            perSampleIterations.append(r.iterations)
        }
        let (mean, median, p90, stddev, rsd) = Stats.compute(perSampleNsPerOp)
        return BenchResult(name: name, iterations: totalIters, totalNanoseconds: totalNS, unitCount: unitCount, samples: samples,
                           meanNsPerOp: mean, medianNsPerOp: median, p90NsPerOp: p90, stddevNsPerOp: stddev, rsdPercent: rsd,
                           sampleNsPerInvocation: perSampleNsPerOp, sampleIterations: perSampleIterations)
    }

    // MARK: - Async measurement

    static func warmupAsync(_ body: @Sendable () async throws -> Void, minWarmupNs: UInt64 = HarnessConfig.defaultWarmupNs, now: @Sendable () -> UInt64 = Clock.now) async throws {
        let recorder = CaseRecorder.current, signals = PhaseSignals(fallbackName: "")
        signals.begin(.warmup)
        defer { signals.end(.warmup) }
        let start = now()
        var iters = 0
        while try BenchmarkArithmetic.duration(start, now(), requirePositive: false) < minWarmupNs {
            try await body(); iters = try BenchmarkArithmetic.sum(iters, 1)
        }
        if iters == 0 { try await body(); iters = 1 }
        recorder?.recordWarmup(iterations: iters)
    }

    /// Measure an async benchmark body with the same timing semantics as the sync variant.
    /// - Parameters:
    ///   - warmupNs: Optional warmup duration before calibration/measurement. Default 0 (disabled).
    static func measureAsync(name: String, minTimeSeconds: Double, repeats: Int?, unitCount: Int = 1, samples: Int = 1, warmupNs: UInt64 = 0,
                             calibrationTickNanoseconds: UInt64 = ClockResolution.tickNanoseconds,
                             now: @Sendable () -> UInt64 = Clock.now,
                             _ body: @Sendable () async throws -> Void) async throws -> BenchResult {
        let spec = MeasureSpec(name: name, minTimeSeconds: minTimeSeconds, repeats: repeats, unitCount: unitCount, samples: samples,
            warmupNs: warmupNs, tickNanoseconds: calibrationTickNanoseconds)
        let result = try await measureAsyncCore(spec, now: now, body)
        CaseRecorder.current?.recordSamples(nsPerInvocation: result.sampleNsPerInvocation, iterations: result.sampleIterations)
        PhaseSignals(fallbackName: name).begin(.validation)
        return result
    }

    private static func measureAsyncCore(_ spec: MeasureSpec, now: @Sendable () -> UInt64,
                                         _ body: @Sendable () async throws -> Void) async throws -> BenchResult {
        let (name, minTimeSeconds, repeats, unitCount, samples, warmupNs) =
            (spec.name, spec.minTimeSeconds, spec.repeats, spec.unitCount, spec.samples, spec.warmupNs)
        let tick = spec.tickNanoseconds
        let signals = PhaseSignals(fallbackName: name)
        guard samples > 0, unitCount > 0, repeats == nil || repeats! > 0 else { throw BenchmarkConfigurationError("Invalid measurement counts") }
        _ = try BenchmarkArithmetic.nanoseconds(minTimeSeconds)
        _ = try BenchmarkArithmetic.product(samples, MemoryLayout<Double>.stride)
        if let repeats { _ = try BenchmarkArithmetic.product(repeats, samples) }

        if warmupNs > 0 { try await warmupAsync(body, minWarmupNs: warmupNs, now: now) }

        // Single-sample fast path if samples == 1
        if samples == 1 {
            if let fixed = repeats, fixed > 0 {
                signals.begin(.timed)
                let start = now()
                for _ in 0..<fixed { try await body() }
                let end = now()
                signals.end(.timed)
                let total = try Calibration.fixedInterval(start, end, repeats: fixed, tick: tick)
                return BenchResult(name: name, iterations: fixed, totalNanoseconds: total, unitCount: unitCount, samples: 1,
                                   sampleNsPerInvocation: [Double(total) / Double(fixed)], sampleIterations: [fixed])
            }

            // Auto-scale iterations until min time reached
            var iterations = 1
            // Calibrate the cost of one invocation: double the invocation count until the interval spans enough clock ticks.
            var start: UInt64 = 0, end: UInt64 = 0
            var single: UInt64 = 0
            var calibrationInvocations = 1
            while true {
                signals.begin(.timed)
                start = now(); for _ in 0..<calibrationInvocations { try await body() }; end = now()
                signals.end(.timed)
                let interval = try BenchmarkArithmetic.duration(start, end, requirePositive: false)
                if let estimate = try Calibration.estimate(interval: interval, invocations: calibrationInvocations,
                                                           tick: tick, name: name) { single = estimate; break }
                calibrationInvocations = try Calibration.grow(calibrationInvocations, interval: interval, tick: tick, name: name)
            }

            let minNs = try BenchmarkArithmetic.nanoseconds(minTimeSeconds)
            let target = max(minNs, HarnessConfig.minMeasurementNsFloor)
            let est = max(UInt64(1), (try BenchmarkArithmetic.product(target, HarnessConfig.calibrationTargetFractionNumerator)) / (try BenchmarkArithmetic.product(single,
                HarnessConfig.calibrationTargetFractionDenominator)))
            iterations = try BenchmarkArithmetic.count(est)

            signals.begin(.timed)
            start = now()
            for _ in 0..<iterations { try await body() }
            end = now()
            signals.end(.timed)
            var elapsed = try BenchmarkArithmetic.duration(start, end)

            if elapsed < minNs {
                let remain = minNs - elapsed
                let proportional = try BenchmarkArithmetic.count(BenchmarkArithmetic.product(UInt64(iterations), remain) / elapsed)
                let more = proportional > 0 ? max(proportional, Calibration.extensionFloor(iterations: iterations, elapsed: elapsed,
                    tick: tick)) : 0
                if more > 0 {
                    signals.begin(.timed)
                    start = now()
                    for _ in 0..<more { try await body() }
                    end = now()
                    signals.end(.timed)
                    elapsed = try BenchmarkArithmetic.sum(elapsed, BenchmarkArithmetic.duration(start, end))
                    iterations = try BenchmarkArithmetic.sum(iterations, more)
                }
            }

            return BenchResult(name: name, iterations: iterations, totalNanoseconds: elapsed, unitCount: unitCount, samples: 1,
                               sampleNsPerInvocation: [Double(elapsed) / Double(max(iterations, 1))], sampleIterations: [iterations])
        }

        // Multi-sample aggregation
        var totalNS: UInt64 = 0
        var totalIters: Int = 0
        var perSampleNsPerOp: [Double] = []
        var perSampleIterations: [Int] = []
        perSampleNsPerOp.reserveCapacity(samples)
        for _ in 0..<samples {
            let r = try await measureAsyncCore(spec.single, now: now, body)
            totalNS = try BenchmarkArithmetic.sum(totalNS, r.totalNanoseconds)
            totalIters = try BenchmarkArithmetic.sum(totalIters, r.iterations)
            let nsPerOp = Double(r.totalNanoseconds) / Double(max(r.iterations, 1))
            perSampleNsPerOp.append(nsPerOp)
            perSampleIterations.append(r.iterations)
        }
        let (mean, median, p90, stddev, rsd) = Stats.compute(perSampleNsPerOp)
        return BenchResult(name: name, iterations: totalIters, totalNanoseconds: totalNS, unitCount: unitCount, samples: samples,
                           meanNsPerOp: mean, medianNsPerOp: median, p90NsPerOp: p90, stddevNsPerOp: stddev, rsdPercent: rsd,
                           sampleNsPerInvocation: perSampleNsPerOp, sampleIterations: perSampleIterations)
    }
}
