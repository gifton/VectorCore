import Foundation

public struct BenchResult: Sendable {
    public let name: String
    public let iterations: Int
    public let totalNanoseconds: UInt64
    public let unitCount: Int // work units per iteration (e.g., candidates per batch). Default 1.
    public init(name: String, iterations: Int, totalNanoseconds: UInt64, unitCount: Int = 1) {
        self.name = name
        self.iterations = iterations
        self.totalNanoseconds = totalNanoseconds
        self.unitCount = unitCount
    }
}

public struct Harness {
    public static func warmup(_ body: () -> Void, minWarmupNs: UInt64 = 200_000_000) {
        let start = Clock.now()
        var iters = 0
        while Clock.now() - start < minWarmupNs {
            body(); iters += 1
        }
        if iters == 0 { body() }
    }

    public static func measure(name: String, minTimeSeconds: Double, repeats: Int?, unitCount: Int = 1, _ body: () -> Void) -> BenchResult {
        let minNs = UInt64(minTimeSeconds * 1_000_000_000)

        if let fixed = repeats, fixed > 0 {
            let start = Clock.now()
            for _ in 0..<fixed { body() }
            let end = Clock.now()
            return BenchResult(name: name, iterations: fixed, totalNanoseconds: end - start, unitCount: unitCount)
        }

        // Auto-scale iterations until min time reached
        var iterations = 1
        // Calibrate a single run cost
        var start = Clock.now(); body(); var end = Clock.now()
        let single = max(end - start, 1)

        // Estimate iterations needed to hit ~80% of minNs
        let target = max(minNs, 100_000)
        let est = max(UInt64(1), (target * 8) / (single * 10))
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

        return BenchResult(name: name, iterations: iterations, totalNanoseconds: elapsed, unitCount: unitCount)
    }

    // MARK: - Async measurement

    public static func warmupAsync(_ body: @Sendable () async -> Void, minWarmupNs: UInt64 = 200_000_000) async {
        let start = Clock.now()
        var iters = 0
        while Clock.now() - start < minWarmupNs {
            await body(); iters += 1
        }
        if iters == 0 { await body() }
    }

    public static func measureAsync(name: String, minTimeSeconds: Double, repeats: Int?, unitCount: Int = 1, _ body: @Sendable () async -> Void) async -> BenchResult {
        let minNs = UInt64(minTimeSeconds * 1_000_000_000)

        if let fixed = repeats, fixed > 0 {
            let start = Clock.now()
            for _ in 0..<fixed { await body() }
            let end = Clock.now()
            return BenchResult(name: name, iterations: fixed, totalNanoseconds: end - start, unitCount: unitCount)
        }

        // Auto-scale iterations until min time reached
        var iterations = 1
        // Calibrate a single run cost
        var start = Clock.now(); await body(); var end = Clock.now()
        let single = max(end - start, 1)

        let target = max(minNs, 100_000)
        let est = max(UInt64(1), (target * 8) / (single * 10))
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

        return BenchResult(name: name, iterations: iterations, totalNanoseconds: elapsed, unitCount: unitCount)
    }
}
