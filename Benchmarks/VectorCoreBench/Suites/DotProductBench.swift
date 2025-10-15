import Foundation
import VectorCore
import VectorCoreBenchmarking

struct DotProductBench: BenchmarkSuite {
    static let name = "dot"

    static func run(options: CLIOptions, progress: ProgressReporter) async -> [BenchResult] {
        var results: [BenchResult] = []

        // Pre-count total cases for progress reporting
        var allCaseNames: [String] = []
        for dim in options.dims {
            switch dim {
            case 512:
                allCaseNames += ["dot.512.generic", "dot.512.optimized"]
            case 768:
                allCaseNames += ["dot.768.generic", "dot.768.optimized"]
            case 1536:
                allCaseNames += ["dot.1536.generic", "dot.1536.optimized"]
            default: break
            }
        }
        // Filter to only cases that will actually run
        let casesToRun = allCaseNames.filter { Filters.shouldRun(name: $0, options: options) }
        let totalCases = casesToRun.count
        var currentIndex = 0

        for dim in options.dims {
            switch dim {
            case 512:
                results += bench512(options, progress: progress, totalCases: totalCases, currentIndex: &currentIndex)
            case 768:
                results += bench768(options, progress: progress, totalCases: totalCases, currentIndex: &currentIndex)
            case 1536:
                results += bench1536(options, progress: progress, totalCases: totalCases, currentIndex: &currentIndex)
            default:
                fputs("[dot] unsupported dimension: \(dim)\n", stderr)
            }
        }

        return results
    }

    // MARK: - Per-dimension benches

    private static func bench512(_ options: CLIOptions, progress: ProgressReporter, totalCases: Int, currentIndex: inout Int) -> [BenchResult] {
        // Pre-generate inputs (deterministic via centralized seeds)
        let seeds = InputSeeds.dot(dim: 512, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 512, seed: seeds.a)
        let b = InputFactory.randomArray(count: 512, seed: seeds.b)

        // Generic
        let g1 = try! Vector<Dim512>(a)
        let g2 = try! Vector<Dim512>(b)
        let nameG = "dot.512.generic"
        guard Filters.shouldRun(name: nameG, options: options) else { return [] }

        let caseStart = Date()
        progress.caseStarted(suite: name, name: nameG, index: currentIndex, total: totalCases)

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

        let caseDuration = Date().timeIntervalSince(caseStart) * 1000.0
        progress.caseCompleted(suite: name, name: nameG, index: currentIndex, total: totalCases, durationMs: caseDuration)
        currentIndex += 1

        // Optimized
        let o1 = try! Vector512Optimized(a)
        let o2 = try! Vector512Optimized(b)
        let nameO = "dot.512.optimized"
        guard Filters.shouldRun(name: nameO, options: options) else { return [rg] }

        let caseStart2 = Date()
        progress.caseStarted(suite: name, name: nameO, index: currentIndex, total: totalCases)

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

        let caseDuration2 = Date().timeIntervalSince(caseStart2) * 1000.0
        progress.caseCompleted(suite: name, name: nameO, index: currentIndex, total: totalCases, durationMs: caseDuration2)
        currentIndex += 1

        return [rg, ro]
    }

    private static func bench768(_ options: CLIOptions, progress: ProgressReporter, totalCases: Int, currentIndex: inout Int) -> [BenchResult] {
        let seeds = InputSeeds.dot(dim: 768, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 768, seed: seeds.a)
        let b = InputFactory.randomArray(count: 768, seed: seeds.b)

        let g1 = try! Vector<Dim768>(a)
        let g2 = try! Vector<Dim768>(b)
        let nameG768 = "dot.768.generic"
        guard Filters.shouldRun(name: nameG768, options: options) else { return [] }

        let caseStart = Date()
        progress.caseStarted(suite: name, name: nameG768, index: currentIndex, total: totalCases)

        Harness.warmup {
            withExtendedLifetime((g1,g2)) { let v = g1.dotProduct(g2); blackHole(v) }
        }
        let rg = Harness.measure(name: nameG768, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
            withExtendedLifetime((g1,g2)) { let v = g1.dotProduct(g2); blackHole(v) }
        }

        let caseDuration = Date().timeIntervalSince(caseStart) * 1000.0
        progress.caseCompleted(suite: name, name: nameG768, index: currentIndex, total: totalCases, durationMs: caseDuration)
        currentIndex += 1

        let o1 = try! Vector768Optimized(a)
        let o2 = try! Vector768Optimized(b)
        let nameO768 = "dot.768.optimized"
        if !Filters.shouldRun(name: nameO768, options: options) { return [rg] }

        let caseStart2 = Date()
        progress.caseStarted(suite: name, name: nameO768, index: currentIndex, total: totalCases)

        Harness.warmup { withExtendedLifetime((o1,o2)) { let v = o1.dotProduct(o2); blackHole(v) } }
        let ro = Harness.measure(name: nameO768, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
            withExtendedLifetime((o1,o2)) { let v = o1.dotProduct(o2); blackHole(v) }
        }

        let caseDuration2 = Date().timeIntervalSince(caseStart2) * 1000.0
        progress.caseCompleted(suite: name, name: nameO768, index: currentIndex, total: totalCases, durationMs: caseDuration2)
        currentIndex += 1

        return [rg, ro]
    }

    private static func bench1536(_ options: CLIOptions, progress: ProgressReporter, totalCases: Int, currentIndex: inout Int) -> [BenchResult] {
        let seeds = InputSeeds.dot(dim: 1536, runSeed: options.runSeed)
        let a = InputFactory.randomArray(count: 1536, seed: seeds.a)
        let b = InputFactory.randomArray(count: 1536, seed: seeds.b)

        let g1 = try! Vector<Dim1536>(a)
        let g2 = try! Vector<Dim1536>(b)
        let nameG1536 = "dot.1536.generic"
        guard Filters.shouldRun(name: nameG1536, options: options) else { return [] }

        let caseStart = Date()
        progress.caseStarted(suite: name, name: nameG1536, index: currentIndex, total: totalCases)

        Harness.warmup { withExtendedLifetime((g1,g2)) { let v = g1.dotProduct(g2); blackHole(v) } }
        let rg = Harness.measure(name: nameG1536, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
            withExtendedLifetime((g1,g2)) { let v = g1.dotProduct(g2); blackHole(v) }
        }

        let caseDuration = Date().timeIntervalSince(caseStart) * 1000.0
        progress.caseCompleted(suite: name, name: nameG1536, index: currentIndex, total: totalCases, durationMs: caseDuration)
        currentIndex += 1

        let o1 = try! Vector1536Optimized(a)
        let o2 = try! Vector1536Optimized(b)
        let nameO1536 = "dot.1536.optimized"
        if !Filters.shouldRun(name: nameO1536, options: options) { return [rg] }

        let caseStart2 = Date()
        progress.caseStarted(suite: name, name: nameO1536, index: currentIndex, total: totalCases)

        Harness.warmup { withExtendedLifetime((o1,o2)) { let v = o1.dotProduct(o2); blackHole(v) } }
        let ro = Harness.measure(name: nameO1536, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
            withExtendedLifetime((o1,o2)) { let v = o1.dotProduct(o2); blackHole(v) }
        }

        let caseDuration2 = Date().timeIntervalSince(caseStart2) * 1000.0
        progress.caseCompleted(suite: name, name: nameO1536, index: currentIndex, total: totalCases, durationMs: caseDuration2)
        currentIndex += 1

        return [rg, ro]
    }
}
