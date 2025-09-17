import Foundation
import VectorCore

struct BatchBench: BenchmarkSuite {
    static let name = "batch"

    static func run(options: CLIOptions) async -> [BenchResult] {
        var results: [BenchResult] = []
        for dim in options.dims {
            switch dim {
            case 512: results += await run512(options)
            case 768: results += await run768(options)
            case 1536: results += await run1536(options)
            default: fputs("[batch] unsupported dimension: \(dim)\n", stderr)
            }
        }
        return results
    }

    private static func providers() -> [(mode: String, provider: CPUComputeProvider)] {
        [
            ("sequential", .sequential),
            ("parallel", .parallel),
            ("automatic", .automatic)
        ]
    }

    private static let counts: [Int] = [100, 1_000, 10_000]

    private static func run512(_ options: CLIOptions) async -> [BenchResult] {
        // Prepare inputs (deterministic)
        let qArr = InputFactory.randomArray(count: 512, seed: 512_777)
        let queryG = try! Vector<Dim512>(qArr)
        let queryO = try! Vector512Optimized(qArr)

        var all: [BenchResult] = []
        for n in counts {
            let baseSeed = UInt64(512_100_000 + n)
            let candsArr: [[Float]] = (0..<n).map { i in
                InputFactory.randomArray(count: 512, seed: baseSeed &+ UInt64(i))
            }
            let candsG: [Vector<Dim512>] = candsArr.map { try! Vector<Dim512>($0) }
            let candsO: [Vector512Optimized] = candsArr.map { try! Vector512Optimized($0) }

            for (mode, provider) in providers() {
                // Generic
                let labelG = "batch.euclidean.512.N\(n).generic.\(mode)"
                let resG = await runBatchGeneric(name: labelG, query: queryG, candidates: candsG, provider: provider, options: options)
                all.append(contentsOf: resG)

                // Optimized
                let labelO = "batch.euclidean.512.N\(n).optimized.\(mode)"
                let resO = await runBatchOptimized(name: labelO, query: queryO, candidates: candsO, provider: provider, options: options)
                all.append(contentsOf: resO)
            }
        }
        return all
    }

    private static func run768(_ options: CLIOptions) async -> [BenchResult] {
        let qArr = InputFactory.randomArray(count: 768, seed: 768_777)
        let queryG = try! Vector<Dim768>(qArr)
        let queryO = try! Vector768Optimized(qArr)

        var all: [BenchResult] = []
        for n in counts {
            let baseSeed = UInt64(768_100_000 + n)
            let candsArr: [[Float]] = (0..<n).map { i in
                InputFactory.randomArray(count: 768, seed: baseSeed &+ UInt64(i))
            }
            let candsG: [Vector<Dim768>] = candsArr.map { try! Vector<Dim768>($0) }
            let candsO: [Vector768Optimized] = candsArr.map { try! Vector768Optimized($0) }

            for (mode, provider) in providers() {
                let labelG = "batch.euclidean.768.N\(n).generic.\(mode)"
                let resG = await runBatchGeneric(name: labelG, query: queryG, candidates: candsG, provider: provider, options: options)
                all.append(contentsOf: resG)

                let labelO = "batch.euclidean.768.N\(n).optimized.\(mode)"
                let resO = await runBatchOptimized(name: labelO, query: queryO, candidates: candsO, provider: provider, options: options)
                all.append(contentsOf: resO)
            }
        }
        return all
    }

    private static func run1536(_ options: CLIOptions) async -> [BenchResult] {
        let qArr = InputFactory.randomArray(count: 1536, seed: 1_536_777)
        let queryG = try! Vector<Dim1536>(qArr)
        let queryO = try! Vector1536Optimized(qArr)

        var all: [BenchResult] = []
        for n in counts {
            let baseSeed = UInt64(1_536_100_000 + n)
            let candsArr: [[Float]] = (0..<n).map { i in
                InputFactory.randomArray(count: 1536, seed: baseSeed &+ UInt64(i))
            }
            let candsG: [Vector<Dim1536>] = candsArr.map { try! Vector<Dim1536>($0) }
            let candsO: [Vector1536Optimized] = candsArr.map { try! Vector1536Optimized($0) }

            for (mode, provider) in providers() {
                let labelG = "batch.euclidean.1536.N\(n).generic.\(mode)"
                let resG = await runBatchGeneric(name: labelG, query: queryG, candidates: candsG, provider: provider, options: options)
                all.append(contentsOf: resG)

                let labelO = "batch.euclidean.1536.N\(n).optimized.\(mode)"
                let resO = await runBatchOptimized(name: labelO, query: queryO, candidates: candsO, provider: provider, options: options)
                all.append(contentsOf: resO)
            }
        }
        return all
    }

    // MARK: - Helpers (generic & optimized)

    private static func runBatchGeneric<D: StaticDimension>(
        name: String,
        query: Vector<D>,
        candidates: [Vector<D>],
        provider: CPUComputeProvider,
        options: CLIOptions
    ) async -> [BenchResult] {
        let n = candidates.count
        await Harness.warmupAsync {
            let arr = try? await provider.parallelExecute(items: 0..<n) { i in
                EuclideanDistance().distance(query, candidates[i])
            }
            var sum: Float = 0
            for v in arr ?? [] { sum += v }
            blackHole(sum)
        }
        let res = await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n) {
            let arr = try? await provider.parallelExecute(items: 0..<n) { i in
                EuclideanDistance().distance(query, candidates[i])
            }
            var sum: Float = 0
            for v in arr ?? [] { sum += v }
            blackHole(sum)
        }
        return [res]
    }

    private static func runBatchOptimized(
        name: String,
        query: Vector512Optimized,
        candidates: [Vector512Optimized],
        provider: CPUComputeProvider,
        options: CLIOptions
    ) async -> [BenchResult] {
        let n = candidates.count
        await Harness.warmupAsync {
            let arr = try? await provider.parallelExecute(items: 0..<n) { i in
                EuclideanDistance().distance(query, candidates[i])
            }
            var sum: Float = 0
            for v in arr ?? [] { sum += v }
            blackHole(sum)
        }
        let res = await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n) {
            let arr = try? await provider.parallelExecute(items: 0..<n) { i in
                EuclideanDistance().distance(query, candidates[i])
            }
            var sum: Float = 0
            for v in arr ?? [] { sum += v }
            blackHole(sum)
        }
        return [res]
    }

    private static func runBatchOptimized(
        name: String,
        query: Vector768Optimized,
        candidates: [Vector768Optimized],
        provider: CPUComputeProvider,
        options: CLIOptions
    ) async -> [BenchResult] {
        let n = candidates.count
        await Harness.warmupAsync {
            let arr = try? await provider.parallelExecute(items: 0..<n) { i in
                EuclideanDistance().distance(query, candidates[i])
            }
            var sum: Float = 0
            for v in arr ?? [] { sum += v }
            blackHole(sum)
        }
        let res = await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n) {
            let arr = try? await provider.parallelExecute(items: 0..<n) { i in
                EuclideanDistance().distance(query, candidates[i])
            }
            var sum: Float = 0
            for v in arr ?? [] { sum += v }
            blackHole(sum)
        }
        return [res]
    }

    private static func runBatchOptimized(
        name: String,
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized],
        provider: CPUComputeProvider,
        options: CLIOptions
    ) async -> [BenchResult] {
        let n = candidates.count
        await Harness.warmupAsync {
            let _ = try? await provider.parallelExecute(items: 0..<n) { i in
                EuclideanDistance().distance(query, candidates[i])
            }
        }
        let res = await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n) {
            let _ = try? await provider.parallelExecute(items: 0..<n) { i in
                EuclideanDistance().distance(query, candidates[i])
            }
        }
        return [res]
    }
}
