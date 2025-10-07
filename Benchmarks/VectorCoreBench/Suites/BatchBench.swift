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

    private static func providers(forDim dim: Int) -> [(mode: String, provider: CPUComputeProvider)] {
        // Tune automatic thresholds per dimension so N≈1000 flips to parallel
        let autoThreshold: Int
        switch dim {
        case 512, 768: autoThreshold = 800
        case 1536: autoThreshold = 800
        default: autoThreshold = 50_000 // fallback conservative
        }
        let automatic = CPUComputeProvider(mode: .automatic, parallelizationThreshold: autoThreshold)
        return [
            ("sequential", .sequential),
            ("parallel", .parallel),
            ("automatic", automatic)
        ]
    }

    private static func abCompareEnabled() -> Bool {
        // Enable A/B compare between euclidean (sqrt) and euclidean2 (squared) when env var set to 1 or by default
        let env = ProcessInfo.processInfo.environment["VC_BATCH_AB"]
        if let env, env == "0" { return false }
        return true
    }

    private static func soaEnabled() -> Bool {
        // Enable SoA (Structure-of-Arrays) kernels when env var set to 1
        let env = ProcessInfo.processInfo.environment["VC_SOA"]
        return env == "1"
    }

    private static func mixedPrecisionEnabled() -> Bool {
        // Enable FP16 mixed-precision kernels when env var set to 1
        let env = ProcessInfo.processInfo.environment["VC_MIXED_PRECISION"]
        return env == "1"
    }

    // Batch sizes are profile-driven (see CLIOptions.batchNs)

    private static func run512(_ options: CLIOptions) async -> [BenchResult] {
        // Prepare inputs (deterministic)
        let qArr = InputFactory.randomArray(count: 512, seed: 512_777)
        let queryG = try! Vector<Dim512>(qArr)
        let queryO = try! Vector512Optimized(qArr)

        var all: [BenchResult] = []
        for n in options.batchNs {
            let baseSeed = UInt64(512_100_000 + n)
            let candsArr: [[Float]] = (0..<n).map { i in
                InputFactory.randomArray(count: 512, seed: baseSeed &+ UInt64(i))
            }
            let candsG: [Vector<Dim512>] = candsArr.map { try! Vector<Dim512>($0) }
            let candsO: [Vector512Optimized] = candsArr.map { try! Vector512Optimized($0) }

            for (mode, provider) in providers(forDim: 512) {
                // Generic
                var labelG = "batch.euclidean2.512.N\(n).generic.\(mode)"
                var resG = await runBatchGeneric(name: labelG, query: queryG, candidates: candsG, provider: provider, options: options, minChunk: 256)
                all.append(contentsOf: resG)

                // Optimized
                var labelO = "batch.euclidean2.512.N\(n).optimized.\(mode)"
                var resO = await runBatchOptimized(name: labelO, query: queryO, candidates: candsO, provider: provider, options: options, minChunk: 256)
                all.append(contentsOf: resO)

                if abCompareEnabled() {
                    // Euclidean with sqrt for A/B
                    labelG = "batch.euclidean.512.N\(n).generic.\(mode)"
                    resG = await runBatchGenericEuclid(name: labelG, query: queryG, candidates: candsG, provider: provider, options: options, minChunk: 256)
                    all.append(contentsOf: resG)

                    labelO = "batch.euclidean.512.N\(n).optimized.\(mode)"
                    resO = await runBatchOptimizedEuclid(name: labelO, query: queryO, candidates: candsO, provider: provider, options: options, minChunk: 256)
                    all.append(contentsOf: resO)
                }

                // Cosine fused (optimized)
                let labelCF = "batch.cosine.512.N\(n).optimized-fused.\(mode)"
                let fused = await runBatchOptimizedCosineFused(name: labelCF, query: queryO, candidates: candsO, provider: provider, options: options, minChunk: 256)
                all.append(contentsOf: fused)

                // Cosine pre-normalized (optimized)
                let qNorm = (try? queryO.normalized().get()) ?? queryO
                let candsNorm512: [Vector512Optimized] = candsO.map { (try? $0.normalized().get()) ?? $0 }
                let labelCP = "batch.cosine.512.N\(n).optimized-preNorm.\(mode)"
                let pren = await runBatchOptimizedCosinePreNorm(name: labelCP, query: qNorm, candidates: candsNorm512, provider: provider, options: options, minChunk: 256)
                all.append(contentsOf: pren)
            }

            // SoA (Structure-of-Arrays) benchmarks when enabled
            if soaEnabled() && BatchKernels_SoA.shouldUseSoA(candidateCount: n, dimension: 512) {
                let labelSoA = "batch.euclidean2.512.N\(n).optimized-soa"
                let soaResult = await runBatchSoA512(name: labelSoA, query: queryO, candidates: candsO, options: options)
                all.append(contentsOf: soaResult)
            }

            // FP16 Mixed-Precision benchmarks when enabled
            if mixedPrecisionEnabled() && MixedPrecisionKernels.shouldUseMixedPrecision(candidateCount: n, dimension: 512) {
                let labelFP16 = "batch.euclidean2.512.N\(n).optimized-fp16"
                let fp16Result = await runBatchFP16_512(name: labelFP16, query: queryO, candidates: candsO, options: options)
                all.append(contentsOf: fp16Result)

                let labelCosineFP16 = "batch.cosine.512.N\(n).optimized-fp16"
                let cosineFP16Result = await runBatchCosineFP16_512(name: labelCosineFP16, query: queryO, candidates: candsO, options: options)
                all.append(contentsOf: cosineFP16Result)
            }
        }
        return all
    }

    private static func run768(_ options: CLIOptions) async -> [BenchResult] {
        let qArr = InputFactory.randomArray(count: 768, seed: 768_777)
        let queryG = try! Vector<Dim768>(qArr)
        let queryO = try! Vector768Optimized(qArr)

        var all: [BenchResult] = []
        for n in options.batchNs {
            let baseSeed = UInt64(768_100_000 + n)
            let candsArr: [[Float]] = (0..<n).map { i in
                InputFactory.randomArray(count: 768, seed: baseSeed &+ UInt64(i))
            }
            let candsG: [Vector<Dim768>] = candsArr.map { try! Vector<Dim768>($0) }
            let candsO: [Vector768Optimized] = candsArr.map { try! Vector768Optimized($0) }

            for (mode, provider) in providers(forDim: 768) {
                let labelG = "batch.euclidean2.768.N\(n).generic.\(mode)"
                let resG = await runBatchGeneric(name: labelG, query: queryG, candidates: candsG, provider: provider, options: options, minChunk: 256)
                all.append(contentsOf: resG)

                let labelO = "batch.euclidean2.768.N\(n).optimized.\(mode)"
                let resO = await runBatchOptimized(name: labelO, query: queryO, candidates: candsO, provider: provider, options: options, minChunk: 256)
                all.append(contentsOf: resO)

                if abCompareEnabled() {
                    let eG = await runBatchGenericEuclid(name: "batch.euclidean.768.N\(n).generic.\(mode)", query: queryG, candidates: candsG, provider: provider, options: options, minChunk: 256)
                    all.append(contentsOf: eG)
                    let eO = await runBatchOptimizedEuclid(name: "batch.euclidean.768.N\(n).optimized.\(mode)", query: queryO, candidates: candsO, provider: provider, options: options, minChunk: 256)
                    all.append(contentsOf: eO)
                }

                // Cosine fused (optimized)
                let labelCF = "batch.cosine.768.N\(n).optimized-fused.\(mode)"
                let fused = await runBatchOptimizedCosineFused(name: labelCF, query: queryO, candidates: candsO, provider: provider, options: options, minChunk: 256)
                all.append(contentsOf: fused)

                // Cosine pre-normalized (optimized)
                let qNorm = (try? queryO.normalized().get()) ?? queryO
                let candsNorm768: [Vector768Optimized] = candsO.map { (try? $0.normalized().get()) ?? $0 }
                let labelCP = "batch.cosine.768.N\(n).optimized-preNorm.\(mode)"
                let pren = await runBatchOptimizedCosinePreNorm(name: labelCP, query: qNorm, candidates: candsNorm768, provider: provider, options: options, minChunk: 256)
                all.append(contentsOf: pren)
            }

            // SoA (Structure-of-Arrays) benchmarks when enabled
            if soaEnabled() && BatchKernels_SoA.shouldUseSoA(candidateCount: n, dimension: 768) {
                let labelSoA = "batch.euclidean2.768.N\(n).optimized-soa"
                let soaResult = await runBatchSoA768(name: labelSoA, query: queryO, candidates: candsO, options: options)
                all.append(contentsOf: soaResult)
            }

            // FP16 Mixed-Precision benchmarks when enabled
            if mixedPrecisionEnabled() && MixedPrecisionKernels.shouldUseMixedPrecision(candidateCount: n, dimension: 768) {
                let labelFP16 = "batch.euclidean2.768.N\(n).optimized-fp16"
                let fp16Result = await runBatchFP16_768(name: labelFP16, query: queryO, candidates: candsO, options: options)
                all.append(contentsOf: fp16Result)

                let labelCosineFP16 = "batch.cosine.768.N\(n).optimized-fp16"
                let cosineFP16Result = await runBatchCosineFP16_768(name: labelCosineFP16, query: queryO, candidates: candsO, options: options)
                all.append(contentsOf: cosineFP16Result)
            }
        }
        return all
    }

    private static func run1536(_ options: CLIOptions) async -> [BenchResult] {
        let qArr = InputFactory.randomArray(count: 1536, seed: 1_536_777)
        let queryG = try! Vector<Dim1536>(qArr)
        let queryO = try! Vector1536Optimized(qArr)

        var all: [BenchResult] = []
        for n in options.batchNs {
            let baseSeed = UInt64(1_536_100_000 + n)
            let candsArr: [[Float]] = (0..<n).map { i in
                InputFactory.randomArray(count: 1536, seed: baseSeed &+ UInt64(i))
            }
            let candsG: [Vector<Dim1536>] = candsArr.map { try! Vector<Dim1536>($0) }
            let candsO: [Vector1536Optimized] = candsArr.map { try! Vector1536Optimized($0) }

            for (mode, provider) in providers(forDim: 1536) {
                let labelG = "batch.euclidean2.1536.N\(n).generic.\(mode)"
                let resG = await runBatchGeneric(name: labelG, query: queryG, candidates: candsG, provider: provider, options: options, minChunk: 512)
                all.append(contentsOf: resG)

                let labelO = "batch.euclidean2.1536.N\(n).optimized.\(mode)"
                let resO = await runBatchOptimized(name: labelO, query: queryO, candidates: candsO, provider: provider, options: options, minChunk: 512)
                all.append(contentsOf: resO)

                if abCompareEnabled() {
                    let eG = await runBatchGenericEuclid(name: "batch.euclidean.1536.N\(n).generic.\(mode)", query: queryG, candidates: candsG, provider: provider, options: options, minChunk: 512)
                    all.append(contentsOf: eG)
                    let eO = await runBatchOptimizedEuclid(name: "batch.euclidean.1536.N\(n).optimized.\(mode)", query: queryO, candidates: candsO, provider: provider, options: options, minChunk: 512)
                    all.append(contentsOf: eO)
                }

                // Cosine fused (optimized)
                let labelCF = "batch.cosine.1536.N\(n).optimized-fused.\(mode)"
                let fused = await runBatchOptimizedCosineFused(name: labelCF, query: queryO, candidates: candsO, provider: provider, options: options, minChunk: 512)
                all.append(contentsOf: fused)

                // Cosine pre-normalized (optimized)
                let qNorm = (try? queryO.normalized().get()) ?? queryO
                let candsNorm1536: [Vector1536Optimized] = candsO.map { (try? $0.normalized().get()) ?? $0 }
                let labelCP = "batch.cosine.1536.N\(n).optimized-preNorm.\(mode)"
                let pren = await runBatchOptimizedCosinePreNorm(name: labelCP, query: qNorm, candidates: candsNorm1536, provider: provider, options: options, minChunk: 512)
                all.append(contentsOf: pren)
            }

            // SoA (Structure-of-Arrays) benchmarks when enabled
            if soaEnabled() && BatchKernels_SoA.shouldUseSoA(candidateCount: n, dimension: 1536) {
                let labelSoA = "batch.euclidean2.1536.N\(n).optimized-soa"
                let soaResult = await runBatchSoA1536(name: labelSoA, query: queryO, candidates: candsO, options: options)
                all.append(contentsOf: soaResult)
            }

            // FP16 Mixed-Precision benchmarks when enabled
            if mixedPrecisionEnabled() && MixedPrecisionKernels.shouldUseMixedPrecision(candidateCount: n, dimension: 1536) {
                let labelFP16 = "batch.euclidean2.1536.N\(n).optimized-fp16"
                let fp16Result = await runBatchFP16_1536(name: labelFP16, query: queryO, candidates: candsO, options: options)
                all.append(contentsOf: fp16Result)

                let labelCosineFP16 = "batch.cosine.1536.N\(n).optimized-fp16"
                let cosineFP16Result = await runBatchCosineFP16_1536(name: labelCosineFP16, query: queryO, candidates: candsO, options: options)
                all.append(contentsOf: cosineFP16Result)
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
        options: CLIOptions,
        minChunk: Int
    ) async -> [BenchResult] {
        let n = candidates.count
        await Harness.warmupAsync {
            let sum = try? await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                var local: Float = 0
                for i in range { local += query.euclideanDistanceSquared(to: candidates[i]) }
                return local
            } _: { $0 + $1 }
            blackHole(sum ?? 0)
        }
        let res = await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
            let total = try? await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                var local: Float = 0
                for i in range { local += query.euclideanDistanceSquared(to: candidates[i]) }
                return local
            } _: { $0 + $1 }
            blackHole(total ?? 0)
        }
        return [res]
    }

    private static func runBatchOptimized(
        name: String,
        query: Vector512Optimized,
        candidates: [Vector512Optimized],
        provider: CPUComputeProvider,
        options: CLIOptions,
        minChunk: Int
    ) async -> [BenchResult] {
        let n = candidates.count
        await Harness.warmupAsync {
            let sum = try? await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                // Compute distances for this range using blocked kernel into shared buffer segment
                var local: Float = 0
                var tmp = [Float](repeating: 0, count: range.count)
                tmp.withUnsafeMutableBufferPointer { sub in
                    BatchKernels.range_euclid2_512(query: query, candidates: candidates, range: range, out: sub)
                    for j in 0..<sub.count { local += sub[j] }
                }
                return local
            } _: { $0 + $1 }
            blackHole(sum ?? 0)
        }
        let res = await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
            let total = try? await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                var local: Float = 0
                var tmp = [Float](repeating: 0, count: range.count)
                tmp.withUnsafeMutableBufferPointer { sub in
                    BatchKernels.range_euclid2_512(query: query, candidates: candidates, range: range, out: sub)
                    for j in 0..<sub.count { local += sub[j] }
                }
                return local
            } _: { $0 + $1 }
            blackHole(total ?? 0)
        }
        return [res]
    }

    private static func runBatchOptimized(
        name: String,
        query: Vector768Optimized,
        candidates: [Vector768Optimized],
        provider: CPUComputeProvider,
        options: CLIOptions,
        minChunk: Int
    ) async -> [BenchResult] {
        let n = candidates.count
        await Harness.warmupAsync {
            let sum = try? await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                var local: Float = 0
                var tmp = [Float](repeating: 0, count: range.count)
                tmp.withUnsafeMutableBufferPointer { sub in
                    BatchKernels.range_euclid2_768(query: query, candidates: candidates, range: range, out: sub)
                    for j in 0..<sub.count { local += sub[j] }
                }
                return local
            } _: { $0 + $1 }
            blackHole(sum ?? 0)
        }
        let res = await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
            let total = try? await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                var local: Float = 0
                var tmp = [Float](repeating: 0, count: range.count)
                tmp.withUnsafeMutableBufferPointer { sub in
                    BatchKernels.range_euclid2_768(query: query, candidates: candidates, range: range, out: sub)
                    for j in 0..<sub.count { local += sub[j] }
                }
                return local
            } _: { $0 + $1 }
            blackHole(total ?? 0)
        }
        return [res]
    }

    private static func runBatchOptimized(
        name: String,
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized],
        provider: CPUComputeProvider,
        options: CLIOptions,
        minChunk: Int
    ) async -> [BenchResult] {
        let n = candidates.count
        await Harness.warmupAsync {
            let _ = try? await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                var local: Float = 0
                var tmp = [Float](repeating: 0, count: range.count)
                tmp.withUnsafeMutableBufferPointer { sub in
                    BatchKernels.range_euclid2_1536(query: query, candidates: candidates, range: range, out: sub)
                    for j in 0..<sub.count { local += sub[j] }
                }
                return local
            } _: { $0 + $1 }
        }
        let res = await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
            let _ = try? await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                var local: Float = 0
                var tmp = [Float](repeating: 0, count: range.count)
                tmp.withUnsafeMutableBufferPointer { sub in
                    BatchKernels.range_euclid2_1536(query: query, candidates: candidates, range: range, out: sub)
                    for j in 0..<sub.count { local += sub[j] }
                }
                return local
            } _: { $0 + $1 }
        }
        return [res]
    }

    // MARK: - Euclidean (sqrt) A/B helpers

    private static func runBatchGenericEuclid<D: StaticDimension>(
        name: String,
        query: Vector<D>,
        candidates: [Vector<D>],
        provider: CPUComputeProvider,
        options: CLIOptions,
        minChunk: Int
    ) async -> [BenchResult] {
        let n = candidates.count
        await Harness.warmupAsync {
            let sum = try? await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                var local: Float = 0
                for i in range { local += EuclideanDistance().distance(query, candidates[i]) }
                return local
            } _: { $0 + $1 }
            blackHole(sum ?? 0)
        }
        let res = await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
            let total = try? await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                var local: Float = 0
                for i in range { local += EuclideanDistance().distance(query, candidates[i]) }
                return local
            } _: { $0 + $1 }
            blackHole(total ?? 0)
        }
        return [res]
    }

    private static func runBatchOptimizedEuclid(
        name: String,
        query: Vector512Optimized,
        candidates: [Vector512Optimized],
        provider: CPUComputeProvider,
        options: CLIOptions,
        minChunk: Int
    ) async -> [BenchResult] {
        let n = candidates.count
        await Harness.warmupAsync {
            let sum = try? await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                var local: Float = 0
                var tmp = [Float](repeating: 0, count: range.count)
                tmp.withUnsafeMutableBufferPointer { sub in
                    BatchKernels.range_euclid_512(query: query, candidates: candidates, range: range, out: sub)
                    for j in 0..<sub.count { local += sub[j] }
                }
                return local
            } _: { $0 + $1 }
            blackHole(sum ?? 0)
        }
        let res = await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
            let total = try? await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                var local: Float = 0
                var tmp = [Float](repeating: 0, count: range.count)
                tmp.withUnsafeMutableBufferPointer { sub in
                    BatchKernels.range_euclid_512(query: query, candidates: candidates, range: range, out: sub)
                    for j in 0..<sub.count { local += sub[j] }
                }
                return local
            } _: { $0 + $1 }
            blackHole(total ?? 0)
        }
        return [res]
    }

    private static func runBatchOptimizedEuclid(
        name: String,
        query: Vector768Optimized,
        candidates: [Vector768Optimized],
        provider: CPUComputeProvider,
        options: CLIOptions,
        minChunk: Int
    ) async -> [BenchResult] {
        let n = candidates.count
        await Harness.warmupAsync {
            let sum = try? await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                var local: Float = 0
                var tmp = [Float](repeating: 0, count: range.count)
                tmp.withUnsafeMutableBufferPointer { sub in
                    BatchKernels.range_euclid_768(query: query, candidates: candidates, range: range, out: sub)
                    for j in 0..<sub.count { local += sub[j] }
                }
                return local
            } _: { $0 + $1 }
            blackHole(sum ?? 0)
        }
        let res = await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
            let total = try? await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                var local: Float = 0
                var tmp = [Float](repeating: 0, count: range.count)
                tmp.withUnsafeMutableBufferPointer { sub in
                    BatchKernels.range_euclid_768(query: query, candidates: candidates, range: range, out: sub)
                    for j in 0..<sub.count { local += sub[j] }
                }
                return local
            } _: { $0 + $1 }
            blackHole(total ?? 0)
        }
        return [res]
    }

    private static func runBatchOptimizedEuclid(
        name: String,
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized],
        provider: CPUComputeProvider,
        options: CLIOptions,
        minChunk: Int
    ) async -> [BenchResult] {
        let n = candidates.count
        await Harness.warmupAsync {
            let _ = try? await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                var local: Float = 0
                var tmp = [Float](repeating: 0, count: range.count)
                tmp.withUnsafeMutableBufferPointer { sub in
                    BatchKernels.range_euclid_1536(query: query, candidates: candidates, range: range, out: sub)
                    for j in 0..<sub.count { local += sub[j] }
                }
                return local
            } _: { $0 + $1 }
        }
        let res = await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
            let _ = try? await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                var local: Float = 0
                var tmp = [Float](repeating: 0, count: range.count)
                tmp.withUnsafeMutableBufferPointer { sub in
                    BatchKernels.range_euclid_1536(query: query, candidates: candidates, range: range, out: sub)
                    for j in 0..<sub.count { local += sub[j] }
                }
                return local
            } _: { $0 + $1 }
        }
        return [res]
    }

    // MARK: - Cosine (fused & pre-normalized) optimized helpers

    private static func runBatchOptimizedCosineFused(
        name: String,
        query: Vector512Optimized,
        candidates: [Vector512Optimized],
        provider: CPUComputeProvider,
        options: CLIOptions,
        minChunk: Int
    ) async -> [BenchResult] {
        let n = candidates.count
        await Harness.warmupAsync {
            let sum = try? await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                var local: Float = 0
                var tmp = [Float](repeating: 0, count: range.count)
                tmp.withUnsafeMutableBufferPointer { sub in
                    BatchKernels.range_cosine_fused_512(query: query, candidates: candidates, range: range, out: sub)
                    for j in 0..<sub.count { local += sub[j] }
                }
                return local
            } _: { $0 + $1 }
            blackHole(sum ?? 0)
        }
        let res = await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
            let total = try? await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                var local: Float = 0
                var tmp = [Float](repeating: 0, count: range.count)
                tmp.withUnsafeMutableBufferPointer { sub in
                    BatchKernels.range_cosine_fused_512(query: query, candidates: candidates, range: range, out: sub)
                    for j in 0..<sub.count { local += sub[j] }
                }
                return local
            } _: { $0 + $1 }
            blackHole(total ?? 0)
        }
        return [res]
    }

    private static func runBatchOptimizedCosineFused(
        name: String,
        query: Vector768Optimized,
        candidates: [Vector768Optimized],
        provider: CPUComputeProvider,
        options: CLIOptions,
        minChunk: Int
    ) async -> [BenchResult] {
        let n = candidates.count
        await Harness.warmupAsync {
            let sum = try? await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                var local: Float = 0
                var tmp = [Float](repeating: 0, count: range.count)
                tmp.withUnsafeMutableBufferPointer { sub in
                    BatchKernels.range_cosine_fused_768(query: query, candidates: candidates, range: range, out: sub)
                    for j in 0..<sub.count { local += sub[j] }
                }
                return local
            } _: { $0 + $1 }
            blackHole(sum ?? 0)
        }
        let res = await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
            let total = try? await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                var local: Float = 0
                var tmp = [Float](repeating: 0, count: range.count)
                tmp.withUnsafeMutableBufferPointer { sub in
                    BatchKernels.range_cosine_fused_768(query: query, candidates: candidates, range: range, out: sub)
                    for j in 0..<sub.count { local += sub[j] }
                }
                return local
            } _: { $0 + $1 }
            blackHole(total ?? 0)
        }
        return [res]
    }

    private static func runBatchOptimizedCosineFused(
        name: String,
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized],
        provider: CPUComputeProvider,
        options: CLIOptions,
        minChunk: Int
    ) async -> [BenchResult] {
        let n = candidates.count
        var out = [Float](repeating: 0, count: n)
        await Harness.warmupAsync {
            let _ = try? await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                var local: Float = 0
                var tmp = [Float](repeating: 0, count: range.count)
                tmp.withUnsafeMutableBufferPointer { sub in
                    BatchKernels.range_cosine_fused_1536(query: query, candidates: candidates, range: range, out: sub)
                    for j in 0..<sub.count { local += sub[j] }
                }
                return local
            } _: { $0 + $1 }
        }
        let res = await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
            let _ = try? await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                var local: Float = 0
                var tmp = [Float](repeating: 0, count: range.count)
                tmp.withUnsafeMutableBufferPointer { sub in
                    BatchKernels.range_cosine_fused_1536(query: query, candidates: candidates, range: range, out: sub)
                    for j in 0..<sub.count { local += sub[j] }
                }
                return local
            } _: { $0 + $1 }
        }
        return [res]
    }

    private static func runBatchOptimizedCosinePreNorm(
        name: String,
        query: Vector512Optimized,
        candidates: [Vector512Optimized],
        provider: CPUComputeProvider,
        options: CLIOptions,
        minChunk: Int
    ) async -> [BenchResult] {
        let n = candidates.count
        var out = [Float](repeating: 0, count: n)
        await Harness.warmupAsync {
            let sum = try? await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                var local: Float = 0
                var tmp = [Float](repeating: 0, count: range.count)
                tmp.withUnsafeMutableBufferPointer { sub in
                    BatchKernels.range_cosine_preNorm_512(query: query, candidates: candidates, range: range, out: sub)
                    for j in 0..<sub.count { local += sub[j] }
                }
                return local
            } _: { $0 + $1 }
            blackHole(sum ?? 0)
        }
        let res = await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
            let total = try? await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                var local: Float = 0
                var tmp = [Float](repeating: 0, count: range.count)
                tmp.withUnsafeMutableBufferPointer { sub in
                    BatchKernels.range_cosine_preNorm_512(query: query, candidates: candidates, range: range, out: sub)
                    for j in 0..<sub.count { local += sub[j] }
                }
                return local
            } _: { $0 + $1 }
            blackHole(total ?? 0)
        }
        return [res]
    }

    private static func runBatchOptimizedCosinePreNorm(
        name: String,
        query: Vector768Optimized,
        candidates: [Vector768Optimized],
        provider: CPUComputeProvider,
        options: CLIOptions,
        minChunk: Int
    ) async -> [BenchResult] {
        let n = candidates.count
        var out = [Float](repeating: 0, count: n)
        await Harness.warmupAsync {
            let sum = try? await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                var local: Float = 0
                var tmp = [Float](repeating: 0, count: range.count)
                tmp.withUnsafeMutableBufferPointer { sub in
                    BatchKernels.range_cosine_preNorm_768(query: query, candidates: candidates, range: range, out: sub)
                    for j in 0..<sub.count { local += sub[j] }
                }
                return local
            } _: { $0 + $1 }
            blackHole(sum ?? 0)
        }
        let res = await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
            let total = try? await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                var local: Float = 0
                var tmp = [Float](repeating: 0, count: range.count)
                tmp.withUnsafeMutableBufferPointer { sub in
                    BatchKernels.range_cosine_preNorm_768(query: query, candidates: candidates, range: range, out: sub)
                    for j in 0..<sub.count { local += sub[j] }
                }
                return local
            } _: { $0 + $1 }
            blackHole(total ?? 0)
        }
        return [res]
    }

    private static func runBatchOptimizedCosinePreNorm(
        name: String,
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized],
        provider: CPUComputeProvider,
        options: CLIOptions,
        minChunk: Int
    ) async -> [BenchResult] {
        let n = candidates.count
        var out = [Float](repeating: 0, count: n)
        await Harness.warmupAsync {
            let _ = try? await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                var local: Float = 0
                var tmp = [Float](repeating: 0, count: range.count)
                tmp.withUnsafeMutableBufferPointer { sub in
                    BatchKernels.range_cosine_preNorm_1536(query: query, candidates: candidates, range: range, out: sub)
                    for j in 0..<sub.count { local += sub[j] }
                }
                return local
            } _: { $0 + $1 }
        }
        let res = await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
            let _ = try? await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                var local: Float = 0
                var tmp = [Float](repeating: 0, count: range.count)
                tmp.withUnsafeMutableBufferPointer { sub in
                    BatchKernels.range_cosine_preNorm_1536(query: query, candidates: candidates, range: range, out: sub)
                    for j in 0..<sub.count { local += sub[j] }
                }
                return local
            } _: { $0 + $1 }
        }
        return [res]
    }

    // MARK: - SoA (Structure-of-Arrays) Benchmark Functions

    private static func runBatchSoA512(
        name: String,
        query: Vector512Optimized,
        candidates: [Vector512Optimized],
        options: CLIOptions
    ) async -> [BenchResult] {
        let n = candidates.count

        // Build SoA once (this cost is part of the benchmark)
        let soa = SoA<Vector512Optimized>.build(from: candidates)

        // Allocate results buffer once to avoid excessive allocations in warmup loop
        // Use UnsafeMutableBufferPointer directly for @Sendable compatibility
        nonisolated(unsafe) let resultsPtr = UnsafeMutableBufferPointer<Float>.allocate(capacity: n)
        defer { resultsPtr.deallocate() }
        resultsPtr.initialize(repeating: 0)

        await Harness.warmupAsync {
            BatchKernels_SoA.euclid2_512(query: query, soa: soa, out: resultsPtr)
            var sum: Float = 0
            for i in 0..<n { sum += resultsPtr[i] }
            blackHole(sum)
        }

        let res = await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
            BatchKernels_SoA.euclid2_512(query: query, soa: soa, out: resultsPtr)
            var total: Float = 0
            for i in 0..<n { total += resultsPtr[i] }
            blackHole(total)
        }
        return [res]
    }

    private static func runBatchSoA768(
        name: String,
        query: Vector768Optimized,
        candidates: [Vector768Optimized],
        options: CLIOptions
    ) async -> [BenchResult] {
        let n = candidates.count

        // Build SoA once (this cost is part of the benchmark)
        let soa = SoA<Vector768Optimized>.build(from: candidates)

        // Allocate results buffer once to avoid excessive allocations in warmup loop
        // Use UnsafeMutableBufferPointer directly for @Sendable compatibility
        nonisolated(unsafe) let resultsPtr = UnsafeMutableBufferPointer<Float>.allocate(capacity: n)
        defer { resultsPtr.deallocate() }
        resultsPtr.initialize(repeating: 0)

        await Harness.warmupAsync {
            BatchKernels_SoA.euclid2_768(query: query, soa: soa, out: resultsPtr)
            var sum: Float = 0
            for i in 0..<n { sum += resultsPtr[i] }
            blackHole(sum)
        }

        let res = await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
            BatchKernels_SoA.euclid2_768(query: query, soa: soa, out: resultsPtr)
            var total: Float = 0
            for i in 0..<n { total += resultsPtr[i] }
            blackHole(total)
        }
        return [res]
    }

    private static func runBatchSoA1536(
        name: String,
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized],
        options: CLIOptions
    ) async -> [BenchResult] {
        let n = candidates.count

        // Build SoA once (this cost is part of the benchmark)
        let soa = SoA<Vector1536Optimized>.build(from: candidates)

        // Allocate results buffer once to avoid excessive allocations in warmup loop
        // Use UnsafeMutableBufferPointer directly for @Sendable compatibility
        nonisolated(unsafe) let resultsPtr = UnsafeMutableBufferPointer<Float>.allocate(capacity: n)
        defer { resultsPtr.deallocate() }
        resultsPtr.initialize(repeating: 0)

        await Harness.warmupAsync {
            BatchKernels_SoA.euclid2_1536(query: query, soa: soa, out: resultsPtr)
            var sum: Float = 0
            for i in 0..<n { sum += resultsPtr[i] }
            blackHole(sum)
        }

        let res = await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
            BatchKernels_SoA.euclid2_1536(query: query, soa: soa, out: resultsPtr)
            var total: Float = 0
            for i in 0..<n { total += resultsPtr[i] }
            blackHole(total)
        }
        return [res]
    }

    // MARK: - FP16 Mixed-Precision Benchmark Functions

    private static func runBatchFP16_512(
        name: String,
        query: Vector512Optimized,
        candidates: [Vector512Optimized],
        options: CLIOptions
    ) async -> [BenchResult] {
        let n = candidates.count

        // Convert candidates to FP16 once (this cost is part of the benchmark)
        let candidatesFP16 = MixedPrecisionKernels.convertToFP16_512(candidates)

        // Allocate results buffer once to avoid excessive allocations in warmup loop
        // Use UnsafeMutableBufferPointer directly for @Sendable compatibility
        nonisolated(unsafe) let resultsPtr = UnsafeMutableBufferPointer<Float>.allocate(capacity: n)
        defer { resultsPtr.deallocate() }
        resultsPtr.initialize(repeating: 0)

        await Harness.warmupAsync {
            MixedPrecisionKernels.range_euclid2_mixed_512(
                query: query,
                candidatesFP16: candidatesFP16,
                range: 0..<n,
                out: resultsPtr
            )
            var sum: Float = 0
            for i in 0..<n { sum += resultsPtr[i] }
            blackHole(sum)
        }

        let res = await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
            MixedPrecisionKernels.range_euclid2_mixed_512(
                query: query,
                candidatesFP16: candidatesFP16,
                range: 0..<n,
                out: resultsPtr
            )
            var total: Float = 0
            for i in 0..<n { total += resultsPtr[i] }
            blackHole(total)
        }
        return [res]
    }

    private static func runBatchFP16_768(
        name: String,
        query: Vector768Optimized,
        candidates: [Vector768Optimized],
        options: CLIOptions
    ) async -> [BenchResult] {
        let n = candidates.count

        // Convert candidates to FP16 once (this cost is part of the benchmark)
        let candidatesFP16 = MixedPrecisionKernels.convertToFP16_768(candidates)

        // Allocate results buffer once to avoid excessive allocations in warmup loop
        // Use UnsafeMutableBufferPointer directly for @Sendable compatibility
        nonisolated(unsafe) let resultsPtr = UnsafeMutableBufferPointer<Float>.allocate(capacity: n)
        defer { resultsPtr.deallocate() }
        resultsPtr.initialize(repeating: 0)

        await Harness.warmupAsync {
            MixedPrecisionKernels.range_euclid2_mixed_768(
                query: query,
                candidatesFP16: candidatesFP16,
                range: 0..<n,
                out: resultsPtr
            )
            var sum: Float = 0
            for i in 0..<n { sum += resultsPtr[i] }
            blackHole(sum)
        }

        let res = await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
            MixedPrecisionKernels.range_euclid2_mixed_768(
                query: query,
                candidatesFP16: candidatesFP16,
                range: 0..<n,
                out: resultsPtr
            )
            var total: Float = 0
            for i in 0..<n { total += resultsPtr[i] }
            blackHole(total)
        }
        return [res]
    }

    private static func runBatchFP16_1536(
        name: String,
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized],
        options: CLIOptions
    ) async -> [BenchResult] {
        let n = candidates.count

        // Convert candidates to FP16 once (this cost is part of the benchmark)
        let candidatesFP16 = MixedPrecisionKernels.convertToFP16_1536(candidates)

        // Allocate results buffer once to avoid excessive allocations in warmup loop
        // Use UnsafeMutableBufferPointer directly for @Sendable compatibility
        nonisolated(unsafe) let resultsPtr = UnsafeMutableBufferPointer<Float>.allocate(capacity: n)
        defer { resultsPtr.deallocate() }
        resultsPtr.initialize(repeating: 0)


        await Harness.warmupAsync {
                MixedPrecisionKernels.range_euclid2_mixed_1536(

                    query: query,

                    candidatesFP16: candidatesFP16,

                    range: 0..<n,

                    out: resultsPtr

                )

                var sum: Float = 0
            for i in 0..<n { sum += resultsPtr[i] }
            blackHole(sum)
        }

        let res = await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
                MixedPrecisionKernels.range_euclid2_mixed_1536(

                    query: query,

                    candidatesFP16: candidatesFP16,

                    range: 0..<n,

                    out: resultsPtr

                )

                var total: Float = 0
            for i in 0..<n { total += resultsPtr[i] }
            blackHole(total)
        }
        return [res]
    }

    private static func runBatchCosineFP16_512(
        name: String,
        query: Vector512Optimized,
        candidates: [Vector512Optimized],
        options: CLIOptions
    ) async -> [BenchResult] {
        let n = candidates.count

        // Convert candidates to FP16 once (this cost is part of the benchmark)
        let candidatesFP16 = MixedPrecisionKernels.convertToFP16_512(candidates)

        // Allocate results buffer once to avoid excessive allocations in warmup loop
        // Use UnsafeMutableBufferPointer directly for @Sendable compatibility
        nonisolated(unsafe) let resultsPtr = UnsafeMutableBufferPointer<Float>.allocate(capacity: n)
        defer { resultsPtr.deallocate() }
        resultsPtr.initialize(repeating: 0)


        await Harness.warmupAsync {
                MixedPrecisionKernels.range_cosine_mixed_512(

                    query: query,

                    candidatesFP16: candidatesFP16,

                    range: 0..<n,

                    out: resultsPtr

                )

                var sum: Float = 0
            for i in 0..<n { sum += resultsPtr[i] }
            blackHole(sum)
        }

        let res = await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
                MixedPrecisionKernels.range_cosine_mixed_512(

                    query: query,

                    candidatesFP16: candidatesFP16,

                    range: 0..<n,

                    out: resultsPtr

                )

                var total: Float = 0
            for i in 0..<n { total += resultsPtr[i] }
            blackHole(total)
        }
        return [res]
    }

    private static func runBatchCosineFP16_768(
        name: String,
        query: Vector768Optimized,
        candidates: [Vector768Optimized],
        options: CLIOptions
    ) async -> [BenchResult] {
        let n = candidates.count

        // Convert candidates to FP16 once (this cost is part of the benchmark)
        let candidatesFP16 = MixedPrecisionKernels.convertToFP16_768(candidates)

        // Allocate results buffer once to avoid excessive allocations in warmup loop
        // Use UnsafeMutableBufferPointer directly for @Sendable compatibility
        nonisolated(unsafe) let resultsPtr = UnsafeMutableBufferPointer<Float>.allocate(capacity: n)
        defer { resultsPtr.deallocate() }
        resultsPtr.initialize(repeating: 0)


        await Harness.warmupAsync {
                MixedPrecisionKernels.range_cosine_mixed_768(

                    query: query,

                    candidatesFP16: candidatesFP16,

                    range: 0..<n,

                    out: resultsPtr

                )

                var sum: Float = 0
            for i in 0..<n { sum += resultsPtr[i] }
            blackHole(sum)
        }

        let res = await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
                MixedPrecisionKernels.range_cosine_mixed_768(

                    query: query,

                    candidatesFP16: candidatesFP16,

                    range: 0..<n,

                    out: resultsPtr

                )

                var total: Float = 0
            for i in 0..<n { total += resultsPtr[i] }
            blackHole(total)
        }
        return [res]
    }

    private static func runBatchCosineFP16_1536(
        name: String,
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized],
        options: CLIOptions
    ) async -> [BenchResult] {
        let n = candidates.count

        // Convert candidates to FP16 once (this cost is part of the benchmark)
        let candidatesFP16 = MixedPrecisionKernels.convertToFP16_1536(candidates)

        // Allocate results buffer once to avoid excessive allocations in warmup loop
        // Use UnsafeMutableBufferPointer directly for @Sendable compatibility
        nonisolated(unsafe) let resultsPtr = UnsafeMutableBufferPointer<Float>.allocate(capacity: n)
        defer { resultsPtr.deallocate() }
        resultsPtr.initialize(repeating: 0)


        await Harness.warmupAsync {
                MixedPrecisionKernels.range_cosine_mixed_1536(

                    query: query,

                    candidatesFP16: candidatesFP16,

                    range: 0..<n,

                    out: resultsPtr

                )

                var sum: Float = 0
            for i in 0..<n { sum += resultsPtr[i] }
            blackHole(sum)
        }

        let res = await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
                MixedPrecisionKernels.range_cosine_mixed_1536(

                    query: query,

                    candidatesFP16: candidatesFP16,

                    range: 0..<n,

                    out: resultsPtr

                )

                var total: Float = 0
            for i in 0..<n { total += resultsPtr[i] }
            blackHole(total)
        }
        return [res]
    }
}
