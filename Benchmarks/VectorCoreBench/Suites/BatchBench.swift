import Foundation
import VectorCore
import VectorCoreBenchmarking

struct BatchBench: BenchmarkSuite {
    static let name = "batch"
    static func catalog(options: ValidatedOptions, progress: BenchmarkProgressReporter) -> [BenchmarkCase] {
        catalog(options: options, provider: nil)
    }
    static func run(options: ValidatedOptions, progress: BenchmarkProgressReporter, provider: (any ComputeProvider)?) async -> [BenchResult] {
        await CaseExecution.run(catalog(options: options, provider: provider), options: options, progress: progress, suite: name)
    }
    private static func providers(forDim dim: Int, injected: (any ComputeProvider)?) -> [(mode: String, provider: any ComputeProvider)] {
        if let injected { return [("injected", injected)] }
        // Automatic-mode threshold per dimension: CPUComputeProvider(.automatic) parallelizes when count >= threshold, so
        // N >= 800 takes the parallel path at 512/768/1536. A benchmark setting, not a measured crossover.
        let autoThreshold: Int
        switch dim {
        case 512, 768: autoThreshold = 800
        case 1536: autoThreshold = 800
        default: autoThreshold = 50_000 // fallback conservative
        }
        let automatic = CPUComputeProvider(mode: .automatic, parallelizationThreshold: autoThreshold)
        return [
            ("sequential", CPUComputeProvider.sequential),
            ("parallel", CPUComputeProvider.parallel),
            ("automatic", automatic)
        ]
    }

    static func catalog(options: ValidatedOptions, provider injected: (any ComputeProvider)?) -> [BenchmarkCase] {
        catalog512(options: options, injected: injected) + catalog768(options: options, injected: injected) + catalog1536(options: options, injected: injected)
    }

    private static func catalog512(options: ValidatedOptions, injected: (any ComputeProvider)?) -> [BenchmarkCase] {
        var cases: [BenchmarkCase] = []
        if options.dims.contains(512) {
            for n in options.batchNs {
                for (mode, provider) in providers(forDim: 512, injected: injected) {
                    if true {
                        let label = "batch.euclidean2.512.N\(n).generic.\(mode)"
                        cases.append(BenchmarkCase(label, boundary: Boundaries.record(label, options)) {
                    let fixture = try BatchFixtures.makeFixture(dim: 512, count: n, runSeed: options.runSeed) { try Vector<Dim512>($0) }
                    let query = fixture.query
                    let candidates = fixture.candidates
                    return await runBatchGeneric(name: label, query: query, candidates: candidates, provider: provider, options: options, minChunk: 256)
                        })
                    }
                    if true {
                        let label = "batch.euclidean2.512.N\(n).optimized.\(mode)"
                        cases.append(BenchmarkCase(label, boundary: Boundaries.record(label, options)) {
                    let fixture = try BatchFixtures.makeFixture(dim: 512, count: n, runSeed: options.runSeed) { try Vector512Optimized($0) }
                    let query = fixture.query
                    let candidates = fixture.candidates
                    return await runBatchOptimized(name: label, query: query, candidates: candidates, provider: provider, options: options, minChunk: 256)
                        })
                    }
                    if options.abCompare == true {
                        let label = "batch.euclidean.512.N\(n).generic.\(mode)"
                        cases.append(BenchmarkCase(label, boundary: Boundaries.record(label, options)) {
                    let fixture = try BatchFixtures.makeFixture(dim: 512, count: n, runSeed: options.runSeed) { try Vector<Dim512>($0) }
                    let query = fixture.query
                    let candidates = fixture.candidates
                    return await runBatchGenericEuclid(name: label, query: query, candidates: candidates, provider: provider, options: options, minChunk: 256)
                        })
                    }
                    if options.abCompare == true {
                        let label = "batch.euclidean.512.N\(n).optimized.\(mode)"
                        cases.append(BenchmarkCase(label, boundary: Boundaries.record(label, options)) {
                    let fixture = try BatchFixtures.makeFixture(dim: 512, count: n, runSeed: options.runSeed) { try Vector512Optimized($0) }
                    let query = fixture.query
                    let candidates = fixture.candidates
                    return await runBatchOptimizedEuclid(name: label, query: query, candidates: candidates, provider: provider, options: options, minChunk: 256)
                        })
                    }
                    if true {
                        let label = "batch.cosine.512.N\(n).optimized-fused.\(mode)"
                        cases.append(BenchmarkCase(label, boundary: Boundaries.record(label, options)) {
                    let fixture = try BatchFixtures.makeFixture(dim: 512, count: n, runSeed: options.runSeed) { try Vector512Optimized($0) }
                    let query = fixture.query
                    let candidates = fixture.candidates
                    return await runBatchOptimizedCosineFused(name: label, query: query, candidates: candidates, provider: provider, options: options, minChunk: 256)
                        })
                    }
                    if true {
                        let label = "batch.cosine.512.N\(n).optimized-preNorm.\(mode)"
                        cases.append(BenchmarkCase(label, boundary: Boundaries.record(label, options)) {
                    let fixture = try BatchFixtures.makeFixture(dim: 512, count: n, runSeed: options.runSeed) { try Vector512Optimized($0) }
                    let query = fixture.query
                    let candidates = fixture.candidates
                    let normalizedQuery = (try? query.normalized().get()) ?? query
                    let normalizedCandidates = candidates.map { (try? $0.normalized().get()) ?? $0 }
                    return await runBatchOptimizedCosinePreNorm(name: label, query: normalizedQuery, candidates: normalizedCandidates, provider: provider, options: options, minChunk: 256)
                        })
                    }
                }
                if options.abOnly != true && options.preferSoA == true && BatchKernels_SoA.shouldUseSoA(candidateCount: n, dimension: 512) {
                    let label = "batch.euclidean2.512.N\(n).optimized-soa"
                    cases.append(BenchmarkCase(label, boundary: Boundaries.record(label, options)) {
                    let fixture = try BatchFixtures.makeFixture(dim: 512, count: n, runSeed: options.runSeed) { try Vector512Optimized($0) }
                    let query = fixture.query
                    let candidates = fixture.candidates
                    return try await runBatchSoA512(name: label, query: query, candidates: candidates, options: options)
                    })
                }
                if options.abOnly != true && options.useMixedPrecision == true && MixedPrecisionKernels.shouldUseMixedPrecision(candidateCount: n, dimension: 512) {
                    let label = "batch.euclidean2.512.N\(n).optimized-fp16"
                    cases.append(BenchmarkCase(label, boundary: Boundaries.record(label, options)) {
                    let fixture = try BatchFixtures.makeFixture(dim: 512, count: n, runSeed: options.runSeed) { try Vector512Optimized($0) }
                    let query = fixture.query
                    let candidates = fixture.candidates
                    return try await runBatchFP16_512(name: label, query: query, candidates: candidates, options: options)
                    })
                }
                if options.abOnly != true && options.useMixedPrecision == true && MixedPrecisionKernels.shouldUseMixedPrecision(candidateCount: n, dimension: 512) {
                    let label = "batch.cosine.512.N\(n).optimized-fp16"
                    cases.append(BenchmarkCase(label, boundary: Boundaries.record(label, options)) {
                    let fixture = try BatchFixtures.makeFixture(dim: 512, count: n, runSeed: options.runSeed) { try Vector512Optimized($0) }
                    let query = fixture.query
                    let candidates = fixture.candidates
                    return try await runBatchCosineFP16_512(name: label, query: query, candidates: candidates, options: options)
                    })
                }
            }
        }
        return cases
    }

    private static func catalog768(options: ValidatedOptions, injected: (any ComputeProvider)?) -> [BenchmarkCase] {
        var cases: [BenchmarkCase] = []
        if options.dims.contains(768) {
            for n in options.batchNs {
                for (mode, provider) in providers(forDim: 768, injected: injected) {
                    if true {
                        let label = "batch.euclidean2.768.N\(n).generic.\(mode)"
                        cases.append(BenchmarkCase(label, boundary: Boundaries.record(label, options)) {
                    let fixture = try BatchFixtures.makeFixture(dim: 768, count: n, runSeed: options.runSeed) { try Vector<Dim768>($0) }
                    let query = fixture.query
                    let candidates = fixture.candidates
                    return await runBatchGeneric(name: label, query: query, candidates: candidates, provider: provider, options: options, minChunk: 256)
                        })
                    }
                    if true {
                        let label = "batch.euclidean2.768.N\(n).optimized.\(mode)"
                        cases.append(BenchmarkCase(label, boundary: Boundaries.record(label, options)) {
                    let fixture = try BatchFixtures.makeFixture(dim: 768, count: n, runSeed: options.runSeed) { try Vector768Optimized($0) }
                    let query = fixture.query
                    let candidates = fixture.candidates
                    return await runBatchOptimized(name: label, query: query, candidates: candidates, provider: provider, options: options, minChunk: 256)
                        })
                    }
                    if options.abCompare == true {
                        let label = "batch.euclidean.768.N\(n).generic.\(mode)"
                        cases.append(BenchmarkCase(label, boundary: Boundaries.record(label, options)) {
                    let fixture = try BatchFixtures.makeFixture(dim: 768, count: n, runSeed: options.runSeed) { try Vector<Dim768>($0) }
                    let query = fixture.query
                    let candidates = fixture.candidates
                    return await runBatchGenericEuclid(name: label, query: query, candidates: candidates, provider: provider, options: options, minChunk: 256)
                        })
                    }
                    if options.abCompare == true {
                        let label = "batch.euclidean.768.N\(n).optimized.\(mode)"
                        cases.append(BenchmarkCase(label, boundary: Boundaries.record(label, options)) {
                    let fixture = try BatchFixtures.makeFixture(dim: 768, count: n, runSeed: options.runSeed) { try Vector768Optimized($0) }
                    let query = fixture.query
                    let candidates = fixture.candidates
                    return await runBatchOptimizedEuclid(name: label, query: query, candidates: candidates, provider: provider, options: options, minChunk: 256)
                        })
                    }
                    if true {
                        let label = "batch.cosine.768.N\(n).optimized-fused.\(mode)"
                        cases.append(BenchmarkCase(label, boundary: Boundaries.record(label, options)) {
                    let fixture = try BatchFixtures.makeFixture(dim: 768, count: n, runSeed: options.runSeed) { try Vector768Optimized($0) }
                    let query = fixture.query
                    let candidates = fixture.candidates
                    return await runBatchOptimizedCosineFused(name: label, query: query, candidates: candidates, provider: provider, options: options, minChunk: 256)
                        })
                    }
                    if true {
                        let label = "batch.cosine.768.N\(n).optimized-preNorm.\(mode)"
                        cases.append(BenchmarkCase(label, boundary: Boundaries.record(label, options)) {
                    let fixture = try BatchFixtures.makeFixture(dim: 768, count: n, runSeed: options.runSeed) { try Vector768Optimized($0) }
                    let query = fixture.query
                    let candidates = fixture.candidates
                    let normalizedQuery = (try? query.normalized().get()) ?? query
                    let normalizedCandidates = candidates.map { (try? $0.normalized().get()) ?? $0 }
                    return await runBatchOptimizedCosinePreNorm(name: label, query: normalizedQuery, candidates: normalizedCandidates, provider: provider, options: options, minChunk: 256)
                        })
                    }
                }
                if options.abOnly != true && options.preferSoA == true && BatchKernels_SoA.shouldUseSoA(candidateCount: n, dimension: 768) {
                    let label = "batch.euclidean2.768.N\(n).optimized-soa"
                    cases.append(BenchmarkCase(label, boundary: Boundaries.record(label, options)) {
                    let fixture = try BatchFixtures.makeFixture(dim: 768, count: n, runSeed: options.runSeed) { try Vector768Optimized($0) }
                    let query = fixture.query
                    let candidates = fixture.candidates
                    return try await runBatchSoA768(name: label, query: query, candidates: candidates, options: options)
                    })
                }
                if options.abOnly != true && options.useMixedPrecision == true && MixedPrecisionKernels.shouldUseMixedPrecision(candidateCount: n, dimension: 768) {
                    let label = "batch.euclidean2.768.N\(n).optimized-fp16"
                    cases.append(BenchmarkCase(label, boundary: Boundaries.record(label, options)) {
                    let fixture = try BatchFixtures.makeFixture(dim: 768, count: n, runSeed: options.runSeed) { try Vector768Optimized($0) }
                    let query = fixture.query
                    let candidates = fixture.candidates
                    return try await runBatchFP16_768(name: label, query: query, candidates: candidates, options: options)
                    })
                }
                if options.abOnly != true && options.useMixedPrecision == true && MixedPrecisionKernels.shouldUseMixedPrecision(candidateCount: n, dimension: 768) {
                    let label = "batch.cosine.768.N\(n).optimized-fp16"
                    cases.append(BenchmarkCase(label, boundary: Boundaries.record(label, options)) {
                    let fixture = try BatchFixtures.makeFixture(dim: 768, count: n, runSeed: options.runSeed) { try Vector768Optimized($0) }
                    let query = fixture.query
                    let candidates = fixture.candidates
                    return try await runBatchCosineFP16_768(name: label, query: query, candidates: candidates, options: options)
                    })
                }
            }
        }
        return cases
    }

    private static func catalog1536(options: ValidatedOptions, injected: (any ComputeProvider)?) -> [BenchmarkCase] {
        var cases: [BenchmarkCase] = []
        if options.dims.contains(1536) {
            for n in options.batchNs {
                for (mode, provider) in providers(forDim: 1536, injected: injected) {
                    if true {
                        let label = "batch.euclidean2.1536.N\(n).generic.\(mode)"
                        cases.append(BenchmarkCase(label, boundary: Boundaries.record(label, options)) {
                    let fixture = try BatchFixtures.makeFixture(dim: 1536, count: n, runSeed: options.runSeed) { try Vector<Dim1536>($0) }
                    let query = fixture.query
                    let candidates = fixture.candidates
                    return await runBatchGeneric(name: label, query: query, candidates: candidates, provider: provider, options: options, minChunk: 512)
                        })
                    }
                    if true {
                        let label = "batch.euclidean2.1536.N\(n).optimized.\(mode)"
                        cases.append(BenchmarkCase(label, boundary: Boundaries.record(label, options)) {
                    let fixture = try BatchFixtures.makeFixture(dim: 1536, count: n,
                        runSeed: options.runSeed) { try Vector1536Optimized($0) }
                    let query = fixture.query
                    let candidates = fixture.candidates
                    return await runBatchOptimized(name: label, query: query, candidates: candidates, provider: provider, options: options, minChunk: 512)
                        })
                    }
                    if options.abCompare == true {
                        let label = "batch.euclidean.1536.N\(n).generic.\(mode)"
                        cases.append(BenchmarkCase(label, boundary: Boundaries.record(label, options)) {
                    let fixture = try BatchFixtures.makeFixture(dim: 1536, count: n, runSeed: options.runSeed) { try Vector<Dim1536>($0) }
                    let query = fixture.query
                    let candidates = fixture.candidates
                    return await runBatchGenericEuclid(name: label, query: query, candidates: candidates, provider: provider, options: options, minChunk: 512)
                        })
                    }
                    if options.abCompare == true {
                        let label = "batch.euclidean.1536.N\(n).optimized.\(mode)"
                        cases.append(BenchmarkCase(label, boundary: Boundaries.record(label, options)) {
                    let fixture = try BatchFixtures.makeFixture(dim: 1536, count: n,
                        runSeed: options.runSeed) { try Vector1536Optimized($0) }
                    let query = fixture.query
                    let candidates = fixture.candidates
                    return await runBatchOptimizedEuclid(name: label, query: query, candidates: candidates, provider: provider, options: options, minChunk: 512)
                        })
                    }
                    if true {
                        let label = "batch.cosine.1536.N\(n).optimized-fused.\(mode)"
                        cases.append(BenchmarkCase(label, boundary: Boundaries.record(label, options)) {
                    let fixture = try BatchFixtures.makeFixture(dim: 1536, count: n,
                        runSeed: options.runSeed) { try Vector1536Optimized($0) }
                    let query = fixture.query
                    let candidates = fixture.candidates
                    return await runBatchOptimizedCosineFused(name: label, query: query, candidates: candidates, provider: provider, options: options, minChunk: 512)
                        })
                    }
                    if true {
                        let label = "batch.cosine.1536.N\(n).optimized-preNorm.\(mode)"
                        cases.append(BenchmarkCase(label, boundary: Boundaries.record(label, options)) {
                    let fixture = try BatchFixtures.makeFixture(dim: 1536, count: n,
                        runSeed: options.runSeed) { try Vector1536Optimized($0) }
                    let query = fixture.query
                    let candidates = fixture.candidates
                    let normalizedQuery = (try? query.normalized().get()) ?? query
                    let normalizedCandidates = candidates.map { (try? $0.normalized().get()) ?? $0 }
                    return await runBatchOptimizedCosinePreNorm(name: label, query: normalizedQuery, candidates: normalizedCandidates, provider: provider, options: options, minChunk: 512)
                        })
                    }
                }
                if options.abOnly != true && options.preferSoA == true && BatchKernels_SoA.shouldUseSoA(candidateCount: n, dimension: 1536) {
                    let label = "batch.euclidean2.1536.N\(n).optimized-soa"
                    cases.append(BenchmarkCase(label, boundary: Boundaries.record(label, options)) {
                    let fixture = try BatchFixtures.makeFixture(dim: 1536, count: n,
                        runSeed: options.runSeed) { try Vector1536Optimized($0) }
                    let query = fixture.query
                    let candidates = fixture.candidates
                    return try await runBatchSoA1536(name: label, query: query, candidates: candidates, options: options)
                    })
                }
                if options.abOnly != true && options.useMixedPrecision == true && MixedPrecisionKernels.shouldUseMixedPrecision(candidateCount: n, dimension: 1536) {
                    let label = "batch.euclidean2.1536.N\(n).optimized-fp16"
                    cases.append(BenchmarkCase(label, boundary: Boundaries.record(label, options)) {
                    let fixture = try BatchFixtures.makeFixture(dim: 1536, count: n,
                        runSeed: options.runSeed) { try Vector1536Optimized($0) }
                    let query = fixture.query
                    let candidates = fixture.candidates
                    return try await runBatchFP16_1536(name: label, query: query, candidates: candidates, options: options)
                    })
                }
                if options.abOnly != true && options.useMixedPrecision == true && MixedPrecisionKernels.shouldUseMixedPrecision(candidateCount: n, dimension: 1536) {
                    let label = "batch.cosine.1536.N\(n).optimized-fp16"
                    cases.append(BenchmarkCase(label, boundary: Boundaries.record(label, options)) {
                    let fixture = try BatchFixtures.makeFixture(dim: 1536, count: n,
                        runSeed: options.runSeed) { try Vector1536Optimized($0) }
                    let query = fixture.query
                    let candidates = fixture.candidates
                    return try await runBatchCosineFP16_1536(name: label, query: query, candidates: candidates, options: options)
                    })
                }
            }
        }
        return cases
    }

    // MARK: - Helpers (generic & optimized)

    private static func runBatchGeneric<D: StaticDimension>(
        name: String,
        query: Vector<D>,
        candidates: [Vector<D>],
        provider: any ComputeProvider,
        options: ValidatedOptions,
        minChunk: Int
    ) async -> [BenchResult] {
        return await BatchExecution.run(name: name) {
            let n = candidates.count
            let refs: [Double] = query.withUnsafeBufferPointer { qb in
                candidates.map { cand in cand.withUnsafeBufferPointer { cb in DoubleRef.euclid2(qb, cb) } }
            }
            let observations = ProviderObservations(reference: refs.reduce(0, +))
            try await Harness.warmupAsync {
                let observedValue = try await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { @Sendable range in
                    var local: Float = 0
                    for i in range { local += query.euclideanDistanceSquared(to: candidates[i]) }
                    return local
                } _: { $0 + $1 }
                observations.observe(observedValue)
                blackHole(observedValue)
            }
            let res = try await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
                let observedValue = try await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { @Sendable range in
                    var local: Float = 0
                    for i in range { local += query.euclideanDistanceSquared(to: candidates[i]) }
                    return local
                } _: { $0 + $1 }
                observations.observe(observedValue)
                blackHole(observedValue)
            }
            // Correctness computation (out-of-band)
            var outputs = [Float](repeating: 0, count: n)
            for i in 0..<n { outputs[i] = query.euclideanDistanceSquared(to: candidates[i]) }
            let stats = Correctness.vector(currents: outputs, references: refs)
            let out = BenchResult(name: res.name, iterations: res.iterations, totalNanoseconds: res.totalNanoseconds, unitCount: res.unitCount, samples: res.samples,
                meanNsPerOp: res.meanNsPerOp, medianNsPerOp: res.medianNsPerOp, p90NsPerOp: res.p90NsPerOp, stddevNsPerOp: res.stddevNsPerOp, rsdPercent: res.rsdPercent,
                correctness: observations.combining(stats))
            return [out]
        }
    }

    private static func runBatchOptimized(
        name: String,
        query: Vector512Optimized,
        candidates: [Vector512Optimized],
        provider: any ComputeProvider,
        options: ValidatedOptions,
        minChunk: Int
    ) async -> [BenchResult] {
        return await BatchExecution.run(name: name) {
            let n = candidates.count
            let refs: [Double] = query.withUnsafeBufferPointer { qb in
                candidates.map { cand in cand.withUnsafeBufferPointer { cb in DoubleRef.euclid2(qb, cb) } }
            }
            let observations = ProviderObservations(reference: refs.reduce(0, +))
            try await Harness.warmupAsync {
                let observedValue = try await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                    // Compute this range's distances with range_euclid2_512 into a per-range [Float] created here, then sum them
                    var local: Float = 0
                    var tmp = [Float](repeating: 0, count: range.count)
                    tmp.withUnsafeMutableBufferPointer { sub in
                        BatchKernels.range_euclid2_512(query: query, candidates: candidates, range: range, out: sub)
                        for j in 0..<sub.count { local += sub[j] }
                    }
                    return local
                } _: { $0 + $1 }
                observations.observe(observedValue)
                blackHole(observedValue)
            }
            let res = try await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
                let observedValue = try await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                    var local: Float = 0
                    var tmp = [Float](repeating: 0, count: range.count)
                    tmp.withUnsafeMutableBufferPointer { sub in
                        BatchKernels.range_euclid2_512(query: query, candidates: candidates, range: range, out: sub)
                        for j in 0..<sub.count { local += sub[j] }
                    }
                    return local
                } _: { $0 + $1 }
                observations.observe(observedValue)
                blackHole(observedValue)
            }
            var outputs = [Float](repeating: 0, count: n)
            outputs.withUnsafeMutableBufferPointer { out in
                BatchKernels.range_euclid2_512(query: query, candidates: candidates, range: 0..<n, out: out)
            }
            let stats = Correctness.vector(currents: outputs, references: refs)
            let out = BenchResult(name: res.name, iterations: res.iterations, totalNanoseconds: res.totalNanoseconds, unitCount: res.unitCount, samples: res.samples,
                meanNsPerOp: res.meanNsPerOp, medianNsPerOp: res.medianNsPerOp, p90NsPerOp: res.p90NsPerOp, stddevNsPerOp: res.stddevNsPerOp, rsdPercent: res.rsdPercent,
                correctness: observations.combining(stats))
            return [out]
        }
    }

    private static func runBatchOptimized(
        name: String,
        query: Vector768Optimized,
        candidates: [Vector768Optimized],
        provider: any ComputeProvider,
        options: ValidatedOptions,
        minChunk: Int
    ) async -> [BenchResult] {
        return await BatchExecution.run(name: name) {
            let n = candidates.count
            let refs: [Double] = query.withUnsafeBufferPointer { qb in
                candidates.map { cand in cand.withUnsafeBufferPointer { cb in DoubleRef.euclid2(qb, cb) } }
            }
            let observations = ProviderObservations(reference: refs.reduce(0, +))
            try await Harness.warmupAsync {
                let observedValue = try await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                    var local: Float = 0
                    var tmp = [Float](repeating: 0, count: range.count)
                    tmp.withUnsafeMutableBufferPointer { sub in
                        BatchKernels.range_euclid2_768(query: query, candidates: candidates, range: range, out: sub)
                        for j in 0..<sub.count { local += sub[j] }
                    }
                    return local
                } _: { $0 + $1 }
                observations.observe(observedValue)
                blackHole(observedValue)
            }
            let res = try await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
                let observedValue = try await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                    var local: Float = 0
                    var tmp = [Float](repeating: 0, count: range.count)
                    tmp.withUnsafeMutableBufferPointer { sub in
                        BatchKernels.range_euclid2_768(query: query, candidates: candidates, range: range, out: sub)
                        for j in 0..<sub.count { local += sub[j] }
                    }
                    return local
                } _: { $0 + $1 }
                observations.observe(observedValue)
                blackHole(observedValue)
            }
            var outputs = [Float](repeating: 0, count: n)
            outputs.withUnsafeMutableBufferPointer { out in
                BatchKernels.range_euclid2_768(query: query, candidates: candidates, range: 0..<n, out: out)
            }
            let stats = Correctness.vector(currents: outputs, references: refs)
            let out = BenchResult(name: res.name, iterations: res.iterations, totalNanoseconds: res.totalNanoseconds, unitCount: res.unitCount, samples: res.samples,
                meanNsPerOp: res.meanNsPerOp, medianNsPerOp: res.medianNsPerOp, p90NsPerOp: res.p90NsPerOp, stddevNsPerOp: res.stddevNsPerOp, rsdPercent: res.rsdPercent,
                correctness: observations.combining(stats))
            return [out]
        }
    }

    private static func runBatchOptimized(
        name: String,
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized],
        provider: any ComputeProvider,
        options: ValidatedOptions,
        minChunk: Int
    ) async -> [BenchResult] {
        return await BatchExecution.run(name: name) {
            let n = candidates.count
            let refs: [Double] = query.withUnsafeBufferPointer { qb in
                candidates.map { cand in cand.withUnsafeBufferPointer { cb in DoubleRef.euclid2(qb, cb) } }
            }
            let observations = ProviderObservations(reference: refs.reduce(0, +))
            try await Harness.warmupAsync {
                let observedValue = try await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                    var local: Float = 0
                    var tmp = [Float](repeating: 0, count: range.count)
                    tmp.withUnsafeMutableBufferPointer { sub in
                        BatchKernels.range_euclid2_1536(query: query, candidates: candidates, range: range, out: sub)
                        for j in 0..<sub.count { local += sub[j] }
                    }
                    return local
                } _: { $0 + $1 }
                observations.observe(observedValue)
                blackHole(observedValue)
            }
            let res = try await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
                let observedValue = try await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                    var local: Float = 0
                    var tmp = [Float](repeating: 0, count: range.count)
                    tmp.withUnsafeMutableBufferPointer { sub in
                        BatchKernels.range_euclid2_1536(query: query, candidates: candidates, range: range, out: sub)
                        for j in 0..<sub.count { local += sub[j] }
                    }
                    return local
                } _: { $0 + $1 }
                observations.observe(observedValue)
                blackHole(observedValue)
            }
            var outputs = [Float](repeating: 0, count: n)
            outputs.withUnsafeMutableBufferPointer { out in
                BatchKernels.range_euclid2_1536(query: query, candidates: candidates, range: 0..<n, out: out)
            }
            let stats = Correctness.vector(currents: outputs, references: refs)
            let out = BenchResult(name: res.name, iterations: res.iterations, totalNanoseconds: res.totalNanoseconds, unitCount: res.unitCount, samples: res.samples,
                meanNsPerOp: res.meanNsPerOp, medianNsPerOp: res.medianNsPerOp, p90NsPerOp: res.p90NsPerOp, stddevNsPerOp: res.stddevNsPerOp, rsdPercent: res.rsdPercent,
                correctness: observations.combining(stats))
            return [out]
        }
    }

    // MARK: - Euclidean (sqrt) A/B helpers

    private static func runBatchGenericEuclid<D: StaticDimension>(
        name: String,
        query: Vector<D>,
        candidates: [Vector<D>],
        provider: any ComputeProvider,
        options: ValidatedOptions,
        minChunk: Int
    ) async -> [BenchResult] {
        return await BatchExecution.run(name: name) {
            let n = candidates.count
            let refs: [Double] = query.withUnsafeBufferPointer { qb in
                candidates.map { cand in cand.withUnsafeBufferPointer { cb in DoubleRef.euclid(qb, cb) } }
            }
            let observations = ProviderObservations(reference: refs.reduce(0, +))
            try await Harness.warmupAsync {
                let observedValue = try await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { @Sendable range in
                    var local: Float = 0
                    for i in range { local += EuclideanDistance().distance(query, candidates[i]) }
                    return local
                } _: { $0 + $1 }
                observations.observe(observedValue)
                blackHole(observedValue)
            }
            let res = try await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
                let observedValue = try await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { @Sendable range in
                    var local: Float = 0
                    for i in range { local += EuclideanDistance().distance(query, candidates[i]) }
                    return local
                } _: { $0 + $1 }
                observations.observe(observedValue)
                blackHole(observedValue)
            }
            var outputs = [Float](repeating: 0, count: n)
            for i in 0..<n { outputs[i] = EuclideanDistance().distance(query, candidates[i]) }
            let stats = Correctness.vector(currents: outputs, references: refs)
            let out = BenchResult(name: res.name, iterations: res.iterations, totalNanoseconds: res.totalNanoseconds, unitCount: res.unitCount, samples: res.samples,
                meanNsPerOp: res.meanNsPerOp, medianNsPerOp: res.medianNsPerOp, p90NsPerOp: res.p90NsPerOp, stddevNsPerOp: res.stddevNsPerOp, rsdPercent: res.rsdPercent,
                correctness: observations.combining(stats))
            return [out]
        }
    }

    private static func runBatchOptimizedEuclid(
        name: String,
        query: Vector512Optimized,
        candidates: [Vector512Optimized],
        provider: any ComputeProvider,
        options: ValidatedOptions,
        minChunk: Int
    ) async -> [BenchResult] {
        return await BatchExecution.run(name: name) {
            let n = candidates.count
            let refs: [Double] = query.withUnsafeBufferPointer { qb in
                candidates.map { cand in cand.withUnsafeBufferPointer { cb in DoubleRef.euclid(qb, cb) } }
            }
            let observations = ProviderObservations(reference: refs.reduce(0, +))
            try await Harness.warmupAsync {
                let observedValue = try await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                    var local: Float = 0
                    var tmp = [Float](repeating: 0, count: range.count)
                    tmp.withUnsafeMutableBufferPointer { sub in
                        BatchKernels.range_euclid_512(query: query, candidates: candidates, range: range, out: sub)
                        for j in 0..<sub.count { local += sub[j] }
                    }
                    return local
                } _: { $0 + $1 }
                observations.observe(observedValue)
                blackHole(observedValue)
            }
            let res = try await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
                let observedValue = try await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                    var local: Float = 0
                    var tmp = [Float](repeating: 0, count: range.count)
                    tmp.withUnsafeMutableBufferPointer { sub in
                        BatchKernels.range_euclid_512(query: query, candidates: candidates, range: range, out: sub)
                        for j in 0..<sub.count { local += sub[j] }
                    }
                    return local
                } _: { $0 + $1 }
                observations.observe(observedValue)
                blackHole(observedValue)
            }
            return [res.withCorrectness(observations.statistics)]
        }
    }

    private static func runBatchOptimizedEuclid(
        name: String,
        query: Vector768Optimized,
        candidates: [Vector768Optimized],
        provider: any ComputeProvider,
        options: ValidatedOptions,
        minChunk: Int
    ) async -> [BenchResult] {
        return await BatchExecution.run(name: name) {
            let n = candidates.count
            let refs: [Double] = query.withUnsafeBufferPointer { qb in
                candidates.map { cand in cand.withUnsafeBufferPointer { cb in DoubleRef.euclid(qb, cb) } }
            }
            let observations = ProviderObservations(reference: refs.reduce(0, +))
            try await Harness.warmupAsync {
                let observedValue = try await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                    var local: Float = 0
                    var tmp = [Float](repeating: 0, count: range.count)
                    tmp.withUnsafeMutableBufferPointer { sub in
                        BatchKernels.range_euclid_768(query: query, candidates: candidates, range: range, out: sub)
                        for j in 0..<sub.count { local += sub[j] }
                    }
                    return local
                } _: { $0 + $1 }
                observations.observe(observedValue)
                blackHole(observedValue)
            }
            let res = try await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
                let observedValue = try await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                    var local: Float = 0
                    var tmp = [Float](repeating: 0, count: range.count)
                    tmp.withUnsafeMutableBufferPointer { sub in
                        BatchKernels.range_euclid_768(query: query, candidates: candidates, range: range, out: sub)
                        for j in 0..<sub.count { local += sub[j] }
                    }
                    return local
                } _: { $0 + $1 }
                observations.observe(observedValue)
                blackHole(observedValue)
            }
            var outputs = [Float](repeating: 0, count: n)
            outputs.withUnsafeMutableBufferPointer { out in
                BatchKernels.range_euclid_768(query: query, candidates: candidates, range: 0..<n, out: out)
            }
            let stats = Correctness.vector(currents: outputs, references: refs)
            let out = BenchResult(name: res.name, iterations: res.iterations, totalNanoseconds: res.totalNanoseconds, unitCount: res.unitCount, samples: res.samples,
                meanNsPerOp: res.meanNsPerOp, medianNsPerOp: res.medianNsPerOp, p90NsPerOp: res.p90NsPerOp, stddevNsPerOp: res.stddevNsPerOp, rsdPercent: res.rsdPercent,
                correctness: observations.combining(stats))
            return [out]
        }
    }

    private static func runBatchOptimizedEuclid(
        name: String,
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized],
        provider: any ComputeProvider,
        options: ValidatedOptions,
        minChunk: Int
    ) async -> [BenchResult] {
        return await BatchExecution.run(name: name) {
            let n = candidates.count
            let refs: [Double] = query.withUnsafeBufferPointer { qb in
                candidates.map { cand in cand.withUnsafeBufferPointer { cb in DoubleRef.euclid(qb, cb) } }
            }
            let observations = ProviderObservations(reference: refs.reduce(0, +))
            try await Harness.warmupAsync {
                let observedValue = try await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                    var local: Float = 0
                    var tmp = [Float](repeating: 0, count: range.count)
                    tmp.withUnsafeMutableBufferPointer { sub in
                        BatchKernels.range_euclid_1536(query: query, candidates: candidates, range: range, out: sub)
                        for j in 0..<sub.count { local += sub[j] }
                    }
                    return local
                } _: { $0 + $1 }
                observations.observe(observedValue)
                blackHole(observedValue)
            }
            let res = try await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
                let observedValue = try await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                    var local: Float = 0
                    var tmp = [Float](repeating: 0, count: range.count)
                    tmp.withUnsafeMutableBufferPointer { sub in
                        BatchKernels.range_euclid_1536(query: query, candidates: candidates, range: range, out: sub)
                        for j in 0..<sub.count { local += sub[j] }
                    }
                    return local
                } _: { $0 + $1 }
                observations.observe(observedValue)
                blackHole(observedValue)
            }
            var outputs = [Float](repeating: 0, count: n)
            outputs.withUnsafeMutableBufferPointer { out in
                BatchKernels.range_euclid_1536(query: query, candidates: candidates, range: 0..<n, out: out)
            }
            let stats = Correctness.vector(currents: outputs, references: refs)
            let out = BenchResult(name: res.name, iterations: res.iterations, totalNanoseconds: res.totalNanoseconds, unitCount: res.unitCount, samples: res.samples,
                meanNsPerOp: res.meanNsPerOp, medianNsPerOp: res.medianNsPerOp, p90NsPerOp: res.p90NsPerOp, stddevNsPerOp: res.stddevNsPerOp, rsdPercent: res.rsdPercent,
                correctness: observations.combining(stats))
            return [out]
        }
    }

    // MARK: - Cosine (fused & pre-normalized) optimized helpers

    private static func runBatchOptimizedCosineFused(
        name: String,
        query: Vector512Optimized,
        candidates: [Vector512Optimized],
        provider: any ComputeProvider,
        options: ValidatedOptions,
        minChunk: Int
    ) async -> [BenchResult] {
        return await BatchExecution.run(name: name) {
            let n = candidates.count
            let refs: [Double] = query.withUnsafeBufferPointer { qb in
                candidates.map { cand in cand.withUnsafeBufferPointer { cb in DoubleRef.cosineDist(qb, cb) } }
            }
            let observations = ProviderObservations(reference: refs.reduce(0, +))
            try await Harness.warmupAsync {
                let observedValue = try await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                    var local: Float = 0
                    var tmp = [Float](repeating: 0, count: range.count)
                    tmp.withUnsafeMutableBufferPointer { sub in
                        BatchKernels.range_cosine_fused_512(query: query, candidates: candidates, range: range, out: sub)
                        for j in 0..<sub.count { local += sub[j] }
                    }
                    return local
                } _: { $0 + $1 }
                observations.observe(observedValue)
                blackHole(observedValue)
            }
            let res = try await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
                let observedValue = try await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                    var local: Float = 0
                    var tmp = [Float](repeating: 0, count: range.count)
                    tmp.withUnsafeMutableBufferPointer { sub in
                        BatchKernels.range_cosine_fused_512(query: query, candidates: candidates, range: range, out: sub)
                        for j in 0..<sub.count { local += sub[j] }
                    }
                    return local
                } _: { $0 + $1 }
                observations.observe(observedValue)
                blackHole(observedValue)
            }
            var outputs = [Float](repeating: 0, count: n)
            outputs.withUnsafeMutableBufferPointer { out in
                BatchKernels.range_cosine_fused_512(query: query, candidates: candidates, range: 0..<n, out: out)
            }
            let stats = Correctness.vector(currents: outputs, references: refs)
            let out = BenchResult(name: res.name, iterations: res.iterations, totalNanoseconds: res.totalNanoseconds, unitCount: res.unitCount, samples: res.samples,
                meanNsPerOp: res.meanNsPerOp, medianNsPerOp: res.medianNsPerOp, p90NsPerOp: res.p90NsPerOp, stddevNsPerOp: res.stddevNsPerOp, rsdPercent: res.rsdPercent,
                correctness: observations.combining(stats))
            return [out]
        }
    }

    private static func runBatchOptimizedCosineFused(
        name: String,
        query: Vector768Optimized,
        candidates: [Vector768Optimized],
        provider: any ComputeProvider,
        options: ValidatedOptions,
        minChunk: Int
    ) async -> [BenchResult] {
        return await BatchExecution.run(name: name) {
            let n = candidates.count
            let refs: [Double] = query.withUnsafeBufferPointer { qb in
                candidates.map { cand in cand.withUnsafeBufferPointer { cb in DoubleRef.cosineDist(qb, cb) } }
            }
            let observations = ProviderObservations(reference: refs.reduce(0, +))
            try await Harness.warmupAsync {
                let observedValue = try await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                    var local: Float = 0
                    var tmp = [Float](repeating: 0, count: range.count)
                    tmp.withUnsafeMutableBufferPointer { sub in
                        BatchKernels.range_cosine_fused_768(query: query, candidates: candidates, range: range, out: sub)
                        for j in 0..<sub.count { local += sub[j] }
                    }
                    return local
                } _: { $0 + $1 }
                observations.observe(observedValue)
                blackHole(observedValue)
            }
            let res = try await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
                let observedValue = try await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                    var local: Float = 0
                    var tmp = [Float](repeating: 0, count: range.count)
                    tmp.withUnsafeMutableBufferPointer { sub in
                        BatchKernels.range_cosine_fused_768(query: query, candidates: candidates, range: range, out: sub)
                        for j in 0..<sub.count { local += sub[j] }
                    }
                    return local
                } _: { $0 + $1 }
                observations.observe(observedValue)
                blackHole(observedValue)
            }
            var outputs = [Float](repeating: 0, count: n)
            outputs.withUnsafeMutableBufferPointer { out in
                BatchKernels.range_cosine_fused_768(query: query, candidates: candidates, range: 0..<n, out: out)
            }
            let stats = Correctness.vector(currents: outputs, references: refs)
            let out = BenchResult(name: res.name, iterations: res.iterations, totalNanoseconds: res.totalNanoseconds, unitCount: res.unitCount, samples: res.samples,
                meanNsPerOp: res.meanNsPerOp, medianNsPerOp: res.medianNsPerOp, p90NsPerOp: res.p90NsPerOp, stddevNsPerOp: res.stddevNsPerOp, rsdPercent: res.rsdPercent,
                correctness: observations.combining(stats))
            return [out]
        }
    }

    private static func runBatchOptimizedCosineFused(
        name: String,
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized],
        provider: any ComputeProvider,
        options: ValidatedOptions,
        minChunk: Int
    ) async -> [BenchResult] {
        return await BatchExecution.run(name: name) {
            let n = candidates.count
            let refs: [Double] = query.withUnsafeBufferPointer { qb in
                candidates.map { cand in cand.withUnsafeBufferPointer { cb in DoubleRef.cosineDist(qb, cb) } }
            }
            let observations = ProviderObservations(reference: refs.reduce(0, +))
            let _ = [Float](repeating: 0, count: n)
            try await Harness.warmupAsync {
                let observedValue = try await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                    var local: Float = 0
                    var tmp = [Float](repeating: 0, count: range.count)
                    tmp.withUnsafeMutableBufferPointer { sub in
                        BatchKernels.range_cosine_fused_1536(query: query, candidates: candidates, range: range, out: sub)
                        for j in 0..<sub.count { local += sub[j] }
                    }
                    return local
                } _: { $0 + $1 }
                observations.observe(observedValue)
                blackHole(observedValue)
            }
            let res = try await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
                let observedValue = try await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                    var local: Float = 0
                    var tmp = [Float](repeating: 0, count: range.count)
                    tmp.withUnsafeMutableBufferPointer { sub in
                        BatchKernels.range_cosine_fused_1536(query: query, candidates: candidates, range: range, out: sub)
                        for j in 0..<sub.count { local += sub[j] }
                    }
                    return local
                } _: { $0 + $1 }
                observations.observe(observedValue)
                blackHole(observedValue)
            }
            var outputs = [Float](repeating: 0, count: n)
            outputs.withUnsafeMutableBufferPointer { out in
                BatchKernels.range_cosine_fused_1536(query: query, candidates: candidates, range: 0..<n, out: out)
            }
            let stats = Correctness.vector(currents: outputs, references: refs)
            let out = BenchResult(name: res.name, iterations: res.iterations, totalNanoseconds: res.totalNanoseconds, unitCount: res.unitCount, samples: res.samples,
                meanNsPerOp: res.meanNsPerOp, medianNsPerOp: res.medianNsPerOp, p90NsPerOp: res.p90NsPerOp, stddevNsPerOp: res.stddevNsPerOp, rsdPercent: res.rsdPercent,
                correctness: observations.combining(stats))
            return [out]
        }
    }

    private static func runBatchOptimizedCosinePreNorm(
        name: String,
        query: Vector512Optimized,
        candidates: [Vector512Optimized],
        provider: any ComputeProvider,
        options: ValidatedOptions,
        minChunk: Int
    ) async -> [BenchResult] {
        return await BatchExecution.run(name: name) {
            let n = candidates.count
            let refs: [Double] = query.withUnsafeBufferPointer { qb in
                candidates.map { cand in cand.withUnsafeBufferPointer { cb in DoubleRef.cosineDist(qb, cb) } }
            }
            let observations = ProviderObservations(reference: refs.reduce(0, +))
            let _ = [Float](repeating: 0, count: n)
            try await Harness.warmupAsync {
                let observedValue = try await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                    var local: Float = 0
                    var tmp = [Float](repeating: 0, count: range.count)
                    tmp.withUnsafeMutableBufferPointer { sub in
                        BatchKernels.range_cosine_preNorm_512(query: query, candidates: candidates, range: range, out: sub)
                        for j in 0..<sub.count { local += sub[j] }
                    }
                    return local
                } _: { $0 + $1 }
                observations.observe(observedValue)
                blackHole(observedValue)
            }
            let res = try await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
                let observedValue = try await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                    var local: Float = 0
                    var tmp = [Float](repeating: 0, count: range.count)
                    tmp.withUnsafeMutableBufferPointer { sub in
                        BatchKernels.range_cosine_preNorm_512(query: query, candidates: candidates, range: range, out: sub)
                        for j in 0..<sub.count { local += sub[j] }
                    }
                    return local
                } _: { $0 + $1 }
                observations.observe(observedValue)
                blackHole(observedValue)
            }
            return [res.withCorrectness(observations.statistics)]
        }
    }

    private static func runBatchOptimizedCosinePreNorm(
        name: String,
        query: Vector768Optimized,
        candidates: [Vector768Optimized],
        provider: any ComputeProvider,
        options: ValidatedOptions,
        minChunk: Int
    ) async -> [BenchResult] {
        return await BatchExecution.run(name: name) {
            let n = candidates.count
            let refs: [Double] = query.withUnsafeBufferPointer { qb in
                candidates.map { cand in cand.withUnsafeBufferPointer { cb in DoubleRef.cosineDist(qb, cb) } }
            }
            let observations = ProviderObservations(reference: refs.reduce(0, +))
            let _ = [Float](repeating: 0, count: n)
            try await Harness.warmupAsync {
                let observedValue = try await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                    var local: Float = 0
                    var tmp = [Float](repeating: 0, count: range.count)
                    tmp.withUnsafeMutableBufferPointer { sub in
                        BatchKernels.range_cosine_preNorm_768(query: query, candidates: candidates, range: range, out: sub)
                        for j in 0..<sub.count { local += sub[j] }
                    }
                    return local
                } _: { $0 + $1 }
                observations.observe(observedValue)
                blackHole(observedValue)
            }
            let res = try await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
                let observedValue = try await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                    var local: Float = 0
                    var tmp = [Float](repeating: 0, count: range.count)
                    tmp.withUnsafeMutableBufferPointer { sub in
                        BatchKernels.range_cosine_preNorm_768(query: query, candidates: candidates, range: range, out: sub)
                        for j in 0..<sub.count { local += sub[j] }
                    }
                    return local
                } _: { $0 + $1 }
                observations.observe(observedValue)
                blackHole(observedValue)
            }
            return [res.withCorrectness(observations.statistics)]
        }
    }

    private static func runBatchOptimizedCosinePreNorm(
        name: String,
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized],
        provider: any ComputeProvider,
        options: ValidatedOptions,
        minChunk: Int
    ) async -> [BenchResult] {
        return await BatchExecution.run(name: name) {
            let n = candidates.count
            let refs: [Double] = query.withUnsafeBufferPointer { qb in
                candidates.map { cand in cand.withUnsafeBufferPointer { cb in DoubleRef.cosineDist(qb, cb) } }
            }
            let observations = ProviderObservations(reference: refs.reduce(0, +))
            let _ = [Float](repeating: 0, count: n)
            try await Harness.warmupAsync {
                let observedValue = try await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                    var local: Float = 0
                    var tmp = [Float](repeating: 0, count: range.count)
                    tmp.withUnsafeMutableBufferPointer { sub in
                        BatchKernels.range_cosine_preNorm_1536(query: query, candidates: candidates, range: range, out: sub)
                        for j in 0..<sub.count { local += sub[j] }
                    }
                    return local
                } _: { $0 + $1 }
                observations.observe(observedValue)
                blackHole(observedValue)
            }
            let res = try await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
                let observedValue = try await provider.parallelReduce(items: 0..<n, initial: Float(0), minChunk: minChunk) { range in
                    var local: Float = 0
                    var tmp = [Float](repeating: 0, count: range.count)
                    tmp.withUnsafeMutableBufferPointer { sub in
                        BatchKernels.range_cosine_preNorm_1536(query: query, candidates: candidates, range: range, out: sub)
                        for j in 0..<sub.count { local += sub[j] }
                    }
                    return local
                } _: { $0 + $1 }
                observations.observe(observedValue)
                blackHole(observedValue)
            }
            return [res.withCorrectness(observations.statistics)]
        }
    }

    // MARK: - SoA (Structure-of-Arrays) Benchmark Functions

    private static func runBatchSoA512(
        name: String,
        query: Vector512Optimized,
        candidates: [Vector512Optimized],
        options: ValidatedOptions
    ) async throws -> [BenchResult] {
        let n = candidates.count

        // Build SoA once, before warmup and timing. This preparation is outside the timed body
        // (see the case's measurement boundary: inputPreparation = excluded).
        let soa = SoA<Vector512Optimized>.build(from: candidates)

        // Allocate results buffer once to avoid excessive allocations in warmup loop
        // Use UnsafeMutableBufferPointer directly for @Sendable compatibility
        nonisolated(unsafe) let resultsPtr = UnsafeMutableBufferPointer<Float>.allocate(capacity: n)
        defer { resultsPtr.deallocate() }
        resultsPtr.initialize(repeating: 0)

        try await Harness.warmupAsync {
            BatchKernels_SoA.euclid2_512(query: query, soa: soa, out: resultsPtr)
            var total: Float = 0
            for i in 0..<n { total += resultsPtr[i] }
            blackHole(total)
        }

        let res = try await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
            BatchKernels_SoA.euclid2_512(query: query, soa: soa, out: resultsPtr)
            var total: Float = 0
            for i in 0..<n { total += resultsPtr[i] }
            blackHole(total)
        }
        // Correctness: compute outputs once and compare to DoubleRef.euclid2
        var outputs = [Float](repeating: 0, count: n)
        outputs.withUnsafeMutableBufferPointer { out in
            BatchKernels_SoA.euclid2_512(query: query, soa: soa, out: out)
        }
        let refs: [Double] = query.withUnsafeBufferPointer { qb in
            candidates.map { cand in cand.withUnsafeBufferPointer { cb in DoubleRef.euclid2(qb, cb) } }
        }
        let stats = Correctness.vector(currents: outputs, references: refs)
        let out = BenchResult(name: res.name, iterations: res.iterations, totalNanoseconds: res.totalNanoseconds, unitCount: res.unitCount, samples: res.samples,
            meanNsPerOp: res.meanNsPerOp, medianNsPerOp: res.medianNsPerOp, p90NsPerOp: res.p90NsPerOp, stddevNsPerOp: res.stddevNsPerOp, rsdPercent: res.rsdPercent, correctness: stats)
        return [out]
    }

    private static func runBatchSoA768(
        name: String,
        query: Vector768Optimized,
        candidates: [Vector768Optimized],
        options: ValidatedOptions
    ) async throws -> [BenchResult] {
        let n = candidates.count

        // Build SoA once, before warmup and timing. This preparation is outside the timed body
        // (see the case's measurement boundary: inputPreparation = excluded).
        let soa = SoA<Vector768Optimized>.build(from: candidates)

        // Allocate results buffer once to avoid excessive allocations in warmup loop
        // Use UnsafeMutableBufferPointer directly for @Sendable compatibility
        nonisolated(unsafe) let resultsPtr = UnsafeMutableBufferPointer<Float>.allocate(capacity: n)
        defer { resultsPtr.deallocate() }
        resultsPtr.initialize(repeating: 0)

        try await Harness.warmupAsync {
            BatchKernels_SoA.euclid2_768(query: query, soa: soa, out: resultsPtr)
            var total: Float = 0
            for i in 0..<n { total += resultsPtr[i] }
            blackHole(total)
        }

        let res = try await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
            BatchKernels_SoA.euclid2_768(query: query, soa: soa, out: resultsPtr)
            var total: Float = 0
            for i in 0..<n { total += resultsPtr[i] }
            blackHole(total)
        }
        var outputs = [Float](repeating: 0, count: n)
        outputs.withUnsafeMutableBufferPointer { out in
            BatchKernels_SoA.euclid2_768(query: query, soa: soa, out: out)
        }
        let refs: [Double] = query.withUnsafeBufferPointer { qb in
            candidates.map { cand in cand.withUnsafeBufferPointer { cb in DoubleRef.euclid2(qb, cb) } }
        }
        let stats = Correctness.vector(currents: outputs, references: refs)
        let out = BenchResult(name: res.name, iterations: res.iterations, totalNanoseconds: res.totalNanoseconds, unitCount: res.unitCount, samples: res.samples,
            meanNsPerOp: res.meanNsPerOp, medianNsPerOp: res.medianNsPerOp, p90NsPerOp: res.p90NsPerOp, stddevNsPerOp: res.stddevNsPerOp, rsdPercent: res.rsdPercent, correctness: stats)
        return [out]
    }

    private static func runBatchSoA1536(
        name: String,
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized],
        options: ValidatedOptions
    ) async throws -> [BenchResult] {
        let n = candidates.count

        // Build SoA once, before warmup and timing. This preparation is outside the timed body
        // (see the case's measurement boundary: inputPreparation = excluded).
        let soa = SoA<Vector1536Optimized>.build(from: candidates)

        // Allocate results buffer once to avoid excessive allocations in warmup loop
        // Use UnsafeMutableBufferPointer directly for @Sendable compatibility
        nonisolated(unsafe) let resultsPtr = UnsafeMutableBufferPointer<Float>.allocate(capacity: n)
        defer { resultsPtr.deallocate() }
        resultsPtr.initialize(repeating: 0)

        try await Harness.warmupAsync {
            BatchKernels_SoA.euclid2_1536(query: query, soa: soa, out: resultsPtr)
            var total: Float = 0
            for i in 0..<n { total += resultsPtr[i] }
            blackHole(total)
        }

        let res = try await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
            BatchKernels_SoA.euclid2_1536(query: query, soa: soa, out: resultsPtr)
            var total: Float = 0
            for i in 0..<n { total += resultsPtr[i] }
            blackHole(total)
        }
        var outputs = [Float](repeating: 0, count: n)
        outputs.withUnsafeMutableBufferPointer { out in
            BatchKernels_SoA.euclid2_1536(query: query, soa: soa, out: out)
        }
        let refs: [Double] = query.withUnsafeBufferPointer { qb in
            candidates.map { cand in cand.withUnsafeBufferPointer { cb in DoubleRef.euclid2(qb, cb) } }
        }
        let stats = Correctness.vector(currents: outputs, references: refs)
        let out = BenchResult(name: res.name, iterations: res.iterations, totalNanoseconds: res.totalNanoseconds, unitCount: res.unitCount, samples: res.samples,
            meanNsPerOp: res.meanNsPerOp, medianNsPerOp: res.medianNsPerOp, p90NsPerOp: res.p90NsPerOp, stddevNsPerOp: res.stddevNsPerOp, rsdPercent: res.rsdPercent, correctness: stats)
        return [out]
    }

    // MARK: - FP16 Mixed-Precision Benchmark Functions

    private static func runBatchFP16_512(
        name: String,
        query: Vector512Optimized,
        candidates: [Vector512Optimized],
        options: ValidatedOptions
    ) async throws -> [BenchResult] {
        let n = candidates.count

        // Convert candidates to FP16 once, before warmup and timing. This preparation is outside the timed body
        // (see the case's measurement boundary: inputPreparation = excluded).
        let candidatesFP16 = MixedPrecisionKernels.convertToFP16_512(candidates)

        // Allocate results buffer once to avoid excessive allocations in warmup loop
        // Use UnsafeMutableBufferPointer directly for @Sendable compatibility
        nonisolated(unsafe) let resultsPtr = UnsafeMutableBufferPointer<Float>.allocate(capacity: n)
        defer { resultsPtr.deallocate() }
        resultsPtr.initialize(repeating: 0)

        try await Harness.warmupAsync {
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

        let res = try await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
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
        // Correctness: copy outputs and compare to double-precision reference
        var outputs = [Float](repeating: 0, count: n)
        outputs.withUnsafeMutableBufferPointer { out in
            MixedPrecisionKernels.range_euclid2_mixed_512(
                query: query, candidatesFP16: candidatesFP16, range: 0..<n,
                out: out
            )
        }
        let refs: [Double] = query.withUnsafeBufferPointer { qb in
            candidates.map { cand in cand.withUnsafeBufferPointer { cb in DoubleRef.euclid2(qb, cb) } }
        }
        let stats = Correctness.vector(currents: outputs, references: refs)
        let out = BenchResult(name: res.name, iterations: res.iterations, totalNanoseconds: res.totalNanoseconds, unitCount: res.unitCount, samples: res.samples,
            meanNsPerOp: res.meanNsPerOp, medianNsPerOp: res.medianNsPerOp, p90NsPerOp: res.p90NsPerOp, stddevNsPerOp: res.stddevNsPerOp, rsdPercent: res.rsdPercent, correctness: stats)
        return [out]
    }

    private static func runBatchFP16_768(
        name: String,
        query: Vector768Optimized,
        candidates: [Vector768Optimized],
        options: ValidatedOptions
    ) async throws -> [BenchResult] {
        let n = candidates.count

        // Convert candidates to FP16 once, before warmup and timing. This preparation is outside the timed body
        // (see the case's measurement boundary: inputPreparation = excluded).
        let candidatesFP16 = MixedPrecisionKernels.convertToFP16_768(candidates)

        // Allocate results buffer once to avoid excessive allocations in warmup loop
        // Use UnsafeMutableBufferPointer directly for @Sendable compatibility
        nonisolated(unsafe) let resultsPtr = UnsafeMutableBufferPointer<Float>.allocate(capacity: n)
        defer { resultsPtr.deallocate() }
        resultsPtr.initialize(repeating: 0)

        try await Harness.warmupAsync {
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

        let res = try await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
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
        var outputs = [Float](repeating: 0, count: n)
        outputs.withUnsafeMutableBufferPointer { out in
            MixedPrecisionKernels.range_euclid2_mixed_768(
                query: query, candidatesFP16: candidatesFP16, range: 0..<n,
                out: out
            )
        }
        let refs: [Double] = query.withUnsafeBufferPointer { qb in
            candidates.map { cand in cand.withUnsafeBufferPointer { cb in DoubleRef.euclid2(qb, cb) } }
        }
        let stats = Correctness.vector(currents: outputs, references: refs)
        let out = BenchResult(name: res.name, iterations: res.iterations, totalNanoseconds: res.totalNanoseconds, unitCount: res.unitCount, samples: res.samples,
            meanNsPerOp: res.meanNsPerOp, medianNsPerOp: res.medianNsPerOp, p90NsPerOp: res.p90NsPerOp, stddevNsPerOp: res.stddevNsPerOp, rsdPercent: res.rsdPercent, correctness: stats)
        return [out]
    }

    private static func runBatchFP16_1536(
        name: String,
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized],
        options: ValidatedOptions
    ) async throws -> [BenchResult] {
        let n = candidates.count

        // Convert candidates to FP16 once, before warmup and timing. This preparation is outside the timed body
        // (see the case's measurement boundary: inputPreparation = excluded).
        let candidatesFP16 = MixedPrecisionKernels.convertToFP16_1536(candidates)

        // Allocate results buffer once to avoid excessive allocations in warmup loop
        // Use UnsafeMutableBufferPointer directly for @Sendable compatibility
        nonisolated(unsafe) let resultsPtr = UnsafeMutableBufferPointer<Float>.allocate(capacity: n)
        defer { resultsPtr.deallocate() }
        resultsPtr.initialize(repeating: 0)


        try await Harness.warmupAsync {
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

        let res = try await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
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
        var outputs = [Float](repeating: 0, count: n)
        outputs.withUnsafeMutableBufferPointer { out in
            MixedPrecisionKernels.range_euclid2_mixed_1536(
                query: query, candidatesFP16: candidatesFP16, range: 0..<n,
                out: out
            )
        }
        let refs: [Double] = query.withUnsafeBufferPointer { qb in
            candidates.map { cand in cand.withUnsafeBufferPointer { cb in DoubleRef.euclid2(qb, cb) } }
        }
        let stats = Correctness.vector(currents: outputs, references: refs)
        let out = BenchResult(name: res.name, iterations: res.iterations, totalNanoseconds: res.totalNanoseconds, unitCount: res.unitCount, samples: res.samples,
            meanNsPerOp: res.meanNsPerOp, medianNsPerOp: res.medianNsPerOp, p90NsPerOp: res.p90NsPerOp, stddevNsPerOp: res.stddevNsPerOp, rsdPercent: res.rsdPercent, correctness: stats)
        return [out]
    }

    private static func runBatchCosineFP16_512(
        name: String,
        query: Vector512Optimized,
        candidates: [Vector512Optimized],
        options: ValidatedOptions
    ) async throws -> [BenchResult] {
        let n = candidates.count

        // Convert candidates to FP16 once, before warmup and timing. This preparation is outside the timed body
        // (see the case's measurement boundary: inputPreparation = excluded).
        let candidatesFP16 = MixedPrecisionKernels.convertToFP16_512(candidates)

        // Allocate results buffer once to avoid excessive allocations in warmup loop
        // Use UnsafeMutableBufferPointer directly for @Sendable compatibility
        nonisolated(unsafe) let resultsPtr = UnsafeMutableBufferPointer<Float>.allocate(capacity: n)
        defer { resultsPtr.deallocate() }
        resultsPtr.initialize(repeating: 0)


        try await Harness.warmupAsync {
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

        let res = try await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
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
        var outputs = [Float](repeating: 0, count: n)
        outputs.withUnsafeMutableBufferPointer { out in
            MixedPrecisionKernels.range_cosine_mixed_512(
                query: query, candidatesFP16: candidatesFP16, range: 0..<n,
                out: out
            )
        }
        let refs: [Double] = query.withUnsafeBufferPointer { qb in
            candidates.map { cand in cand.withUnsafeBufferPointer { cb in DoubleRef.cosineDist(qb, cb) } }
        }
        let stats = Correctness.vector(currents: outputs, references: refs)
        let out = BenchResult(name: res.name, iterations: res.iterations, totalNanoseconds: res.totalNanoseconds, unitCount: res.unitCount, samples: res.samples,
            meanNsPerOp: res.meanNsPerOp, medianNsPerOp: res.medianNsPerOp, p90NsPerOp: res.p90NsPerOp, stddevNsPerOp: res.stddevNsPerOp, rsdPercent: res.rsdPercent, correctness: stats)
        return [out]
    }

    private static func runBatchCosineFP16_768(
        name: String,
        query: Vector768Optimized,
        candidates: [Vector768Optimized],
        options: ValidatedOptions
    ) async throws -> [BenchResult] {
        let n = candidates.count

        // Convert candidates to FP16 once, before warmup and timing. This preparation is outside the timed body
        // (see the case's measurement boundary: inputPreparation = excluded).
        let candidatesFP16 = MixedPrecisionKernels.convertToFP16_768(candidates)

        // Allocate results buffer once to avoid excessive allocations in warmup loop
        // Use UnsafeMutableBufferPointer directly for @Sendable compatibility
        nonisolated(unsafe) let resultsPtr = UnsafeMutableBufferPointer<Float>.allocate(capacity: n)
        defer { resultsPtr.deallocate() }
        resultsPtr.initialize(repeating: 0)


        try await Harness.warmupAsync {
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

        let res = try await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
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
        var outputs = [Float](repeating: 0, count: n)
        outputs.withUnsafeMutableBufferPointer { out in
            MixedPrecisionKernels.range_cosine_mixed_768(
                query: query, candidatesFP16: candidatesFP16, range: 0..<n,
                out: out
            )
        }
        let refs: [Double] = query.withUnsafeBufferPointer { qb in
            candidates.map { cand in cand.withUnsafeBufferPointer { cb in DoubleRef.cosineDist(qb, cb) } }
        }
        let stats = Correctness.vector(currents: outputs, references: refs)
        let out = BenchResult(name: res.name, iterations: res.iterations, totalNanoseconds: res.totalNanoseconds, unitCount: res.unitCount, samples: res.samples,
            meanNsPerOp: res.meanNsPerOp, medianNsPerOp: res.medianNsPerOp, p90NsPerOp: res.p90NsPerOp, stddevNsPerOp: res.stddevNsPerOp, rsdPercent: res.rsdPercent, correctness: stats)
        return [out]
    }

    private static func runBatchCosineFP16_1536(
        name: String,
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized],
        options: ValidatedOptions
    ) async throws -> [BenchResult] {
        let n = candidates.count

        // Convert candidates to FP16 once, before warmup and timing. This preparation is outside the timed body
        // (see the case's measurement boundary: inputPreparation = excluded).
        let candidatesFP16 = MixedPrecisionKernels.convertToFP16_1536(candidates)

        // Allocate results buffer once to avoid excessive allocations in warmup loop
        // Use UnsafeMutableBufferPointer directly for @Sendable compatibility
        nonisolated(unsafe) let resultsPtr = UnsafeMutableBufferPointer<Float>.allocate(capacity: n)
        defer { resultsPtr.deallocate() }
        resultsPtr.initialize(repeating: 0)


        try await Harness.warmupAsync {
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

        let res = try await Harness.measureAsync(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: n, samples: options.samples) {
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
        var outputs = [Float](repeating: 0, count: n)
        outputs.withUnsafeMutableBufferPointer { out in
            MixedPrecisionKernels.range_cosine_mixed_1536(
                query: query, candidatesFP16: candidatesFP16, range: 0..<n,
                out: out
            )
        }
        let refs: [Double] = query.withUnsafeBufferPointer { qb in
            candidates.map { cand in cand.withUnsafeBufferPointer { cb in DoubleRef.cosineDist(qb, cb) } }
        }
        let stats = Correctness.vector(currents: outputs, references: refs)
        let out = BenchResult(name: res.name, iterations: res.iterations, totalNanoseconds: res.totalNanoseconds, unitCount: res.unitCount, samples: res.samples,
            meanNsPerOp: res.meanNsPerOp, medianNsPerOp: res.medianNsPerOp, p90NsPerOp: res.p90NsPerOp, stddevNsPerOp: res.stddevNsPerOp, rsdPercent: res.rsdPercent, correctness: stats)
        return [out]
    }
}
