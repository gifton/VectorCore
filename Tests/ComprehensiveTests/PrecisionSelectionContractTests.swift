import Foundation
import Testing
@testable import VectorCore

@Suite struct PrecisionSelectionContractTests {
    @Test func adaptiveBoundAgainstActualBaseline() {
        let values: [Float] = [0, -0.0, 1e-20, 1, 1.0001, 100, 65504, 70000, .greatestFiniteMagnitude, .infinity, -.infinity, .nan]
        let thresholds: [Float] = [0, -0.0, Float.leastNonzeroMagnitude, 0.000001, 0.001, 0.5, 1, .greatestFiniteMagnitude, -1, .infinity, .nan]
        for a in values {
            for b in values {
                let query = Vector512Optimized(repeating: a), candidate = Vector512Optimized(repeating: b)
                let reference = query.euclideanDistance(to: candidate)
                for threshold in thresholds {
                    let actual = MixedPrecisionKernels.adaptiveEuclideanDistance(query: query, candidate: candidate, threshold: threshold)
                    if !threshold.isFinite || threshold <= 0 || !reference.isFinite || reference == 0 {
                        #expect(actual.bitPattern == reference.bitPattern)
                    } else {
                        #expect(actual.isFinite)
                        #expect(abs(Double(actual)-Double(reference)) <= Double(threshold)*Double(reference))
                    }
                }
            }
        }
    }

    @Test func quantizedHintsAreOrderIndependent() throws {
        let hints: [Float] = [0, -0.0, Float(0.05).nextDown, 0.05, Float(0.1).nextDown, 0.1, 1]
        for count in [0, 1, 1000, 1001, 5000, 5001] {
            for dimension in [1, 512, 768, 1536] {
                for ordering in [hints, hints.reversed().map { $0 }] {
                    let warm = QuantizedAutoTuner()
                    for hint in ordering {
                        let result = try warm.suggestQuantization(for: "opaque_op", dimension: dimension, candidateCount: count, relativeErrorHint: hint)
                        let fresh = try QuantizedAutoTuner().suggestQuantization(for: "opaque_op", dimension: dimension, candidateCount: count, relativeErrorHint: hint)
                        let expected: QuantizedAutoTuner.QuantizationStrategy = count > 5000 && hint >= 0.1 ? .bothINT8 : (count > 1000 && hint >= 0.05 ? .candidatesINT8 : .fullFP32)
                        #expect(result == fresh && result == expected)
                    }
                }
            }
        }
    }
}

private func q02Metric(_ strategy: MixedPrecisionStrategy, dimension: Int = 512,
                       count: Int = 2, error: Double = 0, latency: Double = 0.001) -> StrategyMetrics {
    StrategyMetrics(strategy: strategy, meanLatency: latency, medianLatency: latency,
        throughput: 1 / latency, maxRelativeError: error, meanRelativeError: error,
        candidateCount: count, dimension: dimension, measuredSamples: 1, warmupIterations: 0,
        algorithmVersion: 1, datasetVersion: "gaussian-lcg-boxmuller-v1", seed: 42,
        conversionBoundary: "fp32-input-conversion-per-reduced-euclidean-invocation-v1", timestamp: Date())
}

extension PrecisionSelectionContractTests {
    @Test func explicitFiveStrategyWorkAndLookupOnly() async throws {
        for width in [512, 768, 1536] {
            for count in [1, 2, 3] {
                let tuner = MixedPrecisionAutoTuner(testing: TuningTestHooks(latency: { _, _, _ in 0.001 }))
                for limit: Float in [0, 0.001, 1] {
                    #expect(try await tuner.selectStrategy(candidateCount: count,
                        maximumSyntheticRelativeError: limit, dimension: width) == .fullFP32)
                }
                #expect(await tuner.workCounts == TuningWorkCounts())
                let metrics = try await tuner.calibrate(candidateCount: count, dimension: width,
                                                       iterations: 2, warmupIterations: 1)
                #expect(metrics.count == 5)
                let work = try #require(await tuner.workCounts)
                #expect(work.generatedVectors == count + 1)
                #expect(work.baselineExecutions == 1)
                #expect(work.convertedFP16Vectors == 3 * (4 * count + 2))
                #expect(work.convertedINT8Vectors == 0)
                for strategy in MixedPrecisionStrategy.allCases {
                    #expect(work.warmupExecutions[strategy.rawValue] == 1)
                    #expect(work.measuredExecutions[strategy.rawValue] == 2)
                    let metric = try #require(metrics.first { $0.strategy == strategy })
                    #expect(metric.dimension == width && metric.candidateCount == count)
                    #expect(metric.measuredSamples == 2 && metric.warmupIterations == 1)
                    #expect(metric.meanLatency == 0.001 && metric.medianLatency == 0.001)
                    #expect(metric.maxRelativeError.isFinite && metric.meanRelativeError.isFinite)
                }
                _ = try await tuner.selectStrategy(candidateCount: count, dimension: width)
                #expect(await tuner.workCounts == work)
            }
        }
    }

    @Test func mixedProfilesFiltersOverridesAndExactIdentity() async throws {
        for width in [512, 768, 1536] {
            let tuner = MixedPrecisionAutoTuner(testing: TuningTestHooks())
            await tuner.setProfilesForTesting([
                q02Metric(.queryFP16Standard, dimension: width, error: 0.01, latency: 0.00001),
                q02Metric(.queryFP32Blocked, dimension: width, error: 0.0001),
                q02Metric(.fullFP32, dimension: width, latency: 1)
            ], candidateCount: 2, dimension: width)
            #expect(try await tuner.selectStrategy(candidateCount: 2, maximumSyntheticRelativeError: 0,
                                                   dimension: width) == .fullFP32)
            #expect(try await tuner.selectStrategy(candidateCount: 2, maximumSyntheticRelativeError: 0.001,
                                                   dimension: width) == .queryFP32Blocked)
            #expect(try await tuner.selectStrategy(candidateCount: 3, dimension: width) == .fullFP32)
            try await tuner.setOverride(.fullFP32, candidateCount: 2, dimension: width)
            do {
                try await tuner.setOverride(.queryFP16Standard, candidateCount: 2, dimension: width)
                Issue.record("Inaccurate reduced override accepted")
            } catch let error as VectorError { #expect(error.kind == .invalidConfiguration) }
            #expect(try await tuner.selectStrategy(candidateCount: 2, dimension: width) == .fullFP32)
            try await tuner.setOverride(.queryFP32Blocked, candidateCount: 2, dimension: width)
            await tuner.setProfilesForTesting([], candidateCount: 2, dimension: width)
            do {
                _ = try await tuner.selectStrategy(candidateCount: 2, dimension: width)
                Issue.record("Stale override accepted")
            } catch let error as VectorError { #expect(error.kind == .invalidConfiguration) }
            await tuner.clearCache()
            #expect(try await tuner.selectStrategy(candidateCount: 2, dimension: width) == .fullFP32)
            #expect(await tuner.workCounts == TuningWorkCounts())
        }
    }

    @Test func kernelRepresentationAndResourceMatrix() async throws {
        let modes: [WorkloadCharacteristics.PrecisionMode] = [.fp32Full, .fp16Acceptable, .int8Acceptable]
        for width in [512, 768, 1536] {
            for mode in modes {
                let tuner = KernelAutoTuner()
                for count in [0, 1, 99, 100, 1000] {
                    let workload = WorkloadCharacteristics(dimension: width, batchSize: count, precisionMode: mode)
                    let allowed: [KernelStrategy] = mode == .fp32Full ? [.cpuAoS, .cpuSoA] :
                        (mode == .fp16Acceptable ? [.cpuAoS, .cpuSoA, .cpuMixedPrecision] :
                            [.cpuAoS, .cpuSoA, .cpuMixedPrecision, .cpuQuantized])
                    #expect(allowed.contains(try await tuner.selectStrategy(for: workload)))
                    for strategy in KernelStrategy.allCases {
                        do {
                            try await tuner.setOverride(strategy, for: workload)
                            #expect(allowed.contains(strategy))
                            #expect(try await tuner.selectStrategy(for: workload) == strategy)
                        } catch let error as VectorError {
                            #expect(!allowed.contains(strategy) && error.kind == .invalidConfiguration)
                        }
                    }
                    for filtered in [
                        WorkloadCharacteristics(dimension: width, batchSize: count, precisionMode: mode, maximumEstimatedPayloadBytes: 0),
                        WorkloadCharacteristics(dimension: width, batchSize: count, precisionMode: mode, maximumProfiledMeanLatency: 1)
                    ] {
                        do {
                            _ = try await tuner.selectStrategy(for: filtered)
                            Issue.record("Unmeasured or impossible filter accepted")
                        } catch let error as VectorError { #expect(error.kind == .resourceUnavailable) }
                        do {
                            _ = try await tuner.selectMixedPrecisionStrategy(for: filtered)
                            Issue.record("DOT resource filter accepted for Euclidean integration")
                        } catch let error as VectorError { #expect(error.kind == .unsupportedOperation) }
                    }
                }
            }
        }
    }

    @Test func kernelActualPermittedRepresentations() async throws {
        for width in [512, 768, 1536] {
            for mode in [WorkloadCharacteristics.PrecisionMode.fp32Full, .fp16Acceptable, .int8Acceptable] {
                let tuner = KernelAutoTuner(provenance: KernelProfileProvenance.current(),
                    testing: TuningTestHooks(latency: { _, iteration, _ in [0.003, 0.001, 0.002][iteration] }))
                let workload = WorkloadCharacteristics(dimension: width, batchSize: 2, precisionMode: mode)
                let profiles = try await tuner.profileWorkload(workload, iterations: 3, warmupIterations: 1)
                let expectedCount = mode == .fp32Full ? 2 : (mode == .fp16Acceptable ? 3 : 4)
                #expect(profiles.count == expectedCount)
                let work = try #require(await tuner.workCounts)
                #expect(work.generatedVectors == 3 && work.baselineExecutions == 1)
                #expect(work.convertedFP16Vectors == (mode == .fp32Full ? 0 : 2))
                #expect(work.convertedINT8Vectors == (mode == .int8Acceptable ? 3 : 0))
                for profile in profiles {
                    #expect(work.warmupExecutions[profile.strategy.rawValue] == 1)
                    #expect(work.measuredExecutions[profile.strategy.rawValue] == 3)
                    #expect(profile.latencyP50 == 0.002 && profile.latencyP95 == 0.003 && profile.latencyP99 == 0.003)
                    let bytes = profile.strategy == .cpuMixedPrecision ? width * 8 :
                        (profile.strategy == .cpuQuantized ? width * 3 : width * 12)
                    #expect(profile.estimatedPayloadBytes == bytes)
                    #expect(profile.syntheticAccuracyScore.isFinite && (0...1).contains(profile.syntheticAccuracyScore))
                }
                _ = try await tuner.selectStrategy(for: workload)
                #expect(await tuner.workCounts == work)
            }
        }
    }

    @Test func independentWidenedErrorAndNearestRank() {
        let a: [Float] = [.greatestFiniteMagnitude, -.greatestFiniteMagnitude, 0, 1]
        let b: [Float] = [-.greatestFiniteMagnitude, .greatestFiniteMagnitude, 0, 1.0001]
        let expected = zip(a, b).map { abs(Double($0) - Double($1)) / max(abs(Double($1)), 1e-6) }
        let actual = PrecisionTuningSupport.errors(a, reference: b)
        #expect(actual?.maximum == expected.max())
        #expect(actual?.mean == expected.reduce(0, +) / Double(expected.count))
        #expect(PrecisionTuningSupport.errors([.nan], reference: [0]) == nil)
        for n in 1...20 {
            let values = (1...n).map(Double.init)
            for percentile in [0.5, 0.95, 0.99] {
                let rank = Int((Double(n) * percentile).rounded(.up))
                #expect(PrecisionTuningSupport.percentile(values, percentile) == Double(rank))
            }
        }
    }
}

private func q02Profile(_ strategy: KernelStrategy = .cpuAoS, width: Int = 512, count: Int = 2,
                        mode: WorkloadCharacteristics.PrecisionMode = .fp32Full,
                        latency: Double = 0.001, score: Double = 1,
                        provenance: KernelProfileProvenance = .current()) -> PerformanceProfile {
    let bytes = strategy == .cpuMixedPrecision ? width * 4 + count * width * 2 :
        (strategy == .cpuQuantized ? (count + 1) * width : (count + 1) * width * 4)
    return PerformanceProfile(strategy: strategy,
        workload: KernelProfileWorkload(dimension: width, batchSize: count, precisionMode: mode),
        throughput: Double(count) / latency, latencyMean: latency, latencyP50: latency,
        latencyP95: latency, latencyP99: latency, estimatedPayloadBytes: bytes,
        syntheticAccuracyScore: score, measuredSamples: 1, warmupIterations: 0,
        provenance: provenance, timestamp: Date())
}

extension PrecisionSelectionContractTests {
    @Test func cancellationBeforePublicationPreservesBothTuners() async throws {
        let points: [TuningCheckpoint] = [.generatedVector, .baseline,
            .willExecute(MixedPrecisionStrategy.fullFP32.rawValue, true),
            .didExecute(MixedPrecisionStrategy.fullFP32.rawValue, false), .accuracy, .willPublish]
        for point in points {
            let tuner = MixedPrecisionAutoTuner(testing: TuningTestHooks(observer: { event in
                if event == point { withUnsafeCurrentTask { $0?.cancel() } }
            }, latency: { _, _, _ in 0.001 }))
            await tuner.setProfilesForTesting([q02Metric(.fullFP32)], candidateCount: 2, dimension: 512)
            try await tuner.setOverride(.fullFP32, candidateCount: 2)
            let child = Task { try await tuner.calibrate(candidateCount: 2, iterations: 1, warmupIterations: 1) }
            do { _ = try await child.value; Issue.record("Cancellation did not stop pre-publication work") }
            catch is CancellationError {}
            let retained = await tuner.profilesForTesting(candidateCount: 2, dimension: 512)
            #expect(retained.count == 1 && retained.first?.strategy == .fullFP32)
            #expect(try await tuner.selectStrategy(candidateCount: 2) == .fullFP32)
        }
        let kernelPoints: [TuningCheckpoint] = [.generatedVector, .baseline,
            .willExecute(KernelStrategy.cpuAoS.rawValue, true),
            .didExecute(KernelStrategy.cpuAoS.rawValue, false), .accuracy, .willPublish]
        for point in kernelPoints {
            let tuner = KernelAutoTuner(provenance: .current(), testing: TuningTestHooks(observer: { event in
                if event == point { withUnsafeCurrentTask { $0?.cancel() } }
            }, latency: { _, _, _ in 0.001 }))
            let workload = WorkloadCharacteristics(dimension: 512, batchSize: 2, precisionMode: .fp32Full)
            await tuner.setProfilesForTesting([q02Profile()], for: workload)
            try await tuner.setOverride(.cpuSoA, for: workload)
            let child = Task { try await tuner.profileWorkload(workload, iterations: 1, warmupIterations: 1) }
            do { _ = try await child.value; Issue.record("Cancellation did not stop kernel publication") }
            catch is CancellationError {}
            let retained = await tuner.profilesForTesting(for: workload)
            #expect(retained.count == 1 && retained.first?.strategy == .cpuAoS)
            #expect(try await tuner.selectStrategy(for: workload) == .cpuSoA)
        }
    }

    @Test func cancellationAfterPublicationKeepsCommittedProfiles() async throws {
        let hooks = TuningTestHooks(observer: { event in
            if event == .didPublish { withUnsafeCurrentTask { $0?.cancel() } }
        }, latency: { _, _, _ in 0.001 })
        let mixed = MixedPrecisionAutoTuner(testing: hooks)
        let mixedTask = Task { try await mixed.calibrate(candidateCount: 1, iterations: 1, warmupIterations: 0) }
        #expect(try await mixedTask.value.count == 5)
        #expect(await mixed.profilesForTesting(candidateCount: 1, dimension: 512).count == 5)
        let kernel = KernelAutoTuner(provenance: .current(), testing: hooks)
        let workload = WorkloadCharacteristics(dimension: 512, batchSize: 1, precisionMode: .fp32Full)
        let kernelTask = Task { try await kernel.profileWorkload(workload, iterations: 1, warmupIterations: 0) }
        #expect(try await kernelTask.value.count == 2)
        #expect(await kernel.profilesForTesting(for: workload).count == 2)
    }

    @Test func invalidCalibrationAndSelectionDoNoWork() async throws {
        let mixed = MixedPrecisionAutoTuner(testing: TuningTestHooks())
        for (count, width, iterations, warmups) in [(-1,512,1,0),(0,512,1,0),(1,256,1,0),
            (1,512,0,0),(1,512,1,-1),(Int.max,512,1,0),(1,512,Int.max,1)] {
            do { _ = try await mixed.calibrate(candidateCount: count, dimension: width,
                iterations: iterations, warmupIterations: warmups); Issue.record("Invalid calibration accepted") }
            catch let error as VectorError { #expect(error.kind == .invalidConfiguration) }
        }
        for limit: Float in [-1, .infinity, -.infinity, .nan] {
            do { _ = try await mixed.selectStrategy(candidateCount: 1, maximumSyntheticRelativeError: limit)
                Issue.record("Invalid limit accepted") }
            catch let error as VectorError { #expect(error.kind == .invalidConfiguration) }
        }
        #expect(await mixed.workCounts == TuningWorkCounts())
        let kernel = KernelAutoTuner(provenance: .current())
        for workload in [WorkloadCharacteristics(dimension: 256,batchSize: 1),
            WorkloadCharacteristics(dimension: 512,batchSize: -1),
            WorkloadCharacteristics(dimension: 512,batchSize: Int.max),
            WorkloadCharacteristics(dimension: 512,batchSize: 1,maximumEstimatedPayloadBytes: -1),
            WorkloadCharacteristics(dimension: 512,batchSize: 1,maximumProfiledMeanLatency: .nan),
            WorkloadCharacteristics(dimension: 512,batchSize: 1,maximumProfiledMeanLatency: 0)] {
            do { _ = try await kernel.selectStrategy(for: workload); Issue.record("Invalid workload accepted") }
            catch let error as VectorError { #expect(error.kind == .invalidConfiguration) }
        }
        #expect(await kernel.workCounts == TuningWorkCounts())
        let quantized = QuantizedAutoTuner()
        for (width,count,hint): (Int,Int,Float) in [(0,1,0.1),(-1,1,0.1),(512,-1,0.1),(512,1,-1),(512,1,.nan),(512,1,.infinity)] {
            do { _ = try quantized.suggestQuantization(for: "opaque", dimension: width,
                candidateCount: count, relativeErrorHint: hint); Issue.record("Invalid suggestion accepted") }
            catch let error as VectorError { #expect(error.kind == .invalidConfiguration) }
        }
    }

    @Test func rejectedMetricsPreserveOldProfilesAndOverride() async throws {
        for invalidLatency in [0.0, -1, Double.nan, .infinity] {
            let mixed = MixedPrecisionAutoTuner(testing: TuningTestHooks(latency: { _, _, _ in invalidLatency }))
            await mixed.setProfilesForTesting([q02Metric(.fullFP32)], candidateCount: 1, dimension: 512)
            try await mixed.setOverride(.fullFP32, candidateCount: 1)
            do { _ = try await mixed.calibrate(candidateCount: 1, iterations: 1, warmupIterations: 0)
                Issue.record("All invalid metrics published") }
            catch let error as VectorError { #expect(error.kind == .operationFailed) }
            #expect(await mixed.profilesForTesting(candidateCount: 1, dimension: 512).count == 1)
            #expect(try await mixed.selectStrategy(candidateCount: 1) == .fullFP32)
            let kernel = KernelAutoTuner(provenance: .current(), testing: TuningTestHooks(latency: { _, _, _ in invalidLatency }))
            let workload = WorkloadCharacteristics(dimension: 512,batchSize: 1,precisionMode: .fp32Full)
            await kernel.setProfilesForTesting([q02Profile(count: 1)], for: workload)
            try await kernel.setOverride(.cpuSoA, for: workload)
            do { _ = try await kernel.profileWorkload(workload, iterations: 1, warmupIterations: 0)
                Issue.record("All invalid kernel metrics published") }
            catch let error as VectorError { #expect(error.kind == .operationFailed) }
            #expect(await kernel.profilesForTesting(for: workload).count == 1)
            #expect(try await kernel.selectStrategy(for: workload) == .cpuSoA)
        }
    }
}

extension PrecisionSelectionContractTests {
    @Test func deterministicBitPatternAdaptiveCorpus() throws {
        var state: UInt32 = 0x515032
        for trial in 0..<48 {
            var a: [Float] = [], b: [Float] = []
            for index in 0..<512 {
                state = state &* 1664525 &+ 1013904223
                let bits = trial < 24 ? ((state & 0x807fffff) | (UInt32(110 + trial % 20) << 23)) : state
                let value = Float(bitPattern: bits)
                a.append(value)
                b.append(index % 3 == 0 ? Float(bitPattern: bits &+ 1) : value)
            }
            let lhs = try Vector512Optimized(a), rhs = try Vector512Optimized(b)
            for (query, candidate) in [(lhs,rhs),(rhs,lhs)] {
                let baseline = query.euclideanDistance(to: candidate)
                for threshold: Float in [0, .leastNonzeroMagnitude, 1e-6, 0.001, 0.1, .greatestFiniteMagnitude] {
                    let actual = MixedPrecisionKernels.adaptiveEuclideanDistance(query: query, candidate: candidate, threshold: threshold)
                    if threshold == 0 || !baseline.isFinite || baseline == 0 {
                        #expect(actual.bitPattern == baseline.bitPattern)
                    } else {
                        #expect(actual.isFinite)
                        #expect(abs(Double(actual)-Double(baseline)) <= Double(threshold)*Double(baseline))
                    }
                }
            }
        }
    }

    @Test func everyQuantizedThresholdPermutation() throws {
        func permutations(_ values: [Float]) -> [[Float]] {
            if values.isEmpty { return [[]] }
            return values.indices.flatMap { index -> [[Float]] in
                var rest = values; let first = rest.remove(at: index)
                return permutations(rest).map { [first] + $0 }
            }
        }
        let orders = permutations([0, Float(0.05).nextDown, 0.05, Float(0.1).nextDown, 0.1, 1])
        #expect(orders.count == 720)
        for count in [0,1000,1001,5000,5001] {
            for order in orders {
                let tuner = QuantizedAutoTuner()
                for hint in order {
                    let expected: QuantizedAutoTuner.QuantizationStrategy = count > 5000 && hint >= 0.1 ? .bothINT8 :
                        (count > 1000 && hint >= 0.05 ? .candidatesINT8 : .fullFP32)
                    for operation in ["", "a_512_5001", "a", "opaque\u{0}operation"] {
                        #expect(try tuner.suggestQuantization(for: operation, dimension: 512,
                            candidateCount: count, relativeErrorHint: hint) == expected)
                    }
                }
            }
        }
    }

    @Test func deterministicTiesAndInvalidProfiles() async throws {
        let mixed = MixedPrecisionAutoTuner(testing: TuningTestHooks())
        for order in [MixedPrecisionStrategy.allCases, MixedPrecisionStrategy.allCases.reversed().map { $0 }] {
            await mixed.setProfilesForTesting(order.map { q02Metric($0) }, candidateCount: 2, dimension: 512)
            #expect(try await mixed.selectStrategy(candidateCount: 2) == .fullFP32)
        }
        for bad in [Double.nan, .infinity, -1] {
            await mixed.setProfilesForTesting([q02Metric(.queryFP16Blocked, error: bad)], candidateCount: 2, dimension: 512)
            #expect(try await mixed.selectStrategy(candidateCount: 2) == .fullFP32)
        }
        let kernel = KernelAutoTuner(provenance: .current())
        let plain = WorkloadCharacteristics(dimension: 512,batchSize: 2,precisionMode: .fp32Full)
        await kernel.setProfilesForTesting([q02Profile(.cpuSoA), q02Profile(.cpuAoS)], for: plain)
        #expect(try await kernel.selectStrategy(for: plain) == .cpuAoS)
        for latency in [0.0005,0.001,0.002] {
            for bytes in [0,6143,6144,6145] {
                let filtered = WorkloadCharacteristics(dimension: 512,batchSize: 2,precisionMode: .fp32Full,
                    maximumEstimatedPayloadBytes: bytes,maximumProfiledMeanLatency: latency)
                do {
                    let strategy = try await kernel.selectStrategy(for: filtered)
                    #expect(latency >= 0.001 && bytes >= 6144 && strategy == .cpuAoS)
                } catch let error as VectorError {
                    #expect((latency < 0.001 || bytes < 6144) && error.kind == .resourceUnavailable)
                }
            }
        }
        let measured = WorkloadCharacteristics(dimension: 512,batchSize: 2,precisionMode: .fp32Full,maximumProfiledMeanLatency: 1)
        try await kernel.setOverride(.cpuSoA, for: measured)
        await kernel.setProfilesForTesting([q02Profile(.cpuSoA, latency: .nan)], for: plain)
        do { _ = try await kernel.selectStrategy(for: measured); Issue.record("Invalid profile supported stored override") }
        catch let error as VectorError { #expect(error.kind == .resourceUnavailable) }
    }
}

extension PrecisionSelectionContractTests {
    @Test func alreadyCancelledCalibrationDoesNoWork() async throws {
        for width in [512,768,1536] {
            let mixed = MixedPrecisionAutoTuner(testing: TuningTestHooks())
            await mixed.setProfilesForTesting([q02Metric(.fullFP32, dimension: width)], candidateCount: 2, dimension: width)
            try await mixed.setOverride(.fullFP32, candidateCount: 2, dimension: width)
            let mixedTask = Task {
                withUnsafeCurrentTask { $0?.cancel() }
                return try await mixed.calibrate(candidateCount: 2, dimension: width, iterations: 1, warmupIterations: 0)
            }
            do { _ = try await mixedTask.value; Issue.record("Already-cancelled mixed calibration succeeded") }
            catch is CancellationError {}
            #expect(await mixed.workCounts == TuningWorkCounts())
            #expect(await mixed.profilesForTesting(candidateCount: 2, dimension: width).count == 1)
            #expect(try await mixed.selectStrategy(candidateCount: 2, dimension: width) == .fullFP32)
            let kernel = KernelAutoTuner(provenance: .current())
            let workload = WorkloadCharacteristics(dimension: width,batchSize: 2,precisionMode: .fp32Full)
            await kernel.setProfilesForTesting([q02Profile(width: width)], for: workload)
            try await kernel.setOverride(.cpuSoA, for: workload)
            let kernelTask = Task {
                withUnsafeCurrentTask { $0?.cancel() }
                return try await kernel.profileWorkload(workload, iterations: 1, warmupIterations: 0)
            }
            do { _ = try await kernelTask.value; Issue.record("Already-cancelled kernel profiling succeeded") }
            catch is CancellationError {}
            #expect(await kernel.workCounts == TuningWorkCounts())
            #expect(await kernel.profilesForTesting(for: workload).count == 1)
            #expect(try await kernel.selectStrategy(for: workload) == .cpuSoA)
        }
    }

    @Test func unknownHardwareExplicitlyDisablesDiskReuse() async throws {
        let current = KernelProfileProvenance.current()
        let unknown = KernelProfileProvenance(algorithmVersion: current.algorithmVersion,
            operationBoundary: current.operationBoundary, datasetVersion: current.datasetVersion,
            datasetSeeds: current.datasetSeeds, architecture: current.architecture, hardwareModel: nil,
            processorCount: current.processorCount, operatingSystem: current.operatingSystem)
        let tuner = KernelAutoTuner(persistenceURL: URL(fileURLWithPath: "/Q02-unreachable-profile.json"), provenance: unknown)
        #expect(await tuner.persistenceStatus == .disabledUnavailableHardware)
        _ = try await tuner.profileWorkload(WorkloadCharacteristics(dimension: 512,batchSize: 1,precisionMode: .fp32Full),
                                           iterations: 1,warmupIterations: 0)
        try await tuner.clearCache(removePersistedProfiles: true)
        #expect(await KernelAutoTuner().persistenceStatus == .memoryOnly)
    }
}

extension PrecisionSelectionContractTests {
    @Test func invalidKernelCalibrationDoesNoWork() async throws {
        let tuner = KernelAutoTuner(provenance: .current())
        for (count, width, iterations, warmups) in [(-1,512,1,0),(0,512,1,0),(1,256,1,0),
            (1,512,0,0),(1,512,-1,0),(1,512,1,-1),(Int.max,512,1,0),(1,512,Int.max,1)] {
            let workload = WorkloadCharacteristics(dimension: width,batchSize: count,precisionMode: .fp32Full)
            do {
                _ = try await tuner.profileWorkload(workload,iterations: iterations,warmupIterations: warmups)
                Issue.record("Invalid kernel calibration accepted")
            } catch let error as VectorError { #expect(error.kind == .invalidConfiguration) }
        }
        #expect(await tuner.workCounts == TuningWorkCounts())
    }
}
