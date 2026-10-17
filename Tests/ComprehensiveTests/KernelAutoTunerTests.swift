//
//  KernelAutoTunerTests.swift
//  VectorCore
//
//  Tests for kernel auto-tuning framework
//  Validates strategy selection, profiling, and accuracy measurement
//

import XCTest
@testable import VectorCore

final class KernelAutoTunerTests: XCTestCase {

    // MARK: - Strategy Selection Heuristics

    @MainActor
    func testSmallBatchSelectionFP32() async throws {
        let tuner = KernelAutoTuner()
        // Small batches with FP32 should select cpuAoS (minimal overhead)
        let workload = WorkloadCharacteristics(
            dimension: 512,
            batchSize: 50,
            precisionMode: .fp32Full
        )

        let strategy = try await tuner.selectStrategy(for: workload)
        XCTAssertEqual(strategy, .cpuAoS, "Small FP32 batches should use cpuAoS")
    }

    @MainActor
    func testSmallBatchSelectionFP16() async throws {
        let tuner = KernelAutoTuner()
        // Small batches with FP16 acceptable should use mixed precision
        let workload = WorkloadCharacteristics(
            dimension: 512,
            batchSize: 50,
            precisionMode: .fp16Acceptable
        )

        let strategy = try await tuner.selectStrategy(for: workload)
        XCTAssertEqual(strategy, .cpuMixedPrecision, "Small FP16 batches should use mixed precision")
    }

    @MainActor
    func testMediumBatchSelectionFP32() async throws {
        let tuner = KernelAutoTuner()
        // Medium batches with FP32 should select SoA
        let workload = WorkloadCharacteristics(
            dimension: 512,
            batchSize: 500,
            precisionMode: .fp32Full
        )

        let strategy = try await tuner.selectStrategy(for: workload)
        XCTAssertEqual(strategy, .cpuSoA, "Medium FP32 batches should use SoA")
    }

    @MainActor
    func testMediumBatchSelectionINT8() async throws {
        let tuner = KernelAutoTuner()
        // Medium batches with INT8 acceptable should use quantized
        let workload = WorkloadCharacteristics(
            dimension: 512,
            batchSize: 500,
            precisionMode: .int8Acceptable
        )

        let strategy = try await tuner.selectStrategy(for: workload)
        XCTAssertEqual(strategy, .cpuQuantized, "Medium INT8 batches should use quantized")
    }

    @MainActor
    func testLargeBatchSelectionFP32() async throws {
        let tuner = KernelAutoTuner()
        // Large batches with FP32 should select SoA
        let workload = WorkloadCharacteristics(
            dimension: 512,
            batchSize: 5000,
            precisionMode: .fp32Full
        )

        let strategy = try await tuner.selectStrategy(for: workload)
        XCTAssertEqual(strategy, .cpuSoA, "Large FP32 batches should use SoA")
    }

    @MainActor
    func testLargeBatchSelectionFP16() async throws {
        let tuner = KernelAutoTuner()
        // Large batches with FP16 acceptable should use mixed precision
        let workload = WorkloadCharacteristics(
            dimension: 512,
            batchSize: 5000,
            precisionMode: .fp16Acceptable
        )

        let strategy = try await tuner.selectStrategy(for: workload)
        XCTAssertEqual(strategy, .cpuMixedPrecision, "Large FP16 batches should use mixed precision")
    }

    // MARK: - Multi-Dimensional Support

    @MainActor
    func testDimension768Selection() async throws {
        let tuner = KernelAutoTuner()
        let workload = WorkloadCharacteristics(
            dimension: 768,
            batchSize: 500,
            precisionMode: .fp32Full
        )

        let strategy = try await tuner.selectStrategy(for: workload)
        XCTAssertEqual(strategy, .cpuSoA, "768-dim vectors should follow same heuristics")
    }

    @MainActor
    func testDimension1536Selection() async throws {
        let tuner = KernelAutoTuner()
        let workload = WorkloadCharacteristics(
            dimension: 1536,
            batchSize: 100,
            precisionMode: .fp16Acceptable
        )

        let strategy = try await tuner.selectStrategy(for: workload)
        XCTAssertEqual(strategy, .cpuMixedPrecision, "1536-dim vectors should follow same heuristics")
    }

    // MARK: - Manual Overrides

    @MainActor
    func testManualOverridePersists() async throws {
        let tuner = KernelAutoTuner()
        let workload = WorkloadCharacteristics(
            dimension: 512,
            batchSize: 500,
            precisionMode: .fp32Full
        )

        // Set override
        try await tuner.setOverride(.cpuSoA, for: workload)

        // Verify override is respected
        let selected = try await tuner.selectStrategy(for: workload)
        XCTAssertEqual(selected, .cpuSoA, "Manual override should be respected")

        // Cleanup: remove override to avoid influencing other tests
        await tuner.removeOverride(for: workload)
    }

    // MARK: - Profiling

    @MainActor
    func testProfileWorkload512() async throws {
        let tuner = KernelAutoTuner()
        let workload = WorkloadCharacteristics(
            dimension: 512,
            batchSize: 100,
            precisionMode: .fp16Acceptable
        )

        // Profile with minimal iterations for test speed
        let profiles = try await tuner.profileWorkload(
            workload,
            iterations: 10,
            warmupIterations: 2
        )

        XCTAssertFalse(profiles.isEmpty, "Should generate at least one profile")

        // Verify all profiles have valid data
        for profile in profiles {
            expectXCTestPerformance("KernelAutoTunerTests.testProfileWorkload512.1", profile.throughput > 0, "Throughput should be positive")
            expectXCTestPerformance("KernelAutoTunerTests.testProfileWorkload512.2", profile.latencyMean > 0, "Latency should be positive")
            XCTAssertGreaterThan(profile.estimatedPayloadBytes, 0, "Memory footprint should be positive")
            XCTAssertGreaterThanOrEqual(profile.syntheticAccuracyScore, 0, "Accuracy should be non-negative")
            XCTAssertLessThanOrEqual(profile.syntheticAccuracyScore, 1.0, "Accuracy should be <= 1.0")
        }

        // Verify profiles are sorted by score (descending)
        for i in 0..<(profiles.count - 1) {
            XCTAssertGreaterThanOrEqual(
                profiles[i].score,
                profiles[i + 1].score,
                "Profiles should be sorted by score (descending)"
            )
        }
    }

    @MainActor
    func testProfileWorkload768() async throws {
        let tuner = KernelAutoTuner()
        let workload = WorkloadCharacteristics(
            dimension: 768,
            batchSize: 50,
            precisionMode: .fp32Full
        )

        let profiles = try await tuner.profileWorkload(
            workload,
            iterations: 10,
            warmupIterations: 2
        )

        XCTAssertFalse(profiles.isEmpty, "768-dim profiling should succeed")
    }

    @MainActor
    func testProfileWorkload1536() async throws {
        let tuner = KernelAutoTuner()
        let workload = WorkloadCharacteristics(
            dimension: 1536,
            batchSize: 50,
            precisionMode: .fp16Acceptable
        )

        let profiles = try await tuner.profileWorkload(
            workload,
            iterations: 10,
            warmupIterations: 2
        )

        XCTAssertFalse(profiles.isEmpty, "1536-dim profiling should succeed")
    }

    // MARK: - Accuracy Measurement

    @MainActor
    func testAccuracyBaseline() async throws {
        let tuner = KernelAutoTuner()
        // cpuAoS should have perfect accuracy (1.0) against itself
        let workload = WorkloadCharacteristics(
            dimension: 512,
            batchSize: 50,
            precisionMode: .fp32Full
        )

        let profiles = try await tuner.profileWorkload(
            workload,
            iterations: 5,
            warmupIterations: 1
        )

        let aosProfile = profiles.first { $0.strategy == .cpuAoS }
        XCTAssertNotNil(aosProfile, "cpuAoS should be profiled")
        XCTAssertEqual(Double(aosProfile?.syntheticAccuracyScore ?? 0), 1.0, accuracy: 0.01, "cpuAoS should have perfect accuracy")
    }

    @MainActor
    func testAccuracyMixedPrecision() async throws {
        let tuner = KernelAutoTuner()
        // Mixed precision should have high accuracy (>0.99 for this deterministic synthetic dataset)
        let workload = WorkloadCharacteristics(
            dimension: 512,
            batchSize: 50,
            precisionMode: .fp16Acceptable
        )

        let profiles = try await tuner.profileWorkload(
            workload,
            iterations: 5,
            warmupIterations: 1
        )

        let mixedProfile = profiles.first { $0.strategy == .cpuMixedPrecision }
        XCTAssertNotNil(mixedProfile, "cpuMixedPrecision should be profiled")
        XCTAssertGreaterThan(mixedProfile?.syntheticAccuracyScore ?? 0, 0.99, "Mixed precision should have >99% accuracy")
    }

    @MainActor
    func testAccuracyQuantized() async throws {
        let tuner = KernelAutoTuner()
        // Quantized should have reasonable accuracy (>0.95)
        let workload = WorkloadCharacteristics(
            dimension: 512,
            batchSize: 50,
            precisionMode: .int8Acceptable
        )

        let profiles = try await tuner.profileWorkload(
            workload,
            iterations: 5,
            warmupIterations: 1
        )

        let quantizedProfile = profiles.first { $0.strategy == .cpuQuantized }
        XCTAssertNotNil(quantizedProfile, "cpuQuantized should be profiled")
        XCTAssertGreaterThan(quantizedProfile?.syntheticAccuracyScore ?? 0, 0.95, "Quantized should have >95% accuracy")
    }

    // MARK: - Memory Estimation

    @MainActor
    func testMemoryEstimationFP32() async throws {
        let tuner = KernelAutoTuner()
        // FP32: (N+1) * D * 4 bytes
        let workload = WorkloadCharacteristics(
            dimension: 512,
            batchSize: 100,
            precisionMode: .fp32Full
        )

        // Access the auto-tuner's internal estimation (via profiling)
            let profiles = try await tuner.profileWorkload(
                workload,
                iterations: 1,
                warmupIterations: 0
            )

            let fp32Profile = profiles.first { $0.strategy == .cpuAoS }
            let expectedMemory = (100 + 1) * 512 * 4  // 206,848 bytes

            XCTAssertEqual(
                fp32Profile?.estimatedPayloadBytes,
                expectedMemory,
                "FP32 memory estimation should be accurate"
            )
    }

    @MainActor
    func testMemoryEstimationMixedPrecision() async throws {
        let tuner = KernelAutoTuner()
        // Mixed: Query FP32 + Candidates FP16 = D*4 + N*D*2
        let workload = WorkloadCharacteristics(
            dimension: 512,
            batchSize: 100,
            precisionMode: .fp16Acceptable
        )

            let profiles = try await tuner.profileWorkload(
                workload,
                iterations: 1,
                warmupIterations: 0
            )

            let mixedProfile = profiles.first { $0.strategy == .cpuMixedPrecision }
            let expectedMemory = 512 * 4 + 100 * 512 * 2  // 104,448 bytes

            XCTAssertEqual(
                mixedProfile?.estimatedPayloadBytes,
                expectedMemory,
                "Mixed precision memory estimation should be accurate"
            )
    }

    @MainActor
    func testMemoryEstimationQuantized() async throws {
        let tuner = KernelAutoTuner()
        // INT8: (N+1) * D * 1 byte
        let workload = WorkloadCharacteristics(
            dimension: 512,
            batchSize: 100,
            precisionMode: .int8Acceptable
        )

            let profiles = try await tuner.profileWorkload(
                workload,
                iterations: 1,
                warmupIterations: 0
            )

            let quantizedProfile = profiles.first { $0.strategy == .cpuQuantized }
            let expectedMemory = (100 + 1) * 512 * 1  // 51,712 bytes

            XCTAssertEqual(
                quantizedProfile?.estimatedPayloadBytes,
                expectedMemory,
                "Quantized memory estimation should be accurate"
            )
    }

    // MARK: - Performance Profile Score

    func testScoreCalculation() {
        let tuner = KernelAutoTuner()
        let profile = PerformanceProfile(
            strategy: .cpuAoS,
            workload: KernelProfileWorkload(dimension: 512, batchSize: 100, precisionMode: .fp32Full),
            throughput: 1000,        // 1000 ops/sec
            latencyMean: 0.001,      // 1ms
            latencyP50: 0.001,
            latencyP95: 0.0015,
            latencyP99: 0.002,
            estimatedPayloadBytes: 200_000, // 200KB
            syntheticAccuracyScore: 1.0,            // Perfect
            measuredSamples: 1, warmupIterations: 0, provenance: KernelProfileProvenance.current(),
            timestamp: Date()
        )

        let score = profile.score
        XCTAssertFalse(score.isInfinite, "Score should be finite")
        XCTAssertFalse(score.isNaN, "Score should not be NaN")
        XCTAssertGreaterThan(score, 0, "Score should be positive for good profile")
    }

    func testScoreHandlesZeroThroughput() {
        let tuner = KernelAutoTuner()
        let badProfile = PerformanceProfile(
            strategy: .cpuAoS,
            workload: KernelProfileWorkload(dimension: 512, batchSize: 100, precisionMode: .fp32Full),
            throughput: 0,  // Invalid
            latencyMean: 0.001,
            latencyP50: 0.001,
            latencyP95: 0.001,
            latencyP99: 0.001,
            estimatedPayloadBytes: 200_000,
            syntheticAccuracyScore: 1.0,
            measuredSamples: 1, warmupIterations: 0, provenance: KernelProfileProvenance.current(),
            timestamp: Date()
        )

        XCTAssertEqual(badProfile.score, -Double.infinity, "Zero throughput should result in -inf score")
    }

    // MARK: - Edge Cases

    @MainActor
    func testUnsupportedDimension() async throws {
        let tuner = KernelAutoTuner()
        let workload = WorkloadCharacteristics(
            dimension: 256,  // Unsupported
            batchSize: 100,
            precisionMode: .fp32Full
        )

        do {
            _ = try await tuner.profileWorkload(workload, iterations: 1, warmupIterations: 0)
            XCTFail("Unsupported dimension must throw")
        } catch let error as VectorError {
            XCTAssertEqual(error.kind, .invalidConfiguration)
        }

    }

    @MainActor
    func testZeroBatchSize() async throws {
        let tuner = KernelAutoTuner()
        let workload = WorkloadCharacteristics(
            dimension: 512,
            batchSize: 0,
            precisionMode: .fp32Full
        )

        // Strategy selection should still work (heuristic)
        let strategy = try await tuner.selectStrategy(for: workload)
        XCTAssertEqual(strategy, .cpuAoS, "Zero batch should fall through to cpuAoS")
    }

    // MARK: - Workload Hash Consistency

    @MainActor
    func testWorkloadHashConsistency() async throws {
        let tuner = KernelAutoTuner()
        let workload1 = WorkloadCharacteristics(
            dimension: 512,
            batchSize: 100,
            precisionMode: .fp32Full
        )

        let workload2 = WorkloadCharacteristics(
            dimension: 512,
            batchSize: 100,
            precisionMode: .fp32Full
        )

        // Set override for workload1
        try await tuner.setOverride(.cpuSoA, for: workload1)

        // workload2 should get the same override (same hash)
        let strategy = try await tuner.selectStrategy(for: workload2)
        XCTAssertEqual(strategy, .cpuSoA, "Identical workloads should share overrides")
    }

    // MARK: - Strategy Count Validation

    @MainActor
    func testAvailableStrategiesCount() async throws {
        let tuner = KernelAutoTuner()
        // FP32: Should have 2 strategies (cpuAoS, cpuSoA)
        let fp32Workload = WorkloadCharacteristics(
            dimension: 512,
            batchSize: 100,
            precisionMode: .fp32Full
        )

        let fp32Profiles = try await tuner.profileWorkload(
            fp32Workload,
            iterations: 1,
            warmupIterations: 0
        )

        XCTAssertEqual(fp32Profiles.count, 2, "FP32 should have 2 strategies")

        // FP16 acceptable: Should have 3 strategies (cpuAoS, cpuSoA, cpuMixedPrecision)
        let fp16Workload = WorkloadCharacteristics(
            dimension: 512,
            batchSize: 100,
            precisionMode: .fp16Acceptable
        )

        let fp16Profiles = try await tuner.profileWorkload(
            fp16Workload,
            iterations: 1,
            warmupIterations: 0
        )

        XCTAssertEqual(fp16Profiles.count, 3, "FP16 should have 3 strategies")

        // INT8 acceptable: Should have 4 strategies (all)
        let int8Workload = WorkloadCharacteristics(
            dimension: 512,
            batchSize: 100,
            precisionMode: .int8Acceptable
        )

        let int8Profiles = try await tuner.profileWorkload(
            int8Workload,
            iterations: 1,
            warmupIterations: 0
        )

        XCTAssertEqual(int8Profiles.count, 4, "INT8 should have 4 strategies")
    }
}
