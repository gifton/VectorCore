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

    func testSmallBatchSelectionFP32() async {
        // Small batches with FP32 should select cpuAoS (minimal overhead)
        let workload = WorkloadCharacteristics(
            dimension: 512,
            batchSize: 50,
            precisionMode: .fp32Full
        )

        let strategy = await KernelAutoTuner.shared.selectStrategy(for: workload)
        XCTAssertEqual(strategy, .cpuAoS, "Small FP32 batches should use cpuAoS")
    }

    func testSmallBatchSelectionFP16() async {
        // Small batches with FP16 acceptable should use mixed precision
        let workload = WorkloadCharacteristics(
            dimension: 512,
            batchSize: 50,
            precisionMode: .fp16Acceptable
        )

        let strategy = await KernelAutoTuner.shared.selectStrategy(for: workload)
        XCTAssertEqual(strategy, .cpuMixedPrecision, "Small FP16 batches should use mixed precision")
    }

    func testMediumBatchSelectionFP32() async {
        // Medium batches with FP32 should select SoA
        let workload = WorkloadCharacteristics(
            dimension: 512,
            batchSize: 500,
            precisionMode: .fp32Full
        )

        let strategy = await KernelAutoTuner.shared.selectStrategy(for: workload)
        XCTAssertEqual(strategy, .cpuSoA, "Medium FP32 batches should use SoA")
    }

    func testMediumBatchSelectionINT8() async {
        // Medium batches with INT8 acceptable should use quantized
        let workload = WorkloadCharacteristics(
            dimension: 512,
            batchSize: 500,
            precisionMode: .int8Acceptable
        )

        let strategy = await KernelAutoTuner.shared.selectStrategy(for: workload)
        XCTAssertEqual(strategy, .cpuQuantized, "Medium INT8 batches should use quantized")
    }

    func testLargeBatchSelectionFP32() async {
        // Large batches with FP32 should select SoA
        let workload = WorkloadCharacteristics(
            dimension: 512,
            batchSize: 5000,
            precisionMode: .fp32Full
        )

        let strategy = await KernelAutoTuner.shared.selectStrategy(for: workload)
        XCTAssertEqual(strategy, .cpuSoA, "Large FP32 batches should use SoA")
    }

    func testLargeBatchSelectionFP16() async {
        // Large batches with FP16 acceptable should use mixed precision
        let workload = WorkloadCharacteristics(
            dimension: 512,
            batchSize: 5000,
            precisionMode: .fp16Acceptable
        )

        let strategy = await KernelAutoTuner.shared.selectStrategy(for: workload)
        XCTAssertEqual(strategy, .cpuMixedPrecision, "Large FP16 batches should use mixed precision")
    }

    // MARK: - Multi-Dimensional Support

    func testDimension768Selection() async {
        let workload = WorkloadCharacteristics(
            dimension: 768,
            batchSize: 500,
            precisionMode: .fp32Full
        )

        let strategy = await KernelAutoTuner.shared.selectStrategy(for: workload)
        XCTAssertEqual(strategy, .cpuSoA, "768-dim vectors should follow same heuristics")
    }

    func testDimension1536Selection() async {
        let workload = WorkloadCharacteristics(
            dimension: 1536,
            batchSize: 100,
            precisionMode: .fp16Acceptable
        )

        let strategy = await KernelAutoTuner.shared.selectStrategy(for: workload)
        XCTAssertEqual(strategy, .cpuMixedPrecision, "1536-dim vectors should follow same heuristics")
    }

    // MARK: - Manual Overrides

    func testManualOverridePersists() async {
        let workload = WorkloadCharacteristics(
            dimension: 512,
            batchSize: 500,
            precisionMode: .fp32Full
        )

        // Set override
        await KernelAutoTuner.shared.setOverride(.cpuQuantized, for: workload)

        // Verify override is respected
        let selected = await KernelAutoTuner.shared.selectStrategy(for: workload)
        XCTAssertEqual(selected, .cpuQuantized, "Manual override should be respected")
    }

    // MARK: - Profiling

    func testProfileWorkload512() async {
        let workload = WorkloadCharacteristics(
            dimension: 512,
            batchSize: 100,
            precisionMode: .fp16Acceptable
        )

        // Profile with minimal iterations for test speed
        let profiles = await KernelAutoTuner.shared.profileWorkload(
            workload,
            iterations: 10,
            warmupIterations: 2
        )

        XCTAssertFalse(profiles.isEmpty, "Should generate at least one profile")

        // Verify all profiles have valid data
        for profile in profiles {
            XCTAssertGreaterThan(profile.throughput, 0, "Throughput should be positive")
            XCTAssertGreaterThan(profile.latencyMean, 0, "Latency should be positive")
            XCTAssertGreaterThan(profile.memoryFootprint, 0, "Memory footprint should be positive")
            XCTAssertGreaterThanOrEqual(profile.accuracy, 0, "Accuracy should be non-negative")
            XCTAssertLessThanOrEqual(profile.accuracy, 1.0, "Accuracy should be <= 1.0")
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

    func testProfileWorkload768() async {
        let workload = WorkloadCharacteristics(
            dimension: 768,
            batchSize: 50,
            precisionMode: .fp32Full
        )

        let profiles = await KernelAutoTuner.shared.profileWorkload(
            workload,
            iterations: 10,
            warmupIterations: 2
        )

        XCTAssertFalse(profiles.isEmpty, "768-dim profiling should succeed")
    }

    func testProfileWorkload1536() async {
        let workload = WorkloadCharacteristics(
            dimension: 1536,
            batchSize: 50,
            precisionMode: .fp16Acceptable
        )

        let profiles = await KernelAutoTuner.shared.profileWorkload(
            workload,
            iterations: 10,
            warmupIterations: 2
        )

        XCTAssertFalse(profiles.isEmpty, "1536-dim profiling should succeed")
    }

    // MARK: - Accuracy Measurement

    func testAccuracyBaseline() async {
        // cpuAoS should have perfect accuracy (1.0) against itself
        let workload = WorkloadCharacteristics(
            dimension: 512,
            batchSize: 50,
            precisionMode: .fp32Full
        )

        let profiles = await KernelAutoTuner.shared.profileWorkload(
            workload,
            iterations: 5,
            warmupIterations: 1
        )

        let aosProfile = profiles.first { $0.strategy == .cpuAoS }
        XCTAssertNotNil(aosProfile, "cpuAoS should be profiled")
        XCTAssertEqual(aosProfile?.accuracy, 1.0, accuracy: 0.01, "cpuAoS should have perfect accuracy")
    }

    func testAccuracyMixedPrecision() async {
        // Mixed precision should have high accuracy (>0.99 for typical embeddings)
        let workload = WorkloadCharacteristics(
            dimension: 512,
            batchSize: 50,
            precisionMode: .fp16Acceptable
        )

        let profiles = await KernelAutoTuner.shared.profileWorkload(
            workload,
            iterations: 5,
            warmupIterations: 1
        )

        let mixedProfile = profiles.first { $0.strategy == .cpuMixedPrecision }
        XCTAssertNotNil(mixedProfile, "cpuMixedPrecision should be profiled")
        XCTAssertGreaterThan(mixedProfile?.accuracy ?? 0, 0.99, "Mixed precision should have >99% accuracy")
    }

    func testAccuracyQuantized() async {
        // Quantized should have reasonable accuracy (>0.95)
        let workload = WorkloadCharacteristics(
            dimension: 512,
            batchSize: 50,
            precisionMode: .int8Acceptable
        )

        let profiles = await KernelAutoTuner.shared.profileWorkload(
            workload,
            iterations: 5,
            warmupIterations: 1
        )

        let quantizedProfile = profiles.first { $0.strategy == .cpuQuantized }
        XCTAssertNotNil(quantizedProfile, "cpuQuantized should be profiled")
        XCTAssertGreaterThan(quantizedProfile?.accuracy ?? 0, 0.95, "Quantized should have >95% accuracy")
    }

    // MARK: - Memory Estimation

    func testMemoryEstimationFP32() {
        // FP32: (N+1) * D * 4 bytes
        let workload = WorkloadCharacteristics(
            dimension: 512,
            batchSize: 100,
            precisionMode: .fp32Full
        )

        // Access the auto-tuner's internal estimation (via profiling)
        Task {
            let profiles = await KernelAutoTuner.shared.profileWorkload(
                workload,
                iterations: 1,
                warmupIterations: 0
            )

            let fp32Profile = profiles.first { $0.strategy == .cpuAoS }
            let expectedMemory = (100 + 1) * 512 * 4  // 206,848 bytes

            XCTAssertEqual(
                fp32Profile?.memoryFootprint,
                expectedMemory,
                "FP32 memory estimation should be accurate"
            )
        }
    }

    func testMemoryEstimationMixedPrecision() {
        // Mixed: Query FP32 + Candidates FP16 = D*4 + N*D*2
        let workload = WorkloadCharacteristics(
            dimension: 512,
            batchSize: 100,
            precisionMode: .fp16Acceptable
        )

        Task {
            let profiles = await KernelAutoTuner.shared.profileWorkload(
                workload,
                iterations: 1,
                warmupIterations: 0
            )

            let mixedProfile = profiles.first { $0.strategy == .cpuMixedPrecision }
            let expectedMemory = 512 * 4 + 100 * 512 * 2  // 104,448 bytes

            XCTAssertEqual(
                mixedProfile?.memoryFootprint,
                expectedMemory,
                "Mixed precision memory estimation should be accurate"
            )
        }
    }

    func testMemoryEstimationQuantized() {
        // INT8: (N+1) * D * 1 byte
        let workload = WorkloadCharacteristics(
            dimension: 512,
            batchSize: 100,
            precisionMode: .int8Acceptable
        )

        Task {
            let profiles = await KernelAutoTuner.shared.profileWorkload(
                workload,
                iterations: 1,
                warmupIterations: 0
            )

            let quantizedProfile = profiles.first { $0.strategy == .cpuQuantized }
            let expectedMemory = (100 + 1) * 512 * 1  // 51,712 bytes

            XCTAssertEqual(
                quantizedProfile?.memoryFootprint,
                expectedMemory,
                "Quantized memory estimation should be accurate"
            )
        }
    }

    // MARK: - Performance Profile Score

    func testScoreCalculation() {
        let profile = PerformanceProfile(
            strategy: .cpuAoS,
            workload: "512_100_fp32Full",
            throughput: 1000,        // 1000 ops/sec
            latencyMean: 0.001,      // 1ms
            latencyP50: 0.001,
            latencyP95: 0.0015,
            latencyP99: 0.002,
            memoryFootprint: 200_000, // 200KB
            accuracy: 1.0,            // Perfect
            timestamp: Date()
        )

        let score = profile.score
        XCTAssertFalse(score.isInfinite, "Score should be finite")
        XCTAssertFalse(score.isNaN, "Score should not be NaN")
        XCTAssertGreaterThan(score, 0, "Score should be positive for good profile")
    }

    func testScoreHandlesZeroThroughput() {
        let badProfile = PerformanceProfile(
            strategy: .cpuAoS,
            workload: "512_100_fp32Full",
            throughput: 0,  // Invalid
            latencyMean: 0.001,
            latencyP50: 0.001,
            latencyP95: 0.001,
            latencyP99: 0.001,
            memoryFootprint: 200_000,
            accuracy: 1.0,
            timestamp: Date()
        )

        XCTAssertEqual(badProfile.score, -Double.infinity, "Zero throughput should result in -inf score")
    }

    // MARK: - Edge Cases

    func testUnsupportedDimension() async {
        let workload = WorkloadCharacteristics(
            dimension: 256,  // Unsupported
            batchSize: 100,
            precisionMode: .fp32Full
        )

        let profiles = await KernelAutoTuner.shared.profileWorkload(
            workload,
            iterations: 1,
            warmupIterations: 0
        )

        XCTAssertTrue(profiles.isEmpty, "Unsupported dimension should return empty profiles")
    }

    func testZeroBatchSize() async {
        let workload = WorkloadCharacteristics(
            dimension: 512,
            batchSize: 0,
            precisionMode: .fp32Full
        )

        // Strategy selection should still work (heuristic)
        let strategy = await KernelAutoTuner.shared.selectStrategy(for: workload)
        XCTAssertEqual(strategy, .cpuAoS, "Zero batch should fall through to cpuAoS")
    }

    // MARK: - Workload Hash Consistency

    func testWorkloadHashConsistency() async {
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
        await KernelAutoTuner.shared.setOverride(.cpuQuantized, for: workload1)

        // workload2 should get the same override (same hash)
        let strategy = await KernelAutoTuner.shared.selectStrategy(for: workload2)
        XCTAssertEqual(strategy, .cpuQuantized, "Identical workloads should share overrides")
    }

    // MARK: - Strategy Count Validation

    func testAvailableStrategiesCount() async {
        // FP32: Should have 2 strategies (cpuAoS, cpuSoA)
        let fp32Workload = WorkloadCharacteristics(
            dimension: 512,
            batchSize: 100,
            precisionMode: .fp32Full
        )

        let fp32Profiles = await KernelAutoTuner.shared.profileWorkload(
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

        let fp16Profiles = await KernelAutoTuner.shared.profileWorkload(
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

        let int8Profiles = await KernelAutoTuner.shared.profileWorkload(
            int8Workload,
            iterations: 1,
            warmupIterations: 0
        )

        XCTAssertEqual(int8Profiles.count, 4, "INT8 should have 4 strategies")
    }
}
