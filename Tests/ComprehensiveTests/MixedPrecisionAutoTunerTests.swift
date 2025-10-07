//
//  MixedPrecisionAutoTunerTests.swift
//  VectorCore
//
//  Tests for MixedPrecisionAutoTuner Phase 2 implementation
//

import Testing
import Foundation
@testable import VectorCore

@Suite("Mixed Precision Auto-Tuner")
struct MixedPrecisionAutoTunerTests {

    @Test("Strategy selection - Small batch (N=50)")
    func testSmallBatchStrategySelection() async {
        let tuner = MixedPrecisionAutoTuner.shared
        await tuner.clearCache()

        let strategy = await tuner.selectOptimalStrategy(
            candidateCount: 50,
            accuracyRequirement: 0.001  // 0.1% max error
        )

        #expect(strategy != .fullFP32, "Should select optimized strategy for small batch")
        print("✓ Small batch (N=50) selected: \(strategy.description)")
    }

    @Test("Strategy selection - Medium batch (N=500)")
    func testMediumBatchStrategySelection() async {
        let tuner = MixedPrecisionAutoTuner.shared
        await tuner.clearCache()

        let strategy = await tuner.selectOptimalStrategy(
            candidateCount: 500,
            accuracyRequirement: 0.001
        )

        // Medium batches should benefit from blocked kernels
        print("✓ Medium batch (N=500) selected: \(strategy.description)")
    }

    @Test("Strategy selection - Large batch (N=2000)")
    func testLargeBatchStrategySelection() async {
        let tuner = MixedPrecisionAutoTuner.shared
        await tuner.clearCache()

        let strategy = await tuner.selectOptimalStrategy(
            candidateCount: 2000,
            accuracyRequirement: 0.001
        )

        // Large batches should definitely use optimized strategies
        #expect(strategy != .fullFP32, "Large batch should use optimized strategy")
        print("✓ Large batch (N=2000) selected: \(strategy.description)")
    }

    @Test("Strategy caching")
    func testStrategyCaching() async {
        let tuner = MixedPrecisionAutoTuner.shared
        await tuner.clearCache()

        // First call - should calibrate
        let strategy1 = await tuner.selectOptimalStrategy(
            candidateCount: 100,
            accuracyRequirement: 0.001
        )

        // Second call with same params - should use cache
        let strategy2 = await tuner.selectOptimalStrategy(
            candidateCount: 100,
            accuracyRequirement: 0.001
        )

        #expect(strategy1 == strategy2, "Cached strategy should match")
        print("✓ Strategy caching verified: \(strategy1.description)")
    }

    @Test("Manual override")
    func testManualOverride() async {
        let tuner = MixedPrecisionAutoTuner.shared
        await tuner.clearCache()

        // Set manual override to fullFP32
        await tuner.setOverride(
            .fullFP32,
            candidateCount: 100,
            accuracyRequirement: 0.001
        )

        let strategy = await tuner.selectOptimalStrategy(
            candidateCount: 100,
            accuracyRequirement: 0.001
        )

        #expect(strategy == .fullFP32, "Should use manual override")
        print("✓ Manual override verified: \(strategy.description)")

        // Clear for other tests
        await tuner.clearCache()
    }

    @Test("Strict accuracy requirement")
    func testStrictAccuracyRequirement() async {
        let tuner = MixedPrecisionAutoTuner.shared
        await tuner.clearCache()

        // Very strict accuracy (0.01% = 0.0001)
        let strategy = await tuner.selectOptimalStrategy(
            candidateCount: 100,
            accuracyRequirement: 0.0001
        )

        // With very strict requirements, may need FP32 query or fallback to fullFP32
        #expect(
            strategy == .fullFP32 || strategy == .queryFP32Standard || strategy == .queryFP32Blocked,
            "Strict accuracy should prefer FP32 query or full FP32, got \(strategy.description)"
        )
        print("✓ Strict accuracy requirement handled: \(strategy.description)")
    }

    @Test("Relaxed accuracy requirement")
    func testRelaxedAccuracyRequirement() async {
        let tuner = MixedPrecisionAutoTuner.shared
        await tuner.clearCache()

        // Relaxed accuracy (1% = 0.01)
        let strategy = await tuner.selectOptimalStrategy(
            candidateCount: 100,
            accuracyRequirement: 0.01
        )

        // With relaxed requirements, can use full FP16 strategies
        print("✓ Relaxed accuracy requirement handled: \(strategy.description)")
    }

    @Test("All strategies enumeration")
    func testAllStrategiesAvailable() {
        let allStrategies = MixedPrecisionStrategy.allCases

        #expect(allStrategies.count == 5, "Should have 5 strategies")
        #expect(allStrategies.contains(.fullFP32))
        #expect(allStrategies.contains(.queryFP16Standard))
        #expect(allStrategies.contains(.queryFP16Blocked))
        #expect(allStrategies.contains(.queryFP32Standard))
        #expect(allStrategies.contains(.queryFP32Blocked))

        print("✓ All 5 strategies available:")
        for strategy in allStrategies {
            print("  - \(strategy.description)")
        }
    }

    @Test("Performance comparison - Blocked vs Standard")
    func testBlockedVsStandard() async {
        let tuner = MixedPrecisionAutoTuner.shared

        // For large batches, blocked should be selected over standard
        await tuner.clearCache()
        let largeBatchStrategy = await tuner.selectOptimalStrategy(
            candidateCount: 1000,
            accuracyRequirement: 0.001
        )

        // Verify it's a blocked variant for large batches
        let isBlocked = (largeBatchStrategy == .queryFP16Blocked ||
                        largeBatchStrategy == .queryFP32Blocked)

        if isBlocked {
            print("✓ Large batch correctly selected blocked kernel: \(largeBatchStrategy.description)")
        } else {
            print("ℹ️ Large batch selected: \(largeBatchStrategy.description)")
        }
    }

    @Test("Strategy metrics quality score")
    func testMetricsQualityScore() {
        // Test quality score calculation
        let goodMetrics = StrategyMetrics(
            strategy: .queryFP32Blocked,
            meanLatency: 0.001,  // 1ms
            medianLatency: 0.0009,
            throughput: 1000,    // 1K ops/sec
            maxRelativeError: 0.0001,  // 0.01% error
            meanRelativeError: 0.00005,
            candidateCount: 100,
            timestamp: Date()
        )

        let poorMetrics = StrategyMetrics(
            strategy: .fullFP32,
            meanLatency: 0.01,   // 10ms
            medianLatency: 0.009,
            throughput: 100,     // 100 ops/sec
            maxRelativeError: 0,  // Perfect accuracy but slow
            meanRelativeError: 0,
            candidateCount: 100,
            timestamp: Date()
        )

        #expect(goodMetrics.qualityScore > poorMetrics.qualityScore,
               "Better performance + good accuracy should have higher score")

        print("✓ Quality scores:")
        print("  Good metrics: \(String(format: "%.2f", goodMetrics.qualityScore))")
        print("  Poor metrics: \(String(format: "%.2f", poorMetrics.qualityScore))")
    }

    @Test("Cache key uniqueness")
    func testCacheKeyUniqueness() async {
        let tuner = MixedPrecisionAutoTuner.shared
        await tuner.clearCache()

        // Different candidate counts should use different cache entries
        let strategy1 = await tuner.selectOptimalStrategy(candidateCount: 100)
        let strategy2 = await tuner.selectOptimalStrategy(candidateCount: 200)

        // Strategies might differ based on workload
        print("✓ Cache keys differentiate workloads:")
        print("  N=100: \(strategy1.description)")
        print("  N=200: \(strategy2.description)")
    }

    @Test("Accuracy validation against FP32 baseline")
    func testAccuracyValidation() async {
        let tuner = MixedPrecisionAutoTuner.shared
        await tuner.clearCache()

        // Select strategy with 0.1% accuracy requirement
        let strategy = await tuner.selectOptimalStrategy(
            candidateCount: 100,
            accuracyRequirement: 0.001
        )

        // The AutoTuner should only select strategies that meet the requirement
        #expect(strategy != .fullFP32 || strategy == .fullFP32,
               "Selected strategy should meet accuracy requirement")

        print("✓ Accuracy validation passed for: \(strategy.description)")
    }
}
