
import Testing
import Foundation
@testable import VectorCore

@Suite("Mixed Precision Auto-Tuner")
struct MixedPrecisionAutoTunerTests {

    @Test("Strategy selection - Small batch (N=50)")
    func testSmallBatchStrategySelection() async throws {
        let tuner = MixedPrecisionAutoTuner()
        await tuner.clearCache()

        let strategy = try await tuner.selectStrategy(
            candidateCount: 50,
            maximumSyntheticRelativeError: 0.001  // Synthetic relative-error filter
        )
        #expect(strategy == .fullFP32, "Should return a valid strategy for small batch")
        print("✓ Small batch (N=50) selected: \(strategy.description)")
    }

    @Test("Strategy selection - Medium batch (N=500)")
    func testMediumBatchStrategySelection() async throws {
        let tuner = MixedPrecisionAutoTuner()
        await tuner.clearCache()

        let strategy = try await tuner.selectStrategy(
            candidateCount: 500,
            maximumSyntheticRelativeError: 0.001
        )
        #expect(strategy == .fullFP32)
        print("✓ Medium batch (N=500) selected: \(strategy.description)")
    }

    @Test("Strategy selection - Large batch (N=2000)")
    func testLargeBatchStrategySelection() async throws {
        let tuner = MixedPrecisionAutoTuner()
        await tuner.clearCache()

        let strategy = try await tuner.selectStrategy(
            candidateCount: 2000,
            maximumSyntheticRelativeError: 0.001
        )
        #expect(strategy == .fullFP32, "Large batch should return a valid strategy")
        print("✓ Large batch (N=2000) selected: \(strategy.description)")
    }

    @Test("Repeated unprofiled lookup")
    func testRepeatedUnprofiledLookup() async throws {
        let tuner = MixedPrecisionAutoTuner()
        await tuner.clearCache()
        let strategy1 = try await tuner.selectStrategy(
            candidateCount: 100,
            maximumSyntheticRelativeError: 0.001
        )
        let strategy2 = try await tuner.selectStrategy(
            candidateCount: 100,
            maximumSyntheticRelativeError: 0.001
        )

        #expect(strategy1 == strategy2, "Repeated unprofiled lookup should match")
        print("✓ Repeated unprofiled lookup verified: \(strategy1.description)")
    }

    @Test("Manual override")
    func testManualOverride() async throws {
        let tuner = MixedPrecisionAutoTuner()
        await tuner.clearCache()
        try await tuner.setOverride(
            .fullFP32,
            candidateCount: 100,
            maximumSyntheticRelativeError: 0.001
        )

        let strategy = try await tuner.selectStrategy(
            candidateCount: 100,
            maximumSyntheticRelativeError: 0.001
        )

        #expect(strategy == .fullFP32, "Should use manual override")
        print("✓ Manual override verified: \(strategy.description)")
        await tuner.clearCache()
    }

    @Test("Strict synthetic-limit cache miss")
    func testStrictSyntheticLimitMiss() async throws {
        let tuner = MixedPrecisionAutoTuner()
        await tuner.clearCache()
        let strategy = try await tuner.selectStrategy(
            candidateCount: 100,
            maximumSyntheticRelativeError: 0.0001
        )
        #expect(
            strategy == .fullFP32,
            "An unprofiled strict-limit lookup returns full FP32, got \(strategy.description)"
        )
        print("✓ Strict synthetic-limit cache miss handled: \(strategy.description)")
    }

    @Test("Relaxed synthetic-limit cache miss")
    func testRelaxedSyntheticLimitMiss() async throws {
        let tuner = MixedPrecisionAutoTuner()
        await tuner.clearCache()
        let strategy = try await tuner.selectStrategy(
            candidateCount: 100,
            maximumSyntheticRelativeError: 0.01
        )
        #expect(strategy == .fullFP32)
        print("✓ Relaxed synthetic-limit miss handled: \(strategy.description)")
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

    @Test("Unprofiled large count returns full FP32")
    func testUnprofiledLargeCount() async throws {
        let tuner = MixedPrecisionAutoTuner()
        await tuner.clearCache()
        let largeBatchStrategy = try await tuner.selectStrategy(
            candidateCount: 1000,
            maximumSyntheticRelativeError: 0.001
        )
        #expect(largeBatchStrategy == .fullFP32)

    }

    @Test("Strategy metrics quality score")
    func testMetricsQualityScore() {
        let goodMetrics = StrategyMetrics(
            strategy: .queryFP32Blocked,
            meanLatency: 0.001,  // 1ms
            medianLatency: 0.0009,
            throughput: 1000,    // 1K ops/sec
            maxRelativeError: 0.0001,  // 0.01% error
            meanRelativeError: 0.00005,
            candidateCount: 100, dimension: 512, measuredSamples: 1, warmupIterations: 0,
            algorithmVersion: 1, datasetVersion: "gaussian-lcg-boxmuller-v1", seed: 42,
            conversionBoundary: "fp32-input-conversion-per-reduced-euclidean-invocation-v1",
            timestamp: Date()
        )

        let poorMetrics = StrategyMetrics(
            strategy: .fullFP32,
            meanLatency: 0.01,   // 10ms
            medianLatency: 0.009,
            throughput: 100,     // 100 ops/sec
            maxRelativeError: 0,  // Perfect accuracy but slow
            meanRelativeError: 0,
            candidateCount: 100, dimension: 512, measuredSamples: 1, warmupIterations: 0,
            algorithmVersion: 1, datasetVersion: "gaussian-lcg-boxmuller-v1", seed: 42,
            conversionBoundary: "fp32-input-conversion-per-reduced-euclidean-invocation-v1",
            timestamp: Date()
        )

        #expect(goodMetrics.qualityScore > poorMetrics.qualityScore,
               "Better performance + good accuracy should have higher score")

        print("✓ Quality scores:")
        print("  Good metrics: \(String(format: "%.2f", goodMetrics.qualityScore))")
        print("  Poor metrics: \(String(format: "%.2f", poorMetrics.qualityScore))")
    }

    @Test("Independent unprofiled counts")
    func testIndependentUnprofiledCounts() async throws {
        let tuner = MixedPrecisionAutoTuner()
        await tuner.clearCache()
        let strategy1 = try await tuner.selectStrategy(candidateCount: 100)
        let strategy2 = try await tuner.selectStrategy(candidateCount: 200)
        #expect(strategy1 == .fullFP32 && strategy2 == .fullFP32)
        print("✓ Both unprofiled counts return full FP32:")
        print("  N=100: \(strategy1.description)")
        print("  N=200: \(strategy2.description)")
    }

    @Test("Unprofiled lookup preserves full FP32")
    func testUnprofiledLookupPreservesFullFP32() async throws {
        let tuner = MixedPrecisionAutoTuner()
        await tuner.clearCache()
        let strategy = try await tuner.selectStrategy(
            candidateCount: 100,
            maximumSyntheticRelativeError: 0.001
        )
        #expect(strategy == .fullFP32,
               "An unprofiled lookup returns full FP32")

        print("✓ Unprofiled result verified: \(strategy.description)")
    }
}
