//
//  MixedPrecisionPhase3Tests.swift
//  VectorCore
//
//  Tests for Phase 3: Caching and Validation
//

import Testing
import Foundation
@testable import VectorCore

@Suite("Mixed Precision Phase 3 - Caching & Validation")
struct MixedPrecisionPhase3Tests {

    // MARK: - SoA Cache Tests

    @Test("SoA FP16 Cache - Basic caching")
    func testSoACacheBasic() async {
        let cache = SoAFP16Cache512.shared
        await cache.clear()

        let vectors = generateTestVectors(count: 100)

        // First access - cache miss
        let soa1 = await cache.getOrCreate(for: vectors)
        let stats1 = await cache.statistics()

        #expect(stats1.misses == 1)
        #expect(stats1.hits == 0)
        #expect(stats1.size == 1)

        // Second access - cache hit
        let soa2 = await cache.getOrCreate(for: vectors)
        let stats2 = await cache.statistics()

        #expect(stats2.hits == 1)
        #expect(stats2.misses == 1)
        #expect(stats2.size == 1)

        // Verify same instance returned
        #expect(soa1.vectorCount == soa2.vectorCount)

        print("✓ SoA cache hit/miss tracking works correctly")

        await cache.clear()
    }

    @Test("SoA FP16 Cache - LRU eviction")
    func testSoACacheLRU() async {
        let cache = SoAFP16Cache512(maxSize: 3)

        // Fill cache to capacity
        let vectors1 = generateTestVectors(count: 10, seed: 1)
        let vectors2 = generateTestVectors(count: 10, seed: 2)
        let vectors3 = generateTestVectors(count: 10, seed: 3)

        await cache.store(MixedPrecisionKernels.createSoA512FP16(from: vectors1), for: vectors1)
        await cache.store(MixedPrecisionKernels.createSoA512FP16(from: vectors2), for: vectors2)
        await cache.store(MixedPrecisionKernels.createSoA512FP16(from: vectors3), for: vectors3)

        let stats1 = await cache.statistics()
        #expect(stats1.size == 3, "Cache should be at capacity")

        // Add fourth item - should evict vectors1 (oldest)
        let vectors4 = generateTestVectors(count: 10, seed: 4)
        await cache.store(MixedPrecisionKernels.createSoA512FP16(from: vectors4), for: vectors4)

        let stats2 = await cache.statistics()
        #expect(stats2.size == 3, "Cache should still be at capacity")

        // vectors1 should be evicted
        let result1 = await cache.get(for: vectors1)
        #expect(result1 == nil, "Oldest entry should be evicted")

        // vectors2, 3, 4 should still be present
        let result2 = await cache.get(for: vectors2)
        let result3 = await cache.get(for: vectors3)
        let result4 = await cache.get(for: vectors4)

        #expect(result2 != nil)
        #expect(result3 != nil)
        #expect(result4 != nil)

        print("✓ LRU eviction works correctly")
    }

    @Test("SoA FP16 Cache - Hit rate calculation")
    func testSoACacheHitRate() async {
        let cache = SoAFP16Cache512.shared
        await cache.clear()

        let vectors1 = generateTestVectors(count: 50, seed: 1)
        let vectors2 = generateTestVectors(count: 50, seed: 2)

        // 2 misses
        _ = await cache.getOrCreate(for: vectors1)
        _ = await cache.getOrCreate(for: vectors2)

        // 3 hits
        _ = await cache.getOrCreate(for: vectors1)
        _ = await cache.getOrCreate(for: vectors2)
        _ = await cache.getOrCreate(for: vectors1)

        let hitRate = await cache.hitRate()
        #expect(abs(hitRate - 0.6) < 0.01, "Hit rate should be 3/5 = 0.6, got \(hitRate)")

        print("✓ Hit rate: \(String(format: "%.1f%%", hitRate * 100))")

        await cache.clear()
    }

    // MARK: - Precision Validator Tests

    @Test("Precision Validator - FP16 conversion validation")
    func testConversionValidation() {
        let original = generateTestVectors(count: 1, seed: 42)[0]
        let converted = MixedPrecisionKernels.Vector512FP16(from: original)

        let isValid = PrecisionValidator.validateConversion512(
            original: original,
            converted: converted,
            tolerance: 0.001  // 0.1%
        )

        #expect(isValid, "FP16 conversion should be within 0.1% tolerance")
        print("✓ FP16 conversion validation passed")
    }

    @Test("Precision Validator - Distance accuracy")
    func testDistanceAccuracyValidation() {
        let query = generateTestVectors(count: 1, seed: 1)[0]
        let candidate = generateTestVectors(count: 1, seed: 2)[0]

        // FP32 reference
        let referenceFP32 = EuclideanKernels.distance512(query, candidate)

        // FP16 computation
        let queryFP16 = MixedPrecisionKernels.Vector512FP16(from: query)
        let candidateSoA = MixedPrecisionKernels.createSoA512FP16(from: [candidate])
        var resultFP16: Float = 0
        withUnsafeMutablePointer(to: &resultFP16) { ptr in
            let buffer = UnsafeMutableBufferPointer(start: ptr, count: 1)
            MixedPrecisionKernels.batchEuclidean512(
                query: queryFP16,
                candidates: candidateSoA,
                results: buffer
            )
        }

        let isValid = PrecisionValidator.validateDistanceAccuracy(
            referenceFP32: referenceFP32,
            computedFP16: resultFP16,
            tolerance: 0.01  // 1%
        )

        #expect(isValid, "FP16 distance should be within 1% of FP32")

        let error = abs(referenceFP32 - resultFP16) / max(referenceFP32, 1e-6)
        print("✓ Distance accuracy validated: \(String(format: "%.4f%% error", error * 100))")
    }

    @Test("Precision Validator - Batch distance accuracy")
    func testBatchDistanceAccuracyValidation() {
        let query = generateTestVectors(count: 1, seed: 1)[0]
        let candidates = generateTestVectors(count: 100, seed: 2)

        // FP32 reference
        var referenceFP32 = [Float](repeating: 0, count: 100)
        referenceFP32.withUnsafeMutableBufferPointer { buffer in
            BatchKernels.range_euclid2_512(
                query: query,
                candidates: candidates,
                range: 0..<100,
                out: buffer
            )
        }
        referenceFP32 = referenceFP32.map { sqrt($0) }

        // FP16 computation
        let candidateSoA = MixedPrecisionKernels.createSoA512FP16(from: candidates)
        var resultsFP16 = [Float](repeating: 0, count: 100)
        resultsFP16.withUnsafeMutableBufferPointer { buffer in
            MixedPrecisionKernels.batchEuclidean512(
                query: query,
                candidates: candidateSoA,
                results: buffer
            )
        }

        let (allValid, maxError, meanError) = PrecisionValidator.validateBatchDistanceAccuracy(
            referenceFP32: referenceFP32,
            computedFP16: resultsFP16,
            tolerance: 0.01  // 1%
        )

        #expect(allValid, "All batch distances should be within 1%")
        #expect(maxError < 0.01, "Max error \(maxError) should be < 1%")

        print("✓ Batch validation: max error \(String(format: "%.4f%%", maxError * 100)), mean error \(String(format: "%.4f%%", meanError * 100))")
    }

    @Test("Precision Validator - Accuracy statistics")
    func testAccuracyStatistics() {
        let original = generateTestVectors(count: 1, seed: 42)[0]
        let converted = MixedPrecisionKernels.Vector512FP16(from: original).toFP32()

        let stats = PrecisionValidator.computeAccuracyStatistics512(
            reference: original,
            computed: converted
        )

        #expect(stats.maxRelativeError < 0.001, "Max relative error should be < 0.1%")
        #expect(stats.meanRelativeError < 0.0005, "Mean relative error should be < 0.05%")

        print("✓ Accuracy statistics:")
        print(stats.summary)
    }

    // MARK: - Performance Profiler Tests

    @Test("Performance Profiler - Metrics recording")
    func testPerformanceProfiler() async {
        let profiler = PerformanceProfiler.shared
        await profiler.reset()

        // Record some metrics
        await profiler.recordConversionTime(0.001)  // 1ms
        await profiler.recordComputationTime(0.005)  // 5ms
        await profiler.recordAccuracyLoss(0.0001)    // 0.01%
        await profiler.recordCacheHitRate(0.75)      // 75%

        let metrics = await profiler.getMetrics()

        #expect(metrics.fp16ConversionTime == 0.001)
        #expect(metrics.computationTime == 0.005)
        #expect(metrics.accuracyLoss == 0.0001)
        #expect(metrics.cacheHitRate == 0.75)

        print("✓ Performance metrics recorded:")
        print(metrics.summary)

        await profiler.reset()
    }

    @Test("Performance Profiler - Speedup calculation")
    func testSpeedupCalculation() async {
        let profiler = PerformanceProfiler.shared
        await profiler.reset()

        // Simulate: conversion takes 1ms, computation takes 2ms
        // Without FP16: would take ~4ms (2× computation time)
        // With FP16: takes 3ms total
        // Speedup: 4/3 = 1.33×

        await profiler.recordConversionTime(0.001)
        await profiler.recordComputationTime(0.002)

        let metrics = await profiler.getMetrics()
        let expectedSpeedup = 1.33

        #expect(abs(metrics.totalSpeedup - expectedSpeedup) < 0.1,
               "Speedup should be ~\(expectedSpeedup)×, got \(metrics.totalSpeedup)×")

        print("✓ Speedup calculation: \(String(format: "%.2f×", metrics.totalSpeedup))")

        await profiler.reset()
    }

    @Test("Performance Profiler - Accumulation")
    func testMetricsAccumulation() async {
        let profiler = PerformanceProfiler.shared
        await profiler.reset()

        // Multiple operations accumulate
        await profiler.recordConversionTime(0.001)
        await profiler.recordConversionTime(0.002)
        await profiler.recordComputationTime(0.005)
        await profiler.recordComputationTime(0.003)

        let metrics = await profiler.getMetrics()

        #expect(metrics.fp16ConversionTime == 0.003, "Conversion times should accumulate")
        #expect(metrics.computationTime == 0.008, "Computation times should accumulate")

        print("✓ Metrics accumulation works correctly")

        await profiler.reset()
    }

    @Test("Performance Profiler - Max accuracy loss tracking")
    func testMaxAccuracyLossTracking() async {
        let profiler = PerformanceProfiler.shared
        await profiler.reset()

        await profiler.recordAccuracyLoss(0.0001)
        await profiler.recordAccuracyLoss(0.0005)  // Higher
        await profiler.recordAccuracyLoss(0.0002)  // Lower

        let metrics = await profiler.getMetrics()

        #expect(metrics.accuracyLoss == 0.0005, "Should track maximum accuracy loss")

        print("✓ Max accuracy loss: \(String(format: "%.4f%%", metrics.accuracyLoss * 100))")

        await profiler.reset()
    }

    // MARK: - Helper Functions

    func generateTestVectors(count: Int, seed: UInt32 = 42) -> [Vector512Optimized] {
        var rng = SeededRNG(seed: seed)
        var vectors: [Vector512Optimized] = []

        for _ in 0..<count {
            var data = [Float](repeating: 0, count: 512)
            for i in 0..<512 {
                data[i] = rng.nextGaussian()
            }
            vectors.append(try! Vector512Optimized(data))
        }

        return vectors
    }
}

// MARK: - Seeded RNG (copied from MixedPrecisionAutoTuner for test isolation)

private struct SeededRNG {
    private var state: UInt64

    init(seed: UInt32) {
        self.state = UInt64(seed)
    }

    mutating func next() -> Float {
        state = state &* 1664525 &+ 1013904223
        return Float(state & 0xFFFFFF) / Float(0xFFFFFF)
    }

    mutating func nextGaussian() -> Float {
        var u1 = next()
        while u1 == 0 {
            u1 = next()
        }
        let u2 = next()

        let r = sqrt(-2.0 * log(u1))
        let theta = 2.0 * Float.pi * u2

        return r * cos(theta)
    }
}
