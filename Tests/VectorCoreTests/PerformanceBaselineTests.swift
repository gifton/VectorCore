import XCTest
@testable import VectorCore

/// Comprehensive unit tests for the performance baseline system
final class PerformanceBaselineTests: XCTestCase {
    
    // MARK: - Test Data
    
    private func createMockBenchmarkResults() -> [BenchmarkResult] {
        return [
            BenchmarkResult(
                name: "Vector Addition",
                iterations: 1000,
                totalTime: 0.01,
                averageTime: 0.00001,
                standardDeviation: 0.000001,
                throughput: 100000.0,
                memoryAllocated: 1024
            ),
            BenchmarkResult(
                name: "Dot Product",
                iterations: 1000,
                totalTime: 0.015,
                averageTime: 0.000015,
                standardDeviation: 0.000002,
                throughput: 66666.0,
                memoryAllocated: 512
            )
        ]
    }
    
    private func createMockThroughputMetrics() -> ThroughputMetrics {
        ThroughputMetrics(
            vectorAddition: 200000.0,
            vectorScalarMultiplication: 250000.0,
            vectorElementwiseMultiplication: 180000.0,
            dotProduct: 300000.0,
            euclideanDistance: 260000.0,
            cosineSimilarity: 240000.0,
            manhattanDistance: 285000.0,
            normalization: 208000.0
        )
    }
    
    private func createMockMemoryMetrics() -> MemoryMetrics {
        MemoryMetrics(
            bytesPerOperation: 512,
            peakMemoryUsage: 52428800,
            allocationRate: 10000.0,
            bytesPerVector: ["768": 3072, "1536": 6144]
        )
    }
    
    private func createMockParallelizationMetrics() -> ParallelizationMetrics {
        ParallelizationMetrics(
            parallelSpeedup: 3.2,
            scalingEfficiency: 0.8,
            optimalBatchSize: 1000,
            threadUtilization: 0.75
        )
    }
    
    private func createMockHardwareMetrics() -> HardwareMetrics {
        HardwareMetrics(
            simdUtilization: 0.85,
            l1CacheHitRate: 0.95,
            l2CacheHitRate: 0.88,
            l3CacheHitRate: 0.72,
            memoryBandwidthGBps: 45.2,
            avgCPUFrequencyGHz: 3.2,
            cpuUtilization: 78.5,
            contextSwitchesPerSec: 1250
        )
    }
    
    // MARK: - PerformanceBaseline Tests
    
    func testPerformanceBaselineCreation() {
        let baseline = PerformanceBaseline(
            throughput: createMockThroughputMetrics(),
            memory: createMockMemoryMetrics(),
            parallelization: createMockParallelizationMetrics(),
            benchmarks: createMockBenchmarkResults(),
            hardware: createMockHardwareMetrics()
        )
        
        XCTAssertEqual(baseline.throughput.vectorAddition, 200000.0)
        XCTAssertEqual(baseline.memory.bytesPerOperation, 512)
        XCTAssertEqual(baseline.parallelization.parallelSpeedup, 3.2, accuracy: 0.01)
        XCTAssertNotNil(baseline.hardware)
        XCTAssertEqual(baseline.hardware?.simdUtilization ?? 0, 0.85, accuracy: 0.01)
        XCTAssertEqual(baseline.benchmarks.count, 2)
    }
    
    func testBackwardCompatibility() {
        // Test that baselines without hardware metrics still work
        let baseline = PerformanceBaseline(
            throughput: createMockThroughputMetrics(),
            memory: createMockMemoryMetrics(),
            parallelization: createMockParallelizationMetrics(),
            benchmarks: createMockBenchmarkResults(),
            hardware: nil
        )
        
        XCTAssertNil(baseline.hardware)
        XCTAssertEqual(baseline.benchmarks.count, 2)
    }
    
    // MARK: - JSON Serialization Tests
    
    func testBaselineJSONEncodingDecoding() throws {
        let originalBaseline = PerformanceBaseline(
            throughput: createMockThroughputMetrics(),
            memory: createMockMemoryMetrics(),
            parallelization: createMockParallelizationMetrics(),
            benchmarks: createMockBenchmarkResults(),
            hardware: createMockHardwareMetrics()
        )
        
        // Encode
        let encoder = JSONEncoder.baseline
        let data = try encoder.encode(originalBaseline)
        
        // Decode
        let decoder = JSONDecoder.baseline
        let decodedBaseline = try decoder.decode(PerformanceBaseline.self, from: data)
        
        // Verify
        XCTAssertEqual(originalBaseline.throughput.vectorAddition, 
                      decodedBaseline.throughput.vectorAddition)
        XCTAssertEqual(originalBaseline.memory.bytesPerOperation, 
                      decodedBaseline.memory.bytesPerOperation)
        XCTAssertEqual(originalBaseline.hardware?.simdUtilization,
                      decodedBaseline.hardware?.simdUtilization)
    }
    
    func testJSONFormatting() throws {
        let baseline = PerformanceBaseline(
            throughput: createMockThroughputMetrics(),
            memory: createMockMemoryMetrics(),
            parallelization: createMockParallelizationMetrics(),
            benchmarks: [],
            hardware: nil
        )
        
        let encoder = JSONEncoder.baseline
        let data = try encoder.encode(baseline)
        let json = String(data: data, encoding: .utf8)!
        
        // Check formatting
        XCTAssertTrue(json.contains("  ")) // Pretty printed
        XCTAssertTrue(json.contains("\"benchmarks\"")) // Sorted keys
    }
    
    // MARK: - Comparison Tests
    
    func testPerformanceComparisonNoRegression() {
        let baseline = PerformanceBaseline(
            throughput: createMockThroughputMetrics(),
            memory: createMockMemoryMetrics(),
            parallelization: createMockParallelizationMetrics(),
            benchmarks: createMockBenchmarkResults()
        )
        
        // Current is slightly better
        var improvedThroughput = createMockThroughputMetrics()
        improvedThroughput = ThroughputMetrics(
            vectorAddition: 210000.0, // 5% improvement
            vectorScalarMultiplication: improvedThroughput.vectorScalarMultiplication,
            vectorElementwiseMultiplication: improvedThroughput.vectorElementwiseMultiplication,
            dotProduct: improvedThroughput.dotProduct,
            euclideanDistance: improvedThroughput.euclideanDistance,
            cosineSimilarity: improvedThroughput.cosineSimilarity,
            manhattanDistance: improvedThroughput.manhattanDistance,
            normalization: improvedThroughput.normalization
        )
        
        let current = PerformanceBaseline(
            throughput: improvedThroughput,
            memory: createMockMemoryMetrics(),
            parallelization: createMockParallelizationMetrics(),
            benchmarks: createMockBenchmarkResults()
        )
        
        let comparison = PerformanceComparison(
            baseline: baseline,
            current: current,
            regressionThreshold: 0.05
        )
        
        XCTAssertFalse(comparison.hasRegression())
        XCTAssertTrue(comparison.detectRegressions().isEmpty)
    }
    
    func testPerformanceComparisonWithRegression() {
        let baseline = PerformanceBaseline(
            throughput: createMockThroughputMetrics(),
            memory: createMockMemoryMetrics(),
            parallelization: createMockParallelizationMetrics(),
            benchmarks: createMockBenchmarkResults()
        )
        
        // Current has regression
        var degradedThroughput = createMockThroughputMetrics()
        degradedThroughput = ThroughputMetrics(
            vectorAddition: 180000.0, // 10% regression
            vectorScalarMultiplication: degradedThroughput.vectorScalarMultiplication,
            vectorElementwiseMultiplication: degradedThroughput.vectorElementwiseMultiplication,
            dotProduct: degradedThroughput.dotProduct,
            euclideanDistance: degradedThroughput.euclideanDistance,
            cosineSimilarity: degradedThroughput.cosineSimilarity,
            manhattanDistance: degradedThroughput.manhattanDistance,
            normalization: degradedThroughput.normalization
        )
        
        let current = PerformanceBaseline(
            throughput: degradedThroughput,
            memory: createMockMemoryMetrics(),
            parallelization: createMockParallelizationMetrics(),
            benchmarks: createMockBenchmarkResults()
        )
        
        let comparison = PerformanceComparison(
            baseline: baseline,
            current: current,
            regressionThreshold: 0.05
        )
        
        XCTAssertTrue(comparison.hasRegression())
        let regressions = comparison.detectRegressions()
        XCTAssertEqual(regressions.count, 1)
        XCTAssertEqual(regressions.first?.operation, "Vector Addition")
        XCTAssertEqual(regressions.first?.percentChange ?? 0, -0.1, accuracy: 0.01)
    }
    
    func testMemoryRegression() {
        let baseline = PerformanceBaseline(
            throughput: createMockThroughputMetrics(),
            memory: createMockMemoryMetrics(),
            parallelization: createMockParallelizationMetrics(),
            benchmarks: createMockBenchmarkResults()
        )
        
        // Increase memory usage by 10%
        let degradedMemory = MemoryMetrics(
            bytesPerOperation: 563, // ~10% increase
            peakMemoryUsage: 57671680,
            allocationRate: 11000.0,
            bytesPerVector: ["768": 3380, "1536": 6758]
        )
        
        let current = PerformanceBaseline(
            throughput: createMockThroughputMetrics(),
            memory: degradedMemory,
            parallelization: createMockParallelizationMetrics(),
            benchmarks: createMockBenchmarkResults()
        )
        
        let comparison = PerformanceComparison(
            baseline: baseline,
            current: current,
            regressionThreshold: 0.05
        )
        
        XCTAssertTrue(comparison.hasRegression())
        let regressions = comparison.detectRegressions()
        XCTAssertTrue(regressions.contains { $0.type == .memory })
    }
    
    // MARK: - Hardware Metrics Tests
    
    func testHardwareMetricsValidation() {
        let metrics = createMockHardwareMetrics()
        
        // All rates should be between 0 and 1
        XCTAssertTrue(metrics.simdUtilization >= 0 && metrics.simdUtilization <= 1)
        XCTAssertTrue(metrics.l1CacheHitRate >= 0 && metrics.l1CacheHitRate <= 1)
        XCTAssertTrue(metrics.l2CacheHitRate >= 0 && metrics.l2CacheHitRate <= 1)
        XCTAssertTrue(metrics.l3CacheHitRate >= 0 && metrics.l3CacheHitRate <= 1)
        
        // Reasonable values
        XCTAssertTrue(metrics.memoryBandwidthGBps > 0)
        XCTAssertTrue(metrics.avgCPUFrequencyGHz > 0)
        XCTAssertTrue(metrics.cpuUtilization >= 0 && metrics.cpuUtilization <= 100)
        XCTAssertTrue(metrics.contextSwitchesPerSec >= 0)
    }
    
    // MARK: - Platform Info Tests
    
    func testPlatformInfoCurrent() {
        let platform = PlatformInfo.current
        
        XCTAssertFalse(platform.os.isEmpty)
        XCTAssertFalse(platform.osVersion.isEmpty)
        XCTAssertFalse(platform.architecture.isEmpty)
        XCTAssertTrue(platform.cpuCores > 0)
        XCTAssertTrue(platform.memoryGB > 0)
        
        #if arch(arm64)
        XCTAssertEqual(platform.architecture, "arm64")
        #elseif arch(x86_64)
        XCTAssertEqual(platform.architecture, "x86_64")
        #endif
    }
    
    // MARK: - Benchmark Result Tests
    
    func testBenchmarkResultConsistency() {
        let result = BenchmarkResult(
            name: "Test Benchmark",
            iterations: 1000,
            totalTime: 1.0,
            averageTime: 0.001,
            standardDeviation: 0.0001,
            throughput: 1000.0,
            memoryAllocated: 4096
        )
        
        // Verify consistency
        XCTAssertEqual(result.averageTime, result.totalTime / Double(result.iterations), 
                      accuracy: 0.000001)
        XCTAssertEqual(result.throughput, Double(result.iterations) / result.totalTime,
                      accuracy: 0.01)
    }
    
    // MARK: - Regression Result Tests
    
    func testRegressionResultDescription() {
        let baselineResult = PerformanceResult(
            testName: "Vector Addition",
            dimension: 128,
            meanTime: 0.00001,  // 10 microseconds
            stdDeviation: 0.000001,
            minTime: 0.000009,
            maxTime: 0.000011,
            throughput: 100000.0
        )
        let currentResult = PerformanceResult(
            testName: "Vector Addition",
            dimension: 128,
            meanTime: 0.0000111,  // 11.1 microseconds (10% slower)
            stdDeviation: 0.000001,
            minTime: 0.00001,
            maxTime: 0.0000122,
            throughput: 90000.0
        )
        
        let regression = RegressionResult(
            test: "Vector Addition",
            baseline: baselineResult,
            current: currentResult,
            percentageChange: -10.0,
            isRegression: true,
            isImprovement: false
        )
        
        let summary = regression.summary
        XCTAssertTrue(summary.contains("Vector Addition"))
        XCTAssertTrue(summary.contains("REGRESSION"))
        XCTAssertTrue(summary.contains("-10.0%"))
    }
}