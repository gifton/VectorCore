import XCTest
@testable import VectorCore

/// Integration tests for the benchmark framework
final class BenchmarkIntegrationTests: XCTestCase {
    
    // MARK: - Mock Benchmark Runner
    
    class MockBenchmarkRunner: BenchmarkRunner {
        var shouldFail: Bool = false
        var executionCount: Int = 0
        var lastBenchmarkName: String?
        
        func runBenchmarks() async throws -> [BenchmarkResult] {
            executionCount += 1
            
            if shouldFail {
                throw BenchmarkError.executionFailed("Mock failure")
            }
            
            return [
                BenchmarkResult(
                    name: "Mock Benchmark 1",
                    iterations: 100,
                    totalTime: 0.1,
                    averageTime: 0.001,
                    standardDeviation: 0.0001,
                    throughput: 1000.0,
                    memoryAllocated: 1024
                ),
                BenchmarkResult(
                    name: "Mock Benchmark 2",
                    iterations: 200,
                    totalTime: 0.2,
                    averageTime: 0.001,
                    standardDeviation: 0.0001,
                    throughput: 1000.0,
                    memoryAllocated: 2048
                )
            ]
        }
        
        func runBenchmark(named name: String) async throws -> BenchmarkResult? {
            lastBenchmarkName = name
            executionCount += 1
            
            if shouldFail {
                throw BenchmarkError.executionFailed("Mock failure")
            }
            
            let results = try await runBenchmarks()
            return results.first { $0.name == name }
        }
        
        func availableBenchmarks() -> [String] {
            ["Mock Benchmark 1", "Mock Benchmark 2", "Mock Benchmark 3"]
        }
    }
    
    enum BenchmarkError: Error {
        case executionFailed(String)
    }
    
    // MARK: - Tests
    
    func testBenchmarkRunnerProtocol() async throws {
        let runner = MockBenchmarkRunner()
        
        // Test runBenchmarks
        let results = try await runner.runBenchmarks()
        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(runner.executionCount, 1)
        
        // Test runBenchmark
        let specificResult = try await runner.runBenchmark(named: "Mock Benchmark 1")
        XCTAssertNotNil(specificResult)
        XCTAssertEqual(specificResult?.name, "Mock Benchmark 1")
        XCTAssertEqual(runner.lastBenchmarkName, "Mock Benchmark 1")
        
        // Test availableBenchmarks
        let available = runner.availableBenchmarks()
        XCTAssertEqual(available.count, 3)
        XCTAssertTrue(available.contains("Mock Benchmark 1"))
    }
    
    func testBenchmarkErrorHandling() async {
        let runner = MockBenchmarkRunner()
        runner.shouldFail = true
        
        do {
            _ = try await runner.runBenchmarks()
            XCTFail("Expected error but none was thrown")
        } catch {
            XCTAssertTrue(error is BenchmarkError)
        }
    }
    
    func testBenchmarkConfiguration() {
        // Test default configuration
        let defaultConfig = BenchmarkConfiguration.default
        XCTAssertEqual(defaultConfig.warmupIterations, 10)
        XCTAssertEqual(defaultConfig.measurementIterations, 1000)
        XCTAssertEqual(defaultConfig.timeoutSeconds, 60)
        XCTAssertTrue(defaultConfig.collectHardwareMetrics)
        XCTAssertFalse(defaultConfig.parallelExecution)
        XCTAssertNil(defaultConfig.benchmarkFilter)
        
        // Test quick configuration
        let quickConfig = BenchmarkConfiguration.quick
        XCTAssertEqual(quickConfig.warmupIterations, 5)
        XCTAssertEqual(quickConfig.measurementIterations, 100)
        XCTAssertEqual(quickConfig.timeoutSeconds, 10)
        XCTAssertFalse(quickConfig.collectHardwareMetrics)
        
        // Test comprehensive configuration
        let comprehensiveConfig = BenchmarkConfiguration.comprehensive
        XCTAssertEqual(comprehensiveConfig.warmupIterations, 50)
        XCTAssertEqual(comprehensiveConfig.measurementIterations, 10000)
        XCTAssertEqual(comprehensiveConfig.timeoutSeconds, 300)
        XCTAssertTrue(comprehensiveConfig.collectHardwareMetrics)
        
        // Test custom configuration
        let customConfig = BenchmarkConfiguration(
            warmupIterations: 20,
            measurementIterations: 500,
            timeoutSeconds: 30,
            collectHardwareMetrics: false,
            parallelExecution: true,
            benchmarkFilter: ["Benchmark 1", "Benchmark 2"]
        )
        XCTAssertEqual(customConfig.warmupIterations, 20)
        XCTAssertTrue(customConfig.parallelExecution)
        XCTAssertEqual(customConfig.benchmarkFilter?.count, 2)
    }
    
    func testBaselineCapture() async throws {
        // Simulate baseline capture workflow
        let runner = MockBenchmarkRunner()
        let results = try await runner.runBenchmarks()
        
        // Calculate metrics from results
        let throughputSum = results.reduce(0.0) { $0 + $1.throughput }
        let avgThroughput = throughputSum / Double(results.count)
        
        let memorySum = results.compactMap { $0.memoryAllocated }.reduce(0, +)
        let avgMemory = memorySum / results.count
        
        XCTAssertEqual(avgThroughput, 1000.0)
        XCTAssertEqual(avgMemory, 1536) // (1024 + 2048) / 2
        
        // Create baseline
        let throughput = ThroughputMetrics(
            vectorAddition: avgThroughput,
            vectorScalarMultiplication: avgThroughput,
            vectorElementwiseMultiplication: avgThroughput,
            dotProduct: avgThroughput,
            euclideanDistance: avgThroughput,
            cosineSimilarity: avgThroughput,
            manhattanDistance: avgThroughput,
            normalization: avgThroughput
        )
        
        let memory = MemoryMetrics(
            bytesPerOperation: avgMemory,
            peakMemoryUsage: memorySum * 10,
            allocationRate: 1000.0,
            bytesPerVector: ["768": 3072]
        )
        
        let parallelization = ParallelizationMetrics(
            parallelSpeedup: 1.0,
            scalingEfficiency: 1.0,
            optimalBatchSize: 100,
            threadUtilization: 0.5
        )
        
        let baseline = PerformanceBaseline(
            throughput: throughput,
            memory: memory,
            parallelization: parallelization,
            benchmarks: results
        )
        
        XCTAssertEqual(baseline.benchmarks.count, 2)
        XCTAssertEqual(baseline.throughput.vectorAddition, 1000.0)
    }
    
    func testExtendedBaseline() {
        let baseline = PerformanceBaseline(
            throughput: ThroughputMetrics(
                vectorAddition: 100000,
                vectorScalarMultiplication: 120000,
                vectorElementwiseMultiplication: 90000,
                dotProduct: 150000,
                euclideanDistance: 110000,
                cosineSimilarity: 105000,
                manhattanDistance: 125000,
                normalization: 95000
            ),
            memory: MemoryMetrics(
                bytesPerOperation: 512,
                peakMemoryUsage: 10485760,
                allocationRate: 5000.0,
                bytesPerVector: ["768": 3072]
            ),
            parallelization: ParallelizationMetrics(
                parallelSpeedup: 2.5,
                scalingEfficiency: 0.625,
                optimalBatchSize: 500,
                threadUtilization: 0.7
            ),
            benchmarks: []
        )
        
        // Test that baseline was created successfully
        XCTAssertEqual(baseline.throughput.vectorAddition, 100000)
        XCTAssertEqual(baseline.memory.bytesPerVector["768"], 3072)
        XCTAssertEqual(baseline.parallelization.parallelSpeedup, 2.5)
    }
    
    func testBenchmarkResultAggregation() {
        let results = [
            BenchmarkResult(
                name: "Test 1",
                iterations: 100,
                totalTime: 1.0,
                averageTime: 0.01,
                standardDeviation: 0.001,
                throughput: 100.0,
                memoryAllocated: 1000
            ),
            BenchmarkResult(
                name: "Test 2",
                iterations: 200,
                totalTime: 2.0,
                averageTime: 0.01,
                standardDeviation: 0.002,
                throughput: 100.0,
                memoryAllocated: 2000
            ),
            BenchmarkResult(
                name: "Test 3",
                iterations: 300,
                totalTime: 3.0,
                averageTime: 0.01,
                standardDeviation: 0.001,
                throughput: 100.0,
                memoryAllocated: nil // Test nil handling
            )
        ]
        
        // Aggregate statistics
        let totalIterations = results.reduce(0) { $0 + $1.iterations }
        let totalTime = results.reduce(0.0) { $0 + $1.totalTime }
        let avgThroughput = results.reduce(0.0) { $0 + $1.throughput } / Double(results.count)
        let totalMemory = results.compactMap { $0.memoryAllocated }.reduce(0, +)
        
        XCTAssertEqual(totalIterations, 600)
        XCTAssertEqual(totalTime, 6.0)
        XCTAssertEqual(avgThroughput, 100.0)
        XCTAssertEqual(totalMemory, 3000) // Only non-nil values
    }
    
    func testBenchmarkFilterApplication() async throws {
        let runner = MockBenchmarkRunner()
        let filter = ["Mock Benchmark 1"]
        
        // Simulate filtering
        let allResults = try await runner.runBenchmarks()
        let filteredResults = allResults.filter { filter.contains($0.name) }
        
        XCTAssertEqual(filteredResults.count, 1)
        XCTAssertEqual(filteredResults.first?.name, "Mock Benchmark 1")
    }
}