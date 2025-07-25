import XCTest
@testable import VectorCore

final class MetricsCollectionTests: XCTestCase {
    
    func testHardwareMetricsCollection() {
        // Test that hardware metrics collection returns reasonable values
        let metrics = HardwareMetricsCollector.collect()
        
        // SIMD utilization should be between 0 and 1
        XCTAssertGreaterThanOrEqual(metrics.simdUtilization, 0.0)
        XCTAssertLessThanOrEqual(metrics.simdUtilization, 1.0)
        
        // Cache hit rates should be between 0 and 1
        XCTAssertGreaterThanOrEqual(metrics.l1CacheHitRate, 0.0)
        XCTAssertLessThanOrEqual(metrics.l1CacheHitRate, 1.0)
        
        XCTAssertGreaterThanOrEqual(metrics.l2CacheHitRate, 0.0)
        XCTAssertLessThanOrEqual(metrics.l2CacheHitRate, 1.0)
        
        XCTAssertGreaterThanOrEqual(metrics.l3CacheHitRate, 0.0)
        XCTAssertLessThanOrEqual(metrics.l3CacheHitRate, 1.0)
        
        // Memory bandwidth should be positive
        XCTAssertGreaterThan(metrics.memoryBandwidthGBps, 0.0)
        
        // CPU frequency should be reasonable (0.5 - 10 GHz)
        XCTAssertGreaterThan(metrics.avgCPUFrequencyGHz, 0.5)
        XCTAssertLessThan(metrics.avgCPUFrequencyGHz, 10.0)
        
        // CPU utilization should be between 0 and 100
        XCTAssertGreaterThanOrEqual(metrics.cpuUtilization, 0.0)
        XCTAssertLessThanOrEqual(metrics.cpuUtilization, 100.0)
        
        // Context switches should be positive
        XCTAssertGreaterThan(metrics.contextSwitchesPerSec, 0.0)
        
        print("Hardware Metrics:")
        print("  SIMD Utilization: \(String(format: "%.2f%%", metrics.simdUtilization * 100))")
        print("  L1 Cache Hit: \(String(format: "%.2f%%", metrics.l1CacheHitRate * 100))")
        print("  L2 Cache Hit: \(String(format: "%.2f%%", metrics.l2CacheHitRate * 100))")
        print("  L3 Cache Hit: \(String(format: "%.2f%%", metrics.l3CacheHitRate * 100))")
        print("  Memory Bandwidth: \(String(format: "%.1f GB/s", metrics.memoryBandwidthGBps))")
        print("  CPU Frequency: \(String(format: "%.2f GHz", metrics.avgCPUFrequencyGHz))")
        print("  CPU Utilization: \(String(format: "%.1f%%", metrics.cpuUtilization))")
    }
    
    func testMemoryMetricsExtraction() async throws {
        // Run a quick benchmark to get some results
        let runner = VectorCoreBenchmarkRunner(
            configuration: BenchmarkConfiguration(
                warmupIterations: 1,
                measurementIterations: 3,
                timeoutSeconds: 5
            )
        )
        
        // Run only a few benchmarks
        let result = try await runner.run(benchmarkNamed: "Vector Addition - 768D")
        XCTAssertNotNil(result)
        
        if let result = result {
            let results = [result]
            let baseline = BenchmarkAdapter.createBaseline(from: results)
            
            // Check memory metrics
            XCTAssertGreaterThanOrEqual(baseline.memory.bytesPerOperation, 0)
            XCTAssertGreaterThan(baseline.memory.peakMemoryUsage, 0)
            XCTAssertGreaterThanOrEqual(baseline.memory.allocationRate, 0)
            
            // Check bytes per vector
            let bytesFor768 = baseline.memory.bytesPerVector["768"] ?? 0
            XCTAssertGreaterThan(bytesFor768, 0)
            
            // Should be at least the theoretical size
            let theoreticalSize = 768 * MemoryLayout<Float>.size
            XCTAssertGreaterThanOrEqual(bytesFor768, theoreticalSize)
            
            print("\nMemory Metrics:")
            print("  Bytes per operation: \(baseline.memory.bytesPerOperation)")
            print("  Peak memory: \(baseline.memory.peakMemoryUsage / 1024) KB")
            print("  Allocation rate: \(String(format: "%.0f", baseline.memory.allocationRate)) bytes/sec")
            print("  Bytes for 768D vector: \(bytesFor768)")
        }
    }
    
    func testParallelizationMetricsExtraction() async throws {
        let runner = VectorCoreBenchmarkRunner(
            configuration: BenchmarkConfiguration(
                warmupIterations: 2,
                measurementIterations: 5,
                timeoutSeconds: 10
            )
        )
        
        // Run batch and sequential benchmarks
        let batchResult = try await runner.run(benchmarkNamed: "Batch Euclidean Distance - 100x768D")
        let seqResult = try await runner.run(benchmarkNamed: "Euclidean Distance - 768D")
        
        if let batch = batchResult, let seq = seqResult {
            let results = [batch, seq]
            let baseline = BenchmarkAdapter.createBaseline(from: results)
            
            // Check parallelization metrics
            XCTAssertGreaterThanOrEqual(baseline.parallelization.parallelSpeedup, 1.0)
            XCTAssertGreaterThan(baseline.parallelization.scalingEfficiency, 0.0)
            XCTAssertLessThanOrEqual(baseline.parallelization.scalingEfficiency, 1.0)
            XCTAssertGreaterThan(baseline.parallelization.optimalBatchSize, 0)
            XCTAssertGreaterThanOrEqual(baseline.parallelization.threadUtilization, 0.0)
            XCTAssertLessThanOrEqual(baseline.parallelization.threadUtilization, 100.0)
            
            print("\nParallelization Metrics:")
            print("  Speedup: \(String(format: "%.2fx", baseline.parallelization.parallelSpeedup))")
            print("  Scaling efficiency: \(String(format: "%.1f%%", baseline.parallelization.scalingEfficiency * 100))")
            print("  Optimal batch size: \(baseline.parallelization.optimalBatchSize)")
            print("  Thread utilization: \(String(format: "%.1f%%", baseline.parallelization.threadUtilization))")
        }
    }
}