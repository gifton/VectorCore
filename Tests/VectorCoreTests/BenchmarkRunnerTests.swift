import XCTest
@testable import VectorCore

final class BenchmarkRunnerTests: XCTestCase {
    
    func testBenchmarkRunnerCreation() {
        let runner = VectorCoreBenchmarkRunner()
        XCTAssertFalse(runner.availableBenchmarks.isEmpty)
    }
    
    func testBenchmarkRunnerExecution() async throws {
        let runner = VectorCoreBenchmarkRunner(
            configuration: BenchmarkConfiguration(
                warmupIterations: 2,
                measurementIterations: 5,
                timeoutSeconds: 10
            )
        )
        
        let results = try await runner.run()
        XCTAssertFalse(results.isEmpty)
        
        // Verify we got results for key benchmarks
        let benchmarkNames = results.map { $0.name }
        XCTAssertTrue(benchmarkNames.contains("Vector Addition - 768D"))
        XCTAssertTrue(benchmarkNames.contains("Euclidean Distance - 768D"))
        
        // Verify results have reasonable values
        for result in results {
            XCTAssertGreaterThan(result.throughput, 0)
            XCTAssertGreaterThan(result.averageTime, 0)
            XCTAssertGreaterThan(result.totalTime, 0)
            XCTAssertGreaterThan(result.iterations, 0)
        }
    }
    
    func testBenchmarkAdapter() async throws {
        let runner = VectorCoreBenchmarkRunner(
            configuration: .quick
        )
        
        let results = try await runner.run()
        let baseline = BenchmarkAdapter.createBaseline(from: results)
        
        // Verify baseline has been populated
        XCTAssertGreaterThan(baseline.throughput.vectorAddition, 0)
        XCTAssertGreaterThan(baseline.throughput.euclideanDistance, 0)
        XCTAssertGreaterThan(baseline.memory.bytesPerVector["768"] ?? 0, 0)
        XCTAssertGreaterThan(baseline.parallelization.scalingEfficiency, 0)
    }
    
    func testSpecificBenchmark() async throws {
        let runner = VectorCoreBenchmarkRunner(configuration: .quick)
        
        let result = try await runner.run(benchmarkNamed: "Vector Addition - 768D")
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.name, "Vector Addition - 768D")
        XCTAssertGreaterThan(result?.throughput ?? 0, 0)
    }
    
    func testMemoryTracking() async throws {
        let runner = VectorCoreBenchmarkRunner(configuration: .quick)
        
        let results = try await runner.run()
        let memoryResults = results.filter { $0.memoryAllocated != nil }
        
        // At least some benchmarks should track memory
        XCTAssertFalse(memoryResults.isEmpty)
        
        for result in memoryResults {
            if let memBytes = result.memoryAllocated {
                XCTAssertGreaterThanOrEqual(memBytes, 0)
            }
            // peakMemoryBytes not available in current BenchmarkResult
        }
    }
}