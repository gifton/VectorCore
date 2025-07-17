// VectorCore: Performance Regression Tests
//
// Tests for the performance regression detection system
//

import XCTest
@testable import VectorCore

final class PerformanceRegressionTests: XCTestCase {
    
    // MARK: - Basic Suite Tests
    
    func testRegressionSuiteExecution() {
        let config = RegressionTestConfig(
            iterations: 10, // Reduced for testing
            warmupIterations: 2
        )
        let suite = PerformanceRegressionSuite(config: config)
        
        let results = suite.runAllTests()
        
        // Verify we have results
        XCTAssertFalse(results.isEmpty)
        
        // Check specific tests exist
        XCTAssertNotNil(results["Vector Addition (Dim32)"])
        XCTAssertNotNil(results["Dot Product (Dim1536)"])
        XCTAssertNotNil(results["k-NN Search (100 vectors)"])
        
        // Verify result properties
        if let addResult = results["Vector Addition (Dim32)"] {
            XCTAssertGreaterThan(addResult.meanTime, 0)
            XCTAssertGreaterThanOrEqual(addResult.minTime, 0)
            XCTAssertGreaterThanOrEqual(addResult.maxTime, addResult.minTime)
            XCTAssertGreaterThan(addResult.throughput, 0)
        }
    }
    
    func testPerformanceResultFormatting() {
        let result = PerformanceResult(
            testName: "Test Operation",
            dimension: 128,
            meanTime: 0.000001234, // 1.234 microseconds
            stdDeviation: 0.0000001,
            minTime: 0.000001,
            maxTime: 0.0000015,
            throughput: 810372.77
        )
        
        XCTAssertEqual(result.formattedMeanTime, "1.23 μs")
    }
    
    // MARK: - Baseline Tests
    
    func testBaselineSerialization() throws {
        let results = [
            "Test1": PerformanceResult(
                testName: "Test1",
                dimension: 64,
                meanTime: 0.001,
                stdDeviation: 0.0001,
                minTime: 0.0009,
                maxTime: 0.0011,
                throughput: 1000
            ),
            "Test2": PerformanceResult(
                testName: "Test2",
                dimension: 128,
                meanTime: 0.002,
                stdDeviation: 0.0002,
                minTime: 0.0018,
                maxTime: 0.0022,
                throughput: 500
            )
        ]
        
        let baseline = PerformanceBaseline(
            version: "1.0.0",
            platform: "Test Platform",
            date: Date(),
            results: results
        )
        
        // Test save and load
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_baseline.json")
        
        try baseline.save(to: tempURL)
        let loaded = try PerformanceBaseline.load(from: tempURL)
        
        XCTAssertEqual(loaded.version, baseline.version)
        XCTAssertEqual(loaded.platform, baseline.platform)
        XCTAssertEqual(loaded.results.count, baseline.results.count)
        
        // Clean up
        try? FileManager.default.removeItem(at: tempURL)
    }
    
    // MARK: - Regression Detection Tests
    
    func testRegressionDetection() {
        let config = RegressionTestConfig(acceptableVariance: 0.1) // 10%
        let suite = PerformanceRegressionSuite(config: config)
        
        // Create baseline results
        let baselineResults = [
            "Test1": PerformanceResult(
                testName: "Test1",
                dimension: 64,
                meanTime: 0.001,
                stdDeviation: 0.0001,
                minTime: 0.0009,
                maxTime: 0.0011,
                throughput: 1000
            ),
            "Test2": PerformanceResult(
                testName: "Test2",
                dimension: 128,
                meanTime: 0.002,
                stdDeviation: 0.0002,
                minTime: 0.0018,
                maxTime: 0.0022,
                throughput: 500
            )
        ]
        
        let baseline = PerformanceBaseline(
            version: "1.0.0",
            platform: "Test",
            date: Date(),
            results: baselineResults
        )
        
        // Simulate current results with regression
        suite.results["Test1"] = PerformanceResult(
            testName: "Test1",
            dimension: 64,
            meanTime: 0.0012, // 20% slower - regression
            stdDeviation: 0.0001,
            minTime: 0.0011,
            maxTime: 0.0013,
            throughput: 833.33
        )
        
        suite.results["Test2"] = PerformanceResult(
            testName: "Test2",
            dimension: 128,
            meanTime: 0.0018, // 10% faster - improvement
            stdDeviation: 0.0002,
            minTime: 0.0016,
            maxTime: 0.0020,
            throughput: 555.55
        )
        
        let regressions = suite.compareAgainstBaseline(baseline)
        
        XCTAssertEqual(regressions.count, 2)
        
        // Check Test1 - should be regression
        if let test1Regression = regressions.first(where: { $0.test == "Test1" }) {
            XCTAssertTrue(test1Regression.isRegression)
            XCTAssertFalse(test1Regression.isImprovement)
            XCTAssertEqual(test1Regression.percentageChange, 20.0, accuracy: 0.1)
        }
        
        // Check Test2 - should be improvement
        if let test2Regression = regressions.first(where: { $0.test == "Test2" }) {
            XCTAssertFalse(test2Regression.isRegression)
            XCTAssertTrue(test2Regression.isImprovement)
            XCTAssertEqual(test2Regression.percentageChange, -10.0, accuracy: 0.1)
        }
    }
    
    func testRegressionResultSummary() {
        let baseline = PerformanceResult(
            testName: "Test",
            dimension: 64,
            meanTime: 0.001,
            stdDeviation: 0.0001,
            minTime: 0.0009,
            maxTime: 0.0011,
            throughput: 1000
        )
        
        let current = PerformanceResult(
            testName: "Test",
            dimension: 64,
            meanTime: 0.0012,
            stdDeviation: 0.0001,
            minTime: 0.0011,
            maxTime: 0.0013,
            throughput: 833.33
        )
        
        let regression = RegressionResult(
            test: "Test",
            baseline: baseline,
            current: current,
            percentageChange: 20.0,
            isRegression: true,
            isImprovement: false
        )
        
        XCTAssertTrue(regression.summary.contains("REGRESSION"))
        XCTAssertTrue(regression.summary.contains("20.0%"))
    }
    
    // MARK: - Output Format Tests
    
    func testPlainFormatOutput() {
        let config = RegressionTestConfig(
            iterations: 10,
            outputFormat: .plain
        )
        let suite = PerformanceRegressionSuite(config: config)
        
        suite.results["Test1"] = PerformanceResult(
            testName: "Test1",
            dimension: 64,
            meanTime: 0.000001234,
            stdDeviation: 0.0000001,
            minTime: 0.000001,
            maxTime: 0.0000015,
            throughput: 810372.77
        )
        
        let output = suite.formatResults(suite.results)
        
        XCTAssertTrue(output.contains("Performance Test Results"))
        XCTAssertTrue(output.contains("Test1"))
        XCTAssertTrue(output.contains("1.23 μs"))
        XCTAssertTrue(output.contains("810372.77 ops/sec"))
    }
    
    func testMarkdownFormatOutput() {
        let config = RegressionTestConfig(
            iterations: 10,
            outputFormat: .markdown
        )
        let suite = PerformanceRegressionSuite(config: config)
        
        suite.results["Test1"] = PerformanceResult(
            testName: "Test1",
            dimension: 64,
            meanTime: 0.001,
            stdDeviation: 0.0001,
            minTime: 0.0009,
            maxTime: 0.0011,
            throughput: 1000
        )
        
        let output = suite.formatResults(suite.results)
        
        XCTAssertTrue(output.contains("# Performance Test Results"))
        XCTAssertTrue(output.contains("| Test | Dimension | Mean Time | Std Dev | Throughput |"))
        XCTAssertTrue(output.contains("| Test1"))
    }
    
    func testJSONFormatOutput() throws {
        let config = RegressionTestConfig(
            iterations: 10,
            outputFormat: .json
        )
        let suite = PerformanceRegressionSuite(config: config)
        
        suite.results["Test1"] = PerformanceResult(
            testName: "Test1",
            dimension: 64,
            meanTime: 0.001,
            stdDeviation: 0.0001,
            minTime: 0.0009,
            maxTime: 0.0011,
            throughput: 1000
        )
        
        let output = suite.formatResults(suite.results)
        
        // Verify it's valid JSON
        let data = output.data(using: .utf8)!
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        XCTAssertNotNil(json)
        XCTAssertNotNil(json?["Test1"])
    }
    
    // MARK: - Regression Formatting Tests
    
    func testRegressionResultsFormatting() {
        let suite = PerformanceRegressionSuite()
        
        let regressions = [
            RegressionResult(
                test: "Test1",
                baseline: PerformanceResult(testName: "Test1", dimension: 64, meanTime: 0.001, stdDeviation: 0.0001, minTime: 0.0009, maxTime: 0.0011, throughput: 1000),
                current: PerformanceResult(testName: "Test1", dimension: 64, meanTime: 0.0012, stdDeviation: 0.0001, minTime: 0.0011, maxTime: 0.0013, throughput: 833.33),
                percentageChange: 20.0,
                isRegression: true,
                isImprovement: false
            ),
            RegressionResult(
                test: "Test2",
                baseline: PerformanceResult(testName: "Test2", dimension: 128, meanTime: 0.002, stdDeviation: 0.0002, minTime: 0.0018, maxTime: 0.0022, throughput: 500),
                current: PerformanceResult(testName: "Test2", dimension: 128, meanTime: 0.0018, stdDeviation: 0.0002, minTime: 0.0016, maxTime: 0.0020, throughput: 555.55),
                percentageChange: -10.0,
                isRegression: false,
                isImprovement: true
            )
        ]
        
        let output = suite.formatRegressionResults(regressions)
        
        XCTAssertTrue(output.contains("REGRESSIONS DETECTED"))
        XCTAssertTrue(output.contains("IMPROVEMENTS"))
        XCTAssertTrue(output.contains("Test1: 20.00% slower"))
        XCTAssertTrue(output.contains("Test2: 10.00% faster"))
        XCTAssertTrue(output.contains("Regressions: 1"))
        XCTAssertTrue(output.contains("Improvements: 1"))
    }
    
    // MARK: - Integration Tests
    
    func testRunAndCheckRegressions() throws {
        let config = RegressionTestConfig(
            iterations: 5,
            acceptableVariance: 0.1,
            failOnRegression: false,
            warmupIterations: 1
        )
        let suite = PerformanceRegressionSuite(config: config)
        
        // Run without baseline (should not fail)
        let (results, regressions) = try suite.runAndCheckRegressions()
        
        XCTAssertFalse(results.isEmpty)
        XCTAssertNil(regressions)
        
        // Create baseline
        let baseline = suite.createBaseline()
        XCTAssertEqual(baseline.results.count, results.count)
    }
    
    func testRegressionErrorHandling() {
        // First run a test to get baseline
        let initialConfig = RegressionTestConfig(
            iterations: 5,
            acceptableVariance: 0.1,
            failOnRegression: false,
            warmupIterations: 1
        )
        let initialSuite = PerformanceRegressionSuite(config: initialConfig)
        _ = initialSuite.runAllTests()
        
        // Create baseline from actual test results
        let baseline = initialSuite.createBaseline()
        
        // Modify baseline to simulate much faster historical performance
        var fasterResults = baseline.results
        if let firstTest = fasterResults.first {
            // Make baseline 50% faster (lower time)
            let oldResult = firstTest.value
            fasterResults[firstTest.key] = PerformanceResult(
                testName: oldResult.testName,
                dimension: oldResult.dimension,
                meanTime: oldResult.meanTime * 0.5, // 50% faster in baseline
                stdDeviation: oldResult.stdDeviation * 0.5,
                minTime: oldResult.minTime * 0.5,
                maxTime: oldResult.maxTime * 0.5,
                throughput: oldResult.throughput * 2
            )
        }
        
        let modifiedBaseline = PerformanceBaseline(
            version: baseline.version,
            platform: baseline.platform,
            date: baseline.date,
            results: fasterResults
        )
        
        // Save modified baseline
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_regression_baseline.json")
        try! modifiedBaseline.save(to: tempURL)
        
        // Now run with regression detection enabled
        let config = RegressionTestConfig(
            iterations: 5,
            acceptableVariance: 0.1, // 10% tolerance
            failOnRegression: true,
            warmupIterations: 1
        )
        let suite = PerformanceRegressionSuite(config: config)
        
        // Should throw error because current performance is >10% slower than baseline
        XCTAssertThrowsError(try suite.runAndCheckRegressions(baselineURL: tempURL)) { error in
            guard let regressionError = error as? RegressionError,
                  case .regressionsDetected(let regressions) = regressionError else {
                XCTFail("Wrong error type: \(error)")
                return
            }
            XCTAssertGreaterThan(regressions.filter { $0.isRegression }.count, 0)
        }
        
        // Clean up
        try? FileManager.default.removeItem(at: tempURL)
    }
    
    // MARK: - Performance of Performance Tests
    
    func testSuitePerformance() {
        // Test that the suite itself runs in reasonable time
        let config = RegressionTestConfig(
            iterations: 100,
            warmupIterations: 10
        )
        let suite = PerformanceRegressionSuite(config: config)
        
        measure {
            _ = suite.runAllTests()
        }
    }
}