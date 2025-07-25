#!/usr/bin/swift

// VectorCore: Performance Regression Example
//
// Demonstrates how to use the performance regression suite
//

import VectorCore
import Foundation

// MARK: - Example 1: Basic Performance Testing

print("=== Basic Performance Testing ===\n")

// Create a performance test suite with default configuration
let suite = PerformanceRegressionSuite()

// Run all performance tests
print("Running performance tests...")
let results = suite.runAllTests()

// Display results
print("\nTest Results:")
for (testName, result) in results.sorted(by: { $0.key < $1.key }) {
    print("  \(testName): \(result.formattedMeanTime)")
}

// MARK: - Example 2: Custom Configuration

print("\n\n=== Custom Configuration ===\n")

// Create suite with custom configuration
let customConfig = RegressionTestConfig(
    iterations: 500,           // Fewer iterations for faster testing
    acceptableVariance: 0.15, // Allow 15% variance
    failOnRegression: false,  // Don't fail, just report
    warmupIterations: 50,     // Warmup iterations
    outputFormat: .markdown   // Markdown output
)

let customSuite = PerformanceRegressionSuite(config: customConfig)

// Run specific test categories
print("Running subset of tests...")
// Note: In a real implementation, you might want to add methods to run specific test groups

// MARK: - Example 3: Creating and Using Baselines

print("\n\n=== Baseline Management ===\n")

// Create a baseline from current performance
let baseline = suite.createBaseline()
print("Created baseline with \(baseline.results.count) test results")
print("Platform: \(baseline.platform)")
print("Date: \(baseline.date)")

// Save baseline to file
let baselineURL = FileManager.default.temporaryDirectory
    .appendingPathComponent("vectorcore_baseline.json")

do {
    try baseline.save(to: baselineURL)
    print("Baseline saved to: \(baselineURL.path)")
} catch {
    print("Failed to save baseline: \(error)")
}

// MARK: - Example 4: Regression Detection

print("\n\n=== Regression Detection ===\n")

// For demo purposes, create a modified baseline to simulate regression
// In real usage, you would run tests again and compare
var modifiedResults = baseline.results
if let firstTest = modifiedResults.first {
    let oldResult = firstTest.value
    modifiedResults[firstTest.key] = PerformanceResult(
        testName: oldResult.testName,
        dimension: oldResult.dimension,
        meanTime: oldResult.meanTime * 1.2, // 20% slower
        stdDeviation: oldResult.stdDeviation,
        minTime: oldResult.minTime * 1.2,
        maxTime: oldResult.maxTime * 1.2,
        throughput: oldResult.throughput / 1.2
    )
}

let modifiedBaseline = RegressionTestBaseline(
    version: baseline.version,
    platform: baseline.platform,
    date: Date(),
    results: modifiedResults
)

// Compare baselines
let compareSuite = PerformanceRegressionSuite()
// Simulate that we have new results by comparing modified baseline against original
print("Simulating regression detection (comparing modified results against original):")

// This is a workaround for the demo - normally you'd run tests and compare
let simulatedResults = modifiedBaseline.results
let originalResults = baseline.results

for (testName, currentResult) in simulatedResults {
    if let baselineResult = originalResults[testName] {
        let percentageChange = ((currentResult.meanTime - baselineResult.meanTime) / baselineResult.meanTime) * 100
        if percentageChange > 10 {
            print("  ❌ \(testName): \(String(format: "%.1f%%", percentageChange)) slower")
        } else if percentageChange < -10 {
            print("  ✅ \(testName): \(String(format: "%.1f%%", abs(percentageChange))) faster")
        } else {
            print("  ✓ \(testName): stable (change: \(String(format: "%.1f%%", percentageChange)))")
        }
    }
}

// MARK: - Example 5: Different Output Formats

print("\n\n=== Output Formats ===\n")

// Create a subset of results for demo
let demoResults = Dictionary(uniqueKeysWithValues: results.prefix(3).map { ($0.key, $0.value) })

// Plain text format
let plainConfig = RegressionTestConfig(iterations: 10, outputFormat: .plain)
let plainSuite = PerformanceRegressionSuite(config: plainConfig)

print("Plain Text Output:")
print(plainSuite.formatResults(demoResults))

// Markdown format
let markdownConfig = RegressionTestConfig(iterations: 10, outputFormat: .markdown)
let markdownSuite = PerformanceRegressionSuite(config: markdownConfig)

print("\nMarkdown Output:")
print(markdownSuite.formatResults(demoResults))

// JSON format
let jsonConfig = RegressionTestConfig(iterations: 10, outputFormat: .json)
let jsonSuite = PerformanceRegressionSuite(config: jsonConfig)

print("\nJSON Output (first 200 chars):")
let jsonOutput = jsonSuite.formatResults(demoResults)
print(String(jsonOutput.prefix(200)) + "...")

// MARK: - Example 6: CI/CD Integration

print("\n\n=== CI/CD Integration Example ===\n")

// Example of how to integrate with CI/CD pipelines
func runPerformanceChecksForCI() -> Bool {
    let config = RegressionTestConfig(
        iterations: 1000,
        acceptableVariance: 0.1,
        failOnRegression: true
    )
    
    let suite = PerformanceRegressionSuite(config: config)
    
    do {
        // Try to load baseline
        let baselineURL = URL(fileURLWithPath: "performance_baseline.json")
        let (_, regressions) = try suite.runAndCheckRegressions(baselineURL: baselineURL)
        
        if let regressions = regressions {
            let hasRegressions = regressions.contains { $0.isRegression }
            if hasRegressions {
                print("❌ Performance regressions detected!")
                return false
            }
        }
        
        print("✅ All performance tests passed!")
        return true
        
    } catch {
        print("⚠️  No baseline found or error occurred: \(error)")
        // In CI, you might want to create a baseline on first run
        return true
    }
}

print("CI/CD check would \(runPerformanceChecksForCI() ? "pass" : "fail")")

// MARK: - Example 7: Custom Performance Tests

print("\n\n=== Adding Custom Performance Tests ===\n")

// Example of how to add custom performance measurements
class CustomPerformanceTests {
    static func measureCustomOperation() -> PerformanceResult {
        let iterations = 1000
        var times: [Double] = []
        
        // Custom operation to measure
        let testVector = Vector<Dim512>.random(in: -1...1)
        
        // Warmup
        for _ in 0..<100 {
            _ = testVector.magnitude
        }
        
        // Measure
        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            
            // Your custom operation here
            let _ = testVector
                .normalized()
                .clamped(to: -0.5...0.5)
                .absoluteValue()
                .squareRoot()
            
            let end = CFAbsoluteTimeGetCurrent()
            times.append(end - start)
        }
        
        // Calculate statistics
        let meanTime = times.reduce(0, +) / Double(times.count)
        let variance = times.reduce(0) { $0 + pow($1 - meanTime, 2) } / Double(times.count)
        let stdDev = sqrt(variance)
        
        return PerformanceResult(
            testName: "Custom Complex Operation",
            dimension: 512,
            meanTime: meanTime,
            stdDeviation: stdDev,
            minTime: times.min() ?? 0,
            maxTime: times.max() ?? 0,
            throughput: 1.0 / meanTime
        )
    }
}

let customResult = CustomPerformanceTests.measureCustomOperation()
print("Custom test result: \(customResult.formattedMeanTime) (throughput: \(String(format: "%.2f", customResult.throughput)) ops/sec)")

// MARK: - Example 8: Performance Trends

print("\n\n=== Performance Trend Analysis ===\n")

// Simulate tracking performance over multiple runs
struct PerformanceTrend {
    let date: Date
    let commitHash: String
    let results: [String: PerformanceResult]
}

// In practice, you would load historical data
let trends: [PerformanceTrend] = []

// Function to analyze trends
func analyzeTrends(for testName: String, in trends: [PerformanceTrend]) {
    let times = trends.compactMap { $0.results[testName]?.meanTime }
    
    guard !times.isEmpty else { return }
    
    let firstTime = times.first!
    let lastTime = times.last!
    let change = ((lastTime - firstTime) / firstTime) * 100
    
    print("Performance trend for \(testName):")
    print("  - First measurement: \(formatTime(firstTime))")
    print("  - Latest measurement: \(formatTime(lastTime))")
    print("  - Overall change: \(String(format: "%.1f%%", change))")
}

func formatTime(_ time: Double) -> String {
    if time < 1e-6 {
        return String(format: "%.2f ns", time * 1e9)
    } else if time < 1e-3 {
        return String(format: "%.2f μs", time * 1e6)
    } else {
        return String(format: "%.2f ms", time * 1e3)
    }
}

// Clean up
try? FileManager.default.removeItem(at: baselineURL)

print("\n✅ Performance regression examples completed!")