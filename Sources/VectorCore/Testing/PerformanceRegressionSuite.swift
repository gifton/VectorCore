// VectorCore: Performance Regression Test Suite
//
// Automated performance regression detection for VectorCore operations
//

import Foundation
import Accelerate

/// Performance regression test configuration
public struct RegressionTestConfig {
    /// Number of iterations for each test
    public let iterations: Int
    
    /// Acceptable performance variance (percentage)
    public let acceptableVariance: Double
    
    /// Whether to fail on regression
    public let failOnRegression: Bool
    
    /// Warmup iterations before measurement
    public let warmupIterations: Int
    
    /// Output format for results
    public let outputFormat: OutputFormat
    
    public enum OutputFormat {
        case plain
        case json
        case markdown
    }
    
    public init(
        iterations: Int = 1000,
        acceptableVariance: Double = 0.1, // 10%
        failOnRegression: Bool = true,
        warmupIterations: Int = 100,
        outputFormat: OutputFormat = .plain
    ) {
        self.iterations = iterations
        self.acceptableVariance = acceptableVariance
        self.failOnRegression = failOnRegression
        self.warmupIterations = warmupIterations
        self.outputFormat = outputFormat
    }
}

/// Result of a performance test
public struct PerformanceResult: Sendable {
    public let testName: String
    public let dimension: Int
    public let meanTime: Double
    public let stdDeviation: Double
    public let minTime: Double
    public let maxTime: Double
    public let throughput: Double // operations per second
    
    public init(
        testName: String,
        dimension: Int,
        meanTime: Double,
        stdDeviation: Double,
        minTime: Double,
        maxTime: Double,
        throughput: Double
    ) {
        self.testName = testName
        self.dimension = dimension
        self.meanTime = meanTime
        self.stdDeviation = stdDeviation
        self.minTime = minTime
        self.maxTime = maxTime
        self.throughput = throughput
    }
    
    public var formattedMeanTime: String {
        formatTime(meanTime)
    }
    
    private func formatTime(_ time: Double) -> String {
        if time < 1e-6 {
            return String(format: "%.2f ns", time * 1e9)
        } else if time < 1e-3 {
            return String(format: "%.2f μs", time * 1e6)
        } else {
            return String(format: "%.2f ms", time * 1e3)
        }
    }
}

/// Regression detection result
public struct RegressionResult: Sendable {
    public let test: String
    public let baseline: PerformanceResult
    public let current: PerformanceResult
    public let percentageChange: Double
    public let isRegression: Bool
    public let isImprovement: Bool
    
    public var summary: String {
        let changeStr = percentageChange > 0 ? "+\(percentageChange)%" : "\(percentageChange)%"
        if isRegression {
            return "❌ REGRESSION: \(test) - \(changeStr) slower"
        } else if isImprovement {
            return "✅ IMPROVEMENT: \(test) - \(abs(percentageChange))% faster"
        } else {
            return "✓ STABLE: \(test) - \(changeStr)"
        }
    }
}

/// Performance baseline storage
public struct PerformanceBaseline: Codable {
    public let version: String
    public let platform: String
    public let date: Date
    public let results: [String: PerformanceResult]
    
    public init(
        version: String,
        platform: String,
        date: Date,
        results: [String: PerformanceResult]
    ) {
        self.version = version
        self.platform = platform
        self.date = date
        self.results = results
    }
    
    public static func load(from url: URL) throws -> PerformanceBaseline {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .secondsSince1970
        return try decoder.decode(PerformanceBaseline.self, from: data)
    }
    
    public func save(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .secondsSince1970
        let data = try encoder.encode(self)
        try data.write(to: url)
    }
}

// Make PerformanceResult Codable
extension PerformanceResult: Codable {}

/// Performance regression test suite
public final class PerformanceRegressionSuite {
    private let config: RegressionTestConfig
    internal var results: [String: PerformanceResult] = [:]
    
    public init(config: RegressionTestConfig = RegressionTestConfig()) {
        self.config = config
    }
    
    // MARK: - Test Execution
    
    /// Run all performance tests
    public func runAllTests() -> [String: PerformanceResult] {
        results.removeAll()
        
        // Basic Operations
        runBasicOperationTests()
        
        // Advanced Operations
        runAdvancedOperationTests()
        
        // Storage Tests
        runStorageTests()
        
        // Batch Operations
        runBatchOperationTests()
        
        // Memory Tests
        runMemoryTests()
        
        // Feature Tests
        runFeatureTests()
        
        return results
    }
    
    // MARK: - Basic Operations
    
    private func runBasicOperationTests() {
        // Addition
        measurePerformance(name: "Vector Addition (Dim32)", dimension: 32) {
            let v1 = Vector<Dim32>.random(in: -1...1)
            let v2 = Vector<Dim32>.random(in: -1...1)
            return { _ = v1 + v2 }
        }
        
        measurePerformance(name: "Vector Addition (Dim128)", dimension: 128) {
            let v1 = Vector<Dim128>.random(in: -1...1)
            let v2 = Vector<Dim128>.random(in: -1...1)
            return { _ = v1 + v2 }
        }
        
        measurePerformance(name: "Vector Addition (Dim1536)", dimension: 1536) {
            let v1 = Vector<Dim1536>.random(in: -1...1)
            let v2 = Vector<Dim1536>.random(in: -1...1)
            return { _ = v1 + v2 }
        }
        
        // Dot Product
        measurePerformance(name: "Dot Product (Dim32)", dimension: 32) {
            let v1 = Vector<Dim32>.random(in: -1...1)
            let v2 = Vector<Dim32>.random(in: -1...1)
            return { _ = v1.dotProduct(v2) }
        }
        
        measurePerformance(name: "Dot Product (Dim1536)", dimension: 1536) {
            let v1 = Vector<Dim1536>.random(in: -1...1)
            let v2 = Vector<Dim1536>.random(in: -1...1)
            return { _ = v1.dotProduct(v2) }
        }
        
        // Normalization
        measurePerformance(name: "Normalization (Dim128)", dimension: 128) {
            let v = Vector<Dim128>.random(in: -10...10)
            return { _ = v.normalized() }
        }
        
        measurePerformance(name: "Normalization (Dim1536)", dimension: 1536) {
            let v = Vector<Dim1536>.random(in: -10...10)
            return { _ = v.normalized() }
        }
    }
    
    // MARK: - Advanced Operations
    
    private func runAdvancedOperationTests() {
        // Distance calculations
        measurePerformance(name: "Euclidean Distance (Dim128)", dimension: 128) {
            let v1 = Vector<Dim128>.random(in: -1...1)
            let v2 = Vector<Dim128>.random(in: -1...1)
            let metric = EuclideanDistance()
            return { _ = metric.distance(v1, v2) }
        }
        
        measurePerformance(name: "Cosine Distance (Dim768)", dimension: 768) {
            let v1 = Vector<Dim768>.random(in: -1...1)
            let v2 = Vector<Dim768>.random(in: -1...1)
            let metric = CosineDistance()
            return { _ = metric.distance(v1, v2) }
        }
        
        // Matrix operations
        measurePerformance(name: "Matrix Multiply (128x128)", dimension: 128) {
            let vectors = (0..<128).map { _ in Vector<Dim128>.random(in: -1...1) }
            let query = Vector<Dim128>.random(in: -1...1)
            return {
                _ = vectors.map { $0.dotProduct(query) }
            }
        }
    }
    
    // MARK: - Storage Tests
    
    private func runStorageTests() {
        // COW behavior
        measurePerformance(name: "COW Trigger (Dim1536)", dimension: 1536) {
            let original = Vector<Dim1536>.random(in: -1...1)
            return {
                var copy = original
                copy[0] = 0 // Trigger COW
            }
        }
        
        // Storage allocation
        measurePerformance(name: "Vector Creation (Small)", dimension: 32) {
            return { _ = Vector<Dim32>.random(in: -1...1) }
        }
        
        measurePerformance(name: "Vector Creation (Medium)", dimension: 128) {
            return { _ = Vector<Dim128>.random(in: -1...1) }
        }
        
        measurePerformance(name: "Vector Creation (Large)", dimension: 1536) {
            return { _ = Vector<Dim1536>.random(in: -1...1) }
        }
    }
    
    // MARK: - Batch Operations
    
    private func runBatchOperationTests() {
        // k-NN search
        measurePerformance(name: "k-NN Search (100 vectors)", dimension: 128) {
            let vectors = (0..<100).map { _ in Vector<Dim128>.random(in: -1...1) }
            let query = Vector<Dim128>.random(in: -1...1)
            return {
                _ = SyncBatchOperations.findNearest(to: query, in: vectors, k: 10)
            }
        }
        
        measurePerformance(name: "k-NN Search (1000 vectors)", dimension: 128) {
            let vectors = (0..<1000).map { _ in Vector<Dim128>.random(in: -1...1) }
            let query = Vector<Dim128>.random(in: -1...1)
            return {
                _ = SyncBatchOperations.findNearest(to: query, in: vectors, k: 10)
            }
        }
        
        // Batch transformations
        measurePerformance(name: "Batch Normalization (100 vectors)", dimension: 256) {
            let vectors = (0..<100).map { _ in Vector<Dim256>.random(in: -10...10) }
            return {
                _ = vectors.map { $0.normalized() }
            }
        }
        
        // Centroid calculation
        measurePerformance(name: "Centroid (100 vectors)", dimension: 512) {
            let vectors = (0..<100).map { _ in Vector<Dim512>.random(in: -1...1) }
            return {
                _ = SyncBatchOperations.centroid(of: vectors)
            }
        }
    }
    
    // MARK: - Memory Tests
    
    private func runMemoryTests() {
        // Memory allocation patterns
        measurePerformance(name: "Repeated Allocations (Dim128)", dimension: 128) {
            return {
                var vectors: [Vector<Dim128>] = []
                for _ in 0..<100 {
                    vectors.append(Vector<Dim128>.random(in: -1...1))
                }
                vectors.removeAll(keepingCapacity: false)
            }
        }
        
        // Large vector operations
        measurePerformance(name: "Large Vector Sum (Dim3072)", dimension: 3072) {
            let v1 = Vector<Dim3072>.random(in: -1...1)
            let v2 = Vector<Dim3072>.random(in: -1...1)
            return { _ = v1 + v2 }
        }
    }
    
    // MARK: - Feature Tests
    
    private func runFeatureTests() {
        // Element-wise operations
        measurePerformance(name: "Element-wise Min (Dim512)", dimension: 512) {
            let v1 = Vector<Dim512>.random(in: -10...10)
            let v2 = Vector<Dim512>.random(in: -10...10)
            return { _ = v1.min(v2) }
        }
        
        measurePerformance(name: "Clamp Operation (Dim512)", dimension: 512) {
            let v = Vector<Dim512>.random(in: -100...100)
            return { _ = v.clamped(to: -10...10) }
        }
        
        // Convenience initializers
        measurePerformance(name: "Linspace Creation (Dim256)", dimension: 256) {
            return { _ = Vector<Dim256>.linspace(from: 0, to: 1) }
        }
        
        // NaN handling
        measurePerformance(name: "NaN Check (Dim1536)", dimension: 1536) {
            let v = Vector<Dim1536>.random(in: -1...1)
            return { _ = v.isFinite }
        }
        
        measurePerformance(name: "NaN Replacement (Dim768)", dimension: 768) {
            var values = Array(repeating: Float(1.0), count: 768)
            for i in stride(from: 0, to: 768, by: 50) {
                values[i] = .nan
            }
            let v = Vector<Dim768>(values)
            return { _ = try? v.handleNonFinite(options: .replaceAll) }
        }
    }
    
    // MARK: - Performance Measurement
    
    private func measurePerformance(
        name: String,
        dimension: Int,
        setup: () -> (() -> Void)
    ) {
        let operation = setup()
        
        // Warmup
        for _ in 0..<config.warmupIterations {
            operation()
        }
        
        // Measure
        var times: [Double] = []
        times.reserveCapacity(config.iterations)
        
        for _ in 0..<config.iterations {
            let start = CFAbsoluteTimeGetCurrent()
            operation()
            let end = CFAbsoluteTimeGetCurrent()
            times.append(end - start)
        }
        
        // Calculate statistics
        let meanTime = times.reduce(0, +) / Double(times.count)
        let variance = times.reduce(0) { $0 + pow($1 - meanTime, 2) } / Double(times.count)
        let stdDev = sqrt(variance)
        let minTime = times.min() ?? 0
        let maxTime = times.max() ?? 0
        let throughput = 1.0 / meanTime
        
        let result = PerformanceResult(
            testName: name,
            dimension: dimension,
            meanTime: meanTime,
            stdDeviation: stdDev,
            minTime: minTime,
            maxTime: maxTime,
            throughput: throughput
        )
        
        results[name] = result
    }
    
    // MARK: - Regression Detection
    
    /// Compare current results against baseline
    public func compareAgainstBaseline(
        _ baseline: PerformanceBaseline
    ) -> [RegressionResult] {
        var regressionResults: [RegressionResult] = []
        
        for (testName, currentResult) in results {
            guard let baselineResult = baseline.results[testName] else {
                // New test, no baseline
                continue
            }
            
            let percentageChange = ((currentResult.meanTime - baselineResult.meanTime) / baselineResult.meanTime) * 100
            let isRegression = percentageChange > config.acceptableVariance * 100
            let isImprovement = percentageChange < -config.acceptableVariance * 100
            
            let regression = RegressionResult(
                test: testName,
                baseline: baselineResult,
                current: currentResult,
                percentageChange: percentageChange,
                isRegression: isRegression,
                isImprovement: isImprovement
            )
            
            regressionResults.append(regression)
        }
        
        return regressionResults.sorted { $0.test < $1.test }
    }
    
    // MARK: - Result Formatting
    
    /// Format results based on output format
    public func formatResults(_ results: [String: PerformanceResult]) -> String {
        switch config.outputFormat {
        case .plain:
            return formatPlainResults(results)
        case .json:
            return formatJSONResults(results)
        case .markdown:
            return formatMarkdownResults(results)
        }
    }
    
    private func formatPlainResults(_ results: [String: PerformanceResult]) -> String {
        var output = "=== Performance Test Results ===\n\n"
        
        let sortedResults = results.sorted { $0.key < $1.key }
        for (_, result) in sortedResults {
            output += "\(result.testName):\n"
            output += "  Mean: \(result.formattedMeanTime)\n"
            output += "  Std Dev: \(formatTime(result.stdDeviation))\n"
            output += "  Min: \(formatTime(result.minTime))\n"
            output += "  Max: \(formatTime(result.maxTime))\n"
            output += "  Throughput: \(String(format: "%.2f", result.throughput)) ops/sec\n\n"
        }
        
        return output
    }
    
    private func formatJSONResults(_ results: [String: PerformanceResult]) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        
        guard let data = try? encoder.encode(results),
              let json = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        
        return json
    }
    
    private func formatMarkdownResults(_ results: [String: PerformanceResult]) -> String {
        var output = "# Performance Test Results\n\n"
        output += "| Test | Dimension | Mean Time | Std Dev | Throughput |\n"
        output += "|------|-----------|-----------|---------|------------|\n"
        
        let sortedResults = results.sorted { $0.key < $1.key }
        for (_, result) in sortedResults {
            output += "| \(result.testName) "
            output += "| \(result.dimension) "
            output += "| \(result.formattedMeanTime) "
            output += "| \(formatTime(result.stdDeviation)) "
            output += "| \(String(format: "%.2f", result.throughput)) ops/s |\n"
        }
        
        return output
    }
    
    private func formatTime(_ time: Double) -> String {
        if time < 1e-6 {
            return String(format: "%.2f ns", time * 1e9)
        } else if time < 1e-3 {
            return String(format: "%.2f μs", time * 1e6)
        } else {
            return String(format: "%.2f ms", time * 1e3)
        }
    }
    
    /// Format regression results
    public func formatRegressionResults(_ regressions: [RegressionResult]) -> String {
        var output = "=== Regression Analysis ===\n\n"
        
        let actualRegressions = regressions.filter { $0.isRegression }
        let improvements = regressions.filter { $0.isImprovement }
        let stable = regressions.filter { !$0.isRegression && !$0.isImprovement }
        
        if !actualRegressions.isEmpty {
            output += "❌ REGRESSIONS DETECTED:\n"
            for reg in actualRegressions {
                output += "  - \(reg.test): \(String(format: "%.2f%%", reg.percentageChange)) slower\n"
                output += "    Baseline: \(reg.baseline.formattedMeanTime) → Current: \(reg.current.formattedMeanTime)\n"
            }
            output += "\n"
        }
        
        if !improvements.isEmpty {
            output += "✅ IMPROVEMENTS:\n"
            for imp in improvements {
                output += "  - \(imp.test): \(String(format: "%.2f%%", abs(imp.percentageChange))) faster\n"
                output += "    Baseline: \(imp.baseline.formattedMeanTime) → Current: \(imp.current.formattedMeanTime)\n"
            }
            output += "\n"
        }
        
        output += "📊 SUMMARY:\n"
        output += "  - Total tests: \(regressions.count)\n"
        output += "  - Regressions: \(actualRegressions.count)\n"
        output += "  - Improvements: \(improvements.count)\n"
        output += "  - Stable: \(stable.count)\n"
        
        return output
    }
}

// MARK: - Convenience Methods

public extension PerformanceRegressionSuite {
    /// Run tests and check for regressions in one call
    func runAndCheckRegressions(
        baselineURL: URL? = nil
    ) throws -> (results: [String: PerformanceResult], regressions: [RegressionResult]?) {
        let results = runAllTests()
        
        var regressions: [RegressionResult]? = nil
        if let baselineURL = baselineURL,
           let baseline = try? PerformanceBaseline.load(from: baselineURL) {
            regressions = compareAgainstBaseline(baseline)
            
            if config.failOnRegression {
                let hasRegressions = regressions?.contains { $0.isRegression } ?? false
                if hasRegressions {
                    throw RegressionError.regressionsDetected(regressions!)
                }
            }
        }
        
        return (results, regressions)
    }
    
    /// Create baseline from current results
    func createBaseline() -> PerformanceBaseline {
        PerformanceBaseline(
            version: "1.0.0", // You might want to get this from your package
            platform: "\(ProcessInfo.processInfo.operatingSystemVersionString)",
            date: Date(),
            results: results
        )
    }
}

/// Regression detection error
public enum RegressionError: Error, LocalizedError {
    case regressionsDetected([RegressionResult])
    
    public var errorDescription: String? {
        switch self {
        case .regressionsDetected(let regressions):
            let count = regressions.filter { $0.isRegression }.count
            return "Performance regressions detected in \(count) tests"
        }
    }
}