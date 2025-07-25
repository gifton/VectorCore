import XCTest
@testable import VectorCore

/// Central configuration for all VectorCore tests
public struct TestConfiguration {
    
    // MARK: - Test Execution Settings
    
    /// Global test configuration
    public struct Global {
        /// Enable verbose logging during tests
        public static let verboseLogging = ProcessInfo.processInfo.environment["VECTORCORE_TEST_VERBOSE"] != nil
        
        /// Run extended test suites
        public static let runExtendedTests = ProcessInfo.processInfo.environment["VECTORCORE_TEST_EXTENDED"] != nil
        
        /// Performance test iteration count
        public static let performanceIterations = Int(ProcessInfo.processInfo.environment["VECTORCORE_PERF_ITERATIONS"] ?? "100") ?? 100
        
        /// Property test iteration count  
        public static let propertyTestIterations = Int(ProcessInfo.processInfo.environment["VECTORCORE_PROPERTY_ITERATIONS"] ?? "100") ?? 100
        
        /// Random seed for reproducible tests
        public static let randomSeed = UInt64(ProcessInfo.processInfo.environment["VECTORCORE_TEST_SEED"] ?? "42") ?? 42
    }
    
    // MARK: - Coverage Requirements
    
    /// Coverage targets for different components
    public struct CoverageTargets {
        public static let overall: Double = 0.85
        public static let coreOperations: Double = 0.95
        public static let storageLayer: Double = 0.90
        public static let distanceMetrics: Double = 0.95
        public static let batchOperations: Double = 0.85
        public static let performanceBaseline: Double = 0.90
        public static let errorHandling: Double = 1.00
        public static let protocolConformance: Double = 1.00
    }
    
    // MARK: - Performance Thresholds
    
    /// Performance regression thresholds
    public struct PerformanceThresholds {
        /// Maximum acceptable regression percentage
        public static let regressionThreshold: Double = 0.05 // 5%
        
        /// Minimum improvement to be considered significant
        public static let improvementThreshold: Double = 0.02 // 2%
        
        /// Maximum acceptable memory increase
        public static let memoryRegressionThreshold: Double = 0.10 // 10%
        
        /// Timeout for individual benchmarks
        public static let benchmarkTimeout: TimeInterval = 60.0
    }
    
    // MARK: - Test Categories
    
    /// Test category weights for balanced test suite
    public struct TestCategoryWeights {
        public static let unitTests: Double = 0.60
        public static let integrationTests: Double = 0.20
        public static let propertyTests: Double = 0.15
        public static let performanceTests: Double = 0.05
    }
    
    // MARK: - Quality Metrics
    
    /// Code quality thresholds
    public struct QualityMetrics {
        /// Maximum cyclomatic complexity per function
        public static let maxCyclomaticComplexity = 10
        
        /// Maximum file length (lines)
        public static let maxFileLength = 500
        
        /// Maximum function length (lines)
        public static let maxFunctionLength = 50
        
        /// Minimum assertions per test
        public static let minAssertionsPerTest = 1
        
        /// Maximum assertions per test (to encourage focused tests)
        public static let maxAssertionsPerTest = 10
    }
    
    // MARK: - Test Execution Limits
    
    /// Execution time limits for different test types
    public struct ExecutionLimits {
        /// Maximum time for a unit test
        public static let unitTestTimeout: TimeInterval = 0.001 // 1ms
        
        /// Maximum time for an integration test
        public static let integrationTestTimeout: TimeInterval = 0.1 // 100ms
        
        /// Maximum time for a property test iteration
        public static let propertyTestIterationTimeout: TimeInterval = 0.01 // 10ms
        
        /// Maximum time for the entire test suite
        public static let totalTestSuiteTimeout: TimeInterval = 300.0 // 5 minutes
    }
}

// MARK: - Test Environment Setup

/// Base class for VectorCore tests with common setup
open class VectorCoreTestCase: XCTestCase {
    
    /// Shared random generator for reproducible tests
    public var randomGenerator: SeededRandomGenerator!
    
    /// Performance metrics collector
    public var metricsCollector: TestMetricsCollector!
    
    open override func setUp() {
        super.setUp()
        
        // Initialize random generator with seed
        randomGenerator = SeededRandomGenerator(seed: TestConfiguration.Global.randomSeed)
        
        // Initialize metrics collector
        metricsCollector = TestMetricsCollector()
        
        // Configure test environment
        if TestConfiguration.Global.verboseLogging {
            print("🧪 Starting test: \(name)")
        }
    }
    
    open override func tearDown() {
        super.tearDown()
        
        // Report metrics if verbose
        if TestConfiguration.Global.verboseLogging {
            metricsCollector.report()
        }
    }
    
    /// Measure and assert execution time
    public func measureExecutionTime(
        timeout: TimeInterval,
        file: StaticString = #file,
        line: UInt = #line,
        _ block: () throws -> Void
    ) rethrows {
        let start = CFAbsoluteTimeGetCurrent()
        try block()
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        
        XCTAssertLessThan(elapsed, timeout,
                         "Execution time \(elapsed)s exceeded timeout \(timeout)s",
                         file: file, line: line)
        
        metricsCollector.recordExecutionTime(elapsed)
    }
}

// MARK: - Test Metrics Collection

/// Collects metrics during test execution
public class TestMetricsCollector {
    private var executionTimes: [TimeInterval] = []
    private var memoryUsages: [Int64] = []
    private var startTime: CFAbsoluteTime
    
    init() {
        self.startTime = CFAbsoluteTimeGetCurrent()
    }
    
    func recordExecutionTime(_ time: TimeInterval) {
        executionTimes.append(time)
    }
    
    func recordMemoryUsage(_ bytes: Int64) {
        memoryUsages.append(bytes)
    }
    
    func report() {
        let totalTime = CFAbsoluteTimeGetCurrent() - startTime
        
        if !executionTimes.isEmpty {
            let avgTime = executionTimes.reduce(0, +) / Double(executionTimes.count)
            print("  ⏱️  Average execution time: \(String(format: "%.3f", avgTime * 1000))ms")
        }
        
        if !memoryUsages.isEmpty {
            let avgMemory = memoryUsages.reduce(0, +) / Int64(memoryUsages.count)
            print("  💾 Average memory usage: \(formatBytes(avgMemory))")
        }
        
        print("  ⏱️  Total test time: \(String(format: "%.3f", totalTime))s")
    }
    
    private func formatBytes(_ bytes: Int64) -> String {
        if bytes < 1024 {
            return "\(bytes) B"
        } else if bytes < 1024 * 1024 {
            return String(format: "%.1f KB", Double(bytes) / 1024)
        } else {
            return String(format: "%.1f MB", Double(bytes) / (1024 * 1024))
        }
    }
}

// MARK: - Test Helpers

extension VectorCoreTestCase {
    
    /// Skip test if not running extended tests
    public func skipIfNotExtended() throws {
        if !TestConfiguration.Global.runExtendedTests {
            throw XCTSkip("Skipping extended test")
        }
    }
    
    /// Run test only in specific configurations
    public func requireConfiguration(_ check: () -> Bool, reason: String) throws {
        if !check() {
            throw XCTSkip(reason)
        }
    }
}

// MARK: - Memory Helpers

private func getMemoryUsage() -> Int64 {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
    
    let result = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
            task_info(mach_task_self_,
                     task_flavor_t(MACH_TASK_BASIC_INFO),
                     $0,
                     &count)
        }
    }
    
    return result == KERN_SUCCESS ? Int64(info.resident_size) : 0
}