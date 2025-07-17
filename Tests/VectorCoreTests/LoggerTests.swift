// VectorCore: Logger Tests
//
// Comprehensive tests for the logging system
//

import XCTest
@testable import VectorCore

final class LoggerTests: XCTestCase {
    
    // MARK: - LogLevel Tests
    
    func testLogLevelComparison() {
        // Test ordering
        XCTAssertLessThan(LogLevel.debug, LogLevel.info)
        XCTAssertLessThan(LogLevel.info, LogLevel.warning)
        XCTAssertLessThan(LogLevel.warning, LogLevel.error)
        XCTAssertLessThan(LogLevel.error, LogLevel.critical)
        
        // Test equality
        XCTAssertEqual(LogLevel.debug, LogLevel.debug)
        XCTAssertNotEqual(LogLevel.debug, LogLevel.info)
        
        // Test transitive comparison
        XCTAssertLessThan(LogLevel.debug, LogLevel.critical)
    }
    
    func testLogLevelSymbols() {
        XCTAssertEqual(LogLevel.debug.symbol, "🔍")
        XCTAssertEqual(LogLevel.info.symbol, "ℹ️")
        XCTAssertEqual(LogLevel.warning.symbol, "⚠️")
        XCTAssertEqual(LogLevel.error.symbol, "❌")
        XCTAssertEqual(LogLevel.critical.symbol, "🚨")
    }
    
    func testLogLevelNames() {
        XCTAssertEqual(LogLevel.debug.name, "DEBUG")
        XCTAssertEqual(LogLevel.info.name, "INFO")
        XCTAssertEqual(LogLevel.warning.name, "WARNING")
        XCTAssertEqual(LogLevel.error.name, "ERROR")
        XCTAssertEqual(LogLevel.critical.name, "CRITICAL")
    }
    
    func testLogLevelRawValues() {
        XCTAssertEqual(LogLevel.debug.rawValue, 0)
        XCTAssertEqual(LogLevel.info.rawValue, 1)
        XCTAssertEqual(LogLevel.warning.rawValue, 2)
        XCTAssertEqual(LogLevel.error.rawValue, 3)
        XCTAssertEqual(LogLevel.critical.rawValue, 4)
    }
    
    // MARK: - LogConfiguration Tests
    
    func testLogConfigurationDefaultLevel() {
        let config = LogConfiguration()
        
        #if DEBUG
        XCTAssertEqual(config.minimumLevel, .debug)
        #else
        XCTAssertEqual(config.minimumLevel, .warning)
        #endif
    }
    
    func testLogConfigurationSetLevel() {
        let config = LogConfiguration()
        
        // Test setting different levels
        config.minimumLevel = .error
        XCTAssertEqual(config.minimumLevel, .error)
        
        config.minimumLevel = .debug
        XCTAssertEqual(config.minimumLevel, .debug)
        
        config.minimumLevel = .critical
        XCTAssertEqual(config.minimumLevel, .critical)
    }
    
    func testLogConfigurationThreadSafety() async {
        let config = LogConfiguration()
        let iterations = 1000
        
        // Concurrent reads and writes
        await withTaskGroup(of: Void.self) { group in
            // Writers
            for level in [LogLevel.debug, .info, .warning, .error, .critical] {
                group.addTask {
                    for _ in 0..<iterations {
                        config.minimumLevel = level
                    }
                }
            }
            
            // Readers
            for _ in 0..<5 {
                group.addTask {
                    for _ in 0..<iterations {
                        _ = config.minimumLevel
                    }
                }
            }
        }
        
        // Should complete without crashes
        XCTAssertNotNil(config.minimumLevel)
    }
    
    // MARK: - Logger Creation Tests
    
    func testLoggerInitialization() {
        let logger1 = Logger(category: "Test")
        XCTAssertNotNil(logger1)
        
        let logger2 = Logger(subsystem: "com.test", category: "Test2")
        XCTAssertNotNil(logger2)
    }
    
    func testGlobalLoggers() {
        // Verify all global loggers are accessible
        XCTAssertNotNil(coreLogger)
        XCTAssertNotNil(storageLogger)
        XCTAssertNotNil(batchLogger)
        XCTAssertNotNil(metricsLogger)
        XCTAssertNotNil(performanceLogger)
    }
    
    // MARK: - Logging Methods Tests
    
    func testLogLevelFiltering() {
        let logger = Logger(category: "TestFiltering")
        
        // Save original level
        let originalLevel = Logger.configuration.minimumLevel
        
        // Test that messages below minimum level are filtered
        Logger.configuration.minimumLevel = .error
        
        // These should be filtered out (no crash test)
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        
        // These should pass through (no crash test)
        logger.error("Error message")
        logger.critical("Critical message")
        
        // Restore original level
        Logger.configuration.minimumLevel = originalLevel
    }
    
    func testAutoclosureEvaluation() {
        let logger = Logger(category: "TestAutoclosure")
        
        // Save original level
        let originalLevel = Logger.configuration.minimumLevel
        Logger.configuration.minimumLevel = .critical
        
        var evaluationCount = 0
        func expensiveMessage() -> String {
            evaluationCount += 1
            return "Expensive message"
        }
        
        // This should not evaluate the expensive message
        logger.debug(expensiveMessage())
        XCTAssertEqual(evaluationCount, 0, "Autoclosure should not evaluate when below minimum level")
        
        // This should evaluate
        logger.critical(expensiveMessage())
        XCTAssertEqual(evaluationCount, 1, "Autoclosure should evaluate when at or above minimum level")
        
        // Restore original level
        Logger.configuration.minimumLevel = originalLevel
    }
    
    func testVectorErrorLogging() {
        let logger = Logger(category: "TestError")
        
        // Test various error types
        let dimensionError = VectorError.dimensionMismatch(expected: 128, actual: 256)
        logger.error(dimensionError)
        logger.error(dimensionError, message: "Custom error message")
        
        let indexError = VectorError.indexOutOfBounds(index: 10, dimension: 5)
        logger.error(indexError)
        
        let invalidError = VectorError.invalidDataFormat(expected: "Valid", actual: "Invalid")
        logger.error(invalidError)
    }
    
    // MARK: - PerformanceTimer Tests
    
    func testPerformanceTimer() {
        // Test timer creation and logging
        let timer = PerformanceTimer(operation: "Test Operation")
        
        // Simulate some work
        Thread.sleep(forTimeInterval: 0.01) // 10ms
        
        // Log with default threshold
        timer.log()
        
        // Log with custom threshold
        timer.log(threshold: 0.001)
        timer.log(threshold: 1.0) // Should not log
    }
    
    func testPerformanceTimerWithCustomLogger() {
        let customLogger = Logger(category: "CustomPerf")
        let timer = PerformanceTimer(operation: "Custom Operation", logger: customLogger)
        
        // Simulate minimal work
        Thread.sleep(forTimeInterval: 0.002) // 2ms
        
        timer.log(threshold: 0.001)
    }
    
    // MARK: - VectorError Extension Tests
    
    func testVectorErrorLogExtension() {
        let error = VectorError.dimensionMismatch(expected: 512, actual: 256)
        
        // Test default logger
        error.log()
        
        // Test with custom logger
        let customLogger = Logger(category: "ErrorTest")
        error.log(to: customLogger)
        
        // Test with custom message
        error.log(message: "Failed to create vector")
        error.log(to: customLogger, message: "Custom error context")
    }
    
    // MARK: - Edge Cases
    
    func testEmptyMessages() {
        let logger = Logger(category: "TestEmpty")
        
        // Empty messages should not crash
        logger.debug("")
        logger.info("")
        logger.warning("")
        logger.error("")
        logger.critical("")
    }
    
    func testLongMessages() {
        let logger = Logger(category: "TestLong")
        
        // Very long message
        let longMessage = String(repeating: "A", count: 10000)
        logger.info(longMessage)
        
        // Message with special characters
        let specialMessage = "Special chars: 🚀 \n \t \\ \" ' @ # $ % ^ & * ( ) { } [ ] | : ; < > ? / ~"
        logger.info(specialMessage)
    }
    
    func testUnicodeMessages() {
        let logger = Logger(category: "TestUnicode")
        
        // Various unicode messages
        logger.info("Hello 世界 🌍")
        logger.info("Émojis: 😀😃😄😁😆😅😂🤣")
        logger.info("Math symbols: ∑ ∏ ∫ √ ∞ ≈ ≠ ≤ ≥")
        logger.info("Arrows: ← → ↑ ↓ ↔ ↕ ⇐ ⇒ ⇑ ⇓")
    }
    
    // MARK: - Performance Tests
    
    func testLoggingPerformance() {
        let logger = Logger(category: "PerfTest")
        
        measure {
            for i in 0..<1000 {
                logger.debug("Debug message \(i)")
            }
        }
    }
    
    func testFilteredLoggingPerformance() {
        let logger = Logger(category: "FilterPerfTest")
        
        // Save original level
        let originalLevel = Logger.configuration.minimumLevel
        Logger.configuration.minimumLevel = .critical
        
        // These should all be filtered out
        measure {
            for i in 0..<10000 {
                logger.debug("Filtered message \(i)")
            }
        }
        
        // Restore original level
        Logger.configuration.minimumLevel = originalLevel
    }
    
    // MARK: - Integration Tests
    
    func testLoggingInVectorOperations() {
        // Test that logging works correctly when used within vector operations
        let logger = Logger(category: "VectorOps")
        
        let vector1 = DynamicVector([1, 2, 3, 4, 5])
        let vector2 = DynamicVector([5, 4, 3, 2, 1])
        
        logger.debug("Creating vectors with dimension \(vector1.dimension)")
        
        let dotProduct = vector1.dotProduct(vector2)
        logger.info("Dot product result: \(dotProduct)")
        
        let magnitude = vector1.magnitude
        logger.info("Vector magnitude: \(magnitude)")
        
        // Test error logging
        let errorVector = DynamicVector([1, 2, 3])
        do {
            _ = try VectorFactory.create(Dim128.self, from: errorVector.toArray())
        } catch {
            if let vectorError = error as? VectorError {
                logger.error(vectorError, message: "Failed to create fixed-size vector")
            }
        }
    }
    
    // MARK: - Sendable Conformance Tests
    
    func testLoggerSendable() async {
        let logger = Logger(category: "SendableTest")
        
        // Logger should be usable across concurrent contexts
        await withTaskGroup(of: Void.self) { group in
            for i in 0..<10 {
                group.addTask {
                    logger.info("Message from task \(i)")
                }
            }
        }
    }
    
    func testLogConfigurationSendable() async {
        // LogConfiguration should be safely usable across threads
        let config = Logger.configuration
        
        await withTaskGroup(of: LogLevel.self) { group in
            for _ in 0..<10 {
                group.addTask {
                    return config.minimumLevel
                }
            }
            
            for await level in group {
                XCTAssertNotNil(level)
            }
        }
    }
}