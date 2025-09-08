// VectorCore: Simple Logging System
//
// Lightweight logging for debugging and error reporting
//

import Foundation
import os.log

/// Log levels for VectorCore logging system.
///
/// `LogLevel` defines the severity hierarchy for log messages,
/// from detailed debugging information to critical errors.
///
/// ## Levels (in order of severity)
/// - **debug**: Detailed information for debugging
/// - **info**: General informational messages
/// - **warning**: Potentially harmful situations
/// - **error**: Error events that might still allow continued operation
/// - **critical**: Severe errors that likely cause termination
///
/// ## Example Usage
/// ```swift
/// logger.log("Vector operation completed", level: .info)
/// logger.log("Memory allocation failed", level: .critical)
/// ```
enum LogLevel: Int, Comparable, Sendable {
    /// Detailed information for debugging purposes.
    case debug = 0
    
    /// General informational messages.
    case info = 1
    
    /// Warning messages for potentially harmful situations.
    case warning = 2
    
    /// Error events that might still allow continued operation.
    case error = 3
    
    /// Critical errors that likely cause termination.
    case critical = 4
    
    public static func < (lhs: LogLevel, rhs: LogLevel) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
    
    var symbol: String {
        switch self {
        case .debug: return "🔍"
        case .info: return "ℹ️"
        case .warning: return "⚠️"
        case .error: return "❌"
        case .critical: return "🚨"
        }
    }
    
    var name: String {
        switch self {
        case .debug: return "DEBUG"
        case .info: return "INFO"
        case .warning: return "WARNING"
        case .error: return "ERROR"
        case .critical: return "CRITICAL"
        }
    }
}

/// Configuration for the VectorCore logging system.
///
/// `LogConfiguration` provides thread-safe configuration for controlling
/// logging behavior, including minimum log levels and output options.
///
/// ## Example Usage
/// ```swift
/// LogConfiguration.shared.minimumLevel = .warning
/// LogConfiguration.shared.enableOSLog = true
/// ```
final class LogConfiguration: @unchecked Sendable {
    private var _minimumLevel: LogLevel
    private let lock = NSLock()
    
    /// Minimum log level to output.
    ///
    /// Messages below this level are ignored.
    /// Thread-safe property with internal locking.
    var minimumLevel: LogLevel {
        get {
            lock.lock()
            defer { lock.unlock() }
            return _minimumLevel
        }
        set {
            lock.lock()
            defer { lock.unlock() }
            _minimumLevel = newValue
        }
    }
    
    init() {
        #if DEBUG
        self._minimumLevel = .debug
        #else
        self._minimumLevel = .warning
        #endif
    }
}

/// Simple logger for VectorCore
struct Logger: Sendable {
    private let subsystem: String
    private let category: String
    private let osLog: OSLog
    
    /// Shared logging configuration
    static let configuration = LogConfiguration()
    
    /// Create a logger for a specific subsystem
    init(subsystem: String = "com.vectorcore", category: String) {
        self.subsystem = subsystem
        self.category = category
        self.osLog = OSLog(subsystem: subsystem, category: category)
    }
    
    /// Log a debug message
    func debug(_ message: @autoclosure () -> String, 
                     file: String = #fileID,
                     function: String = #function,
                     line: Int = #line) {
        log(level: .debug, message(), file: file, function: function, line: line)
    }
    
    /// Log an info message
    func info(_ message: @autoclosure () -> String,
                    file: String = #fileID,
                    function: String = #function,
                    line: Int = #line) {
        log(level: .info, message(), file: file, function: function, line: line)
    }
    
    /// Log a warning message
    func warning(_ message: @autoclosure () -> String,
                       file: String = #fileID,
                       function: String = #function,
                       line: Int = #line) {
        log(level: .warning, message(), file: file, function: function, line: line)
    }
    
    /// Log an error message
    func error(_ message: @autoclosure () -> String,
                     file: String = #fileID,
                     function: String = #function,
                     line: Int = #line) {
        log(level: .error, message(), file: file, function: function, line: line)
    }
    
    /// Log a critical message
    func critical(_ message: @autoclosure () -> String,
                        file: String = #fileID,
                        function: String = #function,
                        line: Int = #line) {
        log(level: .critical, message(), file: file, function: function, line: line)
    }
    
    /// Log an error with VectorError
    func error(_ error: VectorError,
                     message: String? = nil,
                     file: String = #fileID,
                     function: String = #function,
                     line: Int = #line) {
        let errorMessage = message ?? error.description
        log(level: .error, "\(errorMessage)", file: file, function: function, line: line)
    }
    
    /// Log with specific level
    private func log(level: LogLevel, 
                    _ message: String,
                    file: String,
                    function: String,
                    line: Int) {
        guard level >= Self.configuration.minimumLevel else { return }
        
        let fileName = URL(fileURLWithPath: file).lastPathComponent
        let location = "\(fileName):\(line)"
        
        // Use os_log for system integration
        let osLogType: OSLogType = {
            switch level {
            case .debug: return .debug
            case .info: return .info
            case .warning, .error: return .error
            case .critical: return .fault
            }
        }()
        
        os_log("%{public}@ [%{public}@] %{public}@ - %{public}@",
               log: osLog,
               type: osLogType,
               level.name,
               location,
               function,
               message)
        
        // Also print to console in debug builds
        #if DEBUG
        print("\(level.symbol) [\(category)] \(message) (\(location))")
        #endif
    }
}

// MARK: - Global Loggers

/// Logger for core vector operations.
///
/// Use for logging vector creation, manipulation, and basic operations.
let coreLogger = Logger(category: "Core")

/// Logger for storage and memory operations.
///
/// Use for logging memory allocation, deallocation, and storage management.
let storageLogger = Logger(category: "Storage")

/// Logger for batch operations.
///
/// Use for logging batch processing, parallel operations, and bulk computations.
let batchLogger = Logger(category: "Batch")

/// Logger for distance metric calculations.
///
/// Use for logging distance computations, similarity measurements, and metric selection.
let metricsLogger = Logger(category: "Metrics")

/// Logger for performance monitoring.
///
/// Use for logging timing information, benchmarks, and performance analysis.
let performanceLogger = Logger(category: "Performance")

// MARK: - Performance Logging

/// Simple performance timer for logging
struct PerformanceTimer {
    private let start: CFAbsoluteTime
    private let operation: String
    private let logger: Logger
    
    /// Start timing an operation
    init(operation: String, logger: Logger = performanceLogger) {
        self.start = CFAbsoluteTimeGetCurrent()
        self.operation = operation
        self.logger = logger
    }
    
    /// Log the elapsed time
    func log(threshold: TimeInterval = 0.001) {
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        if elapsed >= threshold {
            logger.debug("\(operation) took \(String(format: "%.3f", elapsed * 1000))ms")
        }
    }
}

// MARK: - Logging Extensions

extension VectorError {
    /// Log this error with context
    func log(to logger: Logger = coreLogger, 
                   message: String? = nil,
                   file: String = #fileID,
                   function: String = #function,
                   line: Int = #line) {
        logger.error(self, message: message, file: file, function: function, line: line)
    }
}
