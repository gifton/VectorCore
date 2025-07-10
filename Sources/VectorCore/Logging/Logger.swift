// VectorCore: Simple Logging System
//
// Lightweight logging for debugging and error reporting
//

import Foundation
import os.log

/// Log levels for VectorCore
public enum LogLevel: Int, Comparable, Sendable {
    case debug = 0
    case info = 1
    case warning = 2
    case error = 3
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

/// Configuration for logging
public final class LogConfiguration: @unchecked Sendable {
    private var _minimumLevel: LogLevel
    private let lock = NSLock()
    
    public var minimumLevel: LogLevel {
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
public struct Logger: Sendable {
    private let subsystem: String
    private let category: String
    private let osLog: OSLog
    
    /// Shared logging configuration
    public static let configuration = LogConfiguration()
    
    /// Create a logger for a specific subsystem
    public init(subsystem: String = "com.vectorcore", category: String) {
        self.subsystem = subsystem
        self.category = category
        self.osLog = OSLog(subsystem: subsystem, category: category)
    }
    
    /// Log a debug message
    public func debug(_ message: @autoclosure () -> String, 
                     file: String = #fileID,
                     function: String = #function,
                     line: Int = #line) {
        log(level: .debug, message(), file: file, function: function, line: line)
    }
    
    /// Log an info message
    public func info(_ message: @autoclosure () -> String,
                    file: String = #fileID,
                    function: String = #function,
                    line: Int = #line) {
        log(level: .info, message(), file: file, function: function, line: line)
    }
    
    /// Log a warning message
    public func warning(_ message: @autoclosure () -> String,
                       file: String = #fileID,
                       function: String = #function,
                       line: Int = #line) {
        log(level: .warning, message(), file: file, function: function, line: line)
    }
    
    /// Log an error message
    public func error(_ message: @autoclosure () -> String,
                     file: String = #fileID,
                     function: String = #function,
                     line: Int = #line) {
        log(level: .error, message(), file: file, function: function, line: line)
    }
    
    /// Log a critical message
    public func critical(_ message: @autoclosure () -> String,
                        file: String = #fileID,
                        function: String = #function,
                        line: Int = #line) {
        log(level: .critical, message(), file: file, function: function, line: line)
    }
    
    /// Log an error with VectorError
    public func error(_ error: VectorError,
                     message: String? = nil,
                     file: String = #fileID,
                     function: String = #function,
                     line: Int = #line) {
        let errorMessage = message ?? error.description
        log(level: .error, "\(errorMessage): \(error)", file: file, function: function, line: line)
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

/// Core operations logger
public let coreLogger = Logger(category: "Core")

/// Storage operations logger  
public let storageLogger = Logger(category: "Storage")

/// Batch operations logger
public let batchLogger = Logger(category: "Batch")

/// Distance metrics logger
public let metricsLogger = Logger(category: "Metrics")

/// Performance logger
public let performanceLogger = Logger(category: "Performance")

// MARK: - Performance Logging

/// Simple performance timer for logging
public struct PerformanceTimer {
    private let start: CFAbsoluteTime
    private let operation: String
    private let logger: Logger
    
    /// Start timing an operation
    public init(operation: String, logger: Logger = performanceLogger) {
        self.start = CFAbsoluteTimeGetCurrent()
        self.operation = operation
        self.logger = logger
    }
    
    /// Log the elapsed time
    public func log(threshold: TimeInterval = 0.001) {
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        if elapsed >= threshold {
            logger.debug("\(operation) took \(String(format: "%.3f", elapsed * 1000))ms")
        }
    }
}

// MARK: - Logging Extensions

extension VectorError {
    /// Log this error with context
    public func log(to logger: Logger = coreLogger, 
                   message: String? = nil,
                   file: String = #fileID,
                   function: String = #function,
                   line: Int = #line) {
        logger.error(self, message: message, file: file, function: function, line: line)
    }
}