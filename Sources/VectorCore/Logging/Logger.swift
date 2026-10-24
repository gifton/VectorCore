// VectorCore: Simple Logging System
//
// Logging for debugging and error reporting, backed by OSLog
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
/// coreLogger.info("Vector operation completed")
/// coreLogger.critical("Memory allocation failed")
/// ```
public enum LogLevel: Int, Comparable, Sendable {
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
/// Holds the minimum log level. Reads and writes of `minimumLevel` are
/// individually serialized by an internal lock.
///
/// ## Example Usage
/// ```swift
/// Logger.configuration.minimumLevel = .warning
/// ```
public final class LogConfiguration: @unchecked Sendable {
    private var _minimumLevel: LogLevel
    private let lock = NSLock()

    /// Minimum log level to output.
    ///
    /// Messages below this level are ignored.
    ///
    /// Reads and writes are individually serialized by an internal lock. A logging
    /// call reads this value once and releases the lock before it evaluates the
    /// message (see `LoggingContractTests`); a later change affects later calls only.
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

/// One admitted log record, as handed to the backend.
///
/// Internal: exists so `LoggingContractTests` can observe admitted records
/// without the OSLog/console backend. Not public API.
struct LogRecord: Sendable, Equatable {
    let level: LogLevel
    let message: String
    let location: String
    let function: String
    let category: String
}

/// Simple logger for VectorCore.
///
/// ## Message evaluation contract
/// The `message` autoclosures of `debug`/`info`/`warning`/`error`/`critical` (and the
/// fallback `error.description` of the `VectorError` overload) are evaluated only when
/// the call's level is at or above `Logger.configuration.minimumLevel`, and then
/// exactly once, outside the configuration lock. "Admitted" means this library's level
/// gate passed and the backend call was made; OSLog's own persistence/filtering is
/// independent and not promised. Exercised by `LoggingContractTests` (25-case matrix,
/// reentrancy, and `vectorErrorFallbackDescriptionIsLazy` for the `VectorError`
/// fallback) and `LoggingPublicConsumerTests`.
///
/// ## Naming next to `os.Logger`
/// Apple's `os` module also declares a `Logger`, so in a file that imports both `os` and VectorCore a bare
/// `Logger` type is ambiguous. Add `import struct VectorCore.Logger` to that file: there `Logger` names this
/// type, and Apple's stays reachable as `os.Logger`. Spelling this type `VectorCore.Logger` in client code
/// does not compile, because `VectorCore` resolves to the `VectorCore` enum rather than to the module. The
/// consumers `Scripts/ci/tests/D02/LoggerQualification.swift` (the scoped import) and `LoggerNegative.swift`
/// (the ambiguous and enum-qualified spellings) check this with the Swift 6.4 compiler.
public struct Logger: Sendable {
    private let subsystem: String
    private let category: String
    private let osLog: OSLog
    private let activeConfiguration: LogConfiguration
    private let recordSink: (@Sendable (LogRecord) -> Void)?

    /// Shared logging configuration
    public static let configuration = LogConfiguration()

    /// Create a logger for a specific subsystem
    public init(subsystem: String = "com.vectorcore", category: String) {
        self.init(subsystem: subsystem, category: category,
                  configuration: Self.configuration, sink: nil)
    }

    /// Test seam: per-instance configuration and optional record sink.
    ///
    /// With `sink == nil` the actual OSLog/Debug-console backend is used. With a sink,
    /// admitted records go to the sink instead of that backend. The public initializer
    /// always uses the shared configuration and the actual backend.
    init(subsystem: String = "com.vectorcore",
         category: String,
         configuration: LogConfiguration,
         sink: (@Sendable (LogRecord) -> Void)?) {
        self.subsystem = subsystem
        self.category = category
        self.osLog = OSLog(subsystem: subsystem, category: category)
        self.activeConfiguration = configuration
        self.recordSink = sink
    }

    /// Log a debug message
    public func debug(_ message: @autoclosure () -> String,
                      file: String = #fileID,
                      function: String = #function,
                      line: Int = #line) {
        log(level: .debug, message, file: file, function: function, line: line)
    }

    /// Log an info message
    public func info(_ message: @autoclosure () -> String,
                     file: String = #fileID,
                     function: String = #function,
                     line: Int = #line) {
        log(level: .info, message, file: file, function: function, line: line)
    }

    /// Log a warning message
    public func warning(_ message: @autoclosure () -> String,
                        file: String = #fileID,
                        function: String = #function,
                        line: Int = #line) {
        log(level: .warning, message, file: file, function: function, line: line)
    }

    /// Log an error message
    public func error(_ message: @autoclosure () -> String,
                      file: String = #fileID,
                      function: String = #function,
                      line: Int = #line) {
        log(level: .error, message, file: file, function: function, line: line)
    }

    /// Log a critical message
    public func critical(_ message: @autoclosure () -> String,
                         file: String = #fileID,
                         function: String = #function,
                         line: Int = #line) {
        log(level: .critical, message, file: file, function: function, line: line)
    }

    /// Log an error with VectorError.
    ///
    /// `error.description` is only computed when the call is admitted and `message`
    /// is nil. A caller-supplied `message` has already been evaluated by the caller.
    public func error(_ error: VectorError,
                      message: String? = nil,
                      file: String = #fileID,
                      function: String = #function,
                      line: Int = #line) {
        log(level: .error, { message ?? error.description },
            file: file, function: function, line: line)
    }

    /// Log with specific level.
    ///
    /// The minimum level is read once (the getter takes and releases the configuration
    /// lock). The message closure and all formatting run after that, with no logger
    /// lock held, so a message may read/update configuration or log recursively.
    private func log(level: LogLevel,
                     _ message: () -> String,
                     file: String,
                     function: String,
                     line: Int) {
        let minimum = activeConfiguration.minimumLevel
        guard level >= minimum else { return }

        let text = message()
        let fileName = URL(fileURLWithPath: file).lastPathComponent
        let location = "\(fileName):\(line)"

        if let recordSink {
            recordSink(LogRecord(level: level, message: text, location: location,
                                 function: function, category: category))
            return
        }

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
               text)

        // Also print to console in debug builds (reuses the same evaluated text)
        #if DEBUG
        print("\(level.symbol) [\(category)] \(text) (\(location))")
        #endif
    }
}

// MARK: - Global Loggers

/// Logger for core vector operations.
///
/// Use for logging vector creation, manipulation, and basic operations.
public let coreLogger = Logger(category: "Core")

/// Logger for storage and memory operations.
///
/// Use for logging memory allocation, deallocation, and storage management.
public let storageLogger = Logger(category: "Storage")

/// Logger for batch operations.
///
/// Use for logging batch processing, parallel operations, and bulk computations.
public let batchLogger = Logger(category: "Batch")

/// Logger for distance metric calculations.
///
/// Use for logging distance computations, similarity measurements, and metric selection.
public let metricsLogger = Logger(category: "Metrics")

/// Logger for performance monitoring.
///
/// Use for logging timing information, benchmarks, and performance analysis.
public let performanceLogger = Logger(category: "Performance")

// MARK: - Performance Logging

/// Simple performance timer for logging.
///
/// Elapsed time is measured with `ContinuousClock` from a single start `Instant`
/// captured at initialization, so repeated `log` calls report cumulative elapsed time
/// (not lap intervals) and are independent of civil-clock changes. No deadline or exact
/// callback timing is promised.
public struct PerformanceTimer {
    private let start: ContinuousClock.Instant
    private let now: @Sendable () -> ContinuousClock.Instant
    private let operation: String
    private let logger: Logger

    /// Start timing an operation
    public init(operation: String, logger: Logger = performanceLogger) {
        self.init(operation: operation, logger: logger, clock: { ContinuousClock.now })
    }

    /// Test seam: supplies the clock reading. Reads once for the start, then once per
    /// `log` call. Production uses `ContinuousClock.now`.
    init(operation: String,
         logger: Logger,
         clock: @escaping @Sendable () -> ContinuousClock.Instant) {
        self.now = clock
        self.start = clock()
        self.operation = operation
        self.logger = logger
    }

    /// Log the cumulative elapsed time since initialization if it is at least `threshold` seconds.
    ///
    /// A negative or non-finite `threshold` suppresses the call (this method does not
    /// throw), as does a negative elapsed reading. A finite zero threshold is valid.
    /// Elapsed seconds are computed in `Double` from the Duration components
    /// (seconds + attoseconds / 1e18), so no integer or Duration conversion of
    /// `threshold` can trap.
    ///
    /// A rejected call (invalid threshold, or debug gated) does not format the message:
    /// the threshold guard precedes formatting, the formatting lives inside the
    /// `logger.debug` autoclosure, and debug laziness is proven by
    /// `LoggingContractTests.messageEvaluationMatrix`.
    public func log(threshold: TimeInterval = 0.001) {
        let reading = now()
        guard threshold.isFinite, threshold >= 0 else { return }
        let components = start.duration(to: reading).components
        let elapsed = Double(components.seconds) + Double(components.attoseconds) / 1e18
        // A negative reading (only possible from a deliberately invalid test clock)
        // fails this comparison because threshold >= 0.
        guard elapsed >= threshold else { return }
        logger.debug("\(operation) took \(String(format: "%.3f", elapsed * 1000))ms")
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
