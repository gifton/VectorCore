// VectorCore: Error System
//
// Comprehensive error handling with context, chaining, and telemetry support
//

import Foundation

// MARK: - Error Context

/// Rich error context with source location and debugging information.
///
/// `ErrorContext` captures comprehensive debugging information at the
/// point where an error occurs, including source location, timestamp,
/// and additional contextual data. In release builds, source location
/// is omitted for performance and binary size optimization.
///
/// ## Example Usage
/// ```swift
/// let context = ErrorContext(
///     additionalInfo: [
///         "dimension": "128",
///         "operation": "normalization"
///     ]
/// )
/// ```
public struct ErrorContext: Sendable {
    /// Source file where the error occurred (debug builds only).
    public let file: StaticString

    /// Line number where the error occurred (debug builds only).
    public let line: UInt

    /// Function name where the error occurred (debug builds only).
    public let function: StaticString

    /// Timestamp when the error was created.
    public let timestamp: Date

    /// Additional contextual information as key-value pairs.
    ///
    /// Common keys include:
    /// - "message": Human-readable error description
    /// - "dimension": Vector dimension involved
    /// - "index": Array index that caused the error
    public let additionalInfo: [String: String]

    #if DEBUG
    public init(
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function,
        additionalInfo: [String: String] = [:]
    ) {
        self.file = file
        self.line = line
        self.function = function
        self.timestamp = Date()
        self.additionalInfo = additionalInfo
    }
    #else
    @inlinable
    public init(
        file: StaticString = "",
        line: UInt = 0,
        function: StaticString = "",
        additionalInfo: [String: String] = [:]
    ) {
        self.file = ""
        self.line = 0
        self.function = ""
        self.timestamp = Date()
        self.additionalInfo = [:]
    }
    #endif
}

// MARK: - Rich Error Type

/// Error type with enhanced context, chaining, and telemetry support.
///
/// `VectorError` provides a sophisticated error handling system with:
/// - Categorized error types for analytics and handling
/// - Rich context including source location and timestamps
/// - Error chaining for root cause analysis
/// - Telemetry-friendly severity and categorization
///
/// ## Error Handling Pattern
/// ```swift
/// do {
///     try riskyOperation()
/// } catch let error as VectorError {
///     logger.log(error, severity: error.kind.severity)
///     if error.kind == .dimensionMismatch {
///         // Handle specific error type
///     }
/// }
/// ```
///
/// ## Error Chaining
/// ```swift
/// let rootError = VectorError(.invalidData, message: "Corrupted header")
/// let chainedError = VectorError(.operationFailed, message: "Load failed")
///     .chain(with: rootError)
/// ```
public struct VectorError: Error, Sendable {
    /// The specific type of error that occurred.
    public let kind: ErrorKind

    /// Contextual information about where and when the error occurred.
    public let context: ErrorContext

    /// Optional underlying system or library error.
    public let underlyingError: (any Error)?

    /// Chain of errors leading to this error (for root cause analysis).
    public var errorChain: [VectorError]

    /// Error categories for telemetry and systematic handling.
    ///
    /// Each error kind has associated severity and category metadata
    /// for proper handling, logging, and monitoring.
    public enum ErrorKind: String, CaseIterable, Sendable {
        // Dimension errors
        case dimensionMismatch
        case invalidDimension
        case unsupportedDimension

        // Index errors
        case indexOutOfBounds
        case invalidRange

        // Data errors
        case invalidData
        case dataCorruption
        case insufficientData

        // Operation errors
        case invalidOperation
        case unsupportedOperation
        case operationFailed

        // Resource errors
        case allocationFailed
        case resourceExhausted
        case resourceUnavailable

        // Configuration errors
        case invalidConfiguration
        case missingConfiguration

        // System errors
        case systemError
        case unknown
    }

    /// Create error with automatic context capture
    @inlinable
    public init(
        _ kind: ErrorKind,
        message: String? = nil,
        underlying: (any Error)? = nil,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) {
        self.kind = kind
        self.underlyingError = underlying
        self.errorChain = []

        var info = [String: String]()
        if let message = message {
            info["message"] = message
        }

        self.context = ErrorContext(
            file: file,
            line: line,
            function: function,
            additionalInfo: info
        )
    }

    /// Create error with pre-built context (used by ErrorBuilder)
    @inlinable
    internal init(
        kind: ErrorKind,
        context: ErrorContext,
        underlying: (any Error)? = nil
    ) {
        self.kind = kind
        self.context = context
        self.underlyingError = underlying
        self.errorChain = []
    }

    /// Chain errors for root cause analysis
    public func chain(with error: VectorError) -> VectorError {
        var newError = self
        newError.errorChain.append(error)
        return newError
    }
}

// MARK: - Error Builder

/// Fluent builder for constructing detailed error instances.
///
/// `ErrorBuilder` provides a convenient API for creating `VectorError`
/// instances with rich context and proper error chaining. It uses the
/// builder pattern for readable error construction.
///
/// ## Example Usage
/// ```swift
/// throw ErrorBuilder(.dimensionMismatch)
///     .message("Cannot add vectors of different sizes")
///     .dimension(expected: 128, actual: 256)
///     .build()
/// ```
///
/// ## Complex Example
/// ```swift
/// let error = ErrorBuilder(.operationFailed)
///     .message("Matrix multiplication failed")
///     .parameter("matrixA_dims", value: "128x256")
///     .parameter("matrixB_dims", value: "512x128")
///     .underlying(systemError)
///     .chain(previousError)
///     .build()
/// ```
public struct ErrorBuilder {
    private var kind: VectorError.ErrorKind
    private var context: ErrorContext
    private var underlying: (any Error)?
    private var chain: [VectorError] = []
    private var info: [String: String] = [:]

    public init(
        _ kind: VectorError.ErrorKind,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) {
        self.kind = kind
        self.context = ErrorContext(
            file: file,
            line: line,
            function: function
        )
    }

    @discardableResult
    public func message(_ message: String) -> ErrorBuilder {
        var builder = self
        builder.info["message"] = message
        return builder
    }

    @discardableResult
    public func dimension(expected: Int, actual: Int) -> ErrorBuilder {
        var builder = self
        builder.info["expected_dimension"] = String(expected)
        builder.info["actual_dimension"] = String(actual)
        return builder
    }

    @discardableResult
    public func index(_ index: Int, max: Int) -> ErrorBuilder {
        var builder = self
        builder.info["index"] = String(index)
        builder.info["max_index"] = String(max)
        return builder
    }

    @discardableResult
    public func parameter(_ name: String, value: String) -> ErrorBuilder {
        var builder = self
        builder.info[name] = value
        return builder
    }

    @discardableResult
    public func underlying(_ error: any Error) -> ErrorBuilder {
        var builder = self
        builder.underlying = error
        return builder
    }

    @discardableResult
    public func chain(_ errors: VectorError...) -> ErrorBuilder {
        var builder = self
        builder.chain.append(contentsOf: errors)
        return builder
    }

    public func build() -> VectorError {
        // Create error context with all accumulated info
        let errorContext = ErrorContext(
            file: context.file,
            line: context.line,
            function: context.function,
            additionalInfo: info
        )

        var error = VectorError(
            kind: kind,
            context: errorContext,
            underlying: underlying
        )
        error.errorChain = chain
        return error
    }
}

// MARK: - Convenience Factory Methods

public extension VectorError {
    /// Dimension mismatch error with context
    static func dimensionMismatch(
        expected: Int,
        actual: Int,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.dimensionMismatch, file: file, line: line, function: function)
            .message("Expected dimension \(expected), but got \(actual)")
            .dimension(expected: expected, actual: actual)
            .build()
    }

    /// Index out of bounds error
    static func indexOutOfBounds(
        index: Int,
        dimension: Int,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.indexOutOfBounds, file: file, line: line, function: function)
            .message("Index \(index) is out of bounds for dimension \(dimension)")
            .index(index, max: dimension - 1)
            .build()
    }

    /// Invalid operation error
    static func invalidOperation(
        _ operation: String,
        reason: String,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.invalidOperation, file: file, line: line, function: function)
            .message("\(operation) failed: \(reason)")
            .build()
    }

    /// Invalid data error
    static func invalidData(
        _ description: String,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.invalidData, file: file, line: line, function: function)
            .message(description)
            .build()
    }

    /// Allocation failed error
    static func allocationFailed(
        size: Int,
        reason: String? = nil,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        let message = reason ?? "Failed to allocate \(size) bytes"
        return ErrorBuilder(.allocationFailed, file: file, line: line, function: function)
            .message(message)
            .parameter("requested_size", value: String(size))
            .build()
    }

    /// Invalid dimension error
    static func invalidDimension(
        _ dimension: Int,
        reason: String,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.invalidDimension, file: file, line: line, function: function)
            .message("Invalid dimension \(dimension): \(reason)")
            .parameter("dimension", value: String(dimension))
            .parameter("reason", value: reason)
            .build()
    }

    /// Invalid values error
    static func invalidValues(
        indices: [Int],
        reason: String,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.invalidData, file: file, line: line, function: function)
            .message("Invalid values at indices \(indices): \(reason)")
            .parameter("indices", value: indices.map(String.init).joined(separator: ","))
            .parameter("reason", value: reason)
            .build()
    }

    /// Division by zero error
    static func divisionByZero(
        operation: String,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        .invalidOperation(operation, reason: "Division by zero", file: file, line: line, function: function)
    }

    /// Zero vector error
    static func zeroVectorError(
        operation: String,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        .invalidOperation(operation, reason: "Cannot perform operation on zero vector", file: file, line: line, function: function)
    }

    /// Insufficient data error
    static func insufficientData(
        expected: Int,
        actual: Int,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.insufficientData, file: file, line: line, function: function)
            .message("Insufficient data: expected \(expected) bytes, got \(actual)")
            .parameter("expected_bytes", value: String(expected))
            .parameter("actual_bytes", value: String(actual))
            .build()
    }

    /// Invalid data format error
    static func invalidDataFormat(
        expected: String,
        actual: String,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.invalidData, file: file, line: line, function: function)
            .message("Invalid data format: expected \(expected), got \(actual)")
            .parameter("expected_format", value: expected)
            .parameter("actual_format", value: actual)
            .build()
    }

    /// Data corruption error
    static func dataCorruption(
        reason: String,
        file: StaticString = #fileID,
        line: UInt = #line,
        function: StaticString = #function
    ) -> VectorError {
        ErrorBuilder(.dataCorruption, file: file, line: line, function: function)
            .message("Data corruption detected: \(reason)")
            .parameter("reason", value: reason)
            .build()
    }
}

// MARK: - Error Description

extension VectorError: CustomStringConvertible, LocalizedError {
    public var description: String {
        var result = "VectorError.\(kind.rawValue)"

        if let message = context.additionalInfo["message"] {
            result += ": \(message)"
        }

        #if DEBUG
        result += " [at \(context.file):\(context.line) in \(context.function)]"
        #endif

        if !errorChain.isEmpty {
            result += "\nError chain:"
            for (index, error) in errorChain.enumerated() {
                result += "\n  \(index + 1). \(error.kind.rawValue)"
                if let msg = error.context.additionalInfo["message"] {
                    result += ": \(msg)"
                }
            }
        }

        if let underlying = underlyingError {
            result += "\nUnderlying: \(underlying)"
        }

        return result
    }

    public var errorDescription: String? { description }

    public var debugDescription: String {
        var result = description

        result += "\nContext:"
        result += "\n  Timestamp: \(context.timestamp)"

        for (key, value) in context.additionalInfo.sorted(by: { $0.key < $1.key }) where key != "message" {
            result += "\n  \(key): \(value)"
        }

        return result
    }
}

// MARK: - Error Categorization

public extension VectorError.ErrorKind {
    /// Severity level for prioritization
    var severity: ErrorSeverity {
        switch self {
        case .dataCorruption, .systemError:
            return .critical
        case .allocationFailed, .resourceExhausted:
            return .high
        case .dimensionMismatch, .indexOutOfBounds, .invalidOperation:
            return .medium
        case .invalidConfiguration, .unsupportedOperation:
            return .low
        default:
            return .info
        }
    }

    /// Category for grouping
    var category: ErrorCategory {
        switch self {
        case .dimensionMismatch, .invalidDimension, .unsupportedDimension:
            return .dimension
        case .indexOutOfBounds, .invalidRange:
            return .bounds
        case .invalidData, .dataCorruption, .insufficientData:
            return .data
        case .invalidOperation, .unsupportedOperation, .operationFailed:
            return .operation
        case .allocationFailed, .resourceExhausted, .resourceUnavailable:
            return .resource
        case .invalidConfiguration, .missingConfiguration:
            return .configuration
        case .systemError, .unknown:
            return .system
        }
    }
}

public enum ErrorSeverity: String, CaseIterable, Sendable {
    case critical
    case high
    case medium
    case low
    case info
}

public enum ErrorCategory: String, CaseIterable, Sendable {
    case dimension
    case bounds
    case data
    case operation
    case resource
    case configuration
    case system
}

// MARK: - Result Extensions

public extension Result where Failure == VectorError {
    /// Map error with additional context
    func mapErrorContext(
        _ transform: (VectorError) -> VectorError
    ) -> Result<Success, VectorError> {
        mapError(transform)
    }

    /// Chain error if failure
    func chainError(
        _ error: VectorError
    ) -> Result<Success, VectorError> {
        mapError { $0.chain(with: error) }
    }
}
