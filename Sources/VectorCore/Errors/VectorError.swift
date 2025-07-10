// VectorCore: Modern Error System
//
// Rich error handling with context, chaining, and telemetry support
//

import Foundation

// MARK: - Error Context

/// Rich error context with source location
public struct ErrorContext: Sendable {
    public let file: StaticString
    public let line: UInt
    public let function: StaticString
    public let timestamp: Date
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

// MARK: - Modern Error Type

/// Modern error type with rich context and chaining
public struct VectorError: Error, Sendable {
    public let kind: ErrorKind
    public let context: ErrorContext
    public let underlyingError: (any Error)?
    public var errorChain: [VectorError]
    
    /// Error categories for telemetry
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
    
    /// Chain errors for root cause analysis
    public func chain(with error: VectorError) -> VectorError {
        var newError = self
        newError.errorChain.append(error)
        return newError
    }
}

// MARK: - Error Builder

/// Fluent error builder
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
        var error = VectorError(
            kind,
            message: info["message"],
            underlying: underlying,
            file: context.file,
            line: context.line,
            function: context.function
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
        
        for (key, value) in context.additionalInfo.sorted(by: { $0.key < $1.key }) {
            if key != "message" {
                result += "\n  \(key): \(value)"
            }
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