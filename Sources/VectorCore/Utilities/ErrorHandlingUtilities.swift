// VectorCore: Error Handling Utilities
//
// Utilities to support consistent error handling patterns
//

import Foundation

// MARK: - Validation Utilities

/// Namespace for validation utilities
public enum Validation {
    
    /// Validates that two dimensions match
    /// - Parameters:
    ///   - expected: The expected dimension
    ///   - actual: The actual dimension
    /// - Throws: VectorError.dimensionMismatch if dimensions don't match
    @inlinable
    public static func requireDimensionMatch(expected: Int, actual: Int) throws {
        guard expected == actual else {
            throw VectorError.dimensionMismatch(expected: expected, actual: actual)
        }
    }
    
    /// Validates that a dimension is valid (positive)
    /// - Parameters:
    ///   - dimension: The dimension to validate
    /// - Throws: VectorError.invalidDimension if dimension is invalid
    @inlinable
    public static func requireValidDimension(_ dimension: Int) throws {
        guard dimension > 0 else {
            throw VectorError.invalidDimension(dimension, reason: "Dimension must be positive")
        }
    }
    
    /// Validates that an index is within bounds
    /// - Parameters:
    ///   - index: The index to validate
    ///   - dimension: The dimension (upper bound)
    /// - Throws: VectorError.indexOutOfBounds if index is out of bounds
    @inlinable
    public static func requireValidIndex(_ index: Int, dimension: Int) throws {
        guard index >= 0 && index < dimension else {
            throw VectorError.indexOutOfBounds(index: index, dimension: dimension)
        }
    }
    
    /// Validates that a vector is not zero
    /// - Parameters:
    ///   - magnitude: The magnitude of the vector
    ///   - operation: The operation being performed
    /// - Throws: VectorError.zeroVectorError if magnitude is zero
    @inlinable
    public static func requireNonZero(magnitude: Float, operation: String) throws {
        guard magnitude > 0 else {
            throw VectorError.zeroVectorError(operation: operation)
        }
    }
    
    /// Validates that values are within a range
    /// - Parameters:
    ///   - values: The values to check
    ///   - range: The valid range
    /// - Throws: VectorError.outOfRange if any values are outside the range
    public static func requireInRange(_ values: [Float], range: ClosedRange<Float>) throws {
        var invalidIndices: [Int] = []
        for (index, value) in values.enumerated() {
            if !range.contains(value) {
                invalidIndices.append(index)
            }
        }
        
        if !invalidIndices.isEmpty {
            throw VectorError.invalidValues(indices: invalidIndices, reason: "Values outside range \(range.lowerBound)...\(range.upperBound)")
        }
    }
}

// MARK: - Debug Validation

/// Debug-only validation utilities (compiled out in release builds)
public enum DebugValidation {
    
    /// Validates dimension match in debug builds only
    /// - Parameters:
    ///   - expected: The expected dimension
    ///   - actual: The actual dimension
    ///   - message: Optional custom message
    @inlinable
    public static func assertDimensionMatch(expected: Int, actual: Int, message: String? = nil) {
        #if DEBUG
        assert(expected == actual, message ?? "Dimension mismatch: expected \(expected), got \(actual)")
        #endif
    }
    
    /// Validates index bounds in debug builds only
    /// - Parameters:
    ///   - index: The index to validate
    ///   - dimension: The dimension (upper bound)
    ///   - message: Optional custom message
    @inlinable
    public static func assertValidIndex(_ index: Int, dimension: Int, message: String? = nil) {
        #if DEBUG
        assert(index >= 0 && index < dimension, 
               message ?? "Index \(index) out of bounds for dimension \(dimension)")
        #endif
    }
}

// MARK: - Result Extensions

public extension Result where Failure == VectorError {
    
    /// Creates a Result from a throwing closure
    /// - Parameter body: The throwing closure to execute
    /// - Returns: Success with the result or Failure with VectorError
    @inlinable
    static func catching(_ body: () throws -> Success) -> Result<Success, VectorError> {
        do {
            return .success(try body())
        } catch let error as VectorError {
            return .failure(error)
        } catch {
            // This should not happen if all errors are VectorError
            return .failure(.invalidData("Unknown error: \(error)"))
        }
    }
    
    /// Converts a Result to an optional value (nil on failure)
    @inlinable
    var optional: Success? {
        switch self {
        case .success(let value):
            return value
        case .failure:
            return nil
        }
    }
}

// MARK: - Performance Utilities

/// Utilities for performance-critical error handling
public enum PerformanceValidation {
    
    /// Performs validation only in debug builds, using precondition in release
    /// - Parameters:
    ///   - condition: The condition to validate
    ///   - message: The error message
    /// - Note: In release builds, this becomes a precondition for performance
    @inlinable
    public static func require(_ condition: Bool, _ message: @autoclosure () -> String) {
        #if DEBUG
        guard condition else {
            fatalError(message())
        }
        #else
        precondition(condition, message())
        #endif
    }
    
    /// Validates dimension match with performance optimization
    /// - Parameters:
    ///   - lhs: First dimension
    ///   - rhs: Second dimension
    /// - Note: Uses precondition in release builds for performance
    @inlinable
    public static func requireDimensionMatch(_ lhs: Int, _ rhs: Int) {
        #if DEBUG
        guard lhs == rhs else {
            fatalError("Dimension mismatch: \(lhs) != \(rhs)")
        }
        #else
        precondition(lhs == rhs, "Dimension mismatch")
        #endif
    }
}