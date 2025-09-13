//
//  VectorFactory.swift
//  VectorCore
//
//  Protocol for creating vectors from arrays, enabling generic transformations
//

import Foundation

// MARK: - Vector Factory Protocol

/// Protocol for types that can create vectors from arrays
/// This enables generic operations like map and reduce to work with both
/// static (Vector<D>) and dynamic (DynamicVector) vector types
public protocol VectorFactory: VectorProtocol {
    /// Create a new vector from an array of scalars
    /// - Parameter scalars: Array of scalar values
    /// - Returns: New vector instance
    /// - Throws: VectorError if dimensions don't match (for static vectors)
    static func create(from scalars: [Scalar]) throws -> Self
}

// MARK: - DynamicVector Conformance

extension DynamicVector: VectorFactory {
    public static func create(from scalars: [Scalar]) throws -> Self {
        return DynamicVector(scalars)
    }
}

// MARK: - Vector<D> Conformance

extension Vector: VectorFactory {
    public static func create(from scalars: [Scalar]) throws -> Self {
        guard scalars.count == D.value else {
            throw VectorError.dimensionMismatch(
                expected: D.value,
                actual: scalars.count
            )
        }
        return try Vector<D>(scalars)
    }
}

// MARK: - Factory Utilities

public extension VectorFactory {
    // MARK: Operator Implementations using Factory (handles dynamic dimensions safely)

    static func + (lhs: Self, rhs: Self) -> Self {
        try! createByCombining(lhs, rhs, +)
    }

    static func - (lhs: Self, rhs: Self) -> Self {
        try! createByCombining(lhs, rhs, -)
    }

    static func .* (lhs: Self, rhs: Self) -> Self {
        try! createByCombining(lhs, rhs, *)
    }

    static func ./ (lhs: Self, rhs: Self) -> Self {
        try! createByCombining(lhs, rhs) { a, b in
            precondition(b != 0, "Division by zero in element-wise division")
            return a / b
        }
    }

    static func * (lhs: Self, rhs: Scalar) -> Self {
        try! createByTransforming(lhs) { $0 * rhs }
    }

    static func / (lhs: Self, rhs: Scalar) -> Self {
        precondition(rhs != 0, "Division by zero")
        return lhs * (1 / rhs)
    }

    static func * (lhs: Scalar, rhs: Self) -> Self { rhs * lhs }

    static func += (lhs: inout Self, rhs: Self) { lhs = lhs + rhs }
    static func -= (lhs: inout Self, rhs: Self) { lhs = lhs - rhs }
    static func *= (lhs: inout Self, rhs: Scalar) { lhs = lhs * rhs }
    static func /= (lhs: inout Self, rhs: Scalar) { lhs = lhs / rhs }
    /// Create a vector by transforming another vector's elements
    /// - Parameters:
    ///   - vector: Source vector to transform
    ///   - transform: Function to apply to each element
    /// - Returns: New vector with transformed elements
    @inlinable
    static func createByTransforming<V: VectorProtocol>(
        _ vector: V,
        _ transform: (V.Scalar) -> Scalar
    ) throws -> Self where V.Scalar == Scalar {
        var result = Array(repeating: Scalar(0), count: vector.scalarCount)
        vector.withUnsafeBufferPointer { buffer in
            for i in 0..<buffer.count {
                result[i] = transform(buffer[i])
            }
        }
        return try create(from: result)
    }

    /// Create a vector by combining two vectors element-wise
    /// - Parameters:
    ///   - v1: First vector
    ///   - v2: Second vector
    ///   - combine: Function to combine elements
    /// - Returns: New vector with combined elements
    @inlinable
    static func createByCombining<V1: VectorProtocol, V2: VectorProtocol>(
        _ v1: V1,
        _ v2: V2,
        _ combine: (V1.Scalar, V2.Scalar) -> Scalar
    ) throws -> Self where V1.Scalar == Scalar, V2.Scalar == Scalar {
        guard v1.scalarCount == v2.scalarCount else {
            throw VectorError.dimensionMismatch(
                expected: v1.scalarCount,
                actual: v2.scalarCount
            )
        }

        var result = Array(repeating: Scalar(0), count: v1.scalarCount)
        v1.withUnsafeBufferPointer { buffer1 in
            v2.withUnsafeBufferPointer { buffer2 in
                for i in 0..<buffer1.count {
                    result[i] = combine(buffer1[i], buffer2[i])
                }
            }
        }
        return try create(from: result)
    }
}
