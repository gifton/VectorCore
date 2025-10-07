//
//  SIMDOperations.swift
//  VectorCore
//
//  Convenience layer for SIMD operations
//  Provides unified access to platform-specific SIMD implementations
//

import Foundation

/// Global namespace for SIMD operations on Float vectors
internal enum FloatSIMD {
    // MARK: - Arithmetic Operations

    @inlinable
    public static func add(
        _ a: UnsafePointer<Float>,
        _ b: UnsafePointer<Float>,
        result: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        SIMDOperations.FloatProvider.add(a, b, result: result, count: count)
    }

    @inlinable
    public static func subtract(
        _ a: UnsafePointer<Float>,
        _ b: UnsafePointer<Float>,
        result: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        SIMDOperations.FloatProvider.subtract(a, b, result: result, count: count)
    }

    @inlinable
    public static func multiply(
        _ a: UnsafePointer<Float>,
        _ b: UnsafePointer<Float>,
        result: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        SIMDOperations.FloatProvider.multiply(a, b, result: result, count: count)
    }

    @inlinable
    public static func divide(
        _ a: UnsafePointer<Float>,
        _ b: UnsafePointer<Float>,
        result: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        SIMDOperations.FloatProvider.divide(a, b, result: result, count: count)
    }

    @inlinable
    public static func negate(
        _ a: UnsafePointer<Float>,
        result: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        SIMDOperations.FloatProvider.negate(a, result: result, count: count)
    }

    // MARK: - Scalar Operations

    @inlinable
    public static func addScalar(
        _ a: UnsafePointer<Float>,
        scalar: Float,
        result: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        SIMDOperations.FloatProvider.addScalar(a, scalar: scalar, result: result, count: count)
    }

    @inlinable
    public static func multiplyScalar(
        _ a: UnsafePointer<Float>,
        scalar: Float,
        result: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        SIMDOperations.FloatProvider.multiplyScalar(a, scalar: scalar, result: result, count: count)
    }

    @inlinable
    public static func divideByScalar(
        _ a: UnsafePointer<Float>,
        scalar: Float,
        result: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        SIMDOperations.FloatProvider.divideByScalar(a, scalar: scalar, result: result, count: count)
    }

    // MARK: - Reduction Operations

    @inlinable
    public static func dot(
        _ a: UnsafePointer<Float>,
        _ b: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        SIMDOperations.FloatProvider.dot(a, b, count: count)
    }

    @inlinable
    public static func sum(
        _ a: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        SIMDOperations.FloatProvider.sum(a, count: count)
    }

    @inlinable
    public static func sumOfMagnitudes(
        _ a: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        SIMDOperations.FloatProvider.sumOfMagnitudes(a, count: count)
    }

    @inlinable
    public static func sumOfSquares(
        _ a: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        SIMDOperations.FloatProvider.sumOfSquares(a, count: count)
    }

    // MARK: - Statistical Operations

    @inlinable
    public static func maximum(
        _ a: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        SIMDOperations.FloatProvider.maximum(a, count: count)
    }

    @inlinable
    public static func minimum(
        _ a: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        SIMDOperations.FloatProvider.minimum(a, count: count)
    }

    @inlinable
    public static func maximumMagnitude(
        _ a: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        SIMDOperations.FloatProvider.maximumMagnitude(a, count: count)
    }

    // MARK: - Distance Operations

    @inlinable
    public static func distanceSquared(
        _ a: UnsafePointer<Float>,
        _ b: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        SIMDOperations.FloatProvider.distanceSquared(a, b, count: count)
    }

    // MARK: - Utility Operations

    @inlinable
    public static func copy(
        source: UnsafePointer<Float>,
        destination: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        SIMDOperations.FloatProvider.copy(source: source, destination: destination, count: count)
    }

    @inlinable
    public static func fill(
        value: Float,
        destination: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        SIMDOperations.FloatProvider.fill(value: value, destination: destination, count: count)
    }

    @inlinable
    public static func clip(
        _ a: UnsafePointer<Float>,
        low: Float,
        high: Float,
        result: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        SIMDOperations.FloatProvider.clip(a, low: low, high: high, result: result, count: count)
    }
}

/// Global namespace for SIMD operations on Double vectors
internal enum DoubleSIMD {
    // MARK: - Arithmetic Operations

    @inlinable
    public static func add(
        _ a: UnsafePointer<Double>,
        _ b: UnsafePointer<Double>,
        result: UnsafeMutablePointer<Double>,
        count: Int
    ) {
        SIMDOperations.DoubleProvider.add(a, b, result: result, count: count)
    }

    @inlinable
    public static func subtract(
        _ a: UnsafePointer<Double>,
        _ b: UnsafePointer<Double>,
        result: UnsafeMutablePointer<Double>,
        count: Int
    ) {
        SIMDOperations.DoubleProvider.subtract(a, b, result: result, count: count)
    }

    @inlinable
    public static func multiply(
        _ a: UnsafePointer<Double>,
        _ b: UnsafePointer<Double>,
        result: UnsafeMutablePointer<Double>,
        count: Int
    ) {
        SIMDOperations.DoubleProvider.multiply(a, b, result: result, count: count)
    }

    @inlinable
    public static func divide(
        _ a: UnsafePointer<Double>,
        _ b: UnsafePointer<Double>,
        result: UnsafeMutablePointer<Double>,
        count: Int
    ) {
        SIMDOperations.DoubleProvider.divide(a, b, result: result, count: count)
    }

    @inlinable
    public static func negate(
        _ a: UnsafePointer<Double>,
        result: UnsafeMutablePointer<Double>,
        count: Int
    ) {
        SIMDOperations.DoubleProvider.negate(a, result: result, count: count)
    }

    // MARK: - Scalar Operations

    @inlinable
    public static func addScalar(
        _ a: UnsafePointer<Double>,
        scalar: Double,
        result: UnsafeMutablePointer<Double>,
        count: Int
    ) {
        SIMDOperations.DoubleProvider.addScalar(a, scalar: scalar, result: result, count: count)
    }

    @inlinable
    public static func multiplyScalar(
        _ a: UnsafePointer<Double>,
        scalar: Double,
        result: UnsafeMutablePointer<Double>,
        count: Int
    ) {
        SIMDOperations.DoubleProvider.multiplyScalar(a, scalar: scalar, result: result, count: count)
    }

    @inlinable
    public static func divideByScalar(
        _ a: UnsafePointer<Double>,
        scalar: Double,
        result: UnsafeMutablePointer<Double>,
        count: Int
    ) {
        SIMDOperations.DoubleProvider.divideByScalar(a, scalar: scalar, result: result, count: count)
    }

    // MARK: - Reduction Operations

    @inlinable
    public static func dot(
        _ a: UnsafePointer<Double>,
        _ b: UnsafePointer<Double>,
        count: Int
    ) -> Double {
        SIMDOperations.DoubleProvider.dot(a, b, count: count)
    }

    @inlinable
    public static func sum(
        _ a: UnsafePointer<Double>,
        count: Int
    ) -> Double {
        SIMDOperations.DoubleProvider.sum(a, count: count)
    }

    @inlinable
    public static func sumOfMagnitudes(
        _ a: UnsafePointer<Double>,
        count: Int
    ) -> Double {
        SIMDOperations.DoubleProvider.sumOfMagnitudes(a, count: count)
    }

    @inlinable
    public static func sumOfSquares(
        _ a: UnsafePointer<Double>,
        count: Int
    ) -> Double {
        SIMDOperations.DoubleProvider.sumOfSquares(a, count: count)
    }

    // MARK: - Statistical Operations

    @inlinable
    public static func maximum(
        _ a: UnsafePointer<Double>,
        count: Int
    ) -> Double {
        SIMDOperations.DoubleProvider.maximum(a, count: count)
    }

    @inlinable
    public static func minimum(
        _ a: UnsafePointer<Double>,
        count: Int
    ) -> Double {
        SIMDOperations.DoubleProvider.minimum(a, count: count)
    }

    @inlinable
    public static func maximumMagnitude(
        _ a: UnsafePointer<Double>,
        count: Int
    ) -> Double {
        SIMDOperations.DoubleProvider.maximumMagnitude(a, count: count)
    }

    // MARK: - Distance Operations

    @inlinable
    public static func distanceSquared(
        _ a: UnsafePointer<Double>,
        _ b: UnsafePointer<Double>,
        count: Int
    ) -> Double {
        SIMDOperations.DoubleProvider.distanceSquared(a, b, count: count)
    }

    // MARK: - Utility Operations

    @inlinable
    public static func copy(
        source: UnsafePointer<Double>,
        destination: UnsafeMutablePointer<Double>,
        count: Int
    ) {
        SIMDOperations.DoubleProvider.copy(source: source, destination: destination, count: count)
    }

    @inlinable
    public static func fill(
        value: Double,
        destination: UnsafeMutablePointer<Double>,
        count: Int
    ) {
        SIMDOperations.DoubleProvider.fill(value: value, destination: destination, count: count)
    }

    @inlinable
    public static func clip(
        _ a: UnsafePointer<Double>,
        low: Double,
        high: Double,
        result: UnsafeMutablePointer<Double>,
        count: Int
    ) {
        SIMDOperations.DoubleProvider.clip(a, low: low, high: high, result: result, count: count)
    }
}

// MARK: - Provider Type Namespace
// Note: SIMDOperations namespace is defined in SIMDProvider.swift
