//
//  SIMDProvider.swift
//  VectorCore
//
//  Platform-agnostic SIMD operations protocol
//  Enables cross-platform support with Accelerate on Apple platforms
//  and Swift SIMD fallbacks on other platforms
//

import Foundation

/// Protocol defining platform-agnostic SIMD operations for vector computations.
///
/// This protocol abstracts away platform-specific implementations, allowing
/// VectorCore to use hardware-accelerated operations where available (e.g.,
/// Accelerate framework on Apple platforms) while providing portable fallbacks
/// for other platforms.
///
/// ## Design Principles
/// - All methods are static for optimal performance and inlining
/// - Methods use unsafe pointers to avoid copying overhead
/// - Stride parameter is omitted (always 1) for simplicity
/// - Float and Double operations are handled via generic Scalar type
///
/// ## Implementation Notes
/// - AccelerateSIMDProvider: Uses vDSP functions on Apple platforms
/// - SwiftSIMDProvider: Uses Swift's SIMD types for cross-platform support
public protocol SIMDProvider {
    /// The floating-point type this provider operates on
    associatedtype Scalar: BinaryFloatingPoint & SIMDScalar
    
    // MARK: - Arithmetic Operations
    
    /// Adds two vectors element-wise: result[i] = a[i] + b[i]
    @inlinable
    static func add(
        _ a: UnsafePointer<Scalar>,
        _ b: UnsafePointer<Scalar>,
        result: UnsafeMutablePointer<Scalar>,
        count: Int
    )
    
    /// Subtracts two vectors element-wise: result[i] = a[i] - b[i]
    @inlinable
    static func subtract(
        _ a: UnsafePointer<Scalar>,
        _ b: UnsafePointer<Scalar>,
        result: UnsafeMutablePointer<Scalar>,
        count: Int
    )
    
    /// Multiplies two vectors element-wise: result[i] = a[i] * b[i]
    @inlinable
    static func multiply(
        _ a: UnsafePointer<Scalar>,
        _ b: UnsafePointer<Scalar>,
        result: UnsafeMutablePointer<Scalar>,
        count: Int
    )
    
    /// Divides two vectors element-wise: result[i] = a[i] / b[i]
    @inlinable
    static func divide(
        _ a: UnsafePointer<Scalar>,
        _ b: UnsafePointer<Scalar>,
        result: UnsafeMutablePointer<Scalar>,
        count: Int
    )
    
    /// Negates a vector: result[i] = -a[i]
    @inlinable
    static func negate(
        _ a: UnsafePointer<Scalar>,
        result: UnsafeMutablePointer<Scalar>,
        count: Int
    )
    
    // MARK: - Scalar-Vector Operations
    
    /// Adds a scalar to each element: result[i] = a[i] + scalar
    @inlinable
    static func addScalar(
        _ a: UnsafePointer<Scalar>,
        scalar: Scalar,
        result: UnsafeMutablePointer<Scalar>,
        count: Int
    )
    
    /// Multiplies each element by a scalar: result[i] = a[i] * scalar
    @inlinable
    static func multiplyScalar(
        _ a: UnsafePointer<Scalar>,
        scalar: Scalar,
        result: UnsafeMutablePointer<Scalar>,
        count: Int
    )
    
    /// Divides each element by a scalar: result[i] = a[i] / scalar
    @inlinable
    static func divideByScalar(
        _ a: UnsafePointer<Scalar>,
        scalar: Scalar,
        result: UnsafeMutablePointer<Scalar>,
        count: Int
    )
    
    // MARK: - Reduction Operations
    
    /// Computes dot product of two vectors: sum(a[i] * b[i])
    @inlinable
    static func dot(
        _ a: UnsafePointer<Scalar>,
        _ b: UnsafePointer<Scalar>,
        count: Int
    ) -> Scalar
    
    /// Sums all elements in the vector
    @inlinable
    static func sum(
        _ a: UnsafePointer<Scalar>,
        count: Int
    ) -> Scalar
    
    /// Sums the magnitudes (absolute values) of all elements
    @inlinable
    static func sumOfMagnitudes(
        _ a: UnsafePointer<Scalar>,
        count: Int
    ) -> Scalar
    
    /// Sums the squares of all elements
    @inlinable
    static func sumOfSquares(
        _ a: UnsafePointer<Scalar>,
        count: Int
    ) -> Scalar
    
    // MARK: - Statistical Operations
    
    /// Finds the maximum value in the vector
    @inlinable
    static func maximum(
        _ a: UnsafePointer<Scalar>,
        count: Int
    ) -> Scalar
    
    /// Finds the minimum value in the vector
    @inlinable
    static func minimum(
        _ a: UnsafePointer<Scalar>,
        count: Int
    ) -> Scalar
    
    /// Finds the maximum magnitude (absolute value) in the vector
    @inlinable
    static func maximumMagnitude(
        _ a: UnsafePointer<Scalar>,
        count: Int
    ) -> Scalar
    
    // MARK: - Distance Operations
    
    /// Computes squared Euclidean distance between two vectors
    @inlinable
    static func distanceSquared(
        _ a: UnsafePointer<Scalar>,
        _ b: UnsafePointer<Scalar>,
        count: Int
    ) -> Scalar
    
    // MARK: - Utility Operations
    
    /// Copies vector data from source to destination
    @inlinable
    static func copy(
        source: UnsafePointer<Scalar>,
        destination: UnsafeMutablePointer<Scalar>,
        count: Int
    )
    
    /// Fills a vector with a constant value
    @inlinable
    static func fill(
        value: Scalar,
        destination: UnsafeMutablePointer<Scalar>,
        count: Int
    )
    
    /// Clips vector elements to a range: result[i] = min(max(a[i], low), high)
    @inlinable
    static func clip(
        _ a: UnsafePointer<Scalar>,
        low: Scalar,
        high: Scalar,
        result: UnsafeMutablePointer<Scalar>,
        count: Int
    )
}

// MARK: - Type Aliases

/// Type alias for vector length, matching vDSP_Length on Apple platforms
public typealias SIMDLength = Int

// MARK: - Default Implementations

public extension SIMDProvider {
    /// Default implementation of divideByScalar using multiplyScalar
    @inlinable
    static func divideByScalar(
        _ a: UnsafePointer<Scalar>,
        scalar: Scalar,
        result: UnsafeMutablePointer<Scalar>,
        count: Int
    ) {
        let invScalar = 1.0 / scalar
        multiplyScalar(a, scalar: invScalar, result: result, count: count)
    }
}

// MARK: - Provider Selection

/// Namespace for SIMD operations
public enum SIMDOperations {
    /// The active SIMD provider for Float operations
    public typealias FloatProvider = CurrentFloatProvider
    
    /// The active SIMD provider for Double operations
    public typealias DoubleProvider = CurrentDoubleProvider
}

// Platform-specific provider selection
#if canImport(Accelerate)
/// Uses Accelerate framework on Apple platforms
public typealias CurrentFloatProvider = AccelerateFloatProvider
public typealias CurrentDoubleProvider = AccelerateDoubleProvider
#else
/// Uses Swift SIMD on other platforms
public typealias CurrentFloatProvider = SwiftFloatSIMDProvider
public typealias CurrentDoubleProvider = SwiftDoubleSIMDProvider
#endif