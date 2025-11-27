//
//  VectorProtocol.swift
//  VectorCore
//
//  Core protocol for all vector types
//

import Foundation

// MARK: - Vector Protocol

/// The core protocol for all vector types in VectorCore.
///
/// ## Design Philosophy
/// - Minimal required implementations for conformance
/// - Rich functionality through protocol extensions
/// - Type-safe compile-time dimension checking where possible
/// - Efficient storage access patterns for SIMD optimization
///
/// ## Conforming Types
/// - `Vector<D>`: Fixed-dimension vectors with compile-time safety
/// - `DynamicVector`: Runtime-determined dimension vectors
///
public protocol VectorProtocol: Sendable, Hashable, Codable, Collection
where Element == Scalar, Index == Int {

    // MARK: - Associated Types

    /// The scalar type for vector elements
    associatedtype Scalar: BinaryFloatingPoint & Hashable & Codable

    /// The storage type for vector data
    associatedtype Storage: Sendable

    // MARK: - Required Properties

    /// The underlying storage
    var storage: Storage { get set }

    /// Number of elements in the vector
    var scalarCount: Int { get }

    // MARK: - Required Initializers

    /// Initialize with zeros
    init()

    /// Initialize from an array
    /// - Throws: VectorError.dimensionMismatch if array size doesn't match requirements
    init(_ array: [Scalar]) throws

    /// Initialize with repeating value
    init(repeating value: Scalar)

    // MARK: - Required Methods

    /// Convert to array representation
    func toArray() -> [Scalar]

    /// Access storage for reading (enables SIMD optimizations)
    func withUnsafeBufferPointer<R>(
        _ body: (UnsafeBufferPointer<Scalar>) throws -> R
    ) rethrows -> R

    /// Access storage for writing (enables SIMD optimizations)
    mutating func withUnsafeMutableBufferPointer<R>(
        _ body: (UnsafeMutableBufferPointer<Scalar>) throws -> R
    ) rethrows -> R

    // MARK: - Optional Requirements with Defaults

    /// Check if all values are finite (no NaN or infinity)
    var isFinite: Bool { get }

    /// Check if this is the zero vector
    var isZero: Bool { get }
}

// MARK: - Default Implementations

public extension VectorProtocol {

    // MARK: Collection Conformance

    var startIndex: Int { 0 }
    var endIndex: Int { scalarCount }

    func index(after i: Int) -> Int {
        i + 1
    }

    // Default subscript using toArray() - conformers should override for efficiency
    subscript(index: Int) -> Scalar {
        precondition(index >= 0 && index < scalarCount, "Index \(index) out of bounds")
        return toArray()[index]
    }

    // MARK: Validation

    var isFinite: Bool {
        withUnsafeBufferPointer { buffer in
            for element in buffer {
                if !element.isFinite { return false }
            }
            return true
        }
    }

    var isZero: Bool {
        withUnsafeBufferPointer { buffer in
            for element in buffer {
                if element != 0 { return false }
            }
            return true
        }
    }

    // MARK: Factory Methods

    /// Create a zero vector
    static var zero: Self {
        Self()
    }

    /// Create a ones vector
    static var ones: Self {
        Self(repeating: 1)
    }

    /// Create a random vector
    static func random(in range: ClosedRange<Scalar> = 0...1) -> Self where Scalar.RawSignificand: FixedWidthInteger {
        let array = (0..<Self().scalarCount).map { _ in
            Scalar.random(in: range)
        }
        return try! Self(array)
    }
}

// MARK: - Arithmetic Operations

public extension VectorProtocol {

    // MARK: Vector-Vector Operations

    static func + (lhs: Self, rhs: Self) -> Self {
        precondition(lhs.scalarCount == rhs.scalarCount, "Dimension mismatch")
        // Allocate result with correct length
        var result = try! Self(Array(repeating: 0, count: lhs.scalarCount))
        result.withUnsafeMutableBufferPointer { resultPtr in
            lhs.withUnsafeBufferPointer { lhsPtr in
                rhs.withUnsafeBufferPointer { rhsPtr in
                    for i in 0..<lhs.scalarCount {
                        resultPtr[i] = lhsPtr[i] + rhsPtr[i]
                    }
                }
            }
        }
        return result
    }

    static func - (lhs: Self, rhs: Self) -> Self {
        precondition(lhs.scalarCount == rhs.scalarCount, "Dimension mismatch")
        // Allocate result with correct length
        var result = try! Self(Array(repeating: 0, count: lhs.scalarCount))
        result.withUnsafeMutableBufferPointer { resultPtr in
            lhs.withUnsafeBufferPointer { lhsPtr in
                rhs.withUnsafeBufferPointer { rhsPtr in
                    for i in 0..<lhs.scalarCount {
                        resultPtr[i] = lhsPtr[i] - rhsPtr[i]
                    }
                }
            }
        }
        return result
    }

    // MARK: - In-Place Operations

    /// Perform in-place addition with another vector
    @inlinable
    mutating func inPlaceAdd(_ other: Self) {
        let count = scalarCount
        withUnsafeMutableBufferPointer { dst in
            other.withUnsafeBufferPointer { src in
                for i in 0..<count {
                    dst[i] += src[i]
                }
            }
        }
    }

    /// Perform in-place subtraction with another vector
    @inlinable
    mutating func inPlaceSubtract(_ other: Self) {
        let count = scalarCount
        withUnsafeMutableBufferPointer { dst in
            other.withUnsafeBufferPointer { src in
                for i in 0..<count {
                    dst[i] -= src[i]
                }
            }
        }
    }

    /// Perform in-place scalar multiplication
    @inlinable
    mutating func inPlaceMultiply(_ scalar: Scalar) {
        let count = scalarCount
        withUnsafeMutableBufferPointer { dst in
            for i in 0..<count {
                dst[i] *= scalar
            }
        }
    }

    /// Perform in-place scalar division
    @inlinable
    mutating func inPlaceDivide(_ scalar: Scalar) {
        precondition(scalar != 0, "Division by zero")
        let count = scalarCount
        withUnsafeMutableBufferPointer { dst in
            for i in 0..<count {
                dst[i] /= scalar
            }
        }
    }

    /// Perform in-place element-wise multiplication (Hadamard product)
    @inlinable
    mutating func inPlaceElementwiseMultiply(_ other: Self) {
        let count = scalarCount
        withUnsafeMutableBufferPointer { dst in
            other.withUnsafeBufferPointer { src in
                for i in 0..<count {
                    dst[i] *= src[i]
                }
            }
        }
    }

    static func += (lhs: inout Self, rhs: Self) {
        lhs.inPlaceAdd(rhs)
    }

    static func -= (lhs: inout Self, rhs: Self) {
        lhs.inPlaceSubtract(rhs)
    }

    // MARK: Vector-Scalar Operations

    static func * (lhs: Self, rhs: Scalar) -> Self {
        // Allocate result with correct length
        var result = try! Self(Array(repeating: 0, count: lhs.scalarCount))
        result.withUnsafeMutableBufferPointer { resultPtr in
            lhs.withUnsafeBufferPointer { lhsPtr in
                for i in 0..<lhs.scalarCount {
                    resultPtr[i] = lhsPtr[i] * rhs
                }
            }
        }
        return result
    }

    static func / (lhs: Self, rhs: Scalar) -> Self {
        precondition(rhs != 0, "Division by zero")
        return lhs * (1 / rhs)
    }

    static func * (lhs: Scalar, rhs: Self) -> Self {
        rhs * lhs
    }

    static func *= (lhs: inout Self, rhs: Scalar) {
        lhs.inPlaceMultiply(rhs)
    }

    static func /= (lhs: inout Self, rhs: Scalar) {
        lhs.inPlaceDivide(rhs)
    }

    // MARK: Unary Operations

    prefix static func - (vector: Self) -> Self {
        vector * (-1)
    }

    // MARK: Element-wise Operations

    /// Hadamard (element-wise) product
    static func .* (lhs: Self, rhs: Self) -> Self {
        precondition(lhs.scalarCount == rhs.scalarCount, "Dimension mismatch")
        // Allocate result with correct length
        var result = try! Self(Array(repeating: 0, count: lhs.scalarCount))
        result.withUnsafeMutableBufferPointer { resultPtr in
            lhs.withUnsafeBufferPointer { lhsPtr in
                rhs.withUnsafeBufferPointer { rhsPtr in
                    for i in 0..<lhs.scalarCount {
                        resultPtr[i] = lhsPtr[i] * rhsPtr[i]
                    }
                }
            }
        }
        return result
    }

    /// Element-wise division
    static func ./ (lhs: Self, rhs: Self) -> Self {
        precondition(lhs.scalarCount == rhs.scalarCount, "Dimension mismatch")
        // Allocate result with correct length
        var result = try! Self(Array(repeating: 0, count: lhs.scalarCount))
        result.withUnsafeMutableBufferPointer { resultPtr in
            lhs.withUnsafeBufferPointer { lhsPtr in
                rhs.withUnsafeBufferPointer { rhsPtr in
                    for i in 0..<lhs.scalarCount {
                        precondition(rhsPtr[i] != 0, "Division by zero at index \(i)")
                        resultPtr[i] = lhsPtr[i] / rhsPtr[i]
                    }
                }
            }
        }
        return result
    }
}

// MARK: - Vector Mathematics

public extension VectorProtocol {

    /// Compute dot product with another vector
    func dotProduct(_ other: Self) -> Scalar {
        precondition(scalarCount == other.scalarCount, "Dimension mismatch")

        var result: Scalar = 0
        withUnsafeBufferPointer { selfPtr in
            other.withUnsafeBufferPointer { otherPtr in
                for i in 0..<scalarCount {
                    result += selfPtr[i] * otherPtr[i]
                }
            }
        }
        return result
    }

    /// Magnitude (L2 norm) of the vector
    ///
    /// Uses Kahan's two-pass scaling algorithm for numerical stability.
    /// Handles large values (> sqrt(Float.max)) without overflow.
    ///
    /// - Complexity: O(n) where n is the vector dimension
    /// - Note: Approximately 20-30% slower than naive implementation,
    ///         but prevents overflow and silent data corruption
    var magnitude: Scalar {
        // Kahan's two-pass algorithm for numerical stability
        // Phase 1: Find maximum absolute value and detect NaN early
        var maxAbs: Scalar = 0
        var hasNaN = false
        withUnsafeBufferPointer { buffer in
            for element in buffer {
                if element.isNaN { hasNaN = true; continue }
                maxAbs = Swift.max(maxAbs, Swift.abs(element))
            }
        }

        // Propagate NaN if any component is NaN
        if hasNaN { return Scalar.nan }

        // Handle edge cases
        if maxAbs.isNaN { return Scalar.nan }
        guard maxAbs > 0 else { return 0 }  // Zero vector
        guard maxAbs.isFinite else { return Scalar.infinity }  // Infinite components

        // Phase 2: Scale, compute, scale back
        // By scaling all values by 1/maxAbs, we ensure |scaled| ≤ 1
        // This prevents overflow since 1² = 1
        var sumSquares: Scalar = 0
        withUnsafeBufferPointer { buffer in
            for element in buffer {
                let scaled = element / maxAbs
                sumSquares += scaled * scaled
            }
        }

        // Result: maxAbs × sqrt(Σ((x/maxAbs)²))
        //       = sqrt(maxAbs² × Σ((x/maxAbs)²))
        //       = sqrt(Σ(x²))  [mathematically equivalent]
        return maxAbs * Foundation.sqrt(sumSquares)
    }

    /// Squared magnitude
    ///
    /// Uses the stable magnitude calculation and squares the result.
    /// This is numerically stable but requires computing the square root.
    ///
    /// - Note: For the stable algorithm, we must compute sqrt then square it.
    ///         The naive Σ(x²) approach would overflow for large values.
    var magnitudeSquared: Scalar {
        let mag = magnitude  // Uses stable Kahan algorithm
        return mag * mag     // Square the stable result
    }

    /// Normalized (unit) vector
    func normalized() -> Result<Self, VectorError> {
        let mag = magnitude
        guard mag > 0 else {
            return .failure(.invalidOperation("normalize", reason: "Cannot normalize zero vector"))
        }
        return .success(self / mag)
    }

    /// Normalized (unit) vector without error checking.
    ///
    /// This method bypasses the zero-vector check for maximum performance in hot paths
    /// where the caller guarantees the vector is non-zero.
    ///
    /// - Precondition: `magnitude > 0` (asserted in debug builds only)
    /// - Warning: Calling on a zero vector produces undefined results (NaN/Inf values)
    /// - Complexity: O(n) where n is scalarCount
    /// - Note: Use `normalized()` if the vector may be zero
    ///
    /// ## Example Usage
    /// ```swift
    /// // Only use when you KNOW the vector is non-zero
    /// let embedding = model.encode(text)  // Embeddings are typically non-zero
    /// let unit = embedding.normalizedUnchecked()
    /// ```
    @inlinable
    func normalizedUnchecked() -> Self {
        let mag = magnitude
        assert(mag > 0, "normalizedUnchecked() called on zero vector - use normalized() for safe handling")
        return self / mag
    }

    /// L1 norm (Manhattan norm)
    var l1Norm: Scalar {
        var sum: Scalar = 0
        withUnsafeBufferPointer { buffer in
            for element in buffer {
                sum += Swift.abs(element)
            }
        }
        return sum
    }

    /// L∞ norm (maximum norm)
    var lInfinityNorm: Scalar {
        var maxValue: Scalar = 0
        withUnsafeBufferPointer { buffer in
            for element in buffer {
                maxValue = Swift.max(maxValue, Swift.abs(element))
            }
        }
        return maxValue
    }
}

// MARK: - Distance Metrics

public extension VectorProtocol {

    /// Euclidean distance to another vector
    func euclideanDistance(to other: Self) -> Scalar {
        (self - other).magnitude
    }

    /// Squared Euclidean distance (more efficient when square root not needed)
    func euclideanDistanceSquared(to other: Self) -> Scalar {
        (self - other).magnitudeSquared
    }

    /// Manhattan distance to another vector
    func manhattanDistance(to other: Self) -> Scalar {
        (self - other).l1Norm
    }

    /// Chebyshev distance to another vector
    func chebyshevDistance(to other: Self) -> Scalar {
        (self - other).lInfinityNorm
    }

    /// Cosine similarity with another vector
    func cosineSimilarity(to other: Self) -> Scalar {
        let denominator = magnitude * other.magnitude
        guard denominator > 0 else { return 0 }
        return dotProduct(other) / denominator
    }

    /// Cosine distance (1 - cosine similarity)
    func cosineDistance(to other: Self) -> Scalar {
        1 - cosineSimilarity(to: other)
    }
}

// MARK: - Statistical Operations

public extension VectorProtocol {

    /// Sum of all elements
    var sum: Scalar {
        var total: Scalar = 0
        withUnsafeBufferPointer { buffer in
            for element in buffer {
                total += element
            }
        }
        return total
    }

    /// Mean of all elements
    var mean: Scalar {
        guard scalarCount > 0 else { return 0 }
        return sum / Scalar(scalarCount)
    }

    /// Minimum element
    var min: Scalar? {
        guard scalarCount > 0 else { return nil }
        var minValue = Scalar.infinity
        withUnsafeBufferPointer { buffer in
            for element in buffer {
                minValue = Swift.min(minValue, element)
            }
        }
        return minValue
    }

    /// Maximum element
    var max: Scalar? {
        guard scalarCount > 0 else { return nil }
        var maxValue = -Scalar.infinity
        withUnsafeBufferPointer { buffer in
            for element in buffer {
                maxValue = Swift.max(maxValue, element)
            }
        }
        return maxValue
    }
}

// MARK: - Convenience Extensions

public extension VectorProtocol {

    /// Apply a function to each element
    func mapElements(_ transform: (Scalar) throws -> Scalar) rethrows -> Self {
        let array = try toArray().map(transform)
        return try! Self(array)
    }

    /// Check approximate equality with tolerance
    func isApproximatelyEqual(to other: Self, tolerance: Scalar = 1e-6) -> Bool {
        guard scalarCount == other.scalarCount else { return false }

        return withUnsafeBufferPointer { selfPtr in
            other.withUnsafeBufferPointer { otherPtr in
                for i in 0..<scalarCount {
                    if Swift.abs(selfPtr[i] - otherPtr[i]) > tolerance {
                        return false
                    }
                }
                return true
            }
        }
    }
}
