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
    
}

// Intentionally no generic operator overloads here; operators are provided by VectorFactory and specialized vector types.

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
    var magnitude: Scalar {
        Foundation.sqrt(dotProduct(self))
    }
    
    /// Squared magnitude (more efficient when square root not needed)
    var magnitudeSquared: Scalar {
        dotProduct(self)
    }
    
    /// Normalized (unit) vector
    /// - Throws: `VectorError.invalidOperation` if the vector has zero magnitude
    func normalizedThrowing() throws -> Self {
        let mag = magnitude
        guard mag > 0 else {
            throw VectorError.invalidOperation("normalize", reason: "Cannot normalize zero vector")
        }
        let inv = 1 / mag
        // Allocate and scale elements to avoid relying on generic operators
        let scaled = withUnsafeBufferPointer { buffer -> [Scalar] in
            var out = Array(repeating: Scalar(0), count: buffer.count)
            for i in 0..<buffer.count { out[i] = buffer[i] * inv }
            return out
        }
        return try! Self(scaled)
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
        precondition(scalarCount == other.scalarCount, "Dimension mismatch")
        var sum: Scalar = 0
        withUnsafeBufferPointer { a in
            other.withUnsafeBufferPointer { b in
                for i in 0..<scalarCount {
                    let d = a[i] - b[i]
                    sum += d * d
                }
            }
        }
        return Foundation.sqrt(sum)
    }
    
    /// Squared Euclidean distance (more efficient when square root not needed)
    func euclideanDistanceSquared(to other: Self) -> Scalar {
        precondition(scalarCount == other.scalarCount, "Dimension mismatch")
        var sum: Scalar = 0
        withUnsafeBufferPointer { a in
            other.withUnsafeBufferPointer { b in
                for i in 0..<scalarCount {
                    let d = a[i] - b[i]
                    sum += d * d
                }
            }
        }
        return sum
    }
    
    /// Manhattan distance to another vector
    func manhattanDistance(to other: Self) -> Scalar {
        precondition(scalarCount == other.scalarCount, "Dimension mismatch")
        var sum: Scalar = 0
        withUnsafeBufferPointer { a in
            other.withUnsafeBufferPointer { b in
                for i in 0..<scalarCount {
                    sum += Swift.abs(a[i] - b[i])
                }
            }
        }
        return sum
    }
    
    /// Chebyshev distance to another vector
    func chebyshevDistance(to other: Self) -> Scalar {
        precondition(scalarCount == other.scalarCount, "Dimension mismatch")
        var maxVal: Scalar = 0
        withUnsafeBufferPointer { a in
            other.withUnsafeBufferPointer { b in
                for i in 0..<scalarCount {
                    let d = Swift.abs(a[i] - b[i])
                    if d > maxVal { maxVal = d }
                }
            }
        }
        return maxVal
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
