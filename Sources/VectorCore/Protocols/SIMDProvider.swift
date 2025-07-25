// VectorCore: SIMD Provider Protocol
//
// Abstraction for SIMD operations without platform dependencies
//

import Foundation

/// Protocol for SIMD (Single Instruction, Multiple Data) operations
///
/// SIMDProvider abstracts vectorized operations, allowing platform-specific
/// implementations (Accelerate, Intel MKL, etc.) while keeping VectorCore
/// platform-agnostic. A pure Swift implementation ensures cross-platform support.
///
/// ## Design Principles
/// - Operations work on Float arrays (most common case)
/// - No platform-specific types exposed
/// - Performance-critical operations only
/// - Memory-safe interfaces
///
/// ## Performance Expectations
/// - Platform implementations: Native performance
/// - Pure Swift implementation: 80-90% of native
public protocol SIMDProvider: Sendable {
    
    // MARK: - Basic Arithmetic
    
    /// Compute dot product of two vectors
    ///
    /// - Parameters:
    ///   - a: First vector
    ///   - b: Second vector
    /// - Returns: Dot product (sum of element-wise products)
    /// - Precondition: a.count == b.count
    func dot(_ a: [Float], _ b: [Float]) -> Float
    
    /// Element-wise addition
    ///
    /// - Parameters:
    ///   - a: First vector
    ///   - b: Second vector
    /// - Returns: New vector with a[i] + b[i]
    /// - Precondition: a.count == b.count
    func add(_ a: [Float], _ b: [Float]) -> [Float]
    
    /// Element-wise subtraction
    ///
    /// - Parameters:
    ///   - a: First vector
    ///   - b: Second vector
    /// - Returns: New vector with a[i] - b[i]
    /// - Precondition: a.count == b.count
    func subtract(_ a: [Float], _ b: [Float]) -> [Float]
    
    /// Scalar multiplication
    ///
    /// - Parameters:
    ///   - a: Vector to multiply
    ///   - scalar: Scalar value
    /// - Returns: New vector with a[i] * scalar
    func multiply(_ a: [Float], by scalar: Float) -> [Float]
    
    // MARK: - Reduction Operations
    
    /// Sum all elements
    func sum(_ a: [Float]) -> Float
    
    /// Find maximum element
    func max(_ a: [Float]) -> Float
    
    /// Find minimum element
    func min(_ a: [Float]) -> Float
    
    /// Compute mean (average)
    func mean(_ a: [Float]) -> Float
    
    // MARK: - Vector Operations
    
    /// Compute magnitude (L2 norm)
    func magnitude(_ a: [Float]) -> Float
    
    /// Normalize vector to unit length
    ///
    /// - Parameter a: Vector to normalize
    /// - Returns: Normalized vector (magnitude = 1)
    /// - Note: Returns zero vector if input magnitude is zero
    func normalize(_ a: [Float]) -> [Float]
    
    /// Compute squared magnitude (avoids sqrt)
    func magnitudeSquared(_ a: [Float]) -> Float
    
    // MARK: - Distance Operations
    
    /// Euclidean distance between vectors
    func euclideanDistance(_ a: [Float], _ b: [Float]) -> Float
    
    /// Squared Euclidean distance (avoids sqrt)
    func euclideanDistanceSquared(_ a: [Float], _ b: [Float]) -> Float
    
    /// Cosine similarity between vectors
    func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float
    
    // MARK: - Additional Operations
    
    /// Divide vector by scalar
    func divide(_ a: [Float], by scalar: Float) -> [Float]
    
    /// Negate vector (multiply by -1)
    func negate(_ a: [Float]) -> [Float]
    
    /// Element-wise multiplication (Hadamard product)
    func elementWiseMultiply(_ a: [Float], _ b: [Float]) -> [Float]
    
    /// Element-wise division
    func elementWiseDivide(_ a: [Float], _ b: [Float]) -> [Float]
    
    /// Absolute values of vector elements
    func abs(_ a: [Float]) -> [Float]
    
    /// Element-wise minimum of two vectors
    func elementWiseMin(_ a: [Float], _ b: [Float]) -> [Float]
    
    /// Element-wise maximum of two vectors
    func elementWiseMax(_ a: [Float], _ b: [Float]) -> [Float]
    
    /// Find index of minimum element
    func minIndex(_ a: [Float]) -> Int
    
    /// Find index of maximum element
    func maxIndex(_ a: [Float]) -> Int
    
    /// Clamp values to a range
    func clip(_ a: [Float], min: Float, max: Float) -> [Float]
    
    /// Square root of each element
    func sqrt(_ a: [Float]) -> [Float]
}

// MARK: - Default Implementations

public extension SIMDProvider {
    /// Default magnitude using magnitudeSquared
    func magnitude(_ a: [Float]) -> Float {
        Foundation.sqrt(magnitudeSquared(a))
    }
    
    /// Default mean using sum
    func mean(_ a: [Float]) -> Float {
        guard !a.isEmpty else { return 0 }
        return sum(a) / Float(a.count)
    }
    
    /// Default euclideanDistance using euclideanDistanceSquared
    func euclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
        Foundation.sqrt(euclideanDistanceSquared(a, b))
    }
    
    /// Default normalize using magnitude
    func normalize(_ a: [Float]) -> [Float] {
        let mag = magnitude(a)
        guard mag > 0 else { return a }
        return multiply(a, by: 1.0 / mag)
    }
    
    /// Default divide using multiply
    func divide(_ a: [Float], by scalar: Float) -> [Float] {
        guard scalar != 0 else { return [Float](repeating: .infinity, count: a.count) }
        return multiply(a, by: 1.0 / scalar)
    }
    
    /// Default negate using multiply
    func negate(_ a: [Float]) -> [Float] {
        return multiply(a, by: -1.0)
    }
    
    /// Default minIndex implementation
    func minIndex(_ a: [Float]) -> Int {
        guard !a.isEmpty else { return 0 }
        var minIdx = 0
        var minVal = a[0]
        for i in 1..<a.count {
            if a[i] < minVal {
                minVal = a[i]
                minIdx = i
            }
        }
        return minIdx
    }
    
    /// Default maxIndex implementation
    func maxIndex(_ a: [Float]) -> Int {
        guard !a.isEmpty else { return 0 }
        var maxIdx = 0
        var maxVal = a[0]
        for i in 1..<a.count {
            if a[i] > maxVal {
                maxVal = a[i]
                maxIdx = i
            }
        }
        return maxIdx
    }
    
    /// Default clip implementation
    func clip(_ a: [Float], min: Float, max: Float) -> [Float] {
        return a.map { Swift.min(Swift.max($0, min), max) }
    }
    
    /// Default sqrt implementation
    func sqrt(_ a: [Float]) -> [Float] {
        return a.map { Foundation.sqrt($0) }
    }
}

/// Reduction operations for SIMDProvider
public enum ReductionOp: String, Sendable {
    case sum
    case max
    case min
    case mean
}