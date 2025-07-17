// VectorCore: Small Vector Storage (1-64 dimensions)
//
// Unified storage for small vectors using SIMD64 with actualCount
//
// This type consolidates SIMDStorage32 and SIMDStorage64 into a single, flexible
// storage that efficiently handles vectors from 1 to 64 elements using a single
// SIMD64 register with partial usage tracking.
//

import Foundation
import simd
import Accelerate

/// Optimized storage for vectors with dimensions 1-64.
/// 
/// `SmallVectorStorage` consolidates the functionality of SIMDStorage32 and SIMDStorage64
/// into a single, more flexible type that can handle any size from 1-64 elements.
/// It uses a SIMD64<Float> register for storage with an actualCount field to track
/// how many elements are actually used.
///
/// ## Design Rationale
/// - Single SIMD64 register can efficiently handle all small vectors
/// - Reduces code duplication between 32 and 64 element variants
/// - Maintains SIMD performance for all operations
/// - Value semantics with stack allocation
///
/// ## Performance Characteristics
/// - Stack allocated (no heap allocation)
/// - SIMD-accelerated operations via Accelerate
/// - 16-byte aligned for optimal SIMD performance
/// - O(1) element access
///
/// ## Usage Example
/// ```swift
/// // Create a 20-element vector
/// let vec = SmallVectorStorage(count: 20, repeating: 1.0)
/// 
/// // Only the first 20 elements are used in operations
/// let dot = vec.dotProduct(vec)  // Uses only 20 elements
/// ```
public struct SmallVectorStorage: VectorStorage, VectorStorageOperations {
    /// The SIMD64 register holding the vector data.
    /// May contain unused elements when actualCount < 64.
    @usableFromInline
    internal var data: SIMD64<Float>
    
    /// The actual number of valid elements (1-64).
    /// Operations only process this many elements, ignoring the rest.
    @usableFromInline
    internal let actualCount: Int
    
    /// Number of valid elements in the storage.
    public var count: Int { actualCount }
    
    /// Initialize with default capacity (64 elements).
    /// 
    /// Creates a zero-initialized vector using the full SIMD64 capacity.
    /// 
    /// - Complexity: O(1)
    @inlinable
    public init() {
        self.actualCount = 64
        self.data = .zero
    }
    
    /// Initialize with specific element count.
    /// 
    /// Creates a zero-initialized vector with the specified number of elements.
    /// Unused SIMD lanes remain zero for safety.
    /// 
    /// - Parameter count: Number of elements (1-64)
    /// - Precondition: `count` must be between 1 and 64
    /// 
    /// - Complexity: O(1)
    @inlinable
    public init(count: Int) {
        precondition(count > 0 && count <= 64, "Count must be between 1 and 64")
        self.actualCount = count
        self.data = .zero
    }
    
    /// Initialize with repeating value using full capacity.
    /// 
    /// Creates a 64-element vector filled with the specified value.
    /// 
    /// - Parameter value: Value to fill all elements with
    /// 
    /// - Complexity: O(1)
    @inlinable
    public init(repeating value: Float) {
        self.actualCount = 64
        self.data = SIMD64(repeating: value)
    }
    
    /// Initialize with repeating value and specific count.
    /// 
    /// Creates a vector with the specified count, filled with the given value.
    /// Unused SIMD lanes are also filled for consistency.
    /// 
    /// - Parameters:
    ///   - count: Number of valid elements (1-64)
    ///   - value: Value to fill elements with
    /// - Precondition: `count` must be between 1 and 64
    /// 
    /// - Complexity: O(1)
    @inlinable
    public init(count: Int, repeating value: Float) {
        precondition(count > 0 && count <= 64, "Count must be between 1 and 64")
        self.actualCount = count
        self.data = SIMD64(repeating: value)
    }
    
    /// Initialize from an array of values.
    /// 
    /// Creates a vector containing the array values. The array size determines
    /// the actualCount. Unused SIMD lanes remain zero.
    /// 
    /// - Parameter values: Array of values to store (1-64 elements)
    /// - Precondition: Array must have 1-64 elements
    /// 
    /// - Complexity: O(n) where n is values.count
    @inlinable
    public init(from values: [Float]) {
        precondition(!values.isEmpty && values.count <= 64, 
                    "Values count must be between 1 and 64")
        self.actualCount = values.count
        self.data = .zero
        
        // Efficient copy using pointer operations
        values.withUnsafeBufferPointer { src in
            withUnsafeMutablePointer(to: &data) { dst in
                let dstPtr = UnsafeMutableRawPointer(dst).assumingMemoryBound(to: Float.self)
                dstPtr.initialize(from: src.baseAddress!, count: values.count)
            }
        }
    }
    
    // MARK: - VectorStorage Conformance
    
    /// Access elements by index with bounds checking.
    /// 
    /// Only elements within actualCount are valid for access.
    /// 
    /// - Parameter index: Zero-based index
    /// - Returns: Element value at index
    /// - Precondition: `index` must be in range `0..<actualCount`
    /// 
    /// - Complexity: O(1)
    @inlinable
    public subscript(index: Int) -> Float {
        get {
            precondition(index >= 0 && index < actualCount, "Index out of bounds")
            return data[index]
        }
        set {
            precondition(index >= 0 && index < actualCount, "Index out of bounds")
            data[index] = newValue
        }
    }
    
    /// Access valid elements as an unsafe buffer pointer.
    /// 
    /// The buffer only includes actualCount elements, not the full SIMD64.
    /// This ensures operations don't process garbage data in unused lanes.
    /// 
    /// - Parameter body: Closure receiving the buffer
    /// - Returns: Closure result
    /// - Throws: Any error thrown by closure
    /// 
    /// - Complexity: O(1)
    @inlinable
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try withUnsafePointer(to: data) { ptr in
            let floatPtr = UnsafeRawPointer(ptr).assumingMemoryBound(to: Float.self)
            // Only expose actualCount elements
            let buffer = UnsafeBufferPointer(start: floatPtr, count: actualCount)
            return try body(buffer)
        }
    }
    
    /// Access valid elements as a mutable buffer pointer.
    /// 
    /// The buffer only includes actualCount elements for safety.
    /// Modifications are restricted to valid elements only.
    /// 
    /// - Parameter body: Closure receiving the mutable buffer
    /// - Returns: Closure result
    /// - Throws: Any error thrown by closure
    /// 
    /// - Complexity: O(1)
    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try withUnsafeMutablePointer(to: &data) { ptr in
            let floatPtr = UnsafeMutableRawPointer(ptr).assumingMemoryBound(to: Float.self)
            // Only expose actualCount elements
            let buffer = UnsafeMutableBufferPointer(start: floatPtr, count: actualCount)
            return try body(buffer)
        }
    }
    
    // MARK: - VectorStorageOperations Conformance
    
    /// Compute dot product with another SmallVectorStorage.
    /// 
    /// Uses Accelerate's vDSP_dotpr for optimal performance. Only processes
    /// actualCount elements, ignoring any unused SIMD lanes.
    /// 
    /// - Parameter other: Another storage of the same size
    /// - Returns: Dot product result
    /// - Precondition: Both storages must have same actualCount
    /// 
    /// - Complexity: O(n) where n is actualCount
    @inlinable
    public func dotProduct(_ other: Self) -> Float {
        precondition(actualCount == other.actualCount, "Dimensions must match")
        
        var result: Float = 0
        self.withUnsafeBufferPointer { selfBuffer in
            other.withUnsafeBufferPointer { otherBuffer in
                vDSP_dotpr(selfBuffer.baseAddress!, 1,
                          otherBuffer.baseAddress!, 1,
                          &result, vDSP_Length(actualCount))
            }
        }
        return result
    }
}

// MARK: - Additional Operations

extension SmallVectorStorage {
    /// Create a zero vector with specific dimension.
    /// 
    /// Factory method for creating zero-initialized vectors.
    /// 
    /// - Parameter count: Number of elements (1-64)
    /// - Returns: Zero-initialized storage
    @inlinable
    public static func zeros(count: Int) -> Self {
        Self(count: count)
    }
    
    /// Create a vector of ones with specific dimension.
    /// 
    /// Factory method for creating vectors filled with 1.0.
    /// 
    /// - Parameter count: Number of elements (1-64)
    /// - Returns: Storage filled with ones
    @inlinable
    public static func ones(count: Int) -> Self {
        Self(count: count, repeating: 1.0)
    }
    
    /// Add two vectors element-wise.
    /// 
    /// Uses Accelerate's vDSP_vadd for SIMD performance.
    /// 
    /// - Parameters:
    ///   - lhs: First vector
    ///   - rhs: Second vector
    /// - Returns: Sum of the vectors
    /// - Precondition: Both vectors must have same actualCount
    /// 
    /// - Complexity: O(n) where n is actualCount
    @inlinable
    public static func + (lhs: Self, rhs: Self) -> Self {
        precondition(lhs.actualCount == rhs.actualCount, "Dimensions must match")
        
        var result = Self(count: lhs.actualCount)
        result.withUnsafeMutableBufferPointer { dst in
            lhs.withUnsafeBufferPointer { src1 in
                rhs.withUnsafeBufferPointer { src2 in
                    vDSP_vadd(src1.baseAddress!, 1,
                             src2.baseAddress!, 1,
                             dst.baseAddress!, 1,
                             vDSP_Length(lhs.actualCount))
                }
            }
        }
        return result
    }
    
    /// Subtract two vectors element-wise.
    /// 
    /// Uses Accelerate's vDSP_vsub for SIMD performance.
    /// Note: vDSP_vsub computes src2 - src1, so arguments are swapped.
    /// 
    /// - Parameters:
    ///   - lhs: First vector
    ///   - rhs: Second vector to subtract
    /// - Returns: Difference of the vectors
    /// - Precondition: Both vectors must have same actualCount
    /// 
    /// - Complexity: O(n) where n is actualCount
    @inlinable
    public static func - (lhs: Self, rhs: Self) -> Self {
        precondition(lhs.actualCount == rhs.actualCount, "Dimensions must match")
        
        var result = Self(count: lhs.actualCount)
        result.withUnsafeMutableBufferPointer { dst in
            lhs.withUnsafeBufferPointer { src1 in
                rhs.withUnsafeBufferPointer { src2 in
                    vDSP_vsub(src2.baseAddress!, 1,  // Note: order is swapped
                             src1.baseAddress!, 1,
                             dst.baseAddress!, 1,
                             vDSP_Length(lhs.actualCount))
                }
            }
        }
        return result
    }
    
    /// Scale vector by scalar.
    /// 
    /// Multiplies all elements by a scalar value using vDSP_vsmul.
    /// 
    /// - Parameters:
    ///   - lhs: Vector to scale
    ///   - rhs: Scalar multiplier
    /// - Returns: Scaled vector
    /// 
    /// - Complexity: O(n) where n is actualCount
    @inlinable
    public static func * (lhs: Self, rhs: Float) -> Self {
        var result = lhs  // Copy with value semantics
        result.withUnsafeMutableBufferPointer { buffer in
            var scalar = rhs
            vDSP_vsmul(buffer.baseAddress!, 1,
                      &scalar,
                      buffer.baseAddress!, 1,
                      vDSP_Length(lhs.actualCount))
        }
        return result
    }
}