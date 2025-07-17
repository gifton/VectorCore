// VectorCore: Medium Vector Storage (65-512 dimensions)
//
// Unified storage for medium vectors using AlignedValueStorage
//
// This type consolidates SIMDStorage128, SIMDStorage256, and SIMDStorage512 into
// a single storage that efficiently handles vectors from 65 to 512 elements using
// a fixed-size AlignedValueStorage buffer with partial usage tracking.
//

import Foundation
import Accelerate

/// Optimized storage for vectors with dimensions 65-512.
///
/// `MediumVectorStorage` consolidates SIMDStorage128, SIMDStorage256, and SIMDStorage512
/// into a single type that uses AlignedValueStorage for efficiency. It maintains a
/// fixed 512-element buffer internally but tracks the actual number of valid elements.
///
/// ## Design Rationale
/// - Single fixed-size buffer reduces allocation overhead
/// - 512 elements fits well within L1/L2 cache
/// - Padding strategy simplifies memory management
/// - 64-byte alignment for optimal cache performance
///
/// ## Performance Characteristics
/// - Heap allocated with 64-byte alignment
/// - COW semantics from AlignedValueStorage
/// - Cache-friendly size (2KB for 512 floats)
/// - Accelerate-optimized operations
///
/// ## Implementation Details
/// The storage always allocates 512 elements but only uses actualCount elements
/// in operations. This trades some memory for better performance and simpler code.
///
/// ## Usage Example
/// ```swift
/// // Create a 256-element vector
/// let vec = MediumVectorStorage(count: 256, repeating: 1.0)
/// 
/// // Internal buffer is 512 elements, but operations use only 256
/// let dot = vec.dotProduct(vec)  // Processes 256 elements
/// ```
public struct MediumVectorStorage: VectorStorage, VectorStorageOperations {
    /// The underlying aligned storage.
    /// Always contains 512 elements for consistency, regardless of actualCount.
    @usableFromInline
    internal var storage: AlignedValueStorage
    
    /// The actual number of valid elements (65-512).
    /// Operations only process this many elements from the buffer.
    @usableFromInline
    internal let actualCount: Int
    
    /// Number of valid elements in the storage.
    public var count: Int { actualCount }
    
    /// Initialize with default capacity (512 elements).
    /// 
    /// Creates a zero-initialized vector using the full buffer capacity.
    /// NOTE: When used with specific dimension types (e.g., Dim128), 
    /// the Vector type will use init(count:) instead to match the dimension.
    /// 
    /// - Complexity: O(1)
    @inlinable
    public init() {
        self.actualCount = 512
        self.storage = AlignedValueStorage(count: 512)
    }
    
    /// Initialize with specific element count.
    /// 
    /// Creates storage with the specified logical size. The internal buffer
    /// is always 512 elements, but only actualCount elements are valid.
    /// 
    /// - Parameter count: Number of valid elements (65-512)
    /// - Precondition: `count` must be between 65 and 512
    /// 
    /// - Complexity: O(1)
    @inlinable
    public init(count: Int) {
        precondition(count >= 65 && count <= 512, 
                    "Count must be between 65 and 512. Use SmallVectorStorage for ≤64 or LargeVectorStorage for >512")
        self.actualCount = count
        // Always allocate full 512 for cache efficiency and consistency
        self.storage = AlignedValueStorage(count: 512)
    }
    
    /// Initialize uninitialized storage for operations that will overwrite all elements.
    /// 
    /// - Parameters:
    ///   - count: Number of valid elements (65-512)
    ///   - uninitialized: Must be true to create uninitialized storage
    /// - Warning: All actualCount elements must be written before any read
    @usableFromInline
    internal init(count: Int, uninitialized: Bool) {
        precondition(count >= 65 && count <= 512, 
                    "Count must be between 65 and 512")
        precondition(uninitialized, "Use init(count:) for zero-initialized storage")
        self.actualCount = count
        self.storage = AlignedValueStorage(count: 512, uninitialized: true)
    }
    
    /// Initialize with repeating value using full capacity.
    /// 
    /// Creates a 512-element vector filled with the specified value.
    /// 
    /// - Parameter value: Value to fill all elements with
    /// 
    /// - Complexity: O(n) where n is 512
    @inlinable
    public init(repeating value: Float) {
        self.actualCount = 512
        self.storage = AlignedValueStorage(count: 512, repeating: value)
    }
    
    /// Initialize with repeating value and specific count.
    /// 
    /// Creates storage with the specified count. The entire 512-element buffer
    /// is filled with the value for consistency, though only actualCount elements
    /// are considered valid.
    /// 
    /// - Parameters:
    ///   - count: Number of valid elements (65-512)
    ///   - value: Value to fill elements with
    /// - Precondition: `count` must be between 65 and 512
    /// 
    /// - Complexity: O(n) where n is 512
    @inlinable
    public init(count: Int, repeating value: Float) {
        precondition(count >= 65 && count <= 512,
                    "Count must be between 65 and 512")
        self.actualCount = count
        // Fill entire buffer for consistency
        self.storage = AlignedValueStorage(count: 512, repeating: value)
    }
    
    /// Initialize from an array of values.
    /// 
    /// Creates storage containing the array values, padding with zeros to fill
    /// the 512-element buffer. This ensures consistent memory layout.
    /// 
    /// - Parameter values: Array of values to store (65-512 elements)
    /// - Precondition: Array must have 65-512 elements
    /// 
    /// - Complexity: O(n) where n is 512
    @inlinable
    public init(from values: [Float]) {
        precondition(values.count >= 65 && values.count <= 512,
                    "Values count must be between 65 and 512")
        self.actualCount = values.count
        
        // Create storage with zero padding to 512 elements
        var paddedValues = values
        paddedValues.append(contentsOf: Array(repeating: 0, count: 512 - values.count))
        self.storage = AlignedValueStorage(from: paddedValues)
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
            return storage[index]
        }
        set {
            precondition(index >= 0 && index < actualCount, "Index out of bounds")
            storage[index] = newValue
        }
    }
    
    /// Access valid elements as an unsafe buffer pointer.
    /// 
    /// The returned buffer only includes actualCount elements, hiding the
    /// padding from callers to prevent accidental access to invalid data.
    /// 
    /// - Parameter body: Closure receiving the buffer
    /// - Returns: Closure result
    /// - Throws: Any error thrown by closure
    /// 
    /// - Complexity: O(1)
    @inlinable
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeBufferPointer { buffer in
            // Return only the valid portion, hiding padding
            let validBuffer = UnsafeBufferPointer(
                start: buffer.baseAddress,
                count: actualCount
            )
            return try body(validBuffer)
        }
    }
    
    /// Access valid elements as a mutable buffer pointer.
    /// 
    /// The returned buffer only includes actualCount elements. Modifications
    /// to padding areas are prevented by limiting the buffer size.
    /// 
    /// - Parameter body: Closure receiving the mutable buffer
    /// - Returns: Closure result
    /// - Throws: Any error thrown by closure
    /// 
    /// - Complexity: O(1) or O(n) if COW is triggered
    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeMutableBufferPointer { buffer in
            // Return only the valid portion
            let validBuffer = UnsafeMutableBufferPointer(
                start: buffer.baseAddress,
                count: actualCount
            )
            return try body(validBuffer)
        }
    }
    
    // MARK: - VectorStorageOperations Conformance
    
    /// Compute dot product with another MediumVectorStorage.
    /// 
    /// Uses Accelerate's vDSP_dotpr for optimal performance. Only processes
    /// actualCount elements, ignoring any padding.
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

// MARK: - Optimizations for Common Sizes

extension MediumVectorStorage {
    /// Create storage for exactly 128 dimensions.
    /// 
    /// Optimized factory for the common 128-dimension case.
    /// 
    /// - Parameter values: Array with exactly 128 elements
    /// - Returns: Storage containing the values
    /// - Precondition: Array must have exactly 128 elements
    @inlinable
    public static func vector128(from values: [Float]) -> Self {
        precondition(values.count == 128, "Must have exactly 128 values")
        return Self(from: values)
    }
    
    /// Create storage for exactly 256 dimensions.
    /// 
    /// Optimized factory for the common 256-dimension case.
    /// 
    /// - Parameter values: Array with exactly 256 elements
    /// - Returns: Storage containing the values
    /// - Precondition: Array must have exactly 256 elements
    @inlinable
    public static func vector256(from values: [Float]) -> Self {
        precondition(values.count == 256, "Must have exactly 256 values")
        return Self(from: values)
    }
    
    /// Create storage for exactly 512 dimensions.
    /// 
    /// Optimized factory for the full capacity case. No padding needed
    /// since the values exactly fill the internal buffer.
    /// 
    /// - Parameter values: Array with exactly 512 elements
    /// - Returns: Storage containing the values
    /// - Precondition: Array must have exactly 512 elements
    @inlinable
    public static func vector512(from values: [Float]) -> Self {
        precondition(values.count == 512, "Must have exactly 512 values")
        // No padding needed for 512 - direct initialization
        let storage = AlignedValueStorage(from: values)
        var result = Self()
        result.storage = storage
        return result
    }
}

// MARK: - Additional Operations

extension MediumVectorStorage {
    /// Create a zero vector with specific dimension.
    /// 
    /// Factory method for creating zero-initialized vectors.
    /// 
    /// - Parameter count: Number of elements (65-512)
    /// - Returns: Zero-initialized storage
    @inlinable
    public static func zeros(count: Int) -> Self {
        Self(count: count)
    }
    
    /// Create a vector of ones with specific dimension.
    /// 
    /// Factory method for creating vectors filled with 1.0.
    /// 
    /// - Parameter count: Number of elements (65-512)
    /// - Returns: Storage filled with ones
    @inlinable
    public static func ones(count: Int) -> Self {
        Self(count: count, repeating: 1.0)
    }
    
    /// Add two vectors element-wise.
    /// 
    /// Uses Accelerate's vDSP_vadd for SIMD performance.
    /// Result has the same actualCount as the operands.
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
        
        var result = Self(count: lhs.actualCount, uninitialized: true)
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
    
    /// Verify memory alignment for testing.
    /// 
    /// Helper method to confirm the storage maintains proper alignment.
    /// Used primarily in tests to validate our alignment guarantees.
    /// 
    /// - Returns: Alignment in bytes (1, 16, or 64)
    @inlinable
    public func verifyAlignment() -> Int {
        storage.withUnsafeBufferPointer { buffer in
            let address = Int(bitPattern: buffer.baseAddress!)
            return address % 64 == 0 ? 64 : address % 16 == 0 ? 16 : 1
        }
    }
}