// VectorCore: Large Vector Storage (513+ dimensions)
//
// Unified storage for large vectors using COWDynamicStorage
//
// This type consolidates SIMDStorage768, SIMDStorage1536, and SIMDStorage3072 into
// a single flexible storage that efficiently handles vectors with more than 512
// elements using dynamic allocation with Copy-on-Write semantics.
//

import Foundation
import Accelerate

/// Optimized storage for vectors with dimensions 513+.
///
/// `LargeVectorStorage` consolidates SIMDStorage768, SIMDStorage1536, and SIMDStorage3072
/// into a single type that uses COWDynamicStorage for ultimate flexibility. It handles
/// arbitrarily large vectors efficiently through dynamic allocation.
///
/// ## Design Rationale
/// - Dynamic allocation scales to any size
/// - COW semantics prevent unnecessary copies
/// - No fixed-size overhead for varying dimensions
/// - Suitable for embeddings and large ML models
///
/// ## Performance Characteristics
/// - Heap allocated with 16-byte alignment
/// - COW semantics from COWDynamicStorage
/// - Memory scales linearly with dimension
/// - Accelerate-optimized operations
/// - Chunked operations available for cache efficiency
///
/// ## Implementation Details
/// Unlike Small and Medium storage types which use fixed buffers, LargeVectorStorage
/// allocates exactly the memory needed for each dimension. This makes it memory-efficient
/// for varied sizes while maintaining performance through COW optimization.
///
/// ## Usage Example
/// ```swift
/// // Create a 3072-element vector (GPT-style embedding)
/// let vec = LargeVectorStorage(commonSize: .dim3072)
/// 
/// // Or with custom size
/// let customVec = LargeVectorStorage(count: 8192)
/// 
/// // Chunked operations for very large vectors
/// let dot = vec1.chunkedDotProduct(vec2, chunkSize: 2048)
/// ```
public struct LargeVectorStorage: VectorStorage, VectorStorageOperations {
    /// The underlying COW dynamic storage.
    /// Provides value semantics with efficient copying through COW.
    @usableFromInline
    internal var storage: COWDynamicStorage
    
    /// Number of elements in the storage.
    /// This can be any value greater than 512.
    public var count: Int { storage.count }
    
    /// Initialize storage with specific element count.
    /// 
    /// Creates storage with the exact dimension specified. The memory is
    /// dynamically allocated and zero-initialized.
    /// 
    /// - Parameter count: Number of elements (must be > 512)
    /// - Precondition: `count` must be greater than 512
    /// 
    /// - Complexity: O(n) where n is count
    @inlinable
    public init(count: Int) {
        precondition(count > 512,
                    "Count must be > 512. Use MediumVectorStorage for 65-512 or SmallVectorStorage for ≤64")
        self.storage = COWDynamicStorage(dimension: count)
    }
    
    /// Initialize with default size (1024 elements).
    /// 
    /// Creates a zero-initialized vector with a common default size.
    /// 1024 is a power of 2 and commonly used in ML applications.
    /// 
    /// - Complexity: O(n) where n is 1024
    @inlinable
    public init() {
        self.storage = COWDynamicStorage(dimension: 1024)
    }
    
    /// Initialize with repeating value using default size.
    /// 
    /// Creates a 1024-element vector filled with the specified value.
    /// 
    /// - Parameter value: Value to fill all elements with
    /// 
    /// - Complexity: O(n) where n is 1024
    @inlinable
    public init(repeating value: Float) {
        self.storage = COWDynamicStorage(dimension: 1024, repeating: value)
    }
    
    /// Initialize with repeating value and specific count.
    /// 
    /// Creates storage with the specified count, filled with the given value.
    /// Useful for creating constant vectors of any large dimension.
    /// 
    /// - Parameters:
    ///   - count: Number of elements (must be > 512)
    ///   - value: Value to fill elements with
    /// - Precondition: `count` must be greater than 512
    /// 
    /// - Complexity: O(n) where n is count
    @inlinable
    public init(count: Int, repeating value: Float) {
        precondition(count > 512, "Count must be > 512")
        self.storage = COWDynamicStorage(dimension: count, repeating: value)
    }
    
    /// Initialize from an array of values.
    /// 
    /// Creates storage containing a copy of the array values. The storage
    /// dimension is determined by the array size.
    /// 
    /// - Parameter values: Array of values to store (must have > 512 elements)
    /// - Precondition: Array must have more than 512 elements
    /// 
    /// - Complexity: O(n) where n is values.count
    @inlinable
    public init(from values: [Float]) {
        precondition(values.count > 512,
                    "Values count must be > 512")
        self.storage = COWDynamicStorage(from: values)
    }
    
    // MARK: - VectorStorage Conformance
    
    /// Access elements by index.
    /// 
    /// Delegates to the underlying COWDynamicStorage, which handles
    /// bounds checking and COW semantics automatically.
    /// 
    /// - Parameter index: Zero-based index
    /// - Returns: Element value at index
    /// 
    /// - Complexity: O(1) for getter, O(1) or O(n) for setter (if COW triggered)
    @inlinable
    public subscript(index: Int) -> Float {
        get {
            storage[index]
        }
        set {
            storage[index] = newValue
        }
    }
    
    /// Access elements as an unsafe buffer pointer for reading.
    /// 
    /// Delegates to the underlying storage. The pointer is valid only
    /// for the duration of the closure.
    /// 
    /// - Parameter body: Closure receiving the buffer
    /// - Returns: Closure result
    /// - Throws: Any error thrown by closure
    /// 
    /// - Complexity: O(1)
    @inlinable
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeBufferPointer(body)
    }
    
    /// Access elements as a mutable buffer pointer.
    /// 
    /// Triggers COW if needed through the underlying storage.
    /// The pointer is valid only for the duration of the closure.
    /// 
    /// - Parameter body: Closure receiving the mutable buffer
    /// - Returns: Closure result
    /// - Throws: Any error thrown by closure
    /// 
    /// - Complexity: O(1) if unique, O(n) if COW triggered
    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeMutableBufferPointer(body)
    }
    
    // MARK: - VectorStorageOperations Conformance
    
    /// Compute dot product with another LargeVectorStorage.
    /// 
    /// Uses the underlying storage's optimized dot product implementation.
    /// For very large vectors, consider using chunkedDotProduct for better
    /// cache locality.
    /// 
    /// - Parameter other: Another storage of the same size
    /// - Returns: Dot product result
    /// - Precondition: Both storages must have same count
    /// 
    /// - Complexity: O(n) where n is count
    @inlinable
    public func dotProduct(_ other: Self) -> Float {
        precondition(count == other.count, "Dimensions must match")
        return storage.dotProduct(other.storage)
    }
}

// MARK: - Optimizations for Common Large Sizes

extension LargeVectorStorage {
    /// Common vector dimensions used in ML models.
    /// 
    /// These sizes correspond to popular embedding dimensions:
    /// - 768: BERT-base
    /// - 1024: Common power-of-2 size
    /// - 1536: GPT-2 medium, larger BERT models
    /// - 2048: Power-of-2, some vision models
    /// - 3072: GPT-2 large
    /// - 4096: Very large models
    public enum CommonSize: Int {
        case dim768 = 768
        case dim1024 = 1024
        case dim1536 = 1536
        case dim2048 = 2048
        case dim3072 = 3072
        case dim4096 = 4096
    }
    
    /// Create storage for a common dimension.
    /// 
    /// Factory method for creating storage with predefined sizes
    /// commonly used in ML models.
    /// 
    /// - Parameter commonSize: One of the predefined common sizes
    /// 
    /// - Complexity: O(n) where n is the common size value
    @inlinable
    public init(commonSize: CommonSize) {
        self.storage = COWDynamicStorage(dimension: commonSize.rawValue)
    }
    
    /// Create storage for a common dimension with repeating value.
    /// 
    /// Factory method for creating constant vectors with common sizes.
    /// 
    /// - Parameters:
    ///   - commonSize: One of the predefined common sizes
    ///   - value: Value to fill all elements with
    /// 
    /// - Complexity: O(n) where n is the common size value
    @inlinable
    public init(commonSize: CommonSize, repeating value: Float) {
        self.storage = COWDynamicStorage(dimension: commonSize.rawValue, repeating: value)
    }
    
    /// Create storage for exactly 768 dimensions.
    /// 
    /// Optimized factory for BERT-base sized vectors.
    /// 
    /// - Parameter values: Array with exactly 768 elements
    /// - Returns: Storage containing the values
    /// - Precondition: Array must have exactly 768 elements
    @inlinable
    public static func vector768(from values: [Float]) -> Self {
        precondition(values.count == 768, "Must have exactly 768 values")
        return Self(from: values)
    }
    
    /// Create storage for exactly 1536 dimensions.
    /// 
    /// Optimized factory for GPT-2 medium sized vectors.
    /// 
    /// - Parameter values: Array with exactly 1536 elements
    /// - Returns: Storage containing the values
    /// - Precondition: Array must have exactly 1536 elements
    @inlinable
    public static func vector1536(from values: [Float]) -> Self {
        precondition(values.count == 1536, "Must have exactly 1536 values")
        return Self(from: values)
    }
    
    /// Create storage for exactly 3072 dimensions.
    /// 
    /// Optimized factory for GPT-2 large sized vectors.
    /// 
    /// - Parameter values: Array with exactly 3072 elements
    /// - Returns: Storage containing the values
    /// - Precondition: Array must have exactly 3072 elements
    @inlinable
    public static func vector3072(from values: [Float]) -> Self {
        precondition(values.count == 3072, "Must have exactly 3072 values")
        return Self(from: values)
    }
}

// MARK: - Additional Operations

extension LargeVectorStorage {
    /// Create a zero vector with specific dimension.
    /// 
    /// Factory method for creating zero-initialized vectors.
    /// 
    /// - Parameter count: Number of elements (must be > 512)
    /// - Returns: Zero-initialized storage
    @inlinable
    public static func zeros(count: Int) -> Self {
        Self(count: count)
    }
    
    /// Create a vector of ones with specific dimension.
    /// 
    /// Factory method for creating vectors filled with 1.0.
    /// 
    /// - Parameter count: Number of elements (must be > 512)
    /// - Returns: Storage filled with ones
    @inlinable
    public static func ones(count: Int) -> Self {
        Self(count: count, repeating: 1.0)
    }
    
    /// Add two vectors element-wise.
    /// 
    /// Uses Accelerate's vDSP_vadd for SIMD performance.
    /// Result has the same dimension as the operands.
    /// 
    /// - Parameters:
    ///   - lhs: First vector
    ///   - rhs: Second vector
    /// - Returns: Sum of the vectors
    /// - Precondition: Both vectors must have same count
    /// 
    /// - Complexity: O(n) where n is count
    @inlinable
    public static func + (lhs: Self, rhs: Self) -> Self {
        precondition(lhs.count == rhs.count, "Dimensions must match")
        
        var result = Self(count: lhs.count)
        result.withUnsafeMutableBufferPointer { dst in
            lhs.withUnsafeBufferPointer { src1 in
                rhs.withUnsafeBufferPointer { src2 in
                    vDSP_vadd(src1.baseAddress!, 1,
                             src2.baseAddress!, 1,
                             dst.baseAddress!, 1,
                             vDSP_Length(lhs.count))
                }
            }
        }
        return result
    }
    
    /// Compute dot product using chunked processing for cache efficiency.
    /// 
    /// For very large vectors (e.g., > 10K elements), processing in chunks
    /// can improve cache locality and reduce memory bandwidth pressure.
    /// This method divides the computation into smaller chunks that fit
    /// better in CPU cache.
    /// 
    /// - Parameters:
    ///   - other: Another storage of the same size
    ///   - chunkSize: Size of each processing chunk (default: 1024)
    /// - Returns: Dot product result
    /// - Precondition: Both storages must have same count
    /// 
    /// - Note: The result is mathematically identical to regular dotProduct
    ///   but may have better performance for very large vectors.
    /// 
    /// - Complexity: O(n) where n is count, but with better cache behavior
    @inlinable
    public func chunkedDotProduct(_ other: Self, chunkSize: Int = 1024) -> Float {
        precondition(count == other.count, "Dimensions must match")
        
        var result: Float = 0
        let chunks = (count + chunkSize - 1) / chunkSize
        
        for i in 0..<chunks {
            let start = i * chunkSize
            let end = min(start + chunkSize, count)
            let chunkCount = end - start
            
            var chunkResult: Float = 0
            self.withUnsafeBufferPointer { selfBuffer in
                other.withUnsafeBufferPointer { otherBuffer in
                    let selfChunk = selfBuffer.baseAddress! + start
                    let otherChunk = otherBuffer.baseAddress! + start
                    
                    vDSP_dotpr(selfChunk, 1,
                              otherChunk, 1,
                              &chunkResult,
                              vDSP_Length(chunkCount))
                }
            }
            result += chunkResult
        }
        
        return result
    }
}