// VectorCore: Copy-on-Write Dynamic Storage
//
// Wrapper providing value semantics for DynamicArrayStorage
//
// This type wraps the reference-type DynamicArrayStorage to provide value semantics
// with Copy-on-Write (COW) optimization. It enables DynamicVector to behave as a
// value type while maintaining efficient memory usage for large vectors.
//

import Foundation

/// Copy-on-Write wrapper for DynamicArrayStorage providing value semantics.
///
/// `COWDynamicStorage` transforms the reference-type `DynamicArrayStorage` into a
/// value-type storage with Copy-on-Write semantics. This allows dynamic vectors
/// to have predictable value semantics while avoiding unnecessary copies.
///
/// ## Purpose
/// - Provides value semantics for dynamically-sized vectors
/// - Implements COW to avoid expensive copies until mutation
/// - Maintains compatibility with existing DynamicArrayStorage
///
/// ## Performance Characteristics
/// - O(1) copy operations (until mutation)
/// - O(n) copy on first mutation after sharing
/// - No overhead for unique instances
///
/// ## Implementation Details
/// The type wraps DynamicArrayStorage and uses `isKnownUniquelyReferenced` to
/// detect when a copy is needed, implementing the COW pattern efficiently.
///
/// ## Example
/// ```swift
/// var storage1 = COWDynamicStorage(dimension: 1000)
/// let storage2 = storage1  // Cheap copy (no allocation)
/// storage1[0] = 42.0       // Triggers COW, allocates new storage
/// ```
public struct COWDynamicStorage: VectorStorage, VectorStorageOperations {
    /// The underlying reference-type storage.
    /// Marked as var to allow COW reassignment when copying is needed.
    @usableFromInline
    internal var storage: AlignedDynamicArrayStorage

    /// Number of elements in the storage.
    ///
    /// Delegates to the underlying storage's count.
    public var count: Int { storage.count }

    /// Initialize storage with specified dimension.
    ///
    /// Creates a new dynamic storage with zero-initialized values.
    ///
    /// - Parameter dimension: Number of elements to store
    ///
    /// - Complexity: O(n) where n is dimension
    public init(dimension: Int) {
        self.storage = AlignedDynamicArrayStorage(dimension: dimension)
    }

    /// Initialize storage with repeating value.
    ///
    /// Creates a new dynamic storage filled with the specified value.
    ///
    /// - Parameters:
    ///   - dimension: Number of elements to store
    ///   - value: Value to fill all elements with
    ///
    /// - Complexity: O(n) where n is dimension
    public init(dimension: Int, repeating value: Float) {
        self.storage = AlignedDynamicArrayStorage(dimension: dimension, repeating: value)
    }

    /// Initialize storage from an array of values.
    ///
    /// Creates a new dynamic storage containing a copy of the array values.
    /// The storage size is determined by the array size.
    ///
    /// - Parameter values: Array of Float values to store
    ///
    /// - Complexity: O(n) where n is values.count
    public init(from values: [Float]) {
        self.storage = AlignedDynamicArrayStorage(from: values)
    }

    /// Ensure storage is uniquely referenced for COW.
    ///
    /// This private method implements the Copy-on-Write mechanism. It checks if
    /// the underlying storage is shared and creates a copy if needed.
    ///
    /// - Complexity: O(1) if unique, O(n) if copy needed
    @inlinable
    mutating func makeUnique() {
        if !isKnownUniquelyReferenced(&storage) {
            // Storage is shared, create a deep copy
            storage = AlignedDynamicArrayStorage(from: storage.toArray())
        }
    }

    // MARK: - VectorStorage Conformance

    /// Default initializer - not supported.
    /// Use init(dimension:) instead.
    public init() {
        fatalError("COWDynamicStorage requires explicit dimension")
    }

    /// Repeating value initializer - not supported without dimension.
    /// Use init(dimension:repeating:) instead.
    public init(repeating value: Float) {
        fatalError("COWDynamicStorage requires explicit dimension")
    }

    /// Access elements by index.
    ///
    /// Provides read/write access to individual elements. The setter
    /// automatically triggers COW if the storage is shared.
    ///
    /// - Parameter index: Zero-based index of the element
    /// - Returns: Value at the specified index (for getter)
    ///
    /// - Note: Bounds checking is performed by the underlying storage
    ///
    /// - Complexity: O(1) for getter, O(1) or O(n) for setter (if COW triggered)
    @inlinable
    public subscript(index: Int) -> Float {
        get {
            storage[index]
        }
        set {
            makeUnique()  // Ensure exclusive ownership before mutation
            storage[index] = newValue
        }
    }

    /// Access the underlying memory as an unsafe buffer pointer for reading.
    ///
    /// Delegates to the underlying storage. The pointer is valid only for
    /// the duration of the closure.
    ///
    /// - Parameter body: Closure that receives the buffer pointer
    /// - Returns: Result of the closure
    /// - Throws: Any error thrown by the closure
    ///
    /// - Complexity: O(1)
    @inlinable
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeBufferPointer(body)
    }

    /// Access the underlying memory as an unsafe mutable buffer pointer.
    ///
    /// Triggers COW if needed before providing mutable access. The pointer
    /// is valid only for the duration of the closure.
    ///
    /// - Parameter body: Closure that receives the mutable buffer pointer
    /// - Returns: Result of the closure
    /// - Throws: Any error thrown by the closure
    ///
    /// - Complexity: O(1) if unique, O(n) if COW triggered
    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        makeUnique()  // Ensure exclusive ownership before providing mutable access
        return try storage.withUnsafeMutableBufferPointer(body)
    }

    // MARK: - VectorStorageOperations Conformance

    /// Compute dot product with another COWDynamicStorage.
    ///
    /// Delegates to the underlying storage's dot product implementation.
    /// Both storages must have the same count.
    ///
    /// - Parameter other: Another storage to compute dot product with
    /// - Returns: The dot product result
    ///
    /// - Complexity: O(n) where n is count
    @inlinable
    public func dotProduct(_ other: Self) -> Float {
        storage.dotProduct(other.storage)
    }
}

// MARK: - Helper Extensions
