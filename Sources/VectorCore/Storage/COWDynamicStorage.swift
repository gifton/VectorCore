// VectorCore: Copy-on-Write Dynamic Storage

import Foundation

/// Compatibility wrapper delegating value semantics to AlignedDynamicArrayStorage.
/// Copies isolate mutations; see ReferenceStorageValueTests.wrapperCopies.
/// Copying is O(1); the first mutation of shared backing storage copies O(n) elements.
public struct COWDynamicStorage: VectorStorage, VectorStorageOperations {
    /// The underlying copy-on-write value.
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
