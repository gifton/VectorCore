// VectorCore: Aligned Value Storage
//
// Value-type storage with proper alignment and Copy-on-Write semantics
//
// This type provides a value-semantic wrapper around aligned memory allocations,
// implementing Copy-on-Write (COW) for efficient value semantics. It's designed
// to replace the reference-type AlignedStorage while maintaining the same
// performance characteristics and alignment guarantees.
//

import Foundation

/// Value-type storage with guaranteed memory alignment and COW semantics.
///
/// `AlignedValueStorage` provides efficient storage for floating-point vectors with
/// guaranteed memory alignment for SIMD operations. It implements Copy-on-Write (COW)
/// semantics to provide value-type behavior while avoiding unnecessary copies.
///
/// ## Key Features
/// - Guaranteed memory alignment (16-byte minimum, configurable up to 64-byte)
/// - Copy-on-Write optimization for efficient value semantics
/// - Thread-safe through value semantics (each instance is independent)
/// - Compatible with Accelerate framework operations
///
/// ## Performance Characteristics
/// - O(1) copy operations (until mutation)
/// - O(n) copy on first mutation after sharing
/// - Zero overhead for unique instances
///
/// ## Example Usage
/// ```swift
/// // Create storage for 512 floats with 64-byte alignment
/// var storage = AlignedValueStorage(count: 512, alignment: 64)
///
/// // Copying is cheap (no allocation)
/// let copy = storage
///
/// // Mutation triggers COW if needed
/// storage[0] = 42.0  // Allocates new storage if shared
/// ```
@usableFromInline
internal struct AlignedValueStorage: VectorStorage, VectorStorageOperations {
    /// Internal reference type for COW implementation.
    /// This class holds the actual aligned memory and is shared between copies
    /// until mutation occurs.
    @usableFromInline
    final class Storage: @unchecked Sendable {
        /// Aligned pointer to float data.
        /// This pointer is guaranteed to be aligned to the specified alignment
        /// and owns the allocated memory. Deallocated in deinit.
        @usableFromInline
        let ptr: UnsafeMutablePointer<Float>

        /// Number of elements in the storage.
        /// This is the logical size of the vector, not the allocated size
        /// (which may be larger due to alignment requirements).
        @usableFromInline
        let count: Int

        /// Alignment in bytes for this storage.
        /// Common values:
        /// - 16: Minimum for SIMD operations
        /// - 64: Optimal for cache line alignment
        /// Must be a power of 2.
        @usableFromInline
        let alignment: Int

        /// Initialize storage with specified count and alignment.
        ///
        /// Uses posix_memalign for guaranteed alignment. The allocated memory
        /// is zero-initialized for safety.
        ///
        /// - Parameters:
        ///   - count: Number of Float elements to store
        ///   - alignment: Required alignment in bytes (must be power of 2)
        /// - Precondition: alignment must be a power of 2 and at least 16
        @usableFromInline
        convenience init(count: Int, alignment: Int = 64) {
            self.init(count: count, alignment: alignment, zeroed: true)
        }

        /// Initialize storage with optional zero-initialization.
        ///
        /// For operations that will immediately overwrite all elements (like vector
        /// addition), zeroing can be skipped for better performance.
        ///
        /// - Parameters:
        ///   - count: Number of Float elements to store
        ///   - alignment: Required alignment in bytes (must be power of 2)
        ///   - zeroed: Whether to zero-initialize the memory (default: true)
        /// - Precondition: alignment must be a power of 2 and at least 16
        /// - Warning: If zeroed is false, all elements must be written before reading
        @usableFromInline
        init(count: Int, alignment: Int = 64, zeroed: Bool) {
            self.count = count
            self.alignment = alignment

            // Allocate aligned memory using posix_memalign.
            // This guarantees the memory address will be a multiple of alignment.
            var rawPtr: UnsafeMutableRawPointer?
            let result = posix_memalign(&rawPtr, alignment, count * MemoryLayout<Float>.stride)
            precondition(result == 0, "Failed to allocate aligned memory")

            self.ptr = rawPtr!.assumingMemoryBound(to: Float.self)

            if zeroed {
                // Initialize to zero for safety and predictable behavior.
                // This prevents undefined behavior from uninitialized memory.
                self.ptr.initialize(repeating: 0, count: count)
            }
        }

        /// Create a deep copy from another storage instance.
        ///
        /// This is used by COW when mutation is detected on a shared instance.
        /// Allocates new aligned memory and copies all data.
        ///
        /// - Parameter other: Storage instance to copy from
        @usableFromInline
        init(from other: Storage) {
            self.count = other.count
            self.alignment = other.alignment

            // Allocate new aligned memory
            var rawPtr: UnsafeMutableRawPointer?
            let result = posix_memalign(&rawPtr, alignment, count * MemoryLayout<Float>.stride)
            precondition(result == 0, "Failed to allocate aligned memory")

            self.ptr = rawPtr!.assumingMemoryBound(to: Float.self)

            // Copy data from source storage
            self.ptr.initialize(from: other.ptr, count: count)
        }

        deinit {
            // Clean up allocated memory
            ptr.deinitialize(count: count)
            AlignedMemory.deallocate(ptr)
        }
    }

    /// Reference to internal storage.
    /// This is the only stored property, making the struct lightweight.
    /// The 'var' declaration is essential for COW - it allows reassignment
    /// when creating a unique copy.
    @usableFromInline
    internal var storage: Storage

    /// Number of elements in the storage.
    ///
    /// This property provides read-only access to the count of elements.
    /// The count is immutable after initialization to maintain safety.
    public var count: Int { storage.count }

    /// Initialize storage with specified element count and alignment.
    ///
    /// Creates a new aligned storage with zero-initialized memory. The alignment
    /// parameter determines memory alignment for SIMD operations.
    ///
    /// - Parameters:
    ///   - count: Number of Float elements to store
    ///   - alignment: Memory alignment in bytes (default: 64 for cache efficiency)
    ///
    /// - Precondition: `count` must be positive
    /// - Precondition: `alignment` must be a power of 2 and at least 16
    ///
    /// - Complexity: O(n) where n is count
    public init(count: Int, alignment: Int = 64) {
        self.storage = Storage(count: count, alignment: alignment)
    }

    /// Initialize storage without zero-initialization (internal use).
    ///
    /// For operations that will immediately overwrite all elements.
    ///
    /// - Parameters:
    ///   - count: Number of Float elements to store
    ///   - alignment: Memory alignment in bytes
    ///   - uninitialized: Must be true to skip initialization
    /// - Warning: All elements must be written before any read operation
    @usableFromInline
    internal init(count: Int, alignment: Int = 64, uninitialized: Bool) {
        precondition(uninitialized, "Use init(count:alignment:) for zero-initialized storage")
        self.storage = Storage(count: count, alignment: alignment, zeroed: false)
    }

    /// Initialize storage filled with a repeating value.
    ///
    /// Creates aligned storage where all elements are set to the specified value.
    /// Useful for creating vectors of constants.
    ///
    /// - Parameters:
    ///   - count: Number of Float elements to store
    ///   - repeating: Value to fill all elements with
    ///   - alignment: Memory alignment in bytes (default: 64)
    ///
    /// - Complexity: O(n) where n is count
    public init(count: Int, repeating value: Float, alignment: Int = 64) {
        self.storage = Storage(count: count, alignment: alignment)
        self.storage.ptr.initialize(repeating: value, count: count)
    }

    /// Initialize storage from an array of values.
    ///
    /// Creates aligned storage containing a copy of the provided array values.
    /// The array size must match the specified count.
    ///
    /// - Parameters:
    ///   - count: Expected number of elements
    ///   - values: Array of Float values to copy
    ///   - alignment: Memory alignment in bytes (default: 64)
    ///
    /// - Precondition: `values.count` must equal `count`
    ///
    /// - Complexity: O(n) where n is count
    public init(count: Int, from values: [Float], alignment: Int = 64) {
        precondition(values.count == count, "Value count must match specified count")
        self.storage = Storage(count: count, alignment: alignment)
        self.storage.ptr.initialize(from: values, count: count)
    }

    /// Ensure storage is uniquely referenced for COW.
    ///
    /// This method implements the Copy-on-Write optimization. It checks if the
    /// internal storage is uniquely referenced (not shared). If shared, it creates
    /// a new copy before mutation.
    ///
    /// Call this before any mutating operation to maintain value semantics.
    ///
    /// - Complexity: O(1) if unique, O(n) if copy needed
    @inlinable
    mutating func makeUnique() {
        if !isKnownUniquelyReferenced(&storage) {
            storage = Storage(from: storage)
        }
    }

    // MARK: - VectorStorage conformance

    /// Default initializer - not supported for AlignedValueStorage.
    /// Use init(count:) instead.
    public init() {
        fatalError("AlignedValueStorage requires explicit count")
    }

    /// Repeating value initializer - not supported without count.
    /// Use init(count:repeating:) instead.
    public init(repeating value: Float) {
        fatalError("AlignedValueStorage requires explicit count")
    }

    /// Initialize from array, inferring count from array size.
    ///
    /// - Parameter values: Array of values to store
    public init(from values: [Float]) {
        self.init(count: values.count, from: values)
    }

    /// Access elements by index with bounds checking.
    ///
    /// Provides read/write access to individual elements. The setter
    /// automatically triggers COW if the storage is shared.
    ///
    /// - Parameter index: Zero-based index of the element
    /// - Returns: Value at the specified index (for getter)
    ///
    /// - Precondition: `index` must be in range `0..<count`
    ///
    /// - Complexity: O(1) for getter, O(1) or O(n) for setter (if COW triggered)
    @inlinable
    public subscript(index: Int) -> Float {
        get {
            precondition(index >= 0 && index < count)
            return storage.ptr[index]
        }
        set {
            precondition(index >= 0 && index < count)
            makeUnique()  // Trigger COW if storage is shared
            storage.ptr[index] = newValue
        }
    }

    /// Access the underlying memory as an unsafe buffer pointer for reading.
    ///
    /// This method provides direct read-only access to the underlying memory
    /// for performance-critical operations. The pointer is valid only for the
    /// duration of the closure.
    ///
    /// - Parameter body: Closure that receives the buffer pointer
    /// - Returns: Result of the closure
    /// - Throws: Any error thrown by the closure
    ///
    /// - Warning: Do not store or use the pointer outside the closure
    ///
    /// - Complexity: O(1)
    @inlinable
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try body(UnsafeBufferPointer(start: storage.ptr, count: count))
    }

    /// Access the underlying memory as an unsafe mutable buffer pointer.
    ///
    /// This method provides direct read/write access to the underlying memory
    /// for performance-critical operations. Automatically triggers COW if needed.
    /// The pointer is valid only for the duration of the closure.
    ///
    /// - Parameter body: Closure that receives the mutable buffer pointer
    /// - Returns: Result of the closure
    /// - Throws: Any error thrown by the closure
    ///
    /// - Warning: Do not store or use the pointer outside the closure
    ///
    /// - Complexity: O(1) if unique, O(n) if COW triggered
    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        makeUnique()  // Ensure exclusive access before mutation
        return try body(UnsafeMutableBufferPointer(start: storage.ptr, count: count))
    }

    // MARK: - VectorStorageOperations conformance

    /// Compute dot product with another storage of the same size.
    ///
    /// Uses SwiftSIMDProvider for cross-platform performance.
    /// Both storages must have the same count.
    ///
    /// - Parameter other: Another storage to compute dot product with
    /// - Returns: The dot product as a Float
    ///
    /// - Precondition: `self.count == other.count`
    ///
    /// - Complexity: O(n) where n is count
    @inlinable
    public func dotProduct(_ other: Self) -> Float {
        precondition(count == other.count, "Dimensions must match")

        var result: Float = 0
        self.withUnsafeBufferPointer { aBuffer in
            other.withUnsafeBufferPointer { bBuffer in
                result = SIMDOperations.FloatProvider.dot(
                    aBuffer.baseAddress!,
                    bBuffer.baseAddress!,
                    count: count
                )
            }
        }
        return result
    }
}

// MARK: - Convenience factories for specific sizes

extension AlignedValueStorage {
    /// Create zero-initialized storage for 512 elements with 64-byte alignment.
    /// Common size for BERT-style embeddings.
    public static func storage512() -> AlignedValueStorage {
        AlignedValueStorage(count: 512)
    }

    /// Create zero-initialized storage for 768 elements with 64-byte alignment.
    /// Common size for BERT-base embeddings.
    public static func storage768() -> AlignedValueStorage {
        AlignedValueStorage(count: 768)
    }

    /// Create zero-initialized storage for 1536 elements with 64-byte alignment.
    /// Common size for larger transformer models.
    public static func storage1536() -> AlignedValueStorage {
        AlignedValueStorage(count: 1536)
    }

    /// Create zero-initialized storage for 3072 elements with 64-byte alignment.
    /// Common size for GPT-style embeddings.
    public static func storage3072() -> AlignedValueStorage {
        AlignedValueStorage(count: 3072)
    }
}
