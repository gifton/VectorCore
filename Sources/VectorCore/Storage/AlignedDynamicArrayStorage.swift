// VectorCore: Aligned Dynamic Array Storage
//
// Replacement for DynamicArrayStorage with guaranteed memory alignment
//

import Foundation

/// Dynamic array storage with guaranteed memory alignment for SIMD operations
public final class AlignedDynamicArrayStorage: @unchecked Sendable, VectorStorage, VectorStorageOperations {
    /// Aligned pointer to the data
    @usableFromInline
    internal let ptr: UnsafeMutablePointer<Float>

    /// Number of elements
    @usableFromInline
    internal let _count: Int

    /// Alignment used for this storage
    @usableFromInline
    internal let alignment: Int

    public var count: Int { _count }

    /// Tracks whether allocation used posix_memalign (via AlignedMemory)
    @usableFromInline
    internal let usedPosixAlignedAlloc: Bool

    /// Initialize zeroed storage at the requested alignment.
    public convenience init(dimension: Int, alignment: Int = AlignedMemory.optimalAlignment) {
        self.init(dimension: dimension, repeating: 0, alignment: alignment)
    }

    /// Initialize with a repeating value.
    public convenience init(dimension: Int, repeating value: Float, alignment: Int = AlignedMemory.optimalAlignment) {
        self.init(dimension: dimension, repeating: value, alignment: alignment,
                  primaryAllocator: { try AlignedMemory.allocateAligned(count: $0, alignment: $1) })
    }

    /// Initialize from an array using the default alignment.
    public convenience init(from values: [Float]) {
        self.init(from: values, alignment: AlignedMemory.optimalAlignment)
    }

    /// Initialize from an array at the requested alignment.
    public convenience init(from values: [Float], alignment: Int) {
        self.init(dimension: values.count, alignment: alignment)
        values.withUnsafeBufferPointer { buffer in
            if let source = buffer.baseAddress { ptr.update(from: source, count: buffer.count) }
        }
    }

    /// Per-call primary allocator seam. The allocator must return Float-bound,
    /// uninitialized storage suitable for `free`, or throw without transferring storage.
    /// The fallback requests the same alignment; actual exhaustion can still terminate.
    internal init(dimension: Int, repeating value: Float, alignment: Int,
                  primaryAllocator: (Int, Int) throws -> UnsafeMutablePointer<Float>) {
        guard dimension >= 0 else { fatalError("Dimension must be nonnegative") }
        let (byteCount, overflow) = max(dimension, 1).multipliedReportingOverflow(by: MemoryLayout<Float>.stride)
        guard !overflow else { fatalError("Dimension byte count must be representable") }
        guard alignment >= AlignedMemory.minimumAlignment && (alignment & (alignment - 1)) == 0 else {
            fatalError("Alignment must be a power of two and at least 16 bytes")
        }
        self._count = dimension
        self.alignment = alignment
        do {
            self.ptr = try primaryAllocator(dimension, alignment)
            self.usedPosixAlignedAlloc = true
        } catch {
            let raw = UnsafeMutableRawPointer.allocate(byteCount: byteCount, alignment: alignment)
            self.ptr = raw.bindMemory(to: Float.self, capacity: max(dimension, 1))
            self.usedPosixAlignedAlloc = false
        }
        ptr.initialize(repeating: value, count: dimension)
    }

    public required init() {
        fatalError("Dynamic storage requires explicit dimension")
    }

    public init(repeating value: Float) {
        fatalError("Dynamic storage requires explicit dimension")
    }

    deinit {
        ptr.deinitialize(count: _count)
        if usedPosixAlignedAlloc {
            AlignedMemory.deallocate(ptr)
        } else {
            UnsafeMutableRawPointer(ptr).deallocate()
        }
    }

    @inlinable
    public subscript(index: Int) -> Float {
        get {
            precondition(index >= 0 && index < _count, "Index out of bounds")
            return ptr[index]
        }
        set {
            precondition(index >= 0 && index < _count, "Index out of bounds")
            ptr[index] = newValue
        }
    }

    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try body(UnsafeBufferPointer(start: ptr, count: _count))
    }

    public func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try body(UnsafeMutableBufferPointer(start: ptr, count: _count))
    }

    @inlinable
    public func dotProduct(_ other: AlignedDynamicArrayStorage) -> Float {
        precondition(_count == other._count, "Dimensions must match")

        var result: Float = 0
        self.withUnsafeBufferPointer { aBuffer in
            other.withUnsafeBufferPointer { bBuffer in
                result = SIMDOperations.FloatProvider.dot(
                    aBuffer.baseAddress!,
                    bBuffer.baseAddress!,
                    count: _count
                )
            }
        }
        return result
    }

    /// Convert to array for COW operations
    @usableFromInline
    func toArray() -> [Float] {
        Array(UnsafeBufferPointer(start: ptr, count: _count))
    }
}

// Make it compatible with protocols that expect the old type
extension AlignedDynamicArrayStorage {
    /// Bridge method for compatibility with existing DynamicArrayStorage usage
    @inlinable
    func dotProductWithDynamicStorage(_ other: DynamicArrayStorage) -> Float {
        var result: Float = 0
        self.withUnsafeBufferPointer { aBuffer in
            other.withUnsafeBufferPointer { bBuffer in
                result = SIMDOperations.FloatProvider.dot(
                    aBuffer.baseAddress!,
                    bBuffer.baseAddress!,
                    count: _count
                )
            }
        }
        return result
    }
}

// MARK: - AlignedStorage conformance
extension AlignedDynamicArrayStorage: AlignedStorage {
    public var guaranteedAlignment: Int { alignment }


    @inlinable
    public func verifyAlignment() -> Bool {
        AlignedMemory.isAligned(ptr, to: alignment)
    }
}
