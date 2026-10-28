// VectorCore: Aligned Dynamic Array Storage

import Foundation

// Allocation ownership is private to the value wrapper. Shared allocations are read-only;
// both public write paths call makeUnique first. ReferenceStorageValueTests covers copy
// isolation, including throwing borrows and concurrentIndependentCopies.
@usableFromInline
internal final class AlignedDynamicAllocation: @unchecked Sendable {
    @usableFromInline internal let ptr: UnsafeMutablePointer<Float>
    @usableFromInline internal let _count: Int
    @usableFromInline internal let alignment: Int
    internal let usedPosixAlignedAlloc: Bool

    /// Per-call primary allocator seam. The allocator must return Float-bound,
    /// uninitialized storage suitable for `free`, or throw without transferring storage.
    /// The fallback requests the same alignment; actual exhaustion can still terminate.
    init(dimension: Int, repeating value: Float, alignment: Int,
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

    @usableFromInline
    convenience init(copying original: AlignedDynamicAllocation) {
        self.init(dimension: original._count, repeating: 0, alignment: original.alignment,
                  primaryAllocator: { try AlignedMemory.allocateAligned(count: $0, alignment: $1) })
        withExtendedLifetime(original) {
            ptr.update(from: original.ptr, count: original._count)
        }
    }

    deinit {
        ptr.deinitialize(count: _count)
        if usedPosixAlignedAlloc {
            AlignedMemory.deallocate(ptr)
        } else {
            UnsafeMutableRawPointer(ptr).deallocate()
        }
    }

}

/// Copy-on-write value storage with the requested SIMD alignment.
/// Copies isolate scalar and mutable-buffer writes (ReferenceStorageValueTests).
/// Unsafe buffer pointers are valid only during their closure and must not escape.
public struct AlignedDynamicArrayStorage: VectorStorage, VectorStorageOperations {
    @usableFromInline internal var allocation: AlignedDynamicAllocation
    @usableFromInline internal var _count: Int { allocation._count }
    @usableFromInline internal var alignment: Int { allocation.alignment }
    internal var usedPosixAlignedAlloc: Bool { allocation.usedPosixAlignedAlloc }

    public var count: Int { _count }

    public init(dimension: Int, alignment: Int = AlignedMemory.optimalAlignment) {
        self.init(dimension: dimension, repeating: 0, alignment: alignment)
    }

    public init(dimension: Int, repeating value: Float, alignment: Int = AlignedMemory.optimalAlignment) {
        self.init(dimension: dimension, repeating: value, alignment: alignment,
                  primaryAllocator: { try AlignedMemory.allocateAligned(count: $0, alignment: $1) })
    }

    public init(from values: [Float]) {
        self.init(from: values, alignment: AlignedMemory.optimalAlignment)
    }

    public init(from values: [Float], alignment: Int) {
        self.init(dimension: values.count, alignment: alignment)
        withUnsafeMutableBufferPointer { destination in
            values.withUnsafeBufferPointer { source in
                if let base = source.baseAddress { destination.baseAddress!.update(from: base, count: source.count) }
            }
        }
    }

    internal init(dimension: Int, repeating value: Float, alignment: Int,
                  primaryAllocator: (Int, Int) throws -> UnsafeMutablePointer<Float>) {
        allocation = AlignedDynamicAllocation(dimension: dimension, repeating: value,
                                             alignment: alignment, primaryAllocator: primaryAllocator)
    }

    public init() { fatalError("Dynamic storage requires explicit dimension") }
    public init(repeating value: Float) { fatalError("Dynamic storage requires explicit dimension") }

    @inlinable
    internal mutating func makeUnique() {
        if !isKnownUniquelyReferenced(&allocation) {
            allocation = AlignedDynamicAllocation(copying: allocation)
        }
    }

    @inlinable
    public subscript(index: Int) -> Float {
        get {
            precondition(index >= 0 && index < _count, "Index out of bounds")
            return withExtendedLifetime(allocation) { allocation.ptr[index] }
        }
        set {
            precondition(index >= 0 && index < _count, "Index out of bounds")
            makeUnique()
            allocation.ptr[index] = newValue
        }
    }

    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try withExtendedLifetime(allocation) {
            try body(UnsafeBufferPointer(start: allocation.ptr, count: _count))
        }
    }

    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        makeUnique()
        return try withExtendedLifetime(allocation) {
            try body(UnsafeMutableBufferPointer(start: allocation.ptr, count: _count))
        }
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
        withUnsafeBufferPointer { Array($0) }
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
        withUnsafeBufferPointer { AlignedMemory.isAligned($0.baseAddress!, to: alignment) }
    }
}
