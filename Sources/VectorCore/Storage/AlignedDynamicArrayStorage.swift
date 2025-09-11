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
    
    /// Initialize with specified dimension and alignment
    public init(dimension: Int, alignment: Int = AlignedMemory.optimalAlignment) {
        self._count = dimension
        self.alignment = alignment
        if let p = try? AlignedMemory.allocateAligned(count: dimension, alignment: alignment) {
            self.ptr = p
        } else {
            // Fallback: allocate with natural alignment to avoid hard crash
            let raw = UnsafeMutableRawPointer.allocate(byteCount: dimension * MemoryLayout<Float>.stride,
                                                       alignment: MemoryLayout<Float>.alignment)
            self.ptr = raw.assumingMemoryBound(to: Float.self)
        }
        
        // Zero-initialize for safety
        ptr.initialize(repeating: 0, count: dimension)
    }
    
    /// Initialize with repeating value
    public init(dimension: Int, repeating value: Float, alignment: Int = AlignedMemory.optimalAlignment) {
        self._count = dimension
        self.alignment = alignment
        if let p = try? AlignedMemory.allocateAligned(count: dimension, alignment: alignment) {
            self.ptr = p
        } else {
            let raw = UnsafeMutableRawPointer.allocate(byteCount: dimension * MemoryLayout<Float>.stride,
                                                       alignment: MemoryLayout<Float>.alignment)
            self.ptr = raw.assumingMemoryBound(to: Float.self)
        }
        
        ptr.initialize(repeating: value, count: dimension)
    }
    
    /// Initialize from array
    public init(from values: [Float]) {
        self._count = values.count
        self.alignment = AlignedMemory.optimalAlignment
        if let p = try? AlignedMemory.allocateAligned(count: values.count, alignment: alignment) {
            self.ptr = p
        } else {
            let raw = UnsafeMutableRawPointer.allocate(byteCount: values.count * MemoryLayout<Float>.stride,
                                                       alignment: MemoryLayout<Float>.alignment)
            self.ptr = raw.assumingMemoryBound(to: Float.self)
        }
        
        ptr.initialize(from: values, count: values.count)
    }
    
    /// Initialize from array with custom alignment
    public init(from values: [Float], alignment: Int) {
        self._count = values.count
        self.alignment = alignment
        if let p = try? AlignedMemory.allocateAligned(count: values.count, alignment: alignment) {
            self.ptr = p
        } else {
            let raw = UnsafeMutableRawPointer.allocate(byteCount: values.count * MemoryLayout<Float>.stride,
                                                       alignment: MemoryLayout<Float>.alignment)
            self.ptr = raw.assumingMemoryBound(to: Float.self)
        }
        
        ptr.initialize(from: values, count: values.count)
    }
    
    public required init() {
        fatalError("Dynamic storage requires explicit dimension")
    }
    
    public init(repeating value: Float) {
        fatalError("Dynamic storage requires explicit dimension")
    }
    
    deinit {
        ptr.deinitialize(count: _count)
        ptr.deallocate()
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
    
    public func verifyAlignment() -> Bool {
        AlignedMemory.isAligned(ptr, to: alignment)
    }
}
