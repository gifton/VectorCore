//
//  UnifiedStorage.swift
//  VectorCore
//
//

import Foundation

// MARK: - Move-Only Storage (gated off)
#if MOVE_ONLY_EXPERIMENTAL
/// Move-only storage for zero-copy semantics
/// This is the core storage type that provides manual memory management
public struct MoveOnlyStorage<Element: BinaryFloatingPoint & SIMDScalar>: ~Copyable {
    private let pointer: UnsafeMutableRawPointer
    public let capacity: Int
    public let alignment: Int
    
    public init(capacity: Int, alignment: Int = 64) {
        let byteCount = capacity * MemoryLayout<Element>.stride
        self.pointer = UnsafeMutableRawPointer.allocate(
            byteCount: byteCount,
            alignment: alignment
        )
        self.capacity = capacity
        self.alignment = alignment
        pointer.initializeMemory(as: UInt8.self, repeating: 0, count: byteCount)
    }
    
    public init(pointer: UnsafeMutableRawPointer, capacity: Int, alignment: Int) {
        self.pointer = pointer
        self.capacity = capacity
        self.alignment = alignment
    }
    
    public consuming func into() -> UnsafeMutableBufferPointer<Element> {
        let typed = pointer.bindMemory(to: Element.self, capacity: capacity)
        return UnsafeMutableBufferPointer(start: typed, count: capacity)
    }
    
    public borrowing func copy() -> MoveOnlyStorage {
        let byteCount = capacity * MemoryLayout<Element>.stride
        let newPointer = UnsafeMutableRawPointer.allocate(
            byteCount: byteCount,
            alignment: alignment
        )
        newPointer.copyMemory(from: pointer, byteCount: byteCount)
        return MoveOnlyStorage(pointer: newPointer, capacity: capacity, alignment: alignment)
    }
    
    public borrowing func withUnsafeBufferPointer<R>(
        _ body: (UnsafeBufferPointer<Element>) throws -> R
    ) rethrows -> R {
        let typed = pointer.bindMemory(to: Element.self, capacity: capacity)
        let buffer = UnsafeBufferPointer(start: typed, count: capacity)
        return try body(buffer)
    }
    
    public borrowing func withUnsafeMutableBufferPointer<R>(
        _ body: (UnsafeMutableBufferPointer<Element>) throws -> R
    ) rethrows -> R {
        let typed = pointer.bindMemory(to: Element.self, capacity: capacity)
        let buffer = UnsafeMutableBufferPointer(start: typed, count: capacity)
        return try body(buffer)
    }
    
    deinit {
        pointer.deallocate()
    }
}
#endif

// MARK: - Managed Storage with Reference Counting

/// Internal storage buffer with reference counting
final class StorageBuffer<Element: BinaryFloatingPoint & SIMDScalar>: @unchecked Sendable {
    let pointer: UnsafeMutableRawPointer
    let capacity: Int
    let alignment: Int
    
    init(capacity: Int, alignment: Int = 64) {
        let byteCount = capacity * MemoryLayout<Element>.stride
        self.pointer = UnsafeMutableRawPointer.allocate(
            byteCount: byteCount,
            alignment: alignment
        )
        self.capacity = capacity
        self.alignment = alignment
        
        // Initialize to zero
        pointer.initializeMemory(as: UInt8.self, repeating: 0, count: byteCount)
    }
    
    init(copyFrom other: StorageBuffer<Element>) {
        let byteCount = other.capacity * MemoryLayout<Element>.stride
        self.pointer = UnsafeMutableRawPointer.allocate(
            byteCount: byteCount,
            alignment: other.alignment
        )
        self.capacity = other.capacity
        self.alignment = other.alignment
        
        pointer.copyMemory(from: other.pointer, byteCount: byteCount)
    }
    
    deinit {
        pointer.deallocate()
    }
}

// MARK: - Managed Storage with COW

/// Managed storage with copy-on-write semantics
public struct ManagedStorage<Element: BinaryFloatingPoint & SIMDScalar>: @unchecked Sendable {
    private var buffer: StorageBuffer<Element>
    
    public var capacity: Int { buffer.capacity }
    public var alignment: Int { buffer.alignment }
    
    /// Initialize with specified capacity
    public init(capacity: Int, alignment: Int = 64) {
        self.buffer = StorageBuffer(capacity: capacity, alignment: alignment)
    }
    
    /// Initialize from array
    public init(from array: [Element], alignment: Int = 64) {
        self.buffer = StorageBuffer(capacity: array.count, alignment: alignment)
        withUnsafeMutableBufferPointer { dest in
            _ = dest.initialize(from: array)
        }
    }
    
    /// Ensure unique ownership before mutation
    private mutating func ensureUniquelyReferenced() {
        if !isKnownUniquelyReferenced(&buffer) {
            buffer = StorageBuffer(copyFrom: buffer)
        }
    }
    
    /// Borrow for read-only access
    public func withUnsafeBufferPointer<R>(
        _ body: (UnsafeBufferPointer<Element>) throws -> R
    ) rethrows -> R {
        let typed = buffer.pointer.bindMemory(to: Element.self, capacity: capacity)
        let bufferPointer = UnsafeBufferPointer(start: typed, count: capacity)
        return try body(bufferPointer)
    }
    
    /// Borrow for mutable access (triggers COW if needed)
    public mutating func withUnsafeMutableBufferPointer<R>(
        _ body: (UnsafeMutableBufferPointer<Element>) throws -> R
    ) rethrows -> R {
        ensureUniquelyReferenced()
        let typed = buffer.pointer.bindMemory(to: Element.self, capacity: capacity)
        let bufferPointer = UnsafeMutableBufferPointer(start: typed, count: capacity)
        return try body(bufferPointer)
    }
    
    /// Element access
    public subscript(index: Int) -> Element {
        get {
            precondition(index >= 0 && index < capacity, "Index out of bounds")
            return withUnsafeBufferPointer { $0[index] }
        }
        set {
            precondition(index >= 0 && index < capacity, "Index out of bounds")
            ensureUniquelyReferenced()
            let typed = buffer.pointer.bindMemory(to: Element.self, capacity: capacity)
            typed[index] = newValue
        }
    }
}

// MARK: - Storage Protocol Conformance

extension ManagedStorage: VectorStorage {
    public typealias Scalar = Element
    
    public var count: Int { capacity }
    
    // Protocol requirement: init with zeros
    public init() {
        self.init(capacity: 0)
    }
    
    // Protocol requirement: init with repeating value
    public init(repeating value: Element) {
        fatalError("ManagedStorage requires explicit capacity - use init(capacity:) then fill(with:)")
    }
    
    // Protocol requirement: init from array
    public init(from values: [Element]) {
        self.init(capacity: values.count)
        values.withUnsafeBufferPointer { srcBuffer in
            self.withUnsafeMutableBufferPointer { destBuffer in
                _ = destBuffer.initialize(from: srcBuffer)
            }
        }
    }
}

// MARK: - Convenience Methods

extension ManagedStorage {
    /// Fill storage with a single value
    public mutating func fill(with value: Element) {
        withUnsafeMutableBufferPointer { buffer in
            for i in 0..<buffer.count {
                buffer[i] = value
            }
        }
    }
    
    /// Copy from another storage
    public mutating func copyFrom(_ other: ManagedStorage<Element>) {
        precondition(capacity == other.capacity, "Capacity mismatch")
        withUnsafeMutableBufferPointer { dest in
            other.withUnsafeBufferPointer { src in
                _ = dest.initialize(from: src)
            }
        }
    }
    
    /// Check if storage is uniquely referenced
    public var isUniquelyReferenced: Bool {
        mutating get {
            isKnownUniquelyReferenced(&buffer)
        }
    }
}

// MARK: - Equatable Conformance

extension ManagedStorage: Equatable where Element: Equatable {
    public static func == (lhs: ManagedStorage, rhs: ManagedStorage) -> Bool {
        guard lhs.capacity == rhs.capacity else { return false }
        return lhs.withUnsafeBufferPointer { lhsBuffer in
            rhs.withUnsafeBufferPointer { rhsBuffer in
                for i in 0..<lhs.capacity {
                    if lhsBuffer[i] != rhsBuffer[i] {
                        return false
                    }
                }
                return true
            }
        }
    }
}

// MARK: - Slice Support

/// A view into a contiguous region of storage
public struct StorageSlice<Element: BinaryFloatingPoint & SIMDScalar> {
    private let base: UnsafePointer<Element>
    public let count: Int
    
    init(base: UnsafePointer<Element>, count: Int) {
        self.base = base
        self.count = count
    }
    
    /// Access elements by index
    public subscript(index: Int) -> Element {
        precondition(index >= 0 && index < count, "Index out of bounds")
        return base[index]
    }
    
    /// Iterate over elements
    public func withUnsafeBufferPointer<R>(
        _ body: (UnsafeBufferPointer<Element>) throws -> R
    ) rethrows -> R {
        try body(UnsafeBufferPointer(start: base, count: count))
    }
}

extension ManagedStorage {
    /// Create a slice view of the storage
    public func slice(from start: Int, to end: Int) -> StorageSlice<Element> {
        precondition(start >= 0 && end <= capacity && start < end, "Invalid slice bounds")
        return withUnsafeBufferPointer { buffer in
            StorageSlice(base: buffer.baseAddress! + start, count: end - start)
        }
    }
}
