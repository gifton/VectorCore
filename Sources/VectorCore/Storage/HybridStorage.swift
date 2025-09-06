//
//  HybridStorage.swift
//  VectorCore
//
//

import Foundation

// MARK: - Hybrid Storage

/// Storage that uses stack allocation for small vectors and heap for large ones
public struct HybridStorage<Element: BinaryFloatingPoint & SIMDScalar> {
    /// Storage backing - either inline or heap
    private var backing: Backing
    
    /// The number of elements
    public let count: Int
    
    /// Maximum elements that fit in inline storage (64 bytes)
    private static var inlineCapacity: Int {
        64 / MemoryLayout<Element>.stride
    }
    
    /// Storage backing options
    private enum Backing {
        case inline(InlineBuffer)
        case heap(ManagedStorage<Element>)
    }
    
    /// Initialize with specified capacity
    public init(capacity: Int) {
        self.count = capacity
        if capacity <= Self.inlineCapacity {
            self.backing = .inline(InlineBuffer())
        } else {
            self.backing = .heap(ManagedStorage(capacity: capacity))
        }
    }
    
    /// Initialize from array
    public init(from array: [Element]) {
        self.count = array.count
        if array.count <= Self.inlineCapacity {
            var buffer = InlineBuffer()
            buffer.initialize(from: array)
            self.backing = .inline(buffer)
        } else {
            self.backing = .heap(ManagedStorage(from: array))
        }
    }
    
    /// Initialize with repeating value
    public init(capacity: Int, repeating value: Element) {
        self.init(capacity: capacity)
        self.fill(with: value)
    }
}

// MARK: - Inline Buffer

extension HybridStorage {
    /// Fixed-size inline buffer (64 bytes)
    @frozen
    @usableFromInline
    internal struct InlineBuffer {
        // 64 bytes of storage - enough for 16 Float32 or 8 Float64
        private var storage: (
            UInt64, UInt64, UInt64, UInt64,
            UInt64, UInt64, UInt64, UInt64
        ) = (0, 0, 0, 0, 0, 0, 0, 0)
        
        /// Initialize from array
        mutating func initialize(from array: [Element]) {
            precondition(array.count <= HybridStorage.inlineCapacity)
            withUnsafeMutableBytes(of: &storage) { buffer in
                let typed = buffer.bindMemory(to: Element.self)
                _ = typed.initialize(from: array)
            }
        }
        
        /// Access as buffer pointer
        func withUnsafeBufferPointer<R>(
            count: Int,
            _ body: (UnsafeBufferPointer<Element>) throws -> R
        ) rethrows -> R {
            try withUnsafeBytes(of: storage) { buffer in
                let typed = buffer.bindMemory(to: Element.self)
                let bufferPointer = UnsafeBufferPointer(start: typed.baseAddress!, count: count)
                return try body(bufferPointer)
            }
        }
        
        /// Mutable access as buffer pointer
        mutating func withUnsafeMutableBufferPointer<R>(
            count: Int,
            _ body: (UnsafeMutableBufferPointer<Element>) throws -> R
        ) rethrows -> R {
            try withUnsafeMutableBytes(of: &storage) { buffer in
                let typed = buffer.bindMemory(to: Element.self)
                let bufferPointer = UnsafeMutableBufferPointer(start: typed.baseAddress!, count: count)
                return try body(bufferPointer)
            }
        }
        
        /// Get element at index
        func get(at index: Int) -> Element {
            withUnsafeBufferPointer(count: HybridStorage.inlineCapacity) { buffer in
                buffer[index]
            }
        }
        
        /// Set element at index
        mutating func set(_ value: Element, at index: Int) {
            withUnsafeMutableBufferPointer(count: HybridStorage.inlineCapacity) { buffer in
                buffer[index] = value
            }
        }
    }
}

// MARK: - Copy-on-Write Support

extension HybridStorage {
    /// Ensure unique ownership before mutation
    private mutating func ensureUniquelyReferenced() {
        switch backing {
        case .inline:
            // Inline storage is always unique (value type)
            break
        case .heap(var heapStorage):
            if !heapStorage.isUniquelyReferenced {
                // Create a copy
                var newStorage = ManagedStorage<Element>(capacity: count)
                newStorage.copyFrom(heapStorage)
                backing = .heap(newStorage)
            }
        }
    }
    
    /// Check if storage is uniquely referenced
    public var isUniquelyReferenced: Bool {
        mutating get {
            switch backing {
            case .inline:
                return true
            case .heap(var heapStorage):
                return heapStorage.isUniquelyReferenced
            }
        }
    }
}

// MARK: - VectorStorage Conformance

extension HybridStorage: VectorStorage {
    public typealias Scalar = Element
    
    // Protocol requirement: init with zeros
    public init() {
        self.init(capacity: 0)
    }
    
    // Protocol requirement: init with repeating value
    public init(repeating value: Element) {
        fatalError("HybridStorage requires explicit capacity - use init(capacity:repeating:)")
    }
    
    // Protocol requirement: init from array - already defined above
    
    public subscript(index: Int) -> Element {
        get {
            precondition(index >= 0 && index < count, "Index out of bounds")
            switch backing {
            case .inline(let buffer):
                return buffer.get(at: index)
            case .heap(let heapStorage):
                return heapStorage[index]
            }
        }
        set {
            precondition(index >= 0 && index < count, "Index out of bounds")
            ensureUniquelyReferenced()
            switch backing {
            case .inline(var buffer):
                buffer.set(newValue, at: index)
                backing = .inline(buffer)
            case .heap(var heapStorage):
                heapStorage[index] = newValue
                backing = .heap(heapStorage)
            }
        }
    }
    
    public func withUnsafeBufferPointer<R>(
        _ body: (UnsafeBufferPointer<Element>) throws -> R
    ) rethrows -> R {
        switch backing {
        case .inline(let buffer):
            return try buffer.withUnsafeBufferPointer(count: count, body)
        case .heap(let heapStorage):
            return try heapStorage.withUnsafeBufferPointer(body)
        }
    }
    
    public mutating func withUnsafeMutableBufferPointer<R>(
        _ body: (UnsafeMutableBufferPointer<Element>) throws -> R
    ) rethrows -> R {
        ensureUniquelyReferenced()
        switch backing {
        case .inline(var buffer):
            defer { backing = .inline(buffer) }
            return try buffer.withUnsafeMutableBufferPointer(count: count, body)
        case .heap(var heapStorage):
            defer { backing = .heap(heapStorage) }
            return try heapStorage.withUnsafeMutableBufferPointer(body)
        }
    }
}

// MARK: - Convenience Methods

extension HybridStorage {
    /// Fill storage with a value
    public mutating func fill(with value: Element) {
        ensureUniquelyReferenced()
        for i in 0..<count {
            self[i] = value
        }
    }
    
    /// Copy from another storage
    public mutating func copyFrom(_ other: HybridStorage) {
        precondition(count == other.count, "Size mismatch")
        withUnsafeMutableBufferPointer { dest in
            other.withUnsafeBufferPointer { src in
                _ = dest.initialize(from: src)
            }
        }
    }
    
    /// Check if using inline storage
    public var isInline: Bool {
        switch backing {
        case .inline: return true
        case .heap: return false
        }
    }
}

// MARK: - Sendable Conformance

extension HybridStorage: @unchecked Sendable {}

// MARK: - Debug Support

extension HybridStorage: CustomDebugStringConvertible {
    public var debugDescription: String {
        let storageType = isInline ? "inline" : "heap"
        return "HybridStorage<\(Element.self)>(\(storageType), count: \(count))"
    }
}