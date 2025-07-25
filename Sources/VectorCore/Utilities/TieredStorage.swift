//
//  TieredStorage.swift
//  VectorCore
//
//  Adaptive storage that automatically selects optimal backing store based on size and access patterns
//

import Foundation

/// Adaptive storage container that automatically selects the optimal backing store
/// based on element count and access patterns, enabling efficient memory usage for different vector sizes.
public struct TieredStorage<Element> {
    // MARK: - Storage Tiers
    
    /// Storage tier variants optimized for different size ranges
    @usableFromInline
    internal enum Tier {
        /// Stack-allocated storage for tiny vectors (≤ 4 elements)
        case inline(InlineBuffer<Element>)
        /// Heap-allocated cache-line aware storage (5-512 elements)
        case compact(CompactBuffer<Element>)
        /// Standard contiguous array (513-2048 elements)
        case standard(ContiguousArray<Element>)
        /// SIMD-aligned memory for large vectors (> 2048 elements)
        case aligned(AlignedBuffer<Element>)
    }
    
    // MARK: - Performance Hints
    
    /// Hints to optimize storage selection and access patterns
    public enum PerformanceHint {
        /// Optimize for sequential access patterns
        case sequential
        /// Optimize for random access patterns
        case random
        /// Optimize for SIMD operations
        case simd
        /// Short-lived storage, prefer stack allocation
        case temporary
    }
    
    // MARK: - Tier Thresholds
    
    /// Maximum elements for inline storage
    @usableFromInline
    internal static var inlineThreshold: Int { 4 }
    
    /// Maximum elements for compact storage
    @usableFromInline
    internal static var compactThreshold: Int { 512 }
    
    /// Maximum elements for standard storage
    @usableFromInline
    internal static var standardThreshold: Int { 2048 }
    
    // MARK: - Properties
    
    /// Current storage tier
    @usableFromInline
    internal private(set) var tier: Tier
    
    /// Performance hint for optimization
    public var performanceHint: PerformanceHint = .sequential
    
    /// Number of elements in storage
    public var count: Int {
        switch tier {
        case .inline(let buffer):
            return buffer.count
        case .compact(let buffer):
            return buffer.count
        case .standard(let array):
            return array.count
        case .aligned(let buffer):
            return buffer.count
        }
    }
    
    /// Capacity of current storage
    public var capacity: Int {
        switch tier {
        case .inline(let buffer):
            return buffer.capacity
        case .compact(let buffer):
            return buffer.capacity
        case .standard(let array):
            return array.capacity
        case .aligned(let buffer):
            return buffer.capacity
        }
    }
    
    // MARK: - Initialization
    
    /// Initialize with capacity and optional performance hint
    public init(capacity: Int, hint: PerformanceHint = .sequential) {
        self.performanceHint = hint
        
        // Create empty tier with specified capacity
        if hint == .simd && capacity > Self.inlineThreshold {
            self.tier = .aligned(AlignedBuffer(capacity: capacity))
        } else if capacity <= Self.inlineThreshold {
            self.tier = .inline(InlineBuffer())
        } else if capacity <= Self.compactThreshold {
            self.tier = .compact(CompactBuffer(capacity: capacity))
        } else if capacity <= Self.standardThreshold {
            var array = ContiguousArray<Element>()
            array.reserveCapacity(capacity)
            self.tier = .standard(array)
        } else {
            self.tier = .aligned(AlignedBuffer(capacity: capacity))
        }
    }
    
    /// Initialize from a sequence of elements
    public init<S: Sequence>(_ elements: S) where S.Element == Element {
        let array = Array(elements)
        self.performanceHint = .sequential
        
        if array.count <= Self.inlineThreshold {
            self.tier = .inline(InlineBuffer(array))
        } else if array.count <= Self.compactThreshold {
            self.tier = .compact(CompactBuffer(array))
        } else if array.count <= Self.standardThreshold {
            self.tier = .standard(ContiguousArray(array))
        } else {
            self.tier = .aligned(AlignedBuffer(array))
        }
    }
    
    // MARK: - Element Access
    
    /// Access elements by index
    public subscript(index: Int) -> Element {
        get {
            precondition(index >= 0 && index < count, "Index out of bounds")
            switch tier {
            case .inline(let buffer):
                return buffer[index]
            case .compact(let buffer):
                return buffer[index]
            case .standard(let array):
                return array[index]
            case .aligned(let buffer):
                return buffer[index]
            }
        }
        set {
            precondition(index >= 0 && index < count, "Index out of bounds")
            ensureUniqueTier()
            
            switch tier {
            case .inline(var buffer):
                buffer[index] = newValue
                tier = .inline(buffer)
            case .compact(let buffer):
                buffer[index] = newValue
            case .standard(var array):
                array[index] = newValue
                tier = .standard(array)
            case .aligned(let buffer):
                buffer[index] = newValue
            }
        }
    }
    
    // MARK: - Copy-on-Write Support
    
    /// Ensure tier is uniquely owned (for COW semantics)
    @usableFromInline
    internal mutating func ensureUniqueTier() {
        switch tier {
        case .compact(let buffer):
            if !buffer.isUniquelyReferenced() {
                tier = .compact(buffer.copy())
            }
        case .aligned(let buffer):
            if !buffer.isUniquelyReferenced() {
                tier = .aligned(buffer.copy())
            }
        default:
            // Inline and standard tiers are value types, no COW needed
            break
        }
    }
    
    // MARK: - Bulk Operations
    
    /// Access storage as unsafe buffer pointer
    public func withUnsafeBufferPointer<R>(
        _ body: (UnsafeBufferPointer<Element>) throws -> R
    ) rethrows -> R {
        switch tier {
        case .inline(let buffer):
            return try buffer.withUnsafeBufferPointer(body)
        case .compact(let buffer):
            return try buffer.withUnsafeBufferPointer(body)
        case .standard(let array):
            return try array.withUnsafeBufferPointer(body)
        case .aligned(let buffer):
            return try buffer.withUnsafeBufferPointer(body)
        }
    }
    
    /// Mutate storage through unsafe buffer pointer
    public mutating func withUnsafeMutableBufferPointer<R>(
        _ body: (UnsafeMutableBufferPointer<Element>) throws -> R
    ) rethrows -> R {
        ensureUniqueTier()
        
        switch tier {
        case .inline(var buffer):
            let result = try buffer.withUnsafeMutableBufferPointer(body)
            tier = .inline(buffer)
            return result
        case .compact(let buffer):
            return try buffer.withUnsafeMutableBufferPointer(body)
        case .standard(var array):
            return try array.withUnsafeMutableBufferPointer { ptr in
                try body(ptr)
            }
        case .aligned(let buffer):
            return try buffer.withUnsafeMutableBufferPointer(body)
        }
    }
    
    // MARK: - Optimization
    
    /// Optimize storage for a specific performance hint
    public mutating func optimizeFor(_ hint: PerformanceHint) {
        performanceHint = hint
        
        // Transition to aligned tier if SIMD requested and not already aligned
        if hint == .simd && count > Self.inlineThreshold {
            switch tier {
            case .aligned:
                return // Already optimal
            default:
                let elements = Array(self)
                tier = .aligned(AlignedBuffer(elements))
            }
        }
    }
    
    /// Reserve minimum capacity
    public mutating func reserveCapacity(_ minimumCapacity: Int) {
        guard minimumCapacity > capacity else { return }
        
        // Currently not implementing tier transitions for simplicity
        // In a full implementation, would transition to appropriate tier
    }
    
    // MARK: - Platform Specific
    
    #if canImport(Accelerate)
    /// Check if storage is properly aligned for SIMD operations
    public var isAlignedForSIMD: Bool {
        switch tier {
        case .aligned(let buffer):
            return buffer.isAligned
        default:
            return false
        }
    }
    #endif
}

// MARK: - Internal Buffer Types

/// Stack-allocated storage for tiny vectors
@usableFromInline
internal struct InlineBuffer<Element> {
    /// Fixed-size tuple storage for up to 4 elements
    @usableFromInline
    internal var storage: (Element?, Element?, Element?, Element?)
    
    /// Actual number of stored elements
    @usableFromInline
    internal private(set) var count: Int
    
    /// Maximum capacity (always 4)
    @usableFromInline
    internal var capacity: Int { 4 }
    
    /// Initialize empty buffer
    @usableFromInline
    internal init() {
        self.storage = (nil, nil, nil, nil)
        self.count = 0
    }
    
    /// Initialize from array
    @usableFromInline
    internal init(_ elements: [Element]) {
        precondition(elements.count <= 4, "Too many elements for inline storage")
        self.count = elements.count
        
        switch elements.count {
        case 0:
            self.storage = (nil, nil, nil, nil)
        case 1:
            self.storage = (elements[0], nil, nil, nil)
        case 2:
            self.storage = (elements[0], elements[1], nil, nil)
        case 3:
            self.storage = (elements[0], elements[1], elements[2], nil)
        default:
            self.storage = (elements[0], elements[1], elements[2], elements[3])
        }
    }
    
    /// Element access
    @usableFromInline
    internal subscript(index: Int) -> Element {
        get {
            precondition(index >= 0 && index < count, "Index out of bounds")
            switch index {
            case 0: return storage.0!
            case 1: return storage.1!
            case 2: return storage.2!
            default: return storage.3!
            }
        }
        set {
            precondition(index >= 0 && index < count, "Index out of bounds")
            switch index {
            case 0: storage.0 = newValue
            case 1: storage.1 = newValue
            case 2: storage.2 = newValue
            default: storage.3 = newValue
            }
        }
    }
    
    /// Buffer pointer access
    @usableFromInline
    internal func withUnsafeBufferPointer<R>(
        _ body: (UnsafeBufferPointer<Element>) throws -> R
    ) rethrows -> R {
        var array = [Element]()
        array.reserveCapacity(count)
        for i in 0..<count {
            array.append(self[i])
        }
        return try array.withUnsafeBufferPointer(body)
    }
    
    /// Mutable buffer pointer access
    @usableFromInline
    internal mutating func withUnsafeMutableBufferPointer<R>(
        _ body: (UnsafeMutableBufferPointer<Element>) throws -> R
    ) rethrows -> R {
        var array = [Element]()
        array.reserveCapacity(count)
        for i in 0..<count {
            array.append(self[i])
        }
        
        let result = try array.withUnsafeMutableBufferPointer { ptr in
            try body(ptr)
        }
        
        // Copy back
        for i in 0..<count {
            self[i] = array[i]
        }
        
        return result
    }
}

/// Heap-allocated but cache-line aware storage
@usableFromInline
internal final class CompactBuffer<Element> {
    /// Underlying storage using ManagedBuffer
    @usableFromInline
    internal var buffer: ManagedBuffer<CompactBufferHeader, Element>
    
    /// Number of elements
    @usableFromInline
    internal var count: Int { buffer.header.count }
    
    /// Capacity
    @usableFromInline
    internal var capacity: Int { buffer.header.capacity }
    
    /// Initialize with capacity
    @usableFromInline
    internal init(capacity: Int) {
        let actualCapacity = CompactBuffer.alignedCapacity(for: capacity)
        self.buffer = ManagedBuffer<CompactBufferHeader, Element>.create(
            minimumCapacity: actualCapacity
        ) { buffer in
            CompactBufferHeader(count: 0, capacity: actualCapacity)
        }
    }
    
    /// Initialize from array
    @usableFromInline
    internal init(_ elements: [Element]) {
        let actualCapacity = CompactBuffer.alignedCapacity(for: elements.count)
        self.buffer = ManagedBuffer<CompactBufferHeader, Element>.create(
            minimumCapacity: actualCapacity
        ) { buffer in
            CompactBufferHeader(count: elements.count, capacity: actualCapacity)
        }
        
        // Copy elements
        buffer.withUnsafeMutablePointerToElements { ptr in
            ptr.initialize(from: elements, count: elements.count)
        }
    }
    
    /// Calculate cache-line aligned capacity
    @usableFromInline
    internal static func alignedCapacity(for requested: Int) -> Int {
        // Round up to cache line boundary (64 bytes)
        let cacheLineSize = 64
        let elementSize = MemoryLayout<Element>.stride
        let elementsPerCacheLine = cacheLineSize / elementSize
        
        guard elementsPerCacheLine > 0 else { return requested }
        
        return ((requested + elementsPerCacheLine - 1) / elementsPerCacheLine) * elementsPerCacheLine
    }
    
    /// Element access
    @usableFromInline
    internal subscript(index: Int) -> Element {
        get {
            buffer.withUnsafeMutablePointerToElements { ptr in
                ptr[index]
            }
        }
        set {
            buffer.withUnsafeMutablePointerToElements { ptr in
                ptr[index] = newValue
            }
        }
    }
    
    /// Check if uniquely referenced (for COW)
    @usableFromInline
    internal func isUniquelyReferenced() -> Bool {
        isKnownUniquelyReferenced(&buffer)
    }
    
    /// Create a copy
    @usableFromInline
    internal func copy() -> CompactBuffer {
        let newBuffer = CompactBuffer(capacity: capacity)
        newBuffer.buffer.header.count = count
        
        buffer.withUnsafeMutablePointerToElements { src in
            newBuffer.buffer.withUnsafeMutablePointerToElements { dst in
                dst.initialize(from: src, count: count)
            }
        }
        
        return newBuffer
    }
    
    /// Buffer pointer access
    @usableFromInline
    internal func withUnsafeBufferPointer<R>(
        _ body: (UnsafeBufferPointer<Element>) throws -> R
    ) rethrows -> R {
        try buffer.withUnsafeMutablePointerToElements { ptr in
            try body(UnsafeBufferPointer(start: ptr, count: count))
        }
    }
    
    /// Mutable buffer pointer access
    @usableFromInline
    internal func withUnsafeMutableBufferPointer<R>(
        _ body: (UnsafeMutableBufferPointer<Element>) throws -> R
    ) rethrows -> R {
        try buffer.withUnsafeMutablePointerToElements { ptr in
            try body(UnsafeMutableBufferPointer(start: ptr, count: count))
        }
    }
}

/// Header for CompactBuffer
@usableFromInline
internal struct CompactBufferHeader {
    var count: Int
    let capacity: Int
}

/// SIMD-aligned memory buffer for large vectors
@usableFromInline
internal final class AlignedBuffer<Element> {
    /// Aligned pointer to data
    @usableFromInline
    internal let ptr: UnsafeMutablePointer<Element>
    
    /// Number of elements
    @usableFromInline
    internal private(set) var count: Int
    
    /// Capacity
    @usableFromInline
    internal let capacity: Int
    
    /// Alignment used
    @usableFromInline
    internal let alignment: Int = 64  // Cache line size
    
    /// Initialize with capacity
    @usableFromInline
    internal init(capacity: Int) {
        self.capacity = capacity
        self.count = 0
        
        // Allocate aligned memory
        var rawPtr: UnsafeMutableRawPointer?
        let byteCount = capacity * MemoryLayout<Element>.stride
        let result = posix_memalign(&rawPtr, alignment, byteCount)
        
        guard result == 0, let rawPtr = rawPtr else {
            fatalError("Failed to allocate aligned memory")
        }
        
        self.ptr = rawPtr.bindMemory(to: Element.self, capacity: capacity)
    }
    
    /// Initialize from array
    @usableFromInline
    internal init(_ elements: [Element]) {
        self.capacity = elements.count
        self.count = elements.count
        
        // Allocate aligned memory
        var rawPtr: UnsafeMutableRawPointer?
        let byteCount = capacity * MemoryLayout<Element>.stride
        let result = posix_memalign(&rawPtr, alignment, byteCount)
        
        guard result == 0, let rawPtr = rawPtr else {
            fatalError("Failed to allocate aligned memory")
        }
        
        self.ptr = rawPtr.bindMemory(to: Element.self, capacity: capacity)
        ptr.initialize(from: elements, count: count)
    }
    
    deinit {
        ptr.deinitialize(count: count)
        ptr.deallocate()
    }
    
    /// Element access
    @usableFromInline
    internal subscript(index: Int) -> Element {
        get { ptr[index] }
        set { ptr[index] = newValue }
    }
    
    /// Check if aligned
    @usableFromInline
    internal var isAligned: Bool {
        Int(bitPattern: ptr) % alignment == 0
    }
    
    /// Check if uniquely referenced (for COW)
    @usableFromInline
    internal func isUniquelyReferenced() -> Bool {
        // AlignedBuffer is a class, so we can use isKnownUniquelyReferenced
        var mutableSelf = self
        return isKnownUniquelyReferenced(&mutableSelf)
    }
    
    /// Create a copy
    @usableFromInline
    internal func copy() -> AlignedBuffer {
        let newBuffer = AlignedBuffer(capacity: capacity)
        newBuffer.count = count
        newBuffer.ptr.initialize(from: ptr, count: count)
        return newBuffer
    }
    
    /// Buffer pointer access
    @usableFromInline
    internal func withUnsafeBufferPointer<R>(
        _ body: (UnsafeBufferPointer<Element>) throws -> R
    ) rethrows -> R {
        try body(UnsafeBufferPointer(start: ptr, count: count))
    }
    
    /// Mutable buffer pointer access
    @usableFromInline
    internal func withUnsafeMutableBufferPointer<R>(
        _ body: (UnsafeMutableBufferPointer<Element>) throws -> R
    ) rethrows -> R {
        try body(UnsafeMutableBufferPointer(start: ptr, count: count))
    }
}

// MARK: - Collection Conformance

extension TieredStorage: Sequence {
    public func makeIterator() -> IndexingIterator<TieredStorage> {
        IndexingIterator(_elements: self)
    }
}

extension TieredStorage: Collection {
    public var startIndex: Int { 0 }
    public var endIndex: Int { count }
    
    public func index(after i: Int) -> Int {
        i + 1
    }
}