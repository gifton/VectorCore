import Foundation
public enum VectorError: Error { case allocationFailed(size: Int, reason: String) }

// VectorCore: Aligned Memory Utilities
//
// Common utilities for memory alignment across storage types
//

import Foundation

/// Common alignment utilities for VectorCore storage types
public enum AlignedMemory {
    /// Platform-specific optimal alignment for SIMD operations
    public static var optimalAlignment: Int {
        #if arch(arm64)
        // Apple Silicon prefers 64-byte alignment (cache line size)
        return 64
        #elseif arch(x86_64)
        // Intel x86_64 typically uses 32-byte alignment for AVX
        // but 64-byte is still good for cache lines
        return 64
        #else
        // Conservative default
        return 16
        #endif
    }

    /// Minimum alignment required for SIMD operations
    public static let minimumAlignment: Int = 16

    /// Check if a pointer is properly aligned
    @inlinable
    public static func isAligned<T>(_ pointer: UnsafePointer<T>, to alignment: Int) -> Bool {
        Int(bitPattern: pointer) % alignment == 0
    }

    /// Check if a mutable pointer is properly aligned
    @inlinable
    public static func isAligned<T>(_ pointer: UnsafeMutablePointer<T>, to alignment: Int) -> Bool {
        Int(bitPattern: pointer) % alignment == 0
    }

    /// Allocate aligned memory for Float arrays
    /// - Throws: VectorError.allocationFailed if allocation fails
    @inlinable
    public static func allocateAligned(
        count: Int,
        alignment: Int = optimalAlignment
    ) throws -> UnsafeMutablePointer<Float> {
        try allocateAligned(type: Float.self, count: count, alignment: alignment)
    }

    /// Allocate aligned memory for any type
    /// - Throws: VectorError.allocationFailed if allocation fails
    @inlinable
    public static func allocateAligned<T>(
        type: T.Type,
        count: Int,
        alignment: Int = optimalAlignment
    ) throws -> UnsafeMutablePointer<T> {
        precondition(alignment > 0 && (alignment & (alignment - 1)) == 0,
                     "Alignment must be a power of 2")
        precondition(alignment >= minimumAlignment,
                     "Alignment must be at least \(minimumAlignment) bytes")

        var rawPtr: UnsafeMutableRawPointer?
        let byteCount = count * MemoryLayout<T>.stride
        let result = posix_memalign(&rawPtr, alignment, byteCount)

        guard result == 0, let ptr = rawPtr else {
            throw VectorError.allocationFailed(size: byteCount, reason: "posix_memalign error \(result)")
        }

        return ptr.assumingMemoryBound(to: T.self)
    }

    /// Deallocate memory previously allocated via posix_memalign or aligned_alloc.
    ///
    /// - Parameter ptr: Pointer returned by allocateAligned or posix_memalign
    ///
    /// - Important: Memory allocated with `posix_memalign` MUST be freed with `free()`,
    ///   not with Swift's `.deallocate()`. Using `.deallocate()` on memory allocated
    ///   by `posix_memalign` causes undefined behavior and heap corruption.
    @inlinable
    public static func deallocate<T>(_ ptr: UnsafeMutablePointer<T>) {
        free(UnsafeMutableRawPointer(ptr))
    }

    /// Deallocate raw memory previously allocated via posix_memalign or aligned_alloc.
    ///
    /// - Parameter ptr: Raw pointer returned by allocateAligned or posix_memalign
    ///
    /// - Important: Memory allocated with `posix_memalign` MUST be freed with `free()`,
    ///   not with Swift's `.deallocate()`.
    @inlinable
    public static func deallocate(_ ptr: UnsafeMutableRawPointer) {
        free(ptr)
    }
}

/// Protocol for storage types that guarantee memory alignment
public protocol AlignedStorage {
    /// The alignment guarantee provided by this storage
    var guaranteedAlignment: Int { get }

    /// Verify that the storage maintains its alignment guarantee
    func verifyAlignment() -> Bool
}


//
//  TieredStorage.swift
//  VectorCore
//
//  Adaptive storage that automatically selects optimal backing store based on size and access patterns
//

import Foundation

/// Adaptive storage container that automatically selects the optimal backing store
/// based on element count and access patterns, enabling efficient memory usage for different vector sizes.
@usableFromInline
internal struct TieredStorage<Element> {
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
    @usableFromInline
    internal init(capacity: Int, hint: PerformanceHint = .sequential) {
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
    @usableFromInline
    internal init<S: Sequence>(_ elements: S) where S.Element == Element {
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
    @usableFromInline
    internal subscript(index: Int) -> Element {
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
    @usableFromInline
    internal func withUnsafeBufferPointer<R>(
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
    @usableFromInline
    internal mutating func withUnsafeMutableBufferPointer<R>(
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
        ) { _ in
            CompactBufferHeader(count: 0, capacity: actualCapacity)
        }
    }

    /// Initialize from array
    @usableFromInline
    internal init(_ elements: [Element]) {
        let actualCapacity = CompactBuffer.alignedCapacity(for: elements.count)
        self.buffer = ManagedBuffer<CompactBufferHeader, Element>.create(
            minimumCapacity: actualCapacity
        ) { _ in
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

    /// Tracks whether allocation used posix_memalign (via AlignedMemory)
    @usableFromInline
    internal let usedPosixAlignedAlloc: Bool

    /// Initialize with capacity
    @usableFromInline
    internal init(capacity: Int) {
        self.capacity = capacity
        self.count = 0

        // Allocate aligned memory (fallback to natural alignment on failure)
        do {
            let typed = try AlignedMemory.allocateAligned(type: Element.self, count: capacity, alignment: alignment)
            self.ptr = typed
            self.usedPosixAlignedAlloc = true
        } catch {
            // Fallback: allocate with natural alignment to avoid hard crash
            let raw = UnsafeMutableRawPointer.allocate(byteCount: capacity * MemoryLayout<Element>.stride,
                                                       alignment: MemoryLayout<Element>.alignment)
            self.ptr = raw.bindMemory(to: Element.self, capacity: capacity)
            self.usedPosixAlignedAlloc = false
        }
    }

    /// Initialize from array
    @usableFromInline
    internal init(_ elements: [Element]) {
        self.capacity = elements.count
        self.count = elements.count

        // Allocate aligned memory (fallback to natural alignment on failure)
        do {
            let typed = try AlignedMemory.allocateAligned(type: Element.self, count: capacity, alignment: alignment)
            self.ptr = typed
            self.usedPosixAlignedAlloc = true
        } catch {
            let raw = UnsafeMutableRawPointer.allocate(byteCount: capacity * MemoryLayout<Element>.stride,
                                                       alignment: MemoryLayout<Element>.alignment)
            self.ptr = raw.bindMemory(to: Element.self, capacity: capacity)
            self.usedPosixAlignedAlloc = false
        }
        ptr.initialize(from: elements, count: count)
    }

    deinit {
        ptr.deinitialize(count: count)
        if usedPosixAlignedAlloc {
            AlignedMemory.deallocate(ptr)
        } else {
            ptr.deallocate()
        }
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
        Counter.alignedCopies += 1
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

// VectorCore: Triangular Matrix Storage
//
// Efficient storage for symmetric matrices (like distance matrices)
// storing only the upper triangle to save ~50% memory
//

import Foundation

/// Memory-efficient storage for symmetric matrices
///
/// `TriangularMatrix` stores only the upper triangle of a symmetric matrix,
/// reducing memory usage by approximately 50%. Perfect for distance matrices
/// where d(i,j) = d(j,i) and diagonal elements are zero.
///
/// ## Memory Savings
/// - Full matrix: n² elements
/// - Triangular: n*(n-1)/2 elements
/// - Savings: ~50% for large n
///
/// ## Example Usage
/// ```swift
/// var distances = TriangularMatrix<Float>(size: 1000)
/// distances[10, 20] = 5.0  // Automatically handles symmetry
/// let d1 = distances[10, 20]  // Returns 5.0
/// let d2 = distances[20, 10]  // Also returns 5.0
/// ```
public struct TriangularMatrix<T> {
    /// Internal storage for upper triangle elements
    @usableFromInline
    internal var storage: TieredStorage<T>

    /// Matrix dimension (n×n matrix)
    public let size: Int

    /// Number of elements actually stored
    public var elementCount: Int {
        size * (size - 1) / 2
    }

    /// Initialize with default value
    ///
    /// - Parameters:
    ///   - size: Matrix dimension
    ///   - defaultValue: Initial value for all elements
    public init(size: Int, defaultValue: T) {
        precondition(size >= 0, "Size must be non-negative")
        self.size = size
        let count = size * (size - 1) / 2

        // Use TieredStorage for automatic optimization
        let elements = Array(repeating: defaultValue, count: count)
        self.storage = TieredStorage(elements)
    }

    /// Initialize with zero (for numeric types)
    public init(size: Int) where T: Numeric {
        self.init(size: size, defaultValue: .zero)
    }

    /// Convert indices to storage position
    @inlinable
    internal func storageIndex(row: Int, col: Int) -> Int {
        // Ensure we're accessing upper triangle
        let (i, j) = row < col ? (row, col) : (col, row)
        // Formula for upper triangular: i * size - i * (i + 1) / 2 + j - i - 1
        return i * size - i * (i + 1) / 2 + j - i - 1
    }
}

// MARK: - Subscript Access

extension TriangularMatrix {
    /// Access matrix elements
    ///
    /// Automatically handles symmetry: m[i,j] == m[j,i]
    /// Returns zero for diagonal elements when T is Numeric
    @inlinable
    public subscript(row: Int, col: Int) -> T where T: Numeric {
        get {
            precondition(row >= 0 && row < size, "Row index out of bounds")
            precondition(col >= 0 && col < size, "Column index out of bounds")

            // Diagonal is always zero for distance matrices
            guard row != col else { return .zero }

            let index = storageIndex(row: row, col: col)
            return storage[index]
        }
        set {
            precondition(row >= 0 && row < size, "Row index out of bounds")
            precondition(col >= 0 && col < size, "Column index out of bounds")
            precondition(row != col, "Cannot set diagonal elements")

            let index = storageIndex(row: row, col: col)
            storage[index] = newValue
        }
    }

    /// Access matrix elements (non-numeric version)
    @inlinable
    public subscript(row: Int, col: Int) -> T? {
        get {
            precondition(row >= 0 && row < size, "Row index out of bounds")
            precondition(col >= 0 && col < size, "Column index out of bounds")

            // Diagonal returns nil for non-numeric types
            guard row != col else { return nil }

            let index = storageIndex(row: row, col: col)
            return storage[index]
        }
        set {
            precondition(row >= 0 && row < size, "Row index out of bounds")
            precondition(col >= 0 && col < size, "Column index out of bounds")

            if row != col, let value = newValue {
                let index = storageIndex(row: row, col: col)
                storage[index] = value
            }
        }
    }
}

// MARK: - Safe Access

extension TriangularMatrix {
    /// Safely access elements
    ///
    /// - Parameters:
    ///   - row: Row index
    ///   - col: Column index
    /// - Returns: Element value or nil if indices are out of bounds
    @inlinable
    public func at(_ row: Int, _ col: Int) -> T? where T: Numeric {
        guard row >= 0 && row < size else { return nil }
        guard col >= 0 && col < size else { return nil }
        guard row != col else { return .zero }

        let index = storageIndex(row: row, col: col)
        return storage[index]
    }

    /// Safely set element
    ///
    /// - Parameters:
    ///   - row: Row index
    ///   - col: Column index
    ///   - value: Value to set
    /// - Returns: true if successful, false if indices invalid
    @inlinable
    public mutating func setAt(_ row: Int, _ col: Int, to value: T) -> Bool {
        guard row >= 0 && row < size else { return false }
        guard col >= 0 && col < size else { return false }
        guard row != col else { return false }

        let index = storageIndex(row: row, col: col)
        storage[index] = value
        return true
    }
}

// MARK: - Iteration

extension TriangularMatrix: Sequence {
    /// Element iterator that yields (row, col, value) tuples
    public struct Iterator: IteratorProtocol {
        private let matrix: TriangularMatrix<T>
        private var row = 0
        private var col = 1

        init(_ matrix: TriangularMatrix<T>) {
            self.matrix = matrix
        }

        public mutating func next() -> (row: Int, col: Int, value: T)? {
            guard row < matrix.size - 1 else { return nil }

            let value = matrix.storage[matrix.storageIndex(row: row, col: col)]
            let result = (row, col, value)

            // Move to next position
            col += 1
            if col >= matrix.size {
                row += 1
                col = row + 1
            }

            return result
        }
    }

    public func makeIterator() -> Iterator {
        Iterator(self)
    }
}

// MARK: - Conversion

extension TriangularMatrix {
    /// Convert to full matrix representation
    ///
    /// - Returns: Full n×n matrix with symmetric values
    public func toFullMatrix() -> [[T]] where T: Numeric {
        var result = Array(repeating: Array(repeating: T.zero, count: size), count: size)

        for (i, j, value) in self {
            result[i][j] = value
            result[j][i] = value  // Symmetry
        }

        return result
    }

    /// Create from full matrix (takes upper triangle)
    ///
    /// - Parameter matrix: Full matrix (must be square)
    /// - Returns: Triangular matrix with upper triangle values
    public static func fromFullMatrix(_ matrix: [[T]]) -> TriangularMatrix<T>? {
        guard !matrix.isEmpty else { return nil }
        let size = matrix.count
        guard matrix.allSatisfy({ $0.count == size }) else { return nil }

        var result = TriangularMatrix(size: size, defaultValue: matrix[0][1])

        for i in 0..<size {
            for j in (i+1)..<size {
                result[i, j] = matrix[i][j]
            }
        }

        return result
    }
}

// MARK: - Parallel Operations

extension TriangularMatrix where T: Sendable {
    /// Fill matrix using parallel computation
    ///
    /// - Parameter compute: Function to compute element at (i,j)
    public static func buildParallel(
        size: Int,
        compute: @Sendable @escaping (Int, Int) async -> T
    ) async -> TriangularMatrix<T> where T: Numeric {
        var result = TriangularMatrix<T>(size: size)

        await withTaskGroup(of: (Int, Int, T).self) { group in
            // Add tasks for each element
            for i in 0..<size {
                for j in (i+1)..<size {
                    group.addTask {
                        let value = await compute(i, j)
                        return (i, j, value)
                    }
                }
            }

            // Collect results
            for await (i, j, value) in group {
                result[i, j] = value
            }
        }

        return result
    }
}

// VectorCore: Memory Utilities
//
// Memory management and optimization utilities

import Foundation

// MARK: - Memory Management Namespace

/// Namespace for memory management types and utilities
internal enum Memory {
    // MARK: - Pressure Monitor

    /// Monitor system memory pressure
    public final class PressureMonitor {
        private var source: (any DispatchSourceMemoryPressure)?

        public enum Level {
            case normal
            case warning
            case critical
        }

        public init() {}

        public func startMonitoring(handler: @escaping (Level) -> Void) {
            let source = DispatchSource.makeMemoryPressureSource(eventMask: .all)

            source.setEventHandler {
                let event = source.data

                if event.contains(.critical) {
                    handler(.critical)
                } else if event.contains(.warning) {
                    handler(.warning)
                } else {
                    handler(.normal)
                }
            }

            self.source = source
            source.resume()
        }

        public func stopMonitoring() {
            source?.cancel()
            source = nil
        }

        deinit {
            stopMonitoring()
        }
    }
}

// MARK: - Memory Utilities

/// Get memory size of a SIMD vector type.
///
/// - Parameter type: The SIMD vector type to query
/// - Returns: Size in bytes required to store the vector
///
/// ## Example
/// ```swift
/// let size = vectorMemorySize(SIMD16<Float>.self)  // Returns 64 bytes
/// ```
internal func vectorMemorySize<Vector: SIMD>(_ type: Vector.Type) -> Int where Vector.Scalar: BinaryFloatingPoint {
    return MemoryLayout<Vector>.size
}

/// Get memory alignment requirement of a SIMD vector type.
///
/// - Parameter type: The SIMD vector type to query
/// - Returns: Required alignment in bytes for optimal performance
///
/// ## Example
/// ```swift
/// let alignment = vectorMemoryAlignment(SIMD8<Float>.self)  // Returns 32 bytes
/// ```
internal func vectorMemoryAlignment<Vector: SIMD>(_ type: Vector.Type) -> Int where Vector.Scalar: BinaryFloatingPoint {
    return MemoryLayout<Vector>.alignment
}

/// Check if a pointer is properly aligned to a specified boundary.
///
/// Proper alignment is crucial for SIMD operations and can significantly
/// impact performance. Misaligned access may cause crashes on some architectures.
///
/// - Parameters:
///   - pointer: The pointer to check
///   - alignment: Required alignment boundary in bytes
/// - Returns: true if pointer address is a multiple of alignment
///
/// ## Example
/// ```swift
/// let buffer = UnsafeMutablePointer<Float>.allocate(capacity: 256)
/// if isAligned(buffer, to: 32) {
///     // Safe to use with 32-byte aligned SIMD operations
/// }
/// ```
public func isAligned<T>(_ pointer: UnsafePointer<T>, to alignment: Int) -> Bool {
    return Int(bitPattern: pointer) % alignment == 0
}

/// Copy memory with alignment optimization.
///
/// This function checks alignment and uses optimized copy operations
/// when both source and destination meet alignment requirements.
///
/// - Parameters:
///   - source: Source memory pointer
///   - destination: Destination memory pointer
///   - count: Number of elements to copy
///   - preferredAlignment: Preferred alignment for optimization (default: 64)
///
/// ## Performance Notes
/// - Aligned copy uses bulk memory operations
/// - Unaligned copy falls back to element-wise operations
/// - 64-byte alignment optimal for cache lines
///
/// ## Example
/// ```swift
/// alignedCopy(
///     from: sourceBuffer,
///     to: destBuffer,
///     count: 1024,
///     preferredAlignment: 64
/// )
/// ```
public func alignedCopy<T>(
    from source: UnsafePointer<T>,
    to destination: UnsafeMutablePointer<T>,
    count: Int,
    preferredAlignment: Int = 64
) {
    if isAligned(source, to: preferredAlignment) && isAligned(destination, to: preferredAlignment) {
        // Use optimized copy for aligned memory
        destination.initialize(from: source, count: count)
    } else {
        // Fallback to element-wise copy
        for i in 0..<count {
            destination[i] = source[i]
        }
    }
}

// MARK: - Buffer Utilities

/// Create a temporary buffer with automatic cleanup.
///
/// This function allocates an aligned temporary buffer, executes the provided
/// closure with the buffer, and automatically deallocates the memory
/// regardless of how the closure exits (normally or via throw).
///
/// - Parameters:
///   - type: The element type for the buffer
///   - capacity: Number of elements to allocate
///   - alignment: Memory alignment in bytes (default: 64 for cache lines)
///   - body: Closure to execute with the allocated buffer
/// - Returns: Result returned by the closure
/// - Throws: Rethrows any error thrown by the closure
///
/// ## Example
/// ```swift
/// let result = withTemporaryBuffer(of: Float.self, capacity: 1024) { buffer in
///     // Use buffer safely, automatic cleanup guaranteed
///     vDSP_vfill([1.0], buffer.baseAddress!, 1, vDSP_Length(1024))
///     return buffer.reduce(0, +)
/// }
/// ```
internal func withTemporaryBuffer<T, Result>(
    of type: T.Type,
    capacity: Int,
    alignment: Int = 64,
    _ body: (UnsafeMutableBufferPointer<T>) throws -> Result
) rethrows -> Result {
    let rawPointer = UnsafeMutableRawPointer.allocate(
        byteCount: capacity * MemoryLayout<T>.stride,
        alignment: alignment
    )
    defer { rawPointer.deallocate() }

    let pointer = rawPointer.bindMemory(to: T.self, capacity: capacity)
    let buffer = UnsafeMutableBufferPointer(start: pointer, count: capacity)

    return try body(buffer)
}

/// Zero out memory securely to prevent data leakage.
///
/// This function uses memset_s to ensure memory is zeroed in a way that
/// prevents compiler optimizations from removing the operation. Essential
/// for handling sensitive data like cryptographic keys.
///
/// - Parameter buffer: Buffer to securely zero
///
/// ## Security Notes
/// - Uses memset_s for guaranteed zeroing
/// - Cannot be optimized away by compiler
/// - Should be called before deallocating sensitive data
///
/// ## Example
/// ```swift
/// var sensitiveData = UnsafeMutableBufferPointer<UInt8>.allocate(capacity: 32)
/// defer {
///     secureZero(sensitiveData)
///     sensitiveData.deallocate()
/// }
/// // Use sensitive data...
/// ```
internal func secureZero<T>(_ buffer: UnsafeMutableBufferPointer<T>) {
    guard let baseAddress = buffer.baseAddress else { return }

    // Use memset_s for secure zeroing when available
    let byteCount = buffer.count * MemoryLayout<T>.stride
    _ = memset_s(baseAddress, byteCount, 0, byteCount)
}

// MARK: - Sendable Buffer Wrapper

/// Thread-safe wrapper for UnsafeMutablePointer to enable Sendable conformance
///
/// This wrapper provides controlled concurrent access to unsafe memory buffers.
/// The buffer is isolated within the wrapper and only accessed through the
/// nonisolated pointer property, making it safe to use across concurrency domains.
///
/// - Warning: Caller is responsible for ensuring thread-safe access patterns
internal final class SendableBufferWrapper<T> {
    private let buffer: UnsafeMutablePointer<T>
    public let capacity: Int

    /// Initialize with allocated capacity
    public init(capacity: Int) {
        self.capacity = capacity
        self.buffer = UnsafeMutablePointer<T>.allocate(capacity: capacity)
    }

    /// Access the underlying pointer from any isolation domain
    public nonisolated var pointer: UnsafeMutablePointer<T> {
        buffer
    }

    /// Clean up allocated memory
    deinit {
        buffer.deallocate()
    }
}

extension SendableBufferWrapper: @unchecked Sendable {}

/// Thread-safe wrapper for UnsafeMutableBufferPointer to enable Sendable conformance
internal final class SendableMutableBufferWrapper<T> {
    private let buffer: UnsafeMutableBufferPointer<T>

    /// Initialize with allocated capacity
    public init(capacity: Int) {
        let ptr = UnsafeMutablePointer<T>.allocate(capacity: capacity)
        self.buffer = UnsafeMutableBufferPointer(start: ptr, count: capacity)
    }

    /// Access the underlying buffer pointer from any isolation domain
    public nonisolated var bufferPointer: UnsafeMutableBufferPointer<T> {
        buffer
    }

    /// Access the base pointer from any isolation domain
    public nonisolated var pointer: UnsafeMutablePointer<T>? {
        buffer.baseAddress
    }

    /// Clean up allocated memory
    deinit {
        buffer.baseAddress?.deallocate()
    }
}

extension SendableMutableBufferWrapper: @unchecked Sendable {}


enum Counter { nonisolated(unsafe) static var alignedCopies = 0 }
final class Token { let id: Int; init(_ id: Int) { self.id = id } }
@main struct Probe {
 static func main() {
  var uniqueMatrix = TriangularMatrix<Float>(size: 65)
  Counter.alignedCopies = 0
  for j in 1...10 { uniqueMatrix[0,j] = Float(j) }
  print("unique size65 matrix 10 writes aligned copies", Counter.alignedCopies, "last", uniqueMatrix[0,10])

  for n in [2,3,4,16,32,33,64,65] {
   let original = TriangularMatrix<Float>(size: n, defaultValue: 1)
   var copy = original
   copy[0,1] = 99
   print("matrix", n, "original", original[0,1], "copy", copy[0,1])
  }
  weak var weakToken: Token?
  do {
   let token = Token(1)
   weakToken = token
   let matrix = TriangularMatrix<Token>(size: 4, defaultValue: token)
   print("reference matrix count", matrix.elementCount)
   withExtendedLifetime(matrix) {}
  }
  print("compact reference leaked", weakToken != nil)
  for offset in [0, 1] {
   let sourceRaw = UnsafeMutableRawPointer.allocate(byteCount: 128, alignment: 64)
   let destRaw = UnsafeMutableRawPointer.allocate(byteCount: 128, alignment: 64)
   let source = sourceRaw.bindMemory(to: Token.self, capacity: 16).advanced(by: offset)
   let dest = destRaw.bindMemory(to: Token.self, capacity: 16).advanced(by: offset)
   source.initialize(to: Token(2))
   dest.initialize(to: Token(3))
   weak var old = dest.pointee
   alignedCopy(from: source, to: dest, count: 1)
   print("alignedCopy offset", offset, "old destination leaked", old != nil)
   source.deinitialize(count: 1)
   dest.deinitialize(count: 1)
   sourceRaw.deallocate()
   destRaw.deallocate()
  }
  for n in [4,5,512,513,2048,2049] {
   var storage = TieredStorage(Array(repeating: Float(1), count: n))
   let original = storage
   storage.withUnsafeMutableBufferPointer { $0[0] = 19 }
   print("tier buffer", n, "original", original[0], "mutated", storage[0])
  }
 }
}
