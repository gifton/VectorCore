import Foundation
public protocol StaticDimension { static var value: Int { get } }
public enum VectorError: Error { case dimensionMismatch(expected: Int, actual: Int) }
public enum ScalarOps<T: BinaryFloatingPoint> {
 public static func dot(_ a: UnsafePointer<T>, _ b: UnsafePointer<T>, count: Int) -> T { (0..<count).reduce(0) { $0 + a[$1] * b[$1] } }
 public static func fill(value: T, destination: UnsafeMutablePointer<T>, count: Int) { for i in 0..<count { destination[i] = value } }
 public static func addScalar(_ a: UnsafePointer<T>, scalar: T, result: UnsafeMutablePointer<T>, count: Int) { for i in 0..<count { result[i] = a[i] + scalar } }
 public static func multiplyScalar(_ a: UnsafePointer<T>, scalar: T, result: UnsafeMutablePointer<T>, count: Int) { for i in 0..<count { result[i] = a[i] * scalar } }
 public static func copy(source: UnsafePointer<T>, destination: UnsafeMutablePointer<T>, count: Int) { destination.update(from: source, count: count) }
}
public enum SIMDOperations { public typealias FloatProvider = ScalarOps<Float>; public typealias DoubleProvider = ScalarOps<Double> }
public enum Counter { nonisolated(unsafe) static var allocations = 0; nonisolated(unsafe) static var copies = 0 }

// VectorCore: Vector Storage Protocol
//
// Defines the storage abstraction for vectors
//

import Foundation

/// Protocol defining the storage interface for vectors
public protocol VectorStorage: Sendable {
    associatedtype Scalar: BinaryFloatingPoint

    /// Initialize with zeros
    init()

    /// Initialize with a repeating value
    init(repeating value: Scalar)

    /// Initialize from an array of values
    init(from values: [Scalar])

    /// Access elements by index
    subscript(index: Int) -> Scalar { get set }

    /// Number of elements in the storage
    var count: Int { get }

    /// Access the underlying buffer for optimized operations
    func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Scalar>) throws -> R) rethrows -> R

    /// Mutate the underlying buffer
    mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Scalar>) throws -> R) rethrows -> R
}

/// Protocol for storage types that support optimized operations
public protocol VectorStorageOperations: VectorStorage {
    /// Compute dot product with another storage
    func dotProduct(_ other: Self) -> Scalar
}

// MARK: - Storage Operations Extension

extension VectorStorageOperations where Scalar == Float {
    /// Default dot product implementation using SwiftSIMDProvider
    public func dotProduct(_ other: Self) -> Float {
        var result: Float = 0
        self.withUnsafeBufferPointer { aBuffer in
            other.withUnsafeBufferPointer { bBuffer in
                result = SIMDOperations.FloatProvider.dot(
                    aBuffer.baseAddress!,
                    bBuffer.baseAddress!,
                    count: aBuffer.count
                )
            }
        }
        return result
    }
}

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
internal struct MoveOnlyStorage<Element: BinaryFloatingPoint & SIMDScalar>: ~Copyable {
    private let pointer: UnsafeMutableRawPointer
    public let capacity: Int
    public let alignment: Int

    public init(capacity: Int, alignment: Int = 64) {
        Counter.allocations += 1
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
        Counter.allocations += 1
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
        Counter.copies += 1
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
@usableFromInline
internal struct ManagedStorage<Element: BinaryFloatingPoint & SIMDScalar>: @unchecked Sendable {
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
    @usableFromInline
    internal static func == (lhs: ManagedStorage, rhs: ManagedStorage) -> Bool {
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
@usableFromInline
internal struct StorageSlice<Element: BinaryFloatingPoint & SIMDScalar> {
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

//
//  DimensionStorage.swift
//  VectorCore
//
//

import Foundation

// MARK: - Dimension Storage

/// Storage optimization mode
public enum StorageMode {
    /// Use hybrid storage with small vector optimization
    case hybrid
    /// Always use heap storage
    case heap
}

/// Generic storage that knows its dimension at compile time
public struct DimensionStorage<D: StaticDimension, Element: BinaryFloatingPoint & SIMDScalar> {
    /// Internal storage - either hybrid or managed
    private var storage: Storage

    /// Storage implementation
    private enum Storage {
        case hybrid(HybridStorage<Element>)
        case managed(ManagedStorage<Element>)
    }

    /// The dimension type
    public typealias Dimension = D

    /// Default storage mode based on dimension
    private static var defaultMode: StorageMode {
        // Use hybrid for small dimensions, heap for large
        D.value <= 16 ? .hybrid : .heap
    }

    /// Initialize with zeros
    public init(mode: StorageMode? = nil) {
        let actualMode = mode ?? Self.defaultMode
        switch actualMode {
        case .hybrid:
            self.storage = .hybrid(HybridStorage(capacity: D.value))
        case .heap:
            self.storage = .managed(ManagedStorage(capacity: D.value))
        }
    }

    /// Initialize with a fill value
    public init(repeating value: Element, mode: StorageMode? = nil) {
        self.init(mode: mode)
        // Fill manually since fill method is type-specific
        self.withUnsafeMutableBufferPointer { buffer in
            for i in 0..<D.value {
                buffer[i] = value
            }
        }
    }

    /// Initialize from an array
    public init(from array: [Element], mode: StorageMode? = nil) throws {
        guard array.count == D.value else {
            throw VectorError.dimensionMismatch(expected: D.value, actual: array.count)
        }

        let actualMode = mode ?? Self.defaultMode
        switch actualMode {
        case .hybrid:
            self.storage = .hybrid(HybridStorage(from: array))
        case .heap:
            self.storage = .managed(ManagedStorage(from: array))
        }
    }

    /// Initialize from a sequence
    public init<S: Sequence>(_ sequence: S, mode: StorageMode? = nil) throws where S.Element == Element {
        let array = Array(sequence)
        try self.init(from: array, mode: mode)
    }

    /// Check if using hybrid storage
    public var isUsingHybridStorage: Bool {
        switch storage {
        case .hybrid: return true
        case .managed: return false
        }
    }
}

// MARK: - Storage Protocol Conformance

extension DimensionStorage: VectorStorage {
    public typealias Scalar = Element

    public var count: Int { D.value }

    // Protocol requirement: init with zeros
    public init() {
        self.init(mode: nil)
    }

    // Protocol requirement: init with repeating value
    public init(repeating value: Element) {
        self.init(repeating: value, mode: nil)
    }

    // Protocol requirement: init from array
    public init(from values: [Element]) {
        try! self.init(from: values, mode: nil)
    }

    public subscript(index: Int) -> Element {
        get {
            switch storage {
            case .hybrid(let hybrid):
                return hybrid[index]
            case .managed(let managed):
                return managed[index]
            }
        }
        set {
            switch storage {
            case .hybrid(var hybrid):
                hybrid[index] = newValue
                storage = .hybrid(hybrid)
            case .managed(var managed):
                managed[index] = newValue
                storage = .managed(managed)
            }
        }
    }

    public func withUnsafeBufferPointer<R>(
        _ body: (UnsafeBufferPointer<Element>) throws -> R
    ) rethrows -> R {
        switch storage {
        case .hybrid(let hybrid):
            return try hybrid.withUnsafeBufferPointer(body)
        case .managed(let managed):
            return try managed.withUnsafeBufferPointer(body)
        }
    }

    public mutating func withUnsafeMutableBufferPointer<R>(
        _ body: (UnsafeMutableBufferPointer<Element>) throws -> R
    ) rethrows -> R {
        switch storage {
        case .hybrid(var hybrid):
            defer { storage = .hybrid(hybrid) }
            return try hybrid.withUnsafeMutableBufferPointer(body)
        case .managed(var managed):
            defer { storage = .managed(managed) }
            return try managed.withUnsafeMutableBufferPointer(body)
        }
    }
}

// MARK: - SIMD-Optimized Operations

extension DimensionStorage where Element == Float {
    /// Fill storage with a value using SIMD
    public mutating func fill(with value: Float) {
        withUnsafeMutableBufferPointer { buffer in
            SIMDOperations.FloatProvider.fill(
                value: value,
                destination: buffer.baseAddress!,
                count: D.value
            )
        }
    }

    /// Add a scalar to all elements
    public mutating func add(scalar: Float) {
        withUnsafeMutableBufferPointer { buffer in
            SIMDOperations.FloatProvider.addScalar(
                buffer.baseAddress!,
                scalar: scalar,
                result: buffer.baseAddress!,
                count: D.value
            )
        }
    }

    /// Multiply all elements by a scalar
    public mutating func multiply(by scalar: Float) {
        withUnsafeMutableBufferPointer { buffer in
            SIMDOperations.FloatProvider.multiplyScalar(
                buffer.baseAddress!,
                scalar: scalar,
                result: buffer.baseAddress!,
                count: D.value
            )
        }
    }

    /// Copy from another storage using SIMD
    public mutating func copyFrom(_ other: DimensionStorage<D, Float>) {
        withUnsafeMutableBufferPointer { dest in
            other.withUnsafeBufferPointer { src in
                SIMDOperations.FloatProvider.copy(
                    source: src.baseAddress!,
                    destination: dest.baseAddress!,
                    count: D.value
                )
            }
        }
    }
}

extension DimensionStorage where Element == Double {
    /// Fill storage with a value using SIMD
    public mutating func fill(with value: Double) {
        withUnsafeMutableBufferPointer { buffer in
            SIMDOperations.DoubleProvider.fill(
                value: value,
                destination: buffer.baseAddress!,
                count: D.value
            )
        }
    }

    /// Add a scalar to all elements
    public mutating func add(scalar: Double) {
        withUnsafeMutableBufferPointer { buffer in
            SIMDOperations.DoubleProvider.addScalar(
                buffer.baseAddress!,
                scalar: scalar,
                result: buffer.baseAddress!,
                count: D.value
            )
        }
    }

    /// Multiply all elements by a scalar
    public mutating func multiply(by scalar: Double) {
        withUnsafeMutableBufferPointer { buffer in
            SIMDOperations.DoubleProvider.multiplyScalar(
                buffer.baseAddress!,
                scalar: scalar,
                result: buffer.baseAddress!,
                count: D.value
            )
        }
    }

    /// Copy from another storage using SIMD
    public mutating func copyFrom(_ other: DimensionStorage<D, Double>) {
        withUnsafeMutableBufferPointer { dest in
            other.withUnsafeBufferPointer { src in
                SIMDOperations.DoubleProvider.copy(
                    source: src.baseAddress!,
                    destination: dest.baseAddress!,
                    count: D.value
                )
            }
        }
    }
}

// MARK: - Copy-on-Write Behavior

extension DimensionStorage {
    /// Check if storage is uniquely referenced
    public var isUniquelyReferenced: Bool {
        mutating get {
            switch storage {
            case .hybrid(var hybrid):
                return hybrid.isUniquelyReferenced
            case .managed(var managed):
                return managed.isUniquelyReferenced
            }
        }
    }

    /// Force a copy of the storage
    public mutating func makeUnique() {
        if !isUniquelyReferenced {
            switch storage {
            case .hybrid(let hybrid):
                var newHybrid = HybridStorage<Element>(capacity: D.value)
                newHybrid.copyFrom(hybrid)
                storage = .hybrid(newHybrid)
            case .managed(let managed):
                var newManaged = ManagedStorage<Element>(capacity: D.value)
                newManaged.copyFrom(managed)
                storage = .managed(newManaged)
            }
        }
    }
}

// MARK: - Slice Operations

extension DimensionStorage {
    /// Create a slice view of the storage
    @usableFromInline
    internal func slice(from start: Int, to end: Int) -> StorageSlice<Element> {
        switch storage {
        case .hybrid, .managed:
            // Both storage types can provide slices through buffer pointers
            precondition(start >= 0 && end <= D.value && start < end, "Invalid slice bounds")
            return withUnsafeBufferPointer { buffer in
                StorageSlice(base: buffer.baseAddress! + start, count: end - start)
            }
        }
    }

    /// Create a slice with a range
    @usableFromInline
    internal func slice(_ range: Range<Int>) -> StorageSlice<Element> {
        slice(from: range.lowerBound, to: range.upperBound)
    }
}

// MARK: - Convenience Initializers

extension DimensionStorage {
    /// Initialize with a generator function
    public init(generator: (Int) throws -> Element) rethrows {
        var array = [Element]()
        array.reserveCapacity(D.value)
        for i in 0..<D.value {
            array.append(try generator(i))
        }
        self.storage = .managed(ManagedStorage(from: array))
    }

    /// Initialize with random values
    public static func random(in range: ClosedRange<Element> = 0...1) -> DimensionStorage where Element == Float {
        DimensionStorage { _ in Float.random(in: range) }
    }

    public static func random(in range: ClosedRange<Element> = 0...1) -> DimensionStorage where Element == Double {
        DimensionStorage { _ in Double.random(in: range) }
    }
}

// MARK: - Equatable & Hashable

extension DimensionStorage: Equatable {
    public static func == (lhs: DimensionStorage, rhs: DimensionStorage) -> Bool {
        lhs.withUnsafeBufferPointer { lhsBuffer in
            rhs.withUnsafeBufferPointer { rhsBuffer in
                for i in 0..<D.value {
                    if lhsBuffer[i] != rhsBuffer[i] {
                        return false
                    }
                }
                return true
            }
        }
    }
}

extension DimensionStorage: Hashable {
    public func hash(into hasher: inout Hasher) {
        hasher.combine(D.value)
        withUnsafeBufferPointer { buffer in
            for element in buffer {
                hasher.combine(element)
            }
        }
    }
}

// MARK: - Sendable Conformance

extension DimensionStorage: @unchecked Sendable {}

// MARK: - VectorStorageOperations Conformance

extension DimensionStorage: VectorStorageOperations {
    public func dotProduct(_ other: DimensionStorage<D, Element>) -> Element {
        var result: Element = 0

        if Element.self == Float.self {
            self.withUnsafeBufferPointer { aBuffer in
                other.withUnsafeBufferPointer { bBuffer in
                    let floatResult = SIMDOperations.FloatProvider.dot(
                        aBuffer.baseAddress!.withMemoryRebound(to: Float.self, capacity: aBuffer.count) { $0 },
                        bBuffer.baseAddress!.withMemoryRebound(to: Float.self, capacity: bBuffer.count) { $0 },
                        count: aBuffer.count
                    )
                    // swiftlint:disable:next force_cast
                    result = floatResult as! Element
                }
            }
        } else if Element.self == Double.self {
            self.withUnsafeBufferPointer { aBuffer in
                other.withUnsafeBufferPointer { bBuffer in
                    let doubleResult = SIMDOperations.DoubleProvider.dot(
                        aBuffer.baseAddress!.withMemoryRebound(to: Double.self, capacity: aBuffer.count) { $0 },
                        bBuffer.baseAddress!.withMemoryRebound(to: Double.self, capacity: bBuffer.count) { $0 },
                        count: aBuffer.count
                    )
                    // swiftlint:disable:next force_cast
                    result = doubleResult as! Element
                }
            }
        } else {
            // Fallback for other types
            self.withUnsafeBufferPointer { aBuffer in
                other.withUnsafeBufferPointer { bBuffer in
                    for i in 0..<aBuffer.count {
                        result += aBuffer[i] * bBuffer[i]
                    }
                }
            }
        }

        return result
    }
}

// MARK: - Debug Description

extension DimensionStorage: CustomDebugStringConvertible {
    public var debugDescription: String {
        withUnsafeBufferPointer { buffer in
            // swiftlint:disable:next force_cast
            let elements = buffer.prefix(10).map { String(format: "%.4f", $0 as! any CVarArg) }
            if D.value > 10 {
                return "DimensionStorage<\(D.self)>[\(elements.joined(separator: ", ")), ... (\(D.value) total)]"
            } else {
                return "DimensionStorage<\(D.self)>[\(elements.joined(separator: ", "))]"
            }
        }
    }
}


struct D512: StaticDimension { static let value = 512 }
@main struct Probe {
 static func main() {
  var managed = ManagedStorage<Float>(capacity: 512)
  Counter.allocations = 0; Counter.copies = 0
  for i in 0..<10 { managed[i] = Float(i) }
  print("managed 10 writes allocations/copies", Counter.allocations, Counter.copies, "unique", managed.isUniquelyReferenced)
  var dimension = DimensionStorage<D512, Float>()
  Counter.allocations = 0; Counter.copies = 0
  for i in 0..<10 { dimension[i] = Float(i) }
  print("dimension 10 writes allocations/copies", Counter.allocations, Counter.copies, "unique", dimension.isUniquelyReferenced)
  var hybrid = HybridStorage<Float>(capacity: 512)
  Counter.allocations = 0; Counter.copies = 0
  for i in 0..<10 { hybrid[i] = Float(i) }
  print("hybrid 10 writes allocations/copies", Counter.allocations, Counter.copies, "unique", hybrid.isUniquelyReferenced)
  Counter.allocations = 0; Counter.copies = 0
  hybrid.fill(with: 42)
  print("hybrid fill512 allocations/copies", Counter.allocations, Counter.copies, "last", hybrid[511])
  Counter.allocations = 0; Counter.copies = 0
  let repeating = HybridStorage<Float>(capacity: 512, repeating: 3)
  print("hybrid repeating512 allocations/copies", Counter.allocations, Counter.copies, "last", repeating[511])
  Counter.allocations = 0; Counter.copies = 0
  dimension.withUnsafeMutableBufferPointer { $0[0] = 42 }
  print("dimension mutable buffer allocations/copies", Counter.allocations, Counter.copies)
 }
}
