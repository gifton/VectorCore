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
    public func slice(from start: Int, to end: Int) -> StorageSlice<Element> {
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
    public func slice(_ range: Range<Int>) -> StorageSlice<Element> {
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
            let elements = buffer.prefix(10).map { String(format: "%.4f", $0 as! any CVarArg) }
            if D.value > 10 {
                return "DimensionStorage<\(D.self)>[\(elements.joined(separator: ", ")), ... (\(D.value) total)]"
            } else {
                return "DimensionStorage<\(D.self)>[\(elements.joined(separator: ", "))]"
            }
        }
    }
}