//
//  DynamicVector.swift
//  VectorCore
//
//

import Foundation
import Accelerate

// MARK: - Dynamic Vector Type

/// Runtime-dimensioned vector
public struct DynamicVector: Sendable {
    public typealias Scalar = Float
    public typealias Storage = HybridStorage<Float>
    
    public var storage: Storage
    
    @usableFromInline
    internal let dimension: Int
    
    /// The number of elements in the vector
    @inlinable
    public var scalarCount: Int { dimension }
    
    /// Initialize with dimension and storage
    @inlinable
    public init(dimension: Int, storage: Storage) throws {
        guard storage.count == dimension else {
            throw VectorError.dimensionMismatch(expected: dimension, actual: storage.count)
        }
        self.dimension = dimension
        self.storage = storage
    }
    
    /// Initialize with dimension and default value
    @inlinable
    public init(dimension: Int, repeating value: Scalar = 0) {
        self.dimension = dimension
        self.storage = Storage(capacity: dimension, repeating: value)
    }
    
    /// Initialize from array
    @inlinable
    public init(_ elements: [Scalar]) {
        self.dimension = elements.count
        self.storage = Storage(from: elements)
    }
    
    /// Initialize empty (zero-dimensional) vector
    @inlinable
    public init() {
        self.dimension = 0
        self.storage = Storage(capacity: 0, repeating: 0)
    }
    
    /// Initialize with repeating value (required by VectorType)
    @inlinable
    public init(repeating value: Float) {
        // Default to zero dimensions for dynamic vector
        self.dimension = 0
        self.storage = Storage(capacity: 0, repeating: value)
    }
    
    /// Initialize from sequence
    @inlinable
    public init<S: Sequence>(_ scalars: S) where S.Element == Scalar {
        let array = Array(scalars)
        self.init(array)
    }
    
    /// Initialize with generator function
    @inlinable
    public init(dimension: Int, generator: (Int) throws -> Scalar) rethrows {
        self.dimension = dimension
        var elements = [Scalar]()
        elements.reserveCapacity(dimension)
        for i in 0..<dimension {
            elements.append(try generator(i))
        }
        self.storage = Storage(from: elements)
    }
}

// MARK: - VectorProtocol Conformance

extension DynamicVector: VectorProtocol {
    /// Convert to array for VectorProtocol
    public func toArray() -> [Scalar] {
        storage.withUnsafeBufferPointer { Array($0) }
    }
    /// Initialize from storage (protocol requirement)
    @inlinable
    public init(storage: Storage) throws {
        self.dimension = storage.count
        self.storage = storage
    }
    
    /// Element access
    @inlinable
    public subscript(index: Int) -> Scalar {
        get {
            precondition(index >= 0 && index < scalarCount, "Index \(index) out of bounds [0..<\(scalarCount)]")
            return storage[index]
        }
        set {
            precondition(index >= 0 && index < scalarCount, "Index \(index) out of bounds [0..<\(scalarCount)]")
            storage[index] = newValue
        }
    }
    
}

// MARK: - Storage Access

extension DynamicVector {
    /// Access storage for reading (borrowing for efficiency)
    @inlinable
    public func withUnsafeBufferPointer<R>(
        _ body: (UnsafeBufferPointer<Scalar>) throws -> R
    ) rethrows -> R {
        try storage.withUnsafeBufferPointer(body)
    }
    
    /// Access storage for writing (COW semantics)
    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(
        _ body: (UnsafeMutableBufferPointer<Scalar>) throws -> R
    ) rethrows -> R {
        try storage.withUnsafeMutableBufferPointer(body)
    }
}

// MARK: - Dynamic-Specific Features

extension DynamicVector {
    /// Resize vector to new dimension
    @inlinable
    public func resized(to newDimension: Int, fill: Scalar = 0) -> DynamicVector {
        var newStorage = Storage(capacity: newDimension, repeating: fill)
        let copyCount = Swift.min(dimension, newDimension)
        
        if copyCount > 0 {
            withUnsafeBufferPointer { source in
                newStorage.withUnsafeMutableBufferPointer { dest in
                    dest.baseAddress!.initialize(from: source.baseAddress!, count: copyCount)
                }
            }
        }
        
        return try! DynamicVector(dimension: newDimension, storage: newStorage)
    }
    
    /// Append element (returns new vector)
    @inlinable
    public func appending(_ element: Scalar) -> DynamicVector {
        resized(to: dimension + 1, fill: element)
    }
    
    /// Remove last element (returns new vector)
    @inlinable
    public func dropLast() -> DynamicVector {
        guard dimension > 0 else { return self }
        return resized(to: dimension - 1)
    }
    
    /// Create slice of vector
    @inlinable
    public func slice(from start: Int, to end: Int) -> DynamicVector {
        precondition(start >= 0 && end <= dimension && start < end, "Invalid slice bounds")
        let sliceCount = end - start
        
        return DynamicVector(dimension: sliceCount) { i in
            self[start + i]
        }
    }
}

// MARK: - Factory Methods

extension DynamicVector {
    /// Create zero vector
    @inlinable
    public static func zero(dimension: Int) -> DynamicVector {
        DynamicVector(dimension: dimension, repeating: 0)
    }
    
    /// Create ones vector
    @inlinable
    public static func ones(dimension: Int) -> DynamicVector {
        DynamicVector(dimension: dimension, repeating: 1)
    }
    
    /// Generate random vector
    @inlinable
    public static func random(dimension: Int, in range: ClosedRange<Scalar> = 0...1) -> DynamicVector {
        DynamicVector(dimension: dimension) { _ in Scalar.random(in: range) }
    }
    
    /// Generate random unit vector
    @inlinable
    public static func randomUnit(dimension: Int) -> DynamicVector {
        let v = DynamicVector.random(dimension: dimension, in: -1...1)
        return (try? v.normalized().get()) ?? DynamicVector.zero(dimension: dimension)
    }
}

// MARK: - Collection Support

extension DynamicVector: Collection {
    public typealias Index = Int
    public typealias Element = Scalar
    
    @inlinable
    public var startIndex: Int { 0 }
    
    @inlinable
    public var endIndex: Int { dimension }
    
    @inlinable
    public func index(after i: Int) -> Int {
        i + 1
    }
}

// MARK: - Array Conversion

extension DynamicVector {
    // toArray() is provided by VectorProtocol extension
}

// MARK: - Equatable & Hashable

extension DynamicVector: Equatable {
    @inlinable
    public static func == (lhs: DynamicVector, rhs: DynamicVector) -> Bool {
        guard lhs.dimension == rhs.dimension else { return false }
        return lhs.withUnsafeBufferPointer { lhsBuffer in
            rhs.withUnsafeBufferPointer { rhsBuffer in
                for i in 0..<lhs.dimension {
                    if lhsBuffer[i] != rhsBuffer[i] {
                        return false
                    }
                }
                return true
            }
        }
    }
}

extension DynamicVector: Hashable {
    @inlinable
    public func hash(into hasher: inout Hasher) {
        hasher.combine(dimension)
        withUnsafeBufferPointer { buffer in
            for element in buffer {
                hasher.combine(element)
            }
        }
    }
}

// MARK: - Debug Support

extension DynamicVector: CustomDebugStringConvertible {
    public var debugDescription: String {
        let elements = self.prefix(10).map { String(format: "%.4f", $0) }
        if dimension > 10 {
            return "DynamicVector(dim: \(dimension))[\(elements.joined(separator: ", ")), ... (\(dimension) total)]"
        } else {
            return "DynamicVector(dim: \(dimension))[\(elements.joined(separator: ", "))]"
        }
    }
}

// MARK: - VectorType Conformance

extension DynamicVector: VectorType {}
