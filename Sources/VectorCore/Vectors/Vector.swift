//
//  Vector.swift
//  VectorCore
//
//

import Foundation
import Accelerate

// MARK: - Base Vector Type

/// Compile-time dimensioned vector
public struct Vector<D: StaticDimension>: Sendable {
    public typealias Scalar = Float
    public typealias Storage = DimensionStorage<D, Float>
    
    public var storage: Storage
    
    /// The number of elements in the vector
    @inlinable
    public var scalarCount: Int { D.value }
    
    /// Initialize with storage
    @inlinable
    public init(storage: consuming Storage) throws {
        self.storage = storage
    }
    
    /// Initialize with default zero values
    @inlinable
    public init() {
        self.storage = Storage()
    }
    
    /// Initialize with repeating value
    @inlinable
    public init(repeating value: Scalar = 0) {
        self.storage = Storage(repeating: value)
    }
    
    /// Initialize from array
    @inlinable
    public init(_ elements: [Scalar]) throws {
        guard elements.count == D.value else {
            throw VectorError.dimensionMismatch(expected: D.value, actual: elements.count)
        }
        self.storage = Storage(from: elements)
    }
    
    /// Initialize from sequence
    @inlinable
    public init<S: Sequence>(_ scalars: S) throws where S.Element == Scalar {
        let array = Array(scalars)
        try self.init(array)
    }
    
    /// Initialize with generator function
    @inlinable
    public init(generator: (Int) throws -> Scalar) rethrows {
        self.storage = try Storage(generator: generator)
    }
    
    /// Create a zero vector
    @inlinable
    public static func zeros() -> Vector {
        Vector()
    }
}

// MARK: - Unified VectorProtocol Conformance

extension Vector: VectorProtocol {
    /// Convert to array
    public func toArray() -> [Scalar] {
        Array(self)
    }
}

// MARK: - Additional VectorProtocol Requirements

extension Vector {
    
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
    
    /// Check if all values are finite
    @inlinable
    public var isFinite: Bool {
        withUnsafeBufferPointer { buffer in
            for element in buffer {
                if !element.isFinite {
                    return false
                }
            }
            return true
        }
    }
}

// MARK: - Storage Access

extension Vector {
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

// MARK: - Type-Specific Features

extension Vector {
    /// Static factory for zero vector
    @inlinable
    public static var zero: Self {
        Self(repeating: 0)
    }
    
    /// Static factory for ones vector
    @inlinable
    public static var ones: Self {
        Self(repeating: 1)
    }
    
    /// Generate random vector
    @inlinable
    public static func random(in range: ClosedRange<Scalar> = 0...1) -> Self where Scalar.RawSignificand: FixedWidthInteger {
        Self { _ in Scalar.random(in: range) }
    }
    
    /// Generate random unit vector
    @inlinable
    public static func randomUnit() -> Self where Scalar.RawSignificand: FixedWidthInteger {
        let v = Self.random(in: -1...1)
        return (try? v.normalized().get()) ?? Self.zero
    }
}

// MARK: - Convenience Accessors for Low Dimensions

extension Vector where D == Dim2 {
    /// X component
    @inlinable
    public var x: Scalar {
        get { self[0] }
        set { self[0] = newValue }
    }
    
    /// Y component
    @inlinable
    public var y: Scalar {
        get { self[1] }
        set { self[1] = newValue }
    }
    
    /// Initialize from x, y components
    @inlinable
    public init(x: Scalar, y: Scalar) {
        try! self.init(storage: Storage(from: [x, y]))
    }
}

extension Vector where D == Dim3 {
    /// X component
    @inlinable
    public var x: Scalar {
        get { self[0] }
        set { self[0] = newValue }
    }
    
    /// Y component
    @inlinable
    public var y: Scalar {
        get { self[1] }
        set { self[1] = newValue }
    }
    
    /// Z component
    @inlinable
    public var z: Scalar {
        get { self[2] }
        set { self[2] = newValue }
    }
    
    /// Initialize from x, y, z components
    @inlinable
    public init(x: Scalar, y: Scalar, z: Scalar) {
        try! self.init(storage: Storage(from: [x, y, z]))
    }
    
    /// Cross product (3D only)
    @inlinable
    public func cross(_ other: Vector<D>) -> Vector<D> {
        let ax = self.x, ay = self.y, az = self.z
        let bx = other.x, by = other.y, bz = other.z
        
        return Vector(
            x: ay * bz - az * by,
            y: az * bx - ax * bz,
            z: ax * by - ay * bx
        )
    }
}

extension Vector where D == Dim4 {
    /// X component
    @inlinable
    public var x: Scalar {
        get { self[0] }
        set { self[0] = newValue }
    }
    
    /// Y component
    @inlinable
    public var y: Scalar {
        get { self[1] }
        set { self[1] = newValue }
    }
    
    /// Z component
    @inlinable
    public var z: Scalar {
        get { self[2] }
        set { self[2] = newValue }
    }
    
    /// W component
    @inlinable
    public var w: Scalar {
        get { self[3] }
        set { self[3] = newValue }
    }
    
    /// Initialize from x, y, z, w components
    @inlinable
    public init(x: Scalar, y: Scalar, z: Scalar, w: Scalar) {
        try! self.init(storage: Storage(from: [x, y, z, w]))
    }
}

// MARK: - Collection Support

extension Vector: Collection {
    public typealias Index = Int
    public typealias Element = Scalar
    
    @inlinable
    public var startIndex: Int { 0 }
    
    @inlinable
    public var endIndex: Int { D.value }
    
    @inlinable
    public func index(after i: Int) -> Int {
        i + 1
    }
}

// MARK: - Array Conversion

extension Vector {
    // toArray() is provided by VectorProtocol default implementation
}

// MARK: - Equatable & Hashable

extension Vector: Equatable {
    @inlinable
    public static func == (lhs: Vector, rhs: Vector) -> Bool {
        lhs.storage == rhs.storage
    }
}

extension Vector: Hashable {
    @inlinable
    public func hash(into hasher: inout Hasher) {
        storage.hash(into: &hasher)
    }
}

// MARK: - Debug Support

extension Vector: CustomDebugStringConvertible {
    public var debugDescription: String {
        let elements = self.prefix(10).map { String(format: "%.4f", $0) }
        if D.value > 10 {
            return "Vector<\(String(describing: D.self))>[\(elements.joined(separator: ", ")), ... (\(D.value) total)]"
        } else {
            return "Vector<\(String(describing: D.self))>[\(elements.joined(separator: ", "))]"
        }
    }
}

// MARK: - Type Aliases

/// Common vector types
public typealias Vector2 = Vector<Dim2>
public typealias Vector3 = Vector<Dim3>
public typealias Vector4 = Vector<Dim4>
public typealias Vector8 = Vector<Dim8>
public typealias Vector16 = Vector<Dim16>
public typealias Vector32 = Vector<Dim32>
public typealias Vector64 = Vector<Dim64>
public typealias Vector128 = Vector<Dim128>
public typealias Vector256 = Vector<Dim256>
public typealias Vector512 = Vector<Dim512>
public typealias Vector768 = Vector<Dim768>
public typealias Vector1024 = Vector<Dim1024>
public typealias Vector1536 = Vector<Dim1536>
public typealias Vector2048 = Vector<Dim2048>

// MARK: - VectorType Conformance

extension Vector: VectorType {}