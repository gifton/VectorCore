// VectorCore: Generic Vector Implementation
//
// Single implementation for all vector dimensions
//

import Foundation
import simd
import Accelerate

/// Generic vector type supporting arbitrary dimensions
public struct Vector<D: Dimension>: Sendable {
    @usableFromInline
    internal var storage: D.Storage
    
    /// The number of elements in the vector
    @inlinable
    public var scalarCount: Int { D.value }
    
    /// Initialize a zero vector
    @inlinable
    public init() {
        self.storage = D.Storage()
    }
    
    /// Initialize with a repeating value
    @inlinable
    public init(repeating value: Float) {
        self.storage = D.Storage(repeating: value)
    }
    
    /// Initialize from an array
    @inlinable
    public init(_ values: [Float]) {
        precondition(values.count == D.value, "Array count must match dimension")
        self.storage = D.Storage(from: values)
    }
    
    /// Initialize from array literal
    @inlinable
    public init(arrayLiteral elements: Float...) {
        self.init(elements)
    }
    
    /// Access elements by index
    @inlinable
    public subscript(index: Int) -> Float {
        get {
            precondition(index >= 0 && index < D.value, "Index out of bounds")
            return storage[index]
        }
        set {
            precondition(index >= 0 && index < D.value, "Index out of bounds")
            storage[index] = newValue
        }
    }
}

// MARK: - BaseVectorProtocol Conformance

extension Vector: BaseVectorProtocol {
    public typealias Scalar = Float
    
    public static var dimensions: Int { D.value }
    
    @inlinable
    public init(from array: [Float]) {
        self.init(array)
    }
    
    @inlinable
    public func toArray() -> [Float] {
        var result = [Float](repeating: 0, count: D.value)
        _ = storage.withUnsafeBufferPointer { buffer in
            result.withUnsafeMutableBufferPointer { dest in
                dest.initialize(from: buffer)
            }
        }
        return result
    }
}

// MARK: - ExtendedVectorProtocol Conformance

extension Vector: ExtendedVectorProtocol where D.Storage: VectorStorageOperations {
    // All required methods are already implemented in Mathematical Operations section
    // - dotProduct
    // - magnitude
    // - normalized
    // - distance
    // - cosineSimilarity
}

// MARK: - Mathematical Operations

extension Vector where D.Storage: VectorStorageOperations {
    /// Compute dot product with another vector
    @inlinable
    public func dotProduct(_ other: Vector<D>) -> Float {
        storage.dotProduct(other.storage)
    }
    
    /// Compute the magnitude (L2 norm)
    @inlinable
    public var magnitude: Float {
        sqrt(dotProduct(self))
    }
    
    /// Compute the squared magnitude
    @inlinable
    public var magnitudeSquared: Float {
        dotProduct(self)
    }
    
    /// Normalize the vector in place
    @inlinable
    public mutating func normalize() {
        let mag = magnitude
        guard mag > 0 else { return }
        self /= mag
    }
    
    /// Return a normalized copy of the vector
    @inlinable
    public func normalized() -> Vector<D> {
        let mag = magnitude
        guard mag > 0 else { return self }
        return self / mag
    }
    
    /// Compute Euclidean distance to another vector
    @inlinable
    public func distance(to other: Vector<D>) -> Float {
        (self - other).magnitude
    }
    
    /// Compute cosine similarity with another vector
    @inlinable
    public func cosineSimilarity(to other: Vector<D>) -> Float {
        let dot = dotProduct(other)
        let mag1 = magnitude
        let mag2 = other.magnitude
        
        guard mag1 > 0 && mag2 > 0 else { return 0 }
        return dot / (mag1 * mag2)
    }
    
    /// Create a normalized random vector (unit vector with random direction)
    public static func randomNormalized() -> Vector<D> {
        return random(in: -1...1).normalized()
    }
}

// MARK: - Arithmetic Operations

extension Vector {
    /// Add two vectors
    @inlinable
    public static func + (lhs: Vector<D>, rhs: Vector<D>) -> Vector<D> {
        var result = lhs
        result.storage.withUnsafeMutableBufferPointer { dest in
            rhs.storage.withUnsafeBufferPointer { src in
                vDSP_vadd(dest.baseAddress!, 1, src.baseAddress!, 1,
                         dest.baseAddress!, 1, vDSP_Length(D.value))
            }
        }
        return result
    }
    
    /// Subtract two vectors
    @inlinable
    public static func - (lhs: Vector<D>, rhs: Vector<D>) -> Vector<D> {
        var result = lhs
        result.storage.withUnsafeMutableBufferPointer { dest in
            rhs.storage.withUnsafeBufferPointer { src in
                vDSP_vsub(src.baseAddress!, 1, dest.baseAddress!, 1,
                         dest.baseAddress!, 1, vDSP_Length(D.value))
            }
        }
        return result
    }
    
    /// Multiply vector by scalar
    @inlinable
    public static func * (lhs: Vector<D>, rhs: Float) -> Vector<D> {
        var result = lhs
        result.storage.withUnsafeMutableBufferPointer { buffer in
            var scalar = rhs
            vDSP_vsmul(buffer.baseAddress!, 1, &scalar,
                      buffer.baseAddress!, 1, vDSP_Length(D.value))
        }
        return result
    }
    
    /// Multiply scalar by vector
    @inlinable
    public static func * (lhs: Float, rhs: Vector<D>) -> Vector<D> {
        rhs * lhs
    }
    
    /// Divide vector by scalar
    @inlinable
    public static func / (lhs: Vector<D>, rhs: Float) -> Vector<D> {
        var result = lhs
        result.storage.withUnsafeMutableBufferPointer { buffer in
            var scalar = rhs
            vDSP_vsdiv(buffer.baseAddress!, 1, &scalar,
                      buffer.baseAddress!, 1, vDSP_Length(D.value))
        }
        return result
    }
    
    /// Add and assign
    @inlinable
    public static func += (lhs: inout Vector<D>, rhs: Vector<D>) {
        lhs = lhs + rhs
    }
    
    /// Subtract and assign
    @inlinable
    public static func -= (lhs: inout Vector<D>, rhs: Vector<D>) {
        lhs = lhs - rhs
    }
    
    /// Multiply and assign
    @inlinable
    public static func *= (lhs: inout Vector<D>, rhs: Float) {
        lhs = lhs * rhs
    }
    
    /// Divide and assign
    @inlinable
    public static func /= (lhs: inout Vector<D>, rhs: Float) {
        lhs = lhs / rhs
    }
    
    /// Negate vector
    @inlinable
    public static prefix func - (vector: Vector<D>) -> Vector<D> {
        var result = vector
        result.storage.withUnsafeMutableBufferPointer { buffer in
            vDSP_vneg(buffer.baseAddress!, 1,
                     buffer.baseAddress!, 1, vDSP_Length(D.value))
        }
        return result
    }
}

// MARK: - Collection Conformance

extension Vector: Collection {
    public typealias Index = Int
    public typealias Element = Float
    
    public var startIndex: Int { 0 }
    public var endIndex: Int { D.value }
    
    public func index(after i: Int) -> Int {
        i + 1
    }
}

// MARK: - Equatable

extension Vector: Equatable {
    @inlinable
    public static func == (lhs: Vector<D>, rhs: Vector<D>) -> Bool {
        for i in 0..<D.value {
            if lhs[i] != rhs[i] { return false }
        }
        return true
    }
}

// MARK: - Hashable

extension Vector: Hashable {
    public func hash(into hasher: inout Hasher) {
        hasher.combine(D.value)
        for i in 0..<Swift.min(8, D.value) { // Hash first 8 elements for performance
            hasher.combine(self[i])
        }
    }
}

// MARK: - Codable

extension Vector: Codable {
    enum CodingKeys: String, CodingKey {
        case dimension
        case values
    }
    
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let dimension = try container.decode(Int.self, forKey: .dimension)
        let values = try container.decode([Float].self, forKey: .values)
        
        guard dimension == D.value else {
            throw VectorError.dimensionMismatch(expected: D.value, actual: dimension)
        }
        
        self.init(values)
    }
    
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(D.value, forKey: .dimension)
        try container.encode(toArray(), forKey: .values)
    }
}

// MARK: - ExpressibleByArrayLiteral

extension Vector: ExpressibleByArrayLiteral {
    public typealias ArrayLiteralElement = Float
}

// MARK: - CustomStringConvertible

extension Vector: CustomStringConvertible {
    public var description: String {
        let values = toArray()
        let preview = values.prefix(5).map { String(format: "%.3f", $0) }.joined(separator: ", ")
        if values.count > 5 {
            return "Vector<\(D.value)>[\(preview), ... (\(values.count - 5) more)]"
        } else {
            return "Vector<\(D.value)>[\(preview)]"
        }
    }
}

// MARK: - Utility Methods

extension Vector {
    /// Create a standard basis vector (one-hot encoded)
    /// 
    /// - Parameter index: Index of the non-zero element
    /// - Returns: Basis vector with 1.0 at the specified index
    @inlinable
    public static func basis(at index: Int) -> Vector<D> {
        precondition(index >= 0 && index < D.value, "Index out of bounds")
        var vector = Vector<D>()
        vector[index] = 1.0
        return vector
    }
}