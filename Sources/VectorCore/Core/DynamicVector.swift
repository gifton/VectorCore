// VectorCore: Dynamic Vector Implementation
//
// Runtime-determined dimension vectors
//

import Foundation
import Accelerate

/// Vector type with runtime-determined dimensions
public struct DynamicVector: Sendable {
    private let storage: DynamicArrayStorage
    
    /// The number of elements in the vector
    public var dimension: Int { storage.count }
    public var scalarCount: Int { storage.count }
    
    /// Initialize a zero vector with given dimension
    public init(dimension: Int) {
        self.storage = DynamicArrayStorage(dimension: dimension)
    }
    
    /// Initialize with a repeating value
    public init(dimension: Int, repeating value: Float) {
        self.storage = DynamicArrayStorage(dimension: dimension, repeating: value)
    }
    
    /// Initialize from an array
    public init(dimension: Int, values: [Float]) {
        precondition(values.count == dimension, "Value count must match dimension")
        self.storage = DynamicArrayStorage(from: values)
    }
    
    /// Initialize from array (dimension inferred)
    public init(_ values: [Float]) {
        self.storage = DynamicArrayStorage(from: values)
    }
    
    /// Access elements by index
    public subscript(index: Int) -> Float {
        get {
            precondition(index >= 0 && index < dimension, "Index out of bounds")
            return storage[index]
        }
        set {
            precondition(index >= 0 && index < dimension, "Index out of bounds")
            storage[index] = newValue
        }
    }
}

// MARK: - Basic Operations

extension DynamicVector {
    /// Convert to array
    public func toArray() -> [Float] {
        var result = [Float](repeating: 0, count: dimension)
        _ = storage.withUnsafeBufferPointer { buffer in
            result.withUnsafeMutableBufferPointer { dest in
                dest.initialize(from: buffer)
            }
        }
        return result
    }
    
    /// Compute dot product with another vector
    public func dotProduct(_ other: DynamicVector) -> Float {
        precondition(dimension == other.dimension, "Dimensions must match")
        return storage.dotProduct(other.storage)
    }
    
    /// Compute the magnitude (L2 norm)
    public var magnitude: Float {
        sqrt(dotProduct(self))
    }
    
    /// Compute the squared magnitude
    public var magnitudeSquared: Float {
        dotProduct(self)
    }
    
    /// Normalize the vector in place
    public mutating func normalize() {
        let mag = magnitude
        guard mag > 0 else { return }
        
        storage.withUnsafeMutableBufferPointer { buffer in
            var scalar = mag
            vDSP_vsdiv(buffer.baseAddress!, 1, &scalar,
                      buffer.baseAddress!, 1, vDSP_Length(dimension))
        }
    }
    
    /// Return a normalized copy of the vector
    public func normalized() -> DynamicVector {
        var copy = self
        copy.normalize()
        return copy
    }
    
    /// Compute Euclidean distance to another vector
    public func distance(to other: DynamicVector) -> Float {
        (self - other).magnitude
    }
    
    /// Compute cosine similarity with another vector
    public func cosineSimilarity(to other: DynamicVector) -> Float {
        let dot = dotProduct(other)
        let mag1 = magnitude
        let mag2 = other.magnitude
        
        guard mag1 > 0 && mag2 > 0 else { return 0 }
        return dot / (mag1 * mag2)
    }
}

// MARK: - Arithmetic Operations

extension DynamicVector {
    /// Add two vectors
    public static func + (lhs: DynamicVector, rhs: DynamicVector) -> DynamicVector {
        precondition(lhs.dimension == rhs.dimension, "Dimensions must match")
        
        var result = lhs
        result.storage.withUnsafeMutableBufferPointer { dest in
            rhs.storage.withUnsafeBufferPointer { src in
                vDSP_vadd(dest.baseAddress!, 1, src.baseAddress!, 1,
                         dest.baseAddress!, 1, vDSP_Length(lhs.dimension))
            }
        }
        return result
    }
    
    /// Subtract two vectors
    public static func - (lhs: DynamicVector, rhs: DynamicVector) -> DynamicVector {
        precondition(lhs.dimension == rhs.dimension, "Dimensions must match")
        
        var result = lhs
        result.storage.withUnsafeMutableBufferPointer { dest in
            rhs.storage.withUnsafeBufferPointer { src in
                vDSP_vsub(src.baseAddress!, 1, dest.baseAddress!, 1,
                         dest.baseAddress!, 1, vDSP_Length(lhs.dimension))
            }
        }
        return result
    }
    
    /// Multiply vector by scalar
    public static func * (lhs: DynamicVector, rhs: Float) -> DynamicVector {
        var result = lhs
        result.storage.withUnsafeMutableBufferPointer { buffer in
            var scalar = rhs
            vDSP_vsmul(buffer.baseAddress!, 1, &scalar,
                      buffer.baseAddress!, 1, vDSP_Length(lhs.dimension))
        }
        return result
    }
    
    /// Multiply scalar by vector
    public static func * (lhs: Float, rhs: DynamicVector) -> DynamicVector {
        rhs * lhs
    }
    
    /// Divide vector by scalar
    public static func / (lhs: DynamicVector, rhs: Float) -> DynamicVector {
        var result = lhs
        result.storage.withUnsafeMutableBufferPointer { buffer in
            var scalar = rhs
            vDSP_vsdiv(buffer.baseAddress!, 1, &scalar,
                      buffer.baseAddress!, 1, vDSP_Length(lhs.dimension))
        }
        return result
    }
    
    /// Negate vector
    public static prefix func - (vector: DynamicVector) -> DynamicVector {
        var result = vector
        result.storage.withUnsafeMutableBufferPointer { buffer in
            vDSP_vneg(buffer.baseAddress!, 1,
                     buffer.baseAddress!, 1, vDSP_Length(vector.dimension))
        }
        return result
    }
}

// MARK: - Collection Conformance

extension DynamicVector: Collection {
    public typealias Index = Int
    public typealias Element = Float
    
    public var startIndex: Int { 0 }
    public var endIndex: Int { dimension }
    
    public func index(after i: Int) -> Int {
        i + 1
    }
}

// MARK: - Equatable

extension DynamicVector: Equatable {
    public static func == (lhs: DynamicVector, rhs: DynamicVector) -> Bool {
        guard lhs.dimension == rhs.dimension else { return false }
        
        for i in 0..<lhs.dimension {
            if lhs[i] != rhs[i] { return false }
        }
        return true
    }
}

// MARK: - Hashable

extension DynamicVector: Hashable {
    public func hash(into hasher: inout Hasher) {
        hasher.combine(dimension)
        for i in 0..<Swift.min(8, dimension) { // Hash first 8 elements for performance
            hasher.combine(self[i])
        }
    }
}

// MARK: - Codable

extension DynamicVector: Codable {
    enum CodingKeys: String, CodingKey {
        case dimension
        case values
    }
    
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let dimension = try container.decode(Int.self, forKey: .dimension)
        let values = try container.decode([Float].self, forKey: .values)
        
        guard values.count == dimension else {
            throw VectorError.dimensionMismatch(expected: dimension, actual: values.count)
        }
        
        self.init(dimension: dimension, values: values)
    }
    
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(dimension, forKey: .dimension)
        try container.encode(toArray(), forKey: .values)
    }
}

// MARK: - CustomStringConvertible

extension DynamicVector: CustomStringConvertible {
    public var description: String {
        let values = toArray()
        let preview = values.prefix(5).map { String(format: "%.3f", $0) }.joined(separator: ", ")
        if values.count > 5 {
            return "DynamicVector(\(dimension))[\(preview), ... (\(values.count - 5) more)]"
        } else {
            return "DynamicVector(\(dimension))[\(preview)]"
        }
    }
}

// MARK: - VectorType Conformance

extension DynamicVector: VectorType {
    // All requirements are already implemented:
    // - scalarCount (property)
    // - toArray() (method)
    // - dotProduct(_:) (method)
    // - magnitude (property)
    // - normalized() (method)
}

// MARK: - Additional Math Operations

extension DynamicVector {
    /// Element-wise multiplication (Hadamard product)
    public static func .* (lhs: DynamicVector, rhs: DynamicVector) -> DynamicVector {
        precondition(lhs.dimension == rhs.dimension, "Dimensions must match")
        
        var result = lhs
        result.storage.withUnsafeMutableBufferPointer { dest in
            rhs.storage.withUnsafeBufferPointer { src in
                vDSP_vmul(dest.baseAddress!, 1, src.baseAddress!, 1,
                         dest.baseAddress!, 1, vDSP_Length(lhs.dimension))
            }
        }
        return result
    }
    
    /// Element-wise division
    public static func ./ (lhs: DynamicVector, rhs: DynamicVector) -> DynamicVector {
        precondition(lhs.dimension == rhs.dimension, "Dimensions must match")
        
        var result = lhs
        result.storage.withUnsafeMutableBufferPointer { dest in
            rhs.storage.withUnsafeBufferPointer { src in
                vDSP_vdiv(src.baseAddress!, 1, dest.baseAddress!, 1,
                         dest.baseAddress!, 1, vDSP_Length(lhs.dimension))
            }
        }
        return result
    }
    
    /// L1 norm (Manhattan norm)
    public var l1Norm: Float {
        var result: Float = 0
        storage.withUnsafeBufferPointer { buffer in
            var temp = [Float](repeating: 0, count: dimension)
            temp.withUnsafeMutableBufferPointer { tempBuffer in
                vDSP_vabs(buffer.baseAddress!, 1,
                         tempBuffer.baseAddress!, 1, vDSP_Length(dimension))
                vDSP_sve(tempBuffer.baseAddress!, 1, &result, vDSP_Length(dimension))
            }
        }
        return result
    }
    
    /// L2 norm (Euclidean norm) - alias for magnitude
    public var l2Norm: Float {
        magnitude
    }
    
    /// L∞ norm (Maximum norm)
    public var lInfinityNorm: Float {
        var result: Float = 0
        storage.withUnsafeBufferPointer { buffer in
            vDSP_maxmgv(buffer.baseAddress!, 1, &result, vDSP_Length(dimension))
        }
        return result
    }
    
    /// Create a random dynamic vector
    public static func random(dimension: Int, in range: ClosedRange<Float> = -1...1) -> DynamicVector {
        let values = (0..<dimension).map { _ in Float.random(in: range) }
        return DynamicVector(dimension: dimension, values: values)
    }
}