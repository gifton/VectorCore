// VectorCore: Generic Vector Implementation
//
// Single implementation for all vector dimensions
//

import Foundation
import simd

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
    
    /// Safely initialize from an array
    ///
    /// - Parameter values: Array of values (must match dimension)
    /// - Returns: nil if array count doesn't match dimension
    ///
    /// This is the safe alternative to init(_:) that returns nil instead of crashing.
    @inlinable
    public init?(safe values: [Float]) {
        guard values.count == D.value else { return nil }
        self.storage = D.Storage(from: values)
    }
    
    /// Creates a vector from an array literal.
    ///
    /// Enables concise vector initialization using Swift's array literal syntax.
    /// The number of elements must exactly match the vector's dimension.
    ///
    /// ## Example Usage
    /// ```swift
    /// let v1: Vector<Dim32> = [1.0, 2.0, 3.0, /* ... 29 more values ... */]
    /// let v2 = Vector<Dim32>(arrayLiteral: 1.0, 2.0, 3.0, /* ... 29 more values ... */)
    /// ```
    ///
    /// - Parameter elements: Variadic Float values to initialize the vector
    /// - Precondition: `elements.count == D.value`
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
    
    /// Safely access elements by index
    ///
    /// - Parameter index: The index to access
    /// - Returns: The value at the index, or nil if index is out of bounds
    ///
    /// This is the safe alternative to subscript that returns nil instead of crashing.
    @inlinable
    public func at(_ index: Int) -> Float? {
        guard index >= 0 && index < D.value else { return nil }
        return storage[index]
    }
    
    /// Safely set element at index
    ///
    /// - Parameters:
    ///   - index: The index to set
    ///   - value: The value to set
    /// - Returns: true if successful, false if index was out of bounds
    ///
    /// This is the safe alternative to subscript setter that returns a success indicator.
    @inlinable
    public mutating func setAt(_ index: Int, to value: Float) -> Bool {
        guard index >= 0 && index < D.value else { return false }
        storage[index] = value
        return true
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

// MARK: - VectorType Conformance

extension Vector: VectorType where D.Storage: VectorStorageOperations {
    // All requirements are already implemented:
    // - scalarCount (property)
    // - toArray() (method)
    // - dotProduct(_:) (method)
    // - magnitude (property)
    // - normalized() (method)
}

// MARK: - Binary Serialization

extension Vector: BinaryEncodable, BinaryDecodable where D.Storage: VectorStorageOperations {
    /// Decode from binary data
    public static func decodeBinary(from data: Data) throws -> Vector<D> {
        // Step 1: Validate header and get dimension
        let (_, dimension, _) = try BinaryFormat.validateHeader(in: data)
        
        // Step 2: Validate dimension matches type's dimension
        guard Int(dimension) == D.value else {
            throw VectorError.dimensionMismatch(
                expected: D.value,
                actual: Int(dimension)
            )
        }
        
        // Step 3: Validate CRC32 checksum
        try BinaryFormat.validateChecksum(in: data)
        
        // Step 4: Read vector data
        let values = try BinaryFormat.readFloatArray(
            from: data,
            at: BinaryHeader.headerSize,
            count: D.value
        )
        
        // Step 5: Create and return vector
        return Vector(values)
    }
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
        
        // Fast path: if already normalized (magnitude ≈ 1), skip division
        if abs(mag - 1.0) < 1e-6 {
            return
        }
        
        self /= mag
    }
    
    /// Return a normalized copy of the vector
    @inlinable
    public func normalized() -> Vector<D> {
        let mag = magnitude
        guard mag > 0 else { return self }
        
        // Fast path: if already normalized (magnitude ≈ 1), return copy
        if abs(mag - 1.0) < 1e-6 {
            return self
        }
        
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
        var result = lhs  // COW: no allocation yet
        result += rhs     // Will only allocate if lhs is shared
        return result
    }
    
    /// Subtract two vectors
    @inlinable
    public static func - (lhs: Vector<D>, rhs: Vector<D>) -> Vector<D> {
        let lhsArray = lhs.toArray()
        let rhsArray = rhs.toArray()
        let result = Operations.simdProvider.subtract(lhsArray, rhsArray)
        return Vector<D>(result)
    }
    
    /// Multiply vector by scalar
    @inlinable
    public static func * (lhs: Vector<D>, rhs: Float) -> Vector<D> {
        // Fast paths for special values
        switch rhs {
        case 0:
            // Multiplication by 0 returns zero vector
            return Vector<D>()
        case 1:
            // Multiplication by 1 returns copy (COW)
            return lhs
        case -1:
            // Multiplication by -1 uses specialized negation
            return -lhs
        default:
            // General case
            let lhsArray = lhs.toArray()
            let result = Operations.simdProvider.multiply(lhsArray, by: rhs)
            return Vector<D>(result)
        }
    }
    
    /// Multiply scalar by vector
    @inlinable
    public static func * (lhs: Float, rhs: Vector<D>) -> Vector<D> {
        rhs * lhs
    }
    
    /// Divide vector by scalar
    @inlinable
    public static func / (lhs: Vector<D>, rhs: Float) -> Vector<D> {
        // Fast path for division by 1
        if rhs == 1 {
            return lhs
        }
        
        // General case
        let lhsArray = lhs.toArray()
        let result = Operations.simdProvider.divide(lhsArray, by: rhs)
        return Vector<D>(result)
    }
    
    /// Add and assign
    @inlinable
    public static func += (lhs: inout Vector<D>, rhs: Vector<D>) {
        let lhsArray = lhs.toArray()
        let rhsArray = rhs.toArray()
        let result = Operations.simdProvider.add(lhsArray, rhsArray)
        lhs = Vector<D>(result)
    }
    
    /// Subtract and assign
    @inlinable
    public static func -= (lhs: inout Vector<D>, rhs: Vector<D>) {
        let lhsArray = lhs.toArray()
        let rhsArray = rhs.toArray()
        let result = Operations.simdProvider.subtract(lhsArray, rhsArray)
        lhs = Vector<D>(result)
    }
    
    /// Multiply and assign
    @inlinable
    public static func *= (lhs: inout Vector<D>, rhs: Float) {
        // Fast paths for special values
        switch rhs {
        case 0:
            // Multiplication by 0 - set to zero
            lhs = Vector<D>()
        case 1:
            // Multiplication by 1 - no-op
            return
        case -1:
            // Multiplication by -1 - negate in place
            let array = lhs.toArray()
            let result = Operations.simdProvider.negate(array)
            lhs = Vector<D>(result)
        default:
            // General case
            let array = lhs.toArray()
            let result = Operations.simdProvider.multiply(array, by: rhs)
            lhs = Vector<D>(result)
        }
    }
    
    /// Divide and assign
    @inlinable
    public static func /= (lhs: inout Vector<D>, rhs: Float) {
        // Fast path for division by 1
        if rhs == 1 {
            return
        }
        
        // General case
        let array = lhs.toArray()
        let result = Operations.simdProvider.divide(array, by: rhs)
        lhs = Vector<D>(result)
    }
    
    /// Negate vector
    @inlinable
    public static prefix func - (vector: Vector<D>) -> Vector<D> {
        let array = vector.toArray()
        let result = Operations.simdProvider.negate(array)
        return Vector<D>(result)
    }
}

// MARK: - Collection Conformance

extension Vector: Collection {
    /// The type used to index into the vector's elements.
    public typealias Index = Int
    
    /// The type of elements stored in the vector (always Float).
    public typealias Element = Float
    
    /// The position of the first element in the vector.
    ///
    /// Always returns 0 as vectors are zero-indexed.
    public var startIndex: Int { 0 }
    
    /// The position one past the last element in the vector.
    ///
    /// Equal to the vector's dimension, enabling standard Collection iteration.
    public var endIndex: Int { D.value }
    
    /// Returns the position immediately after the given index.
    ///
    /// - Parameter i: A valid index of the collection. `i` must be less than `endIndex`.
    /// - Returns: The index value immediately after `i`.
    ///
    /// - Complexity: O(1)
    public func index(after i: Int) -> Int {
        i + 1
    }
}

// MARK: - Equatable

extension Vector: Equatable {
    /// Exact equality comparison
    ///
    /// Note: This performs exact floating-point comparison.
    /// For approximate equality, use `isApproximatelyEqual(to:tolerance:)`
    @inlinable
    public static func == (lhs: Vector<D>, rhs: Vector<D>) -> Bool {
        for i in 0..<D.value {
            if lhs[i] != rhs[i] { return false }
        }
        return true
    }
}

// MARK: - Approximate Equality

extension Vector {
    /// Check if two vectors are approximately equal within a tolerance
    ///
    /// - Parameters:
    ///   - other: The vector to compare with
    ///   - tolerance: Maximum allowed difference per component (default: 1e-6)
    /// - Returns: true if all components are within tolerance
    ///
    /// This is useful for comparing vectors after arithmetic operations
    /// where floating-point precision may introduce small differences.
    @inlinable
    public func isApproximatelyEqual(to other: Vector<D>, tolerance: Float = 1e-6) -> Bool {
        for i in 0..<D.value {
            if abs(self[i] - other[i]) > tolerance {
                return false
            }
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
    
    /// Safely create a standard basis vector (one-hot encoded)
    /// 
    /// - Parameter index: Index of the non-zero element
    /// - Returns: Basis vector with 1.0 at the specified index, or nil if index is out of bounds
    ///
    /// This is the safe alternative to basis(at:) that returns nil instead of crashing.
    @inlinable
    public static func basis(safe index: Int) -> Vector<D>? {
        guard index >= 0 && index < D.value else { return nil }
        var vector = Vector<D>()
        vector[index] = 1.0
        return vector
    }
}
