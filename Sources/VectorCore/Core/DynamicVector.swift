// VectorCore: Dynamic Vector Implementation
//
// Runtime-determined dimension vectors
//

import Foundation
import Accelerate

/// Vector type with runtime-determined dimensions
public struct DynamicVector: Sendable {
    private var storage: COWDynamicStorage
    
    /// The number of elements in the vector
    public var dimension: Int { storage.count }
    public var scalarCount: Int { storage.count }
    
    /// Initialize a zero vector with given dimension
    public init(dimension: Int) {
        self.storage = COWDynamicStorage(dimension: dimension)
    }
    
    /// Initialize with a repeating value
    public init(dimension: Int, repeating value: Float) {
        self.storage = COWDynamicStorage(dimension: dimension, repeating: value)
    }
    
    /// Initialize from an array
    public init(dimension: Int, values: [Float]) {
        precondition(values.count == dimension, "Value count must match dimension")
        self.storage = COWDynamicStorage(from: values)
    }
    
    /// Safely initialize from an array
    ///
    /// - Parameters:
    ///   - dimension: Expected dimension
    ///   - values: Array of values
    /// - Returns: nil if values count doesn't match dimension
    ///
    /// This is the safe alternative to init(dimension:values:) that returns nil instead of crashing.
    public init?(safe dimension: Int, values: [Float]) {
        guard values.count == dimension else { return nil }
        self.storage = COWDynamicStorage(from: values)
    }
    
    /// Initialize from array (dimension inferred)
    public init(_ values: [Float]) {
        self.storage = COWDynamicStorage(from: values)
    }
    
    /// Access elements by index
    public subscript(index: Int) -> Float {
        get {
            precondition(index >= 0 && index < dimension, "Index out of bounds")
            return storage[index]
        }
        mutating set {
            precondition(index >= 0 && index < dimension, "Index out of bounds")
            storage[index] = newValue
        }
    }
    
    /// Safely access elements by index
    ///
    /// - Parameter index: The index to access
    /// - Returns: The value at the index, or nil if index is out of bounds
    ///
    /// This is the safe alternative to subscript that returns nil instead of crashing.
    public func at(_ index: Int) -> Float? {
        guard index >= 0 && index < dimension else { return nil }
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
    public mutating func setAt(_ index: Int, to value: Float) -> Bool {
        guard index >= 0 && index < dimension else { return false }
        storage[index] = value
        return true
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
        
        let dim = dimension  // Capture dimension before mutable access
        storage.withUnsafeMutableBufferPointer { buffer in
            var scalar = mag
            vDSP_vsdiv(buffer.baseAddress!, 1, &scalar,
                      buffer.baseAddress!, 1, vDSP_Length(dim))
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
    /// Exact equality comparison
    ///
    /// Note: This performs exact floating-point comparison.
    /// For approximate equality, use `isApproximatelyEqual(to:tolerance:)`
    public static func == (lhs: DynamicVector, rhs: DynamicVector) -> Bool {
        guard lhs.dimension == rhs.dimension else { return false }
        
        for i in 0..<lhs.dimension {
            if lhs[i] != rhs[i] { return false }
        }
        return true
    }
}

// MARK: - Approximate Equality

extension DynamicVector {
    /// Check if two vectors are approximately equal within a tolerance
    ///
    /// - Parameters:
    ///   - other: The vector to compare with
    ///   - tolerance: Maximum allowed difference per component (default: 1e-6)
    /// - Returns: true if all components are within tolerance
    ///
    /// This is useful for comparing vectors after arithmetic operations
    /// where floating-point precision may introduce small differences.
    public func isApproximatelyEqual(to other: DynamicVector, tolerance: Float = 1e-6) -> Bool {
        guard dimension == other.dimension else { return false }
        
        for i in 0..<dimension {
            if abs(self[i] - other[i]) > tolerance {
                return false
            }
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

// MARK: - Protocol Conformances

extension DynamicVector: BaseVectorProtocol {
    public typealias Scalar = Float
    public static var dimensions: Int { 0 } // Dynamic, so return 0
    
    public init(from array: [Float]) {
        self.init(array)
    }
}

extension DynamicVector: ExtendedVectorProtocol {
    // All required methods are already implemented above
}

// MARK: - Binary Serialization

extension DynamicVector: BinaryEncodable, BinaryDecodable {
    /// Decode from binary data
    public static func decodeBinary(from data: Data) throws -> DynamicVector {
        // Step 1: Validate header and get dimension
        let (_, dimension, _) = try BinaryFormat.validateHeader(in: data)
        
        // Step 2: Validate CRC32 checksum
        try BinaryFormat.validateChecksum(in: data)
        
        // Step 3: Read vector data
        let values = try BinaryFormat.readFloatArray(
            from: data,
            at: BinaryHeader.headerSize,
            count: Int(dimension)
        )
        
        // Step 4: Create and return vector
        return DynamicVector(values)
    }
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

// MARK: - Quality Metrics

extension DynamicVector {
    /// Calculate sparsity (proportion of near-zero elements)
    ///
    /// - Parameter threshold: Values with absolute value <= threshold are considered zero
    /// - Returns: Proportion of sparse elements (0.0 = dense, 1.0 = all zeros)
    ///
    /// Use cases:
    /// - Compression decisions (sparse vectors can be stored efficiently)
    /// - Quality assessment (very sparse vectors may indicate issues)
    /// - Feature selection (identify uninformative features)
    public func sparsity(threshold: Float = Float.ulpOfOne) -> Float {
        var sparseCount = 0
        for i in 0..<dimension {
            let value = self[i]
            // Non-finite values (NaN, Infinity) are not considered sparse
            if value.isFinite && abs(value) <= threshold {
                sparseCount += 1
            }
        }
        return Float(sparseCount) / Float(dimension)
    }
    
    /// Calculate Shannon entropy
    ///
    /// Treats the vector as a probability distribution after normalization.
    /// Higher values indicate more uniform distribution of values.
    ///
    /// Formula: H(X) = -Σ(p_i * log(p_i)) where p_i = |x_i| / Σ|x_j|
    ///
    /// Returns:
    /// - 0.0 for zero vectors or single-spike vectors
    /// - Higher values for more distributed vectors
    /// - Maximum entropy = log(n) for uniform distribution
    ///
    /// Use cases:
    /// - Measure information content
    /// - Detect concentrated vs distributed patterns
    /// - Feature quality assessment
    public var entropy: Float {
        // Calculate sum of absolute values for normalization
        var absSum: Float = 0
        var hasNonFinite = false
        
        for i in 0..<dimension {
            let value = self[i]
            if !value.isFinite {
                hasNonFinite = true
                break
            }
            absSum += abs(value)
        }
        
        // Handle non-finite values (NaN, Infinity)
        guard !hasNonFinite else {
            return .nan
        }
        
        // Handle zero vector
        guard absSum > Float.ulpOfOne else {
            return 0.0
        }
        
        // Calculate entropy using Shannon formula
        var entropy: Float = 0
        for i in 0..<dimension {
            let p = abs(self[i]) / absSum
            if p > Float.ulpOfOne {  // Skip zero probabilities
                entropy -= p * log(p)
            }
        }
        
        return entropy
    }
    
    /// Comprehensive quality assessment
    ///
    /// Returns a VectorQuality struct containing multiple metrics for
    /// assessing vector characteristics and quality.
    public var quality: VectorQuality {
        // Calculate variance manually for DynamicVector
        let mean = self.toArray().reduce(0, +) / Float(dimension)
        var sumSquaredDiff: Float = 0
        for i in 0..<dimension {
            let diff = self[i] - mean
            sumSquaredDiff += diff * diff
        }
        let variance = sumSquaredDiff / Float(dimension)
        
        return VectorQuality(
            magnitude: magnitude,
            variance: variance,
            sparsity: sparsity(),
            entropy: entropy
        )
    }
}

// MARK: - Serialization

extension DynamicVector {
    /// Base64-encoded representation of the vector
    ///
    /// Uses the binary encoding format with CRC32 checksum for data integrity.
    /// Useful for:
    /// - Transmitting vectors over text-based protocols
    /// - Storing vectors in JSON/XML
    /// - Embedding vectors in URLs
    public var base64Encoded: String {
        guard let data = try? encodeBinary() else {
            return ""  // Should not happen for valid vectors
        }
        return data.base64EncodedString()
    }
    
    /// Decode vector from base64 string
    ///
    /// - Parameter base64String: Base64-encoded vector data
    /// - Returns: Decoded vector
    /// - Throws: VectorError if decoding fails
    public static func base64Decoded(from base64String: String) throws -> DynamicVector {
        guard let data = Data(base64Encoded: base64String) else {
            throw VectorError.invalidDataFormat(
                expected: "base64",
                actual: "invalid base64 string"
            )
        }
        return try decodeBinary(from: data)
    }
}