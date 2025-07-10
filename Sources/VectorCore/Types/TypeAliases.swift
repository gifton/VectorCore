// VectorCore: Type Aliases
//
// Backward-compatible type aliases for common vector dimensions
//

import Foundation

// MARK: - Standard Vector Type Aliases

/// 128-dimensional vector (common for small embeddings)
public typealias Vector128 = Vector<Dim128>

/// 256-dimensional vector
public typealias Vector256 = Vector<Dim256>

/// 512-dimensional vector (common for BERT-style models)
public typealias Vector512 = Vector<Dim512>

/// 768-dimensional vector (BERT base)
public typealias Vector768 = Vector<Dim768>

/// 1536-dimensional vector (larger models)
public typealias Vector1536 = Vector<Dim1536>

/// 3072-dimensional vector (very large models)
public typealias Vector3072 = Vector<Dim3072>

// MARK: - Legacy Compatibility

/// Type alias for distance scores
public typealias DistanceScore = Float

/// Type alias for similarity scores
public typealias SimilarityScore = Float

// MARK: - Collection Type Aliases

/// Type alias for a collection of Vector128
public typealias Vector128Collection = [Vector128]

/// Type alias for a collection of Vector256
public typealias Vector256Collection = [Vector256]

/// Type alias for a collection of Vector512
public typealias Vector512Collection = [Vector512]

/// Type alias for a collection of Vector768
public typealias Vector768Collection = [Vector768]

/// Type alias for a collection of Vector1536
public typealias Vector1536Collection = [Vector1536]

// MARK: - Legacy Protocol Compatibility

/// Protocol that custom vector types must conform to
/// This exists for backward compatibility with code expecting CustomVectorType
public protocol CustomVectorType {
    associatedtype Scalar: BinaryFloatingPoint
    static var dimensions: Int { get }
    func normalized() -> Self
    func dotProduct(_ other: Self) -> Float
    var magnitude: Float { get }
}

// MARK: - Extension for Legacy Compatibility

extension Vector: CustomVectorType where D.Storage: VectorStorageOperations {
    // CustomVectorType conformance
    // All requirements are already satisfied in Vector.swift
}

// MARK: - Helper Extensions

extension Vector128 {
    /// Legacy initializer for compatibility
    public init(unsafeUninitializedCapacity: Int, initializingWith initializer: (inout UnsafeMutableBufferPointer<Float>) -> Void) {
        precondition(unsafeUninitializedCapacity == 128)
        var values = [Float](repeating: 0, count: 128)
        values.withUnsafeMutableBufferPointer { buffer in
            initializer(&buffer)
        }
        self.init(values)
    }
}

extension Vector256 {
    /// Legacy initializer for compatibility
    public init(unsafeUninitializedCapacity: Int, initializingWith initializer: (inout UnsafeMutableBufferPointer<Float>) -> Void) {
        precondition(unsafeUninitializedCapacity == 256)
        var values = [Float](repeating: 0, count: 256)
        values.withUnsafeMutableBufferPointer { buffer in
            initializer(&buffer)
        }
        self.init(values)
    }
}

extension Vector512 {
    /// Legacy initializer for compatibility
    public init(unsafeUninitializedCapacity: Int, initializingWith initializer: (inout UnsafeMutableBufferPointer<Float>) -> Void) {
        precondition(unsafeUninitializedCapacity == 512)
        var values = [Float](repeating: 0, count: 512)
        values.withUnsafeMutableBufferPointer { buffer in
            initializer(&buffer)
        }
        self.init(values)
    }
    
    /// Legacy batch creation method
    public static func createBatch(from values: [Float]) -> [Vector512] {
        guard values.count % 512 == 0 else { return [] }
        
        let count = values.count / 512
        var result: [Vector512] = []
        result.reserveCapacity(count)
        
        for i in 0..<count {
            let start = i * 512
            let end = start + 512
            result.append(Vector512(Array(values[start..<end])))
        }
        
        return result
    }
}

extension Vector768 {
    /// Legacy initializer for compatibility
    public init(unsafeUninitializedCapacity: Int, initializingWith initializer: (inout UnsafeMutableBufferPointer<Float>) -> Void) {
        precondition(unsafeUninitializedCapacity == 768)
        var values = [Float](repeating: 0, count: 768)
        values.withUnsafeMutableBufferPointer { buffer in
            initializer(&buffer)
        }
        self.init(values)
    }
}

extension Vector1536 {
    /// Legacy initializer for compatibility
    public init(unsafeUninitializedCapacity: Int, initializingWith initializer: (inout UnsafeMutableBufferPointer<Float>) -> Void) {
        precondition(unsafeUninitializedCapacity == 1536)
        var values = [Float](repeating: 0, count: 1536)
        values.withUnsafeMutableBufferPointer { buffer in
            initializer(&buffer)
        }
        self.init(values)
    }
}