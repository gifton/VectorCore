// VectorCore: Updated Vector Protocol
//
// Base protocol for vector operations without SIMD requirement
//

import Foundation

/// Base protocol for vector operations
public protocol BaseVectorProtocol: Sendable {
    associatedtype Scalar: BinaryFloatingPoint
    
    /// Number of dimensions in the vector
    static var dimensions: Int { get }
    
    /// Number of elements
    var scalarCount: Int { get }
    
    /// Initialize from an array
    init(from array: [Scalar])
    
    /// Convert to array
    func toArray() -> [Scalar]
    
    /// Access elements by index
    subscript(index: Int) -> Scalar { get }
}

/// Extended vector protocol with operations
public protocol ExtendedVectorProtocol: BaseVectorProtocol where Scalar == Float {
    /// Compute dot product
    func dotProduct(_ other: Self) -> Float
    
    /// Magnitude (L2 norm)
    var magnitude: Float { get }
    
    /// Normalized copy
    func normalized() -> Self
    
    /// Distance to another vector
    func distance(to other: Self) -> Float
    
    /// Cosine similarity
    func cosineSimilarity(to other: Self) -> Float
}

// MARK: - Helper Extensions

extension BaseVectorProtocol {
    /// Validate array dimensions
    public static func validate(_ array: [Scalar]) -> Bool {
        array.count == dimensions
    }
    
    /// Create from array if valid
    public static func create(from array: [Scalar]) -> Self? {
        guard validate(array) else { return nil }
        return Self(from: array)
    }
}