//
//  ProviderProtocols.swift
//  VectorCore
//
//  Provider protocols for distance computation and vector operations
//

import Foundation

// MARK: - Distance Metric Enum

/// Enumeration of supported distance metrics
public enum SupportedDistanceMetric: String, CaseIterable, Sendable {
    case euclidean = "euclidean"
    case cosine = "cosine"
    case dotProduct = "dot_product"
    case manhattan = "manhattan"
    case chebyshev = "chebyshev"
    
    /// Human-readable name
    public var displayName: String {
        switch self {
        case .euclidean: return "Euclidean Distance"
        case .cosine: return "Cosine Distance"
        case .dotProduct: return "Dot Product"
        case .manhattan: return "Manhattan Distance"
        case .chebyshev: return "Chebyshev Distance"
        }
    }
}

// MARK: - Distance Provider Protocol

/// Protocol for providing distance computation services
public protocol DistanceProvider: Sendable {
    
    /// Compute distance between two vectors using the specified metric
    /// - Parameters:
    ///   - vector1: First vector
    ///   - vector2: Second vector  
    ///   - metric: Distance metric to use
    /// - Returns: Distance value
    func distance<T: VectorProtocol>(
        from vector1: T,
        to vector2: T,
        metric: SupportedDistanceMetric
    ) async throws -> Float where T.Scalar == Float
    
    /// Compute distances from query vector to multiple candidate vectors
    /// - Parameters:
    ///   - query: Query vector
    ///   - candidates: Array of candidate vectors
    ///   - metric: Distance metric to use
    /// - Returns: Array of distances
    func batchDistance<T: VectorProtocol>(
        from query: T,
        to candidates: [T],
        metric: SupportedDistanceMetric
    ) async throws -> [Float] where T.Scalar == Float
}

// MARK: - Vector Operations Provider Protocol

/// Protocol for providing vector operations services
public protocol VectorOperationsProvider: Sendable {
    
    /// Add two vectors
    func add<T: VectorProtocol>(_ v1: T, _ v2: T) async throws -> T where T.Scalar == Float
    
    /// Multiply two vectors element-wise
    func multiply<T: VectorProtocol>(_ v1: T, _ v2: T) async throws -> T where T.Scalar == Float
    
    /// Scale vector by scalar
    func scale<T: VectorProtocol>(_ vector: T, by scalar: Float) async throws -> T where T.Scalar == Float
    
    /// Normalize vector to unit length
    func normalize<T: VectorProtocol>(_ vector: T) async throws -> T where T.Scalar == Float
    
    /// Compute dot product between two vectors
    func dotProduct<T: VectorProtocol>(_ v1: T, _ v2: T) async throws -> Float where T.Scalar == Float
}

// MARK: - Default Implementations

public extension VectorProtocol where Scalar == Float {
    
    /// Compute distance using the specified metric
    func distance(to other: Self, metric: SupportedDistanceMetric = .euclidean) -> Float {
        switch metric {
        case .euclidean:
            return euclideanDistance(to: other)
        case .cosine:
            return cosineDistance(to: other)
        case .dotProduct:
            return -dotProduct(other) // Negative because smaller values = more similar
        case .manhattan:
            return manhattanDistance(to: other)
        case .chebyshev:
            return chebyshevDistance(to: other)
        }
    }
}