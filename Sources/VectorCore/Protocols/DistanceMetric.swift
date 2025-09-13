//
//  DistanceMetric.swift
//  VectorCore
//
//  Protocol for distance metrics used in vector operations
//

import Foundation

/// Protocol for distance metrics between vectors
public protocol DistanceMetric<Scalar>: Sendable {
    /// The scalar type for distance calculations
    associatedtype Scalar: BinaryFloatingPoint

    /// Calculate distance between two vectors
    /// - Parameters:
    ///   - a: First vector
    ///   - b: Second vector
    /// - Returns: Distance between vectors
    /// - Precondition: Both vectors must have the same dimension
    func distance<V: VectorProtocol>(_ a: V, _ b: V) -> Scalar
    where V.Scalar == Scalar

    /// Metric name for identification
    var name: String { get }

    /// Unique identifier for the metric
    var identifier: String { get }
}

// MARK: - Default Implementations

public extension DistanceMetric {
    var identifier: String { name }

    /// Batch distance computation with default implementation
    func batchDistance<V: VectorProtocol>(
        query: V,
        candidates: [V]
    ) -> [Scalar] where V.Scalar == Scalar {
        candidates.map { distance(query, $0) }
    }
}
