// VectorCore
//
// Vector types and operations for Swift
//

import Foundation

// MARK: - Core Type Aliases

/// Vector identifier type
public typealias VectorID = String

/// High-precision timestamp
public typealias Timestamp = UInt64

// Type aliases moved to Types/TypeAliases.swift to avoid duplication

// MARK: - Convenience Extensions

public extension Array where Element == Float {
    /// Convert the array using the vector type selected by `VectorTypeFactory` for its count.
    func toVector() -> any VectorType {
        try! VectorTypeFactory.vector(of: count, from: self)
    }
}

public extension Array where Element: VectorType {
    /// Compute pairwise distances between all vectors
    func pairwiseDistances<M: DistanceMetric>(
        metric: M = EuclideanDistance()
    ) async -> [[Float]] where Element: VectorProtocol & Sendable & VectorProtocol, Element.Scalar == Float, M.Scalar == Float {
        await BatchOperations.pairwiseDistances(self, metric: metric)
    }

    /// Find k nearest neighbors to a query vector
    func findNearest<M: DistanceMetric>(
        to query: Element,
        k: Int,
        metric: M = EuclideanDistance()
    ) async -> [(index: Int, distance: Float)]
    where Element: VectorProtocol & Sendable, Element.Scalar == Float, M.Scalar == Float {
        await BatchOperations.findNearest(to: query, in: self, k: k, metric: metric)
    }
}
