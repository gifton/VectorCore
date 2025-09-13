// VectorCore
//
// Modern, high-performance vector operations for Swift
//

import Foundation

// MARK: - Core Type Aliases

/// Vector identifier type
public typealias VectorID = String

/// High-precision timestamp
public typealias Timestamp = UInt64

// Type aliases moved to Types/TypeAliases.swift to avoid duplication

// MARK: - VectorCore Namespace

/// Main entry point for VectorCore functionality
public enum VectorCore {

    /// Create a vector of appropriate type based on dimension
    ///
    /// Automatically selects the optimal storage type for the given dimension.
    /// Common dimensions (128, 256, 512, 768, 1536, 3072) use optimized SIMD storage.
    ///
    /// - Parameters:
    ///   - dimension: The desired vector dimension
    ///   - data: Optional initial values (defaults to zero vector)
    /// - Returns: Optimized vector for the specified dimension
    public static func createVector(dimension: Int, data: [Float]? = nil) -> any VectorType {
        if let data = data {
            return try! VectorTypeFactory.vector(of: dimension, from: data)
        } else {
            return VectorTypeFactory.zeros(dimension: dimension)
        }
    }

    /// Create a batch of vectors from a 2D array
    ///
    /// - Parameters:
    ///   - dimension: Dimension of each vector
    ///   - data: 2D array where each inner array represents a vector
    /// - Returns: Array of optimized vectors
    /// - Throws: VectorCoreError if any vector has incorrect dimension
    public static func createBatch(dimension: Int, from data: [[Float]]) throws -> [any VectorType] {
        try data.map { values in
            try VectorTypeFactory.vector(of: dimension, from: values)
        }
    }

    /// Global configuration for VectorCore behavior
    public struct Configuration: Sendable {
        /// Batch processing configuration
        public var batchOperations = BatchOperations.Configuration()

        /// Memory alignment for optimal SIMD performance
        public var memoryAlignment: Int = 64

        public init() {}
    }

    /// Global configuration instance
    public static let configuration = Configuration()

    /// Get version information
    public static var version: String {
        VectorCoreVersion.versionString
    }

    /// Check if a dimension has optimized SIMD support
    public static func hasOptimizedSupport(for dimension: Int) -> Bool {
        VectorTypeFactory.isSupported(dimension: dimension)
    }

    /// Get the nearest optimized dimension
    public static func optimalDimension(for size: Int) -> Int {
        VectorTypeFactory.optimalDimension(for: size)
    }
}

// MARK: - Convenience Extensions

public extension Array where Element == Float {
    /// Convert array to optimized vector
    func toVector() -> any VectorType {
        VectorCore.createVector(dimension: count, data: self)
    }
}

public extension Array where Element: VectorType {
    /// Compute pairwise distances between all vectors
    func pairwiseDistances<M: DistanceMetric>(
        metric: M = EuclideanDistance()
    ) async -> [[Float]]
    where Element: VectorProtocol & Sendable,
          Element.Scalar == Float,
          M.Scalar == Float {
        await BatchOperations.pairwiseDistances(self, metric: metric)
    }

    /// Find k nearest neighbors to a query vector
    func findNearest<M: DistanceMetric>(
        to query: Element,
        k: Int,
        metric: M = EuclideanDistance()
    ) async -> [(index: Int, distance: Float)]
    where Element: VectorProtocol & Sendable,
          Element.Scalar == Float,
          M.Scalar == Float {
        await BatchOperations.findNearest(to: query, in: self, k: k, metric: metric)
    }
}
