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

// MARK: - VectorCore Namespace

/// Main entry point for VectorCore functionality
public enum VectorCore {

    /// Create a vector of the type `VectorTypeFactory` picks for `dimension`
    ///
    /// Delegates to `VectorTypeFactory.vector(of:from:)` (or `zeros(dimension:)` without data):
    /// `Vector<DimN>` for 128, 256, 384, 512, 768, 1536 and 3072, `DynamicVector` otherwise.
    ///
    /// - Precondition: `dimension >= 0`, and `data`, when given, has exactly `dimension` elements.
    ///   This entry does not throw: a malformed dimension or count stops the process. With `data`, it force-tries
    ///   the checked factory, which throws; without `data`, `zeros(dimension:)` stops at its own `fatalError` for a
    ///   negative dimension. Use `VectorTypeFactory.vector(of:from:)`, which throws `VectorError`
    ///   (`.invalidDimension`, `.dimensionMismatch`), to handle malformed input.
    /// - Parameters:
    ///   - dimension: The desired vector dimension
    ///   - data: Optional initial values (defaults to zero vector)
    /// - Returns: `Vector<DimN>` or `DynamicVector`, as above
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
    /// - Returns: One vector per row, of the type `createVector(dimension:data:)` picks for `dimension`
    /// - Throws: The `VectorError` of `VectorTypeFactory.vector(of:from:)` for the first malformed row
    public static func createBatch(dimension: Int, from data: [[Float]]) throws -> [any VectorType] {
        try data.map { values in
            try VectorTypeFactory.vector(of: dimension, from: values)
        }
    }

    /// Get version information
    public static var version: String {
        VectorCoreVersion.versionString
    }

    /// Whether `VectorTypeFactory` maps `dimension` to a fixed-width `Vector<DimN>`
    /// (128, 256, 384, 512, 768, 1536 or 3072) instead of `DynamicVector`; the same answer as
    /// `VectorTypeFactory.isSupported(dimension:)`. It does not refer to the `Vector*Optimized` types,
    /// and no speed difference is measured (not verified).
    public static func hasOptimizedSupport(for dimension: Int) -> Bool {
        VectorTypeFactory.isSupported(dimension: dimension)
    }

    /// The `VectorTypeFactory.isSupported(dimension:)` width nearest to `size` (`VectorTypeFactory.optimalDimension(for:)`)
    public static func optimalDimension(for size: Int) -> Int {
        VectorTypeFactory.optimalDimension(for: size)
    }
}

// MARK: - Convenience Extensions

public extension Array where Element == Float {
    /// Convert the array with `VectorCore.createVector` (same preconditions)
    func toVector() -> any VectorType {
        VectorCore.createVector(dimension: count, data: self)
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
