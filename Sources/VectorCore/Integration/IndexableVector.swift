//
//  IndexableVector.swift
//  VectorCore
//
//  Protocol for vectors that can be efficiently stored and searched in an index.
//  Extends VectorProtocol with requirements for index storage and retrieval.
//

import Foundation

// MARK: - IndexableVector Protocol

/// Protocol for vectors that can be stored in a search index.
///
/// Extends `VectorProtocol` with additional requirements for efficient
/// index storage, retrieval, and optimization hints.
///
/// ## Design Philosophy
///
/// `IndexableVector` provides a contract between VectorCore (vector operations)
/// and VectorIndex (index structures). It defines:
/// - What information the index can cache (magnitude, normalization status)
/// - How vectors are identified and retrieved
/// - Serialization requirements for persistence
///
/// ## Conformance
///
/// All optimized vector types (Vector384Optimized, etc.) and DynamicVector
/// conform to this protocol. Custom vector types should implement:
/// - VectorProtocol requirements (core vector operations)
/// - Codable for serialization
/// - Optional: Override `isNormalized` and `cachedMagnitude` for optimization hints
///
/// ## Example
/// ```swift
/// // VectorIndex can use these hints for optimization
/// if vector.isNormalized {
///     // Use dot product directly for cosine similarity
///     let similarity = query.dotProduct(vector)
/// } else {
///     // Compute full cosine similarity
///     let similarity = query.cosineSimilarity(to: vector)
/// }
/// ```
public protocol IndexableVector: VectorProtocol, Codable, Sendable where Scalar == Float {
    /// Whether this vector is pre-normalized (magnitude ≈ 1).
    ///
    /// When true, indexes can use optimized cosine similarity computation
    /// via dot product: `cos(θ) = a·b` when `||a|| = ||b|| = 1`.
    ///
    /// Default implementation returns `false`.
    var isNormalized: Bool { get }

    /// Cached magnitude value, if available.
    ///
    /// Indexes may cache this value to avoid recomputation.
    /// Return `nil` if magnitude has not been computed.
    ///
    /// Default implementation returns `nil`.
    var cachedMagnitude: Float? { get }
}

// MARK: - Default Implementation

extension IndexableVector {
    /// Default: Vector is not pre-normalized
    public var isNormalized: Bool { false }

    /// Default: No cached magnitude
    public var cachedMagnitude: Float? { nil }
}

// MARK: - Optimized Vector Conformance

extension Vector384Optimized: IndexableVector {}
extension Vector512Optimized: IndexableVector {}
extension Vector768Optimized: IndexableVector {}
extension Vector1536Optimized: IndexableVector {}

// MARK: - DynamicVector Conformance

extension DynamicVector: IndexableVector {}

// MARK: - Generic Vector Conformance

extension Vector: IndexableVector where D: StaticDimension {}

// MARK: - Normalized Vector Hint

/// Struct to track normalization status alongside a vector.
///
/// Use this when you want to track whether vectors have been pre-normalized
/// without modifying the vector type itself.
///
/// ```swift
/// let hint = NormalizationHint(vector: myVector, isNormalized: true)
/// if hint.isNormalized {
///     // Use dot product for cosine similarity
/// }
/// ```
public struct NormalizationHint<V: IndexableVector>: Sendable {
    /// The vector
    public let vector: V

    /// Whether the vector is pre-normalized
    public let isNormalized: Bool

    /// Cached magnitude (1.0 if normalized)
    public var magnitude: Float {
        isNormalized ? 1.0 : vector.magnitude
    }

    /// Initialize with explicit normalization status
    public init(vector: V, isNormalized: Bool) {
        self.vector = vector
        self.isNormalized = isNormalized
    }

    /// Initialize with auto-detection of normalization
    public init(vector: V) {
        self.vector = vector
        self.isNormalized = abs(vector.magnitude - 1.0) < 0.001
    }

    /// Create from a normalized vector
    public static func normalized(_ vector: V) -> NormalizationHint<V> {
        NormalizationHint(vector: vector, isNormalized: true)
    }
}

// MARK: - Convenience Extensions

extension IndexableVector {
    /// Check if this vector is approximately normalized (magnitude ≈ 1).
    public var isApproximatelyNormalized: Bool {
        abs(magnitude - 1.0) < 0.001
    }

    /// Create a normalization hint for this vector.
    public func withNormalizationHint() -> NormalizationHint<Self> {
        NormalizationHint(vector: self)
    }

    /// Create a normalized copy with hint.
    public func normalizedWithHint() -> NormalizationHint<Self>? {
        guard let normalized = try? normalized().get() else { return nil }
        return NormalizationHint(vector: normalized, isNormalized: true)
    }

    /// Create a normalized copy (unchecked) with hint.
    public func normalizedUncheckedWithHint() -> NormalizationHint<Self> {
        NormalizationHint(vector: normalizedUnchecked(), isNormalized: true)
    }
}
