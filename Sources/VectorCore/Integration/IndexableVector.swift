//
//  IndexableVector.swift
//  VectorCore
//
//  Protocol for vectors that can be stored and searched in an index.
//  Extends VectorProtocol with requirements for index storage and retrieval.
//

import Foundation

// MARK: - IndexableVector Protocol

/// Protocol for vectors that can be stored in a search index.
///
/// Extends `VectorProtocol` with additional requirements for index storage,
/// retrieval, and advisory hints (`isNormalized`, `cachedMagnitude`).
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
/// All optimized vector types (Vector384Optimized, etc.), DynamicVector and
/// `Vector<D>` conform to this protocol. Custom vector types should implement:
/// - VectorProtocol requirements (core vector operations)
/// - Codable for serialization
/// - Optional: Override `isNormalized` and `cachedMagnitude` for optimization hints
///
/// Custom reference-semantic conformers must keep any advertised hints consistent
/// with their coordinates, including mutations through external aliases.
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
    /// Default implementation returns `false`. `Vector<D>` does not use the default: it answers with its own
    /// computed check (`abs(magnitudeSquared - 1) < 1e-6`, O(n) per read), so for `Vector<D>` this is a
    /// computed answer, not a stored hint (D02CosineAndWitnessTests).
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

/// Wrapper that carries normalization metadata alongside a vector.
///
/// `NormalizationHint` conforms to `IndexableVector`, so it can be passed
/// directly to any `<V: IndexableVector>` typed API. Downstream packages
/// (e.g. VectorIndex) read `isNormalized` and `cachedMagnitude` to skip
/// redundant normalization during cosine distance computation.
///
/// Writable access through this wrapper clears both hints before yielding access,
/// including no-write and throwing borrows (see `NormalizationHintMutationTests`).
/// A caller asserting normalization must supply a stable value or snapshot and
/// must not mutate untracked aliases while relying on the assertion. The wrapper
/// cannot observe external aliases in custom reference-semantic conformers.
///
/// Decoding preserves the vector but clears advisory hints without computing its
/// magnitude. Consequently a Codable round trip may change equality and hashing,
/// which include `isNormalized`. Rebuild an explicit trusted hint if needed.
///
/// ```swift
/// let normalized = myVector.normalizedUnchecked()
/// let hint = NormalizationHint(vector: normalized, isNormalized: true)
/// index.insert(id: "doc1", vector: hint)  // the index may compute cosine as a dot product
/// ```
public struct NormalizationHint<V: IndexableVector> {
    private var _vector: V

    /// The wrapped vector. Writable access clears normalization metadata.
    public var vector: V {
        get { _vector }
        set {
            invalidateHints()
            _vector = newValue
        }
        _modify {
            invalidateHints()
            yield &_vector
        }
    }

    /// Whether the vector is pre-normalized (magnitude ≈ 1).
    ///
    /// Reset to `false` before writable access through this wrapper.
    public private(set) var isNormalized: Bool

    /// Internal cache for magnitude value
    private var _cachedMagnitude: Float?

    /// Magnitude of the vector, using cached value when available.
    ///
    /// Returns `1.0` when `isNormalized` is true, the cached value if
    /// available, or computes from the underlying vector.
    public var magnitude: Float {
        if isNormalized { return 1.0 }
        if let cached = _cachedMagnitude { return cached }
        return vector.magnitude
    }

    /// Clear metadata before publishing mutable access.
    private mutating func invalidateHints() {
        isNormalized = false
        _cachedMagnitude = nil
    }

    // MARK: - Initializers

    /// Initialize with caller-asserted normalization status; no numerical validation is performed.
    public init(vector: V, isNormalized: Bool) {
        self._vector = vector
        self.isNormalized = isNormalized
        self._cachedMagnitude = isNormalized ? 1.0 : nil
    }

    /// Initialize with auto-detection of normalization.
    ///
    /// Computes the vector's magnitude and caches it. If the magnitude
    /// is within 0.001 of 1.0, marks the vector as normalized.
    public init(vector: V) {
        self._vector = vector
        let mag = vector.magnitude
        self.isNormalized = abs(mag - 1.0) < 0.001
        self._cachedMagnitude = mag
    }

    /// Create from a vector known to be normalized.
    public static func normalized(_ vector: V) -> NormalizationHint<V> {
        NormalizationHint(vector: vector, isNormalized: true)
    }
}

// MARK: - VectorProtocol Conformance

extension NormalizationHint: VectorProtocol {
    public typealias Scalar = Float
    public typealias Storage = V.Storage

    public var storage: Storage {
        @inlinable get { vector.storage }
        set { vector.storage = newValue }
        _modify {
            invalidateHints()
            yield &_vector.storage
        }
    }

    @inlinable
    public var scalarCount: Int { vector.scalarCount }

    public init() {
        self._vector = V()
        self.isNormalized = false
        self._cachedMagnitude = nil
    }

    public init(_ array: [Scalar]) throws {
        self._vector = try V(array)
        self.isNormalized = false
        self._cachedMagnitude = nil
    }

    public init(repeating value: Scalar) {
        self._vector = V(repeating: value)
        self.isNormalized = false
        self._cachedMagnitude = nil
    }

    @inlinable
    public func toArray() -> [Scalar] {
        vector.toArray()
    }

    @inlinable
    public func withUnsafeBufferPointer<R>(
        _ body: (UnsafeBufferPointer<Scalar>) throws -> R
    ) rethrows -> R {
        try vector.withUnsafeBufferPointer(body)
    }

    public mutating func withUnsafeMutableBufferPointer<R>(
        _ body: (UnsafeMutableBufferPointer<Scalar>) throws -> R
    ) rethrows -> R {
        try vector.withUnsafeMutableBufferPointer(body)
    }

    @inlinable
    public var isFinite: Bool { vector.isFinite }

    @inlinable
    public var isZero: Bool { vector.isZero }
}

// MARK: - Collection Conformance

extension NormalizationHint {
    @inlinable
    public subscript(index: Int) -> Scalar {
        precondition(index >= 0 && index < scalarCount, "Index \(index) out of bounds")
        return vector[index]
    }
}

// MARK: - Equatable & Hashable

extension NormalizationHint: Equatable {
    public static func == (lhs: NormalizationHint, rhs: NormalizationHint) -> Bool {
        lhs.vector == rhs.vector && lhs.isNormalized == rhs.isNormalized
    }
}

extension NormalizationHint: Hashable {
    public func hash(into hasher: inout Hasher) {
        vector.hash(into: &hasher)
        hasher.combine(isNormalized)
    }
}

// MARK: - Codable

extension NormalizationHint: Codable {
    private enum CodingKeys: String, CodingKey {
        case vector
        case isNormalized
        case cachedMagnitude
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self._vector = try container.decode(V.self, forKey: .vector)
        self.isNormalized = false
        self._cachedMagnitude = nil
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(vector, forKey: .vector)
        try container.encode(isNormalized, forKey: .isNormalized)
        try container.encodeIfPresent(_cachedMagnitude, forKey: .cachedMagnitude)
    }
}

// MARK: - IndexableVector Conformance

extension NormalizationHint: IndexableVector {
    /// Cached magnitude: returns 1.0 if normalized, stored cache otherwise.
    public var cachedMagnitude: Float? {
        if isNormalized { return 1.0 }
        return _cachedMagnitude
    }
}

// MARK: - Double-Wrapping Prevention

extension NormalizationHint {
    /// Returns self instead of wrapping in another NormalizationHint layer.
    public func withNormalizationHint() -> NormalizationHint<V> {
        self
    }

    /// Returns a re-hinted copy with normalized status.
    public func normalizedWithHint() -> NormalizationHint<V>? {
        guard let norm = try? vector.normalized().get(), hasUnitFloatCoordinates(norm) else { return nil }
        return NormalizationHint(vector: norm, isNormalized: true)
    }

    /// Returns a re-hinted copy with normalized status (unchecked).
    public func normalizedUncheckedWithHint() -> NormalizationHint<V> {
        let normalized = vector.normalizedUnchecked()
        return NormalizationHint(vector: normalized, isNormalized: hasUnitFloatCoordinates(normalized))
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
        guard let normalized = try? normalized().get(), hasUnitFloatCoordinates(normalized) else { return nil }
        return NormalizationHint(vector: normalized, isNormalized: true)
    }

    /// Create a normalized copy (unchecked) with hint.
    public func normalizedUncheckedWithHint() -> NormalizationHint<Self> {
        let normalized = normalizedUnchecked()
        return NormalizationHint(vector: normalized, isNormalized: hasUnitFloatCoordinates(normalized))
    }
}

// Validate the actual returned coordinates, including custom lossy writeback.
private func hasUnitFloatCoordinates<V: IndexableVector>(_ vector: V) -> Bool {
    vector.withUnsafeBufferPointer { buffer in
        let squares = NormalizationCore.floatSquares(count: buffer.count) { buffer[$0] }
        return squares.isFinite && abs(squares.squareRoot() - 1) <= 8 * Double(Float.ulpOfOne)
    }
}
