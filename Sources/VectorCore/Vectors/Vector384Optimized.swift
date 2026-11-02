//
//  Vector384Optimized.swift
//  VectorCore
//
//  384-dimensional Float vector with SIMD4<Float> storage
//  Standard dimension for MiniLM and Sentence-BERT models
//

import Foundation
import simd
import Accelerate

/// 384-dimensional Float vector stored as 96 `SIMD4<Float>` lanes.
///
/// This is the standard dimension for MiniLM models (all-MiniLM-L6-v2) and
/// many Sentence-BERT variants, making it critical for EmbedKit integration.
///
/// Memory layout:
/// - Stored as 96 SIMD4<Float> values in `_storage` (a ContiguousArray).
/// - Each element is aligned to `MemoryLayout<SIMD4<Float>>.alignment` (16 bytes on Apple platforms): a ContiguousArray
///   stores its elements at their type's alignment. No cache-line placement is promised.
///
public struct Vector384Optimized: Sendable {
    public typealias Scalar = Float

    /// Internal storage: 96 SIMD4<Float> chunks holding the 384 scalar lanes.
    @usableFromInline
    internal var _storage: ContiguousArray<SIMD4<Float>>

    /// SIMD storage. Replacement must contain exactly 96 SIMD4 elements.
    /// Inout mutation must preserve this count, including when unwinding a throw.
    @inlinable
    public var storage: ContiguousArray<SIMD4<Float>> {
        get { _storage }
        set {
            guard newValue.count == 96 else { fatalError("Storage must contain exactly 96 SIMD4 elements") }
            _storage = newValue
        }
        _modify {
            defer {
                guard _storage.count == 96 else { fatalError("Storage must contain exactly 96 SIMD4 elements") }
            }
            yield &_storage
        }
    }

    /// Number of scalar elements (always 384)
    ///
    /// Computed, not stored (P02 RD7): the handle holds only `_storage`
    /// (`P02HandleLayoutTests.handleStrideIsTheStorageStride`).
    @inlinable
    public var scalarCount: Int { 384 }

    // MARK: - Initialization

    /// Initialize with zeros
    @inlinable
    public init() {
        _storage = ContiguousArray(repeating: SIMD4<Float>(), count: 96)
    }

    /// Initialize with repeating value
    @inlinable
    public init(repeating value: Scalar) {
        let simd4 = SIMD4<Float>(repeating: value)
        _storage = ContiguousArray(repeating: simd4, count: 96)
    }

    /// Initialize from an array by copying scalar lanes
    @inlinable
    public init(_ array: [Scalar]) throws {
        guard array.count == 384 else {
            throw VectorError.dimensionMismatch(expected: 384, actual: array.count)
        }

        _storage = ContiguousArray<SIMD4<Float>>()
        _storage.reserveCapacity(96)

        // Copy scalar lanes without imposing SIMD alignment on the Float owner.
        for i in stride(from: 0, to: 384, by: 4) {
            _storage.append(SIMD4<Float>(array[i], array[i + 1], array[i + 2], array[i + 3]))
        }
    }

    /// Initialize from sequence
    @inlinable
    public init<S: Sequence>(_ sequence: S) throws where S.Element == Scalar {
        let array = Array(sequence)
        try self.init(array)
    }

    /// Initialize with generator function
    @inlinable
    public init(generator: (Int) throws -> Scalar) rethrows {
        _storage = ContiguousArray<SIMD4<Float>>()
        _storage.reserveCapacity(96)

        for i in stride(from: 0, to: 384, by: 4) {
            let simd4 = try SIMD4<Float>(
                generator(i),
                generator(i + 1),
                generator(i + 2),
                generator(i + 3)
            )
            _storage.append(simd4)
        }
    }

    /// Initialize from SIMD storage directly
    /// - Precondition: Storage contains exactly 96 SIMD4 elements.
    @inlinable
    public init(fromSIMDStorage storage: ContiguousArray<SIMD4<Float>>) {
        guard storage.count == 96 else { fatalError("Storage must contain exactly 96 SIMD4 elements") }
        self._storage = storage
    }

    // MARK: - Element Access

    /// Access individual elements
    @inlinable
    public subscript(index: Int) -> Scalar {
        get {
            precondition(index >= 0 && index < 384, "Index \(index) out of bounds [0..<384]")
            let vectorIndex = index >> 2  // Divide by 4
            let scalarIndex = index & 3   // Modulo 4
            return _storage[vectorIndex][scalarIndex]
        }
        set {
            precondition(index >= 0 && index < 384, "Index \(index) out of bounds [0..<384]")
            let vectorIndex = index >> 2
            let scalarIndex = index & 3
            _storage[vectorIndex][scalarIndex] = newValue
        }
    }
}

// MARK: - VectorProtocol Conformance

extension Vector384Optimized: VectorProtocol {
    public typealias Storage = ContiguousArray<SIMD4<Float>>

    /// Convert to array
    public func toArray() -> [Scalar] {
        var result = [Scalar]()
        result.reserveCapacity(384)

        _storage.withUnsafeBufferPointer { buffer in
            buffer.baseAddress!.withMemoryRebound(to: Float.self, capacity: 384) { floatBuffer in
                result.append(contentsOf: UnsafeBufferPointer(start: floatBuffer, count: 384))
            }
        }

        return result
    }

    /// Borrow Float lanes for reading; the pointer must not escape the closure.
    @inlinable
    public func withUnsafeBufferPointer<R>(
        _ body: (UnsafeBufferPointer<Scalar>) throws -> R
    ) rethrows -> R {
        try _storage.withUnsafeBufferPointer { simd4Buffer in
            return try simd4Buffer.baseAddress!.withMemoryRebound(to: Float.self, capacity: 384) { floatBuffer in
                try body(UnsafeBufferPointer(start: floatBuffer, count: 384))
            }
        }
    }

    /// Borrow Float lanes for writing; the pointer must not escape the closure.
    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(
        _ body: (UnsafeMutableBufferPointer<Scalar>) throws -> R
    ) rethrows -> R {
        try _storage.withUnsafeMutableBufferPointer { simd4Buffer in
            return try simd4Buffer.baseAddress!.withMemoryRebound(to: Float.self, capacity: 384) { floatBuffer in
                try body(UnsafeMutableBufferPointer(start: floatBuffer, count: 384))
            }
        }
    }
}

// MARK: - Distance, Magnitude and Normalization

extension Vector384Optimized {

    /// Dot product; delegates to `DotKernels.dot384`.
    @inlinable
    @inline(__always)
    public func dotProduct(_ other: Vector384Optimized) -> Scalar {
        DotKernels.dot384(self, other)
    }

    /// Squared Euclidean distance; delegates to `EuclideanKernels.squared384`.
    @inlinable
    @inline(__always)
    public func euclideanDistanceSquared(to other: Vector384Optimized) -> Scalar {
        EuclideanKernels.squared384(self, other)
    }

    /// Euclidean distance (with square root)
    @inlinable
    public func euclideanDistance(to other: Vector384Optimized) -> Scalar {
        EuclideanKernels.distance384(self, other)
    }

    /// Magnitude squared: the compensated Double sum of squares (`NormalizationCore.floatSquares`, through
    /// `NormalizeKernels.magnitudeSquared`) rounded once to Float. `NormalizationRangeContractTests.halfWideningAndScalarMagnitude` checks the bits at the smallest Float
    /// magnitudes on this type; `OptimizedVector512Suite.testMagnitudeRoundsOnceFromTheWidenedSum` checks the single rounding
    /// on `Vector512Optimized` through the same kernel.
    ///
    /// Infinity is returned when the scalar squared norm exceeds Float range.
    @inlinable
    public var magnitudeSquared: Scalar {
        NormalizeKernels.magnitudeSquared(storage: _storage, laneCount: 96)
    }

    /// Magnitude (L2 norm): the square root of the widened (Double) sum of squares, rounded once to Float
    /// (`NormalizeKernels.magnitude`; `NormalizationRangeContractTests.halfWideningAndScalarMagnitude` checks the
    /// widening at the smallest Float magnitudes on this type).
    ///
    /// Infinity is returned when the scalar norm exceeds Float range.
    @inlinable
    public var magnitude: Scalar {
        NormalizeKernels.magnitude(storage: _storage, laneCount: 96)
    }

    /// Normalized vector
    @inlinable
    public func normalized() -> Result<Vector384Optimized, VectorError> {
        NormalizeKernels.normalized384(self)
    }

    /// Normalized vector with a precondition instead of a Result error.
    ///
    /// Validates all coordinates before constructing the normalized copy.
    /// - Precondition: finite nonzero input (enforced in Debug and normal Release; the trap is pinned in a child process by
    ///   `NormalizationRangeContractTests.invalidInputsRejectInChildren` in the configuration the suite runs)
    /// - Warning: Invalid input deliberately traps before mutation.
    @inlinable
    public func normalizedUnchecked() -> Vector384Optimized {
        NormalizeKernels.normalizedUnchecked384(self)
    }

    /// Cosine similarity: 0 when either magnitude is zero or their product underflows to zero, NaN when either
    /// magnitude is NaN or infinite (T15CosineExceptionalTableTests.everyImplementationFollowsThePolicy, sim.opt rows)
    @inlinable
    public func cosineSimilarity(to other: Vector384Optimized) -> Scalar {
        let dot = dotProduct(other)
        let magSelf = magnitude
        let magOther = other.magnitude

        guard magSelf.isFinite && magOther.isFinite else { return .nan }
        guard magSelf > 0 && magOther > 0 else { return 0 }
        let denominator = magSelf * magOther
        guard denominator > 0 else { return 0 }
        return dot / denominator
    }
}

// MARK: - Arithmetic Operations

extension Vector384Optimized {

    /// Addition with SIMD
    @inlinable
    public static func + (lhs: Vector384Optimized, rhs: Vector384Optimized) -> Vector384Optimized {
        var result = Vector384Optimized()
        for i in 0..<96 {
            result._storage[i] = lhs._storage[i] + rhs._storage[i]
        }
        return result
    }

    /// Subtraction with SIMD
    @inlinable
    public static func - (lhs: Vector384Optimized, rhs: Vector384Optimized) -> Vector384Optimized {
        var result = Vector384Optimized()
        for i in 0..<96 {
            result._storage[i] = lhs._storage[i] - rhs._storage[i]
        }
        return result
    }

    /// Scalar multiplication
    @inlinable
    public static func * (lhs: Vector384Optimized, rhs: Scalar) -> Vector384Optimized {
        var result = Vector384Optimized()
        let scalar = SIMD4<Float>(repeating: rhs)
        for i in 0..<96 {
            result._storage[i] = lhs._storage[i] * scalar
        }
        return result
    }

    /// Scalar division
    @inlinable
    public static func / (lhs: Vector384Optimized, rhs: Scalar) -> Vector384Optimized {
        guard rhs != 0 else { fatalError("Division by zero") }
        var result = lhs
        result.inPlaceDivide(rhs)
        return result
    }

    /// Element-wise multiplication (Hadamard product)
    @inlinable
    public static func .* (lhs: Vector384Optimized, rhs: Vector384Optimized) -> Vector384Optimized {
        var result = Vector384Optimized()
        for i in 0..<96 {
            result._storage[i] = lhs._storage[i] * rhs._storage[i]
        }
        return result
    }
}

// MARK: - Collection Conformance

extension Vector384Optimized: Collection {
    public typealias Index = Int
    public typealias Element = Scalar

    public var startIndex: Int { 0 }
    public var endIndex: Int { 384 }

    public func index(after i: Int) -> Int {
        i + 1
    }
}

// MARK: - Equatable & Hashable

extension Vector384Optimized: Equatable {
    @inlinable
    public static func == (lhs: Vector384Optimized, rhs: Vector384Optimized) -> Bool {
        for i in 0..<96 {
            if lhs._storage[i] != rhs._storage[i] {
                return false
            }
        }
        return true
    }
}

extension Vector384Optimized: Hashable {
    /// Hashes every SIMD4 storage block in index order, then the fixed dimension.
    ///
    /// Each block is hashed through `SIMD4<Float>`'s `Hashable` conformance, which hashes lanes as `Float` values,
    /// so `+0` and `-0` hash equally, as `==` requires. Raw bit patterns are not hashed.
    /// Tests: `OptimizedHashContractTests` (block coverage, signed zero, `Set` and `Dictionary` lookup).
    ///
    /// The cost is O(D) per call, one pass over D/4 blocks, and the hash is not cached.
    ///
    /// Hash values are seeded per process and are not persistent identifiers.
    /// Unequal values may still collide; no collision-free promise is made.
    public func hash(into hasher: inout Hasher) {
        for block in _storage {
            hasher.combine(block)
        }
        hasher.combine(384)
    }
}

// MARK: - Codable

extension Vector384Optimized: Codable {
    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        let array = try container.decode([Float].self)
        try self.init(array)
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(toArray())
    }
}

// MARK: - Debug Support

extension Vector384Optimized: CustomDebugStringConvertible {
    public var debugDescription: String {
        let elements = self.prefix(10).map { String(format: "%.4f", $0) }
        return "Vector384Optimized[\(elements.joined(separator: ", ")), ... (384 total)]"
    }
}

// MARK: - OptimizedVector Conformance

extension Vector384Optimized: OptimizedVector {
    public static var laneCount: Int { 4 }
}

// MARK: - VectorType Conformance

extension Vector384Optimized: VectorType {}

// MARK: - AnalyzableVector Conformance

extension Vector384Optimized: MixedPrecisionKernels.AnalyzableVector {}

// MARK: - Arithmetic Extensions for Streaming K-means

extension Vector384Optimized {
    /// Add another vector element-wise
    @inlinable
    public func adding(_ other: Vector384Optimized) -> Vector384Optimized {
        var result = ContiguousArray<SIMD4<Float>>()
        result.reserveCapacity(96)

        for i in 0..<96 {
            result.append(_storage[i] + other._storage[i])
        }

        return Vector384Optimized(fromSIMDStorage: result)
    }

    /// Subtract another vector element-wise
    @inlinable
    public func subtracting(_ other: Vector384Optimized) -> Vector384Optimized {
        var result = ContiguousArray<SIMD4<Float>>()
        result.reserveCapacity(96)

        for i in 0..<96 {
            result.append(_storage[i] - other._storage[i])
        }

        return Vector384Optimized(fromSIMDStorage: result)
    }

    /// Scale by a scalar value
    @inlinable
    public func scaled(by scalar: Float) -> Vector384Optimized {
        var result = ContiguousArray<SIMD4<Float>>()
        result.reserveCapacity(96)

        let simdScalar = SIMD4<Float>(repeating: scalar)
        for i in 0..<96 {
            result.append(_storage[i] * simdScalar)
        }

        return Vector384Optimized(fromSIMDStorage: result)
    }

    /// Static zero vector
    @inlinable
    public static var zero: Vector384Optimized {
        Vector384Optimized()
    }
}
