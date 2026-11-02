//
//  Vector1536Optimized.swift
//  VectorCore
//
//  1536-dimensional Float vector with SIMD4<Float> storage
//  Common dimension for large embedding models
//

import Foundation
import simd
import Accelerate

/// 1536-dimensional Float vector stored as 384 `SIMD4<Float>` lanes.
///
/// Memory layout:
/// - Stored as 384 SIMD4<Float> values in `_storage` (a ContiguousArray).
/// - Each element is aligned to `MemoryLayout<SIMD4<Float>>.alignment` (16 bytes on Apple platforms): a ContiguousArray
///   stores its elements at their type's alignment. No cache-line placement is promised.
///
public struct Vector1536Optimized: Sendable {
    public typealias Scalar = Float

    /// Internal storage: 384 SIMD4<Float> chunks holding the 1536 scalar lanes.
    @usableFromInline
    internal var _storage: ContiguousArray<SIMD4<Float>>

    /// SIMD storage. Replacement must contain exactly 384 SIMD4 elements.
    /// Inout mutation must preserve this count, including when unwinding a throw.
    @inlinable
    public var storage: ContiguousArray<SIMD4<Float>> {
        get { _storage }
        set {
            guard newValue.count == 384 else { fatalError("Storage must contain exactly 384 SIMD4 elements") }
            _storage = newValue
        }
        _modify {
            defer {
                guard _storage.count == 384 else { fatalError("Storage must contain exactly 384 SIMD4 elements") }
            }
            yield &_storage
        }
    }

    /// Number of scalar elements (always 1536)
    ///
    /// Computed, not stored (P02 RD7): the handle holds only `_storage`
    /// (`P02HandleLayoutTests.handleStrideIsTheStorageStride`).
    @inlinable
    public var scalarCount: Int { 1536 }

    // MARK: - Initialization

    /// Initialize with zeros
    @inlinable
    public init() {
        _storage = ContiguousArray(repeating: SIMD4<Float>(), count: 384)
    }

    /// Initialize with repeating value
    @inlinable
    public init(repeating value: Scalar) {
        let simd4 = SIMD4<Float>(repeating: value)
        _storage = ContiguousArray(repeating: simd4, count: 384)
    }

    /// Initialize from an array by copying scalar lanes
    @inlinable
    public init(_ array: [Scalar]) throws {
        guard array.count == 1536 else {
            throw VectorError.dimensionMismatch(expected: 1536, actual: array.count)
        }

        _storage = ContiguousArray<SIMD4<Float>>()
        _storage.reserveCapacity(384)

        // Copy scalar lanes without imposing SIMD alignment on the Float owner.
        for i in stride(from: 0, to: 1536, by: 4) {
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
        _storage.reserveCapacity(384)

        for i in stride(from: 0, to: 1536, by: 4) {
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
    /// - Precondition: Storage contains exactly 384 SIMD4 elements.
    @inlinable
    public init(fromSIMDStorage storage: ContiguousArray<SIMD4<Float>>) {
        guard storage.count == 384 else { fatalError("Storage must contain exactly 384 SIMD4 elements") }
        self._storage = storage
    }

    // MARK: - Element Access

    /// Access individual elements
    @inlinable
    public subscript(index: Int) -> Scalar {
        get {
            precondition(index >= 0 && index < 1536, "Index \(index) out of bounds [0..<1536]")
            let vectorIndex = index >> 2  // Divide by 4
            let scalarIndex = index & 3   // Modulo 4
            return _storage[vectorIndex][scalarIndex]
        }
        set {
            precondition(index >= 0 && index < 1536, "Index \(index) out of bounds [0..<1536]")
            let vectorIndex = index >> 2
            let scalarIndex = index & 3
            _storage[vectorIndex][scalarIndex] = newValue
        }
    }
}

// MARK: - VectorProtocol Conformance

extension Vector1536Optimized: VectorProtocol {
    public typealias Storage = ContiguousArray<SIMD4<Float>>

    /// Convert to array
    public func toArray() -> [Scalar] {
        var result = [Scalar]()
        result.reserveCapacity(1536)

        _storage.withUnsafeBufferPointer { buffer in
            // Scoped rebinding restores the SIMD4<Float> binding on exit instead of permanently retyping the storage.
            // Strict-aliasing safety is not verified by a test; `allWidthScopedBorrows` checks only the
            // element-wise round trip.
            buffer.baseAddress!.withMemoryRebound(to: Float.self, capacity: 1536) { floatPtr in
                result.append(contentsOf: UnsafeBufferPointer(start: floatPtr, count: 1536))
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
            // Scoped rebinding restores the SIMD4<Float> binding on exit. Strict-aliasing safety is not verified by
            // a test; `allWidthScopedBorrows` checks only the element-wise round trip.
            try simd4Buffer.baseAddress!.withMemoryRebound(to: Float.self, capacity: 1536) { floatPtr in
                try body(UnsafeBufferPointer(start: floatPtr, count: 1536))
            }
        }
    }

    /// Borrow Float lanes for writing; the pointer must not escape the closure.
    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(
        _ body: (UnsafeMutableBufferPointer<Scalar>) throws -> R
    ) rethrows -> R {
        try _storage.withUnsafeMutableBufferPointer { simd4Buffer in
            // Scoped rebinding restores the SIMD4<Float> binding on exit. Strict-aliasing safety is not verified by
            // a test; `allWidthScopedBorrows` checks only the element-wise round trip.
            try simd4Buffer.baseAddress!.withMemoryRebound(to: Float.self, capacity: 1536) { floatPtr in
                try body(UnsafeMutableBufferPointer(start: floatPtr, count: 1536))
            }
        }
    }
}

// MARK: - Distance, Magnitude and Normalization

extension Vector1536Optimized {

    /// Dot product; delegates to `DotKernels.dot1536`.
    @inlinable
    @inline(__always)
    public func dotProduct(_ other: Vector1536Optimized) -> Scalar {
        DotKernels.dot1536(self, other)
    }

    /// Squared Euclidean distance; delegates to `EuclideanKernels.squared1536`.
    @inlinable
    @inline(__always)
    public func euclideanDistanceSquared(to other: Vector1536Optimized) -> Scalar {
        EuclideanKernels.squared1536(self, other)
    }

    /// Euclidean distance (with square root)
    @inlinable
    public func euclideanDistance(to other: Vector1536Optimized) -> Scalar {
        EuclideanKernels.distance1536(self, other)
    }

    /// Magnitude squared: the compensated Double sum of squares (`NormalizationCore.floatSquares`, through
    /// `NormalizeKernels.magnitudeSquared`) rounded once to Float. No test checks the rounding at this width;
    /// `OptimizedVector512Suite.testMagnitudeRoundsOnceFromTheWidenedSum` checks the single rounding
    /// on `Vector512Optimized` through the same kernel.
    ///
    /// Infinity is returned when the scalar squared norm exceeds Float range.
    @inlinable
    public var magnitudeSquared: Scalar {
        NormalizeKernels.magnitudeSquared(storage: _storage, laneCount: 384)
    }

    /// Magnitude (L2 norm): the square root of the widened (Double) sum of squares, rounded once to Float
    /// (`NormalizeKernels.magnitude`; no test checks the rounding at this width).
    ///
    /// Infinity is returned when the scalar norm exceeds Float range.
    @inlinable
    public var magnitude: Scalar {
        NormalizeKernels.magnitude(storage: _storage, laneCount: 384)
    }

    /// Normalized vector
    @inlinable
    public func normalized() -> Result<Vector1536Optimized, VectorError> {
        NormalizeKernels.normalized1536(self)
    }

    /// Normalized vector with a precondition instead of a Result error.
    ///
    /// Validates all coordinates before constructing the normalized copy.
    /// - Precondition: finite nonzero input (enforced in Debug and normal Release; the trap is pinned in a child process by
    ///   `NormalizationRangeContractTests.invalidInputsRejectInChildren` in the configuration the suite runs)
    /// - Warning: Invalid input deliberately traps before mutation.
    @inlinable
    public func normalizedUnchecked() -> Vector1536Optimized {
        NormalizeKernels.normalizedUnchecked1536(self)
    }

    /// Cosine similarity: 0 when either magnitude is zero or their product underflows to zero, NaN when either
    /// magnitude is NaN or infinite (T15CosineExceptionalTableTests.everyImplementationFollowsThePolicy, sim.opt rows)
    @inlinable
    public func cosineSimilarity(to other: Vector1536Optimized) -> Scalar {
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

extension Vector1536Optimized {

    /// Addition with SIMD
    @inlinable
    public static func + (lhs: Vector1536Optimized, rhs: Vector1536Optimized) -> Vector1536Optimized {
        var result = Vector1536Optimized()
        for i in 0..<384 {
            result._storage[i] = lhs._storage[i] + rhs._storage[i]
        }
        return result
    }

    /// Subtraction with SIMD
    @inlinable
    public static func - (lhs: Vector1536Optimized, rhs: Vector1536Optimized) -> Vector1536Optimized {
        var result = Vector1536Optimized()
        for i in 0..<384 {
            result._storage[i] = lhs._storage[i] - rhs._storage[i]
        }
        return result
    }

    /// Scalar multiplication
    @inlinable
    public static func * (lhs: Vector1536Optimized, rhs: Scalar) -> Vector1536Optimized {
        var result = Vector1536Optimized()
        let scalar = SIMD4<Float>(repeating: rhs)
        for i in 0..<384 {
            result._storage[i] = lhs._storage[i] * scalar
        }
        return result
    }

    /// Scalar division
    @inlinable
    public static func / (lhs: Vector1536Optimized, rhs: Scalar) -> Vector1536Optimized {
        guard rhs != 0 else { fatalError("Division by zero") }
        var result = lhs
        result.inPlaceDivide(rhs)
        return result
    }

    /// Element-wise multiplication (Hadamard product)
    @inlinable
    public static func .* (lhs: Vector1536Optimized, rhs: Vector1536Optimized) -> Vector1536Optimized {
        var result = Vector1536Optimized()
        for i in 0..<384 {
            result._storage[i] = lhs._storage[i] * rhs._storage[i]
        }
        return result
    }
}

// MARK: - Collection Conformance

extension Vector1536Optimized: Collection {
    public typealias Index = Int
    public typealias Element = Scalar

    public var startIndex: Int { 0 }
    public var endIndex: Int { 1536 }

    public func index(after i: Int) -> Int {
        i + 1
    }
}

// MARK: - Equatable & Hashable

extension Vector1536Optimized: Equatable {
    @inlinable
    public static func == (lhs: Vector1536Optimized, rhs: Vector1536Optimized) -> Bool {
        for i in 0..<384 {
            if lhs._storage[i] != rhs._storage[i] {
                return false
            }
        }
        return true
    }
}

extension Vector1536Optimized: Hashable {
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
        hasher.combine(1536)
    }
}

// MARK: - Codable

extension Vector1536Optimized: Codable {
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

extension Vector1536Optimized: CustomDebugStringConvertible {
    public var debugDescription: String {
        let elements = self.prefix(10).map { String(format: "%.4f", $0) }
        return "Vector1536Optimized[\(elements.joined(separator: ", ")), ... (1536 total)]"
    }
}

// MARK: - OptimizedVector Conformance

extension Vector1536Optimized: OptimizedVector {
    public static var laneCount: Int { 4 }
}

// MARK: - VectorType Conformance

extension Vector1536Optimized: VectorType {}

// MARK: - AnalyzableVector Conformance

extension Vector1536Optimized: MixedPrecisionKernels.AnalyzableVector {}

// MARK: - Arithmetic Extensions for Streaming K-means

extension Vector1536Optimized {
    /// Add another vector element-wise
    @inlinable
    public func adding(_ other: Vector1536Optimized) -> Vector1536Optimized {
        var result = ContiguousArray<SIMD4<Float>>()
        result.reserveCapacity(384)

        for i in 0..<384 {
            result.append(_storage[i] + other._storage[i])
        }

        return Vector1536Optimized(fromSIMDStorage: result)
    }

    /// Subtract another vector element-wise
    @inlinable
    public func subtracting(_ other: Vector1536Optimized) -> Vector1536Optimized {
        var result = ContiguousArray<SIMD4<Float>>()
        result.reserveCapacity(384)

        for i in 0..<384 {
            result.append(_storage[i] - other._storage[i])
        }

        return Vector1536Optimized(fromSIMDStorage: result)
    }

    /// Scale by a scalar value
    @inlinable
    public func scaled(by scalar: Float) -> Vector1536Optimized {
        var result = ContiguousArray<SIMD4<Float>>()
        result.reserveCapacity(384)

        let simdScalar = SIMD4<Float>(repeating: scalar)
        for i in 0..<384 {
            result.append(_storage[i] * simdScalar)
        }

        return Vector1536Optimized(fromSIMDStorage: result)
    }

    /// Static zero vector
    @inlinable
    public static var zero: Vector1536Optimized {
        Vector1536Optimized()
    }
}
