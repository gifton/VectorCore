//
//  Vector512Optimized.swift
//  VectorCore
//
//  512-dimensional Float vector with SIMD4<Float> storage
//  Ported from VectorStoreKit monolith with adaptations for VectorCore
//

import Foundation
import simd
import Accelerate

/// 512-dimensional Float vector stored as 128 `SIMD4<Float>` lanes.
///
/// Memory layout:
/// - Stored as 128 SIMD4<Float> values in `_storage` (a ContiguousArray).
/// - Each element is aligned to `MemoryLayout<SIMD4<Float>>.alignment` (16 bytes on Apple platforms): a ContiguousArray
///   stores its elements at their type's alignment. No cache-line placement is promised.
///
public struct Vector512Optimized: Sendable {
    public typealias Scalar = Float

    /// Internal storage: 128 SIMD4<Float> chunks holding the 512 scalar lanes.
    @usableFromInline
    internal var _storage: ContiguousArray<SIMD4<Float>>

    /// SIMD storage. Replacement must contain exactly 128 SIMD4 elements.
    /// Inout mutation must preserve this count, including when unwinding a throw.
    @inlinable
    public var storage: ContiguousArray<SIMD4<Float>> {
        get { _storage }
        set {
            guard newValue.count == 128 else { fatalError("Storage must contain exactly 128 SIMD4 elements") }
            _storage = newValue
        }
        _modify {
            defer {
                guard _storage.count == 128 else { fatalError("Storage must contain exactly 128 SIMD4 elements") }
            }
            yield &_storage
        }
    }

    /// Number of scalar elements (always 512)
    ///
    /// Computed, not stored (P02 RD7): the handle holds only `_storage`
    /// (`P02HandleLayoutTests.handleStrideIsTheStorageStride`).
    @inlinable
    public var scalarCount: Int { 512 }

    // MARK: - Initialization

    /// Initialize with zeros
    @inlinable
    public init() {
        _storage = ContiguousArray(repeating: SIMD4<Float>(), count: 128)
    }

    /// Initialize with repeating value
    @inlinable
    public init(repeating value: Scalar) {
        let simd4 = SIMD4<Float>(repeating: value)
        _storage = ContiguousArray(repeating: simd4, count: 128)
    }

    /// Initialize from an array by copying scalar lanes
    @inlinable
    public init(_ array: [Scalar]) throws {
        guard array.count == 512 else {
            throw VectorError.dimensionMismatch(expected: 512, actual: array.count)
        }

        _storage = ContiguousArray<SIMD4<Float>>()
        _storage.reserveCapacity(128)

        // Copy scalar lanes without imposing SIMD alignment on the Float owner.
        for i in stride(from: 0, to: 512, by: 4) {
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
        _storage.reserveCapacity(128)

        for i in stride(from: 0, to: 512, by: 4) {
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
    /// - Precondition: Storage contains exactly 128 SIMD4 elements.
    @inlinable
    public init(fromSIMDStorage storage: ContiguousArray<SIMD4<Float>>) {
        guard storage.count == 128 else { fatalError("Storage must contain exactly 128 SIMD4 elements") }
        self._storage = storage
    }

    // MARK: - Element Access

    /// Access individual elements
    @inlinable
    public subscript(index: Int) -> Scalar {
        get {
            precondition(index >= 0 && index < 512, "Index \(index) out of bounds [0..<512]")
            let vectorIndex = index >> 2  // Divide by 4
            let scalarIndex = index & 3   // Modulo 4
            return _storage[vectorIndex][scalarIndex]
        }
        set {
            precondition(index >= 0 && index < 512, "Index \(index) out of bounds [0..<512]")
            let vectorIndex = index >> 2
            let scalarIndex = index & 3
            _storage[vectorIndex][scalarIndex] = newValue
        }
    }
}

// MARK: - VectorProtocol Conformance

extension Vector512Optimized: VectorProtocol {
    public typealias Storage = ContiguousArray<SIMD4<Float>>

    /// Convert to array
    public func toArray() -> [Scalar] {
        var result = [Scalar]()
        result.reserveCapacity(512)

        _storage.withUnsafeBufferPointer { buffer in
            // Scoped rebinding restores the SIMD4<Float> binding on exit instead of permanently retyping the storage.
            // Strict-aliasing safety is not verified by a test; `testBufferAccess_RoundTripIsElementwiseExact` checks
            // only the element-wise round trip.
            buffer.baseAddress!.withMemoryRebound(to: Float.self, capacity: 512) { floatPtr in
                result.append(contentsOf: UnsafeBufferPointer(start: floatPtr, count: 512))
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
            // Scoped rebinding restores the SIMD4<Float> binding on exit. Strict-aliasing safety is not verified by a test;
            // `testBufferAccess_RoundTripIsElementwiseExact` checks only the element-wise round trip.
            try simd4Buffer.baseAddress!.withMemoryRebound(to: Float.self, capacity: 512) { floatPtr in
                try body(UnsafeBufferPointer(start: floatPtr, count: 512))
            }
        }
    }

    /// Borrow Float lanes for writing; the pointer must not escape the closure.
    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(
        _ body: (UnsafeMutableBufferPointer<Scalar>) throws -> R
    ) rethrows -> R {
        try _storage.withUnsafeMutableBufferPointer { simd4Buffer in
            // Scoped rebinding restores the SIMD4<Float> binding on exit. Strict-aliasing safety is not verified by a test;
            // `testBufferAccess_RoundTripIsElementwiseExact` checks only the element-wise round trip.
            try simd4Buffer.baseAddress!.withMemoryRebound(to: Float.self, capacity: 512) { floatPtr in
                try body(UnsafeMutableBufferPointer(start: floatPtr, count: 512))
            }
        }
    }
}

// MARK: - Distance, Magnitude and Normalization

extension Vector512Optimized {

    /// Dot product; delegates to `DotKernels.dot512`.
    @inlinable
    @inline(__always)
    public func dotProduct(_ other: Vector512Optimized) -> Scalar {
        DotKernels.dot512(self, other)
    }

    /// Squared Euclidean distance; delegates to `EuclideanKernels.squared512`.
    @inlinable
    @inline(__always)
    public func euclideanDistanceSquared(to other: Vector512Optimized) -> Scalar {
        EuclideanKernels.squared512(self, other)
    }

    /// Euclidean distance (with square root)
    @inlinable
    public func euclideanDistance(to other: Vector512Optimized) -> Scalar {
        EuclideanKernels.distance512(self, other)
    }

    /// Magnitude squared: the compensated Double sum of squares (`NormalizationCore.floatSquares`) rounded once to
    /// Float. `OptimizedVector512Suite.testMagnitudeRoundsOnceFromTheWidenedSum` checks the bits against one rounding
    /// of an exactly representable sum. On that fixture sequential Float accumulation gives other bits (the test
    /// asserts it); a Float sum that adds the small terms before the 1 gives the single-rounding bits, so the fixture
    /// does not distinguish that order. The fixture's sums are exact, so the test does not check the compensation.
    ///
    /// Infinity is returned when the scalar squared norm exceeds Float range
    /// (`testMagnitudeSquaredOverflowsToInfinityBeyondFloatRange`).
    @inlinable
    public var magnitudeSquared: Scalar {
        NormalizeKernels.magnitudeSquared(storage: _storage, laneCount: 128)
    }

    /// Magnitude (L2 norm): the square root of the widened (Double) sum of squares, rounded once to Float
    /// (`OptimizedVector512Suite.testMagnitudeRoundsOnceFromTheWidenedSum` checks the single rounding;
    /// `NormalizationRangeContractTests.halfWideningAndScalarMagnitude` checks the widening at the smallest Float
    /// magnitudes on Vector384Optimized, through the shared `NormalizeKernels.magnitude` kernel).
    ///
    /// Infinity is returned when the scalar norm exceeds Float range
    /// (`NormalizationRangeContractTests.halfWideningAndScalarMagnitude`).
    @inlinable
    public var magnitude: Scalar {
        NormalizeKernels.magnitude(storage: _storage, laneCount: 128)
    }

    /// Normalized vector
    @inlinable
    public func normalized() -> Result<Vector512Optimized, VectorError> {
        NormalizeKernels.normalized512(self)
    }

    /// Normalized vector with a precondition instead of a Result error.
    ///
    /// Validates all coordinates before constructing the normalized copy.
    /// - Precondition: finite nonzero input (enforced in Debug and normal Release; the trap is pinned in a child process by
    ///   `NormalizationRangeContractTests.invalidInputsRejectInChildren` in the configuration the suite runs)
    /// - Warning: Invalid input deliberately traps before mutation.
    @inlinable
    public func normalizedUnchecked() -> Vector512Optimized {
        NormalizeKernels.normalizedUnchecked512(self)
    }

    /// Cosine similarity: 0 when either magnitude is zero or their product underflows to zero, NaN when either
    /// magnitude is NaN or infinite (T15CosineExceptionalTableTests.everyImplementationFollowsThePolicy, sim.opt rows)
    @inlinable
    public func cosineSimilarity(to other: Vector512Optimized) -> Scalar {
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

extension Vector512Optimized {

    /// Addition with SIMD
    @inlinable
    public static func + (lhs: Vector512Optimized, rhs: Vector512Optimized) -> Vector512Optimized {
        var result = Vector512Optimized()
        for i in 0..<128 {
            result._storage[i] = lhs._storage[i] + rhs._storage[i]
        }
        return result
    }

    /// Subtraction with SIMD
    @inlinable
    public static func - (lhs: Vector512Optimized, rhs: Vector512Optimized) -> Vector512Optimized {
        var result = Vector512Optimized()
        for i in 0..<128 {
            result._storage[i] = lhs._storage[i] - rhs._storage[i]
        }
        return result
    }

    /// Scalar multiplication
    @inlinable
    public static func * (lhs: Vector512Optimized, rhs: Scalar) -> Vector512Optimized {
        var result = Vector512Optimized()
        let scalar = SIMD4<Float>(repeating: rhs)
        for i in 0..<128 {
            result._storage[i] = lhs._storage[i] * scalar
        }
        return result
    }

    /// Scalar division
    @inlinable
    public static func / (lhs: Vector512Optimized, rhs: Scalar) -> Vector512Optimized {
        guard rhs != 0 else { fatalError("Division by zero") }
        var result = lhs
        result.inPlaceDivide(rhs)
        return result
    }

    /// Element-wise multiplication (Hadamard product)
    @inlinable
    public static func .* (lhs: Vector512Optimized, rhs: Vector512Optimized) -> Vector512Optimized {
        var result = Vector512Optimized()
        for i in 0..<128 {
            result._storage[i] = lhs._storage[i] * rhs._storage[i]
        }
        return result
    }
}

// MARK: - Collection Conformance

extension Vector512Optimized: Collection {
    public typealias Index = Int
    public typealias Element = Scalar

    public var startIndex: Int { 0 }
    public var endIndex: Int { 512 }

    public func index(after i: Int) -> Int {
        i + 1
    }
}

// MARK: - Equatable & Hashable

extension Vector512Optimized: Equatable {
    @inlinable
    public static func == (lhs: Vector512Optimized, rhs: Vector512Optimized) -> Bool {
        for i in 0..<128 {
            if lhs._storage[i] != rhs._storage[i] {
                return false
            }
        }
        return true
    }
}

extension Vector512Optimized: Hashable {
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
        hasher.combine(512)
    }
}

// MARK: - Codable

extension Vector512Optimized: Codable {
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

extension Vector512Optimized: CustomDebugStringConvertible {
    public var debugDescription: String {
        let elements = self.prefix(10).map { String(format: "%.4f", $0) }
        return "Vector512Optimized[\(elements.joined(separator: ", ")), ... (512 total)]"
    }
}

// MARK: - OptimizedVector Conformance

extension Vector512Optimized: OptimizedVector {
    public static var laneCount: Int { 4 }
}

// MARK: - VectorType Conformance

extension Vector512Optimized: VectorType {}

// MARK: - AnalyzableVector Conformance

extension Vector512Optimized: MixedPrecisionKernels.AnalyzableVector {}

// MARK: - Arithmetic Extensions for Streaming K-means

extension Vector512Optimized {
    /// Add another vector element-wise
    @inlinable
    public func adding(_ other: Vector512Optimized) -> Vector512Optimized {
        var result = ContiguousArray<SIMD4<Float>>()
        result.reserveCapacity(128)

        for i in 0..<128 {
            result.append(_storage[i] + other._storage[i])
        }

        return Vector512Optimized(fromSIMDStorage: result)
    }

    /// Subtract another vector element-wise
    @inlinable
    public func subtracting(_ other: Vector512Optimized) -> Vector512Optimized {
        var result = ContiguousArray<SIMD4<Float>>()
        result.reserveCapacity(128)

        for i in 0..<128 {
            result.append(_storage[i] - other._storage[i])
        }

        return Vector512Optimized(fromSIMDStorage: result)
    }

    /// Scale by a scalar value
    @inlinable
    public func scaled(by scalar: Float) -> Vector512Optimized {
        var result = ContiguousArray<SIMD4<Float>>()
        result.reserveCapacity(128)

        let simdScalar = SIMD4<Float>(repeating: scalar)
        for i in 0..<128 {
            result.append(_storage[i] * simdScalar)
        }

        return Vector512Optimized(fromSIMDStorage: result)
    }

    /// Static zero vector
    @inlinable
    public static var zero: Vector512Optimized {
        Vector512Optimized()
    }
}
