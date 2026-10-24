//
//  Vector768Optimized.swift
//  VectorCore
//
//  768-dimensional Float vector with SIMD4<Float> storage
//  Common dimension for OpenAI embeddings
//

import Foundation
import simd
import Accelerate

/// 768-dimensional Float vector stored as 192 `SIMD4<Float>` lanes.
///
/// Memory layout:
/// - Stored as 192 SIMD4<Float> values in `_storage` (a ContiguousArray).
/// - Each element is aligned to `MemoryLayout<SIMD4<Float>>.alignment` (16 bytes on Apple platforms): a ContiguousArray
///   stores its elements at their type's alignment. No cache-line placement is promised.
///
public struct Vector768Optimized: Sendable {
    public typealias Scalar = Float

    /// Internal storage: 192 SIMD4<Float> chunks holding the 768 scalar lanes.
    @usableFromInline
    internal var _storage: ContiguousArray<SIMD4<Float>>

    /// SIMD storage. Replacement must contain exactly 192 SIMD4 elements.
    /// Inout mutation must preserve this count, including when unwinding a throw.
    @inlinable
    public var storage: ContiguousArray<SIMD4<Float>> {
        get { _storage }
        set {
            guard newValue.count == 192 else { fatalError("Storage must contain exactly 192 SIMD4 elements") }
            _storage = newValue
        }
        _modify {
            defer {
                guard _storage.count == 192 else { fatalError("Storage must contain exactly 192 SIMD4 elements") }
            }
            yield &_storage
        }
    }

    /// Number of scalar elements (always 768)
    ///
    /// Computed, not stored (P02 RD7): the handle holds only `_storage`
    /// (`P02HandleLayoutTests.handleStrideIsTheStorageStride`).
    @inlinable
    public var scalarCount: Int { 768 }

    // MARK: - Initialization

    /// Initialize with zeros
    @inlinable
    public init() {
        _storage = ContiguousArray(repeating: SIMD4<Float>(), count: 192)
    }

    /// Initialize with repeating value
    @inlinable
    public init(repeating value: Scalar) {
        let simd4 = SIMD4<Float>(repeating: value)
        _storage = ContiguousArray(repeating: simd4, count: 192)
    }

    /// Initialize from an array by copying scalar lanes
    @inlinable
    public init(_ array: [Scalar]) throws {
        guard array.count == 768 else {
            throw VectorError.dimensionMismatch(expected: 768, actual: array.count)
        }

        _storage = ContiguousArray<SIMD4<Float>>()
        _storage.reserveCapacity(192)

        // Copy scalar lanes without imposing SIMD alignment on the Float owner.
        for i in stride(from: 0, to: 768, by: 4) {
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
        _storage.reserveCapacity(192)

        for i in stride(from: 0, to: 768, by: 4) {
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
    /// - Precondition: Storage contains exactly 192 SIMD4 elements.
    @inlinable
    public init(fromSIMDStorage storage: ContiguousArray<SIMD4<Float>>) {
        guard storage.count == 192 else { fatalError("Storage must contain exactly 192 SIMD4 elements") }
        self._storage = storage
    }

    // MARK: - Element Access

    /// Access individual elements
    @inlinable
    public subscript(index: Int) -> Scalar {
        get {
            precondition(index >= 0 && index < 768, "Index \(index) out of bounds [0..<768]")
            let vectorIndex = index >> 2  // Divide by 4
            let scalarIndex = index & 3   // Modulo 4
            return _storage[vectorIndex][scalarIndex]
        }
        set {
            precondition(index >= 0 && index < 768, "Index \(index) out of bounds [0..<768]")
            let vectorIndex = index >> 2
            let scalarIndex = index & 3
            _storage[vectorIndex][scalarIndex] = newValue
        }
    }
}

// MARK: - VectorProtocol Conformance

extension Vector768Optimized: VectorProtocol {
    public typealias Storage = ContiguousArray<SIMD4<Float>>

    /// Convert to array
    public func toArray() -> [Scalar] {
        var result = [Scalar]()
        result.reserveCapacity(768)

        _storage.withUnsafeBufferPointer { buffer in
            // Scoped rebinding restores the SIMD4<Float> binding on exit instead of permanently retyping the storage.
            // Strict-aliasing safety is not verified by a test; `allWidthScopedBorrows` checks only the
            // element-wise round trip.
            buffer.baseAddress!.withMemoryRebound(to: Float.self, capacity: 768) { floatPtr in
                result.append(contentsOf: UnsafeBufferPointer(start: floatPtr, count: 768))
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
            try simd4Buffer.baseAddress!.withMemoryRebound(to: Float.self, capacity: 768) { floatPtr in
                try body(UnsafeBufferPointer(start: floatPtr, count: 768))
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
            try simd4Buffer.baseAddress!.withMemoryRebound(to: Float.self, capacity: 768) { floatPtr in
                try body(UnsafeMutableBufferPointer(start: floatPtr, count: 768))
            }
        }
    }
}

// MARK: - Distance, Magnitude and Normalization

extension Vector768Optimized {

    /// Dot product; delegates to `DotKernels.dot768`.
    @inlinable
    @inline(__always)
    public func dotProduct(_ other: Vector768Optimized) -> Scalar {
        DotKernels.dot768(self, other)
    }

    /// Squared Euclidean distance; delegates to `EuclideanKernels.squared768`.
    @inlinable
    @inline(__always)
    public func euclideanDistanceSquared(to other: Vector768Optimized) -> Scalar {
        EuclideanKernels.squared768(self, other)
    }

    /// Euclidean distance (with square root)
    @inlinable
    public func euclideanDistance(to other: Vector768Optimized) -> Scalar {
        EuclideanKernels.distance768(self, other)
    }

    /// Magnitude squared: the compensated Double sum of squares (`NormalizationCore.floatSquares`, through
    /// `NormalizeKernels.magnitudeSquared`) rounded once to Float. No test checks the rounding at this width;
    /// `OptimizedVector512Suite.testMagnitudeRoundsOnceFromTheWidenedSum` checks the single rounding
    /// on `Vector512Optimized` through the same kernel.
    ///
    /// Infinity is returned when the scalar squared norm exceeds Float range.
    @inlinable
    public var magnitudeSquared: Scalar {
        NormalizeKernels.magnitudeSquared(storage: _storage, laneCount: 192)
    }

    /// Magnitude (L2 norm): the square root of the widened (Double) sum of squares, rounded once to Float
    /// (`NormalizeKernels.magnitude`; no test checks the rounding at this width).
    ///
    /// Infinity is returned when the scalar norm exceeds Float range.
    @inlinable
    public var magnitude: Scalar {
        NormalizeKernels.magnitude(storage: _storage, laneCount: 192)
    }

    /// Normalized vector
    @inlinable
    public func normalized() -> Result<Vector768Optimized, VectorError> {
        NormalizeKernels.normalized768(self)
    }

    /// Normalized vector with a precondition instead of a Result error.
    ///
    /// Validates all coordinates before constructing the normalized copy.
    /// - Precondition: finite nonzero input (enforced in Debug and normal Release; the trap is pinned in a child process by
    ///   `NormalizationRangeContractTests.invalidInputsRejectInChildren` in the configuration the suite runs)
    /// - Warning: Invalid input deliberately traps before mutation.
    @inlinable
    public func normalizedUnchecked() -> Vector768Optimized {
        NormalizeKernels.normalizedUnchecked768(self)
    }

    /// Cosine similarity: 0 when either magnitude is zero or their product underflows to zero, NaN when either
    /// magnitude is NaN or infinite (T15CosineExceptionalTableTests.everyImplementationFollowsThePolicy, sim.opt rows)
    @inlinable
    public func cosineSimilarity(to other: Vector768Optimized) -> Scalar {
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

extension Vector768Optimized {

    /// Addition with SIMD
    @inlinable
    public static func + (lhs: Vector768Optimized, rhs: Vector768Optimized) -> Vector768Optimized {
        var result = Vector768Optimized()
        for i in 0..<192 {
            result._storage[i] = lhs._storage[i] + rhs._storage[i]
        }
        return result
    }

    /// Subtraction with SIMD
    @inlinable
    public static func - (lhs: Vector768Optimized, rhs: Vector768Optimized) -> Vector768Optimized {
        var result = Vector768Optimized()
        for i in 0..<192 {
            result._storage[i] = lhs._storage[i] - rhs._storage[i]
        }
        return result
    }

    /// Scalar multiplication
    @inlinable
    public static func * (lhs: Vector768Optimized, rhs: Scalar) -> Vector768Optimized {
        var result = Vector768Optimized()
        let scalar = SIMD4<Float>(repeating: rhs)
        for i in 0..<192 {
            result._storage[i] = lhs._storage[i] * scalar
        }
        return result
    }

    /// Scalar division
    @inlinable
    public static func / (lhs: Vector768Optimized, rhs: Scalar) -> Vector768Optimized {
        guard rhs != 0 else { fatalError("Division by zero") }
        var result = lhs
        result.inPlaceDivide(rhs)
        return result
    }

    /// Element-wise multiplication (Hadamard product)
    @inlinable
    public static func .* (lhs: Vector768Optimized, rhs: Vector768Optimized) -> Vector768Optimized {
        var result = Vector768Optimized()
        for i in 0..<192 {
            result._storage[i] = lhs._storage[i] * rhs._storage[i]
        }
        return result
    }
}

// MARK: - Collection Conformance

extension Vector768Optimized: Collection {
    public typealias Index = Int
    public typealias Element = Scalar

    public var startIndex: Int { 0 }
    public var endIndex: Int { 768 }

    public func index(after i: Int) -> Int {
        i + 1
    }
}

// MARK: - Equatable & Hashable

extension Vector768Optimized: Equatable {
    @inlinable
    public static func == (lhs: Vector768Optimized, rhs: Vector768Optimized) -> Bool {
        for i in 0..<192 {
            if lhs._storage[i] != rhs._storage[i] {
                return false
            }
        }
        return true
    }
}

extension Vector768Optimized: Hashable {
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
        hasher.combine(768)
    }
}

// MARK: - Codable

extension Vector768Optimized: Codable {
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

extension Vector768Optimized: CustomDebugStringConvertible {
    public var debugDescription: String {
        let elements = self.prefix(10).map { String(format: "%.4f", $0) }
        return "Vector768Optimized[\(elements.joined(separator: ", ")), ... (768 total)]"
    }
}

// MARK: - OptimizedVector Conformance

extension Vector768Optimized: OptimizedVector {
    public static var laneCount: Int { 4 }
}

// MARK: - VectorType Conformance

extension Vector768Optimized: VectorType {}

// MARK: - AnalyzableVector Conformance

extension Vector768Optimized: MixedPrecisionKernels.AnalyzableVector {}

// MARK: - Arithmetic Extensions for Streaming K-means

extension Vector768Optimized {
    /// Add another vector element-wise
    @inlinable
    public func adding(_ other: Vector768Optimized) -> Vector768Optimized {
        var result = ContiguousArray<SIMD4<Float>>()
        result.reserveCapacity(192)

        for i in 0..<192 {
            result.append(_storage[i] + other._storage[i])
        }

        return Vector768Optimized(fromSIMDStorage: result)
    }

    /// Subtract another vector element-wise
    @inlinable
    public func subtracting(_ other: Vector768Optimized) -> Vector768Optimized {
        var result = ContiguousArray<SIMD4<Float>>()
        result.reserveCapacity(192)

        for i in 0..<192 {
            result.append(_storage[i] - other._storage[i])
        }

        return Vector768Optimized(fromSIMDStorage: result)
    }

    /// Scale by a scalar value
    @inlinable
    public func scaled(by scalar: Float) -> Vector768Optimized {
        var result = ContiguousArray<SIMD4<Float>>()
        result.reserveCapacity(192)

        let simdScalar = SIMD4<Float>(repeating: scalar)
        for i in 0..<192 {
            result.append(_storage[i] * simdScalar)
        }

        return Vector768Optimized(fromSIMDStorage: result)
    }

    /// Static zero vector
    @inlinable
    public static var zero: Vector768Optimized {
        Vector768Optimized()
    }
}
