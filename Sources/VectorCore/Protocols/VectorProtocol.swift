//
//  VectorProtocol.swift
//  VectorCore
//
//  Core protocol for all vector types
//

import Foundation

// MARK: - Vector Protocol

/// The core protocol for all vector types in VectorCore.
///
/// ## Design Philosophy
/// - Minimal required implementations for conformance
/// - Rich functionality through protocol extensions
/// - Type-safe compile-time dimension checking where possible
/// - Scoped buffer-pointer access to the storage (`withUnsafeBufferPointer`)
///
/// ## Conforming Types
/// - `Vector<D>`: Fixed-dimension vectors with compile-time safety
/// - `DynamicVector`: Runtime-determined dimension vectors
///
/// ## Dispatch
/// - The requirements below (`storage`, `scalarCount`, `init()`, `init(_:)`, `init(repeating:)`, `toArray()`,
///   `withUnsafeBufferPointer(_:)`, `withUnsafeMutableBufferPointer(_:)`, `isFinite`, `isZero`), and the
///   `Collection` members this protocol's extension supplies as defaults (`startIndex`, `endIndex`,
///   `index(after:)`, `subscript(_:)`), are dynamically dispatched: a call through a generic
///   `V: VectorProtocol` or an `any VectorProtocol` value runs the conforming type's own implementation when it
///   declares one.
/// - Every other member declared in an extension of this protocol (the arithmetic operators, `dotProduct(_:)`,
///   `magnitude`, `magnitudeSquared`, `normalized()`, `normalizedUnchecked()`, `l1Norm`, `lInfinityNorm`, the
///   distance functions including `cosineSimilarity(to:)` and `cosineDistance(to:)`, `sum`, `mean`, `min`,
///   `max`, `zero`, `ones`, `random(in:)`, the `inPlace` mutators, `mapElements(_:)`,
///   `isApproximatelyEqual(to:tolerance:)` and the encoders) is statically selected: generic and existential
///   code always runs the extension's implementation, even when the concrete type declares a member with the
///   same name; only a call on the concrete type binds to that type's own member.
/// - D02WitnessDispatchMatrixTests checks each requirement and each extension member in concrete, generic and
///   existential calls against a shadowing conformer.
///
public protocol VectorProtocol: Sendable, Hashable, Codable, Collection
where Element == Scalar, Index == Int {

    // MARK: - Associated Types

    /// The scalar type for vector elements
    associatedtype Scalar: BinaryFloatingPoint & Hashable & Codable

    /// The storage type for vector data
    associatedtype Storage: Sendable

    // MARK: - Required Properties

    /// The underlying storage.
    ///
    /// Conformers must keep `scalarCount`, collection count, array count and borrowed
    /// scalar buffer counts equal. Fixed-width conformers must validate replacement
    /// shape; dynamic conformers must update their dimension when storage is replaced.
    /// These are obligations of each conformer, including third-party implementations.
    var storage: Storage { get set }

    /// Number of elements in the vector
    var scalarCount: Int { get }

    // MARK: - Required Initializers

    /// Initialize with zeros
    init()

    /// Initialize from an array
    /// - Throws: VectorError.dimensionMismatch if array size doesn't match requirements
    init(_ array: [Scalar]) throws

    /// Initialize with repeating value
    init(repeating value: Scalar)

    // MARK: - Required Methods

    /// Convert to array representation
    func toArray() -> [Scalar]

    /// Access storage for reading (enables SIMD optimizations)
    func withUnsafeBufferPointer<R>(
        _ body: (UnsafeBufferPointer<Scalar>) throws -> R
    ) rethrows -> R

    /// Access storage for writing (enables SIMD optimizations)
    mutating func withUnsafeMutableBufferPointer<R>(
        _ body: (UnsafeMutableBufferPointer<Scalar>) throws -> R
    ) rethrows -> R

    // MARK: - Optional Requirements with Defaults

    /// Check if all values are finite (no NaN or infinity)
    var isFinite: Bool { get }

    /// Check if this is the zero vector
    var isZero: Bool { get }
}

// MARK: - Default Implementations

public extension VectorProtocol {

    // MARK: Collection Conformance

    var startIndex: Int { 0 }
    var endIndex: Int { scalarCount }

    func index(after i: Int) -> Int {
        i + 1
    }

    // Default subscript using toArray() - conformers should override for efficiency
    subscript(index: Int) -> Scalar {
        precondition(index >= 0 && index < scalarCount, "Index \(index) out of bounds")
        return toArray()[index]
    }

    // MARK: Validation

    var isFinite: Bool {
        withUnsafeBufferPointer { buffer in
            for element in buffer {
                if !element.isFinite { return false }
            }
            return true
        }
    }

    var isZero: Bool {
        withUnsafeBufferPointer { buffer in
            for element in buffer {
                if element != 0 { return false }
            }
            return true
        }
    }

    // MARK: Factory Methods

    /// Create a zero vector
    static var zero: Self {
        Self()
    }

    /// Create a ones vector
    static var ones: Self {
        Self(repeating: 1)
    }

    /// Create a random vector
    static func random(in range: ClosedRange<Scalar> = 0...1) -> Self where Scalar.RawSignificand: FixedWidthInteger {
        let array = (0..<Self().scalarCount).map { _ in
            Scalar.random(in: range)
        }
        return try! Self(array)
    }
}

// MARK: - Arithmetic Operations

public extension VectorProtocol {

    // MARK: Vector-Vector Operations

    static func + (lhs: Self, rhs: Self) -> Self {
        precondition(lhs.scalarCount == rhs.scalarCount, "Dimension mismatch")
        // Allocate result with correct length
        var result = try! Self(Array(repeating: 0, count: lhs.scalarCount))
        result.withUnsafeMutableBufferPointer { resultPtr in
            lhs.withUnsafeBufferPointer { lhsPtr in
                rhs.withUnsafeBufferPointer { rhsPtr in
                    for i in 0..<lhs.scalarCount {
                        resultPtr[i] = lhsPtr[i] + rhsPtr[i]
                    }
                }
            }
        }
        return result
    }

    static func - (lhs: Self, rhs: Self) -> Self {
        precondition(lhs.scalarCount == rhs.scalarCount, "Dimension mismatch")
        // Allocate result with correct length
        var result = try! Self(Array(repeating: 0, count: lhs.scalarCount))
        result.withUnsafeMutableBufferPointer { resultPtr in
            lhs.withUnsafeBufferPointer { lhsPtr in
                rhs.withUnsafeBufferPointer { rhsPtr in
                    for i in 0..<lhs.scalarCount {
                        resultPtr[i] = lhsPtr[i] - rhsPtr[i]
                    }
                }
            }
        }
        return result
    }

    // MARK: - In-Place Operations

    /// Perform in-place addition with another vector
    /// - Precondition: Equal scalar counts, checked before either buffer borrow.
    @inlinable
    mutating func inPlaceAdd(_ other: Self) {
        guard scalarCount == other.scalarCount else {
            fatalError("Dimension mismatch")
        }
        let count = scalarCount
        withUnsafeMutableBufferPointer { dst in
            other.withUnsafeBufferPointer { src in
                for i in 0..<count {
                    dst[i] += src[i]
                }
            }
        }
    }

    /// Perform in-place subtraction with another vector
    /// - Precondition: Equal scalar counts, checked before either buffer borrow.
    @inlinable
    mutating func inPlaceSubtract(_ other: Self) {
        guard scalarCount == other.scalarCount else {
            fatalError("Dimension mismatch")
        }
        let count = scalarCount
        withUnsafeMutableBufferPointer { dst in
            other.withUnsafeBufferPointer { src in
                for i in 0..<count {
                    dst[i] -= src[i]
                }
            }
        }
    }

    /// Perform in-place scalar multiplication
    @inlinable
    mutating func inPlaceMultiply(_ scalar: Scalar) {
        let count = scalarCount
        withUnsafeMutableBufferPointer { dst in
            for i in 0..<count {
                dst[i] *= scalar
            }
        }
    }

    /// Perform in-place scalar division
    @inlinable
    mutating func inPlaceDivide(_ scalar: Scalar) {
        guard scalar != 0 else { fatalError("Division by zero") }
        let count = scalarCount
        withUnsafeMutableBufferPointer { dst in
            for i in 0..<count {
                dst[i] /= scalar
            }
        }
    }

    /// Perform in-place element-wise multiplication (Hadamard product)
    /// - Precondition: Equal scalar counts, checked before either buffer borrow.
    @inlinable
    mutating func inPlaceElementwiseMultiply(_ other: Self) {
        guard scalarCount == other.scalarCount else {
            fatalError("Dimension mismatch")
        }
        let count = scalarCount
        withUnsafeMutableBufferPointer { dst in
            other.withUnsafeBufferPointer { src in
                for i in 0..<count {
                    dst[i] *= src[i]
                }
            }
        }
    }

    static func += (lhs: inout Self, rhs: Self) {
        lhs.inPlaceAdd(rhs)
    }

    static func -= (lhs: inout Self, rhs: Self) {
        lhs.inPlaceSubtract(rhs)
    }

    // MARK: Vector-Scalar Operations

    static func * (lhs: Self, rhs: Scalar) -> Self {
        // Allocate result with correct length
        var result = try! Self(Array(repeating: 0, count: lhs.scalarCount))
        result.withUnsafeMutableBufferPointer { resultPtr in
            lhs.withUnsafeBufferPointer { lhsPtr in
                for i in 0..<lhs.scalarCount {
                    resultPtr[i] = lhsPtr[i] * rhs
                }
            }
        }
        return result
    }

    static func / (lhs: Self, rhs: Scalar) -> Self {
        guard rhs != 0 else { fatalError("Division by zero") }
        var result = lhs
        result.inPlaceDivide(rhs)
        return result
    }

    static func * (lhs: Scalar, rhs: Self) -> Self {
        rhs * lhs
    }

    static func *= (lhs: inout Self, rhs: Scalar) {
        lhs.inPlaceMultiply(rhs)
    }

    static func /= (lhs: inout Self, rhs: Scalar) {
        lhs.inPlaceDivide(rhs)
    }

    // MARK: Unary Operations

    prefix static func - (vector: Self) -> Self {
        vector * (-1)
    }

    // MARK: Element-wise Operations

    /// Hadamard (element-wise) product
    static func .* (lhs: Self, rhs: Self) -> Self {
        precondition(lhs.scalarCount == rhs.scalarCount, "Dimension mismatch")
        // Allocate result with correct length
        var result = try! Self(Array(repeating: 0, count: lhs.scalarCount))
        result.withUnsafeMutableBufferPointer { resultPtr in
            lhs.withUnsafeBufferPointer { lhsPtr in
                rhs.withUnsafeBufferPointer { rhsPtr in
                    for i in 0..<lhs.scalarCount {
                        resultPtr[i] = lhsPtr[i] * rhsPtr[i]
                    }
                }
            }
        }
        return result
    }

    /// Element-wise division
    static func ./ (lhs: Self, rhs: Self) -> Self {
        precondition(lhs.scalarCount == rhs.scalarCount, "Dimension mismatch")
        // Allocate result with correct length
        var result = try! Self(Array(repeating: 0, count: lhs.scalarCount))
        result.withUnsafeMutableBufferPointer { resultPtr in
            lhs.withUnsafeBufferPointer { lhsPtr in
                rhs.withUnsafeBufferPointer { rhsPtr in
                    for i in 0..<lhs.scalarCount {
                        precondition(rhsPtr[i] != 0, "Division by zero at index \(i)")
                        resultPtr[i] = lhsPtr[i] / rhsPtr[i]
                    }
                }
            }
        }
        return result
    }
}

// MARK: - Vector Mathematics

public extension VectorProtocol {

    /// Compute dot product with another vector
    func dotProduct(_ other: Self) -> Scalar {
        precondition(scalarCount == other.scalarCount, "Dimension mismatch")

        var result: Scalar = 0
        withUnsafeBufferPointer { selfPtr in
            other.withUnsafeBufferPointer { otherPtr in
                for i in 0..<scalarCount {
                    result += selfPtr[i] * otherPtr[i]
                }
            }
        }
        return result
    }

    /// Magnitude (L2 norm) of the vector
    ///
    /// Two passes: the first finds the largest absolute component `maxAbs` (any NaN
    /// component returns NaN, a zero vector returns 0, an infinite `maxAbs` returns
    /// +infinity); the second sums `(x / maxAbs)²`, each term at most 1, and returns
    /// `maxAbs * sqrt(sum)`. So components above `sqrt(Float.greatestFiniteMagnitude)`
    /// do not overflow the squares.
    ///
    /// - Complexity: O(n) where n is the vector dimension
    /// - Note: Two passes over the data instead of one; the speed difference from a
    ///         one-pass sum of squares is not measured in this repository.
    var magnitude: Scalar {
        // Two-pass scaled norm (see the doc comment above)
        // Phase 1: Find maximum absolute value and detect NaN early
        var maxAbs: Scalar = 0
        var hasNaN = false
        withUnsafeBufferPointer { buffer in
            for element in buffer {
                if element.isNaN { hasNaN = true; continue }
                maxAbs = Swift.max(maxAbs, Swift.abs(element))
            }
        }

        // Propagate NaN if any component is NaN
        if hasNaN { return Scalar.nan }

        // NaN, zero and infinite maxima
        if maxAbs.isNaN { return Scalar.nan }
        guard maxAbs > 0 else { return 0 }  // Zero vector
        guard maxAbs.isFinite else { return Scalar.infinity }  // Infinite components

        // Phase 2: Scale, compute, scale back
        // By scaling all values by 1/maxAbs, we ensure |scaled| ≤ 1
        // This prevents overflow since 1² = 1
        var sumSquares: Scalar = 0
        withUnsafeBufferPointer { buffer in
            for element in buffer {
                let scaled = element / maxAbs
                sumSquares += scaled * scaled
            }
        }

        // Result: maxAbs × sqrt(Σ((x/maxAbs)²))
        //       = sqrt(maxAbs² × Σ((x/maxAbs)²))
        //       = sqrt(Σ(x²))  [mathematically equivalent]
        return maxAbs * Foundation.sqrt(sumSquares)
    }

    /// Squared magnitude
    ///
    /// Squares the two-pass scaled `magnitude`, so it computes a square root first.
    ///
    /// - Note: For the stable algorithm, we must compute sqrt then square it.
    ///         The naive Σ(x²) approach would overflow for large values.
    var magnitudeSquared: Scalar {
        let mag = magnitude  // Uses stable Kahan algorithm
        return mag * mag     // Square the stable result
    }

    /// Normalize finite nonzero coordinates, rejecting empty/zero/nonfinite inputs.
    /// Output fidelity depends on the conformer's storage preserving written scalars.
    func normalized() -> Result<Self, VectorError> {
        let calculation = withUnsafeBufferPointer { buffer in
            NormalizationCore.plan(count: buffer.count, get: { buffer[$0] })
        }
        switch calculation {
        case .failure(let error): return .failure(error)
        case .success(let plan):
            var result = self
            result.withUnsafeMutableBufferPointer { buffer in
                for i in buffer.indices { buffer[i] = plan.coordinate(buffer[i]) }
            }
            return .success(result)
        }
    }

    /// Normalize with a checked precondition, including normal Release builds.
    /// Requires finite, nonzero, nonempty input and a representable internal calculation.
    /// Output fidelity depends on the conformer's storage preserving written scalars.
    @inlinable
    func normalizedUnchecked() -> Self {
        let calculation = withUnsafeBufferPointer { buffer in
            NormalizationCore.plan(count: buffer.count, get: { buffer[$0] })
        }
        guard case .success(let plan) = calculation else {
            fatalError("normalizedUnchecked: requires finite nonzero input and representable calculation")
        }
        var result = self
        result.withUnsafeMutableBufferPointer { buffer in
            for i in buffer.indices { buffer[i] = plan.coordinate(buffer[i]) }
        }
        return result
    }

    /// L1 norm (Manhattan norm)
    var l1Norm: Scalar {
        var sum: Scalar = 0
        withUnsafeBufferPointer { buffer in
            for element in buffer {
                sum += Swift.abs(element)
            }
        }
        return sum
    }

    /// L∞ norm (maximum norm): `Scalar.nan` when any element is NaN, as `magnitude` and `l1Norm` give NaN
    /// (`Swift.max` alone drops a NaN). Pinned through `chebyshevDistance(to:)` by
    /// T15MetricExceptionalTableTests.everyMetricPinsItsExceptionalClass.
    var lInfinityNorm: Scalar {
        // The result is returned from inside the closure; no captured var is written in it and read after it
        // (T15 lost-store class, Scripts/ci/tests/test_lost_store_reads.py).
        withUnsafeBufferPointer { buffer in
            var maxValue: Scalar = 0
            for element in buffer {
                if element.isNaN { return Scalar.nan }
                maxValue = Swift.max(maxValue, Swift.abs(element))
            }
            return maxValue
        }
    }
}

// MARK: - Distance Metrics

public extension VectorProtocol {

    /// Euclidean distance to another vector
    func euclideanDistance(to other: Self) -> Scalar {
        (self - other).magnitude
    }

    /// Squared Euclidean distance: `(self - other).magnitudeSquared`
    func euclideanDistanceSquared(to other: Self) -> Scalar {
        (self - other).magnitudeSquared
    }

    /// Manhattan distance to another vector
    func manhattanDistance(to other: Self) -> Scalar {
        (self - other).l1Norm
    }

    /// Chebyshev distance to another vector
    func chebyshevDistance(to other: Self) -> Scalar {
        (self - other).lInfinityNorm
    }

    /// Cosine similarity with another vector: 0 when either magnitude is zero, NaN when either magnitude is NaN or
    /// infinite (T15CosineExceptionalTableTests, sim.vp rows)
    func cosineSimilarity(to other: Self) -> Scalar {
        let selfMagnitude = magnitude
        let otherMagnitude = other.magnitude
        guard selfMagnitude.isFinite && otherMagnitude.isFinite else { return .nan }
        let denominator = selfMagnitude * otherMagnitude
        guard denominator > 0 else { return 0 }
        return dotProduct(other) / denominator
    }

    /// Cosine distance (1 - cosine similarity), with a finite similarity clamped to [-1, 1] so the distance stays
    /// in [0, 2]; NaN when the similarity is not finite (T15CosineExceptionalTableTests, vpext rows)
    ///
    /// This is its own arithmetic: `cosineSimilarity(to:)` of this extension (`dotProduct(_:)` divided by the
    /// product of the two `magnitude` values), statically selected. It does not call the `CosineDistance` metric
    /// or a concrete type's cosine kernels, so its result is not promised to be bit-identical to theirs; the
    /// difference from `CosineDistance` is bounded in D02WitnessDispatchMatrixTests
    /// (`cosineDistanceStaysWithinTheDerivedBoundOfTheWitness`), which also pins the exceptional cases.
    func cosineDistance(to other: Self) -> Scalar {
        let similarity = cosineSimilarity(to: other)
        guard similarity.isFinite else { return .nan }
        return 1 - Swift.max(-1, Swift.min(1, similarity))
    }
}

// MARK: - Statistical Operations

public extension VectorProtocol {

    /// Sum of all elements
    var sum: Scalar {
        var total: Scalar = 0
        withUnsafeBufferPointer { buffer in
            for element in buffer {
                total += element
            }
        }
        return total
    }

    /// Mean of all elements
    var mean: Scalar {
        guard scalarCount > 0 else { return 0 }
        return sum / Scalar(scalarCount)
    }

    /// Minimum element: nil for an empty vector, `Scalar.nan` when any element is NaN (T16 reduction NaN contract;
    /// `Swift.min` alone drops a NaN). The sign of a zero minimum is unspecified. Pinned by
    /// T16ReductionExceptionalTableTests.
    var min: Scalar? {
        guard scalarCount > 0 else { return nil }
        // The result is returned from inside the closure; no captured var is written in it and read after it
        // (T15 lost-store class, T16 ruling C5).
        return withUnsafeBufferPointer { buffer in
            var minValue = Scalar.infinity
            for element in buffer {
                if element.isNaN { return Scalar.nan }
                minValue = Swift.min(minValue, element)
            }
            return minValue
        }
    }

    /// Maximum element: nil for an empty vector, `Scalar.nan` when any element is NaN (T16 reduction NaN contract;
    /// `Swift.max` alone drops a NaN). The sign of a zero maximum is unspecified. Pinned by
    /// T16ReductionExceptionalTableTests.
    var max: Scalar? {
        guard scalarCount > 0 else { return nil }
        // The result is returned from inside the closure; no captured var is written in it and read after it
        // (T15 lost-store class, T16 ruling C5).
        return withUnsafeBufferPointer { buffer in
            var maxValue = -Scalar.infinity
            for element in buffer {
                if element.isNaN { return Scalar.nan }
                maxValue = Swift.max(maxValue, element)
            }
            return maxValue
        }
    }
}

// MARK: - Convenience Extensions

public extension VectorProtocol {

    /// Apply a function to each element
    func mapElements(_ transform: (Scalar) throws -> Scalar) rethrows -> Self {
        let array = try toArray().map(transform)
        return try! Self(array)
    }

    /// Compare finite coordinates using an absolute, finite, nonnegative tolerance.
    /// Nonfinite coordinates or tolerance return false; signed zeros compare equal.
    /// Equal empty shapes compare true only with a valid tolerance. Dimensions are
    /// checked before borrowing storage. Subtraction remains in `Scalar`; overflow
    /// exceeds every finite tolerance. Complexity: O(n).
    func isApproximatelyEqual(to other: Self, tolerance: Scalar = 1e-6) -> Bool {
        guard tolerance.isFinite, tolerance >= 0, scalarCount == other.scalarCount else { return false }

        return withUnsafeBufferPointer { selfPtr in
            other.withUnsafeBufferPointer { otherPtr in
                for i in 0..<scalarCount {
                    if !selfPtr[i].isFinite || !otherPtr[i].isFinite || !(Swift.abs(selfPtr[i] - otherPtr[i]) <= tolerance) {
                        return false
                    }
                }
                return true
            }
        }
    }
}
