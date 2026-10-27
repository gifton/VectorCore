// VectorCore: Optimized Normalization Operations
//
// Permissive normalization extensions on `Vector` (`normalizedFast`, `normalized(tolerance:)`)
//

import Foundation

extension Vector {
    /// Permissive normalization; empty, zero and nonfinite inputs are unchanged.
    /// Already near-unit input may be returned unnormalized (strict norm tolerance 1e-6).
    @inlinable
    public func normalizedFast() -> Vector<D> {
        normalized(tolerance: 1e-6)
    }

    /// In-place permissive normalization with the same policy as normalizedFast().
    @inlinable
    public mutating func normalizeFast() {
        self = normalizedFast()
    }

    /// Check if vector is approximately normalized
    ///
    /// True when `abs(magnitudeSquared - 1) < 1e-6` (strict). The read computes the squared norm, so it is O(n).
    ///
    /// Since D02 removed this extension's storage gate, this property is `Vector`'s witness for
    /// `IndexableVector.isNormalized`, and code generic over `IndexableVector` or over the dimension
    /// (`func f<D: StaticDimension>(_ v: Vector<D>)`) reads this check for every `Vector<D>`, built-in dimensions
    /// included. Before D02 both read the protocol default `false` (D02CosineAndWitnessTests).
    @inlinable
    public var isNormalized: Bool {
        let magSquared = magnitudeSquared
        return Swift.abs(magSquared - 1.0) < 1e-6
    }

    /// Permissive normalization with strict abs(widenedNorm - 1) < tolerance early return.
    /// Requires finite nonnegative tolerance; either signed zero disables early return.
    /// Empty, zero and nonfinite input is unchanged; large tolerance may return non-unit input.
    @inlinable
    public func normalized(tolerance: Float = 1e-6) -> Vector<D> {
        guard tolerance.isFinite && tolerance >= 0 else { fatalError("normalized: tolerance must be finite and nonnegative") }
        let calculation = withUnsafeBufferPointer { buffer in
            NormalizationCore.plan(count: buffer.count, get: { buffer[$0] })
        }
        guard case .success(let plan) = calculation else { return self }
        if plan.isWithinTolerance(Double(tolerance)) { return self }
        var result = self
        result.withUnsafeMutableBufferPointer { buffer in
            for i in buffer.indices { buffer[i] = plan.coordinate(buffer[i]) }
        }
        return result
    }
}
