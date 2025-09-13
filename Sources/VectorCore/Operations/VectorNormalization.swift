// VectorCore: Optimized Normalization Operations
//
// Extensions for efficient vector normalization
//

import Foundation

extension Vector where D.Storage: VectorStorageOperations {
    /// Normalize using reciprocal multiplication (faster than division)
    ///
    /// This method computes 1/magnitude and multiplies, which is typically
    /// faster than division on modern processors.
    @inlinable
    public func normalizedFast() -> Vector<D> {
        let magSquared = magnitudeSquared

        // Handle zero vector
        guard magSquared > 0 else { return self }

        // Fast path for already normalized vectors
        if Swift.abs(magSquared - 1.0) < 1e-12 {
            return self
        }

        // Use reciprocal square root for efficiency
        let invMag = 1.0 / Foundation.sqrt(magSquared)
        return self * invMag
    }

    /// Normalize in place using reciprocal multiplication
    @inlinable
    public mutating func normalizeFast() {
        let magSquared = magnitudeSquared

        // Handle zero vector
        guard magSquared > 0 else { return }

        // Fast path for already normalized vectors
        if Swift.abs(magSquared - 1.0) < 1e-12 {
            return
        }

        // Use reciprocal square root for efficiency
        let invMag = 1.0 / Foundation.sqrt(magSquared)
        self *= invMag
    }

    /// Check if vector is approximately normalized
    ///
    /// More efficient than computing full magnitude for validation
    @inlinable
    public var isNormalized: Bool {
        let magSquared = magnitudeSquared
        return Swift.abs(magSquared - 1.0) < 1e-6
    }

    /// Normalize with custom tolerance for "already normalized" check
    @inlinable
    public func normalized(tolerance: Float = 1e-6) -> Vector<D> {
        let mag = magnitude
        guard mag > 0 else { return self }

        // Fast path with custom tolerance
        if Swift.abs(mag - 1.0) < tolerance {
            return self
        }

        return self / mag
    }
}
