// VectorCore: Scalar interpolation shared by fixed and dynamic vectors.

@usableFromInline
internal enum VectorInterpolation {
    /// O(1). Finite Float endpoints are widened before subtraction; their difference
    /// fits Double. InterpolationRangeContractTests.oppositeFiniteExtrema covers this boundary.
    @inlinable
    static func scalar(from: Float, to: Float, t: Double) -> Float {
        if t == 0 { return from }
        if t == 1 { return to }
        if from.isFinite && to.isFinite {
            return Float(Double(from) + (Double(to) - Double(from)) * t)
        }
        return from + (to - from) * Float(t)
    }
}
