extension Operations {
    /// Fits the UMAP similarity kernel `1 / (1 + a * d^(2*b))` for embedding distance `d`.
    ///
    /// The calibration target stays at 1 through `minDist` and decays exponentially
    /// thereafter with scale `spread`.
    ///
    /// - Throws: `VectorError` unless both inputs are finite, `spread > 0`, and
    ///   `0 <= minDist <= spread`. A fit that fails to converge or cannot be
    ///   represented with sufficient fidelity in `Float` also throws.
    /// - Note: `UMAPPublicCurveTests.invalidCurveInputsThrowInvalidOperation`,
    ///   `UMAPCurveFitSuite`, and `UMAPRangeContractTests` exercise input validation,
    ///   reference calibration, and the coefficient range, respectively.
    public static func umapCurveParameters(
        spread: Float,
        minDist: Float
    ) throws -> (a: Float, b: Float) {
        try UMAPLayout.fitABParams(spread: spread, minDist: minDist)
    }
}
