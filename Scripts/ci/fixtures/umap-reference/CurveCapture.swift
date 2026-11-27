import VectorCore

// The alpha has no public curve function. Compile this file only for the retained
// accepted revision and candidate; never derive alpha coefficients from candidate code.
func captureCurveParameters() throws -> [UInt32] {
    let pair = try Operations.umapCurveParameters(spread: 1, minDist: 0.1)
    return [pair.a.bitPattern, pair.b.bitPattern]
}
