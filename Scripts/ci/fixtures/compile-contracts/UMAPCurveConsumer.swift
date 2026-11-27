import VectorCore

let curve = try Operations.umapCurveParameters(spread: 1, minDist: 0.1)
precondition(curve.a.isFinite && abs(curve.a - 1.57694) < 0.02)
precondition(curve.b.isFinite && abs(curve.b - 0.89506) < 0.01)

do {
    _ = try Operations.umapCurveParameters(spread: 1, minDist: 2)
    fatalError("Expected UMAP curve calibration to reject minDist > spread")
} catch let error as VectorError {
    precondition(error.kind == .invalidOperation)
}

print("UMAP_PUBLIC_CURVE_OK")
