import Testing
@testable import VectorCore

// Normalization correctness and edge cases (zero vector)

@Suite("Vector Normalization")
struct VectorNormalizationSuite {

    @Test
    func testNormalizedResult_SuccessForNonZero() {
        let v = try! Vector<Dim4>([3, 4, 0, 0])
        let r: Result<Vector<Dim4>, VectorError> = v.normalized()
        switch r {
        case .success(let u):
            #expect(approxEqual(u.magnitude, 1, tol: 1e-6))
            // Direction preserved: components proportional
            #expect(approxEqual(u[0], v[0] / v.magnitude, tol: 1e-6))
            #expect(approxEqual(u[1], v[1] / v.magnitude, tol: 1e-6))
        case .failure(let e):
            Issue.record("Unexpected failure: \(e)")
        }
    }

    @Test
    func testNormalizedResult_FailureForZeroVector() {
        let v = Vector<Dim4>.zero
        let r: Result<Vector<Dim4>, VectorError> = v.normalized()
        switch r {
        case .success:
            Issue.record("Expected failure for zero vector normalization")
        case .failure(let e):
            #expect(e.kind == .invalidOperation)
        }
    }

    @Test
    func testNormalizedFast_UnitVectorReturnsSelf() {
        let v = Vector<Dim3>(x: 1, y: 0, z: 0)
        let u = v.normalizedFast()
        #expect(u == v)
    }

    @Test
    func testNormalizedFast_ZeroVectorReturnsSelf() {
        let v = Vector<Dim8>.zero
        let u = v.normalizedFast()
        #expect(u == v)
    }

    @Test
    func testNormalizeFast_MutatingCorrectness() {
        var v = try! Vector<Dim4>([3, 4, 0, 0])
        v.normalizeFast()
        #expect(approxEqual(v.magnitude, 1, tol: 1e-6))
    }

    @Test
    func testIsNormalizedTrueNearOne() {
        // Start unit, scale slightly within tolerance of isNormalized (1e-6 on mag^2)
        let base = Vector<Dim2>(x: 1, y: 0)
        let s: Float = 1 + 2e-7 // mag^2 ~ 1 + 4e-7 < 1e-6
        let v = base * s
        #expect(v.isNormalized)
    }

    @Test
    func testIsNormalizedFalseWhenNotUnit() {
        let v = Vector<Dim2>(x: 1, y: 0) * 2
        #expect(!v.isNormalized)
    }

    @Test
    func testNormalizedWithTolerance_WithinToleranceReturnsSelf() {
        let base = try! Vector<Dim4>([1, 1, 0, 0])
        let unit = try! base.normalized().get()
        let s: Float = 1 + 5e-7
        let near = unit * s
        let out = near.normalized(tolerance: 1e-6)
        // Within tolerance, should return self (no change)
        #expect(out == near)
    }

    @Test
    func testNormalizedWithTolerance_BeyondToleranceNormalizes() {
        let base = try! Vector<Dim4>([1, 2, 0, 0])
        let unit = try! base.normalized().get()
        let s: Float = 1 + 1e-4
        let far = unit * s
        let out = far.normalized(tolerance: 1e-6)
        // Should change (normalize) and have unit magnitude
        #expect(out != far)
        #expect(approxEqual(out.magnitude, 1, tol: 1e-6))
    }

    @Test
    func testNormalizedConsistencyWithProtocolNormalized() {
        let v = try! Vector<Dim8>((0..<8).map { _ in Float.random(in: -1...1) })
        let f = v.normalizedFast()
        let res: Result<Vector<Dim8>, VectorError> = v.normalized()
        switch res {
        case .success(let u):
            for i in 0..<8 { #expect(approxEqual(f[i], u[i], tol: 1e-5)) }
        case .failure(let e):
            Issue.record("Unexpected failure: \(e)")
        }
    }

    @Test
    func testNormalizationIdempotency() {
        let v = try! Vector<Dim8>((0..<8).map { _ in Float.random(in: -1...1) })
        let n1 = v.normalizedFast()
        let n2 = n1.normalizedFast()
        for i in 0..<8 { #expect(approxEqual(n1[i], n2[i], tol: 1e-6)) }
    }

    @Test
    func testNormalizationPreservesDirection() {
        let v = try! Vector<Dim4>([3, -4, 0, 0])
        let m = v.magnitude
        let n = v.normalizedFast()
        for i in 0..<4 { #expect(approxEqual(n[i], v[i] / m, tol: 1e-6)) }
    }

    // MARK: - normalizedUnchecked() Tests

    @Test
    func testNormalizedUnchecked_ProducesUnitVector() {
        let v = try! Vector<Dim8>([3, 4, 0, 0, 1, 2, 2, 0])
        let u = v.normalizedUnchecked()
        #expect(approxEqual(u.magnitude, 1, tol: 1e-6))
    }

    @Test
    func testNormalizedUnchecked_PreservesDirection() {
        let v = try! Vector<Dim4>([3, -4, 0, 0])
        let m = v.magnitude
        let u = v.normalizedUnchecked()
        for i in 0..<4 { #expect(approxEqual(u[i], v[i] / m, tol: 1e-6)) }
    }

    @Test
    func testNormalizedUnchecked_MatchesNormalized() {
        let v = try! Vector<Dim8>((0..<8).map { _ in Float.random(in: -1...1) })
        let unchecked = v.normalizedUnchecked()
        let checked = try! v.normalized().get()
        for i in 0..<8 { #expect(approxEqual(unchecked[i], checked[i], tol: 1e-6)) }
    }

    @Test
    func testNormalizedUnchecked_DynamicVector() {
        let v = DynamicVector([3, 4, 0, 0, 1, 2, 2, 0])
        let u = v.normalizedUnchecked()
        #expect(approxEqual(u.magnitude, 1, tol: 1e-6))
    }

    @Test
    func testNormalizedUnchecked_Vector384Optimized() {
        let values = Array(repeating: Float(0.1), count: 384)
        let v = try! Vector384Optimized(values)
        let u = v.normalizedUnchecked()
        #expect(approxEqual(u.magnitude, 1, tol: 1e-6))
        // Verify direction preserved
        let m = v.magnitude
        #expect(approxEqual(u[0], v[0] / m, tol: 1e-6))
        #expect(approxEqual(u[383], v[383] / m, tol: 1e-6))
    }

    @Test
    func testNormalizedUnchecked_Vector512Optimized() {
        let values = (0..<512).map { Float($0) * 0.01 }
        let v = try! Vector512Optimized(values)
        let u = v.normalizedUnchecked()
        #expect(approxEqual(u.magnitude, 1, tol: 1e-6))
    }

    @Test
    func testNormalizedUnchecked_Vector768Optimized() {
        let values = (0..<768).map { Float($0) * 0.001 }
        let v = try! Vector768Optimized(values)
        let u = v.normalizedUnchecked()
        #expect(approxEqual(u.magnitude, 1, tol: 1e-6))
    }

    @Test
    func testNormalizedUnchecked_Vector1536Optimized() {
        let values = Array(repeating: Float(0.05), count: 1536)
        let v = try! Vector1536Optimized(values)
        let u = v.normalizedUnchecked()
        #expect(approxEqual(u.magnitude, 1, tol: 1e-6))
    }

    @Test
    func testNormalizedUnchecked_Idempotency() {
        let v = try! Vector<Dim8>((0..<8).map { _ in Float.random(in: -1...1) })
        let n1 = v.normalizedUnchecked()
        let n2 = n1.normalizedUnchecked()
        for i in 0..<8 { #expect(approxEqual(n1[i], n2[i], tol: 1e-6)) }
    }

}
