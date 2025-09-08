import Testing
@testable import VectorCore

// Normalization correctness and edge cases (zero vector)

@Suite("Vector Normalization")
struct VectorNormalizationSuite {

    @Test
    func testNormalizedResult_SuccessForNonZero() {
        let v = try! Vector<Dim4>([3,4,0,0])
        do {
            let u = try v.normalizedThrowing()
            #expect(approxEqual(u.magnitude, 1, tol: 1e-6))
            #expect(approxEqual(u[0], v[0] / v.magnitude, tol: 1e-6))
            #expect(approxEqual(u[1], v[1] / v.magnitude, tol: 1e-6))
        } catch { Issue.record("Unexpected failure: \(error)") }
    }

    @Test
    func testNormalizedResult_FailureForZeroVector() {
        let v = Vector<Dim4>.zero
        do {
            _ = try v.normalizedThrowing()
            Issue.record("Expected failure for zero vector normalization")
        } catch let e as VectorError { #expect(e.kind == .invalidOperation) }
        catch { Issue.record("Unexpected error type: \(error)") }
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
        var v = try! Vector<Dim4>([3,4,0,0])
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
        let base = try! Vector<Dim4>([1,1,0,0])
        let unit = try! base.normalizedThrowing()
        let s: Float = 1 + 5e-7
        let near = unit * s
        let out = near.normalized(tolerance: 1e-6)
        // Within tolerance, should return self (no change)
        #expect(out == near)
    }

    @Test
    func testNormalizedWithTolerance_BeyondToleranceNormalizes() {
        let base = try! Vector<Dim4>([1,2,0,0])
        let unit = try! base.normalizedThrowing()
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
        do {
            let u = try v.normalizedThrowing()
            for i in 0..<8 { #expect(approxEqual(f[i], u[i], tol: 1e-5)) }
        } catch { Issue.record("Unexpected failure: \(error)") }
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

}
