import Foundation
import Testing
@testable import VectorCore

@Suite("Optimized Vector1536")
struct OptimizedVector1536Suite {
    // Helpers
    private func manualDot(_ a: Vector1536Optimized, _ b: Vector1536Optimized) -> Float {
        var s: Float = 0
        for i in 0..<1536 { s += a[i] * b[i] }
        return s
    }

    private func manualL2Squared(_ a: Vector1536Optimized, _ b: Vector1536Optimized) -> Float {
        var s: Float = 0
        for i in 0..<1536 { let d = a[i] - b[i]; s += d * d }
        return s
    }

    private func manualL1(_ a: Vector1536Optimized, _ b: Vector1536Optimized) -> Float {
        var s: Float = 0
        for i in 0..<1536 { s += abs(a[i] - b[i]) }
        return s
    }

    private func manualLinf(_ a: Vector1536Optimized, _ b: Vector1536Optimized) -> Float {
        var m: Float = 0
        for i in 0..<1536 { m = max(m, abs(a[i] - b[i])) }
        return m
    }

    // Construction & Basics
    @Test func testInitZero_AllZeros() {
        let v = Vector1536Optimized()
        #expect(approxEqual(v[0], 0) && approxEqual(v[1535], 0))
        var sum: Float = 0
        for i in stride(from: 0, to: 1536, by: 97) { sum += v[i] }
        #expect(approxEqual(sum, 0))
    }

    @Test func testInitRepeating_FillsAll() {
        let v = Vector1536Optimized(repeating: 4.25)
        for i in stride(from: 0, to: 1536, by: 113) { #expect(approxEqual(v[i], 4.25)) }
        #expect(approxEqual(v[1535], 4.25))
    }

    @Test func testInitFromArray_SuccessAndThrow() {
        let arr = (0..<1536).map { Float($0) }
        let v = try! Vector1536Optimized(arr)
        #expect(approxEqual(v[0], 0) && approxEqual(v[1535], 1535))
        let bad = (0..<10).map { Float($0) }
        do { _ = try Vector1536Optimized(bad); Issue.record("Expected dimensionMismatch not thrown") }
        catch let e as VectorError { #expect(e.kind == .dimensionMismatch) }
        catch { Issue.record("Unexpected error: \(error)") }
    }

    @Test func testGeneratorInit_FillsExpectedValues() {
        let v = Vector1536Optimized(generator: { i in Float((i / 4) % 7) })
        #expect(approxEqual(v[0], 0) && approxEqual(v[4], 1) && approxEqual(v[28], 0))
    }

    @Test func testSubscriptReadWrite() {
        var v = Vector1536Optimized()
        v[0] = -1; v[768] = 2.5; v[1535] = -3
        #expect(approxEqual(v[0], -1) && approxEqual(v[768], 2.5) && approxEqual(v[1535], -3))
    }

    // Arithmetic & Collection
    @Test func testAdditionAndSubtraction_ElementwiseCorrect() {
        let a = try! Vector1536Optimized((0..<1536).map { Float($0 % 3) })
        let b = Vector1536Optimized(repeating: -0.75)
        let c = a + b
        let d = c - a
        for i in stride(from: 0, to: 1536, by: 173) {
            #expect(approxEqual(c[i], Float(i % 3) - 0.75))
            #expect(approxEqual(d[i], -0.75))
        }
    }

    @Test func testScalarMultiplyDivide_Correctness() {
        let a = try! Vector1536Optimized((0..<1536).map { _ in Float(2.0) })
        let k: Float = 1.25
        let m = a * k
        let d = m / k
        for i in stride(from: 0, to: 1536, by: 211) {
            #expect(approxEqual(m[i], 2.5))
            #expect(approxEqual(d[i], 2.0))
        }
    }

    @Test func testHadamardProduct_ElementwiseMultiply() {
        let a = try! Vector1536Optimized((0..<1536).map { Float(($0+1) % 11) })
        let b = try! Vector1536Optimized((0..<1536).map { Float(($0+2) % 13) })
        let h = a .* b
        for i in stride(from: 0, to: 1536, by: 257) { #expect(approxEqual(h[i], a[i] * b[i])) }
    }

    @Test func testCollectionBasics_IndicesAndIteration() {
        let a = try! Vector1536Optimized((0..<1536).map { Float($0) })
        #expect(a.startIndex == 0 && a.endIndex == 1536)
        var sum: Float = 0
        for x in a { sum += x }
        let expected = Float(1535 * 1536) / 2
        #expect(approxEqual(sum, expected))
    }

    @Test func testEquatableAndHashable_Behavior() {
        let a = try! Vector1536Optimized((0..<1536).map { Float($0 % 17) })
        var b = a
        var c = a; c[42] = -5
        #expect(a == b)
        #expect(a != c)
        var set: Set<Vector1536Optimized> = []
        set.insert(a); set.insert(b); set.insert(c)
        #expect(set.count == 2)
    }

    // Dot, Distance, Magnitude
    @Test func testDotProduct_MatchesManual() {
        let a = try! Vector1536Optimized((0..<1536).map { _ in Float.random(in: -1...1) })
        let b = try! Vector1536Optimized((0..<1536).map { _ in Float.random(in: -1...1) })
        #expect(approxEqual(a.dotProduct(b), manualDot(a, b), tol: 5e-4))
    }

    @Test func testEuclideanDistanceSquared_MatchesManual() {
        let a = try! Vector1536Optimized((0..<1536).map { _ in Float.random(in: -1...1) })
        let b = try! Vector1536Optimized((0..<1536).map { _ in Float.random(in: -1...1) })
        #expect(approxEqual(a.euclideanDistanceSquared(to: b), manualL2Squared(a, b), tol: 2e-3))
    }

    @Test func testEuclideanDistance_SqrtOfSquared() {
        let a = try! Vector1536Optimized((0..<1536).map { _ in Float.random(in: -1...1) })
        let b = try! Vector1536Optimized((0..<1536).map { _ in Float.random(in: -1...1) })
        let d = a.euclideanDistance(to: b)
        let s = a.euclideanDistanceSquared(to: b)
        #expect(approxEqual(d * d, s, tol: 2e-3))
    }

    @Test func testMagnitudeAndSquared_Relationships() {
        let a = try! Vector1536Optimized((0..<1536).map { _ in Float.random(in: -1...1) })
        #expect(approxEqual(a.magnitudeSquared, a.dotProduct(a), tol: 5e-4))
        #expect(a.magnitudeSquared >= 0)
        let z = Vector1536Optimized()
        #expect(approxEqual(z.magnitude, 0, tol: 1e-7))
    }

    // Normalization & Cosine
    @Test func testNormalized_SuccessMagnitudeOne() {
        let a = try! Vector1536Optimized((0..<1536).map { _ in Float.random(in: 0.1...1) })
        switch a.normalized() {
        case .success(let u): #expect(approxEqual(u.magnitude, 1, tol: 1e-5))
        case .failure(let e): Issue.record("Unexpected failure: \(e)")
        }
    }

    @Test func testNormalized_ZeroVectorFails() {
        let z = Vector1536Optimized()
        switch z.normalized() {
        case .success: Issue.record("Expected failure for zero normalization")
        case .failure(let e): #expect(e.kind == .invalidOperation)
        }
    }

    @Test func testCosineSimilarity_BoundsAndSpecialCases() {
        let a = Vector1536Optimized(repeating: 1)
        #expect(approxEqual(a.cosineSimilarity(to: a), 1, tol: 1e-6))
        let b = a * -1
        #expect(approxEqual(a.cosineSimilarity(to: b), -1, tol: 1e-6))
        var e1 = Vector1536Optimized(); e1[0] = 1
        var e2 = Vector1536Optimized(); e2[1] = 1
        #expect(approxEqual(e1.cosineSimilarity(to: e2), 0, tol: 1e-6))
    }

    // Optimized Distance Metrics Parity
    @Test func testMetricParity_Euclidean() {
        let a = try! Vector1536Optimized((0..<1536).map { _ in Float.random(in: -1...1) })
        let b = try! Vector1536Optimized((0..<1536).map { _ in Float.random(in: -1...1) })
        let m = EuclideanDistance()
        #expect(approxEqual(a.euclideanDistance(to: b), m.distance(a, b), tol: 1e-5))
    }

    @Test func testMetricParity_Cosine() {
        let a = try! Vector1536Optimized((0..<1536).map { _ in Float.random(in: -1...1) })
        let b = try! Vector1536Optimized((0..<1536).map { _ in Float.random(in: -1...1) })
        let m = CosineDistance()
        #expect(approxEqual(1 - a.cosineSimilarity(to: b), m.distance(a, b), tol: 1e-5))
    }

    @Test func testMetricParity_DotProduct() {
        let a = try! Vector1536Optimized((0..<1536).map { _ in Float.random(in: -1...1) })
        let b = try! Vector1536Optimized((0..<1536).map { _ in Float.random(in: -1...1) })
        let m = DotProductDistance()
        #expect(approxEqual(-a.dotProduct(b), m.distance(a, b), tol: 1e-5))
    }

    @Test func testMetricParity_Manhattan() {
        let a = try! Vector1536Optimized((0..<1536).map { _ in Float.random(in: -1...1) })
        let b = try! Vector1536Optimized((0..<1536).map { _ in Float.random(in: -1...1) })
        let m = ManhattanDistance()
        #expect(approxEqual(manualL1(a, b), m.distance(a, b), tol: 3e-3))
    }

    @Test func testMetricParity_Chebyshev() {
        let a = try! Vector1536Optimized((0..<1536).map { _ in Float.random(in: -1...1) })
        let b = try! Vector1536Optimized((0..<1536).map { _ in Float.random(in: -1...1) })
        let m = ChebyshevDistance()
        #expect(approxEqual(manualLinf(a, b), m.distance(a, b), tol: 1e-6))
    }

    @Test func testBatchDistance_ParityForEuclidean() {
        let q = try! Vector1536Optimized((0..<1536).map { _ in Float.random(in: -1...1) })
        let candidates: [Vector1536Optimized] = (0..<4).map { _ in try! Vector1536Optimized((0..<1536).map { _ in Float.random(in: -1...1) }) }
        let m = EuclideanDistance()
        let batch = m.batchDistance(query: q, candidates: candidates)
        for i in 0..<candidates.count { #expect(approxEqual(batch[i], m.distance(q, candidates[i]), tol: 1e-5)) }
    }

    @Test func testBatchDistance_ParityForCosineAndManhattan() {
        let q = try! Vector1536Optimized((0..<1536).map { _ in Float.random(in: -1...1) })
        let candidates: [Vector1536Optimized] = (0..<4).map { _ in try! Vector1536Optimized((0..<1536).map { _ in Float.random(in: -1...1) }) }
        let cm = CosineDistance(); let mm = ManhattanDistance()
        let bc = cm.batchDistance(query: q, candidates: candidates)
        let bm = mm.batchDistance(query: q, candidates: candidates)
        for i in 0..<candidates.count {
            #expect(approxEqual(bc[i], cm.distance(q, candidates[i]), tol: 1e-5))
            #expect(approxEqual(bm[i], mm.distance(q, candidates[i]), tol: 3e-3))
        }
    }

    // Codable & Debug
    @Test func testCodable_RoundTrip() {
        let v = try! Vector1536Optimized((0..<1536).map { _ in Float.random(in: -1...1) })
        do { let data = try JSONEncoder().encode(v); let u = try JSONDecoder().decode(Vector1536Optimized.self, from: data)
            for i in stride(from: 0, to: 1536, by: 313) { #expect(approxEqual(u[i], v[i], tol: 1e-6)) }
        } catch { Issue.record("Codable round-trip failed: \(error)") }
    }

    @Test func testDebugDescription_ContainsPreviewAndTotal() {
        var v = Vector1536Optimized(); v[0] = 0.1234; v[1] = 2
        let d = v.debugDescription
        #expect(d.contains("1536 total") && d.contains("Vector1536Optimized"))
    }
}
