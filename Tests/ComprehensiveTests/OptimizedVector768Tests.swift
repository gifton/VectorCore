import Testing
@testable import VectorCore

@Suite("Optimized Vector768")
struct OptimizedVector768Suite {
    // Helpers
    private func manualDot(_ a: Vector768Optimized, _ b: Vector768Optimized) -> Float {
        var s: Float = 0
        for i in 0..<768 { s += a[i] * b[i] }
        return s
    }

    private func manualL2Squared(_ a: Vector768Optimized, _ b: Vector768Optimized) -> Float {
        var s: Float = 0
        for i in 0..<768 { let d = a[i] - b[i]; s += d * d }
        return s
    }

    private func manualL1(_ a: Vector768Optimized, _ b: Vector768Optimized) -> Float {
        var s: Float = 0
        for i in 0..<768 { s += abs(a[i] - b[i]) }
        return s
    }

    private func manualLinf(_ a: Vector768Optimized, _ b: Vector768Optimized) -> Float {
        var m: Float = 0
        for i in 0..<768 { m = max(m, abs(a[i] - b[i])) }
        return m
    }

    // Construction & Basics
    @Test func testInitZero_AllZeros() {
        let v = Vector768Optimized()
        #expect(approxEqual(v[0], 0) && approxEqual(v[767], 0))
        var sum: Float = 0
        for i in stride(from: 0, to: 768, by: 31) { sum += v[i] }
        #expect(approxEqual(sum, 0))
    }

    @Test func testInitRepeating_FillsAll() {
        let v = Vector768Optimized(repeating: -2.25)
        for i in stride(from: 0, to: 768, by: 47) { #expect(approxEqual(v[i], -2.25)) }
        #expect(approxEqual(v[767], -2.25))
    }

    @Test func testInitFromArray_SuccessAndThrow() {
        let arr = (0..<768).map { Float($0) }
        let v = try! Vector768Optimized(arr)
        #expect(approxEqual(v[0], 0) && approxEqual(v[767], 767))
        let bad = (0..<100).map { Float($0) }
        do { _ = try Vector768Optimized(bad); Issue.record("Expected dimensionMismatch not thrown") }
        catch let e as VectorError { #expect(e.kind == .dimensionMismatch) }
        catch { Issue.record("Unexpected error: \(error)") }
    }

    @Test func testGeneratorInit_FillsExpectedValues() {
        let v = try! Vector768Optimized(generator: { i in Float(i % 9) })
        #expect(approxEqual(v[0], 0) && approxEqual(v[1], 1) && approxEqual(v[8], 8))
        #expect(approxEqual(v[9], 0))
    }

    @Test func testSubscriptReadWrite() {
        var v = Vector768Optimized()
        v[0] = 1; v[383] = -3.5; v[767] = 2.25
        #expect(approxEqual(v[0], 1) && approxEqual(v[383], -3.5) && approxEqual(v[767], 2.25))
    }

    // Arithmetic & Collection
    @Test func testAdditionAndSubtraction_ElementwiseCorrect() {
        let a = try! Vector768Optimized((0..<768).map { Float($0 % 5) })
        let b = Vector768Optimized(repeating: 0.5)
        let c = a + b
        let d = c - a
        for i in stride(from: 0, to: 768, by: 53) {
            #expect(approxEqual(c[i], Float(i % 5) + 0.5))
            #expect(approxEqual(d[i], 0.5))
        }
    }

    @Test func testScalarMultiplyDivide_Correctness() {
        let a = try! Vector768Optimized((0..<768).map { _ in Float(1.75) })
        let k: Float = 3
        let m = a * k
        let d = m / k
        for i in stride(from: 0, to: 768, by: 49) {
            #expect(approxEqual(m[i], 5.25))
            #expect(approxEqual(d[i], 1.75))
        }
    }

    @Test func testHadamardProduct_ElementwiseMultiply() {
        let a = try! Vector768Optimized((0..<768).map { Float(($0+1) % 7) })
        let b = try! Vector768Optimized((0..<768).map { Float(($0+2) % 5) })
        let h = a .* b
        for i in stride(from: 0, to: 768, by: 61) { #expect(approxEqual(h[i], a[i] * b[i])) }
    }

    @Test func testCollectionBasics_IndicesAndIteration() {
        let a = try! Vector768Optimized((0..<768).map { Float($0) })
        #expect(a.startIndex == 0 && a.endIndex == 768)
        var sum: Float = 0
        for x in a { sum += x }
        let expected = Float(767 * 768) / 2
        #expect(approxEqual(sum, expected))
    }

    @Test func testEquatableAndHashable_Behavior() {
        let a = try! Vector768Optimized((0..<768).map { Float($0 % 11) })
        var b = a
        var c = a; c[10] = -9
        #expect(a == b)
        #expect(a != c)
        var set: Set<Vector768Optimized> = []
        set.insert(a); set.insert(b); set.insert(c)
        #expect(set.count == 2)
    }

    // Dot, Distance, Magnitude
    @Test func testDotProduct_MatchesManual() {
        let a = try! Vector768Optimized((0..<768).map { _ in Float.random(in: -1...1) })
        let b = try! Vector768Optimized((0..<768).map { _ in Float.random(in: -1...1) })
        #expect(approxEqual(a.dotProduct(b), manualDot(a, b), tol: 2e-4))
    }

    @Test func testEuclideanDistanceSquared_MatchesManual() {
        let a = try! Vector768Optimized((0..<768).map { _ in Float.random(in: -1...1) })
        let b = try! Vector768Optimized((0..<768).map { _ in Float.random(in: -1...1) })
        #expect(approxEqual(a.euclideanDistanceSquared(to: b), manualL2Squared(a, b), tol: 7e-4))
    }

    @Test func testEuclideanDistance_SqrtOfSquared() {
        let a = try! Vector768Optimized((0..<768).map { _ in Float.random(in: -1...1) })
        let b = try! Vector768Optimized((0..<768).map { _ in Float.random(in: -1...1) })
        let d = a.euclideanDistance(to: b)
        let s = a.euclideanDistanceSquared(to: b)
        #expect(approxEqual(d * d, s, tol: 1e-3))
    }

    @Test func testMagnitudeAndSquared_Relationships() {
        let a = try! Vector768Optimized((0..<768).map { _ in Float.random(in: -1...1) })
        #expect(approxEqual(a.magnitudeSquared, a.dotProduct(a), tol: 2e-4))
        #expect(a.magnitudeSquared >= 0)
        let z = Vector768Optimized()
        #expect(approxEqual(z.magnitude, 0, tol: 1e-7))
    }

    // Normalization & Cosine
    @Test func testNormalized_SuccessMagnitudeOne() {
        let a = try! Vector768Optimized((0..<768).map { _ in Float.random(in: 0.1...1) })
        do {
            let u = try a.normalizedThrowing()
            #expect(approxEqual(u.magnitude, 1, tol: 1e-5))
        } catch {
            Issue.record("Unexpected failure: \(error)")
        }
    }

    @Test func testNormalized_ZeroVectorFails() {
        let z = Vector768Optimized()
        do {
            _ = try z.normalizedThrowing()
            Issue.record("Expected failure for zero normalization")
        } catch let e as VectorError {
            #expect(e.kind == .invalidOperation)
        } catch { Issue.record("Unexpected error: \(error)") }
    }

    @Test func testCosineSimilarity_BoundsAndSpecialCases() {
        let a = Vector768Optimized(repeating: 1)
        #expect(approxEqual(a.cosineSimilarity(to: a), 1, tol: 1e-6))
        let b = a * -1
        #expect(approxEqual(a.cosineSimilarity(to: b), -1, tol: 1e-6))
        var e1 = Vector768Optimized(); e1[0] = 1
        var e2 = Vector768Optimized(); e2[1] = 1
        #expect(approxEqual(e1.cosineSimilarity(to: e2), 0, tol: 1e-6))
    }

    // Optimized Distance Metrics Parity
    @Test func testMetricParity_Euclidean() {
        let a = try! Vector768Optimized((0..<768).map { _ in Float.random(in: -1...1) })
        let b = try! Vector768Optimized((0..<768).map { _ in Float.random(in: -1...1) })
        let m = EuclideanDistance()
        #expect(approxEqual(a.euclideanDistance(to: b), m.distance(a, b), tol: 1e-5))
    }

    @Test func testMetricParity_Cosine() {
        let a = try! Vector768Optimized((0..<768).map { _ in Float.random(in: -1...1) })
        let b = try! Vector768Optimized((0..<768).map { _ in Float.random(in: -1...1) })
        let m = CosineDistance()
        #expect(approxEqual(1 - a.cosineSimilarity(to: b), m.distance(a, b), tol: 1e-5))
    }

    @Test func testMetricParity_DotProduct() {
        let a = try! Vector768Optimized((0..<768).map { _ in Float.random(in: -1...1) })
        let b = try! Vector768Optimized((0..<768).map { _ in Float.random(in: -1...1) })
        let m = DotProductDistance()
        #expect(approxEqual(-a.dotProduct(b), m.distance(a, b), tol: 1e-5))
    }

    @Test func testMetricParity_Manhattan() {
        let a = try! Vector768Optimized((0..<768).map { _ in Float.random(in: -1...1) })
        let b = try! Vector768Optimized((0..<768).map { _ in Float.random(in: -1...1) })
        let m = ManhattanDistance()
        #expect(approxEqual(manualL1(a, b), m.distance(a, b), tol: 1e-3))
    }

    @Test func testMetricParity_Chebyshev() {
        let a = try! Vector768Optimized((0..<768).map { _ in Float.random(in: -1...1) })
        let b = try! Vector768Optimized((0..<768).map { _ in Float.random(in: -1...1) })
        let m = ChebyshevDistance()
        #expect(approxEqual(manualLinf(a, b), m.distance(a, b), tol: 1e-6))
    }

    @Test func testBatchDistance_ParityForEuclidean() {
        let q = try! Vector768Optimized((0..<768).map { _ in Float.random(in: -1...1) })
        let candidates: [Vector768Optimized] = (0..<6).map { _ in try! Vector768Optimized((0..<768).map { _ in Float.random(in: -1...1) }) }
        let m = EuclideanDistance()
        let batch = m.batchDistance(query: q, candidates: candidates)
        for i in 0..<candidates.count { #expect(approxEqual(batch[i], m.distance(q, candidates[i]), tol: 1e-5)) }
    }

    @Test func testBatchDistance_ParityForCosineAndManhattan() {
        let q = try! Vector768Optimized((0..<768).map { _ in Float.random(in: -1...1) })
        let candidates: [Vector768Optimized] = (0..<5).map { _ in try! Vector768Optimized((0..<768).map { _ in Float.random(in: -1...1) }) }
        let cm = CosineDistance(); let mm = ManhattanDistance()
        let bc = cm.batchDistance(query: q, candidates: candidates)
        let bm = mm.batchDistance(query: q, candidates: candidates)
        for i in 0..<candidates.count {
            #expect(approxEqual(bc[i], cm.distance(q, candidates[i]), tol: 1e-5))
            #expect(approxEqual(bm[i], mm.distance(q, candidates[i]), tol: 1e-3))
        }
    }
}
