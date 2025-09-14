import Foundation
import Testing
@testable import VectorCore

@Suite("Optimized Vector512")
struct OptimizedVector512Suite {
    // MARK: - Helpers
    private func manualDot(_ a: Vector512Optimized, _ b: Vector512Optimized) -> Float {
        var s: Float = 0
        for i in 0..<512 { s += a[i] * b[i] }
        return s
    }

    private func manualL2Squared(_ a: Vector512Optimized, _ b: Vector512Optimized) -> Float {
        var s: Float = 0
        for i in 0..<512 { let d = a[i] - b[i]; s += d * d }
        return s
    }

    private func manualL2(_ a: Vector512Optimized, _ b: Vector512Optimized) -> Float {
        (manualL2Squared(a, b)).squareRoot()
    }

    private func manualL1(_ a: Vector512Optimized, _ b: Vector512Optimized) -> Float {
        var s: Float = 0
        for i in 0..<512 { s += abs(a[i] - b[i]) }
        return s
    }

    private func manualLinf(_ a: Vector512Optimized, _ b: Vector512Optimized) -> Float {
        var m: Float = 0
        for i in 0..<512 { m = max(m, abs(a[i] - b[i])) }
        return m
    }

    // MARK: - Construction & Basics
    @Test
    func testInitZero_AllZeros() {
        let v = Vector512Optimized()
        // Spot check a few indices and aggregate
        #expect(approxEqual(v[0], 0))
        #expect(approxEqual(v[255], 0))
        #expect(approxEqual(v[511], 0))
        var sum: Float = 0
        for i in stride(from: 0, to: 512, by: 17) { sum += v[i] }
        #expect(approxEqual(sum, 0))
    }

    @Test
    func testInitRepeating_FillsAll() {
        let v = Vector512Optimized(repeating: 3.5)
        for i in stride(from: 0, to: 512, by: 29) { #expect(approxEqual(v[i], 3.5)) }
        #expect(approxEqual(v[511], 3.5))
    }

    @Test
    func testInitFromArray_SuccessAndThrow() {
        let arr = (0..<512).map { Float($0) }
        let v = try! Vector512Optimized(arr)
        #expect(approxEqual(v[0], 0) && approxEqual(v[511], 511))

        let bad = (0..<510).map { Float($0) }
        do {
            _ = try Vector512Optimized(bad)
            Issue.record("Expected dimensionMismatch not thrown")
        } catch let e as VectorError {
            #expect(e.kind == .dimensionMismatch)
        } catch { Issue.record("Unexpected error: \(error)") }
    }

    @Test
    func testGeneratorInit_FillsExpectedValues() {
        let v = Vector512Optimized(generator: { Float($0 * 2) })
        #expect(approxEqual(v[0], 0) && approxEqual(v[1], 2) && approxEqual(v[2], 4))
        #expect(approxEqual(v[511], Float(511 * 2)))
    }

    @Test
    func testSubscriptReadWrite() {
        var v = Vector512Optimized()
        v[0] = 1.25
        v[255] = -2
        v[511] = 3.75
        #expect(approxEqual(v[0], 1.25))
        #expect(approxEqual(v[255], -2))
        #expect(approxEqual(v[511], 3.75))
    }

    // MARK: - Arithmetic & Collection
    @Test
    func testAdditionAndSubtraction_ElementwiseCorrect() {
        let a = try! Vector512Optimized((0..<512).map { Float($0) })
        let b = Vector512Optimized(repeating: 2)
        let c = a + b
        let d = c - a
        for i in stride(from: 0, to: 512, by: 41) {
            #expect(approxEqual(c[i], Float(i) + 2))
            #expect(approxEqual(d[i], 2))
        }
    }

    @Test
    func testScalarMultiplyDivide_Correctness() {
        let a = try! Vector512Optimized((0..<512).map { _ in Float(1.5) })
        let k: Float = 2.5
        let m = a * k
        let d = m / k
        for i in stride(from: 0, to: 512, by: 37) {
            #expect(approxEqual(m[i], 1.5 * 2.5))
            #expect(approxEqual(d[i], 1.5))
        }
    }

    @Test
    func testHadamardProduct_ElementwiseMultiply() {
        let a = try! Vector512Optimized((0..<512).map { Float($0 % 7) })
        let b = try! Vector512Optimized((0..<512).map { Float(($0 + 3) % 5) })
        let h = a .* b
        for i in stride(from: 0, to: 512, by: 23) {
            #expect(approxEqual(h[i], a[i] * b[i]))
        }
    }

    @Test
    func testCollectionBasics_IndicesAndIteration() {
        let a = try! Vector512Optimized((0..<512).map { Float($0) })
        #expect(a.startIndex == 0)
        #expect(a.endIndex == 512)
        var sum: Float = 0
        for x in a { sum += x }
        let expected = Float(511 * 512) / 2 // sum 0..511
        #expect(approxEqual(sum, expected))
    }

    @Test
    func testEquatableAndHashable_Behavior() {
        let a = try! Vector512Optimized((0..<512).map { Float($0) })
        let b = a
        var c = a
        c[100] = -1
        #expect(a == b)
        #expect(a != c)
        var set: Set<Vector512Optimized> = []
        set.insert(a)
        set.insert(b) // duplicate
        set.insert(c)
        #expect(set.count == 2)
    }

    // MARK: - Dot, Distance, Magnitude
    @Test
    func testDotProduct_MatchesManual() {
        let a = try! Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
        let b = try! Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
        let dot1 = a.dotProduct(b)
        let dot2 = manualDot(a, b)
        #expect(approxEqual(dot1, dot2, tol: 1e-4))
    }

    @Test
    func testEuclideanDistanceSquared_MatchesManual() {
        let a = try! Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
        let b = try! Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
        let d1 = a.euclideanDistanceSquared(to: b)
        let d2 = manualL2Squared(a, b)
        #expect(approxEqual(d1, d2, tol: 5e-4))
    }

    @Test
    func testEuclideanDistance_SqrtOfSquared() {
        let a = try! Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
        let b = try! Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
        let d = a.euclideanDistance(to: b)
        let s = a.euclideanDistanceSquared(to: b)
        #expect(approxEqual(d * d, s, tol: 1e-4))
    }

    @Test
    func testMagnitudeAndSquared_Relationships() {
        let a = try! Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
        #expect(approxEqual(a.magnitudeSquared, a.dotProduct(a), tol: 1e-4))
        #expect(a.magnitudeSquared >= 0)
        let z = Vector512Optimized()
        #expect(approxEqual(z.magnitude, 0, tol: 1e-7))
    }

    // MARK: - Normalization & Cosine
    @Test
    func testNormalized_SuccessMagnitudeOne() {
        let a = try! Vector512Optimized((0..<512).map { _ in Float.random(in: 0.1...1) })
        switch a.normalized() {
        case .success(let u):
            #expect(approxEqual(u.magnitude, 1, tol: 1e-5))
            // Direction check: u should be proportional to a (compare a[i]/u[i] approximately constant for non-zero)
            var ratios: [Float] = []
            for i in stride(from: 0, to: 512, by: 64) where u[i] != 0 {
                ratios.append(a[i] / u[i])
            }
            if let r0 = ratios.first { for r in ratios { #expect(approxEqual(r, r0, tol: 1e-3)) } }
        case .failure(let e):
            Issue.record("Unexpected failure: \(e)")
        }
    }

    @Test
    func testNormalized_ZeroVectorFails() {
        let z = Vector512Optimized()
        switch z.normalized() {
        case .success:
            Issue.record("Expected failure for zero normalization")
        case .failure(let e):
            #expect(e.kind == .invalidOperation)
        }
    }

    @Test
    func testCosineSimilarity_BoundsAndSpecialCases() {
        // Identical -> 1
        let a = Vector512Optimized(repeating: 1)
        #expect(approxEqual(a.cosineSimilarity(to: a), 1, tol: 1e-6))
        // Opposite -> -1
        let b = a * -1
        #expect(approxEqual(a.cosineSimilarity(to: b), -1, tol: 1e-6))
        // Orthogonal-like: unit basis on different indices
        var e1 = Vector512Optimized(); e1[0] = 1
        var e2 = Vector512Optimized(); e2[1] = 1
        #expect(approxEqual(e1.cosineSimilarity(to: e2), 0, tol: 1e-6))
    }

    // MARK: - Optimized Distance Metrics Parity
    @Test
    func testMetricParity_Euclidean() {
        let a = try! Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
        let b = try! Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
        let m = EuclideanDistance()
        let d1 = a.euclideanDistance(to: b)
        let d2 = m.distance(a, b)
        #expect(approxEqual(d1, d2, tol: 1e-5))
    }

    @Test
    func testMetricParity_Cosine() {
        let a = try! Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
        let b = try! Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
        let m = CosineDistance()
        let d1 = 1 - a.cosineSimilarity(to: b)
        let d2 = m.distance(a, b)
        #expect(approxEqual(d1, d2, tol: 1e-5))
    }

    @Test
    func testMetricParity_DotProduct() {
        let a = try! Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
        let b = try! Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
        let m = DotProductDistance()
        let d1 = -a.dotProduct(b)
        let d2 = m.distance(a, b)
        #expect(approxEqual(d1, d2, tol: 1e-5))
    }

    @Test
    func testMetricParity_Manhattan() {
        let a = try! Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
        let b = try! Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
        let m = ManhattanDistance()
        let d1 = manualL1(a, b)
        let d2 = m.distance(a, b)
        #expect(approxEqual(d1, d2, tol: 5e-4))
    }

    @Test
    func testMetricParity_Chebyshev() {
        let a = try! Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
        let b = try! Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
        let m = ChebyshevDistance()
        let d1 = manualLinf(a, b)
        let d2 = m.distance(a, b)
        #expect(approxEqual(d1, d2, tol: 1e-6))
    }

    @Test
    func testBatchDistance_ParityForEuclidean() {
        let q = try! Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
        let candidates: [Vector512Optimized] = (0..<7).map { _ in try! Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) }) }
        let m = EuclideanDistance()
        let batch = m.batchDistance(query: q, candidates: candidates)
        #expect(batch.count == candidates.count)
        for i in 0..<candidates.count {
            let single = m.distance(q, candidates[i])
            #expect(approxEqual(batch[i], single, tol: 1e-5))
        }
    }

    @Test
    func testBatchDistance_ParityForCosineAndManhattan() {
        let q = try! Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
        let candidates: [Vector512Optimized] = (0..<5).map { _ in try! Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) }) }
        let cm = CosineDistance()
        let mm = ManhattanDistance()
        let bc = cm.batchDistance(query: q, candidates: candidates)
        let bm = mm.batchDistance(query: q, candidates: candidates)
        for i in 0..<candidates.count {
            #expect(approxEqual(bc[i], cm.distance(q, candidates[i]), tol: 1e-5))
            #expect(approxEqual(bm[i], mm.distance(q, candidates[i]), tol: 1e-5))
        }
    }

    // MARK: - Codable & Debug
    @Test
    func testCodable_RoundTrip() {
        let v = try! Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
        do {
            let data = try JSONEncoder().encode(v)
            let u = try JSONDecoder().decode(Vector512Optimized.self, from: data)
            // Spot-check several positions
            for i in stride(from: 0, to: 512, by: 73) { #expect(approxEqual(u[i], v[i], tol: 1e-6)) }
        } catch {
            Issue.record("Codable round-trip failed: \(error)")
        }
    }

    @Test
    func testDebugDescription_ContainsPreviewAndTotal() {
        var v = Vector512Optimized()
        v[0] = 0.1234; v[1] = 2; v[2] = 3; v[3] = 4
        let desc = v.debugDescription
        #expect(desc.contains("512 total"))
        #expect(desc.contains("Vector512Optimized"))
    }
}
