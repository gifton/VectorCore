import Testing
@testable import VectorCore

@Suite("Distance Metrics")
struct VectorDistanceMetricsSuite {

    @Test
    func testEuclideanDistance_IdenticalVectorsZero() {
        let a = try! Vector<Dim8>(Array(0..<8).map { Float($0) })
        let b = a
        #expect(approxEqual(a.euclideanDistance(to: b), 0))
        #expect(approxEqual(a.euclideanDistanceSquared(to: b), 0))
    }

    @Test
    func testManhattanDistance_IdenticalVectorsZero() {
        let a = try! Vector<Dim8>(Array(0..<8).map { Float($0) })
        #expect(approxEqual(a.manhattanDistance(to: a), 0))
    }

    @Test
    func testChebyshevDistance_IdenticalVectorsZero() {
        let a = try! Vector<Dim8>(Array(0..<8).map { Float($0) })
        #expect(approxEqual(a.chebyshevDistance(to: a), 0))
    }

    @Test
    func testEuclideanVsSquaredConsistency() {
        let a = try! Vector<Dim8>([1,2,3,4,5,6,7,8].map(Float.init))
        let b = try! Vector<Dim8>([2,4,6,8,10,12,14,16].map(Float.init))
        let d = a.euclideanDistance(to: b)
        let s = a.euclideanDistanceSquared(to: b)
        #expect(approxEqual(d * d, s, tol: 1e-4))
    }

    @Test
    func testDistances_Simple2DKnownValues() {
        let a = Vector<Dim2>(x: 3, y: 4)
        let b = Vector<Dim2>(x: 0, y: 0)
        #expect(approxEqual(a.euclideanDistance(to: b), 5))
        #expect(approxEqual(a.manhattanDistance(to: b), 7))
        #expect(approxEqual(a.chebyshevDistance(to: b), 4))
    }

    @Test
    func testCosineSimilarity_IdenticalIsOne() {
        let a = try! Vector<Dim8>(Array(0..<8).map { _ in Float.random(in: -1...1) })
        let b = a
        let sim = a.cosineSimilarity(to: b)
        let dist = a.cosineDistance(to: b)
        #expect(approxEqual(sim, 1, tol: 1e-5))
        #expect(approxEqual(dist, 0, tol: 1e-5))
    }

    @Test
    func testCosineSimilarity_OppositeIsMinusOne() {
        let a = try! Vector<Dim8>(Array(0..<8).map { _ in Float.random(in: -1...1) })
        let b = a * -1
        let sim = a.cosineSimilarity(to: b)
        let dist = a.cosineDistance(to: b)
        #expect(approxEqual(sim, -1, tol: 1e-5))
        #expect(approxEqual(dist, 2, tol: 1e-5))
    }

    @Test
    func testCosineSimilarity_OrthogonalIsZero() {
        // 3D orthogonal vectors
        let a = Vector<Dim3>(x: 1, y: 0, z: 0)
        let b = Vector<Dim3>(x: 0, y: 1, z: 0)
        let sim = a.cosineSimilarity(to: b)
        let dist = a.cosineDistance(to: b)
        #expect(approxEqual(sim, 0, tol: 1e-6))
        #expect(approxEqual(dist, 1, tol: 1e-6))
    }

    @Test
    func testCosineSimilarity_ZeroVectorReturnsZero() {
        let z = Vector<Dim8>.zero
        let u = Vector<Dim8>(repeating: 1)
        // Zero with unit → sim = 0, dist = 1
        #expect(approxEqual(z.cosineSimilarity(to: u), 0, tol: 1e-6))
        #expect(approxEqual(z.cosineDistance(to: u), 1, tol: 1e-6))
        // Both zero → sim = 0, dist = 1 (by implementation)
        #expect(approxEqual(z.cosineSimilarity(to: z), 0, tol: 1e-6))
        #expect(approxEqual(z.cosineDistance(to: z), 1, tol: 1e-6))
    }

    @Test
    func testCosineSimilarity_ScaleInvariancePositive() {
        let a = try! Vector<Dim8>(Array(0..<8).map { _ in Float.random(in: -1...1) })
        let b = try! Vector<Dim8>(Array(0..<8).map { _ in Float.random(in: -1...1) })
        let k: Float = 2.7
        let s1 = a.cosineSimilarity(to: b)
        let s2 = (k * a).cosineSimilarity(to: k * b)
        #expect(approxEqual(s1, s2, tol: 1e-6))
    }

    @Test
    func testCosineSimilarity_ScaleByNegativeFlipsSign() {
        let a = try! Vector<Dim8>(Array(0..<8).map { _ in Float.random(in: -1...1) })
        let b = try! Vector<Dim8>(Array(0..<8).map { _ in Float.random(in: -1...1) })
        let s1 = a.cosineSimilarity(to: b)
        let s2 = (-a).cosineSimilarity(to: b)
        #expect(approxEqual(s2, -s1, tol: 1e-6))
    }

    @Test
    func testEuclideanManhattanChebyshev_ScaleEquivariance() {
        let a = try! Vector<Dim8>(Array(0..<8).map { _ in Float.random(in: -2...2) })
        let b = try! Vector<Dim8>(Array(0..<8).map { _ in Float.random(in: -2...2) })
        let k: Float = -3.5
        let dE = a.euclideanDistance(to: b)
        let dM = a.manhattanDistance(to: b)
        let dC = a.chebyshevDistance(to: b)
        #expect(approxEqual((k*a).euclideanDistance(to: k*b), abs(k)*dE, tol: 1e-4))
        #expect(approxEqual((k*a).manhattanDistance(to: k*b), abs(k)*dM, tol: 1e-4))
        #expect(approxEqual((k*a).chebyshevDistance(to: k*b), abs(k)*dC, tol: 1e-4))
    }

    @Test
    func testSymmetry_EuclideanManhattanChebyshev() {
        let a = try! Vector<Dim8>(Array(0..<8).map { _ in Float.random(in: -2...2) })
        let b = try! Vector<Dim8>(Array(0..<8).map { _ in Float.random(in: -2...2) })
        #expect(approxEqual(a.euclideanDistance(to: b), b.euclideanDistance(to: a), tol: 1e-6))
        #expect(approxEqual(a.manhattanDistance(to: b), b.manhattanDistance(to: a), tol: 1e-6))
        #expect(approxEqual(a.chebyshevDistance(to: b), b.chebyshevDistance(to: a), tol: 1e-6))
    }

    @Test
    func testCosineSymmetry() {
        let a = try! Vector<Dim8>(Array(0..<8).map { _ in Float.random(in: -2...2) })
        let b = try! Vector<Dim8>(Array(0..<8).map { _ in Float.random(in: -2...2) })
        #expect(approxEqual(a.cosineSimilarity(to: b), b.cosineSimilarity(to: a), tol: 1e-6))
        #expect(approxEqual(a.cosineDistance(to: b), b.cosineDistance(to: a), tol: 1e-6))
    }

    @Test
    func testTriangleInequality_Euclidean() {
        // Sample a few random triples
        for _ in 0..<5 {
            let a = try! Vector<Dim8>(Array(0..<8).map { _ in Float.random(in: -1...1) })
            let b = try! Vector<Dim8>(Array(0..<8).map { _ in Float.random(in: -1...1) })
            let c = try! Vector<Dim8>(Array(0..<8).map { _ in Float.random(in: -1...1) })
            let lhs = a.euclideanDistance(to: c)
            let rhs = a.euclideanDistance(to: b) + b.euclideanDistance(to: c)
            #expect(lhs <= rhs + 1e-5)
        }
    }

    @Test
    func testTriangleInequality_Manhattan() {
        for _ in 0..<5 {
            let a = try! Vector<Dim8>(Array(0..<8).map { _ in Float.random(in: -1...1) })
            let b = try! Vector<Dim8>(Array(0..<8).map { _ in Float.random(in: -1...1) })
            let c = try! Vector<Dim8>(Array(0..<8).map { _ in Float.random(in: -1...1) })
            let lhs = a.manhattanDistance(to: c)
            let rhs = a.manhattanDistance(to: b) + b.manhattanDistance(to: c)
            #expect(lhs <= rhs + 1e-5)
        }
    }

    @Test
    func testTriangleInequality_Chebyshev() {
        for _ in 0..<5 {
            let a = try! Vector<Dim8>(Array(0..<8).map { _ in Float.random(in: -1...1) })
            let b = try! Vector<Dim8>(Array(0..<8).map { _ in Float.random(in: -1...1) })
            let c = try! Vector<Dim8>(Array(0..<8).map { _ in Float.random(in: -1...1) })
            let lhs = a.chebyshevDistance(to: c)
            let rhs = a.chebyshevDistance(to: b) + b.chebyshevDistance(to: c)
            #expect(lhs <= rhs + 1e-5)
        }
    }

    @Test
    func testEuclideanDistanceToZeroEqualsMagnitude() {
        let a = try! Vector<Dim8>(Array(0..<8).map { _ in Float.random(in: -2...2) })
        #expect(approxEqual(a.euclideanDistance(to: .zero), a.magnitude, tol: 1e-6))
    }

    @Test
    func testManhattanDistanceToZeroEqualsL1() {
        let a = try! Vector<Dim8>(Array(0..<8).map { _ in Float.random(in: -2...2) })
        #expect(approxEqual(a.manhattanDistance(to: .zero), a.l1Norm, tol: 1e-6))
    }

    @Test
    func testChebyshevDistanceToZeroEqualsLInf() {
        let a = try! Vector<Dim8>(Array(0..<8).map { _ in Float.random(in: -2...2) })
        #expect(approxEqual(a.chebyshevDistance(to: .zero), a.lInfinityNorm, tol: 1e-6))
    }

    @Test
    func testFixedVsDynamicVectors_Parity() {
        let vals = Array(0..<8).map { _ in Float.random(in: -1...1) }
        let f = try! Vector<Dim8>(vals)
        let d = DynamicVector(vals)
        let zeroF = Vector<Dim8>.zero
        let zeroD = DynamicVector.zero(dimension: 8)
        #expect(approxEqual(f.euclideanDistance(to: zeroF), d.euclideanDistance(to: zeroD), tol: 1e-6))
        #expect(approxEqual(f.manhattanDistance(to: zeroF), d.manhattanDistance(to: zeroD), tol: 1e-6))
        #expect(approxEqual(f.chebyshevDistance(to: zeroF), d.chebyshevDistance(to: zeroD), tol: 1e-6))
    }

    @Test
    func testCosineRangeBounds() {
        // Generate non-zero vectors to avoid undefined magnitude cases
        for _ in 0..<10 {
            var a: Vector<Dim8>
            var b: Vector<Dim8>
            repeat {
                a = try! Vector<Dim8>(Array(0..<8).map { _ in Float.random(in: -1...1) })
            } while approxEqual(a.magnitude, 0, tol: 1e-6)
            repeat {
                b = try! Vector<Dim8>(Array(0..<8).map { _ in Float.random(in: -1...1) })
            } while approxEqual(b.magnitude, 0, tol: 1e-6)

            let s = a.cosineSimilarity(to: b)
            let d = a.cosineDistance(to: b)
            #expect(s <= 1 + 1e-6 && s >= -1 - 1e-6)
            #expect(d >= -1e-6 && d <= 2 + 1e-6)
        }
    }

}
