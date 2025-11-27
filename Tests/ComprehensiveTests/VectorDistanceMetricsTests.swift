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
        let a = try! Vector<Dim8>([1, 2, 3, 4, 5, 6, 7, 8].map(Float.init))
        let b = try! Vector<Dim8>([2, 4, 6, 8, 10, 12, 14, 16].map(Float.init))
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

    // MARK: - Manhattan SIMD Optimization Tests

    @Test
    func testManhattanDistance_SIMD_DivisibleBy4() {
        // Test dimension divisible by 4 (pure SIMD path)
        let metric = ManhattanDistance()
        let a = try! Vector<Dim8>([1, 2, 3, 4, 5, 6, 7, 8])
        let b = try! Vector<Dim8>([2, 4, 6, 8, 10, 12, 14, 16])
        // Expected: |1-2| + |2-4| + |3-6| + |4-8| + |5-10| + |6-12| + |7-14| + |8-16|
        //         = 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 = 36
        let dist = metric.distance(a, b)
        #expect(approxEqual(dist, 36, tol: 1e-5))
    }

    @Test
    func testManhattanDistance_SIMD_NotDivisibleBy4() {
        // Test dimensions with remainder (exercises scalar tail loop)
        // Use Dim3 (3 elements - not divisible by 4)
        let metric = ManhattanDistance()
        let a = Vector<Dim3>(x: 1, y: 2, z: 3)
        let b = Vector<Dim3>(x: 4, y: 6, z: 10)
        // Expected: |1-4| + |2-6| + |3-10| = 3 + 4 + 7 = 14
        let dist = metric.distance(a, b)
        #expect(approxEqual(dist, 14, tol: 1e-5))
    }

    @Test
    func testManhattanDistance_SIMD_SingleElement() {
        // Test single element (pure scalar remainder)
        let a = DynamicVector([5.0])
        let b = DynamicVector([2.0])
        let metric = ManhattanDistance()
        let dist = metric.distance(a, b)
        #expect(approxEqual(dist, 3, tol: 1e-5))
    }

    @Test
    func testManhattanDistance_SIMD_DynamicVectorVariousDims() {
        let metric = ManhattanDistance()
        // Test various dimensions to ensure SIMD+scalar tail works
        for dim in [1, 2, 3, 5, 7, 9, 13, 17, 100, 127, 128, 129, 255, 256, 257] {
            let aVals = (0..<dim).map { Float($0) }
            let bVals = (0..<dim).map { Float($0 + 1) }
            let a = DynamicVector(aVals)
            let b = DynamicVector(bVals)
            // Each element differs by 1, so Manhattan distance = dim
            let dist = metric.distance(a, b)
            #expect(approxEqual(dist, Float(dim), tol: 1e-4), "Failed for dim=\(dim)")
        }
    }

    @Test
    func testManhattanDistance_SIMD_MatchesScalarReference() {
        // Verify SIMD result matches scalar reference implementation
        let a = DynamicVector((0..<100).map { _ in Float.random(in: -10...10) })
        let b = DynamicVector((0..<100).map { _ in Float.random(in: -10...10) })

        // Compute reference using scalar loop
        var reference: Float = 0
        for i in 0..<100 {
            reference += abs(a[i] - b[i])
        }

        let metric = ManhattanDistance()
        let result = metric.distance(a, b)
        // Use relative tolerance since SIMD accumulation order differs from scalar
        // For sums of ~600, 1e-3 relative tolerance is appropriate
        #expect(approxEqual(result, reference, tol: 1e-3))
    }

    @Test
    func testManhattanDistance_SIMD_NegativeValues() {
        let metric = ManhattanDistance()
        let a = try! Vector<Dim8>([-1, -2, -3, -4, 5, 6, 7, 8])
        let b = try! Vector<Dim8>([1, 2, 3, 4, -5, -6, -7, -8])
        // Expected: |-1-1| + |-2-2| + |-3-3| + |-4-4| + |5-(-5)| + |6-(-6)| + |7-(-7)| + |8-(-8)|
        //         = 2 + 4 + 6 + 8 + 10 + 12 + 14 + 16 = 72
        let dist = metric.distance(a, b)
        #expect(approxEqual(dist, 72, tol: 1e-5))
    }

    @Test
    func testManhattanDistance_SIMD_LargeDimension() {
        // Test larger dimensions (512, 768, 1536)
        let metric = ManhattanDistance()
        for dim in [512, 768, 1536] {
            let aVals = (0..<dim).map { Float($0) * 0.001 }
            let bVals = (0..<dim).map { Float($0) * 0.001 + 0.1 }
            let a = DynamicVector(aVals)
            let b = DynamicVector(bVals)
            // Each element differs by 0.1, so Manhattan distance = dim * 0.1
            let dist = metric.distance(a, b)
            #expect(approxEqual(dist, Float(dim) * 0.1, tol: 1e-3), "Failed for dim=\(dim)")
        }
    }

}
