import Testing
@testable import VectorCore

@Suite("Vector Arithmetic")
struct VectorArithmeticSuite {
    @Test
    func testVectorAdditionBasic() {
        let a = try! Vector<Dim8>(Array(0..<8).map { Float($0) })
        let b = try! Vector<Dim8>(Array(0..<8).map { Float(10 + $0) })
        let c = a + b
        for i in 0..<8 {
            #expect(approxEqual(c[i], a[i] + b[i]))
        }
    }

    @Test
    func testVectorSubtractionBasic() {
        let a = try! Vector<Dim8>(Array(0..<8).map { Float($0) })
        let b = try! Vector<Dim8>(Array(0..<8).map { Float(2 * $0) })
        let c = a - b
        for i in 0..<8 {
            #expect(approxEqual(c[i], a[i] - b[i]))
        }
    }

    @Test
    func testInPlaceAdditionAndSubtractionMutateOriginal() {
        var a = Vector<Dim8>.ones
        let b = Vector<Dim8>(repeating: 2)
        a += b
        for i in 0..<8 { #expect(approxEqual(a[i], 3)) }
        a -= b
        for i in 0..<8 { #expect(approxEqual(a[i], 1)) }
    }

    @Test
    func testScalarMultiplicationRightAndLeft() {
        let a = try! Vector<Dim8>(Array(0..<8).map { Float($0) })
        let s: Float = 2.5
        let r1 = a * s
        let r2 = s * a
        for i in 0..<8 {
            #expect(approxEqual(r1[i], a[i] * s))
            #expect(approxEqual(r2[i], a[i] * s))
            #expect(approxEqual(r1[i], r2[i]))
        }
    }

    @Test
    func testScalarDivision() {
        let a = try! Vector<Dim8>(Array(0..<8).map { Float($0 + 1) }) // avoid zeros
        let s: Float = 2
        let r = a / s
        for i in 0..<8 {
            #expect(approxEqual(r[i], a[i] / s))
            #expect(approxEqual(r[i], a[i] * (1 / s)))
        }
    }

    @Test
    func testUnaryNegation() {
        let a = try! Vector<Dim8>(Array(0..<8).map { Float($0 - 4) })
        let n = -a
        for i in 0..<8 {
            #expect(approxEqual(n[i], -a[i]))
            #expect(approxEqual(n[i], a[i] * -1))
        }
    }

    @Test
    func testHadamardProductMatchesManualComputation() {
        let a = try! Vector<Dim8>(Array(0..<8).map { Float($0 + 1) })
        let b = try! Vector<Dim8>(Array(0..<8).map { Float(2 * ($0 + 1)) })
        let h = a .* b
        for i in 0..<8 {
            #expect(approxEqual(h[i], a[i] * b[i]))
        }
    }

    @Test
    func testElementwiseDivisionMatchesManualComputation() {
        let a = try! Vector<Dim8>(Array(0..<8).map { Float(($0 + 1) * 6) })
        let b = try! Vector<Dim8>(Array(0..<8).map { Float($0 + 1) }) // all non-zero
        let d = a ./ b
        for i in 0..<8 {
            #expect(approxEqual(d[i], a[i] / b[i]))
        }
    }

    @Test
    func testAdditionCommutative() {
        let a = try! Vector<Dim8>(Array(0..<8).map { _ in Float.random(in: -1...1) })
        let b = try! Vector<Dim8>(Array(0..<8).map { _ in Float.random(in: -1...1) })
        let r1 = a + b
        let r2 = b + a
        for i in 0..<8 {
            #expect(approxEqual(r1[i], r2[i], tol: 1e-6))
        }
    }

    @Test
    func testAdditionAssociativeWithinTolerance() {
        let a = try! Vector<Dim8>(Array(0..<8).map { _ in Float.random(in: -1...1) })
        let b = try! Vector<Dim8>(Array(0..<8).map { _ in Float.random(in: -1...1) })
        let c = try! Vector<Dim8>(Array(0..<8).map { _ in Float.random(in: -1...1) })
        let r1 = (a + b) + c
        let r2 = a + (b + c)
        for i in 0..<8 {
            #expect(approxEqual(r1[i], r2[i], tol: 1e-5))
        }
    }

    @Test
    func testScalarDistributivityOverAdditionWithinTolerance() {
        let a = try! Vector<Dim8>(Array(0..<8).map { _ in Float.random(in: -1...1) })
        let b = try! Vector<Dim8>(Array(0..<8).map { _ in Float.random(in: -1...1) })
        let s: Float = 1.7
        let left = s * (a + b)
        let right = s * a + s * b
        for i in 0..<8 {
            #expect(approxEqual(left[i], right[i], tol: 1e-5))
        }
    }

    @Test
    func testElementwiseMinMax() {
        let a = try! Vector<Dim8>([3, -2, 5, 0, 7, -1, 4, 2].map(Float.init))
        let b = try! Vector<Dim8>([1, -3, 6, 2, -8, -1, 10, 2].map(Float.init))
        let mi = a.min(b)
        let ma = a.max(b)
        for i in 0..<8 {
            #expect(approxEqual(mi[i], Swift.min(a[i], b[i])))
            #expect(approxEqual(ma[i], Swift.max(a[i], b[i])))
        }
    }

    @Test
    func testMinElementAndMaxElementIndices() {
        let vals: [Float] = [3, -5, 2, 10, 0, 4, -1, 7]
        let v = try! Vector<Dim8>(vals)
        let minRes = v.minElement()
        let maxRes = v.maxElement()
        #expect(minRes.value == -5)
        #expect(minRes.index == 1)
        #expect(maxRes.value == 10)
        #expect(maxRes.index == 3)
    }

    @Test
    func testClampToRange() {
        let vals: [Float] = [-2, -0.5, 0, 0.5, 2, -3, 3, 1]
        var v = try! Vector<Dim8>(vals)
        let range: ClosedRange<Float> = -1...1
        let cl = v.clamped(to: range)
        v.clamp(to: range)
        for i in 0..<8 {
            let expected = Swift.max(range.lowerBound, Swift.min(range.upperBound, vals[i]))
            #expect(approxEqual(cl[i], expected))
            #expect(approxEqual(v[i], expected))
        }
    }

    @Test
    func testAbsoluteValue() {
        let vals: [Float] = [-3, -2, -1, 0, 1, 2, 3, -4]
        let v = try! Vector<Dim8>(vals)
        let ab = v.absoluteValue()
        for i in 0..<8 {
            #expect(approxEqual(ab[i], Swift.abs(vals[i])))
        }
    }

    @Test
    func testSquareRootPositiveValues() {
        let squares: [Float] = [0, 1, 4, 9, 16, 25, 36, 49]
        let v = try! Vector<Dim8>(squares)
        let r = v.squareRoot()
        let expected: [Float] = [0, 1, 2, 3, 4, 5, 6, 7]
        for i in 0..<8 {
            #expect(approxEqual(r[i], expected[i]))
        }
    }

    @Test
    func testSquareRootNegativeProducesNaN() {
        let vals: [Float] = [-1, -4, -9, -16, 0, 1, 4, 9]
        let v = try! Vector<Dim8>(vals)
        let r = v.squareRoot()
        for i in 0..<4 { #expect(r[i].isNaN) }
        let expected: [Float] = [0, 1, 2, 3]
        for i in 4..<8 { #expect(approxEqual(r[i], expected[i - 4])) }
    }

    @Test
    func testLerpClamped_t0ReturnsSelf() {
        let a = try! Vector<Dim8>(Array(0..<8).map { Float($0) })
        let b = try! Vector<Dim8>(Array(0..<8).map { Float(100 + $0) })
        let r = a.lerp(to: b, t: 0)
        for i in 0..<8 { #expect(approxEqual(r[i], a[i])) }
    }

    @Test
    func testLerpClamped_t1ReturnsOther() {
        let a = try! Vector<Dim8>(Array(0..<8).map { Float($0) })
        let b = try! Vector<Dim8>(Array(0..<8).map { Float(100 + $0) })
        let r = a.lerp(to: b, t: 1)
        for i in 0..<8 { #expect(approxEqual(r[i], b[i])) }
    }

    @Test
    func testLerpClamped_OutOfBoundsTIsClamped() {
        let a = try! Vector<Dim8>(Array(0..<8).map { Float($0) })
        let b = try! Vector<Dim8>(Array(0..<8).map { Float(100 + $0) })
        let rNeg = a.lerp(to: b, t: -5)
        let rBig = a.lerp(to: b, t: 5)
        for i in 0..<8 {
            #expect(approxEqual(rNeg[i], a[i]))
            #expect(approxEqual(rBig[i], b[i]))
        }
    }

    @Test
    func testLerpUnclamped_AllowsExtrapolation() {
        let a = Vector<Dim8>.zero
        let b = Vector<Dim8>(repeating: 10)
        let r2 = a.lerpUnclamped(to: b, t: 2)   // expect 20s
        let rn1 = a.lerpUnclamped(to: b, t: -1) // expect -10s
        for i in 0..<8 {
            #expect(approxEqual(r2[i], 20))
            #expect(approxEqual(rn1[i], -10))
        }
    }

    @Test
    func testSmoothstepEndpointsAndMonotonic() {
        let a = Vector<Dim8>.zero
        let b = Vector<Dim8>.ones
        // Endpoints
        let r0 = a.smoothstep(to: b, t: 0)
        let r1 = a.smoothstep(to: b, t: 1)
        for i in 0..<8 {
            #expect(approxEqual(r0[i], 0))
            #expect(approxEqual(r1[i], 1))
        }
        // Monotonicity for ascending t samples
        let ts: [Float] = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
        var prev: Vector<Dim8>? = nil
        for t in ts {
            let r = a.smoothstep(to: b, t: t)
            if let p = prev {
                for i in 0..<8 { #expect(r[i] >= p[i]) }
            }
            prev = r
        }
    }

    @Test
    func testDynamicVectorMinMaxSuccess() {
        let a = DynamicVector([3, -2, 5, 0, 7, -1, 4, 2].map(Float.init))
        let b = DynamicVector([1, -3, 6, 2, -8, -1, 10, 2].map(Float.init))
        let mi = try! a.min(b)
        let ma = try! a.max(b)
        for i in 0..<a.scalarCount {
            #expect(approxEqual(mi[i], Swift.min(a[i], b[i])))
            #expect(approxEqual(ma[i], Swift.max(a[i], b[i])))
        }
    }

    @Test
    func testDynamicVectorMinMaxDimensionMismatchThrows() {
        let a = DynamicVector([1,2,3].map(Float.init))
        let b = DynamicVector([4,5].map(Float.init))
        do {
            _ = try a.min(b)
            Issue.record("Expected dimensionMismatch not thrown")
        } catch let e as VectorError {
            #expect(e.kind == .dimensionMismatch)
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }

    @Test
    func testDynamicVectorClamped() {
        let vals: [Float] = [-2, -0.5, 0, 0.5, 2]
        let v = DynamicVector(vals)
        let r = v.clamped(to: -1...1)
        for i in 0..<v.scalarCount {
            let expected = Swift.max(-1, Swift.min(1, vals[i]))
            #expect(approxEqual(r[i], expected))
        }
    }

    @Test
    func testDynamicVectorLerpClampedAndMismatchThrows() {
        let a = DynamicVector([0, 0, 0, 0].map(Float.init))
        let b = DynamicVector([10, 10, 10, 10].map(Float.init))
        // Clamped behavior
        let r0 = try! a.lerp(to: b, t: 0)
        let r1 = try! a.lerp(to: b, t: 1)
        for i in 0..<4 { #expect(approxEqual(r0[i], 0)); #expect(approxEqual(r1[i], 10)) }
        // Mismatch throws
        let c = DynamicVector([1,2,3].map(Float.init))
        do {
            _ = try a.lerp(to: c, t: 0.5)
            Issue.record("Expected dimensionMismatch not thrown")
        } catch let e as VectorError {
            #expect(e.kind == .dimensionMismatch)
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }
}
