import Testing
@testable import VectorCore

@Suite("Operations Transform Tests")
struct OperationsTransformTests {
    @Test
    func testMap_StaticVectors_SerialAndParallel() async throws {
        // Serial sized input
        let small: [Vector<Dim4>] = (0..<5).map { i in try! Vector<Dim4>([Float(i), 1, 2, 3]) }
        let smallMapped = try await Operations.map(small) { $0 + 2 }
        #expect(smallMapped.count == small.count)
        for (i, v) in smallMapped.enumerated() {
            #expect(approxEqual(v[0], Float(i) + 2))
            #expect(approxEqual(v[1], 3))
        }

        // Parallel sized input
        let count = 1500
        let large: [Vector<Dim4>] = (0..<count).map { _ in try! Vector<Dim4>([1, 2, 3, 4]) }
        let largeMapped = try await Operations.map(large) { $0 - 1 }
        #expect(largeMapped.count == count)
        for v in largeMapped { #expect(approxEqual(v[0], 0)) }
    }

    @Test
    func testNormalize_StaticVectors_ZeroAndNonZero() async throws {
        let a = try! Vector<Dim4>([3, 4, 0, 0])
        let z = Vector<Dim4>.zero
        let out = try await Operations.normalize([a, z])
        #expect(out.count == 2)
        #expect(approxEqual(out[0].magnitude, 1, tol: 1e-6))
        #expect(approxEqual(out[1].magnitude, 0, tol: 1e-7))
    }

    @Test
    func testCombine_ElementwiseSumAndErrors() async throws {
        let v1: [Vector<Dim2>] = (0..<10).map { i in Vector<Dim2>(x: Float(i), y: 1) }
        let v2: [Vector<Dim2>] = (0..<10).map { i in Vector<Dim2>(x: 1, y: Float(i)) }
        let sum = try await Operations.combine(v1, v2, +)
        #expect(sum.count == 10)
        for i in 0..<10 {
            #expect(approxEqual(sum[i][0], Float(i) + 1))
            #expect(approxEqual(sum[i][1], Float(i) + 1))
        }

        // Count mismatch throws
        do {
            _ = try await Operations.combine(Array(v1.dropLast()), v2, +)
            Issue.record("Expected invalidDimension not thrown")
        } catch let e as VectorError { #expect(e.kind == .invalidDimension) } catch {
            Issue.record("Unexpected error: \(error)")
        }

        // Dimension mismatch throws (DynamicVector with inconsistent dims)
        let d1 = [DynamicVector([1, 2])]
        let d2 = [DynamicVector([1, 2, 3])]
        do {
            _ = try await Operations.combine(d1, d2, +)
            Issue.record("Expected dimensionMismatch not thrown")
        } catch let e as VectorError { #expect(e.kind == .dimensionMismatch) } catch { Issue.record("Unexpected: \(error)") }
    }

    @Test
    func testCentroid_StaticAndDynamic() {
        // Static vectors
        let s1 = Vector<Dim2>(x: 0, y: 0)
        let s2 = Vector<Dim2>(x: 2, y: 2)
        let sc = Operations.centroid(of: [s1, s2])
        #expect(approxEqual(sc[0], 1) && approxEqual(sc[1], 1))

        // Dynamic vectors
        let d1 = DynamicVector([0, 0, 0])
        let d2 = DynamicVector([3, 6, 9])
        let dc: DynamicVector = Operations.centroid(of: [d1, d2])
        #expect(approxEqual(dc[0], 1.5) && approxEqual(dc[1], 3) && approxEqual(dc[2], 4.5))
    }

    @Test
    func testArrayConveniences_MapAndNormalize() async throws {
        let arr: [Vector<Dim4>] = (0..<4).map { _ in try! Vector<Dim4>([1, 2, 3, 4]) }
        let mapped = try await arr.mapElements { $0 * 3 }
        for v in mapped { #expect(approxEqual(v[0], 3)) }
        let normalized = try await arr.normalized()
        for v in normalized { #expect(approxEqual(v.magnitude, 1, tol: 1e-6)) }
    }
}

