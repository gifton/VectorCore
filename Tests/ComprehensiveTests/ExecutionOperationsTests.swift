import Testing
@testable import VectorCore

@Suite("Operations Tests")
struct OperationsTests {

    @Test
    func testFindNearest_BasicTopKOrder() async throws {
        let a = Vector<Dim2>(x: 0, y: 0)
        let b = Vector<Dim2>(x: 3, y: 4) // dist 5
        let c = Vector<Dim2>(x: 1, y: 1) // dist sqrt(2)
        let query = Vector<Dim2>(x: 0, y: 0)
        let vectors = [a, b, c]
        let result = try await Operations.findNearest(to: query, in: vectors, k: 2)
        #expect(result.count == 2)
        #expect(result.map { $0.index }.contains(0))
        #expect(result.map { $0.index }.contains(2))
    }

    @Test
    func testFindNearest_KGreaterThanCount_Clamps() async throws {
        let vectors = [Vector<Dim2>(x: 0, y: 0), Vector<Dim2>(x: 1, y: 1)]
        let query = Vector<Dim2>(x: 0, y: 0)
        let result = try await Operations.findNearest(to: query, in: vectors, k: 10)
        #expect(result.count == vectors.count)
    }

    @Test
    func testFindNearest_CustomMetric_Manhattan() async throws {
        let query = Vector<Dim3>(x: 0, y: 0, z: 0)
        let v1 = Vector<Dim3>(x: 1, y: 0, z: 0) // L1=1
        let v2 = Vector<Dim3>(x: 0, y: 2, z: 0) // L1=2
        let res = try await Operations.findNearest(to: query, in: [v2, v1], k: 1, metric: ManhattanDistance())
        #expect(res.first?.index == 1)
        #expect(approxEqual(res.first?.distance ?? -1, 1))
    }

    @Test
    func testFindNearest_EqualDistances_SetEquality() async throws {
        let query = Vector<Dim2>(x: 0, y: 0)
        let a = Vector<Dim2>(x: 1, y: 0)
        let b = Vector<Dim2>(x: -1, y: 0)
        let res = try await Operations.findNearest(to: query, in: [a, b], k: 2)
        #expect(res.count == 2)
        #expect(approxEqual(res[0].distance, res[1].distance))
        let idxs = Set(res.map { $0.index })
        #expect(idxs == Set([0, 1]))
    }

    @Test
    func testFindNearest_ParallelMatchesSyncBaseline() async throws {
        let count = 1100
        let queryArr = (0..<8).map { _ in Float.random(in: -1...1) }
        let query = try! Vector<Dim8>(queryArr)
        let vectors: [Vector<Dim8>] = (0..<count).map { _ in
            try! Vector<Dim8>((0..<8).map { _ in Float.random(in: -1...1) })
        }
        let k = 5
        let result = try await Operations.findNearest(to: query, in: vectors, k: k)
        let baseline = SyncBatchOperations.findNearest(to: query, in: vectors, k: k)
        #expect(result.count == k && baseline.count == k)
        #expect(Set(result.map { $0.index }) == Set(baseline.map { $0.index }))
    }

    @Test
    func testFindNearestBatch_Basic() async throws {
        let vectors = [Vector<Dim2>(x: 0, y: 0), Vector<Dim2>(x: 3, y: 4), Vector<Dim2>(x: 1, y: 1)]
        let q1 = Vector<Dim2>(x: 0, y: 0)
        let q2 = Vector<Dim2>(x: 2, y: 2)
        let results = try await Operations.findNearestBatch(queries: [q1, q2], in: vectors, k: 1)
        #expect(results.count == 2)
        #expect(results[0].first?.index == 0)
    }

    @Test
    func testDistanceMatrix_SymmetricZeroDiagonal() async throws {
        let a = Vector<Dim2>(x: 0, y: 0)
        let b = Vector<Dim2>(x: 3, y: 4)
        let c = Vector<Dim2>(x: 1, y: 1)
        let m = try await Operations.distanceMatrix(between: [a, b, c], and: [a, b, c])
        #expect(m.count == 3 && m[0].count == 3)
        for i in 0..<3 { #expect(approxEqual(m[i][i], 0)) }
        #expect(approxEqual(m[0][1], 5) && approxEqual(m[1][0], 5))
        #expect(approxEqual(m[0][2], (2 as Float).squareRoot(), tol: 1e-5))
        #expect(approxEqual(m[2][0], (2 as Float).squareRoot(), tol: 1e-5))
    }

    @Test
    func testCentroidBasic() {
        let v1 = Vector<Dim2>(x: 0, y: 0)
        let v2 = Vector<Dim2>(x: 2, y: 2)
        let centroid: Vector<Dim2> = Operations.centroid(of: [v1, v2])
        #expect(approxEqual(centroid[0], 1))
        #expect(approxEqual(centroid[1], 1))
    }

    @Test
    func testNormalize_ReturnsUnitVectors() async throws {
        let vectors = [Vector<Dim2>(x: 3, y: 4)]
        let normalized: [Vector<Dim2>] = try await Operations.normalize(vectors)
        #expect(approxEqual(normalized[0].magnitude, 1, tol: 1e-5))
    }

    @Test
    func testStatistics_BasicMeanAndMagnitudes() {
        let a = Vector<Dim2>(x: 3, y: 4) // mag 5
        let b = Vector<Dim2>(x: 0, y: 0) // mag 0
        let stats = Operations.statistics(for: [a, b])
        #expect(stats.count == 2)
        #expect(stats.dimensions == 2)
        #expect(approxEqual(stats.magnitudes.mean, 2.5, tol: 1e-6))
        #expect(approxEqual(stats.magnitudes.max, 5, tol: 1e-6))
    }
}
