import Testing
@testable import VectorCore

@Suite("Operations Batch Tests")
struct OperationsBatchTests {
    // MARK: - findNearest

    @Test
    func testFindNearest_SmallSet_ReturnsSortedByDistance() async throws {
        let a = Vector<Dim2>(x: 0, y: 0)
        let b = Vector<Dim2>(x: 3, y: 4) // dist 5
        let c = Vector<Dim2>(x: 1, y: 1) // dist sqrt(2)
        let query = Vector<Dim2>(x: 0, y: 0)
        let vectors = [a, b, c]
        let result = try await Operations.findNearest(to: query, in: vectors, k: 3)
        #expect(result.count == 3)
        #expect(result[0].index == 0)
        #expect(result[1].index == 2)
        #expect(result[2].index == 1)
        #expect(approxEqual(result[0].distance, 0))
        #expect(approxEqual(result[1].distance, (2 as Float).squareRoot(), tol: 1e-5))
        #expect(approxEqual(result[2].distance, 5))
    }

    @Test
    func testFindNearest_KGreaterThanCount_ReturnsAll() async throws {
        let vectors = [Vector<Dim2>(x: 0, y: 0), Vector<Dim2>(x: 1, y: 0)]
        let query = Vector<Dim2>(x: 0, y: 0)
        let result = try await Operations.findNearest(to: query, in: vectors, k: 10)
        #expect(result.count == vectors.count)
    }

    @Test
    func testFindNearest_InvalidInputsThrow() async {
        let vectors = [Vector<Dim2>(x: 0, y: 0), Vector<Dim2>(x: 1, y: 1)]
        let query = Vector<Dim2>(x: 0, y: 0)
        do { _ = try await Operations.findNearest(to: query, in: vectors, k: 0); Issue.record("Expected error not thrown") }
        catch let e as VectorError { #expect(e.kind == .invalidDimension) }
        catch { Issue.record("Unexpected error: \(error)") }
        do { _ = try await Operations.findNearest(to: query, in: [Vector<Dim2>](), k: 1); Issue.record("Expected error not thrown") }
        catch let e as VectorError { #expect(e.kind == .invalidDimension) }
        catch { Issue.record("Unexpected error: \(error)") }
    }

    @Test
    func testFindNearest_CustomMetric_ManhattanOrdering() async throws {
        let query = Vector<Dim3>(x: 0, y: 0, z: 0)
        let v1 = Vector<Dim3>(x: 1, y: 0, z: 0) // L1=1
        let v2 = Vector<Dim3>(x: 0, y: 2, z: 0) // L1=2
        let res = try await Operations.findNearest(to: query, in: [v2, v1], k: 2, metric: ManhattanDistance())
        #expect(res.count == 2)
        #expect(res[0].index == 1)
        #expect(approxEqual(res[0].distance, 1))
    }

    @Test
    func testFindNearest_EqualDistances_ContainsAllEqualNeighbors() async throws {
        let query = Vector<Dim2>(x: 0, y: 0)
        let a = Vector<Dim2>(x: 1, y: 0)
        let b = Vector<Dim2>(x: -1, y: 0)
        let res = try await Operations.findNearest(to: query, in: [a, b], k: 2)
        #expect(res.count == 2)
        #expect(approxEqual(res[0].distance, res[1].distance, tol: 1e-6))
        let idxs = Set(res.map { $0.index })
        #expect(idxs == Set([0,1]))
    }

    @Test
    func testFindNearest_LargeDataset_ParallelMatchesSyncBaseline() async throws {
        let count = 1100
        let queryArr = (0..<8).map { _ in Float.random(in: -1...1) }
        let query = try! Vector<Dim8>(queryArr)
        let vectors: [Vector<Dim8>] = (0..<count).map { _ in
            try! Vector<Dim8>((0..<8).map { _ in Float.random(in: -1...1) })
        }
        let k = 5
        let parallel = try await Operations.findNearest(to: query, in: vectors, k: k)
        let baseline = SyncBatchOperations.findNearest(to: query, in: vectors, k: k)
        #expect(parallel.count == k && baseline.count == k)
        #expect(Set(parallel.map { $0.index }) == Set(baseline.map { $0.index }))
    }

    // MARK: - distanceMatrix

    @Test
    func testPairwiseDistances_SmallMatrix_SymmetricZeroDiagonal() async throws {
        let a = Vector<Dim2>(x: 0, y: 0)
        let b = Vector<Dim2>(x: 3, y: 4)
        let c = Vector<Dim2>(x: 1, y: 1)
        let m = try await Operations.distanceMatrix([a, b, c])
        #expect(m.count == 3 && m[0].count == 3)
        for i in 0..<3 { #expect(approxEqual(m[i][i], 0)) }
        #expect(approxEqual(m[0][1], 5) && approxEqual(m[1][0], 5))
        #expect(approxEqual(m[0][2], (2 as Float).squareRoot(), tol: 1e-5))
        #expect(approxEqual(m[2][0], (2 as Float).squareRoot(), tol: 1e-5))
    }

    @Test
    func testPairwiseDistances_LargeMatrix_Symmetric() async throws {
        let n = 120
        let vectors: [Vector<Dim8>] = (0..<n).map { _ in
            try! Vector<Dim8>((0..<8).map { _ in Float.random(in: -1...1) })
        }
        let m = try await Operations.distanceMatrix(vectors)
        #expect(m.count == n && m.first?.count == n)
        for i in stride(from: 0, to: n, by: 23) {
            for j in stride(from: 0, to: n, by: 31) {
                #expect(approxEqual(m[i][j], m[j][i], tol: 1e-4))
            }
        }
    }

    // MARK: - statistics

    @Test
    func testStatistics_Empty_ReturnsZeros() {
        let empty: [Vector<Dim8>] = []
        let stats = Operations.statistics(for: empty)
        #expect(stats.count == 0)
        #expect(stats.dimensions == 0)
    }

    @Test
    func testStatistics_KnownDataset_ComponentwiseMean() {
        let a = try! Vector<Dim2>([1,3])
        let b = try! Vector<Dim2>([3,5])
        let stats = Operations.statistics(for: [a, b])
        #expect(stats.count == 2)
        #expect(stats.dimensions == 2)
        #expect(approxEqual(stats.mean[0], 2))
        #expect(approxEqual(stats.mean[1], 4))
        #expect(approxEqual(stats.min[0], 1) && approxEqual(stats.max[0], 3))
    }
}

