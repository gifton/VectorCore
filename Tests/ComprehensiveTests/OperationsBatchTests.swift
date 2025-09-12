import Testing
@testable import VectorCore

@Suite("Operations Batch Tests")
struct OperationsBatchTests {

    @Test
    func testFindNearest_SortedDistancesContainExpected() async throws {
        let a = Vector<Dim2>(x: 0, y: 0)
        let b = Vector<Dim2>(x: 3, y: 4)
        let c = Vector<Dim2>(x: 1, y: 1)
        let query = Vector<Dim2>(x: 0, y: 0)
        let vectors = [a, b, c]
        let result = try await Operations.findNearest(to: query, in: vectors, k: 3)
        #expect(result.count == 3)
        #expect(result[0].index == 0)
        #expect(result[1].index == 2)
        #expect(result[2].index == 1)
    }

    @Test
    func testFindNearestBatch_TwoQueries() async throws {
        let vectors = [Vector<Dim2>(x: 0, y: 0), Vector<Dim2>(x: 3, y: 4), Vector<Dim2>(x: 1, y: 1)]
        let q1 = Vector<Dim2>(x: 0, y: 0)
        let q2 = Vector<Dim2>(x: 2, y: 2)
        let results = try await Operations.findNearestBatch(queries: [q1, q2], in: vectors, k: 1)
        #expect(results.count == 2)
        #expect(results[0].first?.index == 0)
    }

    @Test
    func testDistanceMatrix_ParallelLargeMatchesSync() async throws {
        let n = 120
        let vectors: [Vector<Dim8>] = (0..<n).map { _ in
            try! Vector<Dim8>((0..<8).map { _ in Float.random(in: -1...1) })
        }
        let parallel = try await Operations.distanceMatrix(between: vectors, and: vectors)
        let serial = SyncBatchOperations.pairwiseDistances(vectors)
        #expect(parallel.count == serial.count && parallel.first?.count == serial.first?.count)
        for i in stride(from: 0, to: n, by: 23) {
            for j in stride(from: 0, to: n, by: 31) {
                #expect(approxEqual(parallel[i][j], serial[i][j], tol: 1e-4))
            }
        }
    }
}

