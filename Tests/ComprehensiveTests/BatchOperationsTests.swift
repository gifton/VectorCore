import Testing
@testable import VectorCore

@Suite("Batch Operations Tests")
struct BatchOperationsTests {
    
    // MARK: - findNearest

    @Test
    func testFindNearest_SmallSet_ReturnsSortedByDistance() async {
        let a = Vector<Dim2>(x: 0, y: 0)
        let b = Vector<Dim2>(x: 3, y: 4) // dist 5
        let c = Vector<Dim2>(x: 1, y: 1) // dist sqrt(2)
        let query = Vector<Dim2>(x: 0, y: 0)
        let vectors = [a, b, c]
        let result = await BatchOperations.findNearest(to: query, in: vectors, k: 3)
        #expect(result.count == 3)
        // Expect a (0), c (~1.414), b (5)
        #expect(result[0].index == 0)
        #expect(result[1].index == 2)
        #expect(result[2].index == 1)
        #expect(approxEqual(result[0].distance, 0))
        #expect(approxEqual(result[1].distance, (2 as Float).squareRoot(), tol: 1e-5))
        #expect(approxEqual(result[2].distance, 5))
    }

    @Test
    func testFindNearest_KGreaterThanCount_ReturnsAll() async {
        let vectors = [Vector<Dim2>(x: 0, y: 0), Vector<Dim2>(x: 1, y: 0)]
        let query = Vector<Dim2>(x: 0, y: 0)
        let result = await BatchOperations.findNearest(to: query, in: vectors, k: 10)
        #expect(result.count == vectors.count)
    }

    @Test
    func testFindNearest_KZeroOrEmptyVectors_ReturnsEmpty() async {
        let vectors = [Vector<Dim2>(x: 0, y: 0), Vector<Dim2>(x: 1, y: 1)]
        let query = Vector<Dim2>(x: 0, y: 0)
        let r0 = await BatchOperations.findNearest(to: query, in: vectors, k: 0)
        #expect(r0.isEmpty)
        let r1 = await BatchOperations.findNearest(to: query, in: [Vector<Dim2>](), k: 1)
        #expect(r1.isEmpty)
    }

    @Test
    func testFindNearest_CustomMetric_ManhattanOrdering() async {
        let query = Vector<Dim3>(x: 0, y: 0, z: 0)
        let v1 = Vector<Dim3>(x: 1, y: 0, z: 0) // L1=1
        let v2 = Vector<Dim3>(x: 0, y: 2, z: 0) // L1=2
        let res = await BatchOperations.findNearest(to: query, in: [v2, v1], k: 2, metric: ManhattanDistance())
        #expect(res.count == 2)
        #expect(res[0].index == 1) // v1 is closer under L1
        #expect(approxEqual(res[0].distance, 1))
    }

    @Test
    func testFindNearest_EqualDistances_ContainsAllEqualNeighbors() async {
        let query = Vector<Dim2>(x: 0, y: 0)
        let a = Vector<Dim2>(x: 1, y: 0)
        let b = Vector<Dim2>(x: -1, y: 0)
        let res = await BatchOperations.findNearest(to: query, in: [a, b], k: 2)
        #expect(res.count == 2)
        #expect(approxEqual(res[0].distance, res[1].distance, tol: 1e-6))
        let idxs = Set(res.map { $0.index })
        #expect(idxs == Set([0,1]))
    }

    @Test
    func testFindNearest_LargeDataset_ParallelMatchesSyncBaseline() async {
        let count = 1100 // > default parallelThreshold (1000)
        let queryArr = (0..<8).map { _ in Float.random(in: -1...1) }
        let query = try! Vector<Dim8>(queryArr)
        let vectors: [Vector<Dim8>] = (0..<count).map { _ in
            try! Vector<Dim8>((0..<8).map { _ in Float.random(in: -1...1) })
        }
        let k = 5
        let parallel = await BatchOperations.findNearest(to: query, in: vectors, k: k)
        let baseline = SyncBatchOperations.findNearest(to: query, in: vectors, k: k)
        #expect(parallel.count == k && baseline.count == k)
        #expect(Set(parallel.map { $0.index }) == Set(baseline.map { $0.index }))
    }

    // MARK: - pairwiseDistances

    @Test
    func testPairwiseDistances_SmallMatrix_SymmetricZeroDiagonal() async {
        let a = Vector<Dim2>(x: 0, y: 0)
        let b = Vector<Dim2>(x: 3, y: 4)
        let c = Vector<Dim2>(x: 1, y: 1)
        let m = await BatchOperations.pairwiseDistances([a, b, c])
        #expect(m.count == 3 && m[0].count == 3)
        for i in 0..<3 { #expect(approxEqual(m[i][i], 0)) }
        #expect(approxEqual(m[0][1], 5) && approxEqual(m[1][0], 5))
        #expect(approxEqual(m[0][2], (2 as Float).squareRoot(), tol: 1e-5))
        #expect(approxEqual(m[2][0], (2 as Float).squareRoot(), tol: 1e-5))
    }

    @Test
    func testPairwiseDistances_LargeMatrix_ParallelMatchesSync() async {
        let n = 120 // >= 100 triggers parallel path
        let vectors: [Vector<Dim8>] = (0..<n).map { _ in
            try! Vector<Dim8>((0..<8).map { _ in Float.random(in: -1...1) })
        }
        let parallel = await BatchOperations.pairwiseDistances(vectors)
        let serial = SyncBatchOperations.pairwiseDistances(vectors)
        #expect(parallel.count == serial.count && parallel.first?.count == serial.first?.count)
        // spot check several entries
        for i in stride(from: 0, to: n, by: 23) {
            for j in stride(from: 0, to: n, by: 31) {
                #expect(approxEqual(parallel[i][j], serial[i][j], tol: 1e-4))
            }
        }
    }

    // MARK: - process (batched transform)

    @Test
    func testProcess_BatchOrderPreserved() async throws {
        // Unique first element encodes original index
        let input: [Vector<Dim2>] = (0..<20).map { i in Vector<Dim2>(x: Float(i), y: 1) }
        let output = try await BatchOperations.process(input, batchSize: 4) { batch in
            batch.map { v in v * 2 }
        }
        #expect(output.count == input.count)
        for i in 0..<input.count { #expect(approxEqual(output[i][0], input[i][0] * 2)) }
    }

    @Test
    func testProcess_CustomBatchSize_RespectsBoundaries() async throws {
        let input: [Vector<Dim2>] = (0..<10).map { i in Vector<Dim2>(x: Float(i), y: 0) }
        let output: [DynamicVector] = try await BatchOperations.process(input, batchSize: 3) { batch in
            // Encode batch size into output so we can verify 3,3,3,1
            let size = Float(batch.count)
            return batch.map { _ in DynamicVector([size]) }
        }
        #expect(output.count == input.count)
        // Expect first 3 -> 3, next 3 -> 3, next 3 -> 3, last -> 1
        for i in 0..<3 { #expect(approxEqual(output[i][0], 3)) }
        for i in 3..<6 { #expect(approxEqual(output[i][0], 3)) }
        for i in 6..<9 { #expect(approxEqual(output[i][0], 3)) }
        #expect(approxEqual(output[9][0], 1))
    }

    @Test
    func testProcess_EmptyInput_ReturnsEmpty() async throws {
        let empty: [Vector<Dim4>] = []
        let out = try await BatchOperations.process(empty) { (batch: [Vector<Dim4>]) in
            batch // identity
        }
        #expect(out.isEmpty)
    }

    @Test
    func testProcess_TransformThrows_PropagatesError() async {
        enum TestError: Error { case boom }
        let input: [Vector<Dim2>] = (0..<5).map { i in Vector<Dim2>(x: Float(i), y: 0) }
        do {
            _ = try await BatchOperations.process(input, batchSize: 2) { batch in
                if batch.contains(where: { approxEqual($0[0], 2) }) { throw TestError.boom }
                return batch
            }
            Issue.record("Expected error not thrown")
        } catch { /* ok */ }
    }

    // MARK: - map (element-wise transform)

    @Test
    func testMap_SerialThreshold_OrderPreserved() async throws {
        let input: [Vector<Dim4>] = (0..<10).map { i in try! Vector<Dim4>([Float(i), 1, 1, 1]) }
        let out = try await BatchOperations.map(input) { v in v * 2 }
        #expect(out.count == input.count)
        for i in 0..<input.count { #expect(approxEqual(out[i][0], input[i][0] * 2)) }
    }

    @Test
    func testMap_ParallelThreshold_OrderPreserved() async throws {
        let count = 1100
        let input: [Vector<Dim4>] = (0..<count).map { i in try! Vector<Dim4>([Float(i), 0, 0, 0]) }
        let out = try await BatchOperations.map(input) { v in v * 2 }
        #expect(out.count == input.count)
        // spot-check order alignment
        for i in stride(from: 0, to: count, by: 137) {
            #expect(approxEqual(out[i][0], input[i][0] * 2))
        }
    }

    @Test
    func testMap_TransformThrows_PropagatesError() async {
        enum TestError: Error { case bad }
        let input: [Vector<Dim2>] = (0..<20).map { i in Vector<Dim2>(x: Float(i), y: 0) }
        do {
            _ = try await BatchOperations.map(input) { v in
                if approxEqual(v[0], 7) { throw TestError.bad }
                return v
            }
            Issue.record("Expected error not thrown")
        } catch { /* ok */ }
    }

    // MARK: - filter

    @Test
    func testFilter_SerialThreshold_CorrectSubset() async throws {
        let input: [Vector<Dim2>] = (0..<10).map { i in Vector<Dim2>(x: Float(i), y: 0) }
        let out = try await BatchOperations.filter(input) { v in v[0] > 4 }
        #expect(out.count == 5)
        #expect(out.map { Int($0[0]) } == [5,6,7,8,9])
    }

    @Test
    func testFilter_ParallelThreshold_CorrectSubset() async throws {
        let count = 1100
        let input: [Vector<Dim2>] = (0..<count).map { i in Vector<Dim2>(x: Float(i), y: 0) }
        let out = try await BatchOperations.filter(input) { v in Int(v[0]) % 2 == 0 }
        #expect(out.count == (count + 1) / 2)
        // spot check
        for i in stride(from: 0, to: out.count, by: 113) {
            #expect(Int(out[i][0]) % 2 == 0)
        }
    }

    @Test
    func testFilter_PredicateThrows_PropagatesError() async {
        enum TestError: Error { case predicate }
        let input: [Vector<Dim2>] = (0..<30).map { i in Vector<Dim2>(x: Float(i), y: 0) }
        do {
            _ = try await BatchOperations.filter(input) { v in
                if approxEqual(v[0], 11) { throw TestError.predicate }
                return true
            }
            Issue.record("Expected error not thrown")
        } catch { /* ok */ }
    }

    // MARK: - statistics

    @Test
    func testStatistics_Empty_ReturnsZeros() async {
        let empty: [Vector<Dim8>] = []
        let stats = await BatchOperations.statistics(for: empty)
        #expect(stats.count == 0)
        #expect(approxEqual(stats.meanMagnitude, 0))
        #expect(approxEqual(stats.stdMagnitude, 0))
    }

    @Test
    func testStatistics_KnownDataset_CorrectMeanStd() async {
        let a = Vector<Dim2>(x: 3, y: 4) // mag 5
        let b = Vector<Dim2>(x: 0, y: 0) // mag 0
        let stats = await BatchOperations.statistics(for: [a, b])
        #expect(stats.count == 2)
        #expect(approxEqual(stats.meanMagnitude, 2.5, tol: 1e-6))
        #expect(approxEqual(stats.stdMagnitude, 2.5, tol: 1e-6))
    }

    @Test
    func testStatistics_LargeDataset_ParallelMatchesSerial() async {
        let count = 1500
        let input: [Vector<Dim8>] = (0..<count).map { _ in
            try! Vector<Dim8>((0..<8).map { _ in Float.random(in: -1...1) })
        }
        let parallel = await BatchOperations.statistics(for: input)
        let serial = SyncBatchOperations.statistics(for: input)
        #expect(parallel.count == serial.count)
        #expect(approxEqual(parallel.meanMagnitude, serial.meanMagnitude, tol: 1e-4))
        #expect(approxEqual(parallel.stdMagnitude, serial.stdMagnitude, tol: 1e-4))
    }

    // MARK: - sample

    @Test
    func testSample_KLessThanCount_UniqueCount() {
        let input = Array(0..<100)
        let out = BatchOperations.sample(input, k: 10)
        #expect(out.count == 10)
        // Ensure uniqueness given unique input
        #expect(Set(out).count == out.count)
    }

    @Test
    func testSample_KGreaterThanCount_ReturnsAll() {
        let input = Array(0..<20)
        let out = BatchOperations.sample(input, k: 50)
        #expect(out == input)
    }

    @Test
    func testSample_KZero_ReturnsEmpty() {
        let input = Array(0..<10)
        let out = BatchOperations.sample(input, k: 0)
        #expect(out.isEmpty)
    }

    // MARK: - configuration

    @Test
    func testConfiguration_UpdateParallelThresholdAffectsBehavior() async {
        // Verify configuration update/read, without asserting internal behavior
        let original = await BatchOperations.configuration()
        let newThreshold = max(10, original.parallelThreshold / 2)
        await BatchOperations.updateConfiguration { config in
            config.parallelThreshold = newThreshold
        }
        let updated = await BatchOperations.configuration()
        #expect(updated.parallelThreshold == newThreshold)
        // Restore
        await BatchOperations.updateConfiguration { config in
            config.parallelThreshold = original.parallelThreshold
        }
    }
}
