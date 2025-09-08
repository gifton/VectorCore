import Testing
@testable import VectorCore

@Suite("Execution Operations Tests")
struct ExecutionOperationsTests {
    // A minimal GPU-like context for testing the GPU branch selection
    private struct FakeGPUContext: ExecutionContext {
        let device: ComputeDevice = .gpu()
        let maxThreadCount: Int = 8
        let preferredChunkSize: Int = 1024
        func execute<T>(_ work: @Sendable @escaping () throws -> T) async throws -> T where T : Sendable { try work() }
        func execute<T>(priority: TaskPriority?, _ work: @Sendable @escaping () throws -> T) async throws -> T where T : Sendable { try work() }
    }

    @Test
    func testFindNearestBasic_TopKCorrectOrder() async throws {
        // Three 2D points and a query
        let a = Vector<Dim2>(x: 0, y: 0)
        let b = Vector<Dim2>(x: 3, y: 4) // dist 5
        let c = Vector<Dim2>(x: 1, y: 1) // dist sqrt(2)
        let query = Vector<Dim2>(x: 0, y: 0)
        let vectors = [a, b, c]
        let result = try await ExecutionOperations.findNearest(to: query, in: vectors, k: 2)
        #expect(result.count == 2)
        // nearest should be a (0.0) then c (~1.414)
        #expect(result[0].index == 0)
        #expect(result[1].index == 2)
    }

    @Test
    func testFindNearest_KGreaterThanCountClamps() async throws {
        let vectors = [Vector<Dim2>(x: 0, y: 0), Vector<Dim2>(x: 1, y: 1)]
        let query = Vector<Dim2>(x: 0, y: 0)
        let result = try await ExecutionOperations.findNearest(to: query, in: vectors, k: 10)
        #expect(result.count == vectors.count)
    }

    @Test
    func testFindNearest_ZeroOrEmptyInputsReturnEmpty() async throws {
        let vectors = [Vector<Dim2>(x: 0, y: 0), Vector<Dim2>(x: 1, y: 1)]
        let query = Vector<Dim2>(x: 0, y: 0)
        let r0 = try await ExecutionOperations.findNearest(to: query, in: vectors, k: 0)
        #expect(r0.isEmpty)
        let r1 = try await ExecutionOperations.findNearest(to: query, in: [Vector<Dim2>](), k: 1)
        #expect(r1.isEmpty)
    }

    @Test
    func testFindNearest_DimensionMismatchThrows() async {
        let query = DynamicVector([0, 0, 0, 0])
        let vectors = [DynamicVector([1, 2, 3])] // dim mismatch
        do {
            _ = try await ExecutionOperations.findNearest(to: query, in: vectors, k: 1)
            Issue.record("Expected dimensionMismatch not thrown")
        } catch let e as VectorError {
            #expect(e.kind == .dimensionMismatch)
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }

    @Test
    func testFindNearest_CustomMetricManhattan() async throws {
        let query = Vector<Dim3>(x: 0, y: 0, z: 0)
        let v1 = Vector<Dim3>(x: 1, y: 0, z: 0) // L1=1
        let v2 = Vector<Dim3>(x: 0, y: 2, z: 0) // L1=2
        let res = try await ExecutionOperations.findNearest(to: query, in: [v2, v1], k: 1, metric: ManhattanDistance())
        #expect(res.first?.index == 1)
        #expect(approxEqual(res.first?.distance ?? -1, 1))
    }

    @Test
    func testFindNearest_StableWithEqualDistances() async throws {
        let query = Vector<Dim2>(x: 0, y: 0)
        let a = Vector<Dim2>(x: 1, y: 0)
        let b = Vector<Dim2>(x: -1, y: 0)
        let res = try await ExecutionOperations.findNearest(to: query, in: [a, b], k: 2)
        #expect(res.count == 2)
        #expect(approxEqual(res[0].distance, res[1].distance))
        let idxs = Set(res.map { $0.index })
        #expect(idxs == Set([0,1]))
    }

    @Test
    func testFindNearest_ParallelPathCorrectness() async throws {
        // Build > parallelThreshold candidates of Dim8
        let count = 1100
        let queryArr = (0..<8).map { _ in Float.random(in: -1...1) }
        let query = try! Vector<Dim8>(queryArr)
        let vectors: [Vector<Dim8>] = (0..<count).map { _ in
            try! Vector<Dim8>((0..<8).map { _ in Float.random(in: -1...1) })
        }
        let k = 5
        let result = try await ExecutionOperations.findNearest(to: query, in: vectors, k: k, context: CPUContext.automatic)
        // Baseline sequential using SyncBatchOperations
        let baseline = SyncBatchOperations.findNearest(to: query, in: vectors, k: k)
        #expect(result.count == k && baseline.count == k)
        // Compare sets of indices (ordering may match, but ensure set equality)
        #expect(Set(result.map { $0.index }) == Set(baseline.map { $0.index }))
    }

    @Test
    func testFindNearest_ParallelRespectsContextCPU() async throws {
        let count = 1100
        let query = try! Vector<Dim8>((0..<8).map { _ in Float.random(in: -1...1) })
        let vectors: [Vector<Dim8>] = (0..<count).map { _ in
            try! Vector<Dim8>((0..<8).map { _ in Float.random(in: -1...1) })
        }
        let _ = try await ExecutionOperations.findNearest(to: query, in: vectors, k: 3, context: CPUContext.highPerformance)
    }

    @Test
    func testFindNearest_GPUPathThrowsUnsupported() async {
        // Force GPU path by using large dataset and a fake GPU context
        let count = 1100
        let query = try! Vector<Dim8>((0..<8).map { _ in Float.random(in: -1...1) })
        let vectors: [Vector<Dim8>] = (0..<count).map { _ in
            try! Vector<Dim8>((0..<8).map { _ in Float.random(in: -1...1) })
        }
        do {
            _ = try await ExecutionOperations.findNearest(to: query, in: vectors, k: 3, context: FakeGPUContext())
            Issue.record("Expected unsupported device path not thrown")
        } catch let e as VectorError {
            #expect(e.kind == .invalidDimension)
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }

    @Test
    func testFindNearestBatch_BasicTwoQueries() async throws {
        let vectors = [Vector<Dim2>(x: 0, y: 0), Vector<Dim2>(x: 3, y: 4), Vector<Dim2>(x: 1, y: 1)]
        let q1 = Vector<Dim2>(x: 0, y: 0)
        let q2 = Vector<Dim2>(x: 2, y: 2)
        let results = try await ExecutionOperations.findNearestBatch(queries: [q1, q2], in: vectors, k: 1)
        #expect(results.count == 2)
        #expect(results[0].first?.index == 0) // nearest to (0,0) is index 0
    }

    @Test
    func testFindNearestBatch_ValidatesNonEmptyQueries() async {
        do {
            _ = try await ExecutionOperations.findNearestBatch(queries: [Vector<Dim2>](), in: [Vector<Dim2>(x: 0, y: 0)])
            Issue.record("Expected invalidDimension for empty queries not thrown")
        } catch let e as VectorError {
            #expect(e.kind == .invalidDimension)
        } catch { Issue.record("Unexpected error: \(error)") }
    }

    @Test
    func testFindNearestBatch_ValidatesNonEmptyVectors() async {
        do {
            _ = try await ExecutionOperations.findNearestBatch(queries: [Vector<Dim2>(x: 0, y: 0)], in: [Vector<Dim2>]())
            Issue.record("Expected invalidDimension for empty vectors not thrown")
        } catch let e as VectorError {
            #expect(e.kind == .invalidDimension)
        } catch { Issue.record("Unexpected error: \(error)") }
    }

    @Test
    func testFindNearestBatch_ValidatesConsistentQueryDims() async {
        // Make 101 queries so index 100 is checked in release sampling
        var queries: [DynamicVector] = (0..<100).map { _ in DynamicVector([0,0,0]) }
        queries.append(DynamicVector([0,0])) // mismatched dimension at index 100
        let vectors = [DynamicVector([1,2,3])]
        do {
            _ = try await ExecutionOperations.findNearestBatch(queries: queries, in: vectors, k: 1)
            Issue.record("Expected dimensionMismatch not thrown for inconsistent queries")
        } catch let e as VectorError {
            #expect(e.kind == .dimensionMismatch)
        } catch { Issue.record("Unexpected error: \(error)") }
    }

    @Test
    func testFindNearestBatch_QueryDimMismatchWithVectors() async {
        let queries = [DynamicVector([0,0,0])]
        let vectors = [DynamicVector([0,0])]
        do {
            _ = try await ExecutionOperations.findNearestBatch(queries: queries, in: vectors, k: 1)
            Issue.record("Expected dimensionMismatch not thrown for query-vectors mismatch")
        } catch let e as VectorError {
            #expect(e.kind == .dimensionMismatch)
        } catch { Issue.record("Unexpected error: \(error)") }
    }

    @Test
    func testMapTransformSequential() async throws {
        let vectors: [Vector<Dim4>] = (0..<5).map { i in try! Vector<Dim4>([Float(i),1,2,3]) }
        let mapped = try await ExecutionOperations.map(vectors, transform: { $0 * 2 }, context: CPUContext.sequential)
        for (i, v) in mapped.enumerated() {
            #expect(approxEqual(v[0], Float(i) * 2))
            #expect(approxEqual(v[1], 2))
        }
    }

    @Test
    func testMapTransformParallel() async throws {
        let count = 1500
        let vectors: [Vector<Dim4>] = (0..<count).map { _ in try! Vector<Dim4>([1,2,3,4]) }
        let mapped = try await ExecutionOperations.map(vectors, transform: { $0 + 1 }, context: CPUContext.automatic)
        #expect(mapped.count == count)
        for v in mapped { #expect(approxEqual(v[0], 2)) }
    }

    @Test
    func testCentroidBasic() async throws {
        let v1 = Vector<Dim2>(x: 0, y: 0)
        let v2 = Vector<Dim2>(x: 2, y: 2)
        let centroid = try await ExecutionOperations.centroid(of: [v1, v2])
        #expect(approxEqual(centroid[0], 1))
        #expect(approxEqual(centroid[1], 1))
    }

    @Test
    func testReduceSequentialAndParallelEquivalence() async throws {
        let count = 1200
        let vectors: [Vector<Dim2>] = (0..<count).map { i in Vector<Dim2>(x: Float(i%3), y: 1) }
        let zero = Vector<Dim2>.zero
        // Sequential
        let seq = try await ExecutionOperations.reduce(vectors, zero, +, context: CPUContext.sequential)
        // Parallel
        let par = try await ExecutionOperations.reduce(vectors, zero, +, context: CPUContext.automatic)
        #expect(approxEqual(seq[0], par[0], tol: 1e-5))
        #expect(approxEqual(seq[1], par[1], tol: 1e-5))
    }

}
