import Testing
@testable import VectorCore

@Suite("Top-K CPU wrapper NaN contract")
struct TopKNaNWrapperTests {
    @Test func batchSelectionPreservesInputEdgeBehavior() async {
        let query = Vector<Dim2>(x: 0, y: 0)
        let candidates = [Float.nan, 2, 1, .nan].map { Vector<Dim2>(x: $0, y: 0) }
        await Operations.$computeProvider.withValue(CPUComputeProvider(mode: .sequential)) {
            let zeroK = await BatchOperations.findNearest(to: query, in: candidates, k: 0)
            let negativeK = await BatchOperations.findNearest(to: query, in: candidates, k: -1)
            let empty = await BatchOperations.findNearest(to: query, in: [Vector<Dim2>](), k: 3)
            #expect(zeroK.isEmpty)
            #expect(negativeK.isEmpty)
            #expect(empty.isEmpty)
            for k in [4, 99] {
                let result = await BatchOperations.findNearest(to: query, in: candidates, k: k)
                #expect(result.map(\.index) == [2, 1, 0, 3])
                #expect(result.count == candidates.count)
                #expect(result[0].distance == 1)
                #expect(result[1].distance == 2)
                #expect(result[2].distance.isNaN)
                #expect(result[3].distance.isNaN)
            }
        }
    }

    @Test func genericOperationsSelectsNumericScoresBeforeNaNs() async throws {
        let query = Vector<Dim2>(x: 0, y: 0)
        let candidates = [Float.nan, 2, 1, 1, .nan].map { Vector<Dim2>(x: $0, y: 0) }
        #expect(EuclideanDistance().distance(query, candidates[0]).isNaN)
        #expect(EuclideanDistance().distance(query, candidates[2]) == 1)
        try await Operations.$computeProvider.withValue(CPUComputeProvider(mode: .sequential)) {
            for k in [2, 4, 99] {
                let result = try await Operations.findNearest(to: query, in: candidates, k: k)
                #expect(result.map(\.index) == Array([2, 3, 1, 0, 4].prefix(k)))
                #expect(result.count == min(k, candidates.count))
                for entry in result where entry.index == 0 || entry.index == 4 {
                    #expect(entry.distance.isNaN)
                }
            }
        }
    }

    @Test func optimizedParallelFiniteTiesUseOriginalIndices() async throws {
        let q512 = try Vector512Optimized([Float](repeating: 1, count: 512))
        let q768 = try Vector768Optimized([Float](repeating: 1, count: 768))
        let q1536 = try Vector1536Optimized([Float](repeating: 1, count: 1536))
        try await Operations.$computeProvider.withValue(CPUComputeProvider(mode: .parallel)) {
            let euclid512 = try await Operations.findNearest(to: q512, in: Array(repeating: q512, count: 1024), k: 3)
            let euclid768 = try await Operations.findNearest(to: q768, in: Array(repeating: q768, count: 1024), k: 3)
            let euclid1536 = try await Operations.findNearest(to: q1536, in: Array(repeating: q1536, count: 1024), k: 3)
            let dot = try await Operations.findNearest(to: q512, in: Array(repeating: q512, count: 1024), k: 3, metric: DotProductDistance())
            let cosine = try await Operations.findNearest(to: q512, in: Array(repeating: q512, count: 1024), k: 3, metric: CosineDistance())
            for result in [euclid512, euclid768, euclid1536, dot, cosine] {
                #expect(result.map(\.index) == [0, 1, 2])
            }
        }
    }

    @Test func optimizedParallelEuclideanRetainsNaNsAfterNumericCandidates() async throws {
        let query = try Vector512Optimized([Float](repeating: 0, count: 512))
        let nan = try Vector512Optimized([Float](repeating: .nan, count: 512))
        var candidates = Array(repeating: nan, count: 1024)
        candidates[900] = query
        #expect(EuclideanKernels.squared512(query, nan).isNaN)
        #expect(EuclideanKernels.squared512(query, query) == 0)
        let inputs = candidates
        try await Operations.$computeProvider.withValue(CPUComputeProvider(mode: .parallel)) {
            let result = try await Operations.findNearest(to: query, in: inputs, k: 3)
            #expect(result.map(\.index) == [900, 0, 1])
            #expect(result.count == 3)
            #expect(result[0].distance == 0)
            #expect(result[1].distance.isNaN)
            #expect(result[2].distance.isNaN)
        }
    }

    @Test func optimizedParallelDotNegationKeepsNaNsLast() async throws {
        let query = try Vector512Optimized([Float](repeating: 1, count: 512))
        let nan = try Vector512Optimized([Float](repeating: .nan, count: 512))
        var candidates = Array(repeating: nan, count: 1024)
        candidates[700] = query
        candidates[900] = query
        #expect(DotKernels.dot512(query, nan).isNaN)
        #expect(DotKernels.dot512(query, query) == 512)
        let inputs = candidates
        try await Operations.$computeProvider.withValue(CPUComputeProvider(mode: .parallel)) {
            let result = try await Operations.findNearest(to: query, in: inputs, k: 4, metric: DotProductDistance())
            #expect(result.map(\.index) == [700, 900, 0, 1])
            #expect(result.count == 4)
            #expect(result[0].distance == -512)
            #expect(result[1].distance == -512)
            #expect(result[2].distance.isNaN)
            #expect(result[3].distance.isNaN)
        }
    }

    @Test func batchSerialHeapAndSortKeepNaNsLast() async {
        let query = Vector<Dim2>(x: 0, y: 0)
        var candidates = Array(repeating: Vector<Dim2>(x: .nan, y: 0), count: 100)
        candidates[99] = query
        #expect(!ParallelHeuristic.shouldParallelize(dim: 2, items: 100, variant: .generic, metric: .euclideanLike))
        #expect(EuclideanDistance().distance(query, candidates[0]).isNaN)
        let inputs = candidates
        await Operations.$computeProvider.withValue(CPUComputeProvider(mode: .sequential)) {
            for k in [3, 10] {
                let result = await BatchOperations.findNearest(to: query, in: inputs, k: k)
                #expect(result.map(\.index) == [99] + Array(0..<(k - 1)))
                #expect(result.count == k)
                #expect(result[0].distance == 0)
                #expect(result.dropFirst().allSatisfy { $0.distance.isNaN })
            }
        }
    }

    @Test func batchParallelHeapAndSortPreserveGlobalIndices() async throws {
        let query = try Vector<Dim512>([Float](repeating: 0, count: 512))
        let nan = try Vector<Dim512>([Float](repeating: .nan, count: 512))
        var candidates = Array(repeating: nan, count: 1024)
        candidates[700] = query
        candidates[900] = query
        #expect(ParallelHeuristic.shouldParallelize(dim: 512, items: 1024, variant: .generic, metric: .euclideanLike))
        #expect(EuclideanDistance().distance(query, nan).isNaN)
        let inputs = candidates
        await Operations.$computeProvider.withValue(CPUComputeProvider(mode: .parallel)) {
            for k in [3, 103] {
                let result = await BatchOperations.findNearest(to: query, in: inputs, k: k)
                #expect(result.map(\.index) == [700, 900] + Array(0..<(k - 2)))
                #expect(result.count == k)
                #expect(result[0].distance == 0)
                #expect(result[1].distance == 0)
                #expect(result.dropFirst(2).allSatisfy { $0.distance.isNaN })
            }
        }
    }

    @Test func gemmBatchEuclideanFormattingPreservesSelectedNaNs() async throws {
        let query = try Vector512Optimized([Float](repeating: 0, count: 512))
        let nan = try Vector512Optimized([Float](repeating: .nan, count: 512))
        let queries = Array(repeating: query, count: 8)
        var candidates = Array(repeating: nan, count: 256)
        candidates[255] = try Vector512Optimized([Float](repeating: 1, count: 512))
        let matrix = MatrixDistance.euclideanSquaredMatrix(queries: queries, candidates: candidates)
        #expect(matrix.count == 8 * 256)
        #expect(matrix[0].isNaN)
        #expect(matrix[255] == 512)
        let inputs = candidates
        try await Operations.$computeProvider.withValue(CPUComputeProvider(mode: .sequential)) {
            let result = try await Operations.findNearestBatch(queries: queries, in: inputs, k: 3)
            #expect(result.count == 8)
            for row in result {
                #expect(row.map(\.index) == [255, 0, 1])
                #expect(row.count == 3)
                #expect(row[0].distance == Float(512).squareRoot())
                #expect(row[1].distance.isNaN)
                #expect(row[2].distance.isNaN)
            }
        }
    }
}
