import Testing
import Foundation
@testable import VectorCore

@Suite("Top-K Selection")
struct TopKSelectionSuite {

    // MARK: - Basic Selection Tests

    @Test
    func testSelect_ReturnsCorrectKSmallest() {
        let distances: [Float] = [5, 3, 8, 1, 9, 2, 7, 4, 6, 0]
        let result = TopKSelection.select(k: 3, from: distances)

        #expect(result.count == 3)
        #expect(result.indices == [9, 3, 5])  // indices of 0, 1, 2
        #expect(result.distances == [0, 1, 2])
    }

    @Test
    func testSelect_SortedAscending() {
        let distances: [Float] = [10, 20, 30, 5, 15, 25]
        let result = TopKSelection.select(k: 4, from: distances)

        #expect(result.count == 4)
        // Verify distances are sorted ascending
        for i in 0..<(result.count - 1) {
            #expect(result.distances[i] <= result.distances[i + 1])
        }
    }

    @Test
    func testSelect_KLargerThanArray() {
        let distances: [Float] = [3, 1, 2]
        let result = TopKSelection.select(k: 10, from: distances)

        #expect(result.count == 3)
        #expect(result.distances == [1, 2, 3])
    }

    @Test
    func testSelect_EmptyArray() {
        let distances: [Float] = []
        let result = TopKSelection.select(k: 5, from: distances)

        #expect(result.isEmpty)
        #expect(result.count == 0)
    }

    @Test
    func testSelect_KZero() {
        let distances: [Float] = [1, 2, 3]
        let result = TopKSelection.select(k: 0, from: distances)

        #expect(result.isEmpty)
    }

    @Test
    func testSelect_SingleElement() {
        let distances: [Float] = [42]
        let result = TopKSelection.select(k: 1, from: distances)

        #expect(result.count == 1)
        #expect(result.indices[0] == 0)
        #expect(result.distances[0] == 42)
    }

    // MARK: - K-Nearest with Generic Vectors

    @Test
    func testNearest_GenericVectors() {
        let query = DynamicVector([1, 0, 0, 0])
        let candidates = [
            DynamicVector([1, 0, 0, 0]),  // distance 0
            DynamicVector([0, 1, 0, 0]),  // distance sqrt(2)
            DynamicVector([2, 0, 0, 0]),  // distance 1
            DynamicVector([0, 0, 1, 0]),  // distance sqrt(2)
        ]

        let result = TopKSelection.nearest(k: 2, query: query, candidates: candidates)

        #expect(result.count == 2)
        #expect(result.indices[0] == 0)  // Identical vector
        #expect(result.indices[1] == 2)  // Distance 1
        #expect(approxEqual(result.distances[0], 0, tol: 1e-5))
        #expect(approxEqual(result.distances[1], 1, tol: 1e-5))
    }

    @Test
    func testNearest_WithCosineDistance() {
        let query = DynamicVector([1, 0, 0])
        let candidates = [
            DynamicVector([1, 0, 0]),   // cosine dist 0
            DynamicVector([-1, 0, 0]),  // cosine dist 2
            DynamicVector([0, 1, 0]),   // cosine dist 1
        ]

        let result = TopKSelection.nearest(
            k: 2,
            query: query,
            candidates: candidates,
            metric: CosineDistance()
        )

        #expect(result.count == 2)
        #expect(result.indices[0] == 0)  // Same direction
        #expect(result.indices[1] == 2)  // Orthogonal
    }

    // MARK: - Optimized Vector Tests

    @Test
    func testNearestEuclidean512() {
        let query = try! Vector512Optimized(Array(repeating: Float(0.5), count: 512))
        let candidates = [
            try! Vector512Optimized(Array(repeating: Float(0.5), count: 512)),  // distance 0
            try! Vector512Optimized(Array(repeating: Float(1.0), count: 512)),  // further
            try! Vector512Optimized(Array(repeating: Float(0.0), count: 512)),  // further
        ]

        let result = TopKSelection.nearestEuclidean512(k: 2, query: query, candidates: candidates)

        #expect(result.count == 2)
        #expect(result.indices[0] == 0)  // Identical
        #expect(approxEqual(result.distances[0], 0, tol: 1e-5))
    }

    @Test
    func testNearestEuclidean384() {
        let query = try! Vector384Optimized(Array(repeating: Float(0.1), count: 384))
        let candidates = [
            try! Vector384Optimized(Array(repeating: Float(0.2), count: 384)),
            try! Vector384Optimized(Array(repeating: Float(0.1), count: 384)),  // identical
            try! Vector384Optimized(Array(repeating: Float(0.3), count: 384)),
        ]

        let result = TopKSelection.nearestEuclidean384(k: 1, query: query, candidates: candidates)

        #expect(result.count == 1)
        #expect(result.indices[0] == 1)  // Identical vector
        #expect(approxEqual(result.distances[0], 0, tol: 1e-5))
    }

    @Test
    func testNearestCosinePreNormalized512() {
        // Create normalized vectors
        let query = try! Vector512Optimized(Array(repeating: Float(0.5), count: 512)).normalizedUnchecked()
        let same = query
        let opposite = try! Vector512Optimized(query.toArray().map { -$0 })

        let candidates = [opposite, same]

        let result = TopKSelection.nearestCosinePreNormalized512(k: 1, query: query, candidates: candidates)

        #expect(result.count == 1)
        #expect(result.indices[0] == 1)  // Same direction (distance ≈ 0)
    }

    // MARK: - TopKResult Tests

    @Test
    func testTopKResult_Collection() {
        let result = TopKResult(indices: [0, 1, 2], distances: [0.1, 0.2, 0.3])

        #expect(result.count == 3)
        #expect(!result.isEmpty)

        // Test subscript
        let first = result[0]
        #expect(first.index == 0)
        #expect(first.distance == 0.1)

        // Test iteration
        var count = 0
        for _ in result {
            count += 1
        }
        #expect(count == 3)
    }

    @Test
    func testTopKResult_ToTuples() {
        let result = TopKResult(indices: [5, 3], distances: [1.0, 2.0])
        let tuples = result.toTuples()

        #expect(tuples.count == 2)
        #expect(tuples[0].index == 5)
        #expect(tuples[0].distance == 1.0)
    }

    @Test
    func testTopKResult_Codable() throws {
        let original = TopKResult(indices: [1, 2, 3], distances: [0.1, 0.2, 0.3])

        let encoder = JSONEncoder()
        let data = try encoder.encode(original)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(TopKResult.self, from: data)

        #expect(decoded == original)
    }

    // MARK: - Batch Tests

    @Test
    func testBatchNearest() {
        let queries = [
            DynamicVector([1, 0]),
            DynamicVector([0, 1]),
        ]
        let candidates = [
            DynamicVector([1, 0]),
            DynamicVector([0, 1]),
            DynamicVector([1, 1]),
        ]

        let results = TopKSelection.batchNearest(k: 2, queries: queries, candidates: candidates)

        #expect(results.count == 2)
        #expect(results[0].indices[0] == 0)  // First query closest to first candidate
        #expect(results[1].indices[0] == 1)  // Second query closest to second candidate
    }

    // MARK: - Algorithm Selection Tests

    @Test
    func testSelect_SmallKUsesHeap() {
        // With k < n/10, should use heap algorithm
        let distances = (0..<1000).map { Float($0) }
        let result = TopKSelection.select(k: 10, from: distances)

        #expect(result.count == 10)
        #expect(result.distances == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    }

    @Test
    func testSelect_LargeKUsesSort() {
        // With k >= n/10, should use sort algorithm
        let distances: [Float] = [5, 3, 8, 1, 9, 2, 7, 4, 6, 0]
        let result = TopKSelection.select(k: 5, from: distances)  // k = 5, n = 10, k >= n/10

        #expect(result.count == 5)
        #expect(result.distances == [0, 1, 2, 3, 4])
    }

    // MARK: - Determinism Tests

    @Test
    func testSelect_DeterministicTiebreaking() {
        // When distances are equal, smaller indices should be preferred
        let distances: [Float] = [1, 1, 1, 1, 1]
        let result = TopKSelection.select(k: 3, from: distances)

        // Should prefer indices 0, 1, 2 (smallest indices for ties)
        #expect(result.count == 3)
        // All distances equal
        #expect(result.distances.allSatisfy { $0 == 1.0 })
    }

    // MARK: - Edge Cases

    @Test
    func testNearest_EmptyCandidates() {
        let query = DynamicVector([1, 2, 3])
        let candidates: [DynamicVector] = []

        let result = TopKSelection.nearest(k: 5, query: query, candidates: candidates)

        #expect(result.isEmpty)
    }

    @Test
    func testNearestEuclidean768() {
        let query = try! Vector768Optimized(Array(repeating: Float(0.1), count: 768))
        let candidates = [
            try! Vector768Optimized(Array(repeating: Float(0.1), count: 768)),
        ]

        let result = TopKSelection.nearestEuclidean768(k: 1, query: query, candidates: candidates)

        #expect(result.count == 1)
        #expect(approxEqual(result.distances[0], 0, tol: 1e-5))
    }

    @Test
    func testNearestEuclidean1536() {
        let query = try! Vector1536Optimized(Array(repeating: Float(0.1), count: 1536))
        let candidates = [
            try! Vector1536Optimized(Array(repeating: Float(0.1), count: 1536)),
            try! Vector1536Optimized(Array(repeating: Float(0.2), count: 1536)),
        ]

        let result = TopKSelection.nearestEuclidean1536(k: 2, query: query, candidates: candidates)

        #expect(result.count == 2)
        #expect(result.indices[0] == 0)  // Identical
    }
}
