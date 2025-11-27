import Testing
import Foundation
@testable import VectorCore

@Suite("Integration Protocols")
struct IntegrationProtocolsSuite {

    // MARK: - SearchResult Tests

    @Test
    func testSearchResult_Creation() {
        let result = SearchResult(id: 42, distance: 0.5, score: 0.8)

        #expect(result.id == 42)
        #expect(result.distance == 0.5)
        #expect(result.score == 0.8)
    }

    @Test
    func testSearchResult_ComputedScore() {
        let result = SearchResult(id: 1, distance: 0.0)
        #expect(result.computedScore == 1.0)  // 1/(1+0) = 1

        let result2 = SearchResult(id: 2, distance: 1.0)
        #expect(result2.computedScore == 0.5)  // 1/(1+1) = 0.5
    }

    @Test
    func testSearchResult_Comparable() {
        let closer = SearchResult(id: 1, distance: 0.1)
        let farther = SearchResult(id: 2, distance: 0.9)

        #expect(closer < farther)
        #expect(!(farther < closer))
    }

    @Test
    func testSearchResult_Codable() throws {
        let original = SearchResult(id: 42, distance: 0.5, score: 0.8)

        let encoder = JSONEncoder()
        let data = try encoder.encode(original)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(IntSearchResult.self, from: data)

        #expect(decoded == original)
    }

    // MARK: - SearchResults Tests

    @Test
    func testSearchResults_Creation() {
        let results = SearchResults(
            results: [
                SearchResult(id: 1, distance: 0.1),
                SearchResult(id: 2, distance: 0.2),
            ],
            candidatesSearched: 100,
            searchTimeNanos: 1_000_000,
            isExhaustive: true
        )

        #expect(results.count == 2)
        #expect(!results.isEmpty)
        #expect(results.candidatesSearched == 100)
        #expect(results.isExhaustive)
        #expect(results.searchTimeMs == 1.0)
    }

    @Test
    func testSearchResults_Accessors() {
        let results = SearchResults(
            results: [
                SearchResult(id: 5, distance: 0.1),
                SearchResult(id: 10, distance: 0.2),
            ],
            candidatesSearched: 50
        )

        #expect(results.ids == [5, 10])
        #expect(results.distances == [0.1, 0.2])
        #expect(results.best?.id == 5)
    }

    @Test
    func testSearchResults_Collection() {
        let results = SearchResults(
            results: [
                SearchResult(id: 1, distance: 0.1),
                SearchResult(id: 2, distance: 0.2),
                SearchResult(id: 3, distance: 0.3),
            ],
            candidatesSearched: 10
        )

        var count = 0
        for result in results {
            count += 1
            #expect(result.distance > 0)
        }
        #expect(count == 3)
    }

    @Test
    func testSearchResults_MapIDs() {
        let results = SearchResults(
            results: [SearchResult(id: 1, distance: 0.5)],
            candidatesSearched: 10
        )

        let mapped = results.mapIDs { "id_\($0)" }

        #expect(mapped.results[0].id == "id_1")
        #expect(mapped.candidatesSearched == 10)
    }

    @Test
    func testSearchResults_FromTopKResult() {
        let topK = TopKResult(indices: [1, 2, 3], distances: [0.1, 0.2, 0.3])

        let results = SearchResults(
            from: topK,
            candidatesSearched: 100,
            searchTimeNanos: 500_000,
            isExhaustive: true
        )

        #expect(results.count == 3)
        #expect(results.ids == [1, 2, 3])
        #expect(results.distances == [0.1, 0.2, 0.3])
        #expect(results.candidatesSearched == 100)
    }

    // MARK: - IndexableVector Tests

    @Test
    func testIndexableVector_DefaultImplementation() {
        let v = try! Vector512Optimized(Array(repeating: Float(0.5), count: 512))

        // Default implementation
        #expect(v.isNormalized == false)
        #expect(v.cachedMagnitude == nil)
    }

    @Test
    func testIndexableVector_IsApproximatelyNormalized() {
        let nonNormalized = try! Vector512Optimized(Array(repeating: Float(0.5), count: 512))
        #expect(!nonNormalized.isApproximatelyNormalized)

        let normalized = nonNormalized.normalizedUnchecked()
        #expect(normalized.isApproximatelyNormalized)
    }

    @Test
    func testIndexableVector_DynamicVectorConformance() {
        let v = DynamicVector([1, 2, 3, 4])

        // Should conform to IndexableVector
        let _: any IndexableVector = v
        #expect(v.isNormalized == false)
    }

    @Test
    func testIndexableVector_Vector384Conformance() {
        let v = try! Vector384Optimized(Array(repeating: Float(0.1), count: 384))

        let _: any IndexableVector = v
        #expect(v.scalarCount == 384)
    }

    // MARK: - NormalizationHint Tests

    @Test
    func testNormalizationHint_Explicit() {
        let v = try! Vector512Optimized(Array(repeating: Float(0.5), count: 512))

        let hint = NormalizationHint(vector: v, isNormalized: false)
        #expect(!hint.isNormalized)

        let normalizedHint = NormalizationHint(vector: v.normalizedUnchecked(), isNormalized: true)
        #expect(normalizedHint.isNormalized)
        #expect(normalizedHint.magnitude == 1.0)
    }

    @Test
    func testNormalizationHint_AutoDetection() {
        let nonNormalized = try! Vector512Optimized(Array(repeating: Float(0.5), count: 512))
        let hint1 = NormalizationHint(vector: nonNormalized)
        #expect(!hint1.isNormalized)

        let normalized = nonNormalized.normalizedUnchecked()
        let hint2 = NormalizationHint(vector: normalized)
        #expect(hint2.isNormalized)
    }

    @Test
    func testNormalizationHint_StaticConstructor() {
        let v = try! Vector512Optimized(Array(repeating: Float(0.5), count: 512))
        let normalized = v.normalizedUnchecked()

        let hint = NormalizationHint.normalized(normalized)
        #expect(hint.isNormalized)
    }

    @Test
    func testIndexableVector_WithNormalizationHint() {
        let v = DynamicVector([3, 4])  // magnitude = 5

        let hint = v.withNormalizationHint()
        #expect(!hint.isNormalized)
        #expect(approxEqual(hint.magnitude, 5, tol: 1e-5))
    }

    @Test
    func testIndexableVector_NormalizedWithHint() {
        let v = DynamicVector([3, 4])  // magnitude = 5

        let hint = v.normalizedWithHint()
        #expect(hint != nil)
        #expect(hint!.isNormalized)
        #expect(approxEqual(hint!.vector.magnitude, 1, tol: 1e-5))
    }

    @Test
    func testIndexableVector_NormalizedUncheckedWithHint() {
        let v = DynamicVector([3, 4])

        let hint = v.normalizedUncheckedWithHint()
        #expect(hint.isNormalized)
    }

    // MARK: - SimpleVectorCollection Tests

    @Test
    func testSimpleVectorCollection_AddAndRetrieve() throws {
        let collection = SimpleVectorCollection<DynamicVector>(dimension: 4)

        let v1 = DynamicVector([1, 2, 3, 4])
        let v2 = DynamicVector([5, 6, 7, 8])

        try collection.add(id: 0, vector: v1)
        try collection.add(id: 1, vector: v2)

        #expect(collection.count == 2)
        #expect(collection.vector(for: 0)?.toArray() == [1, 2, 3, 4])
        #expect(collection.vector(for: 1)?.toArray() == [5, 6, 7, 8])
        #expect(collection.vector(for: 2) == nil)
    }

    @Test
    func testSimpleVectorCollection_AutoID() throws {
        let collection = SimpleVectorCollection<DynamicVector>(dimension: 3)

        let id1 = try collection.add(vector: DynamicVector([1, 2, 3]))
        let id2 = try collection.add(vector: DynamicVector([4, 5, 6]))

        #expect(id1 == 0)
        #expect(id2 == 1)
        #expect(collection.count == 2)
    }

    @Test
    func testSimpleVectorCollection_Remove() throws {
        let collection = SimpleVectorCollection<DynamicVector>(dimension: 2)

        try collection.add(id: 0, vector: DynamicVector([1, 2]))
        try collection.add(id: 1, vector: DynamicVector([3, 4]))

        let removed = collection.remove(id: 0)
        #expect(removed?.toArray() == [1, 2])
        #expect(collection.count == 1)
        #expect(collection.vector(for: 0) == nil)
    }

    @Test
    func testSimpleVectorCollection_DimensionMismatch() {
        let collection = SimpleVectorCollection<DynamicVector>(dimension: 4)

        do {
            try collection.add(id: 0, vector: DynamicVector([1, 2, 3]))  // Wrong dimension
            Issue.record("Expected dimension mismatch error")
        } catch let error as VectorError {
            #expect(error.kind == .dimensionMismatch)
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test
    func testSimpleVectorCollection_Search() throws {
        let collection = SimpleVectorCollection<DynamicVector>(dimension: 2)

        try collection.add(id: 0, vector: DynamicVector([0, 0]))
        try collection.add(id: 1, vector: DynamicVector([1, 0]))
        try collection.add(id: 2, vector: DynamicVector([10, 0]))

        let query = DynamicVector([0.5, 0])
        let results = collection.searchEuclidean(query: query, k: 2)

        #expect(results.count == 2)
        #expect(results.ids.contains(0))
        #expect(results.ids.contains(1))
        #expect(!results.ids.contains(2))  // Too far
    }

    @Test
    func testSimpleVectorCollection_AllIDs() throws {
        let collection = SimpleVectorCollection<DynamicVector>(dimension: 2)

        try collection.add(id: 5, vector: DynamicVector([1, 2]))
        try collection.add(id: 10, vector: DynamicVector([3, 4]))

        let ids = collection.allIDs
        #expect(ids.contains(5))
        #expect(ids.contains(10))
        #expect(ids.count == 2)
    }

    @Test
    func testSimpleVectorCollection_RemoveAll() throws {
        let collection = SimpleVectorCollection<DynamicVector>(dimension: 2)

        try collection.add(id: 0, vector: DynamicVector([1, 2]))
        try collection.add(id: 1, vector: DynamicVector([3, 4]))

        collection.removeAll()

        #expect(collection.isEmpty)
        #expect(collection.count == 0)
    }

    @Test
    func testSimpleVectorCollection_BatchVectors() throws {
        let collection = SimpleVectorCollection<DynamicVector>(dimension: 2)

        try collection.add(id: 0, vector: DynamicVector([1, 2]))
        try collection.add(id: 1, vector: DynamicVector([3, 4]))
        try collection.add(id: 2, vector: DynamicVector([5, 6]))

        let batch = collection.vectors(for: [0, 2, 99])  // 99 doesn't exist

        #expect(batch.count == 2)
        #expect(batch[0]?.toArray() == [1, 2])
        #expect(batch[2]?.toArray() == [5, 6])
        #expect(batch[99] == nil)
    }

    // MARK: - VectorCollection Search Tests

    @Test
    func testVectorCollection_SearchCosine() throws {
        let collection = SimpleVectorCollection<DynamicVector>(dimension: 2)

        try collection.add(id: 0, vector: DynamicVector([1, 0]))  // Same direction as query
        try collection.add(id: 1, vector: DynamicVector([0, 1]))  // Orthogonal
        try collection.add(id: 2, vector: DynamicVector([-1, 0])) // Opposite

        let query = DynamicVector([1, 0])
        let results = collection.searchCosine(query: query, k: 2)

        #expect(results.count == 2)
        #expect(results.ids[0] == 0)  // Same direction, distance ≈ 0
    }

    @Test
    func testVectorCollection_EmptySearch() {
        let collection = SimpleVectorCollection<DynamicVector>(dimension: 2)

        let query = DynamicVector([1, 2])
        let results = collection.searchEuclidean(query: query, k: 5)

        #expect(results.isEmpty)
        #expect(results.candidatesSearched == 0)
    }

    // MARK: - TypeAlias Tests

    @Test
    func testTypeAliases() {
        let intResult: IntSearchResult = SearchResult(id: 1, distance: 0.5)
        #expect(intResult.id == 1)

        let stringResult: StringSearchResult = SearchResult(id: "hello", distance: 0.5)
        #expect(stringResult.id == "hello")
    }
}
