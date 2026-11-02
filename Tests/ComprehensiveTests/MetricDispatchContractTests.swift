import Testing
@testable import VectorCore

@Suite("Metric dispatch contract")
struct MetricDispatchContractTests {
    private struct ConfiguredReverseMetric: DistanceMetric {
        let scale: Float
        let offset: Float
        var name: String { "configured-reverse" }
        func distance<V: VectorProtocol>(_ a: V, _ b: V) -> Float where V.Scalar == Float {
            offset - scale * ManhattanDistance().distance(a, b)
        }
    }

    private struct Collection: VectorCollection {
        let values: [DynamicVector]
        var count: Int { values.count }
        var dimension: Int { 2 }
        var allIDs: [Int] { values.indices.map { 91 - 7 * $0 } }
        func vector(for id: Int) -> DynamicVector? {
            guard let index = allIDs.firstIndex(of: id) else { return nil }
            return values[index]
        }
    }

    private struct OverrideCollection: VectorCollection {
        typealias Vector = DynamicVector
        let count = 0
        let dimension = 2
        let allIDs: [Int] = []
        func vector(for id: Int) -> DynamicVector? { nil }
        func search(query: DynamicVector, k: Int, metric: any DistanceMetric<Float>) -> SearchResults<Int> {
            SearchResults(results: [SearchResult(id: 1001, distance: metric.distance(query, query) + Float(k))],
                          candidatesSearched: 0)
        }
    }

    private func genericSearch<C: VectorCollection>(
        _ collection: C, query: C.Vector, k: Int, metric: any DistanceMetric<Float>
    ) -> SearchResults<C.ID> {
        collection.search(query: query, k: k, metric: metric)
    }

    @Test func constrainedCollectionOverrideDispatchesThroughProtocolWitness() {
        let result = genericSearch(OverrideCollection(), query: DynamicVector([1, 2]), k: 2,
                                   metric: ConfiguredReverseMetric(scale: 2, offset: 30))
        #expect(result.ids == [1001])
        checkScores(result.distances, [32], context: "collection protocol override")
    }

    private var metrics: [(String, any DistanceMetric<Float>)] {
        [
            ("euclidean", EuclideanDistance()), ("cosine", CosineDistance()),
            ("manhattan", ManhattanDistance()), ("dotProduct", DotProductDistance()),
            ("chebyshev", ChebyshevDistance()), ("hamming", HammingDistance()),
            ("minkowski-p3", MinkowskiDistance(p: 3)),
            ("reverse-2", ConfiguredReverseMetric(scale: 2, offset: 30)),
            ("reverse-5", ConfiguredReverseMetric(scale: 5, offset: 80))
        ]
    }

    private func oracle(_ metric: any DistanceMetric<Float>, query: DynamicVector,
                        candidates: [DynamicVector], k: Int) -> [(Int, Float)] {
        let scores: [(Int, Float)] = candidates.enumerated().map { entry in
            (entry.offset, metric.distance(query, entry.element))
        }
        let sorted = scores.sorted { lhs, rhs in
            if lhs.1 == rhs.1 { return lhs.0 < rhs.0 }
            return lhs.1 < rhs.1
        }
        return Array(sorted.prefix(max(0, k)))
    }

    private func checkScores(_ actual: [Float], _ expected: [Float], context: String) {
        #expect(actual.count == expected.count, "\(context)")
        for (value, reference) in zip(actual, expected) {
            #expect(value.isFinite && reference.isFinite, "\(context)")
            #expect(abs(value - reference) <= 1e-6 + 1e-6 * abs(reference), "\(context)")
        }
    }

    @Test func suppliedWitnessAcrossFacades() {
        let queries = [DynamicVector([1, 2]), DynamicVector([-2, 1])]
        let populated = [[3, 3], [4, 0], [1, -2], [-3, 1], [3, 3]].map { DynamicVector($0) }
        for (name, metric) in metrics {
            for candidates in [populated, []] {
                let collection = Collection(values: candidates)
                for k in [0, 1, 3, 9] {
                    let batch = TopKSelection.batchNearest(k: k, queries: queries, candidates: candidates, metric: metric)
                    #expect(batch.count == queries.count)
                    for (queryIndex, query) in queries.enumerated() {
                        let expected = oracle(metric, query: query, candidates: candidates, k: k)
                        let context = "\(name), k=\(k), n=\(candidates.count), query=\(queryIndex)"
                        let nearest = TopKSelection.nearest(k: k, query: query, candidates: candidates, metric: metric)
                        #expect(nearest.indices == expected.map { $0.0 }, "\(context)")
                        checkScores(nearest.distances, expected.map { $0.1 }, context: context)
                        guard batch.indices.contains(queryIndex) else { continue }
                        #expect(batch[queryIndex].indices == expected.map { $0.0 }, "\(context)")
                        checkScores(batch[queryIndex].distances, expected.map { $0.1 }, context: context)
                        let found = collection.search(query: query, k: k, metric: metric)
                        #expect(found.ids == expected.map { collection.allIDs[$0.0] }, "\(context)")
                        checkScores(found.distances, expected.map { $0.1 }, context: context)
                    }
                }
            }
        }
    }

    @Test func literalChebyshevAndConfiguredReverseResults() {
        let query = DynamicVector([0, 0])
        let candidates = [DynamicVector([3, 3]), DynamicVector([4, 0])]
        let chebyshev = TopKSelection.nearest(k: 1, query: query, candidates: candidates, metric: ChebyshevDistance())
        #expect(chebyshev.indices == [0])
        checkScores(chebyshev.distances, [3], context: "literal Chebyshev")
        for (scale, offset, expected): (Float, Float, Float) in [(2, 30, 18), (5, 80, 50)] {
            let result = TopKSelection.nearest(k: 1, query: query, candidates: candidates,
                                               metric: ConfiguredReverseMetric(scale: scale, offset: offset))
            #expect(result.indices == [0])
            checkScores(result.distances, [expected], context: "literal reverse \(scale)")
        }
    }

    @Test func factoryNamesPreserveWitnessCaseInsensitively() throws {
        let query = DynamicVector([1, 2])
        let candidates = [[3, 3], [4, 0], [-3, 1]].map { DynamicVector($0) }
        for type in OptimizedDistanceMetrics.MetricType.allCases {
            let direct: any DistanceMetric<Float> = type.createMetric()
            let alternating = type.rawValue.enumerated().map {
                $0.offset.isMultiple(of: 2) ? String($0.element).uppercased() : String($0.element).lowercased()
            }.joined()
            for name in [type.rawValue, type.rawValue.uppercased(), alternating] {
                let metric = try #require(OptimizedDistanceMetrics.create(named: name))
                checkScores(candidates.map { metric.distance(query, $0) },
                            candidates.map { direct.distance(query, $0) }, context: name)
            }
        }
        #expect(OptimizedDistanceMetrics.create(named: "unknown-metric") == nil)
    }
}
