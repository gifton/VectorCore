import Foundation
import VectorCore

// Public (non-@testable) Swift 6 consumer of the R05 metric-aware relevance API.

/// Conforms without supplying relevanceScore: the defaulted requirement answers nil.
struct R05DefaultedMetric: DistanceMetric {
    var name: String { "euclidean" }  // spoofed name must not select a built-in mapping
    func distance<V: VectorProtocol>(_ a: V, _ b: V) -> Float where V.Scalar == Float { abs(a[0] - b[0]) }
}

/// Supplies its own interpretation; reached through the requirement witness.
struct R05ScoredMetric: DistanceMetric {
    var name: String { "r05-scored" }
    func distance<V: VectorProtocol>(_ a: V, _ b: V) -> Float where V.Scalar == Float { abs(a[0] - b[0]) }
    func relevanceScore(for distance: Float) -> Float? { 10 - distance }
}

/// A custom collection overriding search owns its stored scores.
struct R05OverrideCollection: VectorCollection {
    typealias Vector = DynamicVector
    var count: Int { 0 }
    var dimension: Int { 1 }
    var allIDs: [Int] { [] }
    func vector(for id: Int) -> DynamicVector? { nil }
    func search(query: DynamicVector, k: Int, metric: any DistanceMetric<Float>) -> SearchResults<Int> {
        SearchResults(results: [SearchResult(id: 5, distance: -1, score: -5)], candidatesSearched: 0)
    }
}

@main struct R05PublicConsumer {
    static func main() throws {
        let collection = SimpleVectorCollection<DynamicVector>(dimension: 1)
        for (id, value): (Int, Float) in [(0, 0.5), (1, 1), (2, 2)] {
            try collection.add(id: id, vector: DynamicVector([value]))
        }
        let query = DynamicVector([1])

        // Historical defect scenario: dot-product distances [-2, -1, -0.5] must score finitely.
        let dot = collection.searchDotProduct(query: query, k: 3)
        precondition(dot.ids == [2, 1, 0] && dot.distances == [-2, -1, -0.5])
        let scores = dot.map { $0.score ?? .nan }
        precondition(scores.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 })
        precondition(scores[0] >= scores[1] && scores[1] >= scores[2])

        // Explicit computedScore(using:): stored value wins; otherwise the witness answers.
        let bare = SearchResult(id: 1, distance: 3)
        precondition(bare.computedScore(using: EuclideanDistance()) == 0.25)
        precondition(bare.computedScore(using: DotProductDistance()) != nil)
        precondition(bare.computedScore(using: R05DefaultedMetric()) == nil)
        precondition(bare.computedScore(using: R05ScoredMetric()) == 7)
        precondition(SearchResult(id: 1, distance: 3, score: -9).computedScore(using: R05ScoredMetric()) == -9)

        // Producer populates scores through the supplied metric's own witness.
        precondition(collection.search(query: query, k: 3, metric: R05DefaultedMetric()).allSatisfy { $0.score == nil })
        let custom = collection.search(query: query, k: 3, metric: R05ScoredMetric())
        precondition(custom.allSatisfy { $0.score == 10 - $0.distance })
        precondition(collection.searchEuclidean(query: query, k: 3).allSatisfy { $0.score != nil })

        // Custom override is authoritative; convenience methods forward it unchanged.
        let override = R05OverrideCollection()
        precondition(override.searchCosine(query: query, k: 1).first?.score == -5)
        print("R05_PUBLIC_COMPLETE defaulted conformance, computedScore(using:), search, custom override")
    }
}
