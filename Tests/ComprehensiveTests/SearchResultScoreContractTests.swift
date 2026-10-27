import Foundation
import Testing
@testable import VectorCore

@Suite struct SearchResultScoreContractTests {
    private func closeScore(_ actual: Float?, _ expected: Float) throws {
        let value = try #require(actual)
        #expect(value.isFinite && expected.isFinite)
        // Measure the ULP below a power-of-two expected value, the smaller binade.
        let unit = min(expected.ulp, expected.nextDown.ulp)
        #expect(value.isFinite && abs(value - expected) <= 4 * unit)
    }

    @Test func dotProductSearchRelevanceRegression() throws {
        let collection = SimpleVectorCollection<DynamicVector>(dimension: 1)
        for (id, value): (Int, Float) in [(0, 0.5), (1, 1), (2, 2)] {
            try collection.add(id: id, vector: DynamicVector([value]))
        }
        let results = collection.searchDotProduct(query: DynamicVector([1]), k: 3)
        #expect(results.ids == [2, 1, 0])
        #expect(results.distances == [-2, -1, -0.5])
        let scores = try results.map { try #require($0.score) }
        print("R05_DOT_GREEN ids=\(results.ids) distances=\(results.distances) scores=\(scores)")
        for (distance, score) in zip(results.distances, scores) {
            try closeScore(score, Float(1 / (1 + Foundation.exp(Double(distance)))))
            #expect(score.isFinite && score >= 0 && score <= 1)
        }
        #expect(scores[0] >= scores[1] && scores[1] >= scores[2])
    }

    @Test func inverseBuiltinDomainsAndMonotonicity() throws {
        let metrics: [any DistanceMetric<Float>] = [EuclideanDistance(), CosineDistance(), ManhattanDistance(),
            ChebyshevDistance(), HammingDistance(threshold: 0.75), MinkowskiDistance(p: 3)]
        let values: [Float] = [-0.0, 0, .leastNonzeroMagnitude, .leastNormalMagnitude, 0.125, 0.5, 1, 2, 3, 1024,
                               .greatestFiniteMagnitude]
        for metric in metrics {
            var previous: Float = 1
            for distance in values {
                let score = try #require(metric.relevanceScore(for: distance))
                #expect(score.isFinite && score > 0 && score <= 1 && score <= previous)
                if distance == .greatestFiniteMagnitude {
                    // Offline literal: round-to-nearest binary32 of 1/(1+maxFinite) is 2^-128.
                    #expect(score.bitPattern == Float(bitPattern: 0x0020_0000).bitPattern)
                } else if distance.magnitude <= .leastNormalMagnitude {
                    #expect(score.bitPattern == Float(1).bitPattern)
                } else {
                    try closeScore(score, Float(1 / (1 + Double(distance))))
                }
                previous = score
            }
            #expect(metric.relevanceScore(for: 0) == 1)
            #expect(metric.relevanceScore(for: 1) == 0.5)
            #expect(metric.relevanceScore(for: 3) == 0.25)
            for value: Float in [-.leastNonzeroMagnitude, -1, -.greatestFiniteMagnitude, .nan, .infinity, -.infinity] {
                #expect(metric.relevanceScore(for: value) == nil)
            }
        }
    }

    @Test func dotLogisticDomain() throws {
        let metric = DotProductDistance()
        let values: [Float] = [-100, -20, -2, Float(-1).nextDown, -1, Float(-1).nextUp, -0.5,
            -.leastNormalMagnitude, -.leastNonzeroMagnitude, -0.0, 0, .leastNonzeroMagnitude,
            .leastNormalMagnitude, 0.5, 1, 2, 20, 80, 100]
        var previous: Float = 1
        for distance in values {
            // Independent direct modest-domain formula; no piecewise production branch.
            let reference = Float(1 / (1 + Foundation.exp(Double(distance))))
            let score = try #require(metric.relevanceScore(for: distance))
            #expect(score.isFinite && score >= 0 && score <= 1 && score <= previous)
            try closeScore(score, reference)
            previous = score
        }
        // exp(-1000) is below half a Float subnormal ULP; the rounded limits are exact.
        for distance: Float in [1000, .greatestFiniteMagnitude] {
            #expect(metric.relevanceScore(for: distance)?.bitPattern == Float(0).bitPattern)
            #expect(metric.relevanceScore(for: -distance)?.bitPattern == Float(1).bitPattern)
        }
        #expect(metric.relevanceScore(for: -0.0) == 0.5)
        #expect(metric.relevanceScore(for: 0) == 0.5)
        for value: Float in [.nan, .infinity, -.infinity] { #expect(metric.relevanceScore(for: value) == nil) }
    }

    @Test func customWitnessDispatchAndUnknownContext() {
        let calls = R05ScoreCalls()
        let metric = R05ConfiguredMetric(scale: 2, offset: 17, calls: calls)
        let existential: any DistanceMetric<Float> = metric
        #expect(metric.relevanceScore(for: 4) == 21)
        #expect(r05GenericScore(metric, 4) == 21)
        #expect(existential.relevanceScore(for: 4) == 21)
        #expect(calls.recorded == [4, 4, 4])
        let unknown = R05DefaultMetric()
        let erased: any DistanceMetric<Float> = unknown
        #expect(unknown.relevanceScore(for: 4) == nil)
        #expect(r05GenericScore(unknown, 4) == nil)
        #expect(erased.relevanceScore(for: 4) == nil)
        let result = SearchResult(id: 8, distance: 4)
        #expect(result.computedScore(using: metric) == 21)
        #expect(calls.recorded == [4, 4, 4, 4])
        #expect(result.computedScore(using: erased) == nil)
    }

    @Test func storedScoreBitsAndMappingNeverInvokeWitness() throws {
        let calls = R05ScoreCalls()
        let metric = R05ConfiguredMetric(scale: 2, offset: 17, calls: calls)
        let words: [UInt32] = [0, 0x80000000, 0xbf800000, 0x40000000, 0x7fc01234, 0xffc05678, 0x7f800000, 0xff800000]
        let original = words.enumerated().map {
            SearchResult(id: $0.offset, distance: Float.nan, score: Float(bitPattern: $0.element))
        }
        let mapped = SearchResults(results: original, candidatesSearched: words.count).mapIDs { "id-\($0)" }
        for index in words.indices {
            #expect(original[index].computedScore(using: metric)?.bitPattern == words[index])
            #expect(mapped[index].score?.bitPattern == words[index])
            #expect(mapped[index].computedScore(using: metric)?.bitPattern == words[index])
            #expect(mapped[index].id == "id-\(index)")
        }
        #expect(calls.recorded.isEmpty)
    }

    @Test func codablePreservesExplicitValuesAndMissingContext() throws {
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()
        let original = SearchResult(id: 9, distance: Float(2), score: Float(-3))
        let data = try encoder.encode(original)
        let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(Set(object.keys) == Set(["id", "distance", "score"]))
        let decoded = try decoder.decode(SearchResult<Int>.self, from: data)
        #expect(decoded.id == 9 && decoded.distance.bitPattern == Float(2).bitPattern)
        #expect(decoded.score?.bitPattern == Float(-3).bitPattern)
        for text in [#"{"id":1,"distance":2}"#, #"{"id":1,"distance":2,"score":null}"#] {
            let missing = try decoder.decode(SearchResult<Int>.self, from: Data(text.utf8))
            #expect(missing.score == nil && missing.computedScore(using: R05DefaultMetric()) == nil)
        }
        let noScore = try encoder.encode(SearchResult(id: 1, distance: Float(2)))
        let absent = try #require(JSONSerialization.jsonObject(with: noScore) as? [String: Any])
        #expect(absent["score"] == nil)
        encoder.nonConformingFloatEncodingStrategy = .convertToString(positiveInfinity: "Inf", negativeInfinity: "-Inf", nan: "NaN")
        decoder.nonConformingFloatDecodingStrategy = .convertFromString(positiveInfinity: "Inf", negativeInfinity: "-Inf", nan: "NaN")
        for value: Float in [.nan, .infinity, -.infinity] {
            let wire = try encoder.encode(SearchResult(id: 2, distance: Float(1), score: value))
            let score = try #require(decoder.decode(SearchResult<Int>.self, from: wire).score)
            if value.isNaN { #expect(score.isNaN) } else { #expect(score.bitPattern == value.bitPattern) }
        }
        // JSON's configured NaN token preserves classification, not a NaN payload.
    }

    @Test func topKConversionRequiresExplicitMetricContext() throws {
        let topK = TopKResult(indices: [9, 2], distances: [-1, 0.5])
        let result = SearchResults(from: topK, candidatesSearched: 3)
        #expect(result.ids == [9, 2] && result.distances == [-1, 0.5])
        #expect(result.allSatisfy { $0.score == nil })
        #expect(result.mapIDs(String.init).allSatisfy { $0.score == nil })
        for entry in result {
            #expect(entry.computedScore(using: R05DefaultMetric()) == nil)
            try closeScore(entry.computedScore(using: DotProductDistance()),
                           Float(1 / (1 + Foundation.exp(Double(entry.distance)))))
        }
    }

    @Test func selectedCallbacksAndDefaultNilAcrossProducers() throws {
        let values = [1, 2, 3].map { DynamicVector([Float($0)]) }
        let simple = SimpleVectorCollection<DynamicVector>(dimension: 1)
        for index in values.indices { try simple.add(id: index, vector: values[index]) }
        try checkCallbacks(R05Collection(values: values))
        try checkCallbacks(simple)
        try checkCallbacks(R05Collection(values: []))
        try checkCallbacks(SimpleVectorCollection<DynamicVector>(dimension: 1))
    }

    private func checkCallbacks<C: VectorCollection>(_ collection: C) throws
    where C.Vector == DynamicVector, C.ID == Int {
        for k in [-1, 0, 2, 9] {
            let calls = R05ScoreCalls()
            let metric = R05ConfiguredMetric(scale: -1, offset: 100, calls: calls)
            let result = collection.search(query: DynamicVector([0]), k: k, metric: metric)
            let expectedIDs = collection.count == 0 || k <= 0 ? [] : Array([2, 1, 0].prefix(k))
            #expect(result.ids == expectedIDs)
            #expect(calls.recorded == result.distances)
            for entry in result {
                #expect(entry.distance == -Float(entry.id + 1))
                #expect(entry.score == 100 + entry.distance)
            }
        }
        let unknown = collection.search(query: DynamicVector([0]), k: 9, metric: R05DefaultMetric())
        #expect(unknown.allSatisfy { $0.score == nil })
    }

    @Test func allBuiltinProducerGeometriesAndConfiguredMinkowski() throws {
        let values = [[Float](arrayLiteral: 1, 0), [0, 1], [2, 2]].map { DynamicVector($0) }
        let simple = SimpleVectorCollection<DynamicVector>(dimension: 2)
        for index in values.indices { try simple.add(id: index, vector: values[index]) }
        let cases: [(any DistanceMetric<Float>, [Double], Bool)] = [
            (EuclideanDistance(), [0, sqrt(2), sqrt(5)], false),
            (CosineDistance(), [0, 1, 1 - 1 / sqrt(2)], false),
            (ManhattanDistance(), [0, 2, 3], false), (ChebyshevDistance(), [0, 1, 2], false),
            (HammingDistance(), [0, 2, 1], false),
            (MinkowskiDistance(p: 3), [0, cbrt(2), cbrt(9)], false),
            (DotProductDistance(), [-1, 0, -2], true)
        ]
        for (metric, distances, logistic) in cases {
            try checkGeometry(R05Collection(values: values), metric: metric, distances: distances, logistic: logistic)
            try checkGeometry(simple, metric: metric, distances: distances, logistic: logistic)
        }
    }

    private func checkGeometry<C: VectorCollection>(
        _ collection: C, metric: any DistanceMetric<Float>, distances: [Double], logistic: Bool
    ) throws
    where C.Vector == DynamicVector, C.ID == Int {
        let result = collection.search(query: DynamicVector([1, 0]), k: 3, metric: metric)
        let ids = distances.indices.sorted { distances[$0] < distances[$1] }
        #expect(result.ids == ids)
        for entry in result {
            let expected = distances[entry.id]
            #expect(entry.distance.isFinite && abs(Double(entry.distance) - expected) < 1e-5)
            let score = logistic ? 1 / (1 + exp(expected)) : 1 / (1 + expected)
            try closeScore(entry.score, Float(score))
        }
    }

    @Test func collectionOverrideScoresRemainAuthoritative() {
        let collection = R05OverrideCollection()
        let query = DynamicVector([1])
        let results = [collection.searchEuclidean(query: query, k: 2),
                       collection.searchCosine(query: query, k: 2), collection.searchDotProduct(query: query, k: 2)]
        for result in results {
            #expect(result.ids == [77])
            #expect(result.first?.score == -77)
        }
    }

    @Test func reentrantScoreChildProbe() throws {
        guard ProcessInfo.processInfo.environment["R05_SCORE_CHILD"] == "1" else { return }
        let collection = SimpleVectorCollection<DynamicVector>(dimension: 1)
        for (id, value): (Int, Float) in [(0, 3), (1, 1), (2, 2)] {
            try collection.add(id: id, vector: DynamicVector([value]))
        }
        let calls = R05ScoreCalls()
        let result = collection.search(query: DynamicVector([0]), k: 2,
                                       metric: R05ReentrantMetric(collection: collection, calls: calls))
        #expect(result.ids == [1, 2] && result.distances == [1, 2])
        #expect(result.map(\.score) == [101, 102])
        #expect(result.candidatesSearched == 3 && calls.recorded == [1, 2])
        #expect(collection.allIDs == [99])
        print("R05_REENTRANT_SCORE_COMPLETE")
    }

    @Test func reentrantScoreRunsOutsideLockAndKeepsSelectedSnapshot() throws {
        let diagnostic = try r05RunScoreChild()
        #expect(diagnostic.contains("R05_REENTRANT_SCORE_COMPLETE"))
    }
}

private func r05GenericScore<M: DistanceMetric>(_ metric: M, _ distance: Float) -> Float? where M.Scalar == Float {
    metric.relevanceScore(for: distance)
}

private final class R05ScoreCalls: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [Float] = []
    func record(_ value: Float) {
        lock.lock()
        defer { lock.unlock() }
        values.append(value)
    }
    var recorded: [Float] {
        lock.lock()
        defer { lock.unlock() }
        return values
    }
}

private struct R05ConfiguredMetric: DistanceMetric {
    let scale: Float
    let offset: Float
    let calls: R05ScoreCalls
    var name: String { "euclidean" }
    var identifier: String { "dotproduct" }
    func distance<V: VectorProtocol>(_ a: V, _ b: V) -> Float where V.Scalar == Float { scale * b[0] }
    func relevanceScore(for distance: Float) -> Float? {
        calls.record(distance)
        return offset + distance
    }
}

private struct R05DefaultMetric: DistanceMetric {
    var name: String { "cosine" }
    func distance<V: VectorProtocol>(_ a: V, _ b: V) -> Float where V.Scalar == Float { b[0] }
}

private struct R05Collection: VectorCollection {
    let values: [DynamicVector]
    var count: Int { values.count }
    var dimension: Int { values.first?.scalarCount ?? 1 }
    var allIDs: [Int] { Array(values.indices) }
    func vector(for id: Int) -> DynamicVector? { values.indices.contains(id) ? values[id] : nil }
}

private struct R05OverrideCollection: VectorCollection {
    typealias Vector = DynamicVector
    let count = 0
    let dimension = 1
    let allIDs: [Int] = []
    func vector(for id: Int) -> DynamicVector? { nil }
    func search(query: DynamicVector, k: Int, metric: any DistanceMetric<Float>) -> SearchResults<Int> {
        SearchResults(results: [SearchResult(id: 77, distance: -1, score: -77)], candidatesSearched: 0)
    }
}

private struct R05ReentrantMetric: DistanceMetric {
    let collection: SimpleVectorCollection<DynamicVector>
    let calls: R05ScoreCalls
    var name: String { "reentrant-score" }
    func distance<V: VectorProtocol>(_ a: V, _ b: V) -> Float where V.Scalar == Float { b[0] }
    func relevanceScore(for distance: Float) -> Float? {
        calls.record(distance)
        _ = collection.count
        _ = collection.vector(for: 1)
        _ = collection.vectors(for: [0, 1, 2])
        collection.removeAll()
        try? collection.add(id: 99, vector: DynamicVector([-100]))
        return 100 + distance
    }
}

private enum R05ChildError: Error { case timeout(String), failed(String) }

private func r05RunScoreChild() throws -> String {
    let base = FileManager.default.temporaryDirectory.appendingPathComponent("R05-children")
    let directory = base.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let capture = directory.appendingPathComponent("score.log")
    _ = FileManager.default.createFile(atPath: capture.path, contents: nil)
    let output = try FileHandle(forWritingTo: capture)
    defer { try? output.close() }
    let childFilter = "SearchResultScoreContractTests/reentrantScoreChildProbe"
    var environment = ProcessInfo.processInfo.environment
    environment["R05_SCORE_CHILD"] = "1"
    var record: [String: Any] = ["executable": CommandLine.arguments[0],
                               "capture": capture.path, "deadlineSeconds": 15, "cleanupSeconds": 2]
    defer {
        if let data = try? JSONSerialization.data(withJSONObject: record, options: [.prettyPrinted, .sortedKeys]) {
            try? data.write(to: directory.appendingPathComponent("command.json"))
        }
    }
    let process = try launchChildProbe(.swiftTesting(filter: childFilter), environment: environment, output: output,
                                       deadline: 15, cleanup: .kill(seconds: 2))
    record["arguments"] = process.arguments
    record["pid"] = process.processIdentifier
    print("R05_CHILD_CAPTURE \(capture.path)")
    if process.timedOut {
        record["timeout"] = true
        record["cleanupTimeout"] = !process.cleaned
        throw R05ChildError.timeout(capture.path)
    }
    record["status"] = process.terminationStatus
    record["reason"] = process.terminationReason?.rawValue
    let diagnostic = try String(contentsOf: capture, encoding: .utf8)
    guard process.terminationStatus == 0 else { throw R05ChildError.failed(diagnostic) }
    return diagnostic
}
