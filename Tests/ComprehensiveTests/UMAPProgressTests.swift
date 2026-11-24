import Foundation
import Testing
@testable import VectorCore

enum UMAPProgressFixtures {
    enum Route: CaseIterable, Sendable {
        case builtRandom, builtPCA, suppliedRandom, suppliedPCA, graphRandom, graphProvided

        var initialization: UMAPInitialization {
            self == .builtPCA || self == .suppliedPCA ? .pca : .random
        }

        var stages: [UMAPProgress.Stage] {
            switch self {
            case .builtRandom, .builtPCA: [.graph, .initialization, .fuzzySet, .layout]
            case .suppliedRandom, .suppliedPCA, .graphRandom: [.initialization, .fuzzySet, .layout]
            case .graphProvided: [.fuzzySet, .layout]
            }
        }
    }

    static func vectors(count: Int = 8) -> [DynamicVector] {
        var state: UInt32 = 12345
        return (0..<count).map { _ in
            DynamicVector((0..<6).map { _ -> Float in
                state = state &* 1664525 &+ 1013904223
                return Float(Int((state >> 8) % 2001) - 1000) / 257
            })
        }
    }

    static func config(initialization: UMAPInitialization = .random, epochs: Int? = 7) -> UMAPConfig {
        UMAPConfig(neighbors: 3, epochs: epochs, negativeSampleRate: 2, initialization: initialization, seed: 19)
    }

    static func initial(count: Int = 8, dimensions: Int) -> [[Float]] {
        (0..<count).map { i in (0..<dimensions).map { c in Float(i * 7 - c * 3) / 11 } }
    }

    static func run(_ route: Route, dimensions: Int, metric: KNNGraph.Metric? = nil, progress: ((UMAPProgress) -> Void)? = nil) throws -> UMAPResult {
        let data = vectors()
        var config = config(initialization: route.initialization)
        if let metric { config.metric = metric }
        switch route {
        case .builtRandom, .builtPCA:
            return try Operations.umap(data, dimensions: dimensions, config: config, progress: progress)
        case .suppliedRandom, .suppliedPCA:
            let graph = try KNNGraph.bruteForce(data, neighbors: 3)
            return try Operations.umap(data, dimensions: dimensions, graph: graph, config: config, progress: progress)
        case .graphRandom:
            let graph = try KNNGraph.bruteForce(data, neighbors: 3)
            return try Operations.umap(graph: graph, dimensions: dimensions, config: config, progress: progress)
        case .graphProvided:
            let graph = try KNNGraph.bruteForce(data, neighbors: 3)
            return try Operations.umap(graph: graph, dimensions: dimensions,
                                       initialCoordinates: initial(dimensions: dimensions), config: config, progress: progress)
        }
    }

    static func ring(count: Int) throws -> KNNGraph {
        try KNNGraph(pointCount: count, rowOffsets: Array(0...count),
                     neighborIndices: (0..<count).map { Int32(($0 + 1) % count) },
                     distances: Array(repeating: 1, count: count))
    }
}

@Suite
struct UMAPProgressTests {
    @Test(arguments: UMAPProgressFixtures.Route.allCases, [2, 5])
    func handlerPreservesCoordinates(route: UMAPProgressFixtures.Route, dimensions: Int) throws {
        let plain = try UMAPProgressFixtures.run(route, dimensions: dimensions)
        var events: [UMAPProgress] = []
        let observed = try UMAPProgressFixtures.run(route, dimensions: dimensions) { events.append($0) }
        #expect(observed.coordinates.map(\.bitPattern) == plain.coordinates.map(\.bitPattern))
        #expect(!events.isEmpty)
    }

    @Test(arguments: UMAPProgressFixtures.Route.allCases)
    func stageOrderAndSkippedStages(route: UMAPProgressFixtures.Route) throws {
        var actual: [UMAPProgress] = []
        _ = try UMAPProgressFixtures.run(route, dimensions: 2) { actual.append($0) }
        let expected = route.stages.flatMap { stage -> [UMAPProgress] in
            let counts: [Int]
            let total: Int
            switch stage {
            case .graph: counts = [0, 8]; total = 8
            case .initialization, .fuzzySet: counts = [0, 1]; total = 1
            case .layout: counts = Array(0...7); total = 7
            }
            return counts.map { UMAPProgress(stage: stage, completed: $0, total: total) }
        }
        #expect(actual == expected)
    }

    @Test(arguments: [255, 256, 257, 513])
    func graphReportsBlockBoundaries(count: Int) throws {
        let expected: [Int]
        switch count {
        case 255: expected = [0, 255]
        case 256: expected = [0, 256]
        case 257: expected = [0, 256, 257]
        default: expected = [0, 256, 512, 513]
        }
        var events: [UMAPProgress] = []
        _ = try Operations.umap(UMAPProgressFixtures.vectors(count: count),
                                 config: UMAPProgressFixtures.config(epochs: 1)) { events.append($0) }
        let graph = events.filter { $0.stage == .graph }
        #expect(graph.map(\.completed) == expected)
        #expect(graph.allSatisfy { $0.total == count })
    }

    @Test(arguments: [1, 3, 0, -1])
    func layoutReportsEveryEpoch(selection: Int) throws {
        let count = selection == -1 ? 10_001 : 2
        let expected = selection == -1 ? 200 : (selection == 0 ? 500 : selection)
        let graph = try UMAPProgressFixtures.ring(count: count)
        var config = UMAPProgressFixtures.config(epochs: selection > 0 ? selection : nil)
        config.negativeSampleRate = 0
        var events: [UMAPProgress] = []
        _ = try Operations.umap(graph: graph, dimensions: 1,
                                 initialCoordinates: Array(repeating: [0], count: count), config: config) {
            if $0.stage == .layout { events.append($0) }
        }
        #expect(events.map(\.completed) == Array(0...expected))
        #expect(events.allSatisfy { $0.total == expected })
    }

    @Test func callbackRunsSynchronouslyOnCaller() throws {
        let key = UUID().uuidString
        Thread.current.threadDictionary[key] = key
        defer { Thread.current.threadDictionary.removeObject(forKey: key) }
        var insideCall = true
        var reports = 0
        _ = try UMAPProgressFixtures.run(.builtRandom, dimensions: 2) { _ in
            #expect(insideCall)
            #expect(Thread.current.threadDictionary[key] as? String == key)
            reports += 1
        }
        insideCall = false
        #expect(reports > 0)
    }

    @Test func standaloneGraphBuilderRetainsContract() throws {
        let data = UMAPProgressFixtures.vectors(count: 257)
        let ordinary = try KNNGraph.bruteForce(data, neighbors: 3)
        var events: [UMAPProgress] = []
        let observed = try KNNGraph.bruteForce(data, neighbors: 3,
                                              execution: UMAPExecutionContext(progress: { events.append($0) }))
        #expect(ordinary.rowOffsets == observed.rowOffsets)
        #expect(ordinary.neighborIndices == observed.neighborIndices)
        #expect(ordinary.distances.map(\.bitPattern) == observed.distances.map(\.bitPattern))
        #expect(events.map(\.completed) == [0, 256, 257])
    }

    @Test func failureDoesNotFabricateCompletion() throws {
        var events: [UMAPProgress] = []
        var config = UMAPProgressFixtures.config()
        config.spread = Float(bitPattern: 1)
        config.minDist = 0
        do {
            _ = try Operations.umap(UMAPProgressFixtures.vectors(), config: config) { events.append($0) }
            Issue.record("Expected the curve fit to fail")
        } catch let error as VectorError { #expect(error.kind == .operationFailed) }
        #expect(events.isEmpty)

        let empty = try KNNGraph(pointCount: 3, rowOffsets: [0, 0, 0, 0], neighborIndices: [], distances: [])
        do {
            _ = try Operations.umap(graph: empty, dimensions: 1, initialCoordinates: [[0], [1], [2]],
                                     config: UMAPProgressFixtures.config()) { events.append($0) }
            Issue.record("Expected empty fuzzy output to fail")
        } catch let error as VectorError { #expect(error.kind == .invalidData) }
        #expect(events == [UMAPProgress(stage: .fuzzySet, completed: 0, total: 1)])

        events.removeAll()
        do {
            _ = try Operations.umap(graph: empty, dimensions: 1, initialCoordinates: [[0], [.nan], [2]],
                                     config: UMAPProgressFixtures.config()) { events.append($0) }
            Issue.record("Expected invalid initial coordinates to fail")
        } catch let error as VectorError { #expect(error.kind == .invalidData) }
        #expect(events.isEmpty)

        let ragged = [DynamicVector([0, 1]), DynamicVector([1]), DynamicVector([2, 3])]
        do {
            _ = try Operations.umap(ragged, config: .init(neighbors: 2, epochs: 1)) { events.append($0) }
            Issue.record("Expected ragged vectors to fail")
        } catch let error as VectorError { #expect(error.kind == .dimensionMismatch) }
        #expect(events == [UMAPProgress(stage: .graph, completed: 0, total: 3)])
    }
}
