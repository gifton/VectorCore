import Testing
@testable import VectorCore

@Suite
struct UMAPOptimizerInvariantTests {
    @Test(arguments: [false, true])
    func publicEmptyEdgesFailBeforeLayout(vectorEntry: Bool) throws {
        let graph = try KNNGraph(pointCount: 3, rowOffsets: [0, 0, 0, 0], neighborIndices: [], distances: [])
        let values = [DynamicVector([0]), DynamicVector([1]), DynamicVector([2])]
        let config = UMAPConfig(neighbors: 2, epochs: 1, initialization: .random)
        var events: [UMAPProgress] = []
        do {
            if vectorEntry {
                _ = try Operations.umap(values, dimensions: 2, graph: graph, config: config) { events.append($0) }
            } else {
                _ = try Operations.umap(graph: graph, dimensions: 2, config: config) { events.append($0) }
            }
            Issue.record("Empty fuzzy graph returned a public layout")
        } catch let error as VectorError {
            #expect(error.kind == .invalidData)
        }
        #expect(!events.contains { $0.stage == .layout })
        #expect(events.filter { $0.stage == .fuzzySet }.map(\.completed) == [0])
    }

    @Test(arguments: [false, true])
    func internalEmptyOrZeroEdgesAreNoOps(hasZeroEdge: Bool) throws {
        let edges = UMAPEdgeList(heads: hasZeroEdge ? [0] : [], tails: hasZeroEdge ? [1] : [],
                                 weights: hasZeroEdge ? [0] : [])
        var coordinates: [Float] = [-0.0, 1]
        let original = coordinates.map(\.bitPattern)
        var events: [UMAPProgress] = []
        try UMAPLayout.optimize(edges: edges, coordinates: &coordinates, pointCount: 2, dimension: 1,
                                parameters: .init(a: 1, b: 1, epochs: 11, learningRate: 1, negativeSampleRate: 0, seed: 19),
                                execution: UMAPExecutionContext(progress: { events.append($0) }))
        #expect(coordinates.map(\.bitPattern) == original)
        #expect(events.isEmpty)
    }

    @Test(arguments: [1, 10, 11], [Float(1), Float.leastNonzeroMagnitude])
    func positiveMaximumSurvivesPruning(epochs: Int, maximum: Float) throws {
        let edges = UMAPEdgeList(heads: [0, 1], tails: [1, 0], weights: [0, maximum])
        var coordinates: [Float] = [0, 1]
        var events: [UMAPProgress] = []
        try UMAPLayout.optimize(edges: edges, coordinates: &coordinates, pointCount: 2, dimension: 1,
                                parameters: .init(a: 1, b: 1, epochs: epochs, learningRate: 1, negativeSampleRate: 0, seed: 19),
                                execution: UMAPExecutionContext(progress: { events.append($0) }))
        #expect(events.map(\.stage) == Array(repeating: .layout, count: epochs))
        #expect(events.map(\.completed) == Array(1...epochs))
        #expect(events.allSatisfy { $0.total == epochs && $0.total > 0 && $0.completed <= $0.total })
        #expect(coordinates.allSatisfy { $0.isFinite })
    }

    @Test func cancellationPrecedesInternalNoOp() async {
        await Task {
            withUnsafeCurrentTask { $0?.cancel() }
            var coordinates: [Float] = [0, 1]
            do {
                try UMAPLayout.optimize(edges: UMAPEdgeList(heads: [], tails: [], weights: []),
                                        coordinates: &coordinates, pointCount: 2, dimension: 1,
                                        parameters: .init(a: 1, b: 1, epochs: 1, learningRate: 1, negativeSampleRate: 0, seed: 19),
                                        execution: UMAPExecutionContext(progress: nil))
                Issue.record("Cancelled internal no-op returned success")
            } catch is CancellationError {
            } catch { Issue.record("Unexpected cancellation error: \(error)") }
            #expect(coordinates.map(\.bitPattern) == [Float(0).bitPattern, Float(1).bitPattern])
        }.value
    }
}
