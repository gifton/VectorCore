import Testing
@testable import VectorCore

@Suite
struct UMAPMetricTests {
    @Test(arguments: [UMAPInitialization.random, .pca], [2, 5])
    func metricControlsOnlyBuiltGraph(initialization: UMAPInitialization, dimensions: Int) throws {
        let values = UMAPProgressFixtures.vectors()
        var config = UMAPProgressFixtures.config(initialization: initialization)
        config.metric = .cosine
        let graph = try KNNGraph.bruteForce(values, neighbors: config.neighbors, metric: .cosine)
        var builtEvents: [UMAPProgress] = []
        let built = try Operations.umap(values, dimensions: dimensions, config: config) { builtEvents.append($0) }
        let withoutHandler = try Operations.umap(values, dimensions: dimensions, config: config)
        var suppliedEvents: [UMAPProgress] = []
        let supplied = try Operations.umap(values, dimensions: dimensions, graph: graph, config: config) { suppliedEvents.append($0) }
        #expect(built.coordinates.map(\.bitPattern) == supplied.coordinates.map(\.bitPattern))
        #expect(built.coordinates.map(\.bitPattern) == withoutHandler.coordinates.map(\.bitPattern))
        #expect(builtEvents.filter { $0.stage != .graph } == suppliedEvents)
        #expect(builtEvents.filter { $0.completed == 0 }.map(\.stage) == [.graph, .initialization, .fuzzySet, .layout])

        // Changing only metric cannot rescore a supplied graph or normalize the original PCA inputs.
        config.metric = .euclidean
        var euclideanEvents: [UMAPProgress] = []
        let euclidean = try Operations.umap(values, dimensions: dimensions, graph: graph, config: config) { euclideanEvents.append($0) }
        #expect(euclidean.coordinates.map(\.bitPattern) == supplied.coordinates.map(\.bitPattern))
        #expect(euclideanEvents == suppliedEvents)
        let internalEuclidean = try Operations.umap(values, dimensions: dimensions, config: config)
        #expect(internalEuclidean.coordinates.map(\.bitPattern) != built.coordinates.map(\.bitPattern))

        for initial in [nil, UMAPProgressFixtures.initial(dimensions: dimensions)] {
            var control: [UMAPProgress] = []
            let old = try Operations.umap(graph: graph, dimensions: dimensions, initialCoordinates: initial, config: config) {
                control.append($0)
            }
            config.metric = .cosine
            var observed: [UMAPProgress] = []
            let selected = try Operations.umap(graph: graph, dimensions: dimensions, initialCoordinates: initial, config: config) {
                observed.append($0)
            }
            #expect(old.coordinates.map(\.bitPattern) == selected.coordinates.map(\.bitPattern))
            #expect(control == observed)
            config.metric = .euclidean
        }
    }

    @Test func suppliedGraphDoesNotValidateCosineVectorsForRandomInitialization() throws {
        let values = (0..<4).map { _ in DynamicVector([0, 0]) }
        let graph = try UMAPProgressFixtures.ring(count: 4)
        let config = UMAPConfig(neighbors: 3, epochs: 2, initialization: .random, metric: .cosine)
        let result = try Operations.umap(values, graph: graph, config: config)
        #expect(result.coordinates.allSatisfy { $0.isFinite })
    }

    @Test(arguments: [255, 256, 257, 513])
    func cosineProgressAtBlockBoundaries(count: Int) throws {
        var config = UMAPProgressFixtures.config(epochs: 1)
        config.metric = .cosine
        var events: [UMAPProgress] = []
        _ = try Operations.umap(UMAPProgressFixtures.vectors(count: count), config: config) { events.append($0) }
        let expected = [0] + Array(stride(from: 256, to: count, by: 256)) + [count]
        #expect(events.filter { $0.stage == .graph }.map(\.completed) == expected)
        #expect(events.filter { $0.stage == .graph }.allSatisfy { $0.total == count })
    }

    @Test func invalidCosineRowsDoNotReportCompletion() throws {
        for bad: Float in [0, .nan, .infinity] {
            var values = UMAPProgressFixtures.vectors(count: 257)
            values[256] = DynamicVector(Array(repeating: bad, count: 6))
            var config = UMAPProgressFixtures.config(epochs: 1)
            config.metric = .cosine
            var events: [UMAPProgress] = []
            #expect(throws: VectorError.self) {
                try Operations.umap(values, config: config) { events.append($0) }
            }
            #expect(events == [UMAPProgress(stage: .graph, completed: 0, total: 257)])
        }
    }

    @Test func cosineCancellationAtGraphReports() async throws {
        let values = UMAPProgressFixtures.vectors(count: 513)
        let config = UMAPConfig(neighbors: 3, epochs: 1, initialization: .random, metric: .cosine)
        for completed in [0, 256, 512, 513] {
            await Task {
                var events: [UMAPProgress] = []
                do {
                    _ = try Operations.umap(values, config: config) {
                        events.append($0)
                        if $0.completed == completed { withUnsafeCurrentTask { $0?.cancel() } }
                    }
                    Issue.record("Returned after graph cancellation")
                } catch is CancellationError {
                } catch { Issue.record("Expected cancellation, received \(error)") }
                #expect(events.allSatisfy { $0.stage == .graph })
                #expect(events.map(\.completed) == [0, 256, 512, 513].filter { $0 <= completed })
            }.value
        }
    }

    @Test func cosineEntryAndInitialCallbackCancellationPrecedeValidation() async {
        let values = [DynamicVector([0, 0]), DynamicVector([.nan, 1]), DynamicVector([1, 1])]
        for entry in [true, false] {
            for handler in [true, false] where entry || handler {
                await Task {
                    if entry { withUnsafeCurrentTask { $0?.cancel() } }
                    var events: [UMAPProgress] = []
                    let callback: ((UMAPProgress) -> Void)? = handler ? { event in
                        events.append(event)
                        withUnsafeCurrentTask { $0?.cancel() }
                    } : nil
                    // Entry cancellation must even precede invalid config; initial callback must precede zero/NaN rows.
                    let config = UMAPConfig(neighbors: 2, epochs: entry ? 0 : 1, initialization: .random, metric: .cosine)
                    do {
                        _ = try Operations.umap(values, config: config, progress: callback)
                        Issue.record("Cancelled cosine UMAP returned")
                    } catch is CancellationError {
                    } catch { Issue.record("Validation beat cancellation: \(error)") }
                    #expect(events.count == (entry ? 0 : 1))
                }.value
            }
        }
    }

    @Test func standaloneCosineRunsInCancelledTask() async throws {
        let values = UMAPProgressFixtures.vectors()
        let expected = try KNNGraph.bruteForce(values, neighbors: 3, metric: .cosine)
        try await Task {
            withUnsafeCurrentTask { $0?.cancel() }
            let result = try KNNGraph.bruteForce(values, neighbors: 3, metric: .cosine)
            #expect(result.neighborIndices == expected.neighborIndices)
            #expect(result.distances.map(\.bitPattern) == expected.distances.map(\.bitPattern))
        }.value
    }
}
