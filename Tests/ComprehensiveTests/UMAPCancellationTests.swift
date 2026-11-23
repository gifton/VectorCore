import Foundation
import Testing
@testable import VectorCore

// The lock covers every access to counts; cancelOnBorrow is immutable.
private final class UMAPWorkWitness: @unchecked Sendable {
    private let lock = NSLock()
    private var counts: [String: Int] = [:]
    let cancelOnBorrow: Bool

    init(cancelOnBorrow: Bool = false) { self.cancelOnBorrow = cancelOnBorrow }

    func record(_ kind: String) {
        lock.lock()
        counts[kind, default: 0] += 1
        let cancel = cancelOnBorrow && kind == "borrow" && counts[kind] == 1
        lock.unlock()
        if cancel { withUnsafeCurrentTask { $0?.cancel() } }
    }

    var snapshot: [String: Int] {
        lock.lock()
        defer { lock.unlock() }
        return counts
    }
}

private struct UMAPAccessVector: VectorProtocol {
    var storage: [Float]
    let witness: UMAPWorkWitness
    var scalarCount: Int { witness.record("shape"); return storage.count }
    subscript(index: Int) -> Float { witness.record("element"); return storage[index] }
    init() { storage = [0]; witness = UMAPWorkWitness() }
    init(_ array: [Float]) { storage = array; witness = UMAPWorkWitness() }
    init(repeating value: Float) { storage = [value]; witness = UMAPWorkWitness() }
    init(_ values: [Float], witness: UMAPWorkWitness) { storage = values; self.witness = witness }
    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.storage.map(\.bitPattern) == rhs.storage.map(\.bitPattern)
    }
    func hash(into hasher: inout Hasher) {
        for value in storage { hasher.combine(value.bitPattern) }
    }
    init(from decoder: any Decoder) throws {
        self.init(try decoder.singleValueContainer().decode([Float].self))
    }
    func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(storage)
    }
    func toArray() -> [Float] { witness.record("array"); return storage }
    func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        witness.record("borrow")
        return try storage.withUnsafeBufferPointer(body)
    }
    mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        witness.record("mutable")
        return try storage.withUnsafeMutableBufferPointer { try body($0) }
    }
}

private struct UMAPCancellingProvider: LinearAlgebraProvider {
    let base: any LinearAlgebraProvider
    let witness: UMAPWorkWitness
    let cancelAfterSVD: Bool

    func qrThin(_ a: [Float], rows: Int, columns: Int) throws -> QRFactorization {
        witness.record("qr")
        return try base.qrThin(a, rows: rows, columns: columns)
    }

    func svdThin(_ a: [Float], rows: Int, columns: Int) throws -> SingularValueDecomposition {
        witness.record("svd")
        let result = try base.svdThin(a, rows: rows, columns: columns)
        if cancelAfterSVD { withUnsafeCurrentTask { $0?.cancel() } }
        return result
    }

    func symmetricEigen(_ a: [Float], dimension: Int, computeEigenvectors: Bool) throws -> SymmetricEigenDecomposition {
        witness.record("eigen")
        return try base.symmetricEigen(a, dimension: dimension, computeEigenvectors: computeEigenvectors)
    }
}

@Suite
struct UMAPCancellationTests {
    enum CallStyle: CaseIterable, Sendable { case legacy, nilHandler, handler }

    private func invoke(graphOnly: Bool, style: CallStyle, values: [UMAPAccessVector],
                        graph: KNNGraph, config: UMAPConfig, progress: @escaping (UMAPProgress) -> Void) throws {
        if graphOnly {
            switch style {
            case .legacy: _ = try Operations.umap(graph: graph, config: config)
            case .nilHandler: _ = try Operations.umap(graph: graph, config: config, progress: nil)
            case .handler: _ = try Operations.umap(graph: graph, config: config, progress: progress)
            }
        } else {
            switch style {
            case .legacy: _ = try Operations.umap(values, config: config)
            case .nilHandler: _ = try Operations.umap(values, config: config, progress: nil)
            case .handler: _ = try Operations.umap(values, config: config, progress: progress)
            }
        }
    }

    @Test(arguments: [false, true], CallStyle.allCases)
    func entryCancellationPrecedesWorkAndValidation(graphOnly: Bool, style: CallStyle) async throws {
        for invalid in [false, true] {
            let witness = UMAPWorkWitness()
            let values = UMAPProgressFixtures.vectors().map { UMAPAccessVector($0.toArray(), witness: witness) }
            let graph = try UMAPProgressFixtures.ring(count: values.count)
            var config = UMAPProgressFixtures.config(initialization: .pca)
            if invalid { config.epochs = 0 }
            let selected = config
            await Task {
                withUnsafeCurrentTask { $0?.cancel() }
                var events: [UMAPProgress] = []
                do {
                    try invoke(graphOnly: graphOnly, style: style, values: values, graph: graph, config: selected) {
                        events.append($0)
                    }
                    Issue.record("A pre-cancelled UMAP call returned a result")
                } catch is CancellationError {
                } catch { Issue.record("Expected entry cancellation, received \(error)") }
                #expect(events.isEmpty)
            }.value
            #expect(witness.snapshot.isEmpty)
        }
    }

    @Test(arguments: UMAPProgressFixtures.Route.allCases)
    func cancellingEveryProgressEventStopsExecution(route: UMAPProgressFixtures.Route) async throws {
        var control: [UMAPProgress] = []
        _ = try UMAPProgressFixtures.run(route, dimensions: 2) { control.append($0) }
        for stop in control.indices {
            let expected = Array(control.prefix(stop + 1))
            await Task {
                var events: [UMAPProgress] = []
                do {
                    _ = try UMAPProgressFixtures.run(route, dimensions: 2) {
                        events.append($0)
                        if events.count == stop + 1 { withUnsafeCurrentTask { $0?.cancel() } }
                    }
                    Issue.record("Returned after cancellation at \(route), event \(stop)")
                } catch is CancellationError {
                } catch { Issue.record("Expected cancellation at event \(stop), received \(error)") }
                #expect(events == expected)
            }.value
        }
    }

    @Test func graphCancellationStopsAtEachBlockReport() async throws {
        let values = UMAPProgressFixtures.vectors(count: 513)
        let config = UMAPProgressFixtures.config(epochs: 1)
        for completed in [0, 256, 512, 513] {
            await Task {
                var events: [UMAPProgress] = []
                do {
                    _ = try Operations.umap(values, config: config) {
                        events.append($0)
                        if $0.stage == .graph && $0.completed == completed { withUnsafeCurrentTask { $0?.cancel() } }
                    }
                    Issue.record("Returned after cancelling graph at row \(completed)")
                } catch is CancellationError {
                } catch { Issue.record("Unexpected graph error: \(error)") }
                let expected = [0, 256, 512, 513].filter { $0 <= completed }
                #expect(events.map(\.completed) == expected)
                #expect(events.allSatisfy { $0.stage == .graph && $0.total == 513 })
            }.value
        }
    }

    @Test func cancellationFromOnlyEpochCompletionThrows() async throws {
        let graph = try UMAPProgressFixtures.ring(count: 8)
        await Task {
            #expect(throws: CancellationError.self) {
                _ = try Operations.umap(graph: graph, config: UMAPProgressFixtures.config(epochs: 1)) {
                    if $0.stage == .layout && $0.completed == 1 { withUnsafeCurrentTask { $0?.cancel() } }
                }
            }
        }.value
    }

    @Test func nilHandlerStillChecksCancellation() async throws {
        let witness = UMAPWorkWitness(cancelOnBorrow: true)
        let values = UMAPProgressFixtures.vectors().map { UMAPAccessVector($0.toArray(), witness: witness) }
        let provider = UMAPCancellingProvider(base: Operations.linearAlgebraProvider, witness: witness, cancelAfterSVD: false)
        await Task {
            do {
                _ = try Operations.$linearAlgebraProvider.withValue(provider) {
                    try Operations.umap(values, config: UMAPProgressFixtures.config(initialization: .pca), progress: nil)
                }
                Issue.record("Nil handler disabled cancellation")
            } catch is CancellationError {
            } catch { Issue.record("Expected cancellation during graph preparation, received \(error)") }
        }.value
        // Packing may finish, but initialization must not start after cancellation.
        #expect(witness.snapshot["borrow"] == values.count)
        #expect(witness.snapshot["svd"] == nil)
    }

    @Test func cancellationAfterPCAFittingPrecedesTransform() async throws {
        let data = UMAPProgressFixtures.vectors()
        let graph = try KNNGraph.bruteForce(data, neighbors: 3)
        let witness = UMAPWorkWitness()
        let values = data.map { UMAPAccessVector($0.toArray(), witness: witness) }
        let provider = UMAPCancellingProvider(base: Operations.linearAlgebraProvider, witness: witness, cancelAfterSVD: true)
        await Task {
            do {
                _ = try Operations.$linearAlgebraProvider.withValue(provider) {
                    try Operations.umap(values, graph: graph,
                                         config: UMAPProgressFixtures.config(initialization: .pca), progress: nil)
                }
                Issue.record("Returned after cancellation within PCA fit")
            } catch is CancellationError {
            } catch { Issue.record("Expected cancellation after PCA fit, received \(error)") }
        }.value
        #expect(witness.snapshot["svd"] == 1)
        // Fit packs each row once; transform would borrow every row again.
        #expect(witness.snapshot["borrow"] == values.count)
    }

    @Test func standaloneGraphBuilderRetainsCancelledTaskBehavior() async throws {
        let values = UMAPProgressFixtures.vectors()
        let expected = try KNNGraph.bruteForce(values, neighbors: 3)
        let actual = try await Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try KNNGraph.bruteForce(values, neighbors: 3)
        }.value
        #expect(actual.neighborIndices == expected.neighborIndices)
        #expect(actual.distances.map(\.bitPattern) == expected.distances.map(\.bitPattern))
    }

    @Test func internalNilHandlerContextsCheckEntry() async throws {
        let witness = UMAPWorkWitness()
        let values = UMAPProgressFixtures.vectors().map { UMAPAccessVector($0.toArray(), witness: witness) }
        await Task {
            withUnsafeCurrentTask { $0?.cancel() }
            do {
                _ = try KNNGraph.bruteForce(values, neighbors: 3, execution: UMAPExecutionContext(progress: nil))
                Issue.record("Observed builder ignored entry cancellation")
            } catch is CancellationError {
            } catch { Issue.record("Unexpected observed-builder error: \(error)") }
            var coordinates: [Float] = [0, 1]
            let edges = UMAPEdgeList(heads: [0, 1], tails: [1, 0], weights: [1, 1])
            do {
                try UMAPLayout.optimize(edges: edges, coordinates: &coordinates, pointCount: 2, dimension: 1,
                    parameters: .init(a: 1, b: 1, epochs: 3, learningRate: 1, negativeSampleRate: 0, seed: 19),
                    execution: UMAPExecutionContext(progress: nil))
                Issue.record("Observed optimizer ignored entry cancellation")
            } catch is CancellationError {
            } catch { Issue.record("Unexpected observed-optimizer error: \(error)") }
            #expect(coordinates.map(\.bitPattern) == [Float(0).bitPattern, Float(1).bitPattern])
        }.value
        #expect(witness.snapshot.isEmpty)
    }

    @Test func cancellationPreservesCallerInputs() async throws {
        let graph = try UMAPProgressFixtures.ring(count: 8)
        let initial = UMAPProgressFixtures.initial(dimensions: 2)
        let before = initial.flatMap { $0.map(\.bitPattern) }
        await Task {
            do {
                _ = try Operations.umap(graph: graph, initialCoordinates: initial,
                                         config: UMAPProgressFixtures.config()) {
                    if $0.stage == .layout && $0.completed == 2 { withUnsafeCurrentTask { $0?.cancel() } }
                }
                Issue.record("Expected cancellation during layout")
            } catch is CancellationError {
            } catch { Issue.record("Unexpected layout error: \(error)") }
        }.value
        #expect(initial.flatMap { $0.map(\.bitPattern) } == before)
        #expect(graph.rowOffsets == Array(0...8))
        #expect(graph.neighborIndices == [1, 2, 3, 4, 5, 6, 7, 0])
        #expect(graph.distances.map(\.bitPattern) == Array(repeating: Float(1).bitPattern, count: 8))
    }
}
