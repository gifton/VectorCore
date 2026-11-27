import Foundation
import VectorCore

// This executable's synchronous entry is deliberately outside Swift task execution.
precondition(withUnsafeCurrentTask { $0 == nil })

let vectors: [DynamicVector] = (0..<8).map { index in
    DynamicVector([Float(index % 3), Float(index / 3), Float(index % 2)])
}
let graph = try KNNGraph.bruteForce(vectors, neighbors: 3)
let config = UMAPConfig(neighbors: 3, epochs: 3, negativeSampleRate: 2,
                        initialization: .random, seed: 19)
let legacyGraph: (KNNGraph, Int, [[Float]]?, UMAPConfig) throws -> UMAPResult = Operations.umap
let inferredGraph = Operations.umap
let inferredAsLegacy: (KNNGraph, Int, [[Float]]?, UMAPConfig) throws -> UMAPResult = inferredGraph
let legacyVectors: ([DynamicVector], Int, KNNGraph?, UMAPConfig) throws -> UMAPResult = Operations.umap

let expected = try Operations.umap(graph: graph, dimensions: 2, config: config).coordinates.map(\.bitPattern)
let unchanged = [
    try legacyGraph(graph, 2, nil, config),
    try inferredAsLegacy(graph, 2, nil, config),
    try legacyVectors(vectors, 2, graph, config),
    try Operations.umap(vectors, dimensions: 2, graph: graph, config: config),
    try Operations.umap(graph: graph, dimensions: 2, config: config, progress: nil),
    try Operations.umap(vectors, dimensions: 2, graph: graph, config: config, progress: nil)
]
precondition(unchanged.allSatisfy { $0.coordinates.map(\.bitPattern) == expected })

func requireSendable<T: Sendable>(_ value: T) {}

// Exhaustive consumer mapping pins the closed four-stage source contract.
func stageName(_ stage: UMAPProgress.Stage) -> String {
    switch stage {
    case .graph: return "graph"
    case .initialization: return "initialization"
    case .fuzzySet: return "fuzzySet"
    case .layout: return "layout"
    }
}
let synthetic = UMAPProgress(stage: .layout, completed: 2, total: 4)
precondition(stageName(synthetic.stage) == "layout")
precondition(Double(synthetic.completed) / Double(synthetic.total) == 0.5)
let rawSynthetic = UMAPProgress(stage: .graph, completed: -3, total: 0)
precondition(rawSynthetic.completed == -3 && rawSynthetic.total == 0)

var events: [UMAPProgress] = []
let caller = Thread.current
let observed = try Operations.umap(vectors, dimensions: 2, config: config) { event in
    precondition(Thread.current === caller)
    requireSendable(event)
    requireSendable(event.stage)
    precondition(event.total > 0 && event.completed >= 0 && event.completed <= event.total)
    events.append(event)
}
precondition(observed.coordinates.map(\.bitPattern) == expected)
precondition(events.first == UMAPProgress(stage: .graph, completed: 0, total: vectors.count))
let stages: [UMAPProgress.Stage] = [.graph, .initialization, .fuzzySet, .layout]
precondition(events.filter { $0.completed == 0 }.map(\.stage) == stages)
for stage in stages {
    let stageEvents = events.filter { $0.stage == stage }
    let total = stage == .graph ? vectors.count : (stage == .layout ? 3 : 1)
    precondition(stageEvents.allSatisfy { $0.total == total })
    precondition(stageEvents.map(\.completed) == (stage == .layout ? [0, 1, 2, 3] : [0, total]))
}

let coordinates: [[Float]] = (0..<vectors.count).map { [Float($0), Float(-$0)] }
var suppliedStages: [UMAPProgress.Stage] = []
_ = try Operations.umap(graph: graph, dimensions: 2, initialCoordinates: coordinates, config: config) { event in
    if event.completed == 0 { suppliedStages.append(event.stage) }
}
precondition(suppliedStages == [.fuzzySet, .layout])

// The callback's capture must be released when the synchronous fit returns.
final class Capture {}
weak var captureAfterCall: Capture?
do {
    let capture = Capture()
    captureAfterCall = capture
    _ = try Operations.umap(graph: graph, config: config) { _ in
        withExtendedLifetime(capture) {}
    }
}
precondition(captureAfterCall == nil)

print("UMAP_PROGRESS_OK")
