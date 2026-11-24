import VectorCore

typealias LegacyConfig = (Int, Float, Float, Int?, Float, Int, UMAPInitialization, UInt64) -> UMAPConfig
let inferredConfig = UMAPConfig.init
let retainedInference: LegacyConfig = inferredConfig
let oldConfig: LegacyConfig = UMAPConfig.init
let newConfig: (Int, Float, Float, Int?, Float, Int, UMAPInitialization, UInt64, KNNGraph.Metric) -> UMAPConfig = UMAPConfig.init
let oldGraph: ([DynamicVector], Int) throws -> KNNGraph = KNNGraph.bruteForce
let labeledOldGraph: ([DynamicVector], Int) throws -> KNNGraph = KNNGraph.bruteForce(_:neighbors:)
let newGraph: ([DynamicVector], Int, KNNGraph.Metric) throws -> KNNGraph = KNNGraph.bruteForce
func sendable<T: Sendable>(_ value: T) {}
sendable(KNNGraph.Metric.cosine)

let rows = [[Float(1), 0, 2], [0, 1, 3], [-1, 0, 1], [0, -1, 4], [2, 1, 1], [-2, -3, 1]]
let vectors = rows.map(DynamicVector.init)
let euclidean = try oldGraph(vectors, 3)
let explicit = try newGraph(vectors, 3, .euclidean)
let labeled = try labeledOldGraph(vectors, 3)
precondition(euclidean.neighborIndices == explicit.neighborIndices)
precondition(euclidean.distances.map(\.bitPattern) == explicit.distances.map(\.bitPattern))
precondition(euclidean.distances.map(\.bitPattern) == labeled.distances.map(\.bitPattern))
precondition(UMAPConfig().metric == .euclidean)
for make in [oldConfig, retainedInference] {
    precondition(make(3, 0.1, 1, 3, 1, 2, .random, 19).metric == .euclidean)
}
var config = newConfig(3, 0.1, 1, 3, 1, 2, .random, 19, .cosine)
precondition(config.metric == .cosine)
sendable(config)
let cosine = try KNNGraph.bruteForce(vectors, neighbors: 3, metric: .cosine)
let built = try Operations.umap(vectors, config: config)
var stages: [UMAPProgress.Stage] = []
let observed = try Operations.umap(vectors, config: config) { if $0.completed == 0 { stages.append($0.stage) } }
let supplied = try Operations.umap(vectors, graph: cosine, config: config)
let graphOnly = try Operations.umap(graph: cosine, config: config, progress: nil)
precondition(stages == [.graph, .initialization, .fuzzySet, .layout])
precondition([observed, supplied, graphOnly].allSatisfy { $0.coordinates.map(\.bitPattern) == built.coordinates.map(\.bitPattern) })
config.metric = .euclidean
let authoritative = try Operations.umap(vectors, graph: cosine, config: config)
precondition(authoritative.coordinates.map(\.bitPattern) == built.coordinates.map(\.bitPattern))
do {
    _ = try KNNGraph.bruteForce([DynamicVector([1, 0]), DynamicVector([0, -0.0])], neighbors: 1, metric: .cosine)
    preconditionFailure("Zero cosine input must throw")
} catch let error as VectorError {
    precondition(error.kind == .invalidData)
    precondition(String(describing: error).contains("row 1"))
}
print("KNN_METRIC_OK")
