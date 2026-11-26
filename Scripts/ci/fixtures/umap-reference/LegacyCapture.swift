// Independent fixture client: imports only public API and copies no UMAP math.
// Compiled unchanged against the alpha; modern-only selectors are conditional.
import Foundation
import VectorCore

var state: UInt32 = 12345
let vectors = (0..<8).map { _ in
    DynamicVector((0..<6).map { _ -> Float in
        state = state &* 1664525 &+ 1013904223
        return Float(Int((state >> 8) % 2001) - 1000) / 257
    })
}
let graph = try KNNGraph.bruteForce(vectors, neighbors: 3)
var cases: [String: [UInt32]] = ["knn-legacy": graph.distances.map(\.bitPattern)]
#if VECTORCORE_CURRENT_API
let selections = ["legacy", "explicit"]
cases["knn-explicit"] = try KNNGraph.bruteForce(vectors, neighbors: 3, metric: .euclidean).distances.map(\.bitPattern)
cases["default-coefficients"] = try captureCurveParameters()
#else
let selections = ["legacy"]
#endif

for dimensions in [2, 5] {
    for route in ["builtRandom", "builtPCA", "suppliedRandom", "suppliedPCA", "graphRandom", "graphProvided"] {
        for selection in selections {
            let initialization: UMAPInitialization = route.hasSuffix("PCA") ? .pca : .random
            var config = UMAPConfig(neighbors: 3, epochs: 7, negativeSampleRate: 2,
                                    initialization: initialization, seed: 19)
            #if VECTORCORE_CURRENT_API
            if selection == "explicit" { config.metric = .euclidean }
            #endif
            let result: UMAPResult
            switch route {
            case "builtRandom", "builtPCA":
                result = try Operations.umap(vectors, dimensions: dimensions, config: config)
            case "suppliedRandom", "suppliedPCA":
                result = try Operations.umap(vectors, dimensions: dimensions, graph: graph, config: config)
            case "graphRandom":
                result = try Operations.umap(graph: graph, dimensions: dimensions, config: config)
            default:
                let coordinates = (0..<8).map { row in (0..<dimensions).map { Float(row * 7 - $0 * 3) / 11 } }
                result = try Operations.umap(graph: graph, dimensions: dimensions,
                                             initialCoordinates: coordinates, config: config)
            }
            cases["progress-\(route)-\(dimensions)-\(selection)"] = result.coordinates.map(\.bitPattern)
        }
    }
    for usesDefaultCurve in [true, false] {
        let curveGraph = try KNNGraph(pointCount: 4, rowOffsets: [0, 2, 4, 6, 8],
                                      neighborIndices: [1, 2, 0, 2, 1, 3, 2, 1],
                                      distances: [1, 2, 1, 1, 1, 2, 2, 3])
        let curveVectors = (0..<4).map { DynamicVector([Float($0)]) }
        var config = UMAPConfig(neighbors: 2, epochs: 10, negativeSampleRate: 2, initialization: .random, seed: 17)
        if !usesDefaultCurve { config.spread = 1.5; config.minDist = 0.5 }
        let vectorResult = try Operations.umap(curveVectors, dimensions: dimensions, graph: curveGraph, config: config)
        let graphResult = try Operations.umap(graph: curveGraph, dimensions: dimensions, config: config)
        cases["curve-\(dimensions)-\(usesDefaultCurve)-vectors"] = vectorResult.coordinates.map(\.bitPattern)
        cases["curve-\(dimensions)-\(usesDefaultCurve)-graph"] = graphResult.coordinates.map(\.bitPattern)
    }
}
let capture: [String: Any] = ["schema": 1, "cases": cases,
                             "rowOffsets": graph.rowOffsets, "neighborIndices": graph.neighborIndices,
                             "provider": String(reflecting: type(of: Operations.linearAlgebraProvider))]
let json = try JSONSerialization.data(withJSONObject: capture, options: [.sortedKeys])
print("UMAP_CAPTURE_JSON " + String(decoding: json, as: UTF8.self))
