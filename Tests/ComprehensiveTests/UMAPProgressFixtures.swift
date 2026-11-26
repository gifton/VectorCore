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
