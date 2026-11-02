import Foundation
import VectorCore

@main struct L03PublicConsumer {
    static func reject(_ kind: VectorError.ErrorKind, _ body: () throws -> Void) throws {
        do { try body(); fatalError("Expected typed rejection") }
        catch let error as VectorError { precondition(error.kind == kind) }
    }
    static func main() throws {
        let graph = try KNNGraph(pointCount: 3, rowOffsets: [0, 2, 3, 4],
                                 neighborIndices: [1, 2, 0, 0], distances: [1, 2, 1, 2])
        let input: [[Float]] = [[-1], [0], [2]]
        let bits = input.map { $0.map(\.bitPattern) }
        let config = UMAPConfig(neighbors: 2, epochs: 8, negativeSampleRate: 2, seed: 17)
        let first = try Operations.umap(graph: graph, dimensions: 1, initialCoordinates: input, config: config)
        let second = try Operations.umap(graph: graph, dimensions: 1, initialCoordinates: input, config: config)
        precondition(first.pointCount == 3 && first.dimension == 1 && first.coordinates.count == 3)
        precondition(first.coordinates.allSatisfy(\.isFinite))
        precondition(first.coordinates.map(\.bitPattern) == second.coordinates.map(\.bitPattern))
        precondition(input.map { $0.map(\.bitPattern) } == bits)
        let vectors = [[Float](arrayLiteral: 0, 0), [1, 0], [0, 2]].map { DynamicVector($0) }
        func project() throws {
            let result = try Operations.umap(vectors, dimensions: 1,
                config: UMAPConfig(neighbors: 2, epochs: 5, seed: 19))
            precondition(result.coordinates.count == 3 && result.coordinates.allSatisfy(\.isFinite))
        }
        try project()
        try Operations.$linearAlgebraProvider.withValue(SwiftLinearAlgebraProvider()) { try project() }
        for invalid: Float in [.nan, .infinity, -.infinity] {
            for field in 0..<3 {
                var bad = config
                if field == 0 { bad.spread = invalid }
                if field == 1 { bad.minDist = invalid }
                if field == 2 { bad.learningRate = invalid }
                try reject(.invalidOperation) { _ = try Operations.umap(graph: graph, config: bad) }
                try reject(.invalidOperation) { _ = try Operations.umap(vectors, config: bad) }
            }
        }
        try reject(.invalidData) {
            _ = try KNNGraph(pointCount: 2, rowOffsets: [0, 3, 2], neighborIndices: [1, 0], distances: [1, 1])
        }
        try reject(.invalidDimension) { _ = try Operations.umap(graph: graph, dimensions: Int.max, config: config) }
        try reject(.invalidDimension) { _ = try Operations.umap(vectors, dimensions: Int.max, config: config) }
        for field in 0..<2 {
            var bad = config
            if field == 0 { bad.epochs = Int.max } else { bad.negativeSampleRate = Int.max }
            try reject(.invalidOperation) { _ = try Operations.umap(graph: graph, config: bad) }
            try reject(.invalidOperation) { _ = try Operations.umap(vectors, config: bad) }
        }
        print("L03_PUBLIC_COMPLETE ordinary graph/vector routes, provider override, typed boundaries, immutable input")
    }
}
