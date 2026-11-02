import Foundation
import Testing
@testable import VectorCore

@Suite(.serialized)
struct UMAPPreflightTests {
    private let rejectedCurves: [(UInt32, UInt32)] = [
        (1, 0), (0x7f7fffff, 0), (0x67800000, 0x65cccccd)
    ]

    private func config(_ words: (UInt32, UInt32), initialization: UMAPInitialization = .random) -> UMAPConfig {
        UMAPConfig(neighbors: 2, minDist: Float(bitPattern: words.1), spread: Float(bitPattern: words.0),
                   epochs: 1, negativeSampleRate: 0, initialization: initialization, seed: 17)
    }

    private func graph(empty: Bool = false, count: Int = 3) throws -> KNNGraph {
        if empty {
            return try KNNGraph(pointCount: count, rowOffsets: Array(repeating: 0, count: count + 1),
                                neighborIndices: [], distances: [])
        }
        return try KNNGraph(pointCount: 3, rowOffsets: [0, 2, 2, 2],
                            neighborIndices: [1, 2], distances: [1, 2])
    }

    private func vectors() -> [LC5AccessVector] { [LC5AccessVector([0]), LC5AccessVector([1]), LC5AccessVector([2])] }

    private func expectKind(_ kind: VectorError.ErrorKind, _ body: () throws -> Void) {
        do { try body(); Issue.record("Expected VectorError.\(kind)") } catch let error as VectorError {
            #expect(error.kind == kind)
        } catch { Issue.record("Wrong error family: \(error)") }
    }

    @Test func rejectedCurvePrecedesGraphConstructionForBothInitializers() {
        // Moving fitting behind bruteForce would borrow the vector payloads.
        for words in rejectedCurves {
            for initialization: UMAPInitialization in [.random, .pca] {
                let values = vectors()
                LC5AccessVector.borrows.reset()
                expectKind(.operationFailed) {
                    _ = try Operations.$linearAlgebraProvider.withValue(LC5ThrowingProvider()) {
                        try Operations.umap(values, dimensions: 1, config: config(words, initialization: initialization))
                    }
                }
                #expect(LC5AccessVector.borrows.count == 0)
            }
        }
    }

    @Test func rejectedCurvePrecedesPCAWithSuppliedGraph() throws {
        // A late fit reaches the immediate provider failure, never a hanging LAPACK call.
        let g = try graph()
        for words in rejectedCurves {
            let values = vectors()
            LC5AccessVector.borrows.reset()
            expectKind(.operationFailed) {
                _ = try Operations.$linearAlgebraProvider.withValue(LC5ThrowingProvider()) {
                    try Operations.umap(values, dimensions: 1, graph: g, config: config(words, initialization: .pca))
                }
            }
            #expect(LC5AccessVector.borrows.count == 0)
        }
    }

    @Test func rejectedCurvePrecedesFuzzyEdgesOnGraphRoutes() throws {
        // Empty CSR is valid interchange, but fuzzy construction yields no usable edges.
        let empty = try graph(empty: true)
        for words in rejectedCurves {
            for initial: [[Float]]? in [nil, [[0], [1], [2]]] {
                expectKind(.operationFailed) {
                    _ = try Operations.umap(graph: empty, dimensions: 1, initialCoordinates: initial, config: config(words))
                }
            }
            expectKind(.operationFailed) {
                _ = try Operations.umap(vectors(), dimensions: 1, graph: empty, config: config(words))
            }
        }
    }

    @Test func admittedCurvesReachDownstreamWitnesses() throws {
        let g = try graph()
        let empty = try graph(empty: true)
        for words: (UInt32, UInt32) in [(0x3f800000, 0x3dcccccd), (0x2f000000, 0x2f000000), (0x50000000, 0x50000000)] {
            let values = vectors()
            LC5AccessVector.borrows.reset()
            let result = try Operations.umap(values, dimensions: 1, config: config(words))
            #expect(result.coordinates.count == 3 && result.coordinates.allSatisfy(\.isFinite))
            #expect(LC5AccessVector.borrows.count > 0)
            do {
                _ = try Operations.$linearAlgebraProvider.withValue(LC5ThrowingProvider()) {
                    try Operations.umap(values, dimensions: 1, graph: g, config: config(words, initialization: .pca))
                }
                Issue.record("Expected PCA provider witness")
            } catch is LC5ProviderReached {
            } catch { Issue.record("PCA provider witness was not reached: \(error)") }
            for initial: [[Float]]? in [nil, [[0], [1], [2]]] {
                expectKind(.invalidData) {
                    _ = try Operations.umap(graph: empty, dimensions: 1, initialCoordinates: initial, config: config(words))
                }
            }
        }
    }

    @Test func admittedCurvesPreserveSeededInitializers() throws {
        let g = try graph()
        let values = vectors()
        // One epoch schedules no attractive update, exposing the seeded initialization.
        for words: (UInt32, UInt32) in [(0x3f800000, 0x3dcccccd), (0x2f000000, 0x2f000000), (0x50000000, 0x50000000)] {
            let random = config(words)
            let first = try Operations.umap(values, dimensions: 1, graph: g, config: random)
            let graphOnly = try Operations.umap(graph: g, dimensions: 1, config: random)
            #expect(first.coordinates.map(\.bitPattern) == graphOnly.coordinates.map(\.bitPattern))
            var otherSeed = random
            otherSeed.seed = 18
            let other = try Operations.umap(graph: g, dimensions: 1, config: otherSeed)
            #expect(first.coordinates.map(\.bitPattern) != other.coordinates.map(\.bitPattern))
            let pca = config(words, initialization: .pca)
            try Operations.$linearAlgebraProvider.withValue(SwiftLinearAlgebraProvider()) {
                let firstPCA = try Operations.umap(values, dimensions: 1, graph: g, config: pca)
                let secondPCA = try Operations.umap(values, dimensions: 1, graph: g, config: pca)
                let finite = firstPCA.coordinates.allSatisfy(\.isFinite)
                #expect(finite)
                #expect(firstPCA.coordinates.map(\.bitPattern) == secondPCA.coordinates.map(\.bitPattern))
            }
        }
    }

    @Test func cheapValidationPrecedesRejectedCurve() throws {
        let g = try graph()
        let one = try graph(empty: true, count: 1)
        let two = try graph(empty: true, count: 2)
        let values = vectors()
        for words in rejectedCurves {
            let rejected = config(words)
            var invalidConfig = rejected
            invalidConfig.neighbors = 1
            expectKind(.invalidDimension) { _ = try Operations.umap(values, dimensions: 1, graph: g, config: invalidConfig) }
            expectKind(.invalidDimension) { _ = try Operations.umap(graph: g, dimensions: 1, config: invalidConfig) }
            expectKind(.invalidDimension) { _ = try Operations.umap([values[0]], dimensions: 1, config: rejected) }
            expectKind(.invalidDimension) { _ = try Operations.umap(graph: one, dimensions: 1, config: rejected) }
            for dimension in [0, Int.max] {
                expectKind(.invalidDimension) { _ = try Operations.umap(values, dimensions: dimension, graph: g, config: rejected) }
                expectKind(.invalidDimension) { _ = try Operations.umap(graph: g, dimensions: dimension, config: rejected) }
            }
            var invalidSchedule = rejected
            invalidSchedule.epochs = UMAPChecked.maximumScheduleCount
            invalidSchedule.negativeSampleRate = 2
            expectKind(.invalidOperation) { _ = try Operations.umap(values, dimensions: 1, graph: g, config: invalidSchedule) }
            expectKind(.invalidOperation) { _ = try Operations.umap(graph: g, dimensions: 1, config: invalidSchedule) }
            expectKind(.dimensionMismatch) { _ = try Operations.umap(values, dimensions: 1, graph: two, config: rejected) }
            for initial: [[Float]] in [[[0]], [[.nan], [1], [2, 3]]] {
                expectKind(.dimensionMismatch) {
                    _ = try Operations.umap(graph: g, dimensions: 1, initialCoordinates: initial, config: rejected)
                }
            }
            for value: Float in [.nan, .infinity, -.infinity] {
                expectKind(.invalidData) {
                    _ = try Operations.umap(graph: g, dimensions: 1, initialCoordinates: [[0], [value], [2]], config: rejected)
                }
            }
        }
    }
}

private final class LC5BorrowCount: @unchecked Sendable {
    private let lock = NSLock()
    private var value = 0
    var count: Int { lock.lock(); defer { lock.unlock() }; return value }
    func record() { lock.lock(); defer { lock.unlock() }; value += 1 }
    func reset() { lock.lock(); defer { lock.unlock() }; value = 0 }
}

private struct LC5AccessVector: VectorProtocol {
    static let borrows = LC5BorrowCount()
    var storage: [Float]
    var scalarCount: Int { storage.count }
    subscript(index: Int) -> Float { storage[index] }
    init() { storage = [0] }
    init(_ array: [Float]) { storage = array }
    init(repeating value: Float) { storage = [value] }
    func toArray() -> [Float] { storage }
    func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        Self.borrows.record()
        return try storage.withUnsafeBufferPointer(body)
    }
    mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeMutableBufferPointer { try body($0) }
    }
}

private struct LC5ProviderReached: Error { }
private struct LC5ThrowingProvider: LinearAlgebraProvider {
    func qrThin(_ a: [Float], rows: Int, columns: Int) throws -> QRFactorization { throw LC5ProviderReached() }
    func svdThin(_ a: [Float], rows: Int, columns: Int) throws -> SingularValueDecomposition { throw LC5ProviderReached() }
    func symmetricEigen(_ a: [Float], dimension: Int, computeEigenvectors: Bool) throws -> SymmetricEigenDecomposition {
        throw LC5ProviderReached()
    }
}
