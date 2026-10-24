import Foundation
import Testing
@testable import VectorCore

// T15 test T1 (selection level): on every public cosine top-k and matrix route, a row with a NaN or an infinite
// coordinate scores NaN, ranks after every numeric row, and is never returned for k = 3.
//
// Fixture (the T15 phase 1 probe's): n rows of seeded random coordinates in [-1, 1); row 3 has a NaN coordinate,
// row 11 a +inf, row 19 a -inf, row 33 is all +inf and row 36 has both a +inf and a -inf coordinate; row 27 is the
// zero vector, which has cosine distance 1 to the finite nonzero query (one-zero rule). The query is finite.
// For each route and k in {3, n - 1, n} (where the route takes k):
//   (1) every NaN distance follows every numeric distance (the Top-K NaN contract);
//   (2) every special row that is returned has a NaN distance, and every other row a finite one;
//   (3) k = 3 returns none of the special rows;
//   (4) at k = n the last five entries are exactly rows 3, 11, 19, 33, 36 in that order (NaN ties by index),
//       and at k = n - 1 the last four are 3, 11, 19, 33.
// SimpleVectorCollection snapshots its rows from a Dictionary, so its tie order is unspecified: it is checked for
// (1)-(3) and for membership of the special rows in its last five entries only.
// The GEMM routes (findNearestBatch with the explicit `.gram` strategy since P02, pairwiseDistances at >= 256 vectors,
// MatrixDistance) already rank NaN rows last before T15 and are the control.

private let t15SpecialRows: [Int] = [3, 11, 19, 33, 36]
private let t15ZeroRow = 27

private struct T15LCG {
    var state: UInt64
    mutating func next() -> Float {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        let u = Float((state >> 40) & 0xFFFFFF) / Float(1 << 24)
        return u * 2 - 1
    }
}

private func t15Random(_ d: Int, seed: UInt64) -> [Float] {
    var generator = T15LCG(state: seed &* 0x9E3779B97F4A7C15 &+ 17)
    return (0..<d).map { _ in generator.next() }
}

/// The phase 1 probe's selection fixture.
private func t15SelectionRows(_ d: Int, n: Int) -> (query: [Float], rows: [[Float]]) {
    var rows = (0..<n).map { t15Random(d, seed: UInt64(100 + $0)) }
    rows[3][7] = .nan
    rows[11][7] = .infinity
    rows[19][7] = -.infinity
    rows[t15ZeroRow] = [Float](repeating: 0, count: d)
    rows[33] = [Float](repeating: .infinity, count: d)
    rows[36][7] = .infinity
    rows[36][9] = -.infinity
    return (t15Random(d, seed: 99), rows)
}

/// Checks (1)-(4) for one ranked result. `membershipOnly` relaxes (4) to membership (SimpleVectorCollection).
private func t15CheckRanked(_ route: String, n: Int, k: Int, _ pairs: [(index: Int, distance: Float)],
                            membershipOnly: Bool = false, sourceLocation: SourceLocation = #_sourceLocation) {
    let label = "\(route) n=\(n) k=\(k)"
    #expect(pairs.count == min(k, n), "\(label): returned \(pairs.count)", sourceLocation: sourceLocation)
    // (1) numeric before NaN
    let firstNaN = pairs.firstIndex { $0.distance.isNaN } ?? pairs.count
    let lastNumeric = pairs.lastIndex { !$0.distance.isNaN } ?? -1
    #expect(firstNaN > lastNumeric, "\(label): a NaN distance at rank \(firstNaN) precedes a numeric one at rank \(lastNumeric)",
            sourceLocation: sourceLocation)
    // (2) special rows score NaN; every other row is finite
    for (rank, pair) in pairs.enumerated() {
        if t15SpecialRows.contains(pair.index) {
            #expect(pair.distance.isNaN, "\(label): special row \(pair.index) at rank \(rank) scored \(pair.distance)",
                    sourceLocation: sourceLocation)
        } else {
            #expect(pair.distance.isFinite, "\(label): row \(pair.index) at rank \(rank) scored \(pair.distance)",
                    sourceLocation: sourceLocation)
        }
        if pair.index == t15ZeroRow {
            #expect(pair.distance == 1, "\(label): zero row scored \(pair.distance)", sourceLocation: sourceLocation)
        }
    }
    // (3) k = 3 returns no special row
    if k == 3 {
        let returned = pairs.map(\.index).filter { t15SpecialRows.contains($0) }
        #expect(returned.isEmpty, "\(label): special rows \(returned) returned for k = 3", sourceLocation: sourceLocation)
    }
    // (4) the tail is exactly the special rows, in index order
    let tailCount = k >= n ? t15SpecialRows.count : (k == n - 1 ? t15SpecialRows.count - 1 : 0)
    if tailCount > 0 && pairs.count >= tailCount {
        let tail = pairs.suffix(tailCount).map(\.index)
        let expected = Array(t15SpecialRows.prefix(tailCount))
        if membershipOnly {
            // Unspecified NaN tie order: at k = n - 1 any one special row may be the one left out.
            #expect(Set(tail).count == tailCount && Set(tail).isSubset(of: t15SpecialRows), "\(label): tail \(tail)",
                    sourceLocation: sourceLocation)
        } else {
            #expect(tail == expected, "\(label): tail \(tail), expected \(expected)", sourceLocation: sourceLocation)
        }
    }
}

/// Checks one row of a distance matrix against the fixture: special cells NaN, the zero cell 1, others finite in [0, 2].
private func t15CheckCells(_ route: String, _ cells: [Float], rows: Int, sourceLocation: SourceLocation = #_sourceLocation) {
    #expect(cells.count == rows, "\(route): \(cells.count) cells", sourceLocation: sourceLocation)
    for (index, value) in cells.enumerated() {
        if t15SpecialRows.contains(index) {
            #expect(value.isNaN, "\(route): special row \(index) scored \(value)", sourceLocation: sourceLocation)
        } else if index == t15ZeroRow {
            #expect(value == 1, "\(route): zero row scored \(value)", sourceLocation: sourceLocation)
        } else {
            #expect(value.isFinite && value >= 0 && value <= 2, "\(route): row \(index) scored \(value)",
                    sourceLocation: sourceLocation)
        }
    }
}

private func pairs(_ result: TopKResult) -> [(index: Int, distance: Float)] {
    Array(zip(result.indices, result.distances)).map { (index: $0.0, distance: $0.1) }
}

private func pairs(_ result: [NearestNeighborResult]) -> [(index: Int, distance: Float)] {
    result.map { (index: $0.index, distance: $0.distance) }
}

enum T15SelectionRoute: String, CaseIterable, Sendable, CustomStringConvertible {
    case operationsV512, operationsV768, operationsV1536, operationsV384, operationsDynamic, operationsGeneric512
    case operationsBatchPerQuery, operationsBatchGEMM
    case batchV512, batchV768, batchV1536, batchV512Parallel, batchV384, batchDynamic
    case topKNearestV512, topKNearestDynamic, topKBatchNearestV512, topKFused512, topKPreNormalized512
    case vectorMathWitness, vectorMathExtension
    case collectionSearchCosine, collectionSearchExistential
    case matrixOperationsDistanceMatrix, matrixPairwisePerPair, matrixPairwiseGEMM, matrixMatrixDistance
    var description: String { rawValue }

    enum Group { case singleQueryOperations, batchQueriesAndMatrices, batchOperations, topKSelection, vectorMathAndCollections }

    var group: Group {
        switch self {
        case .operationsV512, .operationsV768, .operationsV1536, .operationsV384, .operationsDynamic, .operationsGeneric512:
            return .singleQueryOperations
        case .operationsBatchPerQuery, .operationsBatchGEMM, .matrixOperationsDistanceMatrix, .matrixPairwisePerPair,
             .matrixPairwiseGEMM, .matrixMatrixDistance:
            return .batchQueriesAndMatrices
        case .batchV512, .batchV768, .batchV1536, .batchV512Parallel, .batchV384, .batchDynamic:
            return .batchOperations
        case .topKNearestV512, .topKNearestDynamic, .topKBatchNearestV512, .topKFused512, .topKPreNormalized512:
            return .topKSelection
        case .vectorMathWitness, .vectorMathExtension, .collectionSearchCosine, .collectionSearchExistential:
            return .vectorMathAndCollections
        }
    }
}

@Suite("T15 cosine selection: NaN and infinite rows rank last on every route")
struct T15CosineSelectionTests {
    static let n = 40
    static let ks = [3, 39, 40]

    @Test(arguments: T15SelectionRoute.allCases)
    func nonFiniteRowsRankLast(_ route: T15SelectionRoute) async throws {
        let n = Self.n
        switch route.group {
        case .singleQueryOperations: try await singleQueryOperations(route, n: n)
        case .batchQueriesAndMatrices: try await batchQueriesAndMatrices(route, n: n)
        case .batchOperations: try await batchOperations(route, n: n)
        case .topKSelection: try await topKSelection(route, n: n)
        case .vectorMathAndCollections: try await vectorMathAndCollections(route, n: n)
        }
    }

    private func singleQueryOperations(_ route: T15SelectionRoute, n: Int) async throws {
        switch route {
        case .operationsV512:
            let (q, rows) = t15SelectionRows(512, n: n)
            let query = try Vector512Optimized(q), candidates = try rows.map { try Vector512Optimized($0) }
            for k in Self.ks {   // k < n: TopK fused kernel; k = n: range-fused full-score route
                t15CheckRanked("Operations.findNearest<V512>", n: n, k: k,
                               pairs(try await Operations.findNearest(to: query, in: candidates, k: k, metric: CosineDistance())))
            }
        case .operationsV768:
            let (q, rows) = t15SelectionRows(768, n: n)
            let query = try Vector768Optimized(q), candidates = try rows.map { try Vector768Optimized($0) }
            for k in Self.ks {
                t15CheckRanked("Operations.findNearest<V768>", n: n, k: k,
                               pairs(try await Operations.findNearest(to: query, in: candidates, k: k, metric: CosineDistance())))
            }
        case .operationsV1536:
            let (q, rows) = t15SelectionRows(1536, n: n)
            let query = try Vector1536Optimized(q), candidates = try rows.map { try Vector1536Optimized($0) }
            for k in Self.ks {
                t15CheckRanked("Operations.findNearest<V1536>", n: n, k: k,
                               pairs(try await Operations.findNearest(to: query, in: candidates, k: k, metric: CosineDistance())))
            }
        case .operationsV384:
            let (q, rows) = t15SelectionRows(384, n: n)
            let query = try Vector384Optimized(q), candidates = try rows.map { try Vector384Optimized($0) }
            for k in Self.ks {   // small k streams, large k scores every row (generic witness)
                t15CheckRanked("Operations.findNearest<V384>", n: n, k: k,
                               pairs(try await Operations.findNearest(to: query, in: candidates, k: k, metric: CosineDistance())))
            }
        case .operationsDynamic:
            let (q, rows) = t15SelectionRows(512, n: n)
            let query = DynamicVector(q), candidates = rows.map { DynamicVector($0) }
            for k in Self.ks {
                t15CheckRanked("Operations.findNearest<DynamicVector>", n: n, k: k,
                               pairs(try await Operations.findNearest(to: query, in: candidates, k: k, metric: CosineDistance())))
            }
        case .operationsGeneric512:
            let (q, rows) = t15SelectionRows(512, n: n)
            let query = try Vector<Dim512>(q), candidates = try rows.map { try Vector<Dim512>($0) }
            for k in Self.ks {
                t15CheckRanked("Operations.findNearest<Vector<Dim512>>", n: n, k: k,
                               pairs(try await Operations.findNearest(to: query, in: candidates, k: k, metric: CosineDistance())))
            }
        default:
            Issue.record("route \(route) is not in the singleQueryOperations group")
        }
    }

    private func batchQueriesAndMatrices(_ route: T15SelectionRoute, n: Int) async throws {
        switch route {
        case .operationsBatchPerQuery:
            let (q, rows) = t15SelectionRows(512, n: n)
            let queries = try [Vector512Optimized(q), Vector512Optimized(t15Random(512, seed: 501))]
            let candidates = try rows.map { try Vector512Optimized($0) }
            for k in Self.ks {   // fewer than 8 queries: per-query findNearest
                let results = try await Operations.findNearestBatch(queries: queries, in: candidates, k: k, metric: CosineDistance())
                for (offset, result) in results.enumerated() {
                    t15CheckRanked("Operations.findNearestBatch.perQuery<V512>.q\(offset)", n: n, k: k, pairs(result))
                }
            }
        case .operationsBatchGEMM:
            let big = 256
            let (q, rows) = t15SelectionRows(512, n: big)
            let queries = try (0..<8).map { try Vector512Optimized($0 == 0 ? q : t15Random(512, seed: UInt64(500 + $0))) }
            let candidates = try rows.map { try Vector512Optimized($0) }
            for k in [3, big - 1, big] {   // 8 queries x 256 candidates: the explicit GEMM (`.gram`) route (control)
                let trace = P02GramTraceLog()
                let results = try await Operations.$gramTrace.withValue({ trace.append($0) }, operation: {
                    try await Operations.findNearestBatch(queries: queries, in: candidates, k: k, metric: CosineDistance(),
                                                          cpuStrategy: .gram(queryTileRows: 8, candidateTileRows: 256,
                                                                             maximumTemporaryPayloadBytes: Int.max))
                })
                #expect(trace.tiles.count == 1 && trace.tiles.allSatisfy { $0.m == 8 && $0.n == big && $0.k == 512 },
                        "operationsBatchGEMM k=\(k) did not run the GEMM tile")
                for (offset, result) in results.enumerated() {
                    t15CheckRanked("Operations.findNearestBatch.GEMM<V512>.q\(offset)", n: big, k: k, pairs(result))
                }
            }
        case .matrixOperationsDistanceMatrix:
            let (q, rows) = t15SelectionRows(512, n: n)
            let matrix = try await Operations.distanceMatrix(between: [try Vector512Optimized(q)],
                                                             and: try rows.map { try Vector512Optimized($0) },
                                                             metric: CosineDistance())
            t15CheckCells("Operations.distanceMatrix<V512>", matrix[0], rows: n)
        case .matrixPairwisePerPair:
            // 41 vectors: below the GEMM threshold, every cell from the metric witness
            let (q, rows) = t15SelectionRows(512, n: n)
            let vectors = try [Vector512Optimized(q)] + rows.map { try Vector512Optimized($0) }
            let matrix = await BatchOperations.pairwiseDistances(vectors, metric: CosineDistance())
            t15CheckCells("BatchOperations.pairwiseDistances(perPair)<V512>", Array(matrix[0].dropFirst()), rows: n)
        case .matrixPairwiseGEMM:
            let big = 256
            let (q, rows) = t15SelectionRows(512, n: big)
            let vectors = try [Vector512Optimized(q)] + rows.map { try Vector512Optimized($0) }
            let matrix = await BatchOperations.pairwiseDistances(vectors, metric: CosineDistance())
            t15CheckCells("BatchOperations.pairwiseDistances(GEMM)<V512>", Array(matrix[0].dropFirst()), rows: big)
        case .matrixMatrixDistance:
            let (q, rows) = t15SelectionRows(512, n: n)
            let cells = MatrixDistance.cosineDistanceMatrix(queries: [try Vector512Optimized(q)],
                                                            candidates: try rows.map { try Vector512Optimized($0) })
            t15CheckCells("MatrixDistance.cosineDistanceMatrix<V512>", cells, rows: n)
        default:
            Issue.record("route \(route) is not in the batchQueriesAndMatrices group")
        }
    }

    private func batchOperations(_ route: T15SelectionRoute, n: Int) async throws {
        switch route {
        case .batchV512:
            let (q, rows) = t15SelectionRows(512, n: n)
            let query = try Vector512Optimized(q), candidates = try rows.map { try Vector512Optimized($0) }
            for k in Self.ks {   // small k: tiled range-fused; large k: serial range-fused
                t15CheckRanked("BatchOperations.findNearest<V512>", n: n, k: k,
                               await BatchOperations.findNearest(to: query, in: candidates, k: k, metric: CosineDistance()))
            }
        case .batchV768:
            let (q, rows) = t15SelectionRows(768, n: n)
            let query = try Vector768Optimized(q), candidates = try rows.map { try Vector768Optimized($0) }
            for k in Self.ks {
                t15CheckRanked("BatchOperations.findNearest<V768>", n: n, k: k,
                               await BatchOperations.findNearest(to: query, in: candidates, k: k, metric: CosineDistance()))
            }
        case .batchV1536:
            let (q, rows) = t15SelectionRows(1536, n: n)
            let query = try Vector1536Optimized(q), candidates = try rows.map { try Vector1536Optimized($0) }
            for k in Self.ks {
                t15CheckRanked("BatchOperations.findNearest<V1536>", n: n, k: k,
                               await BatchOperations.findNearest(to: query, in: candidates, k: k, metric: CosineDistance()))
            }
        case .batchV512Parallel:
            let big = 1200   // at or above the default parallelThreshold (1000)
            let (q, rows) = t15SelectionRows(512, n: big)
            let query = try Vector512Optimized(q), candidates = try rows.map { try Vector512Optimized($0) }
            for k in [3, big - 1, big] {
                t15CheckRanked("BatchOperations.findNearest.n1200<V512>", n: big, k: k,
                               await BatchOperations.findNearest(to: query, in: candidates, k: k, metric: CosineDistance()))
            }
        case .batchV384:
            let (q, rows) = t15SelectionRows(384, n: n)
            let query = try Vector384Optimized(q), candidates = try rows.map { try Vector384Optimized($0) }
            for k in Self.ks {
                t15CheckRanked("BatchOperations.findNearest<V384>", n: n, k: k,
                               await BatchOperations.findNearest(to: query, in: candidates, k: k, metric: CosineDistance()))
            }
        case .batchDynamic:
            let (q, rows) = t15SelectionRows(512, n: n)
            let query = DynamicVector(q), candidates = rows.map { DynamicVector($0) }
            for k in Self.ks {
                t15CheckRanked("BatchOperations.findNearest<DynamicVector>", n: n, k: k,
                               await BatchOperations.findNearest(to: query, in: candidates, k: k, metric: CosineDistance()))
            }
        default:
            Issue.record("route \(route) is not in the batchOperations group")
        }
    }

    private func topKSelection(_ route: T15SelectionRoute, n: Int) async throws {
        switch route {
        case .topKNearestV512:
            let (q, rows) = t15SelectionRows(512, n: n)
            let query = try Vector512Optimized(q), candidates = try rows.map { try Vector512Optimized($0) }
            for k in Self.ks {
                t15CheckRanked("TopKSelection.nearest<V512>", n: n, k: k,
                               pairs(TopKSelection.nearest(k: k, query: query, candidates: candidates, metric: CosineDistance())))
            }
        case .topKNearestDynamic:
            let (q, rows) = t15SelectionRows(512, n: n)
            let query = DynamicVector(q), candidates = rows.map { DynamicVector($0) }
            for k in Self.ks {
                t15CheckRanked("TopKSelection.nearest<DynamicVector>", n: n, k: k,
                               pairs(TopKSelection.nearest(k: k, query: query, candidates: candidates, metric: CosineDistance())))
            }
        case .topKBatchNearestV512:
            let (q, rows) = t15SelectionRows(512, n: n)
            let queries = try [Vector512Optimized(q), Vector512Optimized(t15Random(512, seed: 502))]
            let candidates = try rows.map { try Vector512Optimized($0) }
            for k in Self.ks {
                let results = TopKSelection.batchNearest(k: k, queries: queries, candidates: candidates, metric: CosineDistance())
                for (offset, result) in results.enumerated() {
                    t15CheckRanked("TopKSelection.batchNearest<V512>.q\(offset)", n: n, k: k, pairs(result))
                }
            }
        case .topKFused512:
            let (q, rows) = t15SelectionRows(512, n: n)
            let query = try Vector512Optimized(q), candidates = try rows.map { try Vector512Optimized($0) }
            for k in Self.ks {
                t15CheckRanked("TopKSelection.nearestCosineFused512", n: n, k: k,
                               pairs(TopKSelection.nearestCosineFused512(k: k, query: query, candidates: candidates)))
            }
        case .topKPreNormalized512:
            // Rows that can be normalized are; the zero and non-finite rows are passed as they are.
            let (q, rows) = t15SelectionRows(512, n: n)
            let raw = try rows.map { try Vector512Optimized($0) }
            let candidates = raw.map { (try? $0.normalized().get()) ?? $0 }
            let query = try Vector512Optimized(q).normalized().get()
            for k in Self.ks {
                t15CheckRanked("TopKSelection.nearestCosinePreNormalized512", n: n, k: k,
                               pairs(TopKSelection.nearestCosinePreNormalized512(k: k, query: query, candidates: candidates)))
            }
        default:
            Issue.record("route \(route) is not in the topKSelection group")
        }
    }

    private func vectorMathAndCollections(_ route: T15SelectionRoute, n: Int) async throws {
        switch route {
        case .vectorMathWitness:
            let (q, rows) = t15SelectionRows(512, n: n)
            let query = try Vector<Dim512>(q), candidates = try rows.map { try Vector<Dim512>($0) }
            for k in Self.ks {
                t15CheckRanked("VectorMath.nearestNeighbors(using: CosineDistance().distance)", n: n, k: k,
                               VectorMath.nearestNeighbors(query: query, in: candidates, k: k,
                                                           using: { CosineDistance().distance($0, $1) }))
            }
        case .vectorMathExtension:
            let (q, rows) = t15SelectionRows(512, n: n)
            let query = try Vector<Dim512>(q), candidates = try rows.map { try Vector<Dim512>($0) }
            for k in Self.ks {
                t15CheckRanked("VectorMath.nearestNeighbors(using: cosineDistance(to:))", n: n, k: k,
                               VectorMath.nearestNeighbors(query: query, in: candidates, k: k,
                                                           using: { $0.cosineDistance(to: $1) }))
            }
        case .collectionSearchCosine, .collectionSearchExistential:
            let (q, rows) = t15SelectionRows(512, n: n)
            let query = try Vector512Optimized(q)
            let collection = SimpleVectorCollection<Vector512Optimized>(dimension: 512)
            for (index, row) in rows.enumerated() { try collection.add(id: index, vector: try Vector512Optimized(row)) }
            for k in Self.ks {
                let results = route == .collectionSearchCosine
                    ? collection.searchCosine(query: query, k: k)
                    : collection.search(query: query, k: k, metric: CosineDistance() as any DistanceMetric<Float>)
                t15CheckRanked("SimpleVectorCollection.\(route == .collectionSearchCosine ? "searchCosine" : "search(any)")",
                               n: n, k: k, results.results.map { (index: $0.id, distance: $0.distance) }, membershipOnly: true)
            }
        default:
            Issue.record("route \(route) is not in the vectorMathAndCollections group")
        }
    }
}
