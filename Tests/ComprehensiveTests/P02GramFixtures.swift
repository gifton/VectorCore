import Foundation
import Testing
@testable import VectorCore

// P02 phase C shared support for the explicit Gram route tests (`P02GramRouteTests`, `P02GramCancellationTests`,
// `P02GramAccuracyTests`): the task-local trace capture, an independent sorted-selection oracle and fixtures.

/// Lock-protected list of the Gram seam's events (`Operations.gramTrace`).
final class P02GramTraceLog: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [Operations.GramTraceEvent] = []
    func append(_ event: Operations.GramTraceEvent) { lock.lock(); defer { lock.unlock() }; events.append(event) }
    var snapshot: [Operations.GramTraceEvent] { lock.lock(); defer { lock.unlock() }; return events }
    var tiles: [Operations.GramTileRecord] {
        snapshot.compactMap { if case .tile(let record) = $0 { return record } else { return nil } }
    }
}

/// A short label for an event, used to compare event sequences.
func p02EventLabel(_ event: Operations.GramTraceEvent) -> String {
    switch event {
    case .scratch: return "scratch"
    case .heaps: return "heaps"
    case .queryPack(let tile, _): return "queryPack(\(tile))"
    case .candidatePack(let q, let c, _): return "candidatePack(\(q),\(c))"
    case .tile(let record): return "tile(\(record.queryTile),\(record.candidateTile))"
    case .selected(let q, let c): return "selected(\(q),\(c))"
    case .row(let query, _): return "row(\(query))"
    }
}

/// Runs `findNearestBatch` with `strategy` under a fresh trace log; returns the result or error and the events.
func p02Traced<V: VectorProtocol & Sendable, M: DistanceMetric>(
    _ queries: [V], _ candidates: [V], k: Int, metric: M, strategy: BatchSearchCPUStrategy
) async -> (result: Result<[[NearestNeighborResult]], any Error>, log: P02GramTraceLog)
where V.Scalar == Float, M.Scalar == Float {
    let log = P02GramTraceLog()
    do {
        let rows = try await Operations.$gramTrace.withValue({ log.append($0) }, operation: {
            try await Operations.findNearestBatch(queries: queries, in: candidates, k: k, metric: metric,
                                                  cpuStrategy: strategy)
        })
        return (.success(rows), log)
    } catch {
        return (.failure(error), log)
    }
}

/// `.gram` with an unbounded payload limit.
func p02Gram(_ r: Int, _ c: Int) -> BatchSearchCPUStrategy {
    .gram(queryTileRows: r, candidateTileRows: c, maximumTemporaryPayloadBytes: Int.max)
}

/// The shared score law, restated independently of `TopKSelection` (not called): ascending, every NaN after every
/// number, exact ties (±0 equal, any two NaNs equal) by the smaller index.
func p02BestFirst(_ a: (index: Int, score: Float), _ b: (index: Int, score: Float)) -> Bool {
    switch (a.score.isNaN, b.score.isNaN) {
    case (false, true): return true
    case (true, false): return false
    case (true, true): return a.index < b.index
    case (false, false): return a.score < b.score || (a.score == b.score && a.index < b.index)
    }
}

/// Expected `.gram` rows from the scores the seam saw (E9): every (query, candidate) score must appear in exactly one
/// tile; each row is the sort of that query's scores under the shared law, cut to min(k, N), then formatted (Euclidean:
/// NaN stays, a positive score returns its square root, anything else 0). Returns nil and records an issue when the
/// tiles do not cover every pair exactly once.
func p02RowsFromTileScores(_ tiles: [Operations.GramTileRecord], queries: Int, candidates: Int, k: Int,
                           euclidean: Bool, sourceLocation: SourceLocation = #_sourceLocation) -> [[P01OpsEntry]]? {
    var scores = [[Float?]](repeating: [Float?](repeating: nil, count: candidates), count: queries)
    for tile in tiles {
        for (i, q) in tile.queryRows.enumerated() {
            for (j, c) in tile.candidateRows.enumerated() {
                if scores[q][c] != nil {
                    Issue.record("pair (\(q), \(c)) scored twice", sourceLocation: sourceLocation)
                    return nil
                }
                scores[q][c] = tile.scores[i * tile.n + j]
            }
        }
    }
    let kk = min(k, candidates)
    var rows: [[P01OpsEntry]] = []
    for q in 0..<queries {
        let all = scores[q].enumerated().compactMap { c, s in s.map { (index: c, score: $0) } }
        guard all.count == candidates else {
            Issue.record("query \(q): \(all.count) of \(candidates) pairs scored", sourceLocation: sourceLocation)
            return nil
        }
        rows.append(all.sorted(by: p02BestFirst).prefix(kk).map { entry in
            let value = euclidean ? (entry.score.isNaN ? entry.score : (entry.score > 0 ? entry.score.squareRoot() : 0))
                                  : entry.score
            return P01OpsEntry(index: entry.index, bits: value.bitPattern)
        })
    }
    return rows
}

/// Seeded rows in [-1, 1) (SplitMix64).
func p02RandomRows(_ count: Int, width: Int, seed: UInt64) -> [[Float]] {
    var random = P02SplitMix(state: seed)
    return (0..<count).map { _ in (0..<width).map { _ in random.unit() } }
}

/// Rows with exceptional values: a zero row, rows with a NaN, +inf or -inf coordinate, an all-NaN row, exact
/// duplicates (rows 6 and 7 equal row 5) and seeded rows elsewhere.
func p02ExceptionalRows(_ count: Int, width: Int, seed: UInt64) -> [[Float]] {
    var rows = p02RandomRows(count, width: width, seed: seed)
    if count > 0 { rows[0] = [Float](repeating: 0, count: width) }
    if count > 1 { rows[1][3] = .nan }
    if count > 2 { rows[2][5] = .infinity }
    if count > 3 { rows[3][7] = -.infinity }
    if count > 4 { rows[4] = [Float](repeating: .nan, count: width) }
    if count > 7 { rows[6] = rows[5]; rows[7] = rows[5] }
    return rows
}

/// Calls `body` once for each supported width with the matching optimized type.
func p02ForEachWidth(_ body: (any P02GramWidth.Type) async throws -> Void) async rethrows {
    try await body(Vector512Optimized.self)
    try await body(Vector768Optimized.self)
    try await body(Vector1536Optimized.self)
}

/// The three optimized types the `.gram` route admits.
protocol P02GramWidth: VectorProtocol, UnifiedVectorBuffer, Sendable where Scalar == Float {
    static var p02Width: Int { get }
}
extension Vector512Optimized: P02GramWidth { static var p02Width: Int { 512 } }
extension Vector768Optimized: P02GramWidth { static var p02Width: Int { 768 } }
extension Vector1536Optimized: P02GramWidth { static var p02Width: Int { 1536 } }

extension P02GramWidth {
    static func rows(_ values: [[Float]]) throws -> [Self] { try values.map { try Self($0) } }
}

/// `p02Traced` with `EuclideanDistance` (`euclidean`) or `CosineDistance`.
func p02TracedMetric<V: VectorProtocol & Sendable>(
    _ queries: [V], _ candidates: [V], k: Int, euclidean: Bool, strategy: BatchSearchCPUStrategy
) async -> (result: Result<[[NearestNeighborResult]], any Error>, log: P02GramTraceLog) where V.Scalar == Float {
    if euclidean {
        return await p02Traced(queries, candidates, k: k, metric: EuclideanDistance(), strategy: strategy)
    }
    return await p02Traced(queries, candidates, k: k, metric: CosineDistance(), strategy: strategy)
}

/// `findNearestBatch` with `EuclideanDistance` (`euclidean`) or `CosineDistance`.
func p02Batch<V: VectorProtocol & Sendable>(
    _ queries: [V], _ candidates: [V], k: Int, euclidean: Bool, strategy: BatchSearchCPUStrategy = .perQuery
) async throws -> [[NearestNeighborResult]] where V.Scalar == Float {
    if euclidean {
        return try await Operations.findNearestBatch(queries: queries, in: candidates, k: k, metric: EuclideanDistance(),
                                                     cpuStrategy: strategy)
    }
    return try await Operations.findNearestBatch(queries: queries, in: candidates, k: k, metric: CosineDistance(),
                                                 cpuStrategy: strategy)
}

/// Runs `body` in a new task (task-local values inherited), cancelled before `body` starts when `cancelled`, and returns
/// the `VectorError` it throws (nil when it returns or throws another error).
func p02ErrorOf(cancelled: Bool, _ body: @escaping @Sendable () async throws -> Void) async -> VectorError? {
    await p02AnyErrorOf(cancelled: cancelled, body) as? VectorError
}

/// As `p02ErrorOf`, returning any error.
func p02AnyErrorOf(cancelled: Bool, _ body: @escaping @Sendable () async throws -> Void) async -> (any Error)? {
    let task = Task {
        if cancelled { withUnsafeCurrentTask { $0?.cancel() } }
        try await body()
    }
    do {
        try await task.value
        return nil
    } catch {
        return error
    }
}
