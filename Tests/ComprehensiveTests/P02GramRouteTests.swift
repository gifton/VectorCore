import Foundation
import Testing
@testable import VectorCore

// P02 phase C: the explicit `.gram` route of `Operations.findNearestBatch` (phase 1 check map rows G1-G5, S6-S10,
// T2-T5, T8-T18, T34, T41-T43, E9-E19, A3, RD3, RD16), observed through the task-local seam `Operations.gramTrace`,
// which copies what the production tile code computed and never replaces it.

/// A custom metric that counts its calls (an unsupported `.gram` metric: admission must reject it without a call).
struct P02CountingMetric: DistanceMetric {
    typealias Scalar = Float
    final class Calls: @unchecked Sendable {
        private let lock = NSLock()
        private var value = 0
        func add() { lock.lock(); value += 1; lock.unlock() }
        var count: Int { lock.lock(); defer { lock.unlock() }; return value }
    }
    let calls: Calls
    var identifier: String { "p02-counting" }
    var name: String { "p02-counting" }
    func distance<V: VectorProtocol>(_ a: V, _ b: V) -> Float where V.Scalar == Float {
        calls.add()
        return EuclideanDistance().distance(a, b)
    }
}

@Suite("P02 explicit Gram route (.gram)")
struct P02GramRouteTests {

    /// Tile rows: unit tiles, nondivisible partial tiles, the former implicit shape, one tile, and a mix.
    static let geometries: [(r: Int, c: Int)] = [(1, 1), (3, 7), (8, 256), (Int.max, Int.max), (5, 100)]

    // MARK: E9, E11, G1, G4, T18: rows are the sorted emitted tile scores

    /// Every `.gram` row equals the sort, under the shared score law, of the scores the seam copied from the actual
    /// tiles (each pair scored exactly once), cut to min(k, N) and formatted. All widths, both metrics, four fixtures
    /// (near-duplicate cluster with an exact duplicate, uniform, offset-1000 and an exceptional set with zeros, NaN,
    /// ±inf and duplicates), five tile geometries, k = 10 and k = Int.max.
    @Test func selectionMatchesSortedTileScores() async throws {
        try await p02ForEachWidth { type in try await selection(type) }
    }

    private func selection<V: P02GramWidth>(_ type: V.Type) async throws {
        let w = V.p02Width
        let fixtures: [(String, [[Float]])] = [
            ("near", p02Rows(.near, width: w)), ("rand", p02Rows(.rand, width: w)),
            ("offset1000", p02Rows(.offset1000, width: w)), ("exceptional", p02ExceptionalRows(37, width: w, seed: 7))
        ]
        for (name, values) in fixtures {
            let candidates = try V.rows(values)
            let queries = Array(candidates.prefix(9))
            for (r, c) in Self.geometries {
                for k in [10, Int.max] {
                    try await check(queries, candidates, k: k, metric: EuclideanDistance(), euclidean: true,
                                    label: "w\(w) \(name) euclid r\(r) c\(c) k\(k)", strategy: p02Gram(r, c))
                    try await check(queries, candidates, k: k, metric: CosineDistance(), euclidean: false,
                                    label: "w\(w) \(name) cosine r\(r) c\(c) k\(k)", strategy: p02Gram(r, c))
                }
            }
        }
    }

    private func check<V: P02GramWidth, M: DistanceMetric>( // swiftlint:disable:this function_parameter_count
        _ queries: [V], _ candidates: [V], k: Int, metric: M, euclidean: Bool, label: String,
        strategy: BatchSearchCPUStrategy
    ) async throws where M.Scalar == Float {
        let (result, log) = await p02Traced(queries, candidates, k: k, metric: metric, strategy: strategy)
        let rows = try result.get()
        let expected = try #require(p02RowsFromTileScores(log.tiles, queries: queries.count,
                                                          candidates: candidates.count, k: k, euclidean: euclidean),
                                    "\(label)")
        #expect(rows.count == queries.count, "\(label)")
        #expect(rows.allSatisfy { $0.count == min(k, candidates.count) }, "\(label)")
        #expect(rows.map(p01Entries) == expected, "\(label)")
    }

    // MARK: S10, T16, T17: tile arithmetic is the public prepared/into arithmetic

    /// Each tile's scores equal, bit for bit, the public `MatrixDistance` prepared/into entry run on the same query and
    /// candidate rows (`prepare(_:normalized:)`, then `euclideanSquaredMatrix(queries:prepared:into:)` or
    /// `cosineDistanceMatrix(queries:prepared:into:)`): the Gram route reuses the public row-norm, normalization,
    /// postprocess order `(cross + queryNorm) + candidateNorm` and clamps, with active row counts on partial tiles.
    @Test func tileScoresMatchPublicPreparedEntries() async throws {
        try await p02ForEachWidth { type in try await publicParity(type) }
    }

    private func publicParity<V: P02GramWidth>(_ type: V.Type) async throws {
        let w = V.p02Width
        for (name, values) in [("near", Array(p02Rows(.near, width: w).prefix(40))),
                               ("exceptional", p02ExceptionalRows(40, width: w, seed: 11))] {
            let candidates = try V.rows(values)
            let queries = Array(candidates.prefix(7))
            for (r, c) in [(1, 1), (3, 7), (4, 13), (Int.max, Int.max), (7, 40)] {
                for euclidean in [true, false] {
                    let label = "w\(w) \(name) \(euclidean ? "euclid" : "cosine") r\(r) c\(c)"
                    let (result, log) = await p02TracedMetric(queries, candidates, k: 5, euclidean: euclidean, strategy: p02Gram(r, c))
                    _ = try result.get()
                    #expect(!log.tiles.isEmpty, "\(label)")
                    for tile in log.tiles {
                        let q = Array(queries[tile.queryRows]), cs = Array(candidates[tile.candidateRows])
                        var expected = [Float](repeating: 0, count: q.count * cs.count)
                        if euclidean {
                            MatrixDistance.euclideanSquaredMatrix(
                                queries: q, prepared: MatrixDistance.prepare(cs, normalized: false), into: &expected)
                        } else {
                            MatrixDistance.cosineDistanceMatrix(
                                queries: q, prepared: MatrixDistance.prepare(cs, normalized: true), into: &expected)
                        }
                        #expect(tile.scores.map(\.bitPattern) == expected.map(\.bitPattern),
                                "\(label) tile (\(tile.queryTile), \(tile.candidateTile))")
                    }
                }
            }
        }
    }

    // MARK: T11, T13, T14, T41-T43, S8: tile order, BLAS arguments, packs and padding

    /// The event sequence is the scratch allocation, the heap and result allocation, then for each query tile in increasing order: its pack, then for
    /// each candidate tile in increasing order its pack, one BLAS tile and its selection, then that tile's result rows.
    /// So there are ceil(Q/r) query packs and ceil(Q/r) * ceil(N/c) candidate packs (candidate tiles are packed again
    /// for every query tile). Each BLAS call has M and N equal to the ACTIVE rows (partial tiles included), K = lda =
    /// ldb = D (the whole dimension, no partial-D reduction) and ldc = the planned c, as recorded from the
    /// `MatrixDistance.GemmArguments` binding passed to `gemmCrossTerm` (lexical pin C8 pins that binding and its
    /// forwarding to `cblas_sgemm`); no score-tile slot outside the active block
    /// changes bits during a tile (the padding left by an earlier, larger tile is the sentinel).
    @Test func packCountsAndTileOrder() async throws {
        let shapes: [(q: Int, n: Int, r: Int, c: Int)] = [
            (9, 256, 4, 100), (3, 7, 2, 3), (1, 1, 1, 1), (8, 256, Int.max, Int.max), (5, 9, 1, 4), (7, 30, 3, 8)
        ]
        try await p02ForEachWidth { type in try await order(type, shapes) }
    }

    private func order<V: P02GramWidth>(_ type: V.Type, _ shapes: [(q: Int, n: Int, r: Int, c: Int)]) async throws {
        let d = V.p02Width
        for shape in shapes {
            let candidates = try V.rows(p02RandomRows(shape.n, width: d, seed: UInt64(shape.n * 31 + min(shape.c, 1_000))))
            let queries = Array(try V.rows(p02RandomRows(shape.q, width: d, seed: 99)))
            for euclidean in [true, false] {
                let label = "w\(d) Q\(shape.q) N\(shape.n) r\(shape.r) c\(shape.c) \(euclidean)"
                let (result, log) = await p02TracedMetric(queries, candidates, k: 3, euclidean: euclidean,
                                                          strategy: p02Gram(shape.r, shape.c))
                _ = try result.get()
                let r = min(shape.r, shape.q), c = min(shape.c, shape.n)
                let queryTiles = (shape.q + r - 1) / r, candidateTiles = (shape.n + c - 1) / c
                var expected = ["scratch", "heaps"]
                for qt in 0..<queryTiles {
                    expected.append("queryPack(\(qt))")
                    for ct in 0..<candidateTiles {
                        expected += ["candidatePack(\(qt),\(ct))", "tile(\(qt),\(ct))", "selected(\(qt),\(ct))"]
                    }
                    for q in (qt * r)..<min(shape.q, (qt + 1) * r) { expected.append("row(\(q))") }
                }
                let labels = log.snapshot.map(p02EventLabel)
                #expect(labels == expected, "\(label)")
                #expect(labels.filter { $0.hasPrefix("queryPack") }.count == queryTiles, "\(label)")
                #expect(labels.filter { $0.hasPrefix("candidatePack") }.count == queryTiles * candidateTiles, "\(label)")
                for tile in log.tiles {
                    let activeQ = min(r, shape.q - tile.queryTile * r), activeC = min(c, shape.n - tile.candidateTile * c)
                    #expect(tile.queryRows == (tile.queryTile * r)..<(tile.queryTile * r + activeQ), "\(label)")
                    #expect(tile.candidateRows == (tile.candidateTile * c)..<(tile.candidateTile * c + activeC), "\(label)")
                    #expect(tile.m == activeQ && tile.n == activeC, "\(label) tile (\(tile.queryTile), \(tile.candidateTile))")
                    #expect(tile.k == d && tile.lda == d && tile.ldb == d && tile.ldc == c, "\(label)")
                    #expect(tile.scoreTileElements == r * c && tile.scores.count == activeQ * activeC, "\(label)")
                    #expect(tile.paddingSlotsChanged == 0, "\(label) tile (\(tile.queryTile), \(tile.candidateTile))")
                }
            }
        }
    }

    // MARK: T12, T19, T20: the scratch is the plan

    /// The scratch the search allocates is exactly the planned logical slots, recomputed here from the brief's formula:
    /// r*D and c*D pack Floats, an r*c score tile, r + c Euclidean norms (none for cosine) and r heaps of capacity
    /// K = min(k, N); and it equals `GramPlan`'s counts for the same request.
    @Test func scratchMatchesThePlan() async throws {
        try await p02ForEachWidth { type in try await scratch(type) }
    }

    private func scratch<V: P02GramWidth>(_ type: V.Type) async throws {
        let d = V.p02Width
        let candidates = try V.rows(p02RandomRows(30, width: d, seed: 5))
        let queries = Array(candidates.prefix(7))
        for (r, c, k) in [(3, 8, 4), (Int.max, Int.max, Int.max), (1, 1, 1), (7, 30, 31), (2, 29, 30)] {
            for euclidean in [true, false] {
                let (result, log) = await p02TracedMetric(queries, candidates, k: k, euclidean: euclidean, strategy: p02Gram(r, c))
                _ = try result.get()
                let rr = min(r, 7), cc = min(c, 30), kk = min(k, 30)
                let plan = try GramPlan(queryCount: 7, candidateCount: 30, dimension: d, k: k, euclidean: euclidean,
                                        queryTileRows: r, candidateTileRows: c, maximumTemporaryPayloadBytes: Int.max)
                let events = log.snapshot
                guard events.count > 1, case let .scratch(qp, cp, st, nm) = events[0],
                      case let .heaps(capacities) = events[1] else {
                    Issue.record("w\(d) r\(r) c\(c): no scratch and heaps events first"); continue
                }
                // The capacities are read from the allocated heaps themselves (one entry per heap).
                let observed: [Int] = [qp, cp, st, nm, capacities.count] + capacities
                let formula: [Int] = [rr * d, cc * d, rr * cc, euclidean ? rr + cc : 0, rr]
                    + [Int](repeating: kk, count: rr)
                let planned: [Int] = [plan.queryPackElements, plan.candidatePackElements, plan.scoreTileElements,
                                      plan.normElements, plan.queryTileRows]
                    + [Int](repeating: plan.resultK, count: plan.queryTileRows)
                #expect(observed == formula, "w\(d) r\(r) c\(c) k\(k) \(euclidean)")
                #expect(observed == planned && capacities.reduce(0, +) == plan.heapSlots)
                #expect(events.filter { if case .scratch = $0 { return true } else { return false } }.count == 1)
                #expect(events.filter { if case .heaps = $0 { return true } else { return false } }.count == 1)
            }
        }
    }

    // MARK: T15: cosine normalization touches only the owned scratch

    /// The caller's query and candidate rows keep their exact bits after a cosine (and a Euclidean) `.gram` search.
    @Test func callerRowsAreUnchanged() async throws {
        try await p02ForEachWidth { type in
            func run<V: P02GramWidth>(_: V.Type) async throws {
                let values = p02ExceptionalRows(20, width: V.p02Width, seed: 3)
                let candidates = try V.rows(values)
                let queries = Array(candidates.prefix(5))
                _ = try await Operations.findNearestBatch(queries: queries, in: candidates, k: 4, metric: CosineDistance(),
                                                          cpuStrategy: p02Gram(2, 7))
                _ = try await Operations.findNearestBatch(queries: queries, in: candidates, k: 4,
                                                          cpuStrategy: p02Gram(2, 7))
                #expect(candidates.map { $0.toArray().map(\.bitPattern) } == values.map { $0.map(\.bitPattern) })
                #expect(queries.map { $0.toArray().map(\.bitPattern) } == values.prefix(5).map { $0.map(\.bitPattern) })
            }
            try await run(type)
        }
    }

    // MARK: G5, X8: serial and unscheduled

    /// `.gram` runs on the calling task: the installed scheduling provider is not asked for `parallelExecute` (or any
    /// work) and no bounded worker task is submitted; the default `.perQuery` control on the same provider is.
    @Test func serialAndUnscheduled() async throws {
        let all = try p02Vectors(p02Rows(.near, width: 512), as: Vector512Optimized.self)
        let queries = Array(all.prefix(8))
        for euclidean in [true, false] {
            let log = P02ProviderLog()
            let events = P02EventLog()
            let traces = P02GramTraceLog()
            let rows = try await BoundedExecution.$observation.withValue({ events.append($0) }, operation: {
                try await Operations.$gramTrace.withValue({ traces.append($0) }, operation: {
                    try await Operations.$computeProvider.withValue(P02ExecuteRecorder(log: log)) {
                        try await p02Batch(queries, all, k: 10, euclidean: euclidean, strategy: p02Gram(3, 64))
                    }
                })
            })
            #expect(rows.count == 8 && !traces.tiles.isEmpty)
            #expect(log.executeRanges.isEmpty, "parallelExecute ranges \(log.executeRanges)")
            #expect(events.snapshot.isEmpty, "bounded-execution events \(events.snapshot.count)")
            // Control: the default on the same provider fans out once over the 8 queries and traces no Gram tile.
            let controlTraces = P02GramTraceLog()
            _ = try await Operations.$gramTrace.withValue({ controlTraces.append($0) }, operation: {
                try await Operations.$computeProvider.withValue(P02ExecuteRecorder(log: log)) {
                    try await Operations.findNearestBatch(queries: queries, in: all, k: 10)
                }
            })
            #expect(log.executeRanges.first == 0..<8 && controlTraces.snapshot.isEmpty)
        }
    }

    // MARK: P4, T4, X14: the batch validation precedes every strategy check

    /// Batch errors (k <= 0, empty input, a late malformed candidate row, a query/candidate dimension mismatch) win over
    /// an invalid `.gram(0, 0, -1)` and an infeasible zero limit, and are the errors `.perQuery` throws for the same
    /// input (kind and context), also in an already-cancelled task.
    @Test func batchValidationPrecedesStrategyChecks() async throws {
        let good = try p02Vectors(p02RandomRows(30, width: 512, seed: 1), as: Vector512Optimized.self)
        let dynamic = p02RandomRows(30, width: 8, seed: 2).map { DynamicVector($0) }
        var lateRows = dynamic
        lateRows[29] = DynamicVector([Float](repeating: 1, count: 9))
        let late = lateRows
        let strategies: [BatchSearchCPUStrategy] = [
            .gram(queryTileRows: 0, candidateTileRows: 0, maximumTemporaryPayloadBytes: -1),
            .gram(queryTileRows: 1, candidateTileRows: 1, maximumTemporaryPayloadBytes: 0)
        ]
        for strategy in strategies {
            for cancelled in [false, true] {
                let label = "\(strategy) cancelled \(cancelled)"
                let empty512 = [Vector512Optimized]()
                try await same(label, cancelled, strategy) { s in
                    _ = try await Operations.findNearestBatch(queries: [good[0]], in: good, k: 0, cpuStrategy: s)
                }
                try await same(label, cancelled, strategy) { s in
                    _ = try await Operations.findNearestBatch(queries: empty512, in: good, k: 1, cpuStrategy: s)
                }
                try await same(label, cancelled, strategy) { s in
                    _ = try await Operations.findNearestBatch(queries: [good[0]], in: empty512, k: 1, cpuStrategy: s)
                }
                try await same(label, cancelled, strategy) { s in
                    _ = try await Operations.findNearestBatch(queries: [dynamic[0]], in: late, k: 1, cpuStrategy: s)
                }
                try await same(label, cancelled, strategy) { s in
                    _ = try await Operations.findNearestBatch(queries: [DynamicVector([Float](repeating: 0, count: 9))], in: dynamic,
                                                              k: 1, cpuStrategy: s)
                }
            }
        }
    }

    /// The error `call(strategy)` throws (in a task cancelled first when `cancelled`) equals the error `call(.perQuery)`
    /// throws (kind and context), and no Gram event is traced.
    private func same(_ label: String, _ cancelled: Bool, _ strategy: BatchSearchCPUStrategy,
                      _ call: @escaping @Sendable (BatchSearchCPUStrategy) async throws -> Void) async throws {
        let reference = await p02ErrorOf(cancelled: false) { try await call(.perQuery) }
        let log = P02GramTraceLog()
        let observed = await Operations.$gramTrace.withValue({ log.append($0) }, operation: {
            await p02ErrorOf(cancelled: cancelled) { try await call(strategy) }
        })
        #expect(reference != nil && observed?.kind == reference?.kind, "\(label): \(String(describing: observed))")
        #expect(observed?.context.additionalInfo == reference?.context.additionalInfo, "\(label)")
        #expect(log.snapshot.isEmpty, "\(label)")
    }

    // MARK: T1-T5, G2, G3, E13-E19: field domains, admission and the limit, in order, before any work

    /// `.gram` errors in their order, each before any Gram event, provider scheduling or metric call, and each the
    /// same in an already-cancelled task (every error precedes the first cancellation check):
    /// fields (declaration order) before admission (an unsupported type or metric) before the limit.
    @Test func validationPrecedence() async throws {
        let v512 = try p02Vectors(p02RandomRows(40, width: 512, seed: 4), as: Vector512Optimized.self)
        let v384 = try p02Vectors(p02RandomRows(40, width: 384, seed: 4), as: Vector384Optimized.self)
        let dynamic = p02RandomRows(40, width: 512, seed: 4).map { DynamicVector($0) }
        let generic = try p02RandomRows(40, width: 512, seed: 4).map { try Vector<Dim512>($0) }
        let calls = P02CountingMetric.Calls()
        let counting = P02CountingMetric(calls: calls)
        typealias Case = (label: String, strategy: BatchSearchCPUStrategy, kind: VectorError.ErrorKind, field: String?,
                          run: @Sendable (BatchSearchCPUStrategy) async throws -> Void)
        let q512 = Array(v512.prefix(3)), q384 = Array(v384.prefix(3)), qd = Array(dynamic.prefix(3))
        let qg = Array(generic.prefix(3))
        let euclid512: @Sendable (BatchSearchCPUStrategy) async throws -> Void = {
            _ = try await Operations.findNearestBatch(queries: q512, in: v512, k: 5, cpuStrategy: $0) }
        let cosine512: @Sendable (BatchSearchCPUStrategy) async throws -> Void = {
            _ = try await Operations.findNearestBatch(queries: q512, in: v512, k: 5, metric: CosineDistance(), cpuStrategy: $0) }
        let cases: [Case] = [
            ("fields r", .gram(queryTileRows: 0, candidateTileRows: 0, maximumTemporaryPayloadBytes: -1), .invalidConfiguration,
             "queryTileRows", {
                _ = try await Operations.findNearestBatch(queries: qd, in: dynamic, k: 5, metric: ManhattanDistance(),
                                                          cpuStrategy: $0) }),
            ("fields c", .gram(queryTileRows: 1, candidateTileRows: -3, maximumTemporaryPayloadBytes: -1), .invalidConfiguration,
             "candidateTileRows", { _ = try await Operations.findNearestBatch(queries: q384, in: v384, k: 5, cpuStrategy: $0) }),
            ("fields cap", .gram(queryTileRows: 1, candidateTileRows: 1, maximumTemporaryPayloadBytes: -1), .invalidConfiguration,
             "maximumTemporaryPayloadBytes", euclid512),
            ("dynamic", p02Gram(4, 8), .unsupportedOperation, nil, {
                _ = try await Operations.findNearestBatch(queries: qd, in: dynamic, k: 5, cpuStrategy: $0) }),
            ("vector384", .gram(queryTileRows: 4, candidateTileRows: 8, maximumTemporaryPayloadBytes: 0), .unsupportedOperation, nil, {
                _ = try await Operations.findNearestBatch(queries: q384, in: v384, k: 5, metric: CosineDistance(), cpuStrategy: $0) }),
            ("generic512", p02Gram(4, 8), .unsupportedOperation, nil, {
                _ = try await Operations.findNearestBatch(queries: qg, in: generic, k: 5, cpuStrategy: $0) }),
            ("manhattan512", .gram(queryTileRows: 4, candidateTileRows: 8, maximumTemporaryPayloadBytes: 0), .unsupportedOperation, nil, {
                _ = try await Operations.findNearestBatch(queries: q512, in: v512, k: 5, metric: ManhattanDistance(), cpuStrategy: $0) }),
            ("dot512", p02Gram(4, 8), .unsupportedOperation, nil, {
                _ = try await Operations.findNearestBatch(queries: q512, in: v512, k: 5, metric: DotProductDistance(), cpuStrategy: $0) }),
            ("custom512", .gram(queryTileRows: 4, candidateTileRows: 8, maximumTemporaryPayloadBytes: 0), .unsupportedOperation, nil, {
                _ = try await Operations.findNearestBatch(queries: q512, in: v512, k: 5, metric: counting, cpuStrategy: $0) }),
            ("limit euclid", .gram(queryTileRows: 4, candidateTileRows: 8, maximumTemporaryPayloadBytes: 0), .resourceUnavailable,
             nil, euclid512),
            ("limit cosine", .gram(queryTileRows: 4, candidateTileRows: 8, maximumTemporaryPayloadBytes: 1), .resourceUnavailable,
             nil, cosine512)
        ]
        for item in cases {
            for cancelled in [false, true] {
                let label = "\(item.label) cancelled \(cancelled)"
                let log = P02GramTraceLog()
                let providerLog = P02ProviderLog()
                let error = await Operations.$gramTrace.withValue({ log.append($0) }, operation: {
                    await Operations.$computeProvider.withValue(P02ExecuteRecorder(log: providerLog)) {
                        await p02ErrorOf(cancelled: cancelled) { try await item.run(item.strategy) }
                    }
                })
                #expect(error?.kind == item.kind, "\(label): \(String(describing: error))")
                if let field = item.field { #expect(error?.context.additionalInfo["field"] == field, "\(label)") }
                if item.kind == .unsupportedOperation {
                    #expect(error?.context.additionalInfo["strategy"] == "gram", "\(label)")
                }
                if item.kind == .resourceUnavailable {
                    #expect(error?.context.additionalInfo["boundary"] == "logical temporary payload", "\(label)")
                }
                #expect(log.snapshot.isEmpty, "\(label): \(log.snapshot.map(p02EventLabel))")
                #expect(providerLog.executeRanges.isEmpty, "\(label)")
            }
        }
        #expect(calls.count == 0, "the unsupported custom metric was called")
    }

    // MARK: T8, T9, T10, T34: the limit through the route; k clamping; the result is outside the limit

    /// At Q8/N256 with k = Int.max (K = N = 256 heaps slots per query row), a limit one byte below the exact planned
    /// payload throws `.resourceUnavailable` with the exact `requiredBytes`/`allowedBytes`, the boundary name and "not
    /// a memory cap", before any Gram event; the exact payload and one byte more succeed with the requested (clamped)
    /// tile rows, and the result still holds 8 rows of 256 entries (Q*N, outside the limit).
    @Test func limitBoundaryAndClampedK() async throws {
        let all = try p02Vectors(p02Rows(.rand, width: 512), as: Vector512Optimized.self)
        let queries = Array(all.prefix(8))
        for euclidean in [true, false] {
            for (r, c) in [(8, 256), (3, 100), (Int.max, Int.max)] {
                let plan = try GramPlan(queryCount: 8, candidateCount: 256, dimension: 512, k: Int.max, euclidean: euclidean,
                                        queryTileRows: r, candidateTileRows: c, maximumTemporaryPayloadBytes: Int.max)
                let required = plan.temporaryPayloadBytes
                let rr = min(r, 8), cc = min(c, 256)
                #expect(required == 4 * (rr * 512 + cc * 512 + rr * cc + (euclidean ? rr + cc : 0)) + rr * 256 * 12)
                let label = "\(euclidean) r\(r) c\(c)"
                let below = await p02TracedMetric(queries, all, k: Int.max, euclidean: euclidean,
                    strategy: .gram(queryTileRows: r, candidateTileRows: c, maximumTemporaryPayloadBytes: required - 1))
                guard case .failure(let failure) = below.result, let error = failure as? VectorError else {
                    Issue.record("\(label): required - 1 did not throw a VectorError"); continue
                }
                let info = error.context.additionalInfo
                #expect(error.kind == .resourceUnavailable && info["requiredBytes"] == String(required)
                        && info["allowedBytes"] == String(required - 1), "\(label): \(info)")
                #expect(info["message"]?.contains("logical temporary payload") == true
                        && info["message"]?.contains("not a memory cap") == true, "\(label)")
                #expect(below.log.snapshot.isEmpty, "\(label)")
                for cap in [required, required + 1] {
                    let ok = await p02TracedMetric(queries, all, k: Int.max, euclidean: euclidean,
                        strategy: .gram(queryTileRows: r, candidateTileRows: c, maximumTemporaryPayloadBytes: cap))
                    let rows = try ok.result.get()
                    #expect(rows.count == 8 && rows.allSatisfy { $0.count == 256 }, "\(label) cap \(cap)")
                    #expect(ok.log.tiles.allSatisfy { $0.m <= rr && $0.n <= cc } && ok.log.tiles.contains { $0.m == rr && $0.n == cc },
                            "\(label): tiles were resized")
                }
            }
        }
    }

    // MARK: RD16: equal roots keep the order of their squared scores

    /// Candidate 0 = (1, 1.25 * 2^-12, 0, ...) has squared distance fl(1 + 1.5625 * 2^-24) = 1 + 2^-23 from the zero
    /// query (one rounding: the other terms are exact zeros), candidate 1 = (1, 0, ...) has exactly 1, and both roots
    /// round to 1.0 (sqrt(1 + 2^-23) is below the midpoint 1 + 2^-24). `.gram` selects on the squared scores, so
    /// candidate 1 comes first although the returned distances are equal and its index is larger.
    @Test func equalRootsKeepSquaredOrder() async throws {
        var first = [Float](repeating: 0, count: 512), second = [Float](repeating: 0, count: 512)
        var third = [Float](repeating: 0, count: 512)
        first[0] = 1
        first[1] = 1.25 * 0x1p-12
        second[0] = 1
        third[0] = 3
        let candidates = try [first, second, third].map { try Vector512Optimized($0) }
        let query = try Vector512Optimized([Float](repeating: 0, count: 512))
        let (result, log) = await p02Traced([query], candidates, k: 3, metric: EuclideanDistance(), strategy: p02Gram(1, 3))
        let row = try result.get()[0]
        #expect(log.tiles.first?.scores.map(\.bitPattern) == [Float(1 + 0x1p-23).bitPattern, Float(1).bitPattern,
                                                               Float(9).bitPattern])
        #expect(row.map(\.index) == [1, 0, 2])
        #expect(row.map(\.distance.bitPattern) == [Float(1).bitPattern, Float(1).bitPattern, Float(3).bitPattern])
    }

    // MARK: RD3, A3: BatchOperations configuration does not affect findNearestBatch

    /// In a fresh child (the E06 isolation helper; the configuration is restored afterwards): with
    /// `enableMatrixRouting = false`, `.gram` still runs its tiles and returns the sorted tile scores; with it false and
    /// then true (and `matrixRoutingMinN = 1`), the default runs no Gram tile and every row equals `findNearest`.
    @Test func matrixRoutingFlagDoesNotAffectBatchSearch() async throws {
        try await e06IsolatedScenario("p02-matrix-routing-flag",
                                      filter: "P02GramRouteTests/matrixRoutingFlagDoesNotAffectBatchSearch") {
            let all = try p02Vectors(p02Rows(.near, width: 512), as: Vector512Optimized.self)
            let queries = Array(all.prefix(8))
            for (enabled, minimum) in [(false, 256), (true, 1)] {
                try await BatchOperations.updateConfiguration {
                    $0.enableMatrixRouting = enabled
                    $0.matrixRoutingMinN = minimum
                }
                let gram = await p02Traced(queries, all, k: 10, metric: EuclideanDistance(), strategy: p02Gram(8, 64))
                let rows = try gram.result.get()
                #expect(!gram.log.tiles.isEmpty, "enableMatrixRouting \(enabled): .gram ran no tile")
                #expect(rows.map(p01Entries) == p02RowsFromTileScores(gram.log.tiles, queries: 8, candidates: 256, k: 10,
                                                                       euclidean: true))
                let byDefault = await p02Traced(queries, all, k: 10, metric: EuclideanDistance(), strategy: .perQuery)
                #expect(byDefault.log.snapshot.isEmpty, "enableMatrixRouting \(enabled): the default ran Gram tiles")
                let differing = try await p02BatchRowsDifferingFromSingleQuery(queries: queries, candidates: all, k: 10,
                                                                               metric: EuclideanDistance())
                #expect(differing.isEmpty, "enableMatrixRouting \(enabled): \(differing.prefix(2))")
            }
        }
    }
}
