//
//  GramBatchSearch.swift
//  VectorCore
//

import Foundation

// The explicit tiled Gram route of `Operations.findNearestBatch` (`BatchSearchCPUStrategy.gram`, P02 phase C).
//
// One synchronous active query tile at a time, candidate tiles in increasing index order, scratch sized once by the
// checked `GramPlan` and reused, MatrixDistance's active-row helpers for packing, norms, normalization, the BLAS cross
// term and the postprocess (the arithmetic of the public MatrixDistance prepared/into entries, P02 RD9;
// `P02GramRouteTests.tileScoresMatchPublicPreparedEntries`), and one heap per active query row, drained into its result
// row with `TopKBuffer.drainWorstFirst` (`P02GramRouteTests.selectionMatchesSortedTileScores`).
//
// Swift 6.4.0 lost-store rules (swiftlang/swift#92562; P02 R-LS1 to R-LS5): every scratch array is created with
// `[Float](repeating:count:)`; the four scratch arrays are borrowed once, for the whole search, by nested
// `withUnsafeMutableBufferPointer` closures; every read of BLAS output (postprocess, heap pushes and the trace copy)
// happens inside those closures; the helpers take pointers by value, never `inout`; no scratch array is read after its
// closure, and the result is built inside the closures and returned from them. The search has no `await`, so no
// pointer crosses a suspension point.

extension Operations {

    // MARK: - Test seam

    /// What one `.gram` search did, in order (test seam, P02). Observation only: the seam receives copies made inside
    /// the scratch borrows and never changes the arithmetic, the scratch or the result. `gramTrace` is nil in ordinary
    /// execution and is read once per search, at its start.
    internal enum GramTraceEvent: Sendable {
        /// The four scratch arrays were allocated (once per search, right after the allocation). Counts are the actual
        /// element counts of the allocated arrays.
        case scratch(queryPackElements: Int, candidatePackElements: Int, scoreTileElements: Int, normElements: Int)
        /// The heaps and the result container were allocated (once per search, with no cancellation check between this
        /// and `.scratch`): one entry per allocated heap, read from that heap (`TopKBuffer.k`, the capacity its `vals`
        /// and `idxs` arrays are created with).
        case heaps(capacities: [Int])
        /// Query tile `tile` (`rows`) was packed, then normed (Euclidean) or normalized (cosine).
        case queryPack(tile: Int, rows: Range<Int>)
        /// Candidate tile `candidateTile` (`rows`) was packed for query tile `queryTile`, then normed or normalized.
        case candidatePack(queryTile: Int, candidateTile: Int, rows: Range<Int>)
        /// One BLAS call and its postprocess ran for this tile pair.
        case tile(GramTileRecord)
        /// The tile's active scores were pushed into the query rows' heaps.
        case selected(queryTile: Int, candidateTile: Int)
        /// The result row of query `query` was allocated and filled from its heap (`entries` entries).
        case row(query: Int, entries: Int)
    }

    /// One tile of a `.gram` search (see `GramTraceEvent.tile`).
    internal struct GramTileRecord: Sendable {
        let queryTile: Int
        let candidateTile: Int
        let queryRows: Range<Int>
        let candidateRows: Range<Int>
        /// The integer arguments of this tile's `gemmCrossTerm` call: M (active query rows), N (active candidate rows),
        /// K (the dimension) and the leading dimensions of the query pack, the candidate pack and the score tile. They
        /// are copied from the one `MatrixDistance.GemmArguments` binding (`blas`, built by
        /// `GramPlan.gemmArguments(activeQueryRows:activeCandidateRows:)` from the plan's checked `Int32` fields) that
        /// this route passes to `gemmCrossTerm` (P02 phase C review RA-1), not recomputed from the tile ranges, so a
        /// wrong value in that binding shows here (`P02GramRouteTests.packCountsAndTileOrder`). The seam does not
        /// observe the BLAS call itself. Lexical pin C8 (`Scripts/ci/tests/test_gram_route_lexical.py`) covers the hops
        /// it cannot see, as text: in this file the name `blas` has exactly one binding, `let blas =
        /// plan.gemmArguments(`, passed as `arguments: blas` to the one `gemmCrossTerm` call and as `blas: blas` to the
        /// one record call, with no other binding of the name (no `var`, nested `let`, `for … in`, closure parameter or
        /// pattern; P02 final review FA-1); and `gemmCrossTerm` forwards the six fields to `cblas_sgemm` in BLAS order.
        /// A wrong argument inside that forwarding is caught behaviourally only by the padding check of
        /// `packCountsAndTileOrder` and by `tileScoresMatchPublicPreparedEntries`, not by these fields (P02 phase D-F
        /// review A-1).
        let m: Int
        let n: Int
        let k: Int
        let lda: Int
        let ldb: Int
        let ldc: Int
        /// The postprocessed scores of the active `m × n` block, row-major with `n` columns: squared Euclidean
        /// distances (before the square root) or cosine distances.
        let scores: [Float]
        /// The score tile's element count and how many of its slots OUTSIDE the active block changed bits during this
        /// tile (packing, BLAS and postprocess).
        let scoreTileElements: Int
        let paddingSlotsChanged: Int
    }

    @TaskLocal internal static var gramTrace: (@Sendable (GramTraceEvent) -> Void)?

    // MARK: - Admission and planning

    /// The admitted `.gram` inputs: the caller's arrays as their concrete optimized type (`exactArray`, which shares
    /// the caller's buffers).
    enum GramInputs {
        case width512([Vector512Optimized], [Vector512Optimized])
        case width768([Vector768Optimized], [Vector768Optimized])
        case width1536([Vector1536Optimized], [Vector1536Optimized])
    }

    /// Step 2 of the `.gram` validation: `EuclideanDistance` or `CosineDistance` on `Vector512Optimized`,
    /// `Vector768Optimized` or `Vector1536Optimized`; anything else throws `.unsupportedOperation` naming the metric
    /// and vector types. It reads only types: no packing, BLAS call, heap or result is created here.
    static func gramAdmission<V: VectorProtocol, M: DistanceMetric>(
        queries: [V], candidates: [V], metric: M
    ) throws -> (admitted: GramInputs, euclidean: Bool) {
        let euclidean = metric is EuclideanDistance
        if euclidean || metric is CosineDistance {
            if let q = exactArray(queries, as: Vector512Optimized.self),
               let c = exactArray(candidates, as: Vector512Optimized.self) {
                return (.width512(q, c), euclidean)
            }
            if let q = exactArray(queries, as: Vector768Optimized.self),
               let c = exactArray(candidates, as: Vector768Optimized.self) {
                return (.width768(q, c), euclidean)
            }
            if let q = exactArray(queries, as: Vector1536Optimized.self),
               let c = exactArray(candidates, as: Vector1536Optimized.self) {
                return (.width1536(q, c), euclidean)
            }
        }
        throw VectorError(kind: .unsupportedOperation, context: ErrorContext(additionalInfo: [
            "strategy": "gram",
            "metric": String(describing: M.self),
            "vectorType": String(describing: V.self),
            "message": "BatchSearchCPUStrategy.gram supports EuclideanDistance and CosineDistance on Vector512Optimized, "
                + "Vector768Optimized and Vector1536Optimized only (got \(M.self) on \(V.self)); there is no fallback"
        ]))
    }

    /// The `.gram` route of `findNearestBatch` after its batch validation and provider check: field domains,
    /// admission and the checked plan (`GramPlan.make`), then the tiled search. Every error here precedes the first
    /// cancellation check, packing, BLAS call and heap or result allocation (`P02GramRouteTests.validationPrecedence`).
    static func gramBatch<V: VectorProtocol, M: DistanceMetric>( // swiftlint:disable:this function_parameter_count
        queries: [V], candidates: [V], k: Int, metric: M,
        queryTileRows: Int, candidateTileRows: Int, maximumTemporaryPayloadBytes: Int
    ) throws -> [[NearestNeighborResult]] {
        let (plan, inputs) = try GramPlan.make(
            queryCount: queries.count, candidateCount: candidates.count, dimension: candidates[0].scalarCount, k: k,
            queryTileRows: queryTileRows, candidateTileRows: candidateTileRows,
            maximumTemporaryPayloadBytes: maximumTemporaryPayloadBytes,
            admit: { try gramAdmission(queries: queries, candidates: candidates, metric: metric) })
        switch inputs {
        case let .width512(q, c): return try gramSearch(queries: q, candidates: c, plan: plan)
        case let .width768(q, c): return try gramSearch(queries: q, candidates: c, plan: plan)
        case let .width1536(q, c): return try gramSearch(queries: q, candidates: c, plan: plan)
        }
    }

    // MARK: - The tiled search

    /// Pointers into the four scratch arrays, valid only inside `gramSearch`'s borrows.
    private struct GramScratch {
        let queryPack: UnsafeMutablePointer<Float>
        let candidatePack: UnsafeMutablePointer<Float>
        let scores: UnsafeMutablePointer<Float>
        /// r query norms, then c candidate norms (Euclidean); nil for cosine, which counts no norms.
        let queryNorms: UnsafeMutablePointer<Float>?
        let candidateNorms: UnsafeMutablePointer<Float>?
        let scoreTileElements: Int
    }

    /// The search for a planned `.gram` request. Every row has `plan.dimension` Floats (the batch validation checked
    /// it). Allocates exactly the plan's scratch: `plan.queryPackElements`, `plan.candidatePackElements`,
    /// `plan.scoreTileElements` and `plan.normElements` Floats, and `plan.queryTileRows` heaps of capacity
    /// `plan.resultK`, besides the result.
    static func gramSearch<O: UnifiedVectorBuffer>(
        queries: [O], candidates: [O], plan: GramPlan
    ) throws -> [[NearestNeighborResult]] {
        let trace = gramTrace
        // Checkpoint: after the complete plan and limit, before any scratch or result allocation.
        try Task.checkCancellation()
        // swiftlang/swift#92562 (P02 R-LS4): scratch comes from [Float](repeating:count:), never an array literal.
        var queryPack = [Float](repeating: 0, count: plan.queryPackElements)
        var candidatePack = [Float](repeating: 0, count: plan.candidatePackElements)
        var scoreTile = [Float](repeating: 0, count: plan.scoreTileElements)
        var norms = [Float](repeating: 0, count: plan.normElements)
        trace?(.scratch(queryPackElements: queryPack.count, candidatePackElements: candidatePack.count,
                        scoreTileElements: scoreTile.count, normElements: norms.count))
        return try queryPack.withUnsafeMutableBufferPointer { qp in
            try candidatePack.withUnsafeMutableBufferPointer { cp in
                try scoreTile.withUnsafeMutableBufferPointer { st in
                    try norms.withUnsafeMutableBufferPointer { nb in
                        let euclideanNorms = plan.euclidean ? nb.baseAddress! : nil
                        let scratch = GramScratch(
                            queryPack: qp.baseAddress!, candidatePack: cp.baseAddress!, scores: st.baseAddress!,
                            queryNorms: euclideanNorms, candidateNorms: euclideanNorms.map { $0 + plan.queryTileRows },
                            scoreTileElements: st.count)
                        return try gramTiles(queries: queries, candidates: candidates, plan: plan, scratch: scratch,
                                             trace: trace)
                    }
                }
            }
        }
    }

    /// The tile loops (inside `gramSearch`'s borrows). Cancellation is checked at each query-tile boundary, at each
    /// candidate-tile boundary, after each tile's scores are computed, before each result row and before returning.
    private static func gramTiles<O: UnifiedVectorBuffer>(
        queries: [O], candidates: [O], plan: GramPlan, scratch: GramScratch,
        trace: (@Sendable (GramTraceEvent) -> Void)?
    ) throws -> [[NearestNeighborResult]] {
        let d = plan.dimension, ldc = plan.candidateTileRows, euclidean = plan.euclidean
        var heaps: [TopKBuffer] = []
        heaps.reserveCapacity(plan.queryTileRows)
        for _ in 0..<plan.queryTileRows {
            heaps.append(TopKBuffer(k: plan.resultK, isMinHeap: false, tieBreaker: .smallerIndex))
        }
        var result: [[NearestNeighborResult]] = []
        result.reserveCapacity(plan.queryCount)
        trace?(.heaps(capacities: heaps.map(\.k)))

        for queryTile in 0..<plan.queryTileCount {
            try Task.checkCancellation()   // query-tile boundary
            let queryRows = plan.queryTileRange(queryTile)
            let m = queryRows.count
            MatrixDistance.packRows(queries, rows: queryRows, d: d, into: scratch.queryPack)
            if let queryNorms = scratch.queryNorms {
                MatrixDistance.rowSumOfSquares(scratch.queryPack, rows: m, d: d, into: queryNorms)
            } else {
                MatrixDistance.normalizeRows(scratch.queryPack, rows: m, d: d)
            }
            trace?(.queryPack(tile: queryTile, rows: queryRows))

            for candidateTile in 0..<plan.candidateTileCount {
                try Task.checkCancellation()   // candidate-tile boundary
                let candidateRows = plan.candidateTileRange(candidateTile)
                let n = candidateRows.count
                MatrixDistance.packRows(candidates, rows: candidateRows, d: d, into: scratch.candidatePack)
                if let candidateNorms = scratch.candidateNorms {
                    MatrixDistance.rowSumOfSquares(scratch.candidatePack, rows: n, d: d, into: candidateNorms)
                } else {
                    MatrixDistance.normalizeRows(scratch.candidatePack, rows: n, d: d)
                }
                trace?(.candidatePack(queryTile: queryTile, candidateTile: candidateTile, rows: candidateRows))

                let before = trace == nil ? nil : Array(UnsafeBufferPointer(start: scratch.scores,
                                                                            count: scratch.scoreTileElements))
                // One value, from the plan's checked Int32 fields, for the BLAS call and the trace record (P02 phase C
                // review RA-1; final review C-M3).
                let blas = plan.gemmArguments(activeQueryRows: m, activeCandidateRows: n)
                MatrixDistance.gemmCrossTerm(x: scratch.queryPack, y: scratch.candidatePack, arguments: blas,
                                             alpha: euclidean ? -2 : 1, into: scratch.scores)
                if let queryNorms = scratch.queryNorms, let candidateNorms = scratch.candidateNorms {
                    MatrixDistance.euclideanPostprocess(scratch.scores, rows: m, columns: n, ldc: ldc,
                                                        xNorm: queryNorms, yNorm: candidateNorms)
                } else {
                    MatrixDistance.cosinePostprocess(scratch.scores, rows: m, columns: n, ldc: ldc)
                }
                if let trace, let before {
                    trace(.tile(gramTileRecord(scratch, before: before, queryTile: queryTile,
                                               candidateTile: candidateTile, queryRows: queryRows,
                                               candidateRows: candidateRows, blas: blas, ldc: ldc)))
                }
                try Task.checkCancellation()   // after the tile computation

                for i in 0..<m {
                    let scores = scratch.scores + i * ldc
                    for j in 0..<n {
                        heaps[i].pushIfBetter(val: scores[j], idx: candidateRows.lowerBound + j)
                    }
                }
                trace?(.selected(queryTile: queryTile, candidateTile: candidateTile))
            }

            for i in 0..<m {
                try Task.checkCancellation()   // before each result row
                var row = [NearestNeighborResult](repeating: NearestNeighborResult(index: -1, distance: .nan),
                                                  count: heaps[i].size)
                var slot = row.count
                // Worst first, so the row fills from its end: ascending, NaN last, ties by the smaller index; the
                // Euclidean root is taken after selection on squared scores (P02 RD16).
                heaps[i].drainWorstFirst { index, score in
                    slot -= 1
                    row[slot] = NearestNeighborResult(index: index, distance: euclidean ? gramRoot(score) : score)
                }
                trace?(.row(query: queryRows.lowerBound + i, entries: row.count))
                result.append(row)
            }
        }
        try Task.checkCancellation()   // before returning the full result
        return result
    }

    /// The returned Euclidean distance for a selected squared score: NaN stays NaN, a positive score returns its
    /// square root and every other score returns 0 (the formatting of the former implicit GEMM route).
    @inline(__always)
    static func gramRoot(_ squared: Float) -> Float {
        squared.isNaN ? squared : (squared > 0 ? squared.squareRoot() : 0)
    }

    /// The seam's copy of one tile, read inside the scratch borrow (P02 R-LS1, R-LS3). The BLAS fields come from `blas`,
    /// the value passed to `gemmCrossTerm` (lexical pin C8); the copied block and the padding count use the tile
    /// ranges and the planned `ldc`.
    private static func gramTileRecord( // swiftlint:disable:this function_parameter_count
        _ scratch: GramScratch, before: [Float], queryTile: Int, candidateTile: Int,
        queryRows: Range<Int>, candidateRows: Range<Int>, blas: MatrixDistance.GemmArguments, ldc: Int
    ) -> GramTileRecord {
        let m = queryRows.count, n = candidateRows.count
        var scores: [Float] = []
        scores.reserveCapacity(m * n)
        var changed = 0
        for slot in 0..<scratch.scoreTileElements {
            let (i, j) = slot.quotientAndRemainder(dividingBy: ldc)
            if i < m && j < n {
                scores.append(scratch.scores[slot])
            } else if scratch.scores[slot].bitPattern != before[slot].bitPattern {
                changed += 1
            }
        }
        return GramTileRecord(queryTile: queryTile, candidateTile: candidateTile, queryRows: queryRows,
                              candidateRows: candidateRows, m: Int(blas.m), n: Int(blas.n), k: Int(blas.k),
                              lda: Int(blas.lda), ldb: Int(blas.ldb), ldc: Int(blas.ldc),
                              scores: scores, scoreTileElements: scratch.scoreTileElements, paddingSlotsChanged: changed)
    }
}
