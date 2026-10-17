//
//  GramPlan.swift
//  VectorCore
//

/// Checked plan for the explicit tiled Gram route (`BatchSearchCPUStrategy.gram`) of `Operations.findNearestBatch`.
///
/// Pure count arithmetic: it reads no collection, allocates nothing proportional to its inputs and touches no
/// global state, so representability is tested with counts alone (`P02GramPlanTests`). `Operations.findNearestBatch`
/// plans every `.gram` request with `make(...)` before any CPU cancellation check, scratch or output allocation.
///
/// Validation precedence (`make(...)`; `P02GramPlanTests.precedence*`):
/// 1. Field domains, in declaration order: `queryTileRows > 0`, `candidateTileRows > 0`,
///    `maximumTemporaryPayloadBytes >= 0`; else `.invalidConfiguration` naming the field (`field`, `reason`,
///    `value`, `message`).
/// 2. Admission (supported metric and type), supplied by the caller as `admit`; its error propagates unchanged and
///    nothing later runs.
/// 3. Shape and representability: positive query count, candidate count, k and dimension; the dimension and the
///    clamped tile rows must fit the BLAS `Int32` arguments; every count, byte product and sum below must fit `Int`.
///    Else `.resourceUnavailable` with `reason: "not representable"`, the failing `field`, `requiredBytes: "not
///    representable"` and `allowedBytes`; the message says "must be positive" for a nonpositive count and "does not
///    fit" otherwise (`P02GramPlanTests.precedenceRepresentabilityBeforeLimit`). The positivity guards are defensive:
///    `Operations.findNearestBatch` rejects `k <= 0` and empty collections before it plans, and its admission (step 2)
///    accepts only 512-, 768- and 1536-wide rows, so they are not reached from that route.
/// 4. The limit: a logical temporary payload above `maximumTemporaryPayloadBytes` throws `.resourceUnavailable`
///    with `reason`, `requiredBytes`, `allowedBytes`, `boundary: "logical temporary payload"` and `message`.
///
/// With r = min(queryTileRows, Q), c = min(candidateTileRows, N), D the dimension and K = min(k, N), the logical
/// temporary payload is `Float.stride * (r*D + c*D + r*c + (euclidean ? r + c : 0)) + r*K*(Float.stride +
/// Int.stride)` (`P02GramPlanTests.payloadMatchesIndependentFormula`). It counts named element slots only: it
/// excludes scalar locals, container, runtime and allocator metadata and slack, BLAS-internal allocations, the
/// caller's inputs and the returned result, and it is not an allocator-byte, live-memory or resident-size limit.
/// The result (Q rows of K entries) is outside the limit but its element, byte and row-container counts must still
/// be representable.
///
/// Tiles are caller-selected: a plan that does not fit the limit is rejected, never shrunk. Clamping happens before
/// any product, so `Int.max` tile requests plan like the actual counts (`P02GramPlanTests.clampsBeforeProducts`).
/// Total Q and N need not fit `Int32`; only D, r and c (the BLAS dimensions and leading dimensions) must.
internal struct GramPlan: Equatable, Sendable {
    /// The name of the counted quantity, used in error context and messages.
    static let boundary = "logical temporary payload"

    // MARK: Inputs (tile rows and k clamped)

    let queryCount: Int
    let candidateCount: Int
    let dimension: Int
    let euclidean: Bool
    /// r = min(requested queryTileRows, queryCount).
    let queryTileRows: Int
    /// c = min(requested candidateTileRows, candidateCount); also the padded score-tile row stride.
    let candidateTileRows: Int
    /// K = min(k, candidateCount): the capacity of each heap and the length of each result row.
    let resultK: Int
    let maximumTemporaryPayloadBytes: Int

    // MARK: Logical temporary slots

    let queryPackElements: Int        // r * D
    let candidatePackElements: Int    // c * D
    let scoreTileElements: Int        // r * c
    let normElements: Int             // r + c (Euclidean), else 0
    let heapSlots: Int                // r * K, each one Float and one Int
    let floatPayloadBytes: Int        // Float.stride * (packs + tile + norms)
    let heapPayloadBytes: Int         // heapSlots * (Float.stride + Int.stride)
    let temporaryPayloadBytes: Int    // floatPayloadBytes + heapPayloadBytes

    // MARK: Tiles

    /// ceil(Q / r).
    let queryTileCount: Int
    /// ceil(N / c).
    let candidateTileCount: Int

    // MARK: Result (outside the limit; checked for representability)

    let resultElements: Int           // Q * K
    let resultBytes: Int              // Q * K * MemoryLayout<NearestNeighborResult>.stride
    let resultRowContainerBytes: Int  // Q * MemoryLayout<[NearestNeighborResult]>.stride
    let resultTotalBytes: Int         // resultBytes + resultRowContainerBytes

    // MARK: BLAS arguments (LP64 Int32), the source of every Gram BLAS integer (`gemmArguments`)

    let blasDimension: Int32               // D: the shared inner dimension and both pack leading dimensions
    let blasQueryTileRows: Int32           // r: the largest active query row count
    let blasCandidateTileRows: Int32       // c: the largest active candidate row count and the score leading dimension

    // MARK: Validation

    /// Step 1: the field domains, in declaration order.
    static func validateFields(queryTileRows: Int, candidateTileRows: Int, maximumTemporaryPayloadBytes: Int) throws {
        guard queryTileRows > 0 else { throw invalidField("queryTileRows", "must be positive", queryTileRows) }
        guard candidateTileRows > 0 else { throw invalidField("candidateTileRows", "must be positive", candidateTileRows) }
        guard maximumTemporaryPayloadBytes >= 0 else {
            throw invalidField("maximumTemporaryPayloadBytes", "must be nonnegative", maximumTemporaryPayloadBytes)
        }
    }

    /// Steps 1 to 4 in order: the field domains, then `admit` (which reports whether the admitted metric is
    /// Euclidean), then the plan. `admit` runs only when the fields are valid; the plan only after `admit` returns.
    static func make<Admitted>( // swiftlint:disable:this function_parameter_count
        queryCount: Int, candidateCount: Int, dimension: Int, k: Int,
        queryTileRows: Int, candidateTileRows: Int, maximumTemporaryPayloadBytes: Int,
        admit: () throws -> (admitted: Admitted, euclidean: Bool)
    ) throws -> (plan: GramPlan, admitted: Admitted) {
        try validateFields(queryTileRows: queryTileRows, candidateTileRows: candidateTileRows,
                           maximumTemporaryPayloadBytes: maximumTemporaryPayloadBytes)
        let admission = try admit()
        let plan = try GramPlan(queryCount: queryCount, candidateCount: candidateCount, dimension: dimension, k: k,
                                euclidean: admission.euclidean, queryTileRows: queryTileRows,
                                candidateTileRows: candidateTileRows,
                                maximumTemporaryPayloadBytes: maximumTemporaryPayloadBytes)
        return (plan, admission.admitted)
    }

    /// Steps 1, 3 and 4 (no admission). Every product and sum is checked; the first failing intermediate is named.
    init(queryCount: Int, candidateCount: Int, dimension: Int, k: Int, euclidean: Bool,
         queryTileRows requestedQueryTileRows: Int, candidateTileRows requestedCandidateTileRows: Int,
         maximumTemporaryPayloadBytes cap: Int) throws {
        try Self.validateFields(queryTileRows: requestedQueryTileRows, candidateTileRows: requestedCandidateTileRows,
                                maximumTemporaryPayloadBytes: cap)
        func unrepresentable(_ field: String) -> VectorError { Self.notRepresentable(field, allowed: cap) }
        func nonpositive(_ field: String) -> VectorError {
            Self.notRepresentable(field, allowed: cap, requirement: "must be positive")
        }
        func product(_ a: Int, _ b: Int, _ field: String) throws -> Int {
            let (value, overflow) = a.multipliedReportingOverflow(by: b)
            if overflow { throw unrepresentable(field) }
            return value
        }
        func sum(_ a: Int, _ b: Int, _ field: String) throws -> Int {
            let (value, overflow) = a.addingReportingOverflow(b)
            if overflow { throw unrepresentable(field) }
            return value
        }

        guard queryCount > 0 else { throw nonpositive("queryCount") }
        guard candidateCount > 0 else { throw nonpositive("candidateCount") }
        guard k > 0 else { throw nonpositive("k") }
        guard dimension > 0 else { throw nonpositive("dimension") }
        // Clamp before any product (Int clamps; no NaN is involved).
        let r = Swift.min(requestedQueryTileRows, queryCount)
        let c = Swift.min(requestedCandidateTileRows, candidateCount)
        let kk = Swift.min(k, candidateCount)
        guard let d32 = Int32(exactly: dimension) else { throw unrepresentable("dimension (BLAS Int32)") }
        guard let r32 = Int32(exactly: r) else { throw unrepresentable("queryTileRows (BLAS Int32)") }
        guard let c32 = Int32(exactly: c) else { throw unrepresentable("candidateTileRows (BLAS Int32)") }

        let queryPack = try product(r, dimension, "queryPackElements")
        let candidatePack = try product(c, dimension, "candidatePackElements")
        let tile = try product(r, c, "scoreTileElements")
        let norms = euclidean ? try sum(r, c, "normElements") : 0
        let packs = try sum(queryPack, candidatePack, "packElements")
        let packsAndTile = try sum(packs, tile, "packAndTileElements")
        let floats = try sum(packsAndTile, norms, "floatElements")
        let floatBytes = try product(floats, MemoryLayout<Float>.stride, "floatPayloadBytes")
        let slots = try product(r, kk, "heapSlots")
        let slotStride = try sum(MemoryLayout<Float>.stride, MemoryLayout<Int>.stride, "heapSlotStride")
        let heapBytes = try product(slots, slotStride, "heapPayloadBytes")
        let payload = try sum(floatBytes, heapBytes, "temporaryPayloadBytes")
        let results = try product(queryCount, kk, "resultElements")
        let resultBytes = try product(results, MemoryLayout<NearestNeighborResult>.stride, "resultBytes")
        let rowBytes = try product(queryCount, MemoryLayout<[NearestNeighborResult]>.stride, "resultRowContainerBytes")
        let resultTotal = try sum(resultBytes, rowBytes, "resultTotalBytes")

        guard payload <= cap else { throw Self.exceedsLimit(required: payload, allowed: cap) }

        self.queryCount = queryCount
        self.candidateCount = candidateCount
        self.dimension = dimension
        self.euclidean = euclidean
        self.queryTileRows = r
        self.candidateTileRows = c
        self.resultK = kk
        self.maximumTemporaryPayloadBytes = cap
        self.queryPackElements = queryPack
        self.candidatePackElements = candidatePack
        self.scoreTileElements = tile
        self.normElements = norms
        self.heapSlots = slots
        self.floatPayloadBytes = floatBytes
        self.heapPayloadBytes = heapBytes
        self.temporaryPayloadBytes = payload
        // (count - 1) / rows + 1 is ceil(count / rows) for count, rows > 0 and cannot overflow.
        self.queryTileCount = (queryCount - 1) / r + 1
        self.candidateTileCount = (candidateCount - 1) / c + 1
        self.resultElements = results
        self.resultBytes = resultBytes
        self.resultRowContainerBytes = rowBytes
        self.resultTotalBytes = resultTotal
        self.blasDimension = d32
        self.blasQueryTileRows = r32
        self.blasCandidateTileRows = c32
    }

    // MARK: BLAS arguments of one tile

    /// The `cblas_sgemm` arguments of a tile with `m` active query rows and `n` active candidate rows, built from this
    /// plan's checked `Int32` fields (P02 final review C-M3), so the plan is the one source of every BLAS integer: K,
    /// lda and ldb are `blasDimension` and ldc is `blasCandidateTileRows`; m and n must lie in
    /// `1...blasQueryTileRows` and `1...blasCandidateTileRows` (precondition), so their `Int32` values are exact.
    func gemmArguments(activeQueryRows m: Int, activeCandidateRows n: Int) -> MatrixDistance.GemmArguments {
        precondition(m >= 1 && m <= Int(blasQueryTileRows) && n >= 1 && n <= Int(blasCandidateTileRows),
                     "active tile rows outside the plan")
        return MatrixDistance.GemmArguments(m: Int32(truncatingIfNeeded: m), n: Int32(truncatingIfNeeded: n),
                                            k: blasDimension, lda: blasDimension, ldb: blasDimension,
                                            ldc: blasCandidateTileRows)
    }

    // MARK: Tile ranges

    /// The query rows of tile `tile`, in increasing order; the last tile may be partial.
    /// `start = tile * r <= Q - 1` because `tile <= (Q - 1) / r`, and `start + count <= Q`, so neither overflows
    /// (`P02GramPlanTests.tileRangesCoverInOrderWithoutOverflow`).
    func queryTileRange(_ tile: Int) -> Range<Int> {
        precondition(tile >= 0 && tile < queryTileCount, "query tile index out of range")
        let start = tile * queryTileRows
        return start..<(start + Swift.min(queryTileRows, queryCount - start))
    }

    /// The candidate rows of tile `tile`, in increasing order; the last tile may be partial (same bounds argument).
    func candidateTileRange(_ tile: Int) -> Range<Int> {
        precondition(tile >= 0 && tile < candidateTileCount, "candidate tile index out of range")
        let start = tile * candidateTileRows
        return start..<(start + Swift.min(candidateTileRows, candidateCount - start))
    }

    // MARK: Errors

    static func invalidField(_ field: String, _ requirement: String, _ value: Int) -> VectorError {
        VectorError(kind: .invalidConfiguration, context: ErrorContext(additionalInfo: [
            "field": field,
            "reason": requirement,
            "value": String(value),
            "message": "Invalid Gram strategy field \(field): \(requirement) (got \(value))"
        ]))
    }

    /// `requirement` names what the field fails: "does not fit Int or a BLAS Int32 argument" (the default) for an
    /// overflowing count, byte total or BLAS argument, or "must be positive" for a nonpositive count.
    static func notRepresentable(_ field: String, allowed: Int,
                                 requirement: String = "does not fit Int or a BLAS Int32 argument") -> VectorError {
        VectorError(kind: .resourceUnavailable, context: ErrorContext(additionalInfo: [
            "reason": "not representable",
            "field": field,
            "requiredBytes": "not representable",
            "allowedBytes": String(allowed),
            "boundary": boundary,
            "message": "Gram plan is not representable: \(field) \(requirement), so no \(boundary) can be computed"
        ]))
    }

    static func exceedsLimit(required: Int, allowed: Int) -> VectorError {
        VectorError(kind: .resourceUnavailable, context: ErrorContext(additionalInfo: [
            "reason": "exceeds maximumTemporaryPayloadBytes",
            "requiredBytes": String(required),
            "allowedBytes": String(allowed),
            "boundary": boundary,
            "message": "Gram plan needs \(required) bytes of \(boundary) (packs, norms, score tile and heaps), above "
                + "maximumTemporaryPayloadBytes \(allowed); this counts named element slots and is not a memory cap"
        ]))
    }
}
