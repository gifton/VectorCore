//
//  BatchSearchCPUStrategy.swift
//  VectorCore
//

/// How the CPU route of `Operations.findNearestBatch` searches a query batch when no `BatchKernelProvider` is
/// installed (its final parameter, `cpuStrategy`, default `.perQuery`).
///
/// The policy describes only Core's CPU route. An installed `BatchKernelProvider` receives the original whole batch,
/// `k` and metric before this policy is validated or planned, and the policy (including its payload limit) does not
/// constrain or bypass that provider (`P02BatchProviderTests.providerPrecedesInvalidAndInfeasibleCPUOptions`).
/// `BatchOperations.Configuration` does not select or disable either case.
public enum BatchSearchCPUStrategy: Sendable {
    /// Each query runs the single-query routes of `Operations.findNearest` (the reference for scores, ordering and
    /// returned score bits), after one validation of the whole batch, scheduled through the installed
    /// `ComputeProvider`'s `parallelExecute` (`P02DefaultRouteRedTests`, `P02BatchProviderTests`).
    case perQuery

    /// Explicit tiled Gram search: Euclidean or cosine distance on `Vector512Optimized`, `Vector768Optimized` or
    /// `Vector1536Optimized` rows, scored one query tile at a time with candidate tiles in increasing index order
    /// (`P02GramRouteTests.packCountsAndTileOrder`), on the calling task, without scheduling work on the
    /// `ComputeProvider` (`P02GramRouteTests.serialAndUnscheduled`). Any other metric or vector type throws
    /// `VectorError` `.unsupportedOperation`; there is no fallback to `.perQuery`
    /// (`P02GramRouteTests.validationPrecedence`).
    ///
    /// - `queryTileRows`: query rows per tile; must be positive. Values above the query count are clamped to it,
    ///   so `Int.max` requests one tile.
    /// - `candidateTileRows`: candidate rows per tile; must be positive. Values above the candidate count are
    ///   clamped to it.
    /// - `maximumTemporaryPayloadBytes`: must be nonnegative. A plan whose logical temporary payload exceeds it is
    ///   rejected with `VectorError` `.resourceUnavailable` carrying `requiredBytes` and `allowedBytes`; the tiles
    ///   are never shrunk to fit (`P02GramPlanTests.capBoundary`). Nonpositive tile rows or a negative limit are
    ///   rejected with `.invalidConfiguration` naming the field (`P02GramPlanTests.fieldDomains`).
    ///
    /// The logical temporary payload, with r and c the clamped tile rows, D the dimension and K = min(k, N), is
    /// `Float.stride * (r*D + c*D + r*c + (Euclidean ? r + c : 0)) + r*K*(Float.stride + Int.stride)`: the query and
    /// candidate packs, the score tile, the Euclidean row norms and r heaps of K entries, the scratch the search
    /// allocates (`P02GramRouteTests.scratchMatchesThePlan`). It excludes scalar locals, container, runtime and
    /// allocator metadata and slack, allocations inside BLAS, the caller's inputs and the returned result. It is a
    /// count of named element slots, not an allocator-byte, live-memory, resident-size or deadline limit. The returned
    /// result always holds Q rows of K entries, so k >= N returns Q*N entries outside this limit.
    /// (`P02GramPlanTests.payloadMatchesIndependentFormula` checks the formula.) Candidate tiles are packed again for
    /// every query tile: ceil(Q / r) query packs and ceil(Q / r) * ceil(N / c) candidate packs per search
    /// (`P02GramRouteTests.packCountsAndTileOrder`).
    ///
    /// Accuracy (a derived worst-case bound, P02 RD1). Let u = 2^-24, γ_n = n·u / (1 − n·u) and
    /// η = 2^-24 + 2^-50. The bound assumes that `cblas_sgemm` and `vDSP_svesq` evaluate each length-D inner product
    /// in binary32 operations in some order, with no fast-matrix-multiplication algorithm: an unverified contract about
    /// Accelerate, the one `CorrectnessPolicy` records for `MatrixDistance`.
    /// - Euclidean: provided no binary32 operation in the row norms, the cross term or the two additions overflows,
    ///   and no nonzero product x_k·y_k, x_k² or y_k² underflows (falls below 2^-126 in magnitude), each squared score
    ///   ĝ for rows x and y satisfies |ĝ − ‖x − y‖²| ≤ γ_{D+2}·(‖x‖² + ‖y‖² + 2·Σ|x_k·y_k|). The error does not shrink
    ///   with the distance: an exact duplicate of x can return a distance up to about 2·‖x‖·sqrt(γ_{D+2}) (about 0.011
    ///   at unit norm and D = 512). Outside that domain the bound does not hold:
    ///   - Underflow: products that underflow lose their value, so a pair at a tiny nonzero distance can score 0
    ///     (`P02GramAccuracyTests.underflowedSquaresScoreZero`: 512 coordinates of 1e-30 against a zero row score
    ///     exactly 0, where the true squared distance is about 5.1e-58).
    ///   - Overflow: when ‖q‖² or ‖c‖² overflows to +∞, the score is +∞ or NaN (∞ − ∞ in the cross term and norm
    ///     sum). An exact duplicate can then score NaN and rank LAST under the NaN-last order, where `.perQuery`
    ///     returns 0 and ranks it first (`P02GramAccuracyTests.overflowedNormsGiveNaNForAnExactDuplicate`: 512
    ///     coordinates of 1e20).
    /// - Cosine: on rows normalized by `NormalizationCore` (each coordinate within a relative η of x_k/‖x‖, or within
    ///   2^-150 when it rounds to a subnormal), each distance ĉ satisfies |ĉ − c| ≤ 2η + η² + γ_D·(1 + η)² +
    ///   u·(1 + (1 + γ_D)·(1 + η)²) + D·2^-140, the last term covering underflow.
    ///
    /// Clamping (at 0, or to [0, 2]) only moves a score toward its true value. `P02GramAccuracyTests` checks both bounds
    /// entry by entry against a Double reference on near-duplicate clusters, exact duplicates, the offset-1000 rows,
    /// uniform and unit-norm rows, at every width and at tile rows (1, 1), (3, 7), (8, 256) and (Int.max, Int.max).
    /// Smaller tiles do not repair this cancellation: each score is one full-length inner product (the dimension is
    /// never split across BLAS calls, `P02GramRouteTests.packCountsAndTileOrder` checks K = D on every tile), so the
    /// same bound applies at every tile geometry.
    /// Neighbors whose true squared distances differ by less than the sum of their bounds can therefore be reordered or
    /// left out; the route does no rescoring or reranking and is not equivalent to per-pair distances. The bound is a
    /// worst case: over those fixtures, the largest error that test measured on one host (Apple silicon, macOS 27, Swift
    /// 6.4; P02 phase C evidence, not pinned by a test) was 0.018 of the bound for Euclidean scores and 0.13 of it for
    /// cosine scores.
    ///
    /// Gram scores use a different arithmetic from the per-query routes: they are not guaranteed to be bit-identical to
    /// `Operations.findNearest`, and neither full-matrix nor tile-size-independent bit parity is promised.
    case gram(queryTileRows: Int, candidateTileRows: Int, maximumTemporaryPayloadBytes: Int)
}
