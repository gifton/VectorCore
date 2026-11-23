/// Counts for one stage of a synchronous UMAP fit.
/// Successful event order and counts are exercised by
/// `UMAPProgressTests.stageOrderAndSkippedStages` and `graphReportsBlockBoundaries`.
///
/// Both UMAP routes check task cancellation even without a handler. A checkpoint
/// that observes cancellation throws `CancellationError`; an already-cancelled task is rejected before
/// validation (`UMAPCancellationTests.entryCancellationPrecedesWorkAndValidation`).
/// Checks occur between stages, graph blocks, epochs and preparation passes.
/// Packing, fuzzy-set construction, schedule preparation, and PCA/BLAS/LAPACK
/// calls finish before their next checkpoint; no wall-clock response bound is promised.
///
/// Handlers run synchronously on the calling thread and are not retained after
/// the call (`UMAPProgressConsumer.swift`). Cancellation from a handler is checked when it returns, including
/// a final completion event (`UMAPCancellationTests.cancellationFromOnlyEpochCompletionThrows`).
/// Completion reports work done; cancellation can still prevent returning a result.
public struct UMAPProgress: Sendable, Equatable {
    public enum Stage: Sendable, Equatable {
        /// Exhaustive neighbor construction; units are completed query rows.
        case graph
        /// PCA or random coordinate initialization; one unit.
        case initialization
        /// Fuzzy-set construction; one unit.
        case fuzzySet
        /// Layout optimization; units are completed epochs.
        case layout
    }

    public let stage: Stage
    public let completed: Int
    public let total: Int
}

/// Per-call observer; it owns no numerical state or borrowed storage.
internal struct UMAPExecutionContext {
    let progress: ((UMAPProgress) -> Void)?

    func checkCancellation() throws {
        try Task.checkCancellation()
    }

    func report(_ stage: UMAPProgress.Stage, completed: Int, total: Int) throws {
        try checkCancellation()
        progress?(UMAPProgress(stage: stage, completed: completed, total: total))
        try checkCancellation()
    }
}
