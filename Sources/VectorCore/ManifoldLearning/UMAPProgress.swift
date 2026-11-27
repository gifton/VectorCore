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
/// Library-emitted counts have `total > 0` and `0 <= completed <= total`.
/// Within an emitted stage, total is constant, counts start at zero and increase
/// to total on success (`UMAPProgressTests.stageOrderAndSkippedStages`,
/// `graphReportsBlockBoundaries`, `layoutReportsEveryEpoch`). Unused stages emit
/// nothing; errors/cancellation may end the sequence early. Synthetic values
/// made with the public initializer are not required to satisfy these bounds.
public struct UMAPProgress: Sendable, Equatable {
    /// Closed four-case fit-stage set from retained revision a676d9745dec60450a2f1ff86da94824eb8b4f5e
    /// through pre-release 0.4.0 development and all 0.4.x. Placement cannot add
    /// cases to this enum in that interval. `UMAPProgressConsumer.swift` contains
    /// an exhaustive consumer switch. This is a source contract, not an ABI promise.
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

    /// Constructs a value for consumer mapping/tests, storing counts unchanged.
    /// No validation or clamping is performed (`UMAPProgressConsumer.swift`).
    /// The emitted-event bounds above apply only to real fit callbacks.
    public init(stage: Stage, completed: Int, total: Int) {
        self.stage = stage
        self.completed = completed
        self.total = total
    }
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
