/// Counts for one stage of a synchronous UMAP fit.
/// Successful event order and counts are exercised by
/// `UMAPProgressTests.stageOrderAndSkippedStages` and `graphReportsBlockBoundaries`.
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

    func report(_ stage: UMAPProgress.Stage, completed: Int, total: Int) throws {
        progress?(UMAPProgress(stage: stage, completed: completed, total: total))
    }
}
