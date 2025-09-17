import Foundation

/// Simple runtime heuristic to decide if parallelization is beneficial
/// for per-item workloads in Operations/BatchOperations.
enum ParallelHeuristic {
    enum Variant { case optimized, generic }
    enum MetricClass { case dot, euclideanLike, cosine, manhattan }

    // Estimated nanoseconds per element by variant & metric (coarse constants)
    static func nsPerElement(variant: Variant, metric: MetricClass) -> Double {
        switch (variant, metric) {
        case (.optimized, .dot), (.optimized, .euclideanLike): return 0.20
        case (.optimized, .cosine): return 0.30
        case (.optimized, .manhattan): return 0.35
        case (.generic, .dot): return 45.0
        case (.generic, .euclideanLike): return 55.0
        case (.generic, .cosine): return 50.0
        case (.generic, .manhattan): return 60.0
        }
    }

    // Parallel overhead model ~ constant setup + per-item overhead
    static func parallelOverheadNs(items: Int) -> Double {
        75_000.0 + 1_300.0 * Double(max(items, 0))
    }

    /// Decide whether to run in parallel based on dimension, items, variant, and metric
    static func shouldParallelize(dim: Int, items: Int, variant: Variant, metric: MetricClass, safetyFactor: Double = 1.25) -> Bool {
        guard dim > 0, items > 0 else { return false }
        let perItem = nsPerElement(variant: variant, metric: metric) * Double(dim)
        let total = perItem * Double(items)
        let overhead = parallelOverheadNs(items: items) * safetyFactor
        return total > overhead
    }
}

