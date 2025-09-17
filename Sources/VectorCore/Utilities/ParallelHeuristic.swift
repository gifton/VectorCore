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
        // Calibrated overhead: ~60µs setup + ~1.3µs per item
        60_000.0 + 1_300.0 * Double(max(items, 0))
    }

    /// Decide whether to run in parallel based on dimension, items, variant, and metric
    static func shouldParallelize(dim: Int, items: Int, variant: Variant, metric: MetricClass, safetyFactor: Double = 1.15) -> Bool {
        guard dim > 0, items > 0 else { return false }
        let perItem = nsPerElement(variant: variant, metric: metric) * Double(dim)
        let total = perItem * Double(items)
        let overhead = parallelOverheadNs(items: items) * safetyFactor
        let decision = total > overhead
        debugLog(dim: dim, items: items, variant: variant, metric: metric, perItem: perItem, total: total, overhead: overhead, parallel: decision)
        return decision
    }

    // MARK: - Debug logging
    private static var debugEnabled: Bool {
        ProcessInfo.processInfo.environment["VECTORCORE_HEURISTIC_DEBUG"] == "1"
    }

    private static func debugLog(dim: Int, items: Int, variant: Variant, metric: MetricClass, perItem: Double, total: Double, overhead: Double, parallel: Bool) {
        guard debugEnabled else { return }
        let v = (variant == .optimized) ? "optimized" : "generic"
        let m: String = {
            switch metric { case .dot: return "dot"; case .euclideanLike: return "euclidean"; case .cosine: return "cosine"; case .manhattan: return "manhattan" }
        }()
        let msg = String(format: "[Heuristic] v=%@ m=%@ dim=%d items=%d perItem=%.1f ns total=%.1f ms overhead=%.1f ms -> %@",
                         v, m, dim, items, perItem, total/1e6, overhead/1e6, parallel ? "parallel" : "sequential")
        print(msg)
    }
}
