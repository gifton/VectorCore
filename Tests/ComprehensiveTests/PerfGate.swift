import Foundation

/// Wall-clock performance gates are opt-in: they assert only when the
/// environment sets `VECTORCORE_STRICT_PERF=1` (e.g. a dedicated benchmark
/// CI job on quiet hardware). In normal runs — and especially under
/// ASan/TSan, whose 2–15× slowdowns make wall-clock thresholds meaningless —
/// the measurements are still printed but never fail the suite.
/// Inventory of gated sites: Docs/verification-baseline-0.3.1.md.
let strictPerfGatesEnabled: Bool = ProcessInfo.processInfo.environment["VECTORCORE_STRICT_PERF"] == "1"
