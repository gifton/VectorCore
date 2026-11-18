/// Ordinary probe budgets, centralized for consistent timeout diagnostics.
/// The 15-second deadline matches the existing ordinary 15-second callers and benchmark fixture;
/// cleanup retains their 2-second kill/reap allowance. No flake reduction is claimed.
/// Deliberate timeout self-tests, Q03 sanitizer probes and T16 profiler calibration supply explicit budgets instead.
package struct ChildProbeBudget: Sendable {
    package let deadline: Double
    package let cleanup: ChildProbeCleanup

    package static let standard = ChildProbeBudget(deadline: 15, cleanup: .kill(seconds: 2))
}
