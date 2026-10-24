# N03 independent design review

Verdict: **ACCEPT** for implementation under the current N03 brief. No numerical-design blocking gaps found. This is design acceptance, not implementation verification or NUM-06 closure.

Reviewed the N03 brief, execution POLICY, historical NUM-06, current async/serial and synchronous statistics bodies, Array.batchStatistics forwarding, finiteStatistics/filterFinite, generic VectorProtocol.magnitude, E03/N01 briefs, and the E06 work item. No production/test edits or builds were performed.

## Numerical contract and equations

The quantity is population statistics of the Float measurements actually delivered by the generic magnitude expression. This distinction is necessary: widening aggregation cannot recover rounding or overflow already incurred by magnitude. No component covariance, new protocol witness, or provider dispatch is needed. Single-coordinate DynamicVector fixtures make the delivered finite measurement independently known; ordinary [3,4] controls complement them.

For existing count n, mean u, and centered sum M2, inserting finite x uses d = Double(x) - u, u' = u + d/Double(n+1), and M2' = M2 + d*d*(Double(n)/Double(n+1)). The first value can initialize the mean directly. This is algebraically Welford's update and gives a nonnegative increment without relying on cancellation in x-u'. For two nonempty partials, d = uB-uA, n = nA+nB, u = uA+d*(Double(nB)/Double(n)), and M2 = M2A+M2B+d*d*Double(nA)*(Double(nB)/Double(n)) is Chan's centered merge. Empty partials must be identities before evaluating ratios. Divide M2 by Double(n), take the Double square root, and only then project both outputs to Float.

Identical finite Float inputs widen exactly and produce zero deltas, hence exactly zero M2 regardless of partition. This covers maxFinite, Float(1e18), and leastSubnormal without an approximate-zero allowance. Avoid raw second moments, Float-rounded means, and negative-variance clamps.

Finite Float differences have magnitude below 2^129. Squared differences and accumulation over at most Int.max samples remain below the brief's conservative 2^321 bound, comfortably within Double range. Even products ordered as d*d*Double(nA) remain representable; nevertheless use widened ratios/counts exactly as specified. Smallest distinct Float differences squared remain within Double's normal range. These are range arguments only; the brief correctly limits precision assertions to tested fixtures. Integer count increments/sums still require checked arithmetic or a documented derivation from disjoint ranges of one array. Double count rounding above 2^53 is not integer overflow and does not justify an all-count precision guarantee.

## Domain and partition behavior

Empty input returns count 0 and positive zeros. Singleton/identical finite input returns exact zero deviation. Any nonfinite delivered measurement makes both numeric outputs NaN while preserving total count. An invalid flag or equivalent absorbing state should not stop visiting later vectors: the brief requires every magnitude to be evaluated once. Invalid partials must propagate through every merge position, including singleton and uneven final chunks.

Current generic magnitude returns NaN for NaN coordinates and positive infinity for either sign of infinite coordinate. Thus -infinity coordinates cover the public generic nonfinite policy even though the delivered magnitude is positive infinity. An accumulator-only -infinity test can exercise the broader measurement-domain rule. Finite coordinates such as repeated maxFinite can also deliver infinity; that case must remain distinct from coordinate filtering.

finiteStatistics currently filters coordinate finiteness, then forwards to synchronous statistics. The N03 policy intentionally preserves retained-vector count and can return NaNs for retained finite-coordinate vectors with overflowing magnitude. All-filtered input remains the empty result. No predicate change or magnitude-finiteness filter is appropriate.

The production-used explicit positive-chunk helper is sufficient to test real partitions without activating configuration. The endpoint expression start + min(chunkSize, count-start) is bounded by count and handles Int.max. Avoid overflow-prone auxiliary ceiling division such as (count+chunkSize-1)/chunkSize. Maintain original chunk indices in child results and merge in index order; task completion order must not become reduction order. Determinism is for fixed partition/input, not equality across partition sizes or permutations. Child accumulators should be independent Sendable values; parent-owned collection/merge avoids shared mutable state or new unsafe storage.

## Regression and oracle adequacy

The brief's behavioral RED requirements reproduce both defect forms: alternating 10000/10001 at 999/1000/1024 and repeated actual Float(1e18). Shared synchronous implementation plus its Array forwarding closes the second occurrence of the aggregation defect rather than leaving a parallel API incorrect.

The bounded partition/order matrix is adequate when every listed chunk boundary is actually reached. Public tests must exercise 999 below and 1000/1024 at or above the current threshold, separately from helper tests. Include singleton partitions, uneven tails, larger-than-N and Int.max chunks, with invalid values in different chunk positions. Assertions must prove finite classifications before comparing finite expected values. Exact-zero checks remain separate from the 16-ULP plus eight-subnormal-unit fixture allowance.

Two-point weighted closed forms and independently generated two-pass Decimal references avoid reproducing the production recurrence as an oracle. Preserve input bits and rounded expected result bits, and perform count products in the wider reference. References for max-range, neighboring-offset, and tiny-range fixtures are particularly useful for detecting early Float projection or squaring. Mean/maxFinite classification requires an explicit assertion. For very tiny mathematically positive deviations, distinguish those representable in Float from those whose correct final projection is zero. ULP tolerances are fixture acceptance rules, not universal error bounds.

A custom concrete magnitude shadow cannot establish generic dispatch or call-count behavior because magnitude is an extension expression rather than a protocol requirement. The brief correctly prohibits that fake oracle. Source review plus actual generic public-route tests must establish reachability. No existing tests matching BatchOperations.statistics, SyncBatchOperations.statistics, batchStatistics(), or finiteStatistics() were found in the current Tests inventory; the implementer must repeat inventory on the accepted task base and accurately record any absence instead of claiming existing consumer suites ran.

## Ownership and prerequisite checks

E03's actor transaction design is independent of the centered reduction. N03 must preserve it and must not start reading live settings. E06 remains responsible for validating consumed settings, unsafe scheduling arithmetic, and re-deriving newly reachable routes; E02 retains broader task fan-out ownership. Current default threshold/chunk selection can remain unchanged while the statistics-specific helper safely exercises alternate partitions.

N01 changes numerical producers separately. N03 should summarize whatever the current generic expression delivers and retain concrete/generic controls without changing magnitude or asserting all norm routes are repaired. Public arbitrary row dimensions remain valid because each row contributes one measurement. NaNInfinityHandling is read-only for N03, preventing collision with N01's safeNormalized work.

Source-state observation: at review time this worktree still contains the pre-E03 read/modify/write updateConfiguration body. The brief already makes accepted E03 a prerequisite. Root must use the accepted integrated source state before N03 dispatch; this observation does not require changing the N03 numerical design. Reserve BatchOperations.swift against its other owners throughout implementation/review.

Final acceptance still requires the planned isolated behavioral RED, Debug/Release coverage, complete changed/new diff and hashes, retained oracle provenance, and independent implementation review. No throughput, memory envelope, broader numerical closure, or production readiness conclusion follows from this review.
