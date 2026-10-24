# N03 implementation evidence — final frozen v1

Base commit: `1c55a85b46f026fbcad8d27face1159000be4501`. Binding brief SHA256: `976f77a65c56db104c7187b98cb4d6404ef8d1cf0cf7834d3543acd8870637a5`.

## Implementation and scope

The public async BatchOperations.statistics route now shares MagnitudeStatisticsAccumulator with its internal serial path, internal parallel partition helper, and internal SyncBatchOperations.statistics. Internal Array.batchStatistics forwarding and finiteStatistics consume that synchronous route unchanged. The quantity remains population statistics of Float magnitudes actually delivered by the generic magnitude expression; this is not a component-statistics operation or a new magnitude dispatch contract.

The accumulator widens each measurement exactly to Double. Its update uses delta, mean += delta / newCount, and M2 += delta² * oldCount / newCount. Its merge uses Chan's delta between partial means and widened count ratios. Empty partials are identities, integer count increments/sums are checked, and final division/square root precede Float projection. Identical finite measurements have zero deltas and exact zero deviation. Finite Float differences and sample-count bounds fit the brief's Double range argument; no universal rounding-error guarantee is claimed.

Any invalid measurement marks an absorbing invalid state, but every subsequent magnitude expression is still evaluated once and every vector still contributes to count. Each source loop contains exactly one generic vector.magnitude expression. No shadow magnitude property or fabricated protocol witness is used in tests. Partial results carry their original chunk index and are sorted before merging. The explicit positive-chunk helper is production-used; its endpoint is start + min(chunkSize, remaining), so Int.max chunks do not overflow. No global setting is mutated or newly consumed, and the existing threshold and optimal chunk selection remain unchanged.

Exclusive changes are the two statistics-only existing source files, a new internal accumulator, the new test suite, and N03-prefixed fixture generator/data. No E03 transaction, N01/N02 numerical producer, component-summary statistics, configuration gate, Package, sibling or index changes. `N03-archive-integrity-v1.json` checks all 2,301 tracked base files with none missing and no unexpected changes. The full seven-file diff, hashes and canonical source backup are `N03-review-v1.patch`, `N03-review-v1-manifest.json` and `.stabilization-recovery/2026-09-23/N03/v1`.

## Runtime RED and validation

Exact command, source-hash association, exit and full log hashes are in `N03-command-evidence-final.json`. All builds used stable source bytes; no edit occurred while a compiler or test was running.

- RED setup attempt 1 exited 1 at compilation because Testing's macro could not handle an allSatisfy key-path expression as nonthrowing. The explicit closure correction was made only after the build exited. The setup failure is preserved separately and is not behavioral RED.
- Corrected RED attempt 2 ran five tests and exited 1 with 161 issues. It demonstrates the async threshold discontinuity at 999/1000/1024, repeated-value error, overflow/underflow and invalid-result policy failures on the unchanged production base.
- Initial GREEN ran eight tests and exited 0. Two additional assertions then made empty positive-zero mean and nonzero tiny deviation explicit.
- Final frozen Debug ran eight tests and exited 0.
- Final frozen Release ran eight tests and exited 0.

Repeated inventory on the accepted base found no pre-existing tests invoking BatchOperations.statistics, SyncBatchOperations.statistics, batchStatistics, finiteStatistics or findOutliers. Accordingly the final focused filter is exactly BatchStatisticsRangeTests; no unrelated suite is represented as an existing statistics consumer. No raw memory operations or child probes were introduced, so ASan and child-process harnesses were unnecessary.

## Independent reference and coverage

The checked-in generator `Scripts/fixtures/N03-generate-references.py` decodes Float bits into exact Decimal binary values, computes a two-pass mean and centered deviations at precision 200, takes Decimal square root, then selects the nearest Float using exact Decimal distances and even-bit ties. It does not use Welford/Chan. Its JSON records every input bit pattern, Decimal result and rounded output bits. The Swift fixtures cover maximum range, actual neighboring Float values near 1e18, tiny subnormals, mixed exponents, a small outlier and normal-boundary neighbors. Root independently regenerated the two output files byte-for-byte in the reconstructed archive.

Alternating 10000/10001 fixtures additionally use an independent two-point closed form: mean = 10000 + nB/N; population deviation = sqrt(Double(nA)*Double(nB))/N. The public async path is exercised below threshold at 999 and at/above threshold at 1000/1024. Tests never manufacture distinct 1e18+1 Float measurements.

Eight functions cover:

- Threshold boundaries; identical magnitudes at zero, least subnormal, normal-boundary neighbors, 1, actual Float(1e18), maxFinite; empty and singleton; mixed row dimensions and zero-dimensional rows; ordinary fixed and optimized 3/4 magnitude-5 controls.
- The same measurements through public async statistics and internal synchronous/Array forwarding routes, with explicit finite classification before numeric comparisons. Exact-zero deviation remains a bit-exact assertion; empty mean must be positive zero. Expected nonzero deviation must remain positive.
- A bounded 21-fixture partition matrix (seven size boundaries, six Decimal fixtures, eight identical values) × three orders (original, reversed, deterministic shuffled) × eight chunk selections (1/2/3/127/256/1024/larger-than-N/Int.max), with two independent actual task executions per selection: 1,008 helper calls. Each order also runs async, synchronous and Array routes. Repeated results must have identical bits for a fixed partition; no equivalence across every partition or permutation is claimed.
- Another 216 helper calls place NaN and signed infinities in first/middle/last positions over all eight chunks and three orders. Public async serial/parallel and internal synchronous/Array tests preserve full counts and return two NaNs. Finite-coordinate overflowing magnitudes are independently tested, including a 1,000-row public parallel case.
- finiteStatistics retains coordinate-finite overflowing rows and therefore produces two NaNs with the retained count; NaN/infinity coordinate rows remain filtered, and all-filtered input remains empty. The filtering predicate is unchanged.
- Direct accumulator tests cover signed nonfinite measurement inputs, empty merge identities, and all 95 split positions across the six Decimal fixtures. These supplement actual vector routes.

Bounded nonzero fixture tolerances are 16 ULPs of independently rounded expected values plus eight least-subnormal units. These are fixture acceptance criteria, not a universal precision promise.

## Migration and limits

A nonfinite delivered magnitude now yields count N and NaN for both numeric outputs on every statistics route. Previously some paths returned infinity, skipped meaningful invalid propagation through arithmetic, or clamped invalid/cancelled variance to zero. Coordinate-finite rows with overflowing norms are included by finiteStatistics and follow this new policy. Population denominator N, empty output, public threshold/chunk policy and unrelated outlier/component APIs are unchanged.

This task makes no component covariance, all-numerical-route closure, performance, RSS, downstream or production scale-readiness claim. Independent implementation review and root acceptance remain required.

## Final handoff

Complete diff SHA256: `cfc15ec332f833236e3ed636b753f397c7785b32a9f0633f08686200a9bd9904`. `N03-source-inventory-final.json` records all 261 current source/test/script/package file hashes. Final manifest verification matched all seven owned files. The full retained evidence is backed up under canonical `.stabilization-recovery/2026-09-23/N03/final-evidence`; `N03-evidence-inventory-final.json` hashes that handoff set excluding itself. Source has not changed since v1 freeze.
