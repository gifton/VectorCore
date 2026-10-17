# E05 implementation evidence — final frozen v1

Base: `a558253329933b186ba450b07a67a8a56cb4ca54`. Approved brief SHA256: `6c4b4f3a744aa2fc7c3bd230bdc5b1dd1ed64f9b42382e78204df9666ff19666`.

## State contract and scope

All public AutoTuning calls now use one internal State. Its NSLock protects every cache, reservation and accumulated-duration access. Immutable configuration and a Sendable monotonic-clock closure accompany that state. The unchecked Sendable conformance is attached to this production lock invariant; no unchecked test type or production-global mutable clock hook was introduced. Root explicitly approved the per-instance clock seam.

Admission validates dimension/cores before cache lookup, checks published cache, then in-flight reservation, then the completed-callback cap. A same-key in-flight request returns an unpublished default immediately. A new reservation is created under lock; clock reads, callback, calculation, formatting and output occur outside the lock. Success publishes the existing manual value or measured/fallback tuning, retires the reservation and charges elapsed time in one locked transition. Throw retires/charges and rethrows the identical error without caching a fallback. Manual set never retires a callback-owned reservation. The timer stops immediately after callback return/throw, before calculation. Logging receives a locked immutable total-time snapshot.

The 20 ms cap remains an admission safeguard based on the sum of completed callback durations. Overlap and nesting count individually; already admitted callbacks may overshoot, and a nonreturning callback has no deadline guarantee. The internal injected clock is documented as finite and monotonic and uses the same State implementation as all public calls.

Calculation validates finite positive sequential cost/effective factor, finite nonnegative overhead, finite intermediates and safe nonnegative integer conversions. Double(Int.max).nextDown is a safe conversion upper bound. The inefficient Int.max/2 sentinel is assigned as an integer directly, including large finite raw estimates. Bias, bounds, dimension scaling and 64-alignment retain the existing formulas where representable. Recoverable invalid input/intermediate returns the default through the ordinary publication transition, so manual overrides still win.

Only AutoTuning.swift changed in existing production. Operations.swift required no source migration because rethrows keeps its nonthrowing closure callable unchanged. No Q02 heuristics/key provenance, kernel routing, E02 scheduling, global configuration, Package, sibling or index changes. New files are the isolated child contract suite and supplied-value public compile fixture. Full v1 diff SHA256: `b10dda5a783949f5283b6a0d46580a76e3f33b4fcfddace46e3cbc417774a855`.

## Actual baseline RED

Both required baseline regressions ran before any production edit, from unchanged source throughout compilation/execution:

- Direct callback entered, then called get and same-key calibration. The baseline lock prevented progress at the reentrant get.
- Public Operations.findNearest used DynamicVector dimension 17, the default non-BatchKernelProvider provider and a custom metric. Empty tuner cache was asserted; the custom metric entered during uncached generic calibration and queried the tuner. Dynamic inputs and k equal to candidate count avoid optimized/fused bypasses. This actual public path deadlocked under the baseline lock.

Both parents observed body and callback-entry markers, reached the 15-second deadline and killed only their owned child; cleanup completed within two seconds. The baseline run exited 1 with two runtime issues, across three parent functions (two regression parents plus one intentional no-op child entry). Captures, exact child arguments/executable path/PID, deadlines and cleanup metadata are retained. There were no setup compilation failures in E05.

The original RED compiled test-bundle binary was replaced by the GREEN relink before its binary hash was recorded. This unavailable hash is not reconstructed or claimed. Root accepted the limited RED provenance: immutable source hashes, exact build/run association and actual entry/timeout/cleanup evidence. The original RED test source was later recovered byte-for-byte and verified against its recorded pre-run SHA256; it is retained as E05-red-test-source.swift. Final Debug/Release children additionally record the actual SwiftPM helper host executable SHA256. The --test-bundle-path argument identifies the loaded compiled test binary, whose separate SHA256 is retained in the configuration host record. Both frozen bundles and the helper host are backed up outside /tmp.

## Verification and covered routes

First GREEN: four parent functions passed; 19 isolated children exited normally. Final frozen Debug: eight functions in two suites passed (four E05 parent functions, including the intentional no-op child entry, plus four MetricDispatchContractTests functions). All 21 final E05 scenarios have body/result/completion markers and expected normal exit. Final frozen Release also passed eight functions in two suites with 21 successful E05 child scenarios. Seven narrow existing Operations search controls passed independently in both Debug and Release.

Every global tuner scenario runs in a fresh filtered process. Child environment controls disabled/bias/printing initialization; no environment is changed inside a running child. File output avoids blocking pipe reads. The termination handler is installed before process launch, execution is bounded at 15 seconds, owned-child kill cleanup at two seconds, and fixture start/release/completion gates at three seconds. A repaired-run timeout always fails. E05-child-associations-final.json links every retained capture/metadata file to its source revision and executable record; final metadata includes the executable hash directly. Host/toolchain and full source inventory are recorded separately.

The 21 scenarios are:

- Direct recursion and actual Operations custom-metric reentrancy. The latter asserts complete result count 3, indices [0, 2, 1], distances [5, 7, 9] against the independently predictable metric 3 + 2 times L1; cache publication is also checked.
- Cached and disabled zero-callback shortcuts, diagnostic printing, both environment biases, two-key recursive cycle, recursive manual set.
- Paused active callback with concurrent get and same-key request returning provisional default before release; manual override success, invalid-probe fallback and identical-error throw; unrelated keys both enter before either is released. Distinct manual tuple (321, 192, favorParallel) establishes precedence.
- Throw/retry with identity and no default cache on failure.
- Deterministic completed-time controls using per-instance production State: overlapping callbacks total 22 ms although each contributes 11; only the completed callback is charged during a third admission; failure contributes 21; nested callbacks contribute 13 + 7; a completed successful callback contributes 21. Each cap control verifies a fresh-key callback is skipped and its default published.
- Six invalid parameter pairs, checked both with manual cache and without cache, with no callbacks or new caching; four valid zero/extreme dimension/core controls. Twenty-two probe rows cover NaN, signed infinity, negative/zero fields, permitted zero overhead, inefficient cases, ordinary values and finite overflow paths. A fresh eligible key calibrates afterward. Cached fallback prevents another callback; reservation retirement on cached success/fallback is also directly inspectable in the single locked transition.

The public Swift 6 strict-concurrency fixture consumes supplied public CalibrationProbes/Tuning values, testing nonthrowing, rethrows, MainActor and actor-isolated forms without adding public constructors. Debug and Release typechecks each exited 0. This is compile evidence, not a downstream runtime integration claim.

Existing consumer inventory identified MetricDispatchContractTests metric facade controls, OperationsTests/testFindNearest* and TopKNaNWrapperTests/genericOperationsSelectsNumericScoresBeforeNaNs. The narrow latter controls ran separately in both configurations; no broader unrelated suite is required.

## Migration and limitations

Concurrent/recursive same-key calls can now receive provisional defaults while the initiating callback later returns measured or manual tuning; simultaneous results are not promised identical. Throwing calibration is newly supported with identical-error propagation and reservation cleanup. Invalid dimensions/cores bypass callbacks/cache shortcuts, and invalid probes/intermediates publish safe defaults. Existing nonthrowing call sites remain valid.

This closes the bounded API-29 reentrancy/publication contract only. It does not establish tuner measurement quality, key provenance, precision strategy accuracy, hard callback deadlines, throughput, downstream readiness or production-scale performance. The soft cap can overshoot by design. Independent review and root acceptance remain required.

## Final evidence and caveats

`E05-command-evidence-final.json` records exact invocations, exits, source association and full log hashes. An initial combined shell invocation of three supplemental controls exposed only its aggregate final status; those controls were repeated independently to record each exit 0. Both sets of logs are retained. This was evidence bookkeeping, not a test failure.

Evidence assembly also initially compared the helper-host hash with the loaded test-bundle hash and correctly rejected that mismatched identity. The final records distinguish both: `E05-debug-host.json` and `E05-release-host.json` record host and bundle paths/hashes, while child metadata supplies the exact helper path/hash and --test-bundle-path argument. All 42 final child metadata/capture associations were checked against the appropriate frozen host/bundle record. The 63 total retained children comprise two baseline RED, 19 initial GREEN, 21 final Debug and 21 final Release; the parent no-op child entry is not counted as a scenario.

All three owned files still match the frozen v1 manifest. The 263-file source inventory, complete three-file patch, original RED source/hashes, command logs, 63 child capture/metadata pairs and final host/bundle records are retained. `E05-evidence-inventory-final.json` hashes the worker handoff artifacts, excluding root-owned E05-root-* files and independent design/implementation review reports. Canonical source, test-bundle/host binaries and full worker evidence backups are under `.stabilization-recovery/2026-09-23/E05`. Independent source review accepted v1; final runtime-evidence/root acceptance remains the handoff step.
