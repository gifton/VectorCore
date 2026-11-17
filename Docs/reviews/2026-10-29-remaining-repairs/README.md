# VectorCore: remaining audit repairs

F011, F012 and the two previously dormant FP16 comparison helpers are repaired and validated. This closes the remaining active repairs from the twelve-finding audit on retained platforms. F003 remains a historical defect on the retired watchOS target; it is not represented as fixed.

The changes remain on `codex/audit-top-six-2026-10-27` in `/Users/goftin/dev/projects/VectorCore-fixes-2026-10-27-84798d3`, unmerged and unpushed. This work changes benchmark tooling, consumer evidence tooling and tests. `Sources/` and `Package.swift` have no diff from this turn's baseline; no FP16 kernels or dimensions were added.

## Source and evidence

- Starting revision: `db9329540890f71b53e28f9eb591c1fb51ccf943`, following the [F007 repair](../2026-10-28-progress-repair/README.md).
- Implementation: `c69ce8f` (F012), `0110a53` (F011), `b2968f4` (FP16 helpers), `73d6a5b` (test-policy integration).
- Final tested source: `73d6a5bcd4fa1063e39d467284638b2812a84e7d`. Later changes only complete the plan and this report.
- Raw evidence: `/Users/goftin/dev/projects/VectorCore-remaining-repairs-2026-10-29-evidence`. The stable run label uses the October 29 local start date; execution and completion continued on October 30.
- [Plan](../../superpowers/plans/2026-10-29-remaining-repairs.md), [contracts and traced paths](ARCHITECTURE.md), [coverage ledger](COVERAGE.csv), [validation registry](validation-registry.json), [review record](review-summary.json), [evidence hashes](evidence-manifest.json).
- The [original audit findings](/Users/goftin/dev/projects/VectorCore-audit-2026-10-27-84798d3/evidence/FINDINGS.md) remain frozen at `84798d3`.

## Repairs

| Item | Result | Evidence |
|---|---|---|
| F012: unsupported benchmark rounding-bound domain | Selected executable cases with unsupported derived models return a configuration error before fixture setup. Integer count overflow is checked before gamma evaluation. Overrides cannot admit an undefined model. Defensive result handling and summaries report unsupported names without trapping. | Literal boundaries for four provider families, three dimensions and four providers; actual runner admission/selection controls; bounded injected reports; existing real Release CLI checks. |
| F011: missing D02 consumer fixtures | Three original positive fixtures restored byte-for-byte. Manifest checks precede snapshot/output creation. The driver shares the maintained required/optional Logger diagnostic rule and prints binding status from the SILGen result. | Actual Debug and Release runs each pass all ten fixtures, the 61-symbol binding check and the 77-member API backstop. Seven-driver fixture inventory and orchestration controls pass. |
| Two dormant FP16 comparison helpers | Both reject mismatched nonfinite operands and are now used by real conversion/distance tests. Existing paired NaN/infinity, finite tolerance, signed-zero and strict flush behavior remains. | Literal ordered classification tables, finite boundary/overflow cases, shape/prefix/flush controls, and corruption of every lane of actual conversion/kernel outputs. |

**F012 scope.** The first unsupported 512D generic squared-Euclidean provider count is `N=16,776,696`, where `k32=N+520=2^24`. Tests use the real catalog/runner with a callback that exits before setup; even the failing regression cannot allocate the enormous fixture. Adjacent admitted controls prove the runner is not simply rejecting everything. Filters and unavailable entries are resolved before the new gate. Non-observed and empirical routes retain their existing policies. Gamma formulas and admitted thresholds are unchanged; admission is not a physical-memory budget or a claim of useful sensitivity near the mathematical limit.

Injected result names can differ from the selected catalog name, so the policy also checks their domain before constructing a rule. Pretty/JSON/CSV reporting is exercised at the boundary and at `Int.max`, with strict mode and all override combinations. Internal raw model/rule APIs retain their documented caller preconditions.

**F011 reachability.** Restoring the manifest exposed two downstream reporting/contract defects: the binding message had used the later symbol-graph exit status, and Logger's optional compiler-dependent line 9 had been treated as required. Regressions reproduce both; the fixes preserve required line 10 and compiler exit 1. Other negative fixtures retain exactly one error per code line. Historical `baseline` mode means a pre-D02 module; no historical baseline replay is claimed. The actual executions here are current-module `final` runs, with copied modules/objects, fixtures, stage logs and provenance retained.

**FP16 policy.** Paired NaNs and same-sign infinities are accepted before the new finite guards. All remaining nonfinite pairs fail. The helpers retain their asymmetric reference denominator, inclusive absolute-or-relative acceptance and zero fallback. Conversion's strict `abs(original) < 2^-14` flush allowance is unchanged. Finite operands may overflow the relative-error calculation while satisfying the absolute arm; that case is deliberately retained and tested. This repair assigns no new policy to invalid tolerances.

The existing accuracy tests consume actual converted storage and kernel output, then keep their stronger previous assertions. New standalone kernel tests provide initialized array storage. The inventory now contains 16 finite-gated sites and nine bounded-reference sites; neither helper is dormant. Synthetic dormant fixtures keep direct-use, body-change and ordinary/raw-interpolation detection exercised. The lexical inventory remains a review check, not a dataflow or control-flow proof.

## Validation

Host: Apple M3 Max, Mac15,9, 16 logical CPUs, 48 GiB; macOS 27.0.1 (26A434). Xcode 27.1 (27A9275), Swift 6.4 (`swiftlang-6.4.0.34.1`, clang `2100.3.34.1`), Python 3.14.1. SwiftPM used the native build system, four build jobs, existing caches and `VECTORCORE_TEST_EXTENDED=0`.

| Check | Observed result |
|---|---|
| Final full Debug | Exit 0; 196 XCTest cases reported, 28 skipped, zero failures; 1,742 Swift Testing tests in 262 suites, with 18 skip events. |
| Final full Release | Exit 0; the same counts/skips, with one existing known compiler-canary issue. |
| Final Python policies | All 558 tests passed. |
| Final claim ledger | Zero problems. |
| Focused FP16 validation | Debug and Release: 138 tests in 32 suites each; eight oracle-policy tests passed. |
| F012 focused validation | Initial Debug/Release: 47 XCTest entries each. After consolidating the child fixture, the focused Debug run reports 46; final full Debug/Release cover those final tests. |
| Actual D02 driver | Both configurations: ten fixtures, 61 bindings and 77 members pass, with empty problem lists. |
| Bounded real Release CLI | Established invalid-input, output-failure, strict correctness, A/B, reporting and comparator checks passed. |

The Release known issue is the existing `LostStoreCanaryTests.literalArrayStoreIsObservedAfterOutOfLineClosure` under `withKnownIssue`. No new skip or known-issue exception was added. Existing compiler/build-system warnings remain. Timing printed by ordinary tests is not a controlled performance measurement.

The evidence retains unsuccessful checks as well as final passes:

- F012's initial admission regression reported 96 assertion failures with the allocation-preventing sentinel.
- F011's initial tests reported five failures. The later preflight test-development run had seven assertion failures and two propagated errors, including a mock serialization error; that run is not presented as clean behavioral-red evidence.
- The two actual FP16 helper classification tests reported 44 issues before repair. An interleaved warning split a UTF-8 glyph in their raw output; the recorder saved the raw log and exit record before its tail display failed. Only tail decoding was adjusted; original bytes remain preserved.
- The first full Python run caught two integration requirements: exact benchmark test-file membership and the duplicate child launcher. The repair reuses the existing single-selector benchmark launcher, adds the fifth source to the exact manifest, and pins eight benchmark-CLI filter literals. Existing launch controls and all-five source-removal controls pass. No new process-launch exception was introduced.

All final integrated runs use clean `73d6a5b`. Earlier dirty-source runs retain their exact patches and hashes in the registry. The removed tenth F012 XCTest entry was the standalone child fixture; its body now runs through the existing benchmark child entry. All nine behavioral tests remain.

## Review, coverage and disposition

Three targeted Astra/max discovery reviews and a fresh Astra/max final source challenge inspected the implementation and relevant dependencies. The final reviewer checked all 18 changed-path hashes against the captured v2 input and approved the source at `73d6a5b`. Its two integration findings are resolved. Final Release and evidence reconciliation completed after the review cutoff and were verified by the lead; they are not represented as reviewer-executed tests. This was targeted challenge of known repairs, not blind discovery.

The bounded source ledger records 75 inspected files: 46 reviewed in full and 29 with explicit partial bounds. Other repository inputs are identified as outside this remediation scope, and current audit outputs are excluded. The frozen comprehensive audit remains the broader review. No measured code-coverage percentage is claimed. [Review exclusions and lead rulings](review-summary.json) distinguish remaining platform/experimental limits from unresolved repair work.

Eleven original findings are repaired: F001, F002, F004, F005, F006, F007, F008, F009, F010, F011 and F012. F003 remains on the retired watchOS target. The additional two-helper debt recorded in the F009 report is now closed. Expanded FP16 support remains deferred.

The original checkout retains `main` at `8b02e75` and its recorded status; the frozen source retains clean detached `84798d3`. All 211 first-repair, 528 oracle-repair and 42 progress-repair evidence files still match their previous hashes. New logs, patches, reviews, D02 snapshots, available child output and workflow records are hashed separately. The worktree and branch are preserved; no merge, push, tag or release was performed.

This adds macOS arm64 runtime evidence. It does not add a hosted CI run, whole-package sanitizer coverage, alternate C-mode validation, Intel/device execution, a Swift 6.0 replay, performance certification, or validation of the acceleration package. Completion of these repairs does not establish that the entire library is defect-free.
