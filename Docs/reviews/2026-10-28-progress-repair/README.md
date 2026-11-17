# VectorCore: F007 progress percentage repair

F007 is repaired on `codex/audit-top-six-2026-10-27`. `OperationProgress.percentage` preserves the existing floating-point computation and truncation, including percentages above 100. A result outside the integer range now returns `Int.max` instead of trapping.

## Source and contract

- Baseline: `6e7a136aa7723007b3155caade0c656438a29197`, following the [F009/F010 repairs](../2026-10-28-oracle-repairs/README.md).
- Implementation: `ac1897e`; named-test claim registration: `4fbfaa0`.
- Repair checkout: `/Users/goftin/dev/projects/VectorCore-fixes-2026-10-27-84798d3`.
- Raw evidence: `/Users/goftin/dev/projects/VectorCore-progress-fix-2026-10-28-evidence`.
- [Frozen F007 finding](/Users/goftin/dev/projects/VectorCore-audit-2026-10-27-84798d3/evidence/FINDINGS.md#vc-f007), [plan](../../superpowers/plans/2026-10-28-progress-overflow.md), [validation registry](validation-registry.json), [review summary](review-summary.json), [evidence hashes](evidence-manifest.json).

Only `Sources/VectorCore/Progress/OperationProgress.swift` changes in production. The public property remains a nonthrowing `Int`. Input clamping, stored state, `fraction`, `isComplete`, factories and stream behavior are unchanged. FP16 code is unchanged; consumer discussion is separate from this repair.

**Decision:** saturate the final integer percentage at `Int.max`. Capping at 100 would change existing overcomplete progress behavior. This preserves the current Double rounding; it does not promise exact integer-rational percentages at extreme magnitudes. Consumers needing that representation must calculate it separately.

The guard uses `scaled >= Double(Int.max)`. On a 64-bit target, converting `Int.max` to Double rounds upward to 2^63, so a strict `>` guard would still permit a trapping conversion at equality. Inputs are immutable nonnegative integers after initialization, and zero totals produce a zero fraction; the scaled result is therefore finite and nonnegative over the native input domain. The adjacent safe boundary is exercised by the named regression below.

## Regression and source coverage

The progress test file now uses ordinary `import VectorCore`, exercising the public surface. Three new tests cover:

| Test | Cases and purpose |
|---|---|
| `testOperationProgress_PercentageRange` | 13 rows: current/total 0, 1 and Int.max, plus Int.min initialization. Exposes the large-numerator trap and preserves zero-total/clamping behavior. |
| `testOperationProgress_PercentageConversionBoundary` | Four rows: immediately below the 64-bit floating conversion threshold, at its rounded equality, above it, and `current=Int.max, total=100`. The safe expected integer is Int.max−1023, derived independently of the production floating conversion. |
| `testOperationProgress_PercentagePreservesOvershootAndTruncation` | Retains 150%, 33% and 66%, and verifies the overshoot's stored counts, fraction and completion state. |

In Debug, the range and boundary tests each compiled and terminated with signal 5 and the actual out-of-range Double-to-Int diagnostic before the repair. Both pass after it. The range patch and exact commands are retained. The test has alternate 32-bit expected values, but those branches were not executed on this 64-bit host; watchOS remains retired.

Astra independently reviewed the source, initializer invariants, boundary arithmetic, callers and related conversions. No second reachable occurrence of this unbounded-percentage conversion was found. Related percentile conversions are bounded by allocated collection sizes; an additional test conversion uses fixed ratios. This review does not claim an exhaustive proof of every floating-to-integer conversion in the repository.

The changelog, maintained downstream row BH-67, generated CL-77 mapping and `VC-AUDIT-F007` claim entry record the same behavior and point to the actual tests. The initial changelog check correctly reported an unregistered claim; the named-test entry was then added, and the checker passed. An in-flight Python run was intentionally interrupted before registration, retained as incomplete, and replaced by a complete final run.

## Validation and review

Host: Apple M3 Max, Mac15,9, 16 logical CPUs, 48 GiB; macOS 27.0.1 (26A434). Xcode 27.1 (27A9275), Swift 6.4 (`swiftlang-6.4.0.34.1`, clang `2100.3.34.1`), Python 3.14.1. Package checks use native SwiftPM with four build jobs and `VECTORCORE_TEST_EXTENDED=0`. Existing build caches were reused; prior raw evidence was preserved separately.

| Check | Result |
|---|---|
| Focused Debug progress/stream suites | 27 tests in two suites passed. |
| Focused Release progress/stream suites | 27 tests in two suites passed, including the ordinary-import overflow and boundary cases. |
| Full Debug package suite | Exit 0; 187 XCTest cases reported, 28 skipped, zero failures; Swift Testing reported 1733 tests in 261 suites and 18 skip events. |
| Full Release package suite | Exit 0; 187 XCTest cases reported, 28 skipped, zero failures; Swift Testing reported 1733 tests in 261 suites, 18 skip events and one existing known compiler-canary issue. |
| Final Python CI policy suite | Exit 0; 550 tests passed. |
| Claim ledger and downstream appendices | Both checks pass. |
| Independent final Astra review | No actionable findings in `6e7a136..4fbfaa0`. Source, independent boundary expectations, public import and documentation mapping checked. |

The Release known issue remains `LostStoreCanaryTests.literalArrayStoreIsObservedAfterOutOfLineClosure` under the existing `withKnownIssue` policy. No new skip or exception was added. Existing compiler/build-system deprecation warnings remain in the logs.

The fresh final reviewer inspected source and supplied red/green evidence, without rerunning broad tests. Full-suite results were still pending at review time and were verified separately by the lead. The focused optimized build began at `ac1897e`; the claim-ledger-only commit landed while it ran, with identical Swift source/test content. Full runs use `4fbfaa0`. The [registry](validation-registry.json) retains exact commands, source revisions, tracked patches and outcome classifications.

## Disposition and preservation

Nine of twelve findings are repaired: F001, F002, F004, F005, F006, F007, F008, F009 and F010. Active follow-ups are F011 (historical D02 evidence driver) and F012 (CLI gamma-bound domain). F003 remains a historical defect on the retired watchOS target.

The original checkout and frozen audit source remain unchanged. All 211 first-repair evidence files and all 528 F009/F010 evidence files match their previous hashes. The new manifest preserves raw logs, source patches, review records and workflow evidence, excluding build caches/products. The branch and worktree remain available, unmerged and unpushed.

This repair adds macOS arm64 runtime evidence. It does not add runtime validation on every supported Apple platform, a 32-bit target, whole-package sanitizers, extended lanes, explicit C kernels, GPU providers or the acceleration package.
