# VectorCore: first six audit repairs

Session started 2026-10-27; handoff prepared 2026-10-28. The six selected production repairs are implemented on `codex/audit-top-six-2026-10-27`, together with retirement of watchOS support. Integration validation is recorded below. This is a repair record, not a replacement for the frozen audit or a release certification.

## Source and scope

- Starting source: `84798d3fe3548ea78d4ed5da5dba4d723a85bf32`.
- Reviewed implementation and policy revision: `f6bb24e69bcce999298e7f9dd9065e311e3f8f68`; final test fixture: `982c605`. The handoff is recorded in the subsequent documentation commit.
- Repair checkout: `/Users/goftin/dev/projects/VectorCore-fixes-2026-10-27-84798d3`.
- Frozen audit: [report](/Users/goftin/dev/projects/VectorCore-audit-2026-10-27-84798d3/evidence/REPORT.md), [findings](/Users/goftin/dev/projects/VectorCore-audit-2026-10-27-84798d3/evidence/FINDINGS.md), [validation](/Users/goftin/dev/projects/VectorCore-audit-2026-10-27-84798d3/evidence/VALIDATION.md). Its source checkout remains detached at the starting revision.
- Raw repair evidence: `/Users/goftin/dev/projects/VectorCore-fixes-2026-10-27-evidence`. Logs, command records, bounded child output, external consumer packages and measurement harnesses are kept separately from production source.
- Original checkout: `/Users/goftin/dev/gsuite/VSK/VectorCore`, still on `main` at `8b02e75bca7055ae11170873d15c06f1538709b7`; existing untracked recovery/documentation directories were retained.

The user authorized the six fixes, removal of watchOS support, and conversion of the two public storage classes to copy-on-write values. No merge, push, tag, dependency update or release was performed. The [design](../../superpowers/specs/2026-10-27-audit-repairs-design.md) and [implementation plan](../../superpowers/plans/2026-10-27-audit-repairs.md) record the contracts and sequence.

## Findings and regression coverage

| Finding | Implemented behavior | Regression and boundary coverage |
|---|---|---|
| VC-F001, high: mutable Sendable storage | `AlignedDynamicArrayStorage` and `DynamicArrayStorage` are values; aligned writes establish unique allocation ownership; `COWDynamicStorage` delegates to that ownership layer. | `ReferenceStorageValueTests`: scalar and throwing/nonthrowing buffer writes, original/copy isolation, independent concurrent copies, empty values, requested alignment, self and distinct-operand dot. `UnsafeMemoryContractTests`: allocation fallback and clone release. Ordinary-import negative `let` mutation and positive concurrent-copy consumers, including targeted TSan. |
| VC-F002, high: destruction under collection lock | `update` and `removeAll` retain displaced ownership until after unlocking. | `CollectionDestructionContractTests`: deadline-bounded update, clear and single-removal destruction callbacks read completed state and reenter mutation; existing transaction tests cover rejected updates and preservation. |
| VC-F004, medium: interpolation overflow | Finite endpoint arithmetic widens before subtraction; exact endpoints retain their bit patterns. | `InterpolationRangeContractTests`: fixed/dynamic linspace, lerp, unclamped lerp and smoothstep; opposite finite extrema, widths 0/1/3/5, signed-zero endpoints, quarter fractions, extrapolation, nonfinite controls and shape rejection. |
| VC-F005, medium: tiny queries replaced with zero | The batch shortcut requires actual coordinate zero. Existing approximate-zero predicates elsewhere are unchanged. | `BatchZeroQueryContractTests`: positive/negative tiny queries, signed zero, candidate counts 0–3, exact self-distance and independent Double difference references. Return values remain squared distances. |
| VC-F006, medium: variance and outlier arithmetic | Mean, residuals, variance and three-sigma comparisons retain Double precision until publication. | `PrecisionStatisticsRangeTests`: widths 512/768/1536, adjacent large Float values with population standard deviation 1, opposite extrema with finite expected deviation and one outlier, constants and finite filtering. Recommendation and all-nonfinite policies are unchanged. |
| VC-F008, medium: inaccessible public tuning construction | Explicit public initializers expose `Tuning` and `CalibrationProbes`; tuning requires nonnegative threshold and positive chunk size. | Ordinary-import `F008TuningConsumer.swift`: manual set/get, calibration fallback, all six optimized Euclidean/cosine width routes, sequential/parallel scheduling, thresholds 0/Int.max and chunks 1/Int.max. `TuningInputBoundaryTests`: bounded children reject negative thresholds and nonpositive chunks. |

The initial regressions failed before their corresponding production changes: storage alias writes, collection child deadlines, 72 numerical expectation failures across the three numerical suites, inaccessible tuning constructors, and five invalid tuning inputs that previously reached construction. Exact commands and outcomes are retained in the evidence registry and raw logs. Later coverage additions include distinct-operand dots, clone deallocation and stronger child deadline assertions.

## Compatibility and platform contract

The storage change is source- and layout-breaking. Mutation now requires a `var`; assigning one storage value to another no longer provides shared mutation. Consumers that need shared updates must own and synchronize that state explicitly. Rebuild downstream binaries. Scoped unsafe pointers must stay inside their borrow; ordinary Swift exclusivity still applies. See [Memory Alignment](../../Memory_Alignment.md#pools-copying-and-concurrency), [CHANGELOG](../../../CHANGELOG.md), and the maintained [downstream upgrade guide](../../superpowers/execution/2026-09-18-vectorcore-stabilization/DOWNSTREAM_UPGRADES.md), rows CB-43, BH-62 and AB-09.

watchOS was removed from the manifest support declaration, required CI destinations, README and maintained support documentation. Supported floors remain macOS 14, iOS 17, tvOS 17 and visionOS 1; Mac Catalyst remains in scope. SwiftPM's platform list does not prohibit incidental compilation on an undeclared platform. VC-F003 is still a defect in the audited watchOS snapshot; retiring that platform does not repair its arithmetic.

## Validation

Host: Apple M3 Max, Mac15,9, 16 cores, 48 GiB memory; macOS 27.0.1 (26A434). Main toolchain: Xcode 27.1 (27A9275), Swift 6.4 (`swiftlang-6.4.0.34.1`, clang `2100.3.34.1`). Additional ordinary Release consumers used installed Swift 6.2 (`swiftlang-6.2.0.19.9`, clang `1700.3.19.1`) and its macOS 26 SDK.

| Check | Recorded result |
|---|---|
| Full Debug package suite | Exit 0: 187 XCTest cases reported, 28 skipped, no failures; Swift Testing reported 1727 tests in 261 suites and 18 skip events. Extended performance lanes were disabled with `VECTORCORE_TEST_EXTENDED=0`. |
| Full Release package suite | Exit 0: 187 XCTest cases reported, 28 skipped, no failures; Swift Testing reported 1727 tests in 261 suites, 18 skip events and one existing known compiler-canary issue. All six repair suites are included. |
| Public consumer smoke | Exit 0: all 8 documentation programs and 19 compile contracts passed. The negative mutation contract requires errors on the intended four lines. |
| F001 ordinary consumers | Debug and Release passed, including independent-copy execution under TSan in both configurations. No sanitizer warning was observed in these targeted runs. |
| F008 ordinary consumer | Debug and Release passed, including all traced tuning routes and extreme valid constructor inputs. |
| F004/F005/F006 ordinary optimized consumers | Release passed on Swift 6.4 and Swift 6.2. |
| F002 ordinary optimized consumer | Explicit-subscript fixture passed update, clear and single removal on Swift 6.4 and Swift 6.2. The same fixture against the original Release baseline timed out on update/clear and passed single removal. |
| Python CI policies | 543 tests passed. Includes the 15 platform-policy tests and the claim-history regression below. |
| Storage measurements | Baseline and repaired Release binaries measured sequentially; [results and limitations](STORAGE_MEASUREMENTS.md). |

The original full Debug run used `26b5919`. Subsequent Swift test changes strengthened two timeout assertions and supplied a direct subscript witness in the collection fixture; the final Debug collection and transaction check passed 14 tests in two suites, and the final child-policy check passed 18 Python tests. Production implementation did not change after the full Debug pass. The final policy suite includes the checker fix at `f6bb24e`. The Release known issue is the pre-existing `LostStoreCanaryTests.literalArrayStoreIsObservedAfterOutOfLineClosure`, which deliberately tests a pure-Swift optimizer failure under the repository's existing `withKnownIssue` policy; this repair did not add an exception or suppress a failure.

Early Release builds aborted in Swift's `MandatorySILLinker` with `cannot deserialize shared function`, including native/default build systems and a disabled-CMO attempt. The standalone collection fixture failed on both installed compiler versions. Supplying its own `subscript(index:)` witness made both standalone consumers compile and all three scenarios pass, then allowed the full Swift 6.4 Release package suite to pass. Existing comparable custom-vector fixtures already supply that witness. This is a test-fixture workaround supported by the comparison; it does not establish the compiler's internal root cause or fix the original inherited-default consumer pattern. No production workaround or skipped regression was introduced. The Swift 6.2 command-line installation also lacks the XCTest/Testing modules needed for its attempted package-test runs.

See [validation-registry.json](validation-registry.json) for individual runs and commands, and [evidence-manifest.json](evidence-manifest.json) for SHA-256 hashes of the retained logs, command records, harnesses and child output. Raw paths in those records are relative to the external repair evidence directory. Earlier failed build/harness attempts are retained and distinguished from final successful checks.

## Independent review and evidence maintenance

Astra reviewed the design, ownership changes and complete implementation. The initial locking design was rejected before implementation because it did not cover nested cross-object borrows or Swift array exclusivity; the user then chose value semantics. Ownership review found no critical/important defect and suggested two additional tests, both added. Whole-branch review found no consequential production defect in the six repairs.

Follow-up review found that removed claim-ledger entries bypassed their superseded-history checks. A regression restored each of 24 historical statements and failed before the fix. The checker now validates that history before continuing past removed entries, and every restoration case passes. F001's generated migration mappings and the historical AB-06 note were also corrected. Astra confirmed those corrections at `f6bb24e` and identified the explicit-subscript fixture workaround subsequently verified in Debug and Release. Reviewers inspected source and supplied evidence; they did not independently rerun the tests. See [review-summary.json](review-summary.json).

## Remaining work

VC-F007 (extreme progress overflow), VC-F009 (nonfinite parity-helper gap), VC-F010 (uninvoked safe-division test closure), VC-F011 (stale historical D02 driver), and VC-F012 (CLI gamma-bound input domain) remain deferred. The current consumer-smoke driver passing does not repair the separate historical driver.

This work does not validate all supported Apple device/simulator runtimes, Intel/AVX, explicit `VC_USE_C_KERNELS=1`, extended test lanes, whole-package sanitizers, GPU providers or real acceleration-package integration. The supported-platform policy tests check configuration coverage, not runtime behavior on those platforms. Before a release or the acceleration-package handoff, retain these platform/integration gates and the documented numerical/unsafe ownership limits. Preserve this branch and its evidence for review.
