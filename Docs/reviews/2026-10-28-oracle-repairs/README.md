# VectorCore: F009 and F010 test repairs

F009 and F010 are repaired on `codex/audit-top-six-2026-10-27`. These changes make numerical test failures observable; they do not change production arithmetic or public APIs. The [first six repairs](../2026-10-27-audit-repairs/README.md) remain the preceding historical record.

## Source and evidence

- Starting revision: `25b1616198fecac5e4f1f7ae4c7a5e347673876d`.
- Implementation commits: `1f2c723` (F009), `029902c` (F010), `e575729` (review correction to the F009 policy).
- Repair checkout: `/Users/goftin/dev/projects/VectorCore-fixes-2026-10-27-84798d3`.
- Raw evidence: `/Users/goftin/dev/projects/VectorCore-oracle-fixes-2026-10-28-evidence`.
- Frozen audit: [findings](/Users/goftin/dev/projects/VectorCore-audit-2026-10-27-84798d3/evidence/FINDINGS.md), still describing source `84798d3`.
- [Implementation plan](../../superpowers/plans/2026-10-28-oracle-repairs.md), [validation registry](validation-registry.json), [mutation results](division-mutation-results.json), [review record](review-summary.json), and [evidence hashes](evidence-manifest.json).

`Sources/` and `Package.swift` have no diff from the starting revision. The original checkout remains on `main` at `8b02e75` with its existing untracked files; the frozen audit source remains clean and detached at `84798d3`. All 211 files in the prior repair evidence manifest still match their recorded hashes. Build caches were reused, while new logs and harnesses were stored separately. No merge, push, tag or release was performed.

## F009: reject nonfinite parity mismatches

`AccelerateArraySIMDProviderParityTests.isClose` now rejects nonfinite operands after its exact-equality shortcut. Matching infinities and signed zeros retain their existing equality behavior; finite/infinite pairs, opposite infinities and every NaN pair fail. Finite tolerance arithmetic is unchanged.

Three tests exercise the actual helper and array wrapper:

- `comparatorClassifiesNonfiniteValues`: a literal 7-by-7 expectation matrix covering both infinity signs, finite values, both zeros and NaN, plus empty/unequal array sizes.
- `comparatorPreservesFiniteToleranceBoundaries`: inclusive absolute and relative boundaries and their immediately smaller tolerances, in both argument orders.
- `comparatorRejectsCorruptedProviderOutputs`: finite provider reductions and array outputs deliberately replaced with infinity, negative infinity or NaN, including every array lane and both comparison orders.

Before the fix, the new Swift checks reported 76 issues. The same affected suites pass after the fix in Debug and Release. Five related live assertions also gained explicit finite checks: three SoA dot-product tests and two matrix-distance tests.

The [relative-oracle inventory](../../../Scripts/ci/relative_oracle_inventory.json) records 25 reviewed comparisons: 14 with finite guards, nine with bounded independent references, and two dormant helpers. Its Python policy rejects unreviewed, missing or duplicated sites, removed finite guards, changes to reviewed bounded bodies, and new references to dormant helpers. Controls exercise each of those checks. The original live-inventory check failed on the six missing guards before repair.

This is a lexical review check, not a Swift dataflow proof. It recognizes the documented relative-comparison forms; it does not resolve arbitrary names, aliases, macros, interprocedural behavior or control-flow dominance. Bounded-reference callers and fixture generators were reviewed but are not transitively hashed. A retained guard fragment alone is not proof of execution.

**Scope decision:** `verifyFP16Accuracy` and `compareMixedVsFullPrecision` are unused helpers with their own paired-NaN policy. Their bodies were left unchanged, and the policy requires them to remain declaration-only across `Tests` and `Sources`, including executable string interpolation. Future reuse must repair and test their nonfinite behavior first. They are not represented as repaired numerical helpers.

## F010: execute valid division and check all lanes

`ErrorHandlingValidationTests.testVectorMathSafeDivision` is now a throwing test. It calls `Vector.safeDivide` directly and checks every one of the 32 result lanes against `0.5`. Its invalid-denominator check remains active and unchanged.

The actual test file was copied into an external XCTest package for mutation checks. Mutations were inserted at the success-call/result boundary; production sources were never mutated.

| Variant | Original test, Debug / Release | Repaired test, Debug / Release |
|---|---|---|
| Normal division | Pass / pass | Pass / pass |
| Valid division throws | Incorrectly passes / passes | Fails / fails |
| Wrong lane 0 | Incorrectly passes / passes | Fails / fails |
| Wrong lane 16 | Incorrectly passes / passes | Fails / fails |
| Wrong lane 31 | Incorrectly passes / passes | Fails / fails |

All 20 final observations match those expectations. The root `ErrorHandlingValidationTests` suite also passes all nine tests in both configurations. These are tests of the assertion's sensitivity, not evidence of a production division defect.

## Validation and independent review

Host: Apple M3 Max, Mac15,9, 16 logical CPUs, 48 GiB; macOS 27.0.1 (26A434). Xcode 27.1 (27A9275), Swift 6.4 (`swiftlang-6.4.0.34.1`, clang `2100.3.34.1`); Python 3.14.1. Package tests used the native SwiftPM build system, four build jobs and `VECTORCORE_TEST_EXTENDED=0`.

| Check | Result |
|---|---|
| Full Debug package suite | Exit 0; 187 XCTest cases reported, 28 skipped, zero failures; Swift Testing reported 1730 tests in 261 suites, with 18 skip events. |
| Full Release package suite | Exit 0; the same reported counts and skips, with one existing known compiler-canary issue. |
| Focused F009 suites | Debug and Release: 64 tests in 12 suites reported passing, with four skip events. |
| Focused F010 suite | Debug and Release: nine XCTest cases, zero failures. |
| Final Python policy suite | Exit 0; 550 tests passed, including all seven new oracle-policy tests. |
| Claim ledger and generated downstream appendices | Both exit 0. |
| Mutation controls | All 20 final observations match expectations; detailed source snapshots and hashes retained. |

The full Debug run completed at `029902c`; the later `e575729` correction changes only the Python guard and its tests. Full Release ran with that correction. The final Python run covers the corrected policy. The Release known issue is the pre-existing `LostStoreCanaryTests.literalArrayStoreIsObservedAfterOutOfLineClosure`, under the repository's existing `withKnownIssue` policy. No new exception or skip was introduced.

Astra performed discovery/inventory review and a fresh review of `25b1616..029902c`. It found no consequential Swift repair defect and one P3 policy gap: executable string interpolation was masked while counting dormant-helper references. Four new ordinary/raw-string subcases failed before the correction, then all seven oracle-policy tests passed. The lead verified this correction; no second independent review is claimed. The review record distinguishes source inspection from runtime validation.

Earlier unsuccessful setup attempts are retained: an initial Swift Testing keypath/rethrows macro compilation error, two policy-control failures caused by an overly narrow diagnostic matcher, and six F010 mutation harness records that expected `XCTAssertEqual failed` instead of XCTest's actual `XCTAssertEqualWithAccuracy failed`. The last six tests already failed as intended; their diagnostic matcher was corrected and the runs repeated. They are not counted as final successful harness observations.

The evidence manifest covers logs, command records, harness source, mutation snapshots, review records and archived workflow records. It also retains 390 available child logs referenced by the full suites; eight T10 launcher paths were unavailable when archived and are listed as such in `child-output-index.json`. Build products and caches are excluded. Some task command records capture HEAD while a regression edit was still uncommitted; the implementation commits and retained mutation snapshots identify the tested changes. The registry calls out record provenance rather than treating every HEAD field as an exact clean-tree snapshot.

## Disposition and limits

Eight of the twelve audit findings are now repaired: F001, F002, F004, F005, F006, F008, F009 and F010. Three active findings remain: F007 (extreme progress overflow), F011 (historical D02 evidence driver) and F012 (CLI gamma-bound input domain). F003 remains a historical defect on the retired watchOS target, not a repaired finding.

The unused-helper decision and lexical limits above are retained deliberately. They can require future review if helpers, reference generators or recognized syntax change. This task adds no all-platform, whole-package sanitizer, extended-lane, Intel/AVX, C-kernel, GPU-provider or acceleration-package integration validation. Those release gates and the earlier audit limitations still apply.
