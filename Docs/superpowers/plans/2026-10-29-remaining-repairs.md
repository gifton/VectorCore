# Remaining Audit Repairs Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans. Continue without routine approvals; the user authorized all three repairs.

**Goal:** Repair F012 and F011 and make the two previously dormant FP16 comparison helpers safe for their intended test use.

**Architecture:** Check the selected benchmark routes' derived-bound domain before fixture execution, retain the historical D02 backstops with current fixtures, and migrate the repaired FP16 helpers into exercised finite-gated test coverage. Production vector arithmetic and FP16 dimension support are outside this change.

**Tech Stack:** Swift 6, Swift Testing, XCTest, Python unittest, existing SwiftPM native build and evidence tooling.

**Spec:** The user's authorized repair request; frozen VC-F011 and VC-F012 in `/Users/goftin/dev/projects/VectorCore-audit-2026-10-27-84798d3/evidence/FINDINGS.md`; the F009 dormant-helper disposition in `Docs/reviews/2026-10-28-oracle-repairs/README.md`.

## Global Constraints

- Begin at `db9329540890f71b53e28f9eb591c1fb51ccf943` in the existing `codex/audit-top-six-2026-10-27` isolated worktree. Preserve the original checkout, frozen audit, and previous evidence.
- New raw evidence belongs in `/Users/goftin/dev/projects/VectorCore-remaining-repairs-2026-10-29-evidence`; report and registries belong in `Docs/reviews/2026-10-29-remaining-repairs/`.
- Astra max performs substantive source discovery and a fresh final review. Distinguish source inspection from executed checks.
- Preserve the gamma formula, finite comparison tolerances, paired NaN/equal-sign infinity policy, and the conversion helper's strict subnormal flush allowance.
- Reject selected unsupported derived-bound routes even with threshold overrides; overrides adjust thresholds within an admitted model domain. Filtered-out and unavailable routes must not cause rejection.
- Do not allocate the F012 multi-gigabyte fixture. Prove admission and policy behavior without allocating input vectors.
- Keep the repair branch unmerged and unpushed. No new FP16 kernels, dimensions, dependencies, or platform support.

## Review Focus

- Provider/model/dimension boundaries, integer count overflow, and both threshold overrides: selected-case and defensive policy tests.
- Filtering, unavailable entries, and injected result names: reject only executed unsupported models and keep reporting nontrapping.
- D02 manifest restoration exposes later stages: exercise all fixtures, SILGen binding, and symbol-graph member comparisons.
- FP16 mismatched nonfinite values in either order and any lane: literal classification tables and corrupted actual outputs.
- Finite absolute-or-relative acceptance, overflowed relative error, signed zero, shape mismatch, and the exact flush boundary: behavioral helper tests and mutation controls.

## Task 1: F012 benchmark model admission

**Files:** `Benchmarks/VectorCoreBench/Utils/CorrectnessPolicy.swift`, `Benchmarks/VectorCoreBench/AppMain.swift`, new `Tests/BenchmarkContractTests/BenchmarkModelDomainTests.swift`; maintained benchmark documentation as needed.

**Interfaces:** Preserve existing `errorModel`, `rule`, and `failure` call signatures. Add `CorrectnessPolicy.domainProblem(forCaseName:) -> String?` and non-evaluating model classification; share parsed route/count construction rather than duplicating the model-family formulas.

- [x] Add a runner regression using injected allocation-free cases at the known 512D generic boundary `N=16776696`; assert rejection before the case callback/body and before report creation, with strict mode and threshold overrides.
- [x] Run that regression before repair and retain the actual failure and source patch.
- [x] Add selected available-case domain validation before execution; make defensive result checking and summary handling nontrapping, including parsed counts near `Int.max`. Preserve current finite admitted thresholds.
- [x] Test every modeled provider family at literal below/at/above boundaries for 512/768/1536 and all provider names. Cover filters, non-observed and empirical routes, unavailable cases, overrides, and injected results. Run focused benchmark policy/selection suites in Debug and Release plus bounded real CLI checks. Expected: explicit rejection before setup for unsupported routes; existing admitted cases retain behavior.
- [x] Commit the repair, tests, and its maintained contract documentation.

## Task 2: F011 historical D02 driver

**Files:** `Scripts/ci/tests/D02RunConsumers.py`, `Scripts/ci/tests/D02/{StoragePositive,DeviceSupport,Factory}.swift`, Python driver policy tests and maintained driver documentation.

**Interfaces:** Preserve `BIN_PATH OUTPUT baseline|final`, marker names, and retained SILGen/member backstops. Restore the original positive fixture contracts with current value semantics where applicable.

- [ ] Add a retained-driver manifest membership test; run it against the missing fixtures and record the failure. Include missing/extra controls for the mechanical check.
- [ ] Restore the three positive fixtures. Reconcile the documented compiler-dependent LoggerNegative diagnostics and fix binding-status output to use the SILGen result, with behavioral regressions before each change.
- [ ] Run Python driver guards and the actual final-mode driver against current Debug and Release module snapshots. Expected: all ten fixtures, 61-symbol binding check and 77-member backstop pass; preserve exact evidence. Historical baseline execution is only claimed if actually performed against that baseline.
- [ ] Commit the driver repair and regression coverage.

## Task 3: FP16 comparison helper activation

**Files:** `Tests/ComprehensiveTests/MixedPrecisionKernelTests.swift`, new `Tests/ComprehensiveTests/FP16ComparisonHelperTests.swift`, `Scripts/ci/relative_oracle_inventory.json`, `Scripts/ci/tests/test_relative_oracle_guards.py`.

**Interfaces:** Existing Bool-returning `verifyFP16Accuracy(original:converted:tolerance:)` and `compareMixedVsFullPrecision(mixed:full:relativeTolerance:absoluteTolerance:)`; preserve their platform gates and finite tolerance policy.

- [ ] Add direct tests of both actual helpers with literal special-value classifications, finite tolerance boundaries, shapes, signed zeros, flush behavior and all-lane corrupted conversion/kernel output. Run the nonfinite regressions before repair and record their failures.
- [ ] Insert both-operands-finite guards after allowed paired specials and before flush/arithmetic. Keep error arithmetic and inclusive absolute-or-relative acceptance unchanged.
- [ ] Exercise both helpers from existing real converter/kernel tests while retaining their stronger existing assertions.
- [ ] Migrate both inventory entries from dormant to finite-gated; retain dormant enforcement with synthetic direct and interpolated activation controls. Run the helper tests, existing mixed-precision suites in Debug/Release, and Python oracle policies. Expected: matching specials remain accepted; mismatches fail; no weakening of the inventory guard.
- [ ] Commit the helper repair and coverage.

## Task 4: Integrated review and evidence

**Files:** new repair report, validation registry, bounded coverage record, review summary and evidence hashes under `Docs/reviews/2026-10-29-remaining-repairs/`.

**Interfaces:** Consume the three repaired areas and preserved raw logs; no new production interface.

- [ ] Obtain a fresh Astra review of the complete diff from `db93295`, including gate reachability and regression validity; resolve consequential findings with failing-then-passing evidence.
- [ ] Run complete Debug and Release suites and Python policy checks; retain pre-existing skips/known issues and identify new failures accurately. Run claim/documentation checks applicable to changed claims.
- [ ] Reconcile source coverage and identities, hash all new evidence, verify prior evidence and original/audit checkout preservation, and write the final disposition. Archive this plan's workflow ledger and remove only its temporary workspace.
- [ ] Commit the report and leave the clean repair worktree available for integration.
