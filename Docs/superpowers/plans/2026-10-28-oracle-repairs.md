# F009 and F010 Test Oracle Repairs Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans. Track each verified step below.

**Goal:** Make nonfinite parity failures observable and execute the safe-division success assertion.

**Architecture:** Keep existing numerical tolerance formulas; classify nonfinite operands before relative arithmetic. Add finite checks to repeated live oracle sites and a bounded lexical review guard. Make the XCTest method throwing and assert every result lane directly.

**Tech Stack:** Swift Testing, XCTest, Python unittest and existing Swift lexical helpers.

**Spec:** User-approved F009/F010 scope and the frozen findings at `/Users/goftin/dev/projects/VectorCore-audit-2026-10-27-84798d3/evidence/FINDINGS.md`.

## Global Constraints

- Start at `25b1616198fecac5e4f1f7ae4c7a5e347673876d` on the existing isolated repair branch; no production API or implementation changes.
- Preserve the original audit and first repair evidence; new raw evidence goes in `/Users/goftin/dev/projects/VectorCore-oracle-fixes-2026-10-28-evidence`.
- Preserve matching-infinity equality and signed-zero equality in F009; reject all NaN comparisons there. Finite tolerances stay unchanged.
- Preserve F010's invalid-denominator check; valid division must execute and validate all 32 lanes.
- No routine phase approvals; implement here, use fresh Astra review at integration, preserve branch unmerged.

## Review Focus

- Both argument orders and infinity signs, paired/mismatched NaNs, signed zeros and finite tolerance boundaries: task 1 classification matrix.
- Infinite actual or reference in array and reduction paths: task 1 corrupted-output controls.
- Three SoA and two matrix assertions with production references: task 1 explicit finite assertions and mechanical inventory controls.
- Existing dormant mixed-precision helpers and safe comparators: inventory their status; no silent broad policy changes.
- Thrown valid division or wrong first/interior/last lane: task 2 mutation controls against the actual test method.

## Task 1: Nonfinite parity oracles

**Files:** `Tests/ComprehensiveTests/AccelerateArraySIMDProviderParityTests.swift`, `BatchKernels_SoATests.swift`, `MatrixDistanceTests.swift`; new `Scripts/ci/tests/test_relative_oracle_guards.py`.

**Interfaces:** Keep private `isClose` and `arraysClose` signatures. The policy inventories explicitly documented relative-comparison syntax, checks reviewed finite gates and rejects new/unreviewed sites; it does not claim arbitrary Swift dataflow analysis.

- [ ] Add a literal Float-class expectation matrix and corrupted finite-output controls to the actual helper's suite; run focused Debug and observe F009 failures.
- [ ] Add the mechanical inventory with positive/negative controls, including missing finite-gate mutations, and observe current repeated sites fail.
- [ ] Add the finite guard to `isClose` after exact equality, plus explicit finite checks at the five repeated live assertions. Preserve tolerance arithmetic.
- [ ] Run affected Swift suites in Debug/Release and the Python guard; expect pass. Commit the repair and record red/green evidence.

## Task 2: Executed division assertion

**Files:** `Tests/MinimalTests/ErrorHandlingValidationTests.swift`; external temporary mutation consumers only.

**Interfaces:** `testVectorMathSafeDivision() throws` directly calls the existing public `Vector.safeDivide` and checks all result lanes.

- [ ] Copy the actual test into an external XCTest consumer; demonstrate that deliberate thrown/wrong-result mutations inside the current success arm still pass.
- [ ] Replace the uninvoked closure with a direct throwing call and assertions on all 32 lanes.
- [ ] Verify that the corrected test fails for thrown and first/interior/last-lane corruptions and passes without mutation, in Debug and Release. Commit the test repair.

## Integration

- [ ] Run the full package suite in Debug/Release under the existing default policy, the Python policy suite, and fresh Astra review. Record existing skips/known compiler canaries accurately.
- [ ] Update repair dispositions and preserve commands, hashes, mutation outcomes and review evidence in a separate handoff. Keep F007/F011/F012 and unsupported F003 unchanged.
