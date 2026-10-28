# VectorCore Audit Repairs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Drop watchOS support and repair VC-F001, F002, F004, F005, F006 and F008 without changing the frozen audit baseline.

**Architecture:** Convert public reference storage to copy-on-write values with uniqueness before every write; release collection values outside its lock. Widen vulnerable numerical intermediates and expose deliberate public tuning constructors. Keep each defect's regression independent of its implementation.

**Tech Stack:** Swift6, Foundation/Accelerate/C, Swift Testing/XCTest, Python CI policy tools.

**Spec:** `Docs/superpowers/specs/2026-10-27-audit-repairs-design.md`

## Global Constraints

- Baseline84798d3; isolated repair branch; no merge/push/release.
- Supported deployment floors: macOS14, iOS17, tvOS17, visionOS1; Mac Catalyst retained; watchOS unsupported.
- No new package dependencies; ordinary public import must work.
- Existing approximate-zero, squared-distance and precision recommendation policies remain except the specified repairs.
- Preserve frozen audit artifacts; keep repair logs in a separate evidence directory.

## Review Focus

- Independent value copies must not share writes, including concurrent mutations; task2 covers both storage types and COWDynamicStorage.
- Throwing mutable borrows must preserve copy isolation and usable storage; task2 tests both.
- Destructor re-entry must observe completed collection state; task3 uses child-process deadlines and checked-Sendable hooks.
- Finite extrema, signed-zero endpoints and valid extrapolation must not acquire avoidable nonfinite results; task4 uses literal/Double references.
- Public construction opens input routes previously reachable only internally; task7 covers invalid-domain preconditions and fallback probes, and traces all tuning readers.

## Task 1: Supported-platform contract

**Files:** Package.swift; README.md; Docs/Package_Boundaries.md; .github/workflows/ci.yml; .github/ISSUE_TEMPLATE/bug_report.md; Scripts/ci/tests/test_platform_ci_coverage.py.

**Interfaces:** Existing package_products/CI policy consumes the manifest; remove the watchOS support promise without adding new APIs.

- [ ] Run existing platform policy tests on the baseline.
- [ ] Update supported-platform policy fixtures/assertions to the revised set, then remove the matching manifest/CI/docs declarations. These are policy/config changes; no separate source-text assertion test.
- [ ] Run `python3 -B -m unittest discover -s Scripts/ci/tests -p test_platform_ci_coverage.py -v`; expected PASS.
- [ ] Commit `build: retire watchOS support`.

## Task 2: VC-F001 storage concurrency

**Files:** Sources/VectorCore/Storage/{AlignedDynamicArrayStorage,ArrayStorage,COWDynamicStorage}.swift; Tests/ComprehensiveTests/ReferenceStorageValueTests.swift; public consumer fixtures; maintained ownership docs.

**Interfaces:** Existing names/initializer labels/dot/alignment; public types become structs and write APIs become mutating. Private aligned allocation owns destruction; clone before every mutation when shared.

- [ ] Add scalar and throwing/nonthrowing mutable-borrow copy-isolation tests for both public types and the wrapper. Copy source remains unchanged; mutated copy has expected values; empty storage and alignment remain valid.
- [ ] Run against baseline; expected visible alias mutation failures. Reuse original TSan evidence and ordinary-import shared-let mutation consumer as baseline controls.
- [ ] Convert DynamicArrayStorage to a ContiguousArray-backed value. Convert AlignedDynamicArrayStorage to a COW struct/private allocation; adapt COWDynamicStorage to delegate without double ownership layers. Trace every internal reference use and update intentionally changed client bindings.
- [ ] Run new/StorageBorrow/VectorShape/alignment/lifecycle suites. Normal-import shared-let mutation must fail compilation; concurrent independent-copy consumer must pass TSan in Debug/Release where supported. Record representative Release costs.
- [ ] Commit `fix: give public dynamic storage value semantics`.

## Task 3: VC-F002 collection destruction

**Files:** Sources/VectorCore/Integration/VectorCollection.swift; Tests/ComprehensiveTests/CollectionDestructionContractTests.swift.

**Interfaces:** Existing collection signatures/ID rules; a removed value's destruction occurs after mutation unlocks.

- [ ] Add bounded child scenarios for update/removeAll/single removal using a checked-Sendable lifetime hook that re-enters count and mutation; expect completion and correct state. Include absent update preservation.
- [ ] Run against baseline; expected update/removeAll child deadlines after destructor entry.
- [ ] Retain displaced values until after unlock, explicitly extending lifetime as needed.
- [ ] Run new tests and VectorCollectionTransactionTests; expected PASS in Debug/Release.
- [ ] Commit `fix: release collection values outside the lock`.

## Task 4: VC-F004 interpolation

**Files:** Sources/VectorCore/Initializers/ConvenienceInitializers.swift; Sources/VectorCore/Operations/VectorCoreOperations.swift; focused internal interpolation helper if shared; Tests/ComprehensiveTests/InterpolationRangeContractTests.swift.

**Interfaces:** Existing linspace/lerp/lerpUnclamped/smoothstep signatures remain.

- [ ] Assert linspace(-Float.max,+Float.max,width3) equals [-max,0,max] for fixed and dynamic; midpoint lerp/smoothstep equals0. Add widths0/1, endpoint bit patterns, ordinary quarter fractions, nonfinite controls, valid extrapolation and shape rejection.
- [ ] Run new suite against baseline; expected nonfinite midpoint/linspace failures.
- [ ] Widen finite endpoint arithmetic before subtraction; retain exact endpoint and clamping behavior.
- [ ] Run new suite plus D02DegenerateWidthTests and vector operation tests in Debug/Release; expected PASS.
- [ ] Commit `fix: stabilize finite vector interpolation`.

## Task 5: VC-F005 tiny query identity

**Files:** Sources/VectorCore/Operations/Kernels/EdgeCaseHandler.swift; Tests/ComprehensiveTests/BatchZeroQueryContractTests.swift.

**Interfaces:** safeBatchEuclidean512 still returns squared distances. Only its exact-zero shortcut predicate changes.

- [ ] Assert q=repeat(1e-13), candidates[q] and[q,q] have exact zero self-distance; use signed/tiny/zero queries and direct Double difference references across batch sizes0/1/2/3.
- [ ] Run against baseline; expected positive paired self-distance failure.
- [ ] Check actual coordinate zero for the batch shortcut, leaving other approximate predicates unchanged.
- [ ] Run new suite and affected SoA tests in Debug/Release; expected PASS.
- [ ] Commit `fix: preserve tiny nonzero batch queries`.

## Task 6: VC-F006 precision diagnostics

**Files:** Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift; Tests/ComprehensiveTests/PrecisionStatisticsRangeTests.swift.

**Interfaces:** PrecisionProfile fields and recommendation logic unchanged; calculations retain Double mean/residual/threshold.

- [ ] Assert 256 copies each of16777216/16777218 have population stddev1; one -max and511 +max have finite stddev approximately2*max*sqrt(511)/512 and exactly1 outlier. Include constant data and finite filtering.
- [ ] Run against baseline; expected stddev/outlier failures.
- [ ] Widen before subtraction and classify outliers with Double statistics.
- [ ] Run new suite and precision diagnostics tests in Debug/Release; expected PASS.
- [ ] Commit `fix: retain precision in diagnostic statistics`.

## Task 7: VC-F008 public tuning inputs

**Files:** Sources/VectorCore/Providers/AutoTuning.swift; ordinary-import consumer fixtures and Scripts/ci/consumer_smoke.py; AutoTuningConcurrencyTests.swift where domain checks need children.

**Interfaces:** `public init(breakEvenN:Int,minChunk:Int,modeBias:ModeBias)` and `public init(nsPerCandidateSeq:Double,parallelOverheadNs:Double,effectiveParallelFactor:Double)`.

- [ ] Add a normal-import consumer that constructs, sets/gets a manual override and returns CalibrationProbes from calibrateIfNeeded. Run baseline compile; expected inaccessible-initializer diagnostics.
- [ ] Add explicit initializers; Tuning preconditions breakEvenN>=0/minChunk>0, CalibrationProbes stores observations for existing fallback handling. Trace downstream readers.
- [ ] Run ordinary consumer Debug/Release plus AutoTuningConcurrencyTests and invalid-domain bounded children; expected PASS for valid clients and intended precondition/fallback outcomes.
- [ ] Commit `fix: expose public tuning input initializers`.

## Integration

- [ ] Run full default `swift test` Debug and Release, Python policy suite and affected consumer smoke. Repair affected claim-ledger identities without weakening its controls.
- [ ] Fresh Astra whole-branch review against the spec, with exact evidence and any rulings.
- [ ] Resolve consequential review findings with failing regressions, then re-run affected/full checks as appropriate.
- [ ] Record six finding dispositions, unsupported watchOS status and remaining findings in the repair handoff. Preserve branch/worktree for user review.
