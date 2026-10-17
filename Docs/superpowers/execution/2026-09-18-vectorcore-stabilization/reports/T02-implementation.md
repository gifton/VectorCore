# T02 report — milestone 11 whole-suite legacy failures

> Root-recorded copy: the harness refused the implementer's direct write of this file ("Subagents should return findings as text"), so the implementer returned the full report text and root saved it verbatim below. Evidence files referenced are in this directory and were written by the implementer's commands.

STATUS: DONE_WITH_CONCERNS. All three failures are legacy tests (no production regression). Fixes are test-only. The whole suite exits 0 with 0 issues in Debug and Release.

Workspace: workspaces/T02 @ 3639cc43fec063ba5c5dc5d126b9150b94de92bb; Swift 6.4 (swiftlang-6.4.0.34.1), Xcode 27.0, swiftbuild; capture_swift.py, --jobs 6, VECTORCORE_TEST_EXTENDED=0.
Changed: Tests/ComprehensiveTests/QuantizedKernelsTests.swift (two test bodies + private static helpers expectNonfiniteRejection, q01EndpointBound), Tests/ComprehensiveTests/ArbitraryDimensionTests.swift (one test body). Diff: t02-test-migration.diff sha256 24321742361e695b81aa74df9c4c831bdff6d9308ea7dfd8b293f9897343cc50.
Base comparison: git archive 8b02e75 vs 3639cc4 with an identical diagnostics probe; independent 60-digit Decimal oracle (diagnostics/t02_direction_oracle.py). Base needed a scratch-only typealias shim in the bench executable (ProgressReporter collision); library/test code untouched.

## Failure 1 — testNaNInfinityHandling (trap). Classification (a)
RED red-nan-debug.log: AffineQuantizationCore.swift:19 fatalError, signal 5. Cause: 155 nonfinite coordinates through nonthrowing Vector512INT8(from:params:), whose first statement (QuantizedKernels.swift:93) is requireFinite. Base silently encoded NaN/+Inf as 127 and -Inf as -128 (decoded 10.0 / 10.0 / -10.07874). Q01 brief: nonthrowing routes require finite input (fatalError); throwing initializers reject with invalidValues + indices; Q01 migrated sibling QuantizedKernelsComprehensiveTests.testNaNInfQuantization the same way, but Q01's focused filter selected only INT8QuantizationTests. Fix: the in-process nonthrowing NaN/Inf probes were removed (citing Q01ChildContracts.namedRejectionMatrix: from/supplied routes x 512/768/1536 x NaN/+Inf/-Inf in bounded children); the same fixtures go through throwing Vector512INT8(_:) and assert kind == .invalidData with exact construction-rule indices (155; 51/128/256; 171; 103). The finite 0% fixture is retained (error 0.019848168, bit-identical to base). GREEN Debug/Release.

## Failure 2 — "Normalization Preserves Direction" (ArbitraryDimensionTests:99). Classification (b), not a regression
ratio1 = 4352.9482421875 (0x45880796); ratio2 = ratio3 = 4352.94873046875 (0x45880797): 1 ulp apart. ||x|| = sqrt(18948160) = 4352.948426067095..., ulp32 = 2^-11. The 1e-5 absolute tolerance is about 0.02 ulp, so the test demanded bit-identical ratios. Oracle: base max coordinate error 2.056 ulp (0/384 correctly rounded) vs 3639cc4 0.499 ulp (384/384 correctly rounded); base's three sampled indices agreed by chance. N01 (84f42d0) replaced self*(1/Float magnitude) with fl32(fl64(x_i/N̂)), its accepted contract. Fix: bound 4u(1+2^-20)||x|| (u=2^-24), 1.0378e-3, derived in the comment (integer partial sums < 2^53 give N̂ = ||x||(1+α), |α| ≤ 2^-53; r_i = N̂(1+ε)/((1+β)(1+δ)); the subtraction is exact by Sterbenz). A misplacement of 3 ulp or more in either ratio fails. GREEN Debug/Release.

## Failure 3 — testDegenerateDistributions (:4631). Classification (a)
The failing fixture is "Nearly constant" {0.5, 0.500100016593933 (0x3f00068e)}, not "Binary" (the shared message misattributed it); Binary {0,1} and Single spike {0,10} keep 2 values at base and head. Base: interval [0.5, 0.5001], scale 3.92e-7, zero -1274916, codes [-128, 127]. Q01: zero-inclusive [0, 0.5001], scale 0.0019611765 = 0.5001/255, zero -128, both codes 127, so the pair separation (1.0002e-4) is 0.051 of a step. Q01 brief: zero inclusion intentionally trades narrow far-from-zero resolution for a representable zero point. Fix: the separation-aware assertion uses Q01's endpoint criterion E(x) = max(leastNonzero, scale + 2ulp(x) + 2ulp(x̂)) (as in QuantizationRangeContractTests.testCalibrationEndpoints); the endpoint premise is asserted (paramsUseDataEndpoints); E(x) holds for all 512 coordinates; a pair separated by more than E(x1)+E(x2) must decode to 2 values, a narrower one to 1 or 2. Classification pin: Binary 1.0 vs 7.84e-3, Spike 10 vs 7.84e-2 (resolvable), Nearly constant 1.0002e-4 vs 3.92e-3 (sub-resolution); minimum margin 39.2x. GREEN Debug/Release.

## Whole suite (VECTORCORE_TEST_EXTENDED=0)
Debug (suite-debug.log, 277.7 s, sha 2ce6c9e3…a04c4a5) exit 0: XCTest MinimalTests 59 executed/27 skipped, ComprehensiveTests 49, BenchmarkContractTests 20 (128 executed, 101 not skipped, 0 failures); Swift Testing ComprehensiveTests 1432 tests/196 suites passed (18 conditionally skipped), BenchmarkContractTests 1, MinimalTests 0. Release (suite-release.log, 52.9 s, sha 95532f78…29ebb0): identical totals. 0 issues, no fatal errors or signals.

## Lint and policy
SwiftLint 0.65.1: 21 warnings, 0 serious, the same rule/message pairs as the unmodified baseline. Policy discovery: 44 OK / 1 skip, before and after.

## Defect-class scan
Heuristic scan for nonthrowing quantization combined with NaN/Inf in the same function: 10 candidates, none feeding NaN/Inf in-process. testNaNInfinityHandling was the last instance.

## Concerns
1. Report write refused (root recorded this copy).
2. Base 8b02e75 does not build on Swift 6.4 (ProgressReporter); a scratch-only shim was used.
3. Second occurrence of "legacy test outside the accepting task's focused filter contradicts its contract" (Q01); the whole-suite gate is the mechanical closure; root decision.
4. Latent: dead pre-Q01 (max-min)/255 branches at QuantizedKernelsTests.swift:309 and QuantizedKernelsComprehensiveTests.swift:205 (unreachable, always symmetric); approxEqual default absolute 1e-5 (TestSupport.swift:8) is a hazard for magnitudes of about 100 or more [U]; a fatalError in any Swift Testing test kills the whole ComprehensiveTests process.


> ROOT CORRECTION (independent review Minor-1): the statement above "A misplacement of 3 ulp or more in either ratio fails" is FALSE as stated. Correct: the pairwise test fails iff pair (1,2) or pair (2,3) differs by >= 3 ulps. The code comment is correct. A per-ratio check is being added in fix round 1.

## Fix round 1 (root-recorded from the implementer's returned text)
Fix diff EV/t02-fix1.diff sha256 c147452e020f50214158a0d2025cc49df967e70f1dd90576f1f43cb97c5513bd. New hashes: ArbitraryDimensionTests.swift 11727d78e6191ef1211211243c4258df04dd42f7987be26ed52fb753f8ada6f1; QuantizedKernelsTests.swift e127d2fa000284fcfdf687b8c32c16950e3430e9287584c9fe5888e06a934a3a.
1. Minor-2: per-ratio |Double(x_i/n_i) − norm| ≤ 2u(1+2^-20)·norm for all 384 indices (bound ≈ 5.189e-4 = 1.0627 ulp; the exact bound ||x||(2u+2v)/((1−u)(1−v)), v = 2^-53, exceeds 2u by 1+6.15e-8, inside the margin). It passes iff every ratio is 0x45880796 or 0x45880797; base's 0x45880798 (1.623 ulp) fails. The pairwise check is kept and its comment states exactly what it catches (fails iff pair (1,2) or (2,3) differs by ≥ 3 ulps).
2. Minor-3: the print happens only inside the VectorError catch, with the observed additionalInfo["indices"] count (155, 51, 128, 256, 171, 103).
3. Minor-4: historical claims are labeled "Observed at pre-program revision 8b02e75 during the T02 investigation (not re-verified by a live test)".
4. Minor-5: comments say Q01's ideal scale vs the stored rounded/clamped/capped scale.
Alone runs: 6/6 exit 0 (Debug and Release). NEGATIVE CONTROL (negctl-direction-debug-fix1.log, rc 1): with only normalized(tolerance:) reverted to base's self*(1/mag), exactly one issue, at :126 (per-ratio), "ratios off ||x|| at [4,8,9,17,18,19,20,34] (76 of 384)"; pairwise :128/:129 recorded none; histogram 308 x 0x45880797 + 76 x 0x45880798, identical to base.
Whole suite: Release fix1 rc 0 (XCTest 128/101, 0 failures; Swift Testing 1432/196 + 1; 0 issues). Debug fix1 rc 1: the ComprehensiveTests Swift Testing process exited with signal 5 about 18 s in, with no diagnostic text, no issues and no crash report. Debug rerun1 rc 0, and four backtrace-enabled diagnostic runs (hunt1-4) rc 0 each: 1 trap in 6 Debug runs. R04-deadline hypothesis rejected (all R04 children completed normally); no overlapping capture-driver run; cause unknown [U]: the message-less trap points to cond_fail (e.g. overflow) or a dispatch misuse.
Lint: 21 warnings, 0 serious, the same rule set as 3639cc4. Policy: 44 OK / 1 skip.
Concern: HeapMembershipContractTests.negativeCapacity writes child captures to <package>/.build/R04-child-captures unless R04_CAPTURE_DIRECTORY is set (not caught by test_swift_test_output_paths); round-0 leftovers were moved to EV/ws-dotbuild-left-by-first-suite-runs/.
