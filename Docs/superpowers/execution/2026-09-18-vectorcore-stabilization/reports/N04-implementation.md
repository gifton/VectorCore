# N04 report: scale-invariant entropy

Status: DONE_WITH_CONCERNS. Workspace: .../2026-10-03/workspaces/N04 (detached 0d0bf76). Toolchain: see toolchain.txt (macOS 27 / Xcode 27.0 / Swift 6.4, swiftbuild default).

## Changed files (exclusive-write set only)
- Sources/VectorCore/Operations/VectorEntropy.swift: entropySmall/entropyLarge (retained, @usableFromInline internal) both forward to one new shared `entropyTruncatedContract()`; entropyFast branch structure retained. Provider/array chains removed.
- Sources/VectorCore/Operations/VectorMath.swift: `entropy` doc and forwarding comment only.
- Tests/ComprehensiveTests/VectorEntropyTests.swift: manualEntropy now forwards to the independent reference; UlpThreshold test rewritten to assert the retained policy (small positive nonzero dominant term, not 0).
- Tests/ComprehensiveTests/EntropyRangeContractTests.swift (new, 9 tests).
- Diff: N04/N04.diff. Hashes: N04/hashes.txt (final). No generator script was needed.

## Contract mapping
- Double widening + Neumaier-compensated magnitude sum + Double term accumulation, single Float rounding at end: pass 1 / pass 2 of the shared routine; O(D), O(1) aux.
- Cutoff `p > Double(Float.ulpOfOne)` on normalized p only; zero total check is `total > 0` (no input-sum cutoff). All coords in denominator.
- NaN/+-Inf anywhere -> NaN via `isFinite` (no min/max). All-zero (incl. signed zero) -> +0 (bit pattern asserted). Spike -> +0 (bitPattern 0 asserted; old entropyLarge returned -0).
- Tests: uniform ln(D) at dims 1/4/8/63/64/65/128/1536 over leastSubnormal, largestSubnormal, leastNormal, all 2^-149..2^127, 1e-10, 1, 1e10, maxFinite, alternating signs; spikes (first/middle/last, +-); closed forms [8,4,2,2],[1,1,2],[3,1] with exact power-of-two scalings (premise asserted); unequal patterns with zero padding, reversed, rotated, sign-flipped; cutoff table (below/equal/above, exact-total premise asserted via integers scaled 2^47, omitted-term policy asserted to be distinguishable from tolerance); [1,2^-25] fixture; non-finite at first/middle/last for all dims and fillers (including mixed NaN+Inf); signed zeros. Each checks public `entropy`, `entropyFast`, and direct `entropySmall()`/`entropyLarge()` at every width (dims below/above 64 both exercise both helpers). Reference: Double max-scaled, uncompensated scalar sum (independent structure). Tolerance 8*ulpOfOne*max(1,|ref|).
- No speed claims made. Comment claims name tests (EntropyRangeContractTests / VectorEntropySuite); the 2^191/2^-340 range statement is marked as a range argument.

## RED (unchanged production, new tests + rewritten oracle)
- argv: capture_swift.py --workdir <ws> -- test --jobs 6 --scratch-path N04/build-red-debug --filter 'EntropyRangeContractTests|VectorEntropySuite' (Debug). Exit 1. Log red-debug.log, record red-debug.json (log sha256 d1175c280acf56586e5143867ddf8c767c7da1dc7cae912a76a177e554281494).
- 30 tests, 16566 issues. Failing: uniformMagnitudesReturnLogD (3837; e.g. D=8 s=1e-10 all four routes 0.0 vs 2.0794415; D=8 s=maxFinite entropy/Fast/Small 0.0, entropyLarge NaN), singleSpikes (12240; entropyLarge -0.0), rationalClosedForms (196), unequalFinitePatterns (192), commonPowerOfTwoScaleInvariance (92), tinyCoordinateStaysInDenominator (8), testEntropy_UlpThreshold_IgnoresNearZero (1). Passing under old code: cutoff table, non-finite, signed zeros.
- Public route and both entropyFast branches are in the failing set. Compilation succeeded (not a compile-failure RED).

## GREEN
- Debug, first run: green-debug.log exit 0, 30 tests passed. Release first attempt aborted (argv quoting `-c release` as one arg; logs in N04/aborted/).
- Release: green-release.log exit 0, 30 tests passed.
- Final after trailing-comma lint fix in the new test file: final-debug.log exit 0 (30 tests, 2 suites passed); final-release.log exit 0 (30 tests, 2 suites passed). argv: `-- test [-c release] --jobs 6 --scratch-path N04/build-final-{debug,release} --filter 'EntropyRangeContractTests|VectorEntropySuite'`.

## Mutations (copies at N04/mutation-src/mut{A,B}, workspace untouched; run Debug, same filter)
- A: `guard total > 0` -> `guard total > Double(Float.ulpOfOne)` (absolute input cutoff): exit 1, mutation-A.log; fails uniformMagnitudesReturnLogD (3496 issues), rationalClosedForms, unequalFinitePatterns, commonPowerOfTwoScaleInvariance.
- B: `guard x.isFinite` -> `guard x.isFinite || true` (NaN propagation removed): exit 1, mutation-B.log; fails nonFinitePropagatesAtAnyPosition (1404), testEntropy_NaNInput_ReturnsNaN_Small, testEntropy_InfiniteInput_ReturnsNaN_Small, testEntropy_NonFinite_LargePath_ReturnsNaN.
- Mutations were run on the test file before a whitespace-only trailing-comma lint fix (no semantic change).

## Lint and scanner
- SwiftLint 0.65.1 on the 4 files: exit 0, 0 serious; 1 warning remains, pre-existing at VectorMath.swift:76 (for_where, code not touched by N04). The 2 new-file trailing_comma warnings were fixed (swiftlint-v2.log).
- Bounds scanner test_child_process_bounds.py: 7 tests OK (scanner.log).

## Concerns
- Full suite (VECTORCORE_TEST_EXTENDED=0) and other-target compile not run; focused filter only, per brief. Other tests that call entropy elsewhere were not checked.
- Behavior change: spike/zero results are now +0 (old entropyLarge could produce -0); results shift by <= a few ulps for ordinary inputs.
- Test runtime is ~2.5 s Debug (large nested loops); no deadline-bounded timeouts occurred.
- I ran `git stash list` (read-only) once by mistake; no stash/index state changed.
- Independent Astra review and root adjudication still pending per brief.

## Fix round 1 (edits)
Edits only; NO swift build/test run this round (scheduling). SwiftLint on the 4 files: exit 0, only the pre-existing VectorMath.swift:76 for_where warning. Edits are uncompiled and unverified until root resumes me.
1. VectorMath.swift `entropy` doc: ln D only while 1/D > Float.ulpOfOne (D < 2^23); D >= 2^23 gives +0 by the cutoff; tested range cites uniformMagnitudesReturnLogD, the D >= 2^23 case marked unverified (derived). VectorEntropy.swift docs rechecked: no other uniform overclaim.
2. EntropyRangeContractTests: k=-148 comment corrected (subnormal but exact); silent `continue` replaced by `#expect(finite && >0 && vals[i]/scale == p)` per component.
3. Cutoff table rows: added `#expect(got > 0, ...)` per route.
4. VectorEntropyTests UlpThreshold: `#expect(v.entropy.isFinite)` before the residual comparison.
5. VectorEntropy.swift: tolerance text now 8*ulpOfOne*max(1,|reference|); "normal range for nonzero ratios".
6. Removed "coprime-ish" comment; added a non-cyclic stride permutation (stride = first of 5,3,2,7 coprime to dim, with bijection asserted via a Set) beside reversal and rotation; added private gcd helper.
7. Removed @usableFromInline from entropyTruncatedContract (called only from the @usableFromInline helpers' non-inlinable bodies; compile to be confirmed).
8. Legacy-approximate comments added on both legacy scale tests.
Planned verification: focused Debug and Release with -fix1 suffix, plus the absolute-cutoff mutation against the new table rows.

### Fix round 1 verification
- Compiled cleanly (no errors) including @usableFromInline removal and gcd/stride helper.
- fix1-debug.log: exit 0, 30 tests in 2 suites passed. fix1-release.log: exit 0, 30 tests in 2 suites passed. argv: `-- test [-c release] --jobs 6 --scratch-path N04/build-fix1-{debug,release} --filter 'EntropyRangeContractTests|VectorEntropySuite'`. Hashes: hashes-fix1.txt.
- Mutation A re-run (absolute-input cutoff, scratch mutation-src/mutA): mutation-A-fix1.log exit 1, 30 tests, 3996 issues; fails uniformMagnitudesReturnLogD (3496), rationalClosedForms (168), unequalFinitePatterns... (240, includes new strided case), commonPowerOfTwoScaleInvariance (92). Honest note: normalizedCutoffTable does NOT fail under mutation A, because its totals are 1 and an input-sum cutoff never fires there.
- The new `got > 0` row assertions bite a different mutation C (scratch mutC: `if p > cutoff && p < 0.99999`, drops the retained dominant term): mutation-C-fix1.log exit 1, 41 issues; fails normalizedCutoffTable (32 issues, all 32 are "dropped retained dominant term"), tinyCoordinateStaysInDenominator (8), testEntropy_UlpThreshold_IgnoresNearZero (1). A first C attempt with threshold 0.9999999999 was a no-op (dominant 1-2^-23 stays below it), exit 0, moved to N04/aborted/.

## Fix round 2
Comment-only edit in VectorMath.swift `entropy` doc: "tested widths 1...1536" replaced with "tested at widths 1, 4, 8, 63, 64, 65, 128, 1536 only" (the test named there is EntropyRangeContractTests.uniformMagnitudesReturnLogD). No other change; no rebuild (comment-only). SwiftLint on VectorMath.swift: exit status 0, only the pre-existing line-76 for_where warning.
7db3df6b6f39469862049151b672aa1cf0ed0f331d930a3848c2caf310243e9e  Sources/VectorCore/Operations/VectorMath.swift
