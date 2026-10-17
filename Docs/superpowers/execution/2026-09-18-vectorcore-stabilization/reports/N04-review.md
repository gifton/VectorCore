# N04 independent review: scale-invariant entropy

Reviewer: independent (Astra/medium slot). Date: 2026-10-03.
Inputs: brief N04.md, dispatch addendum, report.md + evidence logs, review-package.diff (BASE 0d0bf76, 4 files), read-only workspace.

Spec: PASS
Quality: CHANGES REQUESTED

The single blocking item is a documentation fix (Important-1). The production contract, numerics, oracle, cutoff table, coverage and mutation evidence all hold. No re-test is needed for Important-1 beyond lint, because the fix is doc-only.

## What I verified

### Evidence integrity
- The SHA-256 of all four workspace files matches `hashes.txt`. The cached diff equals `review-package.diff` apart from context width. Both mutation copies (`mutation-src/mutA`, `mutB`) differ from the final files only by the stated one-line mutation plus the two trailing-comma fixes (both whitespace-only, as the report says).
- Every log SHA-256 matches its `.json` record.
  - RED Debug: exit 1, 30 tests, 16566 issues.
  - Final Debug and final Release: exit 0, "30 tests in 2 suites passed".
  - Mutation A and mutation B: exit 1.
  - 30 = 21 retained `VectorEntropySuite` tests + 9 new tests.
- RED reproduces the defect through the public route and through both `entropyFast` branches, not just the helpers:
  - D=8 (small branch), s=1e-10 and leastNonzero: `entropy`/`entropyFast` = 0.0, reference 2.0794415. At s=maxFinite, `entropy` = 0.0.
  - D=128 (large branch), s=1e-10: `entropy`/`entropyFast` = 0.0. At s=maxFinite, `entropy`/`entropyFast`/`entropyLarge` = nan and `entropySmall` = 0.0.
  - Old large-path spikes returned -0.0.
  - The build compiled, so this is not compile-failure RED.

### (1) Contract
- **Denominator and cutoff.** `total` sums every coordinate (VectorEntropy.swift:67-87). The only test on the input sum is `total > 0` (line 89). The cutoff is `p > Double(Float.ulpOfOne)`, applied only to the normalized `p` (lines 95-96).
  - An absolute-sum cutoff (mutation A) fails `uniformMagnitudesReturnLogD`, `rationalClosedForms`, `unequalFinitePatterns...` and `commonPowerOfTwoScaleInvariance`.
  - Excluding omitted coordinates from the denominator would fail `tinyCoordinateStaysInDenominator` (`got > 0`, line 264) and `testEntropy_UlpThreshold_IgnoresNearZero` (line 150).
- **All-zero and spike inputs return exact +0.**
  - All-zero hits `return 0` at line 89, which is +0.
  - For a spike, Neumaier adds exactly 0 to compensation at every step, so `total == m`, `p == 1` and `log(1) == +0`. Then `0.0 - 0.0 = +0.0`.
  - Both cases are asserted by bit pattern at every scale, at first/middle/last position, with both signs, on all four routes (EntropyRangeContractTests.swift:77, 131-144, 291-298).
- **Non-finite input gives NaN.** `isFinite` fails for NaN and for ±Inf. The closure's early `return` stops pass 1, and `return .nan` follows (lines 71-74, 86). There is no min/max reliance.
  - This is tested at first/middle/last index, for all 8 widths, with 5 fillers (including maxFinite and leastNonzero), plus a NaN+Inf mix, on all four routes (lines 270-289).
  - Mutation B (`|| true`) lets a non-finite value poison `total`, and the `total > 0` guard then returns 0. That fails `nonFinitePropagatesAtAnyPosition` and three legacy NaN/Inf tests.
- **Signs and signed zeros don't matter.** The code takes `abs(Double(x))` after widening, so the sign is irrelevant. This is tested by alternating-sign uniforms, sign-flipped patterns and four signed-zero patterns.
- **One contract on every route.** Both helpers forward to `entropyTruncatedContract()`, and the gate at `D.value <= 64` is unchanged. Tests call `entropy`, `entropyFast`, `entropySmall()` and `entropyLarge()` directly at every width, so both helpers run at small and large widths. `entropySmall` and `entropyLarge` keep their `@usableFromInline internal` symbols.

### (2) Numerics
- **Compensated sum.** Neumaier is implemented correctly: one branch for `|sum| >= m` and one for the reverse, then `total = sum + compensation`.
  - Swift and LLVM keep strict IEEE semantics by default. Package.swift has no `unsafeFlags` and no fast-math options, so `(sum - t) + m` cannot be folded away.
  - This is verified by inspection only; see "Cannot verify" below.
- **Range.** Every finite Float magnitude is below 2^128 and the count is below 2^63, so `sum` stays below 2^191. For a nonzero x, `p` is at least 2^-149/2^191 = 2^-340, well above Double's smallest normal.
  - Pass 2 computes `log` only for `p > 2^-23`, so `log` always sees a finite, normal, positive argument.
  - For finite input, nothing can overflow, underflow to a wrong 0, or produce NaN.
  - `total >= max m` holds: each compensation increment is exact, and the rounding of the accumulator is about Dε² relative, which is far below half an ulp. Hence `p <= 1`, every term is `>= 0`, and the result cannot be negative.
- **Range-argument wording.** The argument is labelled "Range argument (not an exact-sum or universal error theorem)" (VectorEntropy.swift:55-58). The test tolerance is labelled "a regression bound, not an accuracy guarantee" (line 30). There are no speed claims, and the SIMD/"Optimized" wording is gone.
- **Single rounding.** Float rounding happens once, at `return Float(entropy)` (line 101). Every intermediate is Double.

### (3) Oracle independence and cutoff table
- **Independent reference.** `entropyContractReference` (EntropyRangeContractTests.swift:28-44) does not copy the production helper:
  - It detects non-finite input with `isFinite` and only then takes a max.
  - It scales magnitudes by the max, sums them without compensation, and builds normalized probabilities as `s/total`.
  - It is cross-checked against closed forms at line 156.
  - `manualEntropy` now forwards to it, so the old Float absolute-sum oracle is gone.
- **Cutoff table matches the brief** (lines 216-218):
  - below `[2^-23-2^-47, 1-2^-23, 2^-47]`
  - equal `[2^-23, 1-2^-23, 0]`
  - above `[2^-23+2^-46, 1-2^-22, 2^-23-2^-46]`

  I checked independently, with Python `Fraction`, that every component is exact in Float32 and every row sums to exactly 1. The probability-versus-2^-23 signs come out below `[-,+,-]`, equal `[0,+,-]` and above `[+,+,-]`. There is no rounded `Float(1-p)` complement.
- **Rational premise.** The integer premise (lines 207-211: 2^47-scaled integers summing to 2^47) is independent. Line 221 ties each Float component to its integer exactly.
- **Expected terms are specified independently with `log1p`.** For the equal row, an inclusive cutoff adds 1.90e-6, which exceeds the 9.54e-7 tolerance. For the above row, dropping p1 changes the result by 1.90e-6, which also exceeds the tolerance. So `>=` and an over-strict cutoff are both detected. The line-246 premise asserts that the omitted term is larger than the tolerance.
- **`[1, 2^-25]` fixture.** It is not asserted to be exactly 0. The tests assert `got > 0` and closeness to `-q ln q` (lines 253-268; VectorEntropyTests.swift:142-151).
- **Tolerance.** It is `8*ulpOfOne*max(1,|ref|)`, and both `actual` and `reference` must be finite first (lines 72-75).

### (4) Coverage
- **Dimensions.** 1/4/8/63/64/65/128/1536, using private `EDim1`/`EDim63`/`EDim65`.
- **Scale ladder.** leastNonzero, largest subnormal, leastNormal, every power of two from 2^-149 to 2^127, 1e-10, 1, 1e10 and maxFinite, with alternating signs, compared against closed-form ln D.
- **Zero padding.** An 8-element pattern is zero-padded to 8/63/64/65/128/1536. Rational closed forms are padded to every width.
- **Permutation and sign.** Reversed, rotated and sign-flipped variants.
- **Common-scale invariance.** Exact 2^k scalings, k=-140..115 step 5, with the exactness premise asserted at line 199.

### (5) Mutation evidence
- **Mutation A** (`total > Double(Float.ulpOfOne)`) fails 4 named tests.
- **Mutation B** (`x.isFinite || true`) fails `nonFinitePropagatesAtAnyPosition`, `testEntropy_NaNInput_ReturnsNaN_Small`, `testEntropy_InfiniteInput_ReturnsNaN_Small` and `testEntropy_NonFinite_LargePath_ReturnsNaN`.
- Both ran in scratch copies, and the workspace was not mutated. Both required mutations are demonstrated.

## Findings

### Critical
None.

### Important

**Important-1: Sources/VectorCore/Operations/VectorMath.swift:498. The ln D claim is unqualified, and it is false for a reachable domain.**

The doc says "`$\ln D$` for uniform nonzero magnitudes (within test tolerance, see `EntropyRangeContractTests`)".

- Under the retained cutoff, a uniform vector has `p = 1/D`. When D >= 2^23, `p <= 2^-23`, every term is omitted, and the function returns **+0**, not ln D.
  - I checked numerically: D=8388607 gives 15.9424 = ln D, while D=8388608 and D=16777216 give 0.
- The domain is reachable. `StaticDimension` (Core/Dimension.swift:9) and `DimensionStorage` (Storage/DimensionStorage.swift:20) are both public.
- The cited test covers only D <= 1536, so the claim is broader than its evidence.
- Three program sources point the same way:
  - The brief scopes the closed form to "this tested dimension range".
  - The design review (reports/N04-design-review.md:9, :13) requires the old "max-ln(n)" wording to be "qualified for truncation".
  - The claim-discipline rule makes this a defect.

Fix (doc-only), for example: "`ln D` for uniform nonzero magnitudes when `1/D > Float.ulpOfOne` (D < 2^23; tested for D <= 1536 in `EntropyRangeContractTests`). For D >= 2^23 every term is at or below the cutoff and the result is +0." Optionally state the same consequence once in the `entropyFast` contract (VectorEntropy.swift:17-20).

### Minor

**Minor-1: Tests/ComprehensiveTests/EntropyRangeContractTests.swift:159 and :163.** The comment is false, and the premise check skips silently.

- Line 159 says the scaled patterns "stay in the normal/subnormal-free range". But k = -148 produces subnormals for all three patterns, for example `[2^-145, 2^-146, 2^-147, 2^-147]`. That is good coverage, but the comment misdescribes it.
- Line 163 uses `guard ... else { continue }`, which would drop a case silently. It also checks only "finite and > 0", not exactness, yet the report says "(premise asserted)".
- In Python, every chosen (pattern, k) is exact and finite, so no case is skipped today.
- Fix: replace the guard with an `#expect` exactness premise like line 199 (`vals[i] / scale == p`), and correct the comment.

**Minor-2: EntropyRangeContractTests.swift:242 (with :225-226 and :232-233).** The below and equal rows cannot tell their expected value from 0.

- Their expected value is about 1.19e-7, but the absolute tolerance there is 9.54e-7. Returning +0, which drops the retained dominant term, would pass these rows. I confirmed numerically that `close(0, tDomBelow)` is true.
- Strictness of the cutoff is still asserted, and dominant-term retention is covered elsewhere (line 264 and VectorEntropyTests.swift:150), so this is not a coverage hole.
- Fix: add `#expect(got > 0)` to the table rows so the table asserts the full omitted-term policy by itself.

**Minor-3: Tests/ComprehensiveTests/VectorEntropyTests.swift:150-151.** The residual check has no explicit finiteness check before it.

- `abs(v.entropy - expected) <= 8*ulpOfOne` runs after only `v.entropy > 0`, which +Inf passes.
- It fails safe, because the comparison is false for both NaN and Inf. Still, the brief says "check finite output before any residual comparison", and the new file does this through `close()`.
- Fix: add `#expect(v.entropy.isFinite)` before the residual, or use a shared `close`-style helper.

**Minor-4: Sources/VectorCore/Operations/VectorEntropy.swift:30 and :57-58.** Two wording inaccuracies.

- Line 30 calls the tolerance "8 Float ulps". The tolerance is actually `8*ulpOfOne*max(1,|ref|)`: an absolute 9.5e-7 when |ref| < 1 (many ulps of the result), and about 2 result-ulps near ln 1536 ≈ 7.3. Suggest "8·ulpOfOne·max(1,|ref|)".
- Lines 57-58 say the "Double ratios stay in Double's normal range". That holds for nonzero coordinates only, because a zero coordinate gives p = +0. Suggest "nonzero ratios".

**Minor-5: EntropyRangeContractTests.swift:181-183.** The "coprime-ish" comment is false, and only cyclic permutations are tested.

- The comment claims the stride is "coprime-ish" with the width. That is false for D=64 (shift 22, gcd 2), and it is irrelevant anyway, because any rotation is a permutation.
- Permutation coverage is reversal plus one cyclic rotation. A fixed non-cyclic permutation, such as a deterministic stride/interleave, would exercise the "permuted" requirement more directly.
- Fix: correct the comment, and optionally add the non-cyclic case.

**Minor-6: VectorEntropy.swift:59.** `@usableFromInline` on `entropyTruncatedContract` is unnecessary.

- No `@inlinable` body references it, because `entropySmall` and `entropyLarge` are not `@inlinable`. The attribute only widens the exported symbol surface.
- Optional; plain `internal` is enough.

**Minor-7: VectorEntropyTests.swift:64-70 and :203-212.** Two retained legacy tests assert scale invariance with inexact, non-power-of-two scales.

- The scales are `k = 7.5`, and 1e6 versus 1e-3, with an ad-hoc 1e-6 tolerance.
- Keeping them complies with "retain existing tests". They are deterministic and pass. But they fall outside the brief's rule that invariance comparisons use exact power-of-two scaling.
- Root's call: leave them as they are, label them approximate, or switch to 2^k scales.

## Cannot verify from diff / observations for root
- **Who staged the index.** All four files in the worktree are staged in the index (`git status` shows `M `/`A `; index mtime 21:15:44, the same minute as review-package.diff and report.md). The addendum forbids index operations by the implementer, and the report says no index state changed. That fits root staging the files to build the review package, but I cannot attribute it from the evidence.
- **Compensation is not observable in tests.** For D <= 1536, an uncompensated Double sum has relative error of about 1.7e-13, far below the Float tolerance, and the cutoff-table totals are exactly 1 either way. No test would fail if compensation were removed. Its presence, and that it survives Release optimization, is verified by inspection plus the absence of fast-math flags. The brief did not require a compensation mutation.
- **Performance.** The large path changed from the SwiftSIMDProvider array chain to a scalar two-pass Double loop. Nothing was measured, and no speed claim is made, which complies with the brief. The public name `entropyFast` still implies speed, but the docs make no such claim.
- **Full suite not run.** Other test targets were not built or run (the report's own concern). Root's caller check covers source callers. Tests in other targets that assert the old -0 or old ulp-level values were not checked.
- **Out of write scope.** `Sources/VectorCore/Types/VectorData.swift:141` still says "Shannon entropy measure" for a stored `VectorQuality`-style field. It is outside N04's write set and is not the computed property. Noted only.
