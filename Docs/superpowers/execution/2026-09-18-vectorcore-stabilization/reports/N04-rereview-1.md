# N04 fix round 1 re-review

Scope: fix1-package.diff only. No builds run, no edits to source.

## Per-item verdicts
1. [Important] ADDRESSED. VectorMath.swift:498-502 now states ln D only while 1/D > ulpOfOne (D < 2^23), +0 from D >= 2^23, names uniformMagnitudesReturnLogD, and marks the D >= 2^23 case unverified. Arithmetic checks: p = 2^-23 is not > ulpOfOne = 2^-23, and a sum of 2^23 equal values is exact in Double. Other owned doc comments: VectorEntropy.swift has no other uniform claim; line 28 "uniform ln(D) across the Float exponent range" is scale-axis only and not an overclaim.
2. [Minor-1] ADDRESSED. EntropyRangeContractTests.swift ~:160-166: false comment replaced (subnormal but exact), silent guard/continue replaced by per-component #expect(finite && > 0 && vals[i]/scale == p).
3. [Minor-2] ADDRESSED. normalizedCutoffTable loop adds #expect(got > 0) per route (~:256).
4. [Minor-3] ADDRESSED. VectorEntropyTests.swift:~155 #expect(v.entropy.isFinite) precedes the residual comparison.
5. [Minor-4] ADDRESSED. VectorEntropy.swift:30 states 8 * ulpOfOne * max(1,|reference|), which matches close() at EntropyRangeContractTests.swift:~74; line ~57 says "for nonzero ratios".
6. [Minor-5] ADDRESSED. "coprime-ish" removed; strided permutation added (~:187-194).
7. [Minor-6] ADDRESSED. @usableFromInline removed from entropyTruncatedContract; only callers are bodies of the @usableFromInline helpers, which are not @inlinable; report says it compiled and 30 tests passed in Debug and Release (logs not re-read by me beyond the report).
8. [Minor-7] ADDRESSED. Both legacy tests carry "Legacy approximate characterization" comments (VectorEntropyTests.swift ~:61, ~:205).

## Mutation C adequacy (item 3)
Adequate. Table totals are 1, so an input-cutoff mutation cannot fire there, as the implementer said. The tDomBelow/equal expectations are tiny (about ulpOfOne scale), so close() with 8*ulpOfOne absolute slack would accept a result of 0; got > 0 is therefore the only guard against dropping the retained dominant term. Mutation C (`p > cutoff && p < 0.99999`) failed exactly the 32 table assertions with the "dropped retained dominant term" message, plus 2 other tests. The first attempt (threshold 0.9999999999) was a no-op and was discarded and disclosed. Caveat: got > 0 does not detect a wrong positive value; close() covers that for the non-tiny rows.

## New breakage check
- Permutation: loop runs only for dim >= 8 (base.count 8): dims 8, 63, 64, 65, 128, 1536. Stride picked as first of 5,3,2,7 coprime to dim: 8->5, 63->5, 64->5, 65->3 (gcd(5,65)=5), 128->5, 1536->5. All gcd 1, so i -> i*stride mod dim is a bijection; the code also asserts seen.count == dim at runtime. It is non-cyclic (not a rotation) for all of these. True permutation for every tested D. The local `stride` shadows the global stride(from:) only inside this test, which does not use it.
- gcd helper: unique in module (grep), private, terminates.
- Comment at the stride site says "i -> (i * 5) mod dim" and then that the stride varies per dimension; slightly confusing, accurate.
- Doc claim discipline: VectorMath.swift:500 says "tested widths 1...1536", but tested widths are the set {1,4,8,63,64,65,128,1536}, not a range. Minor inaccuracy (see below).

New breakage: 0 Critical, 0 Important, 1 Minor.
- [Minor-N1] VectorMath.swift:500 "tested widths 1...1536" reads as a contiguous range; should say "tested widths 1, 4, 8, 63, 64, 65, 128, 1536" (testedDims, EntropyRangeContractTests.swift:54).

## Out of scope notes
- VectorEntropy.swift:30 is a long single line; may exceed any line-length lint (report says SwiftLint exit 0, so fine).
- Compile/test status rests on implementer logs (fix1-debug.log, fix1-release.log); I did not re-run.
