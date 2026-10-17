# T02 independent review (Opus) — root-recorded copy of the reviewer's returned text

Spec: PASS
Quality: APPROVED (Minor findings only; none blocks integration)

All three classifications are correct; no production regression; the derived bound is mathematically correct; cited coverage exists and ran in both whole-suite logs. One report claim is false as stated (Minor-1).

## Independent checks
- The staged workspace diff is byte-identical to t02-test-migration.diff (sha256 24321742…cc50). The index mtime equals the review-package (root packaging); only the two test files changed.
- NormalizationCore.swift (96090efc…) and VectorNormalization.swift (3d4d23c8…) match N01's frozen hashes. AffineQuantizationCore.swift, QuantizationRangeContractTests.swift and QuantizedKernelsComprehensiveTests.swift match Q01's frozen hashes. The only post-Q01 change to QuantizedKernels.swift (813255c) is the tuner calibration cache.
- Exact simulation of Float(Double(x)/sqrt(KahanSum)) reproduces the head histogram 200 x 0x45880796, 184 x 0x45880797; max coordinate error 0.49913 ulp (384/384 correctly rounded). Q01 calibration, codes and E(x) recomputed for all three two-valued fixtures.

## Classification
- F1 (a): requireFinite is the first statement of init(from:params:) (QuantizedKernels.swift:93); base encoded NaN/+Inf as 127 and -Inf as -128 (base-legacy-and-diag-debug-2.log:173-175).
- F2 (b): ratio1 0x45880796 is the correctly rounded ||x|| (||x|| is 0.3766 ulp above it); base was biased (308 x 0x45880797, 76 x 0x45880798; max error 2.056 ulp); 1e-5 is 0.0205 ulp.
- F3 (a): scale 0x3b008715, zero -128, both codes 127, separation 1.00017e-4 = 0.051 step; the fixture is "Nearly constant".

## Failure details
- F1: namedRejectionMatrix (QuantizationRangeContractTests.swift:128-151) covers from/supplied x 512/768/1536 x NaN/+Inf/-Inf in bounded children, asserting .uncaughtSignal plus the named diagnostic; it passed at suite-debug.log:6831 and suite-release.log:6670. Equivalence to the removed mixed-position probes rests on the guard being position- and params-independent and running before params are read (visible in source). Indices come from construction rules; counts 155; 51/128/256; 171; 103 were recomputed. No finite assertion was dropped.
- F2: every step of the analysis holds (exact Kahan sum 18948160 < 2^53; measured α = 2.9e-17; spread ≤ 4u/(1−u²) + 2.0001·2^-53; computed tolerance 0x3a88079f = 2.1255 ulp; Sterbenz holds; NaN/Inf fail closed). 2-ulp differences pass, 3-ulp differences fail.
- F3: E(x) is faithful to testCalibrationEndpoints (the omitted tolerance.isFinite is equivalent here); the triangle-inequality argument is sound; resolvable pairs require == 2 (stronger than the old >= 2); margins 127.5x / 127.5x / 39.22x.

## Whole-suite totals (log sha matches the run records)
XCTest Minimal 59 executed/27 skipped, Comprehensive 49, BenchmarkContract 20 (128 executed, 101 not skipped). Swift Testing Comprehensive 1432/196 suites (18 skipped), BenchmarkContract 1, Minimal 0. Release identical. No fatal error, signal, recorded issue or failure. Lint has the same rule/message multiset; policy 44 OK / 1 skip before and after.

## Class closure
The whole-suite gate is a sufficient detector for default-enabled tests, with three gaps: (a) EXTENDED-gated tests (18 Swift Testing + 27 XCTest) never run; five exercise INT8(from:) with explicit params (QuantizedKernelsTests:2084/2201/2319/2440, QuantizedKernelsComprehensiveTests:948); a periodic serial EXTENDED=1 run would close this. (b) One trap masks all later results; a static policy scan is a heuristic complement. (c) Print-only legacy code cannot catch contract drift (the "Very small range" fixture now decodes to 1 value, asserted nowhere).

## Findings
- Minor-1 (report.md:15): "A misplacement of 3 ulp or more in either ratio fails" is false; the correct statement is that the test fails iff pair (1,2) or (2,3) differs by >= 3 ulps. The code comment is correct.
- Minor-2 (ArbitraryDimensionTests.swift:110-114): the bound is weaker than the derivation allows and passes at base. Optional strengthening: |ratio_i − norm| ≤ 2u(1+2^-20)·norm over all 384 indices (rejects base's 76 indices at 0x45880798; head max 0.62 ulp).
- Minor-3 (QuantizedKernelsTests.swift:4482): an unconditional print uses the expected count.
- Minor-4 (QuantizedKernelsTests.swift:4393, :4641; ArbitraryDimensionTests.swift:101): historical base claims cite no test or in-repo artifact (confirmed against T02 evidence).
- Minor-5 (QuantizedKernelsTests.swift:4595): it is the ideal scale; the stored scale is rounded and clamped. Wording only.
- Minor-6 (progress.md, not code): three instances across two tasks (Q01 x2, N01 x1).
- Observation: the sub-resolution (1...2).contains(count) check only catches position-dependent decoding; the real check is the per-coordinate endpoint assertion at :4607-4609.

## Cannot verify from diff
That EXTENDED=0 was literally exported (behaviorally equivalent; the gates are == "1"); base numbers (from retained logs; not rebuilt); no negative control was executed for the new assertions; extended-gated behavior under Q01/N01; no artifact retained for the "10 candidates" scan; RED was captured in Debug only (Release from milestone11/release.log).
