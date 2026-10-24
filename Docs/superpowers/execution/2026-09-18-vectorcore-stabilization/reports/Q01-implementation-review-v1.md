# Q01 implementation review package v1

Source and test bytes frozen in `Q01-review-v1-manifest.json`; complete patch including all new files in `Q01-review-v1.diff` (SHA256 `666fd8223cc25be988e338b42aac8e0fa6002dfc9f1be54f99a6740c5de72970`). Base `0dbb18c818f26edd8e28bb303a6fccedbd063a6f`. Isolated archive `/private/tmp/vectorcore-stabilization-Q01-implementation`. No git/index/commit/sibling changes. This package is ready for independent code review; Release and ASan evidence are pending and acceptance is not claimed.

## Implementation

Shared zero-inclusive/asymmetric and symmetric calibration uses Double, preserves requested metadata, clamps new scale to the exact-product finite-decode cap, and uses least-subnormal rather than collapsing small ranges. The original five Codable keys are preserved; decoding validates finite positive scale, finite ordered bounds, and symmetric zero point. All Int32 asymmetric offsets remain accepted. Conversion uses the same checked Double division/round/clamp route in scalar, packed and SoA paths. Decode projects Double affine coordinates once to Float.

Metrics retain the compiled symmetric integer loops. Cosine cancels positive scales; asymmetric dot/squares use signed two-word exact centered accumulation and magnitude-first conversion. Dot scales once in Double. Euclidean's equality gate is exact, and both fallback affine products retain FMA residuals through error-free expansion subtraction. Mixed nonfinite values bypass finite-only expansion transforms. SoA capacity is checked before work and only the logical prefix is written. Quantized mutable borrowing validates only after a normal body return and publishes once; throws retain original storage/parameters, including legacy Float-overflow tuples.

The shared difference helper grows a four-term expansion with two-sum; this implementation does not assert universally correctly rounded integer conversion or metric results. The acceptance evidence is the bounded fixture tolerances from Q01.md. Optional C branches were not enabled or verified.

## Behavioral RED and failed attempts

- `Q01-red-metrics-unsandboxed.log`: 33 expected assertions failed across all three widths before source edits. Includes the21 EX04 witnesses, tiny-scale cosine and exact cancellation fixtures.
- `Q01-red-calibration-outlier.log`: separate small-scale/endpoint/finite-code calibration failures and finite-outlier child trap (`Float value cannot be converted to Int because it is either infinite or NaN`). Child output preserved separately.
- `Q01-red-final-rounding-2.log`: two one-bit failures in the retained symmetric scalar path before widening final sqrt/scaling.
- `Q01-red-metrics.log`: initial sandbox compiler-cache failure; rerun with approved Swift test access.
- `Q01-green-1.log`, `Q01-green-2.log`: generic storage map and missing explicit closure-return compile failures; corrected by indexed storage iteration.
- `Q01-green-3.log`: test generic Scalar/Float literal compile failures; corrected constraints/literals.
- `Q01-green-4.log`: JSON byte-order assertion failure; test now uses sorted-key encoder. All21 EX04 and cancellation cases already passed, as did the named rejection matrix.
- `Q01-red-final-rounding.log`: Float literal compile failure before actual rounding RED.
- `Q01-green-5.log`: exploratory broad filter accidentally included the comprehensive performance group; this run is explicitly **not performance evidence**. It also exposed the approved existing nonfinite-policy migration, and incorrect new-test SoA fixture assumptions. No timing or throughput conclusion is drawn.
- `Q01-green-6.log`: new Q01 suites passed after correcting the reachable SoA codes and limiting powers-of-two mixed Float fixtures to finite inputs.

## Approved expectation migrations

Root approved both changes before editing the existing suite:

1. Negative-only asymmetric calibration formerly expected `(max-min)/255`; for [-100,-10], zero inclusion requires 100/255 rather than90/255. The parameterized expected-scale expression now includes zero.
2. `testNaNInfQuantization` formerly called the nonthrowing `from` initializer on NaN/Inf and asserted only output length512. It now tests the throwing array initializer, specifically `.invalidData` and indices `0,1,2`. Nonthrowing entry points are checked in bounded child processes with intended diagnostics.

No other existing test expectations changed.

## Coverage, distinguishing functions from cases

`Q01-focused-debug-1.log` passed with17 XCTest methods plus50 Swift Testing functions in12 suites (exit0). One of the latter is the parent-process no-op `Q01ChildContracts.childProbe`; actual child executions are separately observed by the two parent tests. Exact compiled symbol inventory is `Q01-test-inventory.log`.

- Calibration: initial signed/positive/negative/cross-zero/constant/all-zero intervals plus43 magnitudes (subnormal/normal boundary neighbors, max and deterministic exponent sweep),5 interval shapes and both modes. Each new tuple checks all256 code projections and finite endpoint bounds. Both calibration APIs agree for asymmetric cases. Signed-zero construction metadata and canonical empty paths are checked.
- Parameter decode:13 invalid tuples, each directly and inside512/768/1536 vector payloads;15 valid legacy scale/offset combinations including both Int32 extrema; scalar/SIMD projection bits for all256 codes. Valid legacy full vectors round-trip at all widths with exact raw bytes and individual parameter fields; existing S01 tests preserve wire structure.
- Conversion: all widths, four explicit parameter families, exact stored half ties/neighbors and finite extreme saturation; counts0...5 for SoA padding and parameter preservation. Array/repeating/from equality covers zero, least-subnormal, ±1 and max. Throwing invalid arrays verify `.invalidData` plus actual first/last invalid indices; shape precedence is checked separately.
- Borrow: all widths, finite normal write, no-op recalibration, retained-copy independence, rollback on sentinel throw after finite/NaN/±Inf writes, legacy-overflow throw rollback. Normal-return invalid and legacy-overflow rejection use children.
- Child matrix:99 intended rejections plus1 finite-outlier success, all file-captured with15-second deadlines and owned-PID cleanup. Nonfinite scalar/scheme/bounds, eight all-width routes ×three nonfinite values, legacy no-op rejection, both capacity routes, percentile invalids and unsupported modes are checked against named diagnostics. No deliberate traps are selected for ASan.
- Metrics:21 EX04 witnesses across512/768/1536; explicit concrete wrappers and generic kernels; self/opposite/orthogonal/zero/cancellation; symmetric/asymmetric/Int32-extreme offsets; tiny/large/equal/unequal scales. Mixed orders and both batch routes are checked. Counts0...5 ×both batch routes ×three widths exercise logical prefixes/sentinels.
- Independent exact fixtures: required dot cancellation, candidate scale1 and2^100 plus neighboring exact-dot1 controls; required affine difference−1 and mixed difference+1 over five powers-of-two scale factors and all widths;96 deterministic arbitrary-dyadic Fraction fixtures for the shared difference helper; positive/negative wide sums/carries/borrow/cancellation, plus negative centered-dot extrema. Provenance and generator: `Q01-exact-oracle.json` and `Q01-exact-oracle.py`.
- SoA exact fixture limitation: the nearest Float to17944036190978560 quantizes to code2 under the required candidate tuple, so code0 cannot be manufactured via that supported conversion. Its SoA test independently expects difference−16777221, while scalar retains the raw-code-zero difference−1. The mixed difference+1 fixture is reachable as SoA code0 and is tested. No unchecked constructor was introduced.

## Pending verification and limits

Focused Release is building in `/private/tmp/vectorcore-Q01-release`. ASan valid raw/SoA/borrow and S01 controls will follow in a distinct scratch. Final evidence will refresh logs and source hashes without replacing the frozen v1 artifact. No generic VectorProtocol metrics, N01 helpers, downstream/GPU runtime, throughput, scale readiness, optional C path, tuner or accuracy-analysis redesign is claimed.
