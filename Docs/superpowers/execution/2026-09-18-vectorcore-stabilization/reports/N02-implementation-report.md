# N02 implementation evidence (final v2)

Base: `84f42d01489a526b8ae40bd77fe7618c0de33f34`. Binding brief SHA256: `c8fcf29152ad6f1627a68533aa997f210807379d598629375786e70d5eb481fe`.

The public approximate-equality predicate and all three precision equality/conversion widths now reject invalid tolerance and nonfinite coordinates. Finite absolute comparisons retain the original scalar type, and shape rejection precedes either borrow. Distance accuracy widens subtraction, denominator and accumulation to Double; invalid batches return the complete invalid sentinel. safeLog first applies unchanged input handling and then its existing nonpositive-domain policy.

Only the three owned source regions and the new NumericalPolicyContractTests file changed. Full v2 diff and base/final SHA256 manifest are `N02-review-v2.patch` and `N02-review-v2-manifest.json`. `N02-archive-integrity-v2.json` checks all 1,899 tracked base files: none missing, no unauthorized changed paths. Frozen bytes and manifest are backed up at canonical `.stabilization-recovery/2026-09-23/N02/v2`.

## Behavioral RED and checks

`N02-command-evidence-final.json` records exact commands, exits and log hashes. All commands ran from the isolated N02 archive with at most six jobs.

- First RED compilation exited 1 because `0.5.nextDown` needed explicit `Float(0.5)`. This is a test-construction failure, not behavioral evidence; the full log is retained. Additional generic dimension cases were added while that initial build was running; the next run rebuilt the test file with stable bytes.
- Corrected behavioral RED exited 1: four tests executed and failed with 72,614 issues. It exposes NaN equality acceptance, invalid tolerance acceptance, overflowing distance differences and incorrect safeLog input staging. The full 12.3 MB log is retained losslessly as `N02-red-02.log.gz`, with compressed and uncompressed SHA256 values.
- Initial implementation GREEN exited 0: four contract tests passed.
- Final focused Debug exited 0: seven tests in two suites passed, including the three existing precision conversion/scalar-distance/batch-distance correctness functions.
- Release attempts 1 and 2 exited 1 before tests because Apple Swift 6.3.3 asserted in MandatorySILLinker (`cannot deserialize shared function`). Both full logs are retained. This is not a test failure or passing Release result.
- Two reduced compiler probes against the actual N02 Release module isolated the same default-subscript witness compiler defect previously identified in S02: the original faithful Double conformer exited -6, and the explicit faithful subscript exited 0. Neither timed out. Both used the original optimization flags; no production workaround or flag suppression was introduced. `N02-compiler-probe-results.json` retains exact commands and source hashes; associated .swift/.log files and the bounded probe driver are retained.
- Root authorized the one-line explicit subscript in the test-only conformer. That is the entire v1-to-v2 source delta. Both frozen revisions and their backups remain available.
- Final v2 focused Debug exited 0: seven tests in two suites. Final v2 focused Release exited 0: seven tests in two suites. No ASan run is necessary: this change adds no unsafe operation.

Existing test bodies and expectations were not changed. Inventory found no existing safeLog or handleNonFinite tests and no pre-existing public isApproximatelyEqual test usage; the existing PrecisionValidator correctness functions were therefore selected explicitly. Performance-profiler and unrelated precision suites were excluded. There are no child helpers or no-op child entries.

## Mechanical coverage

- Dynamic Float and faithful custom Double shapes 0, 1, 3, 4, 5 each execute a 10-value × 10-value × 8-tolerance cross product (4,000 cases per route). Values include signed zeros, finite ordinary/extrema, NaN and both infinities. Tolerances include negative, negative zero, zero, small positive, maximum finite, NaN and both infinities.
- An additional shared width harness exercises Dynamic shapes 0/1/3/4/5, generic Vector shapes 0/1/3/4/5, and optimized widths 384/512/768/1536: 14 routes × 800 matrix evaluations. Internal equality and actual FP16 conversion wrappers each execute 800 evaluations per supported width 512/768/1536 (conversion pairs intentionally repeat across the second-vector axis).
- Mixed nonfinite coordinates cover first, middle, last and all four SIMD lane positions, with both equality operand orders and retained input bits. The custom Double fixture checks native extreme subtraction and signed zeros. Faithful mismatched custom vectors record a failure if either read buffer is borrowed.
- Distance accuracy covers 800 scalar/single-row batch comparisons; all eight empty tolerance cases; bad pairs at every position in a five-row batch in both operand orders; mismatched length; exact threshold neighbors; zero-reference denominator floor; and a complete finite four-row summary with two analytically exact relative errors of 2 for opposite Float maxima.
- The independent safeLog oracle uses raw flag positions, Float-to-Double scalar logarithms and explicitly constructed error-index lists. It never calls production input handling. Shapes 3 and 512 have respectively 42 and 72 input fixtures, each executed against all 32 known flag combinations both with and without an unknown bit: 7,296 calls, plus two default-option calls. Fixtures include least subnormal, maximum finite, signed zero, finite negative, NaN, signed infinities, mixed error classes and negative-first/nonfinite-last. Output classification, finite logarithm tolerance, error family/indices, strict precedence and unchanged original bits are checked.

## Migration and limits

Matching nonfinite values no longer compare approximately equal; NaN or infinite tolerance no longer passes. The conversion default 0.001 is documented as absolute scalar error. Invalid batch rows now invalidate both summaries instead of silently skipping their contribution. Strict safeLog rejects NaN and positive infinity through NonFiniteError before any domain error; explicit replacements apply before logarithms, and default NaN replacement produces zero through the domain policy.

No dormant PrecisionValidator call site was activated, no numerical gate or provider changed, and the accepted N01 normalization routes remain unchanged. This evidence concerns bounded correctness contracts; it makes no performance or downstream validation claim.

## Final handoff

All current owned files match the frozen v2 manifest. Diff SHA256: `8a469f13f434291ccd855c638fd718682d506e6f2464c92caeb5e38debf3ccac`. Independent review accepted v1; root scoped review of the explicit test subscript and final evidence remains the acceptance step. `N02-evidence-inventory-final.json` hashes the complete retained N02 report set (excluding itself). Release-module hashes used by the reduced compiler probes are retained separately.
