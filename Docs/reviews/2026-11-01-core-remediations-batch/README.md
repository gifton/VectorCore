# AP7, LC5 and CI remediation batch

Five bounded audit items are repaired on `codex/audit-top-six-2026-10-27`: **A:AP7, A:LC5, A:CI2, A:CI3 and A:DP3**. The branch retains every commit through the starting revision `ec7f3323529ce3df6c126e8b6416f6e4f67d16ba`. Work remains local; no push, merge to main, tag or release was performed. The record uses the November 1 start date; validation continued on November 2.

## Changes

| Item | Result | Regression evidence |
| --- | --- | --- |
| AP7 | Precision analysis returns the existing zero-statistic FP32 sentinel when a nonempty sample has no finite observations, preserving the first vector's dimension. Zero expected error is a placeholder, not an accuracy estimate. | All NaN, both infinity signs, mixed specials and multiple vectors at 512/768/1536; empty/zero and later-finite controls. Baseline: 45 issues. Fixed: three focused tests pass. |
| LC5 | Both UMAP entries fit coefficients once after cheap validation and before graph/PCA/coordinate/fuzzy work; the private tail consumes those coefficients. | Three literal rejected coefficient fixtures across both entries and initialization routes; graph payload, immediate PCA provider and empty-edge witnesses. Baseline: 24 ordering issues. Fixed: six regression tests and 36 tests in seven UMAP suites pass. |
| CI2 | XCTest representative counts exclude skips and do not add repeated suite summaries. | Skipped-only fails; partial execution and mixed-framework runs pass. Swift 6.0/6.4 summary controls retained. |
| CI3 | All 27 build-mode-driver and 18 benchmark-contract-driver runtime assertions are explicit failures that survive Python optimization. | Negative command/log/report subprocess fixtures and valid controls under normal Python, `-O` and `-OO`; a scoped AST guard covers both drivers. |
| DP3 | Claim-ledger loading rejects duplicate decoded JSON keys recursively with ordinary R8 diagnostics. | Nested/top-level/escaped duplicates, separate-scope controls and both CLI modes. A description fallback also prevents stripped docstrings from crashing the `-OO` CLI. |

UMAP retains cheap config/count/schedule, supplied graph count, and supplied-coordinate shape/finiteness precedence. Vector payload validation remains downstream. Admitted controls retain small-spread and subnormal coefficients and seeded initialization. The coefficient mathematics and optimizer are unchanged. Tests observe graph/PCA/fuzzy work; coordinate allocation ordering follows adjacent source order. No universal spread threshold or timing bound is claimed.

Shared integration adds three test-backed ledger entries, records the public behavior in `CHANGELOG.md`, and refreshes the exact benchmark-driver hash after predicate review. No gate was weakened. Existing duplicate-entry-ID validation remains separate from duplicate JSON-key validation.

## Validation

Validated implementation: `f5a63053fd246e4b9b3a8f14c005889280d6d725`. Final full Debug, Release and Python runs all cover this same clean revision, including the explicit test-conformer witness, catch formatting and regenerated downstream-upgrade appendices.

| Check | Observed result |
| --- | --- |
| Full Debug | 1,750 Swift Testing tests / 263 suites, 18 skip events; 196 XCTest cases reported, 28 skipped; zero failures. |
| Full Release | 1,750 Swift Testing tests / 263 suites, 18 skip events; 196 XCTest cases reported, 28 skipped; zero failures. |
| Full Python policy suite | 576 tests passed. |
| Focused Python optimization matrix | 46 tests pass in each normal/`-O`/`-OO` run; 136 expanded normal-mode tests pass. |
| Final fixture-focused Debug | 39 AP7/UMAP tests in eight suites pass after defining the explicit collection witness. |
| Claim ledger / GitHub policy | Zero ledger problems; nine GitHub YAML/template files and workflow policies pass. |
| SwiftLint 0.65.1 | 5,458 warnings, zero errors. The three new catch-placement warnings were fixed. |
| Independent review | Three task reviews and one integrated review pass without actionable findings; formatting and explicit-witness follow-ups reviewed separately. |

Runs use the [qualified Swift.org 6.4.2 toolchain](../2026-11-01-compiler-qualification/README.md), macOS SDK 27.2, native SwiftPM and three build jobs. Debug uses its existing root cache; final Release uses the newly created `release-clean` scratch after the investigation below. Host: M3 Max, 16 logical CPUs, 48 GiB, macOS 27.0.1 arm64. Extended tests and strict-performance gates are off. The hard compiler canary remains enabled. Existing skips and unrelated build/lint warnings remain; these are not performance measurements or qualification of Xcode's bundled compiler.

## Compiler limitation exposed during validation

Both the existing-cache and clean Release builds at `104a01f` crashed in the compiler's MandatorySILLinker with `cannot deserialize shared function`, before running tests. Bounded outside-sandbox compilation probes isolate the new test vector's inherited `VectorProtocol` collection subscript: the literal conformer crashes, while an explicit subscript and `DynamicVector` compile. A smaller probe reproduces the same assertion by consuming that conformer as a `Collection`, without referencing UMAP; its explicit-subscript control compiles.

The test-only `LC5AccessVector` now defines `subscript(index: Int) -> Float { storage[index] }`, matching the explicit-witness pattern of existing fixtures. Its values, raw-buffer access counter, and all stage assertions remain. Production UMAP, compiler flags and test gates were not changed to accommodate the compiler. This repairs the fixture's Release build, **not the compiler's imported-default-witness failure**. That limitation remains recorded for follow-up. The exact failing SIL symbol and a reproduction against the pre-remediation library module were not established.

`compiler-investigation.md`, `witness-review.md` and `compiler-evidence/` in the archive retain the probes, commands, compiler/module/source hashes, crash logs and passing controls. This new evidence narrows the prior qualification to exercised source paths; it does not establish arbitrary custom-conformer compilation.

## Review and evidence

Three isolated workers implemented disjoint source changes concurrently. Fresh task reviewers checked specification and quality; the coordinator integrated their commits sequentially and ran combined validation, repeating affected checks after the integration and compiler-fixture findings. The final reviewer independently compared every replaced Python predicate/message structurally, verified the byte pin, and checked that the ledger and release notes stay within the evidence.

[Plan](../../superpowers/plans/2026-11-01-core-remediations-batch.md), [validation summary](validation-summary.json), and [evidence manifest](evidence-manifest.json). Raw commands, source hashes, red/green logs, review reports, runner scripts, integrated source archive and source snapshots are retained at `/Users/goftin/dev/projects/VectorCore-remediations-2026-11-01-evidence`. The earlier audit and closure documents remain unchanged history; `original-known-issues.json` copies the five relevant records with their source identity.

Initial worker sandbox manifest failures are retained separately from runtime regression failures. The redundant expanded optimized Python wrapper was intentionally stopped after its normal child passed; wrapper exit 143 is not represented as a passing optimized run. The completed 46-test optimization matrix supplies that evidence. The first full policy run failed only because release-note insertion made the generated downstream appendices stale; regenerating those sections resolved the failure. The initial integrated lint result and its three warnings are retained beside the corrected result. The first Debug recorder counted XCTest and Swift Testing skip lines together; final counts exclude `Test Case` lines, and the old recorder and original result remain unchanged in the archive.

This batch adds local macOS arm64 evidence. It does not add hosted CI, device/Intel/downstream execution, sanitizer qualification, controlled performance results, or a real four-build benchmark replay. Remaining work includes the compiler limitation above and the separately deferred child-launcher/timeout and broader test-hardening batch; completing these five repairs is not a declaration of overall 0.4.0 readiness.
