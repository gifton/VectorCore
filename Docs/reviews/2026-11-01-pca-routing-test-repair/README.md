# PCA routing-test repair — T19 follow-up

The routed pre-0.4.0 test hang is repaired on `codex/audit-top-six-2026-10-27`, starting from `dc4cb0b`. This closes **C:T19:routed:1 / G:A-018**, recorded in the historical [T19 closure](../../superpowers/execution/2026-09-18-vectorcore-stabilization/closures/T19.json). The earlier closure remains unchanged as history.

## Change

`PCACenteringContractTests.sketchBoundariesAndProviderIdentity` deliberately supplies `Int.max` power iterations when the sketch width should select exact PCA. Its original provider recorded calls and checked them only after `fit` returned. Bypassing the exact branch therefore entered the enormous randomized iteration loop before any assertion could report the wrong route.

The test provider now checks each call against the expected sequence **before forwarding** to the real Swift QR/SVD implementation. An unexpected method, shape, order or excess call throws a diagnostic; the final equality still catches missing calls. All six oversampling values, `Int.max` inputs, invalid-configuration assertions and provider-error identity controls remain. The eigen entry is checked too.

Only the test helper and its use changed. Production PCA, public API, compiler flags, test gates and time budgets are unchanged. This bounds unexpected provider calls for these fixtures; it does not promise termination for arbitrary loops that never reach a provider.

## Validation

| Check | Observed result |
|---|---|
| Baseline PCA / T19 suites | 25 tests in six suites pass. |
| Old test, exact branch bypassed | Test starts but reaches the 15-second execution deadline; direct helper is killed and reaped. |
| Repaired test, identical mutation | Fails on the first unexpected QR: `expected svd:6:4, got qr:6:4`. Test reports 0.001 seconds; process exits 1 after 0.514 seconds, without timeout. |
| Full Debug | 1,742 Swift Testing tests / 262 suites, 18 skip events; 196 XCTest cases reported, 28 skipped; zero failures. |
| Full Release | Same counts and skips; zero failures. |
| Python policy suite | 565 tests pass. |
| Claim ledger | Zero problems. |
| SwiftLint 0.65.1 | Exit 0; 5,458 warnings, zero serious violations across 359 files. |
| Independent source review | Approved; no findings. Lead verified final test results after the review. |

Both full suites use the previously [qualified Swift.org 6.4.2 compiler](../2026-11-01-compiler-qualification/README.md), macOS SDK 27.2, macOS 27.0.1 arm64, native SwiftPM, three jobs, existing build caches and normal Debug/Release optimization. Extended and strict-performance gates are off. Both include the repaired PCA test and the unchanged hard lost-store canary. Existing skips and compiler/build warnings remain. These results do not qualify Xcode's bundled compiler or additional platforms, hosted CI, downstream packages or performance.

The mutation changes only `if sketchWidth >= min(n, d)` to `if false` in the actual PCA implementation. Its source hash is identical in the before/after test experiments. Each run restores the original production file in `finally`; final full runs and policy checks have identical before/after source snapshots. The observed failure time is diagnostic evidence, not a timing assertion added to the suite.

The first reproduction harness attempt timed out the SwiftPM parent while its separately grouped test helper survived; that helper was explicitly terminated. Two direct-helper attempts then failed to load test runtime libraries. These attempts remain in the evidence with their failure classifications. The accepted red/green pair supervises the helper directly with the inspected Swift Testing and Xcode test-support library paths. No helper remained after the accepted timeout cleanup.

## Reproduce and evidence

Select the compiler and SDK using the qualification instructions, then run:

```sh
swift test --build-system native --jobs 3 -c debug --sdk "$VC_SDK" --scratch-path "$VC_CHECK_DIR/debug" --filter 'PCA|T19'
swift test --build-system native --jobs 3 -c release --sdk "$VC_SDK" --scratch-path "$VC_CHECK_DIR/release" --filter 'PCA|T19'
python3 -B -m unittest discover -s Scripts/ci/tests -v
python3 -B Scripts/ci/check_claim_ledger.py
```

The retained `check.py` contains the exact full-suite and bounded mutation commands, including SDK, scratch paths, compiler identity, environment and automatic source restoration. Do not execute the old unbounded mutation test without a watchdog. [Validation summary](validation-summary.json) records source hashes and counts. Raw logs, source patches, runner versions and review are retained at `/Users/goftin/dev/projects/VectorCore-pca-routing-repair-2026-11-01-evidence`; [evidence manifest](evidence-manifest.json) hashes that archive.
