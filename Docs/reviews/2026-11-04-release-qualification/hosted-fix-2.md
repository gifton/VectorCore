# Hosted sanitizer dependency search repair — wave 2

The reviewed candidate `90b8d583055fffa729558fb10b96eb7e30e92fda` passed runtime alias discovery in hosted run `41940013803`, then both sanitizer jobs failed before their first test body. The direct helper's environment contained the selected platform's public framework root but omitted its private framework root. This repair supplies both roots, matching the framework roots recorded in those jobs' successful build commands. It does not change manifest selection, runtime selection, helper protocol, sanitizer options, deadlines, or result gates.

## Observed failure and scope

Root retained source-bound hosted logs and downloaded the run's artifacts. The ASan job was `126958111615`; its `body-000.json` reports exit `-5` after 0.8215 seconds and empty stdout. The raw stderr names `swiftpm_testing_helper/Entrypoint.swift:34`, failure to open the aggregate test bundle, and the missing `@rpath/XCTestCore.framework/Versions/A/XCTestCore` dependency of the selected platform's `Developer/usr/lib/libXCTestSwiftSupport.dylib`. The TSan artifact has the same missing dependency and exit `-5`. Neither artifact shows a completed test body. This is launcher/dependency discovery failure, not a workload sanitizer finding.

The logs already show the selected helper, sanitizer, bundle, Swift Testing, and XCTest images loading before the failure. Both successful build logs contain `-F` entries for the selected Xcode 26.3 platform's `Developer/Library/Frameworks` and `Developer/Library/PrivateFrameworks`. The direct-helper environment carried only the former. The downloaded artifact binaries were not executed.

The selected local Xcode installation places XCTest in the public root and XCTestCore plus XCTestSupport in the private root. Local image hashes and `otool -L` records are retained as corroborating local layout evidence; they are not evidence that the hosted installation has identical binaries. The repair adds the selected private root, rather than individual XCTest dependency names or unrelated Xcode roots.

## Changes

`Scripts/ci/check_sanitizers.py` now constructs `DYLD_FRAMEWORK_PATH` from the selected platform's public and private roots, in that order. Existing selected-toolchain Swift/testing library paths and the selected platform's `Developer/usr/lib` remain unchanged. Canonical runtime ambiguity rejection, exact helper invocation, filters, started/passed checks, loaded image gates, and Q03 checks remain unchanged.

`Scripts/ci/tests/test_sanitizer_log.py` checks both roots alongside the existing all-manifest command/options checks. A new case checks both sanitizer modes, platform paths containing spaces, and replacement of an unrelated inherited framework path. It does not assert that a mocked filesystem proves native loader behavior; the native evidence below separately exercises that boundary.

No production Swift, metadata fixture, package, workflow, manifest, or toolchain requirements changed. In particular, the wave 1 ordinary-architecture metadata reader remains untouched.

## Verification

- Mechanical red/green: the original driver failed three assertions across nine tests (`regression-red.*`). With the repair, all nine pass; the final test source is recorded by `regression-final-source.json` and `regression-final.*`. Six CI contract tests and 21 SwiftPM product tests also pass, for 36 direct Python tests total.
- Real native dependency-chain control: locally compiled a public framework depending on a private core framework, itself depending on a second private support framework. Their install names use `@rpath`, with no embedded fallback root. A local executable uses `dlopen` on the public image and calls through the chain to obtain 42. With the original driver's environment, both ASan and TSan runs fail with exit 92 and the missing private-core diagnostic, without a success marker. With the repaired environment, both exit 0, print `TRANSITIVE_FRAMEWORK_CHAIN_PASS value=42`, and show the public, private-core, private-support, and selected sanitizer image in dyld output. Compiler commands, sources, `otool` dependency output, binaries, hashes, and both configurations' raw logs are retained. This demonstrates the transitive framework search class using actual native loading under both selected runtimes.
- Existing qualified bundle coverage: the exact first manifest memory-pool body passes under ASan and TSan; the Q03 TSan parent and its actual child pass with all six workers completing 2,000 loops, stable-total and atomic-batch join markers, 45-second child budget, matching bundle/helper hashes, and PID-specific helper/bundle/runtime dyld lines. The mandatory manifest still contains 77 ASan and 39 TSan rows. Exact filter command checks cover every row.
- Negative diagnostics remain effective: the retained intentional heap overflow and data race executables each exit `-6` and report `AddressSanitizer: heap-buffer-overflow` and `ThreadSanitizer: data race`, respectively, with the selected runtime observed. ASan remains `detect_leaks=0:halt_on_error=1`; TSan remains `halt_on_error=1`.
- `git diff --check` passes. There was no full package rebuild or repeat of the prior full 77/39 manifest qualification.

The reused bundles and diagnostic controls are the prior qualified Swift.org 6.4.2 artifacts. All 386 remaining package inputs match their original source hashes; the sole excluded input is the unrelated wave 1 `Tests/BenchmarkContractTests/RunnerContractTests.swift` fixture repair. This is bounded covering evidence for the changed loader environment, not a fresh build of all current tests. Actual helper, bundle, runtime and diagnostic hashes are recorded in `direct-checks-result.json`.

Local environment: macOS 27.0.1 arm64, qualified Swift.org 6.4.2 payload and installed Xcode SDK 27.2. Qualified frontend SHA-256: `956438a61893525d7ce28c21b6634093f15b0c4bae7fe6eb7d98fba0c1c5e301`; SDKSettings SHA-256: `82165b15ee21419d4cffaba8895797391b908ac9499ba70f561f3e1bd16758d6`. The final source identities and unchanged manifest SHA-256 are in `after-identity.json`.

## Evidence and remaining qualification

Evidence root: `/private/tmp/vectorcore-qualification-2026-11-04/hosted-fix-2`.

- `input-provenance.json` binds the retained hosted JSON/raw stdout/stderr/build records to their original artifact paths and SHA-256 hashes. `inputs/*/framework-build-lines.txt` extracts the successful build's public/private framework search entries.
- `native-fixture-identity.json`, `compile-*`, `dependencies-*`, `before-native-*`, and `after-native-*` retain the native control and observed loader results.
- `direct-checks-result.json`, `direct-address/`, and `direct-thread/` retain actual bundle/runtime/host and diagnostic evidence.
- `regression-red.*`, `regression-final.*`, `ci-contracts.*`, `product-contracts.*`, and source snapshots bind focused test results to implementation state.
- `archive-inclusion.json` lists the relative evidence files and their hashes; no build cache is included.

The next reviewed hosted matrix must execute all mandatory bodies and downstream jobs before this can be called hosted qualification success. Local native loading demonstrates the repaired search-root contract; it cannot substitute for Xcode 26.3 hosted execution. No extra failure class was reported in either sanitizer artifact. Existing full-matrix jobs remain running under the parent's coordination, and this repair does not waive any of their gates.
