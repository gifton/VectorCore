# iOS test-support portability and child sanitizer propagation

The actual `VectorCore-Package` Release scheme builds all three test bundles for the existing iOS simulator. The complete first execution reached real test bodies and exposed the existing Xcode lost-store canary failure plus five test-support capability mistakes. Those capability mistakes were repaired at test entry points; the final focused outcome is recorded below. This is qualification evidence, not release acceptance.

## Changes and platform reachability

`ChildProbeTerminationReason` replaces Foundation's unavailable-on-iOS nested `Process.TerminationReason`; exit/signal diagnostics retain raw values 1/2. Portable argument selection, marker checks, result reads and unreaped-result errors remain compiled and tested on iOS. Process creation/replacement and the reaper implementation are compiled only for macOS; unsupported launch attempts throw an explicit error rather than silently succeeding.

The [capability inventory](capability-inventory.json) names 117 methods and their call chains: 92 Swift Testing methods have individual capability traits, and 25 XCTest methods skip at entry before any assertion or expected-error handler can consume `XCTSkip`. The shared XCTest launch boundaries retain a defensive skip. No test target or whole numerical suite is excluded. The inventory distinguishes 34 process-isolated configuration/routing scenarios, 56 other isolated probes, 21 benchmark process contracts and six process-containment contracts. Isolated numerical scenarios remain unavailable on iOS when their existing contract requires fresh process-global state; they were not changed to concurrently mutate shared state. Ordinary in-process numerical tests remain reachable.

A second defect appeared during broader sanitizer qualification: dyld had preloaded the parent's sanitizer, but `ProcessInfo.environment` no longer contained `DYLD_INSERT_LIBRARIES`. Generic probe children then aborted before their bodies. The launcher now discovers the actually loaded macOS ASan/TSan image and restores that preload. It preserves unrelated caller variables, runtime options and library order; matching explicit preload is unchanged, filesystem aliases deduplicate, conflicting explicit recognized macOS ASan/TSan runtimes and ambiguous loaded runtime identities throw. Zero loaded sanitizers leave the caller environment unchanged. This applies to both selector paths immediately before their common spawn, and does not alter production metadata process policy or marked-child replacement environments.

## Evidence and verification

Raw evidence: `/private/tmp/vectorcore-release-phase-two-2026-11-04/ios/`. Every build/test command uses `Scripts/ci/v03_process.py` with bounded deadlines, fresh command records, selected environment, executable/output hashes and source input manifests. Compiler/SDK hashes are in `compiler-sdk-identities.json`; artifact/source inclusion is in `archive-inclusion.json`.

- `red-launcher-ios`: actual iOS 17 target / iPhoneSimulator 27.2 SDK typecheck fails on `Process`; `green-launcher-ios` passes after the repair. Prior phase's failed package-scheme logs are retained as `prior-package-failure.*`.
- `red-portability-contract` fails on the two leaked result-type references. `green-policy-5` passes all 28 covering launch-policy/process-bounds tests, including native-operation compilation boundaries and individual XCTest entry checks.
- `native-launcher-1`: exact copied launcher/support/test source passes 15 native tests under the verified Swift.org toolchain, including nested-launch refusal and actual exit/SIGTERM/SIGKILL descendant containment. `green-native-launcher-2` passes the expanded 19-test suite.
- `red-sanitizer-environment`: the identity/no-repair control fails six assertions across two tests. The broader worker's `child-red-address` and `child-red-thread` captures independently show actual child startup failures. `native-address-2-testing` and `native-thread-2-testing` each pass 19 Release launcher tests, and their `*-xctest` companions each pass a real XCTest parent/child launch. Captures include actual child image paths and options; `sanitizer-focus-verification.json` verifies exits/counts and absence of sanitizer failure diagnostics. `ASAN_OPTIONS=detect_leaks=0:halt_on_error=1` and `TSAN_OPTIONS=halt_on_error=1` are retained; this is not leak-detection qualification.
- `build-ios-discovery-1` demonstrates that Xcode Release test bundles also require `ENABLE_TESTABILITY=YES`. `build-ios-discovery-2` passes with that setting while retaining `-O`, all three test bundles, the deployment floor and normal assertions.
- The complete simulator execution is `simulator-full-suite-1` and `full-suite-1.xcresult`. Source archive `source-0800986.tar` matches all 403 tracked package/Swift/C/header input hashes recorded for that build (`full-suite-source-match.json`). Actual tested binaries are retained under `artifacts/simulator-full-suite-1/`.

The complete first run has 1,970 declared test cases: 1,807 passed, 157 skipped and six failed. Expanding 49 parameterized methods into 1,678 argument runs gives 3,436 passes, 157 skipped case entries and six failures. These are different counting levels, not independent runs. Exact cases, parameter runs, skips and failure messages are retained in `simulator-test-tree-1.stdout`, `simulator-method-results-1.json` and `simulator-counts-1.json`.

| Bundle | Passed cases | Skipped cases | Failed cases |
|---|---:|---:|---:|
| MinimalTests | 33 | 27 | 0 |
| ComprehensiveTests | 1,694 | 114 | 1 |
| BenchmarkContractTests | 80 | 16 | 5 |

The real compiler failure is `LostStoreCanaryTests.literalArrayStoreIsObservedAfterOutOfLineClosure`: observed 2.0 where the source requires 1.0. The five benchmark failures were `testActualBatchProviderReturnFamily`, `testBadObservationSurvivesLaterGoodResultsAndReference`, `testFixturePreservesEnvironmentAndMarksChild`, `testMetadataEnvironmentReaderRejectsFailedBuild`, and `testMetadataBoundedCommands`. Four consumed a helper-thrown skip inside assertions/error checks; the fifth directly called unavailable host metadata commands. Entry skips and the mechanical entry-check test close that capability-reporting class.

## Final focused simulator result

`simulator-focused-3` / `focused-3.xcresult` passes at source `2bd32ab1ffd626665f90de2dd2d328e2e2243765`: 120 cases, 93 passed, 27 explicitly skipped, zero failures. The complete benchmark bundle contributes 80 passes/21 skips; the launcher contracts contribute 13 passes/six skips. Portable termination, unreaped-result and capability bodies are named passes in the structured result. All five formerly failing benchmark methods now report their platform capability skips at entry. This focused invocation selects the benchmark bundle and launcher contracts; it is not a replacement full-suite run and does not erase the retained compiler-canary failure.

`source-2bd32ab.tar` and `focused-source-match.json` match all 403 tracked package/Swift/C/header input hashes for that invocation. The final uninstrumented native source also passes 19 launcher tests (`native-final`). Actual focused simulator and final native products are retained in `artifacts/`.

## Retained failures and limitations

The failed complete `.xcresult` is retained unchanged. The first focused rebuild exposed the new sanitizer test's missing explicit `MachO` import on iOS; the import was added and actual iOS typechecking then passed. The first ad-hoc full-file typecheck omitted the Testing macro-plugin path; that invocation's failure is retained before the corrected invocation. Initial XCTest sanitizer fixture assertions compared `/private/tmp` to Foundation's canonical `/tmp` spelling; both children loaded the correct runtime, and canonical comparison fixed the qualification fixture without changing the launcher. All failed records remain present.

Full simulator execution used Xcode's installed compiler on an arm64 iPhone 18 Pro simulator running iOS 27.0 (build 24A434), SDK 27.2, and deployment target iOS 17.0. It does not establish minimum-OS execution, physical-device, Intel, tvOS, visionOS or Catalyst coverage. Native instrumented fixture evidence is scoped to the support revision before the final explicit test import; the support implementation itself is unchanged by that import. Root owns final combined-source iOS/sanitizer reruns after benchmark changes settle. No compiler canary, optimization, sanitizer diagnostics, ordinary deadline, public library API or production source was weakened.

The final reporting commit changes only this report and narrows one capability comment; executable verification remains explicitly bound to `2bd32ab` (and the earlier named sanitizer revision), not to unbuilt combined source.

## Review fix round 1: keep arithmetic/copy controls portable

Review I1 found that `concreteProtocolOnlyOperators` combined subprocess mismatch probes with a valid local arithmetic/copy-preservation loop. The mismatch loop is now `concreteProtocolOnlyMismatchesRejectBeforeBorrow`, retaining its macOS capability trait and 24 route/length combinations. The unchanged valid `+=`/`-=` checks for lengths 0, 1, 2, 3 and 2048 now run in the ungated `concreteProtocolOnlyOperatorsPreserveCopy`. The inventory identifies that portable companion. A mechanical positive-control check rejects missing discovery, conditional test traits, enclosing compile gates, capability checks and direct child calls; its RED control failed before extraction, then the affected 17-test policy suite passed.

Evidence is appended under `fix-round-1/`, with a separate inclusion/verification manifest. The original inclusion and verification manifests remain byte-identical. Verified Swift.org native Release execution passed both affected methods. The actual iOS Release run selected exactly those methods: the portable body passed, the mismatch method explicitly skipped, and no method failed. The existing Xcode whole-module test target was rebuilt as required for the changed source; no complete suite was executed in this fix round. Combined-source execution still awaits coordinator integration.

For M1, qualification retains existing diagnostic noise: Accelerate `cblas_sgemm` deprecations in `MatrixMultiply.swift:44` and `MatrixDistance.swift:324`, the weak-variable warning in `BufferPoolOwnershipTests.swift:42`, other existing test deprecations, and build-system platform/App Intents/signing notices. `fix-round-1/existing-diagnostics.json` identifies representative retained log lines and hashes. No warning suppression or unrelated cleanup was applied, and these runs are not described as warning-free.

## Combined-source complete iOS qualification

The final combined Swift/package/benchmark inputs were tested at `1cb903188633926c27cb0300958c2d35e9641ec0`, after the approved review fix. A fresh `VectorCore-Package` Release build completed successfully in 311.9 seconds; a separate complete execution finished in 94.8 seconds with exit 65, no outer timeout, and one genuine failing canary. No test selector, target exclusion or canary suppression was used. The build used two jobs and `ENABLE_TESTABILITY=YES`; `build-settings-evidence.json` preserves compiler commands proving `-O` and `-enable-testing` for VectorCore and all three test bundles.

Evidence is appended under `/private/tmp/vectorcore-release-phase-two-2026-11-04/ios/combined-candidate/`. Original and fix-round inclusion/verification controls remain separate and unchanged. The exact bounded commands, environments, executable hashes, output hashes and complete output are in `build-for-testing.{json,stdout,stderr}` and `full-suite.{json,stdout,stderr}`. `full-suite.xcresult` and the three actual tested `.xctest` bundles, `.xctestrun`, and simulator test-host executable are retained. `retained-products.json` identifies the bundle executables.

| Bundle | Passed cases | Skipped cases | Failed cases |
|---|---:|---:|---:|
| MinimalTests | 33 | 27 | 0 |
| ComprehensiveTests | 1,698 | 115 | 1 |
| BenchmarkContractTests | 87 | 21 | 0 |
| Total | 1,818 | 163 | 1 |

There are 1,982 declared cases. Replacing 49 parameterized methods with their 1,678 argument runs gives 3,447 passed entries, 163 skipped entries and one failure. These are two counting levels for the same execution. `test-summary.stdout`, `test-tree-2.stdout`, `method-results.json`, `counts.json`, and `failed-skipped-results.json` preserve every declared ID, argument run, skip reason and failure; `result-verification.json` reconciles the tree counts with the summary and device counts.

The sole failure is `LostStoreCanaryTests/literalArrayStoreIsObservedAfterOutOfLineClosure()`: observed `2.0` (bits `0x40000000`), expected `1.0`. It remains a real installed-Xcode compiler qualification failure. All seven `BenchmarkNumericalObservationTests` methods pass, including strict observation coverage for all 21 repaired catalog cases and every-coordinate dot-product controls. `concreteProtocolOnlyOperatorsPreserveCopy()` passes; `concreteProtocolOnlyMismatchesRejectBeforeBorrow()` explicitly skips for macOS child-process capability. The earlier five benchmark support failures are absent; that bundle now has zero failures.

The 163 skips comprise 117 macOS subprocess/replacement capability cases, 27 explicitly disabled extended-performance cases, one unavailable C-kernel case, and 18 Swift Testing entries whose structured skip text is only `Test skipped` (their exact IDs are retained). `VECTORCORE_TEST_EXTENDED=0` and `VECTORCORE_STRICT_PERF=0` remain recorded. Timings emitted by ordinary test bodies during overlapping qualification are not performance acceptance.

`source-1cb9031.tar`, pre/post-test source records, clean status and HEAD captures bind execution to immutable source. The explicitly scoped compiler-input projection contains 376 files: `Package.swift` and Swift/C/header/Metal/modulemap files under `Sources`, `Tests`, and `Benchmarks`. The full archived regular-file tree across those paths contains 388 files; every file matched the worktree before and after execution. This projection does not purport to inventory every runtime resource or build setting. The original 403-file historical manifests used a different scope including CI Swift fixtures and are preserved unchanged.

Execution used the existing arm64 iPhone 18 Pro simulator, iOS 27.0 build 24A434, iPhoneSimulator 27.2 SDK and iOS 17.0 deployment target. The Xcode frontend SHA-256 remains `448457da9a8a56741f2c8c4b8931e85b7377fc009c16918cd8ac1a8a3210ea1f`; the SDKSettings hash remains `deb9d0f0fcacac1ba3b606f6b2772fa043ef64814b848cb1deb78f7997f96759`. This establishes neither physical-device nor minimum-OS execution, other-platform coverage, performance acceptance, nor release readiness. Existing Accelerate, weak-variable, test deprecation and build-system warnings remain visible (79 captured build-warning/preview-note lines); no cleanup or suppression was applied.

One evidence-extraction attempt failed: simultaneous `xcresulttool` summary/tree reads raced while materializing `database.sqlite3` inside the result bundle. `test-tree` retains exit 64 and its error; the sequential fresh `test-tree-2` command succeeded. This was a result-reader conflict after execution, not a test failure or a reason to rerun or hide the suite.

Independent final integration check: `final-candidate-source-equality.json` compares immutable archives for tested commit `1cb903188633926c27cb0300958c2d35e9641ec0` and final integrated candidate `6d574f3ff122554ae8bec18f039ffa4a6981920e`. All 388 regular files under `Package.swift`, `Sources/`, `Tests/` and `Benchmarks/` are byte-identical, including non-Swift files; all 376 projection hashes also match. `final-candidate-tree-diff` is empty. These binaries were built at `1cb9031`; no separate compilation of `6d574f3` is claimed.
