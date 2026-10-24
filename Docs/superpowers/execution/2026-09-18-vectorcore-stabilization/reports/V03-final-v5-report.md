# V03 final seal (v5 final bytes): report

Status: DONE_WITH_CONCERNS. Evidence root EV = `.../2026-10-03/V03/final-v5/`. Workspace `.../workspaces/V03-v4`, HEAD e6e1baed3e8af8c95679c84ac9a331440dd088ce.

## Toolchain
Apple Swift 6.4 (swiftlang-6.4.0.34.1), arm64-apple-macosx27.0.0; Xcode 27.0 (27A266a); macOS 27; default build system swiftbuild (driver result.json: buildSystem "toolchain default"). Files: EV/swift-version.txt, xcodebuild-version.txt, sw_vers.txt.

## Freeze
- Before: EV/source-freeze.json, 6130 files. After: EV/source-freeze-after.json, 6130 files. `cmp` reports IDENTICAL; no hash differences. Python bytecode no longer pollutes the freeze.
- Mistake disclosed: my first "after" attempt used a relative output path and wrote `source-freeze-after.json` into the workspace root (an untracked file, 1 extra entry). I moved it out (to EV/stray-freeze-from-relative-path-mistake.json) and recomputed. No tracked or pre-existing file was touched.

## Gates (all exit 0, sequential, --jobs 6, no timeouts, so no load-rule reruns)
| Gate | Exit | Duration | Result |
|---|---|---|---|
| 1 perf gates debug | 0 | 77 s | `V03_PERFORMANCE_GATES_COMPLETE debug 22` |
| 2 perf gates release | 0 | 240 s | `V03_PERFORMANCE_GATES_COMPLETE release 22` |
| 3 ASan | 0 | 119 s | `V03_SANITIZERS_COMPLETE address 77` |
| 4 TSan | 0 | 325 s | `V03_SANITIZERS_COMPLETE thread 39` |
| 5 build modes | carried | not re-run | see below |
| 6 CLI contracts | 0 | 1 s (build 2 steps via capture driver exit 0) | PASS line below |
| 7 extended, strict off | 0 | n/a | 7 tests, 0 failures |
| 8 policy | 0 | 17 s | `Ran 77 tests ... OK` |

Argv and times: EV/gateN-*.json (gates 1-4, 6, 8) and capture-driver records EV/gate6-build.json, gate6-binpath.json, gate7-test.json. Logs beside them.

### Gates 1/2 (accounting)
Cases: 22 per configuration (2 frameworks x 8 controls plus 3 observation runs plus the extra focused cases the driver runs). 2 hosts per configuration (swiftpm-testing-helper and xctest), 2 loaded bundles (ComprehensiveTests, MinimalTests .xctest, per result.json with sha256). Each case ran in its own process and the driver verified the body markers, predicate laziness and the host and loaded-bundle images in the dyld log. Counts cover these 22 cases only, not every extended test.

### Gate 3 ASan
77 of 77 inventory bodies executed, each as its own process. Host swiftpm-testing-helper (sha256 7f75846e...), one loaded bundle ComprehensiveTests.xctest (sha256 b33e1737...), detect_leaks off. Max per-body time well under the 15 s deadline. The no-rebuild check is enforced by the driver on every `--skip-build` run; completion marker was emitted, so it passed. I did not separately inspect each run's no-rebuild record.

### Gate 4 TSan
39 of 39 entries: 38 actual function bodies plus 1 isolated parent (FP16DiagnosticsBoundaryTests/isolatedConcurrency). The parent took 10.9 s against its 49 s budget (cleanup deadline 2 s). The Q03 child record shows markers `Q03_BODY_BEGIN concurrency`, `Q03_WORKERS_JOINED completedLoops=[2000 x 6]`, `Q03_STABLE_TOTALS_JOINED`, `Q03_ATOMIC_BATCHES_JOINED`, `Q03_RESULT`, `Q03_BODY_END`. Loaded bundle ComprehensiveTests sha256 2141b0f1... No timeouts. The 1 parent is a real entry with a real child; there were no no-op parents beyond that accounting.

### Gate 5 (carried, not re-run)
- Driver sha256 of Scripts/ci/check_benchmark_build_modes.py = `2fa77d1206fec1397c3ac9c0211f87e4730118c00c6660cbd41425c0269e6ac7`, equal to the pin in Scripts/ci/validate_github.rb line 179.
- Carried evidence: round-2 CLT Swift 6.2 run, `.../V03/v4-report.md` section "v5 round 2" item 3 (evidence dir `.../V03/v5/`, `modes-r2/`): exit 0, 291 s, `V02_BUILD_MODES_COMPLETE`, 4 cells. I cited it and did not verify it again.
- The build-modes driver remains unsupported under default swiftbuild by design (my earlier v4 run failed at find_executable); gate 5 was not run on this host.

### Gate 6
Release `vectorcore-bench` built, `--show-bin-path` resolved to EV/build-gate6/out/Products/Release, test run exit 0:
`PASS: 10 invalid threshold configurations, JSON/CSV write failures, finite strict real benchmark, 28 provider calls / 14 case boundaries.`

### Gate 7
`VECTORCORE_TEST_EXTENDED=1 VECTORCORE_STRICT_PERF=0`, same filter as final-v4, debug. 7 tests, 0 failures, each emitting exactly one `PERF_OBSERVATION ... bodies=1`: ComprehensiveBenchmarks testBatchDistanceOptimizedPerformance, testVector512OptimizedDotProductPerformance, testVector512OptimizedMemoryFootprint, testVector768GenericDistancePerformance; Vector512Tests testDotProductPerformance, testDistancePerformance, testNormalizationPerformance. Only testVector512OptimizedMemoryFootprint has an in-body numeric assertion (`XCTAssertNotEqual(sum, 0)`), which passed and was not mutation-tested. Log EV/gate7-test.log.

### Gate 8
`python3 -B -m unittest discover -s Scripts/ci/tests -v`: Ran 77 tests, OK (EV/gate8-policy.stderr).

## Failures
None.

## Evidence index
EV/evidence-index.json: sha256 of every file under EV except `build-*` scratch (4067 entries), plus cited scratch binaries: vectorcore-bench (gate 6), MinimalTests (gate 7), debug and release ComprehensiveTests and MinimalTests bundles (gates 1/2), and the ASan and TSan ComprehensiveTests bundles.

## Not claimed
Hosted CI, whole-package sanitizer coverage, all extended tests, performance.
