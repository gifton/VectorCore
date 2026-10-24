# V03 v4 final runtime gates: report

Status: NEEDS_CONTEXT. Gates 1-5 could not run under the swiftbuild default build system. They fail on product-layout assumptions in the drivers, plus one release build failure. Gates 6 and 7 passed.

Evidence root: `/Users/goftin/dev/gsuite/VSK/VectorCore/.stabilization-recovery/2026-10-03/V03/final-v4/` (called EV below).
Workspace: `.../2026-10-03/workspaces/V03-v4`, HEAD `e6e1baed3e8af8c95679c84ac9a331440dd088ce`.

## Toolchain
- Apple Swift 6.4 (swiftlang-6.4.0.34.1), target arm64-apple-macosx27.0.0 (EV/swift-version.txt).
- Xcode 27.0, build 27A266a (EV/xcodebuild-version.txt). macOS: EV/sw_vers.txt.
- SwiftPM default build system: swiftbuild. Build products land in `out/Products/{Debug,Release}`.

## Freeze
- `source-freeze.json` (before): 6127 files, sha256 of the file `2e528f43601c4206216c8147dea4e150bb7b7707c2d813edaa514441f6733c4e`.
- `source-freeze-after.json` (after): 6129 files, sha256 of the file `bd174f229918b0875e9667bb0bf041a0f6b1979cb7d0f3fc924fe08e63107b0e`.
- All 6127 original files are byte-identical. No file was modified or removed.
- The 2 extra entries are untracked, non-ignored Python bytecode files created by running the repo's Python drivers: `Scripts/ci/__pycache__/check_ci_contracts.cpython-314.pyc` and `Scripts/ci/__pycache__/v03_process.cpython-314.pyc`.
- I did not delete them, because that would be a workspace edit. This is a benign artifact, but the "unchanged" proof holds only up to those two caches. Root may want to remove them or exclude `__pycache__`.
- The freeze script is `EV/freeze.py`.

## Gates

Gates 1-5 per-gate record: `EV/gateN-*.json` (argv, cwd, exit code, start, end, duration), with `.stdout` and `.stderr` beside it. Gate 6 and 7 use capture-driver records.

### Gate 1: performance gates, debug. FAIL (exit 1, 38 s)
- argv: `python3 Scripts/ci/check_performance_gates.py --configuration debug --scratch-root EV/build-gates --output EV/gates-debug --jobs 6`
- The build completed ("Build complete! (35.50 sec)"). The driver then died at `check_performance_gates.py` line 40:
```
ValueError: Expected one loaded test bundle
```
- Log: `EV/gate1-debug.stderr`.
- Cause: the driver looks for `rglob('VectorCorePackageTests')` that is a file whose parent is `MacOS`. Under swiftbuild the products are three separate bundles, `Products/Debug/{BenchmarkContractTests,ComprehensiveTests,MinimalTests}.xctest`. No `VectorCorePackageTests` exists.
- No test body ran. Counts are 0 for host, bundle, parent entries and child bodies.

### Gate 2: performance gates, release. FAIL (exit 1, 40 s)
- This is a different failure from gate 1: the `swift build --build-tests -c release` step itself fails.
- Excerpt from `EV/gates-release/build.stderr`:
```
error: .../Tests/MinimalTests/BaselineTests.swift:2:18 unable to resolve Swift module dependency to a compatible module: 'VectorCore'
error: SwiftDriver MinimalTests normal arm64 ... failed
error: .../Tests/ComprehensiveTests/AccelerateArraySIMDProviderParityTests.swift:3:18 unable to resolve Swift module dependency to a compatible module: 'VectorCore'
error: SwiftDriver ComprehensiveTests normal arm64 ... failed
error: Build failed
```
- The `@testable import VectorCore` fails in a release test build, so the test targets cannot compile.
- Driver traceback: `RuntimeError: build: exit 1, expected (0,)` (v03_process.py line 71). Log: `EV/gate2-release.stderr` and `EV/gates-release/build.stderr`.
- I did not diagnose the cause further. It may be that `@testable` needs `-enable-testing` under swiftbuild release.
- No test body ran.

### Gate 3: ASan sanitizer. FAIL (exit 1, 37 s)
- argv: `python3 Scripts/ci/check_sanitizers.py --sanitizer address --scratch-root EV/build-sanitizers --output EV/sanitizer-address --jobs 6`
- Failure at `check_sanitizers.py` line 74: `ValueError: Expected one loaded test bundle`. Same `VectorCorePackageTests` assumption as gate 1. Log: `EV/gate3-asan.stderr`.
- Source inventory is 77 ASan bodies. Actually executed: 0.

### Gate 4: TSan sanitizer. FAIL (exit 1, 33 s)
- argv: as gate 3 with `--sanitizer thread`, output `EV/sanitizer-thread`.
- Same failure at line 74 (`EV/gate4-tsan.stderr`).
- Source inventory is 39 TSan entries. Actually executed: 0. The Q03 49 s parent deadline was never exercised. The load rule was not triggered, because no deadline-bounded body ran.

### Gate 5: benchmark build modes. FAIL (exit 1, 14 s)
- argv: `python3 Scripts/ci/check_benchmark_build_modes.py --scratch-root EV/build-modes --output EV/modes --jobs 6`
- Failure in `find_executable` at line 129:
```
AssertionError: []
```
- `EV/modes/failure.json`: `{"completedCells": [], "error": "[]"}`. Log: `EV/gate5-modes.stderr`.
- Cause: the driver filters for `p.parent.name == configuration` (lowercase `debug`). Swiftbuild places the executable at `build-modes/debug-c-off/out/Products/Debug/vectorcore-bench`, with capital `D`.
- The first cell (debug C-off) built. Completed cells: 0 of 4.

### Gate 6: benchmark CLI contracts. PASS
- Mirrored the workflow's `benchmark-contracts` job step exactly:
  1. `swift build -c release --product vectorcore-bench --jobs 6 --scratch-path EV/build-gate6`, through the capture driver. Exit 0, "Build complete! (56.68 sec)". Record: `EV/gate6-build.json`, log: `EV/gate6-build.log`.
  2. `swift build -c release --show-bin-path ...`, through the capture driver. Exit 0. Output: `EV/build-gate6/out/Products/Release`. Record: `EV/gate6-binpath.json`.
  3. `python3 Scripts/ci/tests/test_benchmark_contracts.py <bin>/vectorcore-bench`. Exit 0, about 2 s. Record: `EV/gate6-contracts.json`.
- stdout: `PASS: 10 invalid threshold configurations, JSON/CSV write failures, finite strict real benchmark, 28 provider calls / 14 case boundaries.`
- The driver's own artifacts went to a `/var/folders` temp directory, not under EV.
- Environment note: the workflow job has no special environment. I used the shell's inherited environment. The Xcode 26.3 pin does not apply locally.
- The built executable's sha256 is in `evidence-index.json`.

### Gate 7: representative real extended tests, strict perf off. PASS
- Command, through the capture driver, with `VECTORCORE_TEST_EXTENDED=1` and `VECTORCORE_STRICT_PERF=0`. This is a focused XCTest filter run in debug (not the `^PerformanceObservationContractTests` harness).
- Filter: `^MinimalTests\.(ComprehensiveBenchmarks/(testVector512OptimizedDotProductPerformance|testVector768GenericDistancePerformance|testBatchDistanceOptimizedPerformance|testVector512OptimizedMemoryFootprint)|Vector512Tests/(testDotProductPerformance|testDistancePerformance|testNormalizationPerformance))$`
- Command: `swift test -c debug --filter <above> --jobs 6 --scratch-path EV/build-gate7`
- Result: exit 0. "Executed 7 tests, with 0 failures". Log: `EV/gate7-test.log`, record: `EV/gate7-test.json`.
- Each of the 7 bodies emitted exactly one `PERF_OBSERVATION <site> ... bodies=1` line, so each ran once and nothing used `XCTest.measure`:
  - ComprehensiveBenchmarks: testBatchDistanceOptimizedPerformance, testVector512OptimizedDotProductPerformance, testVector512OptimizedMemoryFootprint, testVector768GenericDistancePerformance
  - Vector512Tests: testDotProductPerformance, testDistancePerformance, testNormalizationPerformance
- Numeric assertions: only `testVector512OptimizedMemoryFootprint` contains an in-body numeric assertion (`XCTAssertNotEqual(sum, 0)`). It is inside the observed block and passed. The other 6 bodies have no in-body numeric assertions in the source.
- I did not run a mutation to prove the assertion fires. The claim covers only these 7 bodies.
- Host: swift test on the Xcode 27 `xctest` host, loaded bundle `build-gate7/out/Products/Debug/MinimalTests.xctest`. Its binary sha256 is in the evidence index.
- The 0-test ComprehensiveTests and BenchmarkContractTests bundles loaded and executed nothing (expected from the filter).

## Why I am returning NEEDS_CONTEXT
- Gates 1, 3 and 4 assume a single `VectorCorePackageTests` Mach-O in a `MacOS` directory. That is the SwiftPM-native layout. Swiftbuild emits one `.xctest` bundle per test target, so the drivers cannot locate the loaded bundle.
- Gate 5 assumes the executable's parent directory is named exactly the lowercase configuration name (`debug`). Swiftbuild uses `Debug` and `Release`.
- Gate 2 has a genuine release test-build failure under swiftbuild ("unable to resolve Swift module dependency ... 'VectorCore'").
- All of these need a driver or source change, or an explicit decision to run the gates with `--build-system native`. Neither was authorized, so I did not try either. The brief says to stop and return NEEDS_CONTEXT for exactly this class of failure.
- Question for root: authorize rerunning gates 1-5 with the native build system (for example via an environment or flag that the drivers pass through, if any), or accept that the drivers need a swiftbuild-aware patch.

## Not claimed
- No claim about hosted CI, whole-package sanitizer coverage, all extended tests, or performance.
- Gates 1-5 produced no execution evidence. Gates 6 and 7 are single local runs.
- Concurrent load: the host's load average at start was about 3.6 (another worker compiling). No deadline-bounded bodies ran, so no timeout reruns were needed.

## Evidence index
- `EV/evidence-index.json`: sha256 of every file under EV (934 entries) except `build-*` scratch directories. It also holds two scratch products I cite: `build-gate6/out/Products/Release/vectorcore-bench` and `build-gate7/out/Products/Debug/MinimalTests.xctest/Contents/MacOS/MinimalTests`.
- Scratch directories left in place: `build-gates`, `build-sanitizers`, `build-modes`, `build-gate6`, `build-gate7`.
