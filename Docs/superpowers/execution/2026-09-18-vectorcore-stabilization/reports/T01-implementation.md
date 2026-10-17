# T01 report: Xcode 27 / Swift 6.4 toolchain compatibility

Status: DONE_WITH_CONCERNS. Workspace: .../workspaces/T01 (base 9f5fda2). No git index operations.

## Changed files
- Renamed Sources/VectorCoreBenchmarking/ProgressReporter.swift -> BenchmarkProgressReporter.swift; type `ProgressReporter` -> `BenchmarkProgressReporter` (no alias).
- References updated: Benchmarks/VectorCoreBench/{AppMain.swift, Core/BenchmarkCase.swift, Suites/{Batch,Distance,DotProduct,MatrixDistance,Memory,Noop,Normalization}Bench.swift}; Tests/BenchmarkContractTests/{RunnerContractTests,BenchmarkSelectionContractTests}.swift. `ProgressFormat` untouched. No README/Docs (outside historical execution evidence) mention the type.
- CHANGELOG.md: entry under [Unreleased] > Changed (breaking, with migration).
- New: Scripts/ci/check_sdk_name_collisions.py, Scripts/ci/sdk_name_collision_allowlist.json (one entry, see concerns).

## Migration
Replace `ProgressReporter` with `BenchmarkProgressReporter` wherever it names the VectorCoreBenchmarking type (`BenchmarkProgressReporter(format:)`, API unchanged).

## Additional compile fixes
None. The base error cascade (AppMain.swift:246) disappeared with the rename; no hidden Swift 6.4 errors appeared in any target, tests included.

## Commands (all via capture_swift.py; evidence dir = .../2026-10-03/T01, logs `<name>.log`, records `<name>.json`)
| run | args | exit | observed |
|---|---|---|---|
| build-debug-1 | build --build-tests | 0 | Build complete |
| build-release-1 | (quoting mistake `-c release` as one arg) | 64 | invalid invocation, not a build |
| build-release-2 | build --build-tests -c release | 1 | `module not compiled for testing` / unresolved module (see findings) |
| build-release-3-nobuildtests-native | same, --build-system native | 1 | same error under native: not toolchain-related |
| build-release-4-testable | `--enable-testable-imports` | 64 | unknown option |
| build-release-5-enable-testing | build --build-tests -c release -Xswiftc -enable-testing | 0 | Build complete (205 s) |
| test-L03-debug-1 | test --filter L03 | 0 | 39 tests in 9 suites passed (includes 1 parent no-op); 9 L03_CHILD_CAPTURE children, childProbe passed |
| test-M10-debug-1 | test --filter Milestone10 | 0 | XCTest 20 executed, 0 failures; Swift Testing 21 tests in 3 suites + 1 (BenchmarkContractSwiftTestingEntry) = 22 |
| test-L03-release-2 | test -c release --filter L03 | 0 | 39 tests in 9 suites passed; 9 children |
| test-M10-release-2 | test -c release --filter Milestone10 | 0 | 20 XCTest + 22 Swift Testing, all passing |
| swift62-build-1 | build (no DEVELOPER_DIR) | 0 | invalid as a 6.2 check (ran 6.4); ignore |
| swift62-build-2 | DEVELOPER_DIR=/Library/Developer/CommandLineTools build | 0 | Apple Swift 6.2 (swiftlang-6.2.0.19.9); library + bench products compile, Build complete |
Debug filters used the swiftbuild default; no `native` diagnostic pass was needed. Test counts match accepted 6.3.3 results. No performance claims.

## Build-system findings
- Child-process harnesses (the 9 L03 children, Milestone10 BenchmarkContract) all worked under swiftbuild layout; no harness path failure.
- `swift build --build-tests -c release` fails on both swiftbuild and native (pre-existing, not toolchain-caused): tests use `@testable import`, which needs `-enable-testing`; `swift test -c release` adds it, plain `build --build-tests -c release` does not. Workaround used: `-Xswiftc -enable-testing`. Brief item 1 release command as written cannot exit 0.
- L03 children write captures to .../2026-10-03/L03/children/ (outside the T01 evidence dir; harness behavior).

## Name-collision checker
Command for CI (macOS runner, from repo root): `python3 Scripts/ci/check_sdk_name_collisions.py` (optional `--scratch-path`, `--sdk`, `--allowlist`). Exit 0 clean, 1 unreviewed collision, 2 tool failure. Every subprocess has an explicit timeout (1200 s dump-symbol-graph, 300 s per extract, 60 s quick); TimeoutExpired maps to exit 2. It tolerates a failing non-library package target if the two library graphs were produced (needed for the RED base, where the bench target does not compile); prints a warning.
- RED (base 9f5fda2 extracted to T01/base-src, empty allowlist): checker-red.txt, exit 1:
  `COLLISION: VectorCoreBenchmarking.ProgressReporter == Foundation.ProgressReporter`, plus `VectorCore.Logger == os.Logger` and `VectorCoreBenchmarking.Logger == os.Logger` (the latter is the re-exported VectorCore.Logger). Info: SIMDStorage == Swift.SIMDStorage.
- With the allowlist on base: checker-red-with-allowlist.txt, exit 1, only ProgressReporter remains.
- GREEN (fix, allowlist with Logger): checker-green.txt, exit 0. Counts: VectorCore 229, VectorCoreBenchmarking 254 (includes re-exported VectorCore), Foundation 491, Swift 324, Accelerate 305, simd 302, Dispatch 51, os 24. Info: SIMDStorage (Swift stdlib).
- Scripts/ci/tests/test_child_process_bounds.py (`python3 -B -m unittest discover -s Scripts/ci/tests -p 'test_child_process_bounds.py' -v`): 7 tests OK.

## Lint (pinned 0.65.1, 12 changed Swift files)
126 serious violations before and after on the same file set (base 549 total, fix 550 total: one extra non-serious warning, consistent with the longer type name). No new errors.

## Open concerns
1. Allowlist entry: public `VectorCore.Logger` collides with `os.Logger` (pre-existing, not a toolchain regression; module-local shadowing keeps VectorCore itself compiling, but downstream files importing both `os` and VectorCore get an ambiguity). I allowlisted it with a justification rather than renaming, as a rename is outside the approved contract. Root should decide whether to rename (e.g. `VectorCoreLogger`) and then empty the allowlist.
2. Release `build --build-tests` requires `-Xswiftc -enable-testing` (see above); any CI/doc command lacking it fails regardless of toolchain.
3. Checker covers only the active SDK; the allowlist/Swift-stdlib info rule is as specified. CI wiring not done (V03 owns).
4. Scratch dirs (build-*, scratch-*, base-src) remain in the evidence dir.

## Fix round 1
Changes (no Swift source changed, so no rebuild or SwiftLint run; changed files are Python/JSON/Markdown only):
1. Allowlist Logger reason now names Owner D02, D03 preserving the signature, since macOS 11, root ruling 2026-10-03 (T01 brief-addendum item 2); documents (name, SDK module) keying.
2. Stale-graph fix: the checker always dumps into a fresh `run-*` subdirectory (mkdtemp under --scratch-path, or a temp dir) and only reads graphs from it; the "Files written to"/older-run fallbacks are removed. Dump failure with no fresh graphs exits 2. (Cost: every run rebuilds from scratch.)
3. CHANGELOG: iOS 27 claim dropped; now "macOS 27 SDK adds Foundation.ProgressReporter (other OS 27 SDKs are unverified)".
4. Docstring: Swift-stdlib shadowing is written as an explicit unverified rationale.
5. Sentinel self-check (exit 2 if missing): Foundation URL, Data, Date, UUID, AttributedString, ProgressReporter; Swift Array, String; Accelerate DSPComplex, BNNSFilter; simd simd_float4, simd_quatf; Dispatch DispatchQueue, DispatchTime; os Logger, OSAllocatedUnfairLock. All passed on the real Xcode 27 extraction in the first run (no adjustment needed). Negative check: with a bogus os sentinel the checker printed the self-check error and exited 2.
6. Allowlist keying documented in file reason and script docstring; stale entries print `WARNING: stale allowlist entry ...` (not exercised by a run, since Logger still collides; code path is a set difference).
7. `xcrun --show-sdk-path` return code and empty output now exit 2 with a message.
8. swift.actor kept per ruling.

Runs (not via capture driver, since not swift build/test):
- GREEN: `python3 -B Scripts/ci/check_sdk_name_collisions.py --scratch-path T01/scratch-green-fix1` -> exit 0, log checker-green-fix1.txt (same counts as before: 229/254/491/324/305/302/51/24; Logger allowed twice; SIMDStorage info; sentinels passed).
- RED: pristine base via `git -C <repo> archive 9f5fda2 | tar -x -C T01/base-fix1`, `--package-path T01/base-fix1 --scratch-path T01/scratch-red-fix1` -> exit 1, log checker-red-fix1.txt: `COLLISION: VectorCoreBenchmarking.ProgressReporter == Foundation.ProgressReporter` (Logger allowlisted using the fixed workspace's allowlist default); warning that dump exited 1 and used this run's graphs.
- Bounds scanner: `python3 -B -m unittest discover -s Scripts/ci/tests -p 'test_child_process_bounds.py'` -> 7 tests OK.
- SwiftLint: not run (no Swift files changed).
