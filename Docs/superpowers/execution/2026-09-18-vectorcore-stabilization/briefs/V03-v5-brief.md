# V03 v5 brief — build-system-agnostic gate drivers (root ruling, binding)

## Why
The V03 final-gate run on accepted v4 source, Swift 6.4, default `swiftbuild` (report: `.../2026-10-03/V03/final-v4-report.md`, evidence `.../V03/final-v4/`) showed:
- **Gates 1, 3, 4** (`check_performance_gates.py` debug; `check_sanitizers.py` address/thread) fail with `ValueError: Expected one loaded test bundle`. The drivers search for a file `VectorCorePackageTests` whose parent directory is named `MacOS`. That is the native layout. Swiftbuild emits one bundle per test target, `Products/<Config>/<Target>.xctest/Contents/MacOS/<Target>`.
- **Gate 5** (`check_benchmark_build_modes.py`) fails with `AssertionError: []`. `find` requires the executable's parent directory to be named exactly the configuration string (`debug`), but swiftbuild uses `Debug`.
- **Gate 2** (performance release) fails in `swift build --build-tests -c release` with `unable to resolve Swift module dependency to a compatible module: 'VectorCore'` on `@testable import`. T01 showed independently that a Release `build --build-tests` needs `-Xswiftc -enable-testing` under both build systems; `swift test` adds that flag, `swift build` does not.
- Running the drivers created untracked `Scripts/ci/__pycache__/*.pyc`, because `.gitignore` has no Python bytecode rule.

These drivers had never been executed end to end before this run. Hosted CI pins Xcode 16.2/26.3 (native default), and Xcode 27 defaults to swiftbuild. The drivers must work under both layouts.

## Contract
1. **Product discovery from SwiftPM's reported bin path, not from directory-name heuristics.** Add one small shared, stdlib-only helper, in `Scripts/ci/v03_process.py` or a new `Scripts/ci/swiftpm_products.py`. It must:
   - resolve the bin path with `swift build --show-bin-path`, using the same configuration, scratch path and build flags as the build it describes, through the bounded `run` helper with an explicit timeout;
   - locate test bundles under that bin path, accepting exactly two layouts:
     - native: one `VectorCorePackageTests.xctest/Contents/MacOS/VectorCorePackageTests`;
     - swiftbuild: per-target `<Target>.xctest/Contents/MacOS/<Target>`;
   - return the bundle that contains the test target each gate actually runs, for example `ComprehensiveTests` for the performance probes. A sanitizer gate whose manifest spans several targets resolves one bundle per target;
   - locate executables such as `vectorcore-bench` as `<bin>/<name>`;
   - fail with a clear error when a required product is missing or ambiguous. It never silently picks one of several.

   Keep every existing association check: the bundle and host paths must appear in the actual run log, and expected bodies must actually execute. For a multi-bundle layout these checks become per-target. Do not weaken them.
2. **Release testable builds.** For every `--build-tests` build in Release, pass `-Xswiftc -enable-testing`. Pass identical build flags to the matching `swift test --skip-build` invocation, so it reuses the same products. Verify from the log that no rebuild occurs.
3. **Optional build-system override.** Each of the three drivers takes `--swift-build-system {native,swiftbuild}`. When it is given, pass `--build-system <value>` to every swift build/test/show-bin-path invocation. When it is omitted, use the toolchain default and pass no flag. This is used for native-layout confirmation and lets CI pin a value later. Do not change `ci.yml` or `sanitizers.yml` to use it unless validate_github policy requires a matching change. If it does, make the minimal consistent change and say so.
4. **`.gitignore`:** add `__pycache__/` and `*.py[cod]`.
5. **Tests (closing the class):** add stdlib unit tests for the discovery helper on synthetic directory trees:
   - native layout;
   - swiftbuild layout;
   - per-target selection;
   - missing product;
   - two candidate bundles for one target, which must be an error;
   - configuration-directory case differences.

   Put them in a new `Scripts/ci/tests/test_swiftpm_products.py`, or extend an existing V03-owned test. Update any existing policy test or validator fixture that pins the old discovery code or argv strings. Every Python subprocess call keeps an explicit finite timeout; the bounds scanner enforces this.

## Ownership
Workspace: `/Users/goftin/dev/gsuite/VSK/VectorCore/.stabilization-recovery/2026-10-03/workspaces/V03-v4` (accepted v4 rebased on e6e1bae). Root has staged its private index; do not run git commands.

You may edit:
- `Scripts/ci/check_performance_gates.py`, `Scripts/ci/check_sanitizers.py`, `Scripts/ci/check_benchmark_build_modes.py`;
- `Scripts/ci/v03_process.py`, or a new `Scripts/ci/swiftpm_products.py`;
- `.gitignore`;
- new or changed tests under `Scripts/ci/tests/`;
- `validate_github.rb` and `test_validate_github.py`, only if needed for consistency.

**No Swift source or test edits.** If a gate failure needs one, stop and return NEEDS_CONTEXT.

## Verification (evidence: `.../2026-10-03/V03/v5/`)
1. Full policy discovery: `python3 -B -m unittest discover -s Scripts/ci/tests -v`. v4 had 59 tests. Report the new total and explain the delta.
2. Under default swiftbuild (Swift 6.4), run all five driver gates to completion with fresh unique scratch and output paths:
   - perf debug;
   - perf release;
   - ASan (source inventory 77 bodies);
   - TSan (source inventory 39 entries; Q03 parent 49 s / child 45 s exception);
   - four build modes.

   Report actual executed counts. Distinguish host, loaded bundle(s), parent entries, no-op parents and actual child bodies; follow the accounting erratum, which says `BenchmarkContractSwiftTestingEntry` is a real body.
3. Native-layout confirmation: run perf debug and four build modes again with `--swift-build-system native`, using fresh paths.
4. Load rule: another worker compiles concurrently. Re-run a timed-out deadline body once in isolation before calling it a defect, and report both outcomes.

## Report
Append a "v5" section to `/Users/goftin/dev/gsuite/VSK/VectorCore/.stabilization-recovery/2026-10-03/V03/v4-report.md`. Include:
- the diff summary;
- per-command argv, exit code, duration and counts;
- failures verbatim;
- concerns.

Return only: a status (DONE, DONE_WITH_CONCERNS, BLOCKED, or NEEDS_CONTEXT), a one-line summary, and your concerns. No commits, no subagents.
