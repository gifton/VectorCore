# V03 final runtime gates (v4 source) — execution brief

V03 ("Executed CI coverage and performance opt-outs") v4 source is accepted: v3 plus the reviewed v4 required-check/subprocess-deadline delta. Its workspace has been rebased onto the integration revision that contains T01. T01 is the Xcode 27 fix that renamed `VectorCoreBenchmarking.ProgressReporter` to `BenchmarkProgressReporter`. Your job is to **execute** the final gates on that exact source and seal the evidence. You are not here to change source.

- Workspace (git worktree; read and execute only): `/Users/goftin/dev/gsuite/VSK/VectorCore/.stabilization-recovery/2026-10-03/workspaces/V03-v4`
- Evidence root (create it; everything goes under it): `/Users/goftin/dev/gsuite/VSK/VectorCore/.stabilization-recovery/2026-10-03/V03/final-v4/`
- Binding V03 design brief, for the meaning of each gate: `/Users/goftin/dev/gsuite/VSK/VectorCore/.stabilization-recovery/2026-10-02/current-handoff/V03-current-brief.md`. Read its "Final checks" paragraph and the sanitizer/Q03 paragraphs.
- Host toolchain: macOS 27 / Xcode 27.0 / Swift 6.4, SwiftPM default build system `swiftbuild`. Prior V03 development evidence was Swift 6.3.3. Record `swift --version`, `xcodebuild -version` and `sw_vers` into the evidence root first.

## Freeze first

Before any build, write `source-freeze.json`: the sha256 of every tracked and untracked non-ignored file in the workspace (`git ls-files -co --exclude-standard`), plus `git rev-parse HEAD`. After all gates finish, recompute it and prove it is unchanged. Make no source edits at any point. If a gate seems to need a source change, stop and return NEEDS_CONTEXT with the exact failure.

## Gates (run sequentially, each with fresh unique paths, `--jobs 6`)

1. `python3 Scripts/ci/check_performance_gates.py --configuration debug --scratch-root <ev>/build-gates --output <ev>/gates-debug --jobs 6`
2. `python3 Scripts/ci/check_performance_gates.py --configuration release --scratch-root <ev>/build-gates --output <ev>/gates-release --jobs 6`
3. `python3 Scripts/ci/check_sanitizers.py --sanitizer address --scratch-root <ev>/build-sanitizers --output <ev>/sanitizer-address --jobs 6`. Source inventory is 77 ASan bodies. Report how many actually executed.
4. `python3 Scripts/ci/check_sanitizers.py --sanitizer thread --scratch-root <ev>/build-sanitizers --output <ev>/sanitizer-thread --jobs 6`. Source inventory is 39 TSan entries. Q03 `isolatedConcurrency` parent has a 49 s outer deadline and its 45 s child has at most 2 s cleanup. All other bodies have 15 s plus at most 2 s cleanup.
5. `python3 Scripts/ci/check_benchmark_build_modes.py --scratch-root <ev>/build-modes --output <ev>/modes --jobs 6`. These are the four C-off/C-on × Debug/Release cells.
6. Benchmark CLI contracts. Do this exactly as the ci.yml `benchmark-contracts` job does: build the Release `vectorcore-bench` product, resolve its real bin path with `--show-bin-path`, then run `Scripts/ci/tests/test_benchmark_contracts.py` against that executable. Mirror the workflow's exact invocation and environment.
7. Representative real extended tests with strict off. Pick a small representative set of formerly ungated measurement bodies covered by the observation wrapper: for example a few from `Tests/MinimalTests/ComprehensiveBenchmarks.swift` and the three in `Tests/MinimalTests/Vector512Tests.swift`. Run them with `VECTORCORE_TEST_EXTENDED=1` and strict perf off, through a focused `swift test --filter`. Show that each body ran once and that numeric assertions are preserved. Do not claim any untested body.

Run Swift commands that you invoke directly (gates 6 and 7) through the capture driver: `python3 /Users/goftin/dev/gsuite/VSK/VectorCore/.stabilization-recovery/2026-10-02/build-tools/capture_swift.py --workdir <workspace> --log <ev>/<name>.log --record <ev>/<name>.json -- <test|build> ... --jobs 6 --scratch-path <ev>/build-<name>`. For Python driver gates (1–5), save stdout and stderr to files. Record argv, cwd, exit code, start and end times in a JSON per gate.

If a driver fails because of build-system product-layout assumptions (for example a hardcoded `.build/debug`), stop and return NEEDS_CONTEXT with the exact failure. Do not patch.

**Load rule:** another worker may be compiling concurrently. If a deadline-bounded body times out, re-run that single body once in isolation before treating it as a defect, and report both outcomes. A timing-ratio failure is never a correctness verdict.

## Report

Write `/Users/goftin/dev/gsuite/VSK/VectorCore/.stabilization-recovery/2026-10-03/V03/final-v4-report.md` with:
- the toolchain;
- the freeze hashes before and after;
- per gate: argv, exit code, duration, and actual counts. Distinguish SwiftPM host, loaded test binary, parent entries, no-op parents and actual child bodies. Follow the accounting erratum: `BenchmarkContractSwiftTestingEntry` is a real assertion body, not a no-op;
- each failure verbatim, with its log path;
- a final `evidence-index.json` holding the sha256 of every file under the evidence root except build scratch directories. Hash scratch products only for the loaded test bundles and executables you cite.

Claim nothing about hosted CI, whole-package sanitizer coverage, all extended tests, or performance.

Return only: a status (DONE, DONE_WITH_CONCERNS, BLOCKED, or NEEDS_CONTEXT), a one-line gate summary, and your concerns.

## Rules
No git commits, staging, stash or other index operations. No source edits. No subagents. Keep everything local.
