# V03 final seal (v5 final bytes) — execution brief

V03 v5 source is accepted: build-system-agnostic product discovery, Release `-enable-testing`, a no-rebuild check on every `--skip-build` run, swift resolved once through xcrun, and the cancelled-test fix. The build-modes driver is kept at its v4 bytes to preserve its validate_github.rb integrity pin. **Execute the final gates on the exact final source and seal the evidence. Make no source changes.**

- Workspace (read and execute only): `/Users/goftin/dev/gsuite/VSK/VectorCore/.stabilization-recovery/2026-10-03/workspaces/V03-v4`
- Evidence root (create it): `/Users/goftin/dev/gsuite/VSK/VectorCore/.stabilization-recovery/2026-10-03/V03/final-v5/`
- Design brief, for gate meanings: `/Users/goftin/dev/gsuite/VSK/VectorCore/.stabilization-recovery/2026-10-02/current-handoff/V03-current-brief.md`. Your earlier run's brief is `.../V03/final-gates-brief.md`; its gate definitions still apply, except where this document overrides them.

## Freeze
First, write `source-freeze.json`: the sha256 of every file from `git ls-files -co --exclude-standard`, plus `git rev-parse HEAD`. Record the toolchain (`swift --version`, `xcodebuild -version`, `sw_vers`). At the end, recompute the freeze and prove it is unchanged. Python bytecode is now gitignored, so the freeze should be exactly clean.

## Gates (sequential; fresh unique paths; --jobs 6)
Run gates 1–4 under the default swiftbuild build system on Swift 6.4:
1. `python3 Scripts/ci/check_performance_gates.py --configuration debug --scratch-root <ev>/build-gates --output <ev>/gates-debug --jobs 6`
2. `python3 Scripts/ci/check_performance_gates.py --configuration release --scratch-root <ev>/build-gates --output <ev>/gates-release --jobs 6`
3. `python3 Scripts/ci/check_sanitizers.py --sanitizer address --scratch-root <ev>/build-sanitizers --output <ev>/sanitizer-address --jobs 6`. Expect 77/77 bodies, with the no-rebuild check passing on every run.
4. `python3 Scripts/ci/check_sanitizers.py --sanitizer thread --scratch-root <ev>/build-sanitizers --output <ev>/sanitizer-thread --jobs 6`. Expect 39/39. The Q03 parent has a 49 s budget.
5. **Gate 5 carries over and is NOT re-run.** The build-modes driver bytes equal v4, and the bench product is unaffected by any v5 change. Cite the round-2 CLT Swift 6.2 evidence path from the report's "v5 round 2" section, and prove driver identity: the sha256 of `Scripts/ci/check_benchmark_build_modes.py` must equal the pin in `Scripts/ci/validate_github.rb`.
6. Benchmark CLI contracts, exactly as the ci.yml `benchmark-contracts` job runs them, on the final bytes.
7. Representative real extended tests with strict perf off: the same bodies as your earlier final-v4 run, with `VECTORCORE_TEST_EXTENDED=1`. Each must run once.
8. Policy: `python3 -B -m unittest discover -s Scripts/ci/tests -v`. Expect 77 OK.

Run any swift command you invoke directly through the capture driver: `python3 /Users/goftin/dev/gsuite/VSK/VectorCore/.stabilization-recovery/2026-10-02/build-tools/capture_swift.py --workdir <ws> --log <ev>/<n>.log --record <ev>/<n>.json -- <test|build> ... --jobs 6 --scratch-path <ev>/build-<n>`. For Python gates, write stdout, stderr and a JSON record (argv, cwd, exit, start, end).

If a gate fails, record it verbatim and continue with the remaining gates. Do not patch anything. **Load rule:** the T02 worker may be compiling concurrently. Re-run a timed-out deadline body once in isolation before treating it as a defect, and report both outcomes.

## Report
Write `/Users/goftin/dev/gsuite/VSK/VectorCore/.stabilization-recovery/2026-10-03/V03/final-v5-report.md`. Include:
- the toolchain;
- the freeze before and after;
- per gate: argv, exit, duration, and actual counts. Distinguish host, loaded bundles, parent entries, no-op parents and child bodies, following the accounting erratum;
- failures verbatim;
- `evidence-index.json`: the sha256 of every evidence file except build scratch, plus the cited loaded bundles and executables.

Claim nothing about hosted CI, whole-package sanitizer coverage, all extended tests, or performance.

Return only: a status (DONE, DONE_WITH_CONCERNS, BLOCKED, or NEEDS_CONTEXT), a one-line gate summary, and your concerns. No commits, no source edits, no subagents.
