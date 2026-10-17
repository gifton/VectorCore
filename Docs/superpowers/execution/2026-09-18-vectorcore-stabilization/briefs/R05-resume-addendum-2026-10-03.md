# R05 resume addendum (root, 2026-10-03) — binding together with the original brief

## Inputs
- Binding brief (unchanged; sha256 da4ea6c37730403e5178177cac5f6620cde30eee34101b63673f7c869ab45994): `/private/tmp/vectorcore-stabilization-2026-09-18/Docs/superpowers/execution/2026-09-18-vectorcore-stabilization/briefs/R05.md`
- Pause handoff: `.../reports/R05-pause-handoff.md` (same directory as the brief).
- Prior evidence, read-only: `/Users/goftin/dev/gsuite/VSK/VectorCore/.stabilization-recovery/2026-10-02/R05/`. It holds the RED (`red/`), the pause draft (`pause-draft/`) and `caller-inventory.txt`. **The actual unchanged-production RED is already captured, so do not re-run it.** It ran 1 function and found 7 issues: the IDs [2,1,0] and distances [-2,-1,-0.5] were correct, but the metric-free scores were [-1, Inf, 2]. That RED ran on Swift 6.3.3.
- Workspace (git worktree): `/Users/goftin/dev/gsuite/VSK/VectorCore/.stabilization-recovery/2026-10-03/workspaces/R05`. Root has rebased the six pause-draft paths onto the current integration revision, which includes L03 and the T01 Xcode 27 fix. The six paths are byte-identical to `pause-draft/manifest.json`. The draft is **uncompiled, unverified and unreviewed**. Treat it as a starting point, not as accepted work.
- Evidence root for this round: `/Users/goftin/dev/gsuite/VSK/VectorCore/.stabilization-recovery/2026-10-03/R05/`.

## Root rulings for the resumed round
1. **Toolchain.** Use macOS 27 / Xcode 27.0 / Swift 6.4 with the default `swiftbuild` build system. Every child-process harness exercised so far works under it. Record the toolchain in your report.
2. **Single-phase verification.** The focused filters are cheap, so implement, compile and produce the full focused evidence in one round. The independent review then covers source and runtime together. This replaces the earlier plan of a frozen source review followed by an expensive final matrix.
3. **Callers.** Root's grep finds `computedScore` used only in R05-owned files: `SearchResult.swift` and the approved `IntegrationProtocolsTests` migration. No README or Docs Swift block uses it, so `consumer_smoke` is unaffected. If you find any other caller, stop and report it.

## Work
1. Check the drafted test matrix against every clause of the brief, then fill the gaps. In particular verify these brief clauses:
   - the DotProduct logistic piecewise form, with no pole at d = -1;
   - the inverse-distance builtins' domain;
   - nil for negative or nonfinite distances;
   - exact preservation of explicit score bits;
   - `computedScore(using:)` never invoking the witness when a score is stored;
   - existential, generic and concrete witness dispatch;
   - a spoofed-name custom metric;
   - one score call per selected result, and none when k <= 0 or there are no results;
   - a reentrant score witness running outside the collection lock, with a 15 s deadline and at most 2 s owned-child cleanup that throws before any termination-status access;
   - TopK conversion leaving scores nil;
   - mapIDs and Codable preserving scores;
   - all seven builtin metrics in real collection searches, including non-default Minkowski p.

   The independent oracle must not reuse the production piecewise code: use Double `1/(1+exp(d))` for modest values, and limiting endpoints or offline high-precision values for extremes. Assert values are finite before comparing residuals, and use a 4-ULP tolerance.
2. Add an R05-prefixed public Swift 6 consumer under `Scripts/ci/fixtures/`. Use `Scripts/ci/fixtures/L03PublicConsumer.swift` and `L03_public_driver.py` as the pattern. Any Python driver must give every subprocess call an explicit finite timeout, because the program scanner enforces this. The consumer must compile and run: a defaulted metric conformance, the explicit `computedScore(using:)`, search, and a custom collection search override. It must also show that the old `computedScore` property no longer exists, with a negative compile control.
3. Compile, then run the focused tests in Debug and in Release with distinct scratch directories through the capture driver. Use `python3 /Users/goftin/dev/gsuite/VSK/VectorCore/.stabilization-recovery/2026-10-02/build-tools/capture_swift.py --workdir <workspace> --log <ev>/<name>.log --record <ev>/<name>.json -- test [-c release] --jobs 6 --scratch-path <ev>/build-<name> --filter '<filter>'`. The filter is `SearchResultScoreContractTests|IntegrationProtocolsSuite|MetricDispatchContractTests|VectorCollectionTransactionTests`; confirm these symbol names actually exist, and adjust and report if any does not. Keep R05's child captures separate from the existing R02 child captures. Report counts exactly: functions, suites, parent no-ops and actual children.
4. Run pinned SwiftLint 0.65.1 (`/Users/goftin/dev/gsuite/VSK/VectorCore/.stabilization-recovery/2026-10-02/toolchain/swiftlint-0.65.1/swiftlint lint --config <workspace>/.swiftlint.yml <changed swift files>`) and report any new errors. Also run the bounds scanner: `python3 -B -m unittest discover -s Scripts/ci/tests -p 'test_child_process_bounds.py' -v`.
5. Write a GREEN-against-RED comparison: the same dot-product scenario now yields finite, bounded, descending scores, with IDs and distances unchanged.

## Ownership
Edit only the six brief-owned paths plus your new R05 fixture and driver. Make no source edits while a build is running. No git commits, staging, stash or other index operations. No subagents. If a correct implementation needs any other file, stop and return NEEDS_CONTEXT.

## Report
Write `/Users/goftin/dev/gsuite/VSK/VectorCore/.stabilization-recovery/2026-10-03/R05/report.md` with:
- changed files;
- the contract as implemented, mapped to brief clauses;
- the migration;
- per command: argv, exit code, log path and counts;
- the GREEN vs RED comparison;
- lint and scanner results;
- open concerns.

Return only: a status (DONE, DONE_WITH_CONCERNS, BLOCKED, or NEEDS_CONTEXT), a one-line test summary, and your concerns.
