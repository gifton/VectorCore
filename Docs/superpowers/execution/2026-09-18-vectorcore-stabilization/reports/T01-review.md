# T01 independent review

Spec: FAIL (one binding addendum item not met: allowlist justification must name D02)
Quality: CHANGES REQUESTED

## Findings

### Important
1. Scripts/ci/sdk_name_collision_allowlist.json:5 - Addendum item 2 requires the single Logger entry's justification to name D02 as owner. The reason text says "deferred to the program root (T01 report, open concerns)" and never names D02. It also points at a transient report instead of the owning task. Fix: reword to "Owner: D02 (public API disposition, downstream migration); D03 preserves the Logger signature until then", and cite the root ruling.
2. Scripts/ci/check_sdk_name_collisions.py:105-111,121 - With a user-supplied --scratch-path, a failing dump-symbol-graph (nonzero exit) falls back to a scratch.rglob("symbolgraph") / rglob("<module>.symbols.json") search. That can pick up stale graphs from an earlier run, and the code only emits a warning. The checker could then pass on an outdated public surface. Default CI use (fresh temp dir) is not affected. Fix: tolerate nonzero exit only if the graph files are newer than the start of the run, or use a fresh directory.

### Minor
3. CHANGELOG.md:~16 - "macOS 27 / iOS 27 SDKs add Foundation.ProgressReporter": only the macOS 27 SDK is evidenced (the checker output and the build failure). The iOS 27 claim is unverified. Drop it or mark it unverified.
4. Scripts/ci/check_sdk_name_collisions.py:19-20 (docstring) - "other imported modules shadow stdlib declarations" is a correctness claim with no named test and no derivation from adjacent code. Mark it as an unverified contract (it is the root's stated rationale).
5. Scripts/ci/sdk_name_collision_allowlist.json / check_sdk_name_collisions.py:164 - Allowlist matching is keyed on (name, sdk_module) only, not on our module. One entry therefore allows both VectorCore.Logger and the re-exported VectorCoreBenchmarking.Logger (the green log shows both "allowed"). This is deliberate but undocumented in the file. Stale allowlist entries (no longer colliding) are also never flagged; consider reporting them so the allowlist can be emptied after D02.
6. Scripts/ci/check_sdk_name_collisions.py:132-133 - The `xcrun --show-sdk-path` return code is not checked. An empty path becomes "" and fails later with a less clear exit-2 message from extract. Behavior is still correct (exit 2).
7. Scripts/ci/check_sdk_name_collisions.py:68-70 - KINDS also includes swift.actor, which is not in the brief's list (type, protocol, typealias). It is harmless and arguably correct. Note only for completeness.

### Critical
None.

## Checks performed
- Rename completeness: grep of the workspace for the whole word ProgressReporter, excluding .git/.build/execution/audits/reviews dirs. The only remaining hits are the CHANGELOG migration text and the checker docstring, both intentional. Hits in Docs/audits and Docs/superpowers/execution are historical evidence manifests (untracked, not API docs). ProgressFormat is untouched. Diff shows only mechanical type-name substitutions plus a 98% similarity file rename (type decl and the one extension). No behavior change. No deprecated alias, as required.
- Spec items otherwise met: rename and file rename, all listed references updated, no edits to Package.swift, .github, or Scripts/ci/tests; no new tests; CHANGELOG entry under [Unreleased] with migration; no additional compile fixes needed (report states none; builds corroborated by logs); RED and GREEN retained; exact CI command reported; Logger allowlist entry present (apart from finding 1); ruling 3 (release build-tests) is reflected in the report as a pre-existing non-product issue, and release evidence comes from `swift test -c release`.
- Symbol-graph extraction: filters on kind in {struct, class, enum, protocol, typealias, actor}, accessLevel in {public, open}, and len(pathComponents)==1. This is correct for the top-level vs nested distinction. Both modules are handled: package modules from `dump-symbol-graph`, SDK modules via `swift-symbolgraph-extract` with the active SDK and target triple. The VectorCoreBenchmarking count (254) includes re-exported VectorCore symbols, which explains the duplicated Logger collision (benign).
- Exit codes: 0 clean, 1 unreviewed non-Swift collision, 2 tool failure (ToolError, ValueError/JSONDecodeError, KeyError, OSError). Swift-stdlib collisions are informational only. Logs: checker-red.txt shows a RED exit with ProgressReporter; checker-green.txt shows OK.
- Timeouts: all four subprocess call sites go through run() with a finite timeout (1200 s dump, 300 s per extract, 60 s quick); TimeoutExpired maps to ToolError -> exit 2. No other subprocess use (grep). `python3 -B -m unittest discover -s Scripts/ci/tests -p 'test_child_process_bounds.py'` in the workspace: 7 tests OK.
- Counts vs logs: build-debug-1 "Build complete"; L03 debug and release "39 tests in 9 suites passed"; M10 debug and release "Executed 20 tests, 0 failures" plus Swift Testing 21 tests in 3 suites plus 1 in 0 suites (=22); swift62-build-2 "Build complete"; build-release-5 "Build complete (205.40 sec)"; build-release-2 fails with "unable to resolve Swift module dependency" (the testable-import failure, consistent with ruling 3). Checker counts in report (229, 254, 491, 324, 305, 302, 51, 24) match checker-green.txt and checker-red.txt exactly.

## Cannot verify from diff
- That the Foundation graph (491 names) fully covers types that Foundation re-exports from FoundationEssentials/FoundationInternationalization; the ProgressReporter hit proves coverage only for that case. Not rebuilt.
- Lint counts (126 serious before and after; 549 to 550 total) were not re-run.
- The 9 L03 children count was not independently cross-checked beyond the "39 tests in 9 suites" summary line.
- The Swift 6.2 result is taken from swift62-build-2.log's Build complete line; the toolchain version string was not re-checked.
