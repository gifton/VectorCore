# T01 re-review, fix round 1

Verdicts (paths relative to the workspace):
1. Allowlist reason: ADDRESSED. Scripts/ci/sdk_name_collision_allowlist.json:5 names "Owner: D02", "D03 preserves the Logger signature", and "Per root ruling 2026-10-03 (T01 brief-addendum item 2)".
2. Stale graphs: ADDRESSED. check_sdk_name_collisions.py:166-170 always creates a fresh scratch (mkdtemp "run-*" under --scratch-path, or tmp/scratch). Line 97 rglobs only that directory. The "Files written to" and older-run fallbacks are gone. No fresh graph exits 2 (lines 98-101). A nonzero dump exit with fresh graphs still passes with a warning (lines 103-107); those graphs come from this run. Truncated JSON gives JSONDecodeError (a ValueError) and exit 2.
3. CHANGELOG iOS claim: ADDRESSED. CHANGELOG.md:16-17 now says "macOS 27 SDK ... (other OS 27 SDKs are unverified)".
4. Stdlib-shadowing statement: ADDRESSED. Docstring lines 19-21 say "Unverified rationale (root's stated reasoning, no test exercises it)".
5. Sentinel self-check: ADDRESSED. SENTINELS is at lines 54-61, with Foundation URL, Data, Date, UUID, AttributedString and ProgressReporter. The check runs inside sdk_names (lines 124-127) and raises ToolError, so the exit is 2. The green log (checker-green-fix1.txt) shows the sentinels passing on the real SDK. The implementer reports a negative check that exited 2; I did not re-run it.
6. Keying documentation and stale warning: ADDRESSED. Keying is documented in the docstring (lines 22-25) and in the allowlist reason. The stale WARNING is at lines 191-194, computed from allowed minus used, and it does not change the exit code. The path was not exercised by a run, but it is a one-step set difference.
7. xcrun --show-sdk-path: ADDRESSED. Lines 155-161 check the return code and the empty output, and raise ToolError (exit 2).
(Old finding on swift.actor in KINDS: no change, per root ruling. Confirmed unchanged.)

## New-breakage checks
- Stale reads: with the fresh run directory, the only way to read an old graph is for the current run's dump to write one into an old location. That cannot happen, because the scratch is new and empty. No issue.
- Timeouts: every subprocess call goes through run() (subprocess.run at line 70, the only use). The timeouts are 1200 s for the dump, 300 s per extract and 60 s for the quick calls (xcrun --show-sdk-path, xcrun swift -print-target-info). TimeoutExpired becomes ToolError, so the exit is 2. The bounds scanner test passes (7 tests OK).
- Exit codes: 0 clean, 1 collision, 2 for ToolError, ValueError, KeyError and OSError. The new error paths (sentinel, xcrun, no fresh graph) all raise ToolError. A missing xcrun raises FileNotFoundError, an OSError, so the exit is 2. The red log shows exit 1 and the green log shows exit 0.
- Stale-warning false positives: an allowlist entry whose module is "Swift" would always warn, because Swift collisions never consult the allowlist. This is harmless, since it is only a warning.

New breakage: none.

## Out-of-scope notes (not blocking)
- With --scratch-path, the run-* directories are never cleaned up, so they accumulate. Each run is a full rebuild; the implementer acknowledged the cost.
- CHANGELOG.md:16-17 now has an over-long line after the edit. This is cosmetic.
- The stale-warning path and the xcrun/no-graph error paths have no automated test; this is a gap in the class-closing discipline, but T01 forbids adding tests.
