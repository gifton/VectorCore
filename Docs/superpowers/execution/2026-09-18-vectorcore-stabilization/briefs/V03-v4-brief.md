# V03 v4 brief — legacy required-check test migration (root ruling)

V03 ("Executed CI coverage and performance opt-outs") frozen v3 source is independently accepted. Its final runtime gates are pending. Before those gates run, one pending source decision must be resolved, and root has now ruled on it.

## Context

- Workspace: `/Users/goftin/dev/gsuite/VSK/VectorCore/.stabilization-recovery/2026-10-03/workspaces/V03`. This is a git worktree detached at integration `9f5fda2`. Root has copied in all 62 V03 v3 paths (47 changed, 14 added, 1 deleted), and each is byte-identical to `frozen-v3/final-inventory.json`. Historical v1–v3 artifacts live in `/Users/goftin/dev/gsuite/VSK/VectorCore/.stabilization-recovery/2026-10-02/V03/`. Read them if needed, but never modify them.
- Pause handoff: `/private/tmp/vectorcore-stabilization-2026-09-18/Docs/superpowers/execution/2026-09-18-vectorcore-stabilization/reports/V03-pause-handoff.md`.
- The current workflow `.github/workflows/ci.yml` job `required` has `needs: [test, lint, repository-checks, consumer, platforms, benchmark-contracts, sanitizers]`. It runs `python3 Scripts/ci/check_required.py test lint repository-checks consumer platforms benchmark-contracts sanitizers`. The V03 semantic validator (`Scripts/ci/check_ci_contracts.py`) is source-accepted against that seven-job set.
- The legacy `Scripts/ci/tests/test_check_required.py` still has `JOBS` with only five IDs and calls `subprocess.run` with no deadline.

## Root ruling (binding)

Edit only `Scripts/ci/tests/test_check_required.py`:

1. Set `JOBS` to exactly the seven IDs, in the workflow's `needs` order: `["test", "lint", "repository-checks", "consumer", "platforms", "benchmark-contracts", "sanitizers"]`.
2. Close the drift class mechanically. Add one test asserting that `JOBS` equals both the `needs` list of the `required` job in `.github/workflows/ci.yml` and the job-ID argv of its `check_required.py` invocation. If `check_ci_contracts.py` already exposes a parser or helper for these, reuse it. Otherwise use a minimal stdlib parse: PyYAML is not available, and you must not add dependencies.
3. Make `run_gate` bounded and file-backed. Use `Scripts/ci/v03_process.run` (the helper already used by `test_validate_github.py`) with `timeout=15`, `expected=None` (so nonzero exits are returned rather than raised), and a fresh per-test temporary output directory. Remember the helper refuses to overwrite a record, so names must be unique per call. Read the exit code and stderr from the helper's record and files. Preserve the `NEEDS_JSON` environment handling exactly: remove it when the payload is `None`, set it otherwise.
4. Retain every existing negative case, now across all seven jobs:
   - every unsuccessful state for every job;
   - each job missing;
   - an extra job;
   - every malformed payload;
   - every invalid expected-ID set.
   Do not drop or weaken any assertion. Keep the `assertIn(job, result.stderr)` checks.
5. Make the duplicate-key fixture otherwise complete. It must list all seven jobs exactly once, apart from the duplicated `"test"` key (failure first, then success). The duplicate must be the only defect in that payload.

Do not edit any other file. If a correct change appears to need another file, stop and report it to root. Make no Swift builds or tests in this step: Swift work waits for the T01 toolchain fix, which is running concurrently in a different workspace.

## Verification

Evidence dir: `/Users/goftin/dev/gsuite/VSK/VectorCore/.stabilization-recovery/2026-10-03/V03/v4/`. Create it. Write each command's stdout and stderr to files there and record the argv, cwd and exit code in a small JSON file.

1. Run the full policy discovery from the workspace root with the same argv as accepted v3: `python3 -B -m unittest discover -s Scripts/ci/tests -v`. v3 ran 56 tests, all OK. Report the new total and explain the delta, which is your added parity test plus any subtest-structure changes.
2. Run the migrated module alone: `python3 -B -m unittest Scripts.ci.tests.test_check_required -v`, or an equivalent `-s` form if that import path fails. Record it.
3. Mutation check, to prove the drift test bites: temporarily change one ID in a scratch copy of the test file outside the workspace (or mutate the workspace file and restore it byte-identically, verifying with sha256 before and after), and show the parity test fails. Retain the output and the restored hash.

## Global constraints (verbatim from program policy)

- Any correctness/robustness claim in comments must name a test, follow directly from adjacent code, or be explicitly unverified. Close repeated defect classes mechanically.
- Keep changes local: no push, merge, release, publication or sibling-repository writes.
- **No git commits, staging, stash, or other index operations.** Root owns the index. Do not dispatch subagents.

## Report

Write the full report to `/Users/goftin/dev/gsuite/VSK/VectorCore/.stabilization-recovery/2026-10-03/V03/v4-report.md`. It must cover:
- the exact diff summary;
- the before and after sha256 of the file;
- each command's argv, exit code and counts;
- the mutation result;
- any concerns.

Return only: a status (DONE, DONE_WITH_CONCERNS, BLOCKED, or NEEDS_CONTEXT), a one-line test summary, and your concerns.
