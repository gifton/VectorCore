# V03 v4 delta review

Spec: PASS
Quality: APPROVED

## Findings

Critical: none.
Important: none.

Minor:
1. Scripts/ci/tests/test_check_required.py:65-76 - `workflow_required_contract` is regex-based. The block terminator `^  \S` would also end the `required` block at a 2-space-indented YAML comment. The parse needs inline `needs: [...]` and a single-line `run:`. Any mismatch raises AssertionError, so it fails loudly and never passes silently. The current ci.yml (lines 212-226) parses correctly. The parity test binds to the live workflow: it asserts JOBS == needs and JOBS == argv, both order-sensitive, and the implementer's mutation run shows it bites.
2. Scripts/ci/tests/test_child_process_bounds.py:153-173 - the detector matches only `subprocess.<name>(...)` attribute form. It does not catch `from subprocess import run` or `import subprocess as sp`. The implementer disclosed this. A grep of Scripts/ci/**/*.py finds no such imports, so there is no false negative in the current tree. The docstring states the limitation. The only remaining `subprocess.Popen` sites (check_memory_pool.py:17, check_benchmark_build_modes.py:56, probes) are outside the stated four-function rule and are not in scope.
3. Verification evidence: the brief asked for a small JSON per command with argv, cwd and exit code. The report instead records these in notes and sha/exit text files. This is a process deviation only and does not affect the source.

## Checks performed

- JOBS equals the seven IDs in workflow `needs` order.
- `run_gate` uses `v03_process.run` correctly:
  - `timeout=15`, `expected=None`, `cwd=ROOT`, and the argv is mapped to str by the helper.
  - Record names are unique per call: `self._calls` resets in `setUp`, the TemporaryDirectory is per test, and subtests increment the counter.
  - Exit code is read from `record["exit"]`, set on the completed path.
  - stderr is read from `record["stderr"]["path"]`, which exists because the `finally` block populates it.
  - NEEDS_JSON handling is unchanged: popped, then set only when the payload is not None.
  - A timeout raises RuntimeError, so the test errors loudly and does not hang.
- The timeouts are real bounds:
  - `process.wait(timeout)` in v03_process.run, followed by a process-group SIGKILL.
  - The `subprocess.run(..., timeout=)` additions at consumer_smoke.py:75-76 and 80 and in the test_consumer_smoke and test_hooks callers.
- The duplicate-key fixture lists all seven jobs, with `test` duplicated (failure first, then success) as the only defect.
- All negative tests are retained and iterate JOBS, so they now cover seven jobs. `assertIn(job, stderr)` is kept via the SimpleNamespace shim.
- The detector is correct on its own terms:
  - Operator precedence in `timeout is None or isinstance(...) and ...` is fine.
  - A `**kwargs` expansion is treated as unbounded (conservative).
  - The negative control covers missing, None, constant and variable for all four functions.
  - The existing wait/communicate rule is untouched.
- The six violations from the addendum are fixed, adding only the timeout keyword, with the specified values: 1800, 600, 60, and 60 x3.
- No other files changed, and no new dependencies were added.

## Cannot verify from diff

- The reported test counts (57 after the first round, 59 after the extension) and the mutation outputs. I did not rerun them, and the evidence files in V03/v4/ were not inspected.
- Whether `check_ci_contracts.py` has a reusable parser (the implementer says no).
