# AP7, LC5, and CI hardening implementation plan

> For agentic workers: use the subagent-driven-development workflow. The user explicitly authorized three isolated parallel workers; the coordinator integrates and reviews their work.

**Goal:** Close AP7 no-finite precision analysis, LC5 UMAP curve preflight, and the first bounded CI-hardening batch before 0.4.0 readiness validation.

**Architecture:** Three disjoint implementation worktrees start at ec7f3323529ce3df6c126e8b6416f6e4f67d16ba. The coordinator owns shared policy data, evidence, integration, and combined validation on codex/audit-top-six-2026-10-27.

**Tech Stack:** Swift 6.4.2 (qualified official Swift.org toolchain), Swift Testing/XCTest, Python unittest, repository CI policy checkers.

**Spec:** The user's approved AP7 → LC5 → test/CI sequence and parallel breakdown in this conversation. The contracts below make those approved repairs concrete; no separate design document is required.

## Global Constraints

- Preserve all 28 carried local commits; no push, remote PR, release, main-branch changes, or sibling repository edits.
- Workers own only their task files. Do not edit claim_ledger.json, shared documentation, policy byte pins, or other task source. Report required coordinator changes.
- No public API changes, new dependencies, unsafe compiler flags, suppressed canaries, or broadened numerical tolerances.
- Follow the user AGENTS.md review discipline: back correctness comments with named tests or adjacent derivation; a repeated defect class needs a mechanical class test; rederive reachability when changing a gate.
- Focused regression tests must fail against baseline behavior and pass after the repair. Do not run full Swift or Python suites as workers; root runs combined checks once. Preserve exact command/log/result evidence.
- No nested agents. Commit task files on the assigned isolated branch, leave the tree clean, and report commit IDs, checks, shared-data changes needed, and residual limitations.
- Use the task's own build scratch. Never write the shared root build-debug/build-release caches. Do not run Release builds as workers.
- Toolchain/environment and report locations are in /private/tmp/vectorcore-remediations-2026-11-01/worker-context.md; read it before execution.

## Review Focus

- AP7: an all-special first vector followed by finite values must analyze the finite values, preserving supported dimensions.
- LC5: cheap shape/config validation should retain useful precedence while unsupported curves fail before graph/PCA/fuzzy work in both public routes.
- CI counts: mixed XCTest/Swift Testing logs must retain valid representative passes while skipped-only XCTest fails.
- Optimized Python: negative fixtures must still fail under -O and -OO; mechanical protection must cover both runtime drivers.
- JSON keys: escaped aliases and nested duplicate keys must fail, while repeated keys in distinct object scopes remain valid.

### Task 1: AP7 no-finite precision observations

**Worktree:** /private/tmp/vectorcore-remediations-2026-11-01/ap7
**Files:** Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift; Tests/ComprehensiveTests/PrecisionStatisticsRangeTests.swift.
**Interfaces:** analyzePrecision<V: AnalyzableVector>(_ vectors: [V]) -> PrecisionProfile and selectOptimalPrecision(for:errorTolerance:) remain public and unchanged.
**Contract:** Empty input retains its existing all-zero FP32 sentinel with dimension 0. Nonempty input with no finite observations returns the same finite zero statistics, expectedError 0, outlierCount 0, .fp32 recommendation, but preserves the first vector's scalarCount. Zero expected error is a sentinel, not an accuracy measurement. Finite filtering, existing statistics, and heuristic policies remain unchanged.

- [x] Add regression coverage for supported widths 512/768/1536: all NaN, positive/negative infinity, mixed specials, multiple all-special vectors; every numeric profile statistic is finite zero, precision FP32, correct dimension, no outliers. Verify selectOptimalPrecision stays FP32 for default and permissive positive tolerance.
- [x] Pin controls for empty input, constant zero, and an all-special first vector followed by finite observations. Preserve existing large-offset, range and mixed-finite tests.
- [x] Run focused tests against baseline and capture the expected failures.
- [x] Guard the zero finite count before division and downstream heuristics; document sentinel semantics without overstating guarantees.
- [x] Run the focused test suite with the qualified toolchain; self-review; commit only owned files; write task-1-report.md in the shared scratch directory.

### Task 2: LC5 early UMAP curve preflight

**Worktree:** /private/tmp/vectorcore-remediations-2026-11-01/lc5
**Files:** Sources/VectorCore/ManifoldLearning/UMAP.swift; Tests/ComprehensiveTests/UMAPRangeContractTests.swift and/or a focused new UMAPPreflightTests.swift. Existing UMAPTests.swift may be edited only if required for a focused regression.
**Interfaces:** Both public umap overloads retain their signatures. Compute (a: Float, b: Float) once per public call, then pass coefficients to the private layout tail.
**Contract:** After cheap config/count/schedule/graph-count or supplied-coordinate validation, reject unpublishable curve coefficients before constructing graph, invoking PCA, allocating random coordinates, or building fuzzy edges. Preserve existing cheap structural/finite-validation precedence. Do not change curve-fitting math, accepted domain, .operationFailed for coefficient publication failure, graph/optimizer math, or seed behavior.

- [x] Add failing regressions proving preflight happens before expensive downstream work for vector random/PCA and graph-only routes, without public test hooks. Use literal rejected coefficient fixtures already established by curveCoefficientPublicationDomain: (1,0), (0x7f7fffff,0), (0x67800000,0x65cccccd) Float bit patterns. It is acceptable to use a counting/access-guard conformer or existing provider seam; establish an observable downstream witness rather than timing assertions.
- [x] Add valid coefficient/seed controls and cheap validation-precedence coverage; retain valid subnormal coefficients (spread 0x50000000, minDist=spread) and small accepted spread 0x2f000000.
- [x] Run focused tests against baseline and capture the expected failure, bounded so an intentionally wrong PCA path cannot hang.
- [x] Move fitting once to each entry point after cheap guards and pass coefficients privately into the layout tail. Do not duplicate fitting.
- [x] Run UMAP-focused suites with the qualified toolchain; self-review reachability for both entries and initializers; commit owned files; write task-2-report.md in shared scratch.

### Task 3: First CI-hardening batch

**Worktree:** /private/tmp/vectorcore-remediations-2026-11-01/ci
**Files:** Scripts/ci/check_performance_gates.py; its existing test-output parser helper (locate import); Scripts/ci/tests/test_test_output_counts.py; Scripts/ci/check_benchmark_build_modes.py; Scripts/ci/tests/test_benchmark_contracts.py; Scripts/ci/check_claim_ledger.py; Scripts/ci/tests/test_claim_ledger.py; new focused policy tests if justified.
**Interfaces:** Existing CLI contracts and check return conventions stay intact. Shared claim-ledger data and byte pins are coordinator-owned.
**Contracts:** A skipped-only XCTest summary cannot prove representative execution. Any actually executed XCTest or Swift Testing representative can satisfy the existing gate. Every load-bearing runtime assert in both benchmark gate drivers becomes an explicit failure that survives Python optimization. Claim-ledger JSON rejects duplicate object keys recursively, including escaped-equivalent names, via normal diagnostic handling.

- [x] Add baseline-failing count tests for skipped-only, partial-skip, zero/malformed, and mixed framework logs; retain both Swift 6.0 and 6.4 formats.
- [x] Add negative command/log/report fixtures for both benchmark drivers executed under normal Python, -O and -OO. Add a narrowly scoped mechanical guard preventing runtime assert gates returning in these drivers. Do not ban unittest assertion methods or scan unrelated policy code into scope.
- [x] Add duplicate-key parser cases at top level/nested/escaped-equivalent names and valid repeated keys in separate object scopes/arrays. Preserve duplicate-ID validation separately.
- [x] Run tests against baseline and capture expected failures. Implement explicit durable checks and strict JSON object loading. Reuse existing helpers if appropriate without introducing new external dependencies.
- [x] Run focused Python suites, including optimized modes. Self-review; commit owned files; report changed source hashes needed by shared pins and write task-3-report.md in shared scratch.

### Task 4: Integration, shared records, and final validation

**Owner:** Coordinator only.
**Files:** Scripts/ci/claim_ledger.json; shared CI byte pins as required; batch plan and audit remediation report under Docs; integration commits.

- [x] Review each worker diff for specification and code quality using fresh reviewers as slots free; fix findings in isolated worktrees.
- [x] Integrate reviewed commits sequentially into the carried branch; update shared claim data/pins based on exact changed bytes and legitimate claim anchors, never relax gates to hide failures.
- [x] Run focused integrated policy checks, full Debug and Release Swift tests, Python policy suite, pinned lint and diff checks. Serialize builds sharing a cache; capture compiler identity and bounded evidence.
- [x] Broad final independent review of baseline-to-integrated diff. Address findings and rerun affected validation only.
- [x] Commit durable plan/report/evidence references and report outcomes/remaining scope. Keep all work local on the carried branch.

## Completion record

Validated implementation: `f5a63053fd246e4b9b3a8f14c005889280d6d725`. See `Docs/reviews/2026-11-01-core-remediations-batch/README.md` for results and limits.
