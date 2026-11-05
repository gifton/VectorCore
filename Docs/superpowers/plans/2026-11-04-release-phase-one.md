# VectorCore 0.4.0 first-phase remediation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkboxes to track execution.

**Goal:** Complete the approved parallel investigation and repair phase for compiler correctness, runtime/sanitizer qualification, benchmark reporting/comparator execution, and documentation reconciliation.

**Architecture:** Three isolated workers own independent tracks; the coordinator owns shared documentation, review and integration. Final combined qualification and release publication are later phases. An observed failure stays in the evidence even when a separate workaround or environment passes.

**Tech Stack:** Swift 6, SwiftPM, Xcode/Apple SDKs, ASan/TSan, existing Python CI drivers, native managed Git worktrees.

**Spec:** The user's approved three-worker assignment and request to begin the first phase; remaining work in `Docs/reviews/2026-11-04-release-qualification/README.md`. Base: `2dcb24091c6dc21be4db409ecdd0756220e93b8c`; its implementation was hosted-tested at parent `f779d92d8945edd46fe0807607d487ee42c711a3`.

## Global Constraints

- Preserve the carried branch, all prior commits, existing worktrees and frozen evidence. No release tag, main merge or publication in this phase.
- Keep `swift-tools-version: 6.0`, declared platforms, public APIs and numerical contracts. Do not suppress optimization, assertions, sanitizer diagnostics or required gates to obtain a pass. A support restriction is a recommendation until explicitly adopted.
- Retain current known-issue predicates, pinned sanitizer workloads, strict benchmark correctness thresholds and report compatibility validation unless a demonstrated defect and a reviewed narrow repair require otherwise; never weaken them to conceal a failure.
- Workers use isolated worktrees/build roots and at most two compiler jobs each. No global Xcode selection, system toolchain installation, runtime installation or device provisioning. Use already available environments; record unavailable cells explicitly.
- Every meaningful execution has bounded time, exact argv, selected environment, exit/result, source/compiler/SDK identity and retained output hashes. Earlier artifacts may be reused only with a checked input identity and stated scope.
- No timing acceptance during competing builds. Benchmark correctness/tooling may run concurrently; controlled measurements belong to the quiet final-qualification phase.
- Claims in comments need a named test, a one-step adjacent proof or an explicit unverified contract. Repeated defect classes require mechanical class coverage. Gate changes require reachability review of every shielded path.
- Workers own only their task paths and coordinate shared `Package.swift`, `.github/workflows/ci.yml`, root README, claim ledger or production-source changes with the coordinator before editing them. Do not spawn subagents; independent review comes from the coordinator.

## Review Focus

1. A compiler's version label or successful process exit does not establish correct numerical output; retain exact binary identity and expected bit patterns.
2. Platform compile success and simulator smoke are not device/minimum-OS runtime coverage; enumerate executed versus unavailable combinations.
3. Release sanitizer selection must apply to the build, product discovery and executed bundle; preserve pinned bodies and actual image evidence.
4. Comparator imports or mocked reports do not prove execution against the actual benchmark product; keep negative controls and reject incompatible/missing cases.
5. Current documentation must distinguish immutable historical records from unresolved work; changed claims retain their ledger history and evidence links.

### Task 1: Compiler correctness investigation and bounded remediation

**Files:** Own new `Docs/reviews/2026-11-04-release-phase-one/compiler/` probes, driver, report and summary. Read existing `Docs/reviews/2026-11-01-compiler-qualification/`, `Docs/reviews/2026-11-04-release-qualification/`, `Tests/ComprehensiveTests/LostStoreCanaryTests.swift` (locate actual path), and compiler policy sources. Production-source or test-policy edits require coordination.

**Interfaces:** Produces exact-toolchain outcome matrix, a reviewed narrow repair if justified, and a supported-compiler recommendation for later qualification. Consumes no other worker changes.

- [ ] Verify the known failing and passing executable identities; reuse retained qualified payloads without installation.
- [ ] Reproduce the original and inside-borrow lost-store probes in ordinary Debug/Release, retaining `1.0` / `3f800000` acceptance and the `2.0` / `40000000` failure.
- [ ] Investigate the optimizer defect with bounded small probes/IR and public primary-source evidence where useful. Evaluate source-level accommodations without changing the original canary or suppressing optimization; establish each variant's exact scope and failure controls.
- [ ] If a library repair is justified, coordinate the specific paths first, capture a focused regression before the edit, repair narrowly and run covering Debug/Release/public-consumer checks. Otherwise provide concrete compiler-support options and the exact unresolved blocker; do not invent a general workaround from one passing probe.
- [ ] Commit reproducible probes/report and any verified repair. State which compiler/platform/API shapes remain unqualified; write the assigned full report and return the short contract.

### Task 2: Available runtime and expanded sanitizer qualification

**Files:** Own new `Docs/reviews/2026-11-04-release-phase-one/runtime-sanitizers/`, `Scripts/ci/check_sanitizers.py`, and its direct tests in `Scripts/ci/tests/test_sanitizer_log.py`. Read `Scripts/ci/swiftpm_products.py`, the pinned manifest, platform workflows and the prior qualification reports. Shared helper/CI/source changes require coordination.

**Interfaces:** Produces current environment inventory, Release sanitizer capability/execution results, bounded additional platform evidence and an explicit missing-environment matrix. Final acceptance of the combined candidate waits for Task 1 and integration.

- [ ] Inventory available toolchains, SDKs, simulator runtimes, architecture execution and attached usable devices through read-only checks. Map missing minimum-OS/device/tvOS/visionOS/Catalyst/Intel/full-simulator coverage; do not download or provision environments.
- [ ] Add explicit `--configuration {debug,release}` selection to the existing sanitizer driver if required, keeping Debug as default. First add direct tests proving build and product-discovery configuration agree, default compatibility is preserved and invalid values reject. Thread the selected configuration through execution/evidence without changing the manifest or diagnostics.
- [ ] Run covering Python/CI-contract checks and actual Release ASan/TSan pinned workloads using a verified compiler and fresh scratch roots. Retain failed bodies and actual loaded images; do not skip optimizer-induced failures. Preserve existing Q03 deadlines and child identity checks.
- [ ] Probe leak-detection capability with an intentional independent diagnostic control before making any coverage claim. Document unsupported capability instead of treating a clean no-op as a pass.
- [ ] Execute a bounded additional runtime cell where an available environment can add useful evidence (prefer Catalyst runtime and existing iOS full-suite feasibility). Preserve exact test scope and setup failures. No repeat of already accepted unchanged compile matrices.
- [ ] Self-review, commit the driver/direct-test change and the report; return exact completed counts, unsupported cells and requirements for final qualification.

### Task 3: Benchmark reporting and mandatory comparator execution

**Files:** Own `Scripts/compare_benchmarks.swift`, reporting/comparator scripts and direct tests under `Scripts/ci/tests/`, and new `Docs/reviews/2026-11-04-release-phase-one/benchmarks/`. Read `Benchmarks/`, `Sources/VectorCoreBenchmarking/`, CI and historical benchmark contracts; coordinate production benchmark/report schema or workflow changes before editing.

**Interfaces:** Produces reconciled open benchmark findings, tested reporting/comparator repairs and a reproducible actual-product comparison command. Final performance acceptance waits for a settled compiler/source and quiet host.

- [ ] Locate the original open reporting/comparator findings (including the previously referenced D089 where available) and trace each against current code. Mark already repaired findings with evidence rather than redispatching historical work.
- [ ] Exercise the actual comparator against valid current-schema reports and negative controls for incompatible identity, missing cases and invalid numerical records; retain baseline failures and intended nonzero exits.
- [ ] Repair demonstrated reporting/execution gaps with focused red/green tests. Coordinate any mandatory workflow wiring; tests must exercise actual execution, not merely module import.
- [ ] Run the real Release benchmark product to generate small strict-correctness reports with recorded source/compiler/build-mode identity; execute the actual comparator on those reports and preserve results. Do not interpret concurrent-host timing as performance acceptance.
- [ ] Run covering tests once against final changed inputs, commit narrow changes/report and identify the exact remaining controlled-measurement gate.

### Task 4: Coordinator documentation, review and phase completion

**Files:** Own root `README.md`, `Docs/README.md`, current release indexes, matching entries in `Scripts/ci/claim_ledger.json`, this plan and new `Docs/reviews/2026-11-04-release-phase-one/README.md` plus validation/evidence records.

**Interfaces:** Consumes the three worker reports. Historical dated reports/archives remain immutable except a clearly labelled forward link where needed; current support claims are not broadened by draft worker results.

- [ ] Record task ownership, baseline, worktree paths and progress in a plan-specific ledger. Check shared-file/interface conflicts before dispatch.
- [ ] Audit current documentation references/counts against source and accepted evidence; repair demonstrated stale current claims while retaining matching ledger history. Avoid broad style rewrites or historical count rewriting.
- [ ] Obtain independent scoped task reviews, return material findings to the original worker, integrate accepted commits into the carried branch, and run covering integration checks for actual changes.
- [ ] Obtain a final review of the phase diff and conclusions; resolve material findings in one consolidated fix wave with scoped re-review.
- [ ] Preserve bounded evidence and commit the phase report with exact source/test associations, remaining compiler/support decisions, unavailable environments and the next combined-qualification steps. This phase is not release sign-off.
