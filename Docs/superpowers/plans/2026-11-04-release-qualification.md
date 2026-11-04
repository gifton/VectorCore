# VectorCore 0.4.0 qualification plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkboxes to track the qualification evidence.

**Goal:** Execute the remaining compiler, sanitizer and Apple platform qualification cells, identify actual release blockers, and prepare hosted CI against the carried candidate.

**Architecture:** Three independent qualification workers use isolated worktrees and build directories; the coordinator owns hosted CI inspection and the final evidence record. Results distinguish executed passes, observed failures, and unavailable environments. Repairs discovered by qualification require a reproduced failure, a scoped change and covering verification.

**Tech Stack:** Swift 6.0 series and the previously qualified Swift.org 6.4.2 payload; SwiftPM native builds; Apple SDKs/Xcode; ASan/TSan; existing Python CI drivers; GitHub Actions.

**Spec:** User authorization to continue 0.4.0 qualification; remaining boundaries in `Docs/reviews/2026-11-04-release-preparation/README.md`. This is execution of those approved release gates, not a new public API design.

## Global Constraints

- Preserve the carried `codex/audit-top-six-2026-10-27` branch, all previous commits, worktrees and frozen evidence. Base: `d628a2c419300d37c8462d24b2b1069f71867f79`.
- Keep `swift-tools-version: 6.0`, declared platforms, dependencies, public APIs and numerical contracts. Never suppress optimizer passes, assertions, sanitizer reports or required CI gates to obtain a pass.
- `VECTORCORE_TEST_EXTENDED=0` and `VECTORCORE_STRICT_PERF=0` match the accepted baseline; report existing skips/warnings and any cell-specific limitation.
- Use fresh per-cell scratch/output directories, bounded commands, captured stdout/stderr and exact source/toolchain/SDK/runtime identities. Avoid global Xcode selection or installing toolchains system-wide. Verify downloaded toolchain origin/signature before executing it.
- At most two compiler jobs per worker while parallel builds run. Performance timings from concurrent qualification are not benchmark acceptance.
- Workers may repair a demonstrated narrow defect within their owned surface; coordinate shared-file changes first. No subagents spawned by workers. The coordinator supplies independent review.
- Local work is authorized. Hosted execution must use the intended candidate SHA; old remote results cannot qualify new local commits. Prepare the exact remote action before any necessary publication approval.
- Claims require direct evidence; target deployment settings do not prove execution on the oldest supported OS. Compile-only and simulator results do not prove device/runtime compatibility.

## Review Focus

1. A language-mode flag on a new compiler is not a Swift 6.0 compiler run: Task 1 records the executable identity and actually runs it.
2. A sanitizer build or environment variable alone does not prove runtime instrumentation: Task 2 retains actual loaded images, executed body counts, and a diagnostic control.
3. A platform build can accidentally compile only the main product: Task 3 enumerates every library product and checks each supported destination plus Intel compilation.
4. A successful old hosted run can be attached to the wrong candidate: coordinator compares remote run SHA, workflow ref and local candidate.
5. A failed environment must not become a package pass: all tasks retain failure diagnostics and clearly label unsupported/unavailable cells.

## Task 1: Actual Swift 6.0 series qualification

**Files:** Own `Docs/reviews/2026-11-04-release-qualification/swift-6.0.md` and task evidence. Inspect `Package.swift`, existing compiler probes, `Scripts/ci/consumer_smoke.py`, and the Collection witness fixture. Coordinate source/fixture changes before editing them.

**Interfaces:** All workers start from the same candidate; compiler/SDK qualification is independent of sanitizer/platform execution. Use the project's Xcode 16.2 / Swift 6.0-series floor as the first supported combination to seek.

- [ ] Inventory existing official toolchains/SDKs and prior qualification evidence. Obtain an official signed Swift 6.0-series macOS toolchain if absent, extracting it locally and recording exact download/hash/signature/version.
- [ ] Probe real compiler/SDK compatibility and the retained numerical canaries under ordinary Debug/Release optimization. If an installed SDK is too new, distinguish SDK/toolchain incompatibility from library failure; investigate a legitimate available older combination without modifying the declared floor.
- [ ] On a usable combination, execute full Debug/Release suites and required downstream consumers, including inherited/explicit Collection witnesses. Retain actual body counts, skips, diagnostics and exact input identities.
- [ ] If environment access prevents a cell, preserve the concrete failed command and identify the precise hosted/toolchain requirement. Do not claim the floor qualified from Swift 6.4 language mode.
- [ ] Self-review and commit the task report plus any independently verified scoped repair. Return evidence inclusion paths and remaining gaps.

## Task 2: Real sanitizer qualification

**Files:** Own `Docs/reviews/2026-11-04-release-qualification/sanitizers.md`; inspect/repair `Scripts/ci/check_sanitizers.py` and its direct tests if a real defect appears. Coordinate changes to test support or metadata production code before editing.

**Interfaces:** Use `Scripts/ci/sanitizer_bodies.json` and the existing mandatory driver, not a substitute hand-picked list. Qualified Swift.org 6.4.2 identity is in the preceding readiness report.

- [ ] Record the compiler, SDK, test host and ASan/TSan runtime identities. Execute an isolated intentional diagnostic control for each sanitizer to show instrumentation can detect the intended class; retain the expected failing observation.
- [ ] Run `check_sanitizers.py` separately for address and thread with fresh output/build roots, two jobs and native build system. Preserve all pinned bodies, actual loaded host/bundle/runtime evidence, and Q03 child identities/markers.
- [ ] Investigate failures before changing code. A wrapper/runner accommodation must preserve the actual instrumented workload and runtime; a source defect needs a focused red/green regression and review.
- [ ] Verify metadata's new process path with actual sanitizer runtime propagation where the existing mandatory bodies do not cover it; distinguish protected-system-executable behavior from an instrumentable child. Keep process ownership and deadline limitations explicit.
- [ ] Self-review and commit the report and any covering repair; state exact body counts, exclusions such as leak detection, and remaining host/platform limitations.

## Task 3: Apple platform and architecture qualification

**Files:** Own `Docs/reviews/2026-11-04-release-qualification/platforms.md`; inspect `Scripts/ci/package_products.py`, `.github/workflows/ci.yml`, and platform guards. Coordinate workflow/manifest/source edits first.

**Interfaces:** Product/platform inventory comes from Package.swift; hosted workflow is the reference matrix. Compiler qualification remains separate from compile-only platform success.

- [ ] Inventory installed SDKs, simulator runtimes, available architectures and library products. Record Xcode/frontend/SDK identities without changing global selection.
- [ ] Execute Release compilation for every library product at iOS Simulator, tvOS Simulator, visionOS Simulator and Mac Catalyst using the existing workflow commands and fresh destination scratch roots. Retain concrete missing-runtime/SDK/build failures.
- [ ] Compile library and tests for `x86_64-apple-macosx14.0`; perform a bounded runtime smoke only if existing host support makes it available. Label compile-only results explicitly.
- [ ] Run a meaningful simulator library smoke where an already installed runtime is usable; no device provisioning, paid downloads, or oldest-OS claim based solely on deployment flags.
- [ ] Self-review and commit the report and any scoped portability repair, with exact supported/untested cells and evidence paths.

## Coordinator integration and completion

- [ ] Verify the fresh accepted baseline and source identities; create a plan-specific ledger and isolated worker worktrees. Existing full native suites are retained as baseline, not repeated before every qualification cell.
- [ ] Inspect authenticated remote workflow/ref/run availability; prepare exact candidate-specific hosted CI execution. Do not substitute previous remote checks or change branch protections.
- [ ] Independently review each task's evidence and any diff, integrate approved commits, and run covering checks for actual changes. Re-run a cell only when its relevant inputs changed or its evidence is incomplete.
- [ ] Conduct one final review of the qualification conclusions and any repairs; resolve material findings through one consolidated fix wave and scoped re-review.
- [ ] Commit a concise qualification matrix, remaining blockers and a hashed bounded evidence archive. Preserve branches/worktrees and state precisely which release gates passed and which remain open.
