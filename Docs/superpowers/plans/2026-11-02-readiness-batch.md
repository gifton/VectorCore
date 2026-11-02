# Child-probe and compiler readiness implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement the assigned task. Steps use checkbox syntax for tracking.

**Goal:** Complete the user-approved next parallel group: shared child harness and budgets, guard-specific trap-test evidence, and an evidence-backed compiler support decision; then strengthen the two queued weak tests.

**Architecture:** Preserve the carried repair branch. Three isolated workers share only a stable launcher signature, then their reviewed commits are integrated and validated together. The compiler track produces bounded reproducible evidence rather than speculative production changes.

**Tech stack:** Swift 6 package, Darwin child processes, XCTest/Swift Testing, Python policy checks, qualified Swift.org 6.4.2 compiler.

**Spec:** The user approved the prior chat proposal (three workers plus coordinator): (1) finish launcher integration, standardize ordinary probe budgets, verify cleanup/environment handling while preserving short timeout controls; (2) execution markers, signal/diagnostic checks and guard-removal controls so removal of the intended guard fails its test; (3) reduce inherited-subscript compiler crash, compare an earlier library module, establish affected scope and a support decision. The next small assignment is independent accuracy/override controls. This plan records the implementation choices within that approved scope.

## Global Constraints

- Base is `24c4e57c50b83517f620bb39bc5c8561343ee1a6` on carried branch `codex/audit-top-six-2026-10-27`. Preserve all earlier commits. No push, merge to main, release, external issue submission, or sibling library edits.
- Work only in assigned isolated worktree and new scratch evidence directories. Frozen prior archives are read-only. Do not dispatch subagents; the coordinator owns reviews.
- Public VectorCore APIs, numerical algorithms, and optimization flags are unchanged unless separately evidence-backed and coordinated. No suppressed canaries, disabled tests, weaker policy gates, or blanket timeout inflation.
- Correctness/robustness comments name a test, follow in one step from adjacent code, or state an explicit unverified contract. A repeated defect class needs a mechanical class-closing test. Re-derive reachability when routing gates change.
- Retain Swift tools/language floor 6.0. Local testing uses the qualified Swift.org 6.4.2 compiler and SDK in `worker-context.md`; do not use the installed Xcode compiler for qualification.
- Workers run focused baseline/changed-path tests and negative controls. The coordinator runs final full Debug, Release and Python suites once against the integrated unchanged source. Do not duplicate full suites in every worker.
- Preserve intentionally short timeout self-tests, Q03 sanitizer-specific deadlines, and explicit unreaped/error states. Selection shape is not proof of exact single-test matching; do not claim Swift 6.0 or sanitizer runtime validation without executing it.
- Evidence includes source revision/hashes, exact commands, compiler/SDK, bounded process exits, expected failures and controls. No unbounded hanging SwiftPM mutation runs; directly supervise the testing helper where needed.
- Reports return DONE, DONE_WITH_CONCERNS, NEEDS_CONTEXT or BLOCKED, commits, tests and remaining limits. Write full details to the assigned report file.

## Review Focus

- A child exits but leaves descendants: shared launcher must retain group cleanup and bounded returns (Task 1 real controlled child/descendant tests).
- Parent selector or instrumentation environment leaks/losses: retain selector refusal/recursion checks and observe safe environment sentinels (Task 1).
- A different downstream trap mimics the intended guard: include progress/side-effect witnesses and isolated guard deletion (Task 2).
- Release removes diagnostic strings: assertions must retain a meaningful Release witness; record any unclosed guard instead of asserting false coverage (Task 2).
- A compiler failure is misattributed to UMAP or current changes: compare minimal explicit/default witness clients against current and pre-remediation library/module controls (Task 3).

### Task 1: Shared launcher, ordinary budgets and runner integration

**Worktree:** assigned by coordinator. Own `Package.swift`, shared support target, existing `Tests/ComprehensiveTests/ChildProbeLauncher.swift`, `ChildProbeLauncherTests.swift`, `Tests/BenchmarkContractTests/RunnerContractTests.swift`, launch-policy/CI-contract tests and direct consumers outside Task 2's files. Source membership policy updates are allowed when precise and necessary. Coordinator owns shared hash pins, claim ledger and release/report docs.

**Interfaces:** Preserve `launchChildProbe(_ selector: ChildProbeSelector, environment: [String:String], output: FileHandle, deadline: Double, cleanup: ChildProbeCleanup, ...) -> ChildProbeRun` and existing return/selector/cleanup meanings. Move implementation into a test-only `VectorCoreTestSupport` target at `Tests/VectorCoreTestSupport`; expose required declarations with package access. Consumers explicitly `import VectorCoreTestSupport`. Produce `ChildProbeBudget.standard.deadline` and `.cleanup` with types Double and ChildProbeCleanup; choose a documented ordinary budget grounded in current callers. Notify coordinator early with foundation commit so Task 2 can consume it. Do not edit Task 2's five files or `P01Support.swift`; list import/budget integrations they need.

- [x] Inspect runner launch and launcher lifecycle; run focused baseline and add failing migration/policy/environment controls before change. Existing descendant controls are reusable; do not rebuild the harness without a demonstrated defect.
- [x] Share launcher across ComprehensiveTests and BenchmarkContractTests, replace pinned leaf Foundation Process exception, and ratchet policy so it cannot regress. Preserve benchmark test membership and production runner behavior. Add tests for ordinary exit/failure, bounded hang/cleanup and environment sentinels at the migrated boundary.
- [x] Standardize ordinary caller budgets, including the ExecutionBoundary 5-second budget via Task 2 integration. Keep deliberate deadline self-tests and Q03 special budgets explicit. Record this as consistency/diagnosability unless a flake is actually reproduced.
- [x] Trace EnvCapture descendants and report whether group containment reaches them; test what is safely observable. Do not claim sanitizer runtime propagation from environment-string checks alone. Do not redesign production EnvCapture speculatively.
- [x] Run focused Swift tests and affected Python policies; commit source changes, write report with files/commands/results and residual selector/reaper/toolchain limitations.

### Task 2: Guard-specific trap-test reliability

**Files owned:** `Tests/ComprehensiveTests/D02DimensionReachabilityTests.swift`, `D02DegenerateWidthTests.swift`, `HeapMembershipContractTests.swift`, `ExecutionBoundaryTests.swift`, `P01TopKBufferRedTests.swift`; supporting `P01Support.swift` if this is the actual helper name (locate it). New test-only witness helpers and a reproducible bounded mutation runner/manifest under `Scripts/ci/` are allowed; coordinate new support declarations with Task 1. No committed production guard removals.

**Interfaces:** Existing `launchChildProbe` signature/result unchanged. Task 1 supplies `VectorCoreTestSupport` import and `ChildProbeBudget.standard.deadline/.cleanup`; initially investigate against baseline, then cherry-pick its foundation when coordinator supplies it, adding imports and ordinary budgets only to owned files. Preserve every existing valid-input assertion.

- [x] Map each tested invalid path to its intended source guard, required prior stage and possible downstream trap. Record the exact finite target list before changing it; do not claim historical mutations were executed.
- [x] Demonstrate the observation gap with bounded isolated guard-removal or wrong-trap controls, retaining original source untouched. Capture per-case markers, exit reason and logs.
- [x] Strengthen tests with flushed pre-call markers, expected signal class, no-return marker, Debug named diagnostic, and semantic stage witnesses that discriminate downstream failure in Release. Prefer adversarial fixtures whose later access is observable over source-scanning assertions or test-only production hooks.
- [x] Run each targeted intended-guard removal in an isolated disposable mutation copy and require its corresponding strengthened test to fail for the intended missing-guard evidence; restore source after each. Ensure an unrelated controlled signal cannot satisfy the observation helper. If a targeted case cannot meet that criterion, investigate and report it rather than claiming it fixed.
- [x] Run focused suites in Debug and Release (coordinate build-resource use), integration with Task 1 imports/budgets, and mutation-runner tests if added. Commit only restored tests/tooling. Write report including a per-guard mutation table and limits.

### Task 3: Compiler limitation reduction and support decision

**Files owned:** new `Docs/reviews/2026-11-02-readiness-batch/compiler/` reproducer/report files and optional standalone runner within that directory. All probes/build caches go in new assigned scratch; do not mutate production or existing evidence. Prior evidence is read-only at `/private/tmp/vectorcore-remediations-2026-11-01/compiler-evidence` and `/Users/goftin/dev/projects/VectorCore-remediations-2026-11-01-evidence`.

**Interfaces:** A reproducible command accepting compiler/SDK paths, source fixture(s), explicit-vs-inherited witness controls, and machine-readable exit/hash metadata. No public library API change expected.

- [x] Reproduce the existing default-only failing imported VectorProtocol Collection witness and explicit-only passing control under the qualified compiler. Remove LC5 counter/Foundation/UMAP dependencies stepwise to find the smallest supported trigger.
- [x] Compare the same importer against a pre-remediation library revision (`ec7f3323529ce3df6c126e8b6416f6e4f67d16ba`) built with the same compiler/flags/SDK, using an archive/export rather than modifying another worker. Establish current-vs-earlier module behavior and record exact source/module hashes.
- [x] Try a standalone minimal two-module protocol reproduction, separating default Collection witness from unrelated protocol requirements and build flags. Use bounded compiles. Inspect emitted SIL/symbol evidence if it narrows root cause; an unresolved exact compiler symbol is an explicit limit, not a guessed diagnosis.
- [x] Assess affected compiler/API scope using available actual toolchains as comparison probes only; never substitute them into library qualification or weaken flags. Browse primary Swift sources if relying on compiler technical facts not established by local evidence. Do not file an external issue.
- [x] Commit a small rerunnable reproducer, outcome matrix and practical support decision with precise limits (including whether explicit conformer subscript remains the narrow accommodation). Report evidence paths; no claim of compiler fix unless actually demonstrated.

### Task 4: Independent accuracy and override controls

**Files:** `Tests/ComprehensiveTests/KernelAutoTunerTests.swift`, `MixedPrecisionAutoTunerTests.swift` and only genuinely necessary local test fixtures. Start when a worker slot frees; independent of Tasks 1–3.

- [x] Inspect `testAccuracyBaseline` self-comparison and `MixedPrecisionAutoTunerTests.testManualOverride` same-as-fallback selection; identify existing independent reference and precision contracts.
- [x] Change accuracy coverage to a deterministic independent scalar/Double reference and an intentionally perturbed candidate control; choose an override different from the observed fallback. Preserve existing already-repaired KernelAutoTuner override/hash tests.
- [x] Verify controls detect ignored overrides and numerical perturbation without changing production tuning behavior. Focused tests and small isolated negative controls, then commit/report.

### Task 5: Review, integrate, validate and preserve evidence

**Owner:** coordinator with fresh task reviewers and one final batch reviewer. Shared files: claim ledger, exact policy hashes, generated documentation and dated report.

- [x] Review each complete task diff from the common base, resolve findings through its implementer, integrate reviewed commits on carried branch without losing previous work.
- [x] Reconcile shared path references/policy hashes/generated docs where changes require it; exact gate meaning must remain or strengthen. Review shared integration changes.
- [x] Run appropriate final full Debug and Release suites, full Python policies, GitHub validation, pinned lint, claim ledger and diff check against unchanged integrated source. No repetition once passing without a new relevant change.
- [x] Save a dated committed plan/report and a durable hash-verified evidence archive; preserve previous archives and worktrees. State compiler/Swift 6.0/hosted CI/sanitizer/performance limits explicitly.


## Completion record

Final source `a5609bb1acbe50edc5dba089eb2d28ab1d8faa82`, Swift validation `25be224b9e1e5a0e66230e4dad7bad2a83de60ec` with byte-identical compiled inputs; all task/final reviews approved and final full Debug/Release/Python/policy gates passed. Task 2 normal-registration Release acceptance is supplied by that full run. See `Docs/reviews/2026-11-02-readiness-batch/README.md` for exact results and retained limits.
