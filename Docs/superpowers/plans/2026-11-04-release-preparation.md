# 0.4.0 release preparation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox syntax. The user explicitly authorized three independent parallel tracks; use separate worktrees and task-scoped review before integration.

**Goal:** Resolve or precisely bound the imported Collection witness defect, repair automatic row scheduling, and harden metadata subprocess lifetime before the final 0.4.0 qualification pass.

**Architecture:** Preserve the public vector API and normal package compilation modes. Repair each defect at its owning boundary, prove its regression with actual consumers or subprocesses, and integrate onto the existing carried branch. Performance comparisons run after concurrent build work has finished.

**Tech Stack:** Swift 6, SwiftPM, Darwin/POSIX, Python evidence tooling; qualified Swift.org 6.4.2 on macOS arm64.

**Spec:** The three-track description approved in the conversation on 2026-11-04; this plan records its acceptance criteria. Investigation-specific implementation decisions remain with each owner, constrained below.

## Global Constraints

- Base is `d448a3e694ea85ba3e6b7ac6a2f3d28cdaedcafc`; retain every earlier commit on `codex/audit-top-six-2026-10-27`.
- Work remains local. Do not push, merge to main, tag, publish, or submit external messages/issues.
- Preserve existing worktrees and frozen evidence. Each worker owns its new worktree and new evidence directory only.
- No public API removal, dependency addition, platform-floor change, optimization suppression, new skip, or known-issue masking to obtain green tests. Escalate a necessary contract change to the coordinator.
- The package tools/language floor remains Swift 6.0. Qualification uses the exact compiler/SDK in the worker context; Xcode's default compiler is comparison-only because of the earlier numerical canaries.
- Before a production change, reproduce the failure with a behavioral test or real importer. Record baseline/red/green command, exit, logs, source identity and relevant hashes.
- Correctness claims must name a test, follow in one step from adjacent code, or explicitly state an unverified contract. For a second instance, cover the defect class mechanically. Re-derive reachability after routing changes.
- Keep test-only seams in tests. All subprocesses used for experiments need finite execution/cleanup bounds and retained output. Do not assume Foundation.Process descendants share the outer process group.
- Do not expand broad test policy exemptions. If new scripts require exact policy registration, include narrowly scoped controls.
- No worker subagents. Workers perform focused Debug/Release checks; the coordinator owns the combined full Debug/Release, Python, claim, workflow, whitespace and pinned-lint checks.
- Do not run performance comparisons during concurrent builds or tests; prepare a bounded reproducible driver for the coordinator if needed.

## Review Focus

- Imported defaults must work as public API through generic/existential Collection access, not only @testable clients. Task 1 owns compile/link/runtime controls.
- Zero-width, ragged, single-core and nested operations must remain bounded and correct after row chunk conversion. Task 2 owns these tests.
- Triangular distance-matrix rows differ in cost; increasing submissions alone is not a latency claim. Task 2 records actual Release comparisons separately.
- A metadata child may return while a descendant remains alive, or the supervisor may die first. Task 3 owns these lifetime scenarios and explicit containment limits.
- PID/group reuse, signal ownership and descriptor inheritance must not turn cleanup into an unrelated-process kill or an indefinite wait. Task 3 owns implementation reasoning and feasible behavioral controls.

### Task 1: Public consumer compatibility

**Files:**
- Inspect/modify: `Sources/VectorCore/Protocols/VectorProtocol.swift` and only directly implicated helpers.
- Reference: `Docs/reviews/2026-11-02-readiness-batch/compiler/` (historical artifacts must not be rewritten).
- Create: `Scripts/ci/consumer_collection_witness.py` and focused contract tests if a durable driver is appropriate.
- Create: `Docs/reviews/2026-11-04-release-preparation/compiler/` for updated evidence and support scope.

**Interfaces:** Owns the inherited `subscript(index: Int) -> Scalar` behavior without deleting it or requiring all downstream conformers to add a new implementation. Other tracks consume ordinary vector APIs and do not modify this protocol. No shared Package changes without coordination.

- [x] Reproduce the known public importer failure against a fresh base Release module and explicit-subscript control. Establish Debug-built-module behavior, then link and run the passing control. Retain exact module/compiler identities.
- [x] Trace the source-level trigger using controlled comparisons, reading the prior investigation to avoid repeating its 19 paired experiments. Test the smallest source-level accommodation; no speculative broad refactor or build-flag workaround.
- [x] Add durable public-client fixtures covering inherited and explicit witnesses, direct/generic Collection access, and ordinary value correctness. A baseline inherited case must fail for the known compiler reason before accepting a fix.
- [x] If a compatible minimal accommodation succeeds, verify compile/link/runtime across Debug/Release module and client modes on the qualified compiler, and compile-control the installed Xcode compiler where feasible. Verify relevant existing Collection/API tests.
- [x] If no defensible accommodation exists after focused investigation, produce a concrete support recommendation backed by the matrix; report the unresolved release decision without claiming the compiler fixed. Do not silently change the public contract.
- [x] Run covering checks, self-review, commit the scoped result locally, and write the report with the precise remaining limits.

### Task 2: AB3 row scheduling

**Files:**
- Modify: `Sources/VectorCore/Operations/ExecutionOperations.swift`, `TransformOperations.swift`, and the internal chunk-hint declaration/documentation as needed.
- Inspect: `Sources/VectorCore/Utilities/BoundedExecution.swift`, `Operations/CPUContext.swift`, `Operations/ExecutionContext.swift`.
- Test: `Tests/ComprehensiveTests/ExecutionSchedulingContractTests.swift` or a dedicated `AutomaticRowSchedulingTests.swift`.
- Create: a focused bounded Release benchmark driver/fixture under the task evidence or maintained benchmark tooling if appropriate.

**Interfaces:** `preferredChunkSize` currently represents Float elements; `BoundedExecution` minimum represents row items at these call sites. Keep worker-count bounds, order, dimension validation, factory errors, cancellation, and nested-region serialization intact. Task 1 may change default accessor implementation without changing its behavior.

- [x] Reproduce the one-worker case with the real `CPUContext.automatic` path on this host at 1000, 2000, and 4096 nonempty rows. Check actual submission observations and output rather than a copied chunk formula. Keep a one-core control.
- [x] Select and document a consistent unit conversion (or operation-specific row minimum), with overflow-safe arithmetic and explicit empty/zero-width handling. Apply it to map, normalize, and distance-matrix call sites; search all consumers of the hint for the same defect class.
- [x] Add regression coverage demonstrating multiple bounded worker submissions for representative wide rows on a multicore context. Include output order/value checks, single-core and nested controls, empty/zero-width/ragged behavior, cancellation and factory-error propagation using existing tests where adequate.
- [x] Re-derive the newly reachable parallel paths. Review captured data and shared writes; preserve numeric operation order within each output and deterministic assembly.
- [x] Run focused Debug/Release checks and prepare paired base/fixed Release measurement with warm-up, medians/distribution, identical data and compiler identity. Ask the coordinator to schedule measurements after other heavy work; do not claim a speedup from task counts.
- [x] Self-review, commit locally, and write the report with benchmark instructions and any remaining cost-balancing limitations.

### Task 3: EnvCapture subprocess lifetime

**Files:**
- Modify: `Sources/VectorCoreBenchmarking/EnvCapture.swift` and a narrowly scoped internal subprocess helper if required.
- Test: existing `Tests/BenchmarkContractTests/RunnerContractTests.swift` (maintain the five-file membership unless coordinated), with test support helpers if necessary.
- Inspect: `Tests/VectorCoreTestSupport/ChildProbeLauncher.swift`, current metadata tests, and process policy rules.

**Interfaces:** Preserve `collectMetadata(...) -> BenchMetadata` and `observe(command: [String], timeout: TimeInterval = 2) -> RuntimeObservation`, including non-macOS unavailable behavior. Define the supported descendant/parent-death contract explicitly; arbitrary processes that deliberately escape ownership cannot be assumed contained.

- [x] Reproduce real metadata command lifetime with controlled executables: success, nonzero exit, a hanging leaf and a hanging grandchild, including the outer supervisor dying. Retain PID/group/parent observations and clean up owned fixtures independently.
- [x] Compare minimal safe implementation approaches, then implement the least invasive one that gives bounded execution and cleanup under its stated ownership model. Avoid unsafe post-fork Swift/Foundation work, process-wide signal handlers/reapers, PATH/shell interpolation vulnerabilities, and PID reuse races.
- [x] Add behavioral tests for successful output, failure/timeout observation, descendant cleanup after normal leader exit and timeout, and the actual parent-death guarantee chosen. Exercise real processes under a separate bounded controller; use handshakes rather than fixed startup sleeps.
- [x] Ensure returning metadata cannot wait indefinitely on descendant output handles, and inability to observe/clean up is reported truthfully. Preserve executable/argument boundaries and environment semantics; bound output where required by the implementation.
- [x] Run focused Debug/Release and relevant subprocess-policy checks. Explain every ownership limitation; do not claim arbitrary orphan containment without evidence.
- [x] Self-review, commit locally, and write the report, including production resource ownership and retained red/green evidence.

## Integration and completion

- [x] Task-scoped spec/quality reviews on each complete diff; fix findings through the original implementer and scoped re-review.
- [x] Cherry-pick approved task commits onto the carried branch; resolve any shared policy changes explicitly.
- [x] Run combined full Debug and Release, full Python discovery, claim ledger, workflow validation, whitespace and pinned SwiftLint. Record warnings and skips accurately; do not broaden testing after clean results without new cause.
- [x] Run the paired AB3 Release performance experiment with concurrent heavy work stopped; retain the limitations of a single-host measurement.
- [x] One broad final review of this batch plus affected boundaries, then one consolidated fix wave if needed and covering checks.
- [x] Commit a concise readiness report and evidence manifest, preserve branch/evidence, and state which release qualifications remain (Swift 6.0, real sanitizer runtime, hosted CI, other supported platforms and controlled broader performance).


## Completion record

Final implementation `62780666be88554ca2859e2e92cd5d3ad6666dba`; full Swift/D02 source `60a87a11b58c28064626fc48023855c4b9042945` with 387 byte-identical package inputs. All Critical/Important task and final review findings addressed; combined Swift, final Python and required consumers pass. No public compiler-support contract was removed. See `Docs/reviews/2026-11-04-release-preparation/README.md` for the retained ownership/qualification limits, warnings, scoped performance results and deferred Minor follow-ups.
