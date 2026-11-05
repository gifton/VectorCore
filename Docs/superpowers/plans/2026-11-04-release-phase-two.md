# VectorCore 0.4.0 second-phase remediation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkboxes to track execution.

**Goal:** Repair the newly demonstrated iOS test-support portability failure, expand actual sanitizer qualification, and close the benchmark numerical-validation gaps before combined candidate qualification.

**Architecture:** Three isolated workers own test portability, sanitizer execution, and benchmark validation. The coordinator integrates reviewed work and runs the checks that depend on the combined source. Controlled measurements follow competing builds. Compiler support remains a separate explicit release decision.

**Tech Stack:** Swift 6, SwiftPM, Xcode/Apple SDKs, XCTest/Swift Testing, ASan/TSan, existing Python qualification tooling.

**Spec:** The user's request to begin the next phase after the [first-phase report](../../reviews/2026-11-04-release-phase-one/README.md), continuing the requested parallel-agent approach and carried branch. Phase base: `c6124accca922f6ff65c569f3f7b42502eae17ac`. This plan makes the coordinator's sequencing choices explicit; it does not adopt a narrower compiler/platform support policy.

## Global Constraints

- Preserve all existing commits, worktrees and evidence. Integrate on `codex/audit-top-six-2026-10-27`; no main merge, release tag, version bump or publication in this repair phase.
- Keep `swift-tools-version: 6.0`, declared platform floors, public library APIs and numerical contracts. Do not suppress optimization, assertions, sanitizer diagnostics, compiler canaries or required gates to obtain a pass.
- Keep the pinned sanitizer manifest and existing Debug/Release driver defaults. Broader experiments have separately stated discovery, execution, skip and failure counts; they do not replace the mandatory workloads.
- Distinguish platform-inherent subprocess restrictions from ordinary test coverage. Retain process/death-test contracts on supported hosts and keep portable tests reachable on iOS; never blanket-exclude a test target to get a green build.
- Use existing managed worktrees and fresh evidence/build roots, with at most two compiler jobs per worker. No toolchain/runtime installation, global Xcode selection or device provisioning. The existing simulator may be used for the requested tests.
- Capture meaningful commands from the first RED run with bounded deadlines, argv, selected environment, source/compiler/SDK identity, exit/timeout and output hashes using existing capture helpers. Reused artifacts require matching input and executable hashes and a stated source scope.
- Keep correctness observations outside timed benchmark bodies, use independent references, and preserve strict rejection and report-compatibility rules. The six dot routes may replace the demonstrated unsound empirical `1e-6` target with reviewed route-specific absolute rounding bounds, paired with exact-coordinate coverage and retained sensitivity limits; other thresholds stay unchanged. No timing acceptance while builds/tests compete; never override identity or missing-case checks to manufacture a comparison.
- Claims require a named test, an adjacent one-step proof or an explicit unverified contract. Repeated defect classes need mechanical coverage. Re-derive reachability whenever platform/configuration gates change.
- Workers own their assigned paths and coordinate shared files before editing. Coordinator owns `Package.swift`, workflows, current documentation indexes and `Scripts/ci/claim_ledger.json`. No worker subagents; coordinator dispatches independent reviews.

## Review Focus

1. Replacing `Process.TerminationReason` must preserve exit/signal interpretation and unreaped-child behavior on macOS; simulator success must not erase process-containment tests.
2. iOS test builds must contain real test bundles and execute named bodies; empty library schemes and disabled targets are not coverage.
3. Whole-suite sanitizer claims need matching discovery/execution evidence and actual instrumented host/bundle/runtime images, with every failure/skip retained.
4. Benchmark dot/normalize/copy observations must reject deliberately wrong outputs independently of the implementation, without measuring reference work inside timing boundaries.
5. Final source/compiler identities and timing compatibility must match the measured candidate; compiler failures, unsupported leak capability and unavailable runtimes stay visible.

### Task 1: Portable iOS test support and actual simulator execution

**Files:** Own `Tests/VectorCoreTestSupport/ChildProbeLauncher.swift`, directly affected tests under `Tests/ComprehensiveTests/`, platform-only changes in `Tests/BenchmarkContractTests/` after coordinating with Task 3, `Scripts/ci/tests/test_child_probe_launch_policy.py`, `Scripts/ci/tests/test_child_process_bounds.py`, and new `Docs/reviews/2026-11-04-release-phase-two/ios/`. Production sources and package/workflow changes require coordination.

**Interfaces:** Preserve existing launcher selection, deadlines, process-group and result semantics on supported hosts. Produce a portable termination representation and explicit platform capability contract only where needed, plus exact iOS bundle/body/skip/failure evidence. Task 2 consumes the final test-support commit before final candidate sanitizer execution.

- [ ] Retain the existing failed package-scheme evidence and reproduce the minimal `Process` dependency failure against the actual iOS SDK before editing. Inventory all platform-specific APIs and child-launch call sites in the affected test targets.
- [ ] Add focused exit/signal/unreaped-result tests and a mechanical portability/capability contract. Capture failing controls before the repair; preserve the real macOS child-process containment tests.
- [ ] Remove the test-only Foundation type dependency and isolate inherently host-only launch/death-test operations behind explicit, narrow platform contracts. Coordinate any additional compile blocker; keep ordinary numerical tests reachable.
- [ ] Run covering Python and actual macOS launcher tests with the verified toolchain. Build the real `VectorCore-Package` scheme for the existing iOS simulator without target exclusions, then execute the available complete suite with bounded time and retain actual counts and structured results. Resolve demonstrated in-scope portability failures; enumerate any external/toolchain limitation.
- [ ] Commit the narrow repair and report, retain an inclusion list with exact source/output/artifact hashes, and return any integration requirements.

### Task 2: Broader sanitizer execution and leak capability

**Files:** Own new `Docs/reviews/2026-11-04-release-phase-two/sanitizers/` qualification scripts/tests/report. Read existing sanitizer driver, manifests and SwiftPM helpers; coordinate changes to existing shared drivers or Swift tests before editing.

**Interfaces:** Produce a reproducible full-discovery execution record for XCTest and Swift Testing under ASan and TSan, with actual counts, skips, failures and images. Baseline experiments may reuse verified first-phase Release bundles; final candidate evidence must name whether Task 1's changed test inputs are included.

- [ ] Inventory full suite discovery and available runner modes from the exact verified Swift.org compiler. Determine the complete instrumented-bundle inventory and compare selected workloads with the discovered suite.
- [ ] Implement only the bounded qualification support needed to execute and check both test frameworks; test its acceptance against missing/empty/truncated/failing/skipped output controls before trusting a result. Retain every unexecuted or unsupported body explicitly.
- [ ] Execute broader Release ASan and TSan suites with fail-fast runtime diagnostics and actual host/bundle/runtime image proof. Reuse immutable first-phase bundles only after input/hash checks; diagnose failures instead of silently filtering tests. Coordinate test-support defects with Task 1.
- [ ] Probe an available independent leak-detection path with intentional-leak and clean controls. If capability cannot be established, record the exact limitation without weakening ASan settings or claiming clean output proves detection.
- [ ] After Task 1 integration, run the covering changed test-support workloads against the combined source; rebuild broader suites only when their affected inputs require it. Commit scripts/report and retain exact executed versus pending scope and bounded evidence.

### Task 3: Independent benchmark observations and reporting controls

**Files:** Own `Benchmarks/VectorCoreBench/Suites/DotProductBench.swift`, `NormalizationBench.swift`, `MemoryBench.swift`, supporting benchmark correctness/boundary files as required, a new focused test file under `Tests/BenchmarkContractTests/`, direct benchmark Python drivers, and new `Docs/reviews/2026-11-04-release-phase-two/benchmarks/`. Coordinate existing shared test-file, schema, comparator or ledger changes.

**Interfaces:** Preserve case names, timed operation boundaries, strict policy and compatibility validation. Produce independent correctness observations for dot, successful/zero-failure normalization and memory copying; later controlled measurement consumes the final Release product and report identities.

- [ ] Trace the three missing observation families to current code and add focused failing tests for absent/wrong observations. Use independent Double references for arithmetic and complete source/destination comparisons for copying; include negative controls that would pass if the reference reused the implementation.
- [ ] Add validation outside the timing body and thread the resulting observations through existing strict correctness handling. Exercise generic/optimized dimensions and normalization success/failure contracts; retain exact-copy requirements for copying.
- [ ] Resolve the observed seed-0 dot false rejections with the reviewed signed-sum model: `k32=D` for generic, `k32=D/16+5` for optimized, `k64=D`, absolute scale `D*B²`, `B=1`, and the existing evaluation margin. Pin all six counts/domains, real seed-0 outputs and above-bound rejection; add exact single-coordinate contributions over every index for all routes to detect omitted work independently of the conservative random-fixture bound. Preserve the original failures and disclose omission/sign-error sensitivity limits.
- [ ] Run focused contract/policy tests, real strict Release CLI cases for every repaired family, and the actual comparator controls. Execute the existing optional adversarial CSV branch; repair only demonstrated reporting failures with focused controls and compatible documentation.
- [ ] Record any changed boundary/body identity and remaining CSV/provenance/thermal limitations. Prepare a reproducible quiet-host comparison without claiming contended timings as acceptance.
- [ ] Commit the repair and report with source/compiler/product hashes and bounded evidence; coordinate any final combined-source rerun with the coordinator.

### Task 4: Integration, compiler gate and candidate record

**Files:** Own this plan, current documentation/claim indexes and new `Docs/reviews/2026-11-04-release-phase-two/README.md` plus validation/evidence records. Workers supply scoped reports and commits.

- [ ] Record baseline, ownership, worktree reuse and interface conflicts before dispatch; preserve the first-phase archive and prior accepted source-bound evidence.
- [ ] Obtain independent task reviews, route material findings back to their original worker and integrate accepted commits preserving ancestry. Keep compiler acceptance and support options concrete; do not repeat unchanged probe matrices without a new compiler/input or new hypothesis.
- [ ] Run combined Python/policy checks and native/public-consumer checks appropriate to actual changed inputs. Coordinate final iOS/sanitizer execution only after relevant source settles; each result keeps its exact input identity.
- [ ] Once build-heavy work is idle, perform a bounded controlled Release measurement using compatible reports, warm-up, sufficient samples/repeats and recorded host conditions. If identity or host controls prevent acceptance, retain the actual result and precise remaining gate.
- [ ] Obtain final phase review, freeze and verify bounded evidence, update completed plan items, and commit the consolidated report. Final hosted release-candidate acceptance and publication follow only after the compiler/support and remaining qualification gates permit them.
