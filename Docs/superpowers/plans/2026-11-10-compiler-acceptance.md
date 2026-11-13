# Xcode 27.2 Compiler Qualification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce candidate-bound macOS, iOS simulator and physical-iOS evidence under the adopted restricted-support policy, with an independently checked result and explicit Xcode 27.2 hosted coverage.

**Architecture:** Keep collection, numerical assessment, evidence integrity and release admission separate. Add reusable collection/validation tooling alongside the immutable historical checker. Profiles recognize exact diagnostic compositions and configurations; generated receipts bind each execution to the candidate and products actually used. Neither collection success nor scoped qualification grants final release admission.

**Tech Stack:** Python standard library and unittest, Swift 6 language mode, Xcode 27.2 build 27B52, SwiftPM, xcodebuild/xcresult, simctl/devicectl and GitHub Actions.

**Spec:** [Adopted compiler policy](../../reviews/2026-11-10-compiler-acceptance/decision.md)

## Global Constraints

- Work in `/Users/goftin/dev/projects/VectorCore-archive/VectorCore-fixes-2026-10-27-84798d3`; preserve the dirty main checkout.
- Xcode 27.2 build 27B52; frontend SHA-256 `448457da9a8a56741f2c8c4b8931e85b7377fc009c16918cd8ac1a8a3210ea1f`; record and verify every additional SDK/runtime/target identity.
- Prioritize macOS and iOS. Require at least one app-hosted Release consumer execution on a physical iPhone or iPad before stable iOS acceptance, as explicitly selected by the user.
- Preserve deployment declarations and existing required CI. Minimum-OS, Intel, Catalyst, tvOS/visionOS coverage and broader release gates remain unresolved.
- Preserve original canaries, numerical expectations, dated evidence and `bounded_native_qualification.py` with its adjacent profile.
- Do not infer safety from a by-value signature, scanner pass, SDK-only build, simulator run or newer compiler version.
- Preserve production numerical Swift/C. On 2026-11-12 the user separately approved the timing-only batch benchmark opt-in and version="0.4.0", prerelease=nil for a frozen qualification candidate. Earlier alpha evidence remains tooling validation only; all final scopes require the new compiled-input manifest.
- Do not publish, push, dispatch hosted work, install to a device or change signing state merely to prepare the tooling. Prepare those concrete actions and use the session's authorization/permission rules when execution reaches them.

## Review Focus

1. A supported check fails with a known diagnostic's name: reject it unless the entire reviewed diagnostic identity matches (Task 1).
2. A later receipt supplies a new digest for an already pinned canonical file: reject it, including symlink aliases and changes after initial inspection (Task 1).
3. A valid-looking result belongs to another source, SDK, producer configuration, device or invocation: reject the binding (Tasks 1–3).
4. An iOS application launches but tests never execute, or an old result survives: require a new invocation identity and complete numerical records (Task 3).
5. A required CI job is skipped or fails while artifact upload succeeds: reject the qualification aggregate and keep legacy CI independent (Task 4).

## Task 1: Candidate-bound qualification validator

**Files:** Create `Scripts/ci/qualification_records.py`, `Scripts/ci/compiler_qualification.py`, `Scripts/ci/qualification_profiles/README.md`, `Scripts/ci/tests/test_qualification_records.py`, and `Scripts/ci/tests/test_compiler_qualification.py`.

**Interfaces:** `qualification_records.verify_capture(profile: dict, evidence_root: Path, source_root: Path) -> dict` returns verified command/observation records or raises `ValueError`. `compiler_qualification.check(profile_path: Path, evidence_root: Path, source_root: Path) -> dict` returns a report containing `policyId`, `supportPolicyAdopted`, `compilerProfile`, `vectorCoreQualification`, `sourceManifestSHA256`, `scope`, `problems`, and `releaseAdmitted`. CLI: `--profile PATH --evidence-root PATH --source-root PATH --output NEW_FILE`.

- [ ] Add a synthetic complete capture and literal reviewed oracle/profile fixture. Profile fields identify `policyId=vc-0.4.0-xcode27.2-bounded-v1`, source-manifest digest, compiler/SDK/runtime/target/flags, producer/client combinations, required command/artifact pins, full case inventory, positive/negative oracles, and exact excluded diagnostic observations. Expected numbers must come from the fixture contracts, never captured output.
- [ ] Add failing behavioral tests: `test_supported_failure_is_not_a_diagnostic_exception`, `test_missing_duplicate_or_unknown_case_rejects`, `test_incomplete_or_timeout_record_rejects`, `test_unexpected_diagnostic_pass_requires_new_profile`, `test_wrong_source_sdk_runtime_or_flags_rejects`, and `test_no_profile_never_qualifies`. A complete synthetic known-failing profile may yield scoped acceptance; every mutation must yield rejection and `releaseAdmitted is False`.
- [ ] Add integrity tests: duplicate JSON keys, unpinned command receipts, overwritten digest expectations through canonical aliases, producer/module/link mismatches, modified files during verification, output-file overwrite, and attempted evidence-helper execution. Require rejection and no helper side effects.
- [ ] Run `python3 -B -m unittest discover -s Scripts/ci/tests -p 'test_*qualification*.py' -v`; verify the new tests fail for missing functionality while historical tests retain their previous behavior.
- [ ] Implement the interfaces. Use first-registration-wins digest pins, explicit root mapping and a final rehash. Keep the verifier read-only apart from exclusively creating its output. Do not execute historical helper scripts. Start compiler status as `unverified`; set `known-failing` only after checking compiler observations. Always emit `releaseAdmitted=false`.
- [ ] Re-run the focused tests; require all pass. Run the existing lost-store/descriptor policy checks without changing their exceptions. Commit the validator and tests.

## Task 2: Fresh native collection and reviewed profile

**Files:** Create `Scripts/ci/capture_compiler_qualification.py` and `Scripts/ci/tests/test_capture_compiler_qualification.py`. Reuse `Scripts/ci/v03_process.py`, `Scripts/ci/swiftpm_products.py`, `Scripts/ci/consumer_buffer_value.py`, the original compiler probe sources, and `Docs/reviews/2026-11-06-workaround-qualification/ResidualExposure.swift` without modifying their assertions. Create a new native profile under `Scripts/ci/qualification_profiles/` only after capture review.

**Interfaces:** `collect_native(source_root: Path, swiftc: Path, sdk: Path, output: Path, jobs: int) -> Path` produces `capture.json` plus raw receipts/artifacts. CLI requires `--source-root`, `--swiftc`, `--frontend-sha256`, `--sdk`, `--output`, and supports `--jobs` (default 4). The capture envelope contains `schemaVersion`, `invocationId`, source/tool/host identities, command records, artifact inventory, producer bindings, observation groups, and `collectionComplete`.

- [ ] Add failing tests that verify fresh output directories, frontend preflight before a build, source manifests before/after collection, bounded subprocesses, complete Debug/Release producer × Debug/Release client scheduling, and collection of raw diagnostic failures without numerical promotion. A failed compiler invocation or absent product must stop dependent execution.
- [ ] Run `python3 -B -m unittest discover -s Scripts/ci/tests -p 'test_capture_compiler_qualification.py' -v`; require intended failures.
- [ ] Implement the collector using the existing process and product-layout helpers. Run unfiltered normally enabled native suites, the unchanged original compiler matrix and no-write controls, 24-row consumer oracles/migration contracts, and the residual 17-row observations/omitted-write controls. Capture each producer/client configuration explicitly; do not run the historical residual driver that reuses old products.
- [ ] Record full suite identities, skips, issues and native process exits. Keep captured-variable tests strict. Capture optional C-on/off routing observations using `Scripts/ci/check_benchmark_build_modes.py` and bind them separately; they do not establish alternate-architecture or performance coverage.
- [ ] Run the focused tests and an initial real native capture against the chosen current compiled candidate. Review every diagnostic outcome and source/product binding; create the exact native profile only after that review. Validate it with Task 1. Clearly label this tooling-validation candidate if it still contains alpha version constants.
- [ ] Commit the collector, tests, reviewed profile and dated result summary. Keep raw build artifacts outside tracked source and record their canonical locations.

## Task 3: iOS app-hosted collection on simulator and device

**Files:** Create `Scripts/ci/qualification_ios.py`, `Scripts/ci/tests/test_qualification_ios.py`, and `Scripts/ci/fixtures/compiler-qualification-ios/` containing `QualificationHost.xcodeproj/project.pbxproj`, a shared scheme, `App.swift`, `ProbeAdapter.swift`, and a separately compiled witness target. Add per-runtime profiles under `Scripts/ci/qualification_profiles/` after review.

**Interfaces:** `collect_ios(source_root: Path, developer_dir: Path, destination: str, output: Path, producer_mode: str, client_mode: str) -> Path` produces the same capture envelope as Task 2 with `platform=ios-simulator` or `ios-device`. CLI requires these arguments and an explicit destination; it must not select an unrelated attached device automatically.

- [ ] Add failing tests for simulator/device target separation, architecture/SDK/module/object binding, independent producer/client settings, new invocation identifiers, missing completion, stale result files, unknown/duplicate rows, and launch-without-body-execution. A launch exit of zero alone must never pass a numerical assessment.
- [ ] Add adapter checks that compare preserved probe/helper bodies with the reviewed source fixtures. Adapt only program entry/collection plumbing to the app host, record and hash those adapters, and label them as app-hosted compositions. Keep native originals unchanged; app-hosted passes cannot replace native diagnostic evidence.
- [ ] Run `python3 -B -m unittest discover -s Scripts/ci/tests -p 'test_qualification_ios.py' -v`; require intended failures.
- [ ] Implement the minimal app and runner. Include public vector/storage operations, the by-value consumer oracle, separate witness dispatch, negative controls and residual/standalone diagnostic bodies. Write complete numerical records bound to the fresh invocation, binary and runtime; distinguish app lifecycle results from numerical failures. Use matching iOS deployment targets, SDKs and artifact bindings, not the native macOS launch path.
- [ ] Build and execute on an owned simulator destination with Xcode 27.2, preserving fresh full-suite results separately. Exercise Debug/Release producer/client combinations in the app consumer; report any reduced configuration coverage explicitly. Review observations before creating its profile and run Task 1's validator.
- [ ] Prepare the concrete signed-device build and identify its team/destination without changing unrelated signing settings. Once signing/install execution is authorized and resources are available, run at least the Release producer/Release consumer on one physical iPhone or iPad, including complete omitted-write controls and diagnostics. No simulated or build-only result satisfies this requirement.
- [ ] Review the physical observations independently, add its exact profile, validate the capture, and commit the harness/tooling/tests/results. Record device model/OS/architecture while keeping unnecessary unique device identifiers out of public reports.

## Task 4: Explicit Xcode 27.2 hosted qualification

**Files:** Create `.github/workflows/compiler-qualification.yml` and `Scripts/ci/tests/test_compiler_qualification_workflow.py`. Reuse `Scripts/ci/check_required.py`; keep `.github/workflows/ci.yml` and its required jobs unchanged unless a separately reviewed integration requires a change.

**Interfaces:** The user approved two-stage evidence review. A manual collect dispatch receives a full immutable candidate commit and retains unqualified macOS/iOS-simulator captures. A later verify dispatch receives that candidate, the retained run ID and exact reviewed profiles from the immutable workflow revision. Candidate collection and policy verification use separate checkouts. Profile review pins the retained archive digest and provenance before extraction/verification. `qualification-required` consumes both platform results through `check_required.py`; collect mode cannot satisfy it. No fresh capture is accepted using a profile for another invocation.

- [ ] Add failing workflow-contract tests for exact candidate checkout, explicit Xcode selection/fingerprint verification, profile provenance, always-retained raw diagnostics, missing/failed/skipped jobs, and separation from existing `CI Required`. Ensure artifact-upload success cannot satisfy a missing numerical job.
- [ ] Run the new workflow test and `test_check_required.py`; require the new test's intended failure.
- [ ] Implement on the `xcode-27` runner with `/Applications/Xcode_27.2.app`, using the repository's existing pinned actions. Verify actual build 27B52, frontend/SDK/runtime and architecture before using a reviewed profile. A moved runner image needs reviewed scoped evidence; never accept it solely because the label stayed the same.
- [ ] Re-derive and record reachability for the preserved legacy compiler branches, optional C kernels, architecture-specific paths, platform build cells and both aggregates. Inventory exact checks that remain outside the new workflow, including the physical-device receipt and broader release gates.
- [ ] Run focused workflow/aggregate tests, `ruby Scripts/ci/validate_github.rb`, and the full repository Python policy suite once after integrated changes. Fix actual failures before expanding tests. Commit the workflow and its tests.
- [ ] Prepare the exact branch/ref and dispatch inputs for hosted execution. When authorized, run and inspect the actual hosted artifacts, not just the aggregate badge. If runtime identities differ from local profiles, collect and review a new hosted profile before accepting that environment.

The [official runner inventory](https://github.com/actions/runner-images/blob/main/README.md)
lists `xcode-27` as arm64; the [20261104 image record](https://github.com/actions/runner-images/releases/tag/xcode-27-arm64%2F20261104.0337)
lists Xcode 27.2 build 27B52. This identifies a route to investigate, not a promise
that a future scheduled job retains those binaries.

## Task 5: Final candidate qualification and handoff

**Files:** Add dated evidence summaries under `Docs/reviews/2026-11-10-compiler-acceptance/`; update `RELEASE_STATUS.md` and the applicable profiles. No final admission bit is added to the historical checker.

- [ ] Select the actual compiled release candidate before expensive final runs. Record the full source manifest and exact commit; do not substitute alpha receipts or imply documentation-only commits change compiled inputs.
- [ ] Run/revalidate the required macOS, iOS simulator and physical-iOS scopes against that candidate. Reconcile supported checks, all expected diagnostic failures, controls, skips and identities. Re-run hosted acceptance for the same candidate where required.
- [ ] Run `git diff --check` and the affected policy/validator checks, compare the protected historical files with their pre-execution hashes, and verify the dirty main checkout was preserved.
- [ ] Produce a scope table showing each exact combination as accepted, rejected or pending. Retain `compilerProfile=known-failing` wherever diagnostics fail and `releaseAdmitted=false` while other release gates are open.
- [ ] Present remaining minimum-OS/Intel/other-platform, sanitizer, performance and release-reference decisions to the user. Commit the evidence summary and active status. Do not merge, tag or publish as a consequence of compiler qualification alone.

## Plan review status

The user approved this plan and native execution on 2026-11-10. The adopted
policy and physical-device requirement remain binding. Implement in this
session, with an independent review before accepting the integrated result.
