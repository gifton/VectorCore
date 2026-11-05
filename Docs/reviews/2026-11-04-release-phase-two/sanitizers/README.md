# Broader Release sanitizer execution — final v2 results

Final native binaries were built from clean candidate **`e7482a962c67dd0b806059b651c9f59eb678a632`**, using separately bound Release/native builds and the unchanged mandatory ASan/TSan drivers (**77/39 required workloads passed**). Both complete frameworks load the expected host, runtime and aggregate bundle; source and qualification-tool before/after checks match.

| Final observed scope | Discovered | Passed | Skipped | Failed | Pending | Host outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ASan Swift Testing | 1,764 | 1,746 | 18 | 0 | 0 | exit 0, 496.50s / 1,200s budget |
| ASan XCTest | 218 | 190 | 28 | 0 | 0 | exit 0, 173.94s / 1,200s budget |
| TSan Swift Testing, complete additional sweep | 1,764 | 1,746 | 18 | 0 | 0 | exit 0, 1,372.03s / authorized 2,400s budget |
| TSan XCTest, reconciled real selectors | 218 | 190 | 28 | 0 | 0 | exit 0, 662.60s / 1,200s budget |

**Strict no-skips qualification is false in every cell.** Explicit skipped IDs remain in the scope records. Parameterized functions count once; argument cases remain in the native event streams. This is current-host arm64 execution, with no release-readiness, minimum-OS, other-architecture or production-binary-equivalence claim.

The original TSan Swift Testing command genuinely timed out at 1,200.19s, leaving 138 pending functions (1,608 passed / 18 skipped); it remains unchanged. A first additional native attempt launched with an incomplete compiler-fixture environment and was interrupted, unqualified. Its native exit is unrecorded; supervisor/abort/process records document cleanup. The corrected additional sweep starts fresh, keeps all 1,764 independent selectors and normal body/child deadlines, matches the original selected environment except its separate Q03 log directory, and completes without merging prior outcomes.

TSan repeatedly emits **“Invalid dyld module map detected” / “Backtraces may be unreliable.”** These diagnostics limit qualification; recognizing their exact spellings does not establish harmlessness or fully reliable sanitizer reporting. Original TSan XCTest discovery admitted the two warning lines as selectors, producing raw **220 discovered / 2 pending**. The later typed-parser commit **`5a3301055f38d4fbf506985e3feedf3da154c7b4`** separately reconciles the unchanged raw streams to the 218 real methods above and retains both diagnostic records. No native XCTest rerun was made for accounting. Unknown/malformed stdout and ambiguous/incomplete inventories reject mechanically; 31 fixture-only tooling controls pass.

See [final-outcomes.md](final-outcomes.md) for exact source/binary identities, chronology, evidence paths and limits. Final supplemental evidence is `/private/tmp/vectorcore-release-phase-two-2026-11-04/sanitizers-final/`; its explicit manifest preserves original/fix selections and excludes the private environment diagnostic. ASan still uses `detect_leaks=0:halt_on_error=1`; the earlier independent `leaks` positive/negative capability and small public-consumer observations below remain separate, bounded results.

# Historical baseline and tooling development — 2026-11-04

These tools extend the phase-two execution record without changing the pinned sanitizer manifest or CI driver. They report function execution status, not branch coverage, minimum-OS support, production-binary equivalence or release readiness.

`scope_check.py` reads independent discovery plus Swift Testing version-0 events or XCTest started/ended summaries. Missing/empty/truncated output, failed or skipped tests, unknown selectors, nonzero exits and timeouts cannot qualify a complete unskipped scope. `test_scope_check.py` names the negative controls. Failed, skipped and pending selectors are retained individually. Parameterized functions count once; their individual argument cases remain in the raw event stream. Child-probe helper functions can return without an active scenario; their ordinary pass is not evidence that every child scenario executed.

`build_bound_sanitizers.py` creates the admitted source/build association. It requires fresh, separate scratch and evidence roots, captures package/source/CI/producer input hashes before and after the unchanged mandatory driver, fixes only this experiment's explicit Release/native/two-job invocation, and binds its successful outer/build command records, exact mandatory result and matching retained bundle. The effective build root is `<fresh scratch base>/<SHA256(prebuild snapshot)>`; admission checks that digest against both actual command argv and the resulting bundle paths, so replacing source snapshots cannot relabel an old build. Failed commands, changed inputs and stale/swapped result or artifact records cannot produce/admit an accepted association. `test_source_binding.py` exercises these controls through the actual producer with an explicitly synthetic driver, including an unchanged selector whose body changes between builds. No native compilation is part of those controls.

`run_full_suites.py` now **requires that association**. The earlier independent `--driver-result`/`--source-identity` interface is removed. Admission rechecks current inputs against both bound snapshots, then verifies the command/result/artifact links before discovery or execution. It retains the resolved XCTest executable alongside the helper/runtime, plus associated command/source/result records. After execution it rechecks the association. Historical baseline evidence remains unchanged and is not retroactively associated by this producer.

From the exact integrated source root, preserving the established verified-toolchain lookup `PATH`, `SDKROOT` and `SWIFT_EXEC` environment:

```sh
python3 Docs/reviews/2026-11-04-release-phase-two/sanitizers/build_bound_sanitizers.py \
  --sanitizer address \
  --scratch-root /absolute/fresh-address-build \
  --output /absolute/fresh-address-bound
python3 Docs/reviews/2026-11-04-release-phase-two/sanitizers/run_full_suites.py \
  --association /absolute/fresh-address-bound/association.json \
  --scope 'exact integrated commits and changed-input inclusion' \
  --output /absolute/fresh-address-broader
```

Repeat with `thread` and different fresh paths. The producer retains all original mandatory output under `mandatory/`; it does not replace the required workload or alter the driver's Debug default. Source capture includes regular files under Sources, Tests, Benchmarks and Scripts/ci even if gitignored; `.build`, `.swiftpm`, `.git` and `__pycache__` cache directories are excluded, and source-directory symlinks are refused. Keep those inputs, Package.swift and the producer helpers quiescent through build and qualification. The before/after mechanism detects persistent drift; it is not an authenticity signature or a proof against a transient edit reverted between snapshots.

Every command uses `Scripts/ci/v03_process.py` with the existing diagnostic settings: ASan `detect_leaks=0:halt_on_error=1`, TSan `halt_on_error=1`, extended tests and strict performance disabled. A 1200-second full-suite outer deadline is separate from the unchanged mandatory driver and normal child deadlines. The unselected baseline XCTest host additionally auto-runs Swift Testing on this toolchain. Bounded single-method/comma-list probes execute exactly one/two XCTest methods with zero Swift Testing functions. The final runner explicitly selects the complete nonempty XCTest discovery list and still requires exact executed coverage; there is no empty-list fallback.

The independent leak fixtures use Darwin `leaks --atExit`, separately from ASan. `LeakControl.c` requests eight 4096-byte allocations in runtime-selected leak/clean modes. `LeakConsumer.swift` exercises 64 scoped DynamicVector normalization and fixed-vector dot-product iterations, then runs the same C controls through `ConsumerLeakControl.c`. Requested allocation bytes and allocator-reported leak bytes differ. Detector capability and the small clean consumer are separate observations; neither establishes project-wide leak cleanliness.

Raw commands, scope lists, controls, images, exact source/binary hashes and the inclusion list live in `/private/tmp/vectorcore-release-phase-two-2026-11-04/sanitizers/`. The baseline experiments reuse hash-verified first-phase bundles and exclude phase-two test changes. Initial broad baseline runs expose child runtime-preload loss; the portability track owns the shared-launcher repair. Final combined-source execution is a separate required record and must identify those changes explicitly.

## Baseline outcome, before phase-two shared-support integration

The hash-verified first-phase native aggregate bundles contain all three test targets. Independent discovery lists **1,757 Swift Testing functions plus 211 XCTest methods**, 1,968 selectors total. Every pinned ASan/TSan manifest selector maps to exactly one discovered function; the mandatory 77/39 workloads and their defaults remain unchanged.

| Baseline cell | Started | Passed | Completed failures | Skipped | Pending | Command outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ASan Swift Testing | 1,739 | 1,651 | 88 | 18 | 0 | exit 1, 361.58s |
| ASan XCTest | 211 | 159 | 24 | 28 | 0 | exit 1, 296.10s |
| TSan Swift Testing | 1,718 | 1,633 | 84 | 18 | 22 | 600s timeout |
| TSan XCTest | 211 | 159 | 24 | 28 | 0 | 300s host timeout after the XCTest summary |

Both XCTest summaries report 168 assertion failures across 24 failed methods. XCTest also automatically launches Swift Testing: its overlapping execution is retained but not added to these counts. The TSan XCTest command times out in that extra phase. The direct TSan run has 21 untouched functions plus the interrupted `VectorShapeContractTests/fixedStorageBoundaryMatrix()` function pending; assertions recorded by the interrupted function do not count as a completed failure. Every exact ID is retained in `baseline-*-scope.json`. No broad baseline cell qualifies an unskipped clean scope.

The newly exposed shared-support defect is lost sanitizer preloading when launching ordinary children from the instrumented host. Focused `AutoTuningConcurrencyTests/directReentrantCallback` controls fail under both sanitizers: the parent loads the correct runtime, the child aborts with “Interceptors are not working” before its intended body. The portability track owns the generic launcher repair and its dual-framework controls. No production data-race or ASan memory-error diagnostic was observed in these baseline logs; that observation does not establish their absence after the child repair. Skips remain real coverage gaps, including disabled/extended performance tests and the C-kernel stub. No failures were filtered out or relabeled as skips.

`actual-parent-images.json` requires dyld evidence from each captured **parent PID** for its actual host, selected runtime and aggregate bundle. `ScopeControls.test_parent_images_reject_child_only_and_printed_path_controls` rejects evidence limited to children or plain printed paths. `test_interrupted_failing_function_stays_pending_until_end` pins incomplete execution accounting. Nine checker tests pass in the final baseline tooling verification.

The standalone Darwin leak control detects seven of eight requested allocations: **35,840 allocator-reported bytes**, versus 32,768 requested bytes across all eight allocations. Its clean control reports zero. Single-allocation attempts initially report zero in both modes, including a locally debuggable fixture; those nonproving controls remain in the record. The actual VectorCore consumer's clean mode exits zero with zero reported leaks; its intentional-leak mode exits one with **eight roots / 40,960 allocator-reported bytes**. Library inputs, reused object/module snapshots, compiler/SDK and resulting executable identities are retained in `consumer-reuse.json` and command records. This does not change the earlier ASan leak limitation.

One failed unsupported XCTest-discovery invocation dumped the inherited environment. The original diagnostic stays private and is explicitly excluded from shareable evidence; the derivative omits the entire environment section while preserving argv and error context. `sensitive-artifact-exclusion.json` and redaction metadata preserve that exception and hashes. The explicit inclusion list and final exact-value scan exclude the raw file. Reviewers should use only `address-xctest-discovery.redacted.stderr`.

At the earlier baseline handoff, the combined-source rerun after shared-launcher and benchmark-contract integration remained pending; that stage is superseded by the final v2 results above. The coordinator authorized a 1,200-second **whole-suite** budget after observing progress in the baseline; mandatory per-body and child deadlines are unchanged. This record does not qualify the changed source or claim release readiness.

The final runner avoids the baseline overlap through `-XCTest <complete comma-separated discovery list>`. `xctest-selector-mode.json` verifies the observed single/comma modes; `test_selected_xctest_terminal_label_still_requires_all_discovery` rejects an incomplete selected scope even with a successful terminal summary.

## Review fix round 1

Independent discovery is now authoritative for Swift Testing headline selectors and counts. With two independently discovered functions and empty events, discovered/pending are exactly two/two; a partial inventory with one completed function reports discovered/passed/pending exactly two/one/one. Event IDs, source locations and event-only state remain in separate `event*` fields; unexpected event inventory cannot expand the independent universe. Exact-set controls cover these cases. The baseline's previously retained counts/lists and archive selection are immutable; separate fix evidence lives in `/private/tmp/vectorcore-release-phase-two-2026-11-04/sanitizers-fix-1/`.

Review-fix verification passes 21 scoped controls, including real fixture producers for ASan/TSan modes and a both-snapshots replacement control that cannot rebind the original build command. These are fixture-only admission observations; the settled-source native producer/full-suite run was still pending at that review stage; final v2 execution is recorded above.

Review fix round 2 rejects non-injective event-ID-to-selector mappings and retains the full conflicting IDs in `selectorCollisions`. A selector remains pending while any associated event body is pending; completed failure or skip outcomes take precedence over another body's pass, without changing the exact event-level outcomes. Even fully completed collisions remain invalid. Raw event pending state independently prevents acceptance. The two collision controls cover untouched, started-but-unfinished, fully passed, failed and skipped second bodies; 23 scoped tests pass. Binding, executable retention and native execution settings are unchanged.

Post-native discovery repair: `discovery.py` version 1 types both stdout inventories before partitioning or constructing XCTest argv. It preserves the two narrowly recognized module-map/backtrace warnings with stream/line/text/PID, rejects every other nonempty untyped record and duplicate/ambiguous selector, and requires complete consistent framework partitions. Known warnings also remain visible in discovery/result metadata; recognition does not establish reliable sanitizer diagnostics. The native binaries remain candidate `e7482a9`; this later qualification-tool revision only reconciles accounting. Raw native invocations/results, the 1200-second TSan timeout and the interrupted wrong-environment execution remain unchanged.
