# Broader Release sanitizer experiments

These tools extend the phase-two execution record without changing the pinned sanitizer manifest or CI driver. They report function execution status, not branch coverage, minimum-OS support, production-binary equivalence or release readiness.

`scope_check.py` reads independent discovery plus Swift Testing version-0 events or XCTest started/ended summaries. Missing/empty/truncated output, failed or skipped tests, unknown selectors, nonzero exits and timeouts cannot qualify a complete unskipped scope. `test_scope_check.py` names the negative controls. Failed, skipped and pending selectors are retained individually. Parameterized functions count once; their individual argument cases remain in the raw event stream. Child-probe helper functions can return without an active scenario; their ordinary pass is not evidence that every child scenario executed.

`run_full_suites.py` consumes the mandatory driver's native Release result and a prebuild source identity containing every tracked `Sources`, `Tests` and `Benchmarks` file hash plus `packageSHA256`. It verifies retained bundle/helper/runtime identity, independently inventories both frameworks, then executes both hosts with bounded capture and actual loaded-image checks. It performs no compilation. Example, from the exact source tree used for the supplied build:

```sh
python3 Docs/reviews/2026-11-04-release-phase-two/sanitizers/run_full_suites.py \
  --driver-result /absolute/mandatory-output/result.json \
  --source-identity /absolute/prebuild-identity.json \
  --scope 'exact commits and changed-input inclusion' \
  --output /absolute/fresh-broader-output
```

Every command uses `Scripts/ci/v03_process.py` with the existing diagnostic settings: ASan `detect_leaks=0:halt_on_error=1`, TSan `halt_on_error=1`, extended tests and strict performance disabled. A full-suite outer deadline is separate from the unchanged mandatory driver and normal child deadlines. The installed XCTest host additionally auto-runs Swift Testing on this toolchain; that overlapping text is retained, never added to XCTest counts.

The independent leak fixtures use Darwin `leaks --atExit`, separately from ASan. `LeakControl.c` requests eight 4096-byte allocations in runtime-selected leak/clean modes. `LeakConsumer.swift` exercises 64 scoped DynamicVector normalization and fixed-vector dot-product iterations, then runs the same C controls through `ConsumerLeakControl.c`. Requested allocation bytes and allocator-reported leak bytes differ. Detector capability and the small clean consumer are separate observations; neither establishes project-wide leak cleanliness.

Raw commands, scope lists, controls, images, exact source/binary hashes and the inclusion list live in `/private/tmp/vectorcore-release-phase-two-2026-11-04/sanitizers/`. The baseline experiments reuse hash-verified first-phase bundles and exclude phase-two test changes. Initial broad baseline runs expose child runtime-preload loss; the portability track owns the shared-launcher repair. Final combined-source execution is a separate required record and must identify those changes explicitly.
