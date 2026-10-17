# V03 v5 review

Spec: PASS
Quality: APPROVED

Reviewed the brief, the report (v4, v5 and v5 round 2), the 8-file diff and the workspace. I ran `python3 -B -m unittest discover -s Scripts/ci/tests`: 75 tests, OK. I ran no Swift builds.

## Binding constraints
- Association checks are not weakened. Bundle and host paths are still required in the actual run log: check_performance_gates.py:128 and :139 and :161, check_sanitizers.py:279. Expected bodies are still required to execute. The checks are now per target: the sanitizer gate looks up `bundles[targets[row['id']]]`, and the perf gate checks Comprehensive or Minimal as appropriate.
- The only loosening is `in_log`, which also accepts the resolved spelling of the same path. That is the same file and is tested (test_swiftpm_products.py, `test_log_association_accepts_reported_or_resolved_spelling`).
- Product discovery fails on missing or ambiguous products (swiftpm_products.py:365-375). The tests cover missing, duplicate, and aggregate plus per-target together. The search is confined to the bin path SwiftPM reports.
- `run()` bounds every subprocess with a finite timeout. The bin-path call uses 120 s, and swiftpm_products.py has no direct `subprocess.` use.
- No new dependencies.
- `verify_body` never accepts "cancelled". It still requires "passed". Round-2 logs contain no "was cancelled".
- check_benchmark_build_modes.py is byte-identical to v4: its sha256 is 2fa77d12..., which equals the pin at validate_github.rb:179. validate_github.rb is untouched.

## Focus items
- **Discovery helper.** It is correct for both layouts. Native resolves every target to the aggregate bundle. Swiftbuild resolves per target. A per-target bundle together with the aggregate is an explicit error, which is stricter than a silent pick. Unit tests on synthetic trees cover native, swiftbuild, per-target, missing, duplicate, aggregate plus per-target, and Debug/debug/Release case.
- **"(aka ...)" acceptance.** It is exact. The aka text is `re.escape`d and derived from the row id's method, and `test_wrong_aka_missing_pass_and_skips_are_rejected` rejects a wrong aka. It is not a loosened pattern.
- **xcrun swift.** The change is sound. The drivers pass `env` with DEVELOPER_DIR, so xcrun and the `swift` shim select the same toolchain. The gate evidence shows the toolchain swift used for the test runs (body-038.json argv). The host check now matches the path dyld loads.
- **Swift test change.** Every original assertion is retained: the `Issue.record`, the `#expect(error is CancellationError)`, and the do/catch, now inside `await Task { ... }.value` (ExecutionSchedulingContractTests.swift ~L314-320). The test's own task is no longer cancelled. The same child-Task idiom already appears earlier in the test.
- **Gate credibility.** The evidence matches the report:

  | Gate | Exit | Time | Result |
  |---|---|---|---|
  | Perf debug | 0 | 76 s | 22 cases |
  | Perf release | 0 | 256 s | 22 cases |
  | ASan | 0 | 132 s | 77 |
  | TSan | 0 | 332 s | 39 |
  | Build modes (CLT 6.2, native) | 0 | 291 s | V02_BUILD_MODES_COMPLETE |

  The TSan Q03 parent took 12.57 s against its 49 s deadline. result.json lists the loaded bundles (Comprehensive only for the sanitizer gates, Comprehensive and Minimal for perf). Native perf debug is also 22.

## Findings

### Minor
1. check_sanitizers.py:271-280 has no `check_no_rebuild` equivalent. The brief's "verify from the log that no rebuild occurs" is implemented only in the perf gate (check_performance_gates.py:49, :130, :141, :163). The sanitizer flags are identical between build and test, but the no-rebuild property is not asserted.
2. check_sanitizers.py:208 and test_sanitizer_log.py:432-438: the "started" and "passed" lines are matched independently, so one may carry "(aka ...)" and the other not. The aka is also optional for the custom-name case. This is acceptable, but a mixed log is accepted. The tests also lack an explicit `was cancelled` rejection case. The behavior holds only because "passed" is required, so adding that test would pin the constraint.
3. swiftpm_products.py:354 uses bare `swift` for show-bin-path, while the test runs use `xcrun --find swift`. Both resolve to the same toolchain here. Using one resolved binary would remove the reliance on that.
4. Comments asserting that /usr/bin/swift strips DYLD_* (check_performance_gates.py:104, check_sanitizers.py:260) and that the xcrun xctest is a launcher shim (check_performance_gates.py:88) name no test. They are supported only by report evidence (0 vs 6045 dyld lines). Mark them as observed on the toolchain used.
5. check_benchmark_build_modes.py is not build-system-agnostic. This is by root ruling (a) and is documented. Gate 5 is valid only under native (CLT 6.2 here, and Xcode 27 `--build-system native` with the earlier, since-reverted driver). Hosted CI pins native Xcode, so it is not a defect, but the v5 headline "build-system-agnostic" does not cover this driver.

### Critical / Important
None.

## Cannot verify from diff
- The Swift test fix on Swift 6.0 and 6.1 (the report says these were not run).
- That the final byte-identical build-modes driver passes with Xcode native. It is evidenced only by the CLT 6.2 run.
- The `swift test --skip-build` no-rebuild behavior of the sanitizer gate. It is not checked by the driver.
- The contents of the dyld logs beyond the passing exit status. I did not re-parse all 77 and 39 per-body logs.
