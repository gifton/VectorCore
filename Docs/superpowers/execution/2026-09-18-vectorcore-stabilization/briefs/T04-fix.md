# T04 fix brief — the RSS-delta underflow trap and its class (root-approved, binding)

## Root cause (established by the T04 investigation; see the report in this directory)

`Tests/ComprehensiveTests/QuantizedKernelsTests.swift:1217` is in `MemoryCompressionTests.testMemoryFootprintReduction()`. It computes `info.resident_size - baselineMemory` as **UInt64** on whole-process RSS. Swift Testing runs tests in parallel. When other tests free more memory than this test allocates within the window, RSS decreases, the unsigned subtraction underflows, and Swift's overflow check executes `brk #1`. The result is a message-less SIGTRAP that kills the whole ComprehensiveTests Swift Testing process.
- It was reproduced twice with backtraces (`T04/bt/full-A-04`, `full-A2-02`), plus a deterministic standalone reproduction (`T04/experiments/rssprobe`).
- The rate was 3 traps in 12 Debug whole-suite runs.
- Classification: (b), a test defect.

## Required changes (test-only, plus one new policy test)

1. **`QuantizedKernelsTests.swift` ≈ lines 1197–1223:** delete the whole-process RSS measurement block. That is both `task_info` calls, the `largeFP32` allocation, the `:1217` subtraction and its print. Keep the layout prints. The block asserted nothing, and its value is dominated by concurrent tests. Keep every other assertion in that test unchanged.
2. **`MixedPrecisionKernelsTests.swift` ≈ lines 2368–2399:** delete the dead `measureMemoryUsage(operation:)` helper (same UInt64 subtraction at :2398; no callers). Prove it has no callers with a grep in your report.
3. **`BatchKernels_SoATests.swift` `getMemoryUsage` and its caller(s), ≈ :791–800:** the test asserts `memoryGrowth < expected*3` on whole-process RSS. Under parallel load that is a false-failure risk. Determine whether it is EXTENDED-gated; it would run in milestone 12's `EXTENDED=1` pass. Remove the RSS-growth **assertion**. Keep a signed informational print labelled "whole-process RSS, unreliable under parallel tests; isolated-process measurements are V04's memory probe". If the measurement must stay, compute it signed (`Int64(clamping:)`).
4. **E02 watchdog child (`Tests/ComprehensiveTests/ExecutionSchedulingContractTests.swift` ≈ :107–109).** The `watchdog` child scenario deliberately ignores SIGTERM and loops forever, so the parent must escalate to SIGKILL. If the parent dies first, the child is orphaned indefinitely; two orphans from crashed runs were found and killed. Make the child bound its own lifetime without changing what the parent observes:
   - keep ignoring SIGTERM;
   - in its loop, exit when `getppid() == 1` (orphaned) or after a hard ceiling well above the parent's deadline, for example 120 s.

   Re-run that test, and show that the parent still observes the SIGTERM-ignore and SIGKILL-escalation behaviour.
5. **Mechanical closure: new `Scripts/ci/tests/test_process_memory_deltas.py`** (stdlib, in the style of `test_child_process_bounds.py`; any subprocess call bounded). Scan `Tests/`, `Sources/` and `Benchmarks/` `*.swift` for:
   - a subtraction with an operand naming `resident_size`, `resident_size_max`, `virtual_size` or `phys_footprint`;
   - or a `let` bound directly from one of those that is later subtracted,

   unless both operands are converted to a signed type. Include negative controls on synthetic sources reproducing the `:1217` and `:2398` shapes, and a positive control accepting the `Int64(clamping:)` form. The docstring states that it is a heuristic with that scope. Show it is RED on the original `QuantizedKernelsTests.swift` and `MixedPrecisionKernelsTests.swift`, using `git show HEAD:<path>` into a scratch tree.

## Coordination
- T03 is editing `QuantizedKernelsTests.swift` concurrently, at a different hunk (≈ :309, a dead pre-Q01 branch). Edit only your hunks. Root will integrate with a proven 3-way merge.
- V04 will later edit `MixedPrecisionKernelsTests.swift` for handle-stride labels and will rebase onto you.
- Do not touch V04-owned files: `Benchmarks/**`, `Sources/VectorCoreBenchmarking/**`, `Tests/BenchmarkContractTests/**`.

## Verification (evidence root `/Users/goftin/dev/gsuite/VSK/VectorCore/.stabilization-recovery/2026-10-03/T04/fix/`)
Run every swift command through the capture driver with unique scratch paths and `--jobs 6`, with `R04_CAPTURE_DIRECTORY` set under the evidence root.
1. Run each changed test alone in Debug and Release, including the E02 watchdog test.
2. **Trap regression loop:** 6 full Debug ComprehensiveTests Swift Testing runs with `--skip-build`, after one build. Expected: 0 traps (the baseline was 3 in 12). Report exact totals.
3. Run the whole suite with `VECTORCORE_TEST_EXTENDED=0` in Debug and Release.
4. Policy discovery, `python3 -B -m unittest discover -s Scripts/ci/tests -v`: report the total.
5. Pinned SwiftLint on the changed Swift files.

## Rules
Test-only Swift changes. No production edits. No git index operations. No subagents. Return the full report text in your final message, leading with a status line (DONE, DONE_WITH_CONCERNS, BLOCKED, or NEEDS_CONTEXT).
