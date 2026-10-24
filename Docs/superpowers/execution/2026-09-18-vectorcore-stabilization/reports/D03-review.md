# D03 independent review: lazy log messages, monotonic PerformanceTimer

Reviewer: independent reviewer (read-only). Date: 2026-10-03.
Inputs: brief D03.md, dispatch-addendum.md, report.md and evidence logs, review-package.diff (BASE 596774c, 3 files).
Workspace: `.stabilization-recovery/2026-10-03/workspaces/D03`. The files match `manifest.sha256` (Logger a0566627…cdad6, LoggingContractTests 41f66dd7…83f3, LoggingPublicConsumerTests a8b34f4f…25ca). The workspace diff is identical to the review package.

Spec: PASS
Quality: CHANGES REQUESTED

The production change meets every contract item in the brief. I confirmed one item independently: the deferred VectorError fallback description (see the check below). Quality is held back by one Important test and claim gap on the VectorError fallback path. That path can be observed through a public-API spy, contrary to the report's statement. The gap has a small, mechanical fix.

---

## Findings

### I1 (Important): VectorError fallback `error.description` laziness is untested, and the "not observable" claim is false
- Locations:
  - `Tests/ComprehensiveTests/LoggingContractTests.swift:114-115`: "Contract only (not independently observable here)".
  - `Sources/VectorCore/Logging/Logger.swift:124-125` together with `:129-130`. The type doc groups the fallback `error.description` under "Exercised by `LoggingContractTests` (25-case matrix, reentrancy) and `LoggingPublicConsumerTests`", but none of those tests exercises the fallback.
  - report.md, Contract section and Concern 2: "not independently observable by a spy".
- Observability: the fallback can be observed. `VectorError.description` interpolates the underlying error (`Sources/VectorCore/Errors/VectorError.swift:504`, `result += "\nUnderlying: \(underlying)"`). An `underlying:` error that conforms to `CustomStringConvertible` with a counting `description` is therefore a spy that uses only the public API: `VectorError.init(_:message:underlying:…)` at `VectorError.swift:158-161`.
- Proof that the spy discriminates (see "Independent check"). It ran with the public API only:
  - D03 `Logger.swift`: `gated=0 admittedDelta=1 callerMessageDelta=0`.
  - Baseline 596774c `Logger.swift`: `gated=2 admittedDelta=1 callerMessageDelta=0`.
- Existing coverage misses an eager regression. Under mutation (a), which evaluates the message before the gate, `vectorErrorOverloadForwardsAndRespectsGate` still passed (`mutation-a.log`). No current test would catch a regression such as `let d = error.description; log(level: .error, { message ?? d }, …)`.
- Effect: the brief names this path explicitly in the laziness contract ("The VectorError overload defers its fallback error.description until admitted"). The production code is correct, at `Logger.swift:212`. But the evidence package records a false reason for leaving it untested, and the type doc cites tests that do not cover it.
- Requested fix:
  - Add one test with a `SpyError: Error, CustomStringConvertible` passed as `underlying:`, using the sink seam. It asserts:
    - `logger.error(err)` and `err.log(to: logger)` at minimum `.critical` give spy count 0;
    - at `.error` or lower, exactly 1 evaluation per call;
    - with a caller-supplied `message:`, 0 (the caller string wins and `??` does not evaluate the right-hand side).
  - Replace the comment at `LoggingContractTests.swift:114-115`.
  - Make the type doc at `Logger.swift:123-130` name that test, or else derive the fallback from `Logger.swift:212`.
  - Correct report Concern 2.
  - Optional: record the baseline RED for this path. My check shows gated=2 at baseline.

### M1 (Minor): stale doc example for a nonexistent public log method, missing from the inventory
- Location: `Sources/VectorCore/Errors/VectorError.swift:92`: `///     logger.log(error, severity: error.kind.severity)`.
- `Logger` has no public `log(_:severity:)`. The private helper is `log(level:_:file:function:line:)`. The brief says to "Fix existing examples referring to nonexistent … public log methods to actual APIs".
- D03 production ownership is `Logger.swift` only, so the implementer correctly did not edit this file. The caller inventory in report.md should still have listed it as an out-of-ownership residual.
- Action: route to root or the owner of VectorError.swift. The real equivalents are `error.log(to: coreLogger)` and `coreLogger.error(error)`. The code-caller inventory is otherwise complete (see section (6)).

### M2 (Minor): leftover blanket "thread-safe" claim and nonexistent "output options" in a doc block D03 edited
- Location: `Sources/VectorCore/Logging/Logger.swift:69-70`: "`LogConfiguration` provides thread-safe configuration for controlling logging behavior, including minimum log levels and output options."
- The text predates D03. D03 edited the same doc block's example (`:73-75`) and correctly narrowed the property comment (`:84-86`). The type-level sentence still contains:
  - a blanket thread-safety claim, which the program constraints prohibit;
  - "output options", a nonexistent setting. This is the same family as the removed `enableOSLog`.
- Suggested wording: "Holds the minimum log level. Reads and writes of `minimumLevel` are individually serialized by an internal lock."

### M3 (Minor): the timer's "no formatting when rejected" contract is neither documented nor observed
- Locations:
  - `Sources/VectorCore/Logging/Logger.swift:321-327`, the `log(threshold:)` doc;
  - the property itself rests on `:330` (the threshold guard precedes formatting) and `:336` (formatting lives inside the `logger.debug` autoclosure).
- Current coverage: `timerInvalidThresholdSuppressesButStillReadsClock` and `timerSuppressedWhenDebugLevelIsDisabled` assert only that nothing is emitted. Mutation (a) left `timerSuppressedWhenDebugLevelIsDisabled` green.
- Judgement: acceptable as a one-step adjacent-code derivation composed with the matrix-proven laziness of `debug`. A spy cannot prove it without a new seam, because the interpolated values are a `String` (`operation`) and a `Double`, and neither runs user code.
- Residual risk: a regression that pre-formats inside `PerformanceTimer.log` (for example `let m = "…"; logger.debug(m)`) would not be caught.
- Requested: one sentence in the doc that states the contract and its derivation, naming `messageEvaluationMatrix`. Low priority.

Counts: Critical 0, Important 1, Minor 3.

---

## Focus-area assessment

### (1) Lock discipline: correct
- The lock is held only inside the accessor bodies, which never call foreign code:
  - `Logger.swift:88-92`: `get { lock.lock(); defer { lock.unlock() }; return _minimumLevel }`
  - `Logger.swift:93-97`: the setter, same pattern.
- The private helper:
  - `Logger.swift:226`: `let minimum = activeConfiguration.minimumLevel` is the only read per call. The lock is released when the getter returns.
  - `Logger.swift:227`: `guard level >= minimum else { return }`.
  - `Logger.swift:229`: `let text = message()` runs after release, exactly once.
  - `Logger.swift:230-237, 240-259`: location formatting, the sink, `os_log` and `print` all run with no lock held.
- Each wrapper and overload makes exactly one `log` call (`:168, :176, :184, :192, :200, :212`; `VectorError.log` at `:349` forwards once). So each logging call reads the level exactly once, and nested calls take their own snapshots.
- No deadlock from reentrant logging or a configuration update inside a message: the only lock, the NSLock, is never held across user code.
- Evidence for the reentrancy behavior:
  - `childReentrancy` (`LoggingContractTests.swift:150-178`) reads and updates the level from inside a message and logs at depth 2. Counts are `[1,1,0,0]` and records are `["B","A"]`, so the admitted outer record is not retracted. The final level is `.error`.
  - The child ran under a bound and passed in Debug and Release. I re-read the final Release child capture: `D03_REENTRANCY_COMPLETE counts=[1, 1, 0, 0]`.

### (2) Laziness on every path

| Path | Status | Evidence |
|---|---|---|
| `debug`/`info`/`warning`/`error`/`critical` wrappers (`:164-201`) forward the closure, not `message()` | Test-proven | `messageEvaluationMatrix` (25 cases, sink path); `actualBackendEvaluatesAdmittedMessageOnce` (real backend, 25 combinations); public child (25 cases, real global configuration). The `emit` helper re-wraps its autoclosure, so the spy runs only if the wrapper evaluates it. RED shows the baseline failing on exactly the 10 gated cases. |
| Private helper `log` (`:221-261`) | Test-proven | Same tests. Mutation (a) is caught: the matrix fails on exactly the 10 gated cases, and the backend test fails with 10 issues. |
| VectorError overload fallback (`:212`) | Adjacent code only. **A spy can prove it** (I1). | `{ message ?? error.description }` is evaluated only after the gate. Tested only for forwarding and for "gated emits nothing". |
| `VectorError.log` (`:349`) | Inherits from the overload (same gap as I1) | Forwarding is tested at `LoggingContractTests.swift:111-113`. |
| Caller-supplied `String?` message | No promise (brief) | Correctly documented at `:205-206`. |
| Timer, threshold rejection (`:330` before `:336`) | Adjacent code plus absence-of-emission tests | Acceptable; a spy is impossible without a new seam (M3). |
| Timer, level-gate rejection (`:336` autoclosure) | Composition: one-step derivation from `:336` plus the matrix-proven `debug` laziness | Acceptable (M3). |

### (3) Timer: correct
- Conversion (`:331-332`): `Double(components.seconds) + Double(components.attoseconds) / 1e18`. This uses no integer arithmetic. 1e15/1e18 rounds correctly to the literal 0.001, so the equality case at the threshold works; `timerBoundaryReadings` covers 0, 1 ms − 1 ns, 1 ms and 1 ms + 1 ns. Components of a negative duration share a sign or compensate, so the sum is correct either way.
- Cumulative elapsed time: the start is captured once (`:316`) and every log call measures from it (`:331`). `timerReportsCumulativeNotLapElapsed` shows 2.000, 5.000 and 3500.000 ms.
- Thresholds:
  - Negative, NaN and ±Inf are suppressed by `guard threshold.isFinite, threshold >= 0` (`:330`). Covered by 5 parameterized cases.
  - `-0.0` passes the guard as zero, which is IEEE-equal to zero. That is reasonable; it is not a finding.
- Huge finite thresholds are compared as `Double`, never converted (`:335`). Covered by `greatestFiniteMagnitude` and `1e300`.
- Negative injected elapsed time fails `elapsed >= threshold` because `threshold >= 0`. Covered by `timerNegativeInjectedElapsedSuppresses`.
- Clock reads: one at the start (`:316`) and one per call before validation (`:329`). Tests assert counts of 5, 4, 3, 2 and 2, including invalid-threshold calls.
- The milliseconds format `"\(operation) took \(String(format: "%.3f", elapsed * 1000))ms"` is byte-identical to the baseline. The default threshold of 0.001 is unchanged.
- Note, unverified and not executed: the stdlib's `Duration.components` narrows the 128-bit count to `Int64` seconds. That can only trap for durations above about 9.2e18 s, which are reachable only through the internal test seam with absurd instants. `start.duration(to: reading) / .seconds(1)` would avoid the narrowing. Not a finding.

### (4) Test seams: compliant
- `Logger.init(subsystem:category:configuration:sink:)` (`:152-161`) and `PerformanceTimer.init(operation:logger:clock:)` (`:312-319`) are internal and per-instance. `LogRecord` (`:113`) is internal. There is no global or task-local mutable state.
- The test files contain only immutable private globals: `levels` and `d03Levels`.
- Defaults:
  - The public `Logger.init` uses `Self.configuration` with `sink: nil`, which is the real backend (`:142-145`).
  - The public `PerformanceTimer.init` uses `ContinuousClock.now` (`:306-308`).
- Public signatures for Logger, LogLevel, LogConfiguration, PerformanceTimer, VectorError.log and the global loggers are unchanged.

### (5) Tests and evidence
- The 25-case matrix, the backend exactly-once test and the public child together prove the brief's evaluation contract for the five wrappers.
- Dual-output reuse: the real-backend Debug test proves that a single evaluation feeds both `os_log` and `print`. A mutation that evaluates once per output would produce a count of 2. Equal text in both outputs follows trivially from the shared `text` (`:255, :259`). Acceptable.
- Public consumer: `LoggingPublicConsumerTests.swift` uses a plain `import VectorCore`, the real initializer and the global configuration. It changes only `minimumLevel` and restores it (`:91, :94, :111`) in a fresh process. The final Release capture contains 25 cases, `mismatches=0`.
- The child helper follows the E02 style (`:29-69`):
  - capture goes under `FileManager.default.temporaryDirectory`;
  - `terminationHandler` is set at `:47`, before `run()` at `:48`;
  - the deadline is 15 s, then terminate, then 1 s, then SIGKILL, then 1 s (≤ 2 s);
  - it throws at `:59` before `terminationStatus`/`terminationReason` (`:64-65`). Only `isRunning` and `processIdentifier` are read after the deadline.
- Mutations:
  - (a) is convincing: the 10 gated cases fail in the matrix, the backend test, the public child and the reentrancy child.
  - (b) is convincing: the non-recursive NSLock held around `message()` self-deadlocks. The child was killed at the deadline (`D03_CHILD_TIMEOUT reentrancy cleanupReaped=true`), and the test failed after 15.009 s on the thrown timeout while the other tests passed.
  - Both mutations ran on the final test file. The line numbers in the logs match the current file, and the mutation diffs (21:30:48) postdate the last test edit (21:30:47).
- RED is genuinely behavioral. `red-debug-v3` compiled the baseline and the child ran it, giving 10 mismatches, all on disabled debug/info/warning/error. Critical was evaluated under every minimum. The line numbers match the final test file.
- Release coverage gap (FP16DiagnosticsBoundaryTests run in Debug only) is not material:
  - The only production caller, `MixedPrecisionDiagnostics.swift:97, 105, 129, 203, 461`, logs side-effect-free interpolations, and always after releasing its own lock. Making those messages lazy cannot change its state.
  - The Release default (`.warning`) gates exactly those info/debug calls, so the only Release behavior change is skipping pure string construction.
  - `Logger.swift` has no Release-specific branch besides the `#if DEBUG` print.

### (6) Callers and docs
- Code callers: my independent grep over Sources, Tests, Benchmarks, VectorIndex and Package.swift finds only `MixedPrecisionDiagnostics.swift:18, 364` (construction) and `:97, 105, 129, 203, 461` (calls). There are no callers of PerformanceTimer, `VectorError.log` or the global loggers, and no older logging tests. Migration is limited to these contracts.
- Doc examples D03 fixed are real APIs: `coreLogger.info(_:)`, `coreLogger.critical(_:)` and `Logger.configuration.minimumLevel`. One stale example remains outside ownership (M1).

---

## Independent check (one standalone run, public API only, workspace untouched)

- Scratch package: `/private/tmp/claude-501/-Users-goftin-dev-gsuite-VSK-VectorCore/b9be2b44-6a20-4cd9-8b29-ce3af0094ae9/scratchpad/d03check`. It contains an rsync copy of the workspace `Sources/VectorCore` and `Sources/VectorCoreC`, plus an executable that uses only `import VectorCore`.
- Spy: `struct SpyError: Error, CustomStringConvertible { var description: String { counter += 1 … } }`, passed to `VectorError(.operationFailed, message: "m", underlying: SpyError(...))`.
- Sequence:
  1. With `Logger.configuration.minimumLevel = .critical`, call `coreLogger.error(err)` and `err.log()`; record `gated`.
  2. Set `.debug`, call `coreLogger.error(err)`; record `admittedDelta`.
  3. Call `coreLogger.error(err, message: "caller")`; record `callerMessageDelta`.
- Results:
  - D03 Logger.swift (identical to the workspace): `D03CHECK gated=0 admittedDelta=1 callerMessageDelta=0`.
  - Baseline `git show 596774c:Sources/VectorCore/Logging/Logger.swift`: `D03CHECK gated=2 admittedDelta=1 callerMessageDelta=0`.

---

## Cannot verify from diff
1. **Index state.** `git status` in the workspace shows all three files staged (`M `/`A `). The worktree index mtime (21:36:31) equals the mtime of `review-package.diff` and is later than `report.md` (21:36:05). That suggests root staged them while packaging, not the implementer, but the artifacts cannot attribute it. The report says there was no staging.
2. **Lint ordering.** `swiftlint-v2.log` (0 violations) and the final `LoggingContractTests.swift` edit share the same second (21:30:47). Whether lint ran after the final edit cannot be ordered from second-granularity mtimes. The final file hash is the manifest hash.
3. **Older toolchains.** All runs used Swift 6.4 / Xcode 27. CI's matrix is Xcode 16.2 and 26.3, Debug and Release (`.github/workflows/ci.yml:42-43`). Compilation there is unverified, as is the swift-testing handling of the NaN parameterized argument in `timerInvalidThresholdSuppressesButStillReadsClock`.
4. **Inherited child arguments.** CI runs `swift test --xunit-output …` (`ci.yml:67`). `d03RunChild` copies every parent argument except the filter (`LoggingPublicConsumerTests.swift:32-34`), so a child might inherit and rewrite the parent's xunit path. This is the same pre-existing pattern as the accepted child helper `FP16DiagnosticsBoundaryTests.swift:16-19`, so it is not D03-specific, and it is not exercised by the evidence (all runs were filtered and had no xunit output).
5. **OSLog persistence and filtering** of admitted records is not asserted. The brief explicitly does not promise it.
