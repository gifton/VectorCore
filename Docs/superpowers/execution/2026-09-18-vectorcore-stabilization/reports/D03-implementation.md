# D03 report: lazy log messages, monotonic cumulative PerformanceTimer

Status: DONE_WITH_CONCERNS. Toolchain: macOS 27 / Xcode 27.0 (27A266a) / Swift 6.4 (swiftlang-6.4.0.34.1), default swiftbuild. Workspace detached at 596774c.
Evidence root: /Users/goftin/dev/gsuite/VSK/VectorCore/.stabilization-recovery/2026-10-03/D03/ (E below).

## Caller inventory (before freeze)
- Sources: only Sources/VectorCore/Operations/Kernels/MixedPrecisionDiagnostics.swift (lines 18, 364 construct Logger; 97,105,129,203,461 call info/debug with plain String interpolation). It imports OSLog, but VectorCore.Logger shadows os.Logger in-module; compiles unchanged.
- Global loggers coreLogger/storageLogger/batchLogger/metricsLogger/performanceLogger: no callers outside Logger.swift. PerformanceTimer and VectorError.log: no callers outside Logger.swift.
- Benchmarks/, Examples: none. Old logging tests: none (grep of Tests/ for Logger/LogLevel/LogConfiguration/PerformanceTimer found nothing).
- Non-code references: Scripts/ci/sdk_name_collision_allowlist.json (Logger vs os.Logger, owner D02; untouched), audit docs.
- Migration therefore limited to these contracts. FP16DiagnosticsBoundaryTests (the caller's suite) was also run (see GREEN).

## Changed files (no git commit/stage; manifest.sha256, changed.diff in E)
- Sources/VectorCore/Logging/Logger.swift (modified; only production file)
- Tests/ComprehensiveTests/LoggingContractTests.swift (new)
- Tests/ComprehensiveTests/LoggingPublicConsumerTests.swift (new; plain `import VectorCore`, holds bounded child driver d03RunChild)
sha256: Logger a0566627...cdad6, LoggingContractTests 41f66dd7...83f3, LoggingPublicConsumerTests a8b34f4f...25ca (full in manifest.sha256).

## Contract mapped to brief
- Snapshot once, release lock, evaluate once: `Logger.log` reads `activeConfiguration.minimumLevel` (getter takes/releases the existing NSLock), `guard level >= minimum`, then `message()` once, then location/format/backend. Test: messageEvaluationMatrix (5x5, 25 cases), actualBackendEvaluatesAdmittedMessageOnce (real OSLog/Debug print backend, exactly once admitted / zero gated), public child 25 cases.
- All five wrappers forward the autoclosure itself (non-escaping closure param) rather than `message()`. Matrix covers all five.
- VectorError overload: `{ message ?? error.description }` closure, evaluated only when admitted. Tests: vectorErrorOverloadForwardsAndRespectsGate (forwarding, caller message, VectorError.log forwards logger/message/file/function/line, gated emits nothing). The laziness of the fallback description itself is a source-derived contract (adjacent code), not independently observable by a spy; stated as such in the test comment.
- Source-location forwarding: explicit file/function/line in matrix; default #fileID/#line in defaultSourceLocationIsForwarded.
- Reentrancy: reentrantMessageDoesNotDeadlock runs childReentrancy in a bounded child: an admitted message reads and updates configuration and logs nested; asserts counts [1,1,0,0], records ["B","A"] (outer admitted record not retracted by later level change), final level .error. Lock policy retained (existing NSLock).
- Seam: internal `Logger.init(subsystem:category:configuration:sink:)` and internal `LogRecord`; public init uses Logger.configuration and sink nil (actual backend). No global mutable state. Public signatures unchanged.
- Docs fixed: removed nonexistent LogConfiguration.shared / enableOSLog / logger.log examples; replaced with Logger.configuration.minimumLevel and coreLogger.info/critical.
- PerformanceTimer: start is one ContinuousClock.Instant; internal init(operation:logger:clock:) with `@escaping @Sendable () -> ContinuousClock.Instant` (public init supplies ContinuousClock.now). `log(threshold:)` reads the clock once per call (before threshold validation), suppresses negative/NaN/infinite threshold, computes `Double(seconds) + Double(attoseconds)/1e18` (no integer overflow; division chosen so 1ms equals the literal 0.001 exactly), emits iff elapsed >= threshold via `logger.debug` (autoclosure, so no formatting if the level gate rejects). Threshold is never converted to Duration/Int. Tests: timerBoundaryReadings (0, 1ms-1ns, equality, above; 5 reads), timerZeroThresholdEmitsImmediateZero ("0.000ms"), timerReportsCumulativeNotLapElapsed (2.000/5.000/3500.000ms), timerInvalidThresholdSuppressesButStillReadsClock (-1, -tiny, NaN, +inf, -inf; reads == 3 for 2 calls), timerHugeFiniteThresholdDoesNotEmitOrTrap, timerNegativeInjectedElapsedSuppresses, timerSuppressedWhenDebugLevelIsDisabled. No sleeps, no civil-clock alteration. Civil-clock dependency of the baseline is established by source only (CFAbsoluteTimeGetCurrent in the baseline file), not by test.

## RED (unchanged production, behavioral)
Only LoggingPublicConsumerTests.swift existed (compiles against baseline; its public API only).
- Earlier compile failures (not RED, kept): red-debug.log (VectorCore.Logger qualification invalid because VectorCore is a type in the module; level.name internal), red-debug-v2.log (my sed failed, same errors).
- RED: argv `python3 capture_swift.py --workdir <ws> --log E/red-debug-v3.log --record E/red-debug-v3.json -- test --jobs 6 --scratch-path E/build-debug --filter 'LoggingPublicConsumerTests'`; exit 1; log sha 3e9e4de6...c7c859.
  Child (E/red-debug-v3-child.log, status=1): 10 mismatches, all disabled cases evaluated once with expected 0: debug@{info,warning,error,critical}, info@{warning,error,critical}, warning@{error,critical}, error@critical. Critical never mismatched (always admitted). 2 tests, 1 issue (parent), 1 issue (child).

## GREEN
- Debug v1 compile failure (ambiguous == in my #expect on arrays; fixed): green-debug-v1.log, exit 1 (not evidence).
- Debug: --filter 'Logging(Contract|PublicConsumer)Tests' green-debug-v2.log exit 0, 15 tests in 2 suites passed (matrix is 25 cases inside; public child printed 25 D03_PUBLIC_CASE lines, mismatches=0; reentrancy child counts=[1,1,0,0]; child logs green-debug-v2-child-*.log).
- Release: `test -c release ...` green-release-v1.log exit 0, 15 tests passed (children green-release-v1-child-*.log, 25 cases, mismatches=0).
- Final rerun after test-only lint refactor: final-debug.log exit 0 with filter 'Logging(Contract|PublicConsumer)Tests|FP16DiagnosticsBoundaryTests', 22 tests in 3 suites passed; final-release.log exit 0, 15 tests passed. Child statuses all 0.
- All children: argv = same test binary, file capture under FileManager.default.temporaryDirectory, termination handler before run(), 15 s + <=2 s cleanup, throws before reading termination properties on timeout.
- Old suites: none exist for logging. Full suite not run (not required; perf-ratio flake noted in memory).

## Mutation (scratch copies under E, since deleted; diffs in mutation-a.diff / mutation-b.diff)
- (a) message() evaluated before the level check: mutation-a.log exit 1. messageEvaluationMatrix failed (25 cases, 10 issues = the 10 gated cases), actualBackendEvaluatesAdmittedMessageOnce 10 issues, public child failed, reentrancy child also failed (2 issues). Caught.
- (b) lock held around message(): mutation-b.log exit 1. Reentrancy child hung and was killed at the deadline: `D03_CHILD_TIMEOUT reentrancy cleanupReaped=true`, reentrantMessageDoesNotDeadlock failed after 15.009 s with the thrown timeout (child log mutation-b-child-reentrancy-timeout.log). Other tests passed. Caught within bound. No re-run needed (the timeout was the intended outcome; no unrelated timeout occurred).

## Lint and scanners
- Bounds scanner (`python3 -B -m unittest discover -s Scripts/ci/tests -p 'test_child_process_bounds.py'`): 7 tests OK, before (scanner.log) and after final edits (scanner-final.log).
- Pinned SwiftLint 0.65.1 on the 3 files: first run 1 warning (function_parameter_count in my test helper, swiftlint.log), fixed by a Site struct; swiftlint-v2.log: 0 violations.

## Concerns
1. The Logger vs os.Logger name collision forced tests to use unqualified `Logger`; out of scope (D02) per root ruling.
2. Fallback `error.description` laziness is not directly observable; asserted only by source and by the gated-emits-nothing test.
3. The sink replaces the OSLog/print backend, so Debug dual-output reuse is evidenced by exactly-once evaluation with the real backend plus single `text` use in code, not by capturing both outputs. OSLog persistence not asserted (per brief).
4. Release `swift test` with @testable worked here; no claim beyond this toolchain. No thread-safety or performance claim is made.
5. PerformanceTimer "no formatting when gated" is derived from the autoclosure path in adjacent code; tests assert only the absence of emission.
6. Public-consumer child changes the global minimumLevel in its own process only and restores it; mutation scratch dirs were removed (diffs retained). Another worker's concurrent builds did not cause any timeout in my runs.

## Fix round 1 (edits)
Edits only; no swift build/test, no git commands. Verification pending (Swift runs, baseline RED for the new test against 596774c Logger.swift in a scratch copy).
- I1: added `LoggingContractTests.vectorErrorFallbackDescriptionIsLazy` (SpyError: Error, CustomStringConvertible passed as `underlying:`, via the sink seam). Asserts: at minimum .critical, `logger.error(err)` and `err.log(to:)` leave the spy count unchanged; at .error/.warning/.info/.debug exactly +1 per call; with caller `message:` +0 and records are the caller text. Replaced the comment in vectorErrorOverloadForwardsAndRespectsGate; Logger type doc now names the test.
- Corrected Concern 2: the earlier claim that the fallback `error.description` is not observable by a spy was FALSE. It is observable through a counting `description` on the underlying error (VectorError.description interpolates it). Before this round the fallback laziness had no test; mutation (a) would not have been caught for that path. Now tested; baseline RED to follow.
- M1: Sources/VectorCore/Errors/VectorError.swift:92 doc example changed to `error.log(to: coreLogger)` (one line; nothing else in that file). Added to caller inventory as a doc-only residual (previously stale `logger.log(error, severity:)`).
- M2: LogConfiguration type doc now "Holds the minimum log level. Reads and writes of `minimumLevel` are individually serialized by an internal lock." (blanket thread-safe and "output options" removed).
- M3: PerformanceTimer.log(threshold:) doc states that a rejected call does not format the message, with its derivation (threshold guard precedes formatting; formatting is inside the logger.debug autoclosure; debug laziness proven by messageEvaluationMatrix). This remains a derivation, not a spy-proven property.
- Lint: pinned SwiftLint on Logger.swift, LoggingContractTests.swift, VectorError.swift: 0 violations (D03/swiftlint-round1.log).
- Note: manifest.sha256 and changed.diff in E are stale until re-frozen after verification.

### Fix round 1 (verification)
Toolchain unchanged. No git commands run. Same filter in both configs: 'Logging(Contract|PublicConsumer)Tests|FP16DiagnosticsBoundaryTests', --jobs 6, scratch build-{debug,release}-fix1.
- Debug: fix1-debug.log/.json exit 0, 23 tests in 3 suites passed (vectorErrorFallbackDescriptionIsLazy passed); log sha 23c0fd3a...5dea.
- Release: fix1-release.log/.json exit 0, 23 tests in 3 suites passed (new test passed); log sha 6cf0f3d0...2dea... (see json). This closes the earlier Debug-only FP16 coverage asymmetry.
- Baseline RED for the new test: scratch copy of the workspace with Logger.swift changed to base eager behavior only (wrappers pass `message()`, log takes a String, VectorError overload does `let eager = message ?? error.description`), keeping the test seam so the test compiles (diff: baseline-eager-seam.diff). Whole-file seam was needed because the pristine 596774c Logger.swift has no sink/configuration init, so the new tests cannot compile against it; this is a seam-plus-eager-behavior baseline, not byte-identical 596774c. red-fallback-fix1.log/.json exit 1: vectorErrorFallbackDescriptionIsLazy failed ("gated calls must not compute error.description", spy.count == base at LoggingContractTests.swift:138); matrix (10 issues), backend test (10 issues) and reentrancy child also failed as expected. 14 tests, 22 issues. Scratch copy and its build dir were deleted after the run.
- Re-frozen: manifest.sha256 (Logger.swift, VectorError.swift, two test files) and changed.diff (vs. integration checkout /private/tmp/vectorcore-stabilization-2026-09-18 for the two modified sources, plus the new test files as additions). Concern 2 correction stands.
