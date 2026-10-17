# D03 fix round 1 re-review (scoped)

Verdicts: I1 ADDRESSED | M1 ADDRESSED | M2 ADDRESSED | M3 ADDRESSED | Debug/Release same filter ADDRESSED | Baseline RED ADEQUATE
New breakage: none

## I1
- Test: Tests/ComprehensiveTests/LoggingContractTests.swift:122-125 (SpyError, passed as `underlying:` at :131), :129-153 (vectorErrorFallbackDescriptionIsLazy).
  - Gated at .critical: logger.error(err) and err.log(to:) -> spy.count == base (:136-138), records empty (:139).
  - Admitted at .error/.warning/.info/.debug: exactly +1 for each of the two calls (:141-147).
  - Caller message: +0 for both (:150-152), records are the caller text (:153).
  - Observability is real: VectorError.description appends "Underlying: \(underlying)" (VectorError.swift:501-503), and Logger.error(_:message:) wraps `message ?? error.description` in the lazy closure (Logger.swift:213-214).
- False comment replaced: LoggingContractTests.swift:114 now only says gated calls emit nothing and points to the new test.
- Logger type doc names the test: Logger.swift doc block ("vectorErrorFallbackDescriptionIsLazy for the VectorError fallback").
- Report corrected: report.md "Fix round 1 (edits)" says the earlier "not observable" claim was FALSE.
- Baseline RED captured: red-fallback-fix1.log:271 (failure at LoggingContractTests.swift:138, spy.count == base), :377 (test failed), :448 (14 tests, 22 issues).
- Mutation-RED adequacy: ADEQUATE. baseline-eager-seam.diff changes only the evaluation behavior: the wrappers pass `message()`, log takes a String, and the VectorError overload does `let eager = message ?? error.description`. The seam (sink/configuration init) is kept so the test compiles. This is exactly the eager behavior the test must reject, and it fails on the gated assertion. Pristine 596774c cannot compile the seam-based tests, and my prior review independently showed gated=2 on the pristine base through the public API (review.md:24, :147). Together these cover both the seam and the public API. Disclosure in report.md is honest (it says "not byte-identical 596774c").
- Weakness (not blocking): the admitted +1 and caller +0 assertions also pass under the eager mutation. The RED is carried by the gated assertion only. That is the discriminating one, so this is fine.

## M1
VectorError.swift:92 is the only line changed in the file: `logger.log(error, severity: error.kind.severity)` -> `error.log(to: coreLogger)`. Confirmed by fix1-package.diff (1 insertion, 1 deletion, one hunk) and `git diff --stat HEAD` on the workspace (1 insertion, 1 deletion). The API exists: VectorError.log(to:) at Logger.swift:350, and `coreLogger` at Logger.swift:270.

## M2
Logger.swift LogConfiguration doc: "Holds the minimum log level. Reads and writes of `minimumLevel` are individually serialized by an internal lock." The blanket "thread-safe" and "output options" are gone. The new claim follows directly from the adjacent code (private `_minimumLevel` guarded by `private let lock = NSLock()`). Accurate.

## M3
Logger.swift PerformanceTimer.log(threshold:) doc: "A rejected call (invalid threshold, or debug gated) does not format the message: the threshold guard precedes formatting, the formatting lives inside the logger.debug autoclosure, and debug laziness is proven by LoggingContractTests.messageEvaluationMatrix." The derivation is adjacent: the guard at :337, the formatting inside the autoclosure at :342, and the test exists at LoggingContractTests.swift:71. The report states it is a derivation, not spy-proven. Claim-discipline compliant.

## Same filter
run-fix1.sh uses 'Logging(Contract|PublicConsumer)Tests|FP16DiagnosticsBoundaryTests' for both Debug and Release. fix1-debug.log and fix1-release.log each end with "Test run with 23 tests in 3 suites passed", and each lists the new test twice (started and passed). The red run used the narrower 'LoggingContractTests', which is acceptable for a RED.

## New-breakage check
Fix diff touches 3 files (doc lines, one new test, one doc line in VectorError.swift). No behavioral source change. I found no new breakage.

## Out-of-scope / minor notes
1. The M3 parenthetical "(invalid threshold, or debug gated)" omits the third rejection, elapsed < threshold. It is also before formatting (the guard at :341), so the sentence is true but not exhaustive. Optional wording tweak.
2. The messageEvaluationMatrix proof covers Logger.debug laziness, not PerformanceTimer end-to-end. The doc states this honestly as a derivation.
3. The Debug-only `#if DEBUG` suffix in VectorError.description means the spy's single evaluation is unaffected, but the description text differs by config. This is irrelevant to the counts.
4. I ran no Swift builds. Evidence is from the logs and diffs only.
