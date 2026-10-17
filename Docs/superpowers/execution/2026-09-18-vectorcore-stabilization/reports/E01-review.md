# E01 independent spec and quality review

Verdict: spec and quality review PASS; no actionable source defects found. Final Debug and Release evidence passes the E01 verification scope. This reviewer inspected supplied runtime evidence and did not run builds or tests.

## Provenance and scope

Reviewed the complete four-source-file plus new-test diff against brief E01, the implementation report, and bounded surrounding source. All five archive SHA256 values match E01-review-manifest.json. Independently regenerated the unified diff from base `6f364bac978e099feeb2e02693235bbbcdcc88b3` (root Git object database) and frozen archive files; it exactly matches E01-review-package.diff. The archive itself has no usable Git object database; the root supplies the baseline.

## Findings

None requiring changes within E01 scope.

## Spec assessment

- Batch validation uses the actually consumed immutable default and precedes both routes. Both partition endpoints add only a bounded remaining count. Original transform errors escape unchanged; ordering remains batch-index sort followed by flatten. The nil control deliberately mutates the live actor to zero while retaining immutable routing, without activating E06 settings.
- All five CPU helpers and five protocol defaults validate subtraction overflow before range counting, result allocation, or child task creation. Explicit hints validate before empty, sequential, parallel, and nested routing, including forwarding protocol defaults. Empty reductions return before either callback. Empty iteration paths create no work callbacks.
- For a validated range, every remaining difference is at most its representable total. Adding its bounded chunk stays between start and upperBound. Quotient/remainder ceiling avoids the overflowing numerator; task counts are positive even on empty iteration paths. Default hardware doubling checks the bound before multiplication and retains nonpositive hardware normalization.
- Both synchronous loops preserve rethrows, validate domains before work, support arbitrary positive widths, and bound additions by remaining count. Advancing an index to count is representable even at Int.max. No performance claim is introduced.
- Existing CPU sequential reduction initial handling, completion-order combination, per-item default scheduling, initializer settings, and thresholds remain unchanged as required by the E02/E06 exclusions.

## Test quality and evidence

The matrix enumerates all five helpers, four CPU route configurations, four default hardware hints, near-endpoint/empty/nonrepresentable ranges, and valid/invalid explicit hints. Independent array oracles enforce ordered execute output and exact-once coverage; foreach and identity reductions compare unordered coverage. Reduction callback markers separately enforce empty no-work/no-combine behavior. Giant representable reductions inspect chunk counts without enumerating their contents, covering ceiling arithmetic without enormous default scheduling. Loop width/count grids and sentinel positions cover arbitrary widths and synchronous stopping. Batch tests cover both routing sides, changed output cardinality, errors, invalid parameters, and nil routing.

Child tests use fresh processes, a five-second deadline, explicit timeout failure, and no-body markers for domain failures. Debug additionally requires the intended precondition diagnostic, so incidental baseline range traps do not qualify as domain rejection. Release trap checks cannot independently identify the precise precondition message; static guards and the Debug diagnostic checks supplement that limitation. The test runner is macOS/XCTest-specific (Darwin and runner/bundle arguments), consistent with the reviewed execution environment.

Inspected corrected behavioral RED summaries: six tests/490 failures, domain diagnostic failures, and the dedicated large-length baseline child trap. Earlier sandbox/runner setup errors are correctly excluded as behavioral evidence. E01-debug-final.log records seven XCTest cases with zero failures and sixteen Swift Testing controls passing. One XCTest case is the no-op parent probe; substantive child assertions execute through the other cases. The narrow final evidence refresh also inspected E01-release-final.log: seven XCTest entries with zero failures and sixteen Swift Testing controls passing. E01.md records the exact required Debug/Release commands and exit0 for each; the root independently confirms both exit statuses. All entries in E01-evidence-sha256.txt matched before this review refresh. E01-scope-verification.json reports precisely four owned production changes and the new boundary test, with frozen hashes still matching. Root remains responsible for full-source association and integration verification. The evidence checksum entry for this review must be refreshed after this edit. No broad reruns, source modifications, index changes, or commits were performed.
