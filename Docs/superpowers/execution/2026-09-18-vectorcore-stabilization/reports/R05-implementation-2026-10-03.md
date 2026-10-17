# R05 report (resume round, 2026-10-03)

Status: DONE_WITH_CONCERNS. Toolchain: macOS 27 / Xcode 27.0 / Swift 6.4 (swiftlang-6.4.0.34.1), swiftbuild default, arm64.
Workspace: .stabilization-recovery/2026-10-03/workspaces/R05 (detached e6e1bae). No git index operations, no subagents.

## Changed files (sha256 in ev/hashes.txt; full diff in ev/changes.diff)
- Sources/VectorCore/Protocols/DistanceMetric.swift: defaulted requirement relevanceScore(for:) -> Scalar? (default nil).
- Sources/VectorCore/Operations/DistanceMetrics.swift: private inverseDistanceRelevance helper + witnesses on Euclidean, Cosine, Manhattan, Chebyshev, Hamming, Minkowski; logistic piecewise witness on DotProduct.
- Sources/VectorCore/Integration/SearchResult.swift: removed `computedScore` property; added computedScore(using: any DistanceMetric<Float>) -> Float?.
- Sources/VectorCore/Integration/VectorCollection.swift: searchCapturedVectors (shared by default VectorCollection.search and SimpleVectorCollection.search) scores each selected result once via the same metric; SimpleVectorCollection captures snapshot under lock, metric calls happen after unlock.
- Tests/ComprehensiveTests/SearchResultScoreContractTests.swift (new), IntegrationProtocolsTests.swift (approved migration, 0->1 and 1->0.5 kept).
- New: Scripts/ci/fixtures/R05PublicConsumer.swift, R05OldPropertyNegative.swift, R05_public_driver.py.
The pause draft's six paths were reviewed against the brief and needed NO source edits: it compiled and passed first time.

## Contract mapped to brief
- Defaulted requirement, witness dispatch (not extension-only): customWitnessDispatchAndUnknownContext (concrete, generic, any DistanceMetric<Float>; default nil; spoofed name/identifier custom metric).
- Six inverse builtins: Float(1/(1+Double(d))) for finite d>=0, nil for negative/NaN/inf; signed zero -> 1; maxFinite gives positive subnormal: inverseBuiltinDomainsAndMonotonicity (0/1/2/3, leastNonzero/leastNormal/maxFinite, negatives, nonfinite, 4-ULP independent Double oracle, monotone).
- DotProduct logistic, no pole at -1: dotLogisticDomain (-1 and Float neighbors, +-0, tiny, +-20/80/100, +-1000 and +-maxFinite as exact limiting endpoints 0/1, nonfinite nil); reference is direct Double 1/(1+exp(d)), not the piecewise code.
- Explicit scores exact bits (0, -0, negative, out of range, NaN payloads, +-inf), never invoke witness, mapIDs preserves: storedScoreBitsAndMappingNeverInvokeWitness.
- Codable same keys, absent/null score, configured nonfinite strategy (NaN payload not claimed): codablePreservesExplicitValuesAndMissingContext.
- TopK conversion leaves nil: topKConversionRequiresExplicitMetricContext.
- One score call per selected result, none for k<=0 / empty, default-nil; both producer routes (default extension via custom collection, SimpleVectorCollection): selectedCallbacksAndDefaultNilAcrossProducers.
- All seven builtins in real searches incl. Minkowski p=3, both producers: allBuiltinProducerGeometriesAndConfiguredMinkowski.
- Custom search override authoritative; convenience methods do not overwrite: collectionOverrideScoresRemainAuthoritative.
- Reentrant witness reads/mutates SimpleVectorCollection outside lock, selected snapshot kept: reentrantScoreRunsOutsideLockAndKeepsSelectedSnapshot (child process, 15 s deadline, 2 s cleanup, throws on timeout before any terminationStatus access; scanner passes).
- Historical regression: dotProductSearchRelevanceRegression.

## Migration
Removed `SearchResult.computedScore` property. Use `.score` for producer-populated values or `.computedScore(using: metric)` with the metric that produced the distance. Only callers in the repo: SearchResult.swift and IntegrationProtocolsTests (migrated). No README/Docs use.

## Commands (all exit 0; no timeouts, so no isolation re-run was needed)
All under ev/ in this directory.
1. Debug: capture_swift.py ... -- test --jobs 6 --scratch-path ev/build-debug --filter 'SearchResultScoreContractTests|IntegrationProtocolsSuite|MetricDispatchContractTests|VectorCollectionTransactionTests'. Exit 0. Log ev/debug.log, record ev/debug.json. 71 tests in 4 suites passed.
2. Release: same with `-c release --scratch-path ev/build-release`. Exit 0. ev/release.log, ev/release.json. 71 tests in 4 suites passed.
   Symbols confirmed to exist: SearchResultScoreContractTests, IntegrationProtocolsSuite (display "Integration Protocols"), MetricDispatchContractTests, VectorCollectionTransactionTests.
   Counts: new suite has 12 test functions; 2 parent no-ops (R05 reentrantScoreChildProbe, R02 reentrantSearchChildProbe, each returns immediately without its env var); 2 actual children (1 R05, capture in children/*/score.log + command.json, status 0, output R05_REENTRANT_SCORE_COMPLETE; 1 R02, its own temp log). Child process count per run: 2.
3. Public consumer: `python3 Scripts/ci/fixtures/R05_public_driver.py --products <build>/out/Products/{Debug,Release} --output ev/fixture-{debug,release}`. Exit 0 both. Positive compile+run (Swift 6, strict concurrency complete, no @testable) prints R05_PUBLIC_COMPLETE; negative control R05OldPropertyNegative.swift fails with "cannot convert ... '@Sendable (any DistanceMetric<Float>) -> Float?' to 'Float'".
4. Scanner: `python3 -B -m unittest discover -s Scripts/ci/tests -p test_child_process_bounds.py -v`: 7 tests OK.
5. SwiftLint 0.65.1 on the 8 changed Swift files (ev/lint.log): 0 errors, 5 warnings: 3 pre-existing trailing_comma in IntegrationProtocolsTests, 2 new in my test file (statement_position at line ~332, vertical_parameter_alignment ~198). Not fixed, to avoid re-running the matrix; trivial.

## GREEN vs RED
RED (unchanged production, 2026-10-02, Swift 6.3.3, not re-run): ids [2,1,0], distances [-2,-1,-0.5], scores [-1, Inf, 2].
GREEN (Debug and Release identical): ids [2,1,0], distances [-2,-1,-0.5], scores [0.8807971, 0.7310586, 0.62245935]; finite, within [0,1], descending, within 4 ULP of Double 1/(1+exp(d)).

## Open concerns
- Mutation sanity not performed: I did not temporarily break the lock placement or logistic form to confirm the tests go red (reentrancy test would hang to its 15 s deadline if scoring ran under the lock).
- 2 new SwiftLint warnings in the new test file.
- Fixture driver depends on the swiftbuild layout (Products/<cfg>/VectorCore.o, VectorCoreC.o, Intermediates.noindex/GeneratedModuleMaps/VectorCoreC.modulemap) and is not wired into any CI script.
- Extreme-distance kernels of the builtins are untouched, as scoped. Release-mode tests ran with the focused filter only, not the full suite.

## Fix round 1 (edits) -- Python-only checks done; Swift verification PENDING (no build/test run yet)
Edits:
- I1: SearchResultScoreContractTests.swift child captures now go under FileManager.default.temporaryDirectory/R05-children/<UUID>; R05_CHILD_CAPTURE path is still printed. No path-name inference remains.
- I1-class: UMAPRangeContractTests.swift only the evidenceRoot lines: now temporaryDirectory/L03-children (L03_CHILD_CAPTURE still printed). New Scripts/ci/tests/test_swift_test_output_paths.py (stdlib only, no subprocess) rejects `lastPathComponent ==/!=` and the "workspaces" literal in Tests/**/*.swift; includes negative controls on synthetic strings and an env-gated (R05_ORIGINAL_TREE) check for the original files.
  RED: scratch tree with the pre-fix UMAPRangeContractTests.swift (git show HEAD:) and the pre-fix R05 test (the pause-draft copy, sha 654e716c..., identical to the pre-fix file, since the file is new and not in HEAD) flags both files, each with both rules (lines 372 and 320). Log: ev/policy-red-original-fix1.log. Repository scan on current tree: clean.
- M1: DistanceMetrics.swift comment now says single final rounding of the widened value.
- M2: maxFinite expects literal Float(bitPattern: 0x0020_0000); exact == 1 bits for -0.0, leastNonzero, leastNormal; closeScore tolerance uses min(expected.ulp, expected.nextDown.ulp). Other assertions kept.
- M3: negative control is `let _: Float? = result.computedScore`; driver now checks only ': error:' diagnostic lines for "cannot convert", "DistanceMetric" and 'Float?' and raises RuntimeError otherwise.
- M5: CHANGELOG.md Unreleased/Changed bullet added after the T01 entry (score carried by producer results; `producerResult == SearchResult(id:, distance:)` now false; Hashable still hashes id and distance only). Migration section of this report: same note applies -- producer results now carry a score, so equality with a score-less SearchResult is false for builtin metrics.
- M6: both new SwiftLint warnings fixed.
Python checks: policy test 4 tests OK. Full discovery `python3 -B -m unittest discover -s Scripts/ci/tests -v`: 44 tests, OK (skipped=1, the env-gated hook), ev/policy-discovery-fix1.log. SwiftLint 0.65.1 on 9 changed Swift files: 0 errors, 18 warnings, none in the R05 files; 3 pre-existing trailing_comma in IntegrationProtocolsTests and 15 pre-existing line_length/statement_position in UMAPRangeContractTests (not in the lines I changed), ev/lint-fix1.log.
Note: I ran read-only `git show HEAD:<path>` for the RED copy before the no-git message arrived; no index operations.
M4 untouched per ruling. Pending: focused R05 and L03 filters in Debug/Release, public driver, confirm captures land under temp dir.

### Fix round 1 -- Swift verification results (all under ev/, suffix -fix1)
All via capture_swift.py, `--jobs 6`, scratch ev/build-r05-{debug,release}-fix1 (L03 runs reused the same per-config scratch). No timeouts, so no isolated re-runs.
- R05 filter (SearchResultScoreContractTests|IntegrationProtocolsSuite|MetricDispatchContractTests|VectorCollectionTransactionTests), Debug: exit 0, 71 tests in 4 suites passed (r05-debug-fix1.log/.json). Release: exit 0, 71 tests in 4 suites passed (r05-release-fix1.log/.json). Same counts as before: 12 functions in the new suite, 2 parent no-ops, 2 actual children.
- L03 filter (9 suites listed in the task), Debug: exit 0, 39 tests in 9 suites passed, 9 children (l03-debug-fix1.log). Release: exit 0, 39 tests in 9 suites, 9 children (l03-release-fix1.log). Matches the accepted result.
- Public driver, Debug and Release (ev/fixture-{debug,release}-fix1): exit 0, R05_PUBLIC_DRIVER_PASS. First attempt failed in my own new diagnostic check (not the product): the compiler prints "cannot convert value of type '@Sendable (any DistanceMetric<Float>) -> Float?' to specified type 'Float'", so my expectation of a trailing 'Float?' was wrong. I corrected the driver to match "cannot convert value of type" plus the method type text, and re-ran both configurations successfully. The check reads only ': error:' lines, not echoed source.
- Policy test re-run after that driver edit: 4 tests OK (earlier full discovery was 44 tests OK, skipped=1).
- Child captures now land under the temp directory: R05_CHILD_CAPTURE and L03_CHILD_CAPTURE paths printed in the logs are /var/folders/.../T/R05-children/<UUID>/score.log and .../T/L03-children/<UUID>/<probe>.log. No ~/R05 or ~/L03 created, and no new entries in the old evidence-tree children directory (its 2 entries predate the fix).
Note: the driver edit changed R05_public_driver.py after the earlier lint run; it is Python, unaffected by SwiftLint. ev/hashes.txt is from the original round; refresh if the package needs round-1 hashes.
