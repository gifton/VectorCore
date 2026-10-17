# V02 completed-runtime checkpoint review — v7 in progress

**Partial evidence review only; no final acceptance.** Final four-cell completion and sealed package are pending. No builds, tests or consumers were rerun. This record preserves already checked evidence for later final adjudication.

All11 file hashes in shared `V02-completed-runtime-checkpoint.json` match. That record and `V02-implementation-draft.md` predate the v7 repair and are historical partial summaries: final documentation must replace stale31-owned/pending-scope/all-production-identical wording with the accepted33-owned source and the exact three-annotation, two-Interop-file bridge. Benchmark Swift implementation/test bytes remain unchanged; all final Core bytes are not literally identical to v1.

Completed primary logs/commands verify:

- v1 Debug and v3 Release each exit0,20 XCTest methods plus one Swift Testing entry. The XCTest process entry and Swift Testing entry are parent no-ops;19 actual XCTest methods include the inherited child dispatchers. The reported41 inherited executions per configuration include the two directory-write failures. The historical Debug pair of external logs was not retained; parent assertions/full log remain, and final CLI records independently retain their JSON/CSV write failures. Do not call the lost files archived or reconstruct original bytes.
- v6 primary ASan exits0 with two actual XCTest methods and zero Swift Testing functions. Two bounded direct xctest executions also exit0. Their actual output is in stderr; the earlier stdout-only assertion was an offline parser/setup mistake, not a failing test or justification to rerun. Host/runtime/loaded-bundle mapping is explicit and must remain distinct from SwiftPM helper identity.
- **v7 C-on Debug and Release each report10 actual XCTest methods**, not nine: five QuantizationRangeContractTests and five QuantizedMetricRangeTests, including testIndependentDyadicDifferenceFixtures. Both also execute three Swift Testing functions, one childProbe parent no-op and two actual dispatching tests. Both command records exit0. Each retains100 Q01 child captures:99 with named fatal diagnostics and one outlier control with Q01_VALUE127/Q01_ACCEPTED and child pass summary. All200 capture hashes match. Capture-window/source/parent association must not be overstated as independently logged argv/PID for every child.

Rehashed every entry in completed retained manifests: Debug431, Release430, ASan352, C-on Debug352 and C-on Release351; no mismatch. These counts include artifacts/captures and are not test counts. Actual xctest and swiftpm-testing-helper host files are retained separately from bundles and dSYM files with the same basename. The bounded ASan record names xctest and an explicit preload runtime; it does not falsely identify the test bundle as the host. Frozen RunnerContractTests installs its termination handler before launch, uses15+2-second bounds and returns status only after successful completion; inherited Q01 handling remains governed by its accepted bounded harness.

Inspected all completed command records and rehashed stdout/stderr for the following tables; all have completed status and timeout=false:

| Evidence | Actual command count | Observed exits |
| --- | ---: | --- |
| reporting-final-v5 |131 (130 comparators + compiler)|13×0,116×2,2×1|
| selection-debug-v3 |51|36×0,15×1|
| selection-release-v3 |51|36×0,15×1|
| v01-cli-debug-v3 |13|1×0,12×1|
| v01-cli-release-v3 |13|1×0,12×1|
| actual-matrix-comparator-v3 |6|6×2|
| comparator-correctness-red |86|26×0,60×2|

Reporting's132 summary checks additionally count independent Python CSV parsing and pure product discovery; they are not132 comparator processes. Comparator RED source is byte-identical to frozen-v1 and the86 expectations have exactly20 false accepts and66 matching guards/controls. False accepts are malformed correctness shapes/statuses and fractional counts, with and without metadata override. The repaired reporting table executes the guards; six actual failure-bearing matrix reports reject under both override settings. CLI failure exits are expected validation/strict-policy controls, not universally passing numerical benchmark cases. Dense-matrix errors, staged V04 timing claims and unchanged strict thresholds remain qualified as in prior source reviews.

Final work: verify the sealed immutable index, actual completed four-cell v7 source/object/host/consumer mappings, all route/tail markers and C-request controls, final report counts and retained source-version bridges. No final approval is inferred from two completed Debug cells or successful focused runs. Root has been notified of the10-XCTest correction; it is an evidence-summary correction, not a production defect.
