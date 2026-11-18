# Remaining-repair architecture and contracts

This is a bounded remediation review of `db93295` plus the three authorized repairs. The original whole-repository audit remains frozen. The review-input patch and per-file hashes are in `reviewers/final-input.json`; the final coverage ledger identifies inspected bounds. These are source-traced execution paths, not measured code coverage.

## Contracts

| ID | Authority and enforcement | Assumptions / validation |
|---|---|---|
| RC-012-domain | `CorrectnessPolicy` rounding models and gamma precondition; `domainProblem` checks count addition, nonnegative counts, k*u<1, positive relative denominator, finite nonnegative threshold. | Existing modeled dimensions and formulas retained. Raw internal model/rule construction retains its documented precondition. `BenchmarkModelDomainTests.testProviderModelDomainBoundaries` and `testDomainCheckRejectsUnrepresentableProviderCounts`. |
| RC-012-admission | `BenchmarkRunner.run` checks actual selected executable cases before `CaseExecution.execute`. Threshold overrides do not make an undefined model admissible. | Existing option arithmetic checks, catalog availability, filters and AB rules remain earlier gates. Subprocess `onCase` sentinel precedes setup and prevents large allocations even on regression. Selected-invalid, excluded-valid and unavailable controls exercise the real runner. |
| RC-012-results | Defensive result policy checks domain before rule evaluation; summaries classify without constructing a rule. | Injected adapters may return names different from the catalog name. `testUnsupportedInjectedResultsAreReportedWithoutTrapping` covers names at the boundary and Int.max, override combinations, strict modes and pretty/JSON/CSV. Non-strict report-with-failure exit 0 remains intentional. |
| RC-011-manifest | D02 table membership must exactly match Swift fixtures; both required TSV files must exist before output creation or snapshot discovery. | New policy tests remove each input from seven retained explicit drivers and exercise unlisted D02 files and missing TSVs. The AST inventory supports literals and tuple-loop bindings only; it rejects unrecognized fixture-relative expressions and is not a general dataflow proof. |
| RC-011-consumers | Historical driver retains baseline/final fixture expectations and final-mode count/binding/member backstops. | Required Logger line 10, optional line 9 are shared with maintained smoke judgment, with compiler exit 1 additionally required. Other negative fixtures still require exactly one error per code line. Stage output uses SILGen's own status. Actual Debug/Release snapshots pass ten fixtures, 61 bindings, 77 members. No historical baseline replay. |
| RC-FP16-operands | The two test helpers accept paired NaNs/equal-sign infinities and reject remaining nonfinite operands before arithmetic. | Literal ordered 7-by-7 classification matrices and actual-output corruption tests. Native Float16 tests/callers retain the existing Intel macOS/Catalyst exclusion. |
| RC-FP16-finite | Existing asymmetric denominator, zero fallback, inclusive absolute-or-relative tolerance and original-side strict abs(original)<2^-14 flush allowance are retained. | Named finite-boundary, shape/prefix and flush tests; finite relative overflow with absolute success stays accepted. Invalid tolerance values are not assigned a new policy. |
| RC-FP16-inventory | Both active helpers are finite-gated inventory entries. The lexical inventory rejects missing guard fragments and unreviewed comparison sites. | Existing gate-removal controls include both new guards. Synthetic dormant fixtures exercise body edits, direct use and ordinary/raw interpolation in Tests/Sources even though no real dormant entries remain. This does not prove control-flow dominance; the Swift behavioral tests exercise the actual guards. |

## Source-traced paths

### RP-012-A — CLI admission

Entry: `BenchmarkRunner.run` in `Benchmarks/VectorCoreBench/AppMain.swift`. Options pass `ValidatedOptions`, then lazy catalog enumeration, inclusion/exclusion, availability and AB selection. Every executable selected name passes `domainProblem` before any suite or case executes. Unsupported model returns configuration exit 1 without a report. Admitted large controls terminate at `onCase` before setup; no giant fixture or allocation behavior is measured. Empirical routes and non-observed SoA/boundary/matrix models retain their own selection and validation behavior. See the F012 discovery review for all catalog/model inspection bounds.

Ownership: catalogs capture future fixture-building work; enumeration itself does not construct the candidate array. The case executor calls `onCase` before fixture setup. No new shared mutable state, backend route or cancellation policy is introduced. Underlying numerical model assumptions about Accelerate remain explicitly unverified contracts in the existing source.

### RP-012-B — result/report backstop

Entry: `CorrectnessPolicy.failure(for:)`. Missing/invalid comparisons retain their existing failure reasons. Valid statistics first pass domain checking, then existing same-kind threshold overrides and rule comparison. Unsupported injected names become recorded failures. `printCorrectnessSummary` repeats the same decision and separately counts unsupported domains without invoking a trapping threshold. Pretty summary remains present for JSON/CSV requests. The synthetic injected adapter is bounded; its result name alone carries the huge count. No huge execution claim follows.

### RP-011-A — retained consumer evidence

Entry: `Scripts/ci/tests/D02RunConsumers.py BIN OUTPUT final`. Read-only manifest preflight precedes output creation and module/object snapshots. Copied Swift fixtures typecheck; positives link against copied objects and run their required marker; negatives enforce diagnostic contracts. Then the lexical 61-symbol guard, actual SILGen/demangling binding check, and actual symbol-graph 77-member comparison run. Problems accumulate into provenance and nonzero final status. Binding output cannot be changed by later graph exit. Compiler children have existing deadlines/process-group cleanup. Snapshot layout compatibility is policy-tested for native/swiftbuild; this run executes native builds only.

The three restored files are byte-identical to the pre-deletion revision recorded in `f011-restored-fixtures.json`. Debug/Release provenance separately hashes copied inputs, objects, module files and run output. Existing historical records and pre-D02 expectations remain unchanged.

### RP-FP16-A — test helper activation

Entries: the two `MixedPrecisionKernelTests` extension helpers, the existing conversion-accuracy test and existing 512D distance-accuracy test. Equal-length arrays classify paired specials first, reject other nonfinite operands, then apply unchanged finite arithmetic; conversion additionally permits strict original-side tiny-to-zero flushing. Each accepted lane continues to later lanes. Direct tests corrupt every lane of actual 512D converted storage and every lane of a five-candidate initialized mixed-kernel output, in both reference/result directions. The existing tests keep their stronger per-element checks.

No production FP16 function, dimension, storage type, public API or platform gate changes. New kernel-output test storage is initialized before passing its buffer. Expanded FP16 development, external embedding/application architecture, performance measurement and cross-platform runtime validation are outside this repair.
