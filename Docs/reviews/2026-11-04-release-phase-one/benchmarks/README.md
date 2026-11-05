# Benchmark comparator execution, 2026-11-04

The existing required `test_benchmark_contracts.py BINARY` command now executes the comprehensive current-schema comparator table and comparisons of reports produced by the actual benchmark product. Local execution passed with the verified Swift.org 6.4.2 compiler, SDK 27.2 and a fresh Release product built with two compiler jobs. This is tooling and selected-case correctness evidence. Performance acceptance remains pending.

## Reconciliation

The original record is **I:D-089**, in `/Users/goftin/dev/projects/VectorCore-audit-2026-10-27-84798d3/evidence/reviewers/tooling-known-issues-ownership-supplement.json` (retained extraction: `original-D-089.json` under the evidence root below). Its disposition already distinguished existing tests from partial required execution. The broader categorical claim that no comparator tests exist was superseded before this change: required CI already compiled and invoked the comparator for the legacy README examples and configuration/refusal controls. The comprehensive `test_benchmark_reporting.py` table existed, but its main-only entry point was not reached by unittest discovery or the required driver.

This change closes that local execution/wiring gap. It does not claim that the changed required command has run on hosted CI. Related **G:A-027** retains its quiet-host timing limitation. **H:B-130** already describes repaired identity/boundary validation and CSV quoting; those repairs are not redispatched here. Historical baseline files remain historical records. Report p90 remains a percentile of per-sample averages.

## Executed coverage

| Execution | Result |
| --- | --- |
| Fresh `swift build -c release --product vectorcore-bench --jobs 2` and matching `--show-bin-path` | Exit 0; source/compiler/SDK and product hashes retained |
| Required CLI contract driver | Exit 0; 17 actual CLI calls and 19 comparator invocations |
| Current-schema reporting driver, launched by that required command | Exit 0; 302 records: 301 comparator invocations plus product-discovery control |
| Real strict two-case schema-3 reports | Two runs; generic and optimized 512D sequential batch Euclidean-squared cases; valid correctness observations and no failures |
| Real-report self-comparison | Exit 0; two matched cases, median per unit, no compatibility override |
| Distinct real-report comparison | Exit 1 retained; one timing regression on a competing-build host; no performance acceptance |
| Incompatible identity, missing case, invalid median, recorded failure on either report side | Expected exits 2, 1, 2, 2, respectively |
| `BenchmarkGateDurability` | Three tests passed; 36 rejecting probes and six valid probes across normal Python, `-O`, `-OO` |

The reporting table covers hand-derived regression/improvement deltas, per-unit normalization, missing requested statistics, required identities, nonwaivable boundary differences and malformed boundaries, overrides for legacy identities, invalid numerical/correctness records, recorded failures, missing cases and zero matches. Its 23 removable Python `assert` statements became explicit checks before making this driver required. `test_negative_command_log_and_report_fixtures_in_all_python_modes` and `test_runtime_asserts_cannot_return_in_the_benchmark_drivers` cover this recurring class across all three benchmark drivers. RED evidence retains the six expected failures before the repairs.

## Reproduction and evidence

Evidence root: `/private/tmp/vectorcore-release-phase-one-2026-11-04/benchmarks`.

`result-summary.json` records counts, product hash and limitations. Every executed child in the real contract run has exact argv, selected environment, deadline, exit and stdout/stderr hashes in its adjacent command JSON. Top-level commands use `run.py` and the existing `v03_process.run`; nested contract commands use the existing bounded benchmark helper. `source-inputs.json`, `source-stability.json`, `input-identities.json` and retained `swiftbuild-metadata/` describe the built inputs. The source baseline is `2dcb24091c6dc21be4db409ecdd0756220e93b8c`; production sources were unchanged. The worktree had only this track's test edits while building.

With the qualified toolchain directory first in `PATH`, `SWIFT_EXEC` set to that directory's `swiftc`, and `SDKROOT` set to the recorded MacOSX27.2 SDK, the actual contract invocation was:

```sh
python3 -B Scripts/ci/tests/test_benchmark_contracts.py \
  /private/tmp/vectorcore-release-phase-one-2026-11-04/benchmarks/build/out/Products/Release/vectorcore-bench \
  --output /private/tmp/vectorcore-release-phase-one-2026-11-04/benchmarks/actual-cli
```

The actual-product comparison can be repeated without synthesizing reports:

```sh
swift Scripts/compare_benchmarks.swift \
  --baseline /private/tmp/vectorcore-release-phase-one-2026-11-04/benchmarks/actual-cli/product-before.json \
  --current /private/tmp/vectorcore-release-phase-one-2026-11-04/benchmarks/actual-cli/product-after.json \
  --metric median --per-unit --threshold 10
```

The recorded cross-run command used the compiled comparator and default metric/threshold, equivalent to the explicit flags above. Its exit 1 is preserved in `actual-cli/comparator/compare-11.json`.

## Remaining gates and self-review

The final timing gate needs settled merged source and compiler identity, a quiet host, Release builds, warm-up, enough samples and repeat runs, retained raw samples/thermal observations, and the strict comparator without compatibility/missing-case waivers. This run is too small and contended to satisfy it.

Runtime checkout/toolchain metadata remains an observation rather than proof of build provenance; this run supplies external build/input identities. Dot, normalize and mem.copy cases without independent numerical observations are not qualified by these batch controls. CSV completeness, optional adversarial CSV parsing, duplicated comparator boundary codec/scope lists, and thermal/power compatibility remain outside this repair. The optional `--csv` reporting branch was not run here. Existing build warnings are retained, including Swift 6.4.2 Sendable-metatype and unnecessary-`try` diagnostics. No workflow, package, production benchmark schema, numerical threshold, identity guard or comparator implementation changed.

Reachability review: required CI still calls the same CLI driver; that driver now reaches reporting `main()`, which compiles and executes all unconditional table cases. Its CSV branch remains conditional on `--csv`. Discovery remains import-safe and does not compile Swift on repository-checks runners. The new cross-run comparison permits recorded exits 0 or 1 solely within this tooling contract; self-comparison and deterministic negative controls retain their exact expected exits. No performance gate is relaxed.
