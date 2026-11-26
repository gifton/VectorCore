# Test environment controls

`python3 -B Scripts/ci/check_test_environment.py` and the Python policy suite discover
environment reads in every Swift file under `Tests/`. Every key has an explicit
classification in `Scripts/ci/test_environment_controls.json`: external gate,
owned child-process control, or named operating-system control. No prefix is exempt.

New external gates require an executable tracked setter and a documented scope.
Internal selectors require a named setter. A name in a comment is not activation.
Supported reads include direct ProcessInfo accesses, dictionary/process-info aliases,
literal constant keys, literal-key loops and getenv. Unknown dynamic accesses require
a file/function/expression-specific review entry. This lexical check is not a Swift
parser: it does not prove interprocedural data flow, arbitrary reflection, alias
shadowing, regex-literal syntax, or nested-comment correctness. Such new forms must
be reviewed and the discovery controls extended before adoption.

The mandatory UMAP command sets `VECTORCORE_UMAP_CHECK_BASELINE_BITS` and the internal
`VECTORCORE_UMAP_REFERENCE_CONTEXT` after sanitizing its child environment; see
[UMAP reference checks](UMAP_Reference_Checks.md). Ordinary tests still run their
tolerance assertions. The collector's general environment sanitizer is unchanged.

`VECTORCORE_TEST_EXTENDED=1` is set by the manual benchmark workflow. Ordinary PR
CI leaves timing workloads off. `Q03_TSAN=1` is set by `check_sanitizers.py` for
sanitizer-aware child execution.

## Diagnostic commands

These commands describe existing diagnostics, not substitutes for normally enabled
tests or the mandatory exact-reference rule. Skip controls deliberately reduce the
diagnostic scope. The strict-performance fixture and unsafe baseline can fail by
design. Optional output destinations use the caller's temporary directory.

`D02_SKIP_TRAPS` — Skip isolated traps while diagnosing in-process dimension checks.

```sh
env D02_SKIP_TRAPS="1" swift test --filter D02
```

`E02_CAPTURE_DIRECTORY` — Optional diagnostic output destination; does not enable numerical coverage.

```sh
env E02_CAPTURE_DIRECTORY="$TMPDIR" swift test --filter ExecutionSchedulingContractTests
```

`E03_CHILD_LOG_DIR` — Optional diagnostic output destination; does not enable numerical coverage.

```sh
env E03_CHILD_LOG_DIR="$TMPDIR" swift test --filter ConfigurationTransactionTests
```

`E04_CAPTURE_DIRECTORY` — Optional diagnostic output destination; does not enable numerical coverage.

```sh
env E04_CAPTURE_DIRECTORY="$TMPDIR" swift test --filter MemoryPoolBoundaryTests
```

`E04_SKIP_GUARDS` — Skip pool guard-page probes for in-process diagnosis.

```sh
env E04_SKIP_GUARDS="1" swift test --filter MemoryPoolBoundaryTests
```

`E05_CHILD_LOG_DIR` — Optional diagnostic output destination; does not enable numerical coverage.

```sh
env E05_CHILD_LOG_DIR="$TMPDIR" swift test --filter AutoTuningConcurrencyTests
```

`E06_CAPTURE_DIRECTORY` — Optional diagnostic output destination; does not enable numerical coverage.

```sh
env E06_CAPTURE_DIRECTORY="$TMPDIR" swift test --filter BatchConfigurationContractTests
```

`L01_CAPTURE_DIRECTORY` — Optional diagnostic output destination; does not enable numerical coverage.

```sh
env L01_CAPTURE_DIRECTORY="$TMPDIR" swift test --filter LinearAlgebraRangeTests
```

`L02_CAPTURE_DIRECTORY` — Optional diagnostic output destination; does not enable numerical coverage.

```sh
env L02_CAPTURE_DIRECTORY="$TMPDIR" swift test --filter PCACenteringContractTests
```

`L03_VALID_ONLY` — Run only valid-path UMAP range diagnostics.

```sh
env L03_VALID_ONLY="1" swift test --filter UMAPRangeContractTests
```

`N01_SKIP_GUARDS` — Skip normalization guard-page probes for in-process diagnosis.

```sh
env N01_SKIP_GUARDS="1" swift test --filter NormalizationRangeContractTests
```

`P01_RED` — Historical top-k defect reproductions; not ordinary-suite admission.

```sh
env P01_RED="1" swift test --filter P01
```

`Q01_CHILD_LOG_DIR` — Optional diagnostic output destination; does not enable numerical coverage.

```sh
env Q01_CHILD_LOG_DIR="$TMPDIR" swift test --filter QuantizationRangeContractTests
```

`Q03_UNSAFE_BASELINE` — Intentionally unsafe historical sanitizer contrast; never accepting evidence.

```sh
env Q03_UNSAFE_BASELINE="1" swift test --filter FP16DiagnosticsBoundaryTests
```

`R03_CAPTURE_DIRECTORY` — Optional diagnostic output destination; does not enable numerical coverage.

```sh
env R03_CAPTURE_DIRECTORY="$TMPDIR" swift test --filter MatrixMetricContractTests
```

`V02_CSV_OUTPUT` — Optional diagnostic output destination; does not enable numerical coverage.

```sh
env V02_CSV_OUTPUT="$TMPDIR/vectorcore-test.csv" swift test --filter BenchmarkSelectionContractTests
```

`VECTORCORE_STRICT_PERF` — Evaluate performance predicates; controlled failures are expected in the gate fixture.

```sh
env VECTORCORE_STRICT_PERF="1" swift test --filter PerformanceGateContractTests
```

## Owned selectors

The manifest names each child selector/payload/output and the source that sets it.
Its entries are checked against executable Swift/Python setters, including dictionary
literals. ASAN/TSAN and dyld settings are explicitly classified as runtime-owned.
