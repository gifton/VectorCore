# Executed CI contracts

This table describes configured remote commands. Local verification is recorded separately in V03 evidence; YAML and policy success do not establish a hosted run or branch protection.

| Surface | Target / command | Activation and coverage |
| --- | --- | --- |
| Ordinary tests | `swift test -c "$CONFIGURATION"` | Debug/Release, Xcode 16.2/26.3; extended=0 and strict=0. Guarded child entry functions alone are not child coverage. |
| Performance opt-out | `python3 Scripts/ci/check_performance_gates.py --configuration "$CONFIGURATION" --scratch-root DIR --output DIR --jobs 3` | Fresh environments; lazy Swift Testing/XCTest failures, live numeric controls, one-shot observations and named real extended tests. |
| Benchmark contracts | `swift build -c release --product vectorcore-bench`, actual `--show-bin-path`, then `python3 Scripts/ci/tests/test_benchmark_contracts.py BINARY` | Actual built CLI invalid-input/output/strict-correctness cases, the `--ab-only` suite contract and the configuration banner; the compiled `compare_benchmarks.swift` on the baselines README invocations and its configuration warnings. Python discovery is separate and is not CLI execution. |
| C modes and routes | `python3 Scripts/ci/check_benchmark_build_modes.py --scratch-root DIR --output DIR --jobs 3` | Four distinct Debug/Release × optional-C off/on products; compiled `-DVC_USE_C_KERNELS`, the configuration banner per product, actual public forwarding counts and `V02_BUILD_MODES_COMPLETE`. |
| Address sanitizer | `python3 Scripts/ci/check_sanitizers.py --sanitizer address --scratch-root DIR --output DIR --jobs 3` | 77 pinned valid bodies; O05 borrows/ownership, existing selected suites and E02 cancellation/reductions. Leak detection is disabled. |
| Thread sanitizer | Same shared driver with `--sanitizer thread` | 39 entries; Q03 parent requires its actual instrumented child, six × 2000 loops and all result/join markers. Parent deadline 49s, child 45s, cleanup ≤2s; other executions 15s. |
| Repository policy | `python3 -B -m unittest discover -s Scripts/ci/tests -v`; `ruby Scripts/ci/validate_github.rb` | Exact seven-job aggregate, semantic commands, independent sanitizer body set, maintained timing sites, target membership and mutation controls. Narrow lexical/workflow checks, not arbitrary Swift/shell dataflow proof. |
| Lint | Pinned SwiftLint 0.65.1 `lint --config .swiftlint.yml` | Sources, Tests and Benchmarks; no `analyze`, unused-declaration or unused-import analysis claim. |
| Documentation consumers | `python3 Scripts/ci/consumer_smoke.py` | Build and execute extracted supported public examples; compile the nine must-not-compile contract fixtures and their positive controls. |
| Apple platforms | `xcodebuild` library destinations; Intel `swift build --build-tests --triple x86_64-apple-macosx14.0` | Compilation only for these destinations; no device or downstream runtime claim. |
| CodeQL | Separate scheduled/PR CodeQL workflow | Library and extracted dependencies; outside CI Required, not all tests/benchmarks/every file. |
| Perf smoke / full benchmarks | Separate reporting jobs | Outside CI Required; executable failure still fails their own jobs. No speed target. |

CI Required has exactly `test`, `lint`, `repository-checks`, `consumer`, `platforms`, `benchmark-contracts`, `sanitizers`; `always()` runs the exact-result aggregate. Missing, skipped, cancelled and failed results reject.

BenchmarkContractTests was already registered before V03. Its default source membership remains `BenchmarkContractTests.swift`, `BenchmarkSelectionContractTests.swift`, `RunnerContractTests.swift`, `SwiftTestingEntryTests.swift`, with dependencies VectorCoreBenchSupport and VectorCoreBenchmarking. The policy rejects target/source exclusions.

`Scripts/ci/performance_sites.json` inventories 71 existing timing assertions and two intentional gate probes, plus 29 existing measurement bodies and one observation probe. Strict is enabled only by the exact string `1`; disabled lazy predicates do not evaluate. Fixed injected tuning metrics, numerical assertions, allocation counts and liveness deadlines remain active. The unregistered obsolete NSW benchmark source is retired, not registered.

`Scripts/ci/sanitizer_bodies.json` is the exact runtime body manifest; `sanitizer_expected_bodies.txt` is its independently maintained required set. Completion requires actual body events/output and host/bundle/runtime identity, with Q03 child evidence retained separately. This is selected-path coverage, not whole-suite sanitizer coverage.
