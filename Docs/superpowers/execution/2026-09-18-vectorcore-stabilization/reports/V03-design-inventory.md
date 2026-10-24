# V03 local executable-coverage and performance-gate inventory

Read-only source inventory, not a CI execution result. Inspected current Package.swift, all four workflows, both SwiftLint configurations, CI validators/tests, benchmark CLI driver and timing assertion families under Tests. No builds, remote API queries, branch-protection inference or new performance target.

## Local workflow-to-command mapping

| Surface | Actual configured command/coverage | Boundary |
|---|---|---|
| CI repository-checks | Ubuntu Python unittest discovery under Scripts/ci/tests, then Ruby validate_github.rb | Discovers unittest classes, not arbitrary script main bodies |
| CI test | macOS15, Xcode16.2/26.3 x Debug/Release; memory-pool saturation driver then unfiltered swift test | EXTENDED0/STRICT_PERF0; registered normally enabled tests only |
| CI lint | pinned SwiftLint0.65.1 `lint --config .swiftlint.yml` | No `analyze` invocation or compiler-log input |
| CI consumer | Xcode16.2 documentation consumer_smoke.py | Actual examples owned by that driver, not every task-specific public fixture |
| CI platforms | Xcode26.3 Release library xcodebuild for iOS/tvOS/watchOS/visionOS simulators and Catalyst; Intel swift build --build-tests | Compilation only, no platform test execution |
| CI perf-smoke | Release vectorcore-bench dot/distance/normalize,512D,3samples,seed1,JSON artifact | Separate job excluded from CI Required; command success remains checked, no speed target |
| Sanitizers | ASan and TSan targeted swift test filters, separate workflow | Instrumented execution only for matching actual test bodies; no full-suite claim |
| CodeQL | actions/no-build; Swift and C/C++ manual `swift build --target VectorCore` under extraction, then analyze | Static analysis of extracted library/dependencies, not test/benchmark execution or every Swift source in repository |
| Benchmarks workflow | dispatch-only Release full-profile CLI; then Release swift test EXTENDED1/STRICT_PERF0 | Extended timing assertions below currently remain live |

CI Required uses `always()` and depends exactly on test/lint/repository-checks/consumer/platforms; check_required.py receives those names. Sanitizers, CodeQL, perf-smoke and manual benchmarks are not dependencies of this aggregate. Whether external branch protection requires any other check cannot be established from these files. Ruby validate_github.rb fixes the same required-name list and checks job definitions/dependencies plus general YAML/workflow hygiene; existing mutation tests do not establish that each required job still executes its promised semantic command or that sanitizer filters select nonempty intended suites.

Package already registers MinimalTests, ComprehensiveTests and V01's BenchmarkContractTests, the latter depending on VectorCoreBenchSupport/VectorCoreBenchmarking. VectorCoreBenchSupport is the benchmark implementation target and VectorCoreBench is its executable wrapper. Do not propose 'registering' the already registered V01 target as new coverage. Tests/PerformanceTests is not a target path: NSWOptimizationBenchmark.swift there is not compiled/executed by swift test. It references GraphConstructionKernels.NSWOptions/buildNSWIndex, which current Sources searches do not find. It is obsolete source, not an executed failing test; root must choose deletion/archival rather than activating it accidentally.

ASan filter currently names MemoryPoolTests, SIMDProviderBoundsTests, SerializationEdgeCaseTests, SoAPageAlignTests, UnifiedVectorBufferTests, TopKNaNContractTests and QuantizedEuclideanOverflowRegressionTests. TSan names MemoryPoolTests, ProviderConformanceTests and TopKNaNWrapperTests. Many new stabilization suites are outside these filters. A matching parent no-op or sanitizer guard is not proof its child body ran. V03 should pin chosen actual bodies/routes and retain executed counts rather than broaden filter strings and infer coverage.

Root SwiftLint config includes Sources/Tests; nested Sources/.swiftlint.yml exists with additional opt-ins and its own include/exclude settings. Both declare unused_declaration/unused_import as analyzer_rules, but `lint` does not establish analyzer execution. Root's explicit includes do not provide blanket Benchmarks/Scripts coverage. Any desired analyzer/benchmark lint expansion needs a deliberate actual invocation/configuration and evidence, not a declaration-only claim.

## CLI driver wiring gap

Scripts/ci/tests/test_benchmark_contracts.py now safely defines main and calls it only under __main__ (accepted EX07). Unittest discovery imports it but runs none of its CLI assertions. No workflow explicitly invokes it with a built vectorcore-bench executable. Its actual checks are ten invalid threshold configurations, JSON/CSV write failures, one finite strict-correctness batch run, and the28-provider-call/14-case source guard. Add a deliberate built-binary invocation if these are chosen required checks. Perf-smoke running a different CLI command does not replace those cases. BenchmarkContractTests supplies registered library/support tests, not automatically these process contracts.

## Measured timing assertions still live with EXTENDED1/STRICT_PERF0

Paths below are under Tests/ComprehensiveTests. These source conditions do not consult strictPerfGatesEnabled:

| File and line(s) | Measured condition | Current activation |
|---|---|---|
| MixedPrecisionPerformanceBenchmark.swift:100,101,134 | FP16/mixed speedup>1 | Extended test attributes |
| Same:227 |1536D speedup>=512D speedup | Extended |
| MixedPrecisionComprehensiveTests.swift:784,788 | speedup>1 and FP16mean<300ns | Extended |
| Same:824,825 | mixed and FP16means<50000ns | Extended |
| MixedPrecisionKernelTests.swift:2972 | opsPerSecond>1e6 | Normally enabled testSIMDUtilization |
| Same:3040,3045,3054 | measured throughput positive, scaling>=0.8previous, timing-selected optimal batch>=10 | Extended testThroughputScaling |
| Same:3179,3180,3183,3192,3197 | positive measured latency/throughput, batch latency>single, latency/throughput scaling>=0.9previous | Extended testLatencyVsThroughput |
| Same:3208 | timing-selected optimalForLatency>0 | Extended; fallback1 makes this effectively tautological, still derived from measured data |
| QuantizedKernelsTests.swift:2133 | min-latency improvement>1.5 | No strictPerf guard; adjacent throughput gate does have one |

Do not gate entire mixed tests if they also contain numerical/storage checks. For example testSIMDUtilization has actual output distance/storage assertions after its timing assertion. Extended activation and strict speed gating are independent policies. Positive measured-time/profile validity predicates also need an explicit classification: they check timing data validity rather than a speed target, but still depend on the clock and should not be silently mistaken for fixed-data correctness.

Existing strict-gated speed assertions are already disabled at STRICT_PERF0. Mechanical inventory of their source sites (line numbers of guard or assertion):

- ErrorHandlingTests:1719–1728.
- QuantizedKernelsComprehensiveTests:943–985.
- MixedPrecisionKernelTests:2827–2828,3357–3358,3450–3451,3684–3685,4140–4141.
- MixedPrecisionKernelsTests:227–229,930–931,1095–1096,1246–1249,1470–1471,1529–1531,1600–1603,1867–1868.
- QuantizedKernelsTests:629–630,730–731,1459–1460,1622–1623,2166–2167,2267–2268,2489–2490,2534–2535,2642–2643.
- BatchKernels_SoATests:773–774,824–826,1091–1092,1113–1114,1248–1249,1275–1276,1305–1306,1337–1338,1435–1436,1497–1499,1510–1511,1530–1531,1576–1578,1615–1616,2868–2869,3110–3111.

The indirect booleans/ratios in that last list matter: allTimesReasonable, hasPerformanceCliff, cv, scalingFactor and selected best batch can depend on timings without containing 'time' in the assertion. A mechanical guard should cover an explicit maintained inventory or parsed policy marker rather than claim a keyword search proves arbitrary Swift dataflow.

## Conditions that must not be confused with speed targets

- MixedPrecisionPhase3Tests:232/233/280/281 compares explicitly supplied0.001/0.005/etc metric records; IntegrationProtocolsTests:68 checks a supplied searchTimeMs1.0. These are fixed-data model checks.
- QuantizedKernelsTests:2183 energyEfficiencyRatio comes from constant operation-count arithmetic, not measured time or actual energy; QuantizedKernelsComprehensiveTests:917 bandwidthReduction is storage-size arithmetic. Preserve the correctness assertion if desired, but do not call it measured energy/bandwidth evidence.
- MixedPrecisionPerformanceBenchmark accuracy assertions and ordinary norm/ratio/UMAP separation checks are numerical checks, not clock gates. Keep them active when strict speed checks are disabled.
- KernelAutoTunerTests:152–153 checks positive profile throughput/latency from profiling, distinct from fixed injected timing metrics; adjacent memory/accuracy assertions are not time gates. CI cache/persistence bypass in KernelAutoTuner is a separate Q02 concern, not proof of ordinary persistence coverage. Coordinate with Q02's future test migrations.
- ErrorHandlingTests:80 `abs(timestamp.timeIntervalSinceNow)<1s` is a live clock-based freshness assertion, normally enabled and independent of strictPerf. Replace with captured before/after timestamp bounds if root wants to remove elapsed-speed sensitivity without weakening timestamp correctness. Existing timestamp>epoch and OperationProgress before/after bounds test timestamp semantics, not computational throughput.
- Bounded child/process/gate deadlines are liveness/cleanup controls. Do not disable them under STRICT_PERF0. Same for workflow job timeouts and intentional watchdog assertions. E02's helper revisions are pending separate integration and should not be overwritten by V03.
- MinimalTests/ComprehensiveBenchmarks uses XCTest `measure` in26 methods, enabled by EXTENDED1. There is no repository-authored speed threshold in these methods; numeric anti-elision assertions remain. Measurement delegates to XCTest's performance machinery and is not converted into a promise of assertion-free observation merely because no explicit comparison appears. Root should explicitly decide whether to skip XCTest measurement tests at STRICT_PERF0 or replace the measurement harness with observation-only timing; do not claim platform measurement behavior was executed here.
- MinimalTests/PerformanceBenchmark prints manual timings without explicit timing assertions. It is separate from the unregistered NSW XCTest measurement file.

## Bounded design decisions

Choose required local executable checks and align aggregate needs, validator policy and mutation tests with actual commands. Add an explicit CLI contract invocation on the built product; preserve artifact-only performance reporting. Decide lint versus analyzer coverage honestly, and select narrow sanitizer bodies with proof of nonempty execution/no-op distinctions. Retire the obsolete unregistered NSW source deliberately. Close the ungated wall-clock class without removing numerical/storage/model assertions or deadlines. No new speed target, remote protection setting, whole-suite sanitizer claim or downstream execution claim is warranted by this inventory.
