# Validation, benchmarking, CI and tooling review

Revision: `8b02e75bca7055ae11170873d15c06f1538709b7` (main). No repository edits.

## Scope and verification

Read the seven `Sources/VectorCoreBenchmarking/*.swift` files; all benchmark suite/harness/utility/formatter/CLI sources under `Benchmarks/VectorCoreBench`; the standalone Top-K manifest, source, comparison script and measurement record; both baseline JSON metadata/case inventories; `Package.swift`; both SwiftLint configurations; all four workflows; both root Swift scripts; all CI scripts and their Python tests. Representative Swift test review included `MixedPrecisionPerformanceBenchmark`, `PerfGate`, both MinimalTests benchmark files, the unregistered NSW benchmark, pointer seam, SoA layout, Top-K ordering, GEMM routing and provider dispatch tests. Consulted component reviews for missed-class examples and read package-boundary documentation.

The orchestrator reports fresh Debug and Release runs each passing 1,144 Swift Testing tests and 91 XCTest tests (including 27 skips), 25 CI Python tests, Ruby configuration validation, and seven standalone documentation programs. Those are orchestrator-owned results, not reruns by this reviewer. No expensive benchmark or shared build was run.

I compiled a small standalone probe from the current unmodified benchmarking Models/CaseParsing/JSONWriter/ABComparisons/Correctness sources (removed only their now-unnecessary package imports), with its own module cache. Results:

```text
nanScalar samples 1 maxAbs nan maxRel nan violates false
nanVector samples 1 maxAbs 0.0 maxRel 0.0 violates false
emptyVsNonempty samples 0 maxAbs 0.0 maxRel 0.0 violates false
cosineABPairCount 0
```

Probe source/executable: `tooling-probe/main.swift`, `tooling-probe/run` beside this report. Ran the existing Release executable with an optimized-only dot filter; it ran no timed cases and exited zero with “Correctness violations: 0”. `tooling-probe/filter.json` has an empty results array. Ran the existing Debug executable with the same empty selection and inspected its metadata: it falsely records `release`, `releaseSize: true`, `useCKernels: true`. File: `tooling-probe/debug-meta.json`. Finally, `python3 Benchmarks/TopKNaNContract/compare.py` reproduced all checked-in timing table entries and verified all paired checksums; this is reanalysis of historical samples, not new timing.

## Prioritized actionable findings

### B1 — P2: correctness gating accepts NaN and missing output

**Category:** validation correctness. **Locations:** `Benchmarks/VectorCoreBench/Utils/Correctness.swift:15`, `:21`, `:32`; `Benchmarks/VectorCoreBench/AppMain.swift:577`–583 and `:485`.

A kernel returning NaN against a finite reference yields NaN scalar errors; every `error > threshold` comparison is false. The vector helper initializes maxima to zero and updates only for `ae > maxAbs`/`re > maxRel`, so an all-NaN output reports zero maximum error. It also compares only `min(currents.count, references.count)`, and an empty output against a nonempty expected vector is explicitly returned as zero-error/zero-sample. The standalone probe establishes all three. Strict correctness therefore cannot serve as a trustworthy guard against severe numerical/output regressions. A NaN mean also causes JSON encoding to fail, but the CLI catches the write error, prints it, and can still exit successfully.

**Remedy/test:** represent validation failure separately from finite error statistics; require expected output count; explicitly classify non-finite actual/reference values; reject missing validation in strict mode for cases requiring numerical validation. Test scalar and vector NaN, infinities, both length mismatch directions, empty/nonempty, and exact finite outputs through the CLI exit-code/artifact boundary. Apply the same class-wide policy to approximate equality/precision validators identified by the vector and reduced-precision reviewers. Do not simply set a NaN sentinel in a field that JSON cannot encode.

**Additional coverage gap:** dot and normalization suites never attach correctness, nor do optimized Euclidean 512 and pre-normalized cosine batch helpers (`BatchBench.swift:541`, `:802`, `:839`, `:874` return raw results). The CLI silently skips missing correctness (`:577`). A `--strict-correctness` dot-only run is not actually correctness-tested. Timed batch providers can also fail under `try?` and be replaced by zero while out-of-band correctness reruns only the underlying kernels, bypassing the provider execution being measured.

**Confidence:** executable helper reproduction + direct exit/encoding/control-flow derivation; no injected faulty production kernel was needed.

### B2 — P2: reported build/kernel variants do not describe the executed binary

**Category:** measurement provenance / API contract. **Locations:** `Benchmarks/VectorCoreBench/AppMain.swift:94`–104, `:211`–213, `:467`–474; `Sources/VectorCore/Interop/CKernels.swift:17`; `Sources/VectorCore/Interop/FeatureFlags.swift:5`.

`--use-ckernels`, `--use-underscored`, and `--release-size` are advertised as enabling/building alternate execution modes. Repository-wide reference search shows those CLI fields are only parsed and written to metadata. C dispatch is compiled under `VC_USE_C_KERNELS`, a different, compile-time condition; the CLI cannot change it. Size optimization likewise cannot be selected by an already-built process. Independently, every binary records `buildConfig: "release"`, including `.build/debug/vectorcore-bench` (reproduced). Artifacts can thus claim a release/C/size configuration never executed, invalidating A/B attribution and comparison eligibility.

**Remedy/test:** expose compile-time truth as immutable binary metadata; run separate builds through a driver if build variants are desired; reject incompatible requested flags. Record relevant compile definitions/optimization mode, actual executable identity and dirty source status. Test Debug and Release metadata, and paired separately compiled C-enabled/C-disabled binaries with an observable routing hook. Keep requested options distinct from effective configuration.

**Confidence:** direct code reference inventory and existing Debug executable metadata reproduction.

### B3 — P2: extended benchmark workflow re-enables ungated host-dependent performance assertions

**Category:** CI reliability / claim discipline. **Locations:** `Tests/ComprehensiveTests/MixedPrecisionPerformanceBenchmark.swift:100`, `:101`, `:134`, `:227`; `.github/workflows/benchmarks.yml:33`–37; `Tests/ComprehensiveTests/PerfGate.swift:3`–9.

The manual benchmark workflow explicitly sets `VECTORCORE_TEST_EXTENDED=1` and `VECTORCORE_STRICT_PERF=0`, but these mixed-precision tests assert speedup > 1 and that larger dimensions improve the speedup ratio without consulting `strictPerfGatesEnabled`. The latter ordering is not a mathematical invariant and need not hold even on quiet hardware. These assertions become active in the hosted benchmark job despite its opt-out. Normal Debug/Release passes do not exercise them because the extended gate is off.

The timer bodies also discard scalar results, so their ratios lack a robust guarantee that optimized computation remains observable. This compounds the unsupported speedup assertions; no specific compiler elimination or hosted failure was measured here.

**Remedy/test:** move these measurements to the dedicated benchmark executable, consume results, and apply strict performance policy consistently if such gates are retained. A mechanical test should enumerate/assert the enabled assertion policy for `EXTENDED=1, STRICT_PERF=0`; avoid fixing only one of the four sites. Use controlled-run statistical regression comparisons instead of universal FP16-faster and monotonically-improving-dimension claims.

**Confidence:** exact workflow/test reachability proof; not a claim that the current hosted run failed.

### B4 — P2: filtering optimized dot benchmarks suppresses every requested case

**Category:** tooling correctness. **Locations:** `Benchmarks/VectorCoreBench/Suites/DotProductBench.swift:57`, `:116`, `:160`.

Each dimension returns from its whole function when the generic case does not match. The optimized sibling is reachable only after executing the generic sibling. Reproduction:

```sh
.build/release/vectorcore-bench --suites dot --dims 512,768,1536 \
  --filter '*.optimized' --repeats 1 --strict-correctness --format json \
  --out /private/tmp/vectorcore-review-2026-09-14/tooling-probe/filter.json
```

Exit status is zero; JSON contains zero results and the console reports zero correctness violations. This breaks targeted optimized regressions. `Scripts/compare_benchmarks.swift` ignores missing cases unless `--strict` is given, and its documented baseline command omits that flag, compounding silent coverage loss.

**Remedy/test:** independently register/filter each case rather than nesting execution behind a sibling. Test exact-name selection of every registered case, excludes, and selecting one variant across dimensions. Fail or explicitly require an opt-in for a zero-case run. Validate baseline/current case sets for regression use. This is a repeated three-dimension defect, warranting a registry-wide mechanical selection test.

**Confidence:** existing executable reproduction.

### B5 — P2: cosine fused/pre-normalized A/B comparisons can never be emitted

**Category:** report correctness / duplicated logic. **Locations:** `Sources/VectorCoreBenchmarking/ABComparisons.swift:62`; `Benchmarks/VectorCoreBench/AppMain.swift:341`.

The grouping key includes the complete variant (`optimized-fused` versus `optimized-preNorm`). The code then requires both variants in the same group. They necessarily occupy different groups, making that condition unreachable. The exact same mistake exists independently in JSON A/B generation and the pretty summary. A two-case probe produces zero comparisons.

**Remedy/test:** group by shared variant family/dimension/N/provider, using strategy as the value key, and derive both console and JSON from one comparison model. A table-driven test should require exactly one expected pair for each advertised A/B family and reject mismatched provider/dimension/N pairs. Also fix `Models.swift:308`'s delta formula comment: the implementation is `(right-left)/right`, not the displayed `(left-right)/left`.

**Confidence:** standalone current-source reproduction and direct disjoint-key proof.

### B6 — P2: several benchmark paths omit an observable result and measure different operation boundaries

**Category:** measurement validity / claim discipline. **Locations:** `Benchmarks/VectorCoreBench/Suites/BatchBench.swift:449`, `:610`, `:746`, `:864`, `:887`, `:1016`; `Benchmarks/VectorCoreBench/Suites/MatrixDistanceBench.swift:67`; `Tests/MinimalTests/ComprehensiveBenchmarks.swift:24`; `Tests/ComprehensiveTests/MixedPrecisionPerformanceBenchmark.swift:73`.

Four 1536-dimensional optimized batch paths discard the final reduction, whereas corresponding 512/768 paths consume it with `blackHole`. Numerous XCTest/Swift Testing timing loops similarly discard pure operation results. The matrix path only conditionally sends an empty string to stderr based on one self-distance element after the whole run. These are not equivalent observability contracts and do not establish that all intended work survives optimized compilation. This review has **not** inspected generated assembly or measured dead-code elimination, so the actual magnitude is unverified.

There is also a proved timing-boundary discrepancy: SoA/FP16 comments say building/converting candidates “is part of the benchmark”, but those steps occur before both warmup and measurement. Generic AoS work allocates no result array per chunk, optimized AoS allocates a new array per chunk inside each timed invocation, and SoA/FP16 reuse a preallocated buffer. These results are different end-to-end operations and cannot isolate a layout/kernel speedup without stating allocation/packing boundaries. Comparing such results to calibrate production crossover decisions can select the wrong policy.

**Remedy/test:** define a case descriptor containing the operation, effective route, setup scope, timed allocations, work units and output check. Make every result escape consistently; validate every result out-of-band using an order-sensitive consumer as the standalone Top-K harness already does. Maintain separate prepacked-kernel and end-to-end cases, then compare production-like boundaries. Inspect optimized code for representative scalar/batch/matrix cases and measure an empty consumer baseline before claiming timings. Do not attribute a speedup to SoA/FP16 when only the allocation scope changed.

**Confidence:** missing consumption and boundary differences are source-verified; performance consequences require coordinated measurements.

### B7 — P3: printed memory-footprint figures count array handles, not vector payloads

**Category:** measurement correctness. **Location:** `Tests/ComprehensiveTests/MixedPrecisionPerformanceBenchmark.swift:302`–327.

The footprint test uses `MemoryLayout<Vector512Optimized>.stride` and equivalent FP16/768/1536 strides to report bytes and extrapolate 100,000 vectors. FP32 and FP16 types own `ContiguousArray` storage (`Vector512Optimized.swift:29`; `MixedPrecisionKernels.swift:302`), so struct stride excludes the heap payload. A 512D FP32 vector's scalar payload alone is 2,048 bytes, and FP16 is 1,024, before array headers/alignment/allocator overhead. Reported handle sizes are not memory footprint and do not establish actual savings. In addition, `testVector512OptimizedMemoryFootprint` / generic counterpart in MinimalTests time repeated magnitudes, with no memory metric.

**Remedy/test:** report logical payload separately from allocated bytes/RSS, include retained array metadata and allocator alignment in analytical estimates, and use an allocation/RSS instrument for measured footprint. Rename timing-only magnitude cases. A representation test can verify payload counts/types; runtime allocation checks should address realistic independent versus shared COW datasets.

**Confidence:** direct ownership/layout derivation; no memory profiling run.

## Engineering opportunities and known limits

### Trustworthy coverage for downstream scale

The recent regression suites demonstrate good patterns: Top-K checks heap and sort paths, both API forms and non-finite order, with comparator-law checks; pointer seam tests use explicit Double references and tails; provider tests use impossible sentinels to prove delegation rather than only numerical parity; external MemoryPool probes test real production source under bounded process/executor saturation.

Expand these patterns into a small explicit contract matrix: all vector representations (including 384D and runtime dimensions), each public wrapper, optimized/generic/provider paths, empty/singleton/ragged collections, SIMD/chunk boundaries, finite exponent extremes and non-finites, spare output capacity, and every configuration that affects routing or cache identity. Adjacent reviewers found repeated NaN-equality acceptance, forgotten 384D pointer repair, shape validation holes and calibration/cache-key omissions despite the passing suite. Those are specific evidence that large test counts alone are not class-wide closure. Exact integer/bit-pattern checks in ordering tests are appropriate; approximate numeric values need finite-aware tolerance checks.

`BatchGEMMRoutingTests` uses N=300 and mostly 512D/random moderate inputs; its numerical agreement does not by itself prove every concrete route executed. The provider sentinel pattern is stronger for routing claims. Test every newly reachable route when gates change, including boundary N values and custom metric/provider precedence, before using a performance gate to enable it.

The default benchmark CLI and full hosted benchmark omit `matrix` and `memory`; the PR smoke only runs dot/distance/normalization at 512D. SoA and FP16 are opt-in and not enabled by the workflows. There are no dedicated end-to-end streaming nearest-neighbor, prepacked matrix reuse, chunk scaling, quantization, PCA/UMAP or tail/dynamic-dimension cases in this benchmark executable. N=10,000 full batch cases are useful but do not establish million-vector scaling or memory limits. Add controlled workload tiers and allocation measurements around the concrete downstream seams; keep CI artifacts separate from claimed throughput guarantees.

### Matrix suite ignores common harness controls

`MatrixDistanceBench.swift:18` hardcodes N=64/256/1024, ignores `--batch-ns`, `--samples`, `--repeats`, and run seed, and only gets filtering after execution. Its slash-based names are not understood by CaseParsing, so dim/N/execution metadata are absent. Every vector repeats a 17-value pattern, leaving only 17 distinct vectors. Use the shared case registry/harness with explicit samples, repeat controls, seeds and reusable versus one-shot packing cases. Existing p90 values elsewhere are percentiles of sample-average time, not operation latency percentiles; label them accordingly.

### CI and style enforcement

The required CI aggregate rejects missing, skipped, cancelled and unsuccessful expected jobs; its Python tests cover each state and duplicate JSON. Actions are SHA-pinned, credentials disabled, installer checksummed, and YAML duplication/security policy guarded. Two Swift/Xcode generations and Debug/Release are tested; simulator/Catalyst library compilation and Intel compilation broaden coverage. The seven downstream documentation executables exercise a real public package dependency in Release, valuable protection against `@inlinable` visibility/link failures.

The required aggregate does not include sanitizer/CodeQL workflows or artifact-only perf smoke; local review cannot infer repository branch rules requiring those checks separately. Sanitizer coverage is a narrow named filter, not all unsafe/concurrent code. Add selected newly found class regressions and keep an audited filter inventory. `VectorCoreBenchmarking` has no test-target dependency in Package.swift, so its parsing, grouping, JSON and stats have no dedicated tests; the runtime probes above show why a small test target is warranted.

`.swiftlint.yml:39` includes only Sources/Tests, excluding the benchmark executable from the required lint command. The nested Sources config lists Benchmarks, but that does not add the sibling Benchmarks directory to root selection. Analyzer rules are declared, but CI invokes `lint`, not an analysis step with compiler logs. Current linter results belong to the orchestrator. Lint does not enforce the user's evidence-for-claims rule or catch semantic performance assertions. Establish review checks for claims naming tests/evidence and remove stale “skeleton”/“all fixed” comments in Package/configuration files. Do not force weak-self or inlining merely to satisfy regex suggestions; ownership and module-boundary cost need reasoning.

`Tests/PerformanceTests/NSWOptimizationBenchmark.swift` is unregistered in Package.swift and references removed graph construction APIs; it is not part of the passing suite. It belongs with the owning downstream graph/index package or should be explicitly archived/removed. Its presence should not be counted as active performance coverage.

### Metadata and baseline handling

Checked-in canonical/local baselines contain 2025 measurements of 0.1.0-alpha on a different recorded toolchain/environment. They are historical artifacts, not a present-performance guarantee. `compare_benchmarks.swift` reads only results and silently compares incompatible machine/build/seed runs; validate provenance before using it as a gate. Store raw samples and define an acceptable noise policy; current A/B summaries use aggregate time/iterations rather than the sample median displayed separately. `EnvCapture` identifies HEAD from current working directory and asks the current PATH's swiftc for its version, which can differ from the executable's build provenance. CSV metadata does not quote arbitrary labels/newlines or multiline swiftc output even though data rows have a quoting helper.

Public `VectorCoreBenchmarking` environment capture uses macOS Process/Host facilities without a mobile guard; CI's Apple platform loop builds only `VectorCore`. Mobile availability of this separate product was not compiled here and should be explicitly scoped/documented or validated if intended to ship for all package platforms.

## Strengths / rejected suspicions

- The shared harness uses uptime nanoseconds on supported macOS and amortizes clock overhead across repeat loops; warmup, deterministic seeds, medians, nearest-rank p90 and sample standard deviation are implemented coherently. No arithmetic defect found in the ordinary stats computation.
- Main distance benchmarks compare with independent Double reference loops outside the timed region; this is useful once non-finite/shape failure handling is repaired.
- The standalone Top-K harness is the best measurement example: Release instructions, fixed fixtures, warmups, full output checksums outside timing, raw samples, paired baseline/final sequence, exact hardware/toolchain information and candid power/statistical limits. Reanalysis reproduced its record.
- Normal CI deliberately has no wall-clock speed threshold. That is sensible on shared hosted runners; the defect is the extended tests bypassing the opt-out, not the absence of a universal performance gate.
- Documentation consumers are intentionally selected, not advertised as every sibling-package build. `Docs/Package_Boundaries.md` explicitly states there is no cross-package compatibility matrix; lack of VectorIndex/VectorAccelerate end-to-end coverage is a documented limit and future integration opportunity, not a newly discovered false CI claim.
- Benchmark BlackHole/Filters have unchecked mutable global state, but this CLI runs suites/cases sequentially and places blackHole after awaited reductions. No concurrent benchmark-sink race is demonstrated in the current caller structure.

## Verification limits

No hosted CI/security settings inspection, Xcode cross-platform compilation, sanitizer rerun, assembly inspection, thermal-controlled timing, allocation measurement or sibling-package build by this reviewer. No claims of observed speed regressions or speedups beyond reproducing the historical Top-K report. Normal passing tests do not include extended performance assertions. The benchmark probes intentionally ran zero expensive cases; all temporary artifacts are outside the repository. Core numerical correctness findings remain in their component reports; this report identifies why existing tooling can overlook or misreport some of them.
