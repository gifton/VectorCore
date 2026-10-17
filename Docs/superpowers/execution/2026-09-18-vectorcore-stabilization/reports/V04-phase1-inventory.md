# V04 phase 1: path and case inventory, and implementation design

Returned to root before any editing of benchmark companions, CLI selection or boundary metadata, as `briefs/V04.md` and `V04/phase1-addendum.md` require.

## 0. Basis, method and limits

- **Requirements.** The binding brief is `/private/tmp/vectorcore-stabilization-2026-09-18/Docs/superpowers/execution/2026-09-18-vectorcore-stabilization/briefs/V04.md`. It is read together with `phase1-addendum.md`, `reports/V04-design-inventory.md`, `reports/V04-design-review.md` (root's DR1 resolution), WORK_ITEMS V04, and the canonical records TOOL-07/08, API-43, NUM-23 and OWN-23. The closures read were V01, V02 and V03, plus L01, L02, Q02, Q04, O02, O03 and O05.
- **Workspace.** `.../2026-10-03/workspaces/V04` is at `dbd0833044046213e6fb344db756db5a4b640bfc`. `git status --porcelain` was empty.
- **Method.**
  - Read-only source reading and `grep`.
  - Two first-pass lexical scans (vocabularies v0 and v1), run in the session scratchpad outside the workspace.
  - SDK header reads.
  - No source edits, git index operations or subagents. No builds, test runs or measurements were performed.
- **Status of the numbers in this document.** Every "expected" value below (strides, allocation counts, eligibility results, internal paths) is **source-derived**. It is a RED/GREEN target for implementation, not an observation. **No performance claim is made.**
- **Prerequisites.**
  - V01, V02 and V03 are accepted.
  - L01, L02, Q02, Q04, O02, O03 and O05 are all accepted (`closures/*.json`, `state.json`), so final claim reconciliation is no longer blocked on prerequisites.
  - N01 is "accepted-staged". Its Vector512 magnitude wording (`Vector512Optimized.swift:207-211`) is preserved.
  - L01 and L02 already rewrote the PCA and SwiftLinearAlgebraProvider headers that the design inventory flagged.
  - Q02 already rewrote the FP16VectorPool header.

Source bytes this inventory is anchored to (SHA-256, first 16 hex characters):

| Path | sha256 | Path | sha256 |
|---|---|---|---|
| Sources/VectorCoreBenchmarking/Models.swift | 4ac02dee45932315 | Benchmarks/VectorCoreBench/Suites/BatchBench.swift | c4f40e00b1264ec7 |
| Sources/VectorCoreBenchmarking/ABComparisons.swift | d5fc8c12c5bfa231 | Benchmarks/VectorCoreBench/Suites/MatrixDistanceBench.swift | 6ed4ab59f4b009c3 |
| Sources/VectorCoreBenchmarking/JSONWriter.swift | 1887ba4b3cfaeb44 | Scripts/compare_benchmarks.swift | 1131122e5ed53896 |
| Sources/VectorCoreBenchmarking/CaseParsing.swift | 98707d009ea3d6e4 | Tests/ComprehensiveTests/MixedPrecisionPerformanceBenchmark.swift | a8997dcfe688412d |
| Sources/VectorCoreBenchmarking/EnvCapture.swift | 603d9fd44007d1a3 | Tests/MinimalTests/ComprehensiveBenchmarks.swift | 980fe26461611d11 |
| Benchmarks/VectorCoreBench/AppMain.swift | 4b57fdf23b4e8aad | Scripts/ci/performance_sites.json | 1b0b4535852b3598 |
| Benchmarks/VectorCoreBench/ValidatedOptions.swift | df89c48f84ca0e77 | Sources/VectorCore/Vectors/Vector512Optimized.swift | 96f8b02f662ec96c |
| Benchmarks/VectorCoreBench/Core/BenchmarkCase.swift | 5e0ecdead34e413d | Sources/VectorCore/Storage/COWDynamicStorage.swift | 9684f474fd774063 |
| Benchmarks/VectorCoreBench/Core/BenchmarkHarness.swift | 6a8a9da548bbf5bc | Sources/VectorCore/Optimization/OptimizationAttributes.swift | 311a7151aeabdcfa |
| Benchmarks/VectorCoreBench/Formatters/CSVWriter.swift | a5687494acf56710 | Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift | 4af3d208472e7b23 |
| Benchmarks/VectorCoreBench/Utils/BlackHole.swift | 4c10efbc07135f2f | Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift | 8a6f3a0a5bcd6937 |
| Sources/VectorCore/LinearAlgebra/PCA.swift | dc7dffa7ce827ac9 | Sources/VectorCore/LinearAlgebra/SwiftLinearAlgebraProvider.swift | 15d85d858bfd15ea |
| Sources/VectorCore/Vectors/DynamicVector.swift | 6eceda295bd18ae5 | Scripts/ci/tests/O03Allocation.c | 4f21209620aa0e15 |

The workspace bytes match V03's frozen hashes for BatchBench, AppMain, Harness, CSVWriter, the two test files and `performance_sites.json`. MatrixDistanceBench was last changed by T01 (`e6e1bae`). The design inventory's `out[0].isNaN` matrix observation predates V02 and T01. The current consumer is `blackHole(output[0]); blackHole(output[pairs - 1])` (section 4).

---

## 1. Boundary-metadata design

### 1.1 Files and types to extend

| Path | Type / symbol | Change |
|---|---|---|
| `Sources/VectorCoreBenchmarking/MeasurementBoundary.swift` (**new**) | `MeasurementBoundary`, `TimingScope`, `Inclusion`, `BoundaryInclusions`, `FixtureIdentity`, `ComparisonMembership`, `UnavailableCase` | New Codable/Sendable value types (1.2). `canonicalJSON()` (sorted keys) supplies the identity string used for comparison and for the CSV cell. |
| `Sources/VectorCoreBenchmarking/Models.swift` | `BenchMetadata` | `schemaVersion` 2→3. `measurementBoundaryVersion` "V02-boundaries-1"→"V04-boundaries-1". `fixtureVersion` stays "V02-inputs-1" (1.4). Add `unavailableCaseIDs`, `statisticUnit`, and `thermalStateStart`/`thermalStateEnd`/`lowPowerMode` as `RuntimeObservation`. Keep every existing key that `check_benchmark_build_modes.py` reads (`selectedCaseIDs`, `flags.useCKernels`, `flags.requestedCKernels`, `core/benchmarkConfiguration`), because that driver is hash-pinned by `validate_github.rb:179` and must keep working unedited. |
| same | `BenchResult` | Add `boundary` (optional; set only by `CaseExecution`), `sampleNsPerInvocation: [Double]`, `sampleIterations: [Int]`, `warmupIterations: Int?` and `unavailableReason: String?`. Copy helpers `withCorrectness`/`withBoundary` preserve all fields. |
| same | `BenchCase` | Add `boundary` (required in v3), the raw-sample arrays, `warmupIterations` and `statisticUnit`. Name-derived `params`/`execution` are kept as legacy display hints and documented as non-authoritative. |
| same | `ABComparison` | Add `family`, `purpose`, `allowedDifferences`, `leftBoundary` and `rightBoundary`. Keep `leftPreparation`/`rightPreparation`, now derived from the boundaries; the V02 test string "normalization outside timing" is preserved. |
| same | `BenchmarkFailure`, `BenchRun` | `boundary` is optional on a failure (present for case failures and violations). `BenchRun.unavailable: [UnavailableCase]`. |
| `Sources/VectorCoreBenchmarking/ABComparisons.swift` | `ABComparisons` | Pairing comes **only** from catalog-declared `boundary.comparison` (1.5), using a family table of allowed differences. Any cross-scope pair or any difference that is not allowed is a report-integrity error. `missingSides(boundaries:)` serves pre-execution planning (`--ab-only`). Case-name parsing no longer forms pairs. |
| `Sources/VectorCoreBenchmarking/JSONWriter.swift` | `toCases` | Carries the boundary and raw samples. Throws an integrity error when a result has no boundary. |
| `Sources/VectorCoreBenchmarking/EnvCapture.swift` | `collectMetadata` | Records `ProcessInfo.thermalState` and `isLowPowerModeEnabled`, or explicit unavailable reasons. |
| `Benchmarks/VectorCoreBench/Core/BenchmarkCase.swift` | `BenchmarkCase`, `CaseExecution.run` | Signature becomes `BenchmarkCase(name, boundary:, availability:, operation)`. For catalog cases, `CaseExecution` attaches the **catalog** boundary to every result and treats a result whose name differs from its case as an execution failure. Unavailable cases produce a record without invoking `operation`. **Exemption:** the default adapter for injected V01 fixture suites (`RunnerContractTests.FixtureSuite`) returns results named differently from its single case "fixture". It accepts any result name. It keeps a boundary that an inner catalog already attached (the `batch-return-*` scenarios run the real `BatchBench` catalog). Otherwise it stamps `timingScope: fixture-injected`, which never pairs. The V01 process tables therefore keep their exit contracts. |
| `Benchmarks/VectorCoreBench/Core/BenchmarkHarness.swift` | `Harness.measure/measureAsync/warmup/warmupAsync` | Keep per-sample (iterations, ns) pairs. Warmup iteration counts go to a task-local recorder. Phase hooks sit outside the clock reads (1.9). |
| `Benchmarks/VectorCoreBench/Core/MeasurementPhases.swift` (**new**) | `MeasurementPhase`, `MeasurementPhaseObserver`, `MeasurementPhases.$observer`, `CaseRecorder` | Task-local seam (1.9). `public` only inside the benchmark-support module, for the allocation probe (Q9). |
| `Benchmarks/VectorCoreBench/Core/Boundaries.swift` (**new**) | `Boundaries` | One factory function per case family. Every catalog calls it, so it is the single source of each record. |
| `Benchmarks/VectorCoreBench/Core/BatchFixtures.swift` (**new**) | FP32 fixture builders | Same seeds as today (`InputSeeds.batch`, `InputFactory.randomArray`). Used by BatchBench and the new suite, so the "same immutable FP32 fixture" is one function. |
| `Benchmarks/VectorCoreBench/Utils/OutputFold.swift` (**new**) | `OutputFold.fold` | Consumer (section 4). |
| `Benchmarks/VectorCoreBench/Utils/BlackHole.swift` | comments only | Observes value-representation bytes only, is unsynchronized, and is for the serial driver only (6.2). |
| `Benchmarks/VectorCoreBench/Formatters/CSVWriter.swift` | `write` | New columns and an `unavailable` section (1.3). |
| `Benchmarks/VectorCoreBench/AppMain.swift` | `BenchmarkRunner.run`, `printPretty`, `printUsage`, `evaluateCorrectnessWithPolicy`, `reportFailure` | Registry and allowed suites gain `boundary`. Unavailable results are partitioned out. `--ab-only` planning uses boundaries. Pretty output shows the scope and both boundaries of each AB pair. |
| `Benchmarks/VectorCoreBench/ValidatedOptions.swift` | `init` | `boundary` joins the allowed suites with the batch width rule {512, 768, 1536}. Its matrix consumer-only entries get checked N·D·4, N·N·4 and Int32 limits. |
| `Benchmarks/VectorCoreBench/Suites/{Dot,Distance,Normalization,Batch,MatrixDistance,Memory,Noop}Bench.swift` | catalogs | Each `BenchmarkCase` takes its record from `Boundaries`. MatrixDistanceBench gets the consumer change (section 4). BatchBench gets comment corrections (6.2) and the shared fixture builder. The 14 provider run helpers are untouched, so the V01 guard counts (28 provider calls, 14 boundaries, 28 observes) are unchanged. |
| `Benchmarks/VectorCoreBench/Suites/BoundaryBench.swift` (**new**) | `BoundaryBench` | Section 3. |
| `Scripts/compare_benchmarks.swift` | `Run`, `mismatches` | Per-case boundary identity (1.6). |

`CaseParsing.swift` needs no change: every new ID fits the existing `kind.metric.D.N<n>.variant` grammar.

### 1.2 Per-case record (JSON, schema 3)

```json
"boundary": {
  "version": "V04-boundary-1",
  "timingScope": "reused-input",
  "operation": "batch.squared-euclidean",
  "route": "CPUComputeProvider.parallelReduce + BatchKernels.range_euclid2_512(query:candidates:range:out:)",
  "provider": "sequential",
  "workUnit": "candidate",
  "unitsPerInvocation": 1000,
  "inputRepresentation": "aos-fp32:[Vector512Optimized]",
  "outputRepresentation": "provider-range-partials:Float",
  "fixture": {"id": "V02-batch-lcg", "version": "V02-inputs-1", "dimension": 512, "count": 1000,
              "querySeed": "<decimal UInt64>", "candidateBaseSeed": "<decimal UInt64>",
              "values": "uniform[-1,1) LCG", "preparation": "none"},
  "inclusions": {"inputConstruction": "excluded", "inputPreparation": "notApplicable",
                 "routeInternalPacking": "notApplicable", "callerOutputAllocation": "notApplicable",
                 "routeInternalAllocation": "included", "providerScheduling": "included",
                 "providerObservation": "included", "kernelWork": "included",
                 "outputConsumption": "included", "correctnessValidation": "excluded"},
  "consumer": "provider-float-sum-observe-sink-v1",
  "notes": ["per-range [Float](range.count) scratch allocated inside timing"],
  "comparison": {"family": "euclidean2_vs_euclidean", "side": "left", "group": "512|1000|optimized|sequential"},
  "defaultEligible": null
}
```

Field definitions:

- **`timingScope`:** `reused-input` (prepared/reused inputs; the brief's "prepared/reused-input"), `setup-inclusive`, or `consumer-only`. `fixture-injected` is reserved for the V01 test adapter.
- **`inclusions`:** each flag is `included`, `excluded` or `notApplicable`. They cover everything the brief lists:
  - packing/conversion, two flags:
    - `inputPreparation` is preparation the route needs that the **benchmark body** performs (SoA transposition, FP16 conversion, preNorm normalization). It is `excluded` when done before timing and `included` when done inside timing.
    - `routeInternalPacking` is packing or transposition performed **inside the public API call**, such as matrix `packRows` or the SoA convenience API's transposition.
    - Each preparation step is therefore counted under exactly one of the two flags.
  - output allocation: `callerOutputAllocation`, plus `routeInternalAllocation`, which covers provider, API and body intermediates and is recorded separately as the brief requires;
  - provider scheduling and observation: `providerScheduling`, `providerObservation`;
  - kernel work: `kernelWork`;
  - output consumption: `outputConsumption`;
  - correctness validation: `correctnessValidation`, always `excluded`, because validation runs out of band.
- **`consumer` IDs:**
  - `scalar-sink-v1`: blackHole of a scalar result.
  - `provider-float-sum-observe-sink-v1`: in-range Float sum, then `combine`, then `ProviderObservations.observe`, then blackHole.
  - `float-sum-all-outputs-sink-v1`: Float sum over all N outputs, then blackHole.
  - `u64-bitfold-sink-v1`: section 4.
  - `first-coordinate-sink-v1`, `branch-constant-sink-v1`, `single-element-sink-v1`: partial observations, labelled as partial.
- **Correctness method** (vector, sampled, scalar, or provider-only) is **not** part of the boundary. It stays in the result's `correctness` record. Validation is out of band for every case, and including the method would break pairing for reasons unrelated to timed work (see 2.4, row d).
- **`defaultEligible`** is present only on boundary-suite entries (Q1).

### 1.3 Report surfaces

- **JSON**
  - `results[].boundary`, `sampleNsPerInvocation`, `sampleIterations`, `warmupIterations` and `statisticUnit`.
  - `abComparisons[].family/purpose/allowedDifferences/leftBoundary/rightBoundary`.
  - `failures[].boundary`, which is optional.
  - `unavailable[]`, with `{name, boundary, reason}`.
  - `metadata.unavailableCaseIDs`, `statisticUnit`, and the thermal and low-power observations.
- **Pretty**
  - Each result line appends `scope=… unit=… consumer=…`.
  - When there is more than one sample, the raw per-sample values are printed.
  - A header states the statistic: "per-sample average ns per invocation".
  - Each AB pair prints family, purpose and delta, then both boundaries as canonical JSON lines.
  - `UNAVAILABLE <id>: <reason>` lines.
- **CSV** (data columns appended after `suspicious`):
  - `boundary_version, timing_scope, operation, route, provider_mode, work_unit, units_per_invocation, input_representation, output_representation, fixture_id, fixture_preparation`;
  - `incl_input_construction, incl_input_preparation, incl_route_internal_packing, incl_caller_output_allocation, incl_route_internal_allocation, incl_provider_scheduling, incl_provider_observation, incl_kernel_work, incl_output_consumption, incl_correctness_validation`;
  - `consumer, comparison_family, comparison_side, statistic_unit, sample_ns_per_invocation (;-joined), sample_iterations (;-joined), warmup_iterations, boundary_json` (canonical JSON, escaped by the existing V02 rules).
  - A new `unavailable_name,unavailable_reason,timing_scope,route` section sits between measurements and failures. The V02 independent CSV parser reads columns by header, so it stays compatible.

### 1.4 Schema and run-level identity

- Schema version 3.
- `measurementBoundaryVersion` becomes "V04-boundaries-1", because per-case identity is new and the measured work of the matrix case changes (section 4).
- `fixtureVersion` stays "V02-inputs-1": no generator or seed changes, and per-case `fixture.version` references it. A bump is mechanical if root prefers one (Q19).
- The comparator's known values become schemaVersion {"3"} and measurementBoundaryVersion {"V04-boundaries-1"}.

### 1.5 Within-run pairing (V02 families preserved; DR1 resolution)

| Family | Left / right | Must be equal | Allowed to differ | Purpose (printed and stored) |
|---|---|---|---|---|
| `euclidean2_vs_euclidean` | euclidean2 / euclidean | timingScope; provider; workUnit; unitsPerInvocation; input and output representation; every fixture field; consumer; notes; all inclusions | `operation`, `route` | "Compares inclusion of sqrt. The outputs differ numerically (squared versus unsquared distance); inputs, provider and consumer are identical." |
| `cosine_prenorm_vs_fused` | preNorm / fused | timingScope; provider; workUnit; units; output representation; consumer; notes; fixture `id`/`version`/`dimension`/`count`/seeds; all inclusions except `inputPreparation` | `operation`, `route`, `fixture.preparation` ("l2-normalized-outside-timing" vs "none"), `inclusions.inputPreparation` ("excluded" vs "notApplicable") | "Prepared-input strategy comparison. preNorm normalization is excluded from timing; fused norms are computed inside the kernel. This is not an end-to-end strategy-speed claim." |

- **Group key:** (family, D, N, provider, variant family generic/optimized).
- **Rules:**
  - A group needs exactly one left side and one right side.
  - A duplicate side is an error.
  - A declared side whose counterpart is unselected is a missing side; `--ab-only` rejects it, as in V02.
  - Members whose `timingScope` differs are an integrity error. They are never silently skipped.
- **Counts:** with the quick preset and `--ab-only`, the counts stay at 36 + 18.
- **New companions declare no comparisons** (3.5).
- **Tests:**
  - For each family, every field outside the allowed set is mutated on one side, and each mutation must throw.
  - Each allowed field is mutated, and the pair must still form.
  - A fabricated declaration with a different scope must throw.
  - Undeclared cases whose names match the old pattern must not pair. This proves that pairing is no longer inherited from names.

### 1.6 Cross-run compatibility and the legacy override policy

- For each matched case name, both sides must carry a boundary and the canonical JSON strings must be equal.
- A difference prints `Boundary mismatch: <case>: <field paths>` and counts as a metadata incompatibility:
  - exit 2 unless `--allow-incompatible` is given;
  - with the override, each waived case is printed after `OVERRIDE`.
- A missing boundary, such as a legacy schema-2 report or a historical baseline, is "missing boundary identity". It is accepted only under the explicit override, never by default.
- **Not waivable:**
  - a schema-3 report with a missing or malformed boundary;
  - duplicate IDs;
  - numeric validity, failures and units, as in V02.
- `Benchmarks/baselines/*.json` are not rewritten.
- Whether a `timingScope` mismatch should be non-waivable is Q7.

### 1.7 Raw samples and the statistic unit

- The multi-sample harness path keeps each sample's (iterations, total ns). For a single sample there is one element.
- `statisticUnit` = "per-sample-average-ns-per-invocation". One invocation processes `unitsPerInvocation` work units.
- The existing `Stats.compute` definitions are unchanged: the median of per-sample averages, sample stddev, RSD and nearest-rank p90.
- Warmup iteration counts are recorded through the task-local `CaseRecorder`, with no edits to suite bodies.

### 1.8 Unavailable records

- `availability: .unavailable(reason)` entries are selected by filters like any other entry.
- They produce an `unavailable` record and are never invoked, timed or validated.
- They are excluded from results, failures and the correctness policy, and are never counted as tested or substituted.
- **Exit status:**
  - If a selection contains no available case, the run exits 1 with "no executable selected cases". This preserves V02's empty-selection failure.
  - Otherwise, unavailable entries do not change the exit status (Q8).

### 1.9 Sink confinement and phase seam

- **Sink**
  - `blackHole` stays one unsynchronized global sink, called only by the serial measurement driver. Every current call site is after `parallelReduce` returns or in single-threaded bodies.
  - A bounded lexical guard in the V04 Python tests checks that no `blackHole(` appears inside a `{ range in` or `{ @Sendable range in` … `} _:` region under `Benchmarks/`. It is documented as a lexical check, not dataflow.
  - If concurrent consumption is ever needed, the work returns to root, as the brief requires.
- **Phase seam (`MeasurementPhases.$observer`)**
  - `CaseExecution` begins `.setup` at case start.
  - Harness warmup functions bracket `.warmup`.
  - `.timed` begins immediately before each sample's opening `now()` and ends immediately after its closing `now()`, so the task-local lookup is outside every clock interval.
  - `.validation` runs from the last measure return until the operation returns.
  - A nil observer leaves behavior unchanged. Only the allocation probe sets it (5.2).

---

## 2. Case catalog: proposed boundary record for every existing case

Legend for inclusion flags:

| Code | Flag |
|---|---|
| IC | inputConstruction |
| IP | inputPreparation |
| RP | routeInternalPacking |
| OA | callerOutputAllocation |
| RA | routeInternalAllocation |
| PS | providerScheduling |
| PO | providerObservation |
| K | kernelWork |
| C | outputConsumption |
| V | correctnessValidation |

Values are `in`, `ex` and `na`. Counts are for the default quick preset: D ∈ {512, 768, 1536}, N ∈ {100, 1000}. Every existing case is `reused-input`. Source anchors refer to the hashed files above.

### 2.1 Scalar suites

| Case family (count) | Operation / route (timed) | Unit | Input → output | IC IP RP OA RA PS PO K C V | Consumer | Validation (out of band) |
|---|---|---|---|---|---|---|
| `dot.D.{generic,optimized}` (6) | `scalar.dot`; `Vector<DimD>.dotProduct` / `VectorDOptimized.dotProduct` inside `withExtendedLifetime` | invocation (1) | fp32 pair → scalar Float | ex na na na na na na in in na | scalar-sink-v1 | **none.** The current policy reports "Missing validation" (unchanged). |
| `dist.{euclidean,cosine,manhattan,dot,chebyshev,hamming,minkowski}.D.{generic,optimized}` (42) | `scalar.distance.<metric>`; `<Metric>Distance().distance(_:_:)` (Minkowski p=3) | invocation | fp32 pair → scalar Float | ex na na na na na na in in ex | scalar-sink-v1 | scalar against `DoubleRef` |
| `normalize.success.D.{generic,optimized}` (6) | `vector.normalize`; `normalized().get()` on constant 2.0 | invocation | vector → new vector | ex na na na in (result storage) na na in **partial** na | first-coordinate-sink-v1 (`unit[0]` only) | none |
| `normalize.zeroFail.D.{generic,optimized}` (6) | same, on a zero vector | invocation | vector → error | ex na na na na na na in branch-only na | branch-constant-sink-v1 | none |

### 2.2 `batch` suite: provider routes

There are 108 cases: 3 D × 2 N × 3 providers × 6 families. Provider p ∈ {sequential, parallel, automatic}, plus `injected` in tests only. Every provider case has unit **candidate** (N) and output `provider-range-partials:Float`, so OA is `na` and RA is `in` (scheduler partials, and child-task machinery in parallel modes). Flags common to every row: IC ex, PS in, PO in, K in, C in, V ex, consumer `provider-float-sum-observe-sink-v1`.

| Row | Case family | Route | IP | RA notes | Comparison | Validation |
|---|---|---|---|---|---|---|
| a | `batch.euclidean2.D.Nn.generic.p` | `Vector<DimD>.euclideanDistanceSquared(to:)` per candidate | na | scheduler only | e2-vs-e left (generic) | observations + vector |
| b | `batch.euclidean.D.Nn.generic.p` [ab] | `EuclideanDistance().distance` | na | scheduler only | e2-vs-e right | observations + vector |
| c | `batch.euclidean2.D.Nn.optimized.p` | `BatchKernels.range_euclid2_D` into a per-range scratch buffer, then summed | na | + `[Float](range.count)` scratch per range (BatchBench:392 et seq.) | e2-vs-e left (optimized) | observations + vector |
| d | `batch.euclidean.D.Nn.optimized.p` [ab] | `range_euclid_D` (euclid2 + `applySqrt`) | na | + scratch | e2-vs-e right | **512: provider observations only** (BatchBench:602). 768/1536: observations + vector |
| e | `batch.cosine.D.Nn.optimized-fused.p` | `range_cosine_fused_D` (norms inside the kernel) | na | + scratch | cos right | observations + vector |
| f | `batch.cosine.D.Nn.optimized-preNorm.p` | `range_cosine_preNorm_D` on vectors normalized in the case closure before the run helper | **ex** (`fixture.preparation` = l2-normalized-outside-timing; zero vectors fall back to `?? query`) | + scratch | cos left | **provider observations only**, all three widths (BatchBench:910/955/1000) |

### 2.3 `batch` suite: optional reused-output routes

These are included only when the flag and the existing eligibility both hold. Common flags: IC ex, RP na, RA na (expected; to be measured by the 5.2 probe), PS na, PO na, K in, C in, V ex (vector).

| Case family | Gate | Route (timed) | Outside timing | Output | IP | OA | Consumer |
|---|---|---|---|---|---|---|---|
| `batch.euclidean2.D.Nn.optimized-soa` | `--prefer-soa` and `shouldUseSoA` (N ≥ 100, D ≥ 512) | `BatchKernels_SoA.euclid2_D(query:soa:out:)` | `SoA<VectorDOptimized>.build(from:)`. The comment "this cost is part of the benchmark" is **false**; 6.2. | scores:Float32[N], caller-owned reusable `UnsafeMutableBufferPointer` | ex | ex | float-sum-all-outputs-sink-v1 |
| `batch.{euclidean2,cosine}.D.Nn.optimized-fp16` | `--use-mixed-precision` and `shouldUseMixedPrecision` (N·D·4 > 12 MiB, N ≥ 100) | `MixedPrecisionKernels.range_{euclid2,cosine}_mixed_D(query:candidatesFP16:range:out:)`; query stays FP32 | `convertToFP16_D(candidates)`. The comment claiming the opposite is **false**. | caller-owned reusable buffer | ex | ex | float-sum |

For the FP16 rows, validation compares against a Double reference computed from the FP32 inputs, so it measures conversion plus accumulation error on that fixture.

### 2.4 `matrix`, `memory`, `noop` and the test adapter

**`matrix.euclidean2.D.Nn.optimized` (6).**

- Operation `matrix.squared-euclidean.public-api-call`; route `MatrixDistance.euclideanSquaredMatrix(queries:candidates:into:)` with queries == candidates.
- Internally the route runs, on every call:
  - `prepare`: `packRows` and `rowSumOfSquares`;
  - `packRows(queries)` and the query norms;
  - `cblas_sgemm`;
  - the clamp pass.
- Unit: **produced-entry**, N² (checked).
- Flags: IC ex, IP na, **RP in**, OA ex (caller `[Float]` of N², reused), **RA in** (source-derived: four internal Float arrays per call; to be measured), PS na, PO na, K in, C in, V ex (sampled pairs: all pairs for N ≤ 8, 25 pairs otherwise).
- Consumer:
  - today: two diagonal elements, `diagonal-pair-sink-v0`;
  - V04: `u64-bitfold-sink-v1` over all N² entries.
- This is labelled an API-call measurement, never a prepacked GEMM kernel.

**`memory` suite (12 cases).**

- **`mem.alloc.aligned.D`**
  - Operation `memory.aligned-allocate-init8-release`.
  - Unit: invocation.
  - The allocation is the operation, so OA in and RA in.
  - Consumer: `pointer[0]`, `single-element-sink-v1`.
- **`mem.copy.D`**
  - Source and destination buffers of D·1024 floats are allocated outside timing.
  - Route `alignedCopy`; unit: element (D·1024).
  - Consumer: `destination[1]`.
- **`mem.fill.D`**
  - Scalar fill loop; unit: element.
  - Consumer: `destination[count-1]`.
- **`mem.pool.acquire.D`**
  - Route `MemoryPool.shared.acquire`, initialize 1, deinitialize, and release the handle at scope exit.
  - Unit: invocation. The global pool is warmed and quiesced beforehand.
  - Consumer: `pointer[0]`.

None of these has validation. Work retention depends on escaped pointers, not on the consumer. The suite is not in the inspection set.

**`noop.scalar-add` (1).** This is a control. Its loop over constants is foldable at compile time (source-derived), so it measures driver and sink overhead. Consumer `scalar-sink-v1`; no validation.

**V01 injected fixture suites (tests only).** `fixture-injected`; never compared.

### 2.5 Findings for root

These are source-established and do not change the brief.

1. `normalize.success` consumes only `unit[0]`, and `zeroFail` consumes a branch constant (Q6).
2. Four provider helpers have no out-of-band per-candidate check: `euclidean.512.optimized` and `cosine.preNorm` at all three widths. This is V01 territory and is only recorded here. It is also why the correctness method must stay outside the pairing identity.
3. The two "this cost is part of the benchmark" comments appear 9 times and contradict the code.

---

## 3. Setup-inclusive companions

### 3.1 Routes

**Requested metric: squared Euclidean.** It is the representative existing batch metric and the only one with existing benchmark routes in AoS FP32, SoA FP32 and AoS FP16.

**Fixture:** an immutable FP32 query plus N candidates as `[VectorDOptimized]`, built outside timing with the V02 seeds by the shared `BatchFixtures` builder. These are the same objects that the existing reused-input cases measure. Independent Double references are computed from them outside timing.

| Route | 512 | 768 | 1536 | Reused-output counterpart (timed body) | Setup-inclusive companion (timed body) |
|---|---|---|---|---|---|
| Optimized AoS FP32 | supported | supported | supported | `BatchKernels.range_euclid2_D(query:candidates:range: 0..<N, out:)` into a caller-owned reusable buffer (Q4) | `[Float](repeating: 0, count: N)`, then the same kernel |
| SoA FP32 | supported | supported | supported | `BatchKernels_SoA.euclid2_D(query:soa:out:)`; the SoA is built outside timing with `SoA<VectorDOptimized>.build(from:)` | `BatchKernels_SoA.batchEuclideanSquared{512,768,1536}(query:candidates:) -> [Float]`, the public route that builds the SoA, allocates the output and calls `euclid2_D` with its query-lane `precondition` |
| AoS FP16 | supported | supported | supported | `MixedPrecisionKernels.range_euclid2_mixed_D(query:candidatesFP16:range:out:)`; conversion outside timing with `convertToFP16_D` | `convertToFP16_D(candidates)`, then `[Float](repeating: 0, count: N)`, then `range_euclid2_mixed_D`. Query preparation: none (the route takes an FP32 query). |
| SoA FP16 | **unavailable** | **unavailable** | **unavailable** | none | none |

**Why SoA FP16 is unavailable.** There is no squared-Euclidean route for an FP32 query against SoA-FP16 candidates.

- The public SoA-FP16 Euclidean kernels are `MixedPrecisionKernels.batchEuclidean{512,768,1536}` and `batchEuclideanBlocked{512,768,1536}`, each with an FP32-query and an FP16-query overload (12 overloads in total), prepared with `createSoA{D}FP16(from:)`. All 12 write **sqrt** distances:
  - `sqrt(accumulators[i])` at `MixedPrecisionKernels.swift` lines 2004, 2092, 2200, 2308, 2396 and 2504;
  - `sqrt(acc1/acc2[i])` in every blocked overload between lines 2865 and 3357;
  - the `batchEuclideanSoA` doc at L3369 says "Returns the actual Euclidean distance (i.e. with `sqrt`)".
- The SoA-FP16 dot route is `batchDotProductSoA`.
- No SoA-FP16 squared-Euclidean or cosine route exists.
- The record says this and is never substituted (Q2).

**Default eligibility at the representative fixtures** (source arithmetic; Q1):

- `shouldUseSoA(N, D)` = N ≥ 100 and D ≥ 512, so it is true at N = 257 and N = 4096.
- `shouldUseMixedPrecision(N, D)` = N·D·4 > 12 MiB and N ≥ 100:
  - 512D: false at 257 and at 4096 (FP16 needs N ≥ 6145).
  - 768D: false at 4096 (exactly 12 MiB; needs N ≥ 4097).
  - 1536D: true at 4096, false at 257.

**Internal SoA path** (source arithmetic; for the inspection record): the kernel tiles when count·lanes·16 B exceeds the 8 MiB threshold.

- 512D at N = 257 or 4096: 4096 × 128 × 16 B = 8 MiB, which is not above the threshold, so the 4-way path `euclid2_blocked_4way` runs. It calls `prefetch(_:offset:)`.
- 768D at N = 4096: above the threshold, so the tiled path runs.

### 3.2 Selection mechanism (recommendation: a separate documented suite, `--suites boundary`)

How it works:

- The suite is not in the default list `["dot","distance","normalize","batch"]`.
- `--ab-only` forces `batch` and so never selects it.
- `--prefer-soa` and `--use-mixed-precision` do not affect it; they stay default-batch eligibility controls, and this is documented.
- Exact filters select entries, and selection happens before setup.

Why a separate suite rather than a `--timing-scope reused|setup-inclusive|all` option:

1. The V02 default catalogs, case IDs, the 36/18 counts, `--ab-only` and the V01 source guards all stay unaffected.
2. There is no interaction matrix with the prefer-soa and mixed-precision flags or their gates.
3. Counterparts and companions share one catalog function and one fixture builder, so "the case catalog used for execution supplies this record".
4. Unavailable routes stay local to the suite.

The cost is a second reused-shape SoA/FP16 measurement: the default `optimized-soa`/`optimized-fp16` cases keep their Float-sum consumer, which is a different consumer identity (Q5). The alternative would thread a new option through BatchBench's three catalog functions and their branches, rewrite V02 table tests, and add flag interplay. The suite needs no `Package.swift` change because the file lives in the existing `VectorCoreBenchSupport` target path, and it uses the existing runner, so there is no duplicate runner.

### 3.3 `boundary` suite catalog

The catalog is generated for each D in `--dims` and each N in `--batch-ns`. Unit is candidate (N) for rows 1–9 and produced-entry (N²) for row 10. Every timed body ends with `blackHole(OutputFold.fold(output))`.

| # | Case ID | Scope | Timed body | Outside timing | IP | OA | RP | K | Validation |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `batch.euclidean2.D.Nn.aos-fp32-reused` | reused-input | `range_euclid2_D` into the buffer | fixture; caller buffer (allocated, initialized) | na | ex | na | in | vector against `DoubleRef.euclid2` |
| 2 | `…aos-fp32-setup` | setup-inclusive | allocate `[Float]`, run the kernel, release at invocation end | fixture | na | in | na | in | same |
| 3 | `…soa-fp32-reused` | reused-input | `euclid2_D` | SoA build; buffer | ex | ex | na | in | same |
| 4 | `…soa-fp32-setup` | setup-inclusive | `batchEuclideanSquaredD(query:candidates:)` | fixture | na (the API transposes: RP) | na (the API allocates and returns the array, so routeInternalAllocation = in) | **in** | in | same |
| 5 | `…aos-fp16-reused` | reused-input | `range_euclid2_mixed_D` | `convertToFP16_D`; buffer | ex | ex | na | in | vector; reports the FP16 error on this fixture |
| 6 | `…aos-fp16-setup` | setup-inclusive | `convertToFP16_D` (candidates only), allocate `[Float]`, kernel | fixture | in | in | na | in | same |
| 7 | `…soa-fp16-reused` | none | none | none | none | none | none | none | **unavailable** (3.1 reason) |
| 8 | `…soa-fp16-setup` | none | none | none | none | none | none | none | **unavailable** |
| 9 | `batch.euclidean2.D.Nn.consumer-only` | consumer-only | fold of precomputed scores | scores from route 1, computed once | na | ex | na | **ex** | validates the precomputed output |
| 10 | `matrix.euclidean2.D.Nn.consumer-only` | consumer-only | fold of the precomputed N² output | `MatrixDistanceBench.inputs` plus one API call | na | ex | na | ex | sampled pairs (as in the matrix suite) |

- **Counts.** Each (D, N) yields 10 entries: 8 executable and 2 unavailable.
- **Representative evidence** uses 512D only: batch N ∈ {257, 4096}, matrix N ∈ {17, 65}. 768 and 1536 are exercised by contract tests at small N.
- **Correctness for rows 1–8** comes from one extra invocation after timing.
- **Thresholds:** the existing name-based policy applies, 1e-6 relative for FP32 and 2e-3 for FP16, because "fp16" appears in the ID.
- **Pairing:** none of these entries pair.

### 3.4 Wiring: exact paths

- New: `Benchmarks/VectorCoreBench/Suites/BoundaryBench.swift`, which contains `struct BoundaryBench: BenchmarkSuite { static let name = "boundary" }`. Its bodies are value types, `BoundaryBodies`, made `public` only if Q9 is accepted.
- Also new: `Core/Boundaries.swift`, `Core/BatchFixtures.swift`, `Utils/OutputFold.swift`.
- `Benchmarks/VectorCoreBench/AppMain.swift`: the registry entry `BoundaryBench.name: BoundaryBench.self`, the `allowedSuites` set, and a usage line: `--suites boundary  opt-in setup-inclusive companions, reusable-output counterparts and consumer-only records; never default, never auto-paired`.
- `Benchmarks/VectorCoreBench/ValidatedOptions.swift`: the default `allowedSuites`, the width rule, and the products and Int32 limits.
- New doc: `Docs/Benchmark_Measurement_Boundaries.md`. It covers scopes, flags, families and purposes, unavailable semantics, how to run, and what is **not** claimed.

### 3.5 Pairing and reduced precision

- The suite declares no comparisons, so it produces zero AB rows. Different scopes never pair, and consumer-only entries never pair.
- FP16 entries report measured error statistics on their own fixture. Equal requested outputs do not imply bitwise-equivalent arithmetic, and the doc says so.

---

## 4. Matrix consumer

### 4.1 Current code paths (`MatrixDistanceBench.swift`, sha 6ed4ab59…)

- **L48–51:** `measure<V: UnifiedVectorBuffer>` allocates `output` (N² floats) once, outside timing.
- **L52–55:** `once()` calls `MatrixDistance.euclideanSquaredMatrix(queries: vectors, candidates: vectors, into: &output)` and then `blackHole(output[0]); blackHole(output[pairs - 1])`. Index 0 is (0,0) and index N²−1 is (N−1,N−1). Both are self-distance diagonal entries, max(0, ‖x‖²+‖x‖²−2⟨x,x⟩): zero in exact arithmetic and a clamped Float rounding residual in practice (source-derived).
- **L56–57:** `Harness.warmup(once)` and `Harness.measure(… unitCount: pairs …)`.
- **L59:** the test-only hook `onMeasuredOutput?(output)`.
- **L60–73:** sampled-pair correctness after timing.
- Neither the warmup nor the measured body traverses any off-diagonal score.

### 4.2 The fold consumer

```swift
enum OutputFold {
    /// h₀ = 0xcbf29ce484222325; hᵢ₊₁ = (hᵢ ⊕ UInt64(xᵢ.bitPattern)) · 0x100000001b3 (mod 2⁶⁴)
    /// Single-position sensitivity: for a fixed word, h ↦ (h ⊕ w)·p is a bijection on UInt64
    /// (odd p is invertible mod 2⁶⁴); for a fixed h, w ↦ (h ⊕ w)·p is injective on 32-bit w.
    /// Changing exactly one element therefore changes the result (tests: …FoldSensitivity…).
    /// Not a collision-free fingerprint for multi-element changes; not a numerical oracle.
    @inline(never) static func fold(_ values: UnsafeBufferPointer<Float>) -> UInt64
}
```

- The fold uses integers only, so the consumer itself has no floating-point cancellation.
- It is order-dependent and distinguishes −0 from +0 and NaN payloads.
- `@inline(never)` gives a stable inspection anchor and the identical consumer for timed and consumer-only records.
- Matrix body: `once() { MatrixDistance.euclideanSquaredMatrix(...into: &output); blackHole(output.withUnsafeBufferPointer(OutputFold.fold)) }`. Sink use stays in the serial driver.
- The boundary consumer changes to `u64-bitfold-sink-v1`, so identity changes, as the brief requires.

### 4.3 Consumer-only record

`matrix.euclidean2.D.Nn.consumer-only`, in the `boundary` suite (3.3 row 10), with this alternative placement in the matrix suite rejected.

- **Fixture and route.** It uses the same fixture and API to precompute the output, the same `OutputFold.fold`, and N² units.
- **Reporting.**
  - It is reported separately.
  - It is never subtracted and never paired.
- **Placement.** Keeping it out of the `matrix` catalog leaves V02's matrix selection tables unchanged; only the consumer identity changes.

### 4.4 Tests

1. The perturbation test runs on the consumer function used by the timed body, at D = 512 with N ∈ {17, 65}, and at N = 3 for all widths.
   - It changes each of first (0), middle (⌊N²/2⌋) and last (N²−1) **individually**, flipping the lowest bit of `bitPattern`, and asserts that the fold changes. Restoring the value must restore the fold.
   - **Note:** for odd N, ⌊N²/2⌋ = k·N + k with k = (N−1)/2, so first, middle and last are **all diagonal** entries. The test therefore also perturbs an off-diagonal index (1, which is (0,1)) so that off-diagonal observation is shown explicitly.
2. The same three-position test runs for the batch companions at N ∈ {257, 4096} at 512D, and at small N for 768 and 1536.
3. A mutation control: substituting the baseline two-index observation must fail the middle-index and off-diagonal assertions. This is the RED/GREEN pair from section 9.

### 4.5 Inspection anchors

In the specialized `measure`/`once` code of `VectorCoreBenchSupport.o` and `vectorcore-bench`, the inspection looks for:

1. a call to `MatrixDistance.euclideanSquaredMatrix(queries:candidates:into:)`, generic or specialized;
2. a call to `OutputFold.fold` whose arguments are the output base address and count N²;
3. a call to `blackHole` that receives the fold result.

Section 5.3 covers the other routes.

---

## 5. Memory probe and allocation instrumentation

### 5.1 Release public-API process-memory probe

**Paths (new).**

- `Scripts/ci/tests/V04MemoryProbe.swift`.
- `Scripts/ci/tests/V04RunMemoryProbe.py`. It is not named `test_*.py`, so CI's `unittest discover` does not import it, but `test_child_process_bounds.py` still scans it.
- Evidence goes under `.../2026-10-03/V04/memory-<n>/`.

**Build and link.**

1. The capture driver runs `build -c release --product vectorcore-bench --jobs 6 --scratch-path <ev>/build-<n> --verbose`.
2. The driver resolves `--show-bin-path` through `Scripts/ci/swiftpm_products.py` and accepts exactly the native and swiftbuild layouts. Existing recovery builds show the swiftbuild layout:
   - `<scratch>/out/Products/Release/{VectorCore.o, VectorCoreC.o, VectorCore.swiftmodule}`;
   - `<scratch>/out/Intermediates.noindex/GeneratedModuleMaps/VectorCoreC.modulemap`.
3. The probe is compiled with `swiftc -O -swift-version 6 -parse-as-library -I <Products/Release> -Xcc -fmodule-map-file=<VectorCoreC.modulemap> V04MemoryProbe.swift VectorCore.o VectorCoreC.o -framework Accelerate`.
   - It uses the public API only (`import VectorCore`, no `@testable`).
   - The command and the hashes of sources, objects, module and binary are retained.

**Children.**

- One fresh process per case: {fp32, fp16} × {512, 768, 1536} × {1, 257, 4096} × {independent, repeated} = **36**, plus **2 controls**.
- Each child:
  - writes to file captures;
  - has SIGINT/SIGTERM handlers installed before launch;
  - gets a 15 s execution deadline and at most 2 s of owned process-group cleanup;
  - on timeout, raises before any termination property is read (the `v03_process.run` pattern).
- Children run one at a time.
- **Driver deadline, recorded and finite: 12 min** (38 × 17 s ≈ 10.8 min worst case). This is for root review (Q20).

**Child phases.**

1. **Warm.** Construct and drop one vector through the same initializer. Run the fold and both snapshot functions once.
2. **S0: pre-creation snapshot.**
3. **Create the dataset** in a function scope.
   - All four modes use the same D-element seeded finite `values` array ("equal-valued").
   - **fp32 / independent:**
     - The array is reserved to N.
     - N separate `try VectorDOptimized(values)` calls.
   - **fp32 / repeated:** one vector, then `Array(repeating: one, count: N)`.
   - **fp16 / independent:**
     - Each element is converted from its own transient `let source = try VectorDOptimized(values)` with `MixedPrecisionKernels.VectorDFP16(from: source)`.
     - The source ends with its iteration, so no FP32 payload is retained.
   - **fp16 / repeated:**
     - One FP16 value is converted from a transient source.
     - The source is released.
     - Then N copies are made.
   - **Touch and consume.** Every coordinate of every vector is folded (`withUnsafeBufferPointer` / `withUnsafeFP16BufferPointer`) and the fold is printed.
   - **Values preserved.**
     - FP32 must be bit-exact against `values`.
     - FP16 must be bit-exact against `Float16(values[k]).bitPattern`, computed independently. The probe is arm64-only, and this is recorded.
   - **Construction proof, not inferred from equality.**
     - Inside scoped borrows, each storage base address is recorded as an integer. It is never dereferenced or escaped as a pointer.
     - independent: N distinct addresses required.
     - repeated: 1 distinct address required.
     - A mismatch fails the child.
4. **S1: retained snapshot**, with the dataset kept alive by `withExtendedLifetime`.
5. Return from the scope.
6. **S2: post-release snapshot.**

**Snapshot fields.**

- **Mach task info.**
  - Call: `task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), &info, &count)` with `count = MACH_TASK_BASIC_INFO_COUNT`.
  - Fields recorded:
    - the `kern_return_t` and the returned count;
    - `virtual_size`, `resident_size` and `resident_size_max`.
  - All three are in bytes, per SDK `mach/task_info.h` lines 309–323 (sha 97604815…).
  - Any non-`KERN_SUCCESS` result is an explicit unavailable observation, never zero.
- **Allocator statistics.**
  - Call: `malloc_zone_statistics(NULL, &stats)`.
  - `blocks_in_use`, `size_in_use` and `size_allocated` are labelled "process-wide sums across zones". The basis is `malloc/malloc.h:614`, "if !zone, sums up all zones" (sha 30b32f3e…).
  - `max_size_in_use` is reported only as `sumOfZoneHighWaterMarksBytes` (malloc.h:537, "high water mark of touched memory" per zone). It is never presented as a simultaneous process peak.
- **Signed deltas.** S1−S0, S2−S1 and S2−S0, for current values only.
- **Maxima.**
  - Maxima are reported raw at S0, S1 and S2, including the pre-creation maximum.
  - Note recorded with them: "process-lifetime high-water since process start; not dataset-specific; never subtracted".
- **Labelled model columns**, kept alongside:
  - `logicalPayloadBytes` = N·D·4 for fp32 or N·D·2 for fp16 (checked arithmetic);
  - `handleArrayModelBytes` = N·`MemoryLayout<V>.stride`.
  - There is no prediction and no 50% claim.
- **Identity recorded:**
  - workload: representation, D, N, sharing, seed, construction path and value generator;
  - build: executable, source and object hashes;
  - the SDK header hashes;
  - the lifetime markers `V04_MEMORY_PHASE pre-creation|retained|post-release`;
  - API names and units.
- **Failure handling.** An API failure or a missing fixture is an uncompleted measurement, never 0 bytes.

**Controls.**

1. **Known touched allocation.**
   - A 64 MiB `UnsafeMutableRawPointer.allocate`, with one byte written per 16 KiB page.
   - Snapshot S1, deallocate, snapshot S2.
   - Asserts that the APIs succeed and that size_in_use(S1) − size_in_use(S0) ≥ 64 MiB. This is an allocator-accounting liveness check.
   - Resident deltas are recorded, **not asserted**.
2. **No dataset.** Same phases with no allocation, to record runtime noise.

**Not done.** No exact RSS-delta assertion, sampling thread, allocator hook, intentional exhaustion, or SDK or platform-floor change.

### 5.2 Allocation-event probe (O03 collector linked unchanged)

**Paths (new).** `Scripts/ci/tests/V04AllocationProbe.swift` and `Scripts/ci/tests/V04RunAllocationProbe.py`.

**Collector.** `Scripts/ci/tests/O03Allocation.c` is compiled **unchanged** with `clang -O2 -fno-builtin-malloc -fno-builtin-free -c`. Its sha is recorded before and after. The accepted O03 records and fixtures are not modified.

**Linkage.**

- The probe links, from the same Release build:
  - `VectorCoreBenchSupport.o`;
  - `VectorCoreBenchmarking.o`;
  - `VectorCore.o`;
  - `VectorCoreC.o`;
  - `allocation.o`;
  - Accelerate.
- It runs `MeasurementPhases.$observer.withValue(observer) { await BenchmarkRunner.runCLI() }` with exact single-case CLI arguments. The bodies it measures are therefore the **actual catalog closures in the shipped objects**, not copies.

**Observer.**

- `begin` calls `o03_start(threshold: N·D·4)`.
- `end` calls `o03_stop`, then reads events, requested bytes, large events, large bytes and the histogram.
- JSON rows are printed only after the stop.

**Isolation.**

- One case per child process.
- The collector is never shared.
- It asserts rather than replace an existing `malloc_logger`.

**Windows.**

- The phases are setup, warmup, timed and validation.
- Timed runs are at fixed repeats R with 1 sample, and then at 2R in a second child. Comparing the two separates per-invocation events from fixed overhead.
- Results are labelled "allocation events / requested bytes in this window". They are not live heap, peak heap, RSS or elapsed time.

**Cases.** All at 512D:

- `batch.euclidean2.512.N{257,4096}.optimized.{sequential,parallel}`;
- the boundary suite's `aos-fp32`, `soa-fp32` and `aos-fp16` cases in both scopes, plus `consumer-only`;
- `matrix.euclidean2.512.N{17,65}.optimized` and its consumer-only record.

That is 14 case IDs × 2 repeat levels = 28 children. Each child has a 15 s deadline and at most 2 s of cleanup. The driver deadline is 10 min (Q20).

**Controls, in every child before its case.**

- `o03_calibrate_filter()`.
- `o03_calibrate_c()`, which asserts 2 large events and 24 576 bytes.
- A Swift Array window: `[Float](repeating: 1, count: 4096)` is consumed, and the window must show at least 1 event and at least 16 384 requested bytes.
- An empty window, recorded only.

**If Q9 is rejected.** The probe calls `BoundaryBodies` directly instead. Those are still the shipped objects. The provider-AoS item is then recorded as "allocation phase not established", an open acceptance item, and nothing is inferred in its place.

### 5.3 Emitted-code inspection (`V04InspectEmittedCode.py`, new)

**What is retained.** From the same Release build:

- `vectorcore-bench`, `VectorCoreBenchSupport.o`, `VectorCore.o` and the module;
- the verbose build log, which holds the compile and link commands;
- all hashes.

**Tools** (each command recorded): `xcrun nm -m`, `swift demangle`, and `otool -tV` or `xcrun llvm-objdump --disassemble-symbols`.

**Routes and anchors.**

1. **Provider AoS: `batch.euclidean2.512.N4096.optimized.sequential`.** The closures are nested in `BatchBench.runBatchOptimized(name:query:candidates:provider:options:minChunk:)`, the Vector512Optimized overload.
   - The range closure calls or inlines `BatchKernels.range_euclid2_512` and returns a summation over the scratch buffer.
   - The measurement closure calls the existential `parallelReduce`, then `ProviderObservations.observe`, then `blackHole` with the returned Float.
2. **SoA: `batch.euclidean2.512.N4096.soa-fp32-reused`, and optionally the default `optimized-soa`.**
   - The code reaches `euclid2_blocked_4way`, with four accumulator chains.
   - The inspection records whether the dead loads from `prefetch(_:offset:)` at j+4 survive.
   - It also records whether any `prfm` instruction is present. Absence is expected, and the helper is not called a hardware prefetch unless one is found.
   - The fold call and then `blackHole` follow.
3. **Matrix consumer.** Anchors as in 4.5.
4. **Optional: AoS FP16 reused.**

**Rules.**

- When inlining removes a symbol, the caller is inspected and the association recorded.
- The absence of a symbol is never recorded as elimination.
- No throughput conclusion follows from a surviving call or instruction.

### 5.4 Timing records (`V04RunTimingRecords.py`, new)

**Command.** One child per case, one at a time:

```
vectorcore-bench --suites <s> --dims 512 --batch-ns <N> --filter <exact id> \
  --repeats <R> --samples 9 --min-time 0 --format json --out <ev>/<id>.json
```

`--prefer-soa` is added for the default SoA case.

**What the records include.**

- **Warmup.** The existing harness warmup of about 200 ms runs, and its iteration count is recorded.
- **Samples.** At least 9 raw per-sample averages are kept, with R recorded.
- **Statistics.** Median and RSD, with the statistic unit stated.
- **Consumer-only records**, included as separate cases.
- **Host:** `sysctl hw.model machdep.cpu.brand_string hw.ncpu hw.memsize`.
- **Toolchain:** `sw_vers`, `swift --version`.
- **Build:** flags from the capture log.
- **Thermal and low-power state:** from metadata, or unavailable reasons.

**What they do not include:** pass/fail thresholds, derived speedups or subtraction.

**Driver deadline:** 10 min, finite and recorded (Q20).

---

## 6. Claim ledger

### 6.1 Files in scope

**From WORK_ITEMS V04 (12 files):**

- `Benchmarks/VectorCoreBench/Suites/BatchBench.swift`, `MatrixDistanceBench.swift`;
- `Sources/VectorCore/LinearAlgebra/PCA.swift`, `SwiftLinearAlgebraProvider.swift`;
- `Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift`, `MixedPrecisionKernels.swift`;
- `Sources/VectorCore/Optimization/OptimizationAttributes.swift`;
- `Sources/VectorCore/Storage/COWDynamicStorage.swift`;
- `Sources/VectorCore/Vectors/DynamicVector.swift`, `Vector512Optimized.swift`;
- `Tests/ComprehensiveTests/MixedPrecisionPerformanceBenchmark.swift`;
- `Tests/MinimalTests/ComprehensiveBenchmarks.swift`.

**Touched sink, harness and model:**

- `Benchmarks/VectorCoreBench/Utils/BlackHole.swift`;
- `Benchmarks/VectorCoreBench/Core/BenchmarkHarness.swift`;
- `Sources/VectorCoreBenchmarking/Models.swift`.

**Recommended addition (Q21):** the new `Utils/OutputFold.swift`, because it carries a correctness claim.

### 6.2 First-pass inventory

**How the counts were produced.**

- **"v0 lexical"** is a scratch scan of comments, single-line strings longer than 12 characters, and doc anchors. Its vocabulary:
  - quantitative: numbers with x/×/%/~/ns/µs/ms/MB/KB/GB;
  - performance: faster, speedup, throughput, latency, performance, bandwidth, cache, prefetch, overhead, efficient, optimal, amortize, ILP, AMX, L1/L2/L3, zero-cost;
  - robustness: robust, stable, guarantee, always, never, accuracy, precision, error, bound, monotonic, rank, bitwise, identical, exact, thread-safe, lock-free, GC, overflow, underflow, NaN, finite, deterministic, safe.
- **v0 total: 510.**
- **Known v0 misses:**
  - the boundary phrase "this cost is part of the benchmark";
  - "Prevents the compiler…";
  - multi-line `"""` literals, including `estimatePerformanceGain`'s "10-20%".
- **v1** adds compiler, hint, optimize, prevent, eliminate, force, cost, timing, allocate, deallocate, retain, footprint, saving, memory, unroll and blocking. That gives **686** hits, including case-ID strings that must be excluded.
- **"Claims"** below are a manual first-pass count of real performance or robustness statements. The MixedPrecisionKernels figure is an estimate; it was 253 after a noise filter.

| File | v0 | Claims (first pass) | Quantitative |
|---|---|---|---|
| Vector512Optimized.swift | 23 | 14 | 5 |
| COWDynamicStorage.swift | 6 | 8 | 0 |
| OptimizationAttributes.swift | 15 (v1: 31) | 18 | 0 |
| BatchKernels_SoA.swift | 81 (+1 multi-line literal) | ~45 | 6 |
| MixedPrecisionKernels.swift | 313 | ~150 (estimate; exact count in phase F) | ~70 |
| PCA.swift | 20 | 12 | 1 |
| SwiftLinearAlgebraProvider.swift | 3 | 4 | 0 |
| DynamicVector.swift | 4 | 1 | 0 |
| MixedPrecisionPerformanceBenchmark.swift | 32 | ~15 | 4 |
| ComprehensiveBenchmarks.swift | 2 | 3 | 0 |
| BlackHole.swift | 1 (v1: 3) | 3 | 0 |
| BenchmarkHarness.swift | 5 | 5 | 3 |
| Models.swift | 2 (v1: 9) | 4 | 0 |
| BatchBench.swift | 2 (v1: 43) | 19 | 0 |
| MatrixDistanceBench.swift | 1 | 2 | 0 |
| **Total** | **510** | **~300** | **~90** |

Representative anchors (symbol: statement → proposed disposition and evidence), file by file:

**Vector512Optimized.swift**

- `struct Vector512Optimized` doc, L15–18: "~100ns / ~120ns / ~150ns" → **removed**. No dated artifact exists; Performance_Guide carries the same "indicative" figures (Q13).
- `dotProduct` (L188, ~100ns) and `euclideanDistanceSquared` (L196, ~120ns) → **removed**.
- L21–23: "16-byte aligned" is retained, from the adjacent fact that `SIMD4<Float>` is 16-byte aligned. "optimal alignment" and "Cache-line optimized for Apple Silicon" → removed or qualified.
- L28: `_storage` "optimal performance" → corrected to design intent.
- L162/L175: "TBAA-safe" scoped rebinding → retained only with a named O04/O05 test (looked up in phase F); otherwise reworded as an unverified contract.
- L207–211: magnitude wording (N01) → retained.
- L342: hash "for efficiency" → retained as design intent; hash quality belongs to P01.

**COWDynamicStorage.swift**

- L3–7: "Wrapper … for DynamicArrayStorage … enables DynamicVector" → **corrected**. The stored type is `AlignedDynamicArrayStorage` (L42). DynamicVector uses `HybridStorage<Float>` (DynamicVector.swift:15). There are no callers or tests in the package.
- L24–25: O(1) copy and O(n) first-mutation copy → retained from the adjacent `isKnownUniquelyReferenced` logic.
- L26: "No overhead for unique instances" → **removed**.
- L33–35: the example's "(no allocation)" and "allocates new storage" → reworded as an unverified contract (no test).

**OptimizationAttributes.swift**

- **Undefined flags.** `VECTORCORE_ENABLE_SIMD` and `SWIFT_RELEASE_MODE` are never defined in Package.swift, CI or any source (grep). As a result:
  - `isSIMDEnabled` (L10) and `isReleaseMode` (L20) return false in package builds;
  - `releaseOnly` never runs;
  - `performancePrecondition` (L234, "can be disabled in release for performance") is **enforced in Debug and Release**.
  - All of these docs → corrected (Q14).
- `performanceCritical` "always inline" (L32), `likely`/`unlikely` "Hints to the compiler" (L39/L47), and `withLikelyBranch` / `withUnlikelyBranch` "branch prediction hint" → corrected. These are identity wrappers.
- `prefetchRead`/`prefetchWrite`, L57–76 ("Compiler should recognize this pattern for prefetching"; "Touch the memory location to hint at future write") → corrected to: a discarded load or self-assignment that the optimizer may remove; it is not a prefetch instruction, and no new intrinsic is added.
- `assumeAligned` "Ensures…" → corrected to "Debug-only assert".
- `optimizationBarrier` and `doNotOptimize` ("Prevents the compiler from optimizing…") → corrected or reworded as an unverified contract.
- `unrolledLoop`/`vectorizedLoop` → retained. Named tests: `ExecutionBoundaryTests.testArbitraryLoopWidthsExactCoverage` and `testLoopErrorsStopAtFirstInteriorAndTail`.
- None of these helpers has a caller in the package (grep), so the change is comment-only.

**BatchKernels_SoA.swift**

- `euclid2_blocked_4way`, L112–117 ("Achieves ~1.5-2x throughput…") → **removed**. The four-accumulator mechanism is retained from adjacent code.
- The `prefetch(_:offset:)` helper, L456–467 ("ARM64 prefetch instruction … would go here … compiler may auto-prefetch"), and the call-site comments at L142, L482, L487, L521, L588, L957 and L1052 → **corrected** to "discarded pointee load". This is an inspection item (5.3).
- `tilingByteThreshold`, L187–200 ("hardware prefetcher can no longer track L streams… ≈ Apple-silicon L2") → the traversal fact is retained; the microarchitectural rationale is labelled heuristic and unverified.
- `euclid2_tiled` "prefetcher-optimal" and "L1-resident" → qualified.
- The **"bitwise-identical to the 4-way path"** claims at L223–224, L636 and L1110 → **retained**. Named tests: `BatchKernels_SoATests.swift` lines 12–128 compare tiled and 4-way bit patterns. Phase F confirms each variant is covered.
- `estimatePerformanceGain`, L413–432: the doc and the **returned string** "Expected improvement: 10-20%" (Q11).
- `batchManhattan512/768/1536`, L1343–1352 ("SoA transformation with vDSP vsadd/vabs/vadd… ~10-15% faster") → **corrected**. The functions take AoS `[VectorDOptimized]` and call the `manhattanDCore` or `…CoreBlocked` loops (N > 10 000). The speed claim is removed. Phase F re-reads the cores.
- L853 "32KB L1 cache (typical)" and `shouldUseSoA` L1772–1776 → labelled heuristic.
- The header and type doc at L5–6 and L23–28 ("superior cache locality") → qualified.

**MixedPrecisionKernels.swift**

- **Type-level error analysis, L47–76** → corrected or removed:
  - The "Accumulated error bound ≤ ε₁₆√D‖x‖‖y‖ + Dε₃₂…" uses √D, which is a probabilistic growth rate, not a deterministic bound.
  - "Individual term error ≤ ε₁₆|xᵢyᵢ|" holds only for FP32×FP16 in the normal range.
  - The empirical 0.08% / 0.10% / 0.15% figures have no artifact.
  - "Maintains monotonicity: distance ordering preserved for ranking" is false: equal FP16 roundings merge ranks.
  - "FP32 accumulation prevents catastrophic cancellation" is false.
  - "No systematic bias" is unverified.
  - "Special values … propagate correctly" contradicts the cosine contract at L4760–4761, which states that this is not NaN propagation.
  - "<0.1% typical" and "Absolute error bounded by machine epsilon" have no evidence.
  - Unit roundoff 2⁻¹¹ and 2⁻²⁴ are retained, with their domain stated: finite, normal range, round-to-nearest; not through underflow, overflow, zero or cancellation.
- `Vector512FP16`, L298–299: "1024 bytes (50% reduction)" → payload-only label. "Conversion overhead: ~5% of dot product time" → **removed**.
- **Speed and throughput claims, all removed:**
  - SoAFP16, L470: 2-4×.
  - `createSoA*FP16`, L602, L670, L738: ~1-2 μs.
  - L916, L1075, L1128: ~120 ns / ~130 ns.
  - L1875–1878: 1.5-1.8×, ~8.3 μs.
  - `batchEuclidean{512,768,1536}`, L1925–1930, L2013–2018, L2317–2322: 2-4× and ~20-30 M/s.
  - `batchDotProductSoA` ×3, L2521: ~1.5-2×.
  - `batchEuclideanBlocked*` ×3, L2787–2790: "Reduces loop overhead by 50%… ~1.3-1.5×". The loop-control mechanism stays; percentages go.
  - L4142: 2× throughput.
  - L4932: ~50 ns/vector.
  - L4964: 1.5-2× speedup.
  - L5369: 30-50% faster.
  - L5396 and L5427: "vcvt … 1 cycle latency".
- **Accuracy claims, removed or tied to a tested domain:**
  - L2526, L2642, L2718: "<0.1% relative error".
  - L4516: "< 0.01% error".
- "reduces memory bandwidth by 2×" and "2× memory bandwidth" (×6) → payload-bytes-only wording.
- **`FP16VectorPool`** (the header was already corrected by Q02):
  - `release`'s "discarded (auto-deallocated)" → **corrected**.
  - `drain`'s "(forces deallocation)" → **corrected**: `removeAll(keepingCapacity: true)` keeps the array capacity, and payloads are freed only when no other reference remains.
- Platform cache facts at L4953–4999, L5017–5021 and L5210–5221 → labelled unverified hardware assumptions; the threshold constants are kept.
- The **runtime strings** in the `PrecisionProfile` summary, L3485–3487 (Q11).
- **Retained with evidence:**
  - `range_cosine_mixed_*`, L4760–4761 (Q04).
  - FP16 rounding notes, L308–309 (Q03/Q04).
  - `adaptiveEuclideanDistance`, L3397–3401 (Q02).
  - "Preserves full query precision", ×8, from the adjacent fact that the FP32 query is not converted.

**PCA.swift and SwiftLinearAlgebraProvider.swift**

- Already reconciled by L01/L02: thin SVD at the clamp, "No corpus-size memory or throughput envelope is implied", and the named tests PCACenteringContractTests and LinearAlgebraRangeTests.
- Open items:
  - PCA L43, "1–2 is standard": cite or mark unverified.
  - PCA L18–21, "batches parallelize freely": adjacent reasoning, or qualify, because provider GEMM thread-safety is a separate contract.

**MixedPrecisionPerformanceBenchmark.swift**

- The footprint table at L287–327 → section 7.
- The strict-opt-in gate messages ("should be faster", L100–101, L134, L228) → retained as V03-inventoried strict assertions, not facts.
- The accuracy "< 1%" messages at L280–283 → retained as fixture test thresholds.

**ComprehensiveBenchmarks.swift**

- "MARK: - Memory Performance" and the two MemoryFootprint names → renamed (section 7).
- "Prevent optimization" → retained as anti-elision intent.

**BlackHole.swift**

- "A stronger black-hole to prevent DCE in Release builds… force a visible side-effect" → **corrected**. It hashes the argument's value-representation bytes into an unsynchronized global. It does not traverse heap elements, is for the serial driver only, is not thread-safe, and makes no guarantee beyond the value it receives.

**BenchmarkHarness.swift and Models.swift**

- The 200 ms and 100 µs constants, "~80% of minNs", nearest-rank p90, and the `deltaPercent` formula → retained from adjacent code.
- The preparation strings → derived from boundaries.

**BatchBench.swift**

- 9 × "Build SoA once / Convert candidates to FP16 once (this cost is part of the benchmark)" at L1014, L1058, L1101, L1146, L1203, L1259, L1328, L1397 and L1466 → **corrected**: the preparation happens before warmup and measurement, so it is outside timing.
- The buffer-reuse comments ×9 → retained from adjacent code.

**MatrixDistanceBench.swift**

- The sampled-pairs doc and "unitCount counts N*N" → retained from adjacent code.
- The new fold doc → OutputFold tests.

### 6.3 Ledger: `Scripts/ci/claim_ledger.json` (new, checked in)

```json
{
  "schemaVersion": 1,
  "scope": "Inventory and lexical guard backed by full manual source review; not a theorem prover or semantic detector for arbitrary English.",
  "normalization": "comment-text-v1: strip ///, //, /* */, leading *, Swift string quotes; NFC; collapse whitespace; lowercase; statement = one doc paragraph or bullet (wrapped lines joined); fingerprint = first 16 hex of sha256",
  "vocabulary": {"tierA": ["<regexes: quantitative units, %, ×, ~N, speed/throughput/latency, cache/prefetch/L1-L3, guarantee/always/never/robust/stable/bitwise/identical, accuracy/error bound/monotonic/ranking, lock-free/GC/thread-safe, zero overhead, eliminates/forces/deallocat, inside/outside timing, part of the benchmark, compiler hint/prevents optimiz>"],
                 "excludeLiterals": "^[A-Za-z0-9_.\\\\()%-]+$"},
  "regions": [{"path": "<file>", "coverage": "whole-file"}],
  "forbiddenPhrases": [{"pattern": "this cost is part of the benchmark", "paths": ["Benchmarks/**"], "reason": "preparation is outside timing"}],
  "entries": [{
    "id": "V04-C-0001",
    "path": "Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift",
    "symbol": "func euclid2_tiled",
    "fingerprint": "<16 hex>",
    "statement": "<normalized current text>",
    "category": "performance-quantitative | performance-qualitative | robustness | numerical-accuracy | memory-model | measurement-boundary | concurrency | non-claim",
    "disposition": "retained | corrected | unverified-contract | removed | not-a-claim",
    "evidence": {"kind": "adjacent-invariant | named-test | dated-artifact | unverified-contract | none",
                 "reference": "Tests/ComprehensiveTests/BatchKernels_SoATests.swift::<test name>",
                 "domain": "<inputs/widths/build where it holds>", "limitations": "<what does not follow>"},
    "acceptedContract": "L01 | L02 | Q02 | Q03 | Q04 | O02 | O03 | O05 | N01 | null",
    "history": {"originalStatement": "...", "originalFingerprint": "...", "origin": "V04-design-inventory | TOOL-07 | API-43 | NUM-23 | OWN-23 | V04 phase-1"}
  }]
}
```

**Anchors** are path + enclosing or following symbol + fingerprint. They contain no line numbers; an optional `lineHint` is ignored by the check.

**Removed statements** become tombstones: `fingerprint` = the historical fingerprint, with absence required. Retained and corrected entries bind to current text.

**Vocabulary non-claims** that match tier A each get a fingerprinted `not-a-claim` entry with a reason. There is no blanket suppression: any edit changes the fingerprint and forces re-review.

### 6.4 Mechanical check

The check is `Scripts/ci/check_claim_ledger.py`. Its tests are `Scripts/ci/tests/test_claim_ledger.py`, a `unittest` module that CI's existing `python3 -B -m unittest discover -s Scripts/ci/tests` step discovers automatically, so **no `ci.yml` edit** is needed. It is stdlib-only and bounded: it reads at most the listed files.

| Rule | Requirement | Failure reported |
|---|---|---|
| R1 | A retained, corrected or unverified-contract entry's path and symbol exist, and a statement with that fingerprint is attached to that symbol. | "stale anchor" |
| R2 | A tombstone's fingerprint and its forbidden phrase are absent from current source. | tombstone violation |
| R3 | A `named-test` evidence reference must resolve to an existing `func` (XCTest `test…` or Swift Testing `@Test`) in the named file, and the source statement or its adjacent comment block must contain that test name. | "missing named evidence" |
| R4 | A `dated-artifact` reference must name an in-repo path with a sha256 and a date. | missing artifact reference |
| R5 | An `unverified-contract` statement must contain explicit wording from a fixed list ("not verified", "unverified", "not promised", "no … guarantee"). | missing unverified wording |
| R6 | Every tier-A statement in a region has exactly one entry. | "uncovered claim" |
| R7 | No forbidden phrase appears in its path globs. | forbidden phrase |
| R8 | IDs are unique, fingerprints are not double-covered, and every entry has the required fields. | integrity failure |

Statement extraction covers `//`, `///` and `/* */` comments, plus single-line and `"""` string literals, excluding identifier-like literals. Code tokens are never inspected, so exact `== 0` and bit checks in tests are untouched.

### 6.5 Mutation fixtures (temporary copies inside the test)

| ID | Mutation | Expected result |
|---|---|---|
| M1a | Delete an evidence reference in source: remove "LinearAlgebraRangeTests" from the SwiftLinearAlgebraProvider header. | R3 fails |
| M1b | Delete the ledger `evidence.reference`. | fails |
| M1c | Point the reference at a nonexistent test. | fails |
| M2 | Add an unsupported quantitative assertion in a covered region: `/// Achieves ~3x speedup on Apple Silicon.` above `euclid2_512`. | R6 fails |
| M3 | Change a statement without updating its fingerprint: "bitwise-identical" → "bit-identical" in the `euclid2_tiled` doc. | R1 fails |
| M4 | Reintroduce tombstoned text ("this cost is part of the benchmark"). | R2 and R7 fail |
| M5 | Add a forbidden variant ("lock-free fast path"). | R7 fails |
| M6 | Move a retained statement to another symbol. | R1 fails |
| M7 | Duplicate an ID, or cover one fingerprint twice. | R8 fails |
| Control | Unchanged tree. | passes |
| Control | Whitespace or rewrap-only edits. | passes |
| Control | Test code containing exact `== 0`. | not flagged |

### 6.6 Limits, documented in the check and the doc

This is an inventory and lexical guard. It rests on full manual review. It is not a semantic detector for arbitrary English and cannot prove that a claim is true. It only enforces that each tier-A statement has an explicit, reviewed disposition and that the evidence it names exists. A named test supports its claim only within its own fixture domain.

### 6.7 Defect classes found outside the named files (Q10, Q17)

**TOOL-08 class: handle stride used as memory**

These assertions compare handle strides. They pass only because `Vector{D}Optimized` stores `_storage` (8 B) **plus** `let scalarCount: Int` (8 B) while `Vector{D}FP16` stores one handle (8 B). Source-derived strides are 16 and 8; to be confirmed in RED. Instances:

- `Tests/ComprehensiveTests/MixedPrecisionPart1ValidationTests.swift:152-158`: `#expect(ratio < 0.6 && ratio > 0.4, "…not close to 50%")`.
- `Tests/ComprehensiveTests/MixedPrecisionKernelsTests.swift:327-331`: prints "Memory saved".
- Same file, 369-375: `#expect(savings > 45)`.
- Same file, 806-811: `#expect(fp16StorageSize < fp32StorageSize, "FP16 storage should be smaller")`.

Related but outside V04's reach now:

- `QuantizedKernelsTests.swift:1158` labels payload arithmetic "Actual memory usage measurement". That file is **T02-owned**, so V04 does not touch it.

**TOOL-07 class: discarded results in timed test bodies**

There are 39 sites (`_ = …`): 24 in `ComprehensiveBenchmarks.swift` and 15 in `MixedPrecisionPerformanceBenchmark.swift`. The brief limits test edits to labels and memory models.

---

## 7. Test-name migrations

| Old name | New name | Where it changes in lockstep |
|---|---|---|
| `ComprehensiveBenchmarks.testVector512OptimizedMemoryFootprint` | `testVector512OptimizedMagnitudeTraversalTiming` | `Tests/MinimalTests/ComprehensiveBenchmarks.swift`: method (L263), `observePerformance` label (L268), and `// MARK: - Memory Performance` (L261) → `Magnitude Traversal Timing`. `Scripts/ci/performance_sites.json`: `measurements[21]` `id` and `test`. |
| `ComprehensiveBenchmarks.testVector512GenericMemoryFootprint` | `testVector512GenericMagnitudeTraversalTiming` | Method (L277), label (L282); `measurements[22]` `id` and `test`. |

**Preserved without change:**

- the 1000-vector setup outside the observed block;
- `observePerformance`;
- `XCTAssertNotEqual(sum, 0)`, the exact-zero anti-elision check, which is not converted to an epsilon.

**Line counts are unchanged**, so `originalLine` is unchanged. The checker does not validate lines or `originalFileSHA256`.

**No change needed in:**

- `check_performance_policy.py` and `test_performance_policy.py`: counts stay 73/30, and the mutation loops iterate the inventory.
- `sanitizer_bodies.json` and `sanitizer_expected_bodies.txt`: no ComprehensiveBenchmarks or MinimalTests entries.
- `check_performance_gates.py`: its representatives are other tests.
- `Docs/executed-ci-contracts.md`: its counts are unchanged.

Historical logs under `Docs/audits` and the execution reports keep the old names and are not modified.

**MixedPrecisionPerformanceBenchmark.**

- **Rename:** `analyzeMemoryFootprint` / `@Test("Memory footprint comparison")` → `analyzeRepresentationMemoryModel` / `@Test("Representation memory model: handle stride vs logical payload")`. It is not in any inventory; it appears only in historical Q01 test-inventory logs.
- **Table columns:**
  1. `Handle stride FP32 (MemoryLayout<VectorDOptimized>.stride)`.
  2. `Handle stride FP16`.
  3. `Logical scalar payload FP32 (D·MemoryLayout<Float>.stride)`.
  4. `Logical scalar payload FP16 (D·MemoryLayout<UInt16>.stride)`.
  5. `Measured records: not measured here; see V04 memory-probe artifact`.
- **Arithmetic:** checked with `multipliedReportingOverflow`.
- **The 100 000-vector example** stays only as two labelled models: "logical payload model" and "handle-array model". It never says "total footprint" or "savings".
- **Comment:** the 2:1 scalar-width ratio excludes headers, capacity, alignment, padding, conversion temporaries and sharing; equal values do not imply shared storage; and repeated copies need not allocate new payloads.
- **New independent constants:**
  - 512 → 2048/1024;
  - 768 → 3072/1536;
  - 1536 → 6144/3072.
  - They are literal values, not recomputed with the same formula, and include an overflow case.
- **Batch-throughput table** (L144/L167): the "Memory Saved" column becomes "Logical payload difference (model)".

---

## 8. Conflict table

V04 write set, checked against in-flight and planned work. T02 is under review and owns `QuantizedKernelsTests.swift` and `ArbitraryDimensionTests.swift`, plus at most one helper. T03 (future) may touch `check_benchmark_build_modes.py`, `validate_github.rb`, `ci.yml` and the fixture drivers.

| File | V04 action | Prior owner | Conflict |
|---|---|---|---|
| Sources/VectorCoreBenchmarking/{Models,ABComparisons,JSONWriter,EnvCapture}.swift | modify | V02 (accepted) | None in flight. Planned migration of V02's accepted schema and pairing contracts. |
| Sources/VectorCoreBenchmarking/MeasurementBoundary.swift | new | none | None. |
| Benchmarks/VectorCoreBench/AppMain.swift, ValidatedOptions.swift, Core/BenchmarkCase.swift, Core/BenchmarkHarness.swift, Formatters/CSVWriter.swift | modify | V01/V02/V03 (accepted) | None in flight. |
| Benchmarks/VectorCoreBench/Core/{MeasurementPhases,Boundaries,BatchFixtures}.swift, Utils/OutputFold.swift, Suites/BoundaryBench.swift | new | none | None. |
| Benchmarks/VectorCoreBench/Suites/{Batch,Distance,DotProduct,Normalization,MatrixDistance,Memory,Noop}Bench.swift | modify (catalog records; matrix consumer; BatchBench comments) | V01/V02/T01 | None in flight. **P01, P02 and L04 depend on V04**, so they are sequenced after it. |
| Benchmarks/VectorCoreBench/Utils/BlackHole.swift | comments only | V01 | None. |
| Scripts/compare_benchmarks.swift | modify | V02 | None. |
| Tests/BenchmarkContractTests/BenchmarkSelectionContractTests.swift, RunnerContractTests.swift | extend (Q12) | V01/V02/V03 | None. A new file in this directory would **also** need `check_ci_contracts.py` (`BENCHMARK_TEST_SOURCES`) and the generated `Docs/executed-ci-contracts.md` updated. Those are V03 files near T03's area, so extending the existing files is recommended. |
| Scripts/ci/tests/test_benchmark_reporting.py, test_benchmark_selection.py (and test_benchmark_contracts.py only to add boundary assertions; its 28/14/28 counts are unchanged) | modify | V01/V02 | **Medium.** These import `check_benchmark_build_modes.py` as a helper (`run`, `sha`, `write_json`, `find_executable`). If T03 changes those helpers, the edits collide. Sequence V04-B before T03, or coordinate. |
| Scripts/ci/check_benchmark_build_modes.py | **not modified** | V02 (sha pinned in `validate_github.rb:179`) | T03 owns its swiftbuild support. V03 left this "to V04 or the Xcode-27 CI migration"; I recommend T03 (Q16). V04 keeps the metadata keys this driver reads. |
| Scripts/ci/validate_github.rb, .github/workflows/*.yml | **not modified** | V03 / T03 | None. New Python tests run through the existing `unittest discover` step. |
| Scripts/ci/{v03_process,swiftpm_products}.py | imported read-only by new drivers | V03 | **Low.** If T03 changes their API, the V04 drivers adapt. |
| Scripts/ci/tests/O03Allocation.c | linked unchanged (sha recorded) | O03 | None. |
| Scripts/ci/tests/V04{MemoryProbe,AllocationProbe}.swift, V04Run{MemoryProbe,AllocationProbe,TimingRecords}.py, V04InspectEmittedCode.py | new | none | None. `test_child_process_bounds.py` scans them automatically. |
| Scripts/ci/claim_ledger.json, check_claim_ledger.py, tests/test_claim_ledger.py | new | none | None. |
| Tests/ComprehensiveTests/MixedPrecisionPerformanceBenchmark.swift | modify (memory model, labels) | V03 (gate wrapper sites unchanged) | None in flight. |
| Tests/MinimalTests/ComprehensiveBenchmarks.swift, Scripts/ci/performance_sites.json | rename 2 methods; 2 rows | V03 | None in flight. Lockstep in one change. |
| Tests/ComprehensiveTests/{MixedPrecisionPart1ValidationTests,MixedPrecisionKernelsTests}.swift | only if Q10 is approved | Q-series / V03 | None in flight (T02 does not own them). |
| Tests/ComprehensiveTests/QuantizedKernelsTests.swift | **not touched** | **T02 (under review)** | Its TOOL-08-class label (L1158) is deferred to after T02. |
| Sources/VectorCore/{Vectors/Vector512Optimized, Storage/COWDynamicStorage, Optimization/OptimizationAttributes, Operations/Kernels/BatchKernels_SoA, Operations/Kernels/MixedPrecisionKernels, LinearAlgebra/PCA, LinearAlgebra/SwiftLinearAlgebraProvider, Vectors/DynamicVector}.swift | comments only | N01, Q02, Q03, Q04, O05, L01, L02 (all accepted) | None in flight. Accepted wording is preserved. A token-stream diff with comments stripped proves the change is comment-only. |
| Docs/Benchmark_Measurement_Boundaries.md | new | none | None. |
| Docs/Performance_Guide.md (benchmarking section) | only if Q13 is approved | none | None. |
| Package.swift | **not modified** | V02/V03 (`check_ci_contracts` pins target registration) | None. |

---

## 9. Baseline RED plan

All of this runs on unchanged `dbd0833` before any V04 edit. Evidence goes under `.../2026-10-03/V04/A*/`. All builds go through the capture driver. Each probe child has a 15 s deadline and at most 2 s of cleanup.

**A1. Release build.**

```
capture_swift.py --workdir <ws> --log <ev>/A1.log --record <ev>/A1.json -- \
  build -c release --product vectorcore-bench --jobs 6 --scratch-path <ev>/build-A1 --verbose
```

The executable, objects and module hashes are retained.

**A2. Handle/payload mislabeling, from actual report fixtures.**

- **(a) The current footprint report** (Debug):
  - Command: `test --filter "MixedPrecisionPerformanceBenchmark/analyzeMemoryFootprint" --jobs 6 --scratch-path <ev>/build-A2`.
  - Capture the printed table.
  - Expected from source:
    - the "FP32 (bytes)" and "FP16 (bytes)" columns print `MemoryLayout<…>.stride`, 16 and 8;
    - "Savings %" and the 100 000-vector "MB" and "Saved" figures are handle-array arithmetic.
- **(b) A scratch stride/payload probe**, built from the A1 Release objects with the public API, outside the workspace.
  - It prints the strides of the six types and the logical payloads for the three widths.
  - It shows that the table's "bytes" equal handle strides, not D·4 or D·2.
  - It shows that the coincidental 50% comes from the stored `scalarCount` Int.
- **(c) The same run, for the class instances in 6.7** (Q10): the actual pass results of `MixedPrecisionPart1ValidationTests/testMemoryFootprint` and the three MixedPrecisionKernelsTests memory tests, mapped to the probed strides.

**A3. Weak matrix consumption, from actual report and model fixtures.**

- **(a) Actual baseline CLI report:**
  - Command: `vectorcore-bench --suites matrix --dims 512 --batch-ns 17,65 --repeats 3 --samples 2 --min-time 0 --format json`.
  - What it shows: schema 2, "V02-boundaries-1", and no per-case consumer or boundary identity.
- **(b) Model fixture probe** (scratch):
  - It links A1's `VectorCoreBenchSupport.o`, using the public `InputSeeds.matrix` and `InputFactory.randomArray`, so the fixture is the real one.
  - Run seeds 0–3.
  - It computes the actual matrix output and records:
    - the bit patterns of `output[0]` and `output[N²−1]`;
    - their (i, j) index mapping, asserting both are diagonal;
    - the u64 fold of all N² entries.
  - Expected: the two observed scalars are (near-)zero diagonal residuals, while the folds differ across seeds.
  - It also records that ⌊N²/2⌋ is diagonal for odd N.
- **(c) Baseline emitted code** for the `once` closure, from A1:
  - Expected anchors:
    1. the API call;
    2. two element loads at offsets 0 and (N²−1)·4 passed to `blackHole`;
    3. no traversal loop.
  - This is a consumption observation, **not** a dead-code-elimination claim.

**A4. Boundary identity absent.**

- (a) The baseline JSON from A3a and a `dist.euclidean.512.optimized` report have no `results[].boundary`.
- (b) The compiled baseline comparator accepts two such reports without any per-case boundary check.
- (c) The provider and preNorm correctness-method differences in 2.5 are recorded as baseline facts.

**A5. Boundary-label contradiction.**

- BatchBench's "this cost is part of the benchmark" comments sit above calls placed before `Harness.warmupAsync` and `measureAsync`.
- This is recorded as source-established, from one-step adjacent code.
- A timing RED is unnecessary, and none is claimed.

**GREEN counterparts:**

| RED | GREEN |
|---|---|
| A2 | the new memory-model table and the independent constants |
| A3 | the fold, plus perturbation tests at first, middle, last and off-diagonal positions; GREEN emitted-code anchors in phase D |
| A4 | boundary-required JSON and comparator identity tables |

A mutation control re-inserts the baseline two-index observation into a temporary copy, and the middle-index and off-diagonal tests must fail.

---

## 10. Phasing proposal and open questions

### 10.1 Phases

Order: **A → B → C → E → F → D → review.** Measurement (D) runs last, against frozen final source. E is disjoint from B and C and could run in a separate workspace if root wishes.

| Phase | Scope | Acceptance check |
|---|---|---|
| **A: Baseline RED** | Section 9. No source edits. | The evidence manifest has commands, exits, markers and hashes. RED A2, A3 and A4 are shown with actual fixtures. No performance numbers are interpreted. |
| **B: Boundary model and reporting** | Section 1 files, plus `Boundaries` records for every existing case (section 2); OutputFold and the matrix consumer; comparator; BlackHole and BatchBench comment corrections; Swift and Python contract migrations. | Debug and Release `BenchmarkContractTests` pass (Release with `-enable-testing`, per V03). The V01 process tables (41 children) are unchanged. V02 reporting and selection Python suites are migrated to schema 3 and pass. Pair counts under `--ab-only` are 36/18. The allowed-difference mutation table passes, and cross-scope pairs reject. The comparator boundary table covers match, per-field mismatch, legacy/missing (override only) and non-waivable integrity. An independent CSV parse covers the new columns. The fold perturbation tests pass. The sink-placement guard passes. `test_benchmark_contracts.py` still counts 28/14/28. `check_benchmark_build_modes.py` assertions are unaffected (keys preserved). Pinned SwiftLint and `unittest discover` pass. |
| **C: Boundary suite, CLI and docs** | Section 3. | An independent selection table for `--suites boundary` (all D × N, exact IDs). Unavailable `soa-fp16` records carry their reason at all widths. Default suites and case sets are byte-identical to the V02 tables. A route excluded by selection is never invoked. A selection of only unavailable entries exits 1. Every executable companion reports correctness, and FP16 entries report error statistics. The suite produces no AB rows. Companion fold perturbation tests pass. Real Release CLI runs at small N cover all three widths. |
| **E: Memory model and renames** | Section 7 (+ Q10 if approved, with a lexical guard against handle-stride memory arithmetic in tests). | Independent constants and the overflow case pass. Labels are as specified. Performance policy stays 73/30. The modified tests pass in Debug and Release with `EXTENDED=1 STRICT_PERF=0`, with observation-body markers. Lint passes. |
| **F: Claim ledger and comment corrections** | Section 6. | The full manual inventory is reconciled, with exactly one entry per tier-A statement. Every named test exists and runs. Tombstones are present. Mutations M1–M7 fail as designed and the controls pass. A comment-stripped token diff proves the production changes are comment-only. Debug and Release builds pass, the whole suite passes with `EXTENDED=0`, and lint passes. |
| **D: Measurement evidence** (after source freeze; one process at a time) | D1 Release build; D2 emitted-code inspection (5.3); D3 timing records (5.4); D4 allocation-event probe (5.2); D5 memory probe (5.1). | Manifests hold commands, status, markers, raw samples, unavailable fields and source/object/module/binary hashes. The recorded finite driver deadlines are root-reviewed **before** D3–D5 run. There are no thresholds, derived speedups or subtraction. |
| **Review** | Independent source and evidence review, then root verification. | Closure records for TOOL-07, TOOL-08, API-43, NUM-23 and OWN-23, with explicit limitations: no throughput claim, no universal error bound, no target-scale or downstream claim. |

### 10.2 Questions root must rule on

**Needed before phase B or C (they shape the code):**

| # | Question | Recommendation |
|---|---|---|
| Q1 | Should companion availability follow **route support**, recording the existing `shouldUseSoA`/`shouldUseMixedPrecision` result as `defaultEligible`, or **enforce eligibility**? Enforcing it makes the FP16 companions unavailable at 512D for N ≤ 6144, and at 768D for N ≤ 4096. | Route support. If enforcement is chosen, the FP16 representative N must be 6145 at 512D. |
| Q2 | SoA FP16 has no squared-Euclidean route. Report it unavailable, or add a sqrt-Euclidean companion family in which only AoS FP32 (`range_euclid_D`) and SoA FP16 (`batchEuclidean{D}`) are available? | Report it unavailable. |
| Q3 | A separate `boundary` suite, or a `--timing-scope` option? | The separate suite (3.2). |
| Q4 | Add the direct-kernel reusable-output AoS FP32 counterpart (`range_euclid2_D` into a caller buffer) inside the suite, while also labelling the existing provider cases with their actual API boundary? | Yes. |
| Q5 | Keep the default `optimized-soa`/`optimized-fp16` Float-sum consumers, with their consumer identity recorded, or migrate them to the fold? | Keep. |
| Q6 | For partial consumers in `normalize` and `memory`: record them as partial, or migrate `normalize.success` to a full fold, which changes its measured work? | Record as partial. |
| Q7 | Should a cross-run boundary mismatch be waivable with `--allow-incompatible`, following V02 policy? Should a `timingScope` mismatch be non-waivable? | Waivable for missing or legacy boundaries, which follows V02 policy. Root to decide on scope mismatches. |
| Q8 | Unavailable semantics: not a failure, and a selection containing only unavailable entries exits 1? | Yes. |
| Q9 | A public task-local `MeasurementPhases` observer seam in the benchmark-support module, so the allocation probe measures actual catalog bodies? | Yes. Otherwise see the 5.2 fallback. |
| Q12 | Put the new Swift contract tests in existing `BenchmarkContractTests` files (no CI-policy edits), or in a new file, which also requires `check_ci_contracts.py` and the generated doc to change? | Existing files. |
| Q19 | Keep `fixtureVersion` "V02-inputs-1" and bump only the boundary version? | Yes. |

**Needed before phase D, E or F:**

| # | Question | Recommendation |
|---|---|---|
| Q10 | Include the TOOL-08-class handle-stride assertions in `MixedPrecisionPart1ValidationTests` and `MixedPrecisionKernelsTests` (6.7), together with a mechanical guard? | Include them. The T02-owned `QuantizedKernelsTests` label is deferred. |
| Q11 | Runtime API strings that carry claims: `BatchKernels_SoA.estimatePerformanceGain` ("10-20%"), `SoA.memoryEfficiency`, and `MixedPrecisionKernels` `PrecisionProfile` summaries ("<0.1%", "50%", "75%"). Correcting them changes returned text, which is outside a comments-only scope. Authorize, or ledger them as retained-unverified runtime text? | Root to decide. |
| Q13 | Is `Docs/Performance_Guide.md`'s benchmarking section in V04 scope? It has invalid CLI examples (`vectorcore-bench batch …` with positional suites), refers to a nonexistent `scripts/compare_benchmarks.py`, and gives "indicative" ns figures. | Root to decide. |
| Q14 | Confirm a comment-only correction for `OptimizationAttributes`. Its undefined `SWIFT_RELEASE_MODE` and `VECTORCORE_ENABLE_SIMD` flags make `performancePrecondition` always enforced, `releaseOnly` dead, and `isReleaseMode`/`isSIMDEnabled` always false. | Comment-only. Any behavior change is out of V04 scope. |
| Q15 | Timing, allocation and inspection evidence at 512D only, with 768 and 1536 covered by selection and correctness contract tests? | Yes. |
| Q16 | Leave swiftbuild support in `check_benchmark_build_modes.py` (and the `validate_github.rb` re-pin) to T03? | Yes. |
| Q17 | The 39 timed test statements that discard results (6.7): record them as known weak-consumption observation bodies (ledger and limitations only), or authorize V04 to add consumption? Adding it changes the measured work but no numerical assertion. | Record only. |
| Q18 | Accept the raw-sample fields and the statistic name "per-sample-average-ns-per-invocation"? | Yes. |
| Q20 | Finite driver deadlines: memory probe 12 min, allocation probe 10 min, timing records 10 min, with 15 s per child and at most 2 s cleanup. | Root review. |
| Q21 | Include V04's new `OutputFold.swift` in the ledger regions? | Yes. |
