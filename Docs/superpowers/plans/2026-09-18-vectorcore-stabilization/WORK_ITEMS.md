# Stabilization work-item catalog

[Planning draft](README.md) · [Findings](FINDINGS.md) · [Agent workflow](AGENT_WORKFLOW.md)

**43 candidate work items across 12 workstreams; none has started.** A catalog item is a planning boundary, not a ready-to-run prompt. Its design checkpoint must be settled and a bounded brief must supply concrete regression code, exact write ownership and commands before implementation. Split multi-part items at that checkpoint when changes need independent review.

Dependencies below are conservative integration/closure dependencies. A narrow local investigation or regression design may start earlier; root must record any relaxed implementation dependency. File overlap independently prevents simultaneous edits. Source references include existing tests/docs and downstream read-only context; they are not blanket write permission. Proposed tests are named to make the intended coverage concrete and have not been created.

| Item | Scope | Depends on | Records |
|---|---|---|---|
| [V01](#v01) | Strict correctness and observable benchmark failure | None | TOOL-01, TOOL-02 |
| [V02](#v02) | Benchmark selection, provenance and comparison | V01 | TOOL-03, TOOL-05, TOOL-06, TOOL-09, TOOL-12 |
| [V03](#v03) | Executed CI coverage and performance opt-outs | V01 | TOOL-04, TOOL-11 |
| [V04](#v04) | Honest timing, memory and optimization claims | V01, V02 | TOOL-07, TOOL-08, API-43, NUM-23, OWN-23 |
| [S01](#s01) | Public vector shape invariants | None | OWN-01, OWN-02 |
| [S02](#s02) | Dimension guards, aggregation and factory validation | S01 | OWN-06, API-13, API-19 |
| [S03](#s03) | Decoded result and parameter invariants | S01, Q01 | API-18, SER-08 |
| [O01](#o01) | Compact copy isolation and element lifetime | None | OWN-05, OWN-10, API-47 |
| [O02](#o02) | Tiered bulk writeback and mutation cost | O01 | OWN-13, OWN-14 |
| [O03](#o03) | Hybrid ownership traffic and operation temporaries | S01 | OWN-12, OWN-22, OWN-26 |
| [O04](#o04) | Alignment, binding and copy destination contracts | S01 | OWN-08, OWN-09, OWN-11, OWN-25 |
| [O05](#o05) | Scoped borrow and dormant transfer lifetimes | O03 | OWN-15, OWN-16, OWN-17, OWN-24 |
| [I01](#i01) | Normalization hint invalidation | S01 | OWN-03, API-33 |
| [I02](#i02) | Exact cache identity and bounded retention | S01 | OWN-04, OWN-18, OWN-19 |
| [R01](#r01) | Requested metric witness preservation | None | API-01, API-46 |
| [R02](#r02) | Transactional collection replacement and identifiers | R01 | API-09, API-10 |
| [R03](#r03) | Matrix metric semantics and logical output bounds | None | API-14, API-15, API-16 |
| [R04](#r04) | Bounded heap membership and legacy ordering policy | None | API-17, API-26 |
| [R05](#r05) | Metric-aware relevance scores | R01 | API-11 |
| [E01](#e01) | Chunk and loop arithmetic boundaries | None | API-07, API-08, API-44 |
| [E02](#e02) | Bounded scheduling and execution laws | E01, O01, R03 | API-04, API-27, API-28, API-42, API-48 |
| [E03](#e03) | Atomic configuration transactions | None | API-03 |
| [E04](#e04) | Pool teardown, budgets and handle ownership | None | API-05, API-06, API-32, API-37 |
| [E05](#e05) | Reentrant and concurrent calibration | None | API-29 |
| [E06](#e06) | Validated live operation settings | E01, E02, E03, N03 | API-02 |
| [N01](#n01) | Range-safe division and normalization | S01, S02, O04, I01 | NUM-01, NUM-02, API-39 |
| [N02](#n02) | Finite-aware equality and logarithm policy | S02 | NUM-14, NUM-15 |
| [N03](#n03) | Stable batch population statistics | E03 | NUM-06 |
| [N04](#n04) | Scale-invariant entropy | None | NUM-07 |
| [Q01](#q01) | Quantization ranges and INT8 cosine | S01 | NUM-03, NUM-04, NUM-05 |
| [Q02](#q02) | Precision selection, accuracy and tuner provenance | Q01, N01, E05 | NUM-16, NUM-17, NUM-19, NUM-20, NUM-21, NUM-24 |
| [Q03](#q03) | Race-free rounding-aware FP16 diagnostics | None | NUM-18, API-21 |
| [Q04](#q04) | FP16 range work proportional to requested range | Q01, Q03 | NUM-22 |
| [L01](#l01) | Swift factorization range and boundary contracts | N01 | NUM-08, NUM-09, NUM-35 |
| [L02](#l02) | PCA centering and checked sketch dimensions | L01 | NUM-10, API-20 |
| [L03](#l03) | UMAP calibration equations and input boundaries | L02 | NUM-11, NUM-12, NUM-13, NUM-25, NUM-26, NUM-27, NUM-28 |
| [L04](#l04) | Manifold and PCA memory/work envelope | L03, V04 | NUM-29, NUM-30, NUM-31 |
| [P01](#p01) | Hash quality and streaming selection work | R01, R03, R04, O03, V04 | OWN-07, API-23, API-24, API-45 |
| [P02](#p02) | Explicit batch accuracy, memory and reusable packing | R01, R03, N01, E06, V02, V04 | API-22, API-25, API-41, TOOL-10 |
| [B01](#b01) | Binary round trips, real tests and wire contracts | S01, S03 | SER-01, SER-02, SER-03, SER-04, SER-05, SER-06, SER-07 |
| [D01](#d01) | Pinned downstream contract validation | I01, N01, R01, R05 | API-34, API-35, API-36, API-40 |
| [D02](#d02) | Public API, device and C contract disposition | D01, B01, E06 | API-12, API-30, API-38, OWN-20, OWN-21, OWN-27, NUM-32, NUM-33, NUM-34, TOOL-13 |
| [D03](#d03) | Lazy logging and elapsed-time contract | None | API-31 |

<a id="v01"></a>
## V01 — Strict correctness and observable benchmark failure

**State:** planned. **Depends on:** None.

**Design checkpoint:** Define validation status separately from numeric error statistics; invalid/missing results must fail strict mode. Choose failure artifacts that remain JSON-encodable for nonfinite observations.

**Regression design:** Table of finite match/mismatch, NaN, infinities, both length mismatch directions, missing validation and provider error; invalid destination must exit nonzero. Test CLI propagation as well as helper predicates.

**Source/read scope:**

- `Benchmarks/VectorCoreBench/AppMain.swift`
- `Benchmarks/VectorCoreBench/Suites/BatchBench.swift`
- `Benchmarks/VectorCoreBench/Utils/Correctness.swift`

**Test candidates:**

- `Scripts/ci/tests/test_benchmark_contracts.py` — Proposed; not created.

**Audit acceptance to preserve:**

- [TOOL-01](../../../audits/2026-09-14/CLAIMS.md#tool-01): CLI-level table tests: finite matches/mismatches, NaN, infinities, both length mismatch directions, missing validation and provider failure. Each invalid case must fail strict mode and emit a parseable failure artifact.
- [TOOL-02](../../../audits/2026-09-14/CLAIMS.md#tool-02): Unwritable/invalid destination and invalid numeric statistics must produce nonzero exit and explicit error; successful runs must leave decodable artifacts.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="v02"></a>
## V02 — Benchmark selection, provenance and comparison

**State:** planned. **Depends on:** V01.

**Design checkpoint:** Separate requested settings from effective compiled/routed settings; choose explicit incompatible-baseline override and empty-selection policy.

**Regression design:** Enumerate registered cases/filters and expected counts; Debug/Release plus C-on/off route sentinels; A/B pair uniqueness and hand-calculated deltas; CSV commas/quotes/newlines; seeded matrix diversity.

**Source/read scope:**

- `Benchmarks/VectorCoreBench/AppMain.swift`
- `Benchmarks/VectorCoreBench/Formatters/CSVWriter.swift`
- `Benchmarks/VectorCoreBench/Suites/DotProductBench.swift`
- `Benchmarks/VectorCoreBench/Suites/MatrixDistanceBench.swift`
- `Benchmarks/baselines/arm64-os15.4.1-canonical.json`
- `Benchmarks/baselines/arm64-os15.4.1-local.json`
- `Scripts/compare_benchmarks.swift`
- `Sources/VectorCore/Interop/CKernels.swift`
- `Sources/VectorCoreBenchmarking/ABComparisons.swift`
- `Sources/VectorCoreBenchmarking/CaseParsing.swift`
- `Sources/VectorCoreBenchmarking/EnvCapture.swift`
- `Sources/VectorCoreBenchmarking/Models.swift`

**Test candidates:**

- `Scripts/ci/tests/test_benchmark_contracts.py` — Proposed; not created.
- `Tests/ComprehensiveTests/BenchmarkReportingContractTests.swift` — Proposed; not created.

**Audit acceptance to preserve:**

- [TOOL-03](../../../audits/2026-09-14/CLAIMS.md#tool-03): Build Debug/Release and C-on/C-off fixtures; metadata must identify actual settings, and observable route tests must verify enabled C dispatch. Reject incompatible runtime requests.
- [TOOL-05](../../../audits/2026-09-14/CLAIMS.md#tool-05): Table-driven selection test for every registered case, exact names, excludes and variant filters; expected counts and missing-baseline policy must be asserted.
- [TOOL-06](../../../audits/2026-09-14/CLAIMS.md#tool-06): Every supported A/B family must yield exactly one valid pair; mismatched dimensions/N/provider must not pair; check numeric delta against hand-calculated examples.
- [TOOL-09](../../../audits/2026-09-14/CLAIMS.md#tool-09): Changing controls must change scheduled cases/samples; filters must prevent execution; names or structured fields must preserve dim/N; seeded fixtures must have documented diversity.
- [TOOL-12](../../../audits/2026-09-14/CLAIMS.md#tool-12): Incompatible metadata must be rejected or explicitly overridden; round-trip labels containing commas/quotes/newlines; independently verify chosen statistic and comparison formula.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="v03"></a>
## V03 — Executed CI coverage and performance opt-outs

**State:** planned. **Depends on:** V01.

**Design checkpoint:** Choose required executed checks using current CI capability; do not infer remote branch protection. Register tooling tests deliberately without making wall-clock speed a correctness requirement.

**Regression design:** Inventory executable tests/analysis; EXTENDED=1 and STRICT_PERF=0 must remove every timing pass/fail assertion while retaining numeric checks; policy tests reject missing required jobs.

**Source/read scope:**

- `.github/workflows/benchmarks.yml`
- `.github/workflows/ci.yml`
- `.github/workflows/sanitizers.yml`
- `.swiftlint.yml`
- `Package.swift`
- `Tests/ComprehensiveTests/MixedPrecisionPerformanceBenchmark.swift`
- `Tests/ComprehensiveTests/PerfGate.swift`
- `Tests/PerformanceTests/NSWOptimizationBenchmark.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/PerfGate.swift` — Existing candidate.
- `Scripts/ci/tests/test_github_policy.py` — Proposed; not created.

**Audit acceptance to preserve:**

- [TOOL-04](../../../audits/2026-09-14/CLAIMS.md#tool-04): Mechanically enumerate performance assertions; EXTENDED=1/STRICT_PERF=0 must disable every wall-clock pass/fail gate while retaining numerical checks.
- [TOOL-11](../../../audits/2026-09-14/CLAIMS.md#tool-11): CI inventory must map each promised test/analysis to an executed target/command; mechanical policy tests must reject missing required jobs and test/route coverage gaps.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="v04"></a>
## V04 — Honest timing, memory and optimization claims

**State:** planned. **Depends on:** V01, V02.

**Design checkpoint:** Specify end-to-end versus prepacked timing scope and observability of consumed outputs; distinguish payload models from measured allocations/RSS. Unverified throughput assertions receive evidence or qualified wording.

**Regression design:** Retain optimized-code inspection and allocation traces for representative cases; identical work, warmups, raw samples, median/variation and exact host/toolchain. No inferred dead-code-elimination slowdown.

**Conditional prerequisites:**

- L01: Final inventory/sign-off of robustness/performance claims about this repaired algorithm; initial measurement setup can precede it.
- Q02: Final inventory/sign-off of robustness/performance claims about this repaired algorithm; initial measurement setup can precede it.
- O02: Final inventory/sign-off of robustness/performance claims about this repaired algorithm; initial measurement setup can precede it.
- O03: Final inventory/sign-off of robustness/performance claims about this repaired algorithm; initial measurement setup can precede it.

**Source/read scope:**

- `Benchmarks/VectorCoreBench/Suites/BatchBench.swift`
- `Benchmarks/VectorCoreBench/Suites/MatrixDistanceBench.swift`
- `Sources/VectorCore/LinearAlgebra/PCA.swift`
- `Sources/VectorCore/LinearAlgebra/SwiftLinearAlgebraProvider.swift`
- `Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift`
- `Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift`
- `Sources/VectorCore/Optimization/OptimizationAttributes.swift`
- `Sources/VectorCore/Storage/COWDynamicStorage.swift`
- `Sources/VectorCore/Vectors/DynamicVector.swift`
- `Sources/VectorCore/Vectors/Vector512Optimized.swift`
- `Tests/ComprehensiveTests/MixedPrecisionPerformanceBenchmark.swift`
- `Tests/MinimalTests/ComprehensiveBenchmarks.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/MixedPrecisionPerformanceBenchmark.swift` — Existing candidate.

**Audit acceptance to preserve:**

- [TOOL-07](../../../audits/2026-09-14/CLAIMS.md#tool-07): Inspect optimized code and allocation traces for representative cases; record consumer overhead, warmups, raw samples, machine/toolchain and scope. Compare only equivalent workloads.
- [TOOL-08](../../../audits/2026-09-14/CLAIMS.md#tool-08): Check representation payload size; profile realistic independent/shared datasets and report method, peak/retained bytes and workload.
- [API-43](../../../audits/2026-09-14/CLAIMS.md#api-43): Release benchmark artifacts and emitted instruction check support each retained claim.
- [NUM-23](../../../audits/2026-09-14/CLAIMS.md#num-23): Mechanical claim inventory: each guarantee has nearby one-step proof, named test, or explicit unverified-contract wording; performance records carry reproducible provenance.
- [OWN-23](../../../audits/2026-09-14/CLAIMS.md#own-23): Mechanical claim inventory names evidence for each retained correctness/performance assertion; docs reflect actual call graph; no unsupported universal assertion remains.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="s01"></a>
## S01 — Public vector shape invariants

**State:** planned. **Depends on:** None.

**Design checkpoint:** Approve dynamic resize semantics and fixed-width setter/constructor/decoding policy before edits. Preserve valid input compatibility; no silent truncation or padding.

**Regression design:** Short/equal/long construction, replacement, nested mutation and Codable at all FP32/INT8 widths; dimension, owned count and borrowed count agree. Generic fixed and immutable FP16 controls.

**Source/read scope:**

- `Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift`
- `Sources/VectorCore/Protocols/VectorProtocol.swift`
- `Sources/VectorCore/Vectors/DynamicVector.swift`
- `Sources/VectorCore/Vectors/Vector1536Optimized.swift`
- `Sources/VectorCore/Vectors/Vector384Optimized.swift`
- `Sources/VectorCore/Vectors/Vector512Optimized.swift`
- `Sources/VectorCore/Vectors/Vector768Optimized.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/VectorShapeContractTests.swift` — Proposed; not created.
- `Tests/ComprehensiveTests/DynamicVectorTests.swift` — Existing candidate.
- `Tests/ComprehensiveTests/QuantizedKernelsTests.swift` — Existing candidate.

**Audit acceptance to preserve:**

- [OWN-01](../../../audits/2026-09-14/CLAIMS.md#own-01): Mechanical matrix for all families and wrappers: short/equal/long replacement and direct collection edits; dimension/count/buffer/serialization agree; equal values hash equally; invalid shapes are rejected before memory access. Test generic protocol mutation as well.
- [OWN-02](../../../audits/2026-09-14/CLAIMS.md#own-02): Each INT8 width rejects short/long raw storage and JSON storage; valid round trips preserve dimension, payload and parameters. Exercise post-construction collection mutation and all unsafe conversion/kernel entry points.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="s02"></a>
## S02 — Dimension guards, aggregation and factory validation

**State:** planned. **Depends on:** S01.

**Design checkpoint:** Define checked failure behavior for existing throwing/nonthrowing APIs and empty fixed/dynamic aggregates. Complete validation precedes writes or unsafe reads.

**Regression design:** Direct/generic +=/-=/Hadamard, short/long rows at every position, empty input, failing custom factory, zero/negative factory dimensions; preserve receiver on rejected preflight.

**Source/read scope:**

- `Sources/VectorCore/Factory/VectorTypeFactory.swift`
- `Sources/VectorCore/Operations/Operations.swift`
- `Sources/VectorCore/Protocols/VectorProtocol.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/VectorDispatchContractTests.swift` — Proposed; not created.
- `Tests/ComprehensiveTests/OperationsValidationTests.swift` — Existing candidate.
- `Tests/ComprehensiveTests/VectorTypeFactoryTests.swift` — Existing candidate.

**Audit acceptance to preserve:**

- [OWN-06](../../../audits/2026-09-14/CLAIMS.md#own-06): Equal/short/long RHS, empty combinations and direct/generic +=/-= plus all three methods. Invalid dimensions fail before any write; subprocess tests for trapping API or recoverable errors for checked API.
- [API-13](../../../audits/2026-09-14/CLAIMS.md#api-13): Short/long row at first/middle/last, empty dynamic/fixed arrays, failing custom factory; no unsafe operation before all shape checks.
- [API-19](../../../audits/2026-09-14/CLAIMS.md#api-19): Zero/negative dimension with empty/nonempty values, valid multiples and mismatches; ensure throws instead of trap where documented.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="s03"></a>
## S03 — Decoded result and parameter invariants

**State:** planned. **Depends on:** S01, Q01.

**Design checkpoint:** Define valid serialized parallel arrays and derived quantization parameters; coordinate with S01/Q01 instead of duplicating their decoder fixes.

**Regression design:** Both TopK array mismatch directions, valid empty/normal round trips; mechanical representation table includes INT8 widths, shape and quantization parameters.

**Source/read scope:**

- `Sources/VectorCore/Integration/SearchResult.swift`
- `Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift`
- `Sources/VectorCore/Operations/TopKSelection.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/DecodedInvariantTests.swift` — Proposed; not created.
- `Tests/ComprehensiveTests/TopKSelectionTests.swift` — Existing candidate.

**Audit acceptance to preserve:**

- [API-18](../../../audits/2026-09-14/CLAIMS.md#api-18): Both mismatch directions, empty valid data, normal round-trip and collection access; class-wide serialized invariant audit.
- [SER-08](../../../audits/2026-09-14/CLAIMS.md#ser-08): Table-driven invalid shape/derived-parameter/result-pair decoding tests across every representation; valid existing data remains compatible.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="o01"></a>
## O01 — Compact copy isolation and element lifetime

**State:** planned. **Depends on:** None.

**Design checkpoint:** Repair uniqueness at the actual shared owner and destruction at the owner of initialized tail elements. Keep public triangular value semantics.

**Regression design:** Every compact count 5...512, boundaries 0/1/4/5/512/513/2048/2049, public matrix sizes 3/4/32/33/64/65; copies, setAt and iterator snapshots; class sentinels release exactly once; singleton import round trip.

**Staged closure:** O01 compact repair can finish independently; full OWN-05 class-wide acceptance remains open until O02 supplies standard/aligned bulk and single-detach evidence.

**Source/read scope:**

- `Sources/VectorCore/Utilities/TieredStorage.swift`
- `Sources/VectorCore/Utilities/TriangularMatrix.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/TieredStorageOwnershipTests.swift` — Proposed; not created.

**Audit acceptance to preserve:**

- [OWN-05](../../../audits/2026-09-14/CLAIMS.md#own-05): COW tests across tier counts0,1,4,5,512,513,2048,2049 and matrix n3,4,32,33,64,65 for subscript,setAt,bulk and preexisting iterators; original unchanged and first mutation detaches once.
- [OWN-10](../../../audits/2026-09-14/CLAIMS.md#own-10): Deinit/weak sentinel tests for all tiers, compact copy/detach/replacement/destruction and thrown access; each logical initialized reference released exactly once, no early destruction.
- [API-47](../../../audits/2026-09-14/CLAIMS.md#api-47): Empty, singleton,2x2, ragged inputs and singleton roundtrip to zero-diagonal full matrix; no index trap.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="o02"></a>
## O02 — Tiered bulk writeback and mutation cost

**State:** planned. **Depends on:** O01.

**Design checkpoint:** Choose consistent partial-write behavior on thrown mutable borrows. Repair dormant bulk writeback before routing public work through it; repair both standard and aligned tiers.

**Regression design:** Normal/throwing closures at every tier boundary with shared-copy controls; unique scalar writes avoid full copies at matrix 33/64/65; shared values detach correctly. Release allocation evidence, not pointer stability alone.

**Source/read scope:**

- `Sources/VectorCore/Utilities/TieredStorage.swift`
- `Sources/VectorCore/Utilities/TriangularMatrix.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/TieredStorageMutationTests.swift` — Proposed; not created.

**Audit acceptance to preserve:**

- [OWN-13](../../../audits/2026-09-14/CLAIMS.md#own-13): Across all tier boundaries unique writes perform no full-buffer detaches; retained copy produces exactly one detach and remains unchanged. Construction traffic scales linearly with number of stored elements. Benchmark only after allocation and value-semantic checks.
- [OWN-14](../../../audits/2026-09-14/CLAIMS.md#own-14): Normal and throwing closure tests at0,1,4,5,512,513,2048,2049 with retained original copies, consistent mutation policy and independent value semantics.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="o03"></a>
## O03 — Hybrid ownership traffic and operation temporaries

**State:** planned. **Depends on:** S01.

**Design checkpoint:** Keep default generic DimensionStorage as a negative control; OWN-26 is rejected and creates no blanket rewrite. Reduce retained-owner artifacts in actual HybridStorage paths.

**Regression design:** Release allocation scaling for constructor/fill/resize/scalar/bulk writes, Float16/17 and fixed widths, Double8/9/16/17, unique/shared controls and explicit hybrid/default modes.

**Source/read scope:**

- `Sources/VectorCore/Operations/VectorCoreOperations.swift`
- `Sources/VectorCore/Storage/DimensionStorage.swift`
- `Sources/VectorCore/Storage/HybridStorage.swift`
- `Sources/VectorCore/Vectors/DynamicVector.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/StorageAllocationContractTests.swift` — Proposed; not created.
- `Tests/ComprehensiveTests/DynamicVectorTests.swift` — Existing candidate.

**Audit acceptance to preserve:**

- [OWN-12](../../../audits/2026-09-14/CLAIMS.md#own-12): Optimized allocation-scaling checks at Float16/17 and representative384/512/768/1536; unique scalar writes/fill stay bounded, shared first mutation detaches once, subsequent unique writes do not copy. No wall-clock speedup asserted until coordinated measurement.
- [OWN-22](../../../audits/2026-09-14/CLAIMS.md#own-22): Allocation records for representative dimensions show fewer temporaries without changing provider call counts, shape checks or outputs.
- [OWN-26](../../../audits/2026-09-14/CLAIMS.md#own-26): Default and explicit modes at Float16/17 and Double8/9/16/17; constructor,factory,scalar and buffer mutation counts measured in Release with retained-copy controls.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="o04"></a>
## O04 — Alignment, binding and copy destination contracts

**State:** planned. **Depends on:** S01.

**Design checkpoint:** Choose unaligned support versus explicit preconditions, scoped binding rules, initialize/update/overlap policy and allocator fallback behavior. Investigate OWN-25 before claiming a defect or fix.

**Regression design:** Float offsets 0/4/8/12 over SIMD tails; all-width round trips; nontrivial destination lifetime sentinels; allocator failure injection if feasible. Preserve Debug-only reproduced crash qualification.

**Source/read scope:**

- `Sources/VectorCore/Operations/Kernels/NormalizeKernels.swift`
- `Sources/VectorCore/Storage/AlignedDynamicArrayStorage.swift`
- `Sources/VectorCore/Utilities/MemoryUtilities.swift`
- `Sources/VectorCore/Vectors/Vector384Optimized.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/UnsafeMemoryContractTests.swift` — Proposed; not created.
- `Tests/ComprehensiveTests/StorageAlignmentTests.swift` — Existing candidate.

**Audit acceptance to preserve:**

- [OWN-08](../../../audits/2026-09-14/CLAIMS.md#own-08): Offsets0,4,8,12 and dimensions0...5,15...17 in Debug/Release either compute correct finite unit values under supported inputs or reject consistently at a documented boundary; retain numerical reviewer coverage.
- [OWN-09](../../../audits/2026-09-14/CLAIMS.md#own-09): Mechanical audit of typed-owner rebinding, scoped nonescape and alignment obligations plus constructor/read/write/COW Release round trips for all widths. Document the permitted type/layout model.
- [OWN-11](../../../audits/2026-09-14/CLAIMS.md#own-11): Class/deinit and nontrivial-value tests at preferred and natural-only alignment, using only the chosen destination state. Identical lifetime behavior in both branches; overlap behavior explicitly tested if supported.
- [OWN-25](../../../audits/2026-09-14/CLAIMS.md#own-25): Inject allocator failure and validate reported guarantee against returned pointer for several requested alignments; callers never rely on stronger alignment than provided.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="o05"></a>
## O05 — Scoped borrow and dormant transfer lifetimes

**State:** planned. **Depends on:** O03.

**Design checkpoint:** Prefer scoped access and retain disabled experimental gates. Decide deprecation for public escaped getters; retain documented unsafe/custom-conformer duties rather than creating unsupported safety promises.

**Regression design:** Owner/snapshot lifetime through success/throw/cancellation where supported; exactly-once transfer/destruction before any gate activation. No supported-use UAF is currently reproduced.

**Source/read scope:**

- `Docs/Memory_Alignment.md`
- `Docs/SoA_Layout_Contract.md`
- `Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift`
- `Sources/VectorCore/Storage/DimensionStorage.swift`
- `Sources/VectorCore/Storage/SoA.swift`
- `Sources/VectorCore/Storage/UnifiedStorage.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/StorageBorrowContractTests.swift` — Proposed; not created.
- `Tests/ComprehensiveTests/SoALayoutContractTests.swift` — Existing candidate.

**Audit acceptance to preserve:**

- [OWN-15](../../../audits/2026-09-14/CLAIMS.md#own-15): Compile-time nonescape design or scoped lifetime tests for inline/heap slices; no pointer survives its borrow. If retaining owner, prove owner retention and copy isolation.
- [OWN-16](../../../audits/2026-09-14/CLAIMS.md#own-16): Gate-enabled separate fixture proves pointer survives consumed wrapper, is freed once by recipient, and nontransferred wrapper still frees correctly; audit every newly compiled branch.
- [OWN-17](../../../audits/2026-09-14/CLAIMS.md#own-17): All direct kernels and callers use scoped borrows or explicit owned allocation lifetime. Temporary-expression use cannot silently escape through new safe API. Test512/768/1536 content and owner lifetime.
- [OWN-24](../../../audits/2026-09-14/CLAIMS.md#own-24): Transfer identity, logical vs allocated length and frozen layout tests remain; external conformance examples assert required lane/dimension/storage consistency. New checked wrappers validate before arithmetic.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="i01"></a>
## I01 — Normalization hint invalidation

**State:** planned. **Depends on:** S01.

**Design checkpoint:** Invalidate metadata before mutation or on every exit, including throws. Define decoded hints and external alias/custom-conformer obligations. Narrow Core repair may precede S01, but full closure requires its contract.

**Regression design:** Initially valid [1,0] then replacement/subscript/storage writes and throwing buffer writes; all representations/direct/generic routes. Compare hinted/unhinted pinned Index and VA cosine consumers separately.

**Source/read scope:**

- `../VectorIndex/Sources/VectorIndex/HNSWIndex.swift`
- `../VectorIndex/Sources/VectorIndex/TypedOverloads.swift`
- `../future/VectorAccelerate/Sources/VectorAccelerate/Integration/KernelDistanceProviders.swift`
- `Sources/VectorCore/Integration/IndexableVector.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/IntegrationProtocolsTests.swift` — Existing candidate.
- `Tests/ComprehensiveTests/NormalizationHintMutationTests.swift` — Proposed; not created.

**Audit acceptance to preserve:**

- [OWN-03](../../../audits/2026-09-14/CLAIMS.md#own-03): Test direct vector replacement/subscript/storage, wrapper storage, successful and throwing buffer mutation on every built-in representation; after changes metadata must be cleared or recomputed. Downstream insertion/query tests compare fresh normalized reference.
- [API-33](../../../audits/2026-09-14/CLAIMS.md#api-33): Mutate initially correct hints and compare hinted/unhinted insertion+queries, including HNSW live cache and VA cosine, throwing mutations.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="i02"></a>
## I02 — Exact cache identity and bounded retention

**State:** planned. **Depends on:** S01.

**Design checkpoint:** Choose immutable equality-verified snapshots or trusted versioned identity; define zero/negative capacity and count/byte budgets. Include retained identity snapshots in accounting.

**Regression design:** Change each of 512 coordinates, reorder/resize candidates, mutate retained snapshots, force digest collisions; identical input still hits. Capacity 0/1, replacements/evictions and byte accounting.

**Source/read scope:**

- `Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift`
- `Sources/VectorCore/Operations/Kernels/SoAFP16Cache.swift`
- `Sources/VectorCore/Vectors/Vector512Optimized.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/SoAFP16CacheIdentityTests.swift` — Proposed; not created.
- `Tests/ComprehensiveTests/MixedPrecisionPhase3Tests.swift` — Existing candidate.

**Audit acceptance to preserve:**

- [OWN-04](../../../audits/2026-09-14/CLAIMS.md#own-04): For every coordinate, reorder, count change and retained snapshot mutation, getOrCreate must match uncached conversion. Inject deliberate hash collisions to prove equality checking; repeated identical content still hits.
- [OWN-18](../../../audits/2026-09-14/CLAIMS.md#own-18): Fresh empty/nonempty stores at negative,zero,one capacities have documented behavior, repeated eviction and statistics remain consistent.
- [OWN-19](../../../audits/2026-09-14/CLAIMS.md#own-19): Tests with small byte budget cap retained payload; oversized entries follow explicit policy; statistics and eviction accounting agree with logical/allocated sizes. No claim of measured RSS.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="r01"></a>
## R01 — Requested metric witness preservation

**State:** planned. **Depends on:** None.

**Design checkpoint:** Choose source-compatible handling of any DistanceMetric with a non-Float associated scalar; do not silently substitute Euclidean or rebuild stateful witnesses.

**Regression design:** Chebyshev query [0,0], candidates [3,3] and [4,0] selects first, unlike Euclidean; every built-in, nondefault Minkowski and custom reversed metric agree with direct scores/membership through both facades. Factory name round trips.

**Source/read scope:**

- `Sources/VectorCore/Integration/VectorCollection.swift`
- `Sources/VectorCore/Operations/OptimizedDistanceMetrics.swift`
- `Sources/VectorCore/Operations/TopKSelection.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/MetricDispatchContractTests.swift` — Proposed; not created.
- `Tests/ComprehensiveTests/TopKSelectionTests.swift` — Existing candidate.
- `Tests/ComprehensiveTests/IntegrationProtocolsTests.swift` — Existing candidate.

**Audit acceptance to preserve:**

- [API-01](../../../audits/2026-09-14/CLAIMS.md#api-01): Shared facade table for all built-ins, nondefault Minkowski, and custom reverse-order metric; score and member equality versus direct witness. Preserve provider route.
- [API-46](../../../audits/2026-09-14/CLAIMS.md#api-46): Case-insensitive roundtrip for every MetricType.allCases and unknown names returnnil.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="r02"></a>
## R02 — Transactional collection replacement and identifiers

**State:** planned. **Depends on:** R01.

**Design checkpoint:** Decide replacement primitive, snapshot/concurrency semantics, duplicate explicit ID and exhaustion policy; generic remove-then-add cannot promise rollback.

**Regression design:** Inject every add/replacement failure while preserving old value; explicit/generated ID permutations, sparse IDs, remove/reinsert and duplicate policy. Reference collection only.

**Source/read scope:**

- `Sources/VectorCore/Integration/VectorCollection.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/VectorCollectionTransactionTests.swift` — Proposed; not created.

**Audit acceptance to preserve:**

- [API-09](../../../audits/2026-09-14/CLAIMS.md#api-09): Failure injection for all thrown replacement paths preserves old data; concurrent update/search snapshot policy defined.
- [API-10](../../../audits/2026-09-14/CLAIMS.md#api-10): Explicit/generated permutations, sparse IDs, repeated explicit insertion, remove/reinsert, exhausted allocator strategy. Update must use replacement primitive rather than duplicate rejection.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="r03"></a>
## R03 — Matrix metric semantics and logical output bounds

**State:** planned. **Depends on:** None.

**Design checkpoint:** Define cosine zero behavior and metric-specific diagonals; keep documented symmetric-matrix scope. Write only logical output elements.

**Regression design:** Built-in direct-pair oracles across N99/100 and GEMM thresholds; identical/orthogonal/opposite vectors; 512/768/1536 zero-query sentinels at N0/1/3/4/5 and tile255/256/257 with oversized/offset outputs.

**Source/read scope:**

- `Sources/VectorCore/Operations/BatchOperations.swift`
- `Sources/VectorCore/Operations/ExecutionOperations.swift`
- `Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift`
- `Sources/VectorCore/Operations/TransformOperations.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/MatrixMetricContractTests.swift` — Proposed; not created.
- `Tests/ComprehensiveTests/BatchKernels_SoATests.swift` — Existing candidate.

**Audit acceptance to preserve:**

- [API-14](../../../audits/2026-09-14/CLAIMS.md#api-14): Dimension512/768/1536, N0/1/3/4/5, tile255/256/257 and actual byte thresholds; oversized/offset outputs and zero/nonzero query sentinels.
- [API-15](../../../audits/2026-09-14/CLAIMS.md#api-15): All built-ins direct-pair oracle around N99/100 and GEMM gates; include zero vectors and dot-product diagonal. Do not assume every cosine route treats zero self as0.
- [API-16](../../../audits/2026-09-14/CLAIMS.md#api-16): Identical, orthogonal, opposite nonzero vectors and defined zero behavior through serial/parallel matrix gates.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="r04"></a>
## R04 — Bounded heap membership and legacy ordering policy

**State:** planned. **Depends on:** None.

**Design checkpoint:** Define nonnegative capacity and legacy NaN/tie policy without silently transplanting another selector contract.

**Regression design:** Bulk/stream membership for permutations, comparator orientations and capacities0/1/n/>n; ties/nonfinite policy; bounded subprocess for invalid negative capacity.

**Source/read scope:**

- `Sources/VectorCore/Operations/MinHeap.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/HeapMembershipContractTests.swift` — Proposed; not created.

**Audit acceptance to preserve:**

- [API-17](../../../audits/2026-09-14/CLAIMS.md#api-17): Bulk/stream membership for permutations, comparators, ties, capacity0/1/n/>n; bounded subprocess negative capacity rejection.
- [API-26](../../../audits/2026-09-14/CLAIMS.md#api-26): NaN/tie/finite policy tests if retained; route inventory ensures high-level searches stay on intended selector.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="r05"></a>
## R05 — Metric-aware relevance scores

**State:** planned. **Depends on:** R01.

**Design checkpoint:** Choose explicit score or metric metadata policy and documented range. Preserve explicitly supplied scores; no universal metric-free transform over signed distances.

**Regression design:** Search-to-score monotonicity and finiteness across negative-dot pole and nonnegative distance families using actual collection producer.

**Source/read scope:**

- `Sources/VectorCore/Integration/SearchResult.swift`
- `Sources/VectorCore/Integration/VectorCollection.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/SearchResultScoreContractTests.swift` — Proposed; not created.

**Audit acceptance to preserve:**

- [API-11](../../../audits/2026-09-14/CLAIMS.md#api-11): Search-to-score monotonicity, finiteness, documented range across negative-dot pole and nonnegative families; preserve explicitly supplied score.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="e01"></a>
## E01 — Chunk and loop arithmetic boundaries

**State:** planned. **Depends on:** None.

**Design checkpoint:** Specify positive batch/unroll/chunk domains, empty behavior and nonrepresentable Range.count policy before endpoint arithmetic.

**Regression design:** All five CPU helpers on tiny ranges near Int.max, huge hints and exact index coverage; batch0/-1/1/>count around N999/1000; unroll widths1...17 with tails and throws. Invalid hangs/traps isolated.

**Source/read scope:**

- `Sources/VectorCore/Operations/BatchOperations.swift`
- `Sources/VectorCore/Optimization/OptimizationAttributes.swift`
- `Sources/VectorCore/Protocols/ComputeProvider.swift`
- `Sources/VectorCore/Providers/CPUComputeProvider.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/ExecutionBoundaryTests.swift` — Proposed; not created.

**Audit acceptance to preserve:**

- [API-07](../../../audits/2026-09-14/CLAIMS.md#api-07): 0,-1,1,>count on N0,1,999,1000; exact-once ordered coverage when valid. Handle oversized positive endpoint addition too.
- [API-08](../../../audits/2026-09-14/CLAIMS.md#api-08): All five methods, zero/nonzero/near-max lower bounds, huge hints, exact index coverage/order; test protocol defaults separately. Empty and nonrepresentable Range.count cases need a defined policy.
- [API-44](../../../audits/2026-09-14/CLAIMS.md#api-44): Emitted indices/flattened ranges exactly0..<count for widths1...17, tail sizes, zero/negative rejection and throwing body. Avoid executing huge count merely to test endpoint arithmetic.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="e02"></a>
## E02 — Bounded scheduling and execution laws

**State:** planned. **Depends on:** E01, O01, R03.

**Design checkpoint:** Define outstanding-work rather than CPU-thread bounds; ordering, cooperative cancellation, nested calls and reduction identity/associativity. Keep dormant engines dormant unless deliberately adopted.

**Regression design:** Active-body/enqueued-task counters for defaults, batch/map/filter, matrix builders and nested routes; exact ordered coverage, cancellation cleanup and reduction laws. Spy executeBatch routes.

**Source/read scope:**

- `../future/VectorAccelerate/Sources/VectorAccelerate/Integration/MetalComputeProvider.swift`
- `Sources/VectorCore/Operations/BatchOperations.swift`
- `Sources/VectorCore/Operations/ExecutionOperations.swift`
- `Sources/VectorCore/Operations/SyncBatchOperations.swift`
- `Sources/VectorCore/Operations/TransformOperations.swift`
- `Sources/VectorCore/Protocols/BatchKernelProvider.swift`
- `Sources/VectorCore/Protocols/ComputeProvider.swift`
- `Sources/VectorCore/Providers/CPUComputeProvider.swift`
- `Sources/VectorCore/Utilities/TriangularMatrix.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/ExecutionSchedulingContractTests.swift` — Proposed; not created.
- `Tests/ComprehensiveTests/ProviderConformanceTests.swift` — Existing candidate.

**Audit acceptance to preserve:**

- [API-04](../../../audits/2026-09-14/CLAIMS.md#api-04): Instrument active bodies and enqueued tasks for default conformers, public batches, matrix builders, nested routes; verify bound and exact ordered coverage.
- [API-27](../../../audits/2026-09-14/CLAIMS.md#api-27): Cancel before start/during long chunks; bounded new work and cleanup under chosen contract; user body obligations stated.
- [API-28](../../../audits/2026-09-14/CLAIMS.md#api-28): Spy range kernel shows documented calls and settings; Swift6 pointer ownership/concurrency validation before enabling parallel path.
- [API-42](../../../audits/2026-09-14/CLAIMS.md#api-42): Mechanical caller map and property tests for any retained reachable engine; shape validation exhaustive before unsafe operations and reduction identity/associativity documented.
- [API-48](../../../audits/2026-09-14/CLAIMS.md#api-48): Conformance tests for identity and chunk combination; documented limits for nonassociative floating-point reductions.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="e03"></a>
## E03 — Atomic configuration transactions

**State:** planned. **Depends on:** None.

**Design checkpoint:** Choose actor-isolated candidate mutation and Sendable closure contract. If updates reject invalid settings, validate the whole candidate before publication; rejection preserves the complete previous configuration. E06 separately validates consumed settings before unsafe arithmetic.

**Regression design:** 100 concurrent increments and independent-field updates preserve all changes; rejected updates preserve every prior field under the chosen policy; external Swift6 consumer typechecks.

**Source/read scope:**

- `Sources/VectorCore/Configuration/ThreadSafeConfiguration.swift`
- `Sources/VectorCore/Operations/BatchOperations.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/ConfigurationTransactionTests.swift` — Proposed; not created.

**Audit acceptance to preserve:**

- [API-03](../../../audits/2026-09-14/CLAIMS.md#api-03): 100 concurrent increments and independent-field updates retain all changes; Swift6 consumer typecheck.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="e04"></a>
## E04 — Pool teardown, budgets and handle ownership

**State:** planned. **Depends on:** None.

**Design checkpoint:** Separate cached raw buffers from checked-out handles and initialized generic elements. Define active-handle lifetime and zero-budget policy before teardown; profile MemoryPool cleanup before optimizing it.

**Regression design:** Allocation/free balance after return/clear/destruction in each pool, zero/exact/mixed buckets and bursts; live-handle success/throw/cancel return once. Compile buffer-provider docs.

**Source/read scope:**

- `../future/VectorAccelerate/Sources/VectorAccelerate/Core/BufferPool.swift`
- `Sources/VectorCore/Operations/BufferPool.swift`
- `Sources/VectorCore/Protocols/BufferProvider.swift`
- `Sources/VectorCore/Providers/SwiftBufferPool.swift`
- `Sources/VectorCore/Utilities/MemoryPool.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/BufferPoolOwnershipTests.swift` — Proposed; not created.
- `Tests/ComprehensiveTests/MemoryPoolTests.swift` — Existing candidate.

**Audit acceptance to preserve:**

- [API-05](../../../audits/2026-09-14/CLAIMS.md#api-05): Count allocate/free for empty, returned buffers, clear then destruction, destruction without clear for each pool; initialized generic-element lifetime tested separately.
- [API-06](../../../audits/2026-09-14/CLAIMS.md#api-06): Zero/exact/mixed bucket budgets; concurrent burst; no release leaves cache above cap. Verify live-handle and returned-memory accounting.
- [API-32](../../../audits/2026-09-14/CLAIMS.md#api-32): Release allocation/contention profile; if changed, retained-byte invariant across type/size buckets and teardown preserved.
- [API-37](../../../audits/2026-09-14/CLAIMS.md#api-37): Swift6 doc consumer compile; success/throw/cancel each releases once, owner retained through operation. Active-handle policy coordinated with teardown.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="e05"></a>
## E05 — Reentrant and concurrent calibration

**State:** planned. **Depends on:** None.

**Design checkpoint:** Do not invoke arbitrary calibrators while holding a nonrecursive lock; decide single-flight/publication and same-key reentrancy policy.

**Regression design:** Bounded subprocess direct reentrancy and custom metric callback; unrelated-key concurrency, duplicate publication, slow callback and failure cleanup.

**Source/read scope:**

- `Sources/VectorCore/Operations/Operations.swift`
- `Sources/VectorCore/Providers/AutoTuning.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/AutoTuningConcurrencyTests.swift` — Proposed; not created.

**Audit acceptance to preserve:**

- [API-29](../../../audits/2026-09-14/CLAIMS.md#api-29): Bounded subprocess custom metric querying tuner plus direct reentrant calibrator; unrelated-key concurrency; no duplicate publication/race; slow callback policy test.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="e06"></a>
## E06 — Validated live operation settings

**State:** planned. **Depends on:** E01, E02, E03, N03.

**Design checkpoint:** Inventory effective knobs; take one validated live snapshot per operation. Re-derive all newly reachable branches before connecting previously ignored settings.

**Regression design:** Behavioral spy per setting; reject zero/negative/nonfinite/huge oversubscription before division/conversion/shift/addition; rerun serial/parallel statistics and routing boundaries.

**Source/read scope:**

- `Sources/VectorCore/Operations/BatchOperations.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/BatchConfigurationContractTests.swift` — Proposed; not created.

**Audit acceptance to preserve:**

- [API-02](../../../audits/2026-09-14/CLAIMS.md#api-02): Behavior tests per setting; reject zero/nonfinite/negative/huge oversubscription, invalid batch/chunk sizes before division, conversion, shift or addition.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="n01"></a>
## N01 — Range-safe division and normalization

**State:** planned. **Depends on:** S01, S02, O04, I01.

**Design checkpoint:** Set scalar quotient, zero/nonfinite and normalization direction/unit-norm contracts by route; full norm/reciprocal representability is unnecessary. Core and VA CPU parity are separate checks.

**Regression design:** Float/Double exponent sweeps over /, /=, direct divide, generic/concrete/array/pointer routes; [s,0]/s at1e-40, repeated1e-38 and greatestFinite, widths1/4/5/17 plus fixed widths. Use widened references for Float, scaled/exponent-based references for Double extremes, and explicitly finite-aware residuals. vDSP substitution is not closure.

**Source/read scope:**

- `../future/VectorAccelerate/Sources/VectorAccelerate/Core/FallbackProvider.swift`
- `../future/VectorAccelerate/Sources/VectorAccelerate/Integration/VectorCoreIntegration.swift`
- `../future/VectorAccelerate/Sources/VectorAccelerate/Operations/AccelerateFallback.swift`
- `../future/VectorAccelerate/Sources/VectorAccelerate/Operations/SIMDFallback.swift`
- `../future/VectorAccelerate/Sources/VectorAccelerate/Operations/StableNormalization.swift`
- `Sources/VectorCore/Operations/Kernels/NormalizeKernels.swift`
- `Sources/VectorCore/Platform/AccelerateSIMDProvider.swift`
- `Sources/VectorCore/Platform/ArraySIMDProvider.swift`
- `Sources/VectorCore/Platform/SIMDProvider.swift`
- `Sources/VectorCore/Platform/SwiftFloatSIMDProvider.swift`
- `Sources/VectorCore/Protocols/VectorProtocol.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/DivisionRangeContractTests.swift` — Proposed; not created.
- `Tests/ComprehensiveTests/NormalizationRangeContractTests.swift` — Proposed; not created.
- `Tests/ComprehensiveTests/NumericalReferenceFixtures.swift` — Proposed; not created.

**Audit acceptance to preserve:**

- [NUM-01](../../../audits/2026-09-14/CLAIMS.md#num-01): Provider/type/operator conformance matrix Float/Double, normal/subnormal divisors, zero numerators and tails; compare representable quotients to widened scalar reference; test /= separately from inPlaceDivide.
- [NUM-02](../../../audits/2026-09-14/CLAIMS.md#num-02): Across typed/generic/pointer paths and widths/tails, assert unit norm in Double, positive alignment with original direction, zero/nonfinite policy, and unchanged-on-failure if promised. Include least subnormal,1e-38,maxFinite.
- [API-39](../../../audits/2026-09-14/CLAIMS.md#api-39): Exponent-range direction/norm checks through actual Core/VA CPU paths, optional GPU separately; no false true normalization hints.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="n02"></a>
## N02 — Finite-aware equality and logarithm policy

**State:** planned. **Depends on:** S02.

**Design checkpoint:** Resolve strict safeLog ambiguity and matching-infinity/tolerance policy; do not enable dormant validators to repair their contract.

**Regression design:** NaN/finite/±Inf/signed-zero cross product with negative/NaN/infinite tolerances; every log option on zero/negative/nonfinite and mixed values.

**Source/read scope:**

- `Sources/VectorCore/Operations/Kernels/MixedPrecisionValidation.swift`
- `Sources/VectorCore/Operations/NaNInfinityHandling.swift`
- `Sources/VectorCore/Protocols/VectorProtocol.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/NumericalPolicyContractTests.swift` — Proposed; not created.

**Audit acceptance to preserve:**

- [NUM-14](../../../audits/2026-09-14/CLAIMS.md#num-14): Mechanical helpers matrix finite/NaN/±Inf/signedzero and invalid tolerances across dimensions; distances and vectors use consistent declared rules.
- [NUM-15](../../../audits/2026-09-14/CLAIMS.md#num-15): All presets on negative/zero/NaN/±Inf/mixed vectors; if strict keeps general meaning, strict success must imply finite result.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="n03"></a>
## N03 — Stable batch population statistics

**State:** planned. **Depends on:** E03.

**Design checkpoint:** Use a stable serial and mergeable parallel variance design; fix the population statistic, not component covariance. Do not use raw-moment implementation as oracle.

**Regression design:** Alternating10000/10001 and identical Float1e18 at999/1000/1024, shuffled orders and chunk counts against wider/scaled reference; both branches finite and accurate within specified tolerance.

**Source/read scope:**

- `Sources/VectorCore/Operations/BatchOperations.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/BatchStatisticsRangeTests.swift` — Proposed; not created.
- `Tests/ComprehensiveTests/NumericalReferenceFixtures.swift` — Proposed; not created.

**Audit acceptance to preserve:**

- [NUM-06](../../../audits/2026-09-14/CLAIMS.md#num-06): Both sides of999/1000 and multiple chunk counts, shuffled order, constants and large offsets; compare Double population standard deviation and assert finite results where reference is representable.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="n04"></a>
## N04 — Scale-invariant entropy

**State:** planned. **Depends on:** None.

**Design checkpoint:** Separate normalized-probability cutoff policy from absolute input scale. Existing manualEntropy repeats the defect and cannot be the independent oracle.

**Regression design:** Uniform finite nonzero scales produce log(d), zero padding and common-scale invariance around dimension64; extremes and chosen nonfinite/zero policy.

**Source/read scope:**

- `Sources/VectorCore/Operations/VectorEntropy.swift`
- `Sources/VectorCore/Operations/VectorMath.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/VectorEntropyTests.swift` — Existing candidate.
- `Tests/ComprehensiveTests/NumericalReferenceFixtures.swift` — Proposed; not created.

**Audit acceptance to preserve:**

- [NUM-07](../../../audits/2026-09-14/CLAIMS.md#num-07): Scale sweeps over finite exponent range and zero-padding across64; compare normalized Double entropy, include all-zero/nonfinite and existing tiny-probability tests.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="q01"></a>
## Q01 — Quantization ranges and INT8 cosine

**State:** planned. **Depends on:** S01.

**Design checkpoint:** Define calibration interval/constant behavior, finite outlier conversion and valid decoded parameters for both public APIs; cosine uses the decoded-vector mathematical reference.

**Regression design:** Positive/negative/cross-zero/constant intervals, [1,2], finite1e20 outliers; all float-to-int sites; cosine self/opposite/orthogonal at widths512/768/1536 and unequal scales.

**Source/read scope:**

- `Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift`
- `Sources/VectorCore/Quantization/QuantizationSchemes.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/QuantizationRangeContractTests.swift` — Proposed; not created.
- `Tests/ComprehensiveTests/QuantizedKernelsTests.swift` — Existing candidate.

**Audit acceptance to preserve:**

- [NUM-03](../../../audits/2026-09-14/CLAIMS.md#num-03): Both calibration APIs across signed/one-sided/cross-zero/constant intervals must cover endpoints within a defined quantization step; scalar/SIMD/decoded construction agree.
- [NUM-04](../../../audits/2026-09-14/CLAIMS.md#num-04): Mechanical Float-to-integer conversion matrix: finite extremes/constants/outliers, NaN/Inf, scalar versus SIMD, decoded params; no process termination for supported inputs and explicit error/fallback otherwise.
- [NUM-05](../../../audits/2026-09-14/CLAIMS.md#num-05): Self/opposite/orthogonal scale-family matrix all widths, asymmetric and unequal-scale cases, zero/nonfinite policy; compare to Double decoded-vector reference.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="q02"></a>
## Q02 — Precision selection, accuracy and tuner provenance

**State:** planned. **Depends on:** Q01, N01, E05.

**Design checkpoint:** Define bound versus heuristic, absolute/relative error, zero-reference behavior, full-FP32 and resource guarantees. Immutable pool refill and calibration cost are design/profiling questions.

**Regression design:** Threshold0, swapped operands, large candidate/small query, near-tie fixtures; cache every decision input and test call permutations; ordinary non-CI persistence with temporary paths and provenance.

**Conditional prerequisites:**

- O05: Introducing mutable conversion scratch or changing borrowing/storage ownership; correcting unsupported reuse wording alone does not require it.

**Source/read scope:**

- `Sources/VectorCore/Operations/Kernels/KernelAutoTuner.swift`
- `Sources/VectorCore/Operations/Kernels/MixedPrecisionAutoTuner.swift`
- `Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift`
- `Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/PrecisionSelectionContractTests.swift` — Proposed; not created.
- `Tests/ComprehensiveTests/KernelAutoTunerTests.swift` — Existing candidate.

**Audit acceptance to preserve:**

- [NUM-16](../../../audits/2026-09-14/CLAIMS.md#num-16): Operand reversal, high candidate/low query, threshold sweeps and near ties; prove/test conservative selection bounds and compare with defined baseline.
- [NUM-17](../../../audits/2026-09-14/CLAIMS.md#num-17): Change one input at a time, compare warm/fresh answers, exercise call permutations and invalid accuracy values.
- [NUM-19](../../../audits/2026-09-14/CLAIMS.md#num-19): Temporary-path persistence round trip and non-CI isolated selection test; vary constraints with same shape, require feasible or explicit no-feasible response; invalidate incompatible profiles.
- [NUM-20](../../../audits/2026-09-14/CLAIMS.md#num-20): Explicit fp32Full call always returns fullFP32 or a documented error; no synthetic benchmark can weaken exact precision policy.
- [NUM-21](../../../audits/2026-09-14/CLAIMS.md#num-21): Count calibration work and conversion boundaries mechanically; measure coordinated Release first-call latency/peak allocations on representative workloads; preserve error semantics.
- [NUM-24](../../../audits/2026-09-14/CLAIMS.md#num-24): A consumer converts distinct successive vectors while demonstrating actual allocation reuse and correct values; otherwise remove conversion-reuse claim.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="q03"></a>
## Q03 — Race-free rounding-aware FP16 diagnostics

**State:** planned. **Depends on:** None.

**Design checkpoint:** Specify recording snapshots and conversion classification from actual rounded representation; consolidate both unlocked guards.

**Regression design:** Rounding ties and neighbors around zero/subnormal/normal/overflow,1e-12 and65505, signed extrema; scalar/batch totals; concurrent toggling and constructors under supported TSAN.

**Source/read scope:**

- `Sources/VectorCore/Operations/Kernels/MixedPrecisionDiagnostics.swift`
- `Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/FP16DiagnosticsBoundaryTests.swift` — Proposed; not created.

**Audit acceptance to preserve:**

- [NUM-18](../../../audits/2026-09-14/CLAIMS.md#num-18): Scalar/batch classification same on rounding ties, adjacent FP16 normal/subnormal/zero/overflow boundaries, NaNs/infinities and positive/negative-only batches.
- [API-21](../../../audits/2026-09-14/CLAIMS.md#api-21): TSAN concurrent enable/disable plus all live conversion inits; stable state/stat count policy. Eliminate both duplicated unlocked guards.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="q04"></a>
## Q04 — FP16 range work proportional to requested range

**State:** planned. **Depends on:** Q01, Q03.

**Design checkpoint:** Preserve compact-output/global-index conventions while avoiding whole-candidate mapping; serialize shared MixedPrecisionKernels edits with Q02/O05.

**Regression design:** All nine wrappers at empty/nonzero-start/tail ranges and sentinel outputs; vary unrelated N at fixed range, record conversion/allocation work before speed claims.

**Source/read scope:**

- `Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/MixedPrecisionRangeValidationTests.swift` — Existing candidate.

**Audit acceptance to preserve:**

- [NUM-22](../../../audits/2026-09-14/CLAIMS.md#num-22): Empty/single range retained metadata/work independent of unrelated candidate count; all widths and nonzero range starts produce correct scores/sentinels.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="l01"></a>
## L01 — Swift factorization range and boundary contracts

**State:** planned. **Depends on:** N01.

**Design checkpoint:** Choose scaled convergence/normalization rules and dimension-aware tolerances. Keep qualified extreme LAPACK workspace investigation distinct from reproduced Swift defects.

**Regression design:** Analytic golden-ratio, single-column and rectangular full-rank fixtures across exponents; finite factors, scaled spectra, normalized residuals and U/V orthogonality; explicit Swift and PCA override.

**Source/read scope:**

- `Sources/VectorCore/LinearAlgebra/LinearAlgebraProvider.swift`
- `Sources/VectorCore/LinearAlgebra/SwiftLinearAlgebraProvider.swift`
- `Sources/VectorCoreC/common/vc_lapack.c`

**Test candidates:**

- `Tests/ComprehensiveTests/LinearAlgebraRangeTests.swift` — Proposed; not created.
- `Tests/ComprehensiveTests/NumericalReferenceFixtures.swift` — Proposed; not created.

**Audit acceptance to preserve:**

- [NUM-08](../../../audits/2026-09-14/CLAIMS.md#num-08): Scale-equivariant spectra, orthogonality and normalized residuals for full-rank rectangular matrices; output finiteness; isolate rank-deficiency exception.
- [NUM-09](../../../audits/2026-09-14/CLAIMS.md#num-09): Shared exponent sweep QR/SVD/eigen with normalized residual, orthogonality, spectrum, finiteness; include single-column and near-extreme finite vectors.
- [NUM-35](../../../audits/2026-09-14/CLAIMS.md#num-35): Boundary size arithmetic tested without giant allocations; supported envelope documented and clean error outside it.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="l02"></a>
## L02 — PCA centering and checked sketch dimensions

**State:** planned. **Depends on:** L01.

**Design checkpoint:** Decide centering representation and model/transform compatibility before improving qualified offset behavior; validate oversampling before addition.

**Regression design:** Translated representable [0,2] versus [16777216,16777218] against Double covariance; fit/single/batch transform agreement;0/normal/Int.max oversampling on tiny inputs.

**Source/read scope:**

- `Sources/VectorCore/LinearAlgebra/PCA.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/PCATests.swift` — Existing candidate.
- `Tests/ComprehensiveTests/LinearAlgebraBoundaryTests.swift` — Proposed; not created.

**Audit acceptance to preserve:**

- [NUM-10](../../../audits/2026-09-14/CLAIMS.md#num-10): Translation/offset invariance with representable input/residual fixtures; compare Double covariance and projected sample mean; no fit/transform center mismatch.
- [API-20](../../../audits/2026-09-14/CLAIMS.md#api-20): 0,normal,Int.max oversampling on tiny data; correct exact fallback or documented error without overflow.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="l03"></a>
## L03 — UMAP calibration equations and input boundaries

**State:** planned. **Depends on:** L02.

**Design checkpoint:** Define coefficient representability/nonconvergence, checked CSR scope and optional initial-coordinate policy. Brute-force Gram cancellation is documented context, not a new hidden defect.

**Regression design:** Dimensionless curve/SSE and smooth-kNN defining equations over scale families; solvable and tied/duplicate rows; floating config finiteness, malformed offsets and extreme integer options before unsafe arithmetic.

**Conditional prerequisites:**

- P02: Changing the exhaustive Gram reference route or its exactness policy; retaining the documented limitation does not require a new matrix implementation.

**Source/read scope:**

- `Docs/Linear_Algebra_and_Projection.md`
- `Docs/Numerical_Behavior.md`
- `Sources/VectorCore/LinearAlgebra/PCA.swift`
- `Sources/VectorCore/ManifoldLearning/FuzzySimplicialSet.swift`
- `Sources/VectorCore/ManifoldLearning/KNNGraph.swift`
- `Sources/VectorCore/ManifoldLearning/UMAP.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/UMAPRangeContractTests.swift` — Proposed; not created.
- `Tests/ComprehensiveTests/UMAPTests.swift` — Existing candidate.

**Audit acceptance to preserve:**

- [NUM-11](../../../audits/2026-09-14/CLAIMS.md#num-11): Assert b and dimensionless fitted curve/SSE invariant over scale families and check convergence; explicit failure outside supported range. Do not assert entire SGD layout is scale-equivariant.
- [NUM-12](../../../audits/2026-09-14/CLAIMS.md#num-12): Multiplicative-scale tests with unequal degrees/isolated/duplicate rows; assert weights/topology and equation residual where solvable, with supported-range policy for sigma overflow.
- [NUM-13](../../../audits/2026-09-14/CLAIMS.md#num-13): Table of NaN/±Inf per floating option across both overloads must throw; normal configs remain deterministic; include finite extremes to characterize policy.
- [NUM-25](../../../audits/2026-09-14/CLAIMS.md#num-25): Generated small malformed-offset matrix including valid final endpoint but interior overshoot/negative/decrease must throw under chosen checked contract; valid uneven/unsorted rows remain accepted.
- [NUM-26](../../../audits/2026-09-14/CLAIMS.md#num-26): Tiny fixtures atInt.max, safe bounds, zero/negative/normal values either throw documented error or produce bounded intended result without process termination.
- [NUM-27](../../../audits/2026-09-14/CLAIMS.md#num-27): Near-offset fixtures compared with widened/direct reference; any refinement must not lose true nearest candidate before reranking.
- [NUM-28](../../../audits/2026-09-14/CLAIMS.md#num-28): NaN/±Inf coordinate tests and large finite scale fixtures with documented result/error policy; ordinary init deterministic.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="l04"></a>
## L04 — Manifold and PCA memory/work envelope

**State:** planned. **Depends on:** L03, V04.

**Design checkpoint:** Select supported corpus/sample budgets and cancellation contract before redesigning documented synchronous in-memory algorithms.

**Regression design:** Analytical buffers plus measured peak/retained memory, graph sorting/epoch scans and transforms on representative seeded workloads; preserve order/determinism contracts.

**Source/read scope:**

- `Sources/VectorCore/LinearAlgebra/PCA.swift`
- `Sources/VectorCore/LinearAlgebra/SwiftLinearAlgebraProvider.swift`
- `Sources/VectorCore/ManifoldLearning/FuzzySimplicialSet.swift`
- `Sources/VectorCore/ManifoldLearning/UMAP.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/UMAPTests.swift` — Existing candidate.
- `Tests/ComprehensiveTests/PCATests.swift` — Existing candidate.

**Audit acceptance to preserve:**

- [NUM-29](../../../audits/2026-09-14/CLAIMS.md#num-29): Measured peak allocation/RSS alongside predicted payload for representative Release runs; verify identical results/order or explicitly documented numerical changes.
- [NUM-30](../../../audits/2026-09-14/CLAIMS.md#num-30): Coordinated Release time/allocation profiles and result/seed reproducibility checks for any schedule or format change.
- [NUM-31](../../../audits/2026-09-14/CLAIMS.md#num-31): Cancel at meaningful phases and bound latency/resources; legacy synchronous success output remains consistent.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="p01"></a>
## P01 — Hash quality and streaming selection work

**State:** planned. **Depends on:** R01, R03, R04, O03, V04.

**Design checkpoint:** Keep lawful hash collisions distinct from cache identity; preserve common membership/NaN/tie laws, count range and public validation when adopting streaming selection.

**Regression design:** Interior-only optimized-vector variations and signed-zero controls; allocations versus N at fixed k and nearN; scalar/direct reference membership and equivalent-work Release timing.

**Source/read scope:**

- `Sources/VectorCore/Operations/BatchOperations.swift`
- `Sources/VectorCore/Operations/Kernels/TopKSelectionKernels.swift`
- `Sources/VectorCore/Operations/Operations.swift`
- `Sources/VectorCore/Operations/TopKSelection.swift`
- `Sources/VectorCore/Operations/VectorMath.swift`
- `Sources/VectorCore/Vectors/Vector1536Optimized.swift`
- `Sources/VectorCore/Vectors/Vector384Optimized.swift`
- `Sources/VectorCore/Vectors/Vector512Optimized.swift`
- `Sources/VectorCore/Vectors/Vector768Optimized.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/OptimizedHashContractTests.swift` — Proposed; not created.
- `Tests/ComprehensiveTests/TopKSelectionTests.swift` — Existing candidate.

**Audit acceptance to preserve:**

- [OWN-07](../../../audits/2026-09-14/CLAIMS.md#own-07): Large structured/sparse interior-varying corpus must not deterministically collapse; equal values including signed zeros retain equal hashes. Coordinate Release insertion/lookup scaling measurement before speed claims.
- [API-23](../../../audits/2026-09-14/CLAIMS.md#api-23): Allocation/peak memory versus N at fixed k; exact membership/order/NaN/ties match shared selector; Release measurement before performance claims.
- [API-24](../../../audits/2026-09-14/CLAIMS.md#api-24): Release allocation profiles; shared membership/order at k nearN and varied chunk sizes; public invalid inputs still validate exactly before unsafe routes.
- [API-45](../../../audits/2026-09-14/CLAIMS.md#api-45): Metric spy k0 policy, k bounds, common order/member tests; Release allocation/time before performance claim.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="p02"></a>
## P02 — Explicit batch accuracy, memory and reusable packing

**State:** planned. **Depends on:** R01, R03, N01, E06, V02, V04.

**Design checkpoint:** Choose exact/direct versus approximate/Gram policy, scratch budget and real consumer-backed contiguous ownership API before route changes.

**Regression design:** Near-equal large-offset ranking, Q7/8 and N255/256 gates, installed-provider sentinel and fallback; real centroid consumer parity; allocation budgets include packed inputs/output and scratch.

**Source/read scope:**

- `../VectorIndex/Sources/VectorIndex/Kernels/CentroidBatchScore.swift`
- `.github/workflows/benchmarks.yml`
- `.github/workflows/ci.yml`
- `Benchmarks/VectorCoreBench/AppMain.swift`
- `Docs/Numerical_Behavior.md`
- `Sources/VectorCore/Operations/MatrixDistance.swift`
- `Sources/VectorCore/Operations/Operations.swift`
- `Tests/ComprehensiveTests/BatchGEMMRoutingTests.swift`
- `Tests/ComprehensiveTests/BatchKernelProviderTests.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/BatchGEMMRoutingTests.swift` — Existing candidate.
- `Tests/ComprehensiveTests/BatchKernelProviderTests.swift` — Existing candidate.
- `Tests/ComprehensiveTests/MatrixDistanceTests.swift` — Existing candidate.

**Audit acceptance to preserve:**

- [API-22](../../../audits/2026-09-14/CLAIMS.md#api-22): Memory budget invariants and deterministic exact-neighbor fixture; direct precision mode reproduces direct reference across gate; benchmark crossovers only after design.
- [API-25](../../../audits/2026-09-14/CLAIMS.md#api-25): Real centroid consumer parity and removed candidate repacking/allocation; invalid stride/shape checks before BLAS; lifetime tests.
- [API-41](../../../audits/2026-09-14/CLAIMS.md#api-41): Sentinel provider output at actual GEMM gate; default per-query and whole-batch override both exercised.
- [TOOL-10](../../../audits/2026-09-14/CLAIMS.md#tool-10): Publish exact hardware, inputs, routing, raw sample distribution and memory budget for chosen target scale; validate custom-provider precedence at every newly reachable boundary.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="b01"></a>
## B01 — Binary round trips, real tests and wire contracts

**State:** planned. **Depends on:** S01, S03.

**Design checkpoint:** Preserve existing wire bytes/caps and explicit concrete-type precision contracts; define empty dynamic round trip. Conformances and scalar-loop optimization need separate justification.

**Regression design:** Replace15 comment-only bodies and mislabeled concrete types; independent complete golden frame, NaN payload/signed-zero bits, sliced Data, CRC/length/cap boundaries and empty-after-removal; profile before optimizing.

**Conditional prerequisites:**

- V04: Optimizing scalar binary conversion or publishing speed/allocation improvement; compatible round-trip fixes can proceed independently.

**Source/read scope:**

- `Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift`
- `Sources/VectorCore/Serialization/BinaryProtocols.swift`
- `Sources/VectorCore/Utilities/BinaryFormat.swift`
- `Sources/VectorCore/Vectors/DynamicVector.swift`
- `Sources/VectorCore/Vectors/VectorSerialization.swift`
- `Tests/ComprehensiveTests/SerializationEdgeCaseTests.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/SerializationEdgeCaseTests.swift` — Existing candidate.
- `Tests/ComprehensiveTests/VectorSerializationTests.swift` — Existing candidate.

**Audit acceptance to preserve:**

- [SER-01](../../../audits/2026-09-14/CLAIMS.md#ser-01): Real default/last-element-removal empty roundtrips or explicit checked rejection; fixed-size decoders must retain dimension checks; CRC/size failures remain rejected.
- [SER-02](../../../audits/2026-09-14/CLAIMS.md#ser-02): Every active test named for a behavior must execute/assert that behavior. Include signed-zero/NaN payload binary bits, direct sliced Data, valid/invalid CRC and exact frame lengths.
- [SER-03](../../../audits/2026-09-14/CLAIMS.md#ser-03): Before/after Release measurements with allocation counts, CRC included, representative dimensions and unaligned slices; byte-for-byte equality against independent fixtures.
- [SER-04](../../../audits/2026-09-14/CLAIMS.md#ser-04): Examples and tests must name concrete source/destination types and expected byte sizes/precision; existing wire format compatibility preserved unless explicitly versioned.
- [SER-05](../../../audits/2026-09-14/CLAIMS.md#ser-05): External generic helper compile test for supported conformances; clearly define types that encode only versus fully roundtrip.
- [SER-06](../../../audits/2026-09-14/CLAIMS.md#ser-06): Publish independently specified full golden frame; old frames remain decodable.
- [SER-07](../../../audits/2026-09-14/CLAIMS.md#ser-07): Class-wide malformed framing/CRC and cap boundary tests, plus independent bits/slice regression; do not count absent streaming/compression/migration features as implemented.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="d01"></a>
## D01 — Pinned downstream contract validation

**State:** planned. **Depends on:** I01, N01, R01, R05.

**Design checkpoint:** Name supported sibling revisions and direct-vs-bridge score/precision semantics; prepare sibling-owned fixes separately. Core-only authorization does not permit sibling edits.

**Regression design:** Mixed normalized/raw batch hints in both orders versus individual Flat/IVF/optimized-flat searches; compile actual integration examples and dependency combinations; CPU/GPU evidence separate.

**Source/read scope:**

- `../VectorIndex/Package.resolved`
- `../VectorIndex/Package.swift`
- `../VectorIndex/Sources/VectorIndex/FlatIndex.swift`
- `../VectorIndex/Sources/VectorIndex/IVFIndex.swift`
- `../VectorIndex/Sources/VectorIndex/TypedOverloads.swift`
- `../future/VectorAccelerate/Package.resolved`
- `../future/VectorAccelerate/Package.swift`
- `../future/VectorAccelerate/Sources/VectorAccelerate/Integration/MetalComputeProvider.swift`
- `Sources/VectorCore/Integration/VectorCollection.swift`
- `Sources/VectorCore/Protocols/BatchKernelProvider.swift`
- `Sources/VectorCore/Version.swift`

**Test candidates:**

- `Scripts/ci/consumer_smoke.py` — Existing candidate.

**Audit acceptance to preserve:**

- [API-34](../../../audits/2026-09-14/CLAIMS.md#api-34): Mixed normalized/raw hint batches in both orders match individual searches across Flat/IVF/optimized-flat.
- [API-35](../../../audits/2026-09-14/CLAIMS.md#api-35): Direct-vs-bridge sign tests and NaN/tie policy declarations; custom metric witness remains authoritative.
- [API-36](../../../audits/2026-09-14/CLAIMS.md#api-36): Compile actual cross-package integration examples at pinned versions; route map reflects real conformance.
- [API-40](../../../audits/2026-09-14/CLAIMS.md#api-40): Consumer builds against intended release/main separately; do not infer platform lowering from shared protocols.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="d02"></a>
## D02 — Public API, device and C contract disposition

**State:** planned. **Depends on:** D01, B01, E06.

**Design checkpoint:** Document actual effective hooks, factory/storage customization and device placeholders; decide supported surface before wiring dormant hooks. Mobile benchmarking product and helper performance remain investigations.

**Regression design:** External concrete/generic/provider consumers and symbol inventory; every retained runtime metadata claim backed by discovery or explicit unsupported status; C declaration/link and lane-limit checks; actual product target builds.

**Source/read scope:**

- `.github/workflows/ci.yml`
- `Package.swift`
- `Sources/VectorCore/Core/Dimension.swift`
- `Sources/VectorCore/Execution/ComputeDevice.swift`
- `Sources/VectorCore/Factory/VectorTypeFactory.swift`
- `Sources/VectorCore/Interop/CKernels.swift`
- `Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift`
- `Sources/VectorCore/Operations/Kernels/SoAFP16Cache.swift`
- `Sources/VectorCore/Operations/Operations.swift`
- `Sources/VectorCore/Operations/VectorCoreOperations.swift`
- `Sources/VectorCore/Operations/VectorMath.swift`
- `Sources/VectorCore/Optimization/OptimizationAttributes.swift`
- `Sources/VectorCore/Platform/PlatformConfiguration.swift`
- `Sources/VectorCore/Platform/SwiftDoubleSIMDProvider.swift`
- `Sources/VectorCore/Protocols/VectorFactory.swift`
- `Sources/VectorCore/Protocols/VectorProtocol.swift`
- `Sources/VectorCore/VectorCore.swift`
- `Sources/VectorCore/Vectors/Vector.swift`
- `Sources/VectorCoreBenchmarking/EnvCapture.swift`
- `Sources/VectorCoreC/arm64/vc_arm64.c`
- `Sources/VectorCoreC/common/vc_dispatch.c`
- `Sources/VectorCoreC/common/vc_scalar.c`
- `Sources/VectorCoreC/include/VectorCoreC.h`
- `Sources/VectorCoreC/x86_64/vc_x86.c`

**Test candidates:**

- `Tests/ComprehensiveTests/ArbitraryDimensionTests.swift` — Existing candidate.
- `Tests/ComprehensiveTests/VectorTypeFactoryTests.swift` — Existing candidate.
- `Tests/ComprehensiveTests/CKernelsSmokeTests.swift` — Existing candidate.
- `Scripts/ci/consumer_smoke.py` — Existing candidate.

**Audit acceptance to preserve:**

- [API-12](../../../audits/2026-09-14/CLAIMS.md#api-12): Spy provider verifies each claimed active hook; public-entry/provider matrix documents real dispatch.
- [API-30](../../../audits/2026-09-14/CLAIMS.md#api-30): Invalid GPU indices, no-neural hardware and provider creation consistency under fake discovery backend.
- [API-38](../../../audits/2026-09-14/CLAIMS.md#api-38): Generated public symbol diff and downstream consumer compile tests for supported surface.
- [OWN-20](../../../audits/2026-09-14/CLAIMS.md#own-20): Custom dimension conformance fixtures demonstrate actual storage and intended available operations; no advertised customization point has no effect.
- [OWN-21](../../../audits/2026-09-14/CLAIMS.md#own-21): Concrete-output and routing tests remain deliberate; any strategy change reruns all reachable kernels and consumer casts at every supported dimension.
- [OWN-27](../../../audits/2026-09-14/CLAIMS.md#own-27): Concrete/generic/existential compatible calls obey same documented contract; spy/conformance fixtures identify selected customization points.
- [NUM-32](../../../audits/2026-09-14/CLAIMS.md#num-32): All architectures/C gate configurations compare max-magnitude fixed widths and remainder counts to Int64 reference; arbitrary supported lanes must have explicit overflow behavior.
- [NUM-33](../../../audits/2026-09-14/CLAIMS.md#num-33): C consumer symbol/link and nonzero-start sentinel tests; supported public declarations all link.
- [NUM-34](../../../audits/2026-09-14/CLAIMS.md#num-34): Representative Release allocation/throughput and generated-code records, matched semantics, supported-platform conformance.
- [TOOL-13](../../../audits/2026-09-14/CLAIMS.md#tool-13): Product support matrix backed by actual target builds, or clear host-only documentation.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.

<a id="d03"></a>
## D03 — Lazy logging and elapsed-time contract

**State:** planned. **Depends on:** None.

**Design checkpoint:** Preserve logger API while evaluating messages only when emitted; select monotonic elapsed-time semantics without inventing measured overhead.

**Regression design:** Disabled autoclosure evaluated zero times, enabled once; controllable clock fixture for elapsed timing behavior.

**Source/read scope:**

- `Sources/VectorCore/Logging/Logger.swift`

**Test candidates:**

- `Tests/ComprehensiveTests/LoggingContractTests.swift` — Proposed; not created.

**Audit acceptance to preserve:**

- [API-31](../../../audits/2026-09-14/CLAIMS.md#api-31): Disabled message spy remains unevaluated; enabled once; controllable timer clock verifies elapsed policy.

**Before dispatch:**

- [ ] Root resolves the checkpoint and specifies compatibility, observable behavior and exact permitted edits.
- [ ] Root supplies independent regression code and exact focused commands; known-bad audit probes are not passing tests.
- [ ] Scheduler reserves all shared files/build outputs and confirms prerequisites.
- [ ] Later user implementation authorization is present; agent receives Astra/medium and the global constraints.
