# BE3 Failing-Test Triage

**Source run:** `swift test --xunit-output test-results.xml 2>&1 | tee test-run-full.log`
**Run date:** 2026-05-29 13:17 · Swift 6.3.2 · swift-testing 1501 · arm64e-apple-macos
**Totals:** 993 tests / 138 suites · **76 failing test functions** · 3380 issues · 391.8 s

Branch: `fix/be3-audit-confirmed-bugs` (recent commits changed FP16/normalize/cosine numerics — stale-expectation tests are a likely bucket).

Classification key: **SRC** = source bug (test correct) · **TST** = test issue (source correct) · **FLAKY** = env/timing · **?** = not yet investigated.

> **KEY FINDING (all clusters investigated so far):** Every failure triaged below **pre-dates the BE3 audit branch** — git confirms commits `09315d3` and `39012c6` did *not* touch `MixedPrecisionKernels.swift`, `BatchKernels_SoA.swift`, or the FP16 conversion code. Commit `09315d3`'s own message even states "Pre-existing FP16 fuzzing/autotuner failures are unrelated." These are latent pre-existing bugs and stale tests, **not regressions from the audit work.**

## Verdicts — Round 1 (clusters 1, 3, 6, 8) — **RESOLVED & VERIFIED GREEN**
Deeper investigation during the fix flipped two SRC→TST verdicts and surfaced extra sub-issues. Final: **2 real source defects**, rest test issues. All 13 affected tests pass (targeted run, 0 issues).

| Verdict | Count | Tests |
|---------|-------|-------|
| **TST** | 10 | C1 all 4 (FP16 inf is the documented contract); C3 testDotProduct512/768/1536 (1e-5 absolute tol too tight for FP32); C3 testCacheEfficiencySoA (compared distance vs squared); C6 testDenormalizedValues (1e-10 < exact 1.164e-10); C8 testEmptyCandidateSet (empty→empty SoA is valid, no throw) |
| **SRC** | 2 | C6 testAdaptiveKernelSelection (**heap OOB read** → NaN); C8 testSoAFP16VectorExtraction (init missing 768/1536 dispatch) |

testAdaptiveKernelSelection was **MIXED**: 4 sub-issues — NaN/OOB (SRC) + 3 TST (relError compared distance vs squared; false "monotonic" assumption, distances are U-shaped; relError ill-defined at near-zero distance i=9).

**2 source defects fixed:**
1. **Heap out-of-bounds read** — `adaptiveEuclideanDistance` wrapped a flat 512-elem AoS buffer as `SoA512FP16` (needs 2048) → ~75% garbage reads → NaN. Fixed via `createSoA512FP16(from:[candidate])`. `MixedPrecisionKernels.swift`. *Memory-safety bug.*
2. **Under-built `SoAFP16.init(vectors:)`** — only dispatched `Vector512Optimized`; 768/1536 fell through to empty `vectorCount=0`. Added 768/1536 dispatch + unsupported-type throw.

**Two API contract decisions (documented in code):**
- `batchEuclideanSquaredSoA` returns *actual distance* (name is a legacy misnomer, already documented at MPK:3318-3319) — NOT a source bug. Did not remove the `sqrt` (other callers depend on it); tests now compare like-for-like.
- `SoAFP16(vectors: [])` → *empty SoA, no throw* (idiomatic; all 3 builders already do this). Two tests asserted opposite contracts; reconciled on no-throw → testEmptyCandidateSet was the TST.

---

## Cluster 1 — FP16 overflow → `inf` (clamping contract) · 4 tests · **VERDICT: TST (high)**
Source implements IEEE round-to-nearest (overflow & ±inf → ±inf); the doc comment at `MixedPrecisionKernels.swift:303-305` documents exactly this, and the sibling suite `MixedPrecisionKernelTests.swift:305-306` asserts the *correct* inf behavior. The failing suite (`MixedPrecisionKernelsTests.swift`) is even internally contradictory. **Fix tests to expect `±inf`, not 65504.**
- [x] TST AccuracyPrecisionTests.testFP16OverflowUnderflow() — expect `.isInfinite` (`MixedPrecisionKernelsTests.swift:638`, mixed block 663-665)
- [x] TST EdgeCasesErrorHandlingTests.testFP16ConversionErrors() — `MixedPrecisionKernelsTests.swift:1652-1659`
- [x] TST EdgeCasesErrorHandlingTests.testFP16EdgeValues() — `MixedPrecisionKernelsTests.swift:1606`
- [x] TST FP16StorageTypesTests.testFP16Conversion() — `MixedPrecisionKernelsTests.swift:200-204`

## Cluster 2 — SoA / FP16 storage layout sizing · 8 tests · hypothesis: TST (stale) or SRC
`storage.count` / byte-size assertions. Likely tied to SIMD-lane packing expectations vs. actual storage sizing.
- [ ] ? FP16StorageTests.testFP16StorageMemoryEfficiency()
- [ ] ? FP16StorageTests.testVector1536FP16Storage()
- [ ] ? FP16StorageTests.testVector512FP16Storage()
- [ ] ? FP16StorageTests.testVector768FP16Storage()
- [ ] ? FP16StorageTypesTests.testFP16StorageLayout()
- [ ] ? HardwareOptimizationTests.testAppleSiliconNEON()
- [ ] ? IntegrationTests.testOptimizedVectorCompatibility()
- [ ] ? SoALayoutTests.testSoAFP16StorageEfficiency()

## Cluster 3 — SoA vs reference result mismatch · 4 tests · **VERDICT: MIXED (high)**
- [x] TST CoreSoAFunctionalityTests.testDotProduct512/768/1536() — FP32 accumulation-order rounding (kernel correct, simulated bit-for-bit). 1e-5 *absolute* tol too tight for FP32 dot products (expected error ≈ 4e-4…2e-3). Fix: relative tol `1e-5*max(1,|ref|)` at `BatchKernels_SoATests.swift:212,237,262`.
- [x] **SRC** PerformanceValidationTests.testCacheEfficiencySoA() — compares distance `d` vs squared `d²`. `batchEuclideanSquaredSoA`→`batchEuclidean512` returns `sqrt(...)` at `MixedPrecisionKernels.swift:1955` despite "Squared" name. Fails ~93% rel err. Fix source semantics (audit all callers) or test compare.

## Cluster 4 — Accuracy / relative-error tolerance · 12 tests · hypothesis: TST (tolerance) or SRC (regression)
Mixed-precision/quantized relative-error bounds. Need per-test: is the bound reasonable for the FP path, or did accuracy actually regress? Note `relError → inf` cases = possible div-by-zero / inf leak (lean SRC).
- [ ] ? BatchOperationsTests.testBatchDotMixed()
- [ ] ? BenchmarkTests.testAccuracyMeasurement()
- [ ] ? DotProductAccuracyTests.testDotProductAccuracy512()
- [ ] ? DotProductAccuracyTests.testMixedPrecisionAccuracy512()
- [ ] ? DotProductInvariantTests.testLinearity()
- [ ] ? EdgeCasesErrorHandlingTests.testDenormalNumberHandling()
- [ ] ? INT8StorageTests.testVector512INT8Initialization()
- [ ] ? MixedPrecisionDistanceTests.testEuclidean1536MixedAccuracy()
- [ ] ? PerformanceValidationTests.testMemoryBandwidthImprovement()
- [ ] ? QuantizationAccuracyTests.testGaussianDistributionAccuracy()
- [ ] ? QuantizedDistanceComputationTests.testQuantizedDotProduct()
- [ ] ? SoALayoutTests.testBatchEuclideanSoA()

## Cluster 5 — Performance / timing assertions · 11 tests · hypothesis: FLAKY/TST (brittle perf bounds)
Wall-clock speedup/bandwidth/latency thresholds — environment-sensitive. Default suspicion: brittle test assertions, *unless* a real perf regression is shown.
- [ ] ? BatchProcessingTests.testLargeBatchSize() / testMediumBatchSize()
- [ ] ? BenchmarkTests.testBenchmarkBatchOperations() / testBenchmarkDotProduct()
- [ ] ? MemoryEfficiencyTests.testMemoryBandwidthUtilization()
- [ ] ? MixedPrecisionPerformanceBenchmark.benchmarkCosineLatency512() / benchmarkDimensionScaling() / benchmarkEuclideanLatency512()
- [ ] ? PerformanceComparisonTests.testSoAvsAoSPerformance()
- [ ] ? PerformanceOptimizationTests.testCacheEfficiencyQuantized() / testParallelQuantizedOperations()

## Cluster 6 — NaN / distance edge case · 2 tests · **VERDICT: MIXED (very high)**
- [x] **SRC** AutoTuningTests.testAdaptiveKernelSelection() — `adaptiveEuclideanDistance` wraps flat 512-elem AoS `Vector512FP16.storage` as `SoA512FP16` (needs 2048 elems) at `MixedPrecisionKernels.swift:3360` → ~75% **out-of-bounds heap reads** → NaN + 98% error. Memory-safety bug. Fix: build SoA via `createSoA512FP16(from:)`.
- [x] TST EdgeCasesErrorHandlingTests.testDenormalizedValues() — `1e-10` threshold below the exact analytic max `512·(4·eps)² = 1.164e-10`. Kernel correct. Loosen to `1e-9` at `BatchKernels_SoATests.swift:1516`.

## Cluster 7 — AutoTuner / precision heuristic thresholds · 4 tests · hypothesis: TST or SRC (heuristic tuning)
- [ ] ? AutoTuningTests.testPrecisionStrategySelection()
- [ ] ? IntegrationTests.testFP32Fallback()
- [ ] ? MixedPrecisionPerformanceBenchmark.validateHeuristic()
- [ ] ? PrecisionAnalysisTests.testPrecisionAnalysisNormalized()

## Cluster 8 — Error-handling expectations · 2 tests · **VERDICT: SRC (high) — one fix clears both**
Both stem from under-built `SoAFP16.init(vectors:blockSize:)` at `MixedPrecisionKernels.swift:499-508`.
- [x] **SRC** SoALayoutTests.testSoAFP16VectorExtraction() — init only dispatches `Vector512Optimized`; 768/1536 fall through to empty container (`vectorCount=0`), so `getVector(at:0)` throws indexOutOfBounds. `createSoA768FP16`/`createSoA1536FP16` exist but unwired. Fix: add 768/1536 dispatch.
- [x] **SRC** EdgeCaseTests.testEmptyCandidateSet() — init is `throws` but never validates empty input (returns empty SoA). Fix: `guard !vectors.isEmpty else { throw VectorError.invalidData(...) }` (precedent: `EdgeCaseHandler.swift:326`).

## Cluster 9 — Other (quantization / SoA-cache / mixed-accuracy variants) · 29 tests · hypothesis: mixed
Mostly fold into clusters 2/3/4/7 on inspection (INT4/8/16 quant accuracy, SoA cache, mixed-precision accuracy, register blocking).
- [ ] ? AccuracyBoundsTests.testMixedPrecisionAccuracy() / testNormalizedVectorAccuracy()
- [ ] ? DifferentBitWidthTests.testINT16Quantization() / testINT4Quantization()
- [ ] ? EdgeCaseTests.testMismatchedDimensions() / testSingleCandidate()
- [ ] ? EdgeCasesErrorHandlingTests.testDegenerateDistributions()
- [ ] ? HardwareOptimizationTests.testNeuralEngineCompatibility()
- [ ] ? INT8QuantizationTests.testINT8QuantizationBasic()
- [ ] ? MixedPrecisionAutoTunerTests.testLargeBatchStrategySelection() / testSmallBatchStrategySelection()
- [ ] ? MixedPrecisionDistanceTests.testCosine512MixedAccuracy() / testEuclidean768MixedAccuracy()
- [ ] ? MixedPrecisionPhase3Tests.testSoACacheBasic() / testSoACacheHitRate()
- [ ] ? NumericalAccuracyTests.testConsistencyAcrossPlatforms() / testCumulativeErrors()
- [ ] ? NumericalValidationTests.testDistanceOrdering()
- [ ] ? OverflowDetectionTests.testOverflowTracking()
- [ ] ? PerformanceOptimizationTests.testQuantizationOverhead() / testQuantizedComputationPerformance()
- [ ] ? QuantizationSchemesTests.testAsymmetricQuantization() / testPerChannelQuantization()
- [ ] ? QuantizedDistanceComputationTests.testQuantizedCosineDistance()
- [ ] ? QuantizedDistanceTests.testBatchDistanceComputation()
- [ ] ? RegisterBlockingTests.testBlockingWithDifferentDimensions() / testTwoWayBlocking()
- [ ] ? SoALayoutTests.testBatchDotProductSoA() / testSoAFP16Initialization()
