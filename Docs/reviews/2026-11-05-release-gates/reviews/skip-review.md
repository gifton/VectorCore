# Independent review of six mixed correctness/performance skips

**Recommendation: retain these six skips as explicit optional-workload coverage limits; do not refactor their gates or claim a newly discovered production defect.** Four methods' correctness properties already have ordinary, executed coverage. Two methods add distinct concurrent calling contexts, but their only ordinary assertions count outputs from task groups written inside the tests. No production dispatch branch was found that is reachable only through these six methods. The exact optional workloads are not qualified, and running them would not by itself establish concurrent numerical correctness or race freedom.

This is a read-only review of carried source `27a045dd648c01a4ef9c5d8d36c5afc92f36f6fb`, the platform investigator's `extended-body-review.json`, the published phase-two iOS/native reports, and retained v2 iOS `method-results.json`. No builds, tests, source edits, gate changes or performance work were run. [skip-review-support.json](/Users/goftin/dev/projects/VectorCore-release-gates-2026-11-05-evidence/coordinator/skip-review-support.json) retains 25 relevant method results, the raw result-file SHA-256 and seven reviewed source-file hashes. Those seven files are byte-identical between executed iOS source `e7482a9` and carried `27a045d`. Prior binary/result-bundle authenticity was not independently re-established here; the existing phase-two provenance remains the execution authority.

The retained iOS results label all six methods **Skipped** and the ordinary neighboring methods cited below **Passed**. The phase-two native record retains the 18 Swift Testing skips in the broader ASan/TSan scope; its strict no-skips qualification remains false. This review neither relabels those skipped bodies as passed nor broadens sanitizer acceptance. The installed-Xcode canary failure remains separate and unresolved.

## Method-by-method disposition

### 1. `BatchKernels_SoATests/BatchProcessingTests/testMediumBatchSize()`

At `BatchKernels_SoATests.swift:743`, query 0.75 and N = 10, 25, 50, 75, 100 check result count and **every** squared distance against `EuclideanKernels.squared512` with absolute error < 1e-3. Timed comparisons follow those ordinary assertions.

Ordinary coverage includes `testSmallBatchSize` (1…10), `testVaryingBatchSizes` (prime sizes 3…43, powers through 256 with every-result reference checks, and selected positions at 99/101/199/333/777), `testBatchConsistency` (every result at N50, several queries), `testBlockingWithDifferentDimensions` (every result at odd N25), and `testConcurrentBatchProcessing` (eight N100 batches and shared-candidate repetitions). All cited methods selected into the support record passed in the v2 iOS run; small-batch source reachability was inspected but is not one of that record's extracted rows.

The actual production dispatcher (`BatchKernels_SoA.swift:285`) chooses tiling by payload size, then four-way for N≥4, otherwise two-way. All skipped medium sizes use the already-exercised four-way branch. N75/query0.75 is an additional fixture, not a newly uncovered route. The precise skipped fixture matrix remains unexecuted; no new mandatory test is justified solely by this skip.

### 2. `BatchKernels_SoATests/BatchProcessingTests/testLargeBatchSize()`

At `BatchKernels_SoATests.swift:784`, query 0.25 and N = 100, 250, 500, 750, 1000 check count and ten regularly spaced numerical observations per size (<1e-3). RSS deltas are informational; timing checks are separate.

These 512D sizes remain below the 8 MiB tiling threshold: N1000 carries 2,048,000 scalar bytes, so they use four-way blocking, including already-tested remainders. Ordinary `testVaryingBatchSizes` validates public-output counts/reference samples through N777. `tiledEuclid512MatchesFourWayBitwise` compares every direct-kernel result at sizes including 255/256/257/300/512/513/600/1000. `largeBatchRoutesThroughTilingAndMatchesReference` checks the **public** N5000 output count, asserts the tiling predicate and samples the reference. The latter two were confirmed Passed.

These are complementary checks, not exact replacements for the skipped query0.25/N250/500/750/1000 public-wrapper observations: direct tiled-versus-four-way agreement alone cannot rule out a common numerical error. Nevertheless the skip conceals no distinct size-dispatch branch, and public reference checks cover both blocked and tiled routes elsewhere. Preserve the exact scale/fixture limitation; no blocker or forced test extraction follows from it.

### 3. `BatchKernels_SoATests/PerformanceComparisonTests/testSoAvsAoSPerformance()`

At `BatchKernels_SoATests.swift:1230`, its ordinary assertions are `shouldUseSoA(1000,512)==true` and `shouldUseSoA(50,512)==false`. Both are **exactly duplicated** by ungated `testScalingWithBatchSize` at lines 1532–1540, which checks counts 10, 50, 100, 250, 500, 1000 and 2000. That method Passed.

`shouldUseSoA` at production line 1722 is the documented heuristic `candidateCount >= 100 && dimension >= 512`; the numerical public dispatcher does not consult it. The skip therefore hides neither its two asserted recommendations nor a production numerical route. Dimension<512 behavior is not asserted by the skipped test either, so this skip cannot explain any gap there. The timed comparison remains optional and unqualified.

### 4. `MixedPrecisionKernelTests/PerformanceValidationTests/testThroughputScaling()`

At `MixedPrecisionKernelTests.swift:2866`, the sequential timed part calls 768D conversion/range Euclidean² for N = 1,10,50,100,200 but does **not** check their numerical results. Its only ordinary assertion checks that four task-group chunks of 25 produce 100 results. The group is test-authored; each task converts its own candidate slice, allocates its own output, and calls the ordinary serial range function with a shared value query. There is no production parallel batch scheduler here.

Ordinary `testEuclidean768MixedAccuracy` verifies 768D numerical results, `testBatchConversionFP32ToFP16` checks 768D batch conversion at N15 and separately 512D concurrent conversion, and `MixedPrecisionRangeWorkTests/finiteRangeMatrix` covers all dimensions/metrics, pair/tail/empty/nonzero-start ranges, independent Double references and output sentinels. `exceptionalSameCoreParity` adds exceptional-coordinate parity. These methods Passed. Production `convertToFP16_768` is a `map` (line 4887); the public range wrapper (line 4641) validates/slices then calls the same two-way core (line 4081). The count/loop branches are exercised by the ordinary range matrix.

**Residual:** I did not find an ordinary explicit task-group test of this exact shared-query 768D range-kernel context. Existing concurrent 512D conversion is not equivalent, and incidental parallel test-runner scheduling is not proof. The skipped count assertion would not detect wrong computed distances or task completion order, even if enabled. Treat this as optional concurrent-workload validation unless release claims specifically promise that context has been validated. It is not a demonstrated implementation defect or an untested production dispatch route.

### 5. `QuantizedKernelsTests/PerformanceOptimizationTests/testCacheEfficiencyQuantized()`

At `QuantizedKernelsTests.swift:2179`, the ordinary checks at lines 2288–2292 compare one real FP32 vector's stored scalar payload to 2048 bytes, one INT8 vector's to 512 bytes, and their 4× ratio. `pressureTestSize=2000` only multiplies those observations for a printout; no special 2000-vector storage route is exercised by these assertions.

Ungated `MemoryCompressionTests/testMemoryFootprintReduction` (line 1154) already checks the exact 2048- and 512-byte facts from real stored vectors at a 1000-vector dataset, making the 4× scalar ratio an immediate arithmetic consequence. `testQuantizedStorageLayout` (line 1211) separately checks 128 SIMD4<Int8> lanes and 512 stored bytes. Both Passed. No correctness-property gap is exposed by this skip. Cache-hit, bandwidth, or memory-pressure claims remain unestablished by these scalar-payload facts.

### 6. `QuantizedKernelsTests/PerformanceOptimizationTests/testParallelQuantizedOperations()`

At `QuantizedKernelsTests.swift:2425`, the ordinary assertions compare serial versus parallel **counts** for 10,000 vector conversions (eight chunks of 1250) and 1000 direct INT8 distances (eight chunks of 125). Additional 1/2/4/8-worker timing loops discard their result values. Task-group outputs are appended in completion order; the method deliberately does not assert input order, lane contents, numerical agreement or task-index completeness.

The tasks call the same public per-vector `Vector512INT8(from:params:)` initializer and `QuantizedKernels.euclidean512` used by ordinary `INT8QuantizationTests/testINT8QuantizationBasic` and `QuantizedDistanceComputationTests/testQuantizedEuclideanDistance`; both Passed. The initializer builds local packed storage (QuantizedKernels.swift:107–132), while `euclidean512` (line 691) delegates to the ordinary generic scalar kernel. No batch-count/worker-count branch in these production APIs is selected by the test's dataset size. Other ordinary quantized integration methods are not substituted for explicit concurrency evidence; some dequantize before distance computation.

**Residual:** no ordinary explicit concurrent quantized-conversion/direct-INT8-distance counterpart was found in the inspected comprehensive tests. This matters to a claim of validated concurrent usage, but the skipped method checks only a test harness's output lengths and does not establish such correctness even when run. Retain this exact concurrency/scale limit; it does not by itself require reopening production implementation or adding a release blocker.

## Minimal optional strengthening and class discipline

There is no need to unskip lengthy timing bodies to preserve the currently demonstrated ordinary correctness coverage. If the release owner requires explicit concurrent-context evidence, add two **bounded, ordinary, portable** tests before making that stronger claim:

- Mixed precision: a small deterministic 768D input (for example nine candidates in four uneven chunks), shared immutable query, separate initialized output arrays, task results tagged with original indices. Require exactly one result for each index, finite values, every value matching the serial same-route calculation and an independent reference within the existing justified bound, and untouched sentinels. This tests the missing calling context and tails without a throughput assertion.
- Quantized: a small deterministic set using fixed valid quantization parameters, uneven partitions, task-indexed results. Compare each quantized lane/parameter exactly with serial conversion, each finite distance with serial direct-INT8 calculation, and require every index exactly once. Use exact integer/storage comparisons and bit-pattern parity for the same deterministic numerical path as appropriate; do not invent an empirical cross-algorithm epsilon. A targeted TSan execution would add scoped concurrent instrumentation evidence, not universal race freedom.

These are optional improvements, **required only for an explicit stronger concurrency-validation claim**, not repairs compelled by a demonstrated missing branch. No source change or newly passing result is claimed here. Remaining global compiler/platform/minimum-OS/sanitizer/performance release gates are unaffected.

No gate refactor is recommended, so no new defect-class guard is warranted merely because several optional tests mix timing and correctness. A blanket rule banning ordinary assertions in extended workloads would incorrectly reject valid stress checks and duplicate controls. If a later approved change extracts mandatory assertions or alters a gate, treat both concurrent families together: mechanically verify that the new named ordinary methods are discovered and executed with `VECTORCORE_TEST_EXTENDED=0` and strict performance off, with no host/process capability exclusion; retain mutation controls that fail when a task/result is omitted or a computed result is corrupted. Re-derive all bodies affected by that gate (including the other twelve extended methods), preserving explicit optional classifications rather than assuming their names imply timing-only content. This would close the repeated gating class instead of repairing one method by hand.
