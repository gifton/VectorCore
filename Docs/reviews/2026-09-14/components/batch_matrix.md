# Component 06 — batch and matrix operations

Revision: 8b02e75bca7055ae11170873d15c06f1538709b7. Read-only production review. No package builds or production/test edits.

## Scope and validation

Read all of `Operations/{BatchOperations,SyncBatchOperations,ExecutionOperations,TransformOperations,MatrixDistance,VectorEntropy}.swift`, `Operations/Kernels/{BatchKernels,BatchKernels_SoA}.swift`, `Utilities/TriangularMatrix.swift`, and `Operations.swift:773–883`. Traced `DistanceMetric`, `VectorMath` entropy facade, `VectorCore` public wrappers, and downstream `VectorIndex/Kernels/CentroidBatchScore.swift`. Read numerical/boundaries/API-map docs; reviewed MatrixDistanceTests, BatchGEMMRoutingTests, VectorEntropyTests, ExecutionOperationsTests, and searched batch/SoA tests for relevant boundaries. Did not read every 2700-line SoA test case; checked relevant coverage by search and selected source.

Executable evidence: `/private/tmp/vectorcore-review-2026-09-14/batch_probe.swift` linked against existing Debug library objects with root's `build_probe.py`. Output: `batch_probe.log`. Separate crash cases: `batch_singleton.log`, `batch_zero_size.log`, both exit 133. No shared build invocation. Release was not independently repeated.

## Prioritized verified candidates

### 1. P1 correctness / memory safety — public aggregations omit collection shape validation (root-owned lead, corroborated)

Locations: `Operations.swift:780–802,811` and `848–861`.

`centroid` derives dimension from first row then gives that count to an unchecked pointer addition for every subsequent `toArray()`. A short DynamicVector reads outside that allocation; a long row is silently truncated. Empty input calls `try! V.create` with zero elements even for a fixed dimension and traps. `statistics` similarly indexes every row up to the first dimension, trapping on short rows or silently dropping long-row coordinates. These public methods have no documented equal-dimension precondition. Root reproduced these cases; this agent confirmed the exact unsafe call/count and adjacent missing checks.

Remedy: explicit empty and ragged-input policy; validate every row before all aggregations, preserve thrown factory errors, and avoid `try!` at generic factory boundaries. Mechanical coverage should span all public collection reductions, both mismatch directions, first/middle/last bad row, empty collection, and factory rejection. Existing `testCentroidBasic` / `testStatistics_BasicMeanAndMagnitudes` only exercise valid two-row shapes.

### 2. P2 correctness — parallel batch standard deviation loses the variation it measures

Location: `BatchOperations.swift:367–386,392–398`; serial contrast `604–610`; gate `357`.

At 1000 rows, the algorithm changes from centered squared deviations to Float `E[m²]−E[m]²`. Cancellation destroys small spreads around ordinary large magnitudes; summing squared magnitudes can overflow even when individual magnitudes and the correct standard deviation are representable.

Verified public probe: alternating scalar vectors 10000/10001 return standard deviation `0.49999908` at N=999, then exactly `0` at N=1000 and N=1024, where the correct population standard deviation is 0.5. N=1000 identical Float(1e18) vectors return `inf`; the true spread is zero. At N=999 even centered serial results are inaccurate at that scale (`6.25e12`) because Float summation perturbs the computed mean; the parallel issue is not confined to the final clamp.

Remedy: Double or stable online/mergeable mean/M2 accumulation, including stable chunk combination. Test both sides of N=1000, identical inputs, large offsets with small nonzero variance, and large finite magnitudes; compare against Double reference. No BatchOperations.statistics coverage found by repository search.

Confidence: reproduced Debug, algorithmic explanation independent of scheduling. Value at 999 for alternating inputs differs slightly because the population has unequal counts; this is expected and does not explain the N=1000 loss.

### 3. P2 correctness — zero-query SoA cosine overwrites unused output capacity

Locations: `BatchKernels_SoA.swift:930,1025,1126`; safe boundary `1737–1753`; public dispatch `1201–1247`.

Normal branches write exactly `soa.count` values, and the safe wrapper explicitly permits `out.count >= soa.count`. Each zero-query branch calls `out.initialize(repeating: 1.0)`, writing *all* supplied elements, including a reusable output buffer's untouched suffix. This affects 2-way, 4-way, and tiled paths and all three dimension-specific public cosine entry points.

Verified Safe512 public probe: N=1,4,4097 with output length N+2 initialized to −77; both suffix sentinels become 1 in every case. All inputs and output capacity satisfy the public wrapper's validation. N4097 exercises the >8 MiB tiled route for dimension512.

Remedy: initialize/update only the first N elements. Add a mechanical sentinel test across dimensions, zero/nonzero queries, N=0/1/3/4/5, tile edge 255/256/257 and routing boundaries (512:4096/4097; 768:2730/2731; 1536:1365/1366), with oversized and offset output views. Existing zero-query tests mainly use allocating exact-size results and cannot catch this class.

Confidence: reproduced all three kernel routes; sibling dimensions share those implementations directly.

### 4. P2 correctness — entropy is not invariant under finite changes of scale

Locations: `VectorEntropy.swift:37–49,81–89`; public contract `VectorMath.swift:488–507`.

The public entropy definition normalizes absolute values into probabilities, but an absolute `absSum <= Float.ulpOfOne` cutoff declares nonzero uniform vectors zero-entropy. At the other end, Float absolute-sum overflow returns zero on the small path and NaN on the large path even though a normalized distribution and its entropy are finite.

Verified: `Vector<Dim4>(repeating:1e-9).entropy == 0`, expected log4≈1.386; `Vector<Dim128>(repeating:1e-10).entropy == 0`, expected log128≈4.852. Uniform `.greatestFiniteMagnitude` yields0 for Dim4 andNaN for Dim128. Nonfinite *input* handling is a separate issue; these inputs are finite.

Remedy: scale by maximum absolute component or accumulate in Double, distinguish all-zero vectors from merely small values, then compute normalized probabilities. Preserve any explicitly chosen tiny-probability policy separately. Test scale sweeps and zero-padding equivalence across the 64-element route boundary. Existing `testEntropy_ScaleInvariance_Positive` and `testEntropy_ScaleStability_LargeMagnitudesVsSmall_Large` cover moderate scales; the intentional `testEntropy_UlpThreshold_IgnoresNearZero` tests a tiny probability next to1 and does not justify treating a uniform small vector as zero.

Confidence: reproduced.

### 5. P2 correctness — valid 1×1 full matrix traps during triangular conversion

Location: `TriangularMatrix.swift:218–223`.

`fromFullMatrix([[Float(0)]])` passes nonempty and square checks, then indexes `[0][1]` for its initial value although singleton matrices have no upper-triangle element. TriangularMatrix(size:1) otherwise supports zero stored elements. Probe exited133, `Fatal error: Index out of range`.

Remedy: use an existing entry (e.g. `[0][0]`) as the unused initializer value or construct empty storage for size1. Test size0/1/2, ragged matrices, and successful singleton round-trip. Memory agent owns TieredStorage COW/lifetime and large-matrix update complexity; deduplicate if this singleton case is already in that report.

### 6. P2 correctness — nonpositive explicit process batch sizes trap or silently drop every input

Locations: `BatchOperations.swift:175–186,535–539`.

The throwing public `process` accepts an unrestricted Int but does not validate positive size. With one valid DynamicVector, batchSize0 traps in Swift.stride (`Stride size must not be zero`, exit133); batchSize−1 returns an empty array without ever invoking transform. The same stride pattern exists in both serial and parallel routes.

Remedy: reject nonpositive sizes with a documented error before either route. Test 0/−1/1/>count and N999/N1000, asserting every item processed exactly once for valid sizes. This is distinct from configuration not being consulted or unbounded task scheduling, which execution agent owns.

### 7. P2 correctness / API contract — generic pairwise assumes zero self-distance for every metric

Locations: `BatchOperations.swift:550–556,577–581`; related `ExecutionOperations.swift:388–395`, `SyncBatchOperations.swift:341–348`.

Built-in `DotProductDistance` is a negative similarity and does not have zero diagonal. Public `BatchOperations.pairwiseDistances([DynamicVector([2])], metric:DotProductDistance())` returns `[[0]]`, whereas the supplied metric evaluates self to−4. Root's `Array<Vector<Dim768>>.cosineSimilarityMatrix` finding is another manifestation plus a separate sign error: it normalizes and passes negative dot product as a similarity (`TransformOperations.swift:164–169`), obtaining negative off-diagonal similarities and zero diagonal.

There is also route-dependent symmetry behavior for custom `DistanceMetric` implementations. A directed `a[0]−b[0]` metric gives m[1][0]=−1 for N2, and+1 for N100, because serial mirrors the upper triangle whereas parallel computes both orientations. No symmetry/zero-self requirement is declared in DistanceMetric, but `pairwiseDistances` documents a symmetric result; classify directed support as contract clarification rather than demand arbitrary asymmetric support. The built-in negative-dot counterexample requires no custom metric.

Remedy: either restrict the symmetric zero-diagonal shortcut to metrics with an explicit corresponding contract, or compute diagonals with the supplied metric and state required symmetry; correct the similarity convenience separately. Mechanical tests should compare every built-in metric's pairwise matrix to per-pair results around N99/N100 and GEMM thresholds, including zero vectors. Existing BatchGEMMRoutingTests cover Euclidean/cosine/Manhattan on moderate nonzero vectors, not dot-product diagonals.

## Architecture and performance opportunities (source-derived, not measured regressions)

1. `MatrixDistance` requires packing even when downstream data already lives in contiguous matrices. PreparedCandidates avoids repeated candidate packing but still owns a packed copy and every call repacks queries. This is a concrete integration gap: `../VectorIndex/Sources/VectorIndex/Kernels/CentroidBatchScore.swift:10–13` explicitly says it rejected MatrixDistance because of mandatory copies and directly calls BLAS on its existing contiguous cache. Consider a checked contiguous row-major view/prepared-buffer API with explicit ownership and stride; keep GPU scheduling downstream. No timing estimate claimed.
2. `gemmPairwiseTyped` holds a flat N² output and copies it into N row arrays (`BatchOperations.swift:269–282`), producing about8N² bytes of score storage while reshaping, plus two packs of the same input. Output full-matrix O(N²) is already documented, so this is avoidable peak allocation rather than a new undocumented complexity defect. Tiled/streaming consumers or a flat-result API would avoid the extra matrix copy.
3. Public `BatchOperations.findNearest` materializes O(N) distances and tuple pairs even when k is tiny (`416–464,508–524`). Optimized parallel helpers additionally call `candidates.chunked`, materializing arrays merely to obtain boundaries (`773–789` and siblings); kernel calls still index original candidates. Use integer ranges directly and consider per-chunk top-k with deterministic merge for O(chunks*k) retained scores. No measured speedup asserted.
4. Legacy internal `SyncBatchOperations` and most `ExecutionOperations` duplicate newer public implementations, retain independent naive selection rules/shape validation, and have few or no live public callers. `ExecutionOperations.reduce` injects initialResult into every chunk and recursively again (`346,359–364`), wrong for any nonidentity initial accumulator; it also does not declare associativity/order requirements. `validateDimensions` checks only the first candidate in Release (`535–545`), and `validateConsistentDimensions` samples every hundredth (`554–575`). These are dormant defects today for internal methods without public reachability; consolidating/deleting unused engines reduces risk of rearming them in a future routing change. Public array transform extensions remain live; do not remove that functionality accidentally.
5. ExecutionOperations.distanceMatrix and TriangularMatrix.buildParallel create one Task per upper-triangle element (O(N²) tasks). Execution agent owns task fanout; memory agent owns TriangularMatrix's TieredStorage repeated-copy amplification. Do not duplicate their findings.
6. Documentation claims such as SoA ~1.5–2× throughput (`BatchKernels_SoA.swift:115–117`), Manhattan ~10–15% faster (`1343`), and the `prefetch` helper (`459–467`, which merely performs an unused load) need actual benchmark/compiler evidence. The Manhattan methods do not use the advertised vDSP SoA algorithm at all: they traverse original candidate AoS storage and scalar-construct abs SIMD lanes. Classify as stale/unsupported performance claims, not an established runtime regression.

## Strengths

- GEMM path uses opaque prepared candidates, verifies normalized-vs-raw metric preparation and first-query dimension/output size, and exposes reuse directly. Documentation candidly describes precision/shape limits.
- SoA blocked/tiled bodies preserve straightforward accumulation over lanes and cover candidate tails. No finite-input missing-lane/tail defect found in the fully read AoS/SoA kernels.
- Batch nearest selection delegates to shared deterministic TopKSelection for both small-k heap and large-k sort; generic parallel results retain original global indices.
- Current matrix route is constrained to fixed-size optimized types, so documented unchecked packing assumptions do not introduce a ragged DynamicVector route through BatchOperations.pairwiseDistances.

## Rejected / documented limitations

- MatrixDistance ragged rows, zero dimensions, output-size mismatch, and Int/BLAS overflow are explicitly documented unchecked caller contracts in Docs/Numerical_Behavior.md. Not reported as new memory-safety defects of that API.
- GEMM near-coincident cancellation and pairwise Euclidean NaN→zero conversion are already documented. Do not relabel as novel findings; finite moderate-input tests cannot establish universal ranking parity.
- Generic DistanceMetric itself documents equal dimension as a precondition. Batch findNearest's nonthrowing direct-metric call is consequently not automatically a new validation defect.
- TriangularMatrix deliberately stores no diagonal and ignores the lower triangle on import; rejecting nonsymmetric matrices is not part of its advertised import contract. The singleton index trap is independent.
- Range kernels' ordinary nonzero lower-bound indexing correctly uses `i−range.lowerBound`; wrappers pass matching sub-buffer counts. Invalid unsafe output capacity is the caller's boundary obligation. The SoA zero-query suffix overwrite is different: it corrupts values within a valid oversized buffer that all nonzero-query paths leave alone.

## Limits

No performance measurements or exhaustive external provider tests; no claim of Release execution validation from this component. Root owns package-wide Debug/Release results and deduplication. All runnable findings use public APIs; internal-only findings are explicitly separated.
