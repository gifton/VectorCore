# Numerical behavior and validation boundaries

This reference separates score ordering from the floating-point computation
that produces a score. It describes this checkout, not a promise of identical
results across releases, hardware, or external providers.

## Distance and score conventions

| API | Returned score |
|---|---|
| `EuclideanDistance`, Euclidean `Operations.findNearest` | Euclidean distance |
| `MatrixDistance.euclideanSquaredMatrix` | Squared Euclidean distance |
| `CosineDistance`, `MatrixDistance.cosineDistanceMatrix` | Cosine distance, `1 - similarity`, with API-specific edge handling |
| `DotProductDistance`, dot-product `Operations.findNearest` | Negative dot product, so smaller is better |
| `TopKSelection.nearestDotProduct512` | Dot-product similarity, larger is better, even though the result field is named `distances` |

Use the selected API's convention when displaying, thresholding, or combining
scores. See [DistanceMetrics](../Sources/VectorCore/Operations/DistanceMetrics.swift)
and [TopKSelection](../Sources/VectorCore/Operations/TopKSelection.swift).

## Top-K ordering

For precomputed selection, numeric scores (including infinities) precede NaNs
in both ascending and descending selection. Exact equal scores, including
signed zeros and pairs of NaNs, use `TieBreaker`:

- `.smallerIndex` is the default and prefers smaller original input indices.
- `.insertionOrder` is equivalent for public array/pointer scans, whose
  indices are scan positions; it does not describe parallel completion order.
- `.smallerValue` supplies no index tie-break. Membership/order within an
  equal-score or NaN group is unspecified.

For positive `k`, selection retains NaNs when necessary to return
`min(k, count)` candidates. Precomputed selection preserves the original
scores and zero signs. These are exact ordering contracts, not
epsilon-based comparisons. Coverage:
[TopKNaNContractTests](../Tests/ComprehensiveTests/TopKNaNContractTests.swift)
and [TopKTieBreakingTests](../Tests/ComprehensiveTests/TopKTieBreakingTests.swift).

Optimized Euclidean selection can rank squared distances before taking square
roots. Two distinct squared scores can round to the same returned root; that
does not retroactively make their admission a tie. Identical ordering across
kernels is only expected for identical computed scores and tie policies.
Wrapper coverage is in
[TopKNaNWrapperTests](../Tests/ComprehensiveTests/TopKNaNWrapperTests.swift).

## Metric computation is a separate contract

Do not infer a package-wide “NaNs propagate” policy from Top-K's ordering
rule. Metrics, normalization, fused kernels, and matrix wrappers have their
own checks and clamps. For example, the scalar `CosineDistance` returns 1
when its magnitude-product guard fails and both squared norms are finite,
while other paths compute normalized rows. Zero norm is not a meaningful
direction, and a returned number is not evidence that the input represented
a valid cosine comparison.

Cosine distance does have one exceptional-value policy across its paths
(`CosineDistance` and its optimized overloads, the fused, pre-normalized,
SoA, FP16 and mixed-precision kernels, `cosineDistance(to:)`,
`MatrixDistance.cosineDistanceMatrix`, and the searches built on them):
a NaN or infinite coordinate in either vector gives a NaN distance, also
against a zero vector, so Top-K ranks such rows after every numeric
distance; the `[-1, 1]` similarity clamp applies only to a finite
similarity; one or two zero vectors give 1. What counts as a zero vector
(the tiny-magnitude threshold) still differs between paths and is
unchanged: the value 1 applies to whatever a path already treats as zero,
so on the fused kernels two vectors whose squared norms are at or below
`Float.leastNormalMagnitude` also give 1. Vectors whose
squared norm overflows Float (`‖v‖² ≥ Float.greatestFiniteMagnitude`) are
outside the supported domain: the per-pair paths can return NaN or 1
where the GEMM path returns the true distance. The mixed-precision kernels
that pair an FP32 vector with an FP16 vector need the product of the two
squared norms to stay below `Float.greatestFiniteMagnitude`; past it they
return 1 for finite inputs in any direction (this follows from
`MixedPrecisionKernels.cosineMixedCore`; it is not a table row). Tests:
[T15CosineExceptionalTableTests](../Tests/ComprehensiveTests/T15CosineExceptionalTableTests.swift)
(every registered implementation) and
[T15CosineSelectionTests](../Tests/ComprehensiveTests/T15CosineSelectionTests.swift)
(every public cosine search route).

There is also a current wrapper limitation:
`BatchOperations.pairwiseDistances` converts GEMM Euclidean scores with
`v > 0 ? sqrt(v) : 0`; that expression maps a NaN to zero.
`MatrixDistance.euclideanSquaredMatrix` instead clamps with
`v < 0 ? 0 : v`, which preserves a NaN. This difference follows directly
from [BatchOperations](../Sources/VectorCore/Operations/BatchOperations.swift)
and [MatrixDistance](../Sources/VectorCore/Operations/MatrixDistance.swift).
Do not use the pairwise wrapper as a non-finite-input validation mechanism.

If an application requires finite, nonzero embeddings, validate those
requirements before selecting an optimized route. An external
`BatchKernelProvider` needs its own numerical conformance tests.

## Min/max reductions, statistics and clamps

The min/max family has one NaN rule: if any input element is NaN, the result
is NaN. This covers `maximum`, `minimum` and `maximumMagnitude` of the Swift and
Accelerate SIMD providers (and `SIMDOperations`), the array providers' `max` and
`min`, `VectorProtocol.min` and `max`, `SIMDStorage.minMax()`, the column
`min`/`max` and magnitude extremes of `Operations.statistics(for:)`, the
`maxAbsoluteError` of `QuantizationSchemes.analyzeQuantizationError`, and the
`PerformanceProfiler` maxima. Index reductions (`minIndex`/`maxIndex`,
`Vector.minElement()`/`maxElement()`) return the index of the first NaN.
Element-wise `elementWiseMin`/`elementWiseMax` (`Vector.min(_:)`/`max(_:)`) give
NaN in a lane where either operand is NaN, and `clip`/`clamped(to:)` keep a NaN
lane NaN. The NaN's sign and payload are unspecified; test with `isNaN`.

The other edge cases: the sign of a zero maximum or minimum is unspecified
(the Swift and vDSP paths may differ); a zero `maximumMagnitude` is `+0`; an
empty pointer-level reduction returns -infinity (`maximum`), +infinity
(`minimum`) or 0 (`maximumMagnitude`) on both providers (library-defined: the
Accelerate paths guard `count > 0` themselves and do not depend on vDSP's
undocumented `N == 0` output); `SIMDStorage.minMax()`
of empty storage is (+infinity, -infinity); `VectorProtocol.min`/`max` of an
empty vector is nil.

Swift's `max(a, b)`/`min(a, b)` keep a NaN only in their first argument,
`Array.max()` only at index 0, and `SIMD.max()` and `Float.maximum` ignore it;
the vDSP header does not specify NaN handling for `maxv`/`minv`/`maxmgv`. The
library therefore checks for NaN explicitly rather than relying on any of them.
The one exception is the Accelerate `clip`: it relies on the documented
`vDSP_vclip` header formula (`D[n] = A[n]; if (D[n] < B[0]) D[n] = B[0];
if (C[0] < D[n]) D[n] = C[0]`, both comparisons false for NaN), verified on one
host only (Apple silicon); Intel and older OS versions are unverified.
For NaN-ignoring extrema, filter NaNs before reducing. The `lerp`/`smoothstep`
parameter clamp is not covered: `t = NaN` returns the other endpoint.

Tests:
[T16ReductionExceptionalTableTests](../Tests/ComprehensiveTests/T16ReductionExceptionalTableTests.swift)
evaluates every implementation and provider with a NaN at every index of 18
lengths and checks that the Swift and Accelerate providers agree;
`Scripts/ci/tests/test_reduction_nan_guard.py` requires every `Float`/`Double`
min/max fold in `Sources` to be registered with a reviewed NaN decision.

## Matrix distances: shape, memory, and precision

`MatrixDistance` packs queries and candidates into contiguous Float matrices.
Its output is row-major: `out[i * candidateCount + j]`. Prepared candidates
reuse candidate packing and, for Euclidean distance, squared norms; prepare
with `normalized: false` for Euclidean or `true` for cosine.

Callers must ensure every vector has the same positive dimension, every
buffer is initialized and sufficiently sized, and all count products and
BLAS dimensions are representable. The `into:` output must contain exactly
`queryCount * candidateCount` elements for nonempty inputs. These APIs use
preconditions and unchecked packing assumptions; they do not validate every
row before reading it. Empty-input `into:` calls return without modifying
the supplied output. Prefer the throwing `Operations` search APIs when their
shape validation and result form suit the task.

The Euclidean GEMM identity is:

```text
distanceSquared(x, y) = dot(x, x) + dot(y, y) - 2 * dot(x, y)
```

Subtracting large, nearly equal terms can lose accuracy for nearby vectors.
Clamping a negative result to zero prevents a negative squared distance but
does not restore lost precision. Direct differences and GEMM can therefore
produce different near-zero distances and rankings. Cosine matrix output is
clamped to [0, 2] by comparisons; a NaN survives those comparisons.
There is no universal relative-error bound for all Float inputs.

The full score output alone uses `4 * queryCount * candidateCount` bytes,
in addition to packed inputs, norms, and other storage. `into:` and
prepared-candidate overloads do not eliminate all allocation. Matrix tests:
[MatrixDistanceTests](../Tests/ComprehensiveTests/MatrixDistanceTests.swift).

## Reduced precision and projections

FP16 and quantized representations change the error budget. Overflow, small
values, conversion rounding, and near-ties can affect scores and ordering.
Mixed-precision configuration or validation results are not blanket numerical
guarantees for arbitrary data; see the actual path and
[MixedPrecisionRangeValidationTests](../Tests/ComprehensiveTests/MixedPrecisionRangeValidationTests.swift).

PCA retains its compensated Double center in the public `[Double]` mean and
uses `Float(Double(x) - mean)` consistently in fit and single/batch transforms.
The exactly representable translation fixtures in PCACenteringContractTests
exercise the avoided early center rounding. Input quantization, Float residual
rounding and Float matrix products remain; arbitrary translation invariance is
not promised. Nonfinite original inputs are rejected, and successful model
statistics/projections must be finite. Provider failures propagate unchanged.

PCA and UMAP introduce additional approximation and conditioning concerns.
A fixed random seed does not promise bitwise equality across factorization
providers, hardware, or toolchains. Degenerate eigenspaces can have different
valid bases. See [Linear Algebra and Projection](Linear_Algebra_and_Projection.md).

## UMAP calibration and Float Gram context

UMAPRangeContractTests covers dimensionless output-curve fitting against a retained
derivative-free reference and Double smooth-kNN membership/residual controls against
110/160-digit Decimal references. Float coefficient publication has an explicit curve
fidelity bound; smooth bandwidth uses a limiting tie policy followed by a mean floor.
A floor-dominated row is not required to satisfy the unfloored target equation, and
an ill-conditioned equation does not promise accurate sigma identification.

Both UMAP overloads reject nonfinite floating options and supplied coordinates,
preflight schedule/count representation, and throw on nonfinite Float SGD arithmetic
before clipping. This supports finite successful output on the checked routes, not
meaningful/converged layouts for every finite input or stochastic layout-scale
equivariance. The public array inputs are not mutated. Internal optimizer failure
may leave its inout buffer partially updated.

KNNGraph.bruteForce still uses exhaustive Float Gram scoring. Common offsets can
cancel nearby distances; large finite values can overflow norms/dots. The
`[10000,10002,10001]` fixture records actual scores/neighbors against independent
Double differences as a characterization, not a repaired direct-distance guarantee.
No truncated-candidate rerank or hidden routing change is included. See
[the explicit calibration and boundary contracts](Linear_Algebra_and_Projection.md#umap-calibration-and-checked-numerical-domain).

## Choosing checks and tolerances

Use absolute error near zero and relative error at scale, based on the
operation and application's requirements. For search, inspect neighbor
membership and rank stability as well as score error. Use exact comparisons
for contracts such as indices, counts, and Top-K tie ordering; a fuzzy
comparator can violate transitivity.

Throwing construction does not make every later operation checked. Unsafe
storage, integer sizing, lifetime, and concurrency obligations are described
in [Memory Alignment](Memory_Alignment.md). Tests establish the cases they
exercise, not correctness for every dimension, value, or execution route.

L03 curve-step acceptance compares a Neumaier-compensated per-sample SSE
change, with separately formed new/old residuals, rather than subtracting
separately rounded objective totals. Only finite strictly negative changes are
accepted; stationarity remains the true SSE-gradient infinity norm `<= 1e-8`.
The non-dyadic regression controls are named
`UMAPRangeContractTests.curveOrdinaryNonDyadicScales`.
