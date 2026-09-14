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
when its magnitude-product guard fails, while other paths compute normalized
rows. Zero norm is not a meaningful direction, and a returned number is not
evidence that the input represented a valid cosine comparison.

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

PCA and UMAP introduce additional approximation and conditioning concerns.
A fixed random seed does not promise bitwise equality across factorization
providers, hardware, or toolchains. Degenerate eigenspaces can have different
valid bases. See [Linear Algebra and Projection](Linear_Algebra_and_Projection.md).

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
