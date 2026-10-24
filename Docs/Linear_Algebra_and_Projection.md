# Linear algebra and projection

VectorCore provides dense factorization primitives, reusable PCA models, and
UMAP layout. These APIs are CPU-side numerical building blocks, not an ANN
index, embedding model, clustering pipeline, or visualization UI.

## Dense factorization interface

[`LinearAlgebraProvider`](../Sources/VectorCore/LinearAlgebra/LinearAlgebraProvider.swift)
uses flat column-major Float buffers: element (i, j) of an m-by-n matrix is
at `j * m + i`. Do not pass a row-major distance matrix without converting it.

| Operation | Shapes and result conventions |
|---|---|
| `qrThin(_:rows:columns:)` | m ≥ n ≥ 1; Q is m-by-n, R is n-by-n |
| `svdThin(_:rows:columns:)` | m, n ≥ 1; k = min(m, n); U is m-by-k, singular values descend, Vᵀ is k-by-n |
| `symmetricEigen(_:dimension:computeEigenvectors:)` | n-by-n input; only the lower triangle is read; eigenvalues ascend; eigenvectors are columns when requested |

The built-in providers validate matrix shapes and the 32-bit LAPACK dimension
limit, and throw on reported backend failure. They operate on in-memory
matrices, not streamed or distributed storage. A representable shape is not
evidence that the allocation or factorization is practical.

`Operations.linearAlgebraProvider` defaults to
`LAPACKLinearAlgebraProvider()` on Apple platforms. A task-local override can
select `SwiftLinearAlgebraProvider()`. The fallback does not make the package
as a whole Linux-supported.

For exactly rank-deficient SVD, the Swift fallback does not complete directions
corresponding to zero singular values to an orthonormal basis:

- For tall or square input (m ≥ n), the affected directions are U columns.
- For wide input (m < n), the fallback decomposes the transpose and swaps
  factors, so the affected directions are Vᵀ rows instead.

Those directions can be zero vectors. This does not affect their contribution
to `U * diag(s) * Vᵀ`, but callers must not assume both factors provide an
orthonormal basis for zero-singular-value directions. LAPACK completes those
bases. The shape-dependent behavior follows from the transpose branch and
zero-column normalization in
[`SwiftLinearAlgebraProvider.svdThin`](../Sources/VectorCore/LinearAlgebra/SwiftLinearAlgebraProvider.swift).
See
[LinearAlgebraProviderTests](../Tests/ComprehensiveTests/LinearAlgebraProviderTests.swift)
for reconstruction, shape, ordering, and provider comparisons.

## PCA

`PCAModel.fit(_:components:config:)` fits a linear projection
`y = W * (x - mean)`. The public `components` buffer is k-by-d
**row-major**, unlike the factorization interface. The model exposes the mean,
component count, input dimension, explained variance, and explained-variance
ratios. The public `mean` is `[Double]`: it stores the widened center actually
used during fit and both transforms. Components, statistics, and projections
remain Float. Callers migrating from `[Float]` mean should reconstruct in Double
or explicitly choose where to round the center.

`PCAConfig` defaults to centering, oversampling 8, and two power iterations.
Fitting uses randomized SVD with QR-stabilized subspace iteration. If the
sketch width reaches `min(sampleCount, inputDimension)`, it uses an exact
thin SVD instead (the route:
`PCACenteringContractTests.sketchBoundariesAndProviderIdentity`; with the
default LAPACK provider the fit succeeds across `sgesdd`'s path-5 boundary:
`T19LAPACKWorkspaceQueryTests.exactPCASweepCrossesPathFive` and
`reportedCellsSucceed`). Setting `center: false` computes an uncentered truncated
SVD; its reported second-moment quantities should not be interpreted as
centered variances.

Fitting requires at least two equal-dimension, nonempty vectors. The component
count must be in `1...min(n - 1, d)` when centered, or
`1...min(n, d)` when uncentered. Nonnegative oversampling, including `Int.max`,
is clamped before addition; reaching the shape limit selects exact fallback
(the route: `PCACenteringContractTests.sketchBoundariesAndProviderIdentity`;
with the default LAPACK provider the fit succeeds on near-square samples on
both sides of `sgesdd`'s path-5 boundary, tall and wide:
`T19LAPACKWorkspaceQueryTests.exactPCASweepCrossesPathFive` and
`reportedCellsSucceed`).
PCA checks its LP64 dimensions and element/byte representability before its
allocations and GEMM calls. Representability does not guarantee allocation
success or practical memory use.

Fit and both transforms reject nonfinite input after structural checks. Means
use deterministic compensated Double accumulation; every centering path uses
`Float(Double(x) - mean[j])`. Nonfinite centered residuals, model statistics,
components, or projections throw a stage-specific error. Provider errors retain
their identity. Float panels and GEMM still round and can overflow; this is not
an arbitrary translation-invariance or conditioning guarantee. See
[PCACenteringContractTests](../Tests/ComprehensiveTests/PCACenteringContractTests.swift)
for representable translation, finite-boundary, and sketch-shape controls.

These component-count limits are structural, not a measurement of the data's
rank. PCA copies rows of the SVD's Vᵀ into `components`. With the Swift
provider, a wide, rank-deficient SVD panel can therefore yield zero component
rows for zero singular values rather than an orthonormal null-space basis.
This can occur in either the exact or randomized path. For example, two
identical 3D samples become a zero 2-by-3 matrix after centering; fitting one
component with the Swift provider can return a zero axis. Do not treat every
returned component as a unit direction under this fallback limitation.

```swift
import VectorCore

let sample = [
    DynamicVector([Float(0), 0, 0]),
    DynamicVector([Float(1), 0, 0]),
    DynamicVector([Float(0), 2, 0]),
    DynamicVector([Float(1), 2, 1])
]
let model = try PCAModel.fit(sample, components: 2, config: PCAConfig(seed: 7))
let projected: [[Float]] = try model.transform(sample)
let newPoint: [Float] = try model.transform([Float(0.5), 1, 0])
precondition(projected.count == sample.count)
precondition(projected.allSatisfy { $0.count == 2 && $0.allSatisfy { $0.isFinite } })
precondition(newPoint.count == 2 && newPoint.allSatisfy { $0.isFinite })
print("Projected \(projected.count) points to \(model.componentCount) dimensions")
```

`Operations.pca(_:components:config:)` is the one-shot fit-and-transform
convenience. Use `PCAModel` when fitting on a representative sample and
transforming other data in caller-sized batches. Fit still packs the full
sample and allocates intermediate matrices; it is not an online/incremental
fit. Batch transform also packs its input and allocates output.

For sketch width ℓ, the randomized path's matrix passes scale with n·d·ℓ
and the number of power iterations, plus panel factorizations. This explains
why sample size and sketch width matter, but is not a measured throughput
claim. See [PCA implementation](../Sources/VectorCore/LinearAlgebra/PCA.swift)
and [PCATests](../Tests/ComprehensiveTests/PCATests.swift).

## KNNGraph and UMAP

[`KNNGraph`](../Sources/VectorCore/ManifoldLearning/KNNGraph.swift) stores a
directed compressed-sparse-row graph. Row i occupies
`rowOffsets[i]..<rowOffsets[i + 1]`. Its contract requires monotone offsets
from zero to edge count, valid Int32 neighbor indices, no self-loops, and
finite nonnegative distances. Rows can have different degrees and need not
be distance-sorted. A zero distance is allowed for duplicate points.
Construction checks every offset is bounded and nondecreasing before any row
indexes the entry arrays. Malformed CSR throws invalidData; this checked contract
is exercised by UMAPRangeContractTests' bounded malformed-offset matrix. Duplicate
neighbor IDs, unsorted rows and isolated rows remain allowed. Count/byte guards do
not promise successful allocation or a corpus memory budget.

There are two UMAP entry points:

- `Operations.umap(vectors, dimensions:graph:config:)` accepts a supplied
  graph or builds one by brute force. Vectors provide PCA initialization by
  default; random initialization is also available.
- `Operations.umap(graph:dimensions:initialCoordinates:config:)` uses a
  graph directly. Supply matching initial coordinates or let it initialize
  randomly. The graph-only overload does not derive PCA initialization from
  `config.initialization`.

Graph rows, vector rows, and initial-coordinate rows must use the same point
ordering. `config.neighbors` controls the internal graph builder, not the
degree of an already supplied graph; configuration validation still runs.

```swift
import VectorCore

let vectors: [DynamicVector] = (0..<12).map { (i: Int) -> DynamicVector in
    let x = Float(i % 4)
    let y = Float(i / 4)
    let z: Float = Float(i % 3) * 0.1
    return DynamicVector([x, y, z])
}
let graph = try KNNGraph.bruteForce(vectors, neighbors: 3)
let layout = try Operations.umap(
    graph: graph,
    dimensions: 2,
    config: UMAPConfig(neighbors: 3, epochs: 20, seed: 7))
precondition(layout.pointCount == vectors.count && layout.dimension == 2)
precondition(layout.coordinates.allSatisfy { $0.isFinite })
print("Layout coordinates: \(layout.coordinates.count)")
```

This small example checks API usage and output shape, not layout quality.
The exhaustive Float Gram builder considers all pairs: distance work is O(n²·d),
and insertion-based neighbor selection can add O(n²·k) work. Its Gram scratch
panel is bounded to at most `min(n, 256) * n` Floats, but it also holds the
packed data and O(n·k) graph. It is not a corpus-scale ANN substitute.

UMAP builds fuzzy graph weights and optimizes coordinates with seeded SGD.
A supplied ANN graph can avoid brute-force construction, but layout still
has graph-sized work/storage and depends on graph quality, configuration, and
initialization. The API returns coordinates, not a fitted transform for new
points. A low-dimensional visual separation is not proof of semantic clusters.
See [UMAP implementation](../Sources/VectorCore/ManifoldLearning/UMAP.swift)
and [UMAPTests](../Tests/ComprehensiveTests/UMAPTests.swift).

## UMAP calibration and checked numerical domain

`UMAPRangeContractTests` exercises these explicit contracts; they are not a claim
of bitwise equivalence to another UMAP implementation or whole-layout convergence.

The output curve keeps the 300-point target on t in [0,3], with
r = Double(minDist)/Double(spread), and fits `1/(1+alpha*t^(2*beta))` in log-positive
parameters. Damped Double normal equations run for at most 200 iterations and
must meet finite SSE-gradient infinity norm <= 1e-8. Failure to reach stationarity
throws operationFailed. This is an engineering convergence test, not a global
optimum proof. The retained independent derivative-free multi-start fixture
solver compares curve values and SSE at pinned Binary32 inputs.

The returned Float exponent b is rounded before mapping
`log(a)=log(alpha)-2*Double(b)*log(Double(spread))`. Both coefficients must remain
positive and finite, and their dimensionless curve must differ from the converged
Double curve by at most `64*Float.ulpOfOne` on the same grid. Sufficiently accurate
subnormal a is supported; overflow, underflow to zero and coarse quantization
throw operationFailed with a range diagnostic. Positive finite spread alone is
not a guarantee that its physical Float coefficient representation is supported.

Smooth-kNN keeps rho as the smallest positive stored row distance. For k stored
entries the target is log2(Double(k)+1). Duplicate IDs count separately here,
then merge by maximum directed membership during fusion. Isolated rows retain
rho=sigma=0. If zero shifted distances already meet/exceed the target, the explicit
pre-floor bandwidth is zero. Otherwise a checked logarithmic bracket and at most
64 bisections must meet residual `1e-5*max(1,target)` in Double. Nearly flat equations
need not identify sigma accurately; memberships and residual are the controls.
The existing `1e-3` local/global mean floor follows calibration and may intentionally
change the target residual. Zero bandwidth gives weight 1 only for zero shifted
distance, otherwise 0. Fused weights retain canonical ordering and duplicate-max
probabilistic union; Float underflow-to-zero edges are dropped.

Both public entry points validate finite spread/minDist/learningRate, existing
option signs, and an exact-integer Double schedule domain. Effective epochs and
negativeSampleRate are at most 2^53-1 (or native Int.max if smaller), and
`epochs*max(1,negativeSampleRate)` must fit that bound. This is not a work budget:
large accepted counts can still be impractical. Tests use pure boundary helpers,
not enormous accepted loops. Runtime cadence/count arithmetic is checked before
floating-to-Int conversion and throws operationFailed if it leaves the domain.

Point/dimension/count/byte representation failures use invalidDimension;
configuration/schedule-domain failures use invalidOperation. CSR and nonfinite
supplied coordinates use invalidData; mismatching coordinate shapes retain
dimensionMismatch. Every supplied row width is validated before values are copied.
Both APIs reject impossible point counts before initialization allocation. The
KNN builder separately checks LP64 dimensions, data/edge/block products, offsets
and element bytes before allocation or BLAS conversion. These checks do not
establish extreme vendor-internal workspace safety or successful allocation.

PCA initialization uses `10*(Double(value)/Double(maxAbs))` before Float projection,
then the existing jitter; zero projection keeps zero plus jitter. PCA/provider
failures retain their identities. Initial and returned coordinates must be finite.
SGD retains Float coordinates, equations and clipping, but checks differences,
squares, powers, coefficients, unclipped gradients, steps and updates before a
nonfinite value can be hidden by clipping. Thus finite extreme input or learning
rate can throw operationFailed instead of producing a layout. Public input arrays
remain unmodified; internal inout optimizer mutation on failure is not transactional.

The KNN scoring route is unchanged: Float norms and GEMM dots feed Gram subtraction.
It exhaustively enumerates computed scores, not stable direct Euclidean distances.
The pinned `[10000,10002,10001]` characterization compares against independent
Double differences and records the cancellation divergence. There is no reranking
of a truncated candidate set, new distance route or numerical-oracle promise.
A direct Euclidean distance route remains separate P02-dependent work.

The LM acceptance check uses the algebraic objective change
`Σ (newF - oldF) * ((newF - target) + (oldF - target))`, with separately
formed residuals and Neumaier compensation. Every term and the result must be
finite, and the change must be strictly negative. This avoids comparing two
separately rounded total SSEs near stationarity; equality or worsening is still
rejected. The `1e-8` gradient criterion and 200-iteration limit are unchanged.
`UMAPRangeContractTests.curveOrdinaryNonDyadicScales` covers the retained
`.25 / Float(.7)` stall and `.5 / 1.5` control using independent minima.

## Reproducibility and interpretation

Record data ordering, preprocessing, graph construction, configuration,
seed, provider, and toolchain. Seeded randomness does not guarantee bitwise
parity across backends or hardware. Degenerate components can admit multiple
valid bases; small numerical changes can affect neighbor selection and
nonlinear layouts. Check reconstruction/projection error or neighborhood
quality for the intended task, rather than assuming that reduced dimensions
improve retrieval or clustering. See [Numerical Behavior](Numerical_Behavior.md).
