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
ratios.

`PCAConfig` defaults to centering, oversampling 8, and two power iterations.
Fitting uses randomized SVD with QR-stabilized subspace iteration. If the
sketch width reaches `min(sampleCount, inputDimension)`, it uses an exact
thin SVD instead. Setting `center: false` computes an uncentered truncated
SVD; its reported second-moment quantities should not be interpreted as
centered variances.

Fitting requires at least two equal-dimension, nonempty vectors. The component
count must be in `1...min(n - 1, d)` when centered, or
`1...min(n, d)` when uncentered. Configuration and shape checks are not a
comprehensive finite-input or allocation-safety validator.

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
Callers must supply structurally valid buffers; do not treat construction as
a general-purpose parser for arbitrary untrusted data.

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
The reference graph builder considers all pairs: distance work is O(n²·d),
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

## Reproducibility and interpretation

Record data ordering, preprocessing, graph construction, configuration,
seed, provider, and toolchain. Seeded randomness does not guarantee bitwise
parity across backends or hardware. Degenerate components can admit multiple
valid bases; small numerical changes can affect neighbor selection and
nonlinear layouts. Check reconstruction/projection error or neighborhood
quality for the intended task, rather than assuming that reduced dimensions
improve retrieval or clustering. See [Numerical Behavior](Numerical_Behavior.md).
