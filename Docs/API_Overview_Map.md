# VectorCore API overview

This is a map of the public surface in this checkout, not an exhaustive symbol
reference or a release compatibility matrix. Start with the
[README](../README.md); see [Package Boundaries](Package_Boundaries.md) for the
surrounding libraries and [Numerical Behavior](Numerical_Behavior.md) for limits.

## Vector types

| Type | Dimension | Relevant distinction |
|---|---|---|
| `Vector<D>` | Encoded by a `Dimension` type | Generic fixed-dimension vector; dimension safety does not remove runtime validation requirements |
| `DynamicVector` | Runtime | Useful when the dimension is not known at compilation |
| `Vector384Optimized` | 384 | Specialized SIMD4 storage |
| `Vector512Optimized` | 512 | Specialized SIMD4 storage and additional fused search kernels |
| `Vector768Optimized` | 768 | Specialized SIMD4 storage; dedicated Euclidean Top-K and fused cosine scoring routes in `Operations.findNearest` |
| `Vector1536Optimized` | 1536 | Specialized SIMD4 storage; dedicated Euclidean Top-K and fused cosine scoring routes in `Operations.findNearest` |

The optimized types and `DynamicVector` conform to `UnifiedVectorBuffer`.
The four optimized types also conform to `SoACompatible`; `DynamicVector` does
not. These conformances are separate from `VectorProtocol` and do not imply
that every type participates in every optimized route.

Storage and copying depend on the type. Generic `DimensionStorage` defaults to
managed heap storage above 16 elements. There is no universal stack-only,
allocation-free, or fastest-type guarantee. See
[vector implementations](../Sources/VectorCore/Vectors/) and
[memory contracts](Memory_Alignment.md).

## Operations by purpose

| Need | Entry point | Result or boundary |
|---|---|---|
| Arithmetic, dot product, norms | Vector methods | Check the concrete type's checked and unchecked forms |
| Distance policy | `DistanceMetric` and built-in metrics | Euclidean, cosine, Manhattan, Chebyshev, Hamming, Minkowski, negative dot product |
| Search a supplied collection | `Operations.findNearest`, `findNearestBatch` | Async, throwing; candidate indices and scores, not a persistent index; throws for `k <= 0`, empty input or a dimension mismatch before any provider or metric work |
| Search with a closure metric | `VectorMath.nearestNeighbors` | Synchronous, nonthrowing; `k <= 0` or empty input returns `[]` without metric calls; otherwise exactly `n` metric calls in index order, ordered by `TopKSelection`'s default policy |
| Select existing scores | `TopKSelection.select` | Ascending indices/scores, NaN after numeric, ties by `TieBreaker` (default `.smallerIndex`); small `k` streams into a bounded heap of `min(k, n)` |
| Bulk work | `BatchOperations` | Includes processing, pairwise distances, map, filter, and statistics |
| Query-by-candidate distances | `MatrixDistance` | Flat row-major squared-Euclidean or cosine distances; reusable candidate packing |
| Aggregation and transforms | `Operations.centroid`, `normalize`, `statistics` | Requirements differ by vector protocol and scalar type |
| Dense factorizations | `LinearAlgebraProvider` | Thin QR, thin SVD, symmetric eigendecomposition; column-major buffers |
| Linear projection | `PCAModel.fit`, `transform`; `Operations.pca` | Reusable model or one-shot fit-and-transform |
| Nonlinear layout | `Operations.umap` | Coordinates from vectors or a `KNNGraph`; not a fitted out-of-sample transform |
| Graph interchange | `KNNGraph` | Validated CSR neighbor graph, including a reference brute-force builder |
| Quantized representation | [Quantization primitives](../Sources/VectorCore/Quantization/QuantizationSchemes.swift) | Representation/conversion support, not a compressed ANN index |
| Serialization | [Vector serialization](../Sources/VectorCore/Vectors/VectorSerialization.swift) and [binary protocols](../Sources/VectorCore/Serialization/BinaryProtocols.swift) | Vector encoding, not a database durability or migration contract |

See [projection and linear algebra](Linear_Algebra_and_Projection.md) for shape
conventions, scale limits, and complete examples. For distance/search details,
the implementations are [Operations](../Sources/VectorCore/Operations/Operations.swift),
[BatchOperations](../Sources/VectorCore/Operations/BatchOperations.swift), and
[MatrixDistance](../Sources/VectorCore/Operations/MatrixDistance.swift).

## Provider boundaries

Provider values on `Operations` are task-local bindings, not mutable global
settings. Bind them with the corresponding `$provider.withValue` method around
the work that should observe them.

| Binding | Accepted protocol | Default in this checkout |
|---|---|---|
| `Operations.computeProvider` | `ComputeProvider` | `CPUComputeProvider.automatic` |
| `Operations.simdProvider` | `ArraySIMDProvider` | `SwiftSIMDProvider()` |
| `Operations.linearAlgebraProvider` | `LinearAlgebraProvider` | `LAPACKLinearAlgebraProvider()` on Apple platforms |

`SIMDProvider` is a separate typed, low-level protocol. A
`SwiftFloatSIMDProvider` is not the value expected by
`Operations.$simdProvider`. The default provider names also do not mean that
all computation goes through them: some vector methods and matrix operations
call their own kernels or Accelerate directly.

This complete example scopes the CPU execution provider to one search:

```swift
import VectorCore

let query = Vector512Optimized(repeating: 1)
let candidates = [query, Vector512Optimized(repeating: 2)]
let results = try await Operations.$computeProvider.withValue(CPUComputeProvider.sequential) {
    try await Operations.findNearest(to: query, in: candidates, k: 1)
}
precondition(results.count == 1 && results[0].index == 0)
print("Nearest candidate: \(results[0].index)")
```

[`BatchKernelProvider`](../Sources/VectorCore/Protocols/BatchKernelProvider.swift)
extends `ComputeProvider`. `findNearest` and `findNearestBatch` delegate to an
installed conformer before the built-in CPU search routes. Core supplies the
protocol; an external provider owns its implementation, capability checks,
allocation behavior, and synchronization. The protocol's default batch search
performs per-query calls; conformance alone does not imply one fused GPU batch.

## Search and matrix routing

These are current implementation choices, not stable crossover guarantees:

| Entry point | Built-in matrix route |
|---|---|
| `Operations.findNearestBatch` | Only when the caller passes `cpuStrategy: .gram(...)` (the default `.perQuery` never takes it); Euclidean/cosine; optimized 512, 768, or 1536 types; after external provider delegation |
| `BatchOperations.pairwiseDistances` | `Configuration.enableMatrixRouting` and `matrixRoutingMinN` (defaults: `true`, `256`); Euclidean/cosine; optimized 512, 768, or 1536 types |
| `MatrixDistance` | Explicit matrix operation over `UnifiedVectorBuffer`; no automatic crossover gate |

`BatchOperations.updateConfiguration` affects the pairwise gate above. It does
**not** affect `Operations.findNearestBatch`, which has no size gate.
Single-query fused Top-K paths also differ by dimension and metric; do not
extrapolate 512-dimensional coverage to every optimized type.

CPU search routes order results ascending, with computed NaN distances after
numeric ones and exact ties by the smaller original index. On its generic
small-`k` route, `Operations.findNearest` calls the metric once per candidate
without calibration; its full-score route keeps all `n` scores and, on the
first use of a (dimension, kernel) key, may call the metric on up to 256
candidates to calibrate. See the `findNearest` documentation for every route
and for cancellation.

`MatrixDistance` materializes the query-by-candidate score matrix, and the
pairwise matrix route returns the full N × N matrix, computed one query row at
a time straight into the returned rows with no separate score buffer; the
`.gram` route of `findNearestBatch` scores one query tile against one candidate
tile at a time. An
`into:` output or prepared candidates can reuse particular storage, but does
not make the operation allocation-free. Callers of `MatrixDistance` must
validate every input dimension and provide representable sizes and correctly
sized output buffers; it is not a checked replacement for the throwing search
entry points. See [Numerical Behavior](Numerical_Behavior.md).

## Memory and implementation details

`UnifiedVectorBuffer`, `PageAlignedBuffer`, `SoA`, `SoALayout`, `AlignedMemory`,
and `MemoryPool` expose public contracts; they are not all private utilities.
Borrowed contiguous storage is not automatically page-aligned or suitable for
retention by an asynchronous consumer. Read
[Memory Alignment](Memory_Alignment.md) and the
[frozen SoA layout](SoA_Layout_Contract.md) before using raw pointers.

Kernel dispatch heuristics, mixed-precision caches, and C shims are
implementation details unless a public contract explicitly says otherwise.
An implementation symbol appearing in source or a historical design document
is not sufficient evidence of a supported consumer API.
