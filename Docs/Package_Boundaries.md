# Package boundaries

VectorCore is the CPU numerical foundation of a set of separately released
Swift libraries. This document distinguishes the responsibilities of the
current packages from proposals in older design notes. It is not a promise
that every combination of sibling releases is compatible.

## What belongs in VectorCore

Core owns vector representations, distance and selection primitives, CPU batch
math, and the low-level contracts that consumers share:

- Generic fixed-dimension, runtime-dimension, and specialized SIMD vector types.
- Distance metrics, Top-K selection, normalization, statistics, and centroids.
- CPU matrix distances using Accelerate, including reusable candidate packing.
- Dense factorizations, PCA, and UMAP layout over a Core-owned neighbor graph.
- Aligned buffers, SoA layout, allocation/pool utilities, and provider protocols.
- Quantization and vector serialization primitives.

These are implemented in this repository; the [API overview](API_Overview_Map.md)
links the public entry points. Core has no third-party package dependencies.
Its [manifest](../Package.swift) includes Swift and C targets and links Apple's
Accelerate framework. It does not include Metal shaders or an embedding model
runtime.

Core's brute-force neighbor search does not make it a vector database.
Likewise, `KNNGraph` is an interchange representation and reference graph
builder, not a persistent ANN index. Vector serialization is not a
transaction, crash-recovery, or schema-migration service.

## The surrounding libraries

The links below lead to each project's own documentation. They describe roles,
not a synchronized release train or an exhaustive dependency graph.

| Library | Role relative to Core |
|---|---|
| [VectorIndex](https://github.com/gifton/VectorIndex) | CPU vector indexing, including HNSW and IVF; index construction, traversal, and index-specific storage belong here |
| [VectorAccelerate](https://github.com/gifton/VectorAccelerate) | Metal compute, GPU resources, and accelerated operations/indexing; consumes Core's numerical and buffer interfaces |
| [EmbedKit](https://github.com/gifton/EmbedKit) | Embedding generation and its model/tokenization pipeline; produces vectors for downstream numerical work |
| [SwiftTopics](https://github.com/gifton/SwiftTopics) | Topic modeling, clustering, and keyword extraction over embeddings; application-level topic semantics stay above Core |

The libraries have overlapping implementations in places. For example, the
existence of PCA/UMAP in Core does not establish that every sibling uses those
implementations. A future consolidation would require a downstream migration
and verification; it is not accomplished by documenting an intended boundary.

Earlier documents described standalone `VectorStore` and
`VectorIndexAccelerated` packages as part of a four-package architecture.
Those names are not installation requirements or available products of
VectorCore. This reference does not depend on those proposed packages.

## Shared interfaces

### CPU execution and external kernels

`Operations.computeProvider` holds a task-local `ComputeProvider`. A provider
that also conforms to `BatchKernelProvider` receives delegated
`findNearest` / `findNearestBatch` calls before Core's built-in CPU routes.
See the [provider protocol](../Sources/VectorCore/Protocols/BatchKernelProvider.swift)
and [routing reference](API_Overview_Map.md#search-and-matrix-routing).

This is an integration point, not automatic device discovery. A provider owns
its resource lifetime, supported input types, scheduling, failure behavior,
and any CPU fallback. The default batch-search protocol implementation is a
per-query implementation; a provider must override it to offer different batch
execution. Core's CPU tests do not establish external GPU correctness.

### Contiguous buffers and SoA

`UnifiedVectorBuffer` exposes a scoped contiguous read view. It does not
provide a persistent pointer or imply page alignment. `PageAlignedBuffer`
and opt-in page-aligned `SoA` storage provide explicit allocations and
ownership-transfer operations. The [SoA contract](SoA_Layout_Contract.md)
specifies FP32 lane order, stride, and logical versus allocated byte counts.

Core owns these CPU-side contracts. The GPU consumer owns buffer import,
completion tracking, synchronization, and eventual release. “Zero-copy” at
the import boundary does not mean that producing, packing, or converting the
embedding involved no earlier copy. See [Memory Alignment](Memory_Alignment.md).

### Linear algebra and neighbor graphs

`LinearAlgebraProvider` is the factorization interface used by `PCAModel`.
`KNNGraph` is a validated directed CSR representation accepted by UMAP.
An index can produce the graph without Core depending on that index's
implementation. Graph indices and optional initial coordinates must describe
the same point ordering.

These contracts let a consumer reuse numerical work without moving index
ownership or topic-modeling policy into Core. See
[Linear Algebra and Projection](Linear_Algebra_and_Projection.md).

## Compatibility and verification

Use each sibling's `Package.swift`, release notes, and actual consumer builds
to determine compatible versions. A dependency's minimum version is a resolver
constraint, not evidence that all later pre-1.0 versions have been exercised
together. No cross-package compatibility matrix is claimed here.

Core's [CI](../.github/workflows/ci.yml) tests this package and compiles/runs
selected standalone documentation examples. Those checks do not run every
sibling package or validate end-to-end app behavior. The
[dated verification records](README.md#verification-and-measurements)
state what was exercised on particular revisions.

Keep new Core work tied to a concrete numerical need or shared contract.
Index traversal, GPU scheduling, model inference, durable application storage,
and topic interpretation require their own packages and tests. Proposed work
is listed separately in the [roadmap](ROADMAP.md).
