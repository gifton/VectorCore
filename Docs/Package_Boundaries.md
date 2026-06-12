# VectorCore Package Architecture

VectorCore is designed as part of a modular 4-package ecosystem, each with clear responsibilities and boundaries. This document outlines the architecture and explains which functionality belongs in each package.

## 📦 The Four Packages

```
┌─────────────────────────────────────────────────────────────┐
│                                                               │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │ VectorIndex  │      │VectorAccelerate                    │
│  │              │      │              │                     │
│  │ • HNSW/NSW   │      │ • Metal/GPU  │                     │
│  │ • Clustering │      │ • Auto-tuning│                     │
│  │ • Graphs     │      │ • Mixed FP16 │                     │
│  └──────┬───────┘      └──────┬───────┘                     │
│         │                     │                              │
│         │  ┌──────────────────┴────────────────┐            │
│         │  │                                    │            │
│         └─►│        VectorCore                  │            │
│            │                                    │            │
│            │  • Vector types                    │            │
│            │  • SIMD operations                 │            │
│            │  • Distance metrics                │            │
│            │  • CPU providers                   │            │
│            │  • Memory management               │            │
│            └────────────┬───────────────────────┘            │
│                         │                                    │
│                         │                                    │
│                    ┌────▼──────┐                             │
│                    │VectorStore│                             │
│                    │           │                             │
│                    │ • Binary  │                             │
│                    │ • SQLite  │                             │
│                    │ • CRC/val │                             │
│                    └───────────┘                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. VectorCore (Foundation)

**Purpose**: CPU-first vector mathematics and core abstractions

**Scope** (v0.3.1):
- Vector data types (fixed & dynamic dimensions)
- SIMD-optimized operations (512/768/1536 dimensions)
- Distance metrics (Euclidean, Cosine, Manhattan, Chebyshev, Hamming, Minkowski, DotProduct)
- CPU GEMM batch distance (`MatrixDistance`: `euclideanSquaredMatrix` / `cosineDistanceMatrix`, `prepare(_:normalized:)` → `PreparedCandidates`) via Accelerate `cblas_sgemm` (AMX on Apple Silicon)
- Dense linear algebra seam (`LinearAlgebraProvider`: thin QR / thin SVD / symmetric eigen — LAPACK-backed with a pure-Swift fallback) and projection math: PCA (`PCAModel`, randomized SVD) and UMAP layout (`Operations.umap`), consuming the Core-owned `KNNGraph` CSR interchange
- Normalization, statistics, centroid computation
- Top-K selection (deterministic `TieBreaker`: `.smallerIndex` default / `.insertionOrder` / `.smallerValue`)
- Memory management (aligned allocation, buffer pools)
- Provider abstractions (`SIMDProvider`, `ComputeProvider`, `BufferProvider`, `BatchKernelProvider`)
- Zero-copy GPU-bridge buffer contract (`UnifiedVectorBuffer` / `PageAlignedBuffer`) + frozen SoA layout (`SoALayout`)
- Basic quantization primitives (INT8)
- Serialization (Codable, in-memory binary)

### 0.3.0 — "hold the line, sharpen the seams"

The strategy this release is to keep VectorCore the small, CPU-first, allocation-free
foundation while **redirecting** index / GPU / DB features to the siblings
VectorIndex / VectorAccelerate / EmbedKit. Core does *not* grow into those domains; instead
it **owns the seams** they plug into. What Core newly owns in 0.3.0:

- **`MatrixDistance`** — CPU GEMM batch distance (Accelerate `cblas_sgemm` → AMX on Apple
  Silicon), generic over `UnifiedVectorBuffer`. `euclideanSquaredMatrix` /
  `cosineDistanceMatrix`; `prepare(_:normalized:)` → `PreparedCandidates`.
- **Zero-copy GPU-bridge buffer contract** — the `UnifiedVectorBuffer` protocol + the
  page-aligned `PageAlignedBuffer` class, valid for `MTLDevice.makeBuffer(bytesNoCopy:)`.
- **Frozen SoA memory-layout contract** — the `SoALayout` descriptor, reachable as
  `SoA.layoutDescriptor` (live) or `SoALayout.forType` (ahead of allocation). See
  [SoA Layout Contract](SoA_Layout_Contract.md).
- **`BatchKernelProvider`** — a `ComputeProvider` sub-protocol. `Operations.findNearest` /
  `findNearestBatch` downcast the installed `computeProvider` to it and delegate, giving an
  installed GPU provider precedence over the CPU GEMM path.
- **`TieBreaker`** — deterministic Top-K ordering (`.smallerIndex` default / `.insertionOrder`
  / `.smallerValue`).
- **Pointer-level seams** — `SoA.pageAlignedBytes` / `consumeAllocation()`,
  `PlatformConfiguration.roundUpToPage(_:)`.

> **Owns the seam, not the kernel.** Core ships **no GPU/Metal kernels**. What it now owns is
> the GPU *seam*: the `BatchKernelProvider` dispatch hook plus the page-aligned `bytesNoCopy`
> buffer contract. The kernels themselves live in VectorAccelerate.

### 0.3.1 — the projection stack

The P0 band of the HN semantic-search gap analysis. Same philosophy as 0.3.0:
Core owns the *math* and the *contracts*; corpus-scale graph construction stays
in VectorIndex. What Core newly owns in 0.3.1:

- **`LinearAlgebraProvider`** — thin QR / thin SVD / symmetric eigen over column-major
  `[Float]`, task-local-selectable (`Operations.$linearAlgebraProvider`): Accelerate LAPACK
  on Apple platforms, a pure-Swift fallback (Householder QR / cyclic Jacobi / Hestenes SVD)
  elsewhere.
- **PCA** — `PCAModel` / `Operations.pca`, randomized SVD (Halko); fit on a sample,
  stream the corpus through `transform`.
- **UMAP layout** — `Operations.umap`: fuzzy simplicial set + seeded single-threaded SGD.
  Initialization via PCA (default), random, or caller-provided coordinates.
- **`KNNGraph`** — the **ratified CSR interchange contract** (per the gap analysis §3.2
  boundary note): a dumb, validated sparse kNN container that lives in Core so that
  graph *producers* (VectorIndex's ANN indexes) can hand graphs TO Core's UMAP without
  Core ever depending on an index package. **Data flows Index → Core; code never does.**
  Producers must emit the raw directed graph with Euclidean distances and no self-loops —
  Core's fuzzy-set stage owns symmetrization. `KNNGraph.bruteForce` (exact, O(n²·d)) is
  the in-Core reference builder for samples and tests, not the corpus path.

**What's NOT in Core**:
- ❌ Graph algorithms or graph *construction* at scale (the `KNNGraph` CSR *container* is the
  one deliberate exception — a Core-owned data-interchange contract that VectorIndex populates;
  see the 0.3.1 note above)
- ❌ Clustering algorithms (K-means, hierarchical, etc.)
- ❌ GPU/Metal **kernels** (Core owns the seam — `BatchKernelProvider` dispatch + the page-aligned `bytesNoCopy` buffer contract — but not the shaders)
- ❌ Approximate nearest neighbor (ANN) indexes
- ❌ Persistent storage formats
- ❌ Mixed precision auto-tuning (device-specific)

**Dependencies**: None third-party (Swift + Foundation + Accelerate, plus an internal `VectorCoreC` C target)

**Public API Surface**:
- Vector types: `Vector<D>`, `DynamicVector`, `Vector{512,768,1536}Optimized`
- Protocols: `VectorProtocol`, `VectorFactory`, `SIMDProvider`, `ComputeProvider`, `BatchKernelProvider`, `UnifiedVectorBuffer`, `LinearAlgebraProvider`
- Operations: `Operations`, `BatchOperations`, `MatrixDistance`
- Projection stack: `PCAConfig`/`PCAModel`, `UMAPConfig`/`UMAPInitialization`/`UMAPResult`, `KNNGraph`
- Distance metrics: `EuclideanDistance`, `CosineDistance`, etc.
- GPU-bridge: `PageAlignedBuffer`, `SoALayout` (see [SoA Layout Contract](SoA_Layout_Contract.md))

**CPU-First Philosophy (no GPU kernels, owns the GPU seam)**:
VectorCore ships **no GPU/Metal kernels** — it stays CPU-first for simplicity, portability, and
zero GPU *dependencies*. What it does own is the **GPU seam**: the `BatchKernelProvider`
dispatch hook (`Operations.findNearest` / `findNearestBatch` downcast the installed
`computeProvider` and delegate, GPU taking precedence over the CPU GEMM path) and the
zero-copy page-aligned `bytesNoCopy` buffer contract (`UnifiedVectorBuffer` / `PageAlignedBuffer`,
plus the frozen `SoALayout`). The actual kernels are provided by VectorAccelerate through these
seams.

---

## 2. VectorIndex (Graph & Clustering)

**Purpose**: Approximate nearest neighbor (ANN) search and clustering algorithms

**Scope**:
- **Graph-based ANN**:
  - HNSW (Hierarchical Navigable Small World)
  - NSW (Navigable Small World)
  - Graph primitives (adjacency lists, edge management)
  - Graph algorithms (traversal, connectivity, centrality)
  - kNN-graph construction at corpus scale (batch ANN over an index → Core's `KNNGraph` CSR interchange, feeding Core's UMAP)
- **Clustering**:
  - K-means (MiniBatch, Streaming, Distributed)
  - Hierarchical clustering
  - DBSCAN, OPTICS
- **Graph utilities**:
  - Sparse linear algebra
  - Graph properties and analysis
  - Neighbor selection strategies

**Dependencies**:
- VectorCore (for vector types and distance metrics)

**Why separate from Core?**:
- **Domain separation**: Graphs are a distinct domain from vector math
- **Binary size**: Graph algorithms add ~100KB+ of compiled code
- **Use case**: Many users only need vector math, not indexing
- **Complexity**: Index structures have different performance characteristics

**Usage Example**:
```swift
import VectorCore
import VectorIndex

// Build HNSW index
let index = HNSWIndex<Vector512Optimized>(
    metric: EuclideanDistance(),
    m: 16,
    efConstruction: 200
)

// Insert vectors
for vector in database {
    try index.insert(vector)
}

// Query
let neighbors = try index.search(query, k: 10)
```

---

## 3. VectorAccelerate (GPU & Optimization)

**Purpose**: Hardware-accelerated computation and device-specific optimizations

**Scope**:
- **Metal/GPU providers**:
  - GPU-backed `ComputeProvider` implementations
  - Metal shaders for distance computation
  - Fused kernels for batch operations
- **Mixed precision auto-tuning**:
  - FP16 candidate conversion
  - Strategy selection based on device capabilities
  - Performance profiling and adaptive selection
- **Device-adaptive SoA FP16 strategy**:
  - Device-specific auto-tuning over FP16 layouts
  - (Note: the `SoAFP16` cache type itself still ships in Core as an internal type; only the
    device-adaptive *tuning* over it is Accelerate's concern.)
- **Neural Engine support** (future):
  - CoreML-based vector operations

**Dependencies**:
- VectorCore (for protocols and types)
- Metal framework

**Why separate from Core?**:
- **Optional GPU**: Many deployments don't need GPU (server-side, low-power devices)
- **Platform constraints**: Metal is Apple-only; users may want CUDA/Vulkan alternatives
- **Device-specific**: Auto-tuning requires runtime device detection
- **Binary size**: Metal shaders and GPU code add significant binary bloat

**Usage Example**:
```swift
import VectorCore
import VectorAccelerate

// Override compute provider to use GPU
await Operations.$computeProvider.withValue(MetalComputeProvider()) {
    let results = try await Operations.findNearest(
        to: query,
        in: database,
        k: 100
    )
    // GPU-accelerated search
}

// Mixed precision auto-tuning
let workload = WorkloadCharacteristics(
    dimension: 512,
    batchSize: 10000,
    precisionMode: .fp16Acceptable
)

let strategy = await KernelAutoTuner.shared.selectStrategy(for: workload)
// Automatically selects optimal FP16/FP32 strategy for device
```

---

## 4. VectorStore (Persistence)

**Purpose**: Durable storage and serialization of vectors

**Scope**:
- **Binary formats**:
  - Efficient packed binary encoding
  - CRC32 validation
  - Version metadata
- **Storage backends**:
  - Memory-mapped files
  - SQLite adapters
  - Custom file formats
- **Metadata management**:
  - Vector IDs and tags
  - Compression metadata
  - Index metadata (for integration with VectorIndex)

**Dependencies**:
- VectorCore (for vector types)
- SQLite (optional)

**Why separate from Core?**:
- **I/O complexity**: File I/O and database integration are orthogonal to math
- **Deployment flexibility**: Some users want in-memory only
- **Format evolution**: Storage formats need independent versioning
- **Zero third-party dependencies in Core**: Core stays Swift-first (one internal C target, no third-party deps) with no I/O

**Usage Example**:
```swift
import VectorCore
import VectorStore

// Save vectors to disk
let store = VectorBinaryStore<Vector512Optimized>(path: "embeddings.vecstore")
try await store.write(vectors, withCRC: true)

// Load vectors
let loaded = try await store.read()

// SQLite backend
let db = VectorSQLiteStore<Vector768Optimized>(path: "vectors.db")
try await db.insert(vectors, withTags: ["bert", "embeddings"])
let results = try await db.query(tag: "bert")
```

---

## Package Interaction Patterns

### Pattern 1: Core-only (Simple math)

```swift
import VectorCore

let v1 = Vector512Optimized(repeating: 1.0)
let v2 = Vector512Optimized(repeating: 2.0)
let distance = v1.euclideanDistance(to: v2)
```

**Use case**: Simple vector math, embedding computation, ML inference

---

### Pattern 2: Core + Index (ANN search)

```swift
import VectorCore
import VectorIndex

let index = HNSWIndex<Vector768Optimized>(metric: CosineDistance())
try index.bulkInsert(database)
let neighbors = try index.search(query, k: 10)
```

**Use case**: Semantic search, recommendation systems, similarity search

---

### Pattern 3: Core + Accelerate (GPU acceleration)

```swift
import VectorCore
import VectorAccelerate

await Operations.$computeProvider.withValue(MetalComputeProvider()) {
    let results = try await Operations.findNearest(to: query, in: database, k: 100)
}
```

**Use case**: Large-scale batch processing, real-time search, GPU-enabled servers

---

### Pattern 4: Full Stack (Index + GPU + Persistence)

```swift
import VectorCore
import VectorIndex
import VectorAccelerate
import VectorStore

// Load vectors from disk
let store = VectorBinaryStore<Vector1536Optimized>(path: "embeddings.vecstore")
let vectors = try await store.read()

// Build GPU-accelerated index
await Operations.$computeProvider.withValue(MetalComputeProvider()) {
    let index = HNSWIndex<Vector1536Optimized>(metric: EuclideanDistance())
    try index.bulkInsert(vectors)  // Uses GPU for distance computations

    // Query
    let results = try index.search(query, k: 100)
}
```

**Use case**: Production vector search systems, large-scale ML applications

---

## Design Principles

### 1. **Separation of Concerns**
Each package has a single, well-defined responsibility. This prevents scope creep and keeps each package focused.

### 2. **Minimal Dependencies**
VectorCore has zero third-party dependencies. Other packages only depend on Core and standard libraries.

### 3. **Opt-in Complexity**
Users only pay (in binary size and complexity) for what they use. Need just vector math? Use Core. Need GPU? Add Accelerate.

### 4. **Protocol-Based Integration**
Packages integrate via protocols (`ComputeProvider`, `SIMDProvider`, `BufferProvider`, `BatchKernelProvider`), enabling loose coupling and extensibility.

### 5. **Platform Portability**
Core is Swift-first (with one internal `VectorCoreC` C target — so not literally 100% Swift, though it carries zero *third-party* dependencies) and works everywhere. Platform-specific features (Metal kernels, Neural Engine) live in Accelerate; Core owns only the seam they plug into.

---

## Migration from Monolith

VectorCore v0.1.0 started as a monolith with graph/clustering code. This was split for v0.2.0:

**Moved to VectorIndex**:
- All `Graph*.swift` kernels
- All clustering kernels (`*Clustering*.swift`, `*KMeans*.swift`)
- `SparseLinearAlgebraKernels.swift`

**Moved to VectorAccelerate**:
- Mixed precision auto-tuning (`MixedPrecisionAutoTuner.swift`)
- Device-adaptive SoA FP16 *tuning* (the `SoAFP16Cache` type itself remains an internal Core type)

**Moved to EmbedKit** (future):
- Durable storage / embedding-store concerns (CRC-validated binary format, persistence)

### 0.3.0 changes

- **Breaking:** removed the no-op `blockSize:` parameter from `SoA.init(vectors:…)` and
  `SoAFP16.init(vectors:…)`. It was a no-op and is gone (see the [SoA Layout Contract](SoA_Layout_Contract.md)).
- **Deferred (specced, not built):** block-wise quantization `Q8_0` is consumer-gated — it is
  *not* in 0.3.0.

---

## Version Compatibility

| VectorCore | VectorIndex | VectorAccelerate | EmbedKit |
|------------|-------------|------------------|----------|
| 0.1.0      | -           | -                | -        |
| 0.2.0      | 0.1.0       | 0.1.0            | 0.1.0    |
| 0.3.0+     | 0.2.0+      | 0.2.0+           | 0.2.0+   |

**Policy**: All packages use SemVer 2.0. Minor version bumps in Core may require minor version bumps in dependent packages.

---

## FAQ

**Q: Why not one big package?**
A: Binary size, deployment flexibility, and clear boundaries. Users who only need vector math shouldn't ship graph algorithms and GPU code.

**Q: Can I use VectorIndex without VectorAccelerate?**
A: Yes! VectorIndex depends only on VectorCore. GPU acceleration is optional.

**Q: What if I want CUDA instead of Metal?**
A: You can implement a `CUDAComputeProvider` that conforms to `ComputeProvider`. VectorCore's protocol-based design supports custom backends.

**Q: Will there be more packages?**
A: The store/embed sibling is **EmbedKit** (the real embed package — not a speculative "VectorML"/"VectorTransform"). The architecture is designed to be extensible; new packages depend on Core and integrate via protocols.

**Q: Why is quantization in Core but mixed precision auto-tuning in Accelerate?**
A: Basic INT8 quantization is CPU-friendly and device-agnostic. Auto-tuning requires device-specific profiling and strategy selection, which is inherently accelerator-specific. Note: block-wise quantization (`Q8_0`) is **specced but deferred** (consumer-gated) — it is not in 0.3.0.

---

**Document Version**: 1.2
**Last Updated**: 2026-06-11
**Applies to**: VectorCore v0.3.1+
