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

**Purpose**: Pure CPU vector mathematics and core abstractions

**Scope** (v0.1.0):
- Vector data types (fixed & dynamic dimensions)
- SIMD-optimized operations (512/768/1536 dimensions)
- Distance metrics (Euclidean, Cosine, Manhattan, Chebyshev, Hamming, Minkowski, DotProduct)
- Normalization, statistics, centroid computation
- Top-K selection
- Memory management (aligned allocation, buffer pools)
- Provider abstractions (SIMDProvider, ComputeProvider, BufferProvider)
- Basic quantization primitives (INT8)
- Serialization (Codable, in-memory binary)

**What's NOT in Core**:
- ❌ Graph algorithms or data structures
- ❌ Clustering algorithms (K-means, hierarchical, etc.)
- ❌ GPU/Metal code
- ❌ Approximate nearest neighbor (ANN) indexes
- ❌ Persistent storage formats
- ❌ Mixed precision auto-tuning (device-specific)

**Dependencies**: None (pure Swift + Foundation + Accelerate)

**Public API Surface**:
- Vector types: `Vector<D>`, `DynamicVector`, `Vector{512,768,1536}Optimized`
- Protocols: `VectorProtocol`, `VectorFactory`, `SIMDProvider`, `ComputeProvider`
- Operations: `Operations`, `BatchOperations`
- Distance metrics: `EuclideanDistance`, `CosineDistance`, etc.

**CPU-Only Philosophy**:
VectorCore is intentionally CPU-only to maintain simplicity, portability, and zero GPU dependencies. GPU acceleration is provided by VectorAccelerate through provider plugins.

---

## 2. VectorIndex (Graph & Clustering)

**Purpose**: Approximate nearest neighbor (ANN) search and clustering algorithms

**Scope**:
- **Graph-based ANN**:
  - HNSW (Hierarchical Navigable Small World)
  - NSW (Navigable Small World)
  - Graph primitives (adjacency lists, edge management)
  - Graph algorithms (traversal, connectivity, centrality)
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
- **SoA FP16 caching**:
  - Structure-of-Arrays FP16 layouts
  - Cache-efficient batch processing
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
- **Zero dependencies in Core**: Core remains pure Swift with no I/O

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
Packages integrate via protocols (`ComputeProvider`, `SIMDProvider`, `BufferProvider`), enabling loose coupling and extensibility.

### 5. **Platform Portability**
Core is pure Swift and works everywhere. Platform-specific features (Metal, Neural Engine) live in Accelerate.

---

## Migration from Monolith

VectorCore v0.1.0 started as a monolith with graph/clustering code. This was split for v0.2.0:

**Moved to VectorIndex**:
- All `Graph*.swift` kernels
- All clustering kernels (`*Clustering*.swift`, `*KMeans*.swift`)
- `SparseLinearAlgebraKernels.swift`

**Moved to VectorAccelerate**:
- Mixed precision auto-tuning (`MixedPrecisionAutoTuner.swift`)
- SoA FP16 caching (`SoAFP16Cache.swift`)

**Moved to VectorStore** (future):
- Durable binary format with CRC validation

---

## Version Compatibility

| VectorCore | VectorIndex | VectorAccelerate | VectorStore |
|------------|-------------|------------------|-------------|
| 0.1.0      | -           | -                | -           |
| 0.2.0      | 0.1.0       | 0.1.0            | 0.1.0       |
| 0.3.0+     | 0.2.0+      | 0.2.0+           | 0.2.0+      |

**Policy**: All packages use SemVer 2.0. Minor version bumps in Core may require minor version bumps in dependent packages.

---

## FAQ

**Q: Why not one big package?**
A: Binary size, deployment flexibility, and clear boundaries. Users who only need vector math shouldn't ship graph algorithms and GPU code.

**Q: Can I use VectorIndex without VectorAccelerate?**
A: Yes! VectorIndex depends only on VectorCore. GPU acceleration is optional.

**Q: What if I want CUDA instead of Metal?**
A: You can implement a `CUDAComputeProvider` that conforms to `ComputeProvider`. VectorCore's protocol-based design supports custom backends.

**Q: Will there be more packages (e.g., VectorML, VectorTransform)?**
A: Possibly. The 4-package architecture is designed to be extensible. New packages can depend on Core and integrate via protocols.

**Q: Why is quantization in Core but mixed precision auto-tuning in Accelerate?**
A: Basic INT8 quantization is CPU-friendly and device-agnostic. Auto-tuning requires device-specific profiling and strategy selection, which is inherently accelerator-specific.

---

**Document Version**: 1.0
**Last Updated**: October 2025
**Applies to**: VectorCore v0.1.0+
