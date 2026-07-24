# Gap Analysis: VectorCore + VectorIndex for HN Semantic Search

**Date:** 2026-06-09
**Lens:** Semantic search + 2-D projected visualization of the full HackerNews story corpus (~5M stories, 384-dim embeddings from EmbedKit).
**Scope:** Capability gaps in VectorCore and VectorIndex only. The HN app, EmbedKit, and all implementation work are out of scope. Each recommended primitive becomes its own follow-up plan.

> **Verification scope note.** All VectorCore claims below were verified at file level in this repo on the date above. VectorIndex claims (persistence format, HNSW load behavior, filtering strategy) originate from the planning document's code read of the separate VectorIndex checkout and are marked **[VI-unverified-here]**; they must be re-confirmed in that repo before being treated as ground truth.

---

## 1. Executive summary

The HN workload decomposes into: ingest 5M vectors → index → temporal-filtered semantic search → 2-D projection of the corpus → (optional) topic coloring. VectorCore and VectorIndex cover the search stage well. The projection stage — the thing the visualization *is* — has zero coverage, and persistence at 5M scale is reportedly unusable as shipped.

| Priority | Gap | Owner | Status |
|---|---|---|---|
| **Enabler** | LAPACK linkage in VectorCore | Core | Absent — only vDSP used today |
| **P0.1** | PCA / randomized SVD | Core | Absent |
| **P0.2** | kNN-graph construction (CSR output) | Index | Absent |
| **P0.3** | UMAP layout (fuzzy set → init → SGD) | Core (math) + Index (graph) | Absent |
| **P1** | Binary/mmap persistence wired to public save/load; HNSW topology serialization | Index | [VI-unverified-here] JSON-only today |
| **P2** | Temporal/metadata pre-filtering | Index | [VI-unverified-here] Post-search only today |
| **P3** | Public k-means + cluster-quality metrics | **Index** (see §3.6) | Absent in Core; Core has internal building blocks only |

**Placement rule (corrected).** The original plan's rule was "Core owns math, Index owns graph," which placed k-means in Core. This **conflicts with `Package_Boundaries.md`**, which explicitly lists clustering as NOT-in-Core (§1, "What's NOT in Core: ❌ Clustering algorithms (K-means, hierarchical, etc.)") and assigns "K-means (MiniBatch, Streaming, Distributed)" to VectorIndex (§2, "VectorIndex (Graph & Clustering)"). This report follows the boundaries doc: **clustering lands in VectorIndex.** Dense linear algebra (PCA/SVD, eigensolvers) and the UMAP optimization math remain in Core; graph construction and anything touching adjacency structure remain in Index.

**Dependency direction constraint.** `Package.swift` declares zero dependencies for VectorCore; VectorIndex depends on Core, never the reverse. Therefore Core cannot "call Index for its kNN." Instead, Core's UMAP entry point **accepts a prebuilt sparse graph as input** (plain CSR arrays or a Core-owned interchange struct, in the same spirit as the frozen `SoALayout` contract), and Index produces it. Data flows Index → Core; code never does.

---

## 2. Derivation: pipeline stage → capability → present/absent

| HN pipeline stage | Required capability | Present? | Where |
|---|---|---|---|
| Ingest 5M × 384 vectors | Optimized 384-dim vector type | ✅ | `Sources/VectorCore/Vectors/Vector384Optimized.swift` |
| Index for ANN | HNSW / IVF / Flat | ✅ [VI] | VectorIndex repo (this repo's `VectorIndex/` dir holds only `kernel-specs/*.md` — specs, no code) |
| Semantic search | kNN single/batch/GEMM, distance metrics, Top-K | ✅ | `Operations.swift:54,140`, `Operations/Kernels/` (Euclidean, Cosine, Dot, Manhattan, TopKSelection, Batch, Batch_SoA) |
| Memory at 5M | INT8 quantization; mixed precision | ✅ | `Quantization/QuantizationSchemes.swift`, `Kernels/QuantizedKernels.swift`, `Kernels/MixedPrecisionKernels.swift` |
| Cold start at 5M | Binary/mmap index persistence | ❌ [VI-unverified-here] | JSON-only public save/load reported |
| Time-sliced search | Metadata/temporal pre-filter | ❌ [VI-unverified-here] | Post-search predicate filtering reported |
| **2-D projection** | **PCA/SVD, kNN graph, UMAP** | ❌ | **Nowhere — verified zero hits in Core** |
| Topic coloring | Public k-means | ❌ | No k-means implementation in Core, internal or public |

Memory sanity check: 5M × 384 dims × 4 B = **7.68 GB** raw Float32. With PQ at 48 bytes/vector (m=48, u8) → ~240 MB (32×); more aggressive u4/lower-m configs reach 64–128×. Raw mmap of Float32 is viable on a 16 GB+ machine but PQ is the comfortable path. (The 384-dim assumption matches `Vector384Optimized`; confirm against EmbedKit's actual output dim — if it emits 512 or 768, those types also exist and the cost figures scale linearly.)

---

## 3. Per-gap detail

### 3.0 Enabler — LAPACK linkage in VectorCore

**Evidence of absence.** `Sources/VectorCore/Platform/AccelerateSIMDProvider.swift` imports Accelerate (line 11) but every call is `vDSP_*` (lines 26–316). A case-insensitive grep for `lapack|sgesvd|ssyev|eigen|svd|pca` across `Sources/` returns zero genuine hits. No factorization or eigensolver capability exists anywhere in the package.

**What to add.** Accelerate already ships LAPACK; no new dependency is needed (consistent with the zero-third-party-deps policy in `Package_Boundaries.md` §"Minimal Dependencies"). Adopt the modern interface: define `ACCELERATE_NEW_LAPACK=1` (and `ACCELERATE_LAPACK_ILP64=1` for 64-bit indices) via `cSettings`/`swiftSettings`, and route through a new provider seam (e.g., `LinearAlgebraProvider`) parallel to `SIMDProvider`, so the Swift-fallback story is preserved for non-Apple platforms. Required routines for P0: `sgeqrf`/`sorgqr` (QR), `sgesdd` (SVD of small matrices), `ssyevd` (symmetric eigen).

**Why foundational.** PCA/randomized SVD and any spectral computation depend on it. Unblocks the entire P0 band.

### 3.1 P0 — PCA / randomized SVD (VectorCore)

**What's missing.** No projection capability of any kind. The 2-D scatter is literally a projection of 5M×384; PCA is also the standard 384→~50 pre-reduction before UMAP (denoising + 5–10× UMAP speedup).

**Algorithm.** Randomized SVD (Halko–Martinsson–Tropp), not full SVD: Gaussian sketch Ω (384×(k+p)), Y = AΩ via existing GEMM paths, 1–2 power iterations for spectral decay, QR(Y), SVD of the small (k+p)×384 projected matrix via LAPACK. Cost is a few GEMM passes over the data — O(n·d·k) — entirely streamable, no n×n anything. At 5M×384→50 this is seconds-to-a-minute on Apple silicon.

**API sketch.**
```swift
// One-shot
Operations.pca(_ vectors: [V], components: Int)
    -> (projected: [[Float]], explainedVariance: [Float])

// Fittable (fit on a sample, transform the corpus streamingly)
struct PCAModel { 
    static func fit(_ vectors: [V], components: Int, config: PCAConfig) -> PCAModel
    func transform(_ vectors: [V]) -> [[Float]]
}
```
The fittable form matters at 5M: fit on a 100–500k sample, then `transform` the corpus in batches.

### 3.2 P0 — kNN-graph construction (VectorIndex)

**What's missing.** A primitive that runs batch ANN over the indexed corpus and emits a symmetrized, similarity-weighted adjacency in CSR form. Feeds UMAP; independently useful for graph viz and future community detection. (Graph construction is squarely Index territory per `Package_Boundaries.md` §2 "Graph primitives.")

**API sketch.**
```swift
struct KNNGraph {        // CSR; or expose the three arrays raw
    let rowOffsets: [Int]      // n+1
    let neighbors: [Int32]     // nnz
    let weights: [Float]       // nnz
}
func buildKNNGraph(k: Int = 15, symmetrize: Bool = true) async throws -> KNNGraph
```
**Boundary note.** If Core's UMAP consumes this type, the type itself (a dumb CSR container) must live in **Core** as a data-interchange contract — Index populates it. Alternatively Core's API takes the three raw arrays and no shared type is needed. Either is fine; pick one in the follow-up plan and record it in `Package_Boundaries.md`.

### 3.3 P0 — UMAP layout (VectorCore math, Index-fed graph)

**What's missing.** Everything. Decomposes into three composable pieces:

1. **Fuzzy simplicial set** — per-point adaptive bandwidth (binary search for σᵢ), edge weight symmetrization (probabilistic t-conorm). Pure math on the CSR graph → Core.
2. **Initialization** — the plan specified spectral init (eigensolver on the normalized graph Laplacian). **Scale correction:** at 5M nodes a *dense* LAPACK eigensolver is not viable (the Laplacian is 5M×5M sparse). Spectral init requires sparse Lanczos/LOBPCG (Accelerate's Sparse Solvers can support matvec). Recommend **PCA initialization as the default** (reuses 3.1, near-equivalent quality at scale, standard practice for large n) with sparse spectral init as a stretch goal — not a P0 dependency.
3. **SGD layout with negative sampling** — attractive forces along graph edges, repulsive via negative samples, ~200–500 epochs. Embarrassingly parallel per-edge; maps onto existing `ComputeProvider` parallelism and is the natural future `BatchKernelProvider` GPU candidate.

**API sketch.**
```swift
Operations.umapProject(
    graph: KNNGraph,            // produced by VectorIndex
    initial: [[Float]]? = nil,  // e.g. PCA output; random if nil
    targetDim: Int = 2,
    config: UMAPConfig = .default
) -> [SIMD2<Float>]
```

**t-SNE explicitly excluded:** O(n²)/Barnes-Hut variants are dominated by UMAP at 5M; do not build.

### 3.4 P1 — Persistence & cold start (VectorIndex) [VI-unverified-here]

Per the planning document's read of the VectorIndex repo: public `save`/`load` for all three indexes serializes JSON with vectors as text arrays, and `HNSWIndex.load` rebuilds the graph via `batchInsert` (O(n log n), no topology serialization); a binary/mmap path exists but is not wired to public persistence. At 5M vectors JSON is unusable (the text encoding alone would be tens of GB and minutes-to-hours to parse).

**Recommendations** (to re-verify and own in the VectorIndex repo):

1. Wire the existing binary/mmap format into public `save(format: .json | .binary)` / `load`.
2. Either serialize HNSW topology, **or** designate IVF+PQ as the 5M scale path (mmap = effectively instant load; ~32–128× compression) and deprioritize HNSW persistence. This report leans IVF+PQ for the HN corpus: the workload is read-heavy, rebuild-rare, and recall demands are modest for visualization.

Note this is consistent with `Package_Boundaries.md`, which lists persistent storage formats as NOT-in-Core.

### 3.5 P2 — Temporal/metadata pre-filtering (VectorIndex) [VI-unverified-here]

Reported current behavior: predicate filtering applied *after* distance computation in all three indexes. The HN viz slices by time constantly (color/filter/animation), so post-hoc filtering wastes distance work on out-of-window candidates — at small time windows the waste dominates.

**Recommendation:** timestamp-ordered postings within IVF lists (range-prunable) or predicate pushdown into candidate generation. Design belongs to the VectorIndex follow-up plan.

### 3.6 P3 — Public clustering API (VectorIndex — **corrected from plan**)

**Plan correction.** The planning document asserted Lloyd/k-means++ exist internally in VectorCore (`SyncBatchOperations.swift`, "IVF kernels #11/#12") and recommended *promoting* them to public API. **Verified false in this repo:** there is no k-means implementation in VectorCore. The only public cluster-adjacent API is `Operations.centroid(of:)` (`Operations.swift:767`). `SyncBatchOperations.swift` contains internal building blocks — `assignToCentroids` (line 383), `updateCentroids` (line 411), `centroid`/`weightedCentroid` (lines 200, 228) — i.e., the two halves of a Lloyd iteration, but no iteration loop, no k-means++ seeding, no convergence logic. The only "k-means" hits are comments (`Vector{384,512,768,1536}Optimized.swift:381` "Arithmetic Extensions for Streaming K-means"; `VectorMath.swift:375` "Clustering Support"). No IVF kernels numbered #11/#12/#30 exist in this package.

**Corrected recommendation.** This is a **build**, not an expose — and per `Package_Boundaries.md` it builds in **VectorIndex**, which the doc already scopes for "K-means (MiniBatch, Streaming, Distributed)". VectorIndex implements the Lloyd/k-means++/mini-batch loop, calling Core's internal primitives via whatever Core chooses to expose (the `assignToCentroids`/`updateCentroids` pair is a clean candidate for promotion to public *primitive* status — primitives are math, loops are clustering).

**API sketch (VectorIndex).**
```swift
func kMeans<V>(_ vectors: [V], k: Int, config: KMeansConfig)
    -> (assignments: [Int], centroids: [V], inertia: Float)
```
Plus inertia and sampled silhouette for cluster-quality reporting. Priority stays P3: the scatter works without topic coloring.

---

## 4. Not gaps (verified present — keep out of scope)

Verified in this repo: kNN single/batch search with GEMM path (`Operations.swift:54,140`, `BatchOperations.swift`, `MatrixDistance.swift`); distance metrics incl. dedicated Euclidean/Cosine/Dot/Manhattan kernels (`Operations/Kernels/`); Top-K selection (`TopKSelectionKernels.swift`); INT8 quantization and mixed-precision kernels with auto-tuning (`QuantizedKernels.swift`, `MixedPrecisionKernels.swift`, `KernelAutoTuner.swift`); SIMD provider duality (Accelerate vDSP + Swift fallback, `Platform/`); the GPU seam (`Protocols/BatchKernelProvider.swift`, `SoALayout`); centroid math (`Operations.swift:767`); optimized 384/512/768/1536 vector types.

Reported present in VectorIndex [VI-unverified-here]: Flat/HNSW/IVF index structures; PQ compression (u8/u4, residual); internal mmap binary format.

Intentionally excluded: t-SNE (superseded by UMAP at this scale), the HN app itself, EmbedKit.

---

## 5. Build order

```
LAPACK linkage (Core) ──► PCA / randomized SVD (Core) ──► UMAP layout (Core)
                                                              ▲
kNN-graph construction (Index) ──────────────────────────────┘
        (CSR contract type: decide Core-owned struct vs raw arrays)

Parallel track (Index): binary persistence ► (optional) HNSW topology ► temporal pre-filter
Last (Index): public k-means + quality metrics
```

Dependencies: LAPACK gates PCA; PCA and the kNN graph independently gate UMAP (PCA supplies init, graph supplies structure). The persistence track has no coupling to P0 and can proceed concurrently in the VectorIndex repo. P3 depends only on Core promoting `assignToCentroids`/`updateCentroids` to public primitives.

---

## 6. Open items for the VectorIndex agent

1. Re-verify all [VI-unverified-here] claims with file:line citations in that repo (JSON persistence paths, `HNSWIndex.load` rebuild, post-search filtering sites, PQ kernel inventory, mmap format location).
2. Ratify the CSR interchange decision (Core-owned `KNNGraph` struct vs raw arrays) and record it in `Package_Boundaries.md`.
3. Confirm EmbedKit output dimension (this report assumes 384).
