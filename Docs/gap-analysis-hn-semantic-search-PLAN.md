# Plan: Scoped Gap Analysis for VectorCore + VectorIndex (HN semantic-search lens)

> This is the **planning document** for producing a gap-analysis report. It is not the report itself.
> The report it specifies will be written to `Docs/gap-analysis-hn-semantic-search.md`.

## Context

The user is building a small app to **semantically search the entire HackerNews story corpus (~5M stories)** and render a **2-D "globe network" visualization that is a projected scatter** (UMAP/t-SNE/PCA of post embeddings, with topical density emerging spatially and time encoded as color/filter/animation).

Embedding generation is owned by a separate package, **EmbedKit** — it is **out of scope**. The boundary is clean: EmbedKit produces `[Float]` vectors; **VectorCore** stores/searches/groups/projects them and **VectorIndex** indexes them.

The **real deliverable is NOT the app** — it is a **prioritized, scoped gap-analysis report for VectorCore and VectorIndex**, using the HN workload purely as the forcing function to derive *which foundational algorithms/primitives are missing*. Agreed parameters:

- **Output:** a written prioritized gap report only. No implementation in this plan. Each recommended primitive becomes its own follow-up plan.
- **Placement rule:** *Core owns math, Index owns graph.* PCA/SVD, k-means, UMAP layout-math → VectorCore. kNN-graph construction + temporal/metadata pre-filtering → VectorIndex. UMAP (in Core) calls Index for its kNN.

This plan specifies the report's full substance (so writing it is mechanical) and where it lands.

## Deliverable & location

A single Markdown report:

- **Path:** `/Users/goftin/dev/gsuite/VSK/VectorCore/Docs/gap-analysis-hn-semantic-search.md`
- Covers **both** packages; the cwd repo (VectorCore) is the natural home. (Adjust path if the user prefers a shared/VectorIndex location.)

## Verified ground truth (basis for the report — already confirmed by code read)

Present & strong (NOT gaps — report says so explicitly to stay honest):
- kNN search single/batch/GEMM; distance metrics (L2, cosine, dot, manhattan); Top-K selection; pairwise/distance-matrix; quantization (INT8); PQ compression (u8/u4, residual); IVF + HNSW + Flat indexes; k-means++ & mini-batch k-means **(internal to IVF)**; centroid computation; mmap binary format **(exists, internal to IVF Kernel #30 only)**; SIMD providers (Accelerate vDSP + Swift fallback); BatchKernelProvider GPU seam.

Confirmed gaps (load-bearing, verified at file level):
- VectorCore uses **only vDSP** from Accelerate — **no LAPACK linkage**, no eigensolver/factorization (`AccelerateSIMDProvider.swift`).
- **No PCA/SVD/eigen/UMAP/t-SNE/projection anywhere** in VectorCore (grep: zero hits).
- VectorCore exposes **no public clustering API** — only `Operations.centroid(of:)` (`Operations.swift:767`); Lloyd/k-means++ are **internal** (`SyncBatchOperations.swift`, IVF kernels #11/#12).
- VectorIndex `save`/`load` for all three indexes = **JSON, vectors as text arrays** (`Persistence.swift`, `FlatIndex.swift:154`, `HNSWIndex.swift:1002`, `IVFIndex.swift:656`). Binary/mmap (`VIndexMmap.swift`) is **not wired** to public persistence.
- `HNSWIndex.load` **rebuilds graph via `batchInsert`** — O(n log n), no topology serialization (`HNSWIndex.swift:1021`).
- Filtering is **always post-search** (predicate on candidates after distance) — no metadata/temporal pre-filter index (`FlatIndex.swift:60`, `HNSWIndex.swift:213`, `IVFIndex.swift:448`).

## Report content — prioritized gaps

The report derives priority from the HN pipeline: *ingest 5M vectors → index → temporal-filtered semantic search → 2-D projection of the corpus → (optional) topic coloring.* Each stage maps to a capability; the missing ones are ranked.

### P0 — Dimensionality reduction (the visualization literally *is* this)
The 2-D scatter is a projection of 5M×384 embeddings. This is the largest genuine hole and spans both libraries.

1. **PCA / randomized-SVD — VectorCore.** Standalone 2-D/3-D projection *and* the standard 384→~50 pre-step before UMAP (denoise + 5–10× speedup). Use **randomized SVD (Halko et al.)**, not full SVD, for 5M scale. *Prerequisite:* add **LAPACK linkage** to VectorCore (Accelerate ships it; only vDSP is used today) — call this out as a foundational enabler.
   - Sketch: `Operations.pca(_ vectors:, components: Int) -> (projected: [[Float]], explainedVariance: [Float])` and/or a fittable `PCAModel { fit / transform }`.
2. **kNN-graph construction — VectorIndex.** A primitive that emits an adjacency + similarity-weight graph (CSR) from existing ANN, symmetrized. Feeds UMAP and is independently useful (graph viz, community detection later).
   - Sketch: `func buildKNNGraph(k: Int, symmetrize: Bool) async throws -> KNNGraph` on the index protocol.
3. **UMAP layout — VectorCore (math), consuming Index's kNN graph.** Decomposes into three composable pieces: fuzzy simplicial set → **spectral initialization (eigensolver on the graph Laplacian** — needs the LAPACK enabler) → **SGD layout with negative sampling**. t-SNE explicitly de-prioritized (O(n²); UMAP supersedes at 5M).
   - Sketch: `Operations.umapProject(graph: KNNGraph, targetDim: Int = 2, config: UMAPConfig) -> [SIMD2<Float>]`.

### P1 — Persistence & cold-start at 5M (JSON is unusable)
1. **Wire the existing binary/mmap format into public `save`/`load`** for all index types (currently JSON-only; mmap is trapped in IVF Kernel #30). Add a `format:` param (`.json` | `.binary`).
2. **HNSW graph-topology serialization** to eliminate the O(n log n) rebuild on load — *or* the report recommends **IVF+PQ as the scale path** for HN (instant mmap load, 256× compression) and treats HNSW persistence as lower priority.

### P2 — Temporal search ergonomics
- **Metadata/temporal pre-filtering** (timestamp-ordered postings or predicate pushdown) instead of post-search filtering. Relevant because the temporal viz slices the corpus by time; post-hoc filtering wastes distance computation on out-of-window candidates.

### P3 — Clustering (secondary for a scatter; only for topic coloring)
- **Promote the existing internal k-means to a public clustering API in VectorCore** (Lloyd + k-means++ already exist internally — expose cleanly, don't rebuild). Add small cluster-quality metrics (inertia; silhouette on a sample).
   - Sketch: `Operations.kMeans(_ vectors:, k: Int, config:) -> (assignments: [Int], centroids: [V], inertia: Float)`.

### Cross-cutting enabler (prerequisite for P0)
- **Add LAPACK linkage to VectorCore.** Currently only vDSP/BLAS-ish element ops are used. PCA/SVD and the UMAP spectral-init eigensolver both depend on it. Foundational; unblocks the entire P0 band.

### Explicitly NOT gaps (report states this to stay scoped)
kNN/batch/GEMM search, distance metrics, Top-K, PQ/INT8 quantization, centroid math, the three index structures, the GPU seam. t-SNE is intentionally excluded in favor of UMAP at this scale.

## Report structure (sections to write)
1. Executive summary + the priority table (P0–P3 + enabler).
2. Derivation: HN pipeline stages → required capability → present/absent (the forcing-function mapping).
3. Per-gap detail: what's missing, proposed public API sketch, target library (Core/Index), algorithm choice + complexity, scale note at 5M, and file-level evidence of absence.
4. "Not gaps" section (verified-present capabilities) to bound scope.
5. Suggested build order with dependencies (LAPACK → PCA → kNN-graph → UMAP; persistence parallel track).

## Verification (how to confirm the report is correct before finalizing)
- Re-confirm each "absent" claim with a fresh grep at write time (`pca|svd|eigen|umap|tsne|lapack` in VectorCore; `buildKNNGraph|KNNGraph` in VectorIndex) and cite file:line for each present/absent assertion — the report's value is its accuracy.
- Confirm EmbedKit's output dimension (assumed 384, matching `Vector384Optimized`) so projection-cost figures are right; note the assumption if unverified.
- Sanity-check the 5M memory math (5M × 384 × 4B ≈ 7.7 GB raw) and the IVF+PQ compressed footprint in the persistence section.

## Out of scope
- The HN app itself (ingest, UI, the actual globe renderer) — only used as the derivation lens.
- EmbedKit / embedding generation.
- Any code implementation — each recommended primitive is a separate follow-up plan.
