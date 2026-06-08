# DOCUMENT-1 — Redirections & Deferrals

This document records, for every proposal item that VectorCore will **not** implement, *why*
and *where it lives instead*. The goal is that a future contributor (or the external agent) can
see exactly which package owns each capability and what — if anything — Core exposes to support
it. Each redirect cites a confirmed file in a sibling repo.

Two harms motivate redirection over duplication:

- **Format divergence.** A capability like `mmap` storage or ADC look-up tables has a binary
  layout. Two implementations drift, and downstream files written by one cannot be read by the
  other.
- **Double maintenance.** Every numerical fix (the kind the `0.2.2` BE3 audit just did) would
  have to be made twice, in two repos, forever.

---

## REDIRECT

### R1 — `mmap` / out-of-core storage (proposal Task 3.1) → `VectorIndex`

**Already owned.** `VectorIndex/Sources/VectorIndex/Kernels/VIndexMmap.swift` implements a real
on-disk format: 256-byte header + TOC with CRC32, typed section accessors (centroids, codebooks,
IDs, codes, vectors, norms), a write-ahead log with begin/commit/replay, and `msync` with
acquire/release ordering for lock-free readers. Magic `VINDEX\0\0`.

**Why not Core.** The format is index-shaped: it serializes IVF lists, PQ codes, and graph
metadata — concepts Core deliberately does not know about (`Docs/Package_Boundaries.md` assigns
durable formats out of Core). A second `MMapVectorStorage` in Core would be a *different* file
format competing with the one downstream already reads.

**What Core provides instead.** The zero-copy/page-alignment *contract* (DOCUMENT-4, S5): Core
guarantees page-aligned, contiguous raw-pointer views so that whoever owns the mmap (VectorIndex
today, a future `VectorStore`) can map a file region into a Core vector view without copying.

---

### R2 — Pinned Metal arena / ring buffer (proposal Task 3.2) → `VectorAccelerate`

**Already owned.** `VectorAccelerate/Sources/VectorAccelerate/Core/BufferPool.swift` is a
token-based ring-buffer pool with power-of-2 bucketing and `keepAlive(until:)` lifecycle
anchoring; `Core/MetalBufferFactory.swift` selects `MTLStorageModeShared` on unified-memory
devices for zero-copy CPU↔GPU.

**Why not Core.** The proposal explicitly frames this arena as existing "so VectorAccelerate can
stream query vectors to the GPU" — i.e. it is *defined by* its GPU consumer. Putting a
Metal-aware arena in Core would force a Metal dependency and break Core's "zero GPU
dependencies" guarantee for one consumer's benefit.

**What Core provides instead.** A `BufferManagementProvider` protocol (DOCUMENT-4, S4) so Core
code can request/return buffers through an abstraction that `VectorAccelerate` backs with its
pinned Metal pool — the seam, not the implementation.

---

### R3 — ADC look-up tables (proposal Task 4.1) → `VectorIndex`

**Already owned.** `VectorIndex/Sources/VectorIndex/Operations/Quantization/PQLUT.swift` builds
the query LUT and `ADCScan.swift` performs asymmetric distance computation; `PQEncode.swift`
(u8 ks=256, u4 ks=16 packed) and `PQTrain.swift` complete the PQ pipeline.

**Why not Core.** ADC only has meaning relative to a trained PQ codebook and an IVF list layout
— both of which are index concerns. `Docs/Package_Boundaries.md` already lists Product
Quantization as deferred-to-VectorIndex. The proposal's "ADC generator in Core" would need to
reach into index structures Core does not (and should not) model.

**Note on the errata.** The proposal's "zero floating-point drift" requirement for LUT
summation is unattainable; VectorIndex's implementation already accumulates with the
`‖x−c‖² = ‖x‖² + ‖c‖² − 2⟨x,c⟩` dot-trick. If accumulation precision becomes a concern, the fix
(pairwise/Kahan summation) belongs in `ADCScan.swift`, not a new Core type.

---

### R4 — Generation-count visited list (proposal Task 4.2) → `VectorIndex`

**Already owned.** `VectorIndex/Sources/VectorIndex/Kernels/HNSWTraversal.swift` tracks visited
nodes with a `UInt64`-word bitset (`visitedTestAndSet`), reset per search. This is a graph-
traversal primitive with no meaning outside an adjacency structure.

**Why not Core.** Core has no graph, no node IDs, and no traversal — a visited-list here would be
a type with no caller. (The generation-counter variant the proposal describes is a valid
*alternative* to per-search bitset reset; if VectorIndex wants it, it is a local optimization in
`HNSWTraversal.swift`.)

---

### R5 — Concurrent / bounded priority queue (proposal Task 4.3) → `VectorIndex`

**Already owned.** `VectorIndex/Sources/VectorIndex/Operations/Selection/TopK.swift` provides
`TopKHeap` (fixed-capacity, 64-byte-aligned SoA, used under actor isolation) plus `TopKMerge`
for batch results.

**Why not Core.** Core's `TopKBuffer`
(`Sources/VectorCore/Operations/Kernels/TopKSelectionKernels.swift`) already serves single-
threaded selection. A *concurrent* queue is needed only during multi-threaded HNSW graph
*building* — a VectorIndex activity. Per the errata, the proposal's "lock-free via os_unfair_lock"
is self-contradictory; any real concurrent queue should use `swift-atomics` and live with its
caller.

**What Core provides instead.** A pointer-level Top-K entry point (DOCUMENT-4, S1) so VectorIndex
can stop maintaining a second heap purely to avoid Core's allocation overhead.

---

### R6 — Prefetch intrinsics (proposal Task 1.3) → `VectorIndex`

**Already owned (where it matters).** `VectorIndex/Sources/VectorIndex/Operations/Support/Prefetch.swift`
wraps real prefetch builtins and applies them during HNSW neighbor gathering — the exact
random-pointer-chasing workload the proposal cites.

**Why low-value in Core.** Core's batch kernels are *sequential* over contiguous SoA storage,
where the hardware prefetcher already wins; Core's current `prefetchRead/Write`
(`Sources/VectorCore/Optimization/OptimizationAttributes.swift`) are no-op pointer dereferences
precisely because there is little to gain. The benefit lives in graph traversal, which is
downstream. If Core ever needs a *real* `__builtin_prefetch`, the home is the existing
`VectorCoreC` shim — but there is no Core caller today, so this stays deferred.

---

## DEFER

### D1 — Binary packed vectors + Hamming/Jaccard (proposal Task 1.2)

**Sound, but no committed CPU consumer.** GPU Hamming/Jaccard already exist in
`VectorAccelerate`. Core's `HammingDistance`
(`Sources/VectorCore/Operations/DistanceMetrics.swift`) operates on *float* vectors via
threshold, not packed bits — so a real `UInt64`-packed `BinaryVector` with `nonzeroBitCount`
Hamming and a correct Jaccard *is* a genuine gap.

**Why defer, not accept.** Binary screening pays off inside an index's candidate-generation
stage (USearch-style), which is VectorIndex territory, and VectorIndex has not requested it.
Accept it the moment VectorIndex or EmbedKit commits to a binary-screening path; the design is
small (packed storage + two popcount kernels) and can be lifted from this note directly.

### D2 — Sparse CSR vectors + SparseBLAS (proposal Task 2.2)

**Sound, but no producer.** Hybrid dense+sparse (SPLADE/BM25) search is real and valuable, but
**no package in the suite produces sparse embeddings today** — EmbedKit emits dense `[Float]`
only, and lists hybrid search as a *P2 roadmap* item
(`EmbedKit/dev_EKRefactor/Future_Improvements.md`). A `SparseVector` in Core would be an
unexercised type.

**Why defer, not accept.** This is a strategic expansion of the whole suite's mission (into
keyword/lexical retrieval), not a Core-local gap. It should be driven top-down from an EmbedKit
SPLADE model + a VectorIndex hybrid index, with Core's CSR type added last to serve them. Revisit
when EmbedKit schedules SPLADE.

---

## Summary

| Item | Verdict | Owner / trigger |
|---|---|---|
| mmap storage | Redirect | `VectorIndex/.../VIndexMmap.swift` |
| Pinned Metal arena | Redirect | `VectorAccelerate/.../BufferPool.swift` |
| ADC look-up tables | Redirect | `VectorIndex/.../{ADCScan,PQLUT}.swift` |
| Visited list | Redirect | `VectorIndex/.../HNSWTraversal.swift` |
| Concurrent priority queue | Redirect | `VectorIndex/.../TopK.swift` |
| Prefetch | Redirect | `VectorIndex/.../Prefetch.swift` |
| Binary vectors + Hamming/Jaccard | Defer | Trigger: VectorIndex/EmbedKit binary-screening commitment |
| Sparse CSR + SparseBLAS | Defer | Trigger: EmbedKit SPLADE + VectorIndex hybrid index |
