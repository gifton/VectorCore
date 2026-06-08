# Beta-Evolution-4 — Master Decision Review

**Status:** Strategic review + design specs — **implemented on `feature/beta-evo-4`, targeting 0.3.0** (released 2026-06-07)
**Target release:** VectorCore `0.3.0`
**Date:** 2026-06-06 (implementation reconciled 2026-06-07)
**Inputs reviewed:** External "Documents 0–4" strategy (Gemini); VectorCore `0.2.2` source;
downstream packages `VectorIndex` `0.1.5`, `VectorAccelerate` `0.5.0`, `EmbedKit` `0.1.0`.

---

## 1. Thesis: Hold the line, sharpen the seams

The external proposal asks VectorCore to become a SOTA "micro-kernel & memory primitive
provider" that competes with Faiss / USearch / DiskANN / GGML by absorbing block quantization,
binary vectors, prefetch, GEMM, sparse algebra, `mmap` storage, pinned Metal arenas, ADC
look-up tables, HNSW visited-lists, and concurrent priority queues.

The proposal is technically literate, but it was written **without visibility into the three
real downstream packages.** When measured against them, most of its recommendations describe
code that already ships:

- **Documents 3 and 4 are, almost line-for-line, already implemented downstream.**
- The few genuinely *missing* primitives are a small subset of Documents 1–2.
- The proposal **omits the work the ecosystem actually asks Core for** — thin pointer-level
  APIs and shared provider protocols, published as explicit "requests" in the downstream repos.

The correct strategy is therefore not "grow Core to match the proposal," nor "reject everything,"
but: **redirect the already-owned items to their real owners, accept the narrow CPU-primitive
gaps, and invest the real effort in the seams downstream is requesting.** VectorCore's leverage
is not feature breadth — it is being the small, correct, allocation-free foundation that the rest
of the suite is *currently bypassing* because the seams are too coarse.

> **Smell test for "does this belong in Core?"** A primitive belongs in Core when it is
> (a) CPU-only, (b) distance-metric / index-topology agnostic, (c) reusable by ≥2 downstream
> packages, and (d) not already owned by exactly one of them. By this test, GEMM-batch (CPU),
> block-quant format, and the seam APIs qualify; mmap, GPU arenas, ADC, visited-lists, and
> priority queues do not.

---

## 2. Ecosystem maturity map

| Package | Version | Size | Owns today (relevant to the proposal) |
|---|---|---|---|
| **VectorCore** | 0.2.2 | — | Vector types, SIMD/SoA kernels, INT8 + FP16 quant primitives, Top-K heap, aligned memory, provider protocols |
| **VectorIndex** | 0.1.5 | ~25k LoC | HNSW (bitset visited-list), `TopKHeap`, full PQ/IVF + ADC LUTs, `mmap` layout (WIP), prefetch |
| **VectorAccelerate** | 0.5.0 | ~53k LoC, 81 Metal shaders | GPU tiled GEMM, zero-copy `MTLStorageModeShared` bridging, ring-buffer pinned arenas, CPU/GPU routing facade, GPU Hamming/Jaccard |
| **EmbedKit** | 0.1.0 | ~31k LoC | text → dense `[Float]` → `DynamicVector`; INT8/INT4/FP16/binary quant (per-vector, GPU) |

`VectorAccelerate` lives under a `future/` directory only because it requires Metal 4
(macOS 26+) — it is a real, released, production package, not a stub.

Two downstream signals dominate this review:

1. **`VectorIndex` deliberately bypasses Core's typed API** in hot paths (uses raw `[Float]` /
   `UnsafePointer<Float>`) because the typed API allocates. See `VectorIndex/IMPROVEMENTS.md`.
2. **Both `VectorIndex` and `VectorAccelerate` publish concrete "asks" of Core** that the
   proposal never mentions (pointer-level Top-K, raw-buffer normalize, configurable tie-breaking,
   `ComputeProvider`/`BufferManagementProvider`/`QuantizationProvider`). These are the real gaps.

---

## 3. Verdict table

Legend: **Accept** = design spec in this suite, slated for `0.3.0`. **Defer** = sound but no
ready producer/consumer; documented for later. **Redirect** = already owned by a sibling; Core
duplicating it would fork live code. ✅ **shipped 0.3.0** marks rows whose accepted spec is now
implemented on `feature/beta-evo-4`; **⏸ Deferred** marks accepted specs that remain unbuilt
(consumer-gated).

| # | Proposal task | In Core `0.2.2`? | Verdict | Evidence / owner (file pointer) |
|---|---|---|---|---|
| 1.1 | Block-wise quant (`Q8_0`) | Per-vector INT8 only | **⏸ Deferred** (specced, **not built** — consumer-gated) | Gap: `Quantization/QuantizationSchemes.swift` is whole-vector. Spec → DOCUMENT-3; held until a consumer (EmbedKit storage / VectorIndex codes) commits |
| 1.2 | Binary vectors + Hamming/Jaccard | Hamming-on-float only; no packed type | **Defer** → **Redirect** (VectorAccelerate) | GPU versions exist in `VectorAccelerate`; binary-packed + Hamming/Jaccard redirected there, no committed CPU consumer (confirmed not in Core 0.3.0). DOCUMENT-1 |
| 1.3 | Prefetch intrinsics | `prefetchRead/Write` are no-ops | **Redirect** (confirmed not in Core 0.3.0) | `VectorIndex/Sources/VectorIndex/Operations/Support/Prefetch.swift` (real builtins) |
| 2.1 | GEMM batch distance (GPU) | None | **Redirect** (confirmed not in Core 0.3.0) | `VectorAccelerate` tiled GPU GEMM (`Metal/Shaders/OptimizedMatrixOps.metal`) |
| 2.1 | GEMM batch distance (**CPU/AMX**) | None | **Accept** ✅ **shipped 0.3.0** | Gap closed: CPU `cblas_sgemm`→AMX path → `Sources/VectorCore/Operations/MatrixDistance.swift` (`euclideanSquaredMatrix` / `cosineDistanceMatrix`, `prepare(_:normalized:)`, `PreparedCandidates`). Spec → DOCUMENT-2 |
| 2.2 | Sparse CSR + SparseBLAS | None | **Defer** (no producer) | No producer (EmbedKit dense-only; hybrid is a P2 roadmap item); deferred — confirmed not in Core 0.3.0. DOCUMENT-1 |
| 3.1 | `mmap` storage | None | **Redirect** (confirmed not in Core 0.3.0) | `VectorIndex/Sources/VectorIndex/Kernels/VIndexMmap.swift` (real, WIP) |
| 3.2 | Pinned Metal arena | `MemoryPool` (not pinned) | **Redirect** (confirmed not in Core 0.3.0) | `VectorAccelerate/Sources/VectorAccelerate/Core/BufferPool.swift` |
| 4.1 | ADC look-up tables | None; PQ deferred | **Redirect** (confirmed not in Core 0.3.0) | `VectorIndex/Sources/VectorIndex/Operations/Quantization/{ADCScan,PQLUT}.swift` |
| 4.2 | Generation visited-list | None | **Redirect** (confirmed not in Core 0.3.0) | `VectorIndex/Sources/VectorIndex/Kernels/HNSWTraversal.swift` (bitset) |
| 4.3 | Concurrent priority queue | Single-thread `TopKBuffer` | **Redirect** (confirmed not in Core 0.3.0) | `VectorIndex/Sources/VectorIndex/Operations/Selection/TopK.swift` (`TopKHeap`) |
| — | Pointer-level Top-K API | None | **Accept** ✅ **shipped 0.3.0** | Requested: `VectorIndex/IMPROVEMENTS.md` §9.5 → `Sources/VectorCore/Operations/TopKSelection.swift`. Spec → DOCUMENT-4 |
| — | Raw-buffer in-place normalize | Typed only | **Accept** ✅ **shipped 0.3.0** | Requested: `VectorIndex/IMPROVEMENTS.md` §9.5. Spec → DOCUMENT-4 |
| — | Configurable tie-breaking | Insertion-order only | **Accept** ✅ **shipped 0.3.0** | Requested: `VectorIndex/IMPROVEMENTS.md` §9.5 → `TieBreaker` (`.smallerIndex` default / `.insertionOrder` / `.smallerValue`) in `Operations/TopKSelection.swift`. Spec → DOCUMENT-4 |
| — | Provider protocols | Partial | **Accept** ✅ **shipped 0.3.0** | Requested: `VectorAccelerate/docs/ADD/VECTORCORE_ALIGNMENT_ROADMAP.md` → `BatchKernelProvider` (`Sources/VectorCore/Protocols/BatchKernelProvider.swift`). DOCUMENT-4 |
| — | Zero-copy buffer contract | `AlignedMemory` exists | **Accept** ✅ **shipped 0.3.0** | EmbedKit P1 zero-copy (`EmbedKit/dev_EKRefactor/Future_Improvements.md`) → `UnifiedVectorBuffer` + `PageAlignedBuffer` (`Storage/UnifiedVectorBuffer.swift`) and frozen `SoALayout` contract (`Storage/SoALayout.swift`). DOCUMENT-4 |

**Net `0.3.0` Core work (as shipped):** the seam-hardening track (DOCUMENT-4) and the CPU GEMM
batch-distance primitive (DOCUMENT-2) shipped on `feature/beta-evo-4`; the block-quant format
(DOCUMENT-3) was **deferred** (specced, not built — consumer-gated). Everything else is
redirected or deferred with rationale.

Additionally shipped under the DOCUMENT-4 / SoA-layout track in `0.3.0`: the frozen `SoALayout`
descriptor and SoA layout contract (`SoALayout.forType(_:count:pageAligned:)`, `SoA.layoutDescriptor`),
SoA page-alignment APIs (`SoA.build(from:pageAligned:)`, `init(vectors:pageAligned:)`,
`pageAlignedBytes`, `consumeAllocation()`, `withUnsafeRawBuffer`), `BatchOperations.Configuration`
matrix routing (`enableMatrixRouting` default `true`, `matrixRoutingMinN` default `256`),
`PlatformConfiguration.roundUpToPage(_:)`, and the **breaking** removal of the no-op `blockSize:`
parameter from `SoA.init(vectors:…)` and `SoAFP16.init(vectors:…)`.

---

## 4. Corrected execution order

The proposal's Document 0 sequences GEMM → quant → numerical-rigor → mmap, justified as
"GEMM needs no new memory structures." After pruning to the accepted set, the dependency that
actually matters is **layouts before the math that consumes them**, and **seams before the
primitives that ride on them** (so downstream can adopt incrementally):

1. **Seams first (DOCUMENT-4) — ✅ done in `0.3.0`.** Pointer APIs, tie-breaking (`TieBreaker`),
   the provider protocol (`BatchKernelProvider`), the zero-copy buffer contract
   (`UnifiedVectorBuffer`/`PageAlignedBuffer`), and the frozen `SoALayout` contract shipped first
   — they were additive, low-risk, and immediately unblocked `VectorIndex`/`VectorAccelerate` so
   they stop bypassing Core.
2. **GEMM batch-distance (DOCUMENT-2) — ✅ done in `0.3.0`.** Pure CPU (`cblas_sgemm`→AMX),
   reused the existing optimized storage, highest throughput ROI; landed as `MatrixDistance`
   with no new public types beyond the entry point (and transparent routing via
   `BatchOperations.Configuration.enableMatrixRouting`).
3. **Block quant (DOCUMENT-3) — still gated (deferred, not built in `0.3.0`).** Will only
   proceed once a consumer (EmbedKit storage or VectorIndex codes) commits; otherwise it is an
   unexercised format.

This inverts the proposal's "GEMM first" because the seams are what the ecosystem is blocked on
*today*, and they cost the least.

---

## 5. Corrections appendix (technical errata in the proposal)

Recorded so they are not silently propagated into implementation:

1. **"Lock-free … using `os_unfair_lock` (Spinlocks)"** (Task 4.3) is a contradiction. A
   spinlock is a *lock*; lock-free means progress without mutual exclusion (CAS loops). The
   downstream `TopKHeap` is correctly actor-isolated, not "lock-free." If a true concurrent
   queue is ever needed, it belongs in `VectorIndex` and should use `swift-atomics` /
   `Synchronization.Atomic`, not the deprecated `OSAtomic`.
2. **"Zero floating-point accumulation drift"** (Task 4.1) is unattainable — IEEE-754 addition
   is non-associative. The correct contract is *bounded* drift via Kahan/Neumaier or pairwise
   summation. DOCUMENT-2 and DOCUMENT-3 state error bounds rather than promising zero.
3. **GGML `Q8_0` stores a per-block scale only** (symmetric). The proposal's "scale *and*
   offset per block" describes `Q8_1`. DOCUMENT-3 specifies `Q8_0` correctly and notes `Q8_1`
   as a variant.
4. **"GPU pages vector data directly from SSD … without CPU intervention"** (Task 3.1)
   overstates it: `mmap` + `makeBuffer(bytesNoCopy:)` still services page faults through the
   kernel/VM subsystem. The DiskANN benefit is real but is not literally CPU-free.
5. **AMX routing claim is correct.** Accelerate's BLAS (`cblas_sgemm`) does dispatch to the AMX
   coprocessor on Apple Silicon; DOCUMENT-2 relies on this and treats it as the intended path
   (no undocumented intrinsics).

---

## 6. Document index

| Doc | Contents |
|---|---|
| `be4-master.md` | This file — thesis, ecosystem map, verdict table, execution order, errata |
| `DOCUMENT-1_Redirections.md` | Per-item redirect/defer rationale with sibling-repo pointers |
| `DOCUMENT-2_Spec_GEMM_Batch_Distance.md` | CPU AMX GEMM batch-distance design spec |
| `DOCUMENT-3_Spec_Block_Quantization.md` | `Q8_0` block-wise quantization design spec |
| `DOCUMENT-4_Spec_Ecosystem_Seams.md` | Downstream-requested seams (pointer APIs, protocols, zero-copy) |
| `DOCUMENT-4_S4_Provider_Conformance.md` | Provider conformance contract (ComputeProvider/BufferProvider/AccelerationProvider) |
| `DOCUMENT-5_VectorAccelerate_Integration_Requests.md` | Gap analysis + plan for VA's R1–R4 (SoA page-align/accessor, version reply, BatchKernelProvider) |
| `DOCUMENT-6_Page_Alignment_Feasibility.md` | Feasibility of broader `bytesNoCopy`-eligible storage; why batches (SoA/PageAlignedBuffer) not per-vector; `PageBridgeable` sketch |
| `../SoA_Layout_Contract.md` | 🔒 **Frozen** SoA memory-layout contract (0.3.0) — the permanent reference VA's zero-copy Metal kernels pin to. Index formula, `SoALayout` descriptor, page-rounding caveat, free/lifetime contract, golden parity fixture. Promoted out of beta-evo-4 because it is a durable contract, not a planning doc. |
