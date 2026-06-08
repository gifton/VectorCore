# VectorCore: Technical Findings & Implementation Strategy

**Scope:** High-performance CPU-bound vector mathematics via `vDSP` and `simd`.

> **Note (2026-06-07):** This document predates the **"hold the line, sharpen the seams"**
> strategy adopted for 0.3.0. Some primitives it lists as "missing" (FP16, scalar INT8) already
> ship in Core. The authoritative 0.3.0 record of work is
> `Docs/beta-evolution-4/be4-master.md` (+ `CHANGELOG`); findings below are kept for historical
> context and annotated inline where superseded.

## Context
`VectorCore` is the fallback and low-latency engine for this stack. Because Apple Silicon CPUs have incredibly wide and fast SIMD registers (NEON), CPU execution is the optimal path for single-query searches, streaming data, and small micro-batches. It must be brutally fast and allocation-free.

## Finding 1: Swift ARC Overhead in Tight Inner Loops
*   **The Problem:** Passing Swift collections (`Array`, `Data`) into distance calculation functions passes them by reference but enforces boundary checks and potential ARC tracking on every iteration. In loops comparing hundreds of thousands of vectors, this destroys CPU L1/L2 cache performance.
*   **The Solution:** Strict Pointers and Swift 6 Noncopyables.
*   **Agent Implementation Tasks:** 
    *   Refactor core mathematical hot-paths to strictly operate on `UnsafeBufferPointer<Float>`. 
    *   Implement Swift 6 `~Copyable` (noncopyable) structs for internal scratch/vector storage to guarantee deterministic memory ownership transfer without ARC.
    *   Audit all asynchronous boundaries to ensure strict `Sendable` compliance.

## Finding 2: Low-Precision & Quantization Primitives — *mostly shipped; re-scoped for 0.3.0*

> **Status update (2026-06-07):** FP16 and **scalar INT8** are no longer "missing" — both already
> ship in Core (FP16 mixed precision via `MixedPrecisionKernels`; scalar INT8 via
> `QuantizationSchemes`). They are struck from the task list below. The remaining quant asks are
> re-scoped per the 0.3.0 "hold the line, sharpen the seams" decision.

*   **The Problem:** Standardizing entirely on 32-bit floating point (`Float32`) halves potential memory bandwidth. State-of-the-art vector workloads maximize cache locality by scaling down to `Float16` or Scalar Quantization (`Int8`), shrinking downstream RAG caches by up to 4x.
*   **The Solution:** Hardware-optimized precision types.
*   **Agent Implementation Tasks:** 
    *   ~~Add native `Float16` support utilizing Apple's `vDSP` half-precision conversion and dot-product functions (`vDSP_dotpr_f16`).~~ ✅ **Already ships in Core** (`MixedPrecisionKernels`).
    *   ~~**Scalar Quantization:** Add functions to compress `Float32` bounds to `Int8` via min/max scaling, and perform dot products using integer arithmetic.~~ ✅ **Already ships in Core** (`QuantizationSchemes`, scalar INT8).
    *   **Block-wise quantization (`Q8_0`):** the still-open quant gap (per-block scale, not whole-vector). **DEFERRED** for 0.3.0 — specced but **consumer-gated**; will land only once a consumer (EmbedKit storage / VectorIndex codes) commits. See `Docs/beta-evolution-4/DOCUMENT-3_Spec_Block_Quantization.md`.
    *   **Binary Vectors (`[UInt64]` bit-packed + Hamming/Jaccard via popcount):** **REDIRECTED to VectorAccelerate** — GPU versions already exist there and there is no committed CPU consumer, so Core does not ship a packed binary type.