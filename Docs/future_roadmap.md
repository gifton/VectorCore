# VectorCore: Technical Findings & Implementation Strategy

**Scope:** High-performance CPU-bound vector mathematics via `vDSP` and `simd`.

## Context
`VectorCore` is the fallback and low-latency engine for this stack. Because Apple Silicon CPUs have incredibly wide and fast SIMD registers (NEON), CPU execution is the optimal path for single-query searches, streaming data, and small micro-batches. It must be brutally fast and allocation-free.

## Finding 1: Swift ARC Overhead in Tight Inner Loops
*   **The Problem:** Passing Swift collections (`Array`, `Data`) into distance calculation functions passes them by reference but enforces boundary checks and potential ARC tracking on every iteration. In loops comparing hundreds of thousands of vectors, this destroys CPU L1/L2 cache performance.
*   **The Solution:** Strict Pointers and Swift 6 Noncopyables.
*   **Agent Implementation Tasks:** 
    *   Refactor core mathematical hot-paths to strictly operate on `UnsafeBufferPointer<Float>`. 
    *   Implement Swift 6 `~Copyable` (noncopyable) structs for internal scratch/vector storage to guarantee deterministic memory ownership transfer without ARC.
    *   Audit all asynchronous boundaries to ensure strict `Sendable` compliance.

## Finding 2: Missing Low-Precision & Quantization Primitives
*   **The Problem:** Standardizing entirely on 32-bit floating point (`Float32`) halves potential memory bandwidth. State-of-the-art vector workloads maximize cache locality by scaling down to `Float16` or Scalar Quantization (`Int8`), shrinking downstream RAG caches by up to 4x.
*   **The Solution:** Hardware-optimized precision types.
*   **Agent Implementation Tasks:** 
    *   Add native `Float16` support utilizing Apple's `vDSP` half-precision conversion and dot-product functions (`vDSP_dotpr_f16`).
    *   **Scalar Quantization:** Add functions to compress `Float32` bounds to `Int8` via min/max scaling, and perform dot products using integer arithmetic.
    *   **Binary Vectors:** Implement a struct for `[UInt64]` bit-packed vectors and compute Hamming distance using the ultra-fast CPU instruction `nonzeroBitCount` (popcount) via `simd_popcount`.