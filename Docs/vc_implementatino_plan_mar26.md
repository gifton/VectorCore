# VectorCore v0.2.0: Agent Implementation Plan

**Context for the Agent:**
This codebase is a high-performance vector mathematics library for Apple Silicon and Intel. It relies heavily on `SIMD4<Float>` data packing, Structure-of-Arrays (SoA) memory layouts, and strict Swift 6 concurrency. 

Your goal is to implement the next phase of performance optimizations based on the `ROADMAP.md` and `beta-evolution-2` specs. Do not break existing generic fallbacks. Focus on localized, measurable optimizations.

---

## Epic 1: Unblock GPU Acceleration (DistanceMetric Batch Protocol)
**Files:** `Sources/VectorCore/Protocols/DistanceMetric.swift`, `Sources/VectorCore/Operations/DistanceMetrics.swift`

*   **The Problem:** `DistanceMetric` currently provides `batchDistance` inside a protocol *extension*. Because it's not a formal protocol requirement, `VectorAccelerate` cannot provide a Metal-backed metric that dynamically overrides the default CPU implementation (Roadmap Item 2.2).
*   **The Solution:** Make batching a formal protocol requirement to enable polymorphic GPU dispatch.
*   **Agent Tasks:**
    1. In `DistanceMetric.swift`, move the `batchDistance` function signature from the `extension` directly into the `protocol DistanceMetric` definition.
    2. Keep the existing extension as the default implementation so existing conformances don't break.
    3. Ensure `EuclideanDistance`, `CosineDistance`, `ManhattanDistance`, etc., in `DistanceMetrics.swift` and `OptimizedDistanceMetrics.swift` explicitly declare `public func batchDistance...` to fulfill the new protocol requirement, retaining their optimized implementations.

## Epic 2: Zero-Copy Pointer APIs for VectorIndex
**Files:** `Sources/VectorCore/Operations/TopKSelection.swift`, `Sources/VectorCore/Operations/Kernels/NormalizeKernels.swift`

*   **The Problem:** `VectorIndex` needs to perform Top-K selection and normalization directly from memory-mapped files (mmap) or an `IOSurface` from `VectorAccelerate` without copying data into Swift `Array`s (Roadmap Items 9.3 & 4.2).
*   **The Solution:** Add raw pointer-based execution paths.
*   **Agent Tasks:** 
    1. In `TopKSelection.swift`, add a new public static method:
       ```swift
       public static func select(
           k: Int,
           from distances: UnsafePointer<Float>,
           count: Int,
           ids: UnsafePointer<Int32>? = nil
       ) -> (indices: [Int32], distances: [Float])
       ```
    2. Implement the logic using a heap for `k < count/10`. If `ids` is provided, use it for the returned indices; otherwise, use `Int32(0..<count)`.
    3. In `NormalizeKernels.swift`, add `public static func normalizeUnchecked(_ buffer: UnsafeMutablePointer<Float>, dimension: Int)`. Implement it using standard SIMD pointers, skipping the `ContiguousArray` abstraction.

## Epic 3: C-Kernel Prototypes (Hardware-Specific Intrinsics)
**Files:** `Sources/VectorCoreC/arm64/vc_arm64.c`, `Sources/VectorCoreC/x86_64/vc_x86.c`, `Sources/VectorCore/Interop/CKernels.swift`, `Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift`

*   **The Problem:** `VectorCoreC` contains stubs (e.g., `vc_arm64_dot_fp32_512`) that merely fallback to scalar reference logic. To achieve maximum throughput for INT8 quantization, you must utilize ARM NEON's `SDOT` and Intel's `AVX2/VNNI` instructions. 
*   **The Solution:** Fulfill the C-Kernel scaffold.
*   **Agent Tasks:**
    1. In `arm64/vc_arm64.c`, include `<arm_neon.h>`. Implement `vc_arm64_dot_fp32_512` using `vld1q_f32` (load 4 floats) and `vfmaq_f32` (fused multiply-add) with 4 unrolled accumulator registers.
    2. Implement `vc_dot_int8` for ARM using the `vdotq_s32` (SDOT) instruction to accumulate the dot product into an `int32x4_t` register, then horizontally add the result (`vaddvq_s32`).
    3. In `x86_64/vc_x86.c`, implement `vc_dot_int8` using `<immintrin.h>` (AVX2) via `_mm256_maddubs_epi16` and `_mm256_madd_epi16`.
    4. In `CKernels.swift`, expose `static func dotInt8`.
    5. In `QuantizedKernels.swift`, update `euclidean_generic_int8` and `accumulate_fused_generic` to route the dot product calculation through `CKernels.dotInt8` when `FeatureFlags.useCKernels` is true and the hardware supports it.

## Epic 4: SoA 4-Way Register Blocking
**File:** `Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift`

*   **The Problem:** `BatchKernels_SoA.swift` currently uses 2-way register blocking (`euclid2_blocked`, `dot_blocked_2way`). For large candidate batches (N ≥ 1000), 4-way register blocking provides better ILP (Instruction-Level Parallelism) and cache utilization on Apple Silicon (Roadmap Item 4.1).
*   **The Solution:** Expand the unrolling in the Structure-of-Arrays kernels.
*   **Agent Tasks:**
    1. In `BatchKernels_SoA.swift`, create a new internal function `euclid2_blocked_4way` that processes candidates in blocks of 4 (`blockSize = 4`).
    2. Instantiate 4 separate `SIMD4<Float>` accumulators (`acc0`, `acc1`, `acc2`, `acc3`).
    3. Load a single query lane once, load the 4 candidate lanes, compute differences, and use `addProduct(diff, diff)` on all 4 accumulators.
    4. Update `euclid2_512`, `euclid2_768`, and `euclid2_1536` to branch: `if soa.count >= 4 { euclid2_blocked_4way(...) } else { euclid2_blocked(...) }`.

## Epic 5: Concurrency: Nested Parallelism Prevention
**File:** `Sources/VectorCore/Providers/CPUComputeProvider.swift`

*   **The Problem:** `CPUComputeProvider` spawns task groups via `parallelReduce` and `parallelExecute`. If a downstream consumer (like `VectorIndex`) calls a batch operation from *inside* an already concurrent context (e.g., searching multiple nodes in an HNSW graph concurrently), the Swift cooperative thread pool will oversubscribe, causing severe context-switching overhead.
*   **The Solution:** Implement nested parallel detection using `@TaskLocal`.
*   **Agent Tasks:**
    1. In `CPUComputeProvider.swift`, add a `@TaskLocal internal static var isInsideParallelRegion: Bool = false`.
    2. Update `parallelExecute`, `parallelForEach`, and `parallelReduce`. In the `shouldParallelize` boolean logic, add: `if CPUComputeProvider.isInsideParallelRegion { return false }`.
    3. If `shouldParallelize` evaluates to `true`, wrap the `withThrowingTaskGroup` execution in `Self.$isInsideParallelRegion.withValue(true) { ... }`.