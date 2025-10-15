# VectorCore Kernel Audit — C Rewrite Priorities

This document identifies VectorCore kernels that can benefit from C (and arch-specific intrinsics) implementations, outlines expected performance gains, and proposes an interop and validation plan.

## Summary

- High‑ROI for C: INT8 quantized kernels (use SDOT/UDOT on ARM, VNNI/PMADD on x86), SoA/AoS batch loops for large N, and fused cosine.
- Moderate ROI: Top‑K heap inner loops (small‑K), normalization helpers.
- Low ROI: Manhattan and cold/rare metrics.
- Approach: Add a `VectorCoreC` target with per‑arch files, expose a thin C ABI, and wire Swift glue behind a feature flag. Validate with A/B benches and size tracking.

## High‑ROI Candidates

- INT8 Quantized Kernels
  - Why: ARMv8.2+ SDOT/UDOT; x86 AVX2/AVX‑512 (VNNI) accelerate int8 dot/L2^2 beyond Swift simd.
  - Targets:
    - `QuantizedKernels.euclidean_generic_int8` and helper `accumulateEuclidDiffSq`.
    - Mixed precision: `euclidean_generic_mixed` (vectorized dequantize + compute).
    - Dequantize helper: `convertToFP32_NEON` (replace with wider vectorized lanes).
  - Expected gains: 2–4× (ARM SDOT), 1.5–3× (x86 AVX2/AVX‑512) for int8×int8; 1.5–2× for dequantization.

- SoA Batch Kernels (large N)
  - Why: C can use `restrict`, explicit `__builtin_prefetch`, streaming loads/stores, and stronger aliasing assumptions.
  - Targets: `BatchKernels_SoA.euclid2_blocked`, `dot_blocked_2way`, `dot_blocked_4way`, fused cosine variants.
  - Expected gains: 10–25% for N ≥ 1k (memory‑bound improvements).

- AoS Batch Kernels (two‑way blocked)
  - Why: Wider vectors (AVX2/512), explicit unrolling, and zero bounds checks.
  - Targets: `BatchKernels.range_euclid2_512/768/1536` and related.
  - Expected gains: 5–20% depending on CPU and alignment.

- Fused Cosine
  - Why: Fewer temporaries and tighter FMA scheduling in C intrinsics.
  - Targets: `CosineKernels.fused`, `TopKSelectionKernels.range_topk_cosine_fused_512`.
  - Expected gains: 5–15%.

## Moderate‑ROI Candidates

- Top‑K Selection Inner Loops
  - Why: Branchless/selection‑network variants for small K; tighter cache behavior.
  - Targets: `TopKBuffer.heapifyDown`, `mergeTopK`, and `range_topk_*` wrappers when K is small.
  - Expected gains: 0–15% (most for small K, large N).

- Normalize Kernels
  - Why: Wider loads and fewer passes.
  - Targets: `NormalizeKernels.magnitudeSquared`, `scaleInPlace`.
  - Expected gains: 5–10%.

## Lower‑ROI (Keep in Swift unless profiling says otherwise)

- Manhattan kernels: `ManhattanKernels.*`.
- Edge caches/utilities: `NormCache`, `SoAFP16Cache`.
- Rare/cold metrics (Hamming, Minkowski).

## Interop Design

- New C target: `VectorCoreC` with per‑arch files:
  - arm64: NEON (SDOT/UDOT) intrinsics.
  - x86_64: AVX2/AVX‑512 (PMADDUBSW/VPDPBUSD) with CPUID gating; SSE fallback.
  - Common scalar fallback.
- Example C APIs:
  - `float vc_dot_fp32_512(const float* a, const float* b);`
  - `float vc_l2sq_fp32_512(const float* a, const float* b);`
  - `void vc_range_l2sq_fp32_512(const float* q, const float* base, size_t strideFloats, size_t start, size_t end, float* out);`
  - `int32_t vc_dot_int8(const int8_t* a, const int8_t* b, size_t lanes);`
  - `void vc_dequantize_fp32_from_int8(const int8_t* in, float scale, int8_t zp, float* out, size_t lanes);`
- Swift glue:
  - Thin wrappers using `withUnsafeBufferPointer` over `ContiguousArray<SIMD4<Float>>`/`SIMD4<Int8>`; assert alignment and reinterpret as `float*`/`int8_t*`.
  - Feature flag `VC_USE_C_KERNELS`; runtime CPU feature checks for AVX2/AVX‑512.

## Data Layout Notes

- `ContiguousArray<SIMD4<Float>>` is tightly packed and can be reinterpreted as `float*` (512/768/1536 dims = 4×lanes).
- SoA C kernels can accept per‑lane contiguous blocks or a pre‑transposed layout (like existing quantized SoA types).

## Platform Considerations

- Apple platforms: consider vDSP for some BLAS‑like ops (dot, L2) as an alternative; C intrinsics still useful for custom fused kernels.
- Linux x86: prefer AVX2 baseline; enable AVX‑512 guarded by CPUID.

## Validation Plan

- Benchmarks (A/B):
  - 512/768/1536 dot, L2^2, fused cosine; AoS and SoA; N ∈ {128, 1k, 10k}, K ∈ {1, 10, 100}.
  - INT8 paths: compare Swift vs C (SDOT/VNNI).
- Metrics:
  - Throughput deltas (%), distributions, instruction counts (local `perf stat`), code size deltas.
- Guardrail: adopt C path by default only when ≥ 5–10% improvement (or uniquely unlocking SDOT/VNNI).

## Correctness Harness
- Add unit‑style validation for each C kernel with randomized inputs and golden references.
- For Top‑K, validate exact equivalence (indices + ordering) vs Swift path for FP32; define tolerance for quantized paths.
- Assert preconditions in Swift wrappers (alignment, strides, sizes) and trap in debug if violated.

## Adoption & CI Gating
- Keep all C paths behind `VC_USE_C_KERNELS` and runtime CPU feature checks.
- Add CI jobs that run benches with and without `VC_USE_C_KERNELS` and publish deltas; gate adoption on configured thresholds.
- Default to Swift implementations unless C paths clear thresholds on target hardware.

## Risks & Trade‑offs

- Maintenance (arch‑specific code, CPUID, build matrix).
- UB risks (alignment/aliasing): mitigate with asserts, tests, and strict preconditions.
- Binary size growth: keep functions minimal and dimension‑specialized only where used.

## Recommended Next Steps

1. Prototype C kernels:
   - `vc_dot_fp32_512` and `vc_l2sq_fp32_512` (AoS) with NEON and AVX2 backends.
   - `vc_dot_int8` using SDOT (arm64) and VNNI/PMADD (x86).
2. Wire behind `VC_USE_C_KERNELS` and add Swift wrappers in `DotKernels`/`EuclideanKernels`.
3. Add A/B benches and capture deltas on your hardware.
4. If gains are strong, extend to range/SoA kernels and fused cosine; consider small‑K Top‑K selection.
