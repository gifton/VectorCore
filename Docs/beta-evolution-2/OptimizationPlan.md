# VectorCore Performance Optimization Plan

This document captures a phased plan to improve API performance and ergonomics using Swift’s optimization attributes and related techniques.

Applies to: VectorCore package (Sources/VectorCore/*) and Benchmarks/VectorCoreBench.

## Attributes In Scope

- Public/stable: `@inlinable`, `@inline(__always)`, `@frozen`, `@usableFromInline`
- Underscored (guarded): `@_alwaysEmitIntoClient`, `@_transparent`, `@_optimize(speed)`, `@_optimize(size)`, `@_effects(readnone)`, `@_effects(readwrite)`, `@_disfavoredOverload`, `@_unsafeInheritExecutor`, `@_cdecl`, `@_silgen_name`, `@_semantics("…")`
- Future/experimental: `@_alignment(n)`, `@_moveOnly` (Swift 6)

## Guardrails & Flags

- Build flags:
  - `VC_ENABLE_UNDERSCORED`: enables all underscored attributes.
  - `VC_ENABLE_FFI`: enables `@_cdecl`/`@_silgen_name` exports.
- Version checks: wrap uses with `#if swift(>=5.9)` / `#if swift(>=6.0)` as appropriate.
- Bench A/B: keep two baselines (with/without `VC_ENABLE_UNDERSCORED`).
- Keep changes reversible; prefer small, local diffs.

## Phase 1 — Safe + Structural

Goal: Lock stable layouts (selectively), expand safe inlining, remove runtime dispatch where we can via overloads, and strengthen benchmarks.

- [ ] Freeze stable result and enum types (selective)
  - Freeze now (public, value‑like, outward‑facing APIs):
    - [ ] `Operations/Operations.swift`: `NearestNeighborResult`, `VectorStatistics`, `VectorStatistics.MagnitudeStats` → `@frozen`
    - [ ] `Errors/VectorError.swift`: `ErrorContext`, `ErrorKind`, `ErrorSeverity`, `ErrorCategory` → `@frozen`
    - [ ] `Protocols/ProviderProtocols.swift`: `SupportedDistanceMetric` → `@frozen`
    - [ ] `Core/Dimension.swift`: `Dim2…Dim3072`, `DynamicDimension` → `@frozen`
    - [ ] `Version.swift`: `VectorCoreVersion` → `@frozen`
  - Defer to ≥ 0.3.0 (still evolving or likely to change):
    - [ ] `Protocols/CoreProtocols.swift`: `AcceleratedOperation`
    - [ ] `Providers/CPUComputeProvider.swift`: `CPUComputeProvider.Mode`
    - [ ] `Execution/ComputeDevice.swift`: `DeviceCapabilities`, `DeviceCapabilities.Precision`
    - [ ] `Protocols/BufferProvider.swift`: `BufferHandle`, `BufferStatistics`, `BufferError`, `BufferConfiguration`
    - [ ] `Quantization/QuantizationSchemes.swift`: `QuantizationParams`, `QuantizationParams.Strategy`, `QuantizationErrorStats`
  - Guardrail: Only freeze types we commit to keep stable through 1.x; document rationale per type in PRs.

- [ ] Expand safe inlinability
  - [ ] `Optimization/OptimizationAttributes.swift`: keep `@inlinable` and add `@inline(__always)` where appropriate (`likely`, `unlikely`, `performanceCritical`).
  - [ ] `Platform/SIMDProvider.swift`: default `divideByScalar` stays `@inlinable`.
  - [ ] `VectorFactory.swift`: `createByTransforming`, `createByCombining` already `@inlinable` — verify callsites.
  - [ ] `DistanceMetrics.swift`: keep hot `distance` methods `@inlinable`.

- [ ] Replace runtime type checks with compile‑time specialization
  - [ ] `Operations.findNearest(…)`: add overloads for `Vector512Optimized`, `Vector768Optimized`, `Vector1536Optimized` that directly select the fast Top‑K paths.
  - [ ] `Operations.computeDistances(…)`: add the same typed overloads to dispatch straight to batch kernels.
  - [ ] Mark the generic fallbacks as the least preferred with `@_disfavoredOverload` (Phase 2 flag) or keep purely generic in Phase 1.

- [ ] Benchmark coverage improvements
  - [ ] Add cases in `Benchmarks/VectorCoreBench` to measure: specialized overloads vs generic, K ∈ {1,10,100}, N sweeps.
  - [ ] Add microbenchmarks for `DistanceMetrics` per metric.
  - [ ] Capture build size deltas (Release; with/without `VC_ENABLE_UNDERSCORED`).
  - [ ] Add correctness checks (Top‑K recall, distance error bounds) and fail on drift.

## Phase 2 — Targeted Underscored Attributes (Flagged)

Goal: Carefully apply underscored attributes where they consistently improve perf or code size, verified by benchmarks and code size metrics.

- Hinting and emission (hot, tiny, or used cross‑module)
  - [ ] `Optimization/OptimizationAttributes.swift`: `likely`, `unlikely`, `performanceCritical` → `@_transparent` + `@_alwaysEmitIntoClient` (guarded).
  - [ ] `Optimization/OptimizationAttributes.swift`: `assumeAligned` (both overloads) → `@_transparent`.
  - [ ] `Operations.minChunk(forDim:)` → `@inline(__always)` + `@_transparent` + `@_effects(readnone)`.
  - [ ] `Operations.mergeBuffers` → `@_optimize(speed)`.

- Batch helpers and Top‑K kernels
  - [ ] `Operations.batchKernelDistances_*` (euclid/cosine fused × 3 dims) → `@_optimize(speed)`.
  - [ ] `Operations.topk_*` (euclid × 3, cosine 512, dot 512) → `@_optimize(speed)`.
  - [ ] `Operations.toResults` → consider `@_optimize(size)` (reduce code growth).
  - [ ] `Operations` overloads: mark generic fallbacks `@_disfavoredOverload` (guarded).

- Effects annotations (pure or write‑through)
  - [ ] `Optimization/OptimizationAttributes.nextPowerOfTwo` (if moved) or `SwiftBufferPool.nextPowerOfTwo` → `@_effects(readnone)`.
  - [ ] `Execution/ComputeDevice` small computed properties → `@_effects(readnone)` where safe.
  - [ ] Range kernels that only mutate `out` → `@_effects(readwrite)` (verify no global state).

- Concurrency/executor tuning
  - [ ] Evaluate `SwiftBufferPool.statistics()` for `nonisolated`/`@_unsafeInheritExecutor` if truly read‑only and await‑free.
  - [ ] `CPUComputeProvider` `parallel*` helpers → `@_optimize(speed)` on hot overloads.

- FFI (optional, behind `VC_ENABLE_FFI`)
  - [ ] Add thin C ABI shims (e.g., `vc_topk_euclid_512`, `vc_batch_euclid_512`) using `@_cdecl` in `Sources/VectorCore/FFI/Exports.swift`.
  - [ ] Only use `@_silgen_name` for Swift‑only name control; prefer `_cdecl` for C interop.

- Semantics / alignment / move‑only (experimental)
  - [ ] Avoid stdlib `@_semantics` unless matching known patterns.
  - [ ] Consider `@_alignment(n)` for aligned storage when available.
  - [ ] Explore `@_moveOnly` experiments in Swift 6 on internal types; never on public API yet.

## Validation & Metrics

- Performance: compare medians and distributions vs baseline; require >= 3–5% improvement for attribute to stick.
- Code size: track binary size delta; rollback `_alwaysEmitIntoClient`/`_transparent` if size grows without perf win.
- ABI/API stability: `@frozen` only for types we commit to keep stable through 1.x; defer provider/internal types until APIs harden.
- CI: add jobs that build and run benches with `VC_ENABLE_UNDERSCORED` on/off and upload JSON results into `Benchmarks/.bench/`.

## Notes & Rationale

- Underscored attributes are compiler internals; we gate them and keep changes local and measurable.
- Prefer overload‑based specialization to runtime type checks for both ergonomics and optimizer friendliness.
- Use `@_optimize(size)` sparingly to curb code bloat in helper glue.
- Keep all optimizations accompanied by benches to prevent regressions over time.

## Appendix — Representative Candidates (by file)

- Optimization/OptimizationAttributes.swift: `likely`, `unlikely`, `performanceCritical`, `assumeAligned`, loop helpers.
- Operations/Operations.swift: `findNearest`/`findNearestBatch` overloads; `minChunk`, `mergeBuffers`, `toResults`, batch helpers, Top‑K.
- Operations/Kernels/*: `BatchKernels.range_*`, `TopKSelectionKernels.range_*`, `DotKernels`, `EuclideanKernels`, `CosineKernels` hot paths.
- Operations/DistanceMetrics.swift: `distance` methods for Euclidean, Cosine, Dot (hot), Hamming/Minkowski (cold → size bias).
- Platform/SIMDProvider.swift: default `divideByScalar` (transparent), provider typealiases.
- Providers/SwiftBufferPool.swift: `nextPowerOfTwo`, pool bookkeeping (pure helpers).
- Execution/ComputeDevice.swift: tiny computed properties (readnone), alignment constants.
- Core/Dimension.swift: all `Dim*` types and `DynamicDimension` (frozen).
- Protocols/*: public enums/structs (frozen). Version.swift (frozen, inlinable getters).
