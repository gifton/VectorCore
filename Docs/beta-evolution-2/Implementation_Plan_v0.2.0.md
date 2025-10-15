# VectorCore v0.2.0 — Implementation Plan

This document sequences analyze → design → implement/improve → test work for the v0.2.0 beta focused on performance, benchmarking, and API stability. It references authoritative topic docs rather than duplicating details.

Authoritative references:
- Benchmarking: `Perf_Benchmarking.md`
- Concurrency: `Perf_Concurrency_and_Scheduling.md`
- API ergonomics: `Perf_API_Ergonomics.md`
- Build & presets: `Perf_Build_and_Compilation.md`
- Optimization attributes & freezing: `OptimizationPlan.md`
- C rewrite priorities: `Kernel_Audit_C_Rewrite_Priorities.md`

Guiding principles:
- Data‑first and reversible: make small, local diffs with A/B benches at each step.
- Correctness before speed: add recall/error checks; fail on drift.
- Single source of truth: keep specs in the topic docs and cross‑reference here.
- Instrumentation with negligible overhead when disabled; reproducible environments.
- Adopt underscored attributes and C kernels only behind flags and only when thresholds are met.

---

## Phase 1 — Data & Guardrails

Analyze
- Current bench coverage, variance, and correctness gaps.
- Host/toolchain metadata capture; cold vs warm behavior.

Design
- Bench matrix (N × K × dim × metric × layout) and CLI toggles: `--use-underscored`, `--use-ckernels`, `--prefer-soa`, `--release-size`.
- Baseline JSON format with host metadata; regression thresholds (default −3%).
- Warmup + minimum wall‑time per case; optional cold/warm modes.
- Roofline microbenches to contextualize memory‑bound kernels.

Implement/Improve
- Add toggles, baselines, binary size capture, and host metadata.
- Add roofline microbenches and instrumentation hooks (allocs/copies avoided, task counts) where available.

Test/Validate
- Establish reproducible baselines; measure distribution (median/p95).
- Add correctness checks: Top‑K recall and distance error bounds.

Deliverables
- Bench CLI toggles + matrix, baselines with metadata, size capture, roofline calibration, correctness harness.

Acceptance criteria
- Benches stable and reproducible; correctness gates in place; CI stores results and sizes per run.

References: `Perf_Benchmarking.md`

---

## Phase 2 — Concurrency & Scheduling

Analyze
- Task counts, chunk sizes, oversubscription, and crossover points by dim/kernel.

Design
- Adaptive `minChunk` heuristic; cap concurrency ≈ cores; nested‑parallel detection.
- Prefetch‑distance knob for SoA; deterministic reductions.
- Light instrumentation (tasks created, avg chunk size, throttling events, scheduling vs kernel time).

Implement/Improve
- Update `CPUComputeProvider` with adaptive chunking and caps; emit debug signals when throttling nested regions.
- Wire instrumentation to bench outputs via `ExecutionReport`/`VC_TRACE`.

Test/Validate
- Improved scaling vs cores; reduced variance; documented crossover points per dim/kernel.

Deliverables
- Provider upgrades + instrumentation; documented defaults and knobs.

Acceptance criteria
- Demonstrated parallel efficiency gains on representative cases; stable determinism and reporting.

References: `Perf_Concurrency_and_Scheduling.md`

---

## Phase 3 — API Fast Paths & Plans (MVP)

Analyze
- Hot callsites for `findNearest`/`computeDistances`; opportunities to eliminate runtime type checks.

Design
- Typed overloads for `Vector512Optimized`/`768`/`1536` on hot APIs.
- Minimal `DistancePlan`/`TopKPlan` that reuse SoA, norms, and scratch buffers.
- Centralize thresholds (heap vs selection; sequential vs parallel) in one place.

Implement/Improve
- Add typed overloads; keep generic fallbacks (optionally disfavored behind a flag).
- Introduce minimal plan types and wire through hot paths.

Test/Validate
- Verify compile‑time dispatch at callsites (no runtime `is` checks); measure microbench wins and allocation reductions.

Deliverables
- Fast‑path overloads, minimal plans, centralized thresholds.

Acceptance criteria
- Measurable latency/throughput wins vs generic paths; correctness preserved.

References: `Perf_API_Ergonomics.md`

---

## Phase 4 — Buffer & Scratch Management

Analyze
- Allocation hotspots, scratch sizes, and pool behavior.

Design
- `requiredScratchSize()` helpers and APIs to pass pre‑allocated scratch.
- Expose BufferProvider tuning (pool limits, alignment) to advanced callers.

Implement/Improve
- Add APIs with alignment assertions; document ownership and zero‑copy behavior.

Test/Validate
- Reduced allocations and copies in benches; confirm zero‑copy where promised.

Deliverables
- Scratch APIs and BufferProvider tunables with examples.

Acceptance criteria
- Allocation/copy reductions recorded in instrumentation; no correctness regressions.

References: `Perf_API_Ergonomics.md`

---

## Phase 5 — Build Optimization & Presets

Analyze
- Cross‑module inlining opportunities; code size implications of inlining/emit‑into‑client.

Design
- `perf` vs `size` presets via scripts/Makefile/CI; Release CMO; dead_strip for benches.
- Bench uses `perf` preset by default; size recorded for each preset.

Implement/Improve
- Wire flags in `Package.swift` (Release); add scripts/Make targets and CI jobs.

Test/Validate
- Track size/perf per preset; A/B CMO and feature flags; rollback on size growth without wins.

Deliverables
- Presets, CI jobs, size metrics integrated into bench outputs.

Acceptance criteria
- Reproducible builds; clear perf/size trade‑offs published per run.

References: `Perf_Build_and_Compilation.md`

---

## Phase 6 — Underscored Attributes (Flag‑Gated)

Analyze
- Identify tiny, hot helpers where `_transparent`/`_alwaysEmitIntoClient`/`_optimize(speed)` may help; estimate size cost.

Design
- Gate via `VC_ENABLE_UNDERSCORED`; adoption thresholds ≥ 3–5% perf win with acceptable size growth.

Implement/Improve
- Apply attributes to a small set of candidates; keep changes local and reversible.

Test/Validate
- A/B benches and binary size; adopt only wins meeting thresholds.

Deliverables
- Minimal attribute pass with measurements and rollback criteria.

Acceptance criteria
- Retained attributes provide consistent wins; size regressions justified or rolled back.

References: `OptimizationPlan.md`, `Perf_Benchmarking.md`

---

## Phase 7 — C Kernel Prototypes (Flag‑Gated)

Analyze
- Highest ROI kernels: fp32 dot/L2 (512), int8 dot; target CPU features (NEON/SDOT, AVX2/VNNI).

Design
- Thin C ABI and Swift wrappers; runtime CPUID gating; strict preconditions; correctness harness.

Implement/Improve
- Prototype `vc_dot_fp32_512`, `vc_l2sq_fp32_512`, `vc_dot_int8` behind `VC_USE_C_KERNELS`.

Test/Validate
- Randomized correctness vs Swift reference; A/B thresholds ≥ 5–10% win; size tracking.

Deliverables
- Prototypes, wrappers, gating, and CI comparisons.

Acceptance criteria
- Default remains Swift; C paths adopted only when thresholds met on target hardware.

References: `Kernel_Audit_C_Rewrite_Priorities.md`, `Perf_Benchmarking.md`

---

## Phase 8 — Selective ABI Freeze & CI Enhancements

Analyze
- Public types safe to freeze now vs evolving internals (providers, capabilities, quantization).

Design
- Freeze outward, value‑like types; defer internals; PRs document rationale per type.
- Expand CI to run benches per preset/flag and publish results.

Implement/Improve
- Add `@frozen` selectively; add CI jobs for presets and flags.

Test/Validate
- Optional evolution‑ON sanity build; verify API/ABI stability for frozen types.

Deliverables
- Selectively frozen types; expanded CI bench matrix with published artifacts.

Acceptance criteria
- No API/ABI regressions; docs and release notes reflect frozen surfaces.

References: `OptimizationPlan.md`, `Perf_Build_and_Compilation.md`

---

## Milestones & Ordering

- M1: Bench infra & guardrails (Phase 1)
- M2: Concurrency & scheduling (Phase 2)
- M3: API fast paths & minimal plans (Phase 3)
- M4: Buffer & scratch APIs (Phase 4)
- M5: Build presets & size tracking (Phase 5)
- M6: Underscored attribute trial (Phase 6, gated)
- M7: C kernel prototypes (Phase 7, gated)
- M8: Selective ABI freeze & release prep (Phase 8)

---

## CI Gates & Thresholds

- Regression threshold: fail benches if median throughput drops worse than −3% vs baseline unless acknowledged.
- Adoption thresholds:
  - Underscored attributes: ≥ 3–5% win with acceptable size delta.
  - C kernels: ≥ 5–10% win or uniquely unlocking SDOT/VNNI.
- Publish: JSON results, binary sizes, and short markdown summaries per run.

---

## Risks & Rollback

- Code size growth from emit‑into‑client/transparent helpers → track size; rollback on unjustified growth.
- Oversubscription/variance under concurrency → cap tasks and instrument; adjust thresholds.
- C interop UB (alignment/aliasing) → assert preconditions; keep kernels tiny; strong tests.

Rollback policy: any change failing correctness or breaching thresholds reverts to prior behavior; keep diffs small to make rollback easy.

---

## Release Readiness Checklist

- [ ] Bench matrix implemented with correctness gates and baselines stored
- [ ] Concurrency improvements deliver measured scaling with instrumentation
- [ ] Typed fast paths and minimal plans in place with wins
- [ ] Scratch/buffer APIs reduce allocations in hot paths
- [ ] Build presets reproducible; size/perf trade‑offs published
- [ ] Gated features adopted only where thresholds met (or deferred)
- [ ] Selected public types frozen with documented rationale
- [ ] Docs updated to reflect fast paths, buffers, and presets

