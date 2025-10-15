# VectorCore 0.2.0 Performance Opportunities

Focus areas: Build & Compilation, Concurrency & Scheduling, API Ergonomics, and Benchmarking (CPU‑only; GPU lives elsewhere).

Authoritative sources:
- Build details → see `Perf_Build_and_Compilation.md` (single source of truth for flags/presets).
- Concurrency details → see `Perf_Concurrency_and_Scheduling.md`.
- API details → see `Perf_API_Ergonomics.md`.
- Benchmark details → see `Perf_Benchmarking.md`.

This document is an overview and execution order; avoid duplicating detailed requirements here.

---

## 1) Build & Compilation

Opportunities (Release builds by default; see Build doc for specifics):

- Whole‑/Cross‑Module Optimization
  - Enable CMO for Release; validate across‑module inlining gains.

- Library Evolution Toggle
  - Prefer evolution OFF for perf builds; optionally maintain an evolution‑ON product and measure deltas.

- Link‑Time Optimization & Dead Stripping
  - Use dead‑strip flags; consider LTO for C targets when present.

- Dual Release Presets
  - Provide scripts/presets for speed‑first vs size‑first builds used by benches/CI.

- Arch‑specific Builds (when C kernels arrive)
  - arm64: NEON/SDOT/UDOT paths; x86_64: AVX2/AVX‑512; CPUID‑gated dispatch.

- Package Flags and Feature Gates
  - Centralize flags: `VC_ENABLE_UNDERSCORED`, `VC_ENABLE_FFI`, `VC_USE_C_KERNELS`. Provide a single perf preset for benches.

- Codegen Hygiene
  - Ensure Release `-O` benches without `-enable-testing`; audit inlinability for tiny cross‑module glue.

Metrics/Validation:
  - Build time, binary size, benches (ops/s). Maintain A/B with and without CMO and feature flags.

---

## 2) Concurrency & Scheduling

Goals: increase parallel efficiency, reduce overhead/oversubscription, improve cache locality. See Concurrency doc for details and instrumentation.

- Adaptive Chunking (Auto‑Tuned)
  - Extend `CPUComputeProvider` to dynamically compute `minChunk` from micro‑probes (current heuristic + feedback).
  - Persist per‑dim/per‑kernel calibration; adjust at runtime if N distribution changes.

- Coarse Task Grouping
  - Bound number of tasks ≈ cores; avoid per‑item tasks. Continue chunked loops, verify chunk size ≥ 256–1024 items.
  - Reduce captures in task closures to minimize ARC traffic.

- Work‑Stealing Friendly Splits
  - Split ranges into equal contiguous chunks first; allow a small “remainder” pool for tail stealing.
  - Keep chunks cache‑contiguous to favor hardware prefetchers.

- Oversubscription Control
  - Add a provider option to cap concurrency (≤ activeProcessorCount) and detect nested parallel regions to avoid fan‑out.

- NUMA/Pinning (opt‑in; Linux)
  - Add experimental pinning/first‑touch for very large datasets (feature flag, off by default).

- Prefetch‑Aware Scheduling
  - For SoA kernels, prefetch the next 1–2 blocks per lane; allow tuning prefetch distance via provider config.

- Deterministic Reduction
  - Keep per‑task local Top‑K buffers, merge deterministically (already present); document K thresholds for when heap vs selection is chosen.

Metrics/Validation:
  - Parallel efficiency (% of ideal), CPU utilization, task count vs N, variance across runs.

---

## 3) API Ergonomics

Goals: make fast paths obvious, reduce hidden work, and enable pre‑planning/preallocation. See API doc for ExecutionReport, thresholds, and buffer APIs.

- Typed Fast Paths
  - Public overloads for `Vector512Optimized`/`768`/`1536` for `findNearest`, `computeDistances`, normalization, etc.; avoid runtime `is` checks.
  - Mark generic fallbacks as less preferred (disfavored overload under a flag).

- Plans (Pre‑compiled Operations)
  - `DistancePlan`: binds dataset layout (AoS/SoA/INT8 SoA), metric, and buffers; reuses norms/SoA transforms.
  - `TopKPlan`: holds k, heap buffers, and selection strategy (heap vs introselect) per K range.

- Dataset Views & Layouts
  - `VectorDataset` describing alignment, layout, and optional cached norms; provides `.asSoA()` materialization.
  - Zero‑copy views where possible; documented ownership/borrowing rules (prepare for Swift 6 borrowing/consuming).

- Buffer/Scratch Management
  - Public API to pass in scratch buffers (avoid realloc); surface sizes via `requiredScratchSize()`.
  - Expose `BufferProvider` tuning (pool sizes, alignment) at API boundary.

- Introspection & Hints
  - Return a small `ExecutionReport` (kernel kind, layout used, parallel mode) when debugging is enabled.
  - User hints: “preferSoA(N≥X)”, “assumeNormalized”, “preferQuantizedWhenRange≤R”.

Documentation/UX:
  - Clear recipes: building a `DistancePlan`, converting to SoA once, reusing buffers, enabling fast paths.

---

## 4) Benchmarking

Goals: broaden coverage, add realism, and wire comparisons to guide changes. See Benchmarking doc for correctness checks and environment guidance.

- Matrix Coverage
  - N ∈ {128, 1k, 10k, 100k}, K ∈ {1, 10, 100}, dim ∈ {512, 768, 1536}, metrics ∈ {L2, Cosine, Dot}.
  - AoS vs SoA; normalized vs non‑normalized; FP32 vs INT8.

- Harness Improvements
  - Warmup iterations, target min time per case, per‑sample stats (already present; expand defaults per case).
  - Optionally flush caches (large dummy sweep) between runs for cold/warm comparisons.

- Baselines & Trend Tracking
  - Keep baselines per host profile; store hardware/environment metadata next to results.
  - Add binary size measurements for each run (sum of built products used by benches).

- A/B Switches
  - Toggle flags in the bench CLI: `--use-underscored`, `--use-ckernels`, `--prefer-soa`, `--release-size`.
  - Report deltas vs baseline (%; warn on regressions beyond threshold).

- Real‑world Traces
  - Replay workloads: skewed N and K distributions, repeated queries (cache reuse), pre‑normalized embeddings.

- Perf Counters (local developer tooling)
  - Optional `perf stat` / Instruments counters: cycles, L1/L2 misses, branch misses, to validate blocking/prefetch.

Outputs:
  - JSON results (existing), delta reports (median/mean/p95), and a short markdown summary per run.

---

## Execution Order (Suggested)

1. Bench matrix + flags (A/B infra) → data first.
2. Concurrency tuning (adaptive chunks, oversubscription control).
3. API ergonomics (typed overloads, DistancePlan/TopKPlan, buffer APIs).
4. Build knobs (CMO, dual configs, dead_strip) and measure code size vs perf.

## Guardrails

- All changes behind flags where risky; default to conservative behavior.
- Keep diffs small and reversible; benchmark every step.
- Document expectations and roll‑back criteria in the Optimization Plan.
