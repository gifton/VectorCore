# VectorCore 0.2.0 — Concurrency & Scheduling

Focus: improve parallel efficiency, cache locality, and predictability under CPU concurrency. This doc is the source for scheduling strategies and their instrumentation.

## Goals
- Achieve near‑linear scaling up to available cores on large batches.
- Avoid oversubscription and reduce scheduling overhead in hot paths.
- Keep results deterministic and stable.

## Opportunities

- Adaptive Chunking (Auto‑Tuned)
  - Extend `CPUComputeProvider` with micro‑probes per dim/kernel to compute `minChunk` dynamically.
  - Persist calibration per process; allow runtime adjustments based on observed throughput.

- Coarse Task Grouping
  - Bound tasks to ≈ number of cores; chunk size ≥ 256–1024 items to reduce task overhead.
  - Minimize closure captures and ARC activity inside loops.

- Work‑Stealing Friendly Splits
  - Split into contiguous ranges; small remainder pool for tail work.
  - Maintain cache‑friendly contiguous access within each chunk.

- Oversubscription Control
  - Provider option to cap concurrency and detect nested parallel regions; fallback to sequential inside nested regions.
  - Emit a debug signal (via `ExecutionReport` or logging when `VC_TRACE=1`) when a nested parallel region is detected and throttled.

- NUMA/Pinning (opt‑in; Linux)
  - Experimental: pin worker threads and use first‑touch allocation for very large datasets.

- Prefetch‑Aware Scheduling
  - For SoA kernels, prefetch 1–2 blocks ahead per lane; expose prefetch distance in provider config.

- Deterministic Reductions
  - Per‑task local Top‑K buffers; deterministic merges; prefer selection for small K.

## Validation
- Parallel efficiency (% of ideal), CPU utilization, task count vs N, variance across runs.
- Compare sequential vs parallel crossover points per dim/kernel.

## Instrumentation & Debugging
- Track per‑run: tasks created, average chunk size, throttling events, wall‑time spent in scheduling vs kernels.
- Emit these metrics to the bench harness when debugging is enabled; keep overhead negligible when disabled.

## Checklist
- [ ] Add adaptive `minChunk` calibration in provider
- [ ] Cap task count ≈ cores; coarse chunking
- [ ] Optional NUMA/pinning feature flag (off by default)
- [ ] Prefetch distance knob for SoA kernels
- [ ] Document deterministic merge strategy and K thresholds
 - [ ] Add instrumentation for task count/chunk size/throttling and wire to benches
