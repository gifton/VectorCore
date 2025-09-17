# VectorCore Benchmarks — Phased Implementation Plan

This roadmap delivers a robust, reproducible benchmarking suite as a separate executable target living in this repo, evolving from a minimal foundation to a fully fledged system with CI and regression checks.

## Goals
- Stable, reproducible micro/macro benchmarks in Release.
- Clear separation from library code; no extra deps for VectorCore itself.
- Structured outputs (JSON/CSV) for tracking and regression detection.
- Practical local workflow and CI integration.

## Principles
- Warm up before measuring; prevent dead‑code elimination via black‑holing.
- Preallocate and reuse inputs; avoid allocations in hot loops.
- Measure enough time per case; report robust statistics (median, p90, stddev).
- Prefer explicit dimensions (512/768/1536) with optimized vs generic parity.

---

## Phase 1 — Foundation (Target + Skeleton Harness)
Deliver a minimal, dependency‑free executable with a basic harness.

Scope
- Add executable product `vectorcore-bench` with target `VectorCoreBench` under `Benchmarks/VectorCoreBench` (separate from `Sources/`).
- CLI skeleton in `main.swift` with flags:
  - `--suites`, `--dims`, `--provider`, `--profile short|full`, `--min-time`, `--repeats`, `--format json|csv|pretty`, `--out <path>`.
- Utilities:
  - `Timing.swift` for monotonic timers (ns).
  - `BlackHole.swift` with `@inline(never)` sinks to avoid DCE.
  - `Env.swift` to capture run metadata (OS/CPU/Swift/build/git when available).
- Harness: warmup loop + measurement loop, automatic iteration scaling to hit `--min-time`.

Acceptance
- `swift run -c release vectorcore-bench --help` shows flags.
- Dummy “noop” case runs and prints basic timing with no crashes.

---

## Phase 2 — Core Microbenchmarks (Ops Parity)
Measure the most important single‑vector operations.

Scope
- Suites: `DotProductBench`, `DistanceBench` (euclidean, cosine, manhattan, dot-distance), `NormalizationBench`.
- Dims: 512, 768, 1536. Implement both optimized and generic forms.
- Inputs: precreate vectors outside timed region; fixed RNG seed for reproducibility.
- Derive GFLOP/s for dot and distance (assume 2 FLOPs/element for dot).

Acceptance
- Runs produce times for each op × dim × variant (optimized/generic).
- Results black‑holed; no allocations in hot loop.

---

## Phase 3 — Batch + Provider Scaling
Quantify concurrency scaling and batch throughput.

Scope
- `BatchBench`: distance over candidate sets N ∈ {100, 1000, 10_000}.
- Run with `CPUComputeProvider.sequential|parallel|automatic`.
- Metrics: ns/op per candidate and vectors/s throughput.

Acceptance
- Output shows scaling across provider modes and batch sizes.

---

## Phase 4 — Output & Reporting
Produce structured artifacts and human‑friendly console output.

Scope
- Writers: `JSONWriter.swift`, `CSVWriter.swift`, and pretty table printer.
- `.bench/` directory for run artifacts (git‑ignored).
- JSON schema: run metadata + list of cases {name, params, stats, throughput}.

Acceptance
- `--format json --out .bench/results.json` writes valid JSON; same for CSV.
- Pretty console view summarizes key results.

---

## Phase 5 — CI Integration
Run benchmarks on demand and upload artifacts.

Scope
- Update `.github/workflows/benchmarks.yml` to:
  - Build Release
  - Run: `swift run -c release vectorcore-bench --profile full --format json --out .bench/results.json`
  - Upload `.bench/` artifacts
- Keep `workflow_dispatch` (and optionally nightly schedule) to avoid PR noise.

Acceptance
- Manual workflow successfully runs and uploads artifacts on macOS runners.

---

## Phase 6 — Baselines & Regression Checks (Optional)
Track performance and fail on significant regressions.

Scope
- Store blessed baselines under `Benchmarks/baselines/<host>.json`.
- Add comparison script (`Scripts/compare_benchmarks.swift` or shell) to compute deltas vs baseline.
- Configurable thresholds (e.g., >10% slower on key ops → failure).

Acceptance
- CI step compares and clearly reports regressions; can be toggled on/off.

---

## Phase 7 — Extended Coverage
Deeper system characteristics and memory behavior.

Scope
- `MemoryBench`: `AlignedMemory` alloc/free, `MemoryPool` reuse hot loops, `SIMDMemoryUtilities` copy/fill.
- Serialization encode/decode throughput for vectors.
- Larger dims (2048, 4096) to probe cache/bandwidth effects.
- Platform/Accelerate providers if applicable.

Acceptance
- Extended suites run with stable results; no excessive variance.

---

## Phase 8 — Methodology Hardening
Improve statistical stability and guidance.

Scope
- Median‑of‑means or trimmed mean; report mean/median/p90/stddev.
- Better warmup heuristics (run until variance stabilizes or min warmup time reached).
- Developer guidance: DVFS, background load, power settings; pinning hints.

Acceptance
- Repeat runs show reduced variance; documentation reflects best practices.

---

## Phase 9 — Optional: Adopt `swift-benchmark`
If desired, integrate a mature microbenchmark framework (dev‑only).

Scope
- Add `swift-benchmark` dependency to the executable target only.
- Map suites to its API; retain JSON/CSV export compatibility.
- Keep the main library dependency‑free.

Acceptance
- Identical or improved stability with less harness code to maintain.

---

## Phase 10 — Documentation & Publishing
Make it easy to run, interpret, and evolve benchmarks.

Scope
- `BENCHMARKS.md` usage guide: flags, profiles, adding new cases, interpreting metrics.
- Optional badges or generated summaries; version results by git SHA and date.

Acceptance
- New contributors can add a benchmark in minutes and run locally/CI.

---

## Deliverables Checklist
- Executable target + CLI skeleton
- Timing + blackhole utilities
- Harness + stats
- Env/metadata capture
- JSON/CSV writers + pretty print
- Core suites: dot, distance, normalization
- Batch + provider scaling
- `.bench/` artifacts + `.gitignore` update
- CI workflow update
- Baseline comparison (optional)
- Extended suites
- Documentation

## Suggested Target Layout
```
Benchmarks/
  VectorCoreBench/
    main.swift
    Core/
      BenchmarkHarness.swift
    Utils/
      Timing.swift
      BlackHole.swift
      Env.swift
    Formatters/
      JSONWriter.swift
      CSVWriter.swift
    Suites/
      DotProductBench.swift
      DistanceBench.swift
      NormalizationBench.swift
      BatchBench.swift
      MemoryBench.swift
  baselines/               # optional
  BENCHMARKS.md            # usage guide (Phase 10)
  BENCHMARKS_PLAN.md       # this plan
```

## Risks & Mitigations
- Variance from system load → document best practices; longer runs; robust stats.
- Dead‑code elimination → centralized `blackHole` and non‑inlinable sinks.
- Allocation overhead in hot loops → preallocate and reuse inputs.
- CI hardware heterogeneity → keep baselines per host profile; avoid gating on PRs.

---

## Milestones (Time‑boxed Example)
- Week 1: Phases 1–2 (foundation + core microbenches)
- Week 2: Phases 3–4 (batch/provider + outputs)
- Week 3: Phase 5 (+ optional 6) CI + baselines
- Week 4: Phases 7–8 extensions + methodology; Phase 10 docs
- Optional: Phase 9 adoption of `swift-benchmark`

