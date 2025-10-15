# VectorCore 0.2.0 — Benchmarking

Focus: broaden coverage, add realism, and wire A/B comparisons to guide changes. This doc is the source of truth for bench matrix, toggles, correctness checks, and environment guidance.

## Goals
- Systematic matrix over N × K × dim × metric × layout.
- Easy A/B of flags (underscored attrs, C kernels, SoA preference, size‑oriented build).
- Persist baselines per host; track trends and binary size.

## Correctness & Accuracy
- Verify numerical correctness for each case against a reference implementation.
- For Top‑K: check recall (exact match of indices and ordering) or require ≥99.9% agreement when quantized/fused paths are enabled; fail benches on drift.
- For distance metrics: validate relative error bounds when fast math or quantization is enabled.
 - Console summaries: the CLI prints a short correctness summary (pass/fail counts, worst maxRel/maxAbs) and reports the thresholds in effect (defaults or CLI overrides).

Default thresholds (relative error)
- FP32 (generic/optimized): ≤ 1e-6
- FP16 (mixed-precision): ≤ 2e-3
- FP16 Cosine: ≤ 3e-3
- Absolute thresholds default to none; rely on relative error. Override via `--max-rel-error` / `--max-abs-error` (or `VC_MAX_REL_ERROR` / `VC_MAX_ABS_ERROR`).

Outputs
- Each case includes a `correctness` block in JSON when supported:
  - `samples`: number of outputs checked (N for batch, 1 for scalar)
  - `maxAbsError`, `meanAbsError`: absolute error vs double‑precision reference
  - `maxRelError`, `meanRelError`: relative error vs reference (denominator clamped with epsilon)
- Initial coverage: batch Euclidean (squared and sqrt) and batch Cosine (fused and pre‑normalized) across 512/768/1536. Pairwise DistanceBench coverage follows next.

## Environment & Reproducibility
- Record host metadata: CPU model, core count, frequency range, memory, OS, toolchain version.
- Control thermal/power where possible; recommend running on AC power and steady state.
- Warmup iterations and minimum wall‑time per case to stabilize measurements.
- Optionally run cold vs warm modes (cache flush via large sweep between runs).

## Opportunities

- Matrix Coverage
  - N ∈ {128, 1k, 10k, 100k}, K ∈ {1, 10, 100}, dim ∈ {512, 768, 1536}, metrics ∈ {L2, Cosine, Dot}.
  - Layout: AoS vs SoA; normalized vs non‑normalized; FP32 vs INT8.

- Roofline Calibration
  - Add microbenches to estimate memory bandwidth and latency on the host.
  - Use these to contextualize speedups and set expectations for memory‑bound kernels.

- Harness Improvements
  - Warmup iterations; target min time per case; per‑sample stats (median/p95).
  - Optional cache flush (large sweep) between cases for cold/warm comparisons.

- Baselines & Trend Tracking
  - Store JSON baselines under `Benchmarks/baselines/<host>.json` with hardware metadata.
  - Record binary sizes for built products used by benches.

- Instrumentation & Reporting
  - Report allocations avoided, copies avoided, task count, average chunk sizes (when available from the provider).
  - Include which fast path/layout/plan was used (via ExecutionReport) in bench outputs.

- A/B Switches
  - CLI toggles: `--prefer-soa`, `--use-mixed-precision`, `--ab`, `--use-underscored`, `--use-ckernels`, `--release-size`.
  - Report deltas vs baseline (%), warn on regressions beyond threshold.
  - Promote env toggles to first‑class flags (keep env fallback):
    - `--prefer-soa` (fallback `VC_SOA=1`), `--use-mixed-precision` (fallback `VC_MIXED_PRECISION=1`), `--ab` (fallback `VC_BATCH_AB`, default on).

- Regression Thresholds
  - Default adoption thresholds: ≥3–5% improvement for underscored attrs; ≥5–10% for C kernels.
  - Fail CI on regressions worse than a configured threshold (e.g., −3%) unless explicitly acknowledged.

- Real‑World Traces
  - Replay skewed workloads: repeated queries, mixed K values, pre‑normalized embeddings.

- Perf Counters (local dev tooling)
  - Optionally sample cycles, L1/L2/LLC misses, branch misses via Instruments/perf.

## Outputs
- JSON results (existing), delta reports (median/mean/p95), short markdown summary per run.
 - File layout for runs: write to `.bench/runs/<device-tag>/<timestamp>_<git-sha>.json` (unified across packages). Include optional `runLabel` in metadata for tester annotations.
  - CLI support: `--run-label "<label>"` (env fallback `VC_RUN_LABEL`). If `--out` is omitted, default path follows the run layout for both JSON and CSV.

## Modes & CLI Flags (VectorCoreBench)
- Modes: `--mode quick|full|smoke` as presets over `--profile`, `--min-time`, `--dims`, and batch sizes.
  - quick: shorter min time (≈0.2s/case), N ∈ {100, 1k}, dims {512, 768, 1536}
  - full: longer min time (≈1.0s/case), N ∈ {100, 1k, 10k}, dims {512, 768, 1536}
  - smoke: a handful of micro/macro cases to validate setup
- Flags: promote env toggles to CLI (`--prefer-soa`, `--use-mixed-precision`, `--ab`) and keep existing A/B flags (`--use-underscored`, `--use-ckernels`, `--release-size`).
- Always emit machine‑readable output when `--format json|csv` is specified; preserve pretty console output.
 - Overrides: `--min-time`, `--dims`, and `--batch-ns` override mode/profile defaults.
- A/B comparisons: JSON includes `abComparisons[]` with batch euclidean2 vs euclidean and cosine preNorm vs fused deltas. Console prints a brief summary.
- A/B-only runs: `--ab-only` limits generation to A/B pairs (batch euclidean vs euclidean2; cosine fused vs preNorm) for fast iteration; also ensures A/B is enabled.

Filtering
- Use `--filter` and `--exclude` to target subsets of cases by name.
  - Default mode is glob: e.g., `--filter 'batch.*.768.*.optimized*'` or `--filter 'dist.cosine.*'`.
  - Switch to regex with `--filter-mode regex` to use full regular expressions.
- Filtering applies to emitted results and A/B summaries.
- Early gating (runtime saving):
  - Batch and Normalization suites avoid running non‑matching cases.
  - Distance suite avoids non‑matching cases (512/768) and skips entire per‑dimension sets when none match (1536 pending per‑case).

## Run Flags in Metadata
- The run metadata includes a `flags` object capturing the resolved toggles:
  - `preferSoA`, `useMixedPrecision`, `abCompare`, `abOnly`, `useUnderscored`, `useCKernels`, `releaseSize`.
- CLI flags override environment; env fallbacks: `VC_SOA`, `VC_MIXED_PRECISION`, `VC_BATCH_AB` (default on), `VC_USE_UNDERSCORED`, `VC_USE_CKERNELS`, `VC_RELEASE_SIZE`.

## Tester Workflow (VectorCore)
- Build Release and run quick mode first; then full when idle and on AC power.
- Save a device‑tagged baseline using `Scripts/save_baseline.swift`; compare with `Scripts/compare_benchmarks.swift`.
- Submit `.bench/runs/...json` (and baseline if created) via the chosen intake (PR/issue/upload).

## Runner & Viewer (Roadmap)
- Aggregator CLI (later): `vector-bench-runner` orchestrates per‑package benches with a unified schema: `--suite core|index|accelerated|all`, `--mode`, shared flags.
- macOS SwiftUI app (VectorBench):
  - Run: select suites/dims/N/mode/flags; start and monitor runs; uses the CLI under the hood initially.
  - Results: list `.bench/runs` by device tag; filter/search; open details.
  - Compare: pick two runs; show deltas; flag regressions over thresholds.
  - Charts: Swift Charts for ns/op (and per‑unit) and basic trend over time per case.
  - Initial scope VectorCore; VectorIndex shows “coming soon” until its bench target ships.
- Static viewer (BenchDash; optional): single HTML that loads one or more run JSONs to visualize and compare without installing an app.

## Checklist
- [ ] Implement matrix cases and CLI toggles
- [ ] Persist host metadata alongside baselines
- [ ] Capture and report binary sizes
- [ ] Add cold/warm run modes and minimum wall‑time per case
- [ ] Add optional perf‑counter sampling for local runs
- [ ] Add correctness checks (Top‑K recall, distance error bounds)
- [ ] Add roofline microbenches and include calibration in summaries
- [ ] Emit instrumentation and ExecutionReport details with results
 - [ ] Add `--mode quick|full|smoke` presets and promote env toggles to CLI (`--prefer-soa`, `--use-mixed-precision`, `--ab`)
 - [ ] Adopt stable run file naming under `.bench/runs/<device>/<timestamp>_<sha>.json`; include optional `runLabel`
 - [ ] Provide tester one‑pager with commands and best practices
 - [ ] Scaffold aggregator CLI (`vector-bench-runner`) once VectorIndexBench is ready
 - [ ] Scaffold VectorBench macOS app (Run/Results/Compare/Charts) reading `.bench/runs` (VectorCore first)
 - [ ] (Optional) Add static HTML viewer (BenchDash) for quick visualization

## Case Naming & IDs
- Deterministic IDs: Case names are stable and encode only parameters and variants, never random seeds. The global `runSeed` influences inputs, not IDs.
- Canonical forms (grammar):
  - dot: `dot.<dim>.<variant>`
  - distance: `dist.<metric>.<dim>.<variant>` where `<metric>` ∈ {euclidean, cosine, manhattan, dot, chebyshev, hamming, minkowski}
  - normalize: `normalize.<status>.<dim>.<variant>` where `<status>` ∈ {success, zeroFail}
  - batch: `batch.<metric>.<dim>.N<count>.<variant>[.<mode>]`
    - `<metric>` ∈ {euclidean, euclidean2, cosine}
    - `<variant>` examples: `generic`, `optimized`, `optimized-fused`, `optimized-preNorm`, `optimized-soa`, `optimized-fp16`
    - `<mode>` ∈ {sequential, parallel, automatic} (optional for SoA/FP16 one-pass runs)
  - memory: `mem.<op>[.<subop>].<dim>` e.g., `mem.alloc.aligned.512`, `mem.copy.768`, `mem.pool.acquire.1536`
- Parsing and round‑trip:
  - `CaseParsing.parse(name:)` extracts `kind`, `metric`, `dim`, `variant`, `provider`, `n`, and `status` where applicable.
  - Batch parsing accepts both 5‑part and 6‑part forms (provider optional).
  - Memory parsing sets `metric` from op, `variant` from sub‑op (if any), and `dim` from the trailing component.
- Invariants:
  - IDs must be unique per parameter combination within a run.
  - Order of components and separators (`.`) are fixed to ensure stable parsing.
  - New variants must adhere to the same structure; add parsing updates alongside any naming changes.

## Phased Implementation Plan

The phases below sequence work so that each step produces immediate value, de‑risks later steps, and builds toward the full roadmap. Each phase lists outcomes and acceptance criteria. Where helpful, dependencies on earlier phases are noted.

1) Foundation: stabilize harness and schema
- Outcomes: benches run reliably in Release; uniform timing model; stable JSON schema with required metadata.
- Tasks: unify warmup + min wall‑time control; seed random data deterministically; add `RunInfo` (git SHA, device tag, toolchain, OS), `ExecutionReport` (fast path/layout), and `binarySizes` to results; ensure deterministic case IDs.
- Acceptance: running `--format json` emits a valid schema including `runInfo`, `executionReport`, and `binarySizes` for every case; Release builds pass locally.

2) Modes and CLI presets
- Outcomes: `--mode quick|full|smoke` presets configure profiles, min‑time, and case selection consistently.
- Tasks: add `--mode` and map to harness parameters; keep explicit flags (`--min-time`, `--dims`, `--N`, `--K`) to override presets; document defaults in `--help`.
- Acceptance: three modes produce different run counts and durations; overrides work; console and JSON agree on case lists.

3) Stable run layout and metadata
- Outcomes: unified run file placement across machines/packages; simple discoverability.
- Tasks: write runs to `.bench/runs/<device-tag>/<timestamp>_<git-sha>.json`; add optional `--run-label`; include host metadata snapshot per run.
- Acceptance: files appear under the specified layout; `runLabel` and device tag are present in JSON; a simple `ls` shows time‑ordered runs.

4) Promote toggles to CLI and A/B readiness
- Outcomes: environment toggles are first‑class; harness can execute A vs B in one invocation.
- Tasks: add `--prefer-soa`, `--use-mixed-precision`, `--ab`, `--use-underscored`, `--use-ckernels`, `--release-size`; when `--ab` is set, run both variants back‑to‑back and emit deltas per case.
- Acceptance: flags reflect in `RunInfo` and results; `--ab` writes delta fields and a short markdown summary with winners/regressions.

5) Correctness and accuracy gates
- Outcomes: benches fail fast on drift; quantified error when fast paths/quantization are used.
- Tasks: add reference implementations; per‑case checks for Top‑K recall and distance error bounds; thresholds configurable via CLI (e.g., `--max-rel-error`, `--min-recall`).
- Acceptance: known good paths pass; intentionally degraded paths fail with clear diagnostics; JSON carries `correctness` section per case.

6) Matrix coverage expansion
- Outcomes: systematic sweep across N × K × dim × metric × layout with filter controls.
- Tasks: implement param generators; add `--filter` (glob/regex) and `--exclude` for quick targeting; ensure IDs encode all params and flags.
- Acceptance: `--mode full` covers the documented matrix; filters reduce the set without schema changes; case IDs are stable.

7) Instrumentation and reporting
- Outcomes: richer insights into where time goes and which fast path executed.
- Tasks: record allocations avoided/copies avoided when available; task counts/chunk sizes from provider; embed `ExecutionReport` details into JSON and markdown summary.
- Acceptance: reports show non‑zero instrumentation where applicable; summaries include path/layout notes per case.

8) Cold/warm runs and timing stability
- Outcomes: ability to compare cache‑cold vs warmed behavior and stabilize measurements.
- Tasks: optional large‑buffer sweep between cases; per‑case warmup until convergence or cap; expose `--cold`, `--warmup-max`, `--min-time`.
- Acceptance: cold mode shows expected slowdowns; warmup reduces first‑run penalties; variance decreases vs Phase 1.

9) Roofline microbenches and calibration
- Outcomes: contextual bandwidth/latency numbers to frame kernel speedups.
- Tasks: add memory bandwidth and latency microbenches; compute peak‑ish FLOPs for arithmetic kernels; include roofline section in run JSON and summaries.
- Acceptance: microbenches run in smoke/quick; JSON includes a `roofline` block; summaries cite memory‑bound vs compute‑bound hints.

10) Baselines, deltas, and thresholds
- Outcomes: simple workflows for saving baselines and diffing runs; configurable regression gates.
- Tasks: finalize `Scripts/save_baseline.swift` and `Scripts/compare_benchmarks.swift`; add CLI `--baseline <path>` to emit deltas inline; support thresholds per suite/metric; optional size regressions gates.
- Acceptance: deltas appear in JSON and markdown; regressions over threshold cause non‑zero exit when requested (CI‑friendly).

11) Tester workflow and packaging
- Outcomes: frictionless way for external testers to contribute results.
- Tasks: add a one‑pager with commands, best practices (AC power, idle), device tagging; provide a convenience script `scripts/run_quick.sh` that sets sensible flags and labels.
- Acceptance: a new tester can produce a valid run file and optional baseline within minutes, following the doc.

12) CI integration (quick mode)
- Outcomes: continuous guardrails without long runtimes.
- Tasks: run `--mode quick --format json` in CI; archive `.bench/runs` as artifacts; optionally compare against last successful baseline; fail on configured regressions.
- Acceptance: CI surfaces regressions with links to artifacts; flakiness remains low due to min‑time and warmup.

13) Viewer and orchestration (VectorBench + aggregator CLI)
- Outcomes: friendlier UX for running and inspecting results; path to multi‑package.
- Tasks: MVP macOS SwiftUI app (Run/Results/Compare/Charts) that shells out to the CLI and reads `.bench/runs`; later, scaffold `vector-bench-runner` to orchestrate multiple packages with a unified schema.
- Acceptance: app lists runs and renders comparisons; aggregator can invoke VectorCore benches with shared flags.

14) Optional: static HTML viewer (BenchDash)
- Outcomes: zero‑install sharing of results.
- Tasks: single‑file HTML that loads one or more JSON run files; basic tables/charts and delta highlighting.
- Acceptance: opening the file locally renders selected runs and comparisons without a server.

Notes on dependencies
- Phases 1–4 establish the CLI surface, file layout, and A/B capability that later phases rely on.
- Correctness (Phase 5) should land before or alongside matrix expansion (Phase 6) to prevent baking in undetected drift.
- CI (Phase 12) relies on thresholds and deltas from Phase 10 and on stable timing from Phases 1–2 and 8.
- Viewer and aggregator (Phase 13) depend on stable schema (Phase 1) and run layout (Phase 3).
