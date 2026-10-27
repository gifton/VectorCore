# 2026-10-16 PCA, kNN and UMAP workload envelope (one host)

This is a dated one-host record. It reports what was observed on the host and source tree below. It makes no
promise about other hardware, data, parameters, providers or vector types. Inputs outside the grid are not measured.
[Performance Guide: Measured Workload Envelope](../../Docs/Performance_Guide.md#measured-workload-envelope-one-host)
summarizes it.

**Source tree.** Integration commit `31c5845` with the T19 LAPACK workspace-query fix overlaid, which changed 9 files.
Every row below was measured on that tree in one run of the full grid. That tree's `Sources/` and `Package.swift` are
byte-identical to integration `7bb489a`, where T19 is merged as `4c552d9`, except comments in
`Sources/VectorCoreC/common/vc_lapack_workspace.h`. Its comment-masked token streams are identical.

Two earlier runs are kept as records only and are not used here, because the host was most likely in Low Power Mode:
- a run of `31c5845` without the T19 overlay, in which PCA exact threw `operationFailed` ("sgesdd returned an invalid
  workspace query") at N = 500 for D = 384 and D = 768;
- a partial re-run with the overlay.

In this run every case reproduced the output digest those runs recorded for it, and their memory columns agree with
this run's.

**Environment.**
- Hardware and OS: Apple M3 Max (12 performance and 4 efficiency cores), 48 GiB memory, macOS 27.0.1 (26A434).
- Toolchain: Xcode 27.0 (27A266a), Apple Swift 6.4 (`swiftlang-6.4.0.34.1`).
- Build: a standalone harness package built with `swift build -c release`, depending on the source tree by path.
- Provider: the default linear-algebra provider (LAPACK through Accelerate).
- Power and load:
  - The run was 2026-10-16 20:23-20:51 local time, under `caffeinate -i`.
  - Before and after every timing window, `pmset -g` reported `powermode 2` (High Power Mode) and
    `pmset -g batt` reported 'AC Power'. Every run checked this at its start.
  - Each window started with a 1-minute load average below 8.
  - Two windows saw the load rise above 8 (8.52 and 9.36). Both were re-run once without disturbance; the table
    uses the re-runs, and their digests equal the disturbed runs'.
  - No thermal or performance warning was recorded.
  - The system installer daemon `installd` appeared among the five busiest processes in 74 of 128 window-boundary
    samples, at up to 250% CPU. That can add noise to the times.
  - No other build or test ran during the windows.

These are local observations. They are not a controlled power or thermal experiment.

**Data.**
- Generator: harness-owned SplitMix64 and Box-Muller, seed 20261016.
- Shape: 50 unit-norm Gaussian centers in R^D. Each point is a uniformly chosen center plus 0.6·z/sqrt(D), with
  z ~ N(0, I), then L2-normalized and stored as `DynamicVector`.
- Smaller N at the same D is a prefix of larger N.
- Real text-chunk embeddings can form a different neighbor graph, which changes UMAP layout time.

**Calls.**
- PCA: `PCAModel.fit` followed by batch `transform` of all N vectors, which are the two calls `Operations.pca` makes.
  - K = 50, except K = 49 at N = 50, because K must be below N when centered.
  - Exact uses `PCAConfig(oversampling: Int.max)`. Randomized uses `PCAConfig()`: oversampling 8 and powerIterations 2.
  - At N = 50 the randomized sketch reaches min(N, D), so it takes the exact path and gives the same output digest.
- kNN: `KNNGraph.bruteForce(vectors, neighbors: 15)`.
- UMAP: `Operations.umap(vectors, dimensions: 2, config: UMAPConfig(initialization:))`, with every other option at
  its default. That means 15 neighbors from the brute-force builder, and 500 epochs for N up to 10,000 and 200 above.

**Method.**
- Each run is a fresh process, so 5 runs means 5 separate processes.
- Runs per case: 5 at N up to 5,000, 3 at N = 25,000 and 2 at N = 100,000.
- Time is `ContinuousClock` wall time of the call only. It excludes data generation.
- Peak RSS is `getrusage(RUSAGE_SELF).ru_maxrss`, read right after the call. It includes the process baseline
  (about 7 MiB), the caller-owned input vectors and data generation.
- The sampled malloc peak is the highest `malloc_zone_statistics(nil, ...)` `size_in_use` seen by a thread that polls
  every 1 ms during the call. It is a LOWER BOUND: peaks shorter than the poll interval can be missed. The bracket
  shows the increase over the pre-call value. `max_size_in_use` reads 0 on this host, so it is not used. The poll
  thread runs while the call is timed.
- The analytic payload is a sum of the named logical buffers in the route, one live instance each. It does not model
  lifetimes, and it excludes caller input, LAPACK and BLAS workspaces, array headers and sort scratch.
  - PCA fit:
    - always: 8nd + 8d + 4Kd + 8K;
    - exact path: 4nd + 4nr + 4rd, with r = min(n, d);
    - randomized path: Gaussian, sketch, QR and panel terms in n·L, d·L and L².
  - PCA transform: 4nd + 8nK + 8n.
  - kNN: 4nd + 4n + 8nk + 8(n+1) + 8k + 4Bd + 4Bn, with B = min(n, 256).
  - UMAP: kNN + PCA initialization (K = 2) + 16n + 16E + 32E + 12F + 40R + 16n. Here E = 15n, F is the fused edge
    count below, and R is taken as F.
- Reproducibility: every run of a case produced the same SHA-256 digest of its output bits. Every run at a given
  N and D also generated identical input data.
- A run over 15 minutes would have been stopped. None came close.

| Algorithm | N | D | Runs | Median time (s) [min, max] | PCA fit / transform (s) | Peak RSS (MiB) | Sampled malloc peak (MiB, lower bound) [increase over pre-call] | Analytic payload (MiB) |
|---|---:|---:|---:|---|---|---:|---|---:|
| PCA exact | 50 | 384 | 5 | 0.001 [0.001, 0.001] | 0.001 / 0.000 | 7.3 | 0.3 [+0.1] | 0.5 |
| PCA exact | 50 | 768 | 5 | 0.001 [0.001, 0.001] | 0.001 / 0.000 | 7.7 | 0.4 [+0.2] | 0.9 |
| PCA exact | 50 | 1536 | 5 | 0.003 [0.003, 0.003] | 0.002 / 0.000 | 8.5 | 1.7 [+1.3] | 1.8 |
| PCA exact | 500 | 384 | 5 | 0.013 [0.013, 0.013] | 0.012 / 0.001 | 15.6 | 7.0 [+6.1] | 4.5 |
| PCA exact | 500 | 768 | 5 | 0.025 [0.024, 0.026] | 0.024 / 0.001 | 21.2 | 13.1 [+11.5] | 8.6 |
| PCA exact | 500 | 1536 | 5 | 0.041 [0.040, 0.042] | 0.039 / 0.002 | 29.2 | 20.3 [+17.3] | 16.1 |
| PCA exact | 5,000 | 384 | 5 | 0.057 [0.056, 0.057] | 0.052 / 0.006 | 52.4 | 41.5 [+33.5] | 39.2 |
| PCA exact | 5,000 | 768 | 5 | 0.150 [0.145, 0.151] | 0.140 / 0.010 | 98.3 | 88.3 [+73.0] | 77.6 |
| PCA exact | 5,000 | 1536 | 5 | 0.470 [0.468, 0.476] | 0.450 / 0.021 | 216.0 | 203.0 [+173.0] | 157.7 |
| PCA exact | 25,000 | 384 | 3 | 0.243 [0.242, 0.250] | 0.214 / 0.028 | 212.4 | 191.0 [+151.2] | 193.5 |
| PCA exact | 25,000 | 768 | 3 | 0.542 [0.537, 0.542] | 0.487 / 0.055 | 408.7 | 386.5 [+310.1] | 378.3 |
| PCA exact | 25,000 | 1536 | 3 | 1.393 [1.386, 1.398] | 1.282 / 0.109 | 820.9 | 798.8 [+649.2] | 751.5 |
| PCA exact | 100,000 | 384 | 2 | 0.954 [0.935, 0.973] | 0.830 / 0.124 | 822.4 | 753.3 [+594.5] | 772.0 |
| PCA exact | 100,000 | 768 | 2 | 2.140 [2.136, 2.145] | 1.914 / 0.226 | 1585.3 | 1500.2 [+1194.9] | 1506.2 |
| PCA exact | 100,000 | 1536 | 2 | 5.352 [5.316, 5.388] | 4.899 / 0.453 | 3079.7 | 3005.3 [+2407.0] | 2977.9 |
| PCA randomized | 50 | 384 | 5 | 0.001 [0.001, 0.001] | 0.001 / 0.000 | 7.3 | 0.3 [+0.1] | 0.5 |
| PCA randomized | 50 | 768 | 5 | 0.001 [0.001, 0.002] | 0.001 / 0.000 | 7.7 | 0.4 [+0.2] | 0.9 |
| PCA randomized | 50 | 1536 | 5 | 0.003 [0.003, 0.003] | 0.002 / 0.000 | 8.5 | 1.7 [+1.3] | 1.8 |
| PCA randomized | 500 | 384 | 5 | 0.004 [0.004, 0.004] | 0.003 / 0.001 | 10.6 | 2.7 [+1.8] | 3.2 |
| PCA randomized | 500 | 768 | 5 | 0.006 [0.006, 0.007] | 0.005 / 0.001 | 13.4 | 5.2 [+3.6] | 6.0 |
| PCA randomized | 500 | 1536 | 5 | 0.012 [0.011, 0.012] | 0.009 / 0.002 | 17.8 | 10.1 [+7.0] | 11.6 |
| PCA randomized | 5,000 | 384 | 5 | 0.024 [0.024, 0.025] | 0.019 / 0.006 | 39.0 | 26.6 [+18.6] | 26.7 |
| PCA randomized | 5,000 | 768 | 5 | 0.038 [0.038, 0.038] | 0.028 / 0.010 | 59.4 | 49.6 [+34.2] | 49.3 |
| PCA randomized | 5,000 | 1536 | 5 | 0.070 [0.067, 0.071] | 0.048 / 0.021 | 103.6 | 93.7 [+63.7] | 94.4 |
| PCA randomized | 25,000 | 384 | 3 | 0.119 [0.118, 0.124] | 0.090 / 0.029 | 142.1 | 125.3 [+85.5] | 131.3 |
| PCA randomized | 25,000 | 768 | 3 | 0.186 [0.186, 0.194] | 0.132 / 0.054 | 254.7 | 235.8 [+159.4] | 241.7 |
| PCA randomized | 25,000 | 1536 | 3 | 0.343 [0.336, 0.344] | 0.232 / 0.111 | 472.6 | 455.7 [+306.1] | 462.6 |
| PCA randomized | 100,000 | 384 | 2 | 0.480 [0.479, 0.480] | 0.359 / 0.120 | 546.6 | 502.8 [+344.0] | 523.2 |
| PCA randomized | 100,000 | 768 | 2 | 0.761 [0.759, 0.763] | 0.532 / 0.230 | 996.4 | 942.3 [+637.0] | 963.3 |
| PCA randomized | 100,000 | 1536 | 2 | 1.333 [1.324, 1.342] | 0.897 / 0.436 | 1873.4 | 1827.4 [+1229.2] | 1843.3 |
| kNN bruteForce | 50 | 384 | 5 | 0.000 [0.000, 0.000] | - | 7.0 | 0.3 [+0.1] | 0.2 |
| kNN bruteForce | 50 | 768 | 5 | 0.000 [0.000, 0.000] | - | 7.2 | 0.4 [+0.2] | 0.3 |
| kNN bruteForce | 50 | 1536 | 5 | 0.000 [0.000, 0.000] | - | 7.7 | 0.7 [+0.3] | 0.6 |
| kNN bruteForce | 500 | 384 | 5 | 0.002 [0.002, 0.002] | - | 9.4 | 2.5 [+1.6] | 1.7 |
| kNN bruteForce | 500 | 768 | 5 | 0.002 [0.002, 0.002] | - | 11.5 | 4.3 [+2.7] | 2.8 |
| kNN bruteForce | 500 | 1536 | 5 | 0.003 [0.003, 0.003] | - | 15.0 | 8.1 [+5.0] | 5.0 |
| kNN bruteForce | 5,000 | 384 | 5 | 0.061 [0.058, 0.064] | - | 28.7 | 21.3 [+13.2] | 13.2 |
| kNN bruteForce | 5,000 | 768 | 5 | 0.069 [0.068, 0.071] | - | 44.8 | 36.3 [+20.9] | 20.9 |
| kNN bruteForce | 5,000 | 1536 | 5 | 0.090 [0.089, 0.093] | - | 78.8 | 69.7 [+39.7] | 36.3 |
| kNN bruteForce | 25,000 | 384 | 3 | 2.282 [2.223, 2.284] | - | 114.9 | 105.1 [+65.3] | 64.6 |
| kNN bruteForce | 25,000 | 768 | 3 | 2.452 [2.432, 2.480] | - | 193.3 | 179.5 [+103.1] | 101.6 |
| kNN bruteForce | 25,000 | 1536 | 3 | 2.872 [2.852, 2.897] | - | 342.1 | 328.5 [+178.8] | 175.5 |
| kNN bruteForce | 100,000 | 384 | 2 | 42.1 [40.7, 43.5] | - | 434.2 | 416.1 [+257.3] | 257.1 |
| kNN bruteForce | 100,000 | 768 | 2 | 44.4 [44.3, 44.4] | - | 740.0 | 710.3 [+405.0] | 404.0 |
| kNN bruteForce | 100,000 | 1536 | 2 | 54.8 [53.9, 55.6] | - | 1321.3 | 1298.7 [+700.4] | 697.7 |
| UMAP, PCA init | 50 | 384 | 5 | 0.041 [0.040, 0.042] | - | 7.5 | 0.3 [+0.1] | 0.6 |
| UMAP, PCA init | 50 | 768 | 5 | 0.043 [0.041, 0.043] | - | 7.9 | 0.4 [+0.2] | 1.0 |
| UMAP, PCA init | 50 | 1536 | 5 | 0.040 [0.039, 0.040] | - | 8.5 | 0.7 [+0.3] | 1.9 |
| UMAP, PCA init | 500 | 384 | 5 | 0.361 [0.360, 0.369] | - | 10.6 | 1.7 [+0.8] | 4.8 |
| UMAP, PCA init | 500 | 768 | 5 | 0.363 [0.361, 0.371] | - | 13.8 | 4.6 [+3.0] | 8.2 |
| UMAP, PCA init | 500 | 1536 | 5 | 0.362 [0.360, 0.367] | - | 21.4 | 9.1 [+6.0] | 15.0 |
| UMAP, PCA init | 5,000 | 384 | 5 | 4.392 [4.377, 4.395] | - | 45.8 | 25.4 [+17.4] | 44.9 |
| UMAP, PCA init | 5,000 | 768 | 5 | 4.434 [4.416, 4.448] | - | 68.9 | 50.7 [+35.3] | 74.7 |
| UMAP, PCA init | 5,000 | 1536 | 5 | 4.470 [4.453, 4.477] | - | 116.8 | 90.3 [+60.3] | 134.3 |
| UMAP, PCA init | 25,000 | 384 | 3 | 11.5 [11.4, 11.6] | - | 205.3 | 119.1 [+79.3] | 225.7 |
| UMAP, PCA init | 25,000 | 768 | 3 | 12.0 [11.8, 12.0] | - | 317.5 | 229.0 [+152.6] | 372.8 |
| UMAP, PCA init | 25,000 | 1536 | 3 | 13.0 [12.7, 13.2] | - | 537.8 | 448.8 [+299.1] | 666.9 |
| UMAP, PCA init | 100,000 | 384 | 2 | 79.3 [79.2, 79.3] | - | 809.7 | 472.8 [+314.0] | 908.0 |
| UMAP, PCA init | 100,000 | 768 | 2 | 85.5 [83.0, 87.9] | - | 1262.3 | 912.3 [+607.0] | 1495.2 |
| UMAP, PCA init | 100,000 | 1536 | 2 | 94.5 [93.6, 95.4] | - | 2132.9 | 1791.2 [+1193.0] | 2668.7 |
| UMAP, random init | 5,000 | 768 | 5 | 4.397 [4.366, 4.411] | - | 53.8 | 36.3 [+20.9] | 30.1 |
| UMAP, random init | 100,000 | 768 | 2 | 89.9 [89.8, 90.0] | - | 958.4 | 710.3 [+405.0] | 606.2 |

Fused directed UMAP edges F (twice the unique kNN pairs; the upper bound before zero-weight drops) for
E = 15N stored entries, at every measured N, as used in the UMAP payload column:

| N | D = 384 | D = 768 | D = 1536 |
|---:|---:|---:|---:|
| 50 | 868 | 842 | 864 |
| 500 | 8,788 | 8,718 | 8,810 |
| 5,000 | 111,192 | 111,986 | 112,170 |
| 25,000 | 617,748 | 620,694 | 623,928 |
| 100,000 | 2,600,866 | 2,616,408 | 2,630,770 |

**Observations (this host and data only).**
- kNN time grew faster than N² over the two largest measured intervals only. From N = 5,000 to 25,000 (25 times
  the pairs) it rose 31.8 to 37.6 times, and from 25,000 to 100,000 (16 times) 18.1 to 19.1 times. From 500 to 5,000
  (100 times the pairs) it rose 28.9 to 36.0 times, which is less than quadratic. Do not extrapolate.
- At N = 100,000 and D = 768, the UMAP vector entry took 85.5 s. 44.4 s of that is the brute-force kNN, judged from
  the kNN case. That split is a difference between separate cases, not an in-call phase measurement.

**Evidence outside this repository.**
- Raw per-run JSON, window conditions and the harness sources are in the stabilization recovery evidence
  (`.stabilization-recovery/2026-10-03/L04/`). It is not part of this repository.
- Harness sources (sha256):

    342be8f3d43688808607856d49bf7844de38f48ae76b158903d67b8551343d2a  harness/Package.swift
    ae951fe4311161e38fe3ce2c63bce7f1a3e06ac88966b2a2a325c173f0d76d13  harness/Sources/l04probe/Data.swift
    8bec75a19c518c5eb878fb347a620daf1ec2e43d775f62baf643419b126288e9  harness/Sources/l04probe/main.swift
    7485b2e3da5f4f1a5a135ecd5e46ff1a89e859c1dc5fabb4f3c8f4eac6616a04  harness/Sources/l04probe/Measure.swift
    f042773c0347961a55f262aa6cf264bc35b0aaa100fb4a28e3fb0b40a01cddd0  harness/Sources/l04probe/Payload.swift
    0fc291bdcb5be23667c9b62e8b43d1ca69a22623f8d091058c8199d2325ae7c5  harness/run_grid.py
    81c39b154fb798b7ccf02a19f9034edad995936b0d72238547cf0190c8510ea2  harness/summarize.py
    8dc178a3f294e92f23a278311e64f4a0755bb22cb061ea79a78ae8be6e720e2f  scripts/l04_tables.py
    59e8ac83f41fa33539e68c9ccdd30c1a32a76857f95fcfb5b3ae4c383131283d  scripts/crosscheck.py
