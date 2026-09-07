# 2026-09-05 finite Top-K measurement

Baseline: pristine archive of VectorCore `ad39ab6`. Final: the owner's uncommitted
Top-K NaN ordering implementation, copied into an independent source tree.
Both executables were built from this same standalone harness with `swift build
-c release`, with separate source and scratch directories. The downstream
`nearestEuclidean512` and `nearestDotProduct512` smoke checks passed in both builds.

Environment: Apple M3 Max, 48 GiB memory, arm64; macOS 26.5.2 (25F84); Apple Swift
6.3.3 (`swiftlang-6.3.3.1.3`, clang `2100.1.1.101`), Xcode's macOS 26.5 SDK.
No other task builds/tests ran during the recorded timed slot. Power configuration
was left unchanged: `pmset` reported battery discharging before measurement (47%)
and AC power with battery still discharging afterward (45%). No thermal or
performance warning was recorded before the runs. These are local observations,
not a controlled power/thermal experiment or a cross-machine performance claim.

n=100,000; fixed seed `0x5EED202609050042`; default `.smallerIndex`; pointer IDs nil.
Each process used five warmup samples, then fifteen measured samples of ten
selections per case. Two processes per revision ran serially in baseline, final,
final, baseline order. The table uses the median of the pooled thirty sample
averages for each case. Positive deltas mean slower final selection.

| Data | API | k | Baseline ms | Final ms | Delta |
|---|---|---:|---:|---:|---:|
| mixed | array | 10 | 0.4471 | 0.4489 | +0.4% |
| mixed | pointer | 10 | 0.3299 | 0.3318 | +0.6% |
| mixed | array | 20,000 | 7.4085 | 7.8470 | +5.9% |
| mixed | pointer | 20,000 | 6.8817 | 7.5591 | +9.8% |
| duplicates | array | 10 | 0.4515 | 0.4500 | -0.3% |
| duplicates | pointer | 10 | 0.3311 | 0.3341 | +0.9% |
| duplicates | array | 20,000 | 4.4386 | 4.6979 | +5.8% |
| duplicates | pointer | 20,000 | 3.1834 | 3.7762 | +18.6% |

All eight baseline/final result checksums match. Array and pointer checksums match
for each fixture and k. Each run's accumulated checksum is
`12278883740353705392`; output consumption includes every selected index and Float
bit pattern. Raw per-process medians, all sample timings, and checksums are in
[`results/`](results/). `compare.py` independently checks their agreement and
regenerates the table.

The heap cases changed by -0.3% to +0.9% in this run; the sort cases were 5.8% to
18.6% slower. No acceptance threshold is defined, and these measurements do not
establish statistical significance or allocation counts.

Measured final source SHA-256 fingerprints (matched to the owner checkout after
measurement):

```text
Operations/TopKSelection.swift
2cd972d88b8fea57fc601a8740edd28c1a99ffbfc761aa64108f882d42d59d94
Operations/Kernels/TopKSelectionKernels.swift
ec9a6770701201b172d94b9ac696e388d45203e3354b43a2e120db204a4c736d
Operations/Operations.swift
39a34dd72b61605e9818ef09ff04c1654b360feba15505bcc4dee6c222a9ec68
Operations/BatchOperations.swift
432884e8981128ee788177dfbbc23017147e21ae02eaa120a54ab599684ef629
```
