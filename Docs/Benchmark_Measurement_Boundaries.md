# Benchmark measurement boundaries

Every benchmark case in `vectorcore-bench` carries a structured **measurement boundary**: a record of exactly what one timed
invocation does. The boundary is part of the case's identity in JSON, pretty and CSV output, in within-run A/B pairing and in
cross-run comparison. This document describes the record, the opt-in `boundary` suite and what is deliberately not claimed.

The boundary describes the benchmark body as written. It is not proof of what the optimizer emitted, and equal boundaries mean
equal declared work, not equal speed.

## The record

Schema 3 reports (`metadata.schemaVersion` 3, `measurementBoundaryVersion` `V04-boundaries-1`, `fixtureVersion` `V02-inputs-1`)
carry `results[].boundary` (version `V04-boundary-1`) with:

| Field | Meaning |
|---|---|
| `timingScope` | `reused-input`: inputs and any route preparation are built before timing and reused. `setup-inclusive`: every invocation also performs the route's preparation, creates its output and consumes it. `consumer-only`: only the output consumer runs, on a precomputed output. `fixture-injected`: test-injected results, never paired or compared. |
| `operation`, `route` | What is computed and which API or kernel does it, including whether a provider schedules it. |
| `workUnit`, `unitsPerInvocation` | The declared unit (`candidate`, `produced-entry`, `element`, `invocation`) and how many one invocation processes. The harness must report the same count or the case fails (`testResultNameAndUnitContractsFailExplicitly`). |
| `inputRepresentation`, `outputRepresentation` | For example `aos-fp32:[Vector512Optimized]`, `soa-fp32:SoA<...>`, `scores:Float32[N] (caller-owned, reused)`. |
| `fixture` | Fixture id and version, dimension, count, decimal seeds, value generator and the preparation applied before timing. |
| `inclusions` | Ten flags: `inputConstruction`, `inputPreparation` (done by the benchmark body for the route), `routeInternalPacking` (done inside the public API call), `callerOutputAllocation`, `routeInternalAllocation` (provider, API or body intermediates, recorded separately), `providerScheduling`, `providerObservation`, `kernelWork`, `outputConsumption`, `correctnessValidation`. Each is `included`, `excluded`, `notApplicable` or `unverified`. |
| `consumer` | Which observation of the produced result keeps the work alive (below). |
| `notes`, `comparison`, `defaultEligible` | Free-text qualifications, declared A/B membership, and (boundary suite only) the outcome of the existing default-eligibility predicate. |

`unverified` means not established for this case ID (no applicable measurement, or a measurement that cannot be applied, e.g. the
pair constraint: both sides of a declared A/B pair carry the same inclusions). It is a declared gap, not a claim either way. Values
are set per case ID, so one measured case does not resolve its neighbours.

### Measured resolutions

`Boundaries.AllocationProbe.resolutions` sets `routeInternalAllocation` for 28 exact case IDs, all at D = 512: 21 `notApplicable`
and 7 `included`. The values come from allocation events counted in each case's timed window at fixed repeats of 16 and of 32, on
one Release arm64 build (macOS 27.0.1, V04 phase D, 2026-10-04). Every other case ID keeps its family's value. That includes the
other widths, the unprobed N and the generic `euclidean2_vs_euclidean` pair at D = 512, N = 4096: its squared side was measured, but
the pair constraint keeps both sides `unverified`.

Each resolved case carries one note that states its value and observation and names its record,
`V04/phaseD/D4/<set>/manifest.json` (`<set>` is `primary`, `unverified-a` or `unverified-b`). The records are kept outside this
repository, in the stabilization recovery directory, so a reader of a report cannot open them from the checkout. The claim ledger
records the probe notes as unverified contracts (entries V04-C-0692 to V04-C-0713), and each entry whose note cites a record carries
that record's sha256. `testAllocationFlagsDoNotOverclaim` pins the resolved IDs and values, the unprobed neighbours, and each note's
record set and observation against a hand-written table; it reads no record.

A resolved value applies to that case ID in every build configuration and on every platform, but it was measured only on Release
arm64: a Debug or x86_64 report carries the measured value without a measurement of its own. A later library change can make a value
stale without failing a test.

### Consumers

A timed body must consume the numerical result. `u64-bitfold-sink-v1` folds the bit pattern of every produced Float into a UInt64
(`OutputFold`) and sinks that scalar; a change to any single element changes the fold (the one-step argument is in the `OutputFold` doc comment: the step map is a bijection because the FNV prime is odd; checked by `BenchmarkOutputFoldContractTests` and `testCompanionFoldPerturbationAndFixtureAgreementAllWidths`). It is an anti-elision observation, not a
numerical oracle and not a collision-free fingerprint. Other labels: `scalar-sink-v1` (a scalar result), `provider-float-sum-observe-sink-v1`
(the in-range Float sum observed through the provider reduction), `float-sum-all-outputs-sink-v1` (a Float sum over all outputs),
`error-outcome-sink-v1` (the failure outcome is the result) and `escaped-pointer-retention-v1` (memory micro-cases, where retention
rests on the escaped allocator pointer and the sink sees at most one element).

### Statistics

`per-sample-average-ns-per-operation`: for each sample, total nanoseconds over the invocations in that sample, divided by the
invocations and by `unitsPerInvocation`. Per-operation samples, median, mean, nearest-rank p90, sample standard deviation and RSD
are reported next to the V02 per-invocation fields (`per-sample-average-ns-per-invocation`) and the raw samples. Nothing is
subtracted: a consumer-only record is reported next to the route, never deducted from it.

## A/B pairing

Pairs form only from declared `boundary.comparison` membership and only when every boundary field outside the family's allowed
differences is equal. Two families exist: `euclidean2_vs_euclidean` (compares inclusion of sqrt; the outputs differ numerically) and
`cosine_prenorm_vs_fused` (compares prepared-input strategies; preNorm preparation is excluded from timing, so this is not an
end-to-end strategy-speed claim). Members of different timing scopes never pair. The `boundary` suite declares no comparisons.

## The opt-in `boundary` suite

`vectorcore-bench --suites boundary` is never part of the default suites, is not selected by `--ab-only`, and ignores `--prefer-soa`
and `--use-mixed-precision` (those control default batch eligibility only). For squared Euclidean distance, per width in
{512, 768, 1536} and per `--batch-ns` count N, it provides:

| ID (`batch.euclidean2.<D>.N<n>.` +) | Scope | Timed body |
|---|---|---|
| `aos-fp32-reused` | reused-input | `BatchKernels.range_euclid2_D` into a caller-owned reusable buffer, then the fold. Direct kernel: no provider scheduling or observation. |
| `aos-fp32-setup` | setup-inclusive | allocate `[Float](N)`, the same kernel, the fold. |
| `soa-fp32-reused` | reused-input | `BatchKernels_SoA.euclid2_D(query:soa:out:)`; the SoA is built outside timing. |
| `soa-fp32-setup` | setup-inclusive | `BatchKernels_SoA.batchEuclideanSquaredD(query:candidates:)`: the public API that transposes, allocates and returns the output. An API-call measurement. |
| `aos-fp16-reused` | reused-input | `MixedPrecisionKernels.range_euclid2_mixed_D`; candidates converted outside timing. |
| `aos-fp16-setup` | setup-inclusive | `convertToFP16_D(candidates)`, a fresh output, the kernel, the fold (the query stays FP32). |
| `soa-fp16-reused`, `soa-fp16-setup` | n/a | **unavailable**: see below. |
| `consumer-only` | consumer-only | the fold on a precomputed score array. |
| `matrix.euclidean2.<D>.N<n>.consumer-only` | consumer-only | the fold on a precomputed N*N matrix output (N*N produced entries). |

The six route entries (aos-fp32, soa-fp32 and aos-fp16, each reused and setup) start from the same immutable FP32 query and candidates as the default batch cases (both suites build them through `BatchFixtures.makeFixture`, pinned by `testBatchBenchBuildsItsFixtureThroughTheSharedBuilder`), end with every score produced and consumed, and validate against an independent Double reference outside timing. The consumer-only entries fold a precomputed array and have no route of their own. The kernels are the existing ones, called through stored closures; the indirection is the same in the reused and setup
variants. FP16 companions report the measured error on that fixture; equivalent requested distances do not imply bitwise-equivalent
arithmetic. Different scopes never form an automatic speedup pair, and no preparation cost is measured separately and subtracted
to manufacture an end-to-end value.

`defaultEligible` records what `BatchKernels_SoA.shouldUseSoA` and `MixedPrecisionKernels.shouldUseMixedPrecision` return for the
fixture; it is recorded, not enforced, so companions run at any N the route supports.

### Unavailable entries

A selected entry whose route does not exist is reported in `unavailable[]` (and `metadata.unavailableCaseIDs`) with its boundary and
reason. It is never invoked, timed, validated or substituted, and is not counted as tested (`testExcludedAndUnavailableRoutesAreNeverInvoked`).
A selection made up only of unavailable entries exits 1 and prints each entry's reason; otherwise unavailable entries do not change the exit
status (`testOnlyUnavailableSelectionExitsOne`).

SoA-FP16 squared Euclidean is unavailable at every width. Every public SoA-FP16 Euclidean kernel returns sqrt distances:
`MixedPrecisionKernels.batchEuclidean{512,768,1536}`, `batchEuclideanBlocked{512,768,1536}`, the public `batchEuclideanSoA` (its doc
comment says "with `sqrt`") and the deprecated `batchEuclideanSquaredSoA`, whose name is misleading: it returns the non-squared
distance (its own doc comment says so). No SoA-FP16 squared-Euclidean or cosine route exists
(source-derived from those declarations). No sqrt family was added. The misleading API name is routed to the public API disposition task
(D02); this task does not change production code.

### Running

```
vectorcore-bench --suites boundary --dims 512 --batch-ns 257,4096 --repeats 3 --samples 9 --format json --out boundary.json
vectorcore-bench --suites boundary --dims 512,768,1536 --batch-ns 17,65 --filter '*consumer-only' --format csv --out consumers.csv
```

Auto-scaling (the default, no `--repeats`) copes with sub-tick bodies. A single invocation of a consumer-only body (tens of
nanoseconds) can read as zero on a clock that ticks about every 42 ns (the cited Apple Silicon tick, below), so calibration doubles the invocation count until one
calibration interval spans at least 10 clock ticks (cap 2^24 invocations, then an explicit error; an implausibly large interval is also
an explicit error), and recalibrates every sample by the same rule. The loop that tops a sample up to `--min-time` never measures fewer
than 10 ticks at the per-invocation rate observed in the main timed loop. The tick is derived from `mach_timebase_info` (`numer/denom` ns, rounded up, so a timebase of 125/3 gives 42 ns; that Apple
Silicon reports 125/3 is a cited platform fact, not verified by a test here, as `Timing.swift` states),
not probed from the host; auto-scaled reports record the rule and the tick in `metadata.harnessCalibration` (fixed-repeat reports record
"not used"; the recorded rule also states the top-up floor). An explicit `--repeats N` runs exactly N invocations per sample: if that spans no clock tick the case fails with
"Unusable benchmark clock interval" and an actionable message (raise `--repeats` or omit it). Tests: `BenchmarkCalibrationContractTests`.

**Effect on the default suites.** A body whose first single-invocation interval is at least 10 ticks (about 420 ns) calibrates on one
invocation exactly as before. Existing default cases whose first single-invocation interval is shorter now calibrate on more than one
invocation and therefore get different iteration counts than under the previous rule. A steady-state proxy (time per invocation, not the first interval), measured once on one Apple Silicon host and not verified by an
artifact in this repository, put some default cases in the dot, dist, mem and noop families under about 420 ns:
the three optimized dot cases, 29 of the 42 dist cases, mem alloc and pool.acquire, and noop. The other dot, dist and mem cases, and
normalize, matrix and batch at small sizes, were above it. Before this change, bodies of 1 to 9
ticks were calibrated from a one-tick-quantized reading (inaccurate but accepted) and only bodies shorter than one tick could fail.

Selection happens before setup: a case excluded by `--filter` or `--exclude` is never set up or run (the catalog still constructs the case entry and its boundary record). The matrix consumer-only entry uses
`N*N` floats (checked against overflow and the CBLAS Int32 limits), so large `--batch-ns` values for this suite also size that output.

## Comparing reports across runs

`Scripts/compare_benchmarks.swift` requires every matched case to carry a boundary and compares the canonical records for equality.

- A present-but-different boundary (any field, including `timingScope`) exits 2 and is **never** waivable.
- A boundary missing from a legacy report (schema key absent or an integer below 3) is waivable only with `--allow-incompatible`,
  which prints each waived case.
- A schema-3 report with a missing or malformed boundary, a future or non-integer `schemaVersion`, duplicate IDs, invalid timing or
  units, or failures is never waivable.

## Run identity

A report carries no identity of the measured source or of the build. `gitSHA` and `runtimeCheckout` are `git rev-parse HEAD` run in
the benchmark process's working directory, and `swiftVersion` and `runtimeToolchain` are `swiftc -version` resolved through PATH, all
observed at run time: a binary built from another checkout, from uncommitted changes or by another compiler is not detected.
`compare_benchmarks.swift` prints `gitSHA` and `runtimeCheckout` as observations, requires equal `runtimeToolchain` values, and does
not read `swiftVersion`. `binarySizes` holds file sizes, not hashes. A reproducible record pairs the report with an external build
manifest: per-file source hashes, the binary's hash and the build command, as V04 phase D did; `Benchmarks/baselines/README.md` makes
the same distinction between runtime observations and build provenance.

## What is not claimed

No speed ratio, no throughput, no production workload and no RSS budget follow from these records. Boundary flags describe declared
work; they are checked mechanically only for structural validity, unit agreement, body-source fingerprints, consumer vocabulary and
warmup/timed equality (`BenchmarkBoundaryDriftContractTests`, `testWarmupAndTimedBodiesRunTheSameWork`, `testConsumerTokensMatchDeclaredConsumers`, `testIndependentSelectionTableAllWidthsAndCounts`), not for semantic truth after an edit. The matrix consumer-only entries, like the `matrix` suite, have zero references on the diagonal, so the correctness gate holds them to an absolute-error bound derived for the norm-identity route and never divides by a zero reference (`CorrectnessPolicy`, `testMatrixZeroReferenceIsGatedOnAbsoluteError`); the boundary records themselves are unchanged. Memory observations, when taken, are process
snapshots with explicit scope, not object-precise heap attribution.
