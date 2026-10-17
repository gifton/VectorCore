# Baselines

- Store machine- and env-specific benchmark baselines here.
- Filenames are generated from host metadata: <arch>-<model>-os<version>[-label].json
- Create baseline from a run JSON:
  - swift Scripts/save_baseline.swift --input .bench/full_profile.json --label local
- Compare current run to baseline:
  - swift Scripts/compare_benchmarks.swift --baseline Benchmarks/baselines/<file>.json --current .bench/results.json --metric median --per-unit --threshold 10.0

## V02 report compatibility

The existing baseline JSON files are historical artifacts and remain unchanged. Reports written before V04 use schema 2 and measurement boundary identity `V02-boundaries-1`; they carry no per-case boundary. New reports use schema 3, fixture identity `V02-inputs-1` (unchanged), measurement boundary identity `V04-boundaries-1` and a structured per-case `boundary` (see `Docs/Benchmark_Measurement_Boundaries.md`). Matrix inputs and controls changed, so historical numbers are not an equivalent fixture. Runtime checkout and discovered toolchain observations are not binary build provenance; the four-build driver retains compiler commands and object hashes separately.

The comparator rejects missing or mismatched required identity by default. An explicit `--allow-incompatible` permits an exploratory legacy metadata comparison and prints every mismatch; it does not establish comparability or waive duplicate IDs, invalid timing/units, execution/correctness failures, zero matches, or a missing requested statistic. Use `--metric nsPerOp` only when that is the intended statistic; median/p90/mean never fall back. Missing cases fail by default; `--allow-missing` explicitly reports and permits the unmatched sets.

## V04 boundary identity

Per-case comparison now requires the full boundary (timing scope, operation, route, work unit, representations, fixture identity, inclusion flags, consumer) to be equal. The historical baseline files carry no `schemaVersion` key at all and are treated as legacy (a missing schema key or an integer below 3 is legacy; a present non-integer, sentinel or future value is never waivable). They have no boundary, so comparing one against a schema 3 run is possible only with `--allow-incompatible`, which waives the missing legacy identity and prints every waived case. A present-but-different boundary is never waivable. The matrix case's measured work and consumer changed in V04 (every produced entry is now folded), so its historical numbers are not an equivalent measurement even with the override. Per-case statistics: `per-sample-average-ns-per-operation` is reported next to the V02 per-invocation fields. The opt-in `boundary` suite is not part of any baseline.
