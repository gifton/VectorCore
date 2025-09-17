# Baselines

- Store machine- and env-specific benchmark baselines here.
- Filenames are generated from host metadata: <arch>-<model>-os<version>[-label].json
- Create baseline from a run JSON:
  - swift Scripts/save_baseline.swift --input .bench/full_profile.json --label local
- Compare current run to baseline:
  - swift Scripts/compare_benchmarks.swift --baseline Benchmarks/baselines/<file>.json --current .bench/results.json --metric median --per-unit --threshold 10.0
