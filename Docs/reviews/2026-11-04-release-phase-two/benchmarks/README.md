# Benchmark observations and reporting qualification

Source repair: `73b7e7d0638dd0335dae963613a5a6ee1d090ec6`, based on `04b64694297ae5c32c4936c5367b144c674d705e`. This worker record covers benchmark validation and its direct reporting controls; it does not claim release readiness or performance acceptance.

Dot, successful normalization, zero-vector normalization failure and memory copy now attach out-of-band observations. Dot uses an independent Double product-and-sum; successful normalization compares every coordinate to a Double norm reference, and zero input must produce `invalidOperation`. Copy compares every source/destination bit pattern, rejecting mismatches even with numeric threshold overrides. Its input setup now uses the exact pattern `Float(i % 257 - 128) / 256`, making non-leading/order corruption observable. The measured/warm-up closures are unchanged; validation runs after measurement.

Independent observations exposed three seed-0 false rejections by the old dot empirical limit. The coordinator reviewed and authorized the six derived absolute dot rules in [the mathematical analysis](dot-bound-analysis.md). No other threshold, strictness/override behavior, compatibility rule, public library API or production arithmetic changed. The random-fixture bounds are conservative and miss small errors; exact signed one-coordinate tests at every index cover omitted contributions in all six actual dot routes independently of those bounds.

| Executed check | Observed result |
|---|---|
| Initial focused Release test and real strict seed-0 CLI | RED: missing observations; CLI reports all 21 missing validations |
| New observations with original empirical dot policy | Three retained seed-0 dot failures; comparator refuses that actual report with exit 2 |
| Derived-rule RED control | Missing model fails; existing numerical routes remain unchanged |
| Final focused Release contract/policy/boundary checks | 27 tests, zero failures |
| Full normal-default Release `swift test --skip-build` | Exit 0; XCTest: Minimal 60/27 skipped, Comprehensive 52/1 skipped, Benchmark 106/0 skipped; Swift Testing: 1755 comprehensive and 2 benchmark tests passed |
| Real strict Release repaired-family CLI, seed 0 | 21 valid observations, zero recorded failures, all generic/optimized widths and three copy widths; two compatible runs retained |
| Required actual CLI/comparator driver | Exit 0: 19 comparator invocations plus 301 reporting-table comparator invocations; identity, missing-case, numeric and recorded-failure controls on both actual-product sides |
| Optional adversarial CSV branch on final Swift fixture | `V02_REPORTING_COMPLETE 303`: 301 comparator rows, product-discovery control and independent Python CSV check |
| Python driver durability | 3 tests passed, preserving old negative probes and adding a real 21-case validator positive/eight negative fixtures under normal, `-O` and `-OO` |
| Actual repaired-family self-comparison | Exit 0, 21 cases, no regressions |
| Distinct contended repaired-family comparison | Exit 1, three timing regressions; retained without performance acceptance |

The durability fixture originally failed because its old one-row synthetic report reached the new 21-row stage. The repair isolates that older stage and separately runs genuine positive/negative inputs through the new validator. These are scoped validator controls; the actual CLI/report/comparator runs are the integration evidence. An initial new-driver run also exposed a capture/report filename collision; distinct names fixed it, and the failed attempt remains retained. The adversarial CSV branch demonstrated no existing CSV defect, so its writer/parser and comparator were not changed.

The final product was built with the verified Swift.org 6.4.2 frontend SHA-256 `956438a61893525d7ce28c21b6634093f15b0c4bae7fe6eb7d98fba0c1c5e301`, SDKSettings SHA-256 `82165b15ee21419d4cffaba8895797391b908ac9499ba70f561f3e1bd16758d6`. Final `vectorcore-bench` SHA-256: `f37238e062be8e54c5dff1f026e7dd7d13d159b7de65d085ed58f1e501397dfe`. The final build's 366 Package/Swift input hashes match the committed code. Runtime report checkout/Git labels still name the base checkout observed before commit; they are observations, not build provenance. Retained source manifests and product hashes bind this tested dirty-source build to the repair commit.

Body fingerprints intentionally changed:

| Family | Old | New |
|---|---|---|
| dot | `d3eb21ca41deceeaf59f550873c06d13328ecb138f0ab6b0bc6569d41cfd1992` | `ed203803b2587fb340bb781fbc90ec18c3fe841e01265dbeb0dd4ec20651c36d` |
| normalize | `dbe7948200595fb1609c05817ca0fabf61ed85caa1f87abf0787a60a71f601e3` | `c3f766c8871b36bd00d557d3312f6ebeafe48dc5473232ecfe43d9ed70eb371b` |
| memory | `081006cf564ee7550dbbdb6b81311da51907744bb4fcad18336aeffc80a3fe5a` | `ff58a6f15486b6609c96a03a878fccb2bf4793e1c937610fee5453df07c7d645` |

The validation inclusions changed from `notApplicable` to `excluded`; copy has fixture ID `V04-memory-copy-index-pattern`. Schema/boundary version strings and case IDs stay unchanged, while the actual per-case identity changes intentionally prevent comparison with old reports. A [fresh compatible quiet-host baseline/repeat procedure](quiet-host-procedure.md) is provided; it establishes the new measurement baseline, not a historical regression verdict.

Raw bounded evidence is under `/private/tmp/vectorcore-release-phase-two-2026-11-04/benchmarks/`. Each attempt has argv, selected environment, source/compiler/SDK identities, deadline, exit/timeout and output hashes. `archive-inclusion.json` explicitly names bounded retained files and excludes bulk caches/toolchain/SDK payloads; initial RED and final tested executables/bundles are retained. `product-identities.json`, `source-stability.json`, `dot-analysis.json` and `test-skips.json` identify artifacts, source scope, sensitivity and exact skipped test names. Full intermediate failed logs are preserved. One standalone Swift interpreter fingerprint attempt failed in JIT symbol resolution; compiled execution of the same utility succeeded, and the normal body-fingerprint tests passed.

The runs used one or two fixed repeats and two samples while other qualification work competed. They do not establish timing acceptance or thermal stability. The new arithmetic checks observe separate post-measurement calls on the same fixtures; copy checks the final destination. They do not independently inspect every timed invocation. Remaining allocation/retention qualifications are unchanged. No sanitizer, compiler-support-policy, final combined-source, iOS or hosted qualification claim follows from this worker run. The coordinator must review the mathematics/scope and rerun the relevant combined-source checks before integration acceptance.
