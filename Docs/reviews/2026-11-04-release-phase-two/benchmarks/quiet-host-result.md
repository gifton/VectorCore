# Prepared-product repeat measurements

The prescribed three repeats completed without retries. Every repeat passed strict correctness for all 21 repaired benchmark cases. The strict 5% median/per-unit comparison passed for repeat 1 and failed for repeat 2 with three regressions. **Performance acceptance remains false.** This is a fresh compatible baseline/repeat observation; it does not establish an old-to-new code regression verdict or release readiness.

| Comparison against repeat 0 | Exit | Matched cases | Missing cases | Regressions |
|---|---:|---:|---:|---:|
| Repeat 1 | 0 | 21 | 0 | 0 |
| Repeat 2 | 1 | 21 | 0 | 3 |

| Repeat 2 case | Baseline median ns/unit | Current median ns/unit | Change |
|---|---:|---:|---:|
| `dot.512.generic` | 3005.4644192055766 | 3177.920150060406 | +5.738072617090366% |
| `dot.768.generic` | 4516.662756651243 | 4771.18591457824 | +5.6352039468119575% |
| `normalize.success.768.optimized` | 3958.411778667384 | 4164.985613427468 | +5.218603983379102% |

All runs used the prepared unsanitized Release executable, seed 0, 15 estimated-duration samples per case, minimum time 0.25 seconds, normal warm-up and `--strict-correctness`. No fixed-repeat override was supplied. The actual comparator used `--metric median --per-unit --threshold 5`; no identity/missing/statistic waiver or selected retry was used. There were 63 correct case executions and 945 retained raw sample averages, with iteration and warm-up counts. Within-run maximum RSD was 1.419%, 1.827% and 1.061%; low within-run RSD does not remove the observed between-run shift. Each benchmark process completed in approximately 83.1–83.3 seconds.

The coordinator released the quiet window only after native jobs completed and asked other agents to defer hashing, copying, builds and tests. The existing immutable `RESUME.txt`/`resume.py` procedure was followed. Each repeat had a concurrent observer with seven frames: one before, five during and one after. All 63 primary observer command captures exited zero. Initial cooling elapsed 80.523 seconds; the two explicit cooling waits elapsed 45.816 and 44.403 seconds, including orchestration overhead. Full inter-run gaps, measured from the previous completion marker to the next start marker, were 168.644 and 74.525 seconds. Thus cooling exceeded the planned minimum 30 seconds; no uniform exact spacing is claimed. No unrelated process was stopped.

The host was not idle. Captured one-minute load ranged from 4.85 to 7.00. Nonbenchmark PID 40610, captured with command prefix `/usr/libexec/Per`, appeared in all 21 observer frames and reached 197.3% CPU; PID 69556, prefix `/usr/libexec/spo`, reached 122.3%. The captured `ps` comm fields truncate names, so full process identities are not reconstructed. `pmset` supplied no recorded thermal/performance warning level or CPU power status; that absence does not prove no throttling. Observer overhead is retained and not subtracted. A supplemental process snapshot whose filename says “during repeat 0” was actually taken after that repeat completed and is classified post-run only. An additional post-window process-width probe exited 1 for an unsupported format; its failure is retained, and primary observer evidence remains complete.

## Reviewed product-input scope

The measured product remains bound to clean commit `6d574f3ff122554ae8bec18f039ffa4a6981920e`; every before/after source, compiler, SDK, executable and HEAD check passed during the runs. Report metadata correctly retains `6d` for checkout/Git observations. The independently qualified test candidate is `e7482a962c67dd0b806059b651c9f59eb678a632`. The coordinator approved reuse for exactly 182 shipping inputs: the five shipping module/C/header directories, `Package.swift` and comparator source. Those bytes match immutable e748 raw Git blob IDs/bytes, the prepared source hashes and both clean working trees at the association step. Exact original build/comparator flag vectors and actual executed Swift/Clang tool hashes match; SDK identity is the recorded canonical path plus `SDKSettings.json` digest, not a newly invented whole-SDK digest.

The four changed test-only inputs (one method actor annotation and three capability declaration wraps) are explicitly excluded from shipping scope. Documentation differences are also nonshipping. The two commits are **not globally identical**, and reuse does not relabel the measured product as an e748 build. The external association is retained as `reviewed-input-reuse/result.json`, with the complete input comparison, raw Git blob proof and excluded-input inventory.

| Prepared identity | SHA-256 |
|---|---|
| `vectorcore-bench` | `b9d356a1a2217cfa85d0228ad4f267eec4aed0ed7d0e429665784a69dd7a4fd6` |
| Actual comparator | `4961004c80cf6b336256fc62b69e5a4ae08549ba11af66e31370c1810f72e94a` |
| Executed Swift frontend | `956438a61893525d7ce28c21b6634093f15b0c4bae7fe6eb7d98fba0c1c5e301` |
| SDKSettings | `82165b15ee21419d4cffaba8895797391b908ac9499ba70f561f3e1bd16758d6` |
| Immutable `prepared.json` | `373c7e19ef15eaadc1af2b91599678c4b9a41c941ef9a1a5e44c89283702f594` |

Raw evidence is `/private/tmp/vectorcore-release-phase-two-2026-11-04/benchmarks-quiet/`. It includes actual executable copies, every repeat/comparator/observer capture, all raw samples, cooling/precondition/postcondition records, source/compiler/product identities and the reviewed reuse association. `timing-cases.csv`, `timing-summary.json`, `host-summary.json` and the small `result-summary.json` provide derived views without replacing raw reports. The fresh `final-inclusion.json`/`final-verification.json` unify the evidence and exclude bulk caches/toolchain/SDK payloads. The original preparation manifest and every preparation-listed byte remain unchanged and are verified separately.

This documentation record was added only after timing, observations, comparisons and clean-6d postchecks finished. It changes no shipping code, input, flag, executable or preparation artifact. Self-review checked the exact fixed schedule/flags, raw case/sample counts, actual comparator exits/percentages, scoped source binding, host limitations and preservation of the failed gate. No additional run was selected to replace a failure. Timing completion was promptly reported before other agents resumed their deferred work. Broader compiler/support and release decisions remain separate.
