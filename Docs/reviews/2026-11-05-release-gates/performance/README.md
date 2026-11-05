# VectorCore 0.4.0 performance gate investigation — 2026-11-05

**The single formal host-readiness observation completed successfully but rejected admission (`ready=false`). No new benchmark ran, and both diagnostic and release performance acceptance remain false.** A fresh unsanitized native Release product is prepared at carried commit `27a045dd648c01a4ef9c5d8d36c5afc92f36f6fb` with verified Swift.org 6.4.2 and SDK 27.2. The prepared product is native-only; it does not qualify the failing Xcode compiler or the supported platform matrix. No timing was authorized, and the failed admission was not retried.

The parent granted exactly one reviewed five-frame observation after all agent builds/tests/device commands stopped. From **20:07:21.894 to 20:08:21.900 UTC**, every frame reported AC power, nominal OS thermal pressure and low-power mode disabled. Nominal starts were 15 seconds apart, actual lateness was 0.0002–0.0058 seconds, all 15 telemetry commands exited zero, and the four intervals still failed the CPU requirements. A successful observer exit means capture succeeded; it does **not** mean the host qualified.

| Interval | Host idle | PerfPowerServices midpoint CPU | Conservative upper CPU bound |
| --- | ---: | ---: | ---: |
| 0 | 78.274% | 172.488% | 174.198% |
| 1 | 78.615% | 173.365% | 175.101% |
| 2 | 79.142% | 177.045% | 178.830% |
| 3 | 79.164% | 177.806% | 179.608% |

The host-idle requirement was 90%; the per-observed-process limit was 25% of one core. WindowServer also had upper bounds of 45.212–45.289%, and Ghostty exceeded 25% in three intervals (its midpoint estimate was below 25% in two of those). The gate deliberately uses the conservative bound, while midpoint values remain descriptive. All cases and all intervals are retained; no process was stopped. Five frame captures took 0.164–0.176 seconds each; those are wall durations, not a claim of measured CPU overhead. Process births/exits remain explicitly recorded, and endpoint snapshots do not prove absence of short-lived work.

`formal-readiness-grant.json`, `readiness-1/` and `formal-readiness-summary.json` bind the authorization, live commands, full process identities, raw counters, all observations and final rule. The final readiness result SHA-256 is `c22bb1302f956cf9a06728eebace9ffc59b9f727286f909c16b4263d52fb53c0`. This establishes current observed contention under the prespecified criterion, not the unique cause of the historical timing changes.

Future timing needs a documented changed host condition, a newly authorized complete predetermined batch, and complementary direct thermal/power observations before/after each full repeat with source/rule/product identity checks and explicit continuity limits. A separately justified lightweight thermal watch may be appropriate. The inherited runner alone is not fully instrumented for a future acceptance claim; no speculative timing wrapper was added after admission failed.

## Retained measurements: what the data establishes

`analyze_retained.py` reads only the three explicitly named prior reports under `/private/tmp/vectorcore-release-phase-two-2026-11-04/benchmarks-quiet/repeat-{0,1,2}/report.json`. `retained-analysis.json` records their SHA-256 hashes; `retained-case-analysis.csv` preserves all 21 cases and per-repeat descriptive statistics. No excluded private environment diagnostic was read or selected.

All 63 strict correctness records were valid, with 945 raw sample averages. The same case order appears in all three reports. Maximum within-case RSD is 1.419%, 1.827%, 1.061%. The independently recomputed medians exactly equal the reported medians. The original strict comparator outcomes remain repeat1=0 and repeat2=1, with these repeat2 changes versus repeat0:

| Case | Change | Repeat2 range relative to repeat0 |
| --- | ---: | --- |
| dot.512.generic | +5.738% | Every repeat2 sample is above every repeat0 sample |
| dot.768.generic | +5.635% | Every repeat2 sample is above every repeat0 sample |
| normalize.success.768.optimized | +5.219% | Ranges overlap, despite the median change |

The pattern is heterogeneous. Optimized successful normalization shifts +4.981%, +5.219%, +4.288% for dimensions 512/768/1536, while generic successful normalization shifts approximately -0.104%, -0.085%, -0.000%. Copy shifts approximately -1.222%, -1.356%, -1.318%. `dot.1536.optimized` changes -0.037%. Across the 21 cases the median percentage shift is only +0.267%, and 14 cases are slower. Eight cases have completely separated slower repeat2 sample ranges, including several below the 5% rejection threshold.

Therefore the three failing cases cannot be dismissed as an isolated long-tail sample, and the full result is not a uniform multiplicative slowdown. CPU contention, core placement/frequency, process startup state, and operation-specific cache/allocation effects remain hypotheses; neither this sample pattern nor a high-load process establishes which caused an individual shift. No timing was divided by host load or otherwise corrected after the fact.

## Harness and acceptance-design limitations

Read source is the carried checkout `/Users/goftin/dev/projects/VectorCore-fixes-2026-10-27-84798d3`:

- `Benchmarks/VectorCoreBench/AppMain.swift:314` builds plans in supplied suite order; `:335` executes suites serially. Dot and normalization catalogs have a fixed internal case order. Reordering `--dims` does not randomize their order; catalog membership uses `contains`.
- `Benchmarks/VectorCoreBench/Core/BenchmarkHarness.swift:32` computes sample standard deviation. `HarnessConfig.defaultWarmupNs` is 200 ms. The multi-sample loop runs all fifteen samples consecutively for one case after normal warm-up. Each adaptive sample calibrates an invocation count and measures a loop, optionally topping it up. A raw sample is an average over many invocations, not an independently sampled single invocation.
- The reports have sample averages and iteration counts but no per-sample wall timestamps, core placement, frequency, or thermal value. Approximate positions inferred from sample duration cannot identify an exact concurrent process event.
- `Scripts/compare_benchmarks.swift` checks required metadata, boundary identity, validity and selected timing statistic. Its final decision is the deterministic point estimate `(currentMedian / baselineMedian - 1) * 100 > 5`; it does not estimate uncertainty, test stationarity, account for process-level drift or adjust for simultaneous case comparisons. Its correctness/identity rejection behavior must remain unchanged.

Low within-case RSD describes a few contiguous seconds. It does not bound variation between separate processes/minutes. Treating all 945 samples as independent observations would ignore shared process/host states and order confounding. Three complete processes do not establish a stable tail probability for a 5% threshold. Both comparisons share repeat0, so a systematically fast or slow baseline influences them together. Across 21 cases and two comparisons, a family-wide failure is easier to encounter than a single-case failure; a numerical probability would require a distribution and dependence model not established here.

A future experimental design could use predeclared independent process blocks and counterbalanced whole-product order for an old/new compatible comparison, while retaining each strict comparator result. That is a separate design proposal, not permission to reorder/select current cases, pool away a failure, widen thresholds, waive identities, or treat an incompatible historical baseline as a regression reference. This task keeps the existing three complete repeats, seed0, 15 samples, 0.25 s, normal warm-up and 5% median/per-unit comparison if admission is granted.

## Host observability and readiness

`readiness-rule.json` was recorded before any new timing. The parent-approved `readiness-rule-refined.json` retains its hash and adds exact15-second nominal targets with actual starts between target and target+1second, nonoverlapping captures, finite typed metrics, and explicit process-identity scope. The second parent-approved refinement adds the conservative process-rate calculation and CPU-time quantization allowance; `readiness-rule-refined-v1.json` preserves the first refinement and the final rule binds both prior hashes. The refined rule is loaded by the evaluator and bound by SHA-256 in every observer identity/result; it is an **investigative admission criterion**, not as existing project policy. After all compiler/tests and bulk file work stop: collect five frames 15 seconds apart; require nominal current thermal pressure, low-power mode disabled, AC power throughout, at least 90% whole-host idle time in each interval, no comparable observed PID/executable pair above 25% of one core in any interval, and no compiler/test process. Any failed/missing observation invalidates admission. No process is stopped, system power setting changed, service restarted, or superuser power sampler invoked.

`host_observe.py` collects read-only data with a fixed minimal environment and file-backed bounded commands. `host_metrics.py` derives idle percentage from Mach `HOST_CPU_LOAD_INFO` counter deltas and process CPU percentages from cumulative `ps TIME` differences divided by the elapsed time between per-command capture midpoints for descriptive estimates. Admission instead uses `(deltaCPU + 0.02 seconds) / (next capture start - previous capture end)` at the unchanged25% limit. The denominator is the shortest possible interval between the two captured counter observations; a nonpositive denominator rejects. The0.02-second allowance conservatively covers two displayed hundredth-second CPU readings without assuming a rounding direction. This remains conditional on each process counter being sampled within its recorded command span and on comparable process identity. Each ps capture start/end is retained, so sub-command timing skew is visible; it is not an instantaneous reading and centisecond CPU-time quantization remains. It retains full executable names. `ps %CPU` is not used as an interval measurement. There are no process exclusions in the gate. Processes appearing/disappearing between frames are absent from paired process rankings; their CPU usage is still included in whole-host deltas. PID key types are canonicalized before comparison, so JSON string keys cannot hide integer-key pairs. Invalid ps seconds/minutes/day-hour components reject. Changed executables for one observed PID reject admission. New/exited PIDs are retained as unpaired work, not silently called idle. Same-PID/same-executable reuse is a residual limitation. The compiler-name check only detects its listed names; parent-confirmed cessation of agent builds/tests is also required. Neither endpoint snapshots nor process names establish absence of arbitrary short-lived jobs. Aggregate CPU idle does not guarantee a specific core was quiet.

`thermal.swift` reads `ProcessInfo.thermalState.rawValue` and `isLowPowerModeEnabled`. The SDK's `Foundation.framework/Headers/NSProcessInfo.h:457` defines raw value 0 as nominal; `:480` exposes the current thermal state. This is live OS pressure, not temperature, core frequency, or proof of zero throttling. SDK Mach headers define the four CPU tick counters and idle index 2. `pmset -g therm` returns errors on this host; exit0 does not make those values available. `powermetrics --samplers thermal,cpu_power -n 1 -i 1000` rejected ordinary access because it requires superuser; no sudo/elevated power sampler is planned.

Initial observations: Mac15,9; 16 cores, 12 performance and 4 efficiency cores; 48 GiB RAM. A short two-sample `top` probe showed 73.31% idle and PID40610 `PerfPowerService` at 139.4% of one core. `top` itself accounted for 15.4% in that short interval; it is unsuitable as a continuous low-overhead timed-run observer here. The full current name is `/usr/libexec/PerfPowerServices`; the top display truncates it. `perfpower-identity.json` records PID40610, parent1, owner root, elapsed13:30:47 and cumulative593:05.37CPU at capture. This is sustained lifetime activity, not proof of current or historical interval speed. Prior raw records only retained truncated `/usr/libexec/Per`; current matching PID/name is an observation now, not retroactive proof of historical identity. A current full-name snapshot is retained in `observer-preflight/`.

At 19:43:14 UTC, the observer preflight reported thermal nominal, low-power mode false, **Battery Power** (earlier simple probe showed AC). The complete snapshot took 0.169 s, with 0.161 s summed command wall time. This is one preflight overhead observation, not a measured CPU-overhead bound or proof about future frames. It was not the formal five-frame admission. Boundary observations and readiness checks avoid adding the heavier `top` probe to timed work. After the user said they would reconnect AC, the repaired observer independently observed actual **AC Power** at19:50:31UTC, still nominal/lowPower=false; snapshot elapsed0.170s. The intent was not substituted for measurement. Neither snapshot is a formal readiness result. The later formal observation independently rejected readiness; neither earlier preflight was substituted for that result.

## Preparation and verification

`prepared-native/` retains the initial sandboxed attempt: SwiftPM manifest compilation failed because its required user module cache was not writable. `prepared-native-unsandboxed/` is the separately approved, fresh successful preparation. The original preparation helper was reused with only root and candidate configuration changed; commands, selected environment, build logs, before/after 506-input inventories and clean Git checks are retained. The later provenance completion preserved initial JSON records as `.initial`, added actual Clang identities extracted from verbose executed commands, and bound the unchanged prior `resume.py` by hash.

| Identity | SHA-256 |
| --- | --- |
| New benchmark | f0441829cb9d259f6f5f8b3006d9629d250c96f0e7738061eba38704d6311905 |
| Comparator | 4961004c80cf6b336256fc62b69e5a4ae08549ba11af66e31370c1810f72e94a |
| Swift frontend | 956438a61893525d7ce28c21b6634093f15b0c4bae7fe6eb7d98fba0c1c5e301 |
| SDKSettings.json | 82165b15ee21419d4cffaba8895797391b908ac9499ba70f561f3e1bd16758d6 |
| Completed prepared.json | 17ebeceb8a52c71aa64cacc145d1b8f2858f095eeb1cdb75183d19cb5da41168 |

This is a new build at 27a045d; old benchmark hash `b9d356...` and its 6d574f3 metadata/e748 182-shipping-input equivalence remain historical and unchanged. No whole-tree equality is inferred from that old reuse proof. The new executable/comparator are retained separately from build caches. Both actual build frontend and comparator frontend match the expected verified binary; flags include Release, native build system and max2 jobs, with no sanitizer compiler flags/runtime linkage observed in the captured checks.

The task-local observer's five initial controls failed before implementation, and seven rejection mutations plus a nonpositive-time control failed before the readiness evaluator was implemented. `python3 -B -m unittest discover -v` then passed all eight tests, including one live Mach counter schema check. Named controls cover wrap arithmetic, CPU-time formats, full-name parsing, duplicate/bad process rows, missing/nonquiet readiness dimensions and nonpositive time. Independent review then exposed acceptance of empty, nonfinite and malformed telemetry plus unchecked scheduling. `malformed-red.log` and `schedule-red.log` retain those reproductions. The first repair passed16 tests. A further independent review exposed mixed PID-key omission, impossible ps time components and optimistic midpoint rate under long capture spans. `process-interval-red.log` retains the first fixture attempt (one missing-field test error); `process-interval-red-v2.log` records the corrected all-failure RED controls. The final20-test suite passes normal, `-O` and `-OO` modes (`final-controls-*.log`); added class controls cover finite/nonnegative typed metrics, empty inventories, unsigned counter bounds, malformed thermal state, exact nominal schedule, early/late starts, overlapping captures, unchanged counters, executable identity changes and ps-specific time. `refined-green-attempt1.log`, `refined-optimized-green.log` and `refined-doubleoptimized-green.log` retain all outcomes. This verifies the listed cases, not arbitrary observer reliability or whole-project library correctness. Independent final review reran all20 controls in all three Python modes, rechecked mixed-key/long-span reproductions, and verified the original→first→final rule hash chain. `review-association.json` binds that review and verifies every reviewed source hash; final rule SHA-256 is `56fb9a40494081f72e03579be4b36f0c2c89dd7154a17577bbc7b2ae221d80bd`. Approval is for one-shot investigative readiness only, not benchmark timing authorization. No repository implementation or benchmark threshold was changed.

## Retained commands and future procedure

Descriptive analysis (never executes a benchmark):

```sh
python3 /private/tmp/vectorcore-release-gates-2026-11-05/performance/analyze_retained.py
```

The single formal readiness command executed after the explicit parent grant (its output directory now exists and must never be overwritten):

```sh
python3 -B /private/tmp/vectorcore-release-gates-2026-11-05/performance/host_observe.py readiness /private/tmp/vectorcore-release-gates-2026-11-05/performance/readiness-1
```

Readiness failed and no timing command below was executed. These are retained procedure examples only. A future changed-host investigation needs a new named output location and explicit complete-batch authorization after admission and complementary telemetry preparation:

```sh
python3 /private/tmp/vectorcore-release-gates-2026-11-05/performance/prepared-native-unsandboxed/resume.py repeat 0
# Record 30 s cooling and a lightweight boundary observation; no concurrent agent work.
python3 /private/tmp/vectorcore-release-gates-2026-11-05/performance/prepared-native-unsandboxed/resume.py repeat 1
# Record 30 s cooling and a lightweight boundary observation; no concurrent agent work.
python3 /private/tmp/vectorcore-release-gates-2026-11-05/performance/prepared-native-unsandboxed/resume.py repeat 2
python3 /private/tmp/vectorcore-release-gates-2026-11-05/performance/prepared-native-unsandboxed/resume.py compare
```

These commands retain every full outcome and prohibit overwrites. Parent authorization is an explicit operator boundary: the old runner does not consume the new readiness result or mechanically check a permission token. It must not be described as an automatic admission guard. Do not run the inherited `observe` subcommand: its older ps/pmset snapshots lack the new live thermal/interval evidence. Before accepting any batch, the coordinator must finalize a sparse live-thermal observation schedule and account for its overhead; a nominal preflight alone does not prove runtime thermal stability. No automatic timing is built into the readiness tool.

If the current machine cannot admit a quiet window, practical next steps are a dedicated idle session with AC connected and indexing/backup work naturally finished, or a dedicated equivalent test host with matched identities. A reboot after the user saves their work may clear long-lived busy service activity, but would require a fresh documented session and full predetermined measurement batch; it is only an option, not an action taken. User workloads should be closed by the user deliberately, not terminated to force a passing benchmark. Under unchanged busy/battery conditions the defensible result is continued open performance acceptance.


## Frozen evidence

`manifest.json` names the bounded regular-file payload with byte sizes and SHA-256 hashes; `manifest-verification.json` independently rereads those bytes and verifies inventory equality. Build/module caches, Python caches and the two manifest metadata files are excluded. Retained executables, all raw preparation/readiness records, original failure logs, analysis, test controls and review associations are included. The old 945 samples remain in their original report files and are bound by explicit hashes in `retained-analysis.json`; byte-identical copies of all three reports, both strict comparator command/output records and explicit prepared/reuse identities are also retained under `historical-reports/` (12 named files, each checked against its original). They are not rewritten or replaced. `freeze.py` documents the selection and verification operation.

`final-identity-verification.json` rechecks all 506 prepared source inputs, clean carried HEAD, both built and retained executable/comparator hashes, and all 11 independently reviewed readiness inputs after the observation. No new repeat-result directory exists.

This repository copy is accompanied by the [summary](summary.json) and [formal observation summary](formal-readiness-summary.json). Raw artifacts and the reviewed observer scripts are in the [durable evidence directory](/Users/goftin/dev/projects/VectorCore-release-gates-2026-11-05-evidence/performance). Commands above retain the original capture roots; future execution requires fresh paths and verified candidate identities.

**Integration note:** this preparation is bound to clean source `27a045d`. The later report/comment commit changes the carried branch identity, so the retained resume guard must not be bypassed to run against that changed checkout. Prepare a fresh complete batch for the chosen compiler and exact final candidate. The archive preserves evidence and original paths; it is not a relocatable execution package. Historical commands above do not authorize a new timing run or replacement of the rejected readiness record.
