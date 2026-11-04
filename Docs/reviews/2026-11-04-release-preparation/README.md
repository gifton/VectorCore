# Release preparation batch — 2026-11-04

The three authorized tracks are implemented, independently reviewed, integrated and locally validated on `codex/audit-top-six-2026-10-27`. All earlier commits remain ancestors. The branch and evidence remain local; no main-branch merge, push, tag or release was performed.

Final validated implementation: `62780666be88554ca2859e2e92cd5d3ad6666dba`. Full Swift suites and external D02 consumers ran at `60a87a11b58c28064626fc48023855c4b9042945`; all **387 tracked package inputs** under Sources, Tests, Benchmarks and Package.swift are byte-identical at the final revision. The final wave changes consumer fixtures, Python policies and documentation. Its affected checks ran again. The later record-only commit adds this report, validation summary, manifest and completed plan.

## Delivered

| Track | Result | Evidence |
| --- | --- | --- |
| Imported Collection default | Explicit `_read` preserves the inherited public subscript and normal build flags while avoiding the tested Release-module importer crash. Concrete, generic and existential consumers compile, link and run. | Baseline inherited failures and explicit controls; qualified eight-case matrices; four Xcode compile-only controls; [compiler scope](compiler/README.md) |
| AB3 automatic row scheduling | All three consumers convert Float-element hints to row minima using positive overflow-safe arithmetic. Existing worker bounds, ordering, cancellation, factory errors, shape validation and nested serialization remain exercised. | Nine Debug/Release red observations used one worker; final wide-row observations use 16 on this host; 26 focused tests per configuration and full suites |
| Metadata subprocess lifetime | A private guardian arms an independent same-group timer before starting the command. Cleanup survives observer/guardian loss under the stated ownership contract; bounded output/status handling, literal arguments and environment restoration are tested. | Real baseline survivor failures, loader-variable red/green, observer/guardian-death controls, and seven focused metadata tests per configuration |

The first metadata review found ignored SIGINT/SIGQUIT from a background target launch. Foreground target execution plus explicit INT/QUIT/TERM controls repaired it; scoped re-review approved. The broad final review requested inherited-default existential coverage, now exercised through actual opened existential values. Both reviews and the final scoped re-review have no open Critical/Important findings.

## Integrated validation

| Check | Result | Whole command time |
| --- | --- | --- |
| Full Debug | Swift Testing: 1757 tests / 265 suites reported, 18 skip events. XCTest: 207 reported / 28 skipped / 0 failures. | 392.74s including build |
| Full Release | Swift Testing: 1757 tests / 265 suites reported, 18 skip events. XCTest: 207 reported / 28 skipped / 0 failures. | 443.63s including build |
| Full Python discovery, final source | 592 tests passed | 256.38s |
| Required consumer smoke, final source | 8 documentation programs and 21 contracts passed, including inherited/explicit witnesses with existential reads | 85.19s |
| D02 public API consumers, Debug and Release | Ten fixtures, 61 bindings, and 77-member symbolgraph/scan/TSV/dispatch-row agreement in each configuration | Recorded separately |
| Claim ledger / GitHub / whitespace | 0 claim problems; nine YAML/template files and workflow policies pass; no whitespace errors | Recorded separately |
| SwiftLint 0.65.1 | 5483 warnings, 0 errors; warning count unchanged from prior batch | Recorded separately |

Counts are reported in their framework's terms; XCTest totals include skipped cases and Swift Testing skip events are recorded separately. Existing native-build, Swift/Accelerate and lint warnings remain visible. Extended and strict-performance test gates remain disabled as in the baseline. No new skip, known-issue masking, broad policy exemption or optimization suppression was added.

The first full Python run found two integration-policy gaps: an unregistered metadata liveness assertion and the new integer width reduction missing from the expected census. The final wave adds an exact one-occurrence liveness allowance with negative controls and the exact Int census entry with Int/Float controls. Full Python then passed. The original failures and the intermediate corpus-order test-development failure are retained.

Qualification uses Swift.org 6.4.2 frontend SHA256 `956438a61893525d7ce28c21b6634093f15b0c4bae7fe6eb7d98fba0c1c5e301`, macOS27.2 SDK, on an M3 Max arm64 host running macOS27.0.1. Full commands, source snapshots, compiler/SDK identities, selected environment, deadlines, exits and output hashes are in the command records. Task-local tool adapters select that compiler for D02 and the qualified native backend/SDK for the unchanged required consumer runner; no global toolchain setting or shipping Package flag changed. Xcode compiler probes are comparison-only.

## Scoped Release measurement

The isolated AB3 comparison used identical fixtures/compiler, two warmups per block, alternating base/fixed order and fourteen samples per variant for 18 workloads. All paired checksums matched; peak resident memory is retained with the raw samples.

| Workload | Observed base/fixed median ratio |
| --- | --- |
| Map, 1000–4096 rows × 768 | 3.27–3.83× |
| Normalize, 1000–4096 rows × 768 | 5.00–6.15× |
| Distance matrix, 256/1000 rows × 768 | 3.72–4.88× |

Small cases did not uniformly improve: 1000×4 normalization measured 0.2853ms before and 0.3041ms after, about 6.6% slower; map at that boundary was about 1.4% slower. This short single-host run completed after worker builds/tests stopped, but the host snapshot recorded PerfPowerService at 195.8% CPU and ordinary GUI activity. It is a scoped comparison, not controlled whole-library performance certification or allocation-rate profiling. Equal-row triangular partitions still have unequal costs; that separate balancing issue remains.

## Remaining boundaries and next qualifications

- Actual Swift 6.0, oldest supported OS runtime, other supported platforms/architectures, real sanitizer runtime loading and hosted CI remain unqualified by this batch. The package tools/language floor stays 6.0. Source-consumer success does not establish precompiled-client ABI compatibility.
- Metadata cleanup covers descendants that remain in the owned group and do not attack the guardian/timer. Arbitrary escaped groups, hostile signal/credential changes and OS scheduling stalls are excluded. Foreign reapers/ignored SIGCHLD/SA_NOCLDWAIT remain an unverified host contract; a reaper thread can outlive the call if the OS never lets the child terminate. Shell-owned environment bookkeeping is not byte-identical direct-exec behavior.
- Existing test-selector overmatching, foreign-reaper/waitid-fallback limits in the separate test launcher, and previously documented trap-provenance limits were not changed.
- Deferred Minor review items: explicit ceiling/minimum-ragged-width policy-discriminating assertions and inherited warning cleanup. The current arithmetic is directly derivable from the code; all three unit-mismatch call sites have behavioral regression coverage.
- Previously recorded benchmark-reporting/mandatory comparator execution and documentation-reference/count cleanup remain separate follow-up work.

## Execution decisions

- The three authorized tracks ran concurrently in isolated native worktrees. Potential cost: integration rework; task reviews, the broad review and combined checks addressed that risk.
- Workers used the retained clean baseline and focused red/green checks; the coordinator ran full combined Debug/Release suites. Potential cost: discovering integration gaps later; the two actual policy gaps were retained and repaired.
- All old/new worktrees and evidence are preserved. Cost: local disk usage.
- Metadata requires an independently active deadline after supervisor loss. Cost: private guardian complexity and explicit OS/ownership assumptions, tested and documented above.

## Evidence and history

Frozen archive: `/Users/goftin/dev/projects/VectorCore-release-prep-2026-11-04-evidence`. `evidence-manifest.json` lists retained payload hashes and sizes, including task/review reports, original failures and controls, source snapshots, compiler/importer artifacts, lifecycle captures, final gates and paired measurement. Bulk build/module caches and compiler/SDK payloads are excluded. Previous frozen archives remain untouched.

The [completed plan](../../superpowers/plans/2026-11-04-release-preparation.md) and [validation summary](validation-summary.json) identify the exact source/check mapping. The frozen progress ledger records review rulings and completion conditions. No previously completed repair batch was redispatched.
