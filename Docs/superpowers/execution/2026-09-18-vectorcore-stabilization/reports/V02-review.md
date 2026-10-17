# V02 final independent review

**ACCEPT frozen v7 source and final seal 2 runtime/evidence for the approved V02 scope.** No unresolved implementation or evidence blocker. This concludes the earlier versioned source reviews and `V02-runtime-checkpoint-review-v7.md`; it does not erase their findings or relabel earlier binaries as v7. No builds, benchmark reruns, source edits or commits were performed.

## Seal and source

Independently rehashed all **10,312** canonical index entries, **3,037,414,166 bytes**, with no missing files, size mismatches or hash mismatches. Index SHA256: `7009a6beff5877d2b63d8931f0ae0f061a930fe5476ab1e717b5f1702b45ab82`. Final report SHA256: `d33441a9eb450b386c738d7e370f6238ad88d1841222ede2dda407cb788e01d9`. Runtime association SHA256: `3914102021ca43ca6234f101bbaa827648cf10ac7921e6c92d391dd6b5a9deee`.

Accepted v7 manifest remains `fa2f41062175b7bd857932031e3dba7c83603210e2e7549deb5f2f0a5188af7c`, patch `1444b24662e3b791f9b084ab978fca8ebf4ebba86901ef0b43ad7a2a2d1592ef`: 346 base / 356 final code entries and 33 owned paths. Full source/spec and scoped v3/v5/v6/v7 reviews remain the source disposition. V7's only additional Core behavior-contract change is ABI visibility of three existing internal declarations in two Interop files; their source access, numerical bodies, signatures, defaults and gates are preserved. The source-version bridge accurately keeps the benchmark Debug v1, Release v3, reporting driver v5 and ASan v6 provenance distinct from final v7 products and focused C-on tests.

Seal 2 is additive: independently compared every one of the original 10,307 entries to the new index and found them identical; five records were added. The shared historical draft and later canonical working draft are distinct superseded records with explicit hashes in `V02-seal-v2-history.md`. Neither is treated as final acceptance evidence.

## Actual runtime disposition

The prior checkpoint review independently inspected the completed benchmark, CLI, comparator, CSV, memory-ASan and C-on focused records; this final audit rehashed their sealed bytes and completed the four-cell review.

- Benchmark Debug/Release each report 20 XCTest methods plus one Swift Testing entry. The XCTest process entry and Swift Testing entry are parent no-ops; three dispatchers run 41 actual inherited V01 children. Do not turn those reported entries into 21 independently exercised contracts.
- CLI selection runs 51 bounded scenarios per configuration. The separate V01 CLI driver runs 13 commands per configuration. The 28 provider calls / 14 boundaries are mechanical source inventory, not runtime invocation counts.
- Reporting evidence covers 130 comparator executions, independent CSV parsing and executable-discovery controls, with comparator compilation separate. The 86-row actual malformed-schema/count RED contains 20 false accepts and 66 existing guard/control outcomes. Six real failure-bearing matrix comparisons reject even with metadata overrides. Absent/null legacy observations remain permitted as designed; malformed present observations and fractional counts do not.
- Memory ASan executes two actual XCTest methods, with no Swift Testing coverage credited, plus two bounded direct host/bundle controls. The offline stdout-only parser mistake was corrected using already retained stderr; the original actual child had exited successfully.
- Final v7 C-on Debug and Release each execute **10 XCTest methods**, three Swift Testing function results including one no-op entry, and 100 actual children: 99 named rejections and one valid finite-outlier control. All 200 captures are sealed. Earlier nine-method predictions are superseded, not silently adopted. The inherited Q01 child capture/source/parent association is not an independently recorded per-child argv/PID manifest; this limitation does not become a stronger provenance claim in closure.

Actual hosts and loaded bundles are separately retained and hashed. A same-named dSYM DWARF file is not the loaded test executable. The xctest host, SwiftPM helper, actual ASan runtime and versioned loaded bundle associations remain as inspected in the checkpoint. Expected nonzero comparator/CLI/rejection statuses are intentional outcomes, distinct from setup/compiler failures.

## Final four-cell evidence

Independently inspected all four final Debug/Release × optional-C-off/on cells, all 32 cell command records and both output streams, and rehashed all **1,174** retained cell artifact records plus each cell's **133 linked objects** and actual public consumer. Every command completed without timeout with its expected exit: zero for build/compile/link/routes/valid benchmarks, one for contradictory requested C state. Recorded bounds are 1200 seconds for each build, 120 for derivative compilation/link, and 15 for runtime commands; existing driver cleanup remains bounded.

Each cell has 36 public route records and ten separate direct C-tail records, giving **144 + 40** actual rows. Checked finite actual/reference fields, exact gate-dependent call counts, completion markers, actual metadata, selected result cardinality, and consumer/object identities. C-on eligible equal-scale operations and unequal-scale dot/cosine observe three entries; unequal-scale Euclidean and asymmetric fallbacks observe zero. C-off public routes observe zero; direct C tails deliberately bypass the optional Swift gate. These are real dispatcher calls through the forwarding shim, not metadata-only claims. Eight selected benchmark executions and four mismatched-request rejections are distinct from these rows.

`result.json` carries the four-cell completion marker only after the driver's source-integrity check. The required interface now supplies the concrete V03 handoff. No arbitrary raw-C length, FP32 dormant production route, non-arm64 execution or universal numerical envelope is inferred.

## Preserved qualifications and closure boundary

The actual baseline behavioral failures remain separate from development setup/compiler failures and the discovery/public-consumer/C-on visibility failures. Initial source review missed the consumer's internal-namespace visibility defect; v6's public-wrapper correction and actual subsequent consumer compilation close that finding. V7 closes the subsequently discovered serialized visibility errors with the minimal authorized ABI change.

Dense signed matrix fixtures retain the actual Float-Gram cancellation behavior and unchanged V01 strict thresholds. Correctness statistics are truthful observations; a finite status is not a tolerance pass. The independent binary-lattice full-matrix positive control is supplementary, not a replacement chosen to conceal dense-input failure. No numerical repair or NUM-27 closure is claimed.

Two inherited Debug output-directory failure capture files were overwritten before copying. Their parent assertions and complete parent log remain; separately retained Debug CLI controls exercise the same write-failure contract. Release captures and the other 39 inherited child logs per configuration remain. Acceptance preserves this historical evidence limitation rather than claiming recovery.

EX08/EX09 repairs are source-established ownership/bounded-observation changes with scoped controls, not demonstrated prior corruption or a reproduced indefinite hang. No speed target, production-scale guarantee, process memory performance claim, hosted CI execution, remote branch protection or downstream acceptance follows. V03 still owns required CI wiring and instrumented body manifests; V04 owns remaining benchmark timing/claim attribution. Within these explicit limits, specification, implementation quality, source-version mapping, actual runtime outcomes and sealed evidence support V02 integration.
