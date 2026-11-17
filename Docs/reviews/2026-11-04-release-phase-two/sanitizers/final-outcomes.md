# Final native sanitizer execution supplement

Status: **DONE_WITH_CONCERNS**. Clean native source candidate: `e7482a962c67dd0b806059b651c9f59eb678a632`. Posthoc typed-discovery tool revision: `5a3301055f38d4fbf506985e3feedf3da154c7b4`, committed only after the final native sweep and source/tool postchecks. Documentation changes after those points do not change the source identified for the binaries.

## Exact completed scope and limits

The final [README table](README.md) gives the completed counts. Independent discovery contains 1,764 Swift Testing functions and 218 real XCTest methods. Under each sanitizer the complete scope has 1,746 Swift Testing passes / 18 skips and 190 XCTest passes / 28 skips; failed, pending and unknown real selectors are zero. The mandatory drivers pass all 77 ASan and 39 TSan workloads. Actual parent-PID dyld observations verify each broad command's expected host, sanitizer runtime and aggregate bundle, independently of printed paths and child images. All source and six native qualification-tool pre/post checks match.

Every strict no-skips qualification remains **false**. Exact skipped functions are retained under each `*-full/*-scope.json`; argument cases and issues remain in raw event streams. Ordinary helper-function passes do not imply every conditional child scenario or every branch executed. Release test builds use `-O`, `-enable-testing` and the selected `-sanitize=address` / `-sanitize=thread`; these are test binaries, not proof of production-binary equivalence. Only native arm64 macOS 27.0.1 was exercised here.

The original final-v2 TSan Swift Testing command timed out at 1,200.19 seconds: 1,608 passed / 18 skipped / 138 pending, interrupted in `VectorDispatchContractTests/concreteProtocolOnlyMismatchesRejectBeforeBorrow()`. That attempt is not qualified. The coordinator authorized one additional **complete serial** sweep with a 2,400-second whole-host budget based on observed aggregate progress, keeping mandatory/body/child deadlines and diagnostics unchanged. The corrected sweep exits 0 in **1,372.033 seconds**, with all 1,764 independently discovered functions reconciled. It uses a fresh event stream and Q03 directory; no previous outcomes are merged. Successful XCTest is not rerun.

The first additional attempt actually launched the native helper with an incomplete standalone compiler-fixture environment, then was deliberately interrupted. Its original native metadata remains `status=running`, native exit unrecorded, with `Driver interrupted: 15`; the supervisor completed exit -2. `final-v2-thread-preparation/extended-interruption-reconciliation.json` binds termination intent, supervisor metadata and a later exact-owned-PID absence check. It is an interrupted unqualified execution, not merely a pre-launch setup failure. The corrected attempt's selected environment comparison matches the original broad command in every recorded key except the distinct Q03 log directory.

## Runtime warning and posthoc discovery accounting

TSan emits the exact module-map/backtrace warning pair during mandatory bodies, Q03 children, discovery and broad execution. The final explicit-safe-path scan reads 615 retained stdout/stderr/log files: 190 original warning occurrences plus eight copied-log occurrences; its matched records contain no other TSan diagnostic kind. This bounded observation does not establish that every diagnostic would be reliable. Warnings remain visible and limit qualification.

In original final-v2 TSan XCTest discovery, those two stdout warning lines were misclassified as selectors and included in the original selector argv. Thus raw artifacts remain **220 discovered / 2 pending**, while all **218 real methods completed: 190 passed / 28 skipped**. The separately reviewed typed parser classifies the two exact known warnings with stream, line, text, PID and kind, rejects unknown/malformed nonempty stdout, duplicate/ambiguous mappings and inconsistent framework inventories, and prevents diagnostics entering new selection argv. It does not suppress raw diagnostics or change strict qualification.

`sanitizers-discovery-fix/reconciliation-payload.json` is the explicit posthoc reconciliation. It records parser commit/version/SHA, all 18 raw input hashes for v1 ASan/v2 ASan/v2 TSan and exact retained diagnostics. Reconstructed argv is validated but never executed. The original `corpus-control.json` payload/capture basename collision is preserved as a failed evidence attempt; the corrected payload and supervisor use separate names. RED/GREEN capture controls reject metadata/stream alias destinations and verify the detailed payload schema/provenance survives bounded capture. All 31 parser/scope/binding controls pass; no native rerun was made for parser accounting.

Current upstream LLVM [sanitizer_procmaps_mac.cpp](https://github.com/llvm/llvm-project/blob/main/compiler-rt/lib/sanitizer_common/sanitizer_procmaps_mac.cpp#L56-L126) emits this pair when overlapping module mappings fail its verification. This source context is not proof of the exact selected runtime's build source, nor evidence the warning is harmless. Actual-image verification is a separate result.

## Source/build/binary association

Both final-v2 producers use fresh outputs and source-digest-derived scratch roots, retaining before/after source identities, actual unchanged-driver/build commands, exact mandatory results and matching tested bundle copies. Each source identity binds 507 inputs; the candidate source tar contains 516 regular files and independently matches all bound inputs on root and the thread worktree. Sources/tools stayed quiescent through execution. Before/after checks detect persistent drift; they are not signatures or proof against a transient edit reverted between captures.

| Artifact | SHA256 |
| --- | --- |
| v2 ASan aggregate executable | `e7aea05e062738dc75b839f3df80e60ca80fb9ab666453d38f64350b5efc719d` |
| v2 TSan aggregate executable | `27c8350c895416102be22f5d80f85c066a2105bd6482764df643300f2d546c59` |
| Swift.org frontend | `956438a61893525d7ce28c21b6634093f15b0c4bae7fe6eb7d98fba0c1c5e301` |
| SwiftPM testing helper | `9b2b116ce29b756e9a6319526bd803a19946916273544a75ace08de58386f5b8` |
| ASan runtime | `0f77b71ea4342b04be5163a7c02d69376f5ecd5ce71e40278b41473850485180` |
| TSan runtime | `703112657f82fc3522365b529f83919cddc61d114fd7c6bdc8ddaa2088debfe3` |
| Actual XCTest host | `4464ecb489e7c8331f9b384f79aa1677cbb815d1ff3ad42c2b487c9a99ced70644` |
| macOS 27.2 SDK settings | `82165b15ee21419d4cffaba8895797391b908ac9499ba70f561f3e1bd16758d6` |
| v2 candidate source tar | `5046dcc63a09dc09560c1398c124732465fa4189b3fc27ecdc38922e4f70fb12` |

The selected Swift.org frontend and existing Xcode SDK are used through a local lookup adapter. No SDK/toolchain/runtime was installed and global Xcode selection was unchanged. Each package build uses two jobs. The retained coordinator resource records report both package builds and iOS execution finished before authorization for overlapping ASan/TSan full runs, and report performance measurements held until native completion. The no-timing-overlap observation is scoped to those recorded coordinator preconditions and observations. Native timestamps establish a narrower ordering: the XCTest terminal summary at 22:09:49.447 local precedes the corrected additional TSan Swift Testing runStarted event at 22:10:32.169633 by approximately 42.72 seconds. Normal child/body deadlines remain unchanged.

## Chronology and evidence selection

All stages occurred on 2026-11-04; their identities and outcomes remain separate:

1. Baseline reused verified first-phase bundles from source `04b6469`: broader execution exposed generic child sanitizer-preload loss, with real failures/timeouts. The shared-support owner repaired that class. The original 696-entry baseline selection and both review-fix selections are unchanged.
2. Initial integrated `6d574f3` candidate: mandatory ASan 77 / TSan 39 pass; ASan both full frameworks complete with skips. No broad TSan run starts. Subsequent test actor/platform isolation repairs supersede this candidate; it is not relabeled as final source.
3. Final native `e7482a9`: fresh bound producers, complete ASan, original TSan timeout plus successful XCTest, interrupted wrong-environment additional attempt, then corrected complete additional Swift Testing sweep. All attempts remain retained.
4. Later `5a33010` qualification-tool repair and fixture-only reconciliation; final prose/packaging follow native completion. No new native execution is used to repair accounting.

Evidence base: `/private/tmp/vectorcore-release-phase-two-2026-11-04/`. `sanitizers-final/final-results.json` consolidates exact associations, raw summaries, complete additional sweep, warning scan, parser reconciliation and source archive references. `sanitizers-final/archive-manifest.json` uses `entries`, with paths relative to that evidence base; `archive-inclusion.txt` is the explicit selection. Actual XCTest/helper/runtime/bundle executables, source tars, build/source/command proof, raw events/logs and Q03 captures are selected. Bulk builds, module caches, dSYM and SDK/toolchain payloads are excluded.

The private baseline `sanitizers/address-xctest-discovery.stderr` is never read, hashed or selected here. Its environment-free redacted derivative, prior redaction metadata and explicit sensitive-artifact exclusion record are selected instead. Earlier independent Darwin `leaks` controls and the small real public-consumer clean/intentional-leak observations remain bounded baseline results; ASan still uses `detect_leaks=0:halt_on_error=1`. They do not establish project-wide leak cleanliness.

Self-review checked original versus normalized inventories, exact event IDs and pending state, skip retention, actual image proof, source/build/artifact links, immutable prior selections, separated native/posthoc-tool revisions, interrupted attempt accounting, diagnostic limitations, bounded commands and archive exclusions. No release-readiness claim.
