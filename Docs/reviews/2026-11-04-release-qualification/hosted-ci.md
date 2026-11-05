# Hosted candidate qualification

**All 14 jobs passed** in [run 41946052293](https://github.com/gifton/VectorCore/actions/runs/41946052293). Its exact source is `f779d92d8945edd46fe0807607d487ee42c711a3`, published on `codex/qualification-0.4.0-2026-11-04`. No main merge, release tag or release publication was performed. The later documentation commit records these results; it is not the source executed by this run.

## Required results

| Cell | Verified result |
| --- | --- |
| Xcode 16.2 debug | 211 XCTest / 28 skips; 1757 Swift Testing / 18 skips; zero unexpected failures; 0 known canary issues; all 22 downstream contracts pass |
| Xcode 16.2 release | 211 XCTest / 28 skips; 1757 Swift Testing / 18 skips; zero unexpected failures; 0 known canary issues; all 22 downstream contracts pass |
| Xcode 26.3 debug | 211 XCTest / 28 skips; 1757 Swift Testing / 18 skips; zero unexpected failures; 0 known canary issues; all 22 downstream contracts pass |
| Xcode 26.3 release | 211 XCTest / 28 skips; 1757 Swift Testing / 18 skips; zero unexpected failures; 1 known canary issue; all 22 downstream contracts pass |
| ASan / TSan | 77/77 and 39/39 pinned Debug entries; actual Q03 child verified |
| Benchmark contracts | Four build modes, eight strict-correctness runs, four expected mismatch failures, 144 route rows and 40 C-tail cases |
| Repository checks / lint | 596 Python tests; nine GitHub YAML/template files and workflow policies; SwiftLint 0.65.1 passes with zero errors |
| Consumers / platform compilation | Eight documentation programs and 21 contracts; eight library product/destination cells and Intel library/test compilation |
| Other jobs / aggregate | SDK collision check, artifact-only performance smoke and CI Required all pass; overall 14/14 jobs succeed |

Full-suite results were checked in the XCTest output and Swift Testing XML; the XML does not include XCTest. The rehashed test hosts/bundles and loaded-image records belong to the separate strict-performance and sanitizer builds. Counts include the explicitly reported skips; the performance driver additionally executes 22 fresh-process cases per cell: 16 gate controls, three observation controls and three actual representative tests. Five deliberately fail to prove that assertions remain active. Their expected failures are distinct from full-suite failures and from timing acceptance.

Both sanitizer jobs use the unchanged pinned manifest (`91b7eb0850136ff08ac6c0261e80f2db1c0624bfd86607582b1a3bd4546a4f86`): 77 ASan bodies and 39 TSan entries, including Q03. The evidence checks actual host/bundle/runtime loaded paths, command exits and body events, plus the Q03 child's identity, six workers, 2000 loops per worker and bounded cleanup. This is selected Debug instrumentation with leak detection disabled. The hosted runtime hashes are recorded by the producer, but the hosted runtime dylib bytes are not retained for an independent offline rehash. Retained helper and bundle bytes are independently rehashed.

## Failures retained and repaired

| Run and exact source | Observed result | Repair or disposition |
| --- | --- | --- |
| [41934186109](https://github.com/gifton/VectorCore/actions/runs/41934186109), `b6604fc` | Four full-suite cells fail the copied system-reader fixture; both sanitizer jobs reject runtime aliases before bodies | [Wave 1](hosted-fix-1.md) canonicalizes runtime paths and builds a test-owned native reader, retaining all 14 environment rows and explicit setup-failure coverage |
| [41940013803](https://github.com/gifton/VectorCore/actions/runs/41940013803), `90b8d58` | Both sanitizer helpers lack a private XCTest dependency; all four full-suite cells fail only the two protected DYLD expectations (four assertions) | [Wave 2](hosted-fix-2.md) restores the selected private-framework search root; [wave 3](hosted-fix-3.md) measures OS policy through an independent direct-exec baseline, with deterministic preserve/strip/error controls |
| [41944878066](https://github.com/gifton/VectorCore/actions/runs/41944878066), `efcc4ac` | SwiftLint rejects one 256-character diagnostic line; both sanitizer jobs pass | Formatting-only `f779d92` wraps the line with identical emitted text. Six completed jobs pass, lint and the aggregate fail, and six unfinished jobs are intentionally cancelled. This partial run is superseded, not accepted as a full matrix |

The scoped repair reviews found no Critical/Important issue. Local checks at `efcc4ac` passed all 596 Python tests, the claim ledger, GitHub configuration and whitespace checks. At formatting successor `f779d92`, pinned SwiftLint 0.65.1 reports zero errors (5486 existing warnings), and the affected 14-row metadata case passes. The final hosted run, rather than reused local checks, is the exact-source full-matrix evidence.

Production `Sources/`, `Benchmarks/` and `Package.swift` are unchanged from `d628a2c`. The changed executable inputs are the sanitizer driver and its direct Python tests, `RunnerContractTests.swift`, and the test-only `ChildProbeLauncher.swift`. No numerical contract, manifest membership, sanitizer option, test exclusion, workflow requirement or deployment floor was weakened. The two host-dependent environment rows must match an independently measured baseline; execution failures do not count as environment stripping. No claim about hosted SIP configuration is made.

## Acceptance boundaries

The Xcode 26.3 / Swift 6.2.4 Release cell again records the existing lost-store canary issue: **observed `2.0`, expected `1.0`**, at `LostStoreCanaryTests.swift:53:13`. The unchanged pre-6.4 known-issue policy allows that CI cell to pass. The other three cells report no known issues. This remains a numerical compiler qualification limit, not a newly suppressed test.

The local Xcode 27.2 Release numerical canary also fails on macOS and iOS. The passing Swift.org 6.4.2 control is a different compiler and has its own limited qualification scope. A successful CI run does not make a compiler's observed wrong numerical result acceptable. Resolve or explicitly scope the affected compiler combinations before release sign-off.

Minimum-OS and physical-device execution, tvOS/visionOS/Catalyst/Intel runtime, full simulator suites, Release/full-suite sanitizer coverage, leak detection and broad controlled performance remain outside this pass. The previously recorded benchmark-reporting/comparator and documentation cleanup remain separate follow-up work.

## Provenance and retained evidence

`hosted-validation-summary.json` records exact run/job identities and verified results. `hosted-evidence-manifest.json` names the bounded frozen hosted archive, selected original failures, local repair controls, reviews, final artifacts and all verified hashes. Final-run artifact ZIP digests are checked against GitHub's API; reviewed final artifact files are checked against their ZIP members. Bulk ZIPs, intermediate object/module caches and SDK/compiler payloads remain in original scratch or their existing locations, outside the bounded archive. The manifest states omissions and references to the earlier frozen local archive.

The original local `validation-summary.json` and `evidence-manifest.json` remain unchanged historical records. Their local-only “hosted pending” conclusions are superseded by this hosted record, not rewritten as if those earlier commands executed here. All earlier commits, worktrees and evidence remain preserved.

The initial read-only preflight observed old-main run [41378600891](https://github.com/gifton/VectorCore/actions/runs/41378600891) at `7d9c962af774d9aae3cd2445d487ec5a941a7a65`. It established available CI, not acceptance of the carried candidate. The user-approved continuation authorized publishing reviewed candidates on the dedicated qualification branch and dispatching the existing workflow. The upstream-only artifact-action update was not merged as part of qualification.
