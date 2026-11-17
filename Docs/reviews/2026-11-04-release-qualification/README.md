# VectorCore 0.4.0 qualification — 2026-11-04

The local and hosted qualification batch is complete. [CI run 41946052293](https://github.com/gifton/VectorCore/actions/runs/41946052293) passed all 14 jobs, including matching Swift 6.0 full suites and actual hosted ASan/TSan workloads. **This is not a 0.4.0 release sign-off:** the compiler-specific Release numerical failures and the remaining execution limits below still require an explicit release decision.

The tested implementation is `f779d92d8945edd46fe0807607d487ee42c711a3`, carried on `codex/audit-top-six-2026-10-27` and published on `codex/qualification-0.4.0-2026-11-04`. The final record-only commit follows that tested source. All earlier commits and worktrees remain preserved; this pass does not merge main, cut a release branch or publish a release.

## Hosted results

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

The [hosted execution record](hosted-ci.md) and [machine-readable summary](hosted-validation-summary.json) bind the results to the exact candidate, jobs and artifact hashes. Full-suite counts include reported skips. Strict-performance controls run separately and retain their expected assertion failures; they establish gate behavior and numerical checks, not broad timing acceptance.

The original hosted failures exposed test-infrastructure portability defects. The reviewed repairs canonicalize sanitizer runtime aliases, restore selected private XCTest framework paths, replace a copied system executable with a test-owned native reader, and compare the two OS-dependent loader-environment rows with an independent direct-exec baseline. Deterministic preserve/strip/error controls retain strict expectations. A final formatting-only fix resolves one overlong diagnostic line. All failed and superseded runs remain recorded; no numerical assertion, sanitizer option, pinned body set, required workflow gate or deployment floor was weakened.

Production `Sources/`, `Benchmarks/` and `Package.swift` are unchanged from `d628a2c`. Hosted repairs change the sanitizer driver, its Python tests, the metadata XCTest fixture and a test-only marked-child exec helper. Their scoped reviews have no open Critical/Important finding. See the three repair reports: [runtime/native fixture](hosted-fix-1.md), [private frameworks](hosted-fix-2.md), and [independent OS baseline](hosted-fix-3.md).

## Earlier local qualification

The local phase validated tooling at `0ecf6bd392e91b49cb5c7529fea8160d1410e107`. At that point all 387 package inputs matched the preceding repair baseline; later hosted fixture changes are described above. Its [validation summary](validation-summary.json) remains an unchanged historical snapshot.

| Cell | Observed local result and scope |
| --- | --- |
| Genuine Swift.org 6.0.3 | 12 pure-Swift probes, eight Collection witness cases, 20 numerical importer checks, eight documentation programs and 21 contracts pass. Both local full-suite attempts stop before bodies at an independently reproduced newer-XCTest-overlay mismatch; later matching hosted results are recorded above. [Compiler report](swift-6.0.md) |
| Swift.org 6.4.2 sanitizers | 77 ASan bodies and 39 TSan entries pass, including the real Q03 child. Both intentional diagnostic controls are detected. Metadata runtime-propagation checks pass with insertion explicitly restored by the caller. Selected Debug scope; leak detection disabled. [Sanitizer report](sanitizers.md) |
| Apple compilation | Both library products compile in eight Release simulator/Catalyst cells, with measured arm64 and x86_64 slices. Intel library/benchmark/test executables compile; Intel runtime is unavailable. [Platform report](platforms.md) |
| iOS 27 arm64 smoke | Listed public arithmetic, QR/eigen and metadata assertions pass; this is a bounded client, not a full simulator suite |
| iOS compiler canary | Xcode 27.2 Debug passes; **Release returns stale `2.0` instead of `1.0`**. The Swift.org 6.4.2 Release control passes on the same SDK/runtime, but qualifies that canary only |
| Local repository checks | 594 Python tests and claim/GitHub/whitespace checks pass at `0ecf6bd`; after hosted repairs, 596 Python tests pass at `efcc4ac`. Pinned lint and the affected metadata test pass at formatting successor `f779d92` |

The accepted native Debug/Release baseline from the [preceding repair batch](../2026-11-04-release-preparation/README.md) was reused only while all package inputs were identical. Hosted suites execute the final changed fixtures afresh. Existing compiler/build/lint warnings remain visible.

## What still prevents broader acceptance

1. **Resolve or explicitly scope affected compilers.** Xcode 27.2's ordinary Release numerical canary fails on macOS and iOS. The final hosted Xcode 26.3 / Swift 6.2.4 Release run also records the existing lost-store canary issue (`2.0` instead of `1.0`) under unchanged known-issue policy. A successful CI aggregate does not erase an observed wrong numerical result. The passing Swift.org compiler is a different binary; no complete Swift.org simulator-library or App Store qualification is claimed.
2. **Keep execution limits explicit.** Minimum-OS and physical-device runs, tvOS/visionOS/Catalyst/Intel runtime, full simulator suites, Release/full-suite sanitizer coverage, leak detection and broad controlled performance remain unexecuted by this pass. Compile success does not establish runtime support for those cells.
3. **Finish the remaining release follow-up.** Previously recorded benchmark-reporting/mandatory comparator work and documentation-reference/count cleanup remain separate tasks. Existing metadata ownership, foreign-reaper, selector and trap-provenance limitations remain unchanged.

## Evidence

The separate hosted archive is `/Users/goftin/dev/projects/VectorCore-hosted-qualification-2026-11-04-evidence`; its [manifest](hosted-evidence-manifest.json) records selected failures, repair controls, reviews, final command outputs and artifacts. Source/copy hashes and a second destination read verify the frozen bytes. Final-run GitHub artifact ZIP digests and reviewed member bytes were checked; original ZIPs and omitted build caches remain in scratch. Hosted sanitizer runtime image bytes are not retained for an independent offline rehash, so the record distinguishes producer-recorded hashes from locally rehashed helper/bundle bytes.

The original local archive, `/Users/goftin/dev/projects/VectorCore-qualification-2026-11-04-evidence`, and its [manifest](evidence-manifest.json) remain unchanged, as does the preceding release-preparation archive. The [completed plan](../../superpowers/plans/2026-11-04-release-qualification.md) and retained ledger record the execution decisions and limitations.
