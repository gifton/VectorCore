# VectorCore 0.4.0 qualification — 2026-11-04

The local qualification batch is complete, independently reviewed and retained on the carried branch. **This is not a 0.4.0 release sign-off.** Real Swift 6.0 consumer checks, the selected ASan/TSan workloads and Apple compilation cells pass. Full Swift 6.0 suites still need a matching XCTest environment, and Xcode 27.2's Release numerical canary fails on the iOS simulator as well as the previously observed macOS host.

Validated implementation/tooling: `0ecf6bd392e91b49cb5c7529fea8160d1410e107` on `codex/audit-top-six-2026-10-27`. All **387 tracked package inputs** under Sources, Tests, Benchmarks and Package.swift are byte-identical to `d628a2c419300d37c8462d24b2b1069f71867f79`. The only executable repository changes are the sanitizer launcher and its Python regressions. The later record-only commit adds this matrix, completed plan, summary and evidence manifest; all earlier commits and worktrees remain preserved.

## Results

| Gate | Observed result | Exact scope |
| --- | --- | --- |
| Actual Swift.org 6.0.3 | 12 pure-Swift probes, 8 inherited/explicit witness cases, 20 numerical importer checks, 8 documentation programs and 21 contracts pass | Existing SDK 15.2 and current arm64 macOS runtime; [compiler report](swift-6.0.md) |
| Full Swift 6.0 Debug/Release | Both stop before tests; 0 executed bodies | Independent two-line XCTest import reproduces the older compiler crash against the installed Swift 6.4.2 overlay; matching Xcode 16.2 hosted cells remain open |
| AddressSanitizer | 77/77 pinned bodies pass | Actual selected helper, bundle and ASan images verified; intentional overflow diagnostic also detected; leak detection disabled |
| ThreadSanitizer | 39/39 pinned entries pass, including Q03 child | Actual child images, identities, six-worker workload and deadlines verified; intentional race diagnostic also detected |
| Metadata runtime propagation | Both ASan and TSan child observations pass | Caller explicitly restores insertion before the real EnvCapture path; protected-shell behavior and ownership limits remain separate; [sanitizer report](sanitizers.md) |
| Apple library compilation | Both library products at all four simulator/Catalyst destinations pass | Eight Release cells, measured arm64 + x86_64 slices, local Xcode 27.2; [platform report](platforms.md) |
| Intel macOS | Library, benchmark and all three test executables compile | Runtime unavailable on this host; compile-only |
| iOS 27 arm64 public-library smoke | Listed arithmetic, QR/eigen and metadata assertions pass | Bounded external client, not a full simulator suite |
| iOS compiler canary | Xcode Debug passes; **Xcode Release fails**; Swift.org Release passes | Same source/SDK/runtime: Xcode returns stale `2.0 / 40000000`, expected `1.0 / 3f800000` |
| Integrated repository checks | **594 Python tests pass**, claim ledger 0 problems, nine GitHub YAML/template files and workflow policies pass, whitespace clean | Clean `0ecf6bd` source; exact records in [validation summary](validation-summary.json) |

The accepted native Debug/Release baseline from the [preceding repair batch](../2026-11-04-release-preparation/README.md) was not repeated: package inputs are unchanged. Existing compiler/build warnings remain visible. No numerical assertion, optimizer pass, sanitizer report, test membership or required CI gate was suppressed. No Swift production or test sources changed.

## Sanitizer launcher repair

The signed Swift.org `swift-driver`/`swift-package` launch path removed dyld tracing before the test helper, causing the original driver to reject missing image evidence after a passing first body. The driver now invokes the exact selected SwiftPM helper with the same bundle and manifest filter, preloads the selected toolchain's unique runtime, and supplies the needed support-library paths. It retains finite deadlines, fail-fast options, exact started/passed-body checks and Q03 validation, and now records the runtime hash and requires its exact path in the loaded-image log. The original failed attempts, focused red/green tests and both fresh full manifest runs are retained. Task reviews and the final combined review found no open Critical/Important issue in this local batch.

## What still prevents broader acceptance

1. **Run hosted CI on the carried candidate.** Existing Xcode 16.2 full suites can resolve the local XCTest mismatch; Xcode 26.3 execution must qualify the changed sanitizer launcher in its hosted/default-backend environment. The [hosted preflight](hosted-ci.md) verifies an active recent matrix, but its successful old-main run does not qualify these local commits. The next authorized qualification step publishes the reviewed candidate to the new qualification ref and dispatches `ci.yml` against that exact source.
2. **Resolve or explicitly scope the affected Xcode compiler.** Installed Xcode 27.2 returns the wrong canary value under ordinary Release optimization. The passing Swift.org compiler is a different binary and the simulator paired control covers that canary only. No complete Swift.org simulator-library or App Store build qualification is claimed.
3. **Retain the remaining test boundaries.** Minimum-OS and physical-device execution, tvOS/visionOS/Catalyst/Intel runtime, full simulator suites, Release sanitizer builds, leak detection and broader controlled performance remain unexecuted by this pass. Previously documented benchmark-reporting/comparator and documentation cleanup remains separate follow-up work.

## Evidence and execution decisions

Frozen local evidence: `/Users/goftin/dev/projects/VectorCore-qualification-2026-11-04-evidence`. [The manifest](evidence-manifest.json) hashes the selected original failures, passing controls, command records, loaded artifacts, reviews, source snapshots and final checks. Compiler/SDK payloads and bulk build/module caches remain outside this bounded archive; original worktrees and scratch evidence are preserved. This frozen local phase precedes hosted execution; no remote mutation, main merge, tag or release occurred during that phase.

The [plan](../../superpowers/plans/2026-11-04-release-qualification.md) and frozen progress ledger retain the execution decisions: parallel isolated workers following the existing preference (possible integration rework); reuse of the fresh, byte-identical native baseline while running new qualification cells (possible later discovery of an environment difference); and preservation of worktrees/evidence (disk usage). These decisions did not convert unavailable or failing cells into passing gates.
