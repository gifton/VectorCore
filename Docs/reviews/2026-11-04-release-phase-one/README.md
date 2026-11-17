# VectorCore 0.4.0 first-phase remediation — 2026-11-04

The three parallel tracks are implemented and independently reviewed on the carried `codex/audit-top-six-2026-10-27` branch. The combined tooling suite passes at the integrated implementation; the evidence archive and closure verification are identified below. This phase does not provide release sign-off: the Xcode Release numerical failure remains unresolved, and the actual iOS package test scheme exposes a test-support portability failure.

Base: `2dcb24091c6dc21be4db409ecdd0756220e93b8c`. Integrated implementation: `2782f11bee56f70511856dbaebcff68b47819234`. Original worker commits are preserved as ancestors. No main-branch merge, push, release branch, version change, tag or publication is part of this phase. Shipping `Sources/`, `Tests/`, `Benchmarks/` and `Package.swift` are byte-identical to the phase base; new standalone compiler probes live in this review directory.

## Delivered work

| Track | Result | Scope |
| --- | --- | --- |
| Compiler correctness | 32 probe cells across the exact Xcode and Swift.org binaries. Xcode Release fails the original, inside-borrow and copied-buffer/inout shapes; Swift.org passes every positive cell. All four omitted-write controls are rejected. | Investigation is complete; no general source workaround or support restriction is adopted. [Compiler record](compiler/README.md) |
| Release sanitizers | Driver accepts explicit Debug/Release configuration, keeps Debug as default and enables Release testability while retaining `-O`. Actual Release ASan passes 77/77 and TSan 39/39, including Q03. | Selected optimized testable workloads, not whole-suite or production-binary equivalence. [Runtime record](runtime-sanitizers/README.md) |
| Available runtime | Clean Catalyst client passes the listed arm64 public-library assertions. | Current host only; other APIs, Intel and minimum-OS execution remain unqualified. |
| Benchmark comparator | Existing required CLI command now reaches the comprehensive current-schema reporting table and actual-product comparisons: 17 CLI calls, 320 comparator invocations. Removable Python assertions are replaced and checked across normal, `-O` and `-OO` execution. | I:D-089's required-execution gap is repaired and locally executed. No new hosted run or performance acceptance. [Benchmark record](benchmarks/README.md) |
| Documentation and claim evidence | Corrected sanitizer/benchmark gate guidance, consumer coverage wording, current qualification links and tutorial scope. The comparator claim cites dated executed evidence, retaining its old statement/fingerprint in ledger history. | Historical measurements and dated prior reports remain records of their original source. |

Linking the dated comparator artifact exposed an incomplete claim-test fixture: the live checker passed, but its copied repository omitted dated artifacts outside Sources/Tests/Benchmarks. The generic dependency copier now includes every dated-artifact reference. The existing unchanged-copy and corrupt-hash controls establish RED/GREEN; no checker exemption or claim downgrade was added.

## Failures retained as findings

- **Compiler:** Xcode returns stale `2.0` / `40000000` where `1.0` / `3f800000` is required. Copying the buffer does not fix the tested inout shape. Passing nearby formulations and matching version labels do not qualify the failing binary. Prior hosted Xcode 26.3 Release also retains its known numerical issue.
- **iOS full suite:** the `VectorCore` library scheme reports a successful build with an empty test configuration and zero test bundles. The actual `VectorCore-Package` scheme fails compilation because shared `ChildProbeLauncher.swift` uses unavailable `Process`, with zero test bodies executed. Portable test support or explicit platform-specific test contracts is a concrete follow-up; no blanket test exclusion was added.
- **Leaks:** the deliberate 4096-byte allocation survives compilation and executes, but ASan with `detect_leaks=1` emits no leak diagnostic. Standalone leak-sanitizer probes reject the tested Darwin targets. Leak detection remains unqualified.
- **Performance:** two actual strict reports compare with exit 1 on the busy host. That result is retained as a timing observation; the tooling contract does not convert it into performance acceptance.
- **Infrastructure:** original Release sanitizer builds failed before test bodies because testability was absent. The initial Swift.org Catalyst client linked with platform warnings and is not the accepted smoke. Both failures remain separate from the corrected runs.

## Validation and review

The task reviews independently checked compiler matrices, selected sanitizer body/image/child evidence, Catalyst input reuse, benchmark reports and command outcomes, and retained file hashes. The runtime review's expected-argparse-output and warning-disclosure findings are corrected with 11 passing focused tests. Original successful native builds retain existing compiler warnings and native-build-system deprecation diagnostics.

The benchmark track's initial RED/GREEN top-level environment capture is incomplete; its final covering and actual-product runs have complete command records. This limitation is retained rather than reconstructed. At integrated implementation `2782f11`, all **598 Python tests** and **2 compiler-driver tests** pass. Claim-ledger validation reports zero problems; CI-contract, GitHub-policy and whitespace checks exit zero. Before/after hashes agree for all 563 captured inputs. Exact commands, deadlines, output identities and scope are in [validation-summary.json](validation-summary.json). The combined run includes the runtime cleanup and generic dated-artifact fixture repair. Native task executions retain their own exact input manifests; no full native suite or hosted run was repeated for this phase.

The final phase review found no new Critical or Important implementation issues. Its completion conditions are terminal combined-check evidence, verified archive freezing and finalized records. [archive-verification.json](archive-verification.json) records the fulfilled archive conditions and independent closure review; original task and integration reviews are retained in the archive.

## Next release decisions and work

1. Resolve the affected Xcode compiler combinations or explicitly adopt a compiler-support scope, then qualify that exact combination. No support reduction was made here; the previously passing Xcode 16.2 and exact Swift.org results remain bounded evidence.
2. Repair or define portable test-support contracts for full iOS testing. Complete the chosen minimum-OS, device, Intel, tvOS/visionOS and broader Catalyst/simulator execution matrix, or explicitly retain those coverage limits.
3. Decide the required whole-suite sanitizer and leak coverage beyond the selected Debug/Release manifests.
4. Run controlled Release performance comparisons on settled source/compiler and a quiet host. Optional adversarial CSV parsing, report completeness, independent dot/normalize/mem.copy validation, runtime-versus-build provenance and thermal/power compatibility are not closed by the comparator execution repair.
5. Run combined candidate qualification and hosted CI for the integrated changes, finalize release notes/version/support claims, and then perform separately agreed release integration/publication.

## Execution choices and evidence

- Continued the user's approved first-phase scope without another design approval; a mistaken scope interpretation would cost isolated rework.
- Ran independent workers concurrently in managed worktrees, with two compiler jobs per worker; shared-file ownership and review bounded integration/resource-contention risk.
- Reused unchanged source-bound baseline evidence instead of repeating every accepted full native suite; a changed environment could surface later, so fresh relevant controls and the combined tooling suite cover the actual changes.
- Preserved all worktrees and prior/fresh evidence; the tradeoff is local disk usage.

Raw evidence is under `/private/tmp/vectorcore-release-phase-one-2026-11-04`. The bounded archive is `/Users/goftin/dev/projects/VectorCore-release-phase-one-2026-11-04-evidence`; its manifest identity, inclusion counts and verification are recorded in [archive-verification.json](archive-verification.json). The archive snapshots the reviewed implementation, task reports and the coordinator records at freeze time. Its own manifest/receipts and subsequent closure receipt/plan completion are metadata outside that payload, avoiding self-referential hashes. Compiler/SDK payloads and bulk caches are excluded. Actual selected sanitizer runtime/helper/bundle bytes and the real benchmark product/comparators are retained. This record supplements the [earlier qualification](../2026-11-04-release-qualification/README.md) and the [phase plan](../../superpowers/plans/2026-11-04-release-phase-one.md).
