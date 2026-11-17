# Remaining exposure audit and qualification proposal

Status: source review and bounded native experiment complete; support-boundary
decision pending. Release admission, source APIs, CI and support policy are unchanged.

## Finding

The distinct by-value requirement removes the old required inout boundary, but
valid clients can reconstruct that boundary. New experiments reproduce stale reads
through `withUnsafeMutableElements` on both tested Xcode Release compilers.

The next action is a support/qualification decision described in the
[concrete proposal](qualification-proposal.md), not another spelling change.

## Numerical evidence

The [probe source](ResidualExposure.swift) contains 17 independently named first-lane
observations. Every positive row expects Float bits `3f800000` (1); every omitted-write
row expects `40000000` (2). The [driver](run.py) uses the existing complete numerical
judge and requires all rows, exact completion counts and corresponding process exits.

| Compiler | Debug client | Release client | Omitted-write controls |
| --- | --- | --- | --- |
| Xcode 27.2 | 17/17 correct | 14/17 correct; three stale reads | Both executions match all 17 expected rows |
| Swift.org 6.4.2 | 17/17 correct | 17/17 correct | Both executions match all 17 expected rows |
| Xcode 27.3 beta | 17/17 correct | 14/17 correct; same three stale reads | Both executions match all 17 expected rows |

There are 102 positive observations: 96 correct and six incorrect. The seven
by-value controls contribute 42 correct observations. All six omitted-write
executions produce all 17 deliberately unchanged values, report 17 failures and
exit 1. Those deliberate no-write controls are distinct from the valid positive
inout cases that unexpectedly return the wrong value.

The three failing case IDs are:

- `new.concrete.copied-inout`: copy the new callback's descriptor, pass it inout
  to an out-of-line writer, then read the owner after the borrow.
- `new.inside.copied-inout`: the same call, with readback inside the borrow.
- `new.aggregate.inout`: put the pointer in another struct, pass that struct
  inout to its writer, then read the owner after the borrow.

All writes remain within synchronous nonescaping borrows of initialized nonempty
storage. There is no owner reentry or descriptor replacement. Source review found
no violated lifetime, bounds or exclusivity requirement in these probes. This is
a compiler-specific limitation on valid API compositions, not caller misuse.

The tested actual `Vector512Optimized.storage` stdlib/inout, copied-descriptor and
Float callback routes pass. So do generic helper and custom-conformer bridge
cases. Their results are source-shape observations, not evidence that those inout
forms escape the defect class. The proposed restriction would include them.

## Scope and provenance

This experiment reuses the matching, hash-verified **Release** producer products
from the [completed migration](../2026-11-06-buffer-api-migration/README.md), with
both Debug and Release clients. It does not rebuild the library or add Debug
producer coverage. The 389 actual compiled source inputs, compiler/SDK identities,
object/module hashes and producer receipts are rechecked before reuse and after
execution. The source base remains `bdf990847babf567be1cbce22d2606e7102b1a47` plus
the retained migration delta; HEAD alone does not identify the candidate.

The target is arm64 macOS 14, running on macOS 27.0.1 (26A434), arm64. This is
not minimum-OS execution. Custom conformer and generic helper definitions are
client-local; this adds no unspecialized cross-module witness proof. Only the
first lane is observed: no new tail/COW/empty/throw/descriptor-replacement,
arbitrary pointer-bearing state, platform, sanitizer or performance coverage is
claimed. Emitted SIL is a separate compiler emission, not a trace of the executed
binary's individual optimization passes.

Raw records, binaries, SIL and input identities are retained at
`/Users/goftin/dev/projects/VectorCore-workaround-qualification-2026-11-06-evidence`.
The [result summary](summary.json) preserves incorrect positive observations as
failures and records `releaseAdmitted: false`. Driver completion is evidence
collection success, not numerical acceptance of every source shape.

The [independent verification](../../../../VectorCore-workaround-qualification-2026-11-06-evidence/verification.json)
rechecks raw numerical observations, 70 command streams, all 389 compiled source
inputs, matching products and unchanged protected source/policy files. All 1,578
entries in the prior migration's selected evidence manifest still match. The
[new manifest receipt](../../../../VectorCore-workaround-qualification-2026-11-06-evidence/manifest-verification.json)
binds this experiment's retained artifacts and report snapshot. These are evidence
integrity checks; the six incorrect positive observations remain incorrect.

## Source reachability review

Three Astra reviewers independently examined remaining source exposure, admission
policy and platform requirements. A further source review checked the probe's
borrow lifetimes, exact expectations and full omitted-write oracle.

| Surface | Current evidence | Interpretation |
| --- | --- | --- |
| New SIMDStorage requirement and ContiguousArray adapter | `Sources/VectorCore/Protocols/SIMDStorage.swift:43,57` | Adapter passes the descriptor by value. No internal production generic mutation call was found in the scoped review. |
| Concrete optimized storage | `Sources/VectorCore/Vectors/Vector512Optimized.swift:30` and corresponding 384/768/1536 properties | Public ContiguousArray and checked `_modify` expose stdlib access. Actual 512-dimensional first-lane probe now passes in its tested form. |
| Vector/storage callbacks | `Sources/VectorCore/Protocols/VectorProtocol.swift:88`, `Sources/VectorCore/Storage/VectorStorage.swift:31` | By-value signatures cannot prevent callback-local reconstruction of an inout descriptor. |
| Custom conformers | `Sources/VectorCore/Protocols/SIMDStorage.swift:17`, `VectorProtocol.swift:41` | Their internal forwarding code is not constrained by the requirement's function type. |
| Generic mutable state | `Sources/VectorCore/Configuration/ThreadSafeConfiguration.swift:33` | Public inout state may contain pointers; inspected internal BatchOperations configuration is scalar-only. No new numerical probe qualifies arbitrary specializations. |
| Raw handles/storage | `Sources/VectorCore/Protocols/BufferProvider.swift:22`, `Storage/UnifiedVectorBuffer.swift:110,170,184` | Removing optimized storage alone would not eliminate unsafe client composition routes. |

Declaration/readback guards are scoped review aids, not Swift type/alias proofs.
Imported implementations, generated source, arbitrary specialization and helper
chains need separate assessment. Ordinary owning-container inout mutation is not
the same API boundary as a borrowed descriptor; its correctness still needs its
own evidence.

## Qualification gaps identified

The existing required native CI matrix uses Xcode 16.2/26.3, whereas the migration
and this experiment use 27.2/27.3 beta/Swift.org 6.4.2. Existing documentation
consumers run only Xcode 16.2. Both the compiler and numerical consumer drivers are
hardcoded to native arm64 macOS and launch binaries directly; selecting a simulator
SDK does not make them valid platform runners.

Required CI includes eight product/platform Release compilation cells and Intel
test compilation, not runtime numerical acceptance on every advertised platform.
The CI sanitizer driver defaults to Debug; prior investigative Release sanitizer
runs are separate evidence. Optional C kernel modes, x86-specific paths and older
canary known-issue branches must be reconsidered if admission changes.

The smallest additional runtime targets to prepare are current arm64 iOS simulator
and Catalyst, after selecting a qualification scope and checking available
destinations. Their earlier results predate the migration. Minimum OS, real Intel,
tvOS/visionOS runtimes and representative physical-device evidence remain separate
coverage decisions. Physical iOS needs an app-hosted harness or signed public
consumer; the current tool-hosted package test scheme cannot execute there.

No full native-suite repetition was needed for this source-unchanged experiment.
The prior six migration suites retain their own 1,947 executed/46 skipped counts;
those counts are not added to this experiment's observations.
