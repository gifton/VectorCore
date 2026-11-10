# VectorCore 0.4.0 compiler acceptance policy

**Adopted by the user on 2026-11-10.** This policy permits bounded qualification
of VectorCore on the tested Xcode 27.2 compiler despite its retained
descriptor-inout miscompilations. No source candidate is admitted by this decision.

For this restricted qualification, the rule below supersedes the requirement
that every positive compiler probe pass in the earlier release-admission sequence.
That sequence's other release gates remain open. The historical checker and
evidence continue to describe their original, pre-adoption assessment.

## Consumer requirements

The user identified Xcode 27.2 and both macOS and iOS as the initial consumer
targets. Custom storage conformers and consumer-written unsafe-buffer mutation
helpers are not on the roadmap for any consumers currently known to the user.
The user favors a release with a documented limitation. A later correction is
desirable but is not an assumption required to justify this release.

These consumer priorities do not withdraw the package's declared macOS 14,
iOS 17, tvOS 17 or visionOS 1 deployment floors, Intel compatibility, or existing
required CI jobs. Their release coverage remains a separate decision.

## Supported usage boundary

Qualify VectorCore's shipped vector/storage implementations and numerical
operations, including their internal helper chains and generic/inlinable consumer
code, for explicitly recorded compiler, build and runtime combinations. Each
qualification must identify the assessed operations and routes. Naming a shipped
type or passing its outer API signature is insufficient evidence for every route.

Keep the current public extension and borrowed-buffer APIs available. Documented
by-value mutation patterns may be included where assessed. Arbitrary external
conformers, opaque imported helpers and generic pointer-bearing state do not
inherit qualification from the protocol or callback signature.

On an affected qualified combination, exclude additional inout handoffs of a
borrowed pointer, buffer descriptor, span/view or pointer-bearing wrapper to a
mutating helper, transitively through callback, helper and conformer chains.
The reviewed ContiguousArray borrow-entry adapter forwards the stdlib-supplied
descriptor by value; its specific entry boundary is included in the assessment.
This does not create a general exception for other adapters. Owning-array/vector
inout operations remain separately assessed and are not prohibited merely for
using inout. Passing individual inout examples do not earn exceptions.

The restriction concerns valid Swift that affected compilers miscompile. It is
not a lifetime/bounds rule or a finding of caller misuse. The tested patterns do
not establish a complete partition of affected and unaffected programs. Support
wording must preserve that limitation and explain the risk of silent wrong values.

## Initial compiler identity

The first target is the released Xcode 27.2 build 27B52, using Apple Swift 6.4.2
(`swiftlang-6.4.2.1.9`, `clang-2100.4.1.1`). The local frontend was rehashed on
2026-11-10 and matches the retained investigation:

`448457da9a8a56741f2c8c4b8931e85b7377fc009c16918cd8ac1a8a3210ea1f`

A qualification also binds the source manifest, compiler driver/frontend, SDK,
target triple, architecture, language and optimization settings, runtime, and
producer/client artifacts. The marketing version alone does not identify an
accepted combination. Xcode 27.3 beta, other builds, and future releases receive
no automatic acceptance. Existing older CI tuples are not waived or newly
certified by this exception.

## Qualification rule

1. Keep compiler correctness, scoped VectorCore qualification and final release
   admission as separate outcomes. A compiler with the observed failures retains
   `compilerProfile=known-failing` even if scoped qualification succeeds.
2. Require correct results from all positive checks for supported operations and
   consumer routes, including captured-variable checks and generic/inlinable
   consumers. Retain independent omitted-write controls and their complete
   expected outputs. A supported-route failure is never waived by compiler
   version, test name or resemblance to a known issue.
3. Preserve the original compiler canaries and residual probes as raw numerical
   diagnostics. A reviewed profile may recognize only their exact excluded
   compositions, compiler/build configurations, full row inventories and observed
   results. Keep incorrect positive observations labeled as failures; do not
   weaken their numerical assertions or turn their native process exits green.
   Full-suite acceptance must reconcile individual outcomes and distinguish the
   named diagnostic failures from every other failure.
4. Reject qualification on missing or duplicate rows, changed inventories,
   unexplained numerical differences, crashes, timeouts, incomplete controls or
   mismatched identities. An unexpectedly corrected diagnostic triggers review
   and a new profile rather than silently inheriting acceptance. Any new failure
   in supported usage revokes that combination's qualification until investigated.
5. Source, compiler, SDK, flags, architecture or runtime changes require a reviewed
   evidence refresh for the changed scope. Preserve the historical bounded checker
   and profile as an assessment of their original candidate. Do not make their
   `releaseAdmitted` field true or rebind old receipts to a new source candidate.
6. Add explicit Xcode 27.2 qualification coverage. The current required CI runs
   Xcode 16.2/26.3 and cannot supply that acceptance. Re-derive reachability before
   changing admission: legacy known-issue branches, optional C kernels, alternate
   architectures, platform builds and required-job aggregation all remain visible.
   Source scanners are scoped review checks, not proofs about downstream chains.

Every scoped result must publish its covered and uncovered routes and runtimes.
Final release admission additionally requires disposition of the existing
platform/runtime, sanitizer, performance, hosted-CI and release-reference gates.
Adopting this policy supplies no blanket waiver for those gates.

## macOS and iOS evidence

The native investigation covers macOS 27.0.1 build 26A434 on arm64. It includes
Debug/Release producers and consumers for the migrated candidate before the alpha
version change. The alpha has separate macOS Debug/Release public-consumer
receipts. Neither record is automatically evidence for the final 0.4.0 inputs.

The alpha's iOS 27.0 build 24A434 arm64 simulator suite records 1,829 passed,
163 skipped and one failed declared case: the original optimized lost-store
canary. All seven mutation tests passed. These results do not provide the complete
platform-aware residual/consumer matrix or physical-device execution.

The intended next execution sequence is native macOS, an iOS simulator consumer
and diagnostic matrix, then the agreed physical-iOS scope. Native macOS binaries
cannot serve as iOS consumer evidence by changing SDKROOT. Each platform needs
matching targets, linked modules/objects, SDKs and an appropriate runner.

The user additionally requires at least one app-hosted Release consumer run on
a physical iPhone or iPad before stable 0.4.0 iOS acceptance, in addition to the
simulator checks. The previous package-test scheme did not reach test bodies on
a physical device, so a runnable app-hosted consumer or harness is needed. Report
that bounded consumer separately from the full package suite. Minimum-OS runtime
coverage and Intel coverage still require explicit disposition.

## Adoption and implementation scope

The user explicitly selected adoption of the reviewed proposal and subsequently
selected the physical-device requirement above. Xcode 27.2, macOS and iOS are the
priority qualification targets. This permits qualifying an affected compiler
under explicit restrictions; it does not mark any current candidate accepted.

The [implementation plan](../../superpowers/plans/2026-11-10-compiler-acceptance.md)
covers fresh qualification, validation and CI integration. Historical evidence
remains immutable, and the dirty main checkout must be preserved. No release tag
or publication is part of this decision. Broader platform coverage and final
release admission remain open.

## Evidence references

- [Original boundary proposal](../2026-11-06-workaround-qualification/qualification-proposal.md)
- [Migration validation](../2026-11-06-buffer-api-migration/README.md)
- [Bounded native assessment](../2026-11-06-bounded-native-qualification/README.md)
- [Current release status](../../../RELEASE_STATUS.md)
- [Retained alpha summary](/Users/goftin/dev/projects/VectorCore-archive/VectorCore-bounded-native-qualification-2026-11-06-evidence/integration-preview/final-summary.json)
- [Existing release-admission policy](../2026-11-05-release-gates/release-admission.md)
