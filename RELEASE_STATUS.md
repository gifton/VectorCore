# VectorCore release status

## Current consumer development — 2026-11-26

Stable 0.4.0 release preparation and candidate qualification are **paused** while
consumer API work proceeds. Historical qualification below applies only to its
named candidates and does not admit the current development branch.

VectorAccelerate and SwiftTopics may pin the reviewed development snapshot:

```swift
.package(
    url: "https://github.com/gifton/VectorCore.git",
    revision: "a676d9745dec60450a2f1ff86da94824eb8b4f5e"
)
```

The independent remote branch
[`consumer-snapshots/2026-11-24-a676d97`](https://github.com/gifton/VectorCore/tree/consumer-snapshots/2026-11-24-a676d97)
keeps that revision reachable outside the PR stack. Retain it without rebasing,
force-updating or deleting until consumers have moved off it and removal is
explicitly authorized. Remote readback and isolated SwiftPM resolution agreed on
that full revision on 26 November. Keep `Package.resolved` as the consumer's exact
identity: this snapshot reports version `0.4.0` with no prerelease marker, but is
an unqualified development revision, not a stable release.

That snapshot includes the reviewed curve API, fit progress/cancellation and
cosine metric. It does not include the follow-up public `UMAPProgress` initializer.
The verified cleanup will get a separate retained revision; this snapshot will
not be retargeted. Consumers need not wait for that cleanup to re-pin.

Dated reviews below remain historical evidence. See the
[required development reference checks](Docs/UMAP_Reference_Checks.md) for the
current UMAP/graph/PCA merge condition; it is separate from release qualification.

## Platform coverage decision — 2026-11-17

The user adopted the [0.4.0 platform coverage matrix](Docs/reviews/2026-11-17-platform-coverage/decision.md):
Apple silicon macOS and iOS are the primary targets, iOS requires version 18.0
or later, and an iOS 18 simulator run is required in addition to the current
simulator and physical-device scopes. The macOS 14, tvOS 17 and visionOS 1
deployment declarations and all existing required CI compilation checks remain.
Minimum-macOS, Intel, tvOS, visionOS and additional Catalyst runtime qualification
are explicitly deferred for 0.4.0; visionOS remains a future runtime target.

**Required local qualification is complete** for the iOS-18-targeting candidate
`510ea0259998219ef6c0656a2dfcec76190b518a`. [Recorded results](Docs/reviews/2026-11-17-platform-coverage/results.md)
cover native macOS, full Debug/Release suites and app matrices on exact iOS 18.0
and current iOS 27.0 simulators, and a physical iPhone 17 on stable iOS 27.2.
All consumer checkers return `accepted / known-failing / releaseAdmitted=false`;
full Release suites retain the known canary failure. Fresh PR/CI, hosted
numerical qualification and the other release gates remain open. Earlier
local/hosted results below retain their original candidate and evidence.

## Compiler support decision — 2026-11-10

The user adopted [bounded compiler qualification](Docs/reviews/2026-11-10-compiler-acceptance/decision.md)
for exact Xcode 27.2 build 27B52, prioritizing macOS and iOS consumers of the
shipped implementations and assessed by-value buffer patterns. Additional
descriptor/view inout handoffs through helper/conformer chains remain excluded
on affected combinations. Compiler failures stay visible; supported numerical
routes must pass. [Fresh frozen-candidate qualification](Docs/reviews/2026-11-10-compiler-acceptance/stable-candidate.md)
now satisfies the bounded local macOS, iOS simulator and physical-iOS scopes
for `af315c56c52de737f868fd969e6e1f5ae59eaa48`, with version `0.4.0` and
`prerelease=nil`. All checkers retain `compilerProfile=known-failing` and
`releaseAdmitted=false`. [Independent review and the tested verifier fixes](Docs/reviews/2026-11-10-compiler-acceptance/implementation-review.md)
are recorded. Hosted numerical qualification is now accepted for the exact
reviewed macOS and iOS simulator scopes; remaining release gates are still open.
The first hosted collection stopped at a fingerprint mismatch. After the
[byte-level investigation](Docs/reviews/2026-11-10-compiler-acceptance/hosted-compiler-identity.md),
the user adopted the exact hosted compiler pair for fresh qualification on
2026-11-16. The observed file differences are confined to signing records;
metadata collection itself supplies no numerical acceptance.

[Hosted collection, profile review and verification](Docs/reviews/2026-11-10-compiler-acceptance/hosted-evidence-2026-11-16.md)
are complete for tooling candidate `7c115cd927c76275b73a08b0e6fb178636a3c4b9`,
whose tracked compiled inputs match the frozen stable candidate. macOS and iOS
simulator observations match their prior literal expectations, including the
known compiler failures. The collect workflow deliberately reports failure;
[verification run 44030552128](https://github.com/gifton/VectorCore/actions/runs/44030552128)
passed both platform jobs and the required aggregate. Both reports return
`accepted / known-failing / releaseAdmitted=false`, with no verification problems.

Stable iOS acceptance additionally requires at least one app-hosted Release
consumer run on a physical iPhone or iPad. Minimum-OS, Intel and other declared
platform coverage were still undecided on November 10; the November 17
[platform decision](Docs/reviews/2026-11-17-platform-coverage/decision.md) above
now sets their requirements and deferrals. The [implementation plan](Docs/superpowers/plans/2026-11-10-compiler-acceptance.md)
was approved for native execution on 2026-11-10. No candidate or release is admitted by the policy decision.

## Historical integration preview: 0.4.0-alpha.1

The following sections describe the evidence recorded when the published alpha
was prepared. Subsequent qualification of the selected stable candidate is
linked above; changing its version constants has not created a stable tag.

**Status: experimental integration preview.** This snapshot is for fresh
downstream integration testing. Stable 0.4.0 admission remains open.

`${SOURCE_ROOT}` denotes the release-preparation checkout in this record.
The original working checkout remains separate.

This canonical location was reconciled on 2026-11-10. The local
relocation index (`${ARCHIVE_ROOT}/archive-index.json`)
maps earlier checkout and evidence roots to their present locations; the
repair record (`${ARCHIVE_ROOT}/worktree-path-repair-2026-11-10/README.md`)
records preservation checks. Retained originals of reports, manifests and
checker `captureRoots` preserve their original paths and hashes; current public
copies use [portable presentation and archive references](Docs/evidence-storage.md).
Use the relocation map and explicit current-root arguments when reading raw evidence. This path
repair does not refresh qualification or admit the alpha or stable release.

The immutable release tag is `v0.4.0-alpha.1`. Pin its exact
version in the consuming package:

```swift
.package(url: "https://github.com/gifton/VectorCore.git", exact: "0.4.0-alpha.1")
```

Then add `.product(name: "VectorCore", package: "VectorCore")` to the consuming
target's dependencies. In Xcode, use the same repository URL and select **Exact
Version: 0.4.0-alpha.1**. Retain `Package.resolved` so the resolved commit is
reviewable. SwiftPM's [dependency documentation](https://docs.swift.org/package-manager/PackageDescription/PackageDescription.html)
describes version requirements and the resolved file.

## Scope and known limitations

- The distinct-name by-value mutation API is included. Read the
  [SIMDStorage migration](Docs/SIMDStorage_Migration.md) before implementing a
  custom storage conformer or generic mutable-buffer caller.
- The tested Xcode 27.2 and 27.3 beta compilers still have the optimized stale-read
  defect. The original canaries remain unchanged. This alpha does not certify
  those compilers or arbitrary downstream unsafe-pointer compositions.
- For this integration exercise, element-mutating helpers must receive borrowed
  buffer descriptors/views by value. Do not add inout handoffs of those views or
  pointer-bearing wrappers inside callback/helper/conformer chains. The reviewed
  stdlib borrow-entry adapter is the specific starting point described in the
  [adopted policy](Docs/reviews/2026-11-10-compiler-acceptance/decision.md).
  The restriction is not a complete proof against compiler miscompilation.
- Fresh simulator validation targets arm64 iOS. The user deferred further
  Catalyst and Intel runtime qualification for this prerelease effort. Existing
  deployment declarations and required CI jobs remain unchanged; minimum-OS,
  physical-device, tvOS/visionOS runtime and controlled performance acceptance
  are still outside the current preview evidence.
- The [bounded native report](Docs/reviews/2026-11-06-bounded-native-qualification/README.md)
  belongs to the candidate before the version bump. Changing `Version.swift`
  creates a new compiled-input identity. Its pinned checker/profile are preserved;
  they do not automatically admit this alpha candidate.
- Pushing a tag does not trigger this repository's current CI workflow. Local
  validation and any hosted result must be reported separately; the preview does
  not claim final hosted-CI acceptance.

The alpha's 389 compiled-input manifest has SHA-256
`44a5f91ec8b6f182f5795192f9e0654775893c77ae91cf67e00ebdd21a4146eb`.
Only the four version constants in `Sources/VectorCore/Version.swift` changed
from the previously assessed compiled candidate. Raw captures remain local;
public clones contain dated summaries and provenance links, not those local files.

## Validation and integration

The fresh iOS 27.0 (24A434) arm64 simulator Release suite executed all three test
bundles: **1,829 passed, 163 skipped, one failed** out of 1,993 declared cases.
The sole failure is the unchanged optimized lost-store canary. All seven
`SIMDStorageMutationTests` pass. The 163 skip identities and reasons match the
earlier reviewed platform ledger. This is execution on the current simulator,
not iOS 17 or physical-device qualification.

The alpha preparation also passed **644 repository-policy tests**, validation
of nine GitHub workflow/template files and `git diff --check`.

Before publishing the tag, the clean SwiftPM consumer must resolve its exact
version and commit from an isolated Git repository and pass Debug/Release
mutation, version, normalization, dot and distance checks. Deliberately omitted
writes must produce the complete expected failures. The tag annotation records
that prepublication result. After publication, the same checks run with fresh
resolution/build directories against the canonical public repository; those
postpublication receipts remain separate from the immutable tag annotation.

The actual consuming application's repository/path has not yet been provided.
Connecting it is the next application-specific step; the independent consumer
does not establish integration into that application.

All new local artifacts for this effort are grouped under the existing evidence
root's `integration-preview/` subdirectory. No additional release worktree is
needed. This alpha record remains separate from the frozen stable candidate, remaining
+coverage decisions and final release admission.
