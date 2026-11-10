# VectorCore release status

This is the entry point for the active release-preparation work. Dated reviews
are historical evidence; directories ending in `-evidence` contain local records
and build artifacts.

## Integration preview: 0.4.0-alpha.1

**Status: experimental integration preview.** This snapshot is for fresh
downstream integration testing. Stable 0.4.0 admission remains open.

The active local checkout is
`/Users/goftin/dev/projects/VectorCore-archive/VectorCore-fixes-2026-10-27-84798d3`.
The earlier `dev/gsuite/VSK/VectorCore` checkout is not the release-preparation copy.

This canonical location was reconciled on 2026-11-10. The local
[relocation index](/Users/goftin/dev/projects/VectorCore-archive/archive-index.json)
maps earlier checkout and evidence roots to their present locations; the
[repair record](/Users/goftin/dev/projects/VectorCore-archive/worktree-path-repair-2026-11-10/README.md)
records preservation checks. Historical reports, manifests and checker
`captureRoots` retain their original paths and hashes. Use the relocation map
and explicit current-root arguments when reading that evidence. This path
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
  [boundary proposal](Docs/reviews/2026-11-06-workaround-qualification/qualification-proposal.md).
  This testing convention is not a complete proof against compiler miscompilation.
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
needed. Final stable support wording and release admission remain separate work.
