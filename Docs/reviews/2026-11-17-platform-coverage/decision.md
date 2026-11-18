# 0.4.0 platform coverage decision

**Adopted by the user on 2026-11-17.** The initial consumers prioritize macOS
and iOS; Intel is not needed, iOS should require version 18 or later, and
visionOS is useful for future consumers. The user approved an iOS 18 simulator
run as a release requirement and the remaining dispositions below.

| Platform | Deployment declaration | 0.4.0 runtime requirement |
| --- | --- | --- |
| Apple silicon macOS | macOS 14 | Qualify the recorded current macOS runtime; explicitly defer macOS 14 execution |
| iOS / iPadOS | iOS 18.0 | iOS 18 simulator, current simulator and at least one physical-iPhone/iPad Release consumer |
| Intel macOS | macOS 14 declaration retained | Runtime qualification deferred; existing Intel compilation check remains required |
| visionOS | visionOS 1 | Compilation remains required; runtime qualification is a future target |
| tvOS | tvOS 17 | Compilation remains required; runtime qualification deferred |
| Mac Catalyst | Existing iOS-derived deployment configuration | Compilation remains required; additional runtime qualification deferred |

Deployment declarations describe where the package can be selected; they are
not evidence that each runtime works. Deferred combinations remain unqualified
at runtime for this release. This decision does not remove APIs, architecture
branches or existing required CI jobs, and does not treat a compile pass as
numerical acceptance. Benchmark products remain macOS host tooling.

## Required evidence and current status

**Platform acceptance remains pending.** The iOS 18 requirement needs actual
test-body execution. Prefer an exact iOS 18.0 simulator and record its version,
build and architecture; a later 18.x run must not be called an 18.0-floor test.
Run the normally enabled Debug/Release suites and the app-hosted producer/client
matrix, reconciling individual outcomes, omitted-write controls and skips.
Refresh the existing native, current-iOS and physical Release scopes for the
changed candidate. A launch without complete numerical records does not pass.

Raising Package.swift's iOS minimum changes the frozen compiled-input identity.
Existing [local](../2026-11-10-compiler-acceptance/stable-candidate.md) and
[hosted](../2026-11-10-compiler-acceptance/hosted-evidence-2026-11-16.md) profiles
retain their original source, iOS 17 targets, receipts and outcomes. They do not
qualify the new candidate. New profiles require explicit evidence review.
The current collector targets iOS 18.0; historical verification may recognize
only its pinned iOS 17.0 or 18.0 target, with simulator/device separation intact.

The [compiler restrictions](../2026-11-10-compiler-acceptance/decision.md) and
exact adopted Xcode 27.2/27B52 compiler pairs still apply. Supported routes must
be numerically correct. Retained excluded compiler failures remain failures;
scoped reports continue to say `compilerProfile=known-failing` and
`releaseAdmitted=false`.

This decision disposes of the previously open platform scope; it does not close
execution, sanitizer, performance, hosted-CI or final release-reference gates.
No stable tag or publication is approved by it. Fresh raw captures/profiles stay
private under the [evidence-storage rules](../../evidence-storage.md).
