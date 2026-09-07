# Public-readiness verification — September 7, 2026

**Status: implementation and rollout verification in progress.** This record
separates completed checks from pending acceptance. It does not establish that
hardening is complete or that the source is vulnerability-free.

Scope: [approved review](GitHub_Readiness_Review_2026-09-06.md),
[implementation plan](GitHub_Hardening_Implementation_Plan.md), and
[PR #41](https://github.com/gifton/VectorCore/pull/41). Release naming was already
corrected by the owner; this work does not move tags or change the version or
license. Final accepted commit and post-merge run links: **pending**.

## Local tests and repository checks

Local platform validation used Apple M3 Max, macOS 26.5.2, Swift 6.3.3,
Xcode 26.6 (17F113), and 26.5 SDKs. Hosted Xcode 16.2/26.3 results are separate
from this local evidence.

| Check | Observed result | Evidence |
| --- | --- | --- |
| Prior full Debug and Release baseline | Each reported 1,138 Swift Testing tests in 167 suites, including 18 skips, plus 91 XCTest cases, including 27 skips; zero failures | [September 6 verification](verification-topk-nan-contract-2026-09-05.md) |
| FP16 tests after portability fix, Debug | 221 Swift Testing tests in 51 suites, including nine skips, plus two XCTest cases; zero failures; Swift Testing took 376.081 s | `/private/tmp/vectorcore-fp16-debug-green.log` |
| FP16 tests after portability fix, Release | Same counts and skips; zero failures; Swift Testing took 22.792 s | `/private/tmp/vectorcore-fp16-release-green.log` |
| CI gate, YAML policy, and hook tests | 20 tests passed using `python3 -B -m unittest discover -s Scripts/ci/tests -v` | [Test sources](../Scripts/ci/tests) |
| Repository YAML/policy validation | `ruby Scripts/ci/validate_github.rb` passed; nine GitHub YAML/template files validated | [Validator](../Scripts/ci/validate_github.rb) |
| Initial pinned lint baseline | SwiftLint 0.65.1: 293 warnings, zero errors | `/private/tmp/vectorcore-swiftlint-hardening.json` |
| README consumer | Extracted Quick Start built and ran as a downstream Release consumer; initial hosted consumer job also succeeded | [Consumer fixture](../Scripts/ci/consumer_smoke.py), initial CI below |

Counts include reported skips; the prior full baseline executed 1,120 Swift
Testing tests and 64 XCTest cases. It predates the new portability tests and is
not evidence of a full run on the final hardening commit. Latest full-suite and
lint results on the final revision: **pending**.

Active workflow Actions use full commit SHA references with version comments.
The validator checks pin format, explicit token permissions, checkout credential
persistence, and gate wiring. SwiftLint installation checks a fixed archive
SHA-256. Final Action inventory/provenance and live repository SHA enforcement:
**pending final acceptance readback**. Passing local gate tests demonstrates
aggregation behavior, not GitHub branch-protection enforcement.

## Apple platform compilation and portability

The following commands exited zero locally. These are compilation checks, not
simulator or device runtime tests.

| Destination | Configuration / output | Log under `/private/tmp/` |
| --- | --- | --- |
| iOS Simulator | Release, arm64 and x86_64 modules | `vectorcore-platform-ios-simulator.log` |
| tvOS Simulator | Release, arm64 and x86_64 modules | `vectorcore-platform-tvos-simulator.log` |
| watchOS Simulator | Release, arm64 and x86_64 modules | `vectorcore-platform-watchos-simulator.log` |
| visionOS Simulator | Release, arm64 and x86_64 modules | `vectorcore-platform-visionos-simulator.log` |
| Mac Catalyst, after fix | Release, arm64 and x86_64 modules | `vectorcore-platform-mac-catalyst.log` |
| Intel macOS, after fix | Library and all test targets built; test executable verified as Mach-O x86_64 | `vectorcore-intel-tests-build.log` |

Simulator and Catalyst builds used `xcodebuild -scheme VectorCore
-configuration Release -destination 'generic/platform=<destination>'
-derivedDataPath /private/tmp/vectorcore-platform-audit/<name>
CODE_SIGNING_ALLOWED=NO build -quiet`. Catalyst's destination was
`generic/platform=macOS,variant=Mac Catalyst`. The Intel macOS command was
`swift build --build-tests --triple x86_64-apple-macosx14.0 --scratch-path
/private/tmp/vectorcore-intel-tests-build`.

The first Catalyst build failed because the Intel desktop SDK marks `Float16`
and its `Sendable` conformance unavailable. Unconditional uses in
`MixedPrecisionKernels.swift` caused the root error and follow-on initializer
and concurrency diagnostics. A minimal SDK typecheck proved that an unavailable
annotation alone does not make `Float16(value)` compile in the method body.

The narrow fix stores the finite maximum as `Float`, classifies FP16 exponent
bits for all three `validateRange` methods, and excludes native-only
`detectOverflow(value:) -> Float16?` on Intel macOS and Intel Mac Catalyst.
Portable `UInt16` storage, `canRepresent`, and `validateBatch` remain available.
The native conversion compiler flag remains opt-in; no dormant conversion route
was enabled. Existing tests use portable storage sizes and guard native-only
operations. [Range validation tests](../Tests/ComprehensiveTests/MixedPrecisionRangeValidationTests.swift)
cover all widths, signed finite boundaries, infinities, NaN encodings, and native
conversion endpoints on supported targets. An Intel iOS Simulator compile probe
confirmed that the guard retains native APIs there.

The four simulator builds preceded this narrow source fix; Catalyst and Intel
macOS builds followed it. Final hosted platform coverage remains pending.
Existing Accelerate `cblas_sgemm` deprecation and unused-result warnings remain.
No Linux support or device-runtime coverage is inferred from these checks.

## Initial hosted evidence and exposed failures

[Initial CI run 34094838213](https://github.com/gifton/VectorCore/actions/runs/34094838213)
completed repository checks, lint, README consumer, and advisory performance
smoke successfully. Platform compilation exposed the same Intel Catalyst
`Float16` failure reproduced locally. The initial full-test matrix did not
establish a complete green run: jobs stalled or were cancelled during
investigation. A standalone MemoryPool saturation probe reproduced executor starvation;
the correction is described below. Latest full matrix results: **pending**.

[Initial sanitizer run 34094838416](https://github.com/gifton/VectorCore/actions/runs/34094838416)
passed targeted AddressSanitizer coverage: 74 Swift Testing tests in 13 suites,
6.785 s. The initial thread-sanitizer job was cancelled and provides no passing
result. [Initial CodeQL run 34094838331](https://github.com/gifton/VectorCore/actions/runs/34094838331)
completed Actions analysis successfully; both Swift and C/C++ analysis builds exposed the
Intel desktop `Float16` issue. Latest Swift/C/C++ analysis, alert triage, and
thread-sanitizer acceptance: **pending**.

Downloaded initial jobs/logs are in
`/private/tmp/vectorcore-hosted-hardening/`; the ASan success is recorded in
`101656000110.log`. These observations apply to those initial runs, not the
latest PR revision. A passing targeted sanitizer or scanner is not a general
memory-safety or security audit.

## Memory-pool regression evidence

The original pool queued asynchronous bookkeeping and synchronously waited for
that queue from cooperative tasks. A standalone executable compiling the actual
production source timed out under 256 tasks × 1,000 acquire/return/quiesce
iterations. A stack sample showed cooperative workers blocked in
`DispatchQueue.sync` from `acquire` or `quiesce`. Strict executor mode did not
reproduce on every run; the ordinary saturation failure and sampled stacks are
the diagnostic evidence.

State operations now finish synchronously under one `NSLock`. The same harness
also caught allocations leaked by handles released after their weak pool had
expired, retention exceeding its byte budget, and cleanup subtracting element
counts from byte statistics. Handle teardown now frees directly when its pool
has expired, and retention/cleanup use stored byte counts.

`python3 Scripts/ci/check_memory_pool.py --configuration debug` and its Release
variant pass all five modes: late-handle lifetime, byte retention, exact cleanup
accounting, ordinary saturation, and strict saturation. The harness compiles the
production pool source with a tracked real-allocation boundary; it does not copy
the pool implementation. Compiler and probe timeouts terminate their private
process groups. Watchdog tests detect a child process escaping the former timeout
implementation. These checks run automatically in each CI test configuration.

The full local Debug run after synchronization/lifetime fixes passed 1,142 Swift
Testing tests in 168 suites (930.197 s under local contention). That run predates
the two additional byte-accounting tests. Address Sanitizer passed 90 tests in
14 suites and Thread Sanitizer passed 32 tests in three suites on that revision.
Final focused Debug tests passed 22 tests in two suites, including the two
byte-accounting regressions. Final Release and sanitizer results: **pending
final recording**. Independent review found no actionable introduced defects in
the portability, synchronization, accounting, or watchdog changes.

The initial hosted `CI Required` job
[101669910994](https://github.com/gifton/VectorCore/actions/runs/34094838213/job/101669910994)
failed with `required jobs did not succeed: test, platforms`, verifying that
unsuccessful dependencies do not produce a green aggregate. Live merge-rule
enforcement still requires the separate post-integration readback below.

## Live settings and remaining rollout

Initial before/after snapshots are stored outside the repository at
`/private/tmp/vectorcore-github-hardening-20260907/`. The initial application and
readback recorded:

- Private vulnerability reporting enabled.
- Dependabot vulnerability alerts and automated security fixes enabled.
  The initial fixes snapshot reported `paused: true`; the later
  `automated-security-fixes-latest.json` readback confirms `enabled: true` and
  `paused: false`. The initial inactivity pause has cleared.
- Workflow approval required for all external contributors.
- Immutable releases enabled for future publications; this is not retroactive.
- Automatic deletion of merged branches enabled.
- Secret scanning and secret push protection remained enabled.

The following acceptance evidence is still **pending** in this record:

- Latest passing hosted full matrix, consumer/platform checks, and `CI Required`;
  reviewed scanner/sanitizer findings and any narrowly documented limitations.
- Integration of verified workflows into `main`, final commit identity, and
  post-merge verification.
- Effective baseline rules requiring an up-to-date PR, resolved conversations,
  and `CI Required` from GitHub Actions, without baseline bypass actors.
- Separate owner-review rules with the approved PR-only solo-maintainer bypass.
- Version-tag update/deletion protection, allowed Actions restriction, full-SHA
  enforcement, and final CODEOWNERS/API/settings readbacks.
- A documentation-only PR and proof that pending or failing required checks
  prevent merge without merging a deliberately failing change.

This work does not audit all runtime source, historical commits, account 2FA or
recovery, GitHub App grants, credentials, or private application data. Local
checks and initial settings do not establish those controls. Unrelated
`.antigravitycli/` and `GEMINI.md` remain outside this work.
