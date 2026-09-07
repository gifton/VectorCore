# Public-readiness verification — September 7, 2026

[PR #41](https://github.com/gifton/VectorCore/pull/41) was reviewed, tested, and
merged. GitHub protections have been applied, read back, and verified to reject
a merge with pending required checks. [PR #45](https://github.com/gifton/VectorCore/pull/45)
records the final acceptance evidence; its checks provide the documentation
integration result.

Scope: [approved review](GitHub_Readiness_Review_2026-09-06.md) and
[implementation plan](GitHub_Hardening_Implementation_Plan.md), excluding release
naming, which the owner already corrected. Existing tags, releases, package
version, and license were not changed.

## Accepted implementation and hosted validation

Reviewed head: `dd7de41029328b5615d996ad4151d9ee8e092032`.
Merged main: `f4d5b494cd0d3d8697c0ea928b03c828ce864789`.
Both have Git tree `35a451501788e9cf2c0c915cd27c4f074690a0ef`, verified
locally and through GitHub's Git commits API. This exact-tree match and the
successful GitHub Actions `CI Required` on the reviewed head established the
precondition for enabling required checks after integration.

All checks on the reviewed head completed successfully:

| Check | Observed result |
| --- | --- |
| [Full CI](https://github.com/gifton/VectorCore/actions/runs/34149720311) | All required jobs and `CI Required` passed |
| Xcode 16.2 Debug / Release | Each reported 1,144 Swift Testing tests and 91 XCTest cases; zero failures |
| Xcode 26.3 Debug / Release | Each reported 1,144 Swift Testing tests in 168 suites and 91 XCTest cases; zero failures |
| Repository policy, lint, downstream consumer | Passed, including compilation and execution of README Quick Start |
| Apple platform compilation | Four simulators, Mac Catalyst, and Intel macOS library/test targets passed |
| Advisory performance smoke | Passed |
| [CodeQL](https://github.com/gifton/VectorCore/actions/runs/34149720298) | Actions, Swift, and C/C++ passed |
| [AddressSanitizer](https://github.com/gifton/VectorCore/actions/runs/34149720456) | 92 tests in 14 suites passed |
| [ThreadSanitizer](https://github.com/gifton/VectorCore/actions/runs/34149720456) | 34 tests in three suites passed |

Each full configuration includes 18 Swift Testing skips and 27 XCTest skips:
1,126 Swift Testing tests and 64 XCTest cases executed. Existing opt-in skips
were retained; no failing tests were quarantined. Hosted Swift Testing durations
were 1,181.926 s / 28.535 s on Xcode 16.2 Debug / Release and 594.216 s /
33.793 s on Xcode 26.3 Debug / Release.

CodeQL analyses for PR merge commit
`061e251087c1283cec58abb24c7f05b2ed6408ce` recorded 17 Actions rules,
27 Swift rules, and 58 C/C++ rules, each with zero results and empty error and
warning fields. The open-alert API returned an empty list. Swift and C/C++ used
successful manual build extraction. GitHub omitted PR file-coverage metadata;
recorded rule counts and successful extraction are the available evidence.

Post-merge runs: [CI](https://github.com/gifton/VectorCore/actions/runs/34152432291),
[CodeQL](https://github.com/gifton/VectorCore/actions/runs/34152432299), and
[sanitizers](https://github.com/gifton/VectorCore/actions/runs/34152432322).
Their completion is separate from the reviewed-head results above.

## Local validation and defects corrected

Local checks used Apple M3 Max, macOS 26.5.2, Swift 6.3.3, and Xcode 26.6
(17F113) with 26.5 SDKs. Final full Release passed 1,144 Swift Testing tests
in 168 suites (22.698 s). Focused Debug passed 22 tests in two suites.
Final ASan passed 92 tests in 14 suites; final TSan passed 34 in three suites.
Python CI policy, gate, hook, and watchdog tests passed 20/20. The Ruby validator
accepted nine GitHub YAML/template files. Pinned SwiftLint 0.65.1 exited
successfully with 293 existing warnings and zero errors.

Simulator and Catalyst compilation covered arm64 and x86_64. Intel macOS built
the library and all test targets. Compilation checks do not establish Intel,
simulator, or physical-device runtime coverage. Accelerate deprecation and
unused-result warnings remain. No Linux support is inferred.

Expanded testing exposed two areas needing source corrections:

- Intel macOS and Catalyst SDKs reject native `Float16`. Portable `UInt16` FP16
  storage and exponent classification now work across all widths, with native-only
  APIs excluded on Intel desktop targets. The native conversion flag remains
  opt-in. `MixedPrecisionRangeValidationTests` covers signed finite boundaries,
  infinities, NaNs, and supported native endpoints.
- MemoryPool's asynchronous bookkeeping followed by synchronous queue waits
  starved cooperative tasks. State operations now complete under `NSLock`.
  Regression checks also exposed late-handle allocation leaks, retention beyond
  the byte budget, and cleanup subtracting element counts from byte statistics.
  Handle teardown and byte accounting were corrected.

`Scripts/ci/check_memory_pool.py` compiles the production pool with a tracked
real-allocation boundary. Debug and Release pass all five modes: late-handle
lifetime, byte retention, exact cleanup accounting, ordinary saturation, and
strict saturation. Saturation uses 256 tasks × 1,000 iterations. Compiler/probe
process groups have external timeouts; watchdog tests cover child termination.
The original ordinary probe timed out and sampled stacks showed cooperative
workers blocked in `DispatchQueue.sync`. Strict mode did not reproduce on every
run and is not claimed as deterministic failure evidence.

Independent review found no actionable introduced defects in the source,
workflow, policy, or watchdog changes. Targeted sanitizers and zero scanner
findings do not establish general memory safety or absence of vulnerabilities.

## Live GitHub settings

API application and fresh readbacks on September 7 confirmed:

- [Baseline ruleset 10525559](https://github.com/gifton/VectorCore/rules/10525559):
  active on the default branch; PRs and resolved conversations required;
  deletion and force pushes blocked; strict, up-to-date `CI Required` from
  GitHub Actions app **15368** required. No bypass actors; current admin cannot bypass.
- [Review ruleset 22475449](https://github.com/gifton/VectorCore/rules/22475449):
  active on the default branch; one approval and code-owner review required;
  stale approvals dismissed and conversations resolved. Repository admins have
  a **pull-request-only** review exception for solo maintenance. This exception
  does not bypass the separate baseline required-check ruleset.
- [Tag ruleset 22475450](https://github.com/gifton/VectorCore/rules/22475450):
  active for all tags; updates and deletion blocked, no bypass actors, and no
  restriction on creating new tags. Existing tag names and contents were untouched.
- Actions restricted to an explicit allowlist, with full commit SHA pinning
  required. GitHub-owned and verified actions are not globally allowed.
- Private vulnerability reporting, Dependabot alerts, and automated security
  fixes enabled; security fixes read back `paused: false`.
- Secret scanning and push protection enabled. Immutable releases enabled for
  future publications; existing releases are not retroactively immutable.
- Workflow approval required for all external contributors; default workflow
  token permissions remain read-only and Actions cannot approve pull requests.
- Merged branches automatically deleted. CODEOWNERS API returned zero errors.

Allowed Action patterns are `actions/checkout@*`, `actions/upload-artifact@*`,
`maxim-lobanov/setup-xcode@*`, `github/codeql-action/init@*`, and
`github/codeql-action/analyze@*`. Workflow references use verified full SHAs;
SwiftLint installation verifies a fixed archive SHA-256. Under the new allowlist,
[CodeQL Actions job 101843555118](https://github.com/gifton/VectorCore/actions/runs/34154551922/job/101843555118)
successfully completed checkout, initialization, and analysis on PR #45.

Snapshots and enforcement precondition proof are stored locally under
`/private/tmp/vectorcore-github-hardening-20260907/`; downloaded hosted job logs
are under `/private/tmp/vectorcore-hosted-hardening/`. Temporary local evidence
is not a durable public artifact; linked GitHub runs and repository fixtures
provide the public record, subject to GitHub retention settings.

## Required-check enforcement exercise

The initial implementation's failed aggregate
[job 101669910994](https://github.com/gifton/VectorCore/actions/runs/34094838213/job/101669910994)
reported `required jobs did not succeed: test, platforms`. This established
fail-closed workflow aggregation.

At **2026-09-07 19:10:46 UTC**, a server-side merge request for documentation-only
[PR #45](https://github.com/gifton/VectorCore/pull/45), explicitly pinned to head
`44fac3170890deb49aec0d8336c19d789ad81d35`, returned **HTTP 405**:

> Required status check "CI Required" is expected.

The PR's CI jobs were queued or running and GitHub reported `BLOCKED`. The
request came from the repository admin, who could bypass only the separate
review rule. GitHub rejected the merge specifically for required CI; no failing
change was merged. This verifies server-side enforcement in addition to the
API configuration readback and fail-closed aggregate tests.

The subsequent documentation commit records this result. Its required CI must
succeed before integration; the [PR checks](https://github.com/gifton/VectorCore/pull/45/checks)
are the authoritative final run record. No bypass is added to finish this PR.

## Boundaries

This work does not audit all runtime source, historical commits, account 2FA or
recovery, GitHub App grants, credentials, or private application data. New
Dependabot updates remain ordinary reviewable PRs. Contributor and maintainer
guides document support limits without guaranteeing a response SLA. Unrelated
local `.antigravitycli/` and `GEMINI.md` were preserved.
