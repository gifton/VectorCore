# Frozen 0.4.0 candidate qualification — 2026-11-12

The retained local native, simulator-app and physical-iOS evidence satisfies
the adopted bounded compiler policy for candidate `af315c56c52de737f868fd969e6e1f5ae59eaa48`. Every checker
reports **accepted / known-failing / releaseAdmitted=false**. The full simulator
package suites were reconciled separately. Independent review and the tested fix pass are recorded in
[implementation-review.md](implementation-review.md). Actual hosted
qualification remains pending; this is not final release admission.

## Frozen inputs

The user selected `version="0.4.0"`, `prerelease=nil` and approved the existing
`VECTORCORE_TEST_EXTENDED=1` gate for the timing-only Euclidean batch benchmark.
Those are the only changed tracked compiled files relative to the adopted
alpha baseline. Production numerical bodies, compiler canaries and assertions
are unchanged. There are 389 tracked compiled inputs; the conservative local
source-tree manifest also includes three existing `.DS_Store` files.

[Full source manifest](stable-candidate-source-inputs.json), SHA-256:
`50d4861428d63368c309f0e86bfbf376f68a79df08db96a5b164aaaf37abe830`.
Every fresh capture records the selected commit and matching before/after
source manifests. Later tooling/documentation commits do not change that
compiled-input identity. Old alpha receipts were not rebound to it.

## Assessed scopes

| Scope | Result on the frozen candidate |
| --- | --- |
| Native macOS 27.0.1 (26A434), arm64, Debug | 1,946 passed, 47 skipped, no failures; 1,993 declared cases |
| Same native runtime, Release | 1,945 passed, 47 skipped; only the unchanged lost-store canary fails |
| Native Debug/Release producer × client | All 96 by-value observations pass; all 96 by-value omitted-write observations match their control oracle; separate witness and 12 migration contracts complete |
| Native original/residual diagnostics | All recorded groups match the exact reviewed expectations, including the retained optimized failures and complete omitted-write controls; required by-value rows remain strict |
| Native optional C-on/off × Debug/Release | Four build cells complete with separately checked numerical/route, tail and contradictory-request controls |
| Full iOS 27.0 (24A434) arm64 simulator suite, Debug | 1,829 passed, 164 skipped, no failures; all three test bundles and 1,993 declared cases reconciled |
| Same full simulator suite, Release | 1,828 passed, 164 skipped; only the unchanged lost-store canary fails |
| Simulator app Debug/Release producer × client | 52 app groups; 96 supported rows and 168 control rows match; 14 wrong diagnostic rows remain visible |
| Physical iPhone 16 Pro Max, iOS 27.2 (24B119), Release/Release | 13 app groups; all 24 supported rows and 42 control rows match; seven wrong diagnostic rows remain visible |

All scopes use exact Xcode 27.2 build 27B52 and frontend SHA-256
`448457da9a8a56741f2c8c4b8931e85b7377fc009c16918cd8ac1a8a3210ea1f`.
Profiles bind the additional compiler driver, SDK, architecture, target,
language/optimization, producer/client, runtime, app invocation and artifact
identities. The app scopes use the iOS 27.2 SDK and iOS 17 deployment targets;
they are execution evidence only for the recorded runtimes.

The native skip inventory gained exactly the approved timing benchmark.
For the full simulator suites, the only Debug-to-Release status difference is
the original canary changing from passed to failed. Compared with the preceding
failed alpha Debug attempt, the only status difference is that timing benchmark
changing from timeout failure to opt-in skipped. All skip identities/reasons,
raw failures, product hashes and result bundles are retained.

The canary still observes 2.0 (`40000000`) where it expects 1.0 (`3f800000`).
The original optimized CopyInoutReturnInside, LostStoreObserved and
ReturnInsideInout probes, concrete-inout consumer, and the three residual
inout rows remain numerically incorrect. They are diagnostics, not successful
numerical operations. No supported failure was waived.

## Evidence and preservation

[Exact profile hashes, roots and scoped reports](stable-candidate-results.json)
identify the canonical archived evidence. Public profiles are
[`macos-27.0.1-arm64-0.4.0-2026-11-12.json`](../../../Scripts/ci/qualification_profiles/macos-27.0.1-arm64-0.4.0-2026-11-12.json)
and [`ios-simulator-27.0-arm64-0.4.0-2026-11-12.json`](../../../Scripts/ci/qualification_profiles/ios-simulator-27.0-arm64-0.4.0-2026-11-12.json).
The physical-device profile remains in the private local archive, with its
digest published in the summary. Its raw signing/device identifiers are not
included in public reports or the hosted workflow.

Revalidate using `compiler_qualification.py`, the selected profile, its archived
evidence root and `/Users/goftin/dev/projects/VectorCore-archive/VectorCore-compiler-acceptance-2026-11-10-evidence/source-stable-af315c5` as `--source-root`.
Create a new output file. Original receipt paths are preserved and explicitly
remapped. All three archived scopes were revalidated; the full simulator suite
has separate raw streams, `.xcresult`, individual outcomes, product/source
before/after manifests and a complete file inventory under
`/Users/goftin/dev/projects/VectorCore-archive/VectorCore-compiler-acceptance-2026-11-10-evidence/ios-full-suites-stable-af315c5`.

The preserved source view includes every profile-bound source file, checked
against the original profile digest. Its first copy omitted older probe-source
paths; both validators rejected the incomplete copy. The missing files were
copied only after matching their existing pins, and the complete source-pin
closure is now checked mechanically. No receipt, oracle or profile digest was
changed to accept the archive.

[Preservation checks](preservation-result.json) confirm that dirty main's HEAD,
index, status and staged/unstaged diffs match the session baseline. All 25
protected historical evidence/checker/canary/CI files match their prior hashes.
The published alpha tag remains at `44c9ffaf145a59688f420023b28cdcf87c158070`.

## Remaining decisions

- The new [hosted workflow](hosted-qualification.md) implements the adopted
  two-stage review. Its actual hosted runs are still pending. GitHub requires
  workflow introduction on the default branch before manual dispatch; no push,
  merge, dispatch or publication is authorized by these local results.
- Minimum-OS runtime, Intel runtime, Catalyst/tvOS/visionOS scope, current
  sanitizer evidence, controlled performance and final release-reference
  acceptance remain open. Existing deployment declarations and legacy required
  CI are unchanged.
- The consumer application's repository has not been supplied. These results
  establish the named independent consumer compositions, not integration into
  an unidentified application or arbitrary external helpers/conformers.
- Independent review findings were addressed in a tested verifier fix pass.
  The corrected verifier accepts the unchanged archives. No `v0.4.0` tag or
  stable release has been created.
