# Actual Swift 6.0-series qualification — 2026-11-04

> Hosted follow-up: [run 41946052293](hosted-ci.md) at `f779d92` passes matching Xcode 16.2 / Swift 6.0.3 Debug and Release full suites and all downstream contract checks. This closes the matching full-suite gap; the local failed attempts below remain distinct evidence.

**The local attempts below did not close the full Swift 6.0 suite gate.** Genuine Swift.org 6.0.3 compiled
and executed the library's downstream checks with the installed macOS 15.2 SDK.
Both full package test attempts failed before executing tests because importing
the installed Xcode 27.2 XCTest overlay independently crashes Swift 6.0.3. This
record qualifies the listed local compiler/library/consumer cells only; it does
not qualify the Xcode 16.2 full-suite cell.

## Inputs and signed origin

Tested candidate: `77defdf6161ab7600309d3432691fa9d388d088e`. All **387** baseline
inputs listed in `baseline-input-match.json` match accepted implementation
`d628a2c419300d37c8462d24b2b1069f71867f79`. No source, fixture, manifest,
optimization setting, dependency, public API, or platform declaration changed.
The existing Swift.org 6.4.2 qualification was accepted and was not rerun.

| Input | Exact identity |
| --- | --- |
| Compiler | `Apple Swift version 6.0.3 (swift-6.0.3-RELEASE)` |
| Download | [Official Swift.org 6.0.3 macOS package](https://download.swift.org/swift-6.0.3-release/xcode/swift-6.0.3-RELEASE/swift-6.0.3-RELEASE-osx.pkg) |
| Package SHA-256 | `764c3d5ba27473494206278c27d1bb0fdc3a8bca35ed902ed030f6699904e903` |
| Frontend SHA-256 | `ed347b34041039d1ec6c4f0f387d5df668222d44c9a40d344905bfa311609b41` |
| Driver SHA-256 | `ca8c822871ed2d39342b4437a7d7c8e420ace424aeb9804923fd640a2620ff9d` |
| Toolchain bundle | `org.swift.603202412101a`, version `6.0.3.20241210101` |
| Working SDK | `/Library/Developer/CommandLineTools/SDKs/MacOSX15.2.sdk` |
| SDKSettings.json SHA-256 | `2fa5c0ce1bbcd261b132b572b1a9eece3b5905b04640a44deae1a6a8812928fb` |
| Comparison SDK | `/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX27.2.sdk` |
| Comparison SDKSettings.json SHA-256 | `82165b15ee21419d4cffaba8895797391b908ac9499ba70f561f3e1bd16758d6` |
| Runtime/hardware | arm64, Apple M3 Max, 48 GiB, macOS 27.0.1 build `26A434` |
| Deployment target | `arm64-apple-macosx14.0` for package library and direct clients |

The [Swift.org release catalog](https://www.swift.org/install/macos/) links this
6.0.3 package. Before extraction or execution, `pkgutil --check-signature`
reported a trusted Apple-issued distribution certificate, trusted notarization,
timestamp `2024-12-11 23:08:18 +0000`, and
`Developer ID Installer: Swift Open Source (V9AUD2URP3)`, matching the
[official signature instructions](https://www.swift.org/install/macos/package_installer/).
After local extraction, strict `codesign --verify` succeeded for the frontend and
driver; the frontend identifies `Developer ID Application: Swift Open Source
(V9AUD2URP3)`. The package was expanded under the task evidence directory; no
toolchain was installed globally and Xcode selection remained unchanged.

Inventory found Xcode 27.2 build `27B52`, no installed alternative toolchains,
and existing CommandLineTools SDKs 15.2, 15.5, and 26.0. The SDK 15.2 Foundation
interface identifies Apple Swift 6.0.3 (`swiftlang-6.0.3.1.5`). This SDK generation
matches [Apple's Xcode 16.2 requirements table](https://developer.apple.com/xcode/system-requirements),
which lists Swift 6.0, macOS SDK 15.2, and supported host macOS 14.5–15.x.
The local combination uses the Swift.org compiler and existing CLT SDK; it is
not an installed, supported Xcode 16.2 host combination.

## Executed cells

| Cell | Observed result |
| --- | --- |
| Three retained pure-Swift probes, SDK 15.2, each `-Onone` and `-O` | 6 compile/run passes; every result is `1.0`, bits `3f800000` |
| Same probes, SDK 27.2 | 6 compile/run passes; same expected bits; no Foundation import |
| Foundation + Accelerate import, SDK 15.2 | Compile/link/run pass; marker `SDK_IMPORTS_OK 27` |
| Foundation + Accelerate import, SDK 27.2 | Compile exit 1: unsupported SDK arguments/headers before library involvement |
| Debug-built library, inherited/explicit Collection witnesses, each `-Onone` and `-O` client | 4 compile/link/run passes; marker present |
| Release-built library, same witness matrix | 4 compile/link/run passes; marker present |
| Retained ordinary numerical importer, Debug/Release library × `-Onone`/`-O` client | 4 compile/link/run passes; all 20 named checks pass |
| Required `consumer_smoke.py` with a fresh Release dependency build | Exit 0; 8 documentation programs and 21 contracts pass |
| Full Debug `swift test` | Exit 1 in 53.73s; 0 executed test bodies, 0 skip events, no framework completion summary |
| Full Release `swift test` | Exit 1 in 56.31s; 0 executed test bodies, 0 skip events, no framework completion summary |
| Two-line XCTest import control, SDK 15.2 | Compiler signal 6, same assertion as full suites; no VectorCore import |

The retained probes are `LostStoreObserved.swift`, `ReturnInsideInout.swift`, and
`ByValueReturnInside.swift`. Their printed values are checked explicitly by
`verify.py`; probe exit zero alone is insufficient. The public numerical importer
checks the pure-Swift canary, fixed-vector borrow, dynamic-vector borrow, accepted
adaptive reduced-result bits, and public SoA suffix preservation. Every matrix
case prints five `CHECK … PASS` lines. Collection fixtures include the committed
direct, generic, existential, iteration, empty, and copied-value checks.

Consumer smoke exercised **6 executable positive contracts, 2 compile-only
controls, and 13 expected compile failures**, in addition to the 8 documentation
programs. The Logger negative reports only line 10 on this compiler, as permitted
by the existing contract. All expected negative diagnostics are retained.
No downstream check was weakened or skipped.

## Full-suite environment boundary

The full commands used separate fresh Debug/Release scratch directories:

```sh
"$VC60_BIN/swift" test --build-system native --jobs 2 -c debug \
  --scratch-path "$EVIDENCE/full-debug/build" --sdk "$SDK15" -v
"$VC60_BIN/swift" test --build-system native --jobs 2 -c release \
  --scratch-path "$EVIDENCE/full-release/build" --sdk "$SDK15" -v
```

The command records expand every variable and retain `SWIFT_EXEC`, `SDKROOT`,
PATH, temporary/module cache paths, hashes, bounds, elapsed time and outputs.
Both commands stop with `ArgsToFrontendOptionsConverter.cpp, line 208`,
`!inputsAndOutputs->hasInputs()`, while the stack traverses
`extractSwiftInterfaceVersionAndArgs`. An independent source containing only
`import XCTest` and an integer declaration reproduces the assertion with SDK
15.2 and the installed testing framework search path. The installed XCTest
overlay interface identifies Swift 6.4.2 and contains newer module flags; its
SHA-256 is `88e4cb0a889c1d41df4fb1ef18da9c568f2fbf13038dc4736a6aa2fe69b4b26d`.
This isolates a failure independent of VectorCore. It does not establish which
specific newer interface flag triggers the older compiler assertion.

SDK 27.2 is separately incompatible with this compiler for the Foundation
control: diagnostics include unknown argument `-target-arch-variant` and module
feature `found_incompatible_headers__check_search_paths`. Passing pure-Swift
probes against that SDK does not establish framework or package compatibility.

An authenticated older Xcode was unavailable through the attempted official
endpoint: a bounded HEAD request for Apple's `Xcode_16.2.xip` returned HTTP 302
to `https://developer.apple.com/unauthorized/`. The full-suite requirement is a
matching Xcode 16.2 compiler, SDK 15.2, XCTest/framework bundle and supported
macOS 14.5–15.x host, executing the intended candidate. The existing hosted
`macos-15` / Xcode 16.2 Debug and Release jobs are the appropriate remaining
cells. No old remote result is substituted for this candidate.

## Evidence and verification

Raw task evidence: `/private/tmp/vectorcore-qualification-2026-11-04/task-1-evidence`.
`archive-inclusion.txt` selects existing relative paths; `archive-manifest.json`
hashes that selection. Bulk SDKs, compiler packages/payloads and build/module
caches are excluded from the archive selection and remain on disk. The signed
download URL/hash, signature transcript, source archive, exact library
modules/objects, executable probes, generated consumers, diagnostics, command
records and verification scripts are retained.

Principal records are `probe-summary.json`, `import-summary.json`,
`full-debug/result.json`, `full-release/result.json`,
`witness-debug/matrix/summary.json`, `witness-release/matrix/summary.json`,
`boundary/result.json`, `smoke/result.json`, and `verification.json`.
Substantial children use finite process-group-aware supervision from
`Scripts/ci/v03_process.py`, at most two compiler jobs, and per-cell output
directories. Full-suite deadlines are 1800s. Witness compiles/runs are bounded
by the committed driver's 60s per command. Import/probe compiles use 90s.

The smoke-only scratch wrapper selects the exact compiler/SDK and two build
jobs, retains temporary fixtures, and records each build/run and its outputs.
The repository driver and fixture selection are unmodified. Its adapter and
wrapper are included so the qualification harness can be reviewed. One initial
download-wrapper invocation failed before spawning a child because options
followed an argparse remainder argument; `harness-setup-failure.json` records
that setup error and the successful corrected invocation.

All measured cells retained unchanged tracked inputs before/after execution.
`VECTORCORE_TEST_EXTENDED=0` and `VECTORCORE_STRICT_PERF=0` match the accepted
baseline. Existing `cblas_sgemm` deprecation warnings remain in logs. The failed
suite attempts have no test skip results to assess. No optimizer pass,
assertion, test membership, sanitizer or CI gate was suppressed. No performance,
oldest-OS runtime, other architecture, Apple-device runtime, or full-suite
qualification is claimed by these local results.

Self-review verified the numerical output markers, 8 witness executions,
20 importer checks, 8 documentation bodies, all 21 contracts, both concrete
full-suite failures, source identity and all retained command-output hashes.
No independently demonstrated library defect required repair.
