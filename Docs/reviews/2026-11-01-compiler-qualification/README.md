# VectorCore compiler qualification — 2026-11-01

Candidate: `379e610f4365e820454c8c5328c4f87dfaa2031b` on `codex/audit-top-six-2026-10-27`.

**Outcome:** a usable macOS arm64 qualification path exists for this candidate with the exact Swift.org release toolchain and SDK below. Full Debug, full Release and the ordinary downstream client pass. Default Xcode qualification remains failed; this is not an all-platform 0.4.0 release sign-off.

## Finding

The official Swift.org **Swift 6.4.2 release** passes the two retained Array/inout-buffer reproducers under normal optimization on this macOS arm64 host. Xcode 27.2's bundled compiler, which also reports version 6.4.2, still returns stale values in fresh controls. Compiler version numbers alone are insufficient to select the tested compiler.

The repair branch's source, assertions, expected-issue conditions, platform declaration and optimizer settings remain unchanged. The original Xcode Release failure is retained. The absence of a passing released compiler is resolved for this local qualification configuration. Adoption in hosted CI, downstream integration and other platform/runtime qualification remain separate work.

## Toolchain identity

- Official package: https://download.swift.org/swift-6.4.2-release/xcode/swift-6.4.2-RELEASE/swift-6.4.2-RELEASE-osx.pkg
- Package SHA-256: `0a7d8489900d21dbf5f0187cb570e646a4c0226d0ddef2cbde8c7f54dfe79736`.
- Verified signer: `Developer ID Installer: Swift Open Source (V9AUD2URP3)`; Apple notarization trusted; signed 2026-10-28 06:00:23 UTC.
- Toolchain bundle identifier: `org.swift.642202610261a`; bundle version `6.4.2.20261026101`.
- Passing compiler version string: `Apple Swift version 6.4.2 (swift-6.4.2-RELEASE)`.
- Failing Xcode compiler version string: `Apple Swift version 6.4.2 (swiftlang-6.4.2.1.9 clang-2100.4.1.1)`.
- Xcode: 27.2 (27B52); macOS SDK 27.2; deployment target `arm64-apple-macosx14.0`; tested runtime macOS 27.0.1 (26A434). This is not a macOS 14 runtime test.

The signed package was extracted into a temporary directory. No global toolchain installation or Xcode selection was changed. The initial sandboxed signature check reported invalid; the same package verified outside the sandbox before any downloaded executable was run. Both observations are recorded.

## Executed checks

| Check | Observed result |
|---|---|
| Fresh Xcode original repro | `-Onone`: 1; `-O`: stale 2. |
| Fresh Xcode inside-borrow/inout repro | `-O`: stale 2. |
| Official original repro | `-Onone`, `-O`, `-Osize`, `-O -whole-module-optimization`: 1. |
| Official inside-borrow/inout repro | `-O`, `-Osize`, `-O -whole-module-optimization`: 1. |
| Official by-value control | `-O`: 1. |
| Ordinary optimized downstream client | All five checks pass: pure Swift canary, fixed-vector borrow, dynamic-vector borrow, adaptive reduced-result bits, public SoA suffix. |
| Full Release | Exit 0; 196 XCTest cases reported, 28 skipped, zero failures; 1,742 Swift Testing tests in 262 suites, 18 skip events, zero issues. The unchanged hard canary passes. |
| Full Debug | Exit 0; the same reported counts and skips, zero failures/issues; the hard canary passes. |

The standalone sources print observations and exit normally. Acceptance compares `1.0` / `3f800000`, not merely exit status. The by-value control is a control for the tested shape, not a general compiler-safety claim.

## Scope and remaining work

This establishes only the executed host/configuration scope. Device and Intel runtime, the pinned hosted Xcode cells, VectorAccelerate CPU/GPU integration and controlled performance acceptance remain separate. Existing standard test skips remain visible in the final counts. No RLE-disable flag or new expected issue is used for the official toolchain runs.

The compiler investigation's scanner gap VC-C002 still requires its separate repair. Its inside-borrow repro is included in this qualification matrix. The Xcode diagnostic's attribution to a particular missing upstream fix is not established by these results.

The [Swift.org installation documentation](https://www.swift.org/install/macos/package_installer/) states that App Store submissions must use Swift bundled with Xcode. Consequently this official-toolchain validation is not an App Store build qualification or evidence that Xcode's bundled compiler has been repaired.

## Reproduction and evidence

[Validation summary](validation-summary.json) records final counts, commands, identities and raw log hashes. [Probe sources](probes/) retain the small standalone inputs in the repository. Raw command records, logs, the public client, probe binaries, package signature and toolchain metadata are retained at `/Users/goftin/dev/projects/VectorCore-compiler-qualification-2026-11-01-evidence`; [evidence-manifest.json](evidence-manifest.json) hashes that bounded archive. Compiler packages, build caches and complete test bundles are reproducible inputs/outputs rather than archived evidence. The package URL and SHA-256 above identify the download.

Both full runs use SwiftPM's **native** build system, `VECTORCORE_TEST_EXTENDED=0`, `VECTORCORE_STRICT_PERF=0`, three jobs and the explicit SDK. Optional C kernels are not enabled. The initial Release command used `--disable-sandbox` for SwiftPM's manifest/plugin execution in the agent environment; it did not disable an optimizer pass. Debug and the successful client used SwiftPM's normal sandbox outside the agent sandbox. The first client attempt failed at `sandbox_apply` before running the client and is retained separately. Existing compiler/build-system warnings remain; printed test timings are not controlled performance measurements.

The verbose Release log identifies the official compiler for all eight Swift modules, each with `-O` and without `DEBUG`. Both client modules also use that compiler and normal optimization. Debug and the successful client record clean before/after checkout states at the candidate; the Release result records the same revision, and the enclosing checkout checks remain clean. No library source, package manifest or test changed during qualification.

A fresh reviewer independently inspected source, compiler identities, Release flags, the hard-canary pass and the client result, and found no issue invalidating this bounded qualification. The reviewer did not rerun tests. Debug was pending at its review cutoff; the lead subsequently verified the final exit, counts and clean source state. The review explicitly excludes other platforms, App Store eligibility, performance, GPU integration, scanner repair and attribution to a particular upstream fix. The October 30 evidence remains unchanged.


## Reproduce the qualification

Download the exact signed package linked above using the [official installation instructions](https://www.swift.org/install/macos/package_installer/). Verify its checksum and signature before execution. Installing the toolchain is separate from changing Xcode's default selection. The original run extracted the package with `pkgutil --expand-full` and used its payload directly.

From the repository root, set `VC_TOOLCHAIN` to the installed `.xctoolchain` directory or the extracted package's `Payload` directory. Use the same SDK for a reproduction of this record; selecting another SDK is a new qualification cell.

```sh
VC_TOOLCHAIN="/absolute/path/to/the/verified/toolchain"
VC_SDK="/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX27.2.sdk"
VC_CHECK_DIR="$(mktemp -d)"
export SWIFT_EXEC="$VC_TOOLCHAIN/usr/bin/swiftc"
export PATH="$VC_TOOLCHAIN/usr/bin:$PATH"
export VECTORCORE_TEST_EXTENDED=0 VECTORCORE_STRICT_PERF=0

"$SWIFT_EXEC" --version
shasum -a 256 "$VC_TOOLCHAIN/usr/bin/swift-frontend"
```

Require the `swift-6.4.2-RELEASE` identity and frontend SHA-256 `956438a61893525d7ce28c21b6634093f15b0c4bae7fe6eb7d98fba0c1c5e301`; `toolchain-identity.json` in the raw archive and the tracked summary also identify the driver, Clang and Info.plist. A different hash is not covered by this record even if the short version is 6.4.2.

Compile each retained probe with `-O`, `-Osize` and `-O -whole-module-optimization`, using a separate output name for each configuration. For example:

```sh
"$SWIFT_EXEC" -swift-version 6 -target arm64-apple-macosx14.0 \
  -sdk "$VC_SDK" -module-cache-path "$VC_CHECK_DIR/module-cache" -O \
  Docs/reviews/2026-11-01-compiler-qualification/probes/ReturnInsideInout.swift \
  -o "$VC_CHECK_DIR/inside-borrow"
"$VC_CHECK_DIR/inside-borrow"
```

The required observation is exactly `LOST_STORE observed=1.0 bits=3f800000 expected=1.0`. Exit zero with `2.0` / `40000000` is a failure. Repeat for `LostStoreObserved.swift` and the `ByValueReturnInside.swift` control. The recorded matrix above is the authority for configurations actually executed in this run.

Run both full suites with independent fresh scratch directories:

```sh
"$VC_TOOLCHAIN/usr/bin/swift" test --build-system native --jobs 3 \
  -c debug --sdk "$VC_SDK" --scratch-path "$VC_CHECK_DIR/debug"
"$VC_TOOLCHAIN/usr/bin/swift" test --build-system native --jobs 3 \
  -c release --sdk "$VC_SDK" --scratch-path "$VC_CHECK_DIR/release"
```

Retain the full output and require the hard `LostStoreCanaryTests.literalArrayStoreIsObservedAfterOutOfLineClosure` assertion to pass. Do not replace the earlier Xcode failure with this separate result. The ordinary-client source and exact invocation are in the durable raw archive; its five checks and binary hash are in the tracked validation summary.
