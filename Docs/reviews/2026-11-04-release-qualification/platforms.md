# Apple platform and architecture qualification — 2026-11-04

All eight workflow Release library compilation cells and the Intel library/test compilation passed. The iOS 27.0 arm64 public-library smoke passed its specific assertions. **Xcode Release numerical qualification fails:** the independent lost-store canary returns stale `2.0` / `40000000` instead of `1.0` / `3f800000` on that same simulator. The identical canary compiled with the qualified Swift.org 6.4.2 compiler returns the expected value. These results do not establish complete platform or release qualification.

## Source, tools and inventory

Source: clean isolated worktree `/Users/goftin/.codex/worktrees/f86f/VectorCore` at `77defdf6161ab7600309d3432691fa9d388d088e`. Its implementation inputs match the carried qualification base `d628a2c419300d37c8462d24b2b1069f71867f79`; the intervening plan commit changes documentation. Each command's `*-source.json` records before/after tracked Sources, Tests, Benchmarks, Package.swift, workflow and CI-script hashes. No source, manifest, workflow, public API, platform floor or optimizer setting was changed by this task.

- Host: Apple M3 Max, arm64, macOS 27.0.1 (26A434).
- Selected Xcode: `/Applications/Xcode.app`, 27.2 (27B52). Discovery under `/Applications` and `~/Applications` found only this Xcode application. Initial/final `xcode-select -p` agree; commands used task-local `DEVELOPER_DIR` without changing global selection.
- Xcode Swift: `Apple Swift version 6.4.2 (swiftlang-6.4.2.1.9 clang-2100.4.1.1)`; frontend SHA256 `448457da9a8a56741f2c8c4b8931e85b7377fc009c16918cd8ac1a8a3210ea1f`. Clang: Apple clang 21.0.0 (`clang-2100.4.1.1`). This is the compiler used for all library/platform and Intel compilation cells.
- Installed SDK inventory: macOS, iOS/device+simulator, tvOS/device+simulator, visionOS/device+simulator, watchOS/device+simulator and DriverKit, all 27.2. Only declared package destinations are built here. `sdk-list.stdout`, `sdk-inventory.stdout`, `sdk-settings/` and `sdk-compiler-hashes.json` retain paths, settings, available architectures and hashes.
- The iOS/tvOS/visionOS simulator SDKs advertise arm64 and x86_64; macOS/Catalyst SDK targets advertise x86_64, x86_64h, arm64, arm64e and arm64e.x1. Compiled product slices below are measured separately from SDK capabilities.
- The only installed simulator runtime is iOS 27.0 (24A434), identifier `com.apple.CoreSimulator.SimRuntime.iOS-27-0`, available for arm64. No tvOS or visionOS runtime is installed. Initial/final runtime JSON is retained; no runtime was installed or downloaded.
- Package.swift declares macOS 14, iOS 17, tvOS 17, visionOS 1, tools version 6.0, and two library products: `VectorCore`, `VectorCoreBenchmarking`. `vectorcore-bench` is an executable and is outside the library-product matrix.

All compiler commands were sequential within this worker and limited to two jobs. `VECTORCORE_TEST_EXTENDED=0` and `VECTORCORE_STRICT_PERF=0` match the accepted baseline. No timing here is performance acceptance.

## Workflow compilation matrix

The destination list and command structure come from the existing `.github/workflows/ci.yml` platforms job. Products were enumerated by the existing `Scripts/ci/package_products.py`; no hand-maintained replacement list was used for execution. Each product/destination received a new derived-data root, so even dependent products were rebuilt. The only command adaptation was `-jobs 2` and task-owned scratch paths:

```text
xcodebuild -scheme PRODUCT -configuration Release
  -destination generic/platform=DESTINATION -derivedDataPath FRESH_CELL_ROOT
  -jobs 2 CODE_SIGNING_ALLOWED=NO build -quiet
```

`ios`, `tvos`, `visionos`, `catalyst` below mean `iOS Simulator`, `tvOS Simulator`, `visionOS Simulator`, `macOS,variant=Mac Catalyst` respectively.

| Product | Destination | Compile / elapsed | Measured object slices |
|---|---|---|---|
| VectorCore | ios | pass / 40.11s | arm64, x86_64 |
| VectorCore | tvos | pass / 47.13s | arm64, x86_64 |
| VectorCore | visionos | pass / 52.10s | arm64, x86_64 |
| VectorCore | catalyst | pass / 42.93s | arm64, x86_64 |
| VectorCoreBenchmarking | ios | pass / 40.49s | arm64, x86_64 |
| VectorCoreBenchmarking | tvos | pass / 40.45s | arm64, x86_64 |
| VectorCoreBenchmarking | visionos | pass / 40.00s | arm64, x86_64 |
| VectorCoreBenchmarking | catalyst | pass / 41.10s | arm64, x86_64 |

`product-inventory.stdout` records paths, byte counts, SHA256, `lipo` architectures and `vtool` build headers for all produced product/C objects. Headers show SDK 27.2 and minimum 17.0 for iOS/tvOS/Catalyst, 1.0 for visionOS. These deployment settings do not prove execution at those minimum versions. The platform policy suite also passed all 15 tests (`platform-coverage-policy.*`).

These are local Xcode 27.2 compile controls, distinct from the hosted workflow's pinned Xcode 26.3 job. No hosted execution occurred. Existing Accelerate deprecation warnings and Xcode generic-destination warnings remain visible; no warning or assertion was suppressed.

## Intel macOS

The exact workflow command, with the two-job cap and new scratch root, passed in 110.54s:

```text
xcrun swift build --build-tests --triple x86_64-apple-macosx14.0
  --scratch-path FRESH_INTEL_ROOT --jobs 2
```

This uses the workflow's default Debug configuration. `intel-product-inventory.stdout` confirms x86_64-only library/C objects, benchmark executable and all three test executables (`MinimalTests`, `ComprehensiveTests`, `BenchmarkContractTests`). Their minimum build version is macOS 14.0. The library objects report SDK 27.2; the final test/CLI executable load commands report SDK 14.0, retained exactly as emitted by this command. The retained build graph identifies the selected `macosx27.2` SDK.

Runtime is **untested**. The bounded support probe `/usr/bin/arch -x86_64 /usr/bin/uname -m` exits 1 with `Bad CPU type in executable`; the Rosetta package receipt probe also reports no receipt. No Rosetta installation or Intel test execution was attempted. The compile logs retain existing deprecation/test-style warnings.

## Executed iOS library smoke and compiler control

A task-owned iPhone 18 Pro simulator (`E2B7379B-B36C-4DCC-9F90-167A1181A860`) used the already installed iOS 27.0 runtime. It was shut down after checks; no device provisioning was performed. `PlatformSmoke.swift` imports both public library products and links the freshly compiled iOS Release objects. Its optimized external client checks:

- fixed-vector addition at both ends and dot product;
- dynamic-vector magnitude and normalization;
- a 3×2 QR reconstruction residual through both LAPACK and Swift providers;
- a symmetric 2×2 eigenvalue result through both providers;
- benchmarking environment metadata on arm64.

Floating-point checks use explicit absolute tolerances. The client exited 0 and printed `VECTORCORE_PLATFORM_SMOKE_PASS platform=ios-os27.0.0 arch=arm64 version=0.3.3`. Its Mach-O header identifies IOSSIMULATOR, minOS 17.0, SDK 27.2. This is bounded evidence for those assertions, not a full simulator suite or a qualification of other APIs, optional C-kernel modes, devices or old OS versions. The package still reports its current 0.3.3 source version; this task did not change version declarations.

The separate pure-Swift canary is a byte-identical copy of the already retained `Docs/reviews/2026-11-01-compiler-qualification/probes/LostStoreObserved.swift`, with no VectorCore import. **Canary source SHA256: `d6bda153f1ed5d79a393d04ea385690fcdef616f4391d73ab47b2ed2d9221326`. Required observation: `1.0` / `3f800000`.**

| Compiler build | Optimization | Observed value / bits | Process exit | Numeric acceptance |
|---|---|---|---|---|
| Xcode Swift 6.4.2 (`swiftlang-6.4.2.1.9`) | `-Onone` | `1.0` / `3f800000` | 0 | pass |
| Same Xcode compiler | `-O` | **`2.0` / `40000000`** | 0 | **FAIL** |
| Qualified Swift.org Swift 6.4.2 (`swift-6.4.2-RELEASE`) | `-O` | `1.0` / `3f800000` | 0 | pass |

All three target `arm64-apple-ios17.0-simulator`, use the same iPhoneSimulator27.2 SDK, and execute on the same simulator/runtime. `simulator-canary-assessment.json` records exit 1 for the failed installed-compiler acceptance check despite the canary process's zero exit. The paired assessment records actual/expected strings and acceptance explicitly.

The qualified Swift.org frontend SHA256 is `956438a61893525d7ce28c21b6634093f15b0c4bae7fe6eb7d98fba0c1c5e301`, verified before this paired compile. Its existing `usr/lib/swift/iphonesimulator` resources were present; no support was installed. Its version and verbose compile/link invocation are retained. Prior package-origin/signature qualification remains in [the compiler qualification record](../2026-11-01-compiler-qualification/README.md). The paired pass covers this canary only: the eight library cells were built with Xcode, and no complete Swift.org simulator library build was substituted. This diagnostic also makes no App Store toolchain/build eligibility claim.

## Retained failures, boundaries and evidence

The first public smoke compile, using `xcrun swiftc` without selecting an xcrun SDK, emitted a linker sysroot warning. That attempt/binary is retained and was not executed. The accepted retry used `xcrun --sdk iphonesimulator swiftc` with the explicit SDK and emitted no warning. A preliminary simulator `/usr/bin/uname -a` probe failed to spawn (111); the installed-runtime inventory and the successfully executed clients establish the runtime identity. The qualified `--version` probe emitted a default macOS-target versus simulator-SDK warning because that identity command inherited the paired compile SDK; the actual verbose compile targets the simulator correctly. All attempts remain in the evidence.

The numerical Release canary failure is material and remains unfixed. This task establishes eight compile cells, Intel test compilation, the listed iOS smoke assertions and the paired canary observations. It leaves tvOS/visionOS runtime, Catalyst runtime, Intel runtime, all physical devices, minimum-OS execution, full simulator suites, hosted Xcode 26.3 execution and complete Swift.org simulator library qualification untested. The accepted native arm64 full-suite baseline was not duplicated.

Raw evidence: `/private/tmp/vectorcore-qualification-2026-11-04/task-3-evidence`. `qualification-summary.json` contains machine-readable results. Every substantial command has argv, cwd, selected nonsecret environment, deadline, elapsed/exit/timeout and stdout/stderr paths/hashes through the existing process-group-aware `v03_process.run` wrapper. Source snapshots and compiler/SDK hashes supplement those records. `build-graphs/` retains compressed command graphs, build requests and response files; `smoke-inputs/` retains linked library objects/modules. Binaries and fixture sources are retained outside the scratch roots.

`archive-inclusion.txt` explicitly selects existing evidence-relative paths and excludes bulk scratch/module caches and toolchain payloads. Original build roots and prior evidence are preserved in place. Self-review checked all eight cells against both declared library products and all workflow destinations, source hashes, actual object architectures/minimum versions, numeric observations rather than exits alone, and the stated untested boundaries. No portability repair was required.
