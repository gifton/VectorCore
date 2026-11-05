# Runtime and sanitizer phase-one qualification — 2026-11-04

The selected **optimized, testable Release sanitizer workloads pass: 77/77 ASan bodies and 39/39 TSan entries**, including Q03's actual concurrent child. The clean Xcode Catalyst smoke passes its listed public-library assertions. **Full iOS simulator suite qualification is blocked at compilation by `Process` in shared test support. ASan leak detection remains unqualified; standalone LeakSanitizer is unsupported for the tested Darwin targets.** These are first-phase results, not final release acceptance or whole-suite qualification.

## Changes and source identity

Driver/direct-test commit: `e48ee0b` (`Add explicit Release configuration to sanitizer driver`). Assigned tree `/Users/goftin/.codex/worktrees/c0ea/VectorCore` began clean at `2dcb24091c6dc21be4db409ecdd0756220e93b8c`. `Package.swift` and all tracked Sources/Tests/Benchmarks inputs remain unchanged. The independent final source-hash cross-check is `source-identity-check.json`.

`Scripts/ci/check_sanitizers.py` adds `--configuration {debug,release}`, keeping Debug as default. One flag array feeds compilation and SwiftPM product-path discovery; execution uses only the bundles discovered beneath that reported path. `result.json` records the configuration. Release-only `-Xswiftc -enable-testing` permits existing `@testable` imports without replacing `-O`. This is an instrumented, optimized, testable build, not a production-binary-equivalence claim. No manifest body/filter, deadline, assertion setting, sanitizer diagnostic option, production source, package setting or workflow is changed.

Initial original-path ASan and TSan Release builds both failed at `Tests/MinimalTests/BaselineTests.swift:2` because `swift build --build-tests` does not enable Release testability. Those failures remain in `release-address/` and `release-thread/`. Selected SwiftPM help exposes `-Xswiftc`, with no dedicated testability option. The narrow repair follows the existing `check_performance_gates.py` pattern.

`SanitizerConfigurationTests.test_configuration_agrees_across_build_discovery_execution_and_evidence` covers both sanitizers, default Debug, explicit Debug and Release, identical build/discovery flags, Release-only testability, selected execution bundles and output configuration. `test_invalid_configuration_rejects_before_any_build_or_output` checks rejection before lookup/output creation. `configuration-red.*`, `configuration-green.*`, `release-testing-red.*` and `release-testing-green.*` retain RED/GREEN results. An intermediate testability-option expectation was rejected after actual SwiftPM help showed that option absent (`release-testability-red.*`); it was never added to the driver.

The frozen final Python CI tooling suite passes **598 tests** (`python-ci-tests-final.*`, 259.10s test-reported / 259.92s supervisor). The earlier 598-test run predates the testability repair and is retained but not substituted for final verification. Focused final sanitizer tests pass **11 tests**. `check_ci_contracts.py` confirms unchanged **77/39** manifests. `git diff --check` and unchanged production/package-input checks pass. No native Swift full baseline suite was repeated.

## Release execution and identities

Raw evidence: `/private/tmp/vectorcore-release-phase-one-2026-11-04/runtime-sanitizers`.

| Cell | Fresh build | Complete driver | Executed |
| --- | ---: | ---: | --- |
| ASan Release | 342.33s | 388.38s | 77/77 bodies |
| TSan Release | 323.45s | 438.98s | 39/39 entries, including Q03 |

Every selected row has zero exit, exact started/passed evidence, no skip, and actual loaded helper, bundle and selected runtime paths. The independent cross-check verifies command/log hashes, exact manifest selection/deadlines, identical build/discovery flags, selected bundle paths and retained artifact hashes. `verified-release-compiler-invocations.txt` in each accepted output captures emitted `-O`, `-enable-testing`, `VC_SWIFTPM_RELEASE` and the matching `-sanitize` option. No `-Onone` is substituted. Q03's actual child exits zero with cleanup completed, exact host/bundle identity, all six workers completing 2000 loops, stable totals/atomic batches joined, and loaded TSan/helper/bundle images checked in its child log. Its child log SHA256 is `9d059bf0a695ad5a32a94af0fc0ccba8fe1af45449d556f58ce5bbe7ed702e93`.

Each build uses fresh scratch roots, `--jobs 2 --swift-build-system native`, a 1200-second build deadline and a 2400-second outer supervisor. Ordinary bodies retain 15 seconds; Q03 retains a 49-second parent, 45-second child and 2-second cleanup. `VECTORCORE_TEST_EXTENDED=0`, `VECTORCORE_STRICT_PERF=0`, ASan `detect_leaks=0:halt_on_error=1` and TSan `halt_on_error=1` remain unchanged. Compiler work is sequential within this track. Times are evidence of bounded execution, not benchmark acceptance.

The verified Swift.org frontend SHA256 is `956438a61893525d7ce28c21b6634093f15b0c4bae7fe6eb7d98fba0c1c5e301`; MacOSX27.2 SDK settings SHA256 is `82165b15ee21419d4cffaba8895797391b908ac9499ba70f561f3e1bd16758d6`. The retained extracted payload was reused without installation, and its existing compiler qualification applies only to these identical inputs. Per-command records retain selected environment and executable hashes; identity snapshots retain source/SDK/compiler hashes. The actual SwiftPM helper and both sanitizer runtime bytes are copied under `selected-runtime/`, not just named by hash.

| Loaded artifact | ASan SHA256 | TSan SHA256 |
| --- | --- | --- |
| Helper | `9b2b116ce29b756e9a6319526bd803a19946916273544a75ace08de58386f5b8` | `9b2b116ce29b756e9a6319526bd803a19946916273544a75ace08de58386f5b8` |
| Runtime | `0f77b71ea4342b04be5163a7c02d69376f5ecd5ce71e40278b41473850485180` | `703112657f82fc3522365b529f83919cddc61d114fd7c6bdc8ddaa2088debfe3` |
| Aggregate bundle | `cd6fb6413d9b0719f675e56048d99de1b2f4e6c8db3c2f92226f1563294edf9c` | `e84d9216085f94f60985168fa3b0b7d636efe2d7f808724a2c5ff975448ab2e5` |

## Leak capability control

Independent `LeakControl.c`, compiled with `-O1 -g -fsanitize=address`, allocates 4096 bytes, writes/prints its pointer/value, and never frees it. The retained run prints `allocated=0x621000000100 value=42`; emitted disassembly shows `mov w0, #0x1000` followed by `bl _malloc`, and retained symbols/disassembly include `_printf` with no `_free` reference. Source uses a noinline function and volatile access. The allocation therefore survives this compilation; this is supported by `leak-binary-symbols.*`, `leak-binary-disassembly.*` and the runtime marker. Its ASan run with `detect_leaks=1:halt_on_error=1` exits zero without a leak diagnostic. Runtime loading and source/binary/compile/run records are retained; a clean no-op is not a leak-coverage pass. The separate selected-clang `-fsanitize=leak -###` probe exits 1: `unsupported option '-fsanitize=leak' for target 'arm64-apple-darwin27.0.0'`. A second probe explicitly matching the package deployment target, `-target arm64-apple-macosx14.0 -fsanitize=leak -###`, also exits 1 with the unsupported-option diagnostic. The helper's options listing alone does not establish implemented leak detection. The project driver continues to record `leakDetection: false`. No project leak-detection qualification is claimed.

## Additional Catalyst runtime cell

The clean installed-Xcode client compile has zero compiler/linker stderr, targets `arm64-apple-ios17.0-macabi`, and executes on macOS 27.0.1. It prints `VECTORCORE_PLATFORM_SMOKE_PASS platform=maccatalyst-os27.0.1 arch=arm64 version=0.3.3`. `PlatformSmoke.swift` checks vector addition/dot product, dynamic magnitude/normalization, 3×2 QR reconstruction and 2×2 eigenvalues through both LAPACK and Swift providers, and metadata architecture, with explicit numerical tolerances. Actual dyld image evidence, linked libraries, Mach-O platform headers and executable bytes are retained.

The two library products' prior Xcode Release objects were reused to avoid repeating the accepted compile matrix. `Package.swift` and all Sources/Benchmarks inputs hash-match the prior build snapshot; the three linked objects also hash-match its independently retained product inventory. Copied objects/modules and prior records are retained in `catalyst-inputs/`, `catalyst-reuse.json`, `catalyst-prior-object-check.json` and `prior-*.json`. An initial overbroad comparison correctly found two changed test/support files (`RunnerContractTests.swift`, `ChildProbeLauncher.swift`); no old test artifact is reused, and those files are not dependencies of either library product.

The initial Swift.org client link emitted missing-maccatalyst-resource and macOS-dylib platform-mismatch warnings. Although its smoke assertions passed, it is retained as a nonaccepted attempt (`catalyst-compile.*`, `catalyst-run.*`), separate from the clean Xcode retry (`catalyst-xcode-*`). No warning was suppressed. The installed Xcode compiler's known Release lost-store canary failure remains documented in the [prior platform report](../../2026-11-04-release-qualification/platforms.md); the narrow Catalyst assertions do not qualify that compiler's general numerical behavior, all Catalyst APIs, Intel Catalyst or the minimum OS.

## Full iOS suite feasibility and missing environments

The existing arm64 iOS 27.0 simulator is available. `xcodebuild -scheme VectorCore -configuration Release ... build-for-testing` reports success but produces **zero `.xctest` bundles and an empty `TestConfigurations` array**. The actual `.xctestrun` and `ios-empty-testplan.json` preserve this vacuous library-scheme control; it counts as zero test coverage.

The discovered complete `VectorCore-Package` scheme is then built without filters/exclusions in a fresh root. It exits **65**, with `cannot find type 'Process' in scope` at `Tests/VectorCoreTestSupport/ChildProbeLauncher.swift:156`, `:158` and `:193`, and a secondary `ChildProbeTermination` Equatable-conformance error at `:153`. **Zero test bodies execute.** This is a real first-phase support limitation, not a missing simulator. Follow-up requires portable test support or explicit platform-specific test contracts before full iOS suite qualification; this track makes no blanket platform exclusion or shared harness redesign.

| Environment | Current evidence / missing requirement |
| --- | --- |
| Native arm64 macOS 27.0.1 | Selected Release ASan/TSan manifests pass. Whole sanitizer suites and combined-candidate full baseline remain later gates. |
| Native macOS 14 minimum | Runtime unavailable; macOS 14 deployment target is not execution evidence. |
| Catalyst arm64 on macOS 27.0.1 | Listed public-library smoke assertions pass. Full suite/minimum OS and Intel remain untested. |
| iOS 27.0 arm64 simulator | Runtime available; full package test build blocked by shared `Process` support. Prior library smoke is historical scope, not repeated here. |
| iOS 17 minimum | Runtime unavailable. No minimum-OS claim. |
| tvOS 17 / newer, visionOS 1 / newer | SDKs present; no simulator runtimes or corresponding usable devices. No runtime or minimum-OS claim. |
| x86_64 macOS/Catalyst | `arch -x86_64 uname -m` fails `Bad CPU type in executable`; Rosetta receipt absent. No Intel runtime claim. |
| Physical iPhone 17 / iOS 26.6.1 | Paired, connected, Developer Mode enabled; read-only details report developer disk image mount failure. Not provisioned/executed. |
| Physical iPhone 16 Pro Max / iOS 27.2 | Paired, Developer Mode enabled, but connection unavailable; no test execution. |
| Leak sanitizer | Selected Darwin target rejects standalone leak sanitizer; intentional ASan leak control is undiagnosed. No project leak coverage. |

Inventory records cover Xcode 27.2, SDK versions/architectures/settings hashes, selected and installed compiler identities, runtimes, existing simulators, physical-device status and Intel execution. Only the iOS 27.0 arm64 simulator runtime is installed. No toolchain/runtime/SDK was installed, Xcode globally selected, physical device provisioned or simulator booted/created by this track.

## Reachability, self-review and follow-up

`configuration-reachability.txt` inventories every `DEBUG`, `VC_SWIFTPM_DEBUG` and `VC_SWIFTPM_RELEASE` occurrence. Release activates the existing compiled-feature metadata branches and removes existing DEBUG-only checks/logging. The manifest bodies are unchanged; only their optimized instrumented execution is newly selected. No production routing predicate changes. Selected workload success does not establish that every Release conditional or optional C-kernel mode is exercised.

Self-review checked final diff/ownership, default compatibility, supported argument choices, Release testability and emitted optimization flags, shared build/discovery flags, executed/retained product identities, every body/deadline/log hash, actual Q03 child images, no-op leak-control handling, source/SDK/compiler identity, vacuous-scheme rejection and known platform/compiler limitations. Independent coordinator review and combined Task 1 integration/final qualification remain required. No release tag, merge or publication is performed.

`qualification-summary.json` is the machine-readable result. `archive-inclusion.txt` and `archive-manifest.json` explicitly retain exact command/output/source/identity records, accepted/failed diagnostic and smoke attempts, reused Catalyst link inputs, actual selected runtime/helper bytes, loaded sanitizer bundles and Q03 evidence. Bulk build/module caches and compiler/SDK payloads are excluded; originals remain in place. `v03_process.run` records exact argv, environment, bounded execution, exit/timeout and output hashes for every meaningful command.
