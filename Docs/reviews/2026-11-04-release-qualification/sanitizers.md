# Real sanitizer qualification — 2026-11-04

> Hosted follow-up: [run 41946052293](hosted-ci.md) at `f779d92` passes the same 77 ASan / 39 TSan manifest on Xcode 26.3, including Q03. Later runtime-alias and private-framework launcher repairs are documented there. The results and identities below remain the earlier local Swift.org 6.4.2 phase.

The mandatory manifest passed on the qualified Swift.org 6.4.2 toolchain: **77/77 AddressSanitizer bodies and 39/39 ThreadSanitizer entries**, including Q03's actual instrumented child. Every body has a zero exit, its exact started/passed event, and loaded helper, bundle and selected sanitizer image evidence. No selected body was skipped. Separate intentional heap-overflow and concurrent-write controls each produced the expected sanitizer diagnostic and exited with signal 6. Actual metadata child runtime propagation passed for both runtimes.

This qualifies these selected Debug instrumented paths on this arm64 macOS host. It does not qualify leak detection, the complete suite, Release sanitizer builds, older operating systems, other platforms/architectures, Swift 6.0 or hosted CI.

## Source and execution

The package source base is `77defdf6161ab7600309d3432691fa9d388d088e`, the plan commit over the accepted release-preparation implementation. All **387 tracked inputs** under Sources, Tests, Benchmarks and Package.swift are unchanged. The only executable repository changes here are the sanitizer driver and its Python regression tests. Tools version, platform floors, dependencies, public APIs and numerical contracts are unchanged.

The driver ran separately for `address` and `thread`, with `--jobs 2 --swift-build-system native`, fresh scratch/output roots, `VECTORCORE_TEST_EXTENDED=0` and `VECTORCORE_STRICT_PERF=0`. ASan uses `detect_leaks=0:halt_on_error=1`; TSan uses `halt_on_error=1`. Each ordinary body retains a 15-second deadline; the Q03 parent retains 49 seconds, its child 45 seconds and cleanup 2 seconds. The outer supervisor has a finite 2400-second process-group deadline; the existing build deadline remains 1200 seconds.

| Cell | Fresh build | Whole mandatory driver | Completed |
| --- | ---: | ---: | ---: |
| Address | 154.64s | 201.28s | 77 bodies |
| Thread | 127.63s | 244.13s | 39 entries |

These concurrent-host qualification times are not benchmark acceptance. Existing native-backend deprecation, Swift concurrency/unused-value, Float-underflow and Accelerate/API deprecation warnings remain in the build logs; no warning suppression or new test exclusion was introduced.

## Tool and loaded-image identities

The previously verified official package and signature were reused without installation or global Xcode changes. The package origin/signature are retained with the evidence and documented in the [compiler qualification](../2026-11-01-compiler-qualification/README.md). The runtime host is macOS **27.0.1 (26A434)**, arm64; macOS 14 is the compilation deployment target, not an exercised OS runtime.

| Artifact | SHA-256 |
| --- | --- |
| Swift.org 6.4.2 `swift-frontend` | `956438a61893525d7ce28c21b6634093f15b0c4bae7fe6eb7d98fba0c1c5e301` |
| MacOSX27.2.sdk `SDKSettings.json` | `82165b15ee21419d4cffaba8895797391b908ac9499ba70f561f3e1bd16758d6` |
| Selected `swiftpm-testing-helper` | `9b2b116ce29b756e9a6319526bd803a19946916273544a75ace08de58386f5b8` |
| `libclang_rt.asan_osx_dynamic.dylib` | `0f77b71ea4342b04be5163a7c02d69376f5ecd5ce71e40278b41473850485180` |
| `libclang_rt.tsan_osx_dynamic.dylib` | `703112657f82fc3522365b529f83919cddc61d114fd7c6bdc8ddaa2088debfe3` |
| Address loaded aggregate bundle | `b741fa565368825b82131bd423f9262c82b9e9e5ace76d9838cfba8f3bb12d17` |
| Thread loaded aggregate bundle | `3f9e00718350b07e7c5fd13e9618c3b45367c10990939082008f35b7e317caa7` |

The helper and runtimes are from the retained Swift.org payload's `usr/libexec/swift/pm` and `usr/lib/clang/21/lib/darwin` directories. Every body log contains the actual loaded paths. The evidence cross-check also verifies Q03's helper, bundle and TSan image under the actual child PID, **31172**, rather than relying solely on its environment record.

## Narrow runner repair

The original driver successfully built ASan and executed its first pinned body, then rejected missing loaded-image evidence. Direct probes of the signed Swift.org `swift-driver` and `swift-package` emitted zero dyld trace lines with `DYLD_PRINT_LIBRARIES=1`; the unchanged helper emitted 703. Signatures and entitlements are retained. This was a launcher/evidence failure, not a failing VectorCore body.

The driver now launches that exact SwiftPM helper directly with the discovered bundle, unchanged manifest filter and `--testing-library swift-testing`. It preloads exactly one matching sanitizer runtime from the selected toolchain, enables dyld tracing, and supplies Swift Testing and Xcode XCTest support search paths. The first direct probes exposed missing `libTesting.dylib`, then `libXCTestSwiftSupport.dylib`; both failures were retained before the complete search paths passed the original body. No helper, compiler, runtime or test bundle was patched or re-signed.

The manifest, body validation, build flags, deadlines, fail-fast sanitizer options and Q03 checks are preserved. The driver now records the selected runtime hash and requires its exact path in the loaded-image log. Missing or ambiguous toolchain runtimes reject instead of silently selecting one. A task-local `xcrun --find swift` lookup adapter selects the qualified payload; all other xcrun requests delegate to the system tool. The adapter does not wrap actual compilation or test execution.

Focused Python validation passed **7 sanitizer-log/host tests, 6 CI-contract tests and 21 SwiftPM-product tests**. The new construction tests cover both sanitizers and every manifest filter, environment options, and missing/ambiguous runtimes. The initial missing-function red result and an intermediate temporary-path alias assertion failure are retained; the final seven-test run passed. `check_ci_contracts.py` confirms the unchanged 77/39 body sets. The accepted full native suites and compiler canaries were not repeated.

## Diagnostic controls and metadata

`Diagnostic.swift`, compiled separately with each sanitizer using the qualified compiler, intentionally writes beyond a one-element heap allocation for ASan and performs unsynchronized parallel writes for TSan. Both controls exited **-6**, with respectively `AddressSanitizer: heap-buffer-overflow` and `ThreadSanitizer: data race`. Their logs record the loaded matching runtime, and their binaries, source, compile/run records and diagnostics are retained. These are expected failures, separate from the clean mandatory bodies.

For metadata, a small `@testable` parent links the actual instrumented VectorCore, VectorCoreC and VectorCoreBenchmarking objects and invokes `EnvCapture.observe`. An independently compiled child has **no sanitizer linkage**, verified by `otool -L`, and reports `_dyld` image paths. This isolates runtime propagation through the production metadata process path from automatic sanitizer linkage in the child.

The first ASan metadata probe expected the entrypoint environment to retain `DYLD_INSERT_LIBRARIES`; it failed because the parent observed it unset, and therefore did not supply it to metadata. A direct injected-child control likewise observed the insertion variable unset while the ASan image was loaded. The final probe explicitly restores the selected already-loaded runtime's insertion value before `EnvCapture.observe`, as Q03 also explicitly restores insertion for its child. For both ASan and TSan the metadata child then reports the actual matching loaded runtime, while a protected `/bin/sh` target reports `PROTECTED_DYLD=unset`. This is observed runtime propagation, not byte-identical environment preservation or a claim that protected executables accept injection. The ASan metadata parent links the first successful ASan build's objects; those package inputs are identical to the fresh final ASan cell.

Q03's child exited zero, completed cleanup, matched the selected helper/bundle hashes and emitted all required markers: body begin/end, six workers each completing 2000 loops, stable totals joined, atomic batches joined and concurrency result. The unsafe baseline mode was absent. Its actual child log SHA-256 is `781dc0e71af94235805ab80021a7342fc02d341792b2f15272e0b65e4e303fae`.

Metadata ownership limitations remain those of the [accepted lifetime implementation](../2026-11-04-release-preparation/README.md): descendants must remain in the owned group and not attack the guardian/timer; escaped groups, arbitrary signal/credential changes and OS scheduling stalls are excluded. The observer has a bounded observation deadline, but startup/scheduling are not real-time bounded. Foreign reapers, ignored SIGCHLD and SA_NOCLDWAIT remain unverified host assumptions; a reaper thread can outlive the call. These propagation checks do not broaden that contract.

## Retained evidence

Raw root: `/private/tmp/vectorcore-qualification-2026-11-04/task-2-evidence`.

- `address-qualified/result.json`, `thread/result.json`, per-body records/logs, retained loaded bundles and helper copies: mandatory results and identities.
- `thread/Q03-child-concurrency-*.json` and `.log`: actual child identity, runtime image, workload markers and deadlines.
- `address-supplement`, `thread-supplement`: intentional diagnostics and metadata propagation, including the original failed ASan assumption and direct-child control.
- `address`, `loader-investigation`: original launcher evidence failure and isolated direct-host dependency experiments.
- `qualification-summary.json`, `qualification-cross-check.*`, source hashes, original/final source snapshots, command records and fixture sources: final exact-body, hash, deadline and unchanged-package checks.
- `archive-inclusion.txt`: explicit existing relative paths for freezing, excluding bulk build/module caches and toolchain payloads. `archive-manifest.json` hashes those selected files. No previous evidence was modified or deleted.
