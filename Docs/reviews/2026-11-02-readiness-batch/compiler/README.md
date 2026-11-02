# Imported default Collection witness: support limit

On 2026-11-02, the public `PublicImport.swift` client aborts the Swift.org 6.4.2
compiler when it inherits `VectorProtocol`'s scalar subscript. Defining the
subscript directly on the conformer compiles. Both the current library revision
`24c4e57c50b83517f620bb39bc5c8561343ee1a6` and the earlier revision
`ec7f3323529ce3df6c126e8b6416f6e4f67d16ba`, freshly built with that compiler,
SDK and normal Release settings, exhibit this pair of outcomes. The trigger
therefore predates the remediation batch on this toolchain.

The practical accommodation remains an explicit conformer subscript:

```swift
subscript(index: Int) -> Float { storage[index] }
```

This is a compile-tested accommodation for the fixtures here, not a compiler fix
or a guarantee about all conformers. Keep the public default and normal package
flags; third-party conformers depending on that default have a demonstrated
compiler support limitation. No production API or implementation is changed by
this investigation.

## Reproduce

Python 3 and a macOS arm64 Swift toolchain are required. The runner accepts an
explicit compiler and SDK, importer fixture, imported module directories or
standalone library source files, witness choice, and per-process deadline. It
creates a new output directory, records exact argv, source/module/compiler/SDK-settings
hashes, compiler version, process exits and output hashes. A timeout records
`exit: null`, kills the owned process group and records whether it was reaped.
`--allow-failure` allows collecting expected compiler crashes; inspect JSON rather
than treating that runner exit as successful compilation.

Set paths to a **Release-built VectorCore module from the matching compiler**:

```sh
VC_COMPILER_BIN=/path/to/toolchain/usr/bin
VC_SDK=/path/to/MacOSX.sdk
VC_SOURCE=/path/to/exported/VectorCore
VC_BUILD=/path/to/its/arm64-apple-macosx/release
VC_PROBE=Docs/reviews/2026-11-02-readiness-batch/compiler
python3 "$VC_PROBE/run.py" \
  --swiftc "$VC_COMPILER_BIN/swiftc" --sdk "$VC_SDK" \
  --importer "$VC_PROBE/PublicImport.swift" \
  --module-dir "$VC_BUILD/Modules" \
  --extra-flag=-Xcc \
  --extra-flag="-fmodule-map-file=$VC_BUILD/VectorCoreC.build/module.modulemap" \
  --extra-flag=-Xcc --extra-flag=-I --extra-flag=-Xcc \
  --extra-flag="$VC_SOURCE/Sources/VectorCoreC/include" \
  --output /tmp/vectorcore-witness-new-run --allow-failure
```

The recorded baseline uses `-O -whole-module-optimization -enable-testing
-swift-version 6 -enable-experimental-feature StrictConcurrency`, macOS 14 arm64
target, and a 60-second deadline for each compile. `--witness inherited` and
`--witness explicit` select one case; the default runs both from identical source
using `EXPLICIT_WITNESS`. Compilation is object-only; no client is linked or run.

To rebuild either revision, export it with `git archive REVISION` into a new source
directory. With `SWIFT_EXEC` set to `VC_COMPILER_BIN/swiftc`, prepend that bin to
`PATH`, select a new `CLANG_MODULE_CACHE_PATH`, and run:

```sh
"$VC_COMPILER_BIN/swift" build --build-system native --jobs 2 \
  --sdk "$VC_SDK" -c release --scratch-path /tmp/vectorcore-new-build \
  --target VectorCore -v
```

Supervise the build with a deadline (the recorded builds used
`Scripts/ci/v03_process.py:run`, 360 seconds); the importer runner does not launch
SwiftPM. Set `VECTORCORE_TEST_EXTENDED=0` and `VECTORCORE_STRICT_PERF=0` as in the
qualification environment. The builds do not add, remove or override package
optimization settings. Path-derived package names differ between the two archive
roots; exact commands and source-file hashes are retained.

The standalone control builds two modules without VectorCore or Foundation:

```sh
python3 "$VC_PROBE/run.py" \
  --swiftc "$VC_COMPILER_BIN/swiftc" --sdk "$VC_SDK" \
  --library-source "$VC_PROBE/MinimalLibrary.swift" \
  --importer "$VC_PROBE/MinimalImporter.swift" \
  --output /tmp/vectorcore-minimal-new-run
```

Both witnesses **pass** in this control. It is not a source-only reproducer of the
crash. The explicit `Element = Float` alias avoids unrelated associated-type
inference errors. `--profile basic` (`-Onone`) and `--profile optimized` (`-O`)
are comparison controls only; no qualification flags were weakened.

## Outcomes and limits

`outcome-matrix.json` records hashes and individual evidence paths.

| Experiment | Inherited subscript | Explicit subscript |
|---|---:|---:|
| Archived original LC5 fixture | signal 6 | 0 |
| Remove LC5 counter | signal 6 | 0 |
| Also remove Foundation import | signal 6 | 0 |
| Also replace `@testable` with public import | signal 6 | 0 |
| Fresh current Release module | signal 6 | 0 |
| Fresh earlier Release module | signal 6 | 0 |
| Current Release module, `-Onone` importer | signal 6 | 0 |
| Current Release module, `-O` importer without WMO/testing/StrictConcurrency flags | signal 6 | 0 |
| Bare standalone Collection protocol | 0 | 0 |
| Standalone plus bounds check/string interpolation | 0 | 0 |
| Standalone plus VectorProtocol requirements/default validation | 0 | 0 |
| Current full-source module emission, Swift.org 6.4.2 | signal 6 | 0 |
| Same comparison, Xcode Swift 6.4.2 (`swiftlang-6.4.2.1.9`) | signal 6 | 0 |

Both compiler builds reproduce with their own matching modules. The Xcode probes
are comparisons only, not library qualification. No actual Swift 6.0 compiler,
other Swift release, other architecture, other platform, sanitizer runtime, or
Debug-built library module was validated. The package tools/language floor remains
6.0. No runtime or numerical behavior is asserted by these compile-only probes.

Failing logs identify `MandatorySILLinker`, `deserializeAndPushToWorklist` at
`Linker.cpp:88`, and `cannot deserialize shared function`; they do not name the
function being deserialized. Raw SILGen succeeds: the full-module client has a
Collection read witness referencing the shared serialized declaration
`$s10VectorCore0A8ProtocolPAAEy7ElementQzSicir`. The standalone shape control emits
a body for its corresponding read accessor. Symbol inspection finds the public
getter in `VectorProtocol.swift.o`, without a scalar read-accessor definition in
that object's symbol listing. These observations narrow investigation to the
imported accessor path; they **do not establish** which exact function violates
the compiler assertion or why it is emitted differently. The smallest confirmed
failing fixture still requires the full library module.

## Evidence retention

Raw evidence root: `/private/tmp/vectorcore-readiness-2026-11-02/compiler-evidence`.
Prior archives were read-only throughout. The full investigation report is
`/private/tmp/vectorcore-readiness-2026-11-02/task-3-report.md`.

`archive-inclusion.txt` under the raw evidence root lists the portable evidence
subset: source exports and source hashes, current/earlier module exports, both
compiler comparison modules, runner/fixture snapshots, command JSON, logs and SIL.
It excludes SwiftPM object trees, module caches and extracted source trees. C
headers are recoverable from the source archives; generated module maps retain
the original absolute header paths, so rebuild or point the runner at the
corresponding exported headers when relocating evidence.
