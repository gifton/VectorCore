# Public Collection witness accommodation

VectorProtocol retains its default `subscript(index: Int) -> Scalar`. The
implementation now declares `_read` explicitly, keeps the same bounds check,
and yields the same `toArray()[index]` element. Downstream conformers can continue
to inherit the default or supply their own subscript. No protocol requirement,
package flag, dependency, or platform floor changes.

`Scripts/ci/consumer_collection_witness.py` is the durable public-client check.
It builds the same fixture with inherited and explicit witnesses, each with
`-Onone` and `-O`, links against the supplied library objects, and executes direct
access, generic Collection and VectorProtocol access, iteration, empty-collection
and copied-value checks. A compilation failure, runtime failure, or missing
completion marker makes the driver fail. It uses ordinary `import VectorCore`.
Both witness variants are also run contracts in `consumer_smoke.py`, so the
existing required consumer job executes the regression with its Release-built
dependency; the matrix driver adds the explicit Debug/Release configuration check.

## Reproduction and source-level evidence

The fresh base is `d448a3e694ea85ba3e6b7ac6a2f3d28cdaedcafc`. With the qualified
Swift.org 6.4.2 compiler and SDK below, the fresh **Release-built** base module
crashes the compiler for both inherited clients. Diagnostics name signal 6,
`MandatorySILLinker`, `deserializeAndPushToWorklist` at `Linker.cpp:88`, and
`cannot deserialize shared function`. Both explicit controls compile, link and
run. A fresh **Debug-built** base module passes all four cases.

The prior investigation in `Docs/reviews/2026-11-02-readiness-batch/compiler/`
is unchanged. Its comparisons already ruled out the test harness, Foundation
import, testable import, client optimization level, and a simple standalone
Collection protocol as sufficient explanations. This follow-up tests one
source accommodation, an explicit read accessor, rather than repeating those
comparisons or weakening optimization.

The base client SILGen references
`$s10VectorCore0A8ProtocolPAAEy7ElementQzSicir` as a `shared [serialized]`
declaration without a body. Its library object exports the getter but no scalar
read-accessor definition. With the explicit `_read`, the client refers to a public
accessor and the library object exports that exact symbol; the getter symbol
remains exported. This controlled source change removes the failing imported
shared-accessor path in these fixtures. It does **not** establish the internal
compiler defect's full cause or claim that Swift itself is fixed.

## Recorded client outcomes

| Module | Inherited client (`-Onone`, `-O`) | Explicit client (`-Onone`, `-O`) |
|---|---|---|
| Qualified base Debug | compile/link/run pass | compile/link/run pass |
| Qualified base Release | compiler assertion in both | compile/link/run pass |
| Qualified accommodated Debug | compile/link/run pass | compile/link/run pass |
| Qualified accommodated Release | compile/link/run pass | compile/link/run pass |
| Xcode accommodated Release | compile pass | compile pass |

The Xcode row is object compilation only, using its own freshly built Release
module; it makes no runtime or numerical qualification claim. The first Xcode
build was rejected because the explanatory source comment was edited during
compilation; the stable-source retry succeeded and both attempts are retained.
Qualified final module matrices were repeated after that comment and focused
test builds, with all eight compile/link/run cases passing.

The focused existing suites `D02WitnessDispatchMatrixTests`,
`D02CosineAndWitnessTests`, and `VectorCollectionTransactionTests` pass 16 tests
in three suites in both Debug and Release. The dispatch matrix checks 77 rows.
The consumer-smoke, fixture-manifest and child-process-bound Python checks pass
31 tests. The coordinator owns the combined full suites and the D02 external
symbolgraph/member backstop on the integrated build; those are not claimed here.

## Run the check

Build VectorCore separately in both configurations with the intended compiler,
without changing Package.swift optimization settings. On the qualification host:

```sh
VC_BIN=/private/tmp/vectorcore-compiler-qualification-t78w2p5_/swift-6.4.2-expanded/swift-6.4.2-RELEASE-osx-package.pkg/Payload/usr/bin
VC_SDK=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX27.2.sdk
export SWIFT_EXEC="$VC_BIN/swiftc"
export PATH="$VC_BIN:$PATH"
export VECTORCORE_TEST_EXTENDED=0 VECTORCORE_STRICT_PERF=0
# Use a new task-owned scratch path and supervise each build with a finite deadline.
"$VC_BIN/swift" build --build-system native --jobs 2 --sdk "$VC_SDK" \
  -c release --scratch-path /tmp/vectorcore-witness-build --target VectorCore
python3 Scripts/ci/consumer_collection_witness.py \
  --swiftc "$VC_BIN/swiftc" --sdk "$VC_SDK" \
  --bin-path /tmp/vectorcore-witness-build/arm64-apple-macosx/release \
  --output /tmp/vectorcore-witness-clients
```

Repeat with a separate Debug build and output. Each output directory must be new.
The driver retains exact argv, bounded process records, compiler version, compiler,
SDK-settings, fixture, module and object hashes, binaries and output. Its default
compile/run deadline is 60 seconds per command. Build evidence in this investigation
uses `Scripts/ci/v03_process.py` with a 900-second build/test deadline.

## Qualification scope

- Compiler: `Apple Swift version 6.4.2 (swift-6.4.2-RELEASE)`;
  frontend SHA-256 `956438a61893525d7ce28c21b6634093f15b0c4bae7fe6eb7d98fba0c1c5e301`.
- SDK: MacOSX27.2.sdk; SDKSettings.json SHA-256
  `82165b15ee21419d4cffaba8895797391b908ac9499ba70f561f3e1bd16758d6`.
- Target: arm64 macOS 14; runtime: macOS 27.0.1. This is not a macOS 14 runtime test.
- Tools/language floor remains Swift 6.0. No actual Swift 6.0 compiler, other
  architecture, other platform, or sanitizer was qualified by this check.
- Xcode Swift 6.4.2 is a compiler comparison only, not numerical qualification.
- No performance result is claimed; the default still materializes `toArray()`.

`outcome-matrix.json` identifies the retained evidence for individual runs.
Raw evidence: `/private/tmp/vectorcore-release-prep-2026-11-04/task-1-evidence`.
Its `archive-inclusion.txt` selects sources, modules/objects, logs and command
records while excluding build caches. The task report records focused tests,
warnings, failed attempts and integration checks owned by the coordinator.
