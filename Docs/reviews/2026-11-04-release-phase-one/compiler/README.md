# Compiler correctness investigation — 2026-11-04 phase one

**The installed Xcode compiler remains a Release numerical blocker.** The original
pure-Swift canary, the inside-borrow form, and a copied-buffer/inout form all
return `2.0` / `40000000` instead of `1.0` / `3f800000`. The retained Swift.org
compiler passes these exact shapes. No production, compiler-support policy,
canary expectation or optimizer setting is changed by this investigation.

The worktree started clean at `2dcb24091c6dc21be4db409ecdd0756220e93b8c`.
All new files belong to this directory. The earlier November 1 and November 4
qualification evidence remains unchanged; this is a fresh standalone probe
matrix, not a repeated full library qualification.

## Identities and scope

| Input | Identity |
|---|---|
| Xcode compiler | `Apple Swift version 6.4.2 (swiftlang-6.4.2.1.9 clang-2100.4.1.1)` |
| Xcode frontend SHA-256 | `448457da9a8a56741f2c8c4b8931e85b7377fc009c16918cd8ac1a8a3210ea1f` |
| Swift.org compiler | `Apple Swift version 6.4.2 (swift-6.4.2-RELEASE)` |
| Swift.org frontend SHA-256 | `956438a61893525d7ce28c21b6634093f15b0c4bae7fe6eb7d98fba0c1c5e301` |
| SDK | `MacOSX27.2.sdk`, SDKSettings.json SHA-256 `82165b15ee21419d4cffaba8895797391b908ac9499ba70f561f3e1bd16758d6` |
| Target/language | `arm64-apple-macosx14.0`, `-swift-version 6` |
| Executed runtime | macOS 27.0.1 (26A434), arm64 host |
| Debug / Release | `-Onone -D DEBUG` / `-O`; no optimizer pass suppressed |

The Swift.org package payload is the previously verified, extracted toolchain;
its frontend hash was checked before execution. No installation, global Xcode
selection, download or execution of hosted CI binaries was performed. Driver,
frontend, SDK settings, probe, binary and output hashes are in
[outcome-matrix.json](outcome-matrix.json) and its referenced raw summaries.
The short version number is insufficient to distinguish these compilers.

## Numerical outcome matrix

Every number in this table is a runtime observation. All 32 probe programs
compiled and exited zero. Acceptance nevertheless rejects all stale results.
The required positive observation is exactly
`LOST_STORE observed=1.0 bits=3f800000 expected=1.0`.

| Probe shape | Xcode Debug | Xcode Release | Swift.org Debug | Swift.org Release |
|---|---:|---:|---:|---:|
| Original literal array, inout writer, read after borrow | 1 | **2 FAIL** | 1 | 1 |
| Literal array, inout writer, read inside borrow | 1 | **2 FAIL** | 1 | 1 |
| Copy buffer locally, pass copy inout, read original buffer inside borrow | 1 | **2 FAIL** | 1 | 1 |
| By-value writer, read inside borrow | 1 | 1 | 1 | 1 |
| By-value writer, read after borrow | 1 | 1 | 1 | 1 |
| Repeating initializer, inout writer, read after borrow | 1 | 1 | 1 | 1 |
| Literal array, inout writer, out-of-line reader | 1 | 1 | 1 | 1 |
| Deliberately omit write (negative acceptance control) | 2 rejected | 2 rejected | 2 rejected | 2 rejected |

The three historical probes are read directly from
`Docs/reviews/2026-11-01-compiler-qualification/probes/`, copied byte-for-byte to
each run's evidence, and hashed. Five new controls are in [probes/](probes/).
The new copied-buffer control demonstrates that copying a buffer descriptor does
not accommodate this compiler defect. The passing by-value control changes the
callee's parameter convention; it does not establish the correctness of arbitrary
inout or generic APIs. The nonliteral initializer changes the allocation and
optimization shape. The out-of-line reader deliberately introduces a noinline
boundary as an investigative control; adopting such boundaries throughout the
library would not repair client code or the original canary and is not proposed.
No passing probe establishes a general workaround.

## Optimized SIL and IR

Both compilers emitted optimized SIL and LLVM IR for the original, inside-borrow,
copied-inout and by-value-inside probes. Those 16 files are retained and hashed
under each run directory (`*.sil`, `*.ll`).

For the original probe, Xcode's optimized SIL constructs `%13` from the literal
`0x40000000`, calls `writeOne`, then returns `%13`. Swift.org's SIL retains a
post-call `load` and returns its value. In LLVM IR, Xcode's original,
inside-borrow and copied-inout `observedValue` functions retain the writer call
but end with `ret float 2.000000e+00`. The corresponding Swift.org functions load
the element after the call and return that loaded value. The writer's explicit
store remains in its emitted function. These artifacts locate the wrong constant
in optimized SIL before LLVM IR generation; this run does not identify the first
responsible SIL pass or prove whether a specific upstream patch is present.

The public [alias-analysis fix #92580](https://github.com/swiftlang/swift/pull/92580)
explains a related failure: an indirectly passed stored pointer requires global
callee effects to preserve accesses. Its tests retain the load after such a call.
This is consistent with the local evidence, but is an inference about the defect
class, not a binary patch identification. The original
[issue #92562](https://github.com/swiftlang/swift/issues/92562) concerns MutableSpan;
the subsequent [issue #93141](https://github.com/swiftlang/swift/issues/93141)
records another formulation that still failed after #92580 and was fixed
separately. That history is additional reason not to infer correctness from a
patch number or a passing nearby shape. Sources were consulted on 2026-11-04.

## Support options for later approval

1. **Keep broad intended support and block Release sign-off on affected compiler
   builds.** Retain the hard canary and known-issue predicates, submit or track
   the minimized evidence, then qualify a repaired Xcode binary. This preserves
   App Store intent but does not currently provide a passing Xcode path.
2. **Adopt an exact-binary, non-App-Store qualification scope for Swift.org
   6.4.2.** This is the available local macOS arm64 candidate. Adoption would
   require an explicit support decision and the later combined library/client,
   sanitizer and platform gates; this probe run alone is insufficient.
3. **Investigate specific library accommodations only after a concrete library
   failure is reproduced.** A call-convention change may help a private leaf
   operation, but must preserve public APIs and pass actual Debug/Release and
   downstream-client regressions. No such repair is justified by these standalone
   pure-Swift failures, and none can make the unchanged client canary pass.

Recommendation: retain the failing Xcode evidence and canary, and use the exact
Swift.org binary as the candidate for later scoped qualification while support
policy remains unchanged. Do not replace a compiler check with a blanket
`compiler(>=6.4.2)` gate. Swift.org's
[installation documentation](https://www.swift.org/install/macos/package_installer/)
states that App Store submissions must use the Swift compiler included with
Xcode; the extracted-toolchain result therefore does not qualify App Store builds.

This run does not qualify Swift 6.0, Xcode 26.3 / Swift 6.2.4, other compiler
builds, Intel, iOS/simulators/devices, other Apple platforms, minimum-OS runtime,
C kernels, GPU integration, sanitizers, generic/existential public API shapes,
or performance. Earlier evidence for those cells remains distinct. In particular,
previous hosted Xcode 26.3 Release evidence records a known lost store; it is
not turned into a numerical pass by its CI aggregate. Existing public Collection
witness repairs are neither changed nor independently requalified here.

## Reproduction and acceptance controls

Use a new absolute output directory for each compiler; the driver refuses reuse.
It verifies the requested frontend hash before invoking it, executes one compiler
process at a time, gives each compilation a 60-second deadline and each program a
10-second deadline, and uses `Scripts/ci/v03_process.py` for process ownership and
recording. Each command has exact argv, selected environment, exit, output hashes
and bounded cleanup. At most two compiler processes ran concurrently across the
two matrices. No timing acceptance was attempted.

```sh
python3 Docs/reviews/2026-11-04-release-phase-one/compiler/run.py \
  --swiftc /absolute/path/to/toolchain/usr/bin/swiftc \
  --frontend-sha256 <exact-frontend-hash-from-the-table> \
  --sdk /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX27.2.sdk \
  --output /absolute/path/to/new-evidence-directory
python3 Docs/reviews/2026-11-04-release-phase-one/compiler/test_run.py
python3 -m unittest Scripts.ci.tests.test_lost_store_reads
```

The matrix driver exits 1 for Xcode's three genuine numerical failures and 0 for
Swift.org. Omitted-store controls must produce stale bits and be rejected;
they are recorded as successful negative controls, never positive numerical
passes. `test_run.py` pins the conjunction of process success, exact value,
exact bits and one complete output line. Its exit-only mutant failed five
subcases before implementation, then the exact predicate passed. A second test removes each of the eight required
probe files in a temporary copy and requires rejection; its glob-based mutant
failed all eight subcases. The final two-test driver suite passes. The existing
scanner suite remains unchanged and passes its 15 tests; its documented alias
limitations still apply to the copied-buffer probe. This experiment does not
expand or weaken scanner policy.

Raw evidence is retained at
`/private/tmp/vectorcore-release-phase-one-2026-11-04/compiler/`:
`xcode-02/`, `swiftorg-02/` (final driver), `driver-tests/` (RED/GREEN),
and `verification/`. The first matrices (`xcode-01/`, `swiftorg-01/`) are also
retained unchanged and produced the same numerical outcomes; the final rerun
followed the driver inventory check. `archive-inclusion.txt` selects the small
evidence files and excludes module caches and compiler/SDK payloads.
Each summary links the corresponding output hashes. No full baseline suite was
repeated because production and required test policy were not edited; combined
qualification belongs to the coordinator's later gate.
