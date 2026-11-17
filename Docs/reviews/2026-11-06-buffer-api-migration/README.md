# SIMDStorage by-value API migration

Status: migration implemented and validated. The existing compiler release gate remains open.

## Change

`SIMDStorage` now requires `withUnsafeMutableElements`, with an
`UnsafeMutableBufferPointer<Element>` callback argument passed by value. The old
descriptor-inout requirement is removed. ContiguousArray supplies an explicit
public inlinable implementation through its standard-library mutable borrow.
There is no protocol-extension fallback and no same-name overload.

Custom conformers and generic callers must migrate, and library/consumer binaries
must be rebuilt together. [The migration guide](../../SIMDStorage_Migration.md)
shows the signatures, lifetime obligations, error behavior and affected call sites.
`VectorStorage` and `VectorProtocol` keep their existing by-value interfaces.

## Verification design

The new native `SIMDStorageMutationTests` exercises concrete/generic calls, a
new-only conformer, empty/nonempty callbacks and return values, COW copies and tails,
mutation followed by the original error, Double/SIMD elements, and generic
`OptimizedVector.Storage`. All seven tests pass in all six native Debug/Release
configurations across the three tested compilers.

The retained external consumer still has 24 independently checked numerical rows
and the complete omitted-write oracle. Its concrete and generic SIMDStorage routes
now use the new requirement. A separately compiled witness module must retain
`#SIMDStorage.withUnsafeMutableElements` in emitted SIL. This separate emission is
not a pass-by-pass trace of the executed binary.

Three migration compile contracts join the existing consumer CI job: a new-only
conformer and an ordinary concrete stdlib closure compile; an old-only conformer
and an old generic call fail at the expected source lines with specific diagnostics.
Before production changes, those outcomes were reversed as expected. The first two
attempts to typecheck the native test file manually lacked Testing framework/plugin
search paths; the corrected environment produced only the intended missing-new-API
diagnostics. These setup attempts are retained and are not counted as numerical RED.

The descriptor declaration guard now has zero production exceptions. The readback
guard recognizes the new method spelling, including postread/captured-variable and
reentry controls. It retains two explicit test-only entries for the new strict
regression assertions, preserving their literal-owner and callback-count shapes.
No numerical expectation or known-issue suppression was added for those tests.

## Validation results

Native SwiftPM builds use ordinary Debug/Release optimization, four build jobs,
an explicit `arm64-apple-macosx14.0` target and separate build trees for each
compiler. The host is macOS 27.0.1 on arm64; the deployment target is not evidence
of execution on macOS 14. The source inventory binds 389 compiled input files to
the candidate, including the package manifest, production code, tests and benchmarks.

| Compiler | Debug full suite | Release full suite | New API and separate consumers |
| --- | --- | --- | --- |
| Xcode 27.2 | Pass | Original compiler canary fails; no other recorded issue | Pass in both producer/client modes |
| Official Swift.org 6.4.2 | Pass | Pass, including the original canary | Pass in both producer/client modes |
| Xcode 27.3 beta | Pass | Original compiler canary fails; no other recorded issue | Pass in both producer/client modes |

Each full suite executes 1,947 tests, with 46 skips: Swift Testing reports 1,775
tests including 18 disabled tests, and XCTest reports 218 including 28 skipped
tests. Extended and strict performance tests remain disabled. The two Xcode
Release runs exit 1 for `LostStoreCanaryTests.swift:47`; the other four runs exit
0. The canary is not waived or converted to a pass.

The separate-module matrix covers both client optimization modes against both
producer configurations for each compiler:

- All 288 positive numerical observations pass: 24 rows in each of 12 executions.
- All 12 omitted-write controls behave as required: each executes all 24 rows,
  identifies the expected 19 mismatches and exits 1.
- All 36 migration compile contracts have the intended outcome: 12 successful
  new-conformer/concrete-closure compilations and 24 specifically diagnosed
  legacy-call/conformer rejections.
- All six producer bindings verify, and the generic witness SIL refers to the
  new `SIMDStorage.withUnsafeMutableElements` requirement.

The final Python policy discovery passes all 628 tests. Workflow validation and
whitespace checks pass. The actual consumer smoke entry point passes eight
documentation examples and all 24 compile contracts, including the three new
migration contracts. Its temporary build products are not retained; input hashes
and command output are. The numerical consumer matrix retains its own binaries,
modules and object identities.

The first complete Python run found a stale generated downstream-guide appendix.
After reconciling the claim ledger and regenerating that appendix, the focused
checks and final 628-test run pass. Both attempts remain in the evidence directory.

Public symbol declarations were extracted from the old and migrated Xcode 27.2
Release modules. Of 4,088 prior declarations, exactly one is removed: the old
`SIMDStorage.withUnsafeMutableBufferPointer` requirement. Exactly two are added:
the new requirement and ContiguousArray implementation. No other compared public
declaration changes; the new total is 4,089. This is a declaration comparison,
not an execution test of binary compatibility.

Independent Astra review reported no remaining material implementation or
documentation findings. A subsequent evidence review identified two verification omissions:
compiler-control input/artifact hashes needed rechecking, and the unchanged-source
check needed the three original November 1 probes. Both checks were added and the
reviewer confirmed the fixes. Consolidation verifies 157 control artifacts per
compiler, including the probe binaries, and binds the frontend and SDK hashes to
the native identity receipts.

## Compiler and release limits

Concrete ContiguousArray still exposes its standard-library inout method. Its
two-row numerical diagnostic stays separate from the migration results. The driver
requires that diagnostic to compile and execute completely, but a reported stale
value is not converted into a numerical pass. The independent original compiler
canary and release-admission policy are unchanged.

The unchanged standalone compiler matrix has 14 positive cells and two deliberate
no-write controls per compiler. Swift.org 6.4.2 passes all 14 positives. Each Xcode
compiler passes 11 and returns the stale value for the same three optimized inout
shapes. All six no-write controls behave as intended. The concrete stdlib
diagnostic likewise retains its stale-value failure for optimized Xcode clients;
the by-value row passes. These failures remain explicit in the retained results.

Passing this migration's tests is not release approval on an affected compiler.
Platform/device coverage, performance acceptance, exact-candidate hosted CI,
downstream resolution and release materials remain separate gates. No new
sanitizer, leak, minimum-OS runtime or performance claim is made here.

## Provenance

Base HEAD is `bdf990847babf567be1cbce22d2606e7102b1a47` in the selected reviewed
worktree. The migration is an uncommitted source delta; per-file hashes identify
the actual compiled candidate. The previous workaround-audit archive and its
repository snapshot remain unchanged.

New retained evidence is at
`/Users/goftin/dev/projects/VectorCore-buffer-api-migration-2026-11-06-evidence`.
It includes pre-migration compile outcomes, guard RED/GREEN logs, fresh build and
consumer records, exact toolchain/SDK/source identities and producer bindings.
The [validation summary](validation-summary.json) records the checked totals,
receipt hashes, complete compiler versions and remaining limits. The evidence
directory retains the [selected manifest](../../../../VectorCore-buffer-api-migration-2026-11-06-evidence/selected-evidence-manifest.json)
and [verification receipt](../../../../VectorCore-buffer-api-migration-2026-11-06-evidence/selected-evidence-verification.json).
Compiled inputs and selected review/policy files are copied into that directory;
this is a scoped source/evidence snapshot, not a full repository export. Build
caches and binaries are excluded from the selected manifest, with binaries retained
separately and checked against their compiler-control and consumer inventories.
The previous excluded inherited-environment diagnostic was not accessed or included.
