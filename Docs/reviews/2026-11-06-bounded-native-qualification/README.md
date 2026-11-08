# Bounded native qualification — 2026-11-06/07

**The draft checker accepts the pinned native evidence. The compiler remains
known-failing, the proposed support restriction is not adopted, and 0.4.0 is not
admitted for release.** This completes the native evidence assessment and refreshes
the optional C-mode and generic platform compilation records. Runtime expansion
is the next proposed phase; see [coverage-decision.md](coverage-decision.md).

The user selected “Continue bounded qualification” after the
[residual-exposure proposal](../2026-11-06-workaround-qualification/qualification-proposal.md).
That authorized this investigation, not a compiler waiver or release-policy change.
Shipping Swift/C sources, numerical expectations, original compiler canaries and
CI admission workflows were not changed in this phase.

## Candidate and scope

| Identity | Pinned value |
| --- | --- |
| Reviewed worktree | `/Users/goftin/dev/projects/VectorCore-fixes-2026-10-27-84798d3` |
| Base HEAD | `bdf990847babf567be1cbce22d2606e7102b1a47`, plus retained uncommitted migration |
| Compiled inputs | 389 files: `Package.swift`, `Sources`, `Tests`, `Benchmarks` |
| Compiled-input manifest SHA-256 | `f2869bf68bb9286d33f2107566a2fb53d6913f95c76dced68d9affde448bd6eb` |
| Compiler | Released Xcode 27.2, build 27B52 |
| Swift frontend SHA-256 | `448457da9a8a56741f2c8c4b8931e85b7377fc009c16918cd8ac1a8a3210ea1f` |
| Native SDKSettings SHA-256 | `82165b15ee21419d4cffaba8895797391b908ac9499ba70f561f3e1bd16758d6` |
| Native consumer target | `arm64-apple-macosx14.0`, Swift language mode 6, `-Onone` / `-O` |
| Actual runtime | macOS 27.0.1, build 26A434, arm64 |

The base commit alone does not identify the tested candidate. Deployment-target
metadata does not establish execution on macOS 14 or on Intel hardware.

## Native assessment

[`bounded_native_qualification.py`](../../../Scripts/ci/bounded_native_qualification.py)
and its adjacent data profile verify one retained evidence selection. The checker
does not build or execute Swift, import historical evidence scripts, alter test
expectations, or participate in the current CI aggregate. A new source/compiler/
SDK/runtime candidate requires reviewed evidence and a new profile.

The fresh parent invocation verified **1,140 distinct canonical files** and returned:

```text
nativeEvidence=accepted; compilerProfile=known-failing; releaseAdmitted=false
supportPolicyAdopted=false
```

This is acceptance of the selected evidence under the proposed investigation
boundary. It is not proof of correctness for arbitrary callbacks, conformers or
generic/inlinable downstream compositions. The native Swift executions are reused
from the unchanged migrated candidate; they were not rerun merely for changes to
the Python checker.

| Selected Xcode 27.2 evidence | Verified observation |
| --- | --- |
| Original compiler controls | 14 positive cells: 11 correct, three optimized inout shapes wrong; two no-write controls behave as expected |
| Migrated API clients | 96 positive observations across Debug/Release producers and Debug/Release clients; four omitted-write executions reject with the complete expected 24-row outputs |
| Migration contracts | 12 compile contracts: new conformers accepted, legacy generic calls/conformers rejected with the recorded diagnostics |
| Native suites | Each producer reports 1,947 executed and 46 skipped/disabled tests; Debug has no failures, Release retains the single original lost-store canary failure |
| Residual exposure | 34 positive observations: 31 correct and three Release reconstructed-inout failures; two complete omitted-write controls reject |

The residual failures are valid synchronous, nonescaping API compositions. They
are not reclassified as undefined behavior or caller misuse. The proposed boundary
excludes additional descriptor/view/pointer-bearing-wrapper inout handoffs after
the specifically reviewed borrow-entry adapter, transitively through helper and
conformer code. Passing inout shapes do not receive exceptions. That restriction
remains a proposal, not a proved complete partition of safe and unsafe code.

The checker validates pinned receipts, numerical inventories and values, process
completion, native outcomes, producer/client bindings, linked objects, module and
modulemap selection, compiler flags and identities, host identity, and final file
integrity. Hashes establish consistency with reviewed captures; they are not
independent authentication or protection against an actively hostile filesystem.

## Fresh optional C-mode results

All four Debug/Release × C-on/off cells completed with the unchanged existing
driver and exact Xcode frontend:

- 144 public INT8 numerical and routing observations across dimensions 512, 768
  and 1536, four quantization modes and three metrics.
- 40 direct C controls at lengths 0, 1, 3, 4, 15, 16, 17, 31, 32 and 33.
- Eight normal benchmark CLI executions passed; four deliberately mismatched
  configuration requests rejected without producing a report.

Verification checked 37 command records, 52 retained artifact entries and 139
linked objects per cell. The 349-file driver snapshot was unchanged during capture.
Three qualification-tool/test files changed afterward to address review findings;
the parent verification records those exact differences. The 389 compiled inputs
and the C-mode driver/producer identities remain unchanged.

SwiftPM selected **swiftbuild**. Producer commands target arm64 macOS 14; the
standalone route consumers use native compiler defaults without an explicit target.
These runs establish the tested numerical and dispatch results on the current
host. They do not qualify performance, all FP32 C entry points, INT8 extrema/
overflow cases, alternate architecture implementations or platform runtimes.

## Fresh platform compilation and reporting caveat

Both library products completed Release compilation for generic iOS Simulator,
tvOS Simulator, visionOS Simulator and Mac Catalyst: **eight exit-zero actions**,
60 retained artifacts and 12 product objects containing arm64 and x86_64 slices.
Metadata records SDK 27.2 and deployment floors 17.0 for iOS/tvOS/Catalyst and 1.0
for visionOS. No runtime tests executed in these actions.

The four original `VectorCore` console logs contain seven messages saying a
SwiftCompile command failed with exit code zero and produced no further output.
Their structured activity records report zero errors and four warnings per
platform; the warnings concern deprecated CBLAS interfaces. These original logs
remain retained, and **`diagnosticClean=false`** remains the correct aggregate.

One fresh iOS discriminator omitted `-quiet` and added an xcresult bundle. It
completed with zero console/structured errors; the result contains two distinct
CBLAS warning issues. Both architecture driver commands match the originals after
normalizing only the DerivedData root; source lists and normalized output maps
also match. This supports a build-reporting inconsistency. Because two reporting
options changed, it does not isolate `-quiet` as the cause or clear all other
platform diagnostics. It supplies no numerical runtime evidence.

## Tooling review and validation

Independent review identified a material integrity flaw in the initial checker:
later command checks could overwrite earlier pinned digests. Registration now
preserves the first canonical-path expectation, commands require an existing
receipt pin, and the final recheck uses those preserved expectations. Four
regression tests cover altered valid receipts, matching altered streams/binaries,
canonical aliases and changes after successful reads. Independent re-review found
no remaining material issue in the reviewed changes.

Earlier parent review also removed dynamic execution of an evidence helper:
consumer oracles are literal profile data. Tests verify that helper side effects
and preloaded modules cannot supply them. An unverified compiler remains labeled
`unverified` until its observations are checked.

The first full Python run retained 640 tests with two integration failures:
missing subprocess timeouts in two new tests, and the lexical native-layout rule.
The test subprocesses now have 15-second bounds. Two narrowly justified allowlist
entries cover only this candidate-pinned retained-evidence validator and its
synthetic fixtures; build-product discovery drivers remain subject to the rule.
The focused checker/process/layout suites passed 16/15/6 tests. The final full
suite passed **644 tests**; validation of nine GitHub workflow/template files and
`git diff --check` also passed. Frozen policy inputs remained unchanged. Detailed
results are recorded in [validation-summary.json](validation-summary.json).

## Remaining release work

1. Decide the runtime coverage required for release; refresh available iOS and
   Catalyst execution with platform-aware public consumers and fresh skip counts.
2. Prepare an app-hosted physical iOS consumer or test target. The previous
   tool-hosted package attempt never reached test bodies.
3. Obtain or explicitly disposition minimum-OS, native Intel, tvOS/visionOS and
   representative device evidence. The only fresh simulator runtime inventory
   here is iOS 27.0 (24A434); no physical-device inventory was performed.
4. Qualify required legacy CI tuples and one exact candidate in hosted CI. Local
   Xcode 27.2 results do not replace Xcode 16.2/26.3 jobs or authorize changing them.
5. Refresh the chosen sanitizer scope for the migrated candidate, then measure
   performance in an exclusive controlled window. Historical leak and sanitizer
   limits remain; benchmark CLI smoke is not performance acceptance.
6. Review final support wording and any admission-rule change with the user,
   then complete downstream release-reference checks, release materials and branch work.

## Evidence

Evidence root:
`/Users/goftin/dev/projects/VectorCore-bounded-native-qualification-2026-11-06-evidence`.
The directory date reflects the start of this phase; closure continued November 7.

- [Fresh native assessment](/Users/goftin/dev/projects/VectorCore-bounded-native-qualification-2026-11-06-evidence/native-assessment/assessment.json)
- [Parent build verification](/Users/goftin/dev/projects/VectorCore-bounded-native-qualification-2026-11-06-evidence/parent-build-verification.json)
- [C-mode verification](/Users/goftin/dev/projects/VectorCore-bounded-native-qualification-2026-11-06-evidence/c-kernels/verification.json)
- [Platform summary](/Users/goftin/dev/projects/VectorCore-bounded-native-qualification-2026-11-06-evidence/platform-builds/summary.json)
- [Nonquiet discriminator](/Users/goftin/dev/projects/VectorCore-bounded-native-qualification-2026-11-06-evidence/platform-builds/diagnostic-investigation/nonquiet/comparison.json)
- [Final policy command](/Users/goftin/dev/projects/VectorCore-bounded-native-qualification-2026-11-06-evidence/policy-tests-final/discovery.json)

Earlier migration and residual manifests remain immutable. The final selected
manifest and verification receipt accompany this phase; build caches, raw device
identifiers, and the excluded inherited-environment diagnostic are not selected.
