# Unsafe-buffer workaround audit and validation

Status: bounded audit and validation completed November 6, 2026. By-value cases pass; Xcode compiler acceptance and the legacy public API exposure remain unresolved.

## Finding and decision

The proposed by-value boundary is a useful, testable workaround for the examined calls. Most VectorCore numerical interfaces already use it. The audit found one explicit production descriptor-`inout` declaration: the public `SIMDStorage.withUnsafeMutableBufferPointer` callback requirement. It is publicly reachable even though no internal generic numerical caller of that mutable requirement was found.

A separately compiled client of the actual library reproduces stale `2.0` after writing `1.0` through that requirement on both Xcode 27.2 and Xcode 27.3 beta at `-O`. Both its generic and concrete ContiguousArray forms fail. The corresponding by-value writer bridges return the correct `1.0`. The unchanged independent compiler canary remains a separate failure on the affected Xcode compilers.

Production Swift/C code, existing compiler probes, known-issue predicates, release admission and public support policy are unchanged. This audit does not claim to have repaired a compiler, removed the legacy API exposure, or admitted 0.4.0.

## Completed native results

| Compiler | Unchanged positive compiler probes | Full Debug suite | Full Release suite | By-value consumer observations |
|---|---:|---|---|---:|
| Xcode 27.2 | 11/14; three optimized shapes fail | Pass | One failure: unchanged compiler canary | 96/96 pass |
| Xcode 27.3 beta | 11/14; same three shapes fail | Pass | One failure: unchanged compiler canary | 96/96 pass |
| Swift.org 6.4.2 RELEASE, restored complete payload | 14/14 pass | Pass | Pass | 96/96 pass |

Each native suite reports 218 XCTest tests (28 skipped) and 1,768 Swift Testing tests (18 disabled): 1,940 executed tests and 46 skips per configuration. XCTest has zero failures in all six runs. The only Swift Testing issues are the two Apple Release compiler-canary failures. The extra one-test subprocess summaries in the logs are not added to these main-run counts.

There are **288 passing by-value observations**: 24 cases × two client optimization modes × two library configurations × three compilers. All **12 omitted-write executions** matched the entire expected negative table and exited 1. Each compiler's two standalone no-write controls were also correctly rejected. These are numerical checks, not merely successful compilation or process exits.

The legacy public-exposure diagnostic remains separate: optimized clients linked against either Debug or Release library products return stale bits for both concrete and generic inout calls on each Xcode compiler. The matched by-value calls pass. All four diagnostic rows pass in Debug clients and under the restored Swift.org compiler. Changing only the library build configuration therefore does not eliminate client-side exposure.

## What was added

- `Scripts/ci/consumer_buffer_value.py`: a bounded, retained-evidence driver for actual public library consumers, with independent numerical oracles. It compiles a separate witness module and client with ordinary `-Onone` and `-O`, against the supplied exact Debug or Release library products.
- `BufferValueConsumer.swift` and `BufferValueWitness.swift`: 24 cases covering concrete/generic borrowing, an external conformer, retained generic witness dispatch, descriptor aliases, empty and singleton buffers, untouched tails, COW copies, mutation followed by throw, Float/Double/SIMD4 values, inline/heap storage boundaries and representative public vector/inlinable operations.
- `BufferInoutExposure.swift`: a diagnostic of the existing concrete and generic public exposure, reported separately from the by-value cases. Its expected values are never waived into a numerical pass.
- `test_pointer_descriptor_inout.py`: a scoped declaration review gate, discovered by the existing Python CI job. It follows supported source aliases and explicit same-type constraints, and pins the existing exception to the complete top-level SIMDStorage protocol shape. Its exception records unresolved exposure, not safety.
- Python consumer-judge tests reject missing/duplicate/extra rows, stale bits, false completion counts, process failures and partial or corrupted negative controls. Fixture-retention discovery includes the new driver.
- A pre-existing output-path policy false positive is corrected: the two exact reviewed XCTest executable-identity comparisons are allowed, pinned to their file and complete top-level function signature/body hashes. This does not allow new directory-name heuristics or waive other path/write rules.

The declaration guard and driver-judge tests join existing Python discovery. The native consumer driver was run explicitly for this audit; it is not a newly required hosted CI job. Adding it to exact-candidate hosted qualification remains follow-up work.

The omitted-write build changes only the companion writer. The independent oracle requires the complete expected wrong-result table, exit 1, and 19 mismatching cases; the four empty cases and the inlinable arithmetic control remain correct. An arbitrary crash, missing output or corruption is not accepted as a successful negative control.

The witness module's separately emitted SIL must retain a `witness_method` call to `SIMDStorage.withUnsafeMutableBufferPointer`. The retained Xcode output places that call in `withValueBuffer`. This is evidence about the emitted generic body, not a pass-by-pass trace of the executed binary.

Three Astra agents independently reviewed source reachability, consumer/compatibility coverage, and declaration/policy guards. Final native evidence review rechecked the three identical 388-entry source inventories, 522 command streams and six producer receipt hashes. Independent review of the output-path correction found no blocking exception broadening. Earlier failed policy attempts are retained, including the pre-existing false positive and a new negative fixture that was corrected after another guard rejected its configuration-specific path literal.

Final Python discovery passed **624 tests**. The GitHub validator passed all nine YAML/template files and workflow policies, and `git diff --check` passed. The final script/workflow inventory remained unchanged during those checks. The first complete 624-test attempt failed on the new negative fixture; its original receipt and streams remain at `policy-tests/final-discovery.*`. The passing rerun is retained separately under `policy-tests/final-validation/`.

## Compiler, source and host scope

The compiled library/test/benchmark source is `bdf990847babf567be1cbce22d2606e7102b1a47` in the user-selected reviewed worktree. Every native run records executable-input hashes and verifies they are unchanged afterward. The new audit fixtures and Python policies are outside those compiled targets. Each consumer result is bound to its producer's build receipt and module/object hashes; the driver's `--library-configuration` argument alone is not proof of a producer configuration.

| Compiler | Frontend SHA256 | SDK |
|---|---|---|
| Xcode 27.2 | `448457da9a8a56741f2c8c4b8931e85b7377fc009c16918cd8ac1a8a3210ea1f` | Xcode 27.2 macOS SDK |
| Xcode 27.3 beta | `93a7373a7f558b211b9820797d40d7554a0fef2c917d039fa0fa894b60fd55eb` | Beta macOS 27.3 SDK |
| Swift.org 6.4.2 RELEASE | `956438a61893525d7ce28c21b6634093f15b0c4bae7fe6eb7d98fba0c1c5e301` | Same Xcode 27.2 SDK as the first row |

The target is `arm64-apple-macosx14.0`; the execution host is macOS 27.0.1 on arm64. A deployment target of 14 is not a macOS 14 runtime test. The native SwiftPM build system is selected explicitly. Production optimization flags and attributes are unchanged. The new fixtures deliberately use `@inline(never)` to preserve helper boundaries under ordinary `-Onone` and `-O`; they use no `@_optimize(none)` or optimization-pass suppression. `VECTORCORE_TEST_EXTENDED=0` and `VECTORCORE_STRICT_PERF=0` preserve the normally enabled test workload; incidental timings are not performance acceptance.

### Recovered control toolchain

The first fresh Swift.org attempt could not compile because its retained temporary payload had only one shim header and no `SwiftShims` module map. That attempt has four compile-error cells and an invalid-manifest failure; it is a setup failure, not a numerical regression or completed compiler matrix.

The exact official package was downloaded again and unpacked in a new isolated directory. Its SHA256 matches the previously pinned `0a7d8489900d21dbf5f0187cb570e646a4c0226d0ddef2cbde8c7f54dfe79736`; its installer signature and notarization verify. The restored frontend matches the hash above, and all 27 shim files are hashed. The old temporary tree was not changed. This also illustrates why retaining only a compiler executable hash is insufficient to reproduce an operational toolchain.

## Public API migration experiments

The reduced compatibility experiments establish that simply deleting `inout` is not a compatible one-line fix:

1. The empty ContiguousArray conformance no longer satisfies the by-value requirement.
2. An explicit same-name adapter can satisfy it, but an ordinary inferred concrete closure becomes ambiguous between it and the standard-library overload.
3. An existing external conformer fails the changed requirement.
4. A differently named additive adapter compiles with both conformers, while leaving the old API available.

The compiling variants also executed correctly in both optimization modes in their reduced bodies. That does not repair the demonstrated failing actual-library call. The detailed inventory, exact limitations and migration tradeoffs are in [source-audit.md](source-audit.md).

## Remaining release work

The current [release-admission policy](../2026-11-05-release-gates/release-admission.md) still requires the unchanged positive compiler matrix to pass. Passing the by-value consumer suite does not satisfy that requirement on a failing Xcode build.

A future API migration should use an explicitly distinct by-value requirement/name and account for downstream conformers; a same-name overload is contradicted by the compatibility probe. An additive convenience method can guide callers but cannot be described as removing the legacy exposure. Even removal of the protocol requirement would leave the standard-library method available through concrete public ContiguousArray storage. A source-distribution policy that accepts an affected compiler would need a separate, explicit support decision and continued reporting of the failing compiler/public-exposure diagnostics.

Minimum-OS/device/Intel/tvOS/visionOS/Catalyst coverage, controlled performance comparisons, exact-candidate hosted CI, downstream resolution and release materials remain separate gates. This audit supplies no new sanitizer, leak-freedom, physical-device or App Store eligibility claim.

The declaration guard is intentionally narrower than Swift type checking: imported aliases, unconstrained generic instantiations, macros, arbitrary pointer-bearing state and generated source are not proven safe. It scans `Sources/**/*.swift`; the existing Tests canaries and Benchmarks directory remain outside this new prohibition. Existing readback guards stay intact. Some historical source comments describe source shapes rather than durable compiler workarounds; [source-audit.md](source-audit.md) records that distinction.

## Evidence

Durable execution root: `/Users/goftin/dev/projects/VectorCore-workaround-audit-2026-11-06-evidence`.

The root contains fresh compiler matrices, complete native logs and command records, input identities, consumer binaries and SIL, omitted-write controls, reduced compatibility experiments, the preserved failed control setup, and the verified recovery package. Build trees and caches are retained locally; the final review manifest identifies the selected report/command/observation evidence separately.

- [Consolidated machine-readable results](validation-summary.json) record the six native suites, consumer matrices and final policy outcome. The consolidation checks current compiled-source and policy-input hashes, consumer artifacts, command streams and producer bindings.
- [Selected evidence manifest](/Users/goftin/dev/projects/VectorCore-workaround-audit-2026-11-06-evidence/selected-evidence-manifest.json) inventories retained review files by SHA256; its [verification receipt](/Users/goftin/dev/projects/VectorCore-workaround-audit-2026-11-06-evidence/selected-evidence-verification.json) records the final rehash.
- [Final policy execution receipt](/Users/goftin/dev/projects/VectorCore-workaround-audit-2026-11-06-evidence/policy-tests/final-validation/final-discovery.json) retains exact command, executable identity, elapsed time and stream hashes. Native and consumer command records likewise retain exact arguments for replay into fresh output locations.

The selected manifest excludes build/cache trees, the expanded toolchain and binary/package payloads; their retained identities remain in the detailed native, consumer and recovery records. `repository/` is a selected snapshot of audit changes and helper files, not a complete source export. No source outside the explicitly selected audit evidence was added to this manifest.

The earlier `2026-11-06-compiler-research-handoff` directory and ZIP remain unchanged. The external Linux reproduction remains reported corroboration; no Linux execution is counted in this audit.
