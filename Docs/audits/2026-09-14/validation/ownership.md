> Independent reviewer working notes. The [final claim register](../CLAIMS.md) controls priorities and dispositions. Historical temporary paths in replay commands are retained for provenance; portable replay guidance and preserved artifacts are in [the evidence index](../evidence/README.md).

# Independent validation: vectors, storage, hints and cache identity

Revision: `8b02e75bca7055ae11170873d15c06f1538709b7`. Review only. No production/test/repository edits or package builds were performed by this reviewer. Evidence directory: `validate_ownership/` beside this report. Machine-readable register: `validate_ownership.json` (27 deduplicated records, with source locations, exposure, correction, improvement and acceptance for each).

## Decisions that change or sharpen the original review

1. **Keep the public storage, hint, compact COW and cache identity findings.** New actual-library Release probes expand the evidence: every optimized FP32 width exports its fixed count after storage shrinks to four scalars; every compact count from5 through512 aliases; all three INT8 widths decode an empty backing array;496 independent coordinate changes falsely hit the512 FP16 cache. Actual invalid shape/hit/alias is observed. Out-of-bounds writes remain unexecuted static consequences.
2. **Expand PERF1's triangular exposure to the standard tier.** Unique matrix writes at n33 and64 change allocation on every write, just as aligned n65 does. The standard ContiguousArray enum extraction retains a second owner. Fixing only `AlignedBuffer.isUniquelyReferenced()` leaves this full-copy path intact. The public default affected range begins at n33, with m=n(n−1)/2 stored elements. Full scalar construction causes Θ(m²)=Θ(n⁴) element traffic in these paths.
3. **HybridStorage's quadratic traffic survives the instrumentation challenge.** The original counter fixture contains unchanged current storage bodies once the constructor-only integer counters are removed. More importantly, the actual linked Release Float specialization retains storage immediately before its uniqueness test in both outer helper and scalar setter; the failure branches allocate/copy. Actual optimized fill repeats these routes. Counter fixture512 construction still reports514 ordinary plus512 copy allocations=1026. Exact counter totals belong to that fixture; the actual production instruction path independently proves the copy mechanism and complexity.
4. **Pointer stability was an inadequate counterexample.** DynamicVector's before/after address remains the same in fresh Release tests, despite its two-copy instruction path. The allocator can recycle the initial address after the intermediate replacement. The probe only records integer addresses after scoped borrows, so it does not itself retain the storage. Allocation count and generated instructions, not final address alone, decide this question.
5. **C4 is an observed Debug trap and a contract violation; Release crash is not reproduced.** Naturally Float-aligned offset+4 pointers work at dimension3, trap at4/5/16/17 in Debug, and work for all those dimensions on the current ARM Release executable. SIMD `load(as:)` requires stronger alignment than the API's “preferred” wording. Keep exact build/shape qualification.
6. **Keep V4 qualified.** The384 implementation performs four permanent bindings while its siblings use scoped restoration. This warrants a consistent reviewed lifetime strategy. No optimizer miscompile was demonstrated, and a mechanical spelling replacement does not prove Float-array source alignment. Layout compatibility and permitted related-type access must be addressed explicitly.
7. **FP16 shape and FP16 pointer lifetime are different questions.** FP16 vectors have immutable storage and exact-length constructors; there is no public FP16 shape hole equivalent to INT8. Their three escaping pointer getters do have an unsafe lifetime/API gap. The complete-vector FP32 Codable paths reject empty arrays; synthesized INT8 Codable accepts empty backing storage.
8. **Correct performance/cache scope.** Only `SoAFP16Cache512` exists. The384/768/1536 key constructors are unused future surface. Sampled hashes collide deterministically in all four optimized widths, but equal signed zeros have coherent hashes in the fresh checks. Hash collisions themselves are legal; using a digest as identity is not repaired merely by hashing more coordinates.

## Fresh evidence and independent expectations

`Ownership.swift` is an optimized client linked against actual current Release objects with `@testable import VectorCore` for otherwise dormant tier tests. It contains no storage implementation replacements, allocation counters or provider stubs.

- Shape test never dereferences the invalid part of a reported view; it compares fixed advertised count against owned collection count×4.
- COW expectation is ordinary struct copy isolation. Every compact count5...512 fails (508/508). Public matrix sizes4/16/32 fail;3/33/64/65 controls retain the original.
- Weak Token lifetime expects scope-only owners to disappear; compact matrices4/32 retain the token after scope, while3/33/65 release it. No dangling pointer is dereferenced.
- Throwing bulk mutation retains write19 for compact/aligned tiers but loses it in inline1/4 and standard513/2048. The ordinary closure original fixture independently loses standard writes. Both inner InlineBuffer and outer tier writeback must honor the chosen exceptional-exit policy.
- Hint expectations start from valid unit `[1,0]` and ordinary writes2/3; cached magnitude1 is unambiguously stale.
- Signed-zero tests change coordinate0, which is actually hashed, and verify both equality and hash coherence for each optimized width.

`ShapeRoutes.swift` is another actual Release client. It encodes legitimate default values then changes only JSON `storage` to an empty array; all three INT8 decoders accept. Optimized FP32 controls reject `[]`. Its cache test caches one zero candidate then queries512 independent one-coordinate changes without modifying the cached snapshot;496 incorrect hits match exactly the omitted interior coordinates. Binary mutator results independently prove longer RHS truncation across all three mutators and generic operators.

`AlignmentDebug.swift` shares the same source as Ownership at build time and links actual Debug objects. Subprocesses isolate expected misaligned-load traps, providing valid allocated capacity and finite nonzero values. `alignment-results.json` preserves exit statuses and separate full logs. No sanitizer claim is made.

### COW fixture audit

`counter-source-audit.json` checks complete unchanged source bodies for:

- UnifiedStorage.swift
- HybridStorage.swift
- DimensionStorage.swift
- VectorStorage.swift

Only `Counter.allocations += 1` and `Counter.copies += 1` are removed for the comparison. Counters modify no owner/reference values. The stub floating-point provider supplies ordinary copy/fill arithmetic where needed; numerical behavior is not inferred from it. The actual Release instruction evidence closes the remaining concern that different compilation/instrumentation could have manufactured the copying. `hybrid-release-excerpts.txt` contains complete specialized helper/setter/fill disassembly. The full linked-client disassembly remains local for resolving calls.

The superficially similar default `DimensionStorage<D512>` path reports zero allocations for ten scalar writes and one buffer mutation in the original optimized fixture. Its uniqueness-reporting getter still returns false. This does **not** justify a blanket quadratic-cost claim for generic vector convenience factories. Explicit `.hybrid` mode and Float/Double threshold differences need class-wide regression coverage.

## Original subsection crosswalk

### vectors.md

| Original subsection / observation | Disposition |
|---|---|
| Reviewed surface; verification limitations | Historical coverage description, not proof. Fresh probes and current source establish the findings below. |
| V1 storage invariant | OWN-01 and OWN-02; confirmed with expanded width/decoding controls. |
| V2 reciprocal division and generic normalization | Numeric reviewer owns validation; no independent numerical conclusion duplicated here. |
| V3 in-place dimension checks | OWN-06; all longer-RHS operations reproduced in Release; short-buffer overread stays source-derived. |
| V4 384 rebinding | OWN-09; qualified contract/lifetime repair, no miscompile. |
| V5 sampled hashes | OWN-07; all widths checked, signed-zero counterexample rejected. |
| V6 batch factory zero dimension | Execution/API reviewer owns validation. |
| V7 approximate equality | Numeric reviewer owns validation. |
| Extension-only operations / narrow protocol architecture | OWN-27; real generic dispatch consequences, no universal performance claim. |
| StaticDimension.Storage unused customization | OWN-20; source-confirmed API clarity issue. |
| Factory optimal wording versus concrete output | OWN-21; source-confirmed, compatibility-sensitive design opportunity. |
| VectorCoreOperations allocation opportunities | OWN-22; source-visible temporaries, no timing claim. |
| Generic convenience factories repeatedly mutate | OWN-26; include in mechanical coverage; reject blanket default generic quadratic extrapolation. |
| Unmeasured dimension/optimized comments; magnitudeSquared explanation | OWN-23; remove/anchor claims; numeric reviewer owns detailed norm outputs. |
| Strength: fixed constructors validate shape | Source confirmed at entry; fresh empty FP32 decoding controls support this. Post-construction storage remains separately unsafe. |
| Strength: scoped512/768/1536 Float views / ordinary COW | Source confirmed scoped restoration and ContiguousArray ownership; shape setter and alignment obligations limit any blanket safety assertion. |
| Strength: generic norm scaling/nonfinites | Numeric reviewer owns validation; do not infer normalized result safety from norm alone. |
| Strength: nominal shape/construction/factory/producer tests | Existing tests cover nominal scenarios; no new broad “all paths correct” claim. VectorProducer default single-text catches empty output through explicit guard and propagates thrown errors; no additional defect established. |
| No new defects in VectorData/operator declarations/SIMDStorage within contracts | Review disposition only, not a universal correctness guarantee or exhaustive new test campaign. |

### memory_storage.md

| Original subsection / observation | Disposition |
|---|---|
| Outcome/Coverage | Current ownership path confirmed: DynamicVector uses HybridStorage; TriangularMatrix is TieredStorage's production owner; slice/bulk/experimental paths are dormant. |
| MS-1 compact matrix alias | OWN-05; all508 compact counts reproduced plus public tier controls. |
| MS-2 Hybrid copying | OWN-12; fixture source audit and independent actual Release instruction proof. |
| MS-2 rejected DimensionStorage extrapolation / false getter | OWN-26; maintain distinction between default and explicit hybrid modes. |
| MS-3 aligned matrix full copies | OWN-13; expanded to standard tier and public n>=33. |
| MS-4 compact element destruction | OWN-10; fresh actual Release across tier controls. |
| MS-5 alignedCopy | OWN-11; replayed initialized-reference leak and static incompatible branch contracts. |
| MS-6 standard bulk lost writes | OWN-14; ordinary/throwing behavior differentiated; prerequisite to new bulk route. |
| MS-6 inline throw writeback | OWN-14; newly executed across tiers. |
| MS-6 escaped slices | OWN-15; source-confirmed dormant lifetime problem; not DynamicVector.slice. |
| MS-6 move-only into | OWN-16; gated-out static defect; do not enable without separate audit. |
| Existing SoA access-after-transfer contract | OWN-24; already documented invalid access, not new supported-use bug. |
| @unchecked Sendable / synchronization | OWN-24; caller obligations documented; no TSan or supported-caller race reproduction. |
| Count/rounding/type-alignment obligations | OWN-24; documented unsafe constraints; no new representable-input overflow. |
| AlignedDynamicArrayStorage fallback guarantee | OWN-25; source concern retained as unverified failure path. |
| Custom SoACompatible conformers | OWN-24; built-in evidence does not establish arbitrary conformer validity. |
| Strength: frozen SoA formulas / logical versus rounded counts | Source/contract confirmed; existing SoALayoutContractTests and SoAPageAlignTests cover relevant examples; no GPU validation claimed. |
| Strength: AlignedValueStorage detaches; fixed wrappers check count | Current source shows detachment before scalar/buffer mutation and count guards. Limited read-based strength, not a new allocation test across13 wrappers. |
| Strength: COWDynamicStorage detaches but temporary array / obsolete backend | Source confirmed ownership pattern and unused DynamicVector backend description; OWN-23 tracks architecture/docs, not new correctness failure. |
| Strength: matching allocator/deallocator and initialization | Source flags select free versus Swift deallocation in the inspected Float paths; existing page tests cover transfer identity/content. Do not generalize to CompactBuffer's generic destruction. |
| Overlapping ownership architecture and weak class-wide lifetime coverage | OWN-23; consolidation begins with verified failure classes. Acceptance in OWN-05/10/12/13/14 closes storage-tier COW/destruction classes. |
| Verification instrumentation / limitations | Audited/replayed, then strengthened with actual objects and disassembly above. No wall-clock, RSS, allocator-failure, ASan, TSan or GPU experiment. |
| Cross-component Dynamic storage / singleton triangular import | Dynamic storage OWN-01; singleton triangular import is execution/API reviewer scope. |

### Added assigned cross-component observations

A2/IF5→OWN-03; A3/RP1→OWN-04; C4/SIMD2→OWN-08; FP16 escaped pointers→OWN-17; INT8 shape/decoding→OWN-02; FP16 cache capacity→OWN-18; FP16 cache size model/comment/key/LRU complexity→OWN-19. Explicitly false manually supplied hints are caller assertions, while the reproduced stale hints began correct. VectorIndex's first-query-only batch hint issue is downstream-owned and belongs the execution/integration crosswalk.

## Acceptance and sequencing

Use the JSON register's per-record acceptance requirements as the implementation checklist. In particular:

1. Protect public shape, hints and cache identity first. Keep FP32 decoder controls, and ensure an INT8 initializer repair also covers synthesized decoding and public collection mutations.
2. Treat compact sharing/destruction, standard scalar detach, aligned detach and bulk throw/writeback as one tier-boundary ownership suite. A performance repair must not activate dormant lost writes.
3. Fix HybridStorage ownership with optimized allocation evidence and shared-copy controls. Do not promote source intuitions or pointer stability into an allocation result.
4. Resolve unsafe alignment/binding/lifetime contracts explicitly. Behavioral tests supplement, but cannot prove, memory binding and nonescape obligations.
5. Preserve documented SoA ownership and ABI/layout constraints. Consolidation and performance claims need actual consumer compatibility and coordinated Release measurements.

## Reproduction commands

Run from the repository root (compiler cache permission was used when required):

```sh
python3 /private/tmp/vectorcore-review-2026-09-14/build_probe.py /private/tmp/vectorcore-validated-audit-2026-09-14/validate_ownership/Ownership.swift release
/private/tmp/vectorcore-validated-audit-2026-09-14/validate_ownership/Ownership
python3 /private/tmp/vectorcore-review-2026-09-14/build_probe.py /private/tmp/vectorcore-validated-audit-2026-09-14/validate_ownership/ShapeRoutes.swift release
/private/tmp/vectorcore-validated-audit-2026-09-14/validate_ownership/ShapeRoutes
python3 /private/tmp/vectorcore-review-2026-09-14/build_probe.py /private/tmp/vectorcore-validated-audit-2026-09-14/validate_ownership/AlignmentDebug.swift debug
/private/tmp/vectorcore-validated-audit-2026-09-14/validate_ownership/AlignmentDebug 4
/private/tmp/vectorcore-validated-audit-2026-09-14/validate_ownership/Ownership 4
/private/tmp/vectorcore-review-2026-09-14/memory-probes/cow-probe
/private/tmp/vectorcore-review-2026-09-14/memory-probes/tiered-probe
otool -tvV /private/tmp/vectorcore-validated-audit-2026-09-14/validate_ownership/Ownership
```

Logs: `ownership-release.log`, `shape-routes-release.log`, `alignment-results.json`, per-dimension alignment logs, `cow-probe-replay.log`, `tiered-probe-replay.log`, `counter-source-audit.json`, `hybrid-release-excerpts.txt`. `build.log` preserves an initial sandbox/compiler-cache failure; subsequent approved compiler commands succeeded. Initial client syntax errors were corrected before any successful execution and do not establish library behavior.

## Primary memory-model references

The source-contract reasoning distinguishes layout compatibility from valid typed access and scoped pointer lifetime. Swift's [UnsafeRawPointer migration guide](https://www.swift.org/migration-guide-swift3/se-0107-migrate.html) describes temporary rebinding and forbids escaping its pointer. [SE-0107 memory model](https://github.com/swiftlang/swift-evolution/blob/main/proposals/0107-unsaferawpointer.md) explains bound-type requirements and raw-load alignment. These support the API-contract analysis, not a claim of a reproduced384 miscompilation.
