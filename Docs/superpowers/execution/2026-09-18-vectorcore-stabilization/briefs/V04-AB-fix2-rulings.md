# V04 A+B fix round 2 — root rulings (binding)

Source: the scoped re-review of fix round 1 (summary in progress.md). Items I1, I3 and all minors were ADDRESSED. I2 was NOT ADDRESSED for narrow reasons.

1. **I2-F1 (Important).** The `mem.alloc` family's timed and warmup bodies run `BenchmarkAllocator.aligned` (MemoryBench.swift:9-15), a `static let` closure pair the fingerprint scanner never hashes. Dropping `deinitialize` or changing the alignment leaves the hash unchanged.
   - Make the guard cover every code that executes inside a declared timed or warmup body for every family. Either teach the scanner to fingerprint named `static let`/`let` closure declarations, or move such closures into listed `static func`s.
   - Re-audit all 18 families for other non-`func` body helpers, such as closures, computed properties or nested functions.
   - Add a negative control: mutating the allocator's deinitialize or alignment must change the hash.
2. **I2-F2.**
   - Fail when ANY listed symbol is missing, not only when all are missing.
   - Support computed properties such as `ProviderObservations.statistics`, or remove that symbol from the family.
   - Add a negative control for a partly missing symbol list.
   - Correct the BodySource.swift:42 comment.
3. **I2-F3.** Correct BodySource.swift:5-7 to say exactly what traps, which is `"""` only, or make raw strings trap too.
4. **I2 comment.** Re-check Boundaries.swift:12-16 ("editing a body without revisiting its record fails") against the final scanner coverage. State exactly what is covered, and what is not: blackHole and the Harness loops are outside `Suites/*`.
5. **N4.** Reword Boundaries.swift:305 "The generic body allocates nothing itself" to "not established". The @escaping capturing closure passed to `any ComputeProvider.parallelReduce` normally heap-allocates a context.
6. **M4 residual.** A PRESENT `schemaVersion` that is a sentinel string ("", "unknown", "unavailable" in any case) or null must not be classified as legacy. Legacy means the key is absent, or it is an integer below 3. A present value that does not parse as an integer is a hard error that the override cannot waive. Add the tests.
7. **Class closure: warmup/timed divergence** (two instances so far, M1). Add a mechanical guard asserting, for each case, that warmup and timed bodies execute the same body helper or closure, modulo explicitly listed and justified differences. If it is infeasible for some suites, guard the ones it can, and list the rest with a justification.
8. **Nits, all cheap; do them:**
   - refresh the stale manifest entry for test_benchmark_selection.py;
   - correct the "4 pairs" wording in the M13 comment and report to 6;
   - in the CHANGELOG M7 entry, mention that toCases throws on duplicate IDs;
   - tag the pretty-header "ns/op" and the med=/p90= fields with explicit units;
   - in the M12 test, add a failure row with a non-empty boundary.

Same rules as before: VectorCore is comments-only (and untouched here); check_benchmark_build_modes.py is untouched; no git commands; no subagents. Re-run the phase-B acceptance checks: Debug and Release BenchmarkContractTests; 36/18 pairs in-process and in the CLI; V01 28/14/28; the Python suites and discovery; every guard's negative controls; and lint with 0 new warnings. Append a "Fix round 2" section to the report, or return it as text.
