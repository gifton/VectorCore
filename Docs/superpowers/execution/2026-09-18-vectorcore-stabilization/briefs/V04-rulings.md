# V04 root rulings on phase-1 questions (2026-10-03) — binding

These rulings answer the phase-1 inventory (`phase1-inventory.md` §10.2). The approved brief `briefs/V04.md` stays binding wherever these rulings are silent.

## Before phases B and C
- **Q1 — Route support.** Companion availability follows route support. Record the existing `shouldUseSoA`/`shouldUseMixedPrecision` outcome as `defaultEligible`; do not enforce it.
- **Q2 — Unavailable.** SoA FP16 squared-Euclidean is reported as unavailable, with its reason, at all widths. Do not add a sqrt family.
- **Q3 — Separate opt-in suite.** Use `--suites boundary`. It is never default and never auto-paired.
- **Q4 — Yes.** Add a direct-kernel reusable-output AoS FP32 counterpart (`range_euclid2_D` into a caller-owned buffer) using the existing kernel only, with no production change. Label the existing provider cases with their actual API boundary.
- **Q5 — Keep.** Keep the default `optimized-soa`/`optimized-fp16` Float-sum consumers, with their consumer identity recorded. They read every output.
- **Q6 — OVERRULED: migrate, do not merely label.** The brief requires every timed route to consume the numerical result. A D-element output observed at one coordinate does not consume it.
  - `normalize.success` moves to the full `u64-bitfold-sink-v1` fold. Its consumer and boundary identity change; the fixture does not.
  - "Partial" must not label consumption that is actually complete:
    - an error-path outcome (`zeroFail`) is its full result: label it `error-outcome-sink-v1`;
    - a scalar output consumed whole: label it `scalar-sink-v1`.
  - Memory-suite cases whose work retention rests on escaped pointers get an accurate label, e.g. `escaped-pointer-retention-v1`, and the limitation is stated in the ledger.
  - If any multi-element numeric output would still be partially consumed after this, stop and report it to root.
- **Q7 — Split by whether the boundary exists.**
  - Missing or legacy boundaries are waivable only with `--allow-incompatible`, which is V02 policy.
  - Any present-but-different boundary is NOT waivable, including a timingScope mismatch. The brief requires exact full boundary identity, and different scopes never pair.
- **Q8 — Yes.** Unavailable records are not failures. A selection made up only of unavailable entries exits 1.
- **Q9 — Yes, constrained.**
  - The `MeasurementPhases` observer seam lives only in the `VectorCoreBenchSupport` target, never in the `VectorCore` or `VectorCoreBenchmarking` products.
  - It is task-local, with no global mutable state.
  - Use the narrowest access level that works (`package` if sufficient, `public` only if a cross-module test requires it).
- **Q12 — Existing `BenchmarkContractTests` files.** No new test file and no CI-policy edits.
- **Q19 — Yes.** Keep `fixtureVersion` "V02-inputs-1" and bump only the boundary version.

## Before phases D, E and F
- **Q10 — Include, and close the class mechanically.**
  - Include the handle-stride memory assertions in `MixedPrecisionPart1ValidationTests` and `MixedPrecisionKernelsTests`. Change labels and memory model only; numerical assertions stay.
  - Add a lexical guard rejecting handle-stride arithmetic presented as payload or memory facts in tests. It needs a negative control.
  - The `QuantizedKernelsTests` instance is included too, once T02 is integrated. Root will rebase the V04 workspace onto that integration before phase E.
- **Q11 — Ledger as retained-unverified runtime text.** Do not change the returned strings in V04, because production changes are comments only. Each ledger entry names D02 (public API disposition) as the owner of any change to returned text.
- **Q13 — In scope.** Brief: "benchmark docs are allowed". In `Docs/Performance_Guide.md`'s benchmarking section:
  - fix the invalid CLI examples;
  - remove the nonexistent `scripts/compare_benchmarks.py` reference;
  - apply claim policy to the "indicative" ns figures: remove them, or label them with a dated reproducible artifact.
- **Q14 — Comment-only correction in V04, plus an inventory.** In the ledger, inventory every caller of `releaseOnly`, `isReleaseMode`, `isSIMDEnabled` and `performancePrecondition`. Flag to root any correctness-relevant path that is dead or always-on because these flags are undefined, so reachability can be re-derived under the gate rule. The public API disposition belongs to D02.
- **Q15 — Yes.** Timing, allocation and inspection evidence at 512D. Widths 768 and 1536 are covered by selection and correctness contract tests.
- **Q16 — Yes.** Build-modes swiftbuild support and the user-approved `validate_github.rb` re-pin belong to T03. V04 must NOT edit `check_benchmark_build_modes.py`.
- **Q17 — Record only.** The 39 discarded-result timed statements go in the ledger and limitations. They are never used as performance evidence.
- **Q18 — Use the brief's unit.** The statistic is "per-sample average ns per operation", where operation is the declared work unit (candidate, produced entry, or invocation for scalar cases). Record `unitsPerInvocation` so the conversion is explicit. Raw per-invocation samples may be retained as well.
- **Q20 — Approved, as recorded deadlines.** Memory probe 12 min, allocation probe 10 min, timing records 10 min, 15 s per child plus at most 2 s cleanup. Each is recorded in its manifest.
- **Q21 — Yes.** `OutputFold.swift` is a ledger region.

## Phase execution
- **Order and checkpoints.** A → B → C → E → F → D, as proposed. Each phase ends at a checkpoint: write a phase report and stop for root review before the next phase. Phase D runs only after source freeze, and root reviews D3–D5 deadlines before those run.
- **Repeated class.** Concern 1 (handle-stride assertions in other tests) is handled under Q10. Concern 3 (dead release flags) is handled under Q14. Both stay ledger-tracked.
