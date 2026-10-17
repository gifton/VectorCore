# V04 A+B fix round 1 — root rulings (binding)

Review: `phaseB-review.md` (root-recorded) in this directory.

1. **I1 — Allocation flags must not overclaim.**
   - Add an explicit `unverified` value to the inclusion-state enum. This is schema 3, which is new in this task and unreleased.
   - FP16 `routeInternalAllocation` becomes `.included`, with a note naming the per-call handle array (MixedPrecisionKernels range_*_mixed_D).
   - SoA `routeInternalAllocation` and `normalize.zeroFail` allocation become `.unverified`, with a note.
   - Audit every other `notApplicable`/`excluded` allocation flag against its source route. Any flag not established from adjacent source is marked `unverified`.
   - Phase D's allocation probe must later resolve every `unverified` flag for the routes it measures, or keep it unverified with the limitation stated.
2. **I2 — Close the drift class mechanically.**
   - Implement guard (1): per-family body fingerprints. Each Boundaries family declares the helper symbols that implement its warmup and timed bodies, plus the SHA-256 of their comment- and whitespace-stripped source. A contract test recomputes the fingerprints from `Benchmarks/VectorCoreBench/Suites/*.swift` and fails on mismatch, with a negative control on a mutated copy.
   - Implement guard (2): a closed-vocabulary consumer-token check for each measure closure, matched against its declared `consumer`.
   - Fix the overclaiming comment at Boundaries.swift:9-11 so it states exactly what is mechanically checked.
3. **I3 — Statistic per ruling Q18.**
   - Add per-sample nanoseconds per declared work unit, with median and variation, and make that the named statistic: "per-sample-average-ns-per-operation", where operation is the declared work unit.
   - Keep the V02 per-invocation fields, `unitsPerInvocation` and the raw samples.
   - Update the JSON, pretty and CSV output and the pinning tests.
4. **Minors, all included:**
   - M1: align the warmup body with the timed body.
   - M2: extend the normalize fold test to generic and 768/1536, and replace the vacuous assertion with one that shows the old single-coordinate observation is blind to other coordinates.
   - M3: make the flag conventions consistent: caller-owned outputs allocated outside timing are `excluded`; absent validation is `notApplicable` with a note.
   - M4: "legacy" means a missing schema or a version below 3 only. A future or unknown schema is an integrity error and is not waivable.
   - M5: validate nested `fixture` and `inclusions` keys and value types.
   - M6: require exactly one result per production case, which restores duplicate and empty rejection.
   - M7: add a CHANGELOG Unreleased entry for the schema-3 Codable change and the `JSONWriter.toCases` throwing change.
   - M9: balance observer begin/end calls, and document the observer's tolerance.
   - M10: derive left/right preparation from the boundaries.
   - M11: trap or throw on encode failure instead of returning {} or [:].
   - M12: pin the exact 21-column V02 CSV prefix, fix the sections count, and add the boundary to CSV failures.
   - Runtime check: reject the `fixture-injected` scope when `injectedRegistry == nil`.
5. **M13 — Reduce work before raising the deadline.** First reduce the `--ab-only` quick children's work: use smaller N or fewer cases while keeping 36/18 coverage. Only if that is impossible, raise those CLI children to a recorded 30 s deadline with justification. The standard 15 s plus at most 2 s applies to test children; these are full CLI runs.
6. **M14:** annotate or remove the stray phaseA `.exit` artifacts in evidence.
7. **M8:** deferred to phase C. Track it in the report.
8. **Lint:** fix the +77 new warnings on V04-touched lines. Pre-existing warnings are untouched.

Verification is the same as phase B's acceptance checks: Debug and Release BenchmarkContractTests, pairs 36/18 in-process and in a real CLI run, V01 guards 28/14/28, the Python suites, discovery, the new guard negative controls, and lint. Append a "Fix round 1" section to `phaseAB-report.md`, or return it as text.
