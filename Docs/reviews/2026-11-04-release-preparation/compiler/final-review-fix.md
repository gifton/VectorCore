# Final-review existential witness coverage

Review finding I1 identified a coverage gap: the initial public fixture exercised
concrete and generic values, while the D02 existential conformer supplied its own
subscript. Neither established imported inherited-default existential access.

`CollectionWitnessConsumer.swift` now forms both `any Collection<Float>` and
`any VectorProtocol` from the same `ConsumerVector`. Non-inlined generic helpers
implicitly open those values. The Collection helper reads its own `startIndex`
and checks `1.25`; the VectorProtocol helper converts the indexed scalar to
Double inside the generic context and checks `-2.5` at index 1. Both inherited
and `EXPLICIT_WITNESS` variants execute these checks. No production code changed. Retained SILGen also shows `open_existential_addr`
for each existential into its `[noinline]` helper.

All eight qualified compile/link/runtime combinations pass: Debug and Release
modules, each with `-Onone` and `-O` clients, each with inherited and explicit
witnesses. The preserved task-1 library modules and objects were hash-checked
against the prior final matrix before reuse. Consumer, fixture-manifest and
subprocess-bound Python checks pass 31 tests. The existing consumer-smoke rows
execute the enhanced fixture; the coordinator reruns that route after integration.

New records are under
`/private/tmp/vectorcore-release-prep-2026-11-04/task-1-evidence/final-fix/`.
`existential-outcome-matrix.json` records the enhanced fixture's hashes and
outcomes. The original `outcome-matrix.json` and archived fixture snapshots retain
their original scope; they are not retroactively credited with existential checks.
The compiler/platform qualification limits in the original report still apply.
