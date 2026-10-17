# R04 implementation — frozen v1

The bounded heap membership and legacy nearest ordering repair is complete on base `e0e0e74b5128419482d1e1993d58c2184adc01ba`. Production changes are confined to MinHeap.swift. Full patch `R04-v1-diff.patch` has SHA256 `a5dffcd7c82cad065528e1f32e66196cd8cf518cc2336a36fa5f5fa3f15e0049`. Its manifest contains 320 base and 321 final code files, with exactly two changed/new paths: MinHeap.swift and HeapMembershipContractTests.swift. The complete base archive matched all 5,891 regular files before edits; the final source archive has 5,892, excluding generated child captures and Python caches that are retained separately where relevant.

## Repair and migration

Both heap initializers now reject negative capacity before heap work. Nil remains unbounded; zero retains nothing. Empty construction no longer reserves the requested capacity, so a tiny heap with Int.max retention does not request an Int.max allocation. Bulk construction returns immediately for zero capacity, streams insertion when truncation is needed, and retains bottom-up heapify for full-retention inputs with more than one element. It no longer mistakes arbitrary leaf removal for selection. The existing streaming replacement direction is unchanged: retain the last elements under the supplied strict weak ordering. Thus `<` retains largest and `>` retains smallest values. Equivalent newcomers at capacity reject; equivalent-key identity and drain order are not newly stabilized.

KNearestHeap now reverses the existing shared TopKSelection ascending-distance comparator to put the worst retained member at the root. Its reversed drain returns numeric-before-NaN, lower distance first, then smaller index for ties, including signed zeros and NaN/NaN. Duplicate identical entries remain allowed. This is an explicit legacy behavior change. Negative k rejects, zero stays empty, and Int.max with tiny input is supported without eager reservation. No fields or public signatures change.

The production caller inventory finds no current KNearestHeap caller beyond its declaration and implementation; TopKSelection has only a historical comment referring to it. Accepted E02 removed ExecutionOperations.findNearestSequential, and it remains absent. Modern Operations/TopKSelection routes use TopKBuffer and were not edited. No route was resurrected, no score policy changed outside this legacy helper, and no sibling or Package source was touched.

## Actual RED and retained setup failures

The first executed baseline attempt failed test compilation because Swift could not disambiguate an operator-function ternary and Swift Testing's macro could not wrap a mutating insert call directly. The source, full log and partial build artifacts remain retained under red/build-red-debug; this is not behavioral evidence. Two shell-wrapped approval requests were interrupted before execution. Their source/setup state is retained, but no command exit or runtime result is invented. Direct `swift test` with the existing approved prefix then launched normally, and every returned output chunk was persisted without truncation.

Actual unchanged-production RED is red3, exit 1, four test functions and 13 issues:

- Both comparator orientations produce bulk/stream membership mismatches: four assertions fail.
- k1 NaN then numeric fails insertion and membership: two assertions fail.
- Negative-capacity empty and nearest initializers return normally with the accepted marker: three assertions fail per child.
- Negative-capacity empty bulk construction hangs and reaches the 15-second child deadline: one timeout error. Cleanup is bounded to two seconds; the parent throws before any status/reason query.

The baseline hang is a reproduced defect, not a final accepted outcome. No baseline Int.max eager-reserve case ran, and no known hanging operation ran in the parent.

## Final verification

| Check | Result |
| --- | --- |
| Frozen v1 Debug | Exit 0; 72 Swift Testing functions in four suites |
| Frozen v1 Release | Exit 0; same 72 functions in four suites |
| Negative-domain children | Three actual children per configuration; all signal 5/reason 2, no timeout |
| Child-process scanner | Seven checks passed via `python3 -m unittest Scripts.ci.tests.test_child_process_bounds -v` |

Each 72-function parent total includes one no-op childProbe entry. negativeCapacity is one parameterized parent function that actually executes three children. Debug requires the named capacity diagnostic plus started marker, signal rejection, and no accepted marker. Release uses the actual signal and markers together with the frozen precondition guards; standard optimized diagnostics are not assumed to survive. No ASan was required because the repair introduces no pointer scope.

The exact filter is `HeapMembershipContractTests|TopKNaN|IntegrationProtocolsSuite`. It includes the new suite, existing TopK NaN ordering and CPU wrapper controls, and integration protocol controls. Logs and the command record retain actual matched test names. A scanner invocation first returned all seven passes through the tool; a second identical invocation retained the full canonical file log. No source changes occurred during either build, and no final source revision followed verification.

## Independent contract evidence

`R04-inventory.md` supplies the maintained route and case inventory. The numeric suite uses ordinary sorting followed by an independently bounded suffix as its membership oracle. It enumerates both orientations and every positional permutation of empty, singleton, distinct and duplicate-value inputs across nil/0/1/n-1/n/n+1/Int.max capacities. Retained multisets and drained order are compared, not heap layout. Copies can be drained independently. Int.max tests have at most four elements.

Record tests compare keys when comparator-equivalent identities are unspecified, and separately use explicit secondary labels for deterministic identity. Dedicated controls check underfull/replaced/rejected/equivalent insert Bool, zero/unbounded counts and ordinary reference lifetime through copy/mutation.

The nearest oracle independently classifies NaNs and compares numeric values and then indices; it does not call the production comparator. Permutations cover finite values, both infinities, signed zeros, duplicate distances, shuffled indices, NaNs in every insertion position, all-NaN fillers and duplicate identical entries. k includes zero, one, n-1, n, above n and Int.max. NaNs compare by classification; signed-zero ties resolve by index and retained non-NaN bits are checked. Modern public TopKSelection fixtures remain separate controls. No generic Float NaN policy is asserted for an invalid caller comparator.

## Canonical evidence and limitations

Canonical task root: `/Users/goftin/dev/gsuite/VSK/VectorCore/.stabilization-recovery/2026-09-29/R04`.

Actual versioned build trees remain at build-red-debug, build-red3-debug, build-v1-debug and build-v1-release. No tested tree was rebuilt against later source. Baseline source is reconstructible from immutable base.tar plus each retained initial test source; frozen final source is under v1/source. Source, artifact, log and command manifests associate every actual run with its version. Child associations retain exact argv, distinguish the SwiftPM helper executable from the loaded test bundle, and copy the helper binary and each unique capture. Original captures are already canonical within the isolated checkout's .build directory; copies are grouped by source/build version under the task root.

The parent installs its termination handler before launch and captures output through files. Every negative child has a 15-second execution deadline and at most two seconds of cleanup. Any timeout throws before termination status/reason is accessed. Source snapshots, raw complete output chunks, combined logs, setup failures, baseline hang and successful final rejections remain separate. Artifact inventories exclude symlink aliases while retaining their actual targets. The final worker-only index excludes root-owned, independent-review and pre-task design records.

No performance speedup, allocation benchmark, downstream execution, universal comparator correctness, arbitrary identity stability or production-scale readiness claim is made. Capacity is a retention bound, not a memory reservation promise. Root retains independent acceptance and API-17/API-26 closure decisions. No commits, index operations, shared production edits or nested agents were used.
