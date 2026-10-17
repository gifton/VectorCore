# E06 accepted-source settings inventory

Read-only inventory against accepted 433efec. Shared HEAD advanced to 2fb57c8 during inspection; `git diff 433efec -- Sources/VectorCore/Operations/BatchOperations.swift Sources/VectorCore/Configuration/ThreadSafeConfiguration.swift` was empty. No E02 worker source inspected. Read WORK_ITEMS E06, canonical API-02 and approved briefs/E02.md. No build/runtime evidence is asserted here.

## Stored configuration and actual reads

All BatchOperations line references below are in Sources/VectorCore/Operations/BatchOperations.swift.

| Field / declaration | Current effective route |
| --- | --- |
| parallelThreshold=1000, line 57 | Mutable live value is ignored. Immutable defaultConfig controls process:186, map:305, filter:335, statistics:370. |
| oversubscription=2, line 60 | Mutable value ignored. Immutable defaultConfig used by optimalChunkSize:614–615. |
| minimumChunkSize=256, line 63 | Mutable value ignored. Immutable defaultConfig used by optimalChunkSize:616 and 625. |
| defaultBatchSize=1024, line 66 | Mutable value ignored. process nil argument uses immutable defaultConfig:180. Explicit batchSize is independently positive-validated, even for empty input. |
| enableMatrixRouting=true, line 74 | Live snapshot read in pairwiseDistances:240–242 only. |
| matrixRoutingMinN=256, line 79 | Same live pairwise snapshot; no domain validation. |

The actor is private at 87; immutable defaults are separate at 90. configuration() returns a value snapshot (93–95). updateConfiguration forwards once to actor.modify (102–105), preserving E03's synchronous candidate/body/publication transaction and rethrows behavior. Sources/VectorCore/Configuration/ThreadSafeConfiguration.swift:31–35 copies a candidate, calls the throwing body, then publishes; it applies no BatchOperations-specific validation. Public Configuration fields/init likewise validate nothing. Generic actor users must not accidentally inherit Batch-only validation policy.

Search: `rg -n 'parallelThreshold|oversubscription|minimumChunkSize|defaultBatchSize|enableMatrixRouting|matrixRoutingMinN|optimalChunkSize' Sources Tests Benchmarks Scripts --glob '*.swift'`. Matching ExecutionOperations/TransformOperations parallelThreshold and CPUContext chunk helpers are distinct static/context settings, not reads of BatchOperations.Configuration. MatrixDistanceBench mentions calibrating the crossover, but does not make the first four live.

## Existing thresholds and newly reachable work

- process:176–219 changes callback grouping with defaultBatchSize and execution route at 1000 input vectors. Batch endpoints already use `start + min(size, count-start)`. Wiring a live default changes observable callback batch shapes, while explicit argument precedence must remain a declared decision.
- map/filter:301–356 choose serial array traversal below 1000; otherwise currently add one task per input and restore index order. Lowering a live threshold exposes concurrent callbacks at smaller N, including singleton/empty boundary questions. E02 owns replacement scheduling and cancellation behavior.
- statistics:365–402 is serial below 1000 and otherwise uses explicit logical chunks, one magnitude evaluation per vector and original chunk-index merge order. Live minimum/oversubscription changes logical partitioning and can change floating-point rounding without changing the population statistic contract. Preserve N03 Double centered updates, invalid/count policy and internal explicit-partition helper. Never replace logical-chunk merge order with worker partial reduction order.
- pairwise:235–252 has its own hard-coded N100 serial/parallel switch, independent of parallelThreshold. GEMM is attempted first using the live two-field snapshot. `gemmPairwise`:258–271 admits only concrete Euclidean/Cosine and optimized512/768/1536. Custom metrics remain supplied-witness paths; R03's symmetric restriction and metric-derived diagonals must remain. Lowering matrixRoutingMinN exposes packing/BLAS paths at small nonempty N; N0 returns before snapshot. Matrix output remains quadratic. Disabling this field does not disable the separate Operations.findNearestBatch routing gate.
- nearest:138–160 uses ParallelHeuristic rather than parallelThreshold. Sources/VectorCore/Utilities/ParallelHeuristic.swift:10–38 models per-item work versus `1.15*(60000+1300*N)`. **For optimized512/768/1536 Euclidean and cosine, parallel is false for every positive N**: the largest per-item cost is 1536*0.30=460.8, below the overhead slope1495, before its positive fixed term. Therefore the six optimized parallel helper bodies (758–929) currently have no public nearest caller reaching them through this heuristic. Connecting a general threshold to nearest would activate these dormant routes; it is not a routine setting substitution. Generic/custom metrics use the generic heuristic and can reach parallel. Unknown metric classes affect scheduling estimates only; their actual distance must still dispatch through the supplied witness.
- Serial optimized nearest uses hard-coded range steps256 for512/768 and512 for1536 (`minChunk`:631–633 and six serial helpers:636–754). These do not consult minimumChunkSize. Parallel generic nearest (512–528) and six optimized parallel helpers call optimalChunkSize. The optimized helpers invoke AoS `BatchKernels.range_euclid_*` / `range_cosine_fused_*`, **not SoA kernels**. A scheduling change does not justify switching storage/layout or activating SoA/FP16 caches. If E06 introduces such a route, it requires an explicit scope decision and new logical-output/ownership review.

## Arithmetic hazards to resolve before making knobs live

`optimalChunkSize`:612–625 converts `Double(coreCount)*oversubscription` directly to Int, divides by that target, rounds using signed shifts, and subtracts bounds. Nonfinite/huge values can make conversion invalid; zero/tiny positive factors can truncate the denominator to zero; nonpositive minima can reach invalid shift/rounding conditions; large values can exceed signed power-of-two representability. Validating merely oversubscription>0 is insufficient. Do not route unchecked settings into this legacy formula.

Existing helper sites use `(n+chunkSize-1)/chunkSize` reserve calculations (778 and width siblings) and `chunked(by:)` uses `$0+chunkSize` (962–965). An otherwise positive Int.max chunk can overflow intermediate addition even for small input. Chunk count/endpoints must be derived safely; huge settings should be validated/clamped under the chosen policy before arithmetic. These are source hazards on newly activated inputs, not runtime reproductions here.

## E02 coordination and bounded design questions

Approved E02 keeps ignored settings ignored, replaces add-all loops with shared at-most-K coarse workers, and fixes BatchOperations root K to max(1, activeProcessorCount). Nested library regions create no new workers; throwing paths cooperate with cancellation and nonthrowing nearest/pairwise/statistics complete all results. E06 must build on the accepted E02 implementation and repeat this inventory at dispatch. No concurrency/speed claim follows from this accepted-source review.

Root decisions needed:

1. Which settings remain public and precisely which methods do they control? In particular, does parallelThreshold retain only process/map/filter/statistics scope, or supersede nearest's heuristic/pairwise's N100 gate? Do minimumChunkSize and oversubscription govern logical work partitions or worker K? Raising K with oversubscription would change the approved E02 CPU-count cap and needs an explicit new contract; keeping it a partition hint can preserve that bound.
2. Validate at atomic update publication or operation entry? Update validation requires changing today's async rethrows API because validation can throw independently of the user's body. Define invalid-setting errors, successful candidate publication, original user-error identity, and rollback of all fields. Public nonthrowing operations need a coherent invalid-snapshot policy if invalid states remain storable. Define zero threshold semantics, finite upper/lower oversubscription limits and arithmetic-safe treatment of large positive chunk/batch sizes, rather than inferring workload targets.
3. Take one immutable validated snapshot per public operation and pass needed values down; helper re-reads could mix configurations. Define empty-operation validation/snapshot behavior and explicit process batchSize precedence. Configuration updates during callbacks should affect only later operations, testable with finite gates.
4. Preserve metric witnesses/NaN-TopK policies, N03 partition semantics, R03 diagonals/zero rows, and E02 task-local/cancellation/worker bounds on every newly reachable route. Keep independent Operations gates, SoA thresholds/layout and CPU provider settings outside implicit scope.

## Test migrations and acceptance candidates

Global setting tests must use bounded fresh children: .serialized does not isolate other suites. Existing BatchGEMMRoutingTests:105/107/117 mutates the same global enable flag. MatrixMetricContractTests already isolates its45 configuration scenarios and checks six-field restoration; retain this control.

Update-time validation would require scoped migration of ConfigurationTransactionTests:11/27 (zero starting threshold/chunk), 88 (negative-zero oversubscription) and generation-based tuples. Its local generic-actor NaN rollback test at114 should remain legal: it tests generic value binding, not Batch config validity. Scripts/ci/tests/E03ConfigurationConsumer.swift must migrate any newly throwing update signature while preserving positive/negative Sendable evidence. ExecutionBoundaryTests:218 deliberately stores defaultBatchSize0 before testing explicit process arguments; this expectation needs an approved migration, not silent broadening of old tests.

Behavioral spies should prove callback batch shapes for nil versus explicit sizes, threshold-side scheduler observations, actual partitions without equating partitions to workers, custom metric invocation and exact-once magnitude evaluation. Rerun small/empty, threshold neighbors, invalid and Int-limit settings, full configuration transaction rollback, cancelled/nested routes, R03 matrix gates, N03 multiple partitions, all six optimized nearest helpers if newly activated, and preserved kernel compact-output/tail offsets. Use finite-aware oracles and original global indices; no timing-dependent concurrency proof. Removed settings require explicit public migration documentation rather than a fake no-effect test.
