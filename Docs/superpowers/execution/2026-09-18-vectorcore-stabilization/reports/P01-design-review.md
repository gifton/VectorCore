# P01 independent design review

Disposition: two bounded route/contract clarifications are required before dispatch. The hashing, selector, bounded Batch worker-partial design and measurement scope otherwise have no identified design blocker. This is source/design review only; no builds, measurements or implementation were performed.

## DR1 — Preserve the remaining Operations optimized score route explicitly

`Operations.findNearest` dispatches directly to optimized TopK for Euclidean512/768/1536 and cosine/dot512 (Operations.swift:95–123). Its fallback `computeDistances` also dispatches cosine768/1536 to BatchKernels fused range helpers (456–463). Those two routes are therefore neither the existing direct TopK helpers nor generic metric-distance computation. The proposed generic small-k reduction must not capture them and silently replace accepted fused scores with `metric.distance`.

Minimal correction: expressly leave these two routes on their existing full-score computation, recording the remaining N-score materialization, or explicitly authorize bounded same-kernel tiling with per-candidate parity controls for these two Operations routes. The former is sufficient for bounded P01 scope. Tests should identify the route, not just compare top-k answers on ordinary finite values.

## DR2 — Choose the generic Operations calibration/admission behavior

The current fallback determines its serial/parallel route in `computeDistances` (369–439). Its AutoTuning callback invokes the metric on up to256 initial candidates before the actual score computation (405–426), and its final generic branch uses either provider.parallelExecute or synchronous map (467–474). Consequently, unconditional provider.parallelReduce plus exactly one metric call per candidate is an observable policy change, not merely removal of score storage. The brief also says not to change the shared heuristic, leaving the intended local admission/calibration behavior unclear.

Minimal correction: explicitly select whether small-k generic Operations bypasses this calibration and delegates partition admission to provider.parallelReduce (documenting the changed callback/admission behavior), or retain the existing decision/calibration and qualify once-per-candidate as the selection pass, with calibration probes separately observable. Either choice can preserve the shared heuristic implementation; it must be selected before implementation and tested with fresh AutoTuning state/custom metric spies. Large-k behavior and optimized fallback routing should remain separately stated.

## Remaining design assessment

The four hash repairs preserve equality and Float signed-zero hashing while making coordinate coverage explicit; distribution tests do not demand collision freedom. Structural mutation coverage plus consumed Set/Dictionary results appropriately separates the old lawful collision class from semantic correctness and timing.

The Int selector avoids the pointer Int32 limit, preserves callback order and score bits, and gives VectorMath an explicit breaking count/NaN/tie contract. Independent comparator tables and smallerValue equivalence checks are appropriate. Small/large predicate boundary tests must use integer division exactly as the existing code does.

Batch score tiling can preserve the existing range arithmetic without kernel changes: BatchKernels outputs at index minus range.lowerBound. Its fused cosine loops process candidate pairs with scalar tails; the Euclidean square-root wrapper groups four output values before a scalar tail. A256-element tile preserves those group boundaries when restarted within each original logical range; do not tile across or coalesce an original logical boundary. The brief already requires complete accepted-score parity before top-k comparisons, so this is an implementation invariant to inspect rather than permission to relax tolerances. Root-score admission remains distinct from direct optimized squared-score admission.

Grouping logical chunks into at most min(C, chunkCount) successive worker partitions bounds retained heaps without materializing all chunks. Quotient/remainder partition arithmetic and global-index ties make an overflow-free implementation feasible. E06 logical-range observations and E02 real worker accounting must both remain truthful; the stated root approval of exact existing-test migrations is appropriate. The C-based temporary bound excludes custom provider fan-out, full-score large-k paths and GEMM Q*N scores as required.

Local capacities min(k, range.count), an empty-buffer push guard and merge compatibility checks form a coherent class repair. Merge-into can still detach through COW; alias tests and allocation measurement prevent a false zero-allocation claim. Provider override precedence, original k, all-row preflight, original errors and complete versus cooperative-cancellation distinctions are preserved explicitly. The private validated batch worker must retain a single preflight boundary and not become public API.

The Release evidence plan separates score/heap storage from inputs/results, allocation requests from process memory, source models from observed peaks, and hash cost from dictionary throughput. The remaining API-24 reshape is explicitly staged. Dependencies and exact E06/R04/S03 ownership refresh are necessary before implementation; no dormant ExecutionOperations route should be resurrected.

Read basis: full candidate P01 brief and prior P01 inventory; current Operations nearest, computeDistances and helper dispatch; BatchOperations range-output usage; BatchKernels fused cosine and Euclidean sqrt wrappers; TopKSelection strategy gates and TopKBuffer merge/range contracts. No downstream runtime behavior is inferred from sibling source.

## Scoped amended-brief adjudication

ACCEPT — design only. Re-read the amended Operations route paragraphs and original-logical-range tiling invariant. DR1 is resolved by explicitly retaining cosine768/1536 fused full-score computation and its calibration/admission, with the N-score cost qualified. DR2 is resolved by deliberately bypassing computeDistances calibration/admission only for the remaining generic small-k path, delegating to provider.parallelReduce, and requiring exactly N total metric calls/no new calibration publication plus cold-key/provider-spy controls. Large-k and optimized controls remain unchanged. The256 tile now explicitly restarts inside each original logical range and cannot cross a boundary. No remaining design contradiction was identified; implementation/runtime acceptance is separate.
