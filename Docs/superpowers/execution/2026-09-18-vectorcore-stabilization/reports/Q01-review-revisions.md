# Q01 review revisions

## v1 to v2

The first full Release run (`Q01-focused-release-1.log`) passed finite correctness assertions but all99 rejection checks failed to find their intended diagnostic text. Optimized Swift `precondition` removed those strings. Sixteen Q01 guards were changed to explicit `guard`/`fatalError`, retaining the same predicates. This Release failure is class-level behavioral RED for the named rejection requirement. Debug rerun2 passed17 XCTest methods and50 Swift Testing functions (exit0).

The negative centered-extreme dot fixture's handwritten decimal was also corrected to independent precomputed Float bits from exact integer `-(2^31+127)^2 * dimension`; the exact square is4611686563888250625. The original approximate decimal had passed the permitted Float tolerance but was inappropriate provenance. Arbitrary-integer wide accumulator fixtures use separate exact input terms and were unchanged.

Full frozen v2 patch SHA256:24998c1fd21f7770f32afe1f7674d2d5d336ac13c69769f5befa2a8d8552cfe6. v1 retained.

## v2 to v3

Root review identified unbounded `Process.waitUntilExit()` after the nominal child deadline. Replaced polling/sleep/waitUntilExit with a termination handler installed before launch and a semaphore deadline. Execution is capped at15seconds; only the owned child PID is killed if necessary, followed by at most2seconds cleanup. File-backed output is read only after termination; no pipe read blocks cleanup. Only `QuantizationRangeContractTests.swift` changed; production hashes are identical to v2.

Full frozen v3 patch SHA256:6736233f3ed9f373fce8d9626467df457d8dd9b10a6b13e8b2302eb7dfceb228. v2 retained. Affected Debug/Release child checks are pending at this snapshot; Release2 was already building when this test-only revision occurred, so a later explicit rerun will establish frozen-v3 evidence.

Root is independently measuring requested allocation bytes/events in the mixed and unequal-scale fallback routes before ruling on whether the four-term expansion's dynamic temporary arrays require a further correction. No allocation, latency, throughput or readiness claim is made here.

## v3 to v4

Root's actual linked Release allocation probe found the dynamic four-term expansion caused2048 allocations/81920 requested bytes for scalar mixed512,6144/245760 for mixed1536 and30720/1228800 for five-candidate SoA1536. Equal-scale integer control had0allocations. Ordinary calibration of8×1536 regressed from0 to28 allocations/156288 requested bytes through flatMap. These measurements are the RED evidence for the memory-work correction; they are not latency/throughput evidence.

Root authorized two production-only repairs: expansion uses two fixed SIMD4<Double> value buffers plus counts, preserving exact term/two-sum/reduction order; the adjacent bound `nextCount <= count <= termIndex <= 3` justifies capacity. Ordinary calibration now validates finite values and streams extrema directly over SIMD storage. Distribution sorting remains materialized as required. Unsupported-mode and empty behavior are preserved. No test oracle or expected outcome changed for this correction.

Frozen v4 full patch SHA256:03b1df34f94d20807683da228af2d3f382ee6bd389382d0e63b136bdc03a1117. All prior versions remain. Focused Debugv4 passed17 XCTest methods and50 Swift Testing functions (exit0). Release/ASan and root's repeat allocation probe are pending here.

During an unrelated shared-temporary-directory integrity issue, this worker checked all778 base tracked archive paths: none missing; all7 modified/new file hashes still matched v4. No local restoration or regeneration from missing files was required (`Q01-archive-integrity.json`).
