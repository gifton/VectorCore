# E06 independent design review

**Verdict: favorable, pending one explicit scope correction.** No builds, implementation edits or unfinalized E02 source review. Compared the candidate with the accepted settings inventory and approved E02/N03/R03/E03 contracts.

## Concrete blocker

“Every public batch operation obtains exactly one live immutable configuration snapshot” also includes public synchronous `BatchOperations.sample(_:k:)` (BatchOperations.swift:406). That method cannot await the configuration actor and uses none of the settings; the scope table omits it and no async migration is authorized. Explicitly exempt sample and retain its synchronous configuration-independent behavior, or explicitly choose the larger signature migration. The exemption is consistent with the candidate's intended setting scopes and requires no production change to sampling.

## Checked design obligations

The floating target conversion is safe for valid positive finite oversubscription and positive C/N, including Int-limit tests and changed processor counts. C is representable as a finite Double. Multiplication yields a positive value, possible underflow to zero, or positive infinity; it cannot yield NaN from these inputs. Positive infinity takes the target>=Double(N) clamp. Otherwise target<Double(N)<=Double(Int.max), whose rounded value is 2^63 on the supported 64-bit platforms; every nonnegative Double strictly below that bound truncates to a representable Int. max(1,...) handles tiny factors. Rounding Double(N) does not break conversion safety, even if it rounds upward. targetChunks is between1 and N. Ceiling division using quotient/remainder cannot overflow: when remainder is nonzero the quotient is strictly below N. Clamped chunkSize lies in1...N, and start+min(chunkSize,N-start) is bounded by N. Removing signed power-of-two rounding and additive ceiling division addresses the identified hazards.

Update-time validation plus candidate publication preserves nonthrowing operation validity. async throws correctly permits validation errors independent of user closure errors, while running the closure first preserves original-error precedence and atomic rollback. Generic ThreadSafeConfiguration remains unconstrained. Dynamic core counts do not invalidate conversion safety because operation-time clamping provides its own proof rather than relying solely on publication-time processorCount validation.

One captured snapshot/core count per configured async operation is consistent with later updates and nested invocations taking independent snapshots. Nested worker suppression remains a separate E02 property. Oversubscription changes logical chunks, not the accepted hardware worker cap; N03 still merges original logical chunks in index order. Explicit process batch validation precedes cancellation, and nonthrowing complete-result policies are preserved.

Nearest's threshold substitution deliberately reaches all six optimized helpers previously unreachable through the public heuristic. The required public width/metric route evidence, custom witnesses, compact-output offsets, NaN/tie/global-index cases, cancellation, nesting and task bounds correctly treat this as gate activation. No SoA or FP16 activation is inferred. Pairwise retains its independent N100 gate and existing GEMM compatibility restrictions/diagonal semantics.

Migrations explicitly cover E03 valid tuples and throwing consumers, E01 invalid-default setup, and global GEMM tests. Keeping generic actor NaN rollback unchanged is correct. Fresh child isolation is necessary beyond suite serialization. The scoped scheduler observation extension and pure partition helper provide executable arithmetic/route proof without giant allocations or timing assertions. Tests must distinguish logical partitions from workers and permit hardware-limited parallelism as specified.

No additional mathematical, ownership or scheduling contradiction found. Acceptance remains design-only and E02 integration is still a prerequisite; implementation must repeat reachability against that actual base.

## Revised scoped verdict — ACCEPT

Verified the revised snapshot clause explicitly covers the public async compute operations in the scope table and exempts synchronous configuration-independent sample(_:k:). The sole scope blocker is resolved. **ACCEPT design-only**, with accepted E02 integration still required before dispatch. Brief SHA256: `503da2bdb4523637e1dbf9b3509a9b0e106a5b92f2e623ad1f47c2f0bb9c584b`. No further concrete blocker found; no builds or source changes performed.

## E02 coverage and fused-cosine addendum — scoped review

**ACCEPT the addendum, design-only.** Replacing the public nearest heuristic necessarily invalidates E02's Dim64 N34/35/36 routing expectation; migrating that test to a fresh isolated configuration child with restored state is correctly scoped. Its independent finite oracle, NaN/tie membership, cancelled complete-result behavior and actual worker observations must survive. These generic controls do not substitute for the six newly reachable optimized public routes.

The explicit fused-cosine qualification distinguishes independent finite correctness from nonfinite parity with the existing direct kernel. Preserving the established `[NaN, 1, 0, ...]` score0 fixture is an intentional route-specific policy, not a claim of scalar-metric parity or universal NaN propagation. This narrow known behavior does not license ignoring other newly exposed arithmetic defects. Actual optimized-route activation and exceptional-value evidence remain implementation obligations at the accepted E02 base; no new runtime conclusion follows from this brief review.
