# P02 independent candidate design review

**Design disposition: one cancellation contract decision required before dispatch; otherwise acceptable within the stated prerequisites and staged limits.** This is a source/design review, not implementation or runtime acceptance. Reviewed candidate SHA256 `859f1f37459b5960dec8898726e5c3455727f44a9bb3fd05e5568ed7c1a182b0`, prior P02 inventories/policy review, approved P01, and actual Operations batch dispatch, MatrixDistance packing/norm/normalization/GEMM, BatchOperations pairwise helper and TopKBuffer implementation.

## Required decision

**DR1 — Specify explicit Gram cancellation.** The new serial tiled Gram path runs inside async-throwing Operations.findNearestBatch but the brief does not say whether it cooperatively cancels or completes like the old synchronous implicit helper. Per-query fallback retains E02 throwing-route behavior; pairwise is explicitly complete-result. Do not leave this observable distinction to implementation accident.

A narrow consistent choice is: preserve public validation and authoritative provider dispatch first; after successful CPU plan validation, check cancellation before scratch/output allocation, between query/candidate tiles and before final return. A running BLAS call is not interrupted. Cancellation throws CancellationError and never publishes a partial result. Require a pre-cancelled no-work control plus cancellation triggered by the actual tile observation seam, with no subsequent tile execution. Root retains this decision; no new scheduler or deadline is needed.

Also pin unsupported metric/type versus invalid Gram-only options when both are invalid. The current candidate separately specifies both errors but not their precedence. This can be a one-sentence ordering clarification after whole-batch provider precedence.

## Accepted mathematical and dispatch boundaries

Per-query default deliberately removes the old Q8/N256 implicit numerical-method switch. P01's validated worker is the correct reuse point: public preflight remains once per whole batch and ordinary single-query reference retains each specialized admission/formatting route. Provider output/error remains authoritative before CPU policy validation, including invalid CPU-only options. This repairs no nonexistent provider-precedence defect; the present provider-first branch already has that order. Eligible-size sentinels are needed to establish it remains true.

Explicit Gram need not match direct distance or full-matrix bits. Whole-D tiling avoids a new split-reduction algorithm but changes BLAS geometry; neither one-by-one tiles nor postselection reranking establishes direct-neighbor equivalence. Independent finite mathematical controls and exact sorting of actually observed emitted tile scores correctly separate numerical accuracy from selection semantics. NaN-last/index laws, squared-score admission and later root formatting retain their distinct contracts. The candidate correctly preserves N01 owned-row normalization and zero/nonfinite behavior instead of silently using scalar metric witnesses.

P01 is a real prerequisite: its validated worker, streaming generic route and heap merge ownership must be reviewed from accepted source before implementation. V02/V04 measurement tooling is similarly a prerequisite, not assumed implemented. Custom provider callback order and runtime topology are not generalized from Core witnesses.

## Payload and ownership assessment

For one active query tile, the formula accounts for both owned row packs, optional Euclidean norms, active-capacity score tile and all r heaps. Repacking candidates for each query tile trades repeated work for bounded named scratch and is documented. Excluding final Q*K records is explicit; when k>=N, result storage itself is Q*N and cannot be described as eliminated. Checked clamping before arithmetic permits Int.max requested geometry/k without requiring giant allocation in planner tests. Actual element strides, outer row bytes, all products/sums/offsets and active LP64 leading dimensions must be checked, including the planned output stride used on tails.

The proposed internal drain primitive is feasible with existing worst-at-root TopKBuffer comparison. Partially filled buffers need one initial heap build because insertion only heapifies at full capacity. Each removal then restores only the root path. Reverse filling the final row avoids temporary pairs, result arrays and sorting work. Full/partial/empty/refill, both directions, tie-policy equivalence, NaNs/signed zeros and value-copy independence belong in its class-level oracle matrix. Ordinary Swift COW detach/allocator capacity is outside the logical-slot proxy and must not be presented as zero allocation.

No full-array tile slices or escaped borrows are required. The immutable caller arrays remain owned across the synchronous tile scopes. The cap does not constrain installed providers, backend workspace, allocator overhead, RSS or successful allocation. The brief states these limitations plainly and makes observed scratch-slot accounting distinct from Release allocation/peak observations.

## Pairwise and scope assessment

The existing pairwise helper materializes full N*N scores then copies every row. Preparing candidates once and writing each final row through the prepared/into path removes that extra full matrix without needing a new public container. It also changes one full GEMM to N one-query calls; the candidate correctly requires explicit numerical and measured work-cost migration rather than claiming universal parity or speedup. Preserve the outer metric diagonal rewrite, Euclidean off-diagonal NaN-to-zero formatting, cosine comparison behavior, E06 snapshot/gates/phase and nonthrowing complete-result behavior.

The new Gram payload policy does not apply to this separate nonthrowing pairwise route or existing public MatrixDistance unchecked preconditions. No new giant-shape safety promise follows for those APIs. Generic row scheduling, provider protocols, normalization arithmetic and public matrix ownership remain outside scope. API25 contiguous centroid ownership and TOOL10 downstream targets remain staged; no borrow API or production corpus is invented.

## Acceptance feasibility

The candidate's independent fixtures, exact route parity, tile observations, provider sentinels, pure integer-boundary planner tests, heap class tests and bounded Release comparisons are executable without production targets. Baseline numerical mismatch remains qualified if not reproduced on the host; a new API compile failure is not numerical RED. Allocation absence must be tied to actual linked public calls and complete output consumption, with matched flags and retained artifact bytes. No runtime finding, performance improvement or final acceptance is claimed by this review. No builds, source edits, index operations or delegation were performed.

## Root resolution pending on-disk confirmation

Root selected: original public shape/provider precedence; CPU Gram field domains first, then supported metric/type, then complete pure plan/representability/cap; cancellation before allocation, at query/candidate tile boundaries, after actual tile compute, before each finalized row allocation/drain, and before return. Individual synchronous BLAS/packing/drain/allocation calls are not interrupted and no latency bound is promised. Cancellation throws without partial output; perQuery and pairwise retain their separate existing laws. Pre-cancelled no-work, actual tile-observer cancellation and double-invalid precedence fixtures are required. This resolves DR1 and the ordering clarification **conditional on incorporation into the binding brief**. No additional blocker remains.

## Final scoped confirmation

**ACCEPT — design only.** Re-read the updated checked-plan precedence paragraph and complete Throwing search cancellation section in brief SHA256 `f77196427666cd782a3fa69a787d0eed36a30ea41d9b871412b72b52c168787e`. They incorporate the root resolution above, including final-check race qualification, synchronous-operation limits and actual-observer/no-work/invalid-precedence controls. DR1 and its ordering clarification are resolved. Prerequisite acceptance, implementation verification and staged API25/TOOL10 limits remain as stated; no implementation or runtime approval is implied.

Root incorporation: the independently reviewed design is unchanged except the header now records approval. Approved brief SHA256 `82629d23630328e7f56615d0fa0ac95d43e1c23bb35541f23cdd7e79783736a5`; prerequisite and staged-consumer gates remain operative.
