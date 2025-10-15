# NSW: Select Neighbors Heuristic (Prune to M)

Status: Draft spec for kernel generation

Owner: VectorIndex team

Last updated: 2025‑10‑07

Purpose
- Given a node q and a candidate set of neighbor nodes C with distances d(q, c), select at most M neighbors that are both close to q and diverse (avoid redundant neighbors).
- This is the “heuristic neighbor selection” step used in NSW/HNSW literature. Diversity is enforced by rejecting candidates that are too close to already‑selected neighbors.

Non‑goals
- Building the candidate set (handled by upper‑level insertion/search code)
- Computing distances to q (assumed provided)
- Device‑specific acceleration (CPU‑only kernel; accelerated variants can wrap this spec)

---

High‑level behavior
1) Sort C by ascending d(q, c). Resolve ties deterministically (see Tie‑breaking).
2) Iterate candidates in that order. For a candidate c:
   - If for every already‑selected neighbor r ∈ R, we have d(c, r) ≥ d(q, c) (or a relaxed rule with margin α), then accept c, i.e., R ← R ∪ {c}.
   - Otherwise, skip c (redundant w.r.t. a closer neighbor).
3) Stop when |R| = M or C is exhausted.
4) Optional backfill: If R < M and computing d(c, r) is not feasible for some candidates, backfill by taking the best remaining by d(q, c) until R has M elements.
5) Output R in ascending order of d(q, r). Optionally return distances.

Rationale
- Avoids selecting multiple near‑duplicates (neighbors that are very close to each other relative to q), increasing graph navigability and search quality.

---

Inputs
- Node q: not used directly by this kernel (we assume distances d(q, ⋅) are precomputed and passed in), but its ID may be used for logging.
- Candidates (required):
  - `candidateIds`: contiguous array of node IDs (Int32/Int64)
  - `distToQ`: contiguous array of Float32 distances, parallel to `candidateIds`; length = C
- Parameters (required):
  - `M`: Int (target max neighbors)
- Optional inputs (for diversity check):
  - `distanceBetweenCandidates(i, j) -> Float32`: callback to compute d(candidateIds[i], candidateIds[j])
  - OR an `interCandidate` distance accessor (2D packed matrix or on‑demand computation)
  - `alpha`: Float32 ≥ 0.0 (diversity margin; default 0.0). Use rule: accept c if ∀ r ∈ R, d(c, r) ≥ d(q, c) − α.
- Optional constraints:
  - `deletedMask`: optional bitset/map to exclude deleted nodes (skip before sort)
  - `visitedMask`: optional bitset (usually from search), not required here

Outputs
- `selectedIds`: contiguous array (length ≤ M) of selected neighbor IDs
- `selectedDistToQ` (optional): parallel distances d(q, r) for `selectedIds`

Ordering guarantees
- Output sorted ascending by d(q, ⋅), tiebreakers deterministic (see below).

Data types & memory
- IDs: `Int32` (preferred) or `Int64` — specify at compile time; stable across the index
- Distances: `Float32`
- No heap allocation; the kernel receives scratch buffers for temporary work.

Complexity
- Sorting C by d(q, ⋅): O(C log C)
- Selection pass: in worst case O(C × |R|) where |R| ≤ M (small, e.g., 16), thus effectively O(C·M)

---

Deterministic tie‑breaking
When two candidates have identical d(q, c) within a small tolerance (e.g., 1e‑6):
1) Prefer lower ID (ensures deterministic ordering across runs)
2) If IDs also equal (duplicate), keep only one (first occurrence in stable sort)

Numerical stability
- Distances are Float32; treat NaN/Inf as follows:
  - If d(q, c) is NaN or < 0: skip candidate
  - If d(q, c) is +Inf: allow candidate only in backfill (if needed), but place after finite distances
- Distance between candidates: if the callback returns NaN/Inf, treat as non‑diverse (reject candidate unless in backfill stage) to be conservative

Edge cases
- C == 0: output empty
- M <= 0: output empty
- C <= M: output top‑C (sorted by d(q, ⋅)); skip deleted
- Duplicates in `candidateIds`: collapse to first occurrence after sort
- Self‑loops: if a candidate equals q, skip (upper layer should avoid this but kernel should be defensive)

Backfill policy
- If diversity checks reject too many candidates and |R| < M, append the next best remaining by d(q, ⋅) (excluding deleted/invalid). This ensures degree M when possible.

---

Function signature (pseudocode)
```swift
// Swift-like pseudocode (CPU-only)
// Precondition: candidateIds.count == distToQ.count == C
func nsw_select_neighbors_heuristic(
    candidateIds: UnsafeBufferPointer<Int32>,
    distToQ: UnsafeBufferPointer<Float>,
    M: Int,
    deletedMask: (Int32) -> Bool = { _ in false },
    alpha: Float = 0.0,
    // Optional closure; if nil, diversity check is skipped → simple Top‑M
    distBetweenCandidates: ((Int32, Int32) -> Float)? = nil,
    // Scratch/output buffers (preallocated by caller)
    outSelectedIds: UnsafeMutableBufferPointer<Int32>,            // capacity ≥ M
    outSelectedDistToQ: UnsafeMutableBufferPointer<Float>?,       // capacity ≥ M (optional)
    scratchIndices: UnsafeMutableBufferPointer<Int32>,            // capacity ≥ C
    scratchSortedIdx: UnsafeMutableBufferPointer<Int32>           // capacity ≥ C (optional if stable sort done elsewhere)
) -> Int { // returns count of selected neighbors
    // Implementation described below (see pseudocode)
}
```

Notes
- Caller must provide scratch buffers to avoid heap allocation.
- If `distBetweenCandidates == nil`, the kernel should return simple Top‑M by d(q, ⋅) (after filtering) — used as fallback or for FlatIndex.

---

Detailed pseudocode
```text
Inputs:
  candidateIds[0..C-1], distToQ[0..C-1], target M
  deletedMask(id) -> Bool
  alpha ≥ 0, optional distBetweenCandidates(a,b)

1) Build working set W of eligible candidates:
   - For i in 0..C-1:
       id = candidateIds[i]
       d  = distToQ[i]
       if deletedMask(id) == true: continue
       if !(d is finite) or d < 0: continue
       append (id, d, i) to W

2) Sort W by (d ASC, id ASC) deterministically.

3) If distBetweenCandidates == nil:
      // Simple Top‑M (no diversity check)
      take first min(M, |W|) into R
      write into outSelectedIds/outSelectedDistToQ; return count

4) Diversity selection (heuristic):
   R = []
   For each candidate (id_c, d_qc, idx) in W (in sorted order):
       diversify = true
       For each r in R:
           d_cr = distBetweenCandidates(id_c, r.id)
           if !isFinite(d_cr): continue // treat as non-informative; still check others
           if d_cr < d_qc - alpha: // too close to an already-selected neighbor
               diversify = false
               break
       if diversify:
           append c to R
           if |R| == M: break

5) Backfill (if |R| < M):
   For each remaining candidate (in W order):
       if id not already in R:
           append id to R
           if |R| == M: break

6) Output:
   - Write R[0..|R|-1].id into outSelectedIds
   - If outSelectedDistToQ != nil, write d(q, r) accordingly
   - Return |R|
```

---

Determinism requirements
- Sorting by (distance, ID) and set membership checks must yield identical output across runs.
- Floating-point comparisons use an epsilon tolerance for tie detection (e.g., |d1−d2| ≤ 1e‑6 ⇒ equal).

Threading model
- This kernel is small (C typically ≤ a few hundred); implement single-threaded. Upper layers parallelize at coarser granularity.

Memory & capacity
- Caller allocates scratch buffers sized for |C|; out buffers sized for ≥ M.
- The kernel must not allocate memory.

Validation tests (examples)
1) Basic small set
   - q with 6 candidates; M=3; distances: [0.1, 0.11, 0.12, 0.5, 0.51, 0.52]
   - No distBetweenCandidates: expect top 3 lowest distance IDs.
2) Diversity enforcement
   - 6 candidates forming two tight clusters around q; `distBetweenCandidates` small within cluster, large across clusters
   - With M=3, expect selection to include members from both clusters, not 3 from one cluster
3) Ties & determinism
   - Equal distances for several candidates; verify IDs break ties (lower ID wins)
4) Deleted filtering
   - Mark some candidates as deleted; ensure they’re skipped entirely
5) Backfill
   - Diversity reject reduces R below M; verify backfill fills from next best by distance
6) NaN/Inf handling
   - Inject NaN/Inf distances for q or pairs; ensure conservative acceptance/rejection rules are followed

Integration notes
- In NSW/HNSW insertion, call this kernel after collecting construction candidates for a new node to prune its outgoing neighbors to M.
- For search, you may reuse the same heuristic to prune candidate lists when building a candidate frontier.
- For cosine geometry, consider `preNormalize: true` at index build time to simplify distance interpretation.

Optional variants
- Support `alpha > 0` to relax diversity pruning when aggressive pruning hurts recall.
- Provide a fast path using precomputed inter-candidate distances if available; otherwise compute on demand via callback.

Security & safety
- Bounds check all array accesses; verify output buffer capacities.
- Reject negative or NaN distances; clamp extremes if necessary.

Versioning
- v1.0 of this kernel spec. Future versions may add SIMD or device-accelerated variants with the same semantics.
