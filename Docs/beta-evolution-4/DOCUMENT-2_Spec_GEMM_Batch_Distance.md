# DOCUMENT-2 — Design Spec: CPU GEMM Batch-Distance (AMX)

**Verdict:** Accept for `0.3.0`. Highest throughput ROI of the accepted set.
**Scope:** CPU-only. The GPU GEMM path already exists in `VectorAccelerate`; this spec is the
missing *CPU* path that routes to the Apple AMX coprocessor via Accelerate BLAS.

---

## 1. Motivation

`BatchOperations.findNearest` / `pairwiseDistances`
(`Sources/VectorCore/Operations/BatchOperations.swift`) currently compute batched Euclidean and
cosine distances with hand-tuned SoA kernels that evaluate each query–candidate pair directly
(`Operations/Kernels/BatchKernels.swift`, `BatchKernels_SoA.swift`). These are excellent for
small/medium batches but leave the matrix coprocessor idle.

For large batches the standard reformulation turns the whole problem into one matrix multiply:

```
‖x − y‖² = ‖x‖² + ‖y‖² − 2·⟨x, y⟩
```

Stacking queries as `X (q × d)` and candidates as `Y (n × d)`, the cross-term `X·Yᵀ (q × n)` is
a single GEMM. Accelerate's `cblas_sgemm` dispatches to **AMX on Apple Silicon**, so this is the
documented, supported route to coprocessor throughput — no private intrinsics.

This is exactly how Faiss computes its L2 distance matrices.

---

## 2. API surface

Additive only — existing kernels remain the small-batch fast path.

```swift
public enum MatrixDistance {

    /// Squared-Euclidean distance matrix via GEMM. Result is row-major (q × n):
    /// out[i*n + j] = ‖queries[i] − candidates[j]‖², clamped at 0.
    public static func euclideanSquaredMatrix(
        queries: [Vector512Optimized],     // + 768 / 1536 overloads
        candidates: [Vector512Optimized],
        into out: inout [Float]            // caller-owned, count == q*n
    )

    /// Cosine *distance* matrix (1 − cosine similarity) via GEMM on L2-normalized inputs.
    public static func cosineDistanceMatrix(
        queries: [Vector512Optimized],
        candidates: [Vector512Optimized],
        into out: inout [Float]
    )
}
```

`BatchOperations.findNearest` and `pairwiseDistances` gain an **internal** GEMM path selected by
the crossover heuristic (§5); their public signatures do not change. The standalone
`MatrixDistance` entry is public so `VectorIndex` (IVF centroid assignment) can call it directly
without the Top-K wrapper.

**Why `into out:`** — the result matrix dominates allocation cost (`q*n` floats). A caller-owned
buffer keeps the hot path allocation-free and satisfies the malloc-gate (§6).

---

## 3. Algorithm

For each dimension family (512/768/1536), reusing the existing
`public var storage: ContiguousArray<SIMD4<Float>>`:

1. **Pack** `X` and `Y` into contiguous row-major `[Float]` (`q×d`, `n×d`). The optimized
   storage is already contiguous `SIMD4<Float>` — reinterpret via `withUnsafeBufferPointer` +
   `withMemoryRebound(to: Float.self)`; no per-element copy beyond the row gather.
2. **Norms.** Precompute `‖xᵢ‖²` and `‖yⱼ‖²` with `vDSP_svesq` (already used in
   `Platform/AccelerateSIMDProvider.swift`). Candidate norms are cacheable across queries.
3. **GEMM.** `cblas_sgemm(RowMajor, NoTrans, Trans, q, n, d, -2.0, X, d, Y, d, 0.0, out, n)`
   → `out = −2·X·Yᵀ`.
4. **Norm assembly.** `out[i*n + j] += ‖xᵢ‖² + ‖yⱼ‖²` (vectorized via `vDSP_vsadd` per row).
5. **Clamp.** `out = max(out, 0)` with `vDSP_vthr` (see §4).
6. Euclidean (non-squared) defers `sqrt` to the consumer / Top-K stage; cosine path normalizes
   inputs first so the cross-term *is* the similarity.

---

## 4. Numerical contract

- **Cancellation → clamp.** `‖x‖² + ‖y‖² − 2⟨x,y⟩` is catastrophic cancellation when `x ≈ y`:
  the true value ≈ 0 but rounding yields a small **negative** number. Unclamped, `sqrt`
  produces `NaN`. The spec mandates a clamp at 0 (step 5) and documents that distances below
  ~`√(ε·‖x‖²)` are not reliable in absolute terms (they round to 0) — acceptable for
  nearest-neighbor ranking where such pairs are the trivial top match.
- **Accumulation.** BLAS accumulates the dot-product in the coprocessor; we do not promise
  bit-exact agreement with the SoA kernel. The contract is **`|gemm − soa| < 1e-4` relative**
  for normalized embedding magnitudes (consistent with the existing FP16 parity bound).
- This corrects the proposal's "zero drift" wording (see master §5): the contract is *bounded*,
  not zero.

---

## 5. Crossover heuristic

GEMM wins only when the `O(q·n·d)` multiply amortizes packing + BLAS setup. Extend
`Providers/AutoTuning.swift` (`ParallelHeuristic`) with a matrix branch:

- Route to `MatrixDistance` when `q ≥ Q_MIN && n ≥ N_MIN` (initial guess `Q_MIN≈8`,
  `N_MIN≈256`); else keep the candidate-tiled SoA kernel.
- Thresholds are **measured, not guessed**: a crossover benchmark (below) sweeps `q × n × d`
  and picks the breakeven per dimension family. Mirrors how `VectorAccelerate/GPUDecisionEngine.swift`
  calibrates its CPU/GPU crossover.

---

## 6. Validation

1. **Epsilon-parity test.** Identically seeded `X`, `Y`; assert
   `abs(euclideanSquaredMatrix − soaKernel) < 1e-4` element-wise (512/768/1536). This is the
   cross-implementation parity test the original ecosystem doc (Finding 3) asked for, applied
   *within* Core between the GEMM and SoA paths.
2. **Cancellation test.** `X == Y` rows must yield exactly `0.0` (clamped), never `NaN`/negative.
3. **Allocation gate.** A benchmark asserting `mallocCountTotal == 0` inside the GEMM matrix
   call given a caller-owned `out`. Core has **zero external dependencies today**; adding the
   `ordo-one/package-benchmark` plugin is a deliberate dependency decision — alternatively the
   existing `VectorCoreBench` executable can host a lightweight allocation counter. *Decision
   point flagged for implementation: adopt package-benchmark vs. extend the in-house harness.*
4. **Crossover benchmark.** New `MatrixDistanceBench` suite (alongside `BatchBench`) sweeping
   `q ∈ {1,8,64}`, `n ∈ {64,256,1k,10k}`, `d ∈ {512,768,1536}`; emits the threshold table feeding §5.

---

## 7. Reuse / touch-points

| Reuse | Path |
|---|---|
| Optimized storage (`ContiguousArray<SIMD4<Float>>`) | `Vectors/Vector{512,768,1536}Optimized.swift` |
| `vDSP_svesq`, `vDSP_vsadd`, vDSP threshold | `Platform/AccelerateSIMDProvider.swift` |
| Existing batch entry points (internal routing) | `Operations/BatchOperations.swift` |
| Small-batch fast path (unchanged) | `Operations/Kernels/BatchKernels*.swift` |
| Crossover heuristic extension | `Providers/AutoTuning.swift` |

**No new public vector types. No GPU. No new external dependency** (modulo the §6.3 decision).
