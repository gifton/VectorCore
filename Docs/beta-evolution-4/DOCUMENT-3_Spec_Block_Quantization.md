# DOCUMENT-3 — Design Spec: Block-Wise Quantization (`Q8_0`)

**Verdict:** Accept for `0.3.0` — **gated on a committed consumer** (see §6).
**Scope:** A new CPU storage *format* and its dot-product kernel. Sits alongside the existing
whole-vector quantization in `Sources/VectorCore/Quantization/QuantizationSchemes.swift` and
`Operations/Kernels/QuantizedKernels.swift`.

---

## 1. Motivation

Today's INT8 quantization (`LinearQuantizationParams`) derives **one** scale (and zero-point)
for the entire vector. Modern LLM embeddings contain *outlier features* — a few dimensions with
activations far larger than the rest. A whole-vector scale is set by the outlier, so the
remaining ~99% of dimensions collapse into a handful of quantization levels and lose resolution.

GGML's fix, `Q8_0`, is **block-wise**: split the vector into fixed blocks and give each block its
own scale. An outlier corrupts only its own 32-element block.

---

## 2. Format (`Q8_0`, correctly specified)

- **Block size:** 32 elements (GGML's `QK8_0`). A 512-dim vector → 16 blocks; 768 → 24; 1536 → 48.
- **Per block:** one `Float16` **scale** `d`, plus 32 × `Int8` quantized values `q`.
  Reconstruction: `xᵢ ≈ d · qᵢ`.
- **Symmetric, scale-only.** `Q8_0` stores *no* offset/min. (The proposal's "scale *and* offset
  per block" is `Q8_1`, which adds a per-block min for asymmetric ranges — noted as a future
  variant, not the `0.3.0` target. See master §5.)
- **Quantization:** per block, `d = max(|xᵢ|) / 127`; `qᵢ = round(xᵢ / d)` clamped to
  `[−127, 127]` (127, not 128, keeps it symmetric).

```swift
public struct Vector512Q8_0: Sendable {            // + 768 / 1536 variants
    // 16 blocks × (1 Float16 scale + 32 Int8 values)
    public var scales: ContiguousArray<Float16>     // count = 16
    public var codes:  ContiguousArray<Int8>        // count = 512
}
```

Storage: `512·1 + 16·2 = 544` bytes vs `512·4 = 2048` for FP32 → **~3.76×** compression,
materially better fidelity than whole-vector INT8 at the same size.

---

## 3. Dot-product kernel

Block-local accumulation, FP32 reduction:

```
acc_f32 = 0
for each block b:
    acc_i32 = Σ over 32 lanes of  Int32(qA[i]) * Int32(qB[i])     // SIMD widening multiply
    acc_f32 += Float(acc_i32) * Float(scaleA[b]) * Float(scaleB[b])
return acc_f32
```

- Integer products accumulate in `Int32` *within* a block (max magnitude `32·127·127 ≈ 5.2e5`,
  safely inside `Int32`) — this avoids the `Int16` overflow class of bug fixed in the `0.2.2`
  BE3 audit for whole-vector INT8 (`QuantizedKernels.swift`).
- The block scale is applied once per block in FP32, so different blocks' scales never mix in
  integer space.
- Euclidean follows from the dot identity (`‖a−b‖²` expanded), or a direct per-block squared-
  difference kernel; cosine reuses the dot plus stored norms.

Hardware: ARM64 NEON `vmull`/`vdot` and x86 AVX2 `vpmaddubsw`-style widening live in the existing
`VectorCoreC` shim (`Sources/VectorCoreC/arm64/vc_arm64.c`, `x86_64/vc_x86.c`) — the same place
the current INT8 dot (`vc_dot_int8`) already lives. The Swift kernel provides a portable fallback.

---

## 4. Numerical contract

- **Quantization error per block:** `|xᵢ − d·qᵢ| ≤ d/2 = max(|x|_block)/254`. Bounded by the
  block's own dynamic range, not the vector's global outlier — the entire point.
- **Dot-product error:** bounded; accumulation is exact in `Int32` within a block, and the
  cross-block FP32 sum is ordered (optionally Kahan if a parity test shows drift). No "zero
  drift" claim (master §5).
- **Parity test target:** for outlier-free vectors, `Q8_0` dot vs FP32 dot within the empirical
  INT8 bound; for vectors with injected outliers, `Q8_0` must beat whole-vector INT8 on
  reconstruction MSE (this is the test that *justifies* the format).

---

## 5. Validation

1. **Reconstruction MSE test:** synthetic embeddings with planted outlier dimensions; assert
   `MSE(Q8_0) < MSE(wholeVectorINT8)` by a meaningful margin.
2. **Overflow test:** worst-case all-`±127` blocks must not overflow `Int32` accumulation.
3. **Round-trip test:** quantize → dequantize within the per-block bound.
4. **Kernel parity:** Swift fallback vs `VectorCoreC` NEON/AVX path agree to integer-exact.
5. **Allocation gate:** quantize/dot over preallocated buffers with `mallocCountTotal == 0`
   (same harness decision as DOCUMENT-2 §6.3).

---

## 6. Consumer gate (blocking condition)

A storage format is dead weight without a producer and a consumer. **This spec does not ship
until one of the following commits:**

- **EmbedKit** stores embeddings as `Q8_0` (it already has a quantization layer with INT8/INT4/
  FP16/binary — `EmbedKit/Sources/EmbedKit/Optimization/Quantization.swift` — but per-vector;
  `Q8_0` would be an additive format with better outlier behavior), **or**
- **VectorIndex** adopts `Q8_0` as a raw-vector code path (distinct from its existing PQ codes).

Until then, `Q8_0` remains specified-but-unbuilt. **Action for implementation:** confirm a
consumer before scheduling; if none commits this cycle, this document moves to the same "Defer"
status as binary/sparse in DOCUMENT-1, with the design preserved here for immediate pickup.

---

## 7. Reuse / touch-points

| Reuse | Path |
|---|---|
| Existing quant params / schemes | `Quantization/QuantizationSchemes.swift` |
| Existing INT8 kernels + overflow lessons | `Operations/Kernels/QuantizedKernels.swift` |
| C SIMD shim (NEON/AVX) | `Sources/VectorCoreC/{arm64,x86_64}/` (`vc_dot_int8` neighbor) |
| Float16 storage support | `Operations/Kernels/MixedPrecisionKernels.swift` |
