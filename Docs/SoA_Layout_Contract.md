# SoA Memory-Layout Contract

**Status:** 🔒 **Frozen — stable as of VectorCore 0.3.0.**
**Audience:** Any consumer that reads `SoA<Vector>` memory directly — primarily
**VectorAccelerate** Metal kernels indexing a zero-copy `MTLBuffer` built from
`SoA.pageAlignedBytes` / `SoA.consumeAllocation()`.
**Programmatic source of truth:** `SoALayout` (`Sources/VectorCore/Storage/SoALayout.swift`),
reachable as `SoA.layoutDescriptor` (live instance) or `SoALayout.forType(_:count:)` (ahead of allocation).

> **What "frozen" means.** The element type, the index formula, and the lane stride
> described here will not change without a major-version bump and advance notice to
> downstream consumers. The contract is regression-locked by
> `Tests/ComprehensiveTests/SoALayoutContractTests.swift` — if those tests change, this
> document and the consumers must change with them.

---

## 1. The layout

`SoA<Vector>` stores `count` candidate vectors **lane-major, then candidate index**:

```
element(lane ℓ, candidate j)  ==  buffer[ℓ * count + j]
```

- **Element type:** `SIMD4<Float>` — 4 contiguous dimensions, **16 bytes**
  (`MemoryLayout<SIMD4<Float>>.stride == 16`).
- **Lanes per vector:** `lanes = dimension / 4`. Exact for every supported type — see §4 —
  so there are **no partial / tail lanes**. Every lane is a full `SIMD4<Float>`.
- **Lane stride:** consecutive lanes are `count` elements apart, i.e.
  `laneStrideBytes = count * 16`. This confirms VA's assumption exactly
  (`count * MemoryLayout<SIMD4<Float>>.stride`).
- **The candidate axis (`count`, "N") is never padded.** `count` *is* the stride between
  lanes. There is no block padding of N to any multiple — see §3 for the one subtlety
  (whole-buffer page rounding, which is **not** candidate-axis padding).

Both producer and consumer inside VectorCore already agree on this single formula: the
builder writes `buffer[i * count + j]` (`SoA.build(from:)`) and the hot CPU kernel reads
`lanePointer(i)[j] = buffer[i * count + j]` (`BatchKernels_SoA`). Freezing it here changes
nothing at runtime; it only publishes what is already true.

### Byte-offset form (for a shader)

```
byteOffset(lane ℓ, candidate j)  ==  (ℓ * count + j) * 16
```

A Metal kernel indexing the candidate buffer as `device const float4*` reads candidate `j`'s
lane `ℓ` at `buffer[ℓ * count + j]`. Pull `count` and `lanes` from the descriptor (§2); do
**not** hardcode them and do **not** derive `count` from the buffer's byte length (§3).

---

## 2. The descriptor — `SoALayout`

Derive constants from this type, not from magic numbers, so producer and consumer share one
source of truth.

| Member | Meaning | Formula |
|---|---|---|
| `lanes` | SIMD4 lanes per vector | `dimension / 4` |
| `count` | candidate count **N** (the true stride) | — |
| `elementStrideBytes` (static) | size of one element | `16` |
| `laneStrideBytes` | bytes between consecutive lanes | `count * 16` |
| `logicalByteCount` | bytes of real data | `lanes * count * 16` |
| `allocatedByteCount` | bytes the allocation occupies | page-rounded (§3) |
| `elementCount` | logical elements | `lanes * count` |
| `elementIndex(lane:candidate:)` | the frozen index formula | `lane * count + candidate` |

Obtain it two ways, which are guaranteed equal for the same `(type, count, pageAligned)`:

```swift
// From a live SoA:
let layout = soa.layoutDescriptor

// Ahead of allocation (e.g. to size a buffer or precompute shader constants):
let layout = SoALayout.forType(Vector768Optimized.self, count: n, pageAligned: true)
```

> Page-alignment is **opt-in everywhere**: `forType`, `SoA.build`, and `SoA.init` all default
> `pageAligned: false`. A GPU consumer sizing a zero-copy buffer must pass `pageAligned: true`
> to `forType` so its `allocatedByteCount` matches a `build(from:pageAligned: true)` SoA. The
> `lanes`, `count`, `laneStrideBytes`, and `logicalByteCount` are identical either way — only
> `allocatedByteCount` (the `makeBuffer` length) depends on the flag.

---

## 3. Allocated vs. logical bytes — the one sharp edge

For a **page-aligned** SoA, `MTLDevice.makeBuffer(bytesNoCopy:)` on macOS requires the
*length* to be a page multiple, so the **whole buffer** is rounded up:

```
allocatedByteCount == roundUpToPage(logicalByteCount)   ≥   logicalByteCount
```

The trailing `allocatedByteCount - logicalByteCount` bytes are **zero-filled slack** beyond
the data. This rounding is applied once, to the end of the entire buffer — it does **not**
pad the candidate axis and does **not** change the lane stride.

- **Pass `allocatedByteCount`** as the `length:` to `makeBuffer(bytesNoCopy:)`.
- **Bound your kernel by `lanes` and `count`** from the descriptor. The valid indexed region
  is exactly `[0, lanes * count)` elements.
- 🚫 **Never reverse-engineer `count` from `allocatedByteCount`.** It is page-inflated:
  e.g. at 512-dim, N=5 ⇒ `logicalByteCount = 10240`, but `allocatedByteCount = 16384`
  (16 KB page). `16384 / (128 * 16) = 8 ≠ 5`. Take N from `count`.

For a **non-page-aligned** SoA, `allocatedByteCount == logicalByteCount` and
`pageAlignedBytes` is `nil` (there is no zero-copy pointer to offer).

---

## 4. `SoACompatible` coverage

The zero-copy SoA path requires a **static** dimension/lane count, so it applies to the
fixed-dimension optimized vector types — **not** `DynamicVector`:

| Type | `dimension` | `lanes` | SoACompatible |
|---|---|---|---|
| `Vector384Optimized` | 384 | 96 | ✅ |
| `Vector512Optimized` | 512 | 128 | ✅ |
| `Vector768Optimized` | 768 | 192 | ✅ |
| `Vector1536Optimized` | 1536 | 384 | ✅ |
| `DynamicVector` | runtime | — | ❌ (no static `lanes`) |

The two dimensions VA specifically asked about — **768 and 1536 — are both supported.**
`DynamicVector` stays on the staged (copy) path by design.

> Scope note: this contract covers the **FP32** `SoA<Vector>`. The separate `SoAFP16`
> mixed-precision cache is an internal type and is *not* part of this zero-copy bridge
> contract.

---

## 5. Lifetime & deallocation contract

The page-aligned buffer is Core-owned memory. Two supported modes:

### Borrow mode (recommended when the SoA outlives the buffer)
Hold a strong reference to the `SoA` and read `pageAlignedBytes` **without consuming**:

```swift
guard let (base, byteCount) = soa.pageAlignedBytes else { /* not page-aligned */ }
let mtlBuffer = device.makeBuffer(bytesNoCopy: UnsafeMutableRawPointer(mutating: base),
                                  length: byteCount, options: .storageModeShared,
                                  deallocator: nil)   // SoA frees on deinit
```

**Object lifetime is the sole validity guarantee** for the pointer: the `SoA` frees the
memory on `deinit`, so it **must outlive** the `MTLBuffer`.

### Transfer mode (hand ownership to Metal)
Call `consumeAllocation()` to take ownership, then free from the Metal deallocator:

```swift
guard let (base, byteCount) = soa.consumeAllocation() else { /* not page-aligned */ }
let mtlBuffer = device.makeBuffer(bytesNoCopy: base, length: byteCount,
                                  options: .storageModeShared) { ptr, _ in
    AlignedMemory.deallocate(ptr)   // ≡ free(ptr); see below
}
```

- **The deallocator must call `AlignedMemory.deallocate(base)`**, which is exactly
  `free(base)` — the buffer comes from `posix_memalign`, which is freed with `free`.
  (`free(base)` directly is equivalent.)
- **It is safe to free on an arbitrary thread** at `MTLBuffer`-release time — `free` is
  thread-safe.
- After `consumeAllocation()`, the `SoA` no longer frees on `deinit` and `pageAlignedBytes`
  returns `nil`; do not use the `SoA` for CPU compute afterward.

🚫 **Do not** mix modes (consume *and* rely on deinit) — that double-frees.

---

## 6. Golden parity fixture

A small, fully reproducible fixture to validate a downstream shader against VectorCore's CPU
kernel. Values below are regression-locked in `SoALayoutContractTests.swift`.

**Construction** (512-dim, N = 5):

```
query     q   = [1, 1, …, 1]              (512 dims)
candidate c_k = [1+k, 1+k, …, 1+k]        (512 dims), k = 0 … 4
```

Since `q − c_k = [−k, …]` over 512 dims, the Euclidean **squared** distance is `512 · k²`.

**Expected outputs** (`BatchKernels_SoA.euclid2_512`, squared distance):

| k | candidate value | ‖q − c_k‖² |
|---|---|---|
| 0 | 1.0 | 0 |
| 1 | 2.0 | 512 |
| 2 | 3.0 | 2048 |
| 3 | 4.0 | 4608 |
| 4 | 5.0 | 8192 |

**Descriptor for this fixture** — all values are **host-invariant** except `allocatedByteCount`:

| field | value | host-invariant? |
|---|---|---|
| `lanes` | 128 | ✅ |
| `count` | 5 | ✅ |
| `elementStrideBytes` | 16 | ✅ |
| `laneStrideBytes` | 80 | ✅ |
| `logicalByteCount` | 10240 | ✅ |
| `allocatedByteCount` | 16384 (16 KB page) / 12288 (4 KB page) | ❌ depends on `getpagesize()` |
| `elementIndex(lane: 1, candidate: 0)` | 5 | ✅ |
| `elementIndex(lane: 127, candidate: 4)` | 639 | ✅ |

> `allocatedByteCount` is `roundUpToPage(10240)` — **16384** on Apple Silicon (16 KB pages,
> the VA target) but **12288** on a 4 KB-page host (most x86). It is the only value here that
> varies by platform; always read it from the live descriptor and never hardcode it. Every
> other value — the entire indexing core — is identical on every platform.

**Buffer spot-check:** candidate `j` is the constant `1+j`, so for every lane `ℓ`,
`buffer[ℓ * 5 + j] == SIMD4<Float>(repeating: Float(1 + j))`.

---

## 7. Notes & change policy

- **`blockSize` removed (0.3.0).** The former `SoA.init(vectors:blockSize:pageAligned:)`
  `blockSize` parameter was a **no-op** and was removed to eliminate any implication of
  candidate-axis padding. (The only real block size is an internal CPU-kernel compute-tiling
  constant that has no effect on this layout.)
- **Breaking change = major bump + notice.** Any change to the index formula, element type,
  or stride is breaking. The frozen tests are the tripwire.
- **R4 / transparent dispatch is orthogonal to this contract.** The
  `BatchKernelProvider` hook (`Sources/VectorCore/Protocols/BatchKernelProvider.swift`,
  dispatched from `Operations.findNearest` / `Operations.findNearestBatch`) lets an installed
  GPU provider service Core's k-NN entry points. It is independent of buffer layout; see
  `Docs/beta-evolution-4/DOCUMENT-5_VectorAccelerate_Integration_Requests.md` (R4).

### Related documents
- `Docs/beta-evolution-4/DOCUMENT-5_VectorAccelerate_Integration_Requests.md` — R1/R2/R4 (page-align + accessor + dispatch).
- `Docs/beta-evolution-4/DOCUMENT-6_Page_Alignment_Feasibility.md` — why batches (not per-vector) are page-aligned; the `PageBridgeable` option (not required for an SoA-only consumer).
