# DOCUMENT-6 — Page-Alignment Feasibility: making more storage `bytesNoCopy`-eligible

**Question (from VA, follow-up to DOCUMENT-5 R3):** VectorAccelerate observed that
`AlignedMemory` / `AlignedDynamicArrayStorage` are not `MTLDevice.makeBuffer(bytesNoCopy:)`-eligible.
What's the complexity/viability of supporting page-aligned storage more broadly?

**Short answer:** The complexity is *low* — the alignment machinery already exists. But for the
**per-vector** storages it's the wrong unit (catastrophic memory bloat), and the **batch** case it's
actually meant for is already shipped (`SoA`, `PageAlignedBuffer`). Recommendation below, plus a
`PageBridgeable` extraction if we expect to bridge more types.

---

## 1. Findings — the alignment machinery is already there

The base-alignment half is essentially free; it's already plumbed:

- `AlignedMemory.allocateAligned(type:count:alignment:)` (`Storage/AlignedMemory.swift:53`) accepts
  **arbitrary power-of-two alignment ≥ 16**. `PlatformConfiguration.pageSize` (16 KB on Apple
  Silicon) qualifies. `deallocate` is `free()`.
- `AlignedDynamicArrayStorage.init(dimension:alignment:)` (`Storage/AlignedDynamicArrayStorage.swift:29`)
  and `AlignedValueStorage.init(count:alignment:)` (`Storage/AlignedValueStorage.swift:168`) **already
  take an `alignment:` parameter** that flows straight into `posix_memalign`. Passing `pageSize`
  page-aligns the *base* today.
- Both already **track which allocator they used** and free correctly (`AlignedMemory.deallocate(ptr)`
  vs `.deallocate()` — `AlignedDynamicArrayStorage.swift:109–111`). The allocator-correct-deinit
  hazard (the BE3 bug class) is already solved in these types.

So "support page alignment" is **not** blocked on the allocator. The two genuine gaps are:

1. **Length page-rounding.** These allocate exactly `count * MemoryLayout<Float>.stride` bytes.
   `makeBuffer(bytesNoCopy:)` on macOS requires the **length** to be a page multiple *as well as* the
   base — so a page-aligned base with an unrounded length is still rejected. Fix: when page-aligning,
   over-allocate to `roundUpToPage(count*stride)` and track the padded length (exactly what
   `PageAlignedBuffer`/`SoA` do).
2. **Escaping accessor + ownership handoff.** Need `pageAlignedBytes` (escaping base+length) and
   `consumeAllocation()` (transfer to a Metal deallocator without double-free). Today these types
   expose only scoped `withUnsafeMutableBufferPointer`.

Both are the **exact pattern already shipped** in `SoA` and `PageAlignedBuffer` this cycle.

---

## 2. The catch — economics, not difficulty

A page is **16 KB**. Page-aligning *per-vector* storage means a 384-dim `DynamicVector`
(1,536 bytes) rounds up to a full page — **~10.7× memory bloat per vector**. For a database of
millions of individually-stored vectors that is a non-starter.

Page alignment only pays for itself when the per-page overhead **amortizes across a contiguous
batch** — one page-rounded slab holding many vectors, where the rounding waste is a single trailing
page (well under 0.01%). So the unit that should be page-aligned is a **batch/contiguous layout**,
never per-vector storage.

This is already covered:

- **`SoA<Vector>`** — opt-in page-aligned (`build(from:pageAligned: true)`); the batch candidate
  database. The high-value GPU-bridge target. ✅ shipped (DOCUMENT-5 R1/R2).
- **`PageAlignedBuffer`** — an arbitrary page-aligned `[Float]` slab (`Storage/UnifiedVectorBuffer.swift`).
  This is the answer for the **EmbedKit → VectorAccelerate dense handoff**: EmbedKit allocates
  `PageAlignedBuffer(elementCount: N*dim)`, writes its N embeddings in once, and hands the
  page-aligned pointer to VA — no per-vector bloat, no retrofit. ✅ shipped (DOCUMENT-4 S5).

---

## 3. Foreseeable approaches

| # | Approach | Complexity | Memory | When |
|---|---|---|---|---|
| **A** | **Status quo** — use `SoA`/`PageAlignedBuffer` for batches; don't page-align per-vector storage | — (done) | optimal | **Default. Recommended.** Covers the batch handoff today. |
| **B** | **Add an opt-in page-aligned batch type on demand** — e.g. a contiguous page-aligned `[Vector512Optimized]` slab, or a page-aligned `DynamicVector` *batch* | ~½ day/type | optimal (batch) | When a concrete bridge target appears that `SoA`/`PageAlignedBuffer` don't fit |
| **C** | **Extract a shared `PageBridgeable` helper**, then adopt it in `SoA` + future types | ~1 day once | optimal | If we expect ≥ 2 more bridged types — makes each subsequent one ~1–2 h and contains the double-free risk in one place |
| **D** | **Page-align the per-vector storages** (`AlignedDynamicArrayStorage`/`AlignedValueStorage`) | ~½ day/type | **~10× bloat at scale** | **Not recommended** — wrong unit |

### Approach C sketch — `PageBridgeable`

Today the page-round + accessor + `consumeAllocation` + allocator-correct-free logic is implemented
*twice* (`SoA`, `PageAlignedBuffer`) and would be copy-pasted a third time per new type — each copy
re-risks the double-free. Centralize it:

```swift
/// The zero-copy GPU-bridge surface: a contiguous region whose base + length are
/// page multiples, handable to MTLDevice.makeBuffer(bytesNoCopy:).
public protocol PageBridgeable: AnyObject {
    /// Page-aligned base + page-rounded length, or nil if not page-aligned.
    var pageAlignedBytes: (base: UnsafeRawPointer, byteCount: Int)? { get }
    /// Transfer ownership to the caller (for a Metal bytesNoCopy deallocator). After
    /// this the object no longer frees on deinit and `pageAlignedBytes` returns nil;
    /// the caller frees via AlignedMemory.deallocate / free.
    func consumeAllocation() -> (base: UnsafeMutableRawPointer, byteCount: Int)?
}

/// Embeddable owner of a page-aligned, page-length-rounded allocation. A storage
/// type composes this instead of re-implementing page alignment + the free contract.
struct PageAlignedRegion {            // value type, embedded in the (class) storage
    let base: UnsafeMutableRawPointer
    let byteCount: Int                // page multiple
    private(set) var owns: Bool = true

    init?(logicalBytes: Int) {        // nil for empty
        guard logicalBytes > 0 else { return nil }
        let page = PlatformConfiguration.pageSize
        let padded = ((logicalBytes + page - 1) / page) * page
        let raw = try? AlignedMemory.allocateAligned(type: UInt8.self, count: padded, alignment: page)
        guard let raw else { return nil }
        UnsafeMutableRawPointer(raw).initializeMemory(as: UInt8.self, repeating: 0, count: padded)
        self.base = UnsafeMutableRawPointer(raw); self.byteCount = padded
    }
    mutating func consume() -> (UnsafeMutableRawPointer, Int)? {
        guard owns else { return nil }; owns = false; return (base, byteCount)
    }
    func freeIfOwned() { if owns { AlignedMemory.deallocate(base) } }   // call from deinit
}
```

A bridged storage then holds an optional `PageAlignedRegion` (non-nil only on the opt-in path),
forwards `pageAlignedBytes`/`consumeAllocation` to it, and calls `region?.freeIfOwned()` in `deinit`.
`SoA`'s current bespoke fields collapse into this; new types adopt it in a few lines. (Swift protocols
can't add stored state, so composition — not a protocol default — is the right tool for the shared
*implementation*; the protocol just names the public contract.)

---

## 4. Recommendation

1. **Default to Approach A.** The GPU-bridge value is in batches, and both batch primitives ship in
   0.3.0. Point VA at `SoA(pageAligned: true)` for candidate databases and `PageAlignedBuffer` for the
   EmbedKit dense handoff. **Do not** page-align per-vector storage (Approach D).
2. **Reach for Approach C** if a second/third bridge target is on the horizon — the `PageBridgeable`
   extraction is a 1-day investment that de-risks every future adoption and removes the copy-pasted
   double-free hazard. Otherwise defer it (YAGNI).
3. **Approach B** is the per-need fallback: a concrete new batch slab is ~½ day with tests, low risk,
   because the alignment + allocator-correct-free machinery already exists.

**Net:** technically cheap, already solved for the cases that matter; the only thing to actively avoid
is page-aligning the per-vector storages.
