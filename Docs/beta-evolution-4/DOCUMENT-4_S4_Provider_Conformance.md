# DOCUMENT-4 / S4 — Provider Conformance Contract

**Status:** Implemented (contract + conformance tests) for `0.3.0`.
**Audience:** Anyone conforming a type to VectorCore's provider protocols — primarily
`VectorAccelerate` (`ComputeEngine: ComputeProvider`, `BufferPool: BufferProvider`,
`MetalContext: AccelerationProvider`, per its `VECTORCORE_ALIGNMENT_ROADMAP.md`).

The three protocols already exist in Core (`Protocols/ComputeProvider.swift`,
`Protocols/BufferProvider.swift`, `Protocols/CoreProtocols.swift`). S4 does **not** add
protocols — it pins the *semantic laws* a conformer must satisfy and ships an executable
conformance suite (`Tests/ComprehensiveTests/ProviderConformanceTests.swift`) that validates
Core's own conformers and minimal mocks. Downstream packages should mirror that suite against
their conformers.

A type that compiles against a protocol is not the same as a *correct* conformer. The laws
below are the difference.

---

## `ComputeProvider`

A strategy for executing work (CPU, GPU, Neural). Conformers must satisfy:

- **C1 — execute fidelity.** `execute { x }` returns `x`; a thrown error propagates unchanged.
- **C2 — parallelExecute order & completeness.** `parallelExecute(items:work:)` returns an
  array of length `items.count` where `result[j] == work(items.lowerBound + j)`. Results are in
  **index order regardless of completion order** — a conformer that returns results in the order
  tasks finished is non-conformant. Each index's `work` is invoked exactly once.
- **C3 — parallelForEach completeness.** `body` is invoked **exactly once per item** — no index
  skipped, none duplicated.
- **C4 — parallelReduce partition.** `rangeWork` is invoked on a set of disjoint sub-ranges
  whose union is exactly `items` (or the whole range once). The result is the fold of the
  partial results with `combine`, seeded by `initial`.
- **C5 — reduce algebra (caller + conformer contract).** The caller MUST supply a `combine`
  that is **associative and commutative**, and an `initial` that is the **identity element**
  for `combine` (e.g. `0` for `+`, `[]` for concat, an empty `TopKBuffer` for a top-k merge).
  Under this contract the result is deterministic for any chunking. *Conformer note:* because
  `initial` is the identity, a conformer may fold it once, many times, or omit it in a
  single-chunk fast path — all are equivalent. (Core's `CPUComputeProvider` omits it on the
  sequential path; this is conformant precisely because `initial` is the identity. A conformer
  that wants to be robust to misuse may fold `initial` uniformly.)
- **C6 — empty range.** `parallelExecute → []`; `parallelForEach → ` no `body` calls;
  `parallelReduce → initial`.
- **C7 — metadata sanity.** `maxConcurrency ≥ 1`, `deviceInfo.maxThreads ≥ 1`,
  `deviceInfo.preferredChunkSize ≥ 1`.
- **C8 — error propagation.** A throw from any `work`/`body`/`rangeWork` propagates out of the
  call. No rollback is promised: side effects already performed by other items may persist.

*GPU conformer guidance.* A provider may dispatch `execute`/`parallel*` to the GPU, but the
observable semantics above are mandatory. In particular, batching many items into one GPU
dispatch is fine **only if** results are returned in index order (C2).

---

## `BufferProvider`

Managed, poolable memory. Conformers must satisfy:

- **B1 — size.** `acquire(size:)` returns a handle whose backing buffer is **at least** `size`
  bytes (`handle.size` may report the requested size; the allocation is ≥ that).
- **B2 — alignment.** `handle.pointer` is aligned to `alignment`, and `alignment` is a power of
  two ≥ 1. (Core's default is 64.)
- **B3 — writability.** The first `size` bytes are writable and read back what was written.
- **B4 — non-aliasing while active.** Two buffers that are simultaneously *active* (acquired,
  not yet released) never overlap; their handle ids and pointers differ. After `release`, the
  memory may be reused by a later `acquire`.
- **B5 — release contract.** `release` returns the buffer to the pool; its contents are
  undefined afterward and its pointer must not be used.
- **B6 — clear keeps active handles valid.** `clear()` frees *cached* (released) buffers only;
  buffers still active remain valid.
- **B7 — statistics sanity.** Counters are non-negative; `hitRate ∈ [0, 1]`.

*GPU integration caveat (stability finding).* `BufferHandle.pointer` is a **CPU-addressable**
`UnsafeMutableRawPointer`. This maps cleanly onto Metal `.shared` storage on unified memory
(`MTLBuffer.contents()` yields such a pointer). It does **not** map onto GPU-*private* buffers,
which have no CPU pointer — `BufferProvider` is not the right seam for those, and VectorAccelerate
should expose private-storage buffers through a separate, Metal-typed API rather than forcing
them through `BufferHandle`.

---

## `AccelerationProvider`

Hardware acceleration with a provider-specific `Config` (a PAT; `init(configuration:) async throws`).
Conformers must satisfy:

- **A1 — isSupported purity.** `isSupported(for:)` is deterministic and side-effect free.
- **A2 — correctness parity (the critical law).** For a supported operation, `accelerate`
  returns a value of the **same type** as `input` whose value matches the CPU reference within a
  documented tolerance (`|accelerated − reference| < ε`, e.g. `1e-4` for FP32 distance — see
  DOCUMENT-2 and the ecosystem doc's "Finding 3"). An accelerator that is fast but wrong is
  non-conformant.
- **A3 — unsupported throws.** `accelerate` for an unsupported operation **throws** rather than
  returning an incorrect or silently-CPU result; callers gate on `isSupported` first.
- **A4 — init is all-or-nothing.** `init(configuration:)` returns a ready provider or throws
  (e.g. hardware unavailable). No half-initialized state.

*Parity is VA's responsibility to test against its own kernels* — Core cannot run Metal. The
conformance suite validates A1/A3/A4 and type-preservation with a mock; A2 (numerical parity)
must be asserted in VectorAccelerate using identically-seeded inputs versus a Core reference.

---

## Conformance suite

`Tests/ComprehensiveTests/ProviderConformanceTests.swift` provides reusable checkers —
`checkComputeProvider`, `checkBufferProvider`, `checkAccelerationProvider` — and runs them
against:

- `CPUComputeProvider` in `.sequential`, `.parallel`, and `.automatic` modes;
- the protocol **default** implementations (via `MockComputeProvider`, which implements only
  `execute` and inherits the rest) — so VA can rely on the defaults;
- `SwiftBufferPool` and a `MockBufferProvider`;
- a `MockAccelerator` for A1/A3/A4 + type preservation.

Downstream packages should copy these checkers and run them against their own conformers
(`ComputeEngine`, `BufferPool`, `MetalContext`), adding the A2 numerical-parity assertion that
only they can run.
