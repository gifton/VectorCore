# VectorAccelerate Integration (R1, R2, R4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unblock VectorAccelerate's zero-copy batch GPU search and transparent GPU dispatch by page-aligning the `SoA` candidate buffer (R1), exposing it publicly with a lifetime contract (R2), and adding a `BatchKernelProvider` hook so an installed GPU provider transparently services `Operations.findNearest`/`findNearestBatch` (R4).

**Architecture:** R1/R2 add an *opt-in* page-aligned allocation mode to `SoA` (default stays 16-byte, no memory regression) plus public accessors — reusing the existing `AlignedMemory` (`posix_memalign`/`free`) + `PlatformConfiguration.pageSize`, mirroring the `PageAlignedBuffer` page-rounding pattern. R4 defines a `ComputeProvider` sub-protocol and downcasts the `@TaskLocal` `Operations.computeProvider` to it, taking precedence over the CPU paths (including the new GEMM routing).

**Tech Stack:** Swift 6 (strict concurrency), swift-testing (`import Testing`), Accelerate (already a dep), `@testable import VectorCore`.

---

## Context (why)

VectorAccelerate filed `future/VectorAccelerate/docs/VECTORCORE_INTEGRATION_REQUESTS.md`. Verification (see `DOCUMENT-5_VectorAccelerate_Integration_Requests.md`) found **none of R1/R2/R4 are supported**, and R3 rests on a wrong premise (page alignment never landed on `AlignedMemory`; the only page-aligned storage is `PageAlignedBuffer`, new on this branch). VA's `makeNoCopyBuffer` is ready to consume aligned Core storage; the `SoA` candidate database is the high-value object to bridge but is neither page-aligned nor publicly addressable.

## File structure

- **Modify:** `Sources/VectorCore/Storage/SoA.swift` — add opt-in page-aligned allocation + `pageAlignedBytes` / `withUnsafeRawBuffer` accessors + allocator-correct `deinit`. (R1, R2)
- **Create:** `Sources/VectorCore/Protocols/BatchKernelProvider.swift` — the `ComputeProvider` sub-protocol. (R4)
- **Modify:** `Sources/VectorCore/Operations/Operations.swift` — downcast `computeProvider` to `BatchKernelProvider` in `findNearest` and `findNearestBatch`. (R4)
- **Create:** `Tests/ComprehensiveTests/SoAPageAlignTests.swift` — R1/R2 tests.
- **Create:** `Tests/ComprehensiveTests/BatchKernelProviderTests.swift` — R4 tests.
- **Modify:** `Docs/beta-evolution-4/DOCUMENT-5_VectorAccelerate_Integration_Requests.md` — mark R1/R2/R4 done; record the R3 reply. (R3)

Facts this plan relies on (verified):
- `SoA.swift:57` `@usableFromInline internal let buffer: UnsafeMutablePointer<SIMD4<Float>>`; `:58` `private let bufferCapacity: Int`; allocated `:67`/`:95` via `.allocate(capacity:)`; freed `:76–81` via `.deinitialize`+`.deallocate()`. Two inits: `private init(count:)` and `public init(vectors:blockSize:)`; `static func build(from:)`. `SoACompatible.lanes` = dimension/4 (512 → 128).
- `AlignedMemory.allocateAligned(type:count:alignment:)` (posix_memalign, throws `VectorError.allocationFailed`) and `AlignedMemory.deallocate(_ ptr:)` (→ `free`). `PlatformConfiguration.pageSize` = `getpagesize()`.
- `Operations.computeProvider` is `@TaskLocal public static var ... = CPUComputeProvider.automatic`. `Operations.findNearest` returns `[NearestNeighborResult]` (`.index`/`.distance`, public init). `findNearestBatch` returns `[[NearestNeighborResult]]` and currently routes to `gemmFindNearestBatch` then a per-query `parallelExecute`.

---

## Task 1: Page-aligned SoA buffer + public accessors (R1, R2)

**Files:**
- Modify: `Sources/VectorCore/Storage/SoA.swift`
- Test: `Tests/ComprehensiveTests/SoAPageAlignTests.swift`

- [ ] **Step 1: Write the failing tests**

Create `Tests/ComprehensiveTests/SoAPageAlignTests.swift`:

```swift
//
//  SoAPageAlignTests.swift
//  VectorCore
//
//  R1/R2: opt-in page-aligned SoA buffer + public bytesNoCopy-ready accessor.
//

import Testing
import Foundation
@testable import VectorCore

@Suite("SoA page alignment (R1/R2)")
struct SoAPageAlignTests {
    private func makeCandidates(_ n: Int) -> [Vector512Optimized] {
        (0..<n).map { k in try! Vector512Optimized((0..<512).map { Float(($0 + k) % 13) }) }
    }

    @Test("Page-aligned SoA exposes a page-aligned, page-length pointer")
    func pageAlignedExposesBytes() {
        let page = PlatformConfiguration.pageSize
        let soa = SoA512.build(from: makeCandidates(300), pageAligned: true)
        let bytes = soa.pageAlignedBytes
        #expect(bytes != nil)
        if let (base, byteCount) = bytes {
            #expect(Int(bitPattern: base) % page == 0, "base must be page-aligned")
            #expect(byteCount % page == 0, "length must be a page multiple")
            #expect(byteCount >= 300 * 128 * 16)   // count * lanes * sizeof(SIMD4<Float>)
        }
    }

    @Test("Non-page-aligned SoA returns nil for pageAlignedBytes")
    func defaultReturnsNil() {
        let soa = SoA512.build(from: makeCandidates(300))   // pageAligned defaults false
        #expect(soa.pageAlignedBytes == nil)
    }

    @Test("Page alignment preserves the SoA data layout")
    func dataIntegrity() {
        let candidates = makeCandidates(64)
        let aligned = SoA512.build(from: candidates, pageAligned: true)
        let plain = SoA512.build(from: candidates, pageAligned: false)
        for lane in 0..<aligned.lanes {
            let a = aligned.lanePointer(lane)
            let p = plain.lanePointer(lane)
            for j in 0..<64 { #expect(a[j] == p[j], "lane \(lane) candidate \(j)") }
        }
    }

    @Test("withUnsafeRawBuffer exposes the logical data length")
    func rawBufferLength() {
        let soa = SoA512.build(from: makeCandidates(10), pageAligned: true)
        soa.withUnsafeRawBuffer { base, byteCount in
            #expect(byteCount == 10 * 128 * 16)
            #expect(Int(bitPattern: base) % 16 == 0)
        }
    }

    @Test("Empty page-aligned SoA has no bytesNoCopy pointer")
    func emptyAligned() {
        let soa = SoA512.build(from: [], pageAligned: true)
        #expect(soa.pageAlignedBytes == nil)
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `swift test --filter SoAPageAlignTests`
Expected: FAIL to compile — `extra argument 'pageAligned'`, `value of type 'SoA<…>' has no member 'pageAlignedBytes'` / `withUnsafeRawBuffer`.

- [ ] **Step 3: Add stored properties and the allocation helper to `SoA`**

In `Sources/VectorCore/Storage/SoA.swift`, add two stored properties immediately after `private let bufferCapacity: Int` (line 58):

```swift
    /// True when `buffer` was allocated via AlignedMemory (posix_memalign) and so must
    /// be freed with free(); false for the default Swift allocation.
    private let ownsViaAlignedMemory: Bool

    /// Total allocated bytes. Page-rounded when page-aligned (the length to pass to
    /// makeBuffer(bytesNoCopy:)); otherwise bufferCapacity * sizeof(SIMD4<Float>).
    private let allocatedByteCount: Int
```

Add this static helper (place it just above `deinit`):

```swift
    /// Allocate (and zero) the SoA buffer. When `pageAligned` and non-empty, uses
    /// AlignedMemory (page-aligned base + page-rounded length) so the region is valid
    /// for MTLDevice.makeBuffer(bytesNoCopy:); otherwise the default 16-byte allocation.
    private static func allocateBuffer(
        capacity: Int, pageAligned: Bool
    ) -> (buffer: UnsafeMutablePointer<SIMD4<Float>>, ownsViaAligned: Bool, allocatedBytes: Int) {
        let elemSize = MemoryLayout<SIMD4<Float>>.stride   // 16
        if pageAligned && capacity > 0 {
            let page = PlatformConfiguration.pageSize
            let logical = capacity * elemSize
            let padded = ((logical + page - 1) / page) * page   // round up to a whole page
            let paddedCapacity = padded / elemSize
            let ptr: UnsafeMutablePointer<SIMD4<Float>>
            do {
                ptr = try AlignedMemory.allocateAligned(type: SIMD4<Float>.self,
                                                        count: paddedCapacity, alignment: page)
            } catch {
                fatalError("SoA: page-aligned allocation of \(padded) bytes failed: \(error)")
            }
            ptr.initialize(repeating: .zero, count: paddedCapacity)   // zero logical + padding
            return (ptr, true, padded)
        } else {
            let ptr = UnsafeMutablePointer<SIMD4<Float>>.allocate(capacity: max(0, capacity))
            if capacity > 0 { ptr.initialize(repeating: .zero, count: capacity) }
            return (ptr, false, max(0, capacity) * elemSize)
        }
    }
```

- [ ] **Step 4: Route both initializers through the helper and add `pageAligned`**

Replace the `private init(count:)` (lines 60–74) with:

```swift
    private init(count: Int, pageAligned: Bool = false) {
        self.count = count
        self.lanes = Vector.lanes
        self.bufferCapacity = self.lanes * self.count
        let alloc = SoA.allocateBuffer(capacity: self.bufferCapacity, pageAligned: pageAligned)
        self.buffer = alloc.buffer
        self.ownsViaAlignedMemory = alloc.ownsViaAligned
        self.allocatedByteCount = alloc.allocatedBytes
    }
```

Replace the allocation block of `public init(vectors:blockSize:)` (lines 88–99) so the signature and allocation become:

```swift
    public init(vectors: [Vector], blockSize: Int = 32, pageAligned: Bool = false) {
        let count = vectors.count
        self.count = count
        self.lanes = Vector.lanes
        self.bufferCapacity = self.lanes * self.count
        let alloc = SoA.allocateBuffer(capacity: self.bufferCapacity, pageAligned: pageAligned)
        self.buffer = alloc.buffer
        self.ownsViaAlignedMemory = alloc.ownsViaAligned
        self.allocatedByteCount = alloc.allocatedBytes

        // Populate the SoA structure from vectors
        guard count > 0 else { return }
        let lanes = Vector.lanes
        let bufferPtr = self.buffer
        for j in 0..<count {
            let vectorStorage = vectors[j].storage
            for i in 0..<lanes {
                bufferPtr[i * count + j] = vectorStorage[i]
            }
        }
    }
```

(Leave the `blockSize` doc comment as-is; only the allocation lines and signature change.)

- [ ] **Step 5: Make `deinit` free with the matching allocator**

Replace `deinit` (lines 76–81) with:

```swift
    deinit {
        if ownsViaAlignedMemory {
            // posix_memalign memory MUST be freed with free(), never .deallocate().
            AlignedMemory.deallocate(buffer)
        } else {
            if bufferCapacity > 0 { buffer.deinitialize(count: bufferCapacity) }
            buffer.deallocate()
        }
    }
```

- [ ] **Step 6: Add `pageAligned` to `build(from:)`**

Replace the `static func build(from:)` signature + `SoA<Vector>(count:)` call (lines 122–124) so they read:

```swift
    public static func build(from candidates: [Vector], pageAligned: Bool = false) -> SoA<Vector> {
        let N = candidates.count
        let soa = SoA<Vector>(count: N, pageAligned: pageAligned)
```

(The rest of `build` is unchanged.)

- [ ] **Step 7: Add the public accessors (R2)**

Add these two public members (place after `lanePointer(_:)`, around line 159):

```swift
    /// Page-aligned base pointer + page-rounded byte length for zero-copy GPU import
    /// via `MTLDevice.makeBuffer(bytesNoCopy:)`, or `nil` if this SoA was not built
    /// page-aligned (`build(from:pageAligned: true)` / `init(..., pageAligned: true)`).
    ///
    /// - Important: The `SoA` MUST outlive any `MTLBuffer` created from this pointer.
    ///   The memory is freed on `SoA` deinit, so hold a strong reference to the `SoA`
    ///   for the buffer's lifetime (or arrange a deallocator handshake).
    public var pageAlignedBytes: (base: UnsafeRawPointer, byteCount: Int)? {
        guard ownsViaAlignedMemory else { return nil }
        return (UnsafeRawPointer(buffer), allocatedByteCount)
    }

    /// Scoped read-only access to the raw SoA data (logical bytes = `count * lanes * 16`).
    /// The pointer is valid only for the duration of `body`.
    public func withUnsafeRawBuffer<R>(_ body: (UnsafeRawPointer, Int) throws -> R) rethrows -> R {
        try body(UnsafeRawPointer(buffer), bufferCapacity * MemoryLayout<SIMD4<Float>>.stride)
    }
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `swift test --filter SoAPageAlignTests`
Expected: PASS — 5 tests.

- [ ] **Step 9: Run the existing SoA suites to confirm no regression**

Run: `swift test --filter "SoA Memory Layout" --filter BatchKernels_SoATests`
Expected: PASS (the default-path allocation/deinit behavior is unchanged).

- [ ] **Step 10: Commit**

```bash
git add Sources/VectorCore/Storage/SoA.swift Tests/ComprehensiveTests/SoAPageAlignTests.swift
git commit -m "feat(soa): opt-in page-aligned buffer + bytesNoCopy accessors (R1/R2)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `BatchKernelProvider` protocol (R4, part 1)

**Files:**
- Create: `Sources/VectorCore/Protocols/BatchKernelProvider.swift`
- Test: `Tests/ComprehensiveTests/BatchKernelProviderTests.swift`

- [ ] **Step 1: Write the failing conformance test**

Create `Tests/ComprehensiveTests/BatchKernelProviderTests.swift`:

```swift
//
//  BatchKernelProviderTests.swift
//  VectorCore
//
//  R4: a ComputeProvider sub-protocol that supplies real batch kernels, so an
//  installed GPU provider transparently services Operations.findNearest.
//

import Testing
import Foundation
@testable import VectorCore

/// A mock GPU provider that returns an impossible-from-CPU sentinel, so tests can
/// detect that dispatch was delegated to it.
private struct MockGPUProvider: BatchKernelProvider {
    let device: ComputeDevice = .cpu
    var maxConcurrency: Int { 1 }
    var deviceInfo: ComputeDeviceInfo {
        ComputeDeviceInfo(name: "mock-gpu", availableMemory: nil, maxThreads: 1, preferredChunkSize: 1)
    }
    func execute<T: Sendable>(_ work: @Sendable @escaping () async throws -> T) async throws -> T {
        try await work()
    }
    func batchDistance<V: VectorProtocol>(query: V, candidates: [V], metric: any DistanceMetric)
        async throws -> [Float] where V.Scalar == Float {
        Array(repeating: Float(-1), count: candidates.count)
    }
    func findNearest<V: VectorProtocol>(query: V, candidates: [V], k: Int, metric: any DistanceMetric)
        async throws -> [(index: Int, distance: Float)] where V.Scalar == Float {
        [(index: 999, distance: -42)]   // impossible from the CPU path
    }
}

@Suite("BatchKernelProvider GPU dispatch (R4)")
struct BatchKernelProviderTests {

    @Test("A BatchKernelProvider is usable as a ComputeProvider")
    func conformsToComputeProvider() async throws {
        let p: any ComputeProvider = MockGPUProvider()
        let mapped = try await p.parallelExecute(items: 0..<4) { $0 * 2 }
        #expect(mapped == [0, 2, 4, 6])   // inherits the ComputeProvider default
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `swift test --filter BatchKernelProviderTests`
Expected: FAIL to compile — `cannot find type 'BatchKernelProvider' in scope`.

- [ ] **Step 3: Define the protocol**

Create `Sources/VectorCore/Protocols/BatchKernelProvider.swift`:

```swift
// VectorCore: Batch Kernel Provider Protocol
//
// beta-evolution-4, DOCUMENT-5 (R4). A ComputeProvider that can substitute real
// batch kernels (e.g. Metal) for VectorCore's CPU kernels. `Operations` downcasts
// the installed `computeProvider` to this protocol and, when present, delegates —
// so GPU acceleration becomes transparent through VectorCore's own findNearest.
//
// VectorCore defines the protocol; the GPU-backed conformance lives in VectorAccelerate.

import Foundation

/// A `ComputeProvider` that supplies hardware batch kernels rather than only a
/// scheduling strategy. Conformers compute the kernel themselves (CPU SIMD, Metal,
/// etc.) instead of running an opaque closure.
///
/// Semantics a conformer must honor (see DOCUMENT-4 S4 conformance contract):
/// - `batchDistance` returns one distance per candidate, in candidate order.
/// - `findNearest` returns up to `k` `(index, distance)` pairs sorted ascending by
///   distance, indexing into `candidates`; results match the CPU reference within a
///   documented tolerance.
public protocol BatchKernelProvider: ComputeProvider {
    /// Distance from `query` to each candidate, in candidate order.
    func batchDistance<V: VectorProtocol>(
        query: V, candidates: [V], metric: any DistanceMetric
    ) async throws -> [Float] where V.Scalar == Float

    /// Up to `k` nearest candidates, sorted ascending by distance.
    func findNearest<V: VectorProtocol>(
        query: V, candidates: [V], k: Int, metric: any DistanceMetric
    ) async throws -> [(index: Int, distance: Float)] where V.Scalar == Float
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `swift test --filter BatchKernelProviderTests`
Expected: PASS — 1 test.

- [ ] **Step 5: Commit**

```bash
git add Sources/VectorCore/Protocols/BatchKernelProvider.swift Tests/ComprehensiveTests/BatchKernelProviderTests.swift
git commit -m "feat(providers): add BatchKernelProvider protocol (R4)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Wire transparent GPU dispatch into `Operations` (R4, part 2)

**Files:**
- Modify: `Sources/VectorCore/Operations/Operations.swift`
- Test: `Tests/ComprehensiveTests/BatchKernelProviderTests.swift` (extend)

- [ ] **Step 1: Add the delegation tests**

Append these tests inside `struct BatchKernelProviderTests` (before its closing brace):

```swift
    @Test("Operations.findNearest delegates to an installed BatchKernelProvider")
    func findNearestDelegates() async throws {
        let q = try Vector512Optimized(Array(repeating: 1, count: 512))
        let cs = (0..<10).map { _ in try! Vector512Optimized(Array(repeating: 0, count: 512)) }
        let result = try await Operations.$computeProvider.withValue(MockGPUProvider()) {
            try await Operations.findNearest(to: q, in: cs, k: 3)
        }
        #expect(result.count == 1)
        #expect(result[0].index == 999)        // the sentinel ⇒ delegation happened
        #expect(result[0].distance == -42)
    }

    @Test("findNearestBatch delegates per query (GPU precedence over CPU GEMM)")
    func findNearestBatchDelegates() async throws {
        // n = 300 ≥ 256 would normally hit the CPU GEMM path; the GPU provider must win.
        let qs = (0..<5).map { _ in try! Vector512Optimized(Array(repeating: 1, count: 512)) }
        let cs = (0..<300).map { _ in try! Vector512Optimized(Array(repeating: 0, count: 512)) }
        let result = try await Operations.$computeProvider.withValue(MockGPUProvider()) {
            try await Operations.findNearestBatch(queries: qs, in: cs, k: 3)
        }
        #expect(result.count == 5)
        for row in result { #expect(row.first?.index == 999) }
    }

    @Test("Default provider uses the CPU path (no sentinel)")
    func defaultUsesCPU() async throws {
        let q = try Vector512Optimized((0..<512).map { Float($0) })
        let cs = (0..<10).map { k in try! Vector512Optimized((0..<512).map { Float($0 + k) }) }
        let result = try await Operations.findNearest(to: q, in: cs, k: 3)
        #expect(result.count == 3)
        #expect(result[0].index != 999)        // real CPU result, not the sentinel
    }
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `swift test --filter BatchKernelProviderTests`
Expected: `findNearestDelegates` / `findNearestBatchDelegates` FAIL (CPU path runs; index is not 999). `defaultUsesCPU` and `conformsToComputeProvider` still pass.

- [ ] **Step 3: Downcast in `Operations.findNearest`**

In `Sources/VectorCore/Operations/Operations.swift`, in `findNearest` (line 46), insert the downcast immediately after the input-validation block (after the `for v in vectors where v.scalarCount != expectedDim { … }` loop, i.e. after line 66, before the `// Optimized Top‑K fast paths` comment on line 68):

```swift
        // R4 — transparent GPU dispatch: an installed BatchKernelProvider supplies real
        // batch kernels; delegate to it (precedence over the CPU fast paths below).
        if let gpu = computeProvider as? BatchKernelProvider {
            let pairs = try await gpu.findNearest(query: query, candidates: vectors, k: k, metric: metric)
            return pairs.map { NearestNeighborResult(index: $0.index, distance: $0.distance) }
        }
```

- [ ] **Step 4: Downcast in `Operations.findNearestBatch`**

In `findNearestBatch` (line 125), insert this immediately before the `// GEMM fast path (DOCUMENT-2)` block (i.e. right after the dimension-validation loops and before `if let routed = gemmFindNearestBatch(...)`):

```swift
        // R4 — transparent GPU dispatch: if a GPU kernel provider is installed, dispatch
        // each query through it (its findNearest); this takes precedence over CPU GEMM.
        if computeProvider as? BatchKernelProvider != nil {
            return try await computeProvider.parallelExecute(items: 0..<queries.count) { i in
                try await findNearest(to: queries[i], in: vectors, k: k, metric: metric)
            }
        }
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `swift test --filter BatchKernelProviderTests`
Expected: PASS — 4 tests (`conformsToComputeProvider`, `findNearestDelegates`, `findNearestBatchDelegates`, `defaultUsesCPU`).

- [ ] **Step 6: Run the GEMM routing/batch suites to confirm the default path still works**

Run: `swift test --filter BatchKNNGEMMTests --filter MatrixDistanceTests`
Expected: PASS (no provider installed ⇒ `as? BatchKernelProvider` is nil ⇒ unchanged behavior).

- [ ] **Step 7: Commit**

```bash
git add Sources/VectorCore/Operations/Operations.swift Tests/ComprehensiveTests/BatchKernelProviderTests.swift
git commit -m "feat(operations): downcast computeProvider to BatchKernelProvider for transparent GPU dispatch (R4)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: R3 reply + status update + full-suite verification

**Files:**
- Modify: `Docs/beta-evolution-4/DOCUMENT-5_VectorAccelerate_Integration_Requests.md`

- [ ] **Step 1: Mark R1/R2/R4 done and record the R3 reply**

In `DOCUMENT-5_VectorAccelerate_Integration_Requests.md`, change the recommendation table's status column to reflect completion, and append this section at the end:

```markdown
## Status (implemented on feature/beta-evo-4)

- **R1 — page-align SoA:** ✅ `SoA.build(from:pageAligned:)` / `init(vectors:…:pageAligned:)`
  allocate via `AlignedMemory` (page-aligned base, page-rounded length, allocator-correct deinit).
- **R2 — public accessor:** ✅ `SoA.pageAlignedBytes` (bytesNoCopy base+length, lifetime contract
  documented) and `withUnsafeRawBuffer { (ptr, byteCount) in … }`.
- **R4 — BatchKernelProvider:** ✅ protocol added; `Operations.findNearest` / `findNearestBatch`
  downcast the installed `computeProvider` and delegate (precedence over CPU GEMM).
- **R3 — version reply (sent):** page alignment was never on `AlignedMemory`/`AlignedDynamicArrayStorage`
  (those are 64-byte). Page-aligned storage (`PageAlignedBuffer`, and now opt-in `SoA`) ships in the
  **0.3.0** release that includes beta-evo-4 — **pin your floor to 0.3.0**, not 0.2.2.
```

- [ ] **Step 2: Run the full correctness suite**

Run: `VECTORCORE_TEST_EXTENDED=0 swift test`
Expected: PASS — all tests, 0 issues. (Use `VECTORCORE_TEST_EXTENDED=0`: the extended wall-clock `PerformanceComparisonTests` are load-flaky and excluded from the correctness gate, matching CI.)

- [ ] **Step 3: Commit**

```bash
git add Docs/beta-evolution-4/DOCUMENT-5_VectorAccelerate_Integration_Requests.md
git commit -m "docs(beta-evo-4): mark VA integration R1/R2/R4 done + R3 version reply

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Verification (end-to-end)

1. `swift test --filter SoAPageAlignTests` → 5 pass (R1/R2).
2. `swift test --filter BatchKernelProviderTests` → 4 pass (R4).
3. `VECTORCORE_TEST_EXTENDED=0 swift test` → full suite green, 0 issues.
4. Manual contract check (optional): a page-aligned `SoA512.build(from: …, pageAligned: true)` returns `pageAlignedBytes` with `base % pageSize == 0` and `byteCount % pageSize == 0` — the two invariants `makeBuffer(bytesNoCopy:)` requires.

## Notes / non-goals

- Default `SoA` allocation stays 16-byte (no memory regression); page alignment is strictly opt-in for the GPU-bridge path.
- This plan does not implement a GPU `BatchKernelProvider` conformance — that lives in VectorAccelerate (`MetalComputeProvider` is "a thin adapter away" per their doc). Core only owns the protocol + the downcast.
- `batchDistance` is defined on the protocol for VA's use; Core wires the downcast into the k-NN paths (`findNearest`/`findNearestBatch`). Routing a Core entry point to `batchDistance` is deferred (no public Core "distance vector" entry beyond `BatchOperations`, which already has its own path).
