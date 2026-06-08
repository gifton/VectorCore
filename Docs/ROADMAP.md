# VectorCore Improvement Roadmap

**Current Version:** 0.2.2 → **0.3.0 in progress** (releasing 2026-06-07)
**Last Updated:** 2026-06-07
**Target Packages:** VectorIndex, VectorAccelerate, VectorIndexAccelerated, EmbedKit

> **Strategy note (0.3.0):** "Hold the line, sharpen the seams." Several items previously
> listed here as open Core backlog have either **shipped in 0.3.0** (see the "0.3.0 — Done"
> section immediately below) or been **redirected to a sibling package** (VectorIndex /
> VectorAccelerate). The authoritative 0.3.0 record of work is
> `Docs/beta-evolution-4/be4-master.md` (+ `CHANGELOG`).

---

## ✅ 0.3.0 — Done (shipped on `feature/beta-evo-4`)

The following items from the backlog below are **shipped in 0.3.0**. They are marked inline at
their original locations as well; collected here for quick reference:

- **Matrix / GEMM batch distance (CPU/AMX)** — §9.2 "Matrix Distance Computation" and §2.2
  "Add `distanceMatrix(a:b:)`". Shipped as `MatrixDistance`
  (`Sources/VectorCore/Operations/MatrixDistance.swift`): `euclideanSquaredMatrix` /
  `cosineDistanceMatrix` (with `into:` / allocating / `prepared:` overloads),
  `prepare(_:normalized:)`, `PreparedCandidates`. Uses Accelerate `cblas_sgemm` → AMX.
- **Configurable tie-breaking** — §9.3. Shipped as `TieBreaker`
  (`.smallerIndex` default / `.insertionOrder` / `.smallerValue`) in
  `Sources/VectorCore/Operations/TopKSelection.swift`.
- **Batch-distance protocol seam + provider seam** — §2.2 / §11.2. Shipped as
  `BatchKernelProvider` (a `ComputeProvider` sub-protocol) in
  `Sources/VectorCore/Protocols/BatchKernelProvider.swift`; enables transparent GPU dispatch
  from `Operations.findNearest` / `findNearestBatch`.
- **Zero-copy buffer contract + shared CPU↔GPU types** — §11.2 (ComputeProvider / Metal-buffer
  compatibility / shared CPU↔GPU types). Shipped as the `UnifiedVectorBuffer` protocol +
  `PageAlignedBuffer` (`Sources/VectorCore/Storage/UnifiedVectorBuffer.swift`) and the **frozen
  `SoALayout` contract** (`Sources/VectorCore/Storage/SoALayout.swift`,
  `Docs/SoA_Layout_Contract.md`), plus SoA page-alignment APIs in
  `Sources/VectorCore/Storage/SoA.swift` (`build(from:pageAligned:)`,
  `init(vectors:pageAligned:)`, `pageAlignedBytes`, `consumeAllocation()`,
  `withUnsafeRawBuffer`) and `PlatformConfiguration.roundUpToPage(_:)`.
- **Transparent matrix routing** — `BatchOperations.Configuration.enableMatrixRouting`
  (default `true`) + `matrixRoutingMinN` (default `256`) route `pairwiseDistances` and
  `Operations.findNearestBatch` through `MatrixDistance`.

> **⚠️ BREAKING (0.3.0):** the no-op `blockSize:` parameter was removed from
> `SoA.init(vectors:…)` **and** `SoAFP16.init(vectors:…)`.

**Redirected to sibling packages (not open Core backlog):** `mmap` storage → VectorIndex (§10.2),
prefetching → VectorIndex (§4.4), PQ / ADC → VectorIndex (§5.1), binary-packed + Hamming →
VectorAccelerate (§5.2), sparse CSR → deferred (no producer). FP16 mixed precision and scalar
INT8 quantization already shipped in Core pre-0.3.0.

This document tracks identified improvement opportunities for VectorCore, organized by component and priority. Items marked with 🔒 may require API changes.

---

## Table of Contents

0. [🚨 CRITICAL: Missing Dimension Support for EmbedKit](#-critical-missing-dimension-support-for-embedkit)
1. [Storage Layer](#1-storage-layer)
2. [Protocol Design](#2-protocol-design)
3. [Vector Types](#3-vector-types)
4. [Kernel Optimizations](#4-kernel-optimizations)
5. [Quantization System](#5-quantization-system)
6. [Factory & Construction](#6-factory--construction)
7. [Error Handling](#7-error-handling)
8. [Distance Metrics](#8-distance-metrics)
9. [Batch Operations](#9-batch-operations)
10. [Memory Management](#10-memory-management)
11. [Cross-Package Integration](#11-cross-package-integration)
12. [Testing & Benchmarking](#12-testing--benchmarking)
13. [Documentation](#13-documentation)

---

## ✅ RESOLVED: Dimension 384 Support Added

### Issue: 384-Dimensional Vectors Fall Back to DynamicVector

**Priority:** P0 - Critical → **IMPLEMENTED**
**Impact:** Performance (EmbedKit integration), API Surface
**Discovered:** Analysis of EmbedKit `ModelDiscovery.swift` and `LocalCoreMLModel.swift`
**Resolution Date:** November 2025

EmbedKit's default embedding dimension is **384** (MiniLM models), but VectorCore only has optimized types for 128, 256, 512, 768, 1536, and 3072. This means:

1. **All MiniLM embeddings use `DynamicVector`** instead of optimized SIMD storage
2. **No SoA batch kernels** for 384-dimensional vectors
3. **No BatchKernels.range_* functions** for 384

#### EmbedKit Dimension Usage (from `ModelDiscovery.swift`)

| Model Family | Common Dimensions | VectorCore Status |
|--------------|-------------------|-------------------|
| **MiniLM** | **384**, 256, 128 | ❌ 384 missing |
| **SBERT** | **384**, 768, 512 | ❌ 384 missing |
| **BERT** | 768, 512 | ✅ Supported |
| **MPNet** | 768 | ✅ Supported |
| **E5** | 768, 1024 | ❌ 1024 missing |

#### SIMD Alignment Analysis

384 dimensions align perfectly for SIMD optimization:
- **SIMD4 lanes:** 96 (384 ÷ 4)
- **SIMD8 lanes:** 48 (384 ÷ 8)
- **Cache lines:** 24 (384 × 4 bytes ÷ 64)

#### Required Changes (All Completed ✅)

**New Files:**
- [x] `Sources/VectorCore/Vectors/Vector384Optimized.swift`
- [x] `Sources/VectorCore/Storage/Dim384Storage.swift` (if needed) - Not needed, uses DimensionStorage

**Updates:**
- [x] `Sources/VectorCore/Core/Dimension.swift` - Add `Dim384`
- [x] `Sources/VectorCore/Storage/SoA.swift` - Add `Vector384Optimized: SoACompatible`
- [x] `Sources/VectorCore/Operations/Kernels/BatchKernels.swift` - Add 384 variants
- [x] `Sources/VectorCore/Operations/Kernels/DotKernels.swift` - Add `dot384`
- [x] `Sources/VectorCore/Operations/Kernels/EuclideanKernels.swift` - Add 384 variants
- [x] `Sources/VectorCore/Operations/Kernels/CosineKernels.swift` - Add 384 variants
- [x] `Sources/VectorCore/Operations/Kernels/NormalizeKernels.swift` - Add 384 variants
- [x] `Sources/VectorCore/Factory/VectorTypeFactory.swift` - Add 384 case

**API Addition (non-breaking):**
```swift
public enum Dim384: StaticDimension {
    public static let value = 384
}

public typealias Vector384Optimized = Vector<Dim384>
public typealias SoA384 = SoA<Vector384Optimized>
```

#### Secondary: 1024-Dimensional Support

E5-large models use 1024 dimensions. Lower priority than 384 but worth considering:
- **SIMD4 lanes:** 256
- **SIMD8 lanes:** 128
- **Cache lines:** 64

---

## 1. Storage Layer

### 1.1 SoA Container Thread Safety

**File:** `Sources/VectorCore/Storage/SoA.swift:47`
**Priority:** Medium
**Impact:** Correctness, Concurrency

The `SoA<Vector>` class uses `@unchecked Sendable` with manually managed `UnsafeMutablePointer<SIMD4<Float>>`.

```swift
public final class SoA<Vector: SoACompatible>: @unchecked Sendable {
    @usableFromInline internal let buffer: UnsafeMutablePointer<SIMD4<Float>>
```

**Issue:** While the current implementation is safe because the buffer is only written during initialization, the `@unchecked` annotation suppresses compiler verification.

**Recommendations:**
- [ ] Consider using `ManagedBuffer` or `ManagedBufferPointer` for automatic memory management
- [ ] Add explicit documentation about thread-safety guarantees
- [ ] Consider making `SoA` immutable after construction (current design intent)
- [ ] Evaluate `UnsafeRawBufferPointer` for read-only access in concurrent scenarios

### 1.2 Aligned Memory Allocation Strategy

**File:** `Sources/VectorCore/Storage/AlignedMemory.swift`
**Priority:** Low
**Impact:** Performance, Platform Compatibility

Current implementation uses `posix_memalign` which is POSIX-specific.

**Recommendations:**
- [ ] Add Windows support via `_aligned_malloc` when platform expands beyond Apple
- [ ] Consider Swift's `UnsafeMutableRawPointer.allocate(byteCount:alignment:)` for simpler code
- [ ] Benchmark whether 64-byte vs 128-byte alignment matters for Apple Silicon

### 1.3 HybridStorage Threshold Tuning

**File:** `Sources/VectorCore/Storage/HybridStorage.swift`
**Priority:** Low
**Impact:** Performance

HybridStorage selects between optimized and dynamic paths but threshold selection could be data-driven.

**Recommendations:**
- [ ] Add runtime profiling to determine optimal thresholds per dimension
- [ ] Consider lazy migration between storage strategies
- [ ] Document when to use hybrid vs specialized storage

### 1.4 COW Implementation Verification

**File:** `Sources/VectorCore/Storage/COWDynamicStorage.swift`
**Priority:** Medium
**Impact:** Performance, Memory

Verify Copy-on-Write semantics under concurrent access patterns.

**Recommendations:**
- [ ] Add stress tests for COW under concurrent modifications
- [ ] Profile memory allocation patterns in hot loops
- [ ] Consider memory pooling for frequently mutated vectors

---

## 2. Protocol Design

### 2.1 ✅ VectorProtocol Normalized Return Type

**File:** `Sources/VectorCore/Protocols/VectorProtocol.swift`
**Priority:** High → **IMPLEMENTED**
**Impact:** API Ergonomics, Performance
**Resolution Date:** November 2025

The `normalized()` method returns `Result<Self, VectorError>` which adds complexity in hot paths.

**Solution Implemented:**
Added `normalizedUnchecked() -> Self` method that:
- Bypasses zero-vector validation for maximum performance
- Uses debug-only assertions to catch misuse during development
- Implemented with optimized SIMD kernels for Vector384/512/768/1536Optimized
- DynamicVector and generic vectors use the default VectorProtocol extension

**Files Modified:**
- `Sources/VectorCore/Protocols/VectorProtocol.swift` - Added default extension
- `Sources/VectorCore/Operations/Kernels/NormalizeKernels.swift` - Added unchecked variants
- `Sources/VectorCore/Vectors/Vector384Optimized.swift` - Override with kernel
- `Sources/VectorCore/Vectors/Vector512Optimized.swift` - Override with kernel
- `Sources/VectorCore/Vectors/Vector768Optimized.swift` - Override with kernel
- `Sources/VectorCore/Vectors/Vector1536Optimized.swift` - Override with kernel

**Remaining Items:**
- [ ] Consider adding `isNormalized` cached flag for pre-normalized vectors
- [ ] Document performance characteristics of Result vs throwing variants

### 2.2 🔒 DistanceMetric Batch Requirements

**File:** `Sources/VectorCore/Protocols/DistanceMetric.swift`
**Priority:** Medium → **P1 (VectorAccelerate request)**
**Impact:** API Completeness, GPU Acceleration

The `DistanceMetric` protocol doesn't require batch operations, so implementations may not optimize for batch scenarios.

```swift
public protocol DistanceMetric: Sendable {
    func distance<V: VectorProtocol>(_ a: V, _ b: V) -> Float where V.Scalar == Float
}
```

**Recommendations:**
- [x] Add optional `batchDistance(query:candidates:)` with default implementation ✅ (v0.2.0 — promoted to protocol requirement)
- [x] Add `distanceMatrix(a:b:)` for cross-product computations ✅ **shipped 0.3.0** as `MatrixDistance.euclideanSquaredMatrix` / `cosineDistanceMatrix` (`Sources/VectorCore/Operations/MatrixDistance.swift`, CPU `cblas_sgemm`→AMX)
- [ ] Document when batch vs single operations should be used

**VectorAccelerate Request:**
This enables VectorAccelerate to provide GPU-optimized batch distance implementations that VectorCore can dispatch to. Without protocol-based batch operations, VectorAccelerate must wrap static functions which prevents proper polymorphism.

```swift
// Proposed addition (~30 lines)
public protocol DistanceMetric {
    func distance<V>(_ a: V, _ b: V) -> Float where V.Scalar == Float

    // NEW - optional with default sequential implementation
    func batchDistance<V>(from query: V, to candidates: [V]) -> [Float]
}

extension DistanceMetric {
    func batchDistance<V>(from query: V, to candidates: [V]) -> [Float] {
        candidates.map { distance(query, $0) }
    }
}
```

### 2.3 Generic Scalar Type Support

**File:** `Sources/VectorCore/Protocols/CoreProtocols.swift`
**Priority:** Low
**Impact:** Flexibility

Currently hardcoded to `Float` in many places. `Double` support exists but is incomplete.

**Recommendations:**
- [ ] Audit all kernels for Double variants
- [ ] Consider `Float16` support for memory-constrained scenarios
- [ ] Add benchmark comparing Float vs Double precision impact on similarity search

### 2.4 SIMDProvider Protocol Completeness

**File:** `Sources/VectorCore/Protocols/ProviderProtocols.swift`
**Priority:** Low
**Impact:** Extensibility

Missing some Accelerate operations that could be useful.

**Recommendations:**
- [ ] Add `fma` (fused multiply-add) operation
- [ ] Add `reciprocalSquareRoot` for faster normalization
- [ ] Add `linearInterpolation` for vector blending
- [ ] Consider `batchDot` for matrix-like operations

---

## 3. Vector Types

### 3.1 Optimized Vector Code Generation

**Files:** `Sources/VectorCore/Vectors/Vector512Optimized.swift`, `Vector768Optimized.swift`, `Vector1536Optimized.swift`
**Priority:** Medium
**Impact:** Maintainability

The three optimized vector types (512, 768, 1536) contain significant code duplication.

**Recommendations:**
- [ ] Create a code generation script to produce dimension-specific implementations
- [ ] Use Swift macros (Swift 5.9+) to reduce boilerplate
- [ ] Consider adding `Vector256Optimized` and `Vector384Optimized` for emerging embedding models
- [ ] Document the templating process for new dimensions

### 3.2 DynamicVector Magnitude Caching

**File:** `Sources/VectorCore/Vectors/DynamicVector.swift`
**Priority:** Medium
**Impact:** Performance

Vectors used in similarity search are often pre-normalized, but magnitude is recomputed each time.

**Recommendations:**
- [ ] Add optional `cachedMagnitude: Float?` property
- [ ] Implement `NormalizedVector` wrapper type that guarantees unit length
- [ ] Add `DynamicVector.normalized(caching: true)` variant

### 3.3 Vector Slicing and Views

**Priority:** Low
**Impact:** API Flexibility

No support for vector slicing or creating views into existing storage.

**Recommendations:**
- [ ] Add `VectorSlice<V>` type for zero-copy views
- [ ] Implement `subscript(range:)` returning a slice
- [ ] Consider implications for memory safety and COW

### 3.4 🔒 Vector Arithmetic Operators

**File:** `Sources/VectorCore/Core/Operators.swift`
**Priority:** Low
**Impact:** API Ergonomics

Operators like `+`, `-`, `*` exist but don't always use optimal kernels.

**Recommendations:**
- [ ] Ensure operator implementations dispatch to SIMD kernels
- [ ] Add compound assignment operators (`+=`, `-=`, `*=`)
- [ ] Profile operator vs explicit method call performance

---

## 4. Kernel Optimizations

### 4.1 4-Way Register Blocking for Large Batches

**File:** `Sources/VectorCore/Operations/Kernels/BatchKernels.swift`
**Priority:** High
**Impact:** Performance

Current implementation uses 2-way register blocking. For large batches, 4-way could improve ILP.

```swift
// Current: processes 2 candidates per iteration
while i + 1 < end {
    let c0 = candidates[i].storage
    let c1 = candidates[i+1].storage
```

**Recommendations:**
- [x] Implement 4-way blocked variants for batches > 64 ✅ (v0.2.0 — euclid2_blocked_4way in BatchKernels_SoA)
- [x] Add adaptive blocking that selects 2-way vs 4-way based on batch size ✅ (v0.2.0 — branches on soa.count >= 4)
- [ ] Benchmark optimal blocking factor on M1/M2/M3 chips
- [ ] Consider 8-way for very large batches (cache permitting)

### 4.2 In-Place Normalization

**File:** `Sources/VectorCore/Operations/Kernels/NormalizeKernels.swift`
**Priority:** Medium
**Impact:** Memory, Performance

All normalization creates new vectors. In-place would reduce allocations.

**Recommendations:**
- [ ] Add `normalizeInPlace(vector: inout Vector)` variants
- [ ] For optimized vectors, modify storage directly
- [ ] Add batch in-place normalization

**VectorIndex Request (Section 9.5 in VectorIndex/IMPROVEMENTS.md):**
- [ ] **Raw buffer normalization** - For mmap/GPU buffer compatibility:
  ```swift
  public static func normalizeUnchecked(
      _ buffer: UnsafeMutablePointer<Float>,
      dimension: Int
  )
  ```

### 4.3 MixedPrecisionKernels Modularization

**File:** `Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift`
**Priority:** Medium
**Impact:** Maintainability, Compile Time

File is extremely large (64k+ tokens) which impacts compile time and maintainability.

**Recommendations:**
- [ ] Split into separate files per dimension (MixedPrecision512.swift, etc.)
- [ ] Extract common patterns into shared utilities
- [ ] Consider using conditional compilation for dimension-specific code
- [ ] Add compilation time benchmarks to CI

### 4.4 Prefetching for Batch Operations → **REDIRECTED to VectorIndex (0.3.0)**

**Priority:** Medium → **Redirected (not Core backlog)**
**Impact:** Performance

Explicit memory prefetching could improve cache utilization in batch loops.

> **0.3.0 decision:** prefetch intrinsics are owned by VectorIndex
> (`VectorIndex/Sources/VectorIndex/Operations/Support/Prefetch.swift`, real builtins). Core's
> `prefetchRead/Write` remain no-ops by design; this is **not** a Core deliverable.

**Recommendations (for the owning package, VectorIndex):**
- [ ] Add `__builtin_prefetch` wrappers via C interop
- [ ] Experiment with prefetch distance (1-3 cache lines ahead)
- [ ] Measure impact on different batch sizes

### 4.5 SIMD8/SIMD16 Investigation

**Priority:** Low
**Impact:** Performance

Current kernels primarily use SIMD4. Larger SIMD widths might benefit some operations.

**Recommendations:**
- [ ] Benchmark SIMD8<Float> for dot product on Apple Silicon
- [ ] Investigate SIMD16<Float> for very high dimensions (4096+)
- [ ] Profile register pressure with larger SIMD widths

---

## 5. Quantization System

### 5.1 Product Quantization (PQ) Support → **REDIRECTED to VectorIndex (0.3.0)**

**File:** `Sources/VectorCore/Quantization/QuantizationSchemes.swift`
**Priority:** High → **Redirected (not Core backlog)**
**Impact:** Memory, Performance (VectorIndex integration)

Currently only scalar INT8 quantization (already shipped in Core). Product quantization is
essential for billion-scale search.

> **0.3.0 decision:** PQ and ADC look-up tables are owned by VectorIndex
> (`VectorIndex/Sources/VectorIndex/Operations/Quantization/{ADCScan,PQLUT}.swift`). PQ/ADC are
> index-topology-coupled and are **not** Core deliverables. Core keeps the topology-agnostic
> scalar INT8 path only.

**Recommendations (for the owning package, VectorIndex):**
- [ ] Implement PQ encoding with configurable subspace count (typically 8-32)
- [ ] Add PQ codebook training (k-means on subspaces)
- [ ] Implement asymmetric distance computation (ADC) for PQ
- [ ] Consider OPQ (Optimized Product Quantization) for better accuracy

### 5.2 Binary Quantization → **REDIRECTED to VectorAccelerate (0.3.0)**

**Priority:** Medium → **Redirected (not Core backlog)**
**Impact:** Memory, Performance

1-bit quantization for extreme memory efficiency.

> **0.3.0 decision:** binary-packed vectors and Hamming/Jaccard are redirected to
> VectorAccelerate (GPU versions already exist there); there is **no committed CPU consumer**, so
> Core does not ship a packed binary type. Confirmed not in Core 0.3.0.

**Recommendations (for the owning package, VectorAccelerate):**
- [ ] Implement sign-based binary quantization
- [ ] Add popcount-based Hamming distance
- [ ] Document accuracy trade-offs

### 5.3 INT4 Quantization

**Priority:** Medium
**Impact:** Memory

4-bit quantization as middle ground between INT8 and binary.

**Recommendations:**
- [ ] Implement symmetric INT4 quantization
- [ ] Handle packing (2 values per byte)
- [ ] Add specialized INT4 distance kernels

### 5.4 Per-Dimension Calibration API

**File:** `Sources/VectorCore/Quantization/QuantizationSchemes.swift:76`
**Priority:** Medium
**Impact:** Accuracy

Per-dimension calibration is stubbed with `fatalError`.

```swift
case .perDimension:
    fatalError(".perDimension calibration requires a dataset-level API")
```

**Recommendations:**
- [ ] Design `QuantizationCalibrator` that operates on vector collections
- [ ] Implement percentile-based range estimation
- [ ] Add outlier handling strategies

---

## 6. Factory & Construction

### 6.1 Protocol-Based Dimension Dispatch

**File:** `Sources/VectorCore/Factory/VectorTypeFactory.swift`
**Priority:** Low
**Impact:** Maintainability

Large switch statements for dimension routing.

```swift
switch dimension {
case 128: return try Vector<Dim128>(values)
case 256: return try Vector<Dim256>(values)
// ...
```

**Recommendations:**
- [ ] Create `DimensionRegistry` with registered dimension types
- [ ] Allow runtime registration of custom dimensions
- [ ] Consider using protocol witnesses for dispatch

### 6.2 RandomNormalized Factory Method

**File:** `Sources/VectorCore/Factory/VectorTypeFactory.swift:160`
**Priority:** Low
**Impact:** API Completeness

`randomNormalized` is commented out due to existential type issue.

```swift
// TODO: Fix normalized() on existential type
// public static func randomNormalized(dimension: Int) -> any VectorType {
```

**Recommendations:**
- [ ] Investigate using type erasure wrappers
- [ ] Consider generic factory methods instead of existential returns
- [ ] Add dimension-specific randomNormalized methods as workaround

### 6.3 Builder Pattern for Complex Vector Construction

**Priority:** Low
**Impact:** API Ergonomics

No fluent builder for constructing vectors with specific properties.

**Recommendations:**
- [ ] Add `VectorBuilder` for: dimension, values, normalization, storage type
- [ ] Support construction from iterator/sequence
- [ ] Add validation hooks in builder

---

## 7. Error Handling

### 7.1 ErrorContext Performance

**File:** `Sources/VectorCore/Errors/VectorError.swift:37`
**Priority:** Low
**Impact:** Performance

`ErrorContext` captures `Date()` which has non-trivial cost.

```swift
public let timestamp: Date
// ...
self.timestamp = Date()
```

**Recommendations:**
- [ ] Make timestamp capture opt-in or lazy
- [ ] Use monotonic time instead of wall clock for performance analysis
- [ ] Consider removing timestamp in release builds (like source location)

### 7.2 🔒 Consistent Error Handling Pattern

**Priority:** Medium
**Impact:** API Consistency

Some APIs return `Result`, others throw, others return Optional.

**Recommendations:**
- [ ] Audit all public APIs for consistency
- [ ] Document when each pattern is appropriate
- [ ] Consider adding throwing versions with `try` prefix: `tryNormalized()`
- [ ] Add `VectorError` to `Result` pattern guidelines

### 7.3 Error Recovery Suggestions

**Priority:** Low
**Impact:** Developer Experience

Errors don't suggest recovery actions.

**Recommendations:**
- [ ] Add `recoverySuggestion` property to `VectorError`
- [ ] Integrate with SwiftUI/AppKit error presentation
- [ ] Document common error scenarios and fixes

---

## 8. Distance Metrics

### 8.1 ✅ SIMD Manhattan Distance

**File:** `Sources/VectorCore/Operations/DistanceMetrics.swift:162`
**Priority:** Medium → **IMPLEMENTED**
**Impact:** Performance
**Resolution Date:** November 2025

Manhattan distance used scalar loop instead of SIMD.

**Solution Implemented:**
Replaced scalar loop with SIMD4-vectorized implementation:
- Processes 4 elements per iteration using `SIMD4<Float>`
- Uses `abs(a4 - b4)` for vectorized absolute difference
- Includes scalar tail loop for dimensions not divisible by 4
- No temporary buffer allocation required

```swift
// New SIMD4 implementation
var acc = SIMD4<Float>.zero
for i in 0..<simdCount {
    let offset = i * 4
    let a4 = SIMD4<Float>(aBuffer[offset], aBuffer[offset+1], ...)
    let b4 = SIMD4<Float>(bBuffer[offset], bBuffer[offset+1], ...)
    acc += abs(a4 - b4)
}
result = acc.sum()
// + scalar remainder loop
```

**Files Modified:**
- `Sources/VectorCore/Operations/DistanceMetrics.swift` - Replaced scalar loop
- `Tests/ComprehensiveTests/VectorDistanceMetricsTests.swift` - Added comprehensive tests

**Note:** Optimized vectors (384/512/768/1536) already had highly optimized SIMD4 implementations in `ManhattanKernels.swift` with 4-way unrolling. This change improves the generic path for `DynamicVector` and arbitrary dimensions.

### 8.2 SIMD Chebyshev Distance

**File:** `Sources/VectorCore/Operations/DistanceMetrics.swift:226`
**Priority:** Low
**Impact:** Performance

Chebyshev uses scalar loop for max finding.

**Recommendations:**
- [ ] Use `vDSP_maxmgv` after computing difference
- [ ] Consider SIMD4 max reduction

### 8.3 Angular Distance

**Priority:** Low
**Impact:** API Completeness

Missing angular distance (arccos of cosine similarity).

**Recommendations:**
- [ ] Add `AngularDistance` metric
- [ ] Document relationship to cosine distance
- [ ] Handle edge cases (acos domain)

### 8.4 Weighted Distance Metrics

**Priority:** Low
**Impact:** Flexibility

No support for dimension-weighted distances.

**Recommendations:**
- [ ] Add `WeightedEuclidean` with per-dimension weights
- [ ] Add `WeightedCosine` for attention-weighted similarity
- [ ] Consider Mahalanobis distance

---

## 9. Batch Operations

### 9.1 Async Batch Operations

**File:** `Sources/VectorCore/Operations/BatchOperations.swift`
**Priority:** High
**Impact:** Concurrency, Performance

No native async support for batch operations.

**Recommendations:**
- [ ] Add `async` variants: `batchDistancesAsync(query:candidates:)`
- [ ] Implement using Swift Concurrency `TaskGroup`
- [ ] Add progress reporting via `AsyncSequence`
- [ ] Support cancellation

### 9.2 Matrix Distance Computation → ✅ **SHIPPED 0.3.0**

**Priority:** Medium → **IMPLEMENTED**
**Impact:** Performance (VectorIndex)

Computing pairwise distances previously required nested loops.

**Solution Implemented (0.3.0):**
Shipped `MatrixDistance` (`Sources/VectorCore/Operations/MatrixDistance.swift`) — a CPU GEMM
batch-distance path using Accelerate `cblas_sgemm` (→ AMX on Apple Silicon):
`euclideanSquaredMatrix` / `cosineDistanceMatrix` with `into:` / allocating / `prepared:`
overloads, `prepare(_:normalized:)`, and `PreparedCandidates`. `BatchOperations` routes
`pairwiseDistances` and `Operations.findNearestBatch` through it via
`Configuration.enableMatrixRouting` (default `true`) above `matrixRoutingMinN` (default `256`).

- [x] Add `distanceMatrix(queries:candidates:)` ✅ (as `euclideanSquaredMatrix` / `cosineDistanceMatrix`)
- [x] Implement using blocked matrix algorithms ✅ (`cblas_sgemm` → AMX)
- [~] Metal acceleration for large matrices — provided via the `BatchKernelProvider` seam (VectorAccelerate hook), not in Core

### 9.3 ✅ Top-K Selection Kernel

**Priority:** High → **IMPLEMENTED**
**Impact:** Performance (VectorIndex integration)
**Resolution Date:** November 2025

**Solution Implemented:**
Created public `TopKSelection` API with:
- `select(k:from:)` - O(n log k) heap-based selection from pre-computed distances
- `nearest(k:query:candidates:metric:)` - Generic k-NN with any distance metric
- `nearestEuclidean384/512/768/1536()` - SIMD-optimized for specific vector types
- `nearestCosinePreNormalized512()` - Optimized for pre-normalized vectors
- `nearestCosineFused512()` - One-pass cosine distance
- `nearestDotProduct512()` - For maximum inner product search
- `batchNearest()` - Multiple queries in batch
- `TopKResult` struct with Collection/Codable conformance

**Algorithm:**
- k < n/10: Uses max-heap for O(n log k) complexity
- k >= n/10: Uses partial sort (better constants)

**Files Created:**
- `Sources/VectorCore/Operations/TopKSelection.swift` - Full public API
- `Tests/ComprehensiveTests/TopKSelectionTests.swift` - 27 tests

**Remaining Items:**
- [ ] Add SIMD-accelerated argmin for K=1
- [ ] Add threshold-based filtering (only keep distances < threshold)

**VectorIndex Requests (Section 9.5 in VectorIndex/IMPROVEMENTS.md):**
- [x] **Pointer-based TopK API** - Accept `UnsafePointer<Float>` for zero-copy from GPU/mmap buffers: ✅ (v0.2.0 — select(k:from:count:ids:) returning [Int32])
  ```swift
  public static func select(
      k: Int,
      from distances: UnsafePointer<Float>,
      count: Int,
      ids: UnsafePointer<Int32>?  // Optional custom ID array
  ) -> (indices: [Int32], distances: [Float])
  ```
- [x] **Configurable tie-breaking** - Deterministic behavior for reproducibility ✅ **shipped 0.3.0** (`TieBreaker` in `Sources/VectorCore/Operations/TopKSelection.swift`; `.smallerIndex` is the default):
  ```swift
  public enum TieBreaker { case smallerIndex /* default */, insertionOrder, smallerValue }
  ```

### 9.4 Batch Normalization

**Priority:** Medium
**Impact:** Performance

Normalizing many vectors sequentially is inefficient.

**Recommendations:**
- [ ] Add `normalizeAll(vectors: inout [Vector])`
- [ ] Implement parallel normalization with GCD
- [ ] Consider memory-mapped I/O for very large sets

---

## 10. Memory Management

### 10.1 Memory Pool for Temporary Allocations

**File:** `Sources/VectorCore/Providers/SwiftBufferPool.swift`
**Priority:** Medium
**Impact:** Performance, Memory

Frequent allocations in hot paths can cause memory pressure.

**Recommendations:**
- [ ] Expand `SwiftBufferPool` usage to all temporary allocations
- [ ] Add per-thread buffer pools to avoid contention
- [ ] Implement pool size limits and eviction
- [ ] Profile allocation patterns in VectorIndex workloads

### 10.2 Memory-Mapped Vector Storage → **REDIRECTED to VectorIndex (0.3.0)**

**Priority:** Medium → **Redirected (not Core backlog)**
**Impact:** Scalability (VectorIndex)

Large vector collections don't fit in RAM.

> **0.3.0 decision:** `mmap` storage is owned by VectorIndex
> (`VectorIndex/Sources/VectorIndex/Kernels/VIndexMmap.swift`, real/WIP). It is persistence- and
> index-coupled, so it is **not** a Core deliverable. Core's contribution is the zero-copy
> page-aligned buffer contract (`PageAlignedBuffer` / frozen `SoALayout`) that an mmap backend
> can build on.

**Recommendations (for the owning package, VectorIndex):**
- [ ] Add `MMapStorage` backend for vectors
- [ ] Implement lazy loading with LRU cache
- [ ] Support append-only growth for indexing
- [ ] Coordinate with VectorIndex for index persistence

### 10.3 Compressed In-Memory Storage

**Priority:** Low
**Impact:** Memory

Vectors could be stored compressed with on-demand decompression.

**Recommendations:**
- [ ] Investigate LZ4 for fast decompression
- [ ] Cache decompressed vectors
- [ ] Benchmark compression ratio vs access latency

---

## 11. Cross-Package Integration

### 11.1 ✅ VectorIndex Integration Points

**Priority:** High → **IMPLEMENTED**
**Impact:** Ecosystem
**Resolution Date:** November 2025

**Solution Implemented:**
Created Integration module with protocols and types for VectorIndex:

**IndexableVector Protocol:**
- Extends `VectorProtocol` with `isNormalized` and `cachedMagnitude` hints
- All optimized vectors and DynamicVector conform
- `NormalizationHint<V>` wrapper for tracking normalization status

**SearchResult Types:**
- `SearchResult<ID>` - Single result with id, distance, score
- `SearchResults<ID>` - Collection with metadata (candidatesSearched, searchTimeNanos, isExhaustive)
- `IntSearchResult`/`StringSearchResult` type aliases
- Conversion from `TopKResult`

**VectorCollection Protocol:**
- `vector(for:)` / `vectors(for:)` - Retrieval
- `search(query:k:metric:)` - Default brute-force implementation
- `searchEuclidean()` / `searchCosine()` / `searchDotProduct()` conveniences
- `MutableVectorCollection` for add/remove/update operations
- `SimpleVectorCollection<V>` reference implementation

**Files Created:**
- `Sources/VectorCore/Integration/IndexableVector.swift`
- `Sources/VectorCore/Integration/SearchResult.swift`
- `Sources/VectorCore/Integration/VectorCollection.swift`
- `Tests/ComprehensiveTests/IntegrationProtocolsTests.swift` - 24 tests

**Design Notes:**
- VectorIndex will override `search()` with ANN algorithms
- VectorCore provides brute-force fallback
- Clear ownership: VectorCore owns vectors, VectorIndex owns indices

### 11.2 VectorAccelerate Metal Hooks → ✅ **SHIPPED 0.3.0**

**Priority:** High → **IMPLEMENTED**
**Impact:** Performance

Integration points for GPU acceleration.

**Solution Implemented (0.3.0):**
- [x] `ComputeProvider` protocol abstraction ✅ — `BatchKernelProvider` (a `ComputeProvider`
  sub-protocol) in `Sources/VectorCore/Protocols/BatchKernelProvider.swift`; enables transparent
  GPU dispatch from `Operations.findNearest` / `findNearestBatch`.
- [x] Define Metal buffer compatibility requirements ✅ — `UnifiedVectorBuffer` protocol +
  `PageAlignedBuffer` (`Sources/VectorCore/Storage/UnifiedVectorBuffer.swift`): page-aligned
  storage valid for `MTLDevice.makeBuffer(bytesNoCopy:)`.
- [x] Create shared types for CPU↔GPU data transfer ✅ — frozen `SoALayout` descriptor + SoA
  layout contract (`Sources/VectorCore/Storage/SoALayout.swift`, `Docs/SoA_Layout_Contract.md`),
  plus SoA page-alignment APIs (`SoA.build(from:pageAligned:)`, `init(vectors:pageAligned:)`,
  `pageAlignedBytes`, `consumeAllocation()`, `withUnsafeRawBuffer`) and
  `PlatformConfiguration.roundUpToPage(_:)`.
- [~] Add fallback mechanism when Metal unavailable — routing falls back to the CPU
  `MatrixDistance` / kernel path when no `BatchKernelProvider` is installed.

### 11.3 EmbedKit Type Compatibility

**Priority:** Medium
**Impact:** Ecosystem

Ensure VectorCore types work seamlessly with embedding models.

**Recommendations:**
- [ ] Add initializer from `MLMultiArray` (CoreML)
- [ ] Support ONNX tensor conversion
- [ ] Add batched conversion utilities
- [ ] Document memory ownership in conversions

---

## 12. Testing & Benchmarking

### 12.1 Numerical Stability Test Suite

**Priority:** High
**Impact:** Correctness

Need comprehensive tests for edge cases.

**Recommendations:**
- [ ] Add tests for: near-zero vectors, very large values, denormals
- [ ] Test precision loss accumulation in long computations
- [ ] Compare against reference implementations (NumPy, BLAS)
- [ ] Add property-based tests for mathematical identities

### 12.2 Performance Regression Tests

**Priority:** Medium
**Impact:** Quality

No automated performance regression detection.

**Recommendations:**
- [ ] Add benchmark baselines to CI
- [ ] Implement percentage-based regression alerts
- [ ] Track performance across Swift versions
- [ ] Create benchmark dashboard

### 12.3 Memory Profiling

**Priority:** Medium
**Impact:** Resource Usage

No systematic memory tracking.

**Recommendations:**
- [ ] Add memory high-water-mark tests
- [ ] Profile allocation frequency in benchmarks
- [ ] Test for memory leaks in long-running operations
- [ ] Add COW verification tests

### 12.4 Cross-Platform Testing

**Priority:** Low
**Impact:** Portability

Limited testing on non-macOS platforms.

**Recommendations:**
- [ ] Add Linux CI job
- [ ] Test on iOS/visionOS simulators
- [ ] Verify Accelerate fallbacks work correctly

---

## 13. Documentation

### 13.1 Performance Guide

**Priority:** Medium
**Impact:** Developer Experience

No guidance on optimal usage patterns.

**Recommendations:**
- [ ] Document when to use each vector type
- [ ] Explain batch size selection
- [ ] Add memory vs speed trade-off guidance
- [ ] Include benchmarks in documentation

### 13.2 Integration Examples

**Priority:** Medium
**Impact:** Adoption

Limited real-world usage examples.

**Recommendations:**
- [ ] Add similarity search example
- [ ] Create clustering example
- [ ] Show CoreML integration
- [ ] Document VectorIndex usage patterns

### 13.3 Architecture Decision Records

**Priority:** Low
**Impact:** Maintainability

No record of design decisions.

**Recommendations:**
- [ ] Document why SIMD4<Float> was chosen
- [ ] Explain storage strategy trade-offs
- [ ] Record protocol design rationale

---

## Implementation Priority Matrix

| Priority | Items | Rationale |
|----------|-------|-----------|
| **P0 - Critical** | ~~Dim384 Support~~ ✅, 4.1, ~~5.1~~ → VectorIndex, 9.1, ~~9.3~~ ✅, ~~11.1~~ ✅, ~~11.2~~ ✅ 0.3.0 | Direct impact on EmbedKit/VectorIndex/VectorAccelerate |
| **P1 - High** | ~~2.1~~ ✅, 3.1, ~~8.1~~ ✅, 12.1, Dim1024 Support, **~~9.3-Pointer API~~ ✅**, **~~2.2-Batch Protocol~~ ✅ 0.3.0** | Performance or API quality |
| **P2 - Medium** | ~~1.1~~ deferred (Q8_0), 1.4, 3.2, 4.2, 4.3, ~~4.4~~ → VectorIndex, ~~5.2~~ → VectorAccelerate, 5.3, 5.4, 7.2, ~~9.2~~ ✅ 0.3.0, 9.4, 10.1, ~~10.2~~ → VectorIndex, 11.3, 12.2, 12.3, 13.1, 13.2, **~~4.2-Raw Buffer Normalize~~ ✅ 0.3.0**, **~~9.3-TieBreaker~~ ✅ 0.3.0** | Important but not blocking |
| **P3 - Low** | 1.2 → VectorAccelerate, ~~1.3~~ → VectorIndex, 2.3, 2.4, 3.3, 3.4, 4.5, 6.1, 6.2, 6.3, 7.1, 7.3, 8.2, 8.3, 8.4, 10.3, 12.4, 13.3 | Nice to have |

### Downstream Package Requests

**VectorIndex** (documented in `VectorIndex/IMPROVEMENTS.md` Section 9.5):

| Request | Section | Priority | Rationale | Status |
|---------|---------|----------|-----------|--------|
| Pointer-based TopK API | 9.3 | P1 | Zero-copy from GPU/mmap buffers | ✅ v0.2.0 |
| Raw buffer normalization | 4.2 | P2 | Mmap/GPU buffer compatibility | ✅ 0.3.0 |
| Configurable tie-breaking | 9.3 | P2 | Reproducibility in tests/benchmarks | ✅ 0.3.0 (`TieBreaker`) |

**VectorAccelerate** (analysis of integration patterns):

| Request | Section | Priority | Rationale | Status |
|---------|---------|----------|-----------|--------|
| Batch distance protocol | 2.2 | P1 | Enables GPU-optimized batch operations via protocol dispatch | ✅ 0.3.0 (`BatchKernelProvider`) |

**Note:** Other VectorAccelerate suggestions were either already implemented in VectorCore (`withUnsafeBufferPointer`, `StaticDimension.value`, `VectorError`) or belong in VectorAccelerate's layer (GPU buffer protocols, acceleration providers). See analysis in VectorAccelerate agent prompt.

---

## Version Milestones

### v0.1.5 (Released November 2025) ✅
- [x] **Add Dim384 optimized vector type (EmbedKit critical)** ✅ DONE
- [x] **Add normalizedUnchecked() API (Item 2.1)** ✅ DONE
- [x] **Optimize generic Manhattan distance (Item 8.1)** ✅ DONE
- [x] **Add Top-K Selection public API (Item 9.3)** ✅ DONE
- [x] **Add VectorIndex Integration protocols (Item 11.1)** ✅ DONE

### v0.2.0 (API Enhancements)
- [x] **Pointer-based TopK API** (VectorIndex request) - Zero-copy from GPU/mmap ✅ v0.2.0
- [x] **Batch distance protocol** (VectorAccelerate request) - GPU-optimized batch ops ✅ **0.3.0** (`BatchKernelProvider`)
- [x] **Raw buffer normalization** (VectorIndex request) - Mmap compatibility ✅ **0.3.0**
- [x] **Configurable tie-breaking** (VectorIndex request) - Reproducibility ✅ **0.3.0** (`TieBreaker`)
- [ ] Resolve all 🔒 marked API changes
- [ ] Complete remaining P0 items (4.1, 9.1) — 5.1 redirected to VectorIndex; 11.2 ✅ shipped 0.3.0
- [ ] Add Dim1024 optimized vector type (E5-large models)
- [ ] Finalize protocol designs

### v0.3.0 — "Hold the line, sharpen the seams" (released 2026-06-07)

**Shipped:**
- [x] **CPU/AMX GEMM batch distance** — `MatrixDistance` (§9.2, §2.2)
- [x] **Configurable tie-breaking** — `TieBreaker` (§9.3)
- [x] **Provider seam** — `BatchKernelProvider` (§11.2)
- [x] **Zero-copy buffer contract** — `UnifiedVectorBuffer` / `PageAlignedBuffer` + frozen `SoALayout` (§11.2)
- [x] **SoA page-alignment APIs** + transparent matrix routing (`BatchOperations.Configuration.enableMatrixRouting` / `matrixRoutingMinN`)
- [x] **⚠️ BREAKING:** removed no-op `blockSize:` from `SoA.init(vectors:…)` and `SoAFP16.init(vectors:…)`

**Deferred / redirected:**
- [ ] Block-wise quantization (`Q8_0`) — **deferred** (specced, consumer-gated; see DOCUMENT-3)
- [→] `mmap` storage, prefetch, PQ/ADC → **VectorIndex**; binary-packed + Hamming → **VectorAccelerate**; sparse CSR → **deferred** (no producer)

**Still open (carried forward):**
- [ ] P1 kernel optimizations (3.1, 12.1 remaining)
- [ ] Memory management enhancements (10.1, 10.3)

### v1.0.0 (Stable Release)
- [ ] All P0-P2 items resolved
- [ ] Comprehensive documentation
- [ ] Full test coverage
- [ ] VectorIndex/VectorAccelerate integration validated

---

*Last updated: 2026-06-07 (0.3.0 reconciliation)*
