# VectorCore Performance Guide

This guide explains how to maximize performance in VectorCore, covering optimized vector types, provider configuration, and performance benchmarking.

---

## Table of Contents

1. [Choosing the Right Vector Type](#choosing-the-right-vector-type)
2. [Optimized Vectors Deep Dive](#optimized-vectors-deep-dive)
3. [Provider Configuration](#provider-configuration)
4. [Batch Operations](#batch-operations)
5. [GEMM / Matrix Batch-Distance](#gemm--matrix-batch-distance)
6. [Memory Management](#memory-management)
7. [Benchmarking](#benchmarking)
8. [Performance Pitfalls](#performance-pitfalls)

---

## Choosing the Right Vector Type

VectorCore provides several vector types optimized for different use cases:

### Decision Matrix

| Use Case | Recommended Type | Why? |
|----------|------------------|------|
| **BERT embeddings (768-dim)** | `Vector768Optimized` | Specialized SIMD implementation |
| **OpenAI embeddings (1536-dim)** | `Vector1536Optimized` | Specialized SIMD implementation |
| **Custom 512-dim** | `Vector512Optimized` | Specialized SIMD implementation |
| **Dynamic dimensions** | `DynamicVector` | Flexible but slower |
| **Compile-time safety** | `Vector<D>` | Type-safe with generic overhead |

### Performance Hierarchy

```
Vector{512,768,1536}Optimized    ← Fastest (5-10x)
          ↓
      Vector<D>                   ← Good (type-safe)
          ↓
    DynamicVector                 ← Flexible (runtime overhead)
```

**Rule of Thumb**: If your dimension is 512, 768, or 1536, **always** use the optimized variant.

---

## Optimized Vectors Deep Dive

### Architecture

Optimized vectors use **SIMD4<Float>** storage with specialized unrolling:

```swift
public struct Vector512Optimized {
    // 128 SIMD4<Float> chunks = 512 scalars
    public var storage: ContiguousArray<SIMD4<Float>>  // 128 chunks

    public init { /* ... */ }
}
```

### Key Performance Features

#### 1. **SIMD4 Storage Layout**

```swift
// Generic Vector<Dim512>
storage: [Float] = [f0, f1, f2, f3, f4, ...]  // Sequential scalars

// Vector512Optimized
storage: [SIMD4<Float>] = [
    SIMD4(f0, f1, f2, f3),     // Lane 0
    SIMD4(f4, f5, f6, f7),     // Lane 1
    // ... 128 lanes total
]
```

**Benefit**: Single instruction processes 4 scalars simultaneously

#### 2. **Loop Unrolling with Multiple Accumulators**

```swift
// Dot product with 4-way unrolling and 4 accumulators
public func dotProduct(_ other: Vector512Optimized) -> Float {
    var acc0: SIMD4<Float> = .zero
    var acc1: SIMD4<Float> = .zero
    var acc2: SIMD4<Float> = .zero
    var acc3: SIMD4<Float> = .zero

    for i in stride(from: 0, to: 128, by: 4) {
        acc0 += storage[i] * other.storage[i]
        acc1 += storage[i+1] * other.storage[i+1]
        acc2 += storage[i+2] * other.storage[i+2]
        acc3 += storage[i+3] * other.storage[i+3]
    }

    let sum = acc0 + acc1 + acc2 + acc3
    return sum.x + sum.y + sum.z + sum.w
}
```

**Benefit**: Maximizes instruction-level parallelism (ILP), prevents pipeline stalls

#### 3. **Cache-Aware Memory Access**

- **16-byte alignment**: SIMD4<Float> is naturally 16-byte aligned
- **Contiguous storage**: `ContiguousArray` ensures no ARC metadata in the middle
- **Sequential access**: Prefetcher-friendly memory patterns

**Typical Performance** (Apple Silicon M-series) — *indicative only; these are
order-of-magnitude figures, not 0.3.0 measurements. Profile your own hardware with
`vectorcore-bench` (the numbers below are flagged for re-measurement):*
- **Dot product**: ~100ns for 512 dimensions *(indicative)*
- **Euclidean distance**: ~120ns *(indicative)*
- **Normalization**: ~150ns *(indicative)*

---

## Provider Configuration

VectorCore uses `@TaskLocal` for zero-cost provider abstraction:

### Default Providers

```swift
Operations.$simdProvider.withValue(SwiftFloatSIMDProvider()) {
    Operations.$computeProvider.withValue(CPUComputeProvider.automatic) {
        // Your code here
    }
}
```

### Provider Types

#### 1. **SIMDProvider**

Controls low-level SIMD operations (add, multiply, reduce, etc.)

**Available Implementations**:
- `SwiftFloatSIMDProvider` (default): Pure Swift SIMD
- `AccelerateSIMDProvider` (VectorAccelerate): vDSP-backed operations

**When to override**:
- Large vectors (>1000 dimensions): Use Accelerate for vDSP
- Small vectors (<100 dimensions): Swift SIMD is fine

**Example**:
```swift
await Operations.$simdProvider.withValue(AccelerateSIMDProvider()) {
    let centroid = Operations.centroid(of: largeVectors)
    // Uses vDSP for aggregation
}
```

#### 2. **ComputeProvider**

Controls batch operations and parallelization strategy

**Available Implementations**:
- `CPUComputeProvider.automatic`: Auto-parallelizes based on batch size
- `CPUComputeProvider.sequential`: Single-threaded (for debugging)
- `CPUComputeProvider.parallel`: Force multi-threaded execution
- `CPUComputeProvider.performance` / `.efficiency`: Bias toward P- or E-cores
- A `BatchKernelProvider` conformer (e.g. a GPU/Metal provider in VectorAccelerate):
  supplies real hardware batch kernels (see below)

**When to override**:
- GPU-enabled devices: Install a `BatchKernelProvider` for large brute-force search
- Debugging: Use `.sequential` for reproducibility
- Low latency: Use `.automatic` for adaptive parallelization

**Transparent GPU dispatch via `BatchKernelProvider`**

`Operations.findNearest` / `findNearestBatch` *downcast* the installed `computeProvider`
to the `BatchKernelProvider` protocol. When a conformer is present, they delegate the
k-NN query to its kernel — and this path takes **precedence over the CPU GEMM routing**
and the optimized Top-K fast paths. This makes GPU acceleration transparent through
VectorCore's own entry points: no separate API, just a different provider.

```swift
public protocol BatchKernelProvider: ComputeProvider {
    func batchDistance<V: VectorProtocol>(
        query: V, candidates: [V], metric: any DistanceMetric
    ) async throws -> [Float] where V.Scalar == Float

    func findNearest<V: VectorProtocol>(
        query: V, candidates: [V], k: Int, metric: any DistanceMetric
    ) async throws -> [(index: Int, distance: Float)] where V.Scalar == Float

    // Default loops `findNearest`; a true batched GPU kernel should override this
    // (one dispatch for the whole query set is the actual GPU win).
    func findNearestBatch<V: VectorProtocol>(
        queries: [V], candidates: [V], k: Int, metric: any DistanceMetric
    ) async throws -> [[(index: Int, distance: Float)]] where V.Scalar == Float
}
```

**Example** — install a GPU provider; existing `Operations` calls route to it:
```swift
await Operations.$computeProvider.withValue(yourGPUProvider) {
    let results = try await Operations.findNearest(
        to: query,
        in: database,  // 100K vectors
        k: 100
    )
    // Delegates to yourGPUProvider.findNearest (GPU brute-force search),
    // ahead of the CPU GEMM / Top-K paths.
}
```

> `yourGPUProvider` is any type conforming to `BatchKernelProvider`. VectorCore defines
> the protocol; the GPU-backed conformance ships in VectorAccelerate. To feed the GPU
> efficiently, build candidate buffers with the zero-copy SoA layout described in
> [Zero-Copy GPU Batches](#zero-copy-gpu-batches).

#### 3. **BufferProvider**

Controls temporary buffer allocation for operations

**Available Implementations**:
- `SwiftBufferPool` (default): Actor-based pooling with automatic cleanup
- Custom implementations: For specialized memory management

**When to override**:
- Custom allocators (e.g., huge pages, NUMA)
- Memory-constrained environments (embedded, serverless)

---

## Batch Operations

### API Choice

| API | Best For | Characteristics |
|-----|----------|-----------------|
| `Operations` | Single queries, simple ops | Sync or async, minimal overhead |
| `BatchOperations` | Large batches (>100) | Auto-parallelization, async only |

### BatchOperations Auto-Tuning

`BatchOperations` automatically selects parallelization strategy based on:
- Batch size
- Vector dimensionality
- Available CPU cores
- Current system load

**Heuristics**:
```swift
// Small batch: Sequential execution
let results = await BatchOperations.findNearest(to: query, in: vectors, k: 10)
// Batch size < 100: Single-threaded

// Large batch: Parallel execution
let results = await BatchOperations.findNearest(to: query, in: vectors, k: 10)
// Batch size > 1000: Multi-threaded with chunk size = batch / cores
```

### Manual Parallelization Control

`BatchOperations` exposes a thread-safe global `Configuration` updated via a mutating
closure. The tunable fields are `parallelThreshold` (default `1000`), `minimumChunkSize`
(default `256`), `enableMatrixRouting` (default `true`), and `matrixRoutingMinN`
(default `256`).

```swift
// Override automatic tuning
await BatchOperations.updateConfiguration { config in
    config.parallelThreshold = 500   // Parallelize sooner
    config.minimumChunkSize = 128    // Smaller chunks
}

let results = await BatchOperations.findNearest(to: query, in: vectors, k: 10)
// Uses your configuration
```

> The `enableMatrixRouting` / `matrixRoutingMinN` fields control the GEMM batch-distance
> fast path covered in [GEMM / Matrix Batch-Distance](#gemm--matrix-batch-distance).

---

## GEMM / Matrix Batch-Distance

For large query × candidate matrices, computing each pair with a per-pair SIMD kernel
leaves throughput on the table. VectorCore instead computes the **whole distance matrix
with a single matrix multiply**, using the identity:

```
‖x − y‖² = ‖x‖² + ‖y‖² − 2⟨x, y⟩
```

The cross-term `⟨x, y⟩` is one `cblas_sgemm` call, which **routes to the AMX coprocessor
on Apple Silicon**. This is the high-throughput path; the per-pair kernels remain the
small-batch path.

### `MatrixDistance` API

`MatrixDistance` is generic over any `UnifiedVectorBuffer` — the optimized
`Vector384/512/768/1536Optimized` types and `DynamicVector` all conform, so one
implementation serves every dimension. Results are written **row-major**: `out[i*n + j]`
is the distance from `queries[i]` to `candidates[j]`.

```swift
// Allocating convenience overloads (Euclidean SQUARED, clamped ≥ 0):
let flat: [Float] = MatrixDistance.euclideanSquaredMatrix(
    queries: queries, candidates: candidates
)

// Hot path — caller owns `out`; count MUST equal queries.count * candidates.count:
var out = [Float](repeating: 0, count: queries.count * candidates.count)
MatrixDistance.euclideanSquaredMatrix(queries: queries, candidates: candidates, into: &out)

// Cosine has the same three overloads (result = 1 − cos, clamped to [0, 2]):
MatrixDistance.cosineDistanceMatrix(queries: queries, candidates: candidates, into: &out)
```

### Reusing candidates with `PreparedCandidates`

For **repeated** search against a fixed candidate set, pack the candidates **once** with
`prepare` and reuse the `PreparedCandidates` across many query batches. This avoids
re-packing and re-norming the (large) candidate side on every call. Pass
`normalized: false` for Euclidean (raw rows + squared norms) and `normalized: true` for
Cosine (L2-normalized rows). `PreparedCandidates` exposes `.count` and `.dimension`.

```swift
// Pack once (normalized: false → Euclidean; true → Cosine).
let prepared = MatrixDistance.prepare(candidates, normalized: false)

// Reuse across many query batches:
var out = [Float](repeating: 0, count: queries.count * prepared.count)
MatrixDistance.euclideanSquaredMatrix(queries: queries, prepared: prepared, into: &out)
```

### Automatic routing (crossover)

You usually don't call `MatrixDistance` directly. `BatchOperations.pairwiseDistances` and
`Operations.findNearestBatch` **auto-route** through it when both:

- `enableMatrixRouting` is `true` (the default), **and**
- the per-side count clears `matrixRoutingMinN` (default **256**),

for optimized vector types with the Euclidean or Cosine metric. Below the threshold the
per-pair kernels win; above it the matrix multiply amortizes the packing cost. This is the
same auto-tuning machinery as the parallelization heuristics above — tune it the same way:

```swift
await BatchOperations.updateConfiguration { config in
    config.matrixRoutingMinN = 512   // Raise the GEMM crossover
    // config.enableMatrixRouting = false  // Force the exact per-pair path
}
```

> An installed `BatchKernelProvider` (GPU) takes **precedence** over this CPU GEMM
> routing in `Operations.findNearest` / `findNearestBatch` — see
> [Transparent GPU dispatch](#2-computeprovider).

### Accuracy caveat

The GEMM result agrees with the per-pair kernels to **~1e-3 relative**, **not
bit-identical**. The difference comes from evaluating distance through the
`‖x‖² + ‖y‖² − 2⟨x, y⟩` identity (rather than summing `(xᵢ − yᵢ)²` directly) plus the final
clamp: Euclidean is **squared** and clamped `≥ 0` (the identity can round slightly negative
when `x ≈ y`); cosine is `1 − cos` clamped to `[0, 2]`. If you need exact per-pair values,
set `enableMatrixRouting = false`.

### Calibrating the crossover

The ideal `matrixRoutingMinN` depends on dimension and hardware. The `vectorcore-bench`
**`matrix`** suite (`MatrixDistanceBench`) measures GEMM vs per-pair kernels across N and
dimension so you can pick the crossover for your target:

```bash
.build/release/vectorcore-bench --suites matrix
```

---

## Memory Management

### Aligned Allocation

VectorCore uses 64-byte aligned allocations for SIMD operations:

```swift
// Automatically aligned
let vector = Vector512Optimized(repeating: 1.0)
// storage is 64-byte aligned

// Manual alignment (advanced) — Float convenience overload
let ptr = try AlignedMemory.allocateAligned(count: 512, alignment: 64)
defer { AlignedMemory.deallocate(ptr) }

// For other element types, pass the type via the `type:` label:
// let ptr = try AlignedMemory.allocateAligned(type: Float.self, count: 512, alignment: 64)
```

**Why alignment matters**:
- SIMD load/store instructions require 16-byte alignment
- Cache lines are 64 bytes; aligned access avoids cache line splits
- Performance penalty for misaligned access: ~2-5x slowdown

See [Memory_Alignment.md](Memory_Alignment.md) for details.

### Buffer Pooling

Avoid repeated allocations in hot loops:

```swift
// ❌ Bad: Allocates on every iteration
for query in queries {
    let distances = candidates.map { query.euclideanDistance(to: $0) }
    // `distances` array allocated every iteration
}

// ✅ Good: Use withUnsafeBufferPointer
var distances = [Float](repeating: 0, count: candidates.count)
for query in queries {
    distances.withUnsafeMutableBufferPointer { distBuffer in
        for (i, candidate) in candidates.enumerated() {
            distBuffer[i] = query.euclideanDistance(to: candidate)
        }
    }
}
```

**Even better**: Use `BatchOperations` which handles pooling internally:

```swift
let allDistances = await BatchOperations.pairwiseDistances(queries, candidates)
// Internally uses buffer pool, zero user-visible allocations
```

---

## Benchmarking

### Using vectorcore-bench

VectorCore includes a comprehensive benchmark CLI:

```bash
# Build benchmark executable
swift build --product vectorcore-bench -c release

# Run all benchmarks
.build/release/vectorcore-bench batch --profile short

# Specific benchmark suite
.build/release/vectorcore-bench distance --dims 512 --repeats 50

# Save results to CSV
.build/release/vectorcore-bench batch --profile short --format csv > results.csv
```

### Benchmark Suites

| Suite | What It Measures | Use Case |
|-------|------------------|----------|
| `distance` | Distance metric performance | Validate optimization claims |
| `dot` | Dot product performance | SIMD effectiveness |
| `batch` | Batch operation throughput | Parallelization efficiency |
| `normalize` | Normalization performance | vDSP integration |
| `memory` | Allocation overhead | Buffer pool efficiency |
| `matrix` | GEMM batch-distance vs per-pair kernels | Calibrate `matrixRoutingMinN` crossover |

Run the GEMM calibration suite with:

```bash
.build/release/vectorcore-bench --suites matrix
```

### Performance Regression Detection

Integrate benchmarks into CI:

```yaml
# .github/workflows/performance.yml
- name: Run performance benchmarks
  run: |
    .build/release/vectorcore-bench batch --profile short --format json > bench.json

- name: Check for regressions
  run: |
    python scripts/compare_benchmarks.py bench.json baseline.json
    # Fails if any operation is >10% slower
```

---

## Performance Pitfalls

### ❌ Pitfall 1: Using Generic Vectors for Common Dimensions

```swift
// ❌ Slow (generic overhead)
let embedding = Vector<Dim512>(array)
let distance = embedding.euclideanDistance(to: other)

// ✅ Fast (specialized implementation)
let embedding = try Vector512Optimized(array)
let distance = embedding.euclideanDistance(to: other)
```

**Impact**: 5-10x performance difference

---

### ❌ Pitfall 2: Allocating in Hot Loops

```swift
// ❌ Allocates 2 arrays per iteration
for candidate in candidates {
    let dist = EuclideanDistance().distance(query, candidate)
}

// ✅ Uses withUnsafeBufferPointer (zero allocations)
query.withUnsafeBufferPointer { queryBuf in
    for candidate in candidates {
        candidate.withUnsafeBufferPointer { candBuf in
            // Direct memory access, no allocations
            let dist = computeDistance(queryBuf, candBuf)
        }
    }
}
```

**Impact**: 10-100x speedup for large batches

---

### ❌ Pitfall 3: Not Using BatchOperations for Large Datasets

```swift
// ❌ Slow (sequential)
var results: [[Int]] = []
for query in queries {
    let nearest = try await Operations.findNearest(to: query, in: database, k: 10)
    results.append(nearest.map { $0.index })
}

// ✅ Fast (auto-parallelized)
let results = try await Operations.findNearestBatch(
    queries: queries,
    in: database,
    k: 10
)
```

**Impact**: Near-linear speedup with CPU cores (4x on 4-core, 8x on 8-core)

---

### ❌ Pitfall 4: Ignoring Cache Effects

```swift
// ❌ Cache-unfriendly (random access)
for i in randomIndices {
    let dist = query.euclideanDistance(to: candidates[i])
}

// ✅ Cache-friendly (sequential access)
for candidate in candidates {
    let dist = query.euclideanDistance(to: candidate)
}
```

**Impact**: 2-3x slowdown for random access on large datasets

---

### ❌ Pitfall 5: Over-Normalizing Vectors

```swift
// ❌ Normalizes every time
for query in queries {
    let normalized = try query.normalized().get()
    let dist = normalized.cosineSimilarity(to: candidate)
}

// ✅ Normalize once, reuse
let normalizedQueries = try await Operations.normalize(queries)
for normalized in normalizedQueries {
    let dist = normalized.cosineSimilarity(to: candidate)
}
```

**Impact**: Normalization costs roughly ~150ns per 512-dim vector (*indicative — profile
your hardware*); pre-normalizing avoids repeating that work on every comparison.

---

## Performance Checklist

Before deploying performance-critical code, verify:

- [ ] Using `Vector{512,768,1536}Optimized` for common dimensions
- [ ] Using `BatchOperations` for batches >100
- [ ] No allocations in hot loops (use `withUnsafeBufferPointer`)
- [ ] Sequential memory access patterns (cache-friendly)
- [ ] Pre-normalized vectors when doing cosine similarity
- [ ] Benchmarked with `vectorcore-bench` to validate
- [ ] Profiled with Instruments to find bottlenecks

---

## Advanced: Custom Providers

To plug in specialized hardware (GPU, accelerator), conform to `BatchKernelProvider`.
Because it refines `ComputeProvider`, `Operations.findNearest` / `findNearestBatch`
downcast to it and delegate — your kernel runs ahead of the CPU GEMM and Top-K paths.

A conformer must honor the kernel contract:
- `batchDistance` returns one distance per candidate, in candidate order.
- `findNearest` returns up to `k` `(index, distance)` pairs sorted ascending by distance,
  indexing into `candidates`, matching the CPU reference within a documented tolerance.
- `findNearestBatch` has a default that loops `findNearest`; override it with a single
  batched dispatch over the whole query set — that one-dispatch form is the real GPU win.

```swift
struct GPUKernelProvider: BatchKernelProvider {
    func batchDistance<V: VectorProtocol>(
        query: V, candidates: [V], metric: any DistanceMetric
    ) async throws -> [Float] where V.Scalar == Float {
        // Upload, run the distance kernel, read back one distance per candidate.
        return try await gpuBatchDistance(query, candidates, metric)
    }

    func findNearest<V: VectorProtocol>(
        query: V, candidates: [V], k: Int, metric: any DistanceMetric
    ) async throws -> [(index: Int, distance: Float)] where V.Scalar == Float {
        let distances = try await gpuBatchDistance(query, candidates, metric)
        return gpuTopK(distances, k)  // (index, distance) sorted ascending by distance
    }

    func findNearestBatch<V: VectorProtocol>(
        queries: [V], candidates: [V], k: Int, metric: any DistanceMetric
    ) async throws -> [[(index: Int, distance: Float)]] where V.Scalar == Float {
        // Override the default: one GPU dispatch for the whole query set.
        return try await gpuFindNearestBatch(queries, candidates, k, metric)
    }
}

// Install it — existing Operations calls now route through the GPU kernel.
await Operations.$computeProvider.withValue(GPUKernelProvider()) {
    let results = try await Operations.findNearest(to: query, in: database, k: 100)
}
```

### Zero-Copy GPU Batches

Feeding a GPU kernel is bandwidth-bound, so the buffer layout is itself a throughput
lever. Build the candidate set with the page-aligned SoA layout to obtain
`MTLDevice.makeBuffer(bytesNoCopy:)`-eligible storage (no host→device staging copy):

```swift
// Page-aligned SoA → bytesNoCopy-eligible backing for the Metal path.
let soa = SoA.build(from: candidates, pageAligned: true)
```

`PageAlignedBuffer` / `SoA.build(from:pageAligned: true)` allocate page-aligned, page-
rounded storage so the GPU can map the bytes directly. See
[SoA_Layout_Contract.md](SoA_Layout_Contract.md) for the exact byte-count contract that
`SoALayout` and `makeBuffer(bytesNoCopy:)` must agree on.

---

## Profiling with Instruments

### CPU Profiling

```bash
# Build with release + debug symbols
swift build -c release -Xswiftc -g

# Profile with Instruments
open -a Instruments .build/release/YourApp

# Or use command-line profiling
xcrun xctrace record --template 'Time Profiler' --launch .build/release/YourApp
```

**What to look for**:
- Hot functions (>5% of time): Optimize these first
- Allocation spikes: Use buffer pooling
- Cache misses: Improve memory access patterns

### Allocations Profiling

```bash
xcrun xctrace record --template 'Allocations' --launch .build/release/YourApp
```

**What to look for**:
- Persistent allocations: Should be minimal
- Allocation rate: <100 allocs/sec for steady-state
- Peak memory: Ensure it fits in available RAM

---

## Platform-Specific Considerations

### Apple Silicon (M-series)

**Advantages**:
- Larger L1 cache (128KB+): Better for large working sets
- Unified memory: Zero-copy GPU transfers (with Metal)
- Wide SIMD (NEON): 128-bit SIMD ops

**Optimizations**:
- Use larger chunk sizes (16KB vs 8KB for Intel)
- Leverage Metal for batch >5000
- Exploit fast memory bandwidth

### Intel (x86_64)

**Advantages**:
- AVX2/AVX-512: Wider SIMD (256/512-bit)
- Higher clock speeds: Better single-threaded perf

**Optimizations**:
- Smaller chunk sizes (8KB)
- More aggressive loop unrolling
- Careful with thermal throttling on sustained workloads

---

**Document Version**: 0.3.0
**Last Updated**: 2026-06-07
**Applies to**: VectorCore v0.3.0+
