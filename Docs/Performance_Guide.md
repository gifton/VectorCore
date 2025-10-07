# VectorCore Performance Guide

This guide explains how to maximize performance in VectorCore, covering optimized vector types, provider configuration, and performance benchmarking.

---

## Table of Contents

1. [Choosing the Right Vector Type](#choosing-the-right-vector-type)
2. [Optimized Vectors Deep Dive](#optimized-vectors-deep-dive)
3. [Provider Configuration](#provider-configuration)
4. [Batch Operations](#batch-operations)
5. [Memory Management](#memory-management)
6. [Benchmarking](#benchmarking)
7. [Performance Pitfalls](#performance-pitfalls)

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

**Typical Performance** (Apple Silicon M-series):
- **Dot product**: ~100ns for 512 dimensions
- **Euclidean distance**: ~120ns
- **Normalization**: ~150ns

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
- `MetalComputeProvider` (VectorAccelerate): GPU-accelerated

**When to override**:
- GPU-enabled devices: Use Metal for batches >1000
- Debugging: Use sequential for reproducibility
- Low latency: Use automatic for adaptive parallelization

**Example**:
```swift
await Operations.$computeProvider.withValue(MetalComputeProvider()) {
    let results = try await Operations.findNearest(
        to: query,
        in: database,  // 100K vectors
        k: 100
    )
    // GPU-accelerated brute-force search
}
```

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

```swift
// Override automatic tuning
await BatchOperations.configure(
    minBatchSizeForParallel: 500,
    preferredChunkSize: 100
)

let results = await BatchOperations.findNearest(to: query, in: vectors, k: 10)
// Uses your configuration
```

---

## Memory Management

### Aligned Allocation

VectorCore uses 64-byte aligned allocations for SIMD operations:

```swift
// Automatically aligned
let vector = Vector512Optimized(repeating: 1.0)
// storage is 64-byte aligned

// Manual alignment (advanced)
let ptr = try AlignedMemory.allocateAligned(Float.self, count: 512, alignment: 64)
defer { AlignedMemory.deallocate(ptr) }
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

**Impact**: Normalization is ~150ns per 512-dim vector; pre-normalize to save ~10% overhead

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

Implement custom providers for specialized hardware:

```swift
struct CUDAComputeProvider: ComputeProvider {
    func findNearest<V: VectorProtocol>(
        to query: V,
        in vectors: [V],
        k: Int
    ) async throws -> [(index: Int, distance: Float)] {
        // Transfer to GPU
        let gpuQuery = cudaAllocate(query)
        let gpuVectors = cudaAllocate(vectors)

        // Compute distances on GPU
        let gpuDistances = cudaDistances(gpuQuery, gpuVectors)

        // Top-K selection on GPU
        let gpuResults = cudaTopK(gpuDistances, k)

        // Transfer back to CPU
        return cudaCopyToHost(gpuResults)
    }
}

// Use it
await Operations.$computeProvider.withValue(CUDAComputeProvider()) {
    let results = try await Operations.findNearest(to: query, in: database, k: 100)
}
```

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

**Document Version**: 1.0
**Last Updated**: October 2025
**Applies to**: VectorCore v0.1.0+
