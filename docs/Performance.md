# VectorCore Performance Guide

This guide provides detailed performance characteristics, optimization strategies, and best practices for using VectorCore effectively.

## Performance Overview

VectorCore is designed for maximum performance through:
- **SIMD Operations**: Hardware-accelerated vector instructions
- **Memory Optimization**: Aligned allocations and cache-friendly layouts
- **Compile-Time Optimization**: Zero-cost abstractions via generics
- **Parallel Processing**: Efficient batch operations using Swift concurrency

## Performance by Vector Size

### Small Vectors (1-64 dimensions)
**Storage**: `SmallVectorStorage` using SIMD64<Float>

| Operation | Complexity | Typical Performance |
|-----------|-----------|-------------------|
| Creation | O(1) | ~5 ns |
| Element Access | O(1) | ~1 ns |
| Dot Product | O(n) | ~10 ns (n=32) |
| Addition | O(n) | ~8 ns (n=32) |
| Normalization | O(n) | ~20 ns (n=32) |

**Characteristics**:
- Stack allocated (no heap overhead)
- Fits in CPU registers
- Optimal SIMD utilization
- No memory allocation costs

### Medium Vectors (65-512 dimensions)
**Storage**: `MediumVectorStorage` with aligned heap allocation

| Operation | Complexity | Typical Performance |
|-----------|-----------|-------------------|
| Creation | O(n) | ~100 ns (n=256) |
| Element Access | O(1) | ~2 ns |
| Dot Product | O(n) | ~50 ns (n=256) |
| Addition | O(n) | ~40 ns (n=256) |
| Normalization | O(n) | ~100 ns (n=256) |

**Characteristics**:
- Single heap allocation
- Good cache locality
- Efficient SIMD operations
- 16-byte memory alignment

### Large Vectors (513+ dimensions)
**Storage**: `LargeVectorStorage` with page-aligned allocation

| Operation | Complexity | Typical Performance |
|-----------|-----------|-------------------|
| Creation | O(n) | ~500 ns (n=1536) |
| Element Access | O(1) | ~3 ns |
| Dot Product | O(n) | ~300 ns (n=1536) |
| Addition | O(n) | ~250 ns (n=1536) |
| Normalization | O(n) | ~600 ns (n=1536) |

**Characteristics**:
- Page-aligned allocation
- Optimized for bulk operations
- May span multiple cache lines
- Benefits from prefetching

### Dynamic Vectors (Runtime dimensions)
**Storage**: `COWDynamicStorage` with copy-on-write

| Operation | Complexity | Typical Performance |
|-----------|-----------|-------------------|
| Creation | O(n) | ~200 ns (n=1000) |
| Copy (COW) | O(1) | ~5 ns |
| First Write | O(n) | ~400 ns (n=1000) |
| Element Access | O(1) | ~4 ns |

**Characteristics**:
- Copy-on-write optimization
- Flexible runtime sizing
- Additional indirection overhead
- Efficient for read-heavy workloads

## Operation Performance

### Basic Operations

#### Element-wise Operations
```swift
// Addition: O(n), ~0.15 ns per element
let sum = vector1 + vector2

// Scalar multiplication: O(n), ~0.12 ns per element
let scaled = vector * 2.5

// Subtraction: O(n), ~0.15 ns per element
let diff = vector1 - vector2
```

#### Mathematical Operations
```swift
// Dot product: O(n), ~0.2 ns per element
let dot = vector1.dotProduct(vector2)

// Magnitude: O(n), ~0.25 ns per element
let mag = vector.magnitude

// Normalization: O(n), ~0.4 ns per element
let normalized = vector.normalized()
```

### Distance Metrics

Performance comparison for 512-dimensional vectors:

| Metric | Time | Notes |
|--------|------|--------|
| Euclidean | ~120 ns | sqrt(sum((a-b)²)) |
| Manhattan | ~80 ns | sum(abs(a-b)) |
| Cosine | ~150 ns | 1 - (a·b)/(‖a‖‖b‖) |
| Dot Product | ~50 ns | Direct hardware support |

### Batch Operations

Batch operations leverage parallelism for better throughput:

```swift
// Finding k-nearest neighbors
// Time: O(n log k) where n = dataset size
let nearest = await BatchOperations.findNearest(
    to: query,
    in: dataset,  // 10,000 vectors
    k: 50
)
// ~50ms for 10,000 x 768-dimensional vectors
```

## Optimization Strategies

### 1. Choose the Right Vector Type

```swift
// GOOD: Use static dimensions when known at compile time
let embedding: Vector<Dim768> = model.encode(text)

// AVOID: Using DynamicVector for fixed dimensions
let embedding = DynamicVector(dimension: 768)  // Unnecessary overhead
```

### 2. Leverage Batch Operations

```swift
// GOOD: Process multiple vectors together
let normalized = BatchOperations.normalizeInPlace(&vectors)

// SLOWER: Individual operations in a loop
for i in vectors.indices {
    vectors[i] = vectors[i].normalized()
}
```

### 3. Minimize Allocations

```swift
// GOOD: Reuse buffers for temporary calculations
var workspace = Vector<Dim512>()
for item in dataset {
    workspace = item - query  // Reuses workspace memory
    let distance = workspace.magnitude
}

// SLOWER: Creating new vectors repeatedly
for item in dataset {
    let diff = item - query  // Allocates new vector
    let distance = diff.magnitude
}
```

### 4. Use Appropriate Parallelism

```swift
// Optimal chunk size depends on vector dimension and operation
let chunkSize = max(1000, dataset.count / (ProcessInfo.processInfo.activeProcessorCount * 4))

// Process in parallel chunks
await withTaskGroup(of: [Float].self) { group in
    for chunk in dataset.chunked(into: chunkSize) {
        group.addTask {
            chunk.map { $0.magnitude }
        }
    }
}
```

### 5. Fast Path Optimizations

VectorCore includes fast paths for common cases:

```swift
// These operations have optimized implementations:
vector * 0.0   // Returns zero vector without computation
vector * 1.0   // Returns copy without computation
vector * -1.0  // Uses specialized negation
vector / 1.0   // Returns copy without computation

// Already normalized vectors skip computation
if abs(vector.magnitude - 1.0) < 1e-6 {
    return vector  // Already normalized
}
```

## Memory Usage Patterns

### Storage Overhead

| Vector Type | Dimension | Memory Usage | Allocation |
|------------|-----------|--------------|------------|
| Vector<Dim32> | 32 | 256 bytes | Stack |
| Vector<Dim64> | 64 | 256 bytes | Stack |
| Vector<Dim128> | 128 | 512 bytes | Heap |
| Vector<Dim256> | 256 | 1 KB | Heap |
| Vector<Dim512> | 512 | 2 KB | Heap |
| Vector<Dim768> | 768 | 3 KB | Heap |
| Vector<Dim1536> | 1536 | 6 KB | Heap |
| DynamicVector | n | 4n + 24 bytes | Heap |

### Cache Considerations

```swift
// Good cache utilization: Process vectors that fit in L1 cache together
let l1Size = 32 * 1024  // 32KB typical L1 cache
let vectorSize = 768 * 4  // 768 floats * 4 bytes
let vectorsPerL1 = l1Size / vectorSize  // ~10 vectors

// Process in cache-friendly batches
for batch in vectors.chunked(into: vectorsPerL1) {
    processBatch(batch)
}
```

## Benchmarking Best Practices

### 1. Warm-up Iterations
```swift
// Warm up to ensure consistent measurements
for _ in 0..<100 {
    _ = vector1.dotProduct(vector2)
}

// Actual measurement
let start = CFAbsoluteTimeGetCurrent()
for _ in 0..<iterations {
    _ = vector1.dotProduct(vector2)
}
let elapsed = CFAbsoluteTimeGetCurrent() - start
```

### 2. Statistical Analysis
```swift
// Collect multiple samples
var times: [Double] = []
for _ in 0..<samples {
    let time = measureOperation()
    times.append(time)
}

// Use median instead of mean to reduce outlier impact
let median = times.sorted()[times.count / 2]
let p95 = times.sorted()[Int(Double(times.count) * 0.95)]
```

### 3. Memory Pressure
```swift
// Test under memory pressure
let largeDataset = (0..<100_000).map { _ in
    Vector<Dim768>.random(in: -1...1)
}

// This will show realistic performance with cache misses
let results = largeDataset.map { $0.magnitude }
```

## Platform-Specific Optimizations

### Apple Silicon (M1/M2/M3)
- Excellent SIMD performance with NEON
- Unified memory architecture benefits large vectors
- Efficient atomic operations for COW

### Intel x86_64
- AVX2/AVX512 support through Accelerate
- Larger L3 cache benefits batch operations
- NUMA considerations for multi-socket systems

## Common Performance Pitfalls

### 1. Unnecessary Normalization
```swift
// BAD: Normalizing already-normalized vectors
let normalized1 = vector.normalized()
let normalized2 = normalized1.normalized()  // Wasteful!

// GOOD: Check if needed
if abs(vector.magnitude - 1.0) > 1e-6 {
    vector = vector.normalized()
}
```

### 2. Repeated Calculations
```swift
// BAD: Computing magnitude multiple times
if vector.magnitude > 0 {
    let normalized = vector / vector.magnitude  // Recomputes!
}

// GOOD: Store computed values
let mag = vector.magnitude
if mag > 0 {
    let normalized = vector / mag
}
```

### 3. Wrong Parallelism Granularity
```swift
// BAD: Too fine-grained parallelism
await withTaskGroup(of: Float.self) { group in
    for element in vector {
        group.addTask { element * 2.0 }  // Overhead > benefit!
    }
}

// GOOD: Appropriate granularity
let scaled = vector * 2.0  // Let SIMD handle element parallelism
```

## Performance Monitoring

### Using Instruments
```swift
// Mark regions for profiling
os_signpost(.begin, log: perfLog, name: "VectorOperation")
let result = expensiveVectorOperation()
os_signpost(.end, log: perfLog, name: "VectorOperation")
```

### Custom Metrics
```swift
struct PerformanceMetrics {
    let operationsPerSecond: Double
    let bytesPerSecond: Double
    let cacheHitRate: Double
    
    static func measure<T>(_ operation: () throws -> T) rethrows -> (T, PerformanceMetrics) {
        let start = CFAbsoluteTimeGetCurrent()
        let result = try operation()
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        
        // Calculate metrics...
        return (result, metrics)
    }
}
```

## Summary

Key performance guidelines:

1. **Use static dimensions** when possible for compile-time optimizations
2. **Batch operations** for better throughput with large datasets
3. **Consider cache effects** when processing many vectors
4. **Profile your code** to identify actual bottlenecks
5. **Minimize allocations** in hot loops
6. **Leverage COW** for read-heavy workloads with DynamicVector
7. **Use appropriate parallelism** based on operation cost

VectorCore is optimized for the common case while providing flexibility for specialized needs. By understanding these performance characteristics, you can write efficient vector processing code that fully utilizes modern hardware capabilities.