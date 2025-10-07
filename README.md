# VectorCore

[![Swift](https://img.shields.io/badge/Swift-6.0-orange.svg?style=flat)](https://swift.org)
[![Platforms](https://img.shields.io/badge/Platforms-macOS%2014%2B%20|%20iOS%2017%2B%20|%20tvOS%2017%2B%20|%20watchOS%2010%2B%20|%20visionOS%201%2B-blue.svg?style=flat)](https://swift.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat)](LICENSE)
[![SPM](https://img.shields.io/badge/SPM-compatible-brightgreen.svg?style=flat)](https://swift.org/package-manager/)
[![CI](https://github.com/gifton/VectorCore/actions/workflows/ci.yml/badge.svg)](https://github.com/gifton/VectorCore/actions/workflows/ci.yml)

High-performance, type-safe vector operations for Swift with zero third‑party dependencies. VectorCore provides fast vector math optimized for machine learning, scientific computing, and real-time applications.

## 🚀 Why VectorCore?

VectorCore is designed from the ground up for performance and ease of use:

- **Blazing Fast SIMD** - Optimized implementations for 512/768/1536 dimensions with unrolled loops
- **Hardware-optimized** - Leverages SIMD4<Float>, Accelerate-backed primitives, and cache-aware memory
- **Type-safe dimensions** - Compile-time dimension checking prevents runtime errors
- **Specialized Vectors** - Dedicated Vector512Optimized, Vector768Optimized, Vector1536Optimized types
- **Zero third‑party dependencies** - Pure Swift with no external packages
- **Unified Protocol** - Clean VectorProtocol for consistent API across all vector types

Indicative performance (Apple Silicon M‑series): optimized 512/768/1536‑dimensional operations are significantly faster than generic fixed‑dimension vectors due to SIMD storage and loop unrolling.

Whether you're building ML pipelines, processing embeddings, or need fast numerical computing, VectorCore delivers the performance you need with an API you'll love.

## ⚡️ GPU Acceleration

VectorCore is CPU-only and contains no GPU/Metal code. If you need GPU acceleration, use the separate VectorAccelerate package, which implements GPU-backed compute while integrating seamlessly with VectorCore types and protocols.

## 📦 Installation

Add VectorCore to your Swift Package Manager dependencies:

```swift
dependencies: [
    .package(url: "https://github.com/gifton/VectorCore.git", from: "0.1.0")
]
```

Then add it to your target:

```swift
.target(
    name: "YourTarget",
    dependencies: ["VectorCore"]
)
```

## 🏃 Quick Start

```swift
import VectorCore

// Use optimized vectors for best performance
let v1 = Vector512Optimized(repeating: 1.0)
let v2 = Vector512Optimized(repeating: 2.0)

// Fast operations with SIMD
let dotProduct = v1.dotProduct(v2)
let distance = v1.euclideanDistance(to: v2)
let normalized = try v1.normalized().get()

// Or use generic vectors with compile-time safety
let bert = Vector<Dim768>(repeating: 0.5)
let gpt = Vector<Dim1536>(repeating: 0.5)

// Distance metrics work with all vector types
let metric = EuclideanDistance()
let dist = metric.distance(v1, v2)
```

## ✨ Key Features

### 1. **Type-Safe Dimensions**
Prevent dimension mismatch errors at compile time:

```swift
let bert = Vector768.randomUnit()           // BERT embeddings
let gpt = Vector1536.random(in: -1...1)     // GPT embeddings  
// let invalid = bert + gpt  // Compile error! Dimensions don't match
```

### 2. **Zero-Allocation Math**
All core operations avoid heap allocations:

```swift
let v1 = Vector256.random(in: -1...1)
let v2 = Vector256.random(in: -1...1)

// These operations allocate nothing on the heap
let sum = v1 + v2
let product = v1 * 2.5
let dotProduct = v1.dotProduct(v2)
```

### 3. **Hardware Acceleration**
Optimized for Apple Silicon and Intel processors:

```swift
// SIMD-optimized operations
let magnitude = vector.magnitude
let normalized = vector.normalized()
let distance = v1.euclideanDistance(to: v2)
```

### 4. **Automatic Parallelization**
Large batch operations scale across cores:

```swift
// Automatically parallelizes for datasets >= 1000 vectors (configurable)
let neighbors = await BatchOperations.findNearest(
    to: query,
    in: vectors,
    k: 10
)

// Batch processing with transformation
let normalized: [Vector<Dim512>] = try await BatchOperations.process(vectors) { batch in
    try batch.map { try $0.normalized().get() }
}

// Pairwise distances (parallelized for large inputs)
let distances = await BatchOperations.pairwiseDistances(vectors)
```

### 5. **Flexible Distance Metrics**
Multiple distance metrics for different use cases:

```swift
// Built-in metrics
let euclidean = v1.euclideanDistance(to: v2)
let cosine = v1.cosineSimilarity(to: v2)
let dot = v1.dotProduct(v2)

// Use built-in metrics
let manhattan = ManhattanDistance()
let dist = manhattan.distance(v1, v2)

// Or create custom metrics
struct CustomMetric: DistanceMetric {
    typealias Scalar = Float
    func distance<V: VectorProtocol>(_ a: V, _ b: V) -> Float where V.Scalar == Float {
        // Your implementation
        return customCalculation(a, b)
    }
}
```

## 🛡️ Error Handling Philosophy

VectorCore follows Swift conventions: **fast by default, safe by opt-in**.

### Fast Path (Default)
```swift
// Standard operations use preconditions for maximum performance
let value = vector[10]                    // Fast subscript access
let v = try Vector<Dim128>(array)        // Fast, throwing initialization on mismatch
let result = v1 / scalar                  // Fast scalar division
let elementwise = v1 ./ v2               // Fast element-wise division (no zero check)
```

### Safe Path (Opt-in)
```swift
// Safe variants available when you need graceful error handling
let normalized = try v.normalized().get()       // Returns Result; throws on zero magnitude when unwrapped
let result = try Vector.safeDivide(v1, by: v2)  // Throws on division by zero
```

Choose the right tool for your use case - performance in hot paths, safety when handling untrusted input.

## 📖 API Overview

### Core Types

```swift
// Optimized vectors for common dimensions (fastest)
Vector512Optimized   // 512-dim with SIMD4 storage
Vector768Optimized   // 768-dim (BERT embeddings)
Vector1536Optimized  // 1536-dim (GPT embeddings)

// Generic fixed-dimension vectors (compile-time safe)
Vector<Dim128>   // 128-dimensional vector
Vector<Dim256>   // 256-dimensional vector
Vector<Dim512>   // 512-dimensional vector
Vector<Dim768>   // 768-dimensional vector
Vector<Dim1024>  // 1024-dimensional vector
Vector<Dim1536>  // 1536-dimensional vector

// Convenience type aliases
typealias Vector512 = Vector<Dim512>
typealias Vector768 = Vector<Dim768>
// ... and more

// Dynamic vectors (runtime dimensions)
DynamicVector([1.0, 2.0, 3.0, 4.0])
```

### Basic Operations

```swift
// Arithmetic
let sum = v1 + v2                        // Vector addition
let diff = v1 - v2                       // Vector subtraction
let scaled = v1 * 2.5                    // Scalar multiplication
let divided = v1 / 2.0                   // Scalar division

// Element-wise operations
let product = v1 .* v2                   // Element-wise multiplication
let quotient = v1 ./ v2                  // Element-wise division

// Vector operations
let magnitude = vector.magnitude
let normalized = vector.normalized()
let dotProduct = v1.dotProduct(v2)

// Distance metrics
let euclidean = v1.euclideanDistance(to: v2)
let cosine = v1.cosineSimilarity(to: v2)
```

### Batch Operations

```swift
// Auto-parallelized batch operations for large datasets
let nearest = await BatchOperations.findNearest(to: query, in: vectors, k: 10)
let distances = await BatchOperations.pairwiseDistances(vectors)

// Centroid and statistics
let centroid: Vector<Dim512> = Operations.centroid(of: vectors)
let stats = await BatchOperations.statistics(for: vectors)
```

### Provider Configuration

VectorCore uses `@TaskLocal` for zero-cost provider abstraction:

```swift
// Override SIMD provider (e.g., for custom Accelerate integration)
await Operations.$simdProvider.withValue(AccelerateSIMDProvider()) {
    let centroid = Operations.centroid(of: vectors)
    // All operations in this scope use Accelerate vDSP
}

// Override compute provider (e.g., for GPU via VectorAccelerate)
await Operations.$computeProvider.withValue(MetalComputeProvider()) {
    let results = try await Operations.findNearest(to: query, in: database, k: 100)
    // GPU-accelerated search
}

// Multiple provider overrides
await Operations.$simdProvider.withValue(AccelerateSIMDProvider()) {
    await Operations.$computeProvider.withValue(CPUComputeProvider.automatic) {
        // Fully customized execution environment
        let normalized = try await Operations.normalize(vectors)
    }
}
```

### Error Handling

```swift
// Safe operations with error handling
do {
    let normalized = try vector.normalized().get()
} catch let error as VectorError {
    print("Error: \(error.localizedDescription)")
}
```

## ⚡ Performance Characteristics

VectorCore's optimized implementations deliver exceptional performance:

### Performance Optimization

The optimized vector types (`Vector512Optimized`, `Vector768Optimized`, `Vector1536Optimized`) provide significant performance improvements over generic vectors thanks to:
- SIMD4<Float> storage layout for optimal vectorization
- 4x loop unrolling with multiple accumulators to maximize instruction-level parallelism
- Cache-line optimized memory access patterns
- Specialized implementations for common dimensions (512, 768, 1536)
- Zero-copy operations where possible
- Accelerate framework integration for large operations

### Performance Notes

- Optimized vectors can be **5-10x faster** than generic implementations for core operations
- Performance varies based on hardware (Apple Silicon vs Intel) and dimension size
- The repository includes comprehensive test suites that validate performance characteristics
- For detailed benchmarks, profile your specific use case with Instruments

## 📱 Requirements

- **Swift:** 6.0+
- **Swift Language Mode**: Swift 6 with **StrictConcurrency** enabled
- **Concurrency**: Full `Sendable` conformance for all public types
- **Platforms:**
  - macOS 14.0+
  - iOS 17.0+
  - tvOS 17.0+
  - watchOS 10.0+
  - visionOS 1.0+

### Concurrency Safety

VectorCore is built with **Swift 6 Strict Concurrency** from the ground up:

- ✅ All public types conform to `Sendable` (data race safety)
- ✅ Async operations use structured concurrency (`async`/`await`)
- ✅ Provider configuration via `@TaskLocal` (zero-cost, thread-safe)
- ✅ Buffer pooling via `actor` (isolation guarantees)
- ✅ No global mutable state (except thread-safe caches)

**Migration from Swift 5**:
If your project uses Swift 5.x, you may see warnings about `Sendable` conformance. To migrate:

```swift
// Enable strict concurrency in your Package.swift
swiftSettings: [
    .enableExperimentalFeature("StrictConcurrency")
]

// Or in Xcode: Build Settings → Swift Compiler → Strict Concurrency Checking → Complete
```

See [Swift Evolution SE-0337](https://github.com/apple/swift-evolution/blob/main/proposals/0337-support-incremental-migration-to-concurrency-checking.md) for details.

## 📚 Documentation

### Guides

- **[Package Boundaries](Docs/Package_Boundaries.md)** - Understanding the 4-package architecture (Core, Index, Accelerate, Store) and when to use each
- **[Performance Guide](Docs/Performance_Guide.md)** - Optimized vector types, provider tuning, benchmarking, and performance best practices
- **[Memory Alignment](Docs/Memory_Alignment.md)** - Aligned allocation/deallocation, SIMD requirements, and avoiding common pitfalls

### API Documentation

- **Vector Types**: `Vector<D>`, `DynamicVector`, `Vector{512,768,1536}Optimized`
- **Operations**: `Operations` (primary API), `BatchOperations` (auto-parallelized batches)
- **Distance Metrics**: `EuclideanDistance`, `CosineDistance`, `ManhattanDistance`, `ChebyshevDistance`, `HammingDistance`, `MinkowskiDistance`, `DotProductDistance`
- **Providers**: `SIMDProvider`, `ComputeProvider`, `BufferProvider` (plug-in architecture via `@TaskLocal`)

## 🤝 Contributing

Contributions are welcome! Please open an issue or pull request with a clear description and minimal reproduction where applicable. Coding style aims for clarity, performance, and Swift 6 concurrency safety.

## 📄 License

VectorCore is released under the MIT License. See [LICENSE](LICENSE) for details.
