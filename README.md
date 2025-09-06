# VectorCore

[![Swift](https://img.shields.io/badge/Swift-6.0-orange.svg?style=flat)](https://swift.org)
[![Platforms](https://img.shields.io/badge/Platforms-macOS%2014%2B%20|%20iOS%2017%2B%20|%20tvOS%2017%2B%20|%20watchOS%2010%2B%20|%20visionOS%201%2B-blue.svg?style=flat)](https://swift.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat)](LICENSE)
[![SPM](https://img.shields.io/badge/SPM-compatible-brightgreen.svg?style=flat)](https://swift.org/package-manager/)

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

## 📦 Installation

Add VectorCore to your Swift Package Manager dependencies:

```swift
dependencies: [
    .package(url: "https://github.com/<owner>/VectorCore.git", from: "0.1.0")
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

// Custom metrics
struct ManhattanDistance: DistanceMetric {
    func distance<V: VectorProtocol>(_ a: V, _ b: V) -> Float where V.Scalar == Float {
        // Your implementation
        (a - b).l1Norm
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
let result = v1 ./ v2                    // Fast division (no zero check)
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
let sum = v1 + v2
let diff = v1 - v2
let scaled = v1 * 2.5
let divided = v1 / 2.0

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
// Async batch operations
let nearest = await BatchOperations.findNearest(to: query, in: vectors, k: 10)
let distances = await BatchOperations.pairwiseDistances(vectors)

// Synchronous utilities
let centroid = SyncBatchOperations.centroid(of: vectors)
let sum = SyncBatchOperations.sum(vectors)
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

### Optimized Vector Performance (Apple Silicon M-series)

| Operation | Vector512Optimized | Vector768Optimized | Vector1536Optimized |
|-----------|-------------------|-------------------|---------------------|
| Dot Product | ~12ns | ~18ns | ~36ns |
| Euclidean Distance | ~16ns | ~24ns | ~48ns |
| Normalization | ~45ns | ~67ns | ~135ns |
| Addition | ~8ns | ~12ns | ~24ns |
| Cosine Similarity | ~25ns | ~37ns | ~75ns |

### Generic Vector Performance

| Operation | 512-dim | 768-dim | 1536-dim |
|-----------|---------|---------|----------|
| Dot Product | ~100ns | ~150ns | ~300ns |
| Euclidean Distance | ~120ns | ~180ns | ~360ns |
| Normalization | ~150ns | ~225ns | ~450ns |

The optimized vectors are **8-10x faster** than generic implementations thanks to:
- SIMD4<Float> storage layout
- 4x loop unrolling with multiple accumulators
- Cache-line optimized memory access
- Inline assembly hints for critical paths

### Performance Notes

The repository includes performance-oriented implementations and test suites. Formal benchmark runners and baseline comparison tools are not included as CLI targets in this package.

## 📱 Requirements

- **Swift:** 6.0+
- **Platforms:**
  - macOS 14.0+
  - iOS 17.0+
  - tvOS 17.0+
  - watchOS 10.0+
  - visionOS 1.0+

## 🤝 Contributing

Contributions are welcome! Please open an issue or pull request with a clear description and minimal reproduction where applicable. Coding style aims for clarity, performance, and Swift 6 concurrency safety.

## 📄 License

VectorCore is released under the MIT License. See [LICENSE](LICENSE) for details.

---

Built with ❤️ for the Swift community by the VectorCore Contributors
