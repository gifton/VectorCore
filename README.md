# VectorCore

[![Swift](https://img.shields.io/badge/Swift-6.0-orange.svg?style=flat)](https://swift.org)
[![Platforms](https://img.shields.io/badge/Platforms-macOS%2014%2B%20|%20iOS%2017%2B%20|%20tvOS%2017%2B%20|%20watchOS%2010%2B%20|%20visionOS%201%2B-blue.svg?style=flat)](https://swift.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat)](LICENSE)
[![SPM](https://img.shields.io/badge/SPM-compatible-brightgreen.svg?style=flat)](https://swift.org/package-manager/)
[![CI](https://github.com/yourusername/VectorCore/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/VectorCore/actions/workflows/ci.yml)
[![Benchmarks](https://github.com/yourusername/VectorCore/actions/workflows/benchmarks.yml/badge.svg)](https://github.com/yourusername/VectorCore/actions/workflows/benchmarks.yml)
[![codecov](https://codecov.io/gh/yourusername/VectorCore/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/VectorCore)

High-performance, type-safe vector operations for Swift with zero dependencies. VectorCore provides blazing-fast vector math optimized for machine learning, scientific computing, and real-time applications.

## 🚀 Why VectorCore?

VectorCore is designed from the ground up for performance and ease of use:

- **Zero-allocation operations** - All core operations are stack-allocated
- **Hardware-optimized** - Leverages SIMD, Accelerate framework, and cache-aligned memory
- **Type-safe dimensions** - Compile-time dimension checking prevents runtime errors
- **Automatic parallelization** - Batch operations scale across CPU cores
- **Zero dependencies** - Pure Swift with no external dependencies
- **Comprehensive API** - Full suite of vector operations, distance metrics, and utilities

Whether you're building ML pipelines, processing embeddings, or need fast numerical computing, VectorCore delivers the performance you need with an API you'll love.

## 📦 Installation

Add VectorCore to your Swift Package Manager dependencies:

```swift
dependencies: [
    .package(url: "https://github.com/yourusername/VectorCore.git", from: "0.1.0")
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

// Create vectors with compile-time dimension safety
let v1 = Vector512.random(in: -1...1)
let v2 = Vector512.random(in: -1...1)

// All operations are zero-allocation and optimized
let distance = v1.distance(to: v2)
let similarity = v1.cosineSimilarity(to: v2)
let normalized = v1.normalized()

// Batch operations automatically parallelize
let vectors = (0..<10_000).map { _ in Vector512.random(in: -1...1) }
let query = Vector512.random(in: -1...1)

// Find k-nearest neighbors (automatically parallel for large datasets)
let nearest = await vectors.findNearest(to: query, k: 100)
```

## ✨ Key Features

### 1. **Type-Safe Dimensions**
Prevent dimension mismatch errors at compile time:

```swift
let bert = Vector768.randomNormalized()     // BERT embeddings
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
// Automatically uses Accelerate framework for best performance
let magnitude = vector.magnitude           // Uses vDSP
let normalized = vector.normalized()       // SIMD-optimized
let distance = v1.euclideanDistance(to: v2) // Cache-aligned
```

### 4. **Automatic Parallelization**
Large batch operations scale across cores:

```swift
// Automatically parallel for datasets > 1000 vectors
let results = await BatchOperations.map(largeDataset) { vector in
    vector.normalized()
}

// Parallel similarity search
let similar = await BatchOperations.findSimilar(
    to: query,
    in: database,
    threshold: 0.8
)
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
    func distance<V: ExtendedVectorProtocol>(_ a: V, _ b: V) -> Float {
        // Your implementation
    }
}
```

## 🛡️ Error Handling Philosophy

VectorCore follows Swift conventions: **fast by default, safe by opt-in**.

### Fast Path (Default)
```swift
// Standard operations use preconditions for maximum performance
let value = vector[10]                    // Fast subscript access
let v = Vector<Dim128>(array)            // Fast initialization
let result = v1 ./ v2                    // Fast division (no zero check)
```

### Safe Path (Opt-in)
```swift
// Safe variants available when you need graceful error handling
if let value = vector.at(10) { }         // Returns nil if out of bounds
if let v = Vector<Dim128>(safe: array) { } // Returns nil if wrong dimension
let result = try Vector.safeDivide(v1, by: v2) // Throws on division by zero
```

Choose the right tool for your use case - performance in hot paths, safety when handling untrusted input.

## 📖 API Overview

### Core Types

```swift
// Fixed-dimension vectors (compile-time safe)
Vector<Dim128>   // 128-dimensional vector
Vector<Dim256>   // 256-dimensional vector
Vector<Dim512>   // 512-dimensional vector
Vector<Dim768>   // 768-dimensional vector (BERT)
Vector<Dim1024>  // 1024-dimensional vector
Vector<Dim1536>  // 1536-dimensional vector (GPT)

// Convenience type aliases
typealias Vector128 = Vector<Dim128>
typealias Vector256 = Vector<Dim256>
// ... and more

// Dynamic vectors (runtime dimensions)
DynamicVector(dimension: 384)
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
// Parallel transformations
let normalized = await BatchOperations.map(vectors) { $0.normalized() }

// Similarity search
let nearest = await BatchOperations.findNearest(to: query, in: vectors, k: 10)
let similar = await BatchOperations.findSimilar(to: query, in: vectors, threshold: 0.9)

// Reductions
let average = await BatchOperations.average(vectors)
let sum = await BatchOperations.sum(vectors)
```

### Error Handling

```swift
// Safe operations with error handling
do {
    let normalized = try vector.normalizedThrowing()
} catch VectorError.zeroMagnitude {
    // Handle zero vector case
}

// Rich error information
catch let error as VectorError {
    print("Error: \(error.localizedDescription)")
    print("Context: \(error.context)")
}
```

## ⚡ Performance Characteristics

VectorCore is optimized for real-world performance:

| Operation | 512-dim Vectors | Performance |
|-----------|----------------|-------------|
| Addition | v1 + v2 | ~50ns |
| Dot Product | v1 · v2 | ~80ns |
| Euclidean Distance | ‖v1 - v2‖ | ~100ns |
| Cosine Similarity | cos(θ) | ~120ns |
| Normalization | v/‖v‖ | ~150ns |

Batch operations scale linearly with core count:
- 10K vector search: ~5ms on Apple Silicon
- 100K vector normalization: ~50ms with 8 cores
- 1M vector filtering: ~200ms parallelized

## 📱 Requirements

- **Swift:** 6.0+
- **Platforms:**
  - macOS 14.0+
  - iOS 17.0+
  - tvOS 17.0+
  - watchOS 10.0+
  - visionOS 1.0+

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

VectorCore is released under the MIT License. See [LICENSE](LICENSE) for details.

---

Built with ❤️ for the Swift community by the VectorCore Contributors