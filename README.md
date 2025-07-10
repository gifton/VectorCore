# VectorCore

The foundational Swift package for high-performance vector operations, providing zero-dependency vector types, distance metrics, and mathematical operations optimized for Apple platforms.

## Overview

VectorCore is the foundation of the VectorStoreKit ecosystem, providing:

- 🚀 **Generic vector types** with compile-time dimension safety
- 📏 **Optimized distance metrics** (Euclidean, Cosine, Manhattan, etc.)
- 🧮 **Mathematical operations** with Accelerate framework integration
- 💾 **SIMD-optimized storage** for common dimensions
- 🔒 **Type-safe API** with Swift 6 concurrency support
- 📦 **Zero dependencies** - pure Swift implementation

## Installation

### Swift Package Manager

Add VectorCore to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/yourusername/VSK.git", from: "0.1.0")
],
targets: [
    .target(
        name: "YourTarget",
        dependencies: [
            .product(name: "VectorCore", package: "VSK")
        ]
    )
]
```

## Quick Start

```swift
import VectorCore

// Create vectors using type aliases (recommended)
let embedding1 = Vector256.random(in: -1...1)
let embedding2 = Vector256([1.0, 2.0, 3.0, ...]) // 256 values

// Or use dynamic vectors for runtime dimensions
let dynamicVector = DynamicVector(dimension: 384, repeating: 0.5)

// Mathematical operations
let sum = embedding1 + embedding2
let difference = embedding1 - embedding2
let scaled = embedding1 * 2.5
let normalized = embedding1.normalized()

// Distance calculations
let euclideanDist = embedding1.distance(to: embedding2)
let cosineSim = embedding1.cosineSimilarity(to: embedding2)

// Use specific distance metrics
let metric = EuclideanDistance()
let distance = metric.distance(embedding1, embedding2)

// Batch operations for efficiency (async API)
let vectors = (0..<1000).map { _ in Vector<Dim256>.random(in: -1...1) }
let neighbors = await BatchOperations.findNearest(
    to: query,
    in: vectors,
    k: 10,
    metric: CosineDistance()
)
```

## Architecture

VectorCore is the foundation of the VectorStoreKit ecosystem:

```
VectorCore (You are here)
    ↑
    ├── VectorAccelerate (Hardware acceleration)
    ├── VectorIndex (Search algorithms & quantization)
    ├── VectorStore (Storage & persistence)
    └── VectorAI (ML & intelligence)
```

## Supported Vector Types

### Vector Types

VectorCore provides optimized vector types for common dimensions:

```swift
// Standard vector types (recommended)
Vector128   // 128 dimensions
Vector256   // 256 dimensions  
Vector512   // 512 dimensions
Vector768   // 768 dimensions (BERT embeddings)
Vector1536  // 1536 dimensions (GPT embeddings)
Vector3072  // 3072 dimensions
```

**Advanced**: For custom dimensions, you can use the generic syntax:
```swift
// Define custom dimension
struct Dim384: Dimension {
    static let value = 384
    typealias Storage = DynamicArrayStorage
}

// Use with generic syntax
let customVector = Vector<Dim384>()
```

### Dynamic Vectors
- `DynamicVector` - Runtime-determined dimensions
- `VectorFactory` - Automatic type selection based on dimension

## Distance Metrics

All metrics are optimized for performance:

- **Euclidean Distance** - L2 norm, standard geometric distance
- **Cosine Distance** - Angular distance, great for embeddings
- **Dot Product** - Inner product (negative for similarity)
- **Manhattan Distance** - L1 norm, city-block distance
- **Chebyshev Distance** - L∞ norm, maximum coordinate difference
- **Hamming Distance** - Count of differing elements
- **Minkowski Distance** - Generalized Lp norm
- **Jaccard Distance** - Set similarity for binary vectors

## Performance

VectorCore achieves exceptional performance through:

- SIMD intrinsics for parallel computation
- Accelerate framework integration (vDSP)
- Cache-friendly memory layouts
- Zero-copy operations where possible
- Optimal storage selection based on dimension

Key optimizations:
- Stack allocation for vectors ≤ 256 dimensions
- SIMD32/SIMD64 types for small vectors
- Aligned memory for large vectors
- Batch processing with parallelization

## Advanced Features

### Generic Vector Implementation

VectorCore uses a sophisticated generic design:

```swift
// Single generic type for all fixed dimensions
public struct Vector<D: Dimension>: Sendable {
    internal var storage: D.Storage
}

// Dimension protocol for compile-time safety
public protocol Dimension {
    static var value: Int { get }
    associatedtype Storage: VectorStorage
}

// Predefined dimensions with optimal storage
public struct Dim768: Dimension {
    public static let value = 768
    public typealias Storage = SIMDStorage768
}
```

Benefits:
- Compile-time dimension checking
- Automatic storage optimization
- Zero runtime overhead
- Type-safe operations

### Batch Operations (Async API)
```swift
// Find k-nearest neighbors
let results = await BatchOperations.findNearest(
    to: queryVector,
    in: vectorDatabase,
    k: 10,
    metric: CosineDistance()
)

// Compute pairwise distances efficiently
let distances = await BatchOperations.pairwiseDistances(vectors)

// Get statistics
let stats = await BatchOperations.statistics(for: vectors)
```

### Vector Math Extensions
```swift
// Element-wise operations
let hadamard = vector1 .* vector2  // Element-wise multiplication
let divided = vector1 ./ vector2    // Element-wise division

// Norms
let l1Norm = vector.l1Norm
let l2Norm = vector.magnitude      // Euclidean norm
let lInfNorm = vector.lInfinityNorm

// Statistical operations (on DynamicVector)
let mean = dynamicVector.mean
let variance = dynamicVector.variance
let stdDev = dynamicVector.standardDeviation
```

### Custom Dimensions
```swift
// Define your own dimensions
struct Dim384: Dimension {
    static let value = 384
    typealias Storage = DynamicArrayStorage
}

// Use it like built-in types
let customVector = Vector<Dim384>()
```

## API Guidelines

### When to Use Each Type

- **Standard Vector Types** (e.g., `Vector768`): Recommended for most use cases
  - Cleaner, more readable code
  - Better IDE autocompletion
  - Covers all common embedding dimensions
  - Best performance with compile-time optimizations

- **`DynamicVector`**: When dimension is determined at runtime
  - Flexible dimension support
  - Slightly lower performance due to heap allocation
  - Use for user-defined or variable dimensions

- **Generic Syntax** (e.g., `Vector<Dim768>`): Advanced use cases only
  - Creating custom dimensions
  - Library internals and extensions
  - When you need to work with the Dimension protocol directly

## Design Principles

1. **Zero Dependencies**: No external packages required
2. **Platform Agnostic**: Works on all Swift platforms
3. **Protocol-Oriented**: Extensible architecture
4. **Performance Focused**: SIMD and hardware acceleration
5. **Type Safe**: Strong typing with compile-time checks

## API Design Rationale

### Why Type Aliases?

VectorCore provides both type aliases (`Vector768`) and generic syntax (`Vector<Dim768>`), but **type aliases are the recommended approach** for several reasons:

1. **Developer Experience**: Cleaner, more readable code with better IDE support
2. **Industry Convention**: Matches Swift patterns like `Float32`, `Int64`, `SIMD32`
3. **Simplicity**: Reduces cognitive load and typing
4. **Discoverability**: Easier to explore available vector types

The generic syntax remains available for advanced use cases like custom dimensions, but for the standard dimensions used in machine learning (128, 256, 512, 768, 1536, 3072), type aliases provide the best experience.

## Requirements

- Swift 6.0+ 
- macOS 14.0+ / iOS 17.0+ / tvOS 17.0+ / watchOS 10.0+ / visionOS 1.0+
- No external dependencies

## Current Limitations

- No built-in serialization (planned for v0.2.0)
- Limited initialization options (improvements planned)
- Async-only batch operations (sync alternatives being considered)

## Performance Benchmarks

VectorCore includes comprehensive performance benchmarks in a separate target to keep the core library lean.

### Running Benchmarks

```bash
# Run all benchmarks
swift run VectorCoreBenchmarks

# Run specific suites
swift run VectorCoreBenchmarks --simd
swift run VectorCoreBenchmarks --parallel

# Run in release mode for accurate results
swift run -c release VectorCoreBenchmarks
```

See [Benchmarks/README.md](Benchmarks/README.md) for detailed information.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

VectorCore is available under the MIT license. See the LICENSE file for details.

## Version History

### v0.1.0 (Current Development)
- Initial release
- Generic vector implementation
- SIMD-optimized storage
- Complete distance metrics
- Async batch operations
- Swift 6 concurrency support

---

Built with ❤️ for the Swift community