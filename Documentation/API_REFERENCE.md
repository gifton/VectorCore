# VectorCore API Reference

## Overview

VectorCore is the foundational package for vector operations in VectorStoreKit. It provides high-performance vector types, distance metrics, and core protocols with zero external dependencies.

## Core Types

### Vector<D: Dimension>

The generic vector type that supports compile-time dimension checking and optimal storage selection.

```swift
// Create vectors with type aliases
let v128 = Vector128(repeating: 1.0)
let v256 = Vector256([1.0, 2.0, 3.0, ...]) // 256 elements

// Access elements
let firstElement = v128[0]
v128[0] = 2.0

// Mathematical operations
let sum = v1 + v2
let difference = v1 - v2
let scaled = v1 * 2.0
let divided = v1 / 2.0

// Vector operations
let dotProduct = v1.dotProduct(v2)
let magnitude = v1.magnitude
let normalized = v1.normalized()
let distance = v1.distance(to: v2)
let cosineSim = v1.cosineSimilarity(to: v2)
```

#### Type Aliases
- `Vector128` - 128-dimensional vector
- `Vector256` - 256-dimensional vector  
- `Vector512` - 512-dimensional vector
- `Vector768` - 768-dimensional vector
- `Vector1536` - 1536-dimensional vector

### DynamicVector

A vector type with runtime-determined dimensions for flexibility.

```swift
// Create with any dimension
let vector = DynamicVector(dimension: 100, repeating: 1.0)
let fromArray = DynamicVector([1.0, 2.0, 3.0, ...])

// All standard operations supported
let dot = v1.dotProduct(v2)
let normalized = vector.normalized()
```

### VectorFactory

Factory methods for creating vectors with optimal type selection.

```swift
// Create vector with automatic type selection
let vector = try VectorFactory.vector(of: 256, from: values)

// Convenience methods
let zeros = VectorFactory.zeros(dimension: 128)
let ones = VectorFactory.ones(dimension: 256)
let random = VectorFactory.random(dimension: 512, range: -1...1)
let normalized = VectorFactory.randomNormalized(dimension: 768)

// Create basis vectors (one-hot)
let basis = try VectorFactory.basis(dimension: 128, index: 5)

// Batch creation
let batch = try VectorFactory.batch(dimension: 128, from: flatArray)
```

## Distance Metrics

### Built-in Metrics

All distance metrics implement the `DistanceMetric` protocol and are optimized for performance.

```swift
// Euclidean distance (L2 norm)
let euclidean = EuclideanDistance()
let distance = euclidean.distance(v1, v2)

// Cosine distance (1 - cosine similarity)
let cosine = CosineDistance()
let distance = cosine.distance(v1, v2)

// Dot product distance (negative for similarity to distance conversion)
let dotProduct = DotProductDistance()
let distance = dotProduct.distance(v1, v2)

// Manhattan distance (L1 norm)
let manhattan = ManhattanDistance()
let distance = manhattan.distance(v1, v2)

// Other metrics
let hamming = HammingDistance()
let chebyshev = ChebyshevDistance()
let minkowski = MinkowskiDistance(p: 3.0)
let jaccard = JaccardDistance(threshold: 0.01)
```

### Batch Operations

All batch operations are async and automatically parallelize for large datasets:

```swift
// Find k-nearest neighbors (async)
let neighbors = await BatchOperations.findNearest(
    to: query,
    in: vectors,
    k: 10,
    metric: EuclideanDistance()
)

// Compute pairwise distances (async)
let distances = await BatchOperations.pairwiseDistances(
    vectors,
    metric: CosineDistance()
)

// Transform vectors with automatic parallelization (async)
let transformed = try await BatchOperations.map(vectors) { vector in
    vector.normalized()
}

// Filter vectors (async)
let filtered = try await BatchOperations.filter(vectors) { vector in
    vector.magnitude > threshold
}

// Compute statistics (async)
let stats = await BatchOperations.statistics(for: vectors)

// Random sampling (synchronous - no benefit from parallelization)
let sampled = BatchOperations.sample(vectors, k: 100)
```

#### Auto-Parallelization Behavior

- Operations automatically parallelize when datasets exceed 1000 vectors
- Pairwise distances parallelize at 100 vectors (due to O(n²) complexity)
- Configure thresholds via `BatchOperations.configuration`

## Protocols

### BaseVectorProtocol

Base protocol for vector operations without SIMD requirement.

```swift
public protocol BaseVectorProtocol: Sendable {
    associatedtype Scalar: BinaryFloatingPoint
    static var dimensions: Int { get }
    var scalarCount: Int { get }
    init(from array: [Scalar])
    func toArray() -> [Scalar]
    subscript(index: Int) -> Scalar { get }
}
```

### ExtendedVectorProtocol

Extended protocol with mathematical operations.

```swift
public protocol ExtendedVectorProtocol: BaseVectorProtocol {
    func dotProduct(_ other: Self) -> Float
    var magnitude: Float { get }
    func normalized() -> Self
    func distance(to other: Self) -> Float
    func cosineSimilarity(to other: Self) -> Float
}
```

### DistanceMetric

Protocol for implementing custom distance metrics.

```swift
public protocol DistanceMetric: Sendable {
    var identifier: String { get }
    func distance<Vector: ExtendedVectorProtocol>(_ a: Vector, _ b: Vector) -> DistanceScore
    func batchDistance<Vector: ExtendedVectorProtocol>(query: Vector, candidates: [Vector]) -> [DistanceScore]
}
```

## Performance Considerations

### Storage Strategy

VectorCore automatically selects optimal storage based on dimension:
- 1-32 dimensions: `SIMD32<Float>`
- 33-64 dimensions: `SIMD64<Float>`
- 65-128 dimensions: 2×`SIMD64<Float>`
- 129-256 dimensions: 4×`SIMD64<Float>`
- 257+ dimensions: Contiguous array with aligned access

### Optimization Tips

1. **Use type aliases** for common dimensions to benefit from SIMD optimization
2. **Batch operations** when processing multiple vectors
3. **Reuse vector instances** when possible to avoid allocations
4. **Use `@inlinable`** functions for hot paths
5. **Prefer dot product** over cosine similarity when vectors are pre-normalized

## Error Handling

```swift
// VectorCoreError provides structured error information
do {
    let vector = try VectorFactory.vector(of: 128, from: values)
} catch let error as VectorCoreError {
    print("Error: \(error.code) - \(error.message)")
    if let underlying = error.underlyingError {
        print("Underlying: \(underlying)")
    }
}
```

Common error codes:
- `DIMENSION_MISMATCH` - Vector dimensions don't match
- `INVALID_PARAMETER` - Invalid parameter provided
- `INDEX_OUT_OF_BOUNDS` - Array index out of bounds
- `SERIALIZATION_FAILED` - Serialization/deserialization error

## Serialization

Vectors support `Codable` for easy serialization:

```swift
// JSON encoding
let encoder = JSONEncoder()
let data = try encoder.encode(vector)

// JSON decoding
let decoder = JSONDecoder()
let vector = try decoder.decode(Vector256.self, from: data)

// Archive to file
let url = URL(fileURLWithPath: "vector.json")
try data.write(to: url)
```

## Platform Support

VectorCore supports:
- macOS 14.0+
- iOS 17.0+
- tvOS 17.0+
- watchOS 10.0+
- visionOS 1.0+

All operations are optimized for Apple Silicon and Intel processors using:
- SIMD intrinsics
- Accelerate framework (vDSP)
- Compiler optimizations