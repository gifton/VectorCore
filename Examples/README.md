# VectorCore Examples

This directory contains comprehensive examples demonstrating how to use VectorCore effectively in various scenarios.

## 📚 Example Categories

### 1. Getting Started
- **File**: `01_GettingStarted.swift`
- **Topics**: Creating vectors, basic operations, mathematical functions, distance metrics, quality assessment
- **Perfect for**: New users learning VectorCore basics

### 2. Advanced Operations
- **File**: `02_AdvancedOperations.swift`
- **Topics**: Element-wise operations, special functions, batch processing, serialization, performance patterns
- **Perfect for**: Users ready to explore advanced features

### 3. Machine Learning Use Cases
- **File**: `03_MachineLearningUseCases.swift`
- **Topics**: Text embeddings, image similarity, clustering, recommendation systems, anomaly detection
- **Perfect for**: ML/AI developers using VectorCore for embeddings

### 4. Error Handling & Robustness
- **File**: `04_ErrorHandlingAndRobustness.swift`
- **Topics**: Non-finite value handling, safe operations, dimension validation, serialization errors, defensive programming
- **Perfect for**: Building production-ready applications

### 5. Performance Optimization
- **File**: `05_PerformanceOptimization.swift`
- **Topics**: Vector type selection, batch processing, memory efficiency, SIMD optimization, caching strategies
- **Perfect for**: Optimizing performance-critical code

### 6. Integration Patterns
- **File**: `06_IntegrationPatterns.swift`
- **Topics**: Vector databases, embedding services, recommendation engines, real-time search
- **Perfect for**: Integrating VectorCore into larger applications

## 🚀 Running the Examples

### Individual Examples
```bash
# Navigate to VectorCore directory
cd /path/to/VectorCore

# Run a specific example
swift run --package-path . 01_GettingStarted
```

### All Examples
```bash
# Run all examples in sequence
for i in {1..6}; do
    echo "Running Example $i..."
    swift run --package-path . 0${i}_*
    echo -e "\n\n"
done
```

### As Swift Package
Add to your `Package.swift`:
```swift
.executableTarget(
    name: "MyVectorApp",
    dependencies: [
        .product(name: "VectorCore", package: "VectorCore")
    ]
)
```

## 📖 Quick Start Guide

### Creating Vectors
```swift
// Using type aliases (recommended)
let v128 = Vector128.random(in: -1...1)
let v256 = Vector256.zeros()
let v768 = Vector768(repeating: 0.5)

// From arrays
let values = [1.0, 2.0, 3.0, ...] // 512 values
let vector = Vector512(values)

// Dynamic vectors
let dynamic = DynamicVector(dimension: 1000)
```

### Basic Operations
```swift
let v1 = Vector256.random(in: -1...1)
let v2 = Vector256.random(in: -1...1)

// Arithmetic
let sum = v1 + v2
let diff = v1 - v2
let scaled = v1 * 2.0

// Mathematical operations
let dot = v1.dotProduct(v2)
let magnitude = v1.magnitude
let normalized = v1.normalized()

// Distance metrics
let euclidean = v1.distance(to: v2)
let cosine = v1.cosineSimilarity(to: v2)
```

### Quality Assessment
```swift
let vector = Vector512.random(in: -1...1)
let quality = vector.quality

print("Magnitude: \(quality.magnitude)")
print("Sparsity: \(quality.sparsity * 100)%")
print("Entropy: \(quality.entropy)")
print("Quality Score: \(quality.score)")
```

## 🏃‍♂️ Performance Tips

1. **Use strongly-typed vectors** when dimensions are known at compile time
2. **Pre-normalize vectors** for multiple similarity computations
3. **Use batch operations** for processing multiple vectors
4. **Cache expensive computations** like magnitude and normalization
5. **Choose appropriate distance metrics** (dot product is fastest)

## 🔧 Common Patterns

### Similarity Search
```swift
let query = Vector768.random(in: -1...1).normalized()
let database = loadVectors() // [Vector768]

let results = database
    .map { (vector: $0, similarity: query.cosineSimilarity(to: $0)) }
    .sorted { $0.similarity > $1.similarity }
    .prefix(10)
```

### Batch Processing
```swift
let batchOps = SyncBatchOperations()
let distances = batchOps.batchDistance(
    from: query,
    to: candidates,
    using: EuclideanDistance()
)
```

### Error Handling
```swift
do {
    let vector = try VectorFactory.create(Dim128.self, from: data)
} catch let error as VectorError {
    print("Error: \(error.localizedDescription)")
    print("Recovery: \(error.recoverySuggestion ?? "None")")
}
```

## 📚 Additional Resources

- [API Reference](../Documentation/API_REFERENCE.md)
- [Performance Guide](../Documentation/PERFORMANCE_CHARACTERISTICS.md)
- [SIMD Optimization Guide](../Documentation/SIMD_OPTIMIZATION_GUIDE.md)

## 🤝 Contributing

Have an interesting use case or example? We'd love to include it! Please see our [Contributing Guidelines](../CONTRIBUTING.md) for details on submitting new examples.

## 📝 License

All examples are provided under the same license as VectorCore. See the main [LICENSE](../LICENSE) file for details.