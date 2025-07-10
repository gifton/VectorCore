// VectorCore: Basic Usage Example
//
// Demonstrates the new extension-based API

import VectorCore
import Foundation

// Example: Working with vectors using the new API
func demonstrateNewAPI() {
    print("=== VectorCore New API Demo ===\n")
    
    // Create some vectors
    let vector1 = SIMD32<Float>(repeating: 1.0)
    let vector2 = SIMD32<Float>(repeating: 2.0)
    
    print("Vector 1: \(vector1.toString(precision: 1))")
    print("Vector 2: \(vector2.toString(precision: 1))")
    
    // Mathematical operations - now properties!
    print("\n--- Mathematical Operations ---")
    print("Magnitude of vector1: \(vector1.magnitude)")
    print("Mean of vector1: \(vector1.mean)")
    print("Standard deviation: \(vector1.standardDeviation)")
    
    // Normalization
    let randomVector = SIMD32<Float>((0..<32).map { _ in Float.random(in: -1...1) })
    let normalized = randomVector.normalized
    print("\nNormalized vector magnitude: \(normalized.magnitude)")
    
    // Distance calculations
    print("\n--- Distance Calculations ---")
    let euclideanDist = vector1.distance(to: vector2)
    let manhattanDist = vector1.distance(to: vector2, using: ManhattanDistance())
    print("Euclidean distance: \(euclideanDist)")
    print("Manhattan distance: \(manhattanDist)")
    
    // Quality assessment
    print("\n--- Quality Assessment ---")
    let sparseVector = SIMD32<Float>((0..<32).map { i in i % 4 == 0 ? Float(i) : 0 })
    print("Sparsity: \(sparseVector.sparsity)")
    print("Entropy: \(sparseVector.entropy)")
    let quality = sparseVector.quality
    print("Quality metrics: magnitude=\(quality.magnitude), variance=\(quality.variance)")
    
    // Validation
    print("\n--- Validation ---")
    print("Is normalized? \(normalized.isNormalized())")
    print("Is valid? \(normalized.isValid)")
    print("Has 32 dimensions? \(normalized.hasDimensions(32))")
    
    // Serialization
    print("\n--- Serialization ---")
    let base64 = vector1.base64Encoded
    print("Base64 encoded (first 50 chars): \(String(base64.prefix(50)))...")
}

// Example: Using the VectorFactory
func demonstrateFactory() throws {
    print("\n=== VectorFactory Demo ===\n")
    
    // Create vectors of different dimensions
    let vec128 = try VectorFactory.zeros(dimension: 128)
    let vec256 = try VectorFactory.ones(dimension: 256)
    let vec512 = try VectorFactory.random(dimension: 512)
    
    print("Created vectors:")
    print("- 128D zeros: magnitude = \(vec128.magnitude)")
    print("- 256D ones: magnitude = \(vec256.magnitude)")
    print("- 512D random: magnitude = \(vec512.magnitude)")
    
    // Pattern-based creation
    let pattern = try VectorFactory.withPattern(dimension: 768) { i in
        Float(i) / 768.0
    }
    print("- 768D pattern: mean = \(pattern.mean)")
    
    // Optimal dimension selection
    let optimal = VectorFactory.optimalDimension(for: 1000)
    print("\nOptimal dimension for ~1000: \(optimal)")
}

// Example: Batch operations
func demonstrateBatchOps() async {
    print("\n=== Batch Operations Demo ===\n")
    
    // Create a batch of vectors
    let vectors = (0..<100).map { _ in
        SIMD32<Float>((0..<32).map { _ in Float.random(in: -1...1) })
    }
    
    // Find nearest neighbors
    let query = vectors[0]
    let nearest = await BatchOperations.findNearest(to: query, in: vectors, k: 5)
    
    print("Found \(nearest.count) nearest neighbors:")
    for (index, distance) in nearest {
        print("  Vector \(index): distance = \(distance)")
    }
    
    // Batch statistics
    let stats = await BatchOperations.statistics(for: vectors)
    print("\nBatch statistics:")
    print("- Count: \(stats.count)")
    print("- Mean magnitude: \(stats.meanMagnitude)")
    print("- Std magnitude: \(stats.stdMagnitude)")
    // Note: meanSparsity not available in BatchStatistics
}

// Example: Performance comparison
func demonstratePerformance() throws {
    print("\n=== Performance Comparison ===\n")
    
    let vector1 = SIMD64<Float>(repeating: 1.0)
    let vector2 = SIMD64<Float>(repeating: 2.0)
    
    let implementations = [
        ("Euclidean", { vector1.distance(to: vector2, using: EuclideanDistance()) }),
        ("Manhattan", { vector1.distance(to: vector2, using: ManhattanDistance()) }),
        ("Cosine", { vector1.distance(to: vector2, using: CosineDistance()) }),
        ("Dot Product", { vector1.distance(to: vector2, using: DotProductDistance()) })
    ]
    
    let results = try PerformanceTest.compareImplementations(implementations)
    
    print("Distance metric performance:")
    for (name, time, relativeSpeed) in results {
        print("- \(name): \(time * 1000)ms (relative speed: \(String(format: "%.2fx", relativeSpeed)))")
    }
}

// Run all examples
@main
struct BasicUsageExample {
    static func main() async {
        print("VectorCore New API Examples\n")
        print("This demonstrates the new extension-based API that replaces utility functions\n")
        
        demonstrateNewAPI()
        
        do {
            try demonstrateFactory()
            await demonstrateBatchOps()
            try demonstratePerformance()
        } catch {
            print("Error: \(error)")
        }
    }
}