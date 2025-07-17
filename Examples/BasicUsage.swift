// VectorCore: Basic Usage Example
//
// Demonstrates the new extension-based API

import VectorCore
import Foundation

// Example: Working with vectors using the VectorCore API
func demonstrateVectorAPI() {
    print("=== VectorCore API Demo ===\n")
    
    // Create vectors using type-safe dimensions
    let vector1 = Vector32(repeating: 1.0)
    let vector2 = Vector32(repeating: 2.0)
    
    print("Vector 1: \(vector1)")
    print("Vector 2: \(vector2)")
    
    // Mathematical operations
    print("\n--- Mathematical Operations ---")
    print("Magnitude of vector1: \(vector1.magnitude)")
    print("Mean of vector1: \(vector1.mean)")
    print("Standard deviation: \(vector1.standardDeviation)")
    
    // Normalization
    let randomVector = Vector32.random(in: -1...1)
    let normalized = randomVector.normalized()
    print("\nNormalized vector magnitude: \(normalized.magnitude)")
    
    // Distance calculations
    print("\n--- Distance Calculations ---")
    let euclideanDist = vector1.distance(to: vector2)
    let manhattanDist = vector1.manhattanDistance(to: vector2)
    print("Euclidean distance: \(euclideanDist)")
    print("Manhattan distance: \(manhattanDist)")
    
    // Quality assessment
    print("\n--- Quality Assessment ---")
    var sparseValues = [Float](repeating: 0, count: 32)
    for i in stride(from: 0, to: 32, by: 4) {
        sparseValues[i] = Float(i)
    }
    let sparseVector = Vector32(sparseValues)
    print("Sparsity: \(sparseVector.sparsity())")
    print("Entropy: \(sparseVector.entropy)")
    let quality = sparseVector.quality
    print("Quality metrics: magnitude=\(quality.magnitude), variance=\(quality.variance)")
    print("Quality score: \(quality.score)")
    
    // Validation through quality metrics
    print("\n--- Validation ---")
    print("Is zero? \(quality.isZero)")
    print("Is sparse? \(quality.isSparse)")
    print("Is concentrated? \(quality.isConcentrated)")
    
    // Serialization
    print("\n--- Serialization ---")
    let base64 = vector1.base64Encoded
    print("Base64 encoded (first 50 chars): \(String(base64.prefix(50)))...")
}

// Example: Using the VectorFactory
func demonstrateFactory() {
    print("\n=== VectorFactory Demo ===\n")
    
    // Create vectors of different dimensions
    let vec128 = VectorFactory.zeros(dimension: 128)
    let vec256 = VectorFactory.ones(dimension: 256)
    let vec512 = VectorFactory.random(dimension: 512)
    
    print("Created vectors:")
    print("- 128D zeros: magnitude = \(vec128.magnitude)")
    print("- 256D ones: magnitude = \(vec256.magnitude)")
    print("- 512D random: magnitude = \(vec512.magnitude)")
    
    // Pattern-based creation
    let pattern = VectorFactory.withPattern(dimension: 768) { i in
        Float(i) / 768.0
    }
    
    // Cast to specific type to access mean property
    if let vector768 = pattern as? Vector768 {
        print("- 768D pattern: mean = \(vector768.mean)")
    }
    
    // Optimal dimension selection
    let optimal = VectorFactory.optimalDimension(for: 1000)
    print("\nOptimal dimension for ~1000: \(optimal)")
    
    // Demonstrate quality metrics with factory-created vectors
    print("\n--- Factory Vector Quality ---")
    if let vec256Typed = vec256 as? Vector256 {
        let quality = vec256Typed.quality
        print("256D ones quality: \(quality)")
    }
}

// Example: Batch operations
func demonstrateBatchOps() async {
    print("\n=== Batch Operations Demo ===\n")
    
    // Create a batch of vectors
    let vectors = (0..<100).map { _ in
        Vector32.random(in: -1...1)
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
    
    // Demonstrate quality metrics on batch
    print("\n--- Batch Quality Analysis ---")
    let qualities = vectors.prefix(5).map { $0.quality }
    for (i, quality) in qualities.enumerated() {
        print("Vector \(i) - Score: \(String(format: "%.2f", quality.score)), Sparsity: \(String(format: "%.1f%%", quality.sparsity * 100))")
    }
}

// Example: Performance comparison
func demonstratePerformance() {
    print("\n=== Performance Comparison ===\n")
    
    let vector1 = Vector64(repeating: 1.0)
    let vector2 = Vector64(repeating: 2.0)
    
    // Time different distance calculations
    let startTime = CFAbsoluteTimeGetCurrent()
    
    // Euclidean distance
    let euclideanStart = CFAbsoluteTimeGetCurrent()
    for _ in 0..<10000 {
        _ = vector1.distance(to: vector2)
    }
    let euclideanTime = CFAbsoluteTimeGetCurrent() - euclideanStart
    
    // Manhattan distance
    let manhattanStart = CFAbsoluteTimeGetCurrent()
    for _ in 0..<10000 {
        _ = vector1.manhattanDistance(to: vector2)
    }
    let manhattanTime = CFAbsoluteTimeGetCurrent() - manhattanStart
    
    // Cosine similarity
    let cosineStart = CFAbsoluteTimeGetCurrent()
    for _ in 0..<10000 {
        _ = vector1.cosineSimilarity(to: vector2)
    }
    let cosineTime = CFAbsoluteTimeGetCurrent() - cosineStart
    
    print("Distance metric performance (10,000 iterations):")
    print("- Euclidean: \(String(format: "%.3f", euclideanTime * 1000))ms")
    print("- Manhattan: \(String(format: "%.3f", manhattanTime * 1000))ms")
    print("- Cosine: \(String(format: "%.3f", cosineTime * 1000))ms")
    
    // Demonstrate metrics performance
    print("\n--- Quality Metrics Performance ---")
    let qualityVector = Vector768.random(in: -1...1)
    
    let sparsityStart = CFAbsoluteTimeGetCurrent()
    for _ in 0..<1000 {
        _ = qualityVector.sparsity()
    }
    let sparsityTime = CFAbsoluteTimeGetCurrent() - sparsityStart
    
    let entropyStart = CFAbsoluteTimeGetCurrent()
    for _ in 0..<1000 {
        _ = qualityVector.entropy
    }
    let entropyTime = CFAbsoluteTimeGetCurrent() - entropyStart
    
    print("Quality metrics (1,000 iterations on 768D vector):")
    print("- Sparsity: \(String(format: "%.3f", sparsityTime * 1000))ms")
    print("- Entropy: \(String(format: "%.3f", entropyTime * 1000))ms")
}

// Example: Demonstrate advanced features
func demonstrateAdvancedFeatures() {
    print("\n=== Advanced Features Demo ===\n")
    
    // Create a test vector with mixed values
    var values = [Float](repeating: 0, count: 512)
    // Add some non-zero values
    for i in stride(from: 0, to: 512, by: 10) {
        values[i] = Float.random(in: 1...10)
    }
    let testVector = Vector512(values)
    
    print("--- Quality Metrics ---")
    let quality = testVector.quality
    print("Quality Assessment: \(quality)")
    print("- Is sparse? \(quality.isSparse)")
    print("- Is concentrated? \(quality.isConcentrated)")
    print("- Overall score: \(String(format: "%.2f", quality.score))")
    
    print("\n--- Base64 Serialization ---")
    let encoded = testVector.base64Encoded
    print("Encoded length: \(encoded.count) characters")
    
    // Decode and verify
    if let decoded = try? Vector512.base64Decoded(from: encoded) {
        let areEqual = (0..<512).allSatisfy { testVector[$0] == decoded[$0] }
        print("Round-trip successful: \(areEqual)")
    }
    
    // Dynamic vector example
    print("\n--- Dynamic Vector Quality ---")
    let dynamicVector = DynamicVector.random(dimension: 100, in: -2...2)
    let dynQuality = dynamicVector.quality
    print("Dynamic vector (100D) quality:")
    print("- Magnitude: \(String(format: "%.3f", dynQuality.magnitude))")
    print("- Entropy: \(String(format: "%.3f", dynQuality.entropy))")
    print("- Sparsity: \(String(format: "%.1f%%", dynQuality.sparsity * 100))")
}

// Run all examples
@main
struct BasicUsageExample {
    static func main() async {
        print("VectorCore API Examples\n")
        print("Demonstrating the complete VectorCore API with quality metrics\n")
        
        demonstrateVectorAPI()
        demonstrateFactory()
        await demonstrateBatchOps()
        demonstratePerformance()
        demonstrateAdvancedFeatures()
    }
}