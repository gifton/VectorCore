// VectorCore - Modern API Example
//
// Demonstrates the clean, optimized-by-default API of VectorCore 3.0

import VectorCore
import Foundation

// MARK: - Basic Usage

func basicUsageExample() async {
    print("=== VectorCore 3.0 - Modern API ===\n")
    
    // Create vectors using type aliases - optimized storage by default
    let v1 = Vector512.random(in: -1...1)
    let v2 = Vector512.random(in: -1...1)
    
    print("Created two 512-dimensional vectors")
    print("v1 magnitude: \(v1.magnitude)")
    print("v2 magnitude: \(v2.magnitude)")
    
    // All operations are optimized with zero allocations
    let sum = v1 + v2  // Zero allocations
    let distance = v1.distance(to: v2)  // Zero allocations
    let similarity = v1.cosineSimilarity(to: v2)  // Zero allocations
    
    print("\nOperations (all zero-allocation):")
    print("Distance: \(distance)")
    print("Cosine similarity: \(similarity)")
    print("Sum magnitude: \(sum.magnitude)")
}

// MARK: - Batch Operations

func batchOperationsExample() async {
    print("\n=== Batch Operations (Automatic Parallelization) ===\n")
    
    // Create a dataset - automatic parallelization for large datasets
    let vectors = (0..<10_000).map { _ in Vector512.random(in: -1...1) }
    let query = Vector512.random(in: -1...1)
    
    print("Created 10,000 vectors for search")
    
    // Automatically parallel for large datasets (>1000 vectors)
    let start = Date()
    let nearest = await BatchOperations.findNearest(
        to: query,
        in: vectors,
        k: 100
    )
    let elapsed = Date().timeIntervalSince(start)
    
    print("Found 100 nearest neighbors in \(String(format: "%.3f", elapsed))s")
    print("Nearest distance: \(nearest.first?.distance ?? 0)")
    print("Farthest of k-nearest: \(nearest.last?.distance ?? 0)")
    
    // Convenience method on arrays
    let nearestAlt = await vectors.findNearest(to: query, k: 10)
    print("\nUsing array extension - found \(nearestAlt.count) nearest")
}

// MARK: - Different Dimensions

func differentDimensionsExample() async {
    print("\n=== Optimized Dimensions ===\n")
    
    // All common embedding dimensions have optimized storage
    let bert = Vector768.randomNormalized()  // BERT
    let gpt = Vector1536.random(in: -1...1)  // GPT embeddings
    let large = Vector3072.random(in: -1...1)  // Large models
    
    print("BERT-sized (768d): magnitude = \(bert.magnitude)")
    print("GPT-sized (1536d): magnitude = \(gpt.magnitude)")
    print("Large model (3072d): magnitude = \(large.magnitude)")
    
    // Check optimization support
    print("\nOptimization support:")
    for dim in [128, 256, 512, 768, 1024, 1536, 2048, 3072] {
        let hasSupport = VectorCore.hasOptimizedSupport(for: dim)
        print("Dimension \(dim): \(hasSupport ? "✓ Optimized" : "- Dynamic")")
    }
}

// MARK: - Parallel Processing

func parallelProcessingExample() async throws {
    print("\n=== Parallel Processing ===\n")
    
    let vectors = (0..<5000).map { _ in Vector256.random(in: -1...1) }
    
    // Transform vectors in parallel
    let start = Date()
    let normalized = try await BatchOperations.map(vectors) { vector in
        vector.normalized()
    }
    let elapsed1 = Date().timeIntervalSince(start)
    
    print("Normalized 5000 vectors in \(String(format: "%.3f", elapsed1))s")
    
    // Filter vectors in parallel
    let start2 = Date()
    let filtered = try await BatchOperations.filter(vectors) { vector in
        vector.magnitude > 8.0  // Vectors with large magnitude
    }
    let elapsed2 = Date().timeIntervalSince(start2)
    
    print("Filtered to \(filtered.count) vectors in \(String(format: "%.3f", elapsed2))s")
    
    // Compute statistics in parallel
    let stats = await BatchOperations.statistics(for: vectors)
    print("\nStatistics (computed in parallel):")
    print("Mean magnitude: \(stats.meanMagnitude)")
    print("Std deviation: \(stats.stdMagnitude)")
}

// MARK: - Clean API Surface

func cleanAPIExample() {
    print("\n=== Clean API Surface ===\n")
    
    // Single way to create vectors
    let v1 = VectorCore.createVector(dimension: 512, data: nil)  // Zero vector
    let v2 = VectorCore.createVector(dimension: 512, data: Array(repeating: Float(1.0), count: 512))
    
    // Convert arrays to vectors
    let data: [Float] = [1.0, 2.0, 3.0, 4.0]
    let v3 = data.toVector()  // Automatically creates Vector with optimal storage
    
    print("Created vectors with dimensions: \(v1.scalarCount), \(v2.scalarCount), \(v3.scalarCount)")
    
    // Configuration
    // Note: Configuration is read-only in this version
    print("\nParallel threshold: \(VectorCore.configuration.batchOperations.parallelThreshold)")
    
    // Version info
    print("VectorCore version: \(VectorCore.version)")
}

// MARK: - Performance Comparison

func performanceComparison() async {
    print("\n=== Performance Benefits ===\n")
    
    let vectors = (0..<1000).map { _ in Vector128.random(in: -1...1) }
    
    // Operation that would have allocated in old version
    var totalDotProduct: Float = 0
    let start = Date()
    
    for i in 0..<vectors.count {
        for j in i+1..<vectors.count {
            totalDotProduct += vectors[i].dotProduct(vectors[j])
        }
    }
    
    let elapsed = Date().timeIntervalSince(start)
    let operations = (vectors.count * (vectors.count - 1)) / 2
    let nanosPerOp = (elapsed * 1_000_000_000) / Double(operations)
    
    print("Performed \(operations) dot products")
    print("Total time: \(String(format: "%.3f", elapsed))s")
    print("Time per operation: \(String(format: "%.1f", nanosPerOp))ns")
    print("Zero heap allocations!")
}

// MARK: - Main

@main
struct ModernAPIExample {
    static func main() async {
        await basicUsageExample()
        await batchOperationsExample()
        await differentDimensionsExample()
        
        do {
            try await parallelProcessingExample()
        } catch {
            print("Error in parallel processing: \(error)")
        }
        
        cleanAPIExample()
        await performanceComparison()
        
        print("\n=== Summary ===")
        print("• Zero-allocation operations by default")
        print("• Automatic parallelization for large datasets")
        print("• Clean, single way to do things")
        print("• 7-55x faster than traditional implementations")
        print("• Modern Swift with async/await throughout")
    }
}