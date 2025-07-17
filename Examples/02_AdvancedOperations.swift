// VectorCore: Advanced Operations
//
// This example demonstrates advanced vector operations and techniques

import VectorCore
import Foundation

// MARK: - 1. Element-wise Operations

func elementWiseOperations() {
    print("=== Element-wise Operations ===\n")
    
    let v1 = Vector64.random(in: 0...2)
    let v2 = Vector64.random(in: 0...2)
    
    // Hadamard product (element-wise multiplication)
    let hadamard = v1 .* v2
    print("Hadamard product: \(hadamard)")
    
    // Element-wise division
    let elementDiv = v1 ./ (v2 + Vector64(repeating: 0.1)) // Add small value to avoid division by zero
    print("Element-wise division: \(elementDiv)")
    
    // Clamping values
    let unclamped = Vector64.random(in: -5...5)
    let clamped = unclamped.clamped(to: -1...1)
    print("\nOriginal range: [\(unclamped.lInfinityNorm)]")
    print("Clamped range: [\(clamped.lInfinityNorm)]")
}

// MARK: - 2. Softmax and Special Functions

func specialFunctions() {
    print("\n\n=== Special Functions ===\n")
    
    // Softmax transformation
    let logits = Vector32.random(in: -2...2)
    let probabilities = logits.softmax()
    
    print("Logits: \(logits)")
    print("Softmax: \(probabilities)")
    print("Sum of probabilities: \(probabilities.sum)")
    
    // Basis vectors (one-hot encoded)
    let basis5 = Vector32.basis(at: 5)
    print("\nBasis vector at index 5:")
    print("Non-zero elements: \(basis5.toArray().enumerated().filter { $0.1 != 0 }.map { "[\($0.0)]=\($0.1)" })")
}

// MARK: - 3. Batch Operations

func batchOperations() {
    print("\n\n=== Batch Operations ===\n")
    
    // Create a batch of vectors
    let batchSize = 10
    let vectors = (0..<batchSize).map { _ in Vector128.random(in: -1...1) }
    
    // Synchronous batch operations
    let batchOps = SyncBatchOperations()
    
    // Batch normalization
    let normalized = batchOps.batchNormalize(vectors)
    print("Batch normalized \(normalized.count) vectors")
    print("All unit vectors: \(normalized.allSatisfy { abs($0.magnitude - 1.0) < 0.0001 })")
    
    // Batch distance computation
    let query = Vector128.random(in: -1...1)
    let distances = batchOps.batchDistance(
        from: query,
        to: vectors,
        using: EuclideanDistance()
    )
    
    print("\nDistances from query:")
    for (i, dist) in distances.enumerated().prefix(5) {
        print("  Vector \(i): \(dist)")
    }
    
    // Find nearest neighbors
    let k = 3
    let nearest = batchOps.findNearestNeighbors(
        to: query,
        in: vectors,
        k: k,
        using: CosineDistance()
    )
    
    print("\n\(k) nearest neighbors (by cosine distance):")
    for result in nearest {
        print("  Index \(result.index): distance = \(result.distance)")
    }
}

// MARK: - 4. Serialization

func serializationDemo() {
    print("\n\n=== Serialization ===\n")
    
    let original = Vector256.random(in: -10...10)
    
    // Binary serialization with CRC32 checksum
    do {
        let binaryData = try original.encodeBinary()
        print("Binary size: \(binaryData.count) bytes")
        
        let decoded = try Vector256.decodeBinary(from: binaryData)
        print("Decoded successfully: \(decoded == original)")
        
        // Base64 encoding for text transport
        let base64 = original.base64Encoded
        print("\nBase64 encoded (first 50 chars): \(String(base64.prefix(50)))...")
        
        let fromBase64 = try Vector256.base64Decoded(from: base64)
        print("Base64 round-trip successful: \(fromBase64 == original)")
        
    } catch {
        print("Serialization error: \(error)")
    }
}

// MARK: - 5. Performance Patterns

func performancePatterns() {
    print("\n\n=== Performance Patterns ===\n")
    
    // Pattern 1: Reuse normalized vectors
    let vectors = (0..<100).map { _ in Vector512.random(in: -1...1) }
    let normalizedCache = vectors.map { $0.normalized() }
    
    // Now use normalizedCache for multiple operations
    let query = Vector512.random(in: -1...1).normalized()
    
    // Pattern 2: Use appropriate distance metrics
    print("Comparing distance metrics for normalized vectors:")
    
    let v1 = normalizedCache[0]
    let v2 = normalizedCache[1]
    
    // For normalized vectors, these are related:
    let cosineDistance = CosineDistance().distance(v1, v2)
    let dotProductSimilarity = -DotProductDistance().distance(v1, v2)
    
    print("Cosine distance: \(cosineDistance)")
    print("Negative dot product: \(dotProductSimilarity)")
    print("Difference: \(abs(cosineDistance - (1 - dotProductSimilarity)))")
    
    // Pattern 3: Choose the right vector type
    print("\nVector type selection:")
    
    // For known dimensions, use strongly-typed vectors
    let embedding = Vector768.zeros() // Typical transformer embedding
    print("Transformer embedding: \(type(of: embedding))")
    
    // For runtime dimensions, use DynamicVector
    let runtimeDim = 1000
    let dynamic = DynamicVector(dimension: runtimeDim)
    print("Dynamic vector (\(runtimeDim) dims): \(type(of: dynamic))")
}

// MARK: - 6. Working with Vector Collections

func vectorCollections() {
    print("\n\n=== Vector Collections ===\n")
    
    let vectors = (0..<5).map { i in 
        Vector32(repeating: Float(i))
    }
    
    // Vectors conform to Collection protocol
    for vector in vectors {
        print("First element: \(vector.first!), Last: \(vector.last!)")
    }
    
    // Can use standard collection operations
    let v = Vector32.random(in: 0...10)
    let filtered = v.filter { $0 > 5 }
    let mapped = v.map { $0 * 2 }
    let sum = v.reduce(0, +)
    
    print("\nCollection operations:")
    print("Elements > 5: \(filtered.count)")
    print("Sum via reduce: \(sum)")
    print("Sum via method: \(v.sum)")
}

// MARK: - Main

func main() {
    elementWiseOperations()
    specialFunctions()
    batchOperations()
    serializationDemo()
    performancePatterns()
    vectorCollections()
}

// Run the example
main()