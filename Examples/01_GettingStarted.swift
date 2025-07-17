// VectorCore: Getting Started Guide
//
// This example demonstrates the basics of VectorCore

import VectorCore
import Foundation

// MARK: - 1. Creating Vectors

func creatingVectors() {
    print("=== Creating Vectors ===\n")
    
    // Method 1: Type aliases (recommended for common dimensions)
    let embedding128 = Vector128.zeros()
    let embedding256 = Vector256.ones()
    let embedding768 = Vector768.random(in: -1...1)
    
    print("128-dim zeros: \(embedding128)")
    print("256-dim ones: \(embedding256)") 
    print("768-dim random: \(embedding768)")
    
    // Method 2: Generic type with specific dimension
    let vector = Vector<Dim512>(repeating: 0.5)
    print("\n512-dim vector: \(vector)")
    
    // Method 3: From array (compile-time safety)
    let values = Array(stride(from: 0.0, to: 32.0, by: 1.0))
    let fromArray = Vector32(values)
    print("\nFrom array: \(fromArray)")
    
    // Method 4: Dynamic vectors (runtime dimensions)
    let dynamic = DynamicVector(dimension: 100, repeating: 0.1)
    print("\nDynamic vector (100 dims): \(dynamic)")
}

// MARK: - 2. Basic Operations

func basicOperations() {
    print("\n\n=== Basic Operations ===\n")
    
    let v1 = Vector128.random(in: 0...1)
    let v2 = Vector128.random(in: 0...1)
    
    // Arithmetic
    let sum = v1 + v2
    let difference = v1 - v2
    let scaled = v1 * 2.0
    let divided = v1 / 2.0
    
    print("Sum magnitude: \(sum.magnitude)")
    print("Difference magnitude: \(difference.magnitude)")
    print("Scaled magnitude: \(scaled.magnitude)")
    print("Divided magnitude: \(divided.magnitude)")
    
    // Element access
    print("\nFirst element of v1: \(v1[0])")
    print("Last element of v1: \(v1[127])")
    
    // In-place operations
    var mutable = v1
    mutable[0] = 99.0
    print("\nModified first element: \(mutable[0])")
}

// MARK: - 3. Mathematical Operations

func mathematicalOperations() {
    print("\n\n=== Mathematical Operations ===\n")
    
    let v1 = Vector64.random(in: -2...2)
    let v2 = Vector64.random(in: -2...2)
    
    // Core operations
    let dotProduct = v1.dotProduct(v2)
    let magnitude = v1.magnitude
    let normalized = v1.normalized()
    
    print("Dot product: \(dotProduct)")
    print("Magnitude: \(magnitude)")
    print("Normalized magnitude: \(normalized.magnitude)")
    
    // Statistical operations
    print("\nStatistics:")
    print("Mean: \(v1.mean)")
    print("Sum: \(v1.sum)")
    print("Variance: \(v1.variance)")
    print("Std deviation: \(v1.standardDeviation)")
    
    // Norms
    print("\nNorms:")
    print("L1 norm: \(v1.l1Norm)")
    print("L2 norm: \(v1.l2Norm)")
    print("L∞ norm: \(v1.lInfinityNorm)")
}

// MARK: - 4. Distance Metrics

func distanceMetrics() {
    print("\n\n=== Distance Metrics ===\n")
    
    let query = Vector256.random(in: -1...1)
    let candidate1 = Vector256.random(in: -1...1)
    let candidate2 = Vector256.random(in: -1...1)
    
    // Direct distance methods
    let euclidean = query.distance(to: candidate1)
    let manhattan = query.manhattanDistance(to: candidate1)
    let cosine = query.cosineSimilarity(to: candidate1)
    
    print("Euclidean distance: \(euclidean)")
    print("Manhattan distance: \(manhattan)")
    print("Cosine similarity: \(cosine)")
    
    // Using distance metric objects
    let metrics: [any DistanceMetric] = [
        EuclideanDistance(),
        CosineDistance(),
        ManhattanDistance(),
        DotProductDistance()
    ]
    
    print("\nUsing metric objects:")
    for metric in metrics {
        let distance = metric.distance(query, candidate2)
        print("\(metric.identifier): \(distance)")
    }
}

// MARK: - 5. Quality Assessment

func qualityAssessment() {
    print("\n\n=== Quality Assessment ===\n")
    
    // Create vectors with different characteristics
    let dense = Vector128.random(in: -1...1)
    
    var sparseValues = [Float](repeating: 0, count: 128)
    for i in stride(from: 0, to: 128, by: 8) {
        sparseValues[i] = Float.random(in: -1...1)
    }
    let sparse = Vector128(sparseValues)
    
    let uniform = Vector128(repeating: 0.5)
    
    // Analyze quality
    let vectors = [
        ("Dense", dense),
        ("Sparse", sparse),
        ("Uniform", uniform)
    ]
    
    for (name, vector) in vectors {
        let quality = vector.quality
        print("\n\(name) vector:")
        print("  Magnitude: \(quality.magnitude)")
        print("  Variance: \(quality.variance)")
        print("  Sparsity: \(quality.sparsity * 100)%")
        print("  Entropy: \(quality.entropy)")
        print("  Quality score: \(quality.score)")
    }
}

// MARK: - Main

func main() {
    creatingVectors()
    basicOperations()
    mathematicalOperations()
    distanceMetrics()
    qualityAssessment()
}

// Run the example
main()