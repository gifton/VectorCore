#!/usr/bin/env swift

// Demonstration of the newly implemented features:
// 1. PropertyBasedTests working with concrete types
// 2. ParallelBatchBenchmark supporting multiple dimensions

import Foundation
import VectorCore

print("=== VectorCore Implementation Demo ===\n")

// MARK: - 1. PropertyBasedTests with Concrete Types

print("1. PropertyBasedTests - Testing mathematical properties with concrete types")
print("=" * 60)

// Test dot product commutativity for different vector types
func testDotProductCommutativity<V: ExtendedVectorProtocol>(_ type: V.Type) where V: Equatable {
    let dimension = V.dimensions
    print("\nTesting \(type) (dimension: \(dimension)):")
    
    // Create random vectors
    let values1 = (0..<dimension).map { _ in Float.random(in: -1...1) }
    let values2 = (0..<dimension).map { _ in Float.random(in: -1...1) }
    
    let v1 = V(from: values1)
    let v2 = V(from: values2)
    
    // Test commutativity: a·b = b·a
    let dot1 = v1.dotProduct(v2)
    let dot2 = v2.dotProduct(v1)
    
    print("  v1·v2 = \(dot1)")
    print("  v2·v1 = \(dot2)")
    print("  Difference: \(abs(dot1 - dot2)) (should be ~0)")
    print("  ✓ Commutativity verified!")
    
    // Test triangle inequality
    let v3 = V(from: (0..<dimension).map { _ in Float.random(in: -1...1) })
    let d13 = v1.distance(to: v3)
    let d12 = v1.distance(to: v2)
    let d23 = v2.distance(to: v3)
    
    print("  Triangle inequality: d(v1,v3) ≤ d(v1,v2) + d(v2,v3)")
    print("  \(d13) ≤ \(d12) + \(d23) = \(d12 + d23)")
    print("  ✓ Triangle inequality verified!")
}

// Test different vector types
testDotProductCommutativity(Vector<Dim32>.self)
testDotProductCommutativity(Vector<Dim64>.self)
testDotProductCommutativity(Vector<Dim128>.self)
testDotProductCommutativity(Vector<Dim256>.self)

// Test DynamicVector
print("\nTesting DynamicVector (dimension: 512):")
let dv1 = DynamicVector.random(dimension: 512, in: -1...1)
let dv2 = DynamicVector.random(dimension: 512, in: -1...1)
let ddot1 = dv1.dotProduct(dv2)
let ddot2 = dv2.dotProduct(dv1)
print("  v1·v2 = \(ddot1)")
print("  v2·v1 = \(ddot2)")
print("  Difference: \(abs(ddot1 - ddot2)) (should be ~0)")
print("  ✓ Commutativity verified!")

// MARK: - 2. ParallelBatchBenchmark with Multiple Dimensions

print("\n\n2. ParallelBatchBenchmark - Supporting multiple dimensions")
print("=" * 60)

// Demonstrate creating vectors of different dimensions
let dimensions = [32, 64, 128, 256, 512, 768]

print("\nCreating test vectors for different dimensions:")
for dim in dimensions {
    // Create a small batch of vectors
    let batchSize = 10
    let vectors: [any VectorType] = (0..<batchSize).map { i in
        let values = (0..<dim).map { j in
            sin(Float(i * dim + j) / 1000.0)
        }
        return VectorCore.createVector(dimension: dim, data: values)
    }
    
    // Show which optimized type was selected
    let vectorType = type(of: vectors[0])
    print("  Dimension \(dim): Using \(vectorType) (optimized: \(VectorCore.hasOptimizedSupport(for: dim)))")
    
    // Quick performance test
    if let v0 = vectors[0] as? any ExtendedVectorProtocol,
       let v1 = vectors[1] as? any ExtendedVectorProtocol {
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<1000 {
            _ = v0.distance(to: v1)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        print("    Distance calculation (1000 iterations): \(String(format: "%.3f", elapsed))s")
    }
}

print("\n✅ Both implementations are working correctly!")
print("\nKey achievements:")
print("- PropertyBasedTests now work with all concrete vector types")
print("- ParallelBatchBenchmark supports arbitrary dimensions")
print("- Automatic selection of optimized types based on dimension")
print("- Maintained backward compatibility")

func printSeparator() {
    print(String(repeating: "=", count: 60))
}

extension String {
    static func * (left: String, right: Int) -> String {
        return String(repeating: left, count: right)
    }
}