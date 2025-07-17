// VectorCore: Performance Regression Test
//
// Quick test to measure performance of new operations
//

import Foundation
import VectorCore

// Test core operations
func testCoreOperationsPerformance() {
    print("=== Core Operations Performance ===\n")
    
    let v1 = Vector256.random(in: -1...1)
    let v2 = Vector256.random(in: -1...1)
    let iterations = 10000
    
    // Min/Max operations
    let minStart = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations {
        _ = v1.min(v2)
    }
    let minTime = CFAbsoluteTimeGetCurrent() - minStart
    print("Min operation: \(String(format: "%.3f", minTime * 1000))ms for \(iterations) iterations")
    print("  → \(String(format: "%.3f", minTime / Double(iterations) * 1_000_000))µs per operation")
    
    let maxStart = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations {
        _ = v1.max(v2)
    }
    let maxTime = CFAbsoluteTimeGetCurrent() - maxStart
    print("Max operation: \(String(format: "%.3f", maxTime * 1000))ms for \(iterations) iterations")
    print("  → \(String(format: "%.3f", maxTime / Double(iterations) * 1_000_000))µs per operation")
    
    // Clamp operation
    let clampStart = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations {
        _ = v1.clamped(to: -0.5...0.5)
    }
    let clampTime = CFAbsoluteTimeGetCurrent() - clampStart
    print("Clamp operation: \(String(format: "%.3f", clampTime * 1000))ms for \(iterations) iterations")
    print("  → \(String(format: "%.3f", clampTime / Double(iterations) * 1_000_000))µs per operation")
    
    // Lerp operation
    let lerpStart = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations {
        _ = v1.lerp(to: v2, t: 0.5)
    }
    let lerpTime = CFAbsoluteTimeGetCurrent() - lerpStart
    print("Lerp operation: \(String(format: "%.3f", lerpTime * 1000))ms for \(iterations) iterations")
    print("  → \(String(format: "%.3f", lerpTime / Double(iterations) * 1_000_000))µs per operation")
    
    // Absolute value
    let absStart = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations {
        _ = v1.absoluteValue()
    }
    let absTime = CFAbsoluteTimeGetCurrent() - absStart
    print("Absolute value operation: \(String(format: "%.3f", absTime * 1000))ms for \(iterations) iterations")
    print("  → \(String(format: "%.3f", absTime / Double(iterations) * 1_000_000))µs per operation")
    
    // Square root
    let sqrtStart = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations {
        _ = v1.absoluteValue().squareRoot()  // Use abs first to ensure positive values
    }
    let sqrtTime = CFAbsoluteTimeGetCurrent() - sqrtStart
    print("Square root operation: \(String(format: "%.3f", sqrtTime * 1000))ms for \(iterations) iterations")
    print("  → \(String(format: "%.3f", sqrtTime / Double(iterations) * 1_000_000))µs per operation")
}

// Test convenience initializers
func testInitializerPerformance() {
    print("\n\n=== Convenience Initializer Performance ===\n")
    
    let iterations = 1000
    
    // Basis vectors
    let basisStart = CFAbsoluteTimeGetCurrent()
    for i in 0..<iterations {
        _ = Vector256.basis(axis: i % 256)
    }
    let basisTime = CFAbsoluteTimeGetCurrent() - basisStart
    print("Basis vector creation: \(String(format: "%.3f", basisTime * 1000))ms for \(iterations) iterations")
    print("  → \(String(format: "%.3f", basisTime / Double(iterations) * 1_000_000))µs per creation")
    
    // Linspace
    let linspaceStart = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations {
        _ = Vector256.linspace(from: 0, to: 100)
    }
    let linspaceTime = CFAbsoluteTimeGetCurrent() - linspaceStart
    print("Linspace creation: \(String(format: "%.3f", linspaceTime * 1000))ms for \(iterations) iterations")
    print("  → \(String(format: "%.3f", linspaceTime / Double(iterations) * 1_000_000))µs per creation")
    
    // Geometric sequence
    let geoStart = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations {
        _ = Vector256.geometric(initial: 1, ratio: 1.01)
    }
    let geoTime = CFAbsoluteTimeGetCurrent() - geoStart
    print("Geometric sequence creation: \(String(format: "%.3f", geoTime * 1000))ms for \(iterations) iterations")
    print("  → \(String(format: "%.3f", geoTime / Double(iterations) * 1_000_000))µs per creation")
    
    // One-hot
    let oneHotStart = CFAbsoluteTimeGetCurrent()
    for i in 0..<iterations {
        _ = Vector256.oneHot(at: i % 256)
    }
    let oneHotTime = CFAbsoluteTimeGetCurrent() - oneHotStart
    print("One-hot creation: \(String(format: "%.3f", oneHotTime * 1000))ms for \(iterations) iterations")
    print("  → \(String(format: "%.3f", oneHotTime / Double(iterations) * 1_000_000))µs per creation")
    
    // Function-based generation
    let genStart = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations {
        _ = Vector256.generate { i in Float(i) }
    }
    let genTime = CFAbsoluteTimeGetCurrent() - genStart
    print("Function-based generation: \(String(format: "%.3f", genTime * 1000))ms for \(iterations) iterations")
    print("  → \(String(format: "%.3f", genTime / Double(iterations) * 1_000_000))µs per creation")
}

// Compare with baseline operations
func compareWithBaseline() {
    print("\n\n=== Performance Comparison with Baseline ===\n")
    
    let v1 = Vector256.random(in: -1...1)
    let v2 = Vector256.random(in: -1...1)
    let iterations = 10000
    
    // Baseline: Addition
    let addStart = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations {
        _ = v1 + v2
    }
    let addTime = CFAbsoluteTimeGetCurrent() - addStart
    let addPerOp = addTime / Double(iterations) * 1_000_000
    
    // Baseline: Dot product
    let dotStart = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations {
        _ = v1.dotProduct(v2)
    }
    let dotTime = CFAbsoluteTimeGetCurrent() - dotStart
    let dotPerOp = dotTime / Double(iterations) * 1_000_000
    
    // Min operation
    let minStart = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations {
        _ = v1.min(v2)
    }
    let minTime = CFAbsoluteTimeGetCurrent() - minStart
    let minPerOp = minTime / Double(iterations) * 1_000_000
    
    // Clamp operation
    let clampStart = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations {
        _ = v1.clamped(to: -0.5...0.5)
    }
    let clampTime = CFAbsoluteTimeGetCurrent() - clampStart
    let clampPerOp = clampTime / Double(iterations) * 1_000_000
    
    print("Baseline operations:")
    print("  Addition: \(String(format: "%.3f", addPerOp))µs per operation")
    print("  Dot product: \(String(format: "%.3f", dotPerOp))µs per operation")
    print("\nCore operations:")
    print("  Min: \(String(format: "%.3f", minPerOp))µs per operation (\(String(format: "%.1fx", minPerOp/addPerOp)) vs addition)")
    print("  Clamp: \(String(format: "%.3f", clampPerOp))µs per operation (\(String(format: "%.1fx", clampPerOp/addPerOp)) vs addition)")
    
    // Check that new operations are reasonably fast
    if minPerOp > addPerOp * 3 {
        print("\n⚠️  WARNING: Min operation is significantly slower than expected!")
    }
    if clampPerOp > addPerOp * 2 {
        print("\n⚠️  WARNING: Clamp operation is significantly slower than expected!")
    } else {
        print("\n✅ All core operations perform within expected bounds")
    }
}

// Main
print("VectorCore Performance Regression Test")
print("======================================\n")

testCoreOperationsPerformance()
testInitializerPerformance()
compareWithBaseline()

print("\n\nRegression test completed!")