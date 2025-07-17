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
    
    // Compare with baseline
    let addStart = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations {
        _ = v1 + v2
    }
    let addTime = CFAbsoluteTimeGetCurrent() - addStart
    
    print("\nComparison with baseline:")
    print("  Addition: \(String(format: "%.3f", addTime / Double(iterations) * 1_000_000))µs per operation")
    print("  Min is \(String(format: "%.1fx", minTime/addTime)) slower than addition")
    print("  Max is \(String(format: "%.1fx", maxTime/addTime)) slower than addition")
    print("  Clamp is \(String(format: "%.1fx", clampTime/addTime)) slower than addition")
    
    if minTime/addTime < 3 && maxTime/addTime < 3 && clampTime/addTime < 2 {
        print("\n✅ All core operations perform within expected bounds")
    }
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
    print("Basis vector: \(String(format: "%.3f", basisTime / Double(iterations) * 1_000_000))µs per creation")
    
    // Linspace
    let linspaceStart = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations {
        _ = Vector256.linspace(from: 0, to: 100)
    }
    let linspaceTime = CFAbsoluteTimeGetCurrent() - linspaceStart
    print("Linspace: \(String(format: "%.3f", linspaceTime / Double(iterations) * 1_000_000))µs per creation")
    
    // Compare with zeros
    let zerosStart = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations {
        _ = Vector256.zeros()
    }
    let zerosTime = CFAbsoluteTimeGetCurrent() - zerosStart
    
    print("\nComparison with baseline:")
    print("  Zeros: \(String(format: "%.3f", zerosTime / Double(iterations) * 1_000_000))µs per creation")
    print("  Basis is \(String(format: "%.1fx", basisTime/zerosTime)) slower than zeros")
    print("  Linspace is \(String(format: "%.1fx", linspaceTime/zerosTime)) slower than zeros")
}

// Main
print("VectorCore Performance Regression Test")
print("======================================\n")

testCoreOperationsPerformance()
testInitializerPerformance()

print("\n\nRegression test completed!")