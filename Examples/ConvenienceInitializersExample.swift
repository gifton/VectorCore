#!/usr/bin/swift

// VectorCore: Convenience Initializers Example
//
// Demonstrates various ways to create vectors with convenience initializers
//

import VectorCore
import Foundation

// MARK: - Basis Vectors

print("=== Basis Vectors ===")

// Standard basis vectors (unit vectors along axes)
let e0 = Vector<Dim32>.basis(axis: 0)  // [1, 0, 0, ..., 0]
let e1 = Vector<Dim32>.basis(axis: 1)  // [0, 1, 0, ..., 0]
let e2 = Vector<Dim32>.basis(axis: 2)  // [0, 0, 1, ..., 0]

print("e0 magnitude:", e0.magnitude)  // 1.0
print("e0 · e1:", e0.dotProduct(e1))  // 0.0 (orthogonal)

// Alternative notation
let ex = Vector<Dim128>.unitX  // Same as basis(axis: 0)
let ey = Vector<Dim128>.unitY  // Same as basis(axis: 1)
let ez = Vector<Dim128>.unitZ  // Same as basis(axis: 2)

// MARK: - Sequential Values

print("\n=== Sequential Values ===")

// Linearly spaced values
let linear = Vector<Dim32>.linspace(from: 0, to: 10)
print("Linear[0]:", linear[0])    // 0.0
print("Linear[31]:", linear[31])  // 10.0

// Simple range
let range1 = Vector<Dim32>.range()        // [0, 1, 2, ..., 31]
let range2 = Vector<Dim32>.range(from: 5) // [5, 6, 7, ..., 36]

// MARK: - Function-based Initialization

print("\n=== Function-based ===")

// Generate values using a function
let squares = Vector<Dim32>.generate { i in Float(i * i) }
print("Squares:", squares[0...4].map { $0 })  // [0, 1, 4, 9, 16]

// Using index map initializer
let sineWave = Vector<Dim128>(indexMap: { i in
    sin(Float(i) * 2 * .pi / 128)
})
print("Sine wave peak:", sineWave[32])  // ~1.0 (at π/2)

// MARK: - Mathematical Sequences

print("\n=== Mathematical Sequences ===")

// Geometric sequence
let powers2 = Vector<Dim32>.geometric(initial: 1, ratio: 2)
print("Powers of 2:", powers2[0...4].map { $0 })  // [1, 2, 4, 8, 16]

// Shorthand for powers
let powers3 = Vector<Dim32>.powers(of: 3)
print("Powers of 3:", powers3[0...3].map { $0 })  // [1, 3, 9, 27]

// Alternating signs
let alternating = Vector<Dim64>.alternating(magnitude: 5)
print("Alternating:", alternating[0...3].map { $0 })  // [5, -5, 5, -5]

// MARK: - Special Patterns

print("\n=== Special Patterns ===")

// One-hot encoding
let oneHot = Vector<Dim32>.oneHot(at: 10)
print("One-hot sum:", oneHot.magnitude)  // 1.0
print("One-hot[10]:", oneHot[10])       // 1.0

// Repeating pattern
let pattern = Vector<Dim32>.repeatingPattern([1, 2, 3])
print("Pattern:", pattern[0...8].map { $0 })  // [1, 2, 3, 1, 2, 3, 1, 2, 3]

// Sparse vector
let sparse = Vector<Dim128>.sparse(value: 10.0, at: [5, 10, 20])
print("Sparse non-zero count:", sparse.toArray().filter { $0 != 0 }.count)  // 3
print("Sparse magnitude:", sparse.magnitude)  // ~17.32 (sqrt(3 * 10²))

// MARK: - DynamicVector Examples

print("\n=== DynamicVector ===")

// Dynamic dimensions allow runtime-determined sizes
let dynamicBasis = DynamicVector.basis(dimension: 10, axis: 3)
print("Dynamic basis[3]:", dynamicBasis[3])  // 1.0

let dynamicLinspace = DynamicVector.linspace(dimension: 5, from: 0, to: 100)
print("Dynamic linspace:", dynamicLinspace.toArray())  // [0, 25, 50, 75, 100]

let dynamicGeometric = DynamicVector.geometric(dimension: 6, initial: 1, ratio: 0.5)
print("Dynamic geometric:", dynamicGeometric.toArray())  // [1, 0.5, 0.25, 0.125, 0.0625, 0.03125]

// MARK: - Practical Examples

print("\n=== Practical Examples ===")

// Create a Gaussian-like bell curve
let gaussian = Vector<Dim128>(indexMap: { i in
    let x = Float(i - 64) / 16.0  // Center at 64, scale by 16
    return exp(-x * x / 2)
})
print("Gaussian peak:", gaussian[64])  // ~1.0 (center)

// Create a decay envelope for audio
let envelope = Vector<Dim256>.geometric(initial: 1.0, ratio: 0.99)
print("Envelope start:", envelope[0])    // 1.0
print("Envelope end:", envelope[255])    // ~0.078

// Create a frequency spectrum with harmonics
let harmonics = Vector<Dim64>(indexMap: { i in
    let fundamental = 1.0
    let harmonic = Float(i + 1)
    return fundamental / harmonic  // 1/n amplitude for nth harmonic
})
print("Harmonic series:", harmonics[0...4].map { $0 })  // [1, 0.5, 0.33, 0.25, 0.2]

// MARK: - Performance Considerations

print("\n=== Performance ===")

// Measure creation time
let start = Date()
for _ in 0..<1000 {
    _ = Vector<Dim512>.zeros()           // Fastest - just allocation
    _ = Vector<Dim512>.basis(axis: 100)  // Fast - single element set
    _ = Vector<Dim512>.linspace(from: 0, to: 1)  // Moderate - linear computation
    _ = Vector<Dim512>.generate { i in sin(Float(i)) }  // Slowest - function calls
}
let elapsed = Date().timeIntervalSince(start)
print("Created 4000 vectors in \(String(format: "%.3f", elapsed))s")

// Memory efficiency example
let sparseData = Vector<Dim1536>.sparse(value: 1.0, at: Set(0..<10))
print("Sparse vector has \(sparseData.sparsity() * 100)% zero elements")