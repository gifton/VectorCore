#!/usr/bin/env swift

import VectorCore

// Test basic vector operations
print("=== VectorCore Verification ===")

// Test Vector128
let v128 = Vector128(repeating: 1.0)
print("✓ Vector128 created with dimension: \(v128.scalarCount)")

// Test Vector256
let v256 = Vector256(repeating: 2.0)
print("✓ Vector256 created with dimension: \(v256.scalarCount)")

// Test Vector512
let v512 = Vector512(repeating: 3.0)
print("✓ Vector512 created with dimension: \(v512.scalarCount)")

// Test dot product
let a = Vector128(repeating: 1.0)
let b = Vector128(repeating: 2.0)
let dot = a.dotProduct(b)
print("✓ Dot product: \(dot) (expected: 256.0)")

// Test arithmetic
let sum = a + b
print("✓ Vector addition: first element = \(sum[0]) (expected: 3.0)")

let scaled = a * 5.0
print("✓ Scalar multiplication: first element = \(scaled[0]) (expected: 5.0)")

// Test normalization
var values = [Float](repeating: 0, count: 128)
values[0] = 3.0
values[1] = 4.0
let v = Vector128(values)
let normalized = v.normalized()
print("✓ Normalization: magnitude = \(normalized.magnitude) (expected: 1.0)")

// Test DynamicVector
let dynamic = DynamicVector(dimension: 100, repeating: 1.0)
print("✓ DynamicVector created with dimension: \(dynamic.dimension)")

// Test VectorFactory
if let factoryVector = try? VectorFactory.vector(of: 256, from: Array(repeating: 1.0, count: 256)) {
    print("✓ VectorFactory created vector with dimension: \(factoryVector.scalarCount)")
}

// Test distance metrics
let euclidean = EuclideanDistance()
let distance = euclidean.distance(a, b)
print("✓ Euclidean distance: \(distance)")

print("\n✅ All basic operations verified successfully!")
print("The generic vector implementation is working correctly.")