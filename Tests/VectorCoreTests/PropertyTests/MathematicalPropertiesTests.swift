import XCTest
@testable import VectorCore

// Import Dimension explicitly to disambiguate from Foundation.Dimension (NSDimension)
import protocol VectorCore.Dimension

/// Tests for fundamental mathematical laws and properties that should hold for all vector operations
final class MathematicalPropertiesTests: XCTestCase {
    
    // MARK: - Configuration
    
    private let iterations = 100
    private let accuracy: Float = 1e-5
    private let dimensions = [32, 64, 128, 256, 512]
    
    // MARK: - Property Testing Framework
    
    /// Simple property testing helper that generates random inputs and verifies properties
    private func forAll<T>(
        iterations: Int = 100,
        generator: () -> T,
        property: (T) throws -> Bool,
        message: String,
        file: StaticString = #file,
        line: UInt = #line
    ) {
        var failures = 0
        var lastFailure: T?
        
        for i in 0..<iterations {
            let input = generator()
            do {
                if try !property(input) {
                    failures += 1
                    lastFailure = input
                }
            } catch {
                XCTFail("Property test threw error: \(error) on iteration \(i)", file: file, line: line)
                return
            }
        }
        
        if failures > 0 {
            XCTFail("\(message) - Failed \(failures)/\(iterations) times. Last failure: \(String(describing: lastFailure))", 
                   file: file, line: line)
        }
    }
    
    // MARK: - Vector Addition Properties
    
    func testAdditionCommutativity() {
        // Test for fixed-size vectors
        testFixedVectorAdditionCommutativity()
        
        // Test for dynamic vectors
        testDynamicVectorAdditionCommutativity()
    }
    
    private func testFixedVectorAdditionCommutativity() {
        // a + b == b + a
        forAll(
            generator: { 
                (Vector<Dim128>.random(in: -10...10), 
                 Vector<Dim128>.random(in: -10...10))
            },
            property: { (a, b) in
                let ab = a + b
                let ba = b + a
                return vectorsEqual(ab, ba, accuracy: accuracy)
            },
            message: "Addition commutativity failed for Vector<Dim128>"
        )
    }
    
    private func testDynamicVectorAdditionCommutativity() {
        for dim in dimensions {
            forAll(
                iterations: 50,
                generator: { 
                    (DynamicVector.random(dimension: dim, in: -10...10),
                     DynamicVector.random(dimension: dim, in: -10...10))
                },
                property: { (a, b) in
                    let ab = a + b
                    let ba = b + a
                    return dynamicVectorsEqual(ab, ba, accuracy: accuracy)
                },
                message: "Addition commutativity failed for DynamicVector[\(dim)]"
            )
        }
    }
    
    func testAdditionAssociativity() {
        // (a + b) + c == a + (b + c)
        testFixedVectorAdditionAssociativity()
        testDynamicVectorAdditionAssociativity()
    }
    
    private func testFixedVectorAdditionAssociativity() {
        forAll(
            generator: {
                (Vector<Dim64>.random(in: -10...10),
                 Vector<Dim64>.random(in: -10...10),
                 Vector<Dim64>.random(in: -10...10))
            },
            property: { (a, b, c) in
                let left = (a + b) + c
                let right = a + (b + c)
                return vectorsEqual(left, right, accuracy: accuracy)
            },
            message: "Addition associativity failed for Vector<Dim64>"
        )
    }
    
    private func testDynamicVectorAdditionAssociativity() {
        forAll(
            generator: {
                let dim = 128
                return (DynamicVector.random(dimension: dim, in: -10...10),
                        DynamicVector.random(dimension: dim, in: -10...10),
                        DynamicVector.random(dimension: dim, in: -10...10))
            },
            property: { (a, b, c) in
                let left = (a + b) + c
                let right = a + (b + c)
                return dynamicVectorsEqual(left, right, accuracy: accuracy)
            },
            message: "Addition associativity failed for DynamicVector"
        )
    }
    
    func testAdditionIdentity() {
        // v + zeros == v
        testFixedVectorAdditionIdentity()
        testDynamicVectorAdditionIdentity()
    }
    
    private func testFixedVectorAdditionIdentity() {
        forAll(
            generator: { Vector<Dim256>.random(in: -100...100) },
            property: { v in
                let zeros = Vector<Dim256>(repeating: 0)
                let result = v + zeros
                return vectorsEqual(v, result, accuracy: accuracy)
            },
            message: "Addition identity failed for Vector<Dim256>"
        )
    }
    
    private func testDynamicVectorAdditionIdentity() {
        forAll(
            generator: {
                let dim = 256
                return DynamicVector.random(dimension: dim, in: -100...100)
            },
            property: { (v: DynamicVector) in
                let zeros = DynamicVector(dimension: v.dimension, repeating: 0)
                let result = v + zeros
                return dynamicVectorsEqual(v, result, accuracy: accuracy)
            },
            message: "Addition identity failed for DynamicVector"
        )
    }
    
    // MARK: - Scalar Multiplication Properties
    
    func testScalarMultiplicationDistributivity() {
        // a * (b + c) == a * b + a * c (scalar over vector addition)
        testFixedVectorScalarDistributivity()
        testDynamicVectorScalarDistributivity()
    }
    
    private func testFixedVectorScalarDistributivity() {
        forAll(
            generator: {
                (Float.random(in: -10...10),
                 Vector<Dim128>.random(in: -10...10),
                 Vector<Dim128>.random(in: -10...10))
            },
            property: { (a, b, c) in
                let left = a * (b + c)
                let right = a * b + a * c
                return vectorsEqual(left, right, accuracy: accuracy * 10) // Slightly relaxed for compound operations
            },
            message: "Scalar distributivity failed for Vector<Dim128>"
        )
    }
    
    private func testDynamicVectorScalarDistributivity() {
        forAll(
            generator: {
                let dim = 128
                return (Float.random(in: -10...10),
                        DynamicVector.random(dimension: dim, in: -10...10),
                        DynamicVector.random(dimension: dim, in: -10...10))
            },
            property: { (a, b, c) in
                let left = a * (b + c)
                let right = a * b + a * c
                return dynamicVectorsEqual(left, right, accuracy: accuracy * 10)
            },
            message: "Scalar distributivity failed for DynamicVector"
        )
    }
    
    func testScalarMultiplicationIdentity() {
        // v * 1 == v
        forAll(
            generator: { Vector<Dim64>.random(in: -100...100) },
            property: { v in
                let result = v * 1.0
                return vectorsEqual(v, result, accuracy: accuracy)
            },
            message: "Scalar multiplication identity failed"
        )
    }
    
    func testScalarMultiplicationZero() {
        // v * 0 == zeros
        forAll(
            generator: { Vector<Dim64>.random(in: -100...100) },
            property: { v in
                let result = v * 0.0
                let zeros = Vector<Dim64>(repeating: 0)
                return vectorsEqual(result, zeros, accuracy: accuracy)
            },
            message: "Scalar multiplication by zero failed"
        )
    }
    
    // MARK: - Element-wise Multiplication Properties
    
    func testElementWiseMultiplicationCommutativity() {
        // a .* b == b .* a
        forAll(
            generator: {
                (Vector<Dim128>.random(in: -10...10),
                 Vector<Dim128>.random(in: -10...10))
            },
            property: { (a, b) in
                let ab = a .* b
                let ba = b .* a
                return vectorsEqual(ab, ba, accuracy: accuracy)
            },
            message: "Element-wise multiplication commutativity failed"
        )
    }
    
    func testElementWiseMultiplicationAssociativity() {
        // (a .* b) .* c == a .* (b .* c)
        forAll(
            generator: {
                (Vector<Dim64>.random(in: -2...2), // Smaller range to avoid overflow
                 Vector<Dim64>.random(in: -2...2),
                 Vector<Dim64>.random(in: -2...2))
            },
            property: { (a, b, c) in
                let left = (a .* b) .* c
                let right = a .* (b .* c)
                return vectorsEqual(left, right, accuracy: accuracy * 10)
            },
            message: "Element-wise multiplication associativity failed"
        )
    }
    
    func testElementWiseMultiplicationIdentity() {
        // v .* ones == v
        forAll(
            generator: { Vector<Dim128>.random(in: -100...100) },
            property: { v in
                let ones = Vector<Dim128>(repeating: 1)
                let result = v .* ones
                return vectorsEqual(v, result, accuracy: accuracy)
            },
            message: "Element-wise multiplication identity failed"
        )
    }
    
    // MARK: - Norm Properties
    
    func testNormalizationMagnitude() {
        // ||normalize(v)|| ≈ 1.0 (for non-zero vectors)
        testFixedVectorNormalizationMagnitude()
        testDynamicVectorNormalizationMagnitude()
    }
    
    private func testFixedVectorNormalizationMagnitude() {
        forAll(
            generator: { Vector<Dim128>.random(in: -100...100) },
            property: { v in
                guard v.magnitude > 1e-6 else { return true } // Skip near-zero vectors
                
                let normalized = v.normalized()
                return abs(normalized.magnitude - 1.0) < accuracy
            },
            message: "Normalized magnitude != 1 for Vector<Dim128>"
        )
    }
    
    private func testDynamicVectorNormalizationMagnitude() {
        forAll(
            generator: { DynamicVector.random(dimension: 256, in: -100...100) },
            property: { v in
                guard v.magnitude > 1e-6 else { return true }
                
                let normalized = v.normalized()
                return abs(normalized.magnitude - 1.0) < accuracy
            },
            message: "Normalized magnitude != 1 for DynamicVector"
        )
    }
    
    func testTriangleInequality() {
        // ||a + b|| ≤ ||a|| + ||b||
        forAll(
            generator: {
                (Vector<Dim128>.random(in: -100...100),
                 Vector<Dim128>.random(in: -100...100))
            },
            property: { (a, b) in
                let sumMagnitude = (a + b).magnitude
                let magnitudeSum = a.magnitude + b.magnitude
                return sumMagnitude <= magnitudeSum + accuracy
            },
            message: "Triangle inequality failed"
        )
    }
    
    func testCauchySchwarzInequality() {
        // |a·b| ≤ ||a|| * ||b||
        forAll(
            generator: {
                (Vector<Dim256>.random(in: -50...50),
                 Vector<Dim256>.random(in: -50...50))
            },
            property: { (a, b) in
                let dotProduct = abs(a.dotProduct(b))
                let magnitudeProduct = a.magnitude * b.magnitude
                return dotProduct <= magnitudeProduct + accuracy
            },
            message: "Cauchy-Schwarz inequality failed"
        )
    }
    
    // MARK: - Distance Properties
    
    func testDistanceSymmetry() {
        // d(a, b) == d(b, a)
        forAll(
            generator: {
                (Vector<Dim128>.random(in: -100...100),
                 Vector<Dim128>.random(in: -100...100))
            },
            property: { (a, b) in
                let d1 = a.distance(to: b)
                let d2 = b.distance(to: a)
                return abs(d1 - d2) < accuracy
            },
            message: "Distance symmetry failed"
        )
    }
    
    func testDistanceTriangleInequality() {
        // d(a, c) ≤ d(a, b) + d(b, c)
        forAll(
            generator: {
                (Vector<Dim64>.random(in: -50...50),
                 Vector<Dim64>.random(in: -50...50),
                 Vector<Dim64>.random(in: -50...50))
            },
            property: { (a, b, c) in
                let dac = a.distance(to: c)
                let dab = a.distance(to: b)
                let dbc = b.distance(to: c)
                return dac <= dab + dbc + accuracy
            },
            message: "Distance triangle inequality failed"
        )
    }
    
    func testManhattanDistanceProperties() {
        // Manhattan distance specific properties
        forAll(
            generator: {
                (Vector<Dim128>.random(in: -50...50),
                 Vector<Dim128>.random(in: -50...50))
            },
            property: { (a, b) in
                let manhattan = a.manhattanDistance(to: b)
                let euclidean = a.distance(to: b)
                // Manhattan distance >= Euclidean distance
                return manhattan >= euclidean - accuracy
            },
            message: "Manhattan distance < Euclidean distance"
        )
    }
    
    // MARK: - Dot Product Properties
    
    func testDotProductCommutativity() {
        // a·b == b·a
        forAll(
            generator: {
                (Vector<Dim256>.random(in: -50...50),
                 Vector<Dim256>.random(in: -50...50))
            },
            property: { (a, b) in
                let ab = a.dotProduct(b)
                let ba = b.dotProduct(a)
                return abs(ab - ba) < accuracy
            },
            message: "Dot product commutativity failed"
        )
    }
    
    func testDotProductDistributivity() {
        // a·(b + c) == a·b + a·c
        forAll(
            generator: {
                (Vector<Dim64>.random(in: -10...10),
                 Vector<Dim64>.random(in: -10...10),
                 Vector<Dim64>.random(in: -10...10))
            },
            property: { (a, b, c) in
                let left = a.dotProduct(b + c)
                let right = a.dotProduct(b) + a.dotProduct(c)
                return abs(left - right) < accuracy * 10
            },
            message: "Dot product distributivity failed"
        )
    }
    
    func testDotProductScalarAssociativity() {
        // (k*a)·b == k*(a·b)
        forAll(
            generator: {
                (Float.random(in: -10...10),
                 Vector<Dim128>.random(in: -10...10),
                 Vector<Dim128>.random(in: -10...10))
            },
            property: { (k, a, b) in
                let left = (k * a).dotProduct(b)
                let right = k * a.dotProduct(b)
                return abs(left - right) < accuracy * 10
            },
            message: "Dot product scalar associativity failed"
        )
    }
    
    // MARK: - Helper Methods
    
    private func vectorsEqual<D>(_ a: Vector<D>, _ b: Vector<D>, accuracy: Float) -> Bool where D: Dimension {
        guard a.scalarCount == b.scalarCount else { return false }
        for i in 0..<a.scalarCount {
            if abs(a[i] - b[i]) > accuracy {
                return false
            }
        }
        return true
    }
    
    private func dynamicVectorsEqual(_ a: DynamicVector, _ b: DynamicVector, accuracy: Float) -> Bool {
        guard a.dimension == b.dimension else { return false }
        for i in 0..<a.dimension {
            if abs(a[i] - b[i]) > accuracy {
                return false
            }
        }
        return true
    }
}

// MARK: - Test Extensions

extension Vector {
    static func random(in range: ClosedRange<Float>) -> Self {
        let values = (0..<D.value).map { _ in Float.random(in: range) }
        return Self(values)
    }
}

extension DynamicVector {
    static func random(dimension: Int, in range: ClosedRange<Float>) -> Self {
        let values = (0..<dimension).map { _ in Float.random(in: range) }
        return Self(values)
    }
}