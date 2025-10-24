import Testing
import Foundation
@testable import VectorCore

// Define an arbitrary dimension not in the predefined list
// This tests that the generic Vector<D> system works for ANY dimension
struct Dim384: StaticDimension {
    static let value: Int = 384
    typealias Storage = DimensionStorage<Dim384, Float>
}

@Suite("Arbitrary Dimension Support")
struct ArbitraryDimensionSuite {

    @Test("Vector<Dim384> - Creation")
    func testArbitraryDimCreation() {
        let values: [Float] = (0..<384).map { Float($0) + 1.0 }
        let vector = try! Vector<Dim384>(values)
        #expect(vector.scalarCount == 384)
        #expect(vector[0] == 1.0)
        #expect(vector[383] == 384.0)
    }

    @Test("Vector<Dim384> - Scalar Multiplication")
    func testArbitraryDimScalarMultiplication() {
        let values: [Float] = (0..<384).map { Float($0) + 1.0 }
        let vector = try! Vector<Dim384>(values)

        let scaled = vector * 2.0
        #expect(scaled[0] == 2.0)
        #expect(scaled[100] == 202.0)
        #expect(approxEqual(scaled[383], 768.0))
    }

    @Test("Vector<Dim384> - Division Operator")
    func testArbitraryDimDivision() {
        let values: [Float] = (0..<384).map { Float($0) + 1.0 }
        let vector = try! Vector<Dim384>(values)

        let divided = vector / 2.0
        #expect(approxEqual(divided[0], 0.5))
        #expect(approxEqual(divided[100], 50.5))
        #expect(approxEqual(divided[383], 192.0))
    }

    @Test("Vector<Dim384> - Magnitude Calculation")
    func testArbitraryDimMagnitude() {
        let values: [Float] = (0..<384).map { Float($0) + 1.0 }
        let vector = try! Vector<Dim384>(values)

        let mag = vector.magnitude
        // Manual calculation: sqrt(sum(i^2 for i in 1..384))
        // = sqrt(1^2 + 2^2 + ... + 384^2)
        // = sqrt(n(n+1)(2n+1)/6) where n=384
        let expected = Float(sqrt(384.0 * 385.0 * 769.0 / 6.0))
        #expect(approxEqual(mag, expected, tol: 0.1))
    }

    @Test("Vector<Dim384> - Normalization with Tolerance")
    func testArbitraryDimNormalization() {
        let values: [Float] = (0..<384).map { Float($0) + 1.0 }
        let vector = try! Vector<Dim384>(values)

        // This is the key test - normalization requires division operator
        let normalized = vector.normalized(tolerance: 1e-6)

        // Verify it's a unit vector
        let mag = normalized.magnitude
        #expect(approxEqual(mag, 1.0, tol: 0.001))
    }

    @Test("Vector<Dim384> - Protocol-based Normalization (Result)")
    func testArbitraryDimProtocolNormalization() {
        let values: [Float] = (0..<384).map { Float($0) + 1.0 }
        let vector = try! Vector<Dim384>(values)

        let result: Result<Vector<Dim384>, VectorError> = vector.normalized()

        switch result {
        case .success(let norm):
            let mag = norm.magnitude
            #expect(approxEqual(mag, 1.0, tol: 0.001))
        case .failure:
            Issue.record("Normalization should not fail for non-zero vector")
        }
    }

    @Test("Vector<Dim384> - Normalization Preserves Direction")
    func testArbitraryDimNormalizationPreservesDirection() {
        let values: [Float] = (0..<384).map { Float($0) + 1.0 }
        let vector = try! Vector<Dim384>(values)
        let normalized = vector.normalized(tolerance: 1e-6)

        // Check that normalized vector is proportional to original
        let ratio1 = vector[0] / normalized[0]
        let ratio2 = vector[100] / normalized[100]
        let ratio3 = vector[383] / normalized[383]

        #expect(approxEqual(ratio1, ratio2))
        #expect(approxEqual(ratio2, ratio3))
    }

    @Test("Vector<Dim384> - Zero Vector Normalization")
    func testArbitraryDimZeroVectorNormalization() {
        let zero = Vector<Dim384>.zero

        // Extension method returns self for zero vectors
        let norm1 = zero.normalized(tolerance: 1e-6)
        #expect(norm1.magnitude == 0.0)

        // Protocol method returns failure
        let result: Result<Vector<Dim384>, VectorError> = zero.normalized()
        switch result {
        case .success:
            Issue.record("Zero vector normalization should fail")
        case .failure(let error):
            // Expected behavior
            #expect(error.kind == .invalidOperation)
        }
    }

    @Test("Vector<Dim384> - All Arithmetic Operations")
    func testArbitraryDimFullArithmetic() {
        let a = try! Vector<Dim384>((0..<384).map { Float($0) + 1.0 })
        let b = try! Vector<Dim384>((0..<384).map { Float($0) + 10.0 })

        // Addition
        let sum = a + b
        #expect(approxEqual(sum[0], 11.0))

        // Subtraction
        let diff = a - b
        #expect(approxEqual(diff[0], -9.0))

        // Scalar multiplication
        let scaled = a * 3.0
        #expect(approxEqual(scaled[0], 3.0))

        // Division (THE KEY OPERATION)
        let divided = a / 2.0
        #expect(approxEqual(divided[0], 0.5))

        // Hadamard product
        let hadamard = a .* b
        #expect(approxEqual(hadamard[0], 10.0))

        // Element-wise division
        let elemDiv = b ./ a
        #expect(approxEqual(elemDiv[0], 10.0))
    }
}
