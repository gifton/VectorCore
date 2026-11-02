import Testing
import Foundation
@testable import VectorCore

// Define an arbitrary dimension not in the predefined list
// This tests that the generic Vector<D> system works for ANY dimension
struct Dim384: StaticDimension {
    static let value: Int = 384
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

        // Each ratio approximates ||x|| = sqrt(18948160) ≈ 4352.948 after two binary32 roundings, so ratios
        // agree only to about one ulp of ||x|| (ulp = 2^-11 ≈ 4.9e-4); the former absolute 1e-5 (≈ 0.02 ulp)
        // demanded bit-identical ratios. Observed at pre-program revision 8b02e75 during the T02
        // investigation (not re-verified by a live test): that route also left 76 of 384 ratios one ulp from
        // the rest, and only the three sampled indices happened to agree.
        // Derivation, with u = 2^-24 and v = 2^-53: N01 (closures/N01.json; briefs/N01.md "Numerical core")
        // forms the binary64 norm N̂ and returns n_i = fl32(fl64(x_i / N̂)). Every square and partial sum here
        // is an integer below 2^53, so N̂ = ||x||(1 + α) with |α| <= v. Then n_i = (x_i / N̂)(1 + β_i)(1 + δ_i)
        // with |β_i| <= v and |δ_i| <= u, and ratio_i = fl32(x_i / n_i) = N̂(1 + ε_i) / ((1 + β_i)(1 + δ_i))
        // with |ε_i| <= u. Each assertion below adds a 2^-20 relative margin to its exact bound.
        // (1) Per ratio, all 384 indices: |ratio_i - ||x||| <= ||x||(2u + 2v) / ((1 - u)(1 - v))
        //     < 2u(1 + 2^-20)||x|| ≈ 5.19e-4 ≈ 1.06 ulp. Evaluated in binary64 against `norm` (within 2^-53
        //     of ||x||; the subtraction is exact by Sterbenz). ||x|| lies 0.377 ulp above 0x45880796, so this
        //     passes iff every ratio is one of its two binary32 neighbours, 0x45880796 or 0x45880797. It
        //     rejects the pre-N01 route self * (1 / Float magnitude), which puts 76 ratios at 0x45880798,
        //     1.62 ulp away (T02 negative control: a scratch copy with only that route reverted, not a live
        //     test). The pairwise check (2) passes on that route, because its three sampled ratios coincide.
        // (2) Pairwise: |ratio_i - ratio_j| <= N̂(4u / (1 - u²) + 2.0001 · v) < 4u(1 + 2^-20)||x|| ≈ 1.04e-3,
        //     between 2 and 3 ulps. The Float tolerance costs < 2^-23 relative to evaluate and approxEqual's
        //     difference is exact by Sterbenz, so it fails iff pair (1, 2) or (2, 3) differs by 3 or more
        //     ulps of ||x||.
        let norm = (384.0 * 385.0 * 769.0 / 6.0).squareRoot()  // exact sum of squares 18948160
        let unitRoundoff = Double(Float.ulpOfOne) / 2
        let perRatioTolerance = 2 * unitRoundoff * (1 + 0x1p-20) * norm
        let offNormIndices = (0..<384).filter { i in
            abs(Double(vector[i] / normalized[i]) - norm) > perRatioTolerance
        }
        #expect(offNormIndices.isEmpty, "ratios off ||x|| at \(offNormIndices.prefix(8)) (\(offNormIndices.count) of 384)")
        let directionTolerance = Float(4 * unitRoundoff * (1 + 0x1p-20) * norm)
        #expect(approxEqual(ratio1, ratio2, tol: directionTolerance))
        #expect(approxEqual(ratio2, ratio3, tol: directionTolerance))
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
