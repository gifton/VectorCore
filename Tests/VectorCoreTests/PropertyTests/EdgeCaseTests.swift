import XCTest
@testable import VectorCore

/// Tests for boundary conditions and edge cases
final class EdgeCaseTests: XCTestCase {
    
    // MARK: - Configuration
    
    private let accuracy: Float = 1e-5
    
    // MARK: - Zero Vector Edge Cases
    
    func testZeroVectorOperations() {
        // Test operations with zero vectors
        testFixedZeroVectorOperations()
        testDynamicZeroVectorOperations()
    }
    
    private func testFixedZeroVectorOperations() {
        let zero = Vector<Dim128>(repeating: 0)
        let nonZero = Vector<Dim128>.random(in: -10...10)
        
        // Magnitude of zero vector is 0
        XCTAssertEqual(zero.magnitude, 0, accuracy: accuracy, "Zero vector magnitude != 0")
        
        // Operations with zero vector
        assertVectorsEqual(zero + nonZero, nonZero, "zero + v != v")
        assertVectorsEqual(nonZero + zero, nonZero, "v + zero != v")
        assertVectorsEqual(nonZero - zero, nonZero, "v - zero != v")
        assertVectorsEqual(zero - nonZero, -nonZero, "zero - v != -v")
        
        // Scalar operations
        assertVectorsEqual(zero * 5.0, zero, "zero * scalar != zero")
        assertVectorsEqual(zero / 3.0, zero, "zero / scalar != zero")
        
        // Dot product with zero
        XCTAssertEqual(zero.dotProduct(nonZero), 0, accuracy: accuracy, "zero · v != 0")
        XCTAssertEqual(nonZero.dotProduct(zero), 0, accuracy: accuracy, "v · zero != 0")
        
        // Distance from zero
        XCTAssertEqual(zero.distance(to: nonZero), nonZero.magnitude, accuracy: accuracy,
                      "d(zero, v) != ||v||")
        
        // Normalization of zero vector returns zero
        let normalizedZero = zero.normalized()
        assertVectorsEqual(normalizedZero, zero, "normalize(zero) != zero")
        
        // Cosine similarity with zero is 0
        XCTAssertEqual(zero.cosineSimilarity(to: nonZero), 0, accuracy: accuracy,
                      "cos(zero, v) != 0")
    }
    
    private func testDynamicZeroVectorOperations() {
        let dimensions = [32, 128, 512]
        
        for dim in dimensions {
            let zero = DynamicVector(repeating: 0, dimension: dim)
            let nonZero = DynamicVector.random(dimension: dim, in: -10...10)
            
            XCTAssertEqual(zero.magnitude, 0, accuracy: accuracy, 
                          "Zero DynamicVector[\(dim)] magnitude != 0")
            
            // Test preservation of dimension
            XCTAssertEqual((zero + nonZero).dimension, dim)
            XCTAssertEqual((zero * 5.0).dimension, dim)
            XCTAssertEqual(zero.normalized().dimension, dim)
        }
    }
    
    // MARK: - Unit Vector Edge Cases
    
    func testUnitVectorOperations() {
        // Test operations with unit vectors (magnitude = 1)
        let dimensions = [64, 128, 256]
        
        for dim in dimensions {
            // Create unit vectors along different axes
            for axis in 0..<min(3, dim) {
                var values = [Float](repeating: 0, count: dim)
                values[axis] = 1.0
                
                switch dim {
                case 64:
                    let unit = Vector<Dim64>(values)
                    verifyUnitVectorProperties(unit, axis: axis)
                case 128:
                    let unit = Vector<Dim128>(values)
                    verifyUnitVectorProperties(unit, axis: axis)
                case 256:
                    let unit = Vector<Dim256>(values)
                    verifyUnitVectorProperties(unit, axis: axis)
                default:
                    break
                }
                
                let dynamicUnit = DynamicVector(values)
                verifyDynamicUnitVectorProperties(dynamicUnit, axis: axis)
            }
        }
    }
    
    private func verifyUnitVectorProperties<D: Dimension>(_ unit: Vector<D>, axis: Int) {
        // Magnitude is 1
        XCTAssertEqual(unit.magnitude, 1.0, accuracy: accuracy, "Unit vector magnitude != 1")
        
        // Normalization returns self
        let normalized = unit.normalized()
        assertVectorsEqual(normalized, unit, "normalize(unit) != unit")
        
        // Dot product with self is 1
        XCTAssertEqual(unit.dotProduct(unit), 1.0, accuracy: accuracy, "unit · unit != 1")
        
        // L1 norm is 1 for axis-aligned unit vectors
        XCTAssertEqual(unit.l1Norm, 1.0, accuracy: accuracy, "||unit||₁ != 1")
        
        // L∞ norm is 1
        XCTAssertEqual(unit.lInfinityNorm, 1.0, accuracy: accuracy, "||unit||∞ != 1")
    }
    
    private func verifyDynamicUnitVectorProperties(_ unit: DynamicVector, axis: Int) {
        XCTAssertEqual(unit.magnitude, 1.0, accuracy: accuracy, "Dynamic unit magnitude != 1")
        XCTAssertEqual(unit.dotProduct(unit), 1.0, accuracy: accuracy, "Dynamic unit · unit != 1")
    }
    
    // MARK: - Maximum/Minimum Value Edge Cases
    
    func testExtremeValueOperations() {
        // Test with very large and very small values
        testLargeValueOperations()
        testSmallValueOperations()
        testMixedScaleOperations()
    }
    
    private func testLargeValueOperations() {
        // Test with large values that won't overflow Float
        let largeValue: Float = 1e10
        let values = [Float](repeating: largeValue, count: 128)
        let largeVector = Vector<Dim128>(values)
        
        // Magnitude should be computable
        let expectedMagnitude = largeValue * sqrt(128)
        XCTAssertEqual(largeVector.magnitude, expectedMagnitude, 
                      accuracy: expectedMagnitude * accuracy,
                      "Large vector magnitude incorrect")
        
        // Normalization should work
        let normalized = largeVector.normalized()
        XCTAssertEqual(normalized.magnitude, 1.0, accuracy: accuracy,
                      "Large vector normalization failed")
        
        // All normalized components should be equal
        let expectedComponent = 1.0 / sqrt(128)
        for i in 0..<normalized.scalarCount {
            XCTAssertEqual(normalized[i], expectedComponent, accuracy: accuracy,
                          "Normalized component incorrect")
        }
    }
    
    private func testSmallValueOperations() {
        // Test with very small values
        let smallValue: Float = 1e-10
        let values = [Float](repeating: smallValue, count: 64)
        let smallVector = Vector<Dim64>(values)
        
        // Operations should preserve small values
        let doubled = smallVector * 2.0
        for i in 0..<doubled.scalarCount {
            XCTAssertEqual(doubled[i], smallValue * 2.0, accuracy: smallValue,
                          "Small value multiplication failed")
        }
        
        // Distance between small vectors
        let otherSmall = Vector<Dim64>(repeating: smallValue * 2)
        let distance = smallVector.distance(to: otherSmall)
        let expectedDistance = smallValue * sqrt(64)
        XCTAssertEqual(distance, expectedDistance, accuracy: expectedDistance * 0.1,
                      "Small vector distance incorrect")
    }
    
    private func testMixedScaleOperations() {
        // Test with mixed large and small values
        var values = [Float]()
        for i in 0..<128 {
            values.append(i % 2 == 0 ? 1e8 : 1e-8)
        }
        let mixedVector = Vector<Dim128>(values)
        
        // Operations should handle mixed scales
        XCTAssertFalse(mixedVector.magnitude.isNaN, "Mixed scale magnitude is NaN")
        XCTAssertFalse(mixedVector.magnitude.isInfinite, "Mixed scale magnitude is Infinite")
        
        // Normalization should work
        let normalized = mixedVector.normalized()
        XCTAssertEqual(normalized.magnitude, 1.0, accuracy: accuracy * 10,
                      "Mixed scale normalization failed")
    }
    
    // MARK: - Dimension Boundary Edge Cases
    
    func testDimensionBoundaries() {
        // Test at specific dimension boundaries
        testPowerOfTwoDimensions()
        testNonPowerOfTwoDimensions()
        testSIMDBoundaries()
    }
    
    private func testPowerOfTwoDimensions() {
        // Test dimensions that are powers of 2 (often optimized)
        let powerOfTwoDims = [32, 64, 128, 256, 512, 1024]
        
        for dim in powerOfTwoDims {
            let v1 = DynamicVector.random(dimension: dim, in: -10...10)
            let v2 = DynamicVector.random(dimension: dim, in: -10...10)
            
            // Basic operations should work correctly
            let sum = v1 + v2
            XCTAssertEqual(sum.dimension, dim, "Sum dimension incorrect for \(dim)")
            
            // Verify element-wise correctness
            for i in 0..<dim {
                XCTAssertEqual(sum[i], v1[i] + v2[i], accuracy: accuracy,
                              "Addition incorrect at index \(i) for dimension \(dim)")
            }
            
            // Dot product
            var expectedDot: Float = 0
            for i in 0..<dim {
                expectedDot += v1[i] * v2[i]
            }
            XCTAssertEqual(v1.dotProduct(v2), expectedDot, accuracy: accuracy * Float(dim),
                          "Dot product incorrect for dimension \(dim)")
        }
    }
    
    private func testNonPowerOfTwoDimensions() {
        // Test dimensions that are not powers of 2
        let nonPowerOfTwoDims = [31, 63, 127, 255, 511, 768, 1536]
        
        for dim in nonPowerOfTwoDims {
            let v = DynamicVector.random(dimension: dim, in: -5...5)
            
            // Magnitude computation
            var expectedMagSquared: Float = 0
            for i in 0..<dim {
                expectedMagSquared += v[i] * v[i]
            }
            let expectedMag = sqrt(expectedMagSquared)
            XCTAssertEqual(v.magnitude, expectedMag, accuracy: expectedMag * accuracy,
                          "Magnitude incorrect for non-power-of-2 dimension \(dim)")
        }
    }
    
    private func testSIMDBoundaries() {
        // Test at SIMD width boundaries (typically 4, 8, 16 for Float)
        let simdBoundaries = [4, 8, 16, 24, 32, 48, 64]
        
        for dim in simdBoundaries {
            let v = DynamicVector.random(dimension: dim, in: -10...10)
            
            // Test operations that use SIMD
            let scaled = v * 2.5
            for i in 0..<dim {
                XCTAssertEqual(scaled[i], v[i] * 2.5, accuracy: accuracy,
                              "SIMD boundary scaling failed at dimension \(dim)")
            }
            
            // Test reduction operations
            let sum = v.sum
            var expectedSum: Float = 0
            for i in 0..<dim {
                expectedSum += v[i]
            }
            XCTAssertEqual(sum, expectedSum, accuracy: abs(expectedSum) * accuracy,
                          "Sum incorrect at SIMD boundary \(dim)")
        }
    }
    
    // MARK: - Special Floating Point Edge Cases
    
    func testSpecialFloatingPointCases() {
        // Test subnormal numbers
        testSubnormalNumbers()
        
        // Test operations near epsilon
        testNearEpsilonOperations()
        
        // Test catastrophic cancellation scenarios
        testCatastrophicCancellation()
    }
    
    private func testSubnormalNumbers() {
        // Subnormal (denormalized) numbers
        let subnormal = Float.leastNonzeroMagnitude
        let values = [Float](repeating: subnormal, count: 64)
        let subnormalVector = Vector<Dim64>(values)
        
        // Operations should handle subnormals
        let doubled = subnormalVector * 2.0
        for i in 0..<doubled.scalarCount {
            XCTAssertEqual(doubled[i], subnormal * 2.0, 
                          "Subnormal multiplication failed")
        }
    }
    
    private func testNearEpsilonOperations() {
        // Test operations with values near machine epsilon
        let epsilon = Float.ulpOfOne
        let v1 = Vector<Dim32>(repeating: 1.0)
        let v2 = Vector<Dim32>(repeating: 1.0 + epsilon)
        
        // Subtraction should preserve the difference
        let diff = v2 - v1
        for i in 0..<diff.scalarCount {
            XCTAssertEqual(diff[i], epsilon, accuracy: epsilon * 0.5,
                          "Near-epsilon subtraction failed")
        }
    }
    
    private func testCatastrophicCancellation() {
        // Test scenarios that could cause catastrophic cancellation
        let large: Float = 1e10
        let small: Float = 1.0
        
        // Create vector with large value and add small perturbation
        var values1 = [Float](repeating: large, count: 32)
        var values2 = [Float](repeating: large, count: 32)
        values2[0] += small
        
        let v1 = Vector<Dim32>(values1)
        let v2 = Vector<Dim32>(values2)
        
        // Distance should capture the small difference
        let distance = v1.distance(to: v2)
        XCTAssertEqual(distance, small, accuracy: small * 0.1,
                      "Catastrophic cancellation in distance calculation")
    }
    
    // MARK: - Empty and Single Element Edge Cases
    
    func testSingleElementVectors() {
        // While our vectors have minimum dimensions, test smallest supported sizes
        struct Dim1: Dimension {
            static let value = 1
            typealias Storage = SmallVectorStorage<Dim1>
        }
        
        let single = Vector<Dim1>([5.0])
        
        // Magnitude equals absolute value
        XCTAssertEqual(single.magnitude, 5.0, "Single element magnitude incorrect")
        
        // Normalized is ±1
        let normalized = single.normalized()
        XCTAssertEqual(normalized[0], 1.0, "Single element normalization incorrect")
        
        // Operations
        let doubled = single * 2.0
        XCTAssertEqual(doubled[0], 10.0, "Single element multiplication incorrect")
    }
    
    // MARK: - Orthogonal Vector Edge Cases
    
    func testOrthogonalVectors() {
        // Test with orthogonal vectors (dot product = 0)
        let v1 = Vector<Dim64>([1.0] + [Float](repeating: 0, count: 63))
        let v2 = Vector<Dim64>([0.0, 1.0] + [Float](repeating: 0, count: 62))
        
        // Dot product should be 0
        XCTAssertEqual(v1.dotProduct(v2), 0, accuracy: accuracy,
                      "Orthogonal vectors dot product != 0")
        
        // Cosine similarity should be 0
        XCTAssertEqual(v1.cosineSimilarity(to: v2), 0, accuracy: accuracy,
                      "Orthogonal vectors cosine similarity != 0")
        
        // Distance follows Pythagorean theorem
        let distance = v1.distance(to: v2)
        let expectedDistance = sqrt(2.0) // √(1² + 1²)
        XCTAssertEqual(distance, expectedDistance, accuracy: accuracy,
                      "Orthogonal vector distance incorrect")
    }
    
    // MARK: - Helper Methods
    
    private func assertVectorsEqual<D: Dimension>(_ a: Vector<D>, _ b: Vector<D>, 
                                                  _ message: String,
                                                  file: StaticString = #file, 
                                                  line: UInt = #line) {
        XCTAssertEqual(a.scalarCount, b.scalarCount, message, file: file, line: line)
        for i in 0..<a.scalarCount {
            XCTAssertEqual(a[i], b[i], accuracy: accuracy, 
                          "\(message) - differ at index \(i)", file: file, line: line)
        }
    }
}