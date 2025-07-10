// VectorCore: Vector Math Tests
//
// Comprehensive tests for mathematical operations on vectors
//

import XCTest
@testable import VectorCore

final class VectorMathTests: XCTestCase {
    
    // MARK: - Test Vectors
    
    let v1_128 = Vector128(repeating: 2.0)
    let v2_128 = Vector128(repeating: 3.0)
    
    var v1_256: Vector256!
    var v2_256: Vector256!
    var v3_256: Vector256!
    
    override func setUp() {
        super.setUp()
        
        // Create test vectors with known values
        let values1 = (0..<256).map { Float($0) / 256.0 }
        let values2 = (0..<256).map { Float(255 - $0) / 256.0 }
        let values3 = (0..<256).map { i in Float(sin(Double(i) * .pi / 128)) }
        
        v1_256 = Vector256(values1)
        v2_256 = Vector256(values2)
        v3_256 = Vector256(values3)
    }
    
    // MARK: - Element-wise Operations
    
    func testElementWiseMultiplication() {
        // Test basic multiplication
        let result = v1_128 .* v2_128
        let expected = Vector128(repeating: 6.0) // 2 * 3
        
        for i in 0..<128 {
            XCTAssertEqual(result[i], expected[i], accuracy: 1e-6)
        }
        
        // Test with varying values
        let result256 = v1_256 .* v2_256
        for i in 0..<256 {
            let expected = v1_256[i] * v2_256[i]
            XCTAssertEqual(result256[i], expected, accuracy: 1e-6)
        }
    }
    
    func testElementWiseDivision() {
        // Test basic division
        let result = v1_128 ./ v2_128
        let expected = Vector128(repeating: 2.0/3.0)
        
        for i in 0..<128 {
            XCTAssertEqual(result[i], expected[i], accuracy: 1e-6)
        }
        
        // Test with varying values (avoiding division by zero)
        let nonZero = Vector256(repeating: 0.5) + v2_256
        let result256 = v1_256 ./ nonZero
        
        for i in 0..<256 {
            let expected = v1_256[i] / nonZero[i]
            XCTAssertEqual(result256[i], expected, accuracy: 1e-6)
        }
    }
    
    func testElementWiseOperationsWithZero() {
        let zeros = Vector128(repeating: 0.0)
        let ones = Vector128(repeating: 1.0)
        
        // Multiplication with zero
        let multResult = ones .* zeros
        for i in 0..<128 {
            XCTAssertEqual(multResult[i], 0.0)
        }
        
        // Division by zero should produce infinity
        let divResult = ones ./ zeros
        for i in 0..<128 {
            XCTAssertTrue(divResult[i].isInfinite)
        }
    }
    
    func testElementWiseOperationsWithSpecialValues() {
        var special = Vector128(repeating: 1.0)
        special[0] = .nan
        special[1] = .infinity
        special[2] = -.infinity
        
        let result = special .* v1_128
        
        XCTAssertTrue(result[0].isNaN)
        XCTAssertTrue(result[1].isInfinite && result[1] > 0)
        XCTAssertTrue(result[2].isInfinite && result[2] < 0)
    }
    
    // MARK: - Norm Tests
    
    func testL1Norm() {
        // Test with known values
        let v = Vector128([-1, 2, -3, 4] + Array(repeating: Float(0), count: 124))
        let l1 = v.l1Norm
        
        XCTAssertEqual(l1, 10.0, accuracy: 1e-6) // |−1| + |2| + |−3| + |4| = 10
        
        // Test with all positive
        let positive = Vector128(repeating: 2.0)
        XCTAssertEqual(positive.l1Norm, 256.0, accuracy: 1e-6) // 2 * 128
        
        // Test with mixed values
        let l1_256 = v3_256.l1Norm
        var expectedL1: Float = 0
        for i in 0..<256 {
            expectedL1 += abs(v3_256[i])
        }
        XCTAssertEqual(l1_256, expectedL1, accuracy: 1e-5)
    }
    
    func testL2Norm() {
        // L2 norm should equal magnitude
        let v = Vector128([3, 4] + Array(repeating: Float(0), count: 126))
        
        XCTAssertEqual(v.l2Norm, 5.0, accuracy: 1e-6) // sqrt(9 + 16) = 5
        XCTAssertEqual(v.l2Norm, v.magnitude, accuracy: 1e-6)
        
        // Test larger vector
        let l2_256 = v1_256.l2Norm
        XCTAssertEqual(l2_256, v1_256.magnitude, accuracy: 1e-6)
    }
    
    func testLInfinityNorm() {
        // Test with known values
        let v = Vector128([-5, 3, -8, 2] + Array(repeating: Float(0), count: 124))
        let lInf = v.lInfinityNorm
        
        XCTAssertEqual(lInf, 8.0, accuracy: 1e-6) // max(|−5|, |3|, |−8|, |2|) = 8
        
        // Test with all same magnitude
        let uniform = Vector128(repeating: -3.0)
        XCTAssertEqual(uniform.lInfinityNorm, 3.0, accuracy: 1e-6)
        
        // Test edge case with zero
        let zeros = Vector128(repeating: 0.0)
        XCTAssertEqual(zeros.lInfinityNorm, 0.0, accuracy: 1e-6)
    }
    
    func testNormConsistency() {
        // L∞ ≤ L2 ≤ L1
        let v = Vector256.random(in: -1...1)
        
        let l1 = v.l1Norm
        let l2 = v.l2Norm
        let lInf = v.lInfinityNorm
        
        XCTAssertLessThanOrEqual(lInf, l2)
        XCTAssertLessThanOrEqual(l2, l1)
    }
    
    // MARK: - Statistical Operations
    
    func testMean() {
        // Test with uniform values
        let uniform = Vector128(repeating: 5.0)
        XCTAssertEqual(uniform.mean, 5.0, accuracy: 1e-6)
        
        // Test with known sequence
        let sequence = Vector128((0..<128).map { Float($0) })
        let expectedMean = Float(127) / 2.0 // (0 + 127) / 2
        XCTAssertEqual(sequence.mean, expectedMean, accuracy: 1e-6)
        
        // Test with mixed positive/negative
        XCTAssertEqual(v3_256.mean, v3_256.sum / Float(256), accuracy: 1e-6)
    }
    
    func testSum() {
        // Test basic sum
        let ones = Vector128(repeating: 1.0)
        XCTAssertEqual(ones.sum, 128.0, accuracy: 1e-6)
        
        // Test with known sequence
        let sequence = Vector128((0..<128).map { Float($0) })
        let expectedSum = Float(127 * 128 / 2) // n(n-1)/2
        XCTAssertEqual(sequence.sum, expectedSum, accuracy: 1e-6)
        
        // Test empty-ish (all zeros)
        let zeros = Vector128(repeating: 0.0)
        XCTAssertEqual(zeros.sum, 0.0, accuracy: 1e-6)
    }
    
    func testVariance() {
        // Test zero variance (all same values)
        let uniform = Vector128(repeating: 3.0)
        XCTAssertEqual(uniform.variance, 0.0, accuracy: 1e-6)
        
        // Test with known values
        let v = Vector128([1, 2, 3, 4] + Array(repeating: Float(2.5), count: 124))
        let mean = v.mean
        
        var expectedVariance: Float = 0
        for i in 0..<128 {
            expectedVariance += pow(v[i] - mean, 2)
        }
        expectedVariance /= 128
        
        XCTAssertEqual(v.variance, expectedVariance, accuracy: 1e-5)
    }
    
    func testStandardDeviation() {
        // Standard deviation should be sqrt of variance
        let v = Vector256.random(in: -1...1)
        
        let std = v.standardDeviation
        let variance = v.variance
        
        XCTAssertEqual(std, sqrt(variance), accuracy: 1e-6)
        
        // Test zero standard deviation
        let uniform = Vector128(repeating: 7.0)
        XCTAssertEqual(uniform.standardDeviation, 0.0, accuracy: 1e-6)
    }
    
    // MARK: - Distance Metric Extensions
    
    func testManhattanDistanceExtension() {
        let v1 = Vector128([1, 2, 3, 4] + Array(repeating: Float(0), count: 124))
        let v2 = Vector128([2, 4, 1, 8] + Array(repeating: Float(0), count: 124))
        
        let distance = v1.manhattanDistance(to: v2)
        let expected: Float = 1 + 2 + 2 + 4 // |1-2| + |2-4| + |3-1| + |4-8|
        
        XCTAssertEqual(distance, expected, accuracy: 1e-6)
        
        // Should match ManhattanDistance metric
        let metric = ManhattanDistance()
        XCTAssertEqual(distance, metric.distance(v1, v2), accuracy: 1e-6)
    }
    
    func testChebyshevDistanceExtension() {
        let v1 = Vector128([1, 2, 3, 4] + Array(repeating: Float(0), count: 124))
        let v2 = Vector128([2, 4, 1, 8] + Array(repeating: Float(0), count: 124))
        
        let distance = v1.chebyshevDistance(to: v2)
        let expected: Float = 4 // max(|1-2|, |2-4|, |3-1|, |4-8|) = 4
        
        XCTAssertEqual(distance, expected, accuracy: 1e-6)
        
        // Should match ChebyshevDistance metric
        let metric = ChebyshevDistance()
        XCTAssertEqual(distance, metric.distance(v1, v2), accuracy: 1e-6)
    }
    
    // MARK: - Special Operations
    
    func testSoftmax() {
        // Test basic softmax properties
        let v = Vector128([1, 2, 3, 4] + Array(repeating: Float(0), count: 124))
        let softmaxed = v.softmax()
        
        // Sum should be 1
        XCTAssertEqual(softmaxed.sum, 1.0, accuracy: 1e-5)
        
        // All values should be positive
        for i in 0..<128 {
            XCTAssertGreaterThan(softmaxed[i], 0)
            XCTAssertLessThanOrEqual(softmaxed[i], 1.0)
        }
        
        // Larger values should have larger softmax
        XCTAssertGreaterThan(softmaxed[3], softmaxed[2])
        XCTAssertGreaterThan(softmaxed[2], softmaxed[1])
        XCTAssertGreaterThan(softmaxed[1], softmaxed[0])
    }
    
    func testSoftmaxNumericalStability() {
        // Test with large values (should not overflow)
        var large = Vector128(repeating: 0.0)
        large[0] = 1000
        large[1] = 1001
        
        let softmaxed = large.softmax()
        
        // Should not contain NaN or infinity
        for i in 0..<128 {
            XCTAssertFalse(softmaxed[i].isNaN)
            XCTAssertFalse(softmaxed[i].isInfinite)
        }
        
        // Sum should still be 1
        XCTAssertEqual(softmaxed.sum, 1.0, accuracy: 1e-5)
    }
    
    func testClamped() {
        // Test basic clamping
        let v = Vector128([-2, -1, 0, 1, 2] + Array(repeating: Float(0.5), count: 123))
        let clamped = v.clamped(to: -1...1)
        
        XCTAssertEqual(clamped[0], -1.0) // -2 clamped to -1
        XCTAssertEqual(clamped[1], -1.0) // -1 stays -1
        XCTAssertEqual(clamped[2], 0.0)  // 0 stays 0
        XCTAssertEqual(clamped[3], 1.0)  // 1 stays 1
        XCTAssertEqual(clamped[4], 1.0)  // 2 clamped to 1
        
        // Values in range should be unchanged
        for i in 5..<128 {
            XCTAssertEqual(clamped[i], 0.5)
        }
    }
    
    func testClampedWithSpecialValues() {
        var v = Vector128(repeating: 0.0)
        v[0] = .nan
        v[1] = .infinity
        v[2] = -.infinity
        
        let clamped = v.clamped(to: -1...1)
        
        // NaN should remain NaN
        XCTAssertTrue(clamped[0].isNaN)
        
        // Infinities should be clamped
        XCTAssertEqual(clamped[1], 1.0)
        XCTAssertEqual(clamped[2], -1.0)
    }
    
    // MARK: - Vector Creation Helpers
    
    func testRandomVectorCreation() {
        let range: ClosedRange<Float> = -2...3
        let v = Vector128.random(in: range)
        
        // All values should be in range
        for i in 0..<128 {
            XCTAssertGreaterThanOrEqual(v[i], range.lowerBound)
            XCTAssertLessThanOrEqual(v[i], range.upperBound)
        }
        
        // Should have some variety (not all the same)
        let uniqueValues = Set(v.toArray())
        XCTAssertGreaterThan(uniqueValues.count, 1)
    }
    
    func testBasisVectorCreation() {
        // Test basis vector at index 5
        let basis = Vector128.basis(at: 5)
        
        for i in 0..<128 {
            if i == 5 {
                XCTAssertEqual(basis[i], 1.0)
            } else {
                XCTAssertEqual(basis[i], 0.0)
            }
        }
        
        // Test edge cases
        let first = Vector128.basis(at: 0)
        XCTAssertEqual(first[0], 1.0)
        
        let last = Vector128.basis(at: 127)
        XCTAssertEqual(last[127], 1.0)
    }
    
    func testZeroAndOneVectors() {
        let zero = Vector128.zero
        let one = Vector128.one
        
        for i in 0..<128 {
            XCTAssertEqual(zero[i], 0.0)
            XCTAssertEqual(one[i], 1.0)
        }
        
        // Test properties
        XCTAssertEqual(zero.magnitude, 0.0)
        XCTAssertEqual(zero.sum, 0.0)
        XCTAssertEqual(one.sum, 128.0)
    }
    
    // MARK: - VectorMath Batch Operations
    
    func testNearestNeighbors() {
        let vectors = (0..<100).map { i in
            Vector128(repeating: Float(i))
        }
        let query = Vector128(repeating: 50.5)
        
        let neighbors = VectorMath.nearestNeighbors(
            query: query,
            in: vectors,
            k: 5,
            using: { v1, v2 in v1.distance(to: v2) }
        )
        
        XCTAssertEqual(neighbors.count, 5)
        
        // Should find vectors 50 and 51 as closest
        XCTAssertTrue(neighbors[0].index == 50 || neighbors[0].index == 51)
        XCTAssertTrue(neighbors[1].index == 50 || neighbors[1].index == 51)
        
        // Check ordering
        for i in 1..<neighbors.count {
            XCTAssertLessThanOrEqual(neighbors[i-1].distance, neighbors[i].distance)
        }
    }
    
    func testPairwiseDistances() {
        let vectors = [
            Vector128([1, 0, 0] + Array(repeating: Float(0), count: 125)),
            Vector128([0, 1, 0] + Array(repeating: Float(0), count: 125)),
            Vector128([0, 0, 1] + Array(repeating: Float(0), count: 125))
        ]
        
        let distances = VectorMath.pairwiseDistances(vectors) { v1, v2 in
            v1.distance(to: v2)
        }
        
        // Check dimensions
        XCTAssertEqual(distances.count, 3)
        XCTAssertEqual(distances[0].count, 3)
        
        // Diagonal should be 0
        for i in 0..<3 {
            XCTAssertEqual(distances[i][i], 0.0, accuracy: 1e-6)
        }
        
        // Off-diagonal should be sqrt(2) for these unit vectors
        let expectedDist = sqrt(2.0)
        for i in 0..<3 {
            for j in 0..<3 where i != j {
                XCTAssertEqual(distances[i][j], expectedDist, accuracy: 1e-6)
            }
        }
    }
    
    // MARK: - Performance Tests
    
    func testElementWiseOperationPerformance() {
        let v1 = Vector256.random(in: -1...1)
        let v2 = Vector256.random(in: -1...1)
        
        measure {
            for _ in 0..<1000 {
                _ = v1 .* v2
            }
        }
    }
    
    func testNormPerformance() {
        let v = Vector512.random(in: -1...1)
        
        measure {
            for _ in 0..<1000 {
                _ = v.l1Norm
                _ = v.l2Norm
                _ = v.lInfinityNorm
            }
        }
    }
    
    func testStatisticalOperationsPerformance() {
        let v = Vector512.random(in: -1...1)
        
        measure {
            for _ in 0..<1000 {
                _ = v.mean
                _ = v.variance
                _ = v.standardDeviation
            }
        }
    }
    
    func testSoftmaxPerformance() {
        let v = Vector256.random(in: -10...10)
        
        measure {
            for _ in 0..<1000 {
                _ = v.softmax()
            }
        }
    }
}