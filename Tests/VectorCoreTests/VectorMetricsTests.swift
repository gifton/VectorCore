// VectorCore: Vector Metrics Tests
//
// Tests for quality metrics and serialization
//

import XCTest
@testable import VectorCore

final class VectorMetricsTests: XCTestCase {
    
    // MARK: - Sparsity Tests
    
    func testSparsityAllZeros() {
        let vector = Vector128()
        XCTAssertEqual(vector.sparsity(), 1.0, accuracy: 0.001)
    }
    
    func testSparsityNonZeros() {
        let vector = Vector128(repeating: 1.0)
        XCTAssertEqual(vector.sparsity(), 0.0, accuracy: 0.001)
    }
    
    func testSparsityMixed() {
        var values = [Float](repeating: 0, count: 128)
        // Set 32 elements to non-zero (25% dense, 75% sparse)
        for i in 0..<32 {
            values[i] = 1.0
        }
        let vector = Vector128(values)
        XCTAssertEqual(vector.sparsity(), 0.75, accuracy: 0.001)
    }
    
    func testSparsityWithThreshold() {
        let values = [Float](repeating: 0.1, count: 128)
        let vector = Vector128(values)
        
        // With small threshold, values are not sparse
        XCTAssertEqual(vector.sparsity(threshold: 0.01), 0.0, accuracy: 0.001)
        
        // With larger threshold, all values are sparse
        XCTAssertEqual(vector.sparsity(threshold: 0.2), 1.0, accuracy: 0.001)
    }
    
    func testSparsityDynamicVector() {
        var values = [Float](repeating: 0, count: 50)
        values[10] = 1.0
        values[20] = 2.0
        values[30] = 3.0
        // 3 non-zero out of 50 = 94% sparse
        
        let vector = DynamicVector(values)
        XCTAssertEqual(vector.sparsity(), 0.94, accuracy: 0.001)
    }
    
    // MARK: - Entropy Tests
    
    func testEntropyZeroVector() {
        let vector = Vector128()
        XCTAssertEqual(vector.entropy, 0.0, accuracy: 0.001)
    }
    
    func testEntropyUniformDistribution() {
        // Uniform distribution has maximum entropy
        let vector = Vector128(repeating: 1.0)
        let expectedEntropy = log(Float(128)) // Maximum entropy for 128 elements
        XCTAssertEqual(vector.entropy, expectedEntropy, accuracy: 0.01)
    }
    
    func testEntropySingleSpike() {
        // Single non-zero element has zero entropy
        var values = [Float](repeating: 0, count: 128)
        values[64] = 100.0
        let vector = Vector128(values)
        XCTAssertEqual(vector.entropy, 0.0, accuracy: 0.001)
    }
    
    func testEntropyBinaryDistribution() {
        // Our entropy implementation treats each element as a separate outcome
        // For a vector with 64 ones and 64 twos:
        // Total sum = 64*1 + 64*2 = 192
        // Each "1" has probability 1/192
        // Each "2" has probability 2/192 = 1/96
        // Entropy = -64*(1/192)*log(1/192) - 64*(1/96)*log(1/96)
        
        var values = [Float](repeating: 1.0, count: 64)
        values.append(contentsOf: [Float](repeating: 2.0, count: 64))
        let vector = Vector128(values)
        
        // Calculate expected entropy based on our implementation
        let totalSum: Float = 192.0
        let p1 = 1.0 / totalSum  // probability for each element with value 1
        let p2 = 2.0 / totalSum  // probability for each element with value 2
        let expectedEntropy = -64 * p1 * log(p1) - 64 * p2 * log(p2)
        
        XCTAssertEqual(vector.entropy, expectedEntropy, accuracy: 0.01)
    }
    
    func testEntropyNegativeValues() {
        // Entropy should handle negative values by using absolute values
        let values = [Float]([-1.0, -2.0, -3.0, -4.0])
        let vector = DynamicVector(values)
        
        // Should be same as positive values
        let positiveVector = DynamicVector([1.0, 2.0, 3.0, 4.0])
        XCTAssertEqual(vector.entropy, positiveVector.entropy, accuracy: 0.001)
    }
    
    // MARK: - Quality Tests
    
    func testQualityZeroVector() {
        let vector = Vector128()
        let quality = vector.quality
        
        XCTAssertEqual(quality.magnitude, 0.0, accuracy: 0.001)
        XCTAssertEqual(quality.variance, 0.0, accuracy: 0.001)
        XCTAssertEqual(quality.sparsity, 1.0, accuracy: 0.001)
        XCTAssertEqual(quality.entropy, 0.0, accuracy: 0.001)
        XCTAssertTrue(quality.isZero)
        XCTAssertTrue(quality.isSparse)
        XCTAssertTrue(quality.isConcentrated)
    }
    
    func testQualityUniformVector() {
        let vector = Vector256(repeating: 2.0)
        let quality = vector.quality
        
        XCTAssertGreaterThan(quality.magnitude, 0.0)
        XCTAssertEqual(quality.variance, 0.0, accuracy: 0.001) // No variance in uniform
        XCTAssertEqual(quality.sparsity, 0.0, accuracy: 0.001)
        XCTAssertGreaterThan(quality.entropy, 0.0) // Maximum entropy
        XCTAssertFalse(quality.isZero)
        XCTAssertFalse(quality.isSparse)
        XCTAssertFalse(quality.isConcentrated)
    }
    
    func testQualityScore() {
        // Test that quality score is reasonable
        let goodVector = Vector512.random(in: -1...1)
        let quality = goodVector.quality
        
        XCTAssertGreaterThanOrEqual(quality.score, 0.0)
        XCTAssertLessThanOrEqual(quality.score, 1.0)
    }
    
    func testQualityComparison() {
        let sparseVector = Vector256([1.0] + [Float](repeating: 0, count: 255))
        let denseVector = Vector256.random(in: -1...1)
        
        let sparseQuality = sparseVector.quality
        let denseQuality = denseVector.quality
        
        // Dense vector should have better quality score
        XCTAssertGreaterThan(denseQuality.score, sparseQuality.score)
    }
    
    func testQualityDynamicVector() {
        let vector = DynamicVector.random(dimension: 100, in: -2...2)
        let quality = vector.quality
        
        XCTAssertGreaterThan(quality.magnitude, 0.0)
        XCTAssertGreaterThan(quality.variance, 0.0)
        XCTAssertLessThan(quality.sparsity, 0.1) // Random vectors are rarely sparse
        XCTAssertGreaterThan(quality.entropy, 0.0)
    }
    
    // MARK: - Base64 Serialization Tests
    
    func testBase64EncodingDecoding() throws {
        let original = Vector768.random(in: -10...10)
        let encoded = original.base64Encoded
        
        XCTAssertFalse(encoded.isEmpty)
        
        let decoded = try Vector768.base64Decoded(from: encoded)
        
        // Compare element by element due to potential floating point precision
        for i in 0..<768 {
            XCTAssertEqual(original[i], decoded[i], accuracy: 0.0001)
        }
    }
    
    func testBase64RoundTripDifferentDimensions() throws {
        // Test various dimensions
        let vec128 = Vector128.random(in: -1...1)
        let vec256 = Vector256.random(in: -1...1)
        let vec512 = Vector512.random(in: -1...1)
        let vec768 = Vector768.random(in: -1...1)
        let vec1536 = Vector1536.random(in: -1...1)
        let vec3072 = Vector3072.random(in: -1...1)
        
        // Test each dimension
        XCTAssertFalse(vec128.base64Encoded.isEmpty)
        XCTAssertFalse(vec256.base64Encoded.isEmpty)
        XCTAssertFalse(vec512.base64Encoded.isEmpty)
        XCTAssertFalse(vec768.base64Encoded.isEmpty)
        XCTAssertFalse(vec1536.base64Encoded.isEmpty)
        XCTAssertFalse(vec3072.base64Encoded.isEmpty)
        
        // Test round-trip for each
        let decoded128 = try Vector128.base64Decoded(from: vec128.base64Encoded)
        for i in 0..<128 {
            XCTAssertEqual(vec128[i], decoded128[i], accuracy: 0.0001)
        }
        
        let decoded256 = try Vector256.base64Decoded(from: vec256.base64Encoded)
        for i in 0..<256 {
            XCTAssertEqual(vec256[i], decoded256[i], accuracy: 0.0001)
        }
    }
    
    func testBase64DynamicVector() throws {
        let original = DynamicVector.random(dimension: 200, in: -5...5)
        let encoded = original.base64Encoded
        
        XCTAssertFalse(encoded.isEmpty)
        
        let decoded = try DynamicVector.base64Decoded(from: encoded)
        
        XCTAssertEqual(original.dimension, decoded.dimension)
        for i in 0..<original.dimension {
            XCTAssertEqual(original[i], decoded[i], accuracy: 0.0001)
        }
    }
    
    func testBase64InvalidString() {
        XCTAssertThrowsError(try Vector256.base64Decoded(from: "invalid base64!@#$")) { error in
            if let vectorError = error as? VectorError {
                switch vectorError {
                case .invalidDataFormat:
                    break // Expected
                default:
                    XCTFail("Unexpected error type: \(vectorError)")
                }
            } else {
                XCTFail("Unexpected error type: \(error)")
            }
        }
    }
    
    func testBase64EmptyVector() throws {
        let vector = Vector128()
        let encoded = vector.base64Encoded
        let decoded = try Vector128.base64Decoded(from: encoded)
        
        for i in 0..<128 {
            XCTAssertEqual(decoded[i], 0.0, accuracy: 0.0001)
        }
    }
    
    // MARK: - VectorQuality Tests
    
    func testVectorQualityDescription() {
        let quality = VectorQuality(
            magnitude: 10.5,
            variance: 2.3,
            sparsity: 0.754,
            entropy: 1.234
        )
        
        let description = quality.description
        XCTAssertTrue(description.contains("10.500"))
        XCTAssertTrue(description.contains("2.300"))
        XCTAssertTrue(description.contains("75.4%"))
        XCTAssertTrue(description.contains("1.234"))
    }
    
    func testVectorQualityCodable() throws {
        let quality = VectorQuality(
            magnitude: 5.0,
            variance: 1.0,
            sparsity: 0.5,
            entropy: 2.0
        )
        
        let encoder = JSONEncoder()
        let data = try encoder.encode(quality)
        
        let decoder = JSONDecoder()
        let decoded = try decoder.decode(VectorQuality.self, from: data)
        
        XCTAssertEqual(quality, decoded)
    }
    
    func testVectorQualityComparable() {
        let lowQuality = VectorQuality(
            magnitude: 0.1,
            variance: 0.001,
            sparsity: 0.9,
            entropy: 0.1
        )
        
        let highQuality = VectorQuality(
            magnitude: 10.0,
            variance: 2.0,
            sparsity: 0.3,
            entropy: 3.0
        )
        
        XCTAssertLessThan(lowQuality, highQuality)
        XCTAssertGreaterThan(highQuality, lowQuality)
    }
    
    // MARK: - Edge Cases
    
    func testMetricsWithInfinityAndNaN() {
        var values = [Float](repeating: 1.0, count: 128)
        values[0] = .infinity
        values[1] = -.infinity
        values[2] = .nan
        
        let vector = Vector128(values)
        
        // Sparsity should count only finite values
        let sparsity = vector.sparsity()
        XCTAssertEqual(sparsity, 0.0, accuracy: 0.001) // Non-zero finite values
        
        // Entropy should return NaN when vector contains non-finite values
        let entropy = vector.entropy
        XCTAssertTrue(entropy.isNaN)
        
        // Quality should handle NaN values appropriately
        let quality = vector.quality
        XCTAssertTrue(quality.entropy.isNaN)
        XCTAssertEqual(quality.sparsity, 0.0, accuracy: 0.001)
    }
    
    func testVeryLargeValues() {
        let values = [Float](repeating: 1e20, count: 256)
        let vector = Vector256(values)
        
        let quality = vector.quality
        XCTAssertFalse(quality.magnitude.isNaN)
        // Very large values will have infinite magnitude - this is expected
        XCTAssertTrue(quality.magnitude.isInfinite)
        XCTAssertFalse(quality.entropy.isNaN)
        // For uniform distribution, entropy should be log(256) regardless of scale
        XCTAssertEqual(quality.entropy, log(Float(256)), accuracy: 0.01)
    }
    
    // MARK: - Approximate Equality Tests
    
    func testApproximateEqualityVector() {
        let v1 = Vector128(repeating: 1.0)
        var v2 = Vector128(repeating: 1.0)
        
        // Exact equality
        XCTAssertTrue(v1 == v2)
        XCTAssertTrue(v1.isApproximatelyEqual(to: v2))
        
        // Small difference within tolerance
        v2[0] = 1.0 + 1e-7
        XCTAssertFalse(v1 == v2) // Exact equality fails
        XCTAssertTrue(v1.isApproximatelyEqual(to: v2)) // Approximate equality succeeds
        
        // Difference exceeds default tolerance
        v2[0] = 1.0 + 2e-6
        XCTAssertFalse(v1.isApproximatelyEqual(to: v2))
        
        // Custom tolerance
        XCTAssertTrue(v1.isApproximatelyEqual(to: v2, tolerance: 1e-5))
    }
    
    func testApproximateEqualityDynamicVector() {
        let v1 = DynamicVector([1.0, 2.0, 3.0])
        var v2 = DynamicVector([1.0, 2.0, 3.0])
        
        // Exact equality
        XCTAssertTrue(v1 == v2)
        XCTAssertTrue(v1.isApproximatelyEqual(to: v2))
        
        // Small difference
        v2[1] = 2.0 + 5e-7
        XCTAssertFalse(v1 == v2)
        XCTAssertTrue(v1.isApproximatelyEqual(to: v2))
        
        // Different dimensions
        let v3 = DynamicVector([1.0, 2.0])
        XCTAssertFalse(v1.isApproximatelyEqual(to: v3))
    }
}

// MARK: - Performance Tests

extension VectorMetricsTests {
    func testSparsityPerformance() {
        let vector = Vector3072.random(in: -1...1)
        
        measure {
            for _ in 0..<1000 {
                _ = vector.sparsity()
            }
        }
    }
    
    func testEntropyPerformance() {
        let vector = Vector3072.random(in: -1...1)
        
        measure {
            for _ in 0..<1000 {
                _ = vector.entropy
            }
        }
    }
    
    func testQualityPerformance() {
        let vector = Vector3072.random(in: -1...1)
        
        measure {
            for _ in 0..<100 {
                _ = vector.quality
            }
        }
    }
    
    func testBase64Performance() {
        let vector = Vector3072.random(in: -1...1)
        
        measure {
            for _ in 0..<100 {
                _ = vector.base64Encoded
            }
        }
    }
}