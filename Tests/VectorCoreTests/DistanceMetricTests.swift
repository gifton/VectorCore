// VectorCoreTests: Distance Metric Tests
//
// Comprehensive tests for distance metrics
//

import XCTest
@testable import VectorCore

final class DistanceMetricTests: XCTestCase {
    
    // MARK: - Test Vectors
    
    let vectorA = DynamicVector([1, 2, 3, 4, 5, 6, 7, 8])
    let vectorB = DynamicVector([2, 4, 6, 8, 10, 12, 14, 16])
    let vectorC = DynamicVector([1, 0, 1, 0, 1, 0, 1, 0])
    let vectorD = DynamicVector([0, 1, 0, 1, 0, 1, 0, 1])
    
    // MARK: - Euclidean Distance Tests
    
    func testEuclideanDistance() {
        let metric = EuclideanDistance()
        
        // Same vector should have distance 0
        let sameDist = metric.distance(vectorA, vectorA)
        XCTAssertEqual(sameDist, 0.0, accuracy: 0.001)
        
        // Known distance calculation
        let dist = metric.distance(vectorA, vectorB)
        // vectorB = 2 * vectorA, so distance = ||vectorA|| = sqrt(1+4+9+16+25+36+49+64) = sqrt(204)
        let expected = sqrt(204.0)
        XCTAssertEqual(Double(dist), expected, accuracy: 0.001)
        
        // Test with orthogonal binary vectors
        let orthoDist = metric.distance(vectorC, vectorD)
        // Each position differs by 1, so distance = sqrt(8)
        XCTAssertEqual(orthoDist, sqrt(8.0), accuracy: 0.001)
    }
    
    func testEuclideanBatchDistance() {
        let metric = EuclideanDistance()
        let query = vectorA
        let candidates = [vectorA, vectorB, vectorC, vectorD]
        
        let distances = metric.batchDistance(query: query, candidates: candidates)
        
        XCTAssertEqual(distances.count, 4)
        XCTAssertEqual(distances[0], 0.0, accuracy: 0.001) // Same vector
        XCTAssertEqual(distances[1], sqrt(204.0), accuracy: 0.001)
    }
    
    // MARK: - Cosine Distance Tests
    
    func testCosineDistance() {
        let metric = CosineDistance()
        
        // Same vector should have distance 0 (cosine similarity = 1)
        let sameDist = metric.distance(vectorA, vectorA)
        XCTAssertEqual(sameDist, 0.0, accuracy: 0.001)
        
        // Parallel vectors (vectorB = 2 * vectorA) should have distance 0
        let parallelDist = metric.distance(vectorA, vectorB)
        XCTAssertEqual(parallelDist, 0.0, accuracy: 0.001)
        
        // Test with normalized vectors
        let normalized1 = SIMD4<Float>(1, 0, 0, 0)
        let normalized2 = SIMD4<Float>(0, 1, 0, 0)
        let orthogonalDist = metric.distance(normalized1, normalized2)
        XCTAssertEqual(orthogonalDist, 1.0, accuracy: 0.001) // Orthogonal vectors
    }
    
    func testCosineDistanceEdgeCases() {
        let metric = CosineDistance()
        
        // Zero vector handling
        let zeroVector = SIMD8<Float>(repeating: 0)
        let dist = metric.distance(vectorA, zeroVector)
        XCTAssertEqual(dist, 1.0) // Maximum distance for zero vector
        
        // Opposite vectors
        let opposite = SIMD4<Float>(1, 0, 0, 0)
        let negativeOpposite = SIMD4<Float>(-1, 0, 0, 0)
        let oppositeDist = metric.distance(opposite, negativeOpposite)
        XCTAssertEqual(oppositeDist, 2.0, accuracy: 0.001) // 1 - (-1) = 2
    }
    
    // MARK: - Dot Product Distance Tests
    
    func testDotProductDistance() {
        let metric = DotProductDistance()
        
        // Dot product of orthogonal vectors should be 0, so distance = 0
        let orthoDist = metric.distance(vectorC, vectorD)
        XCTAssertEqual(orthoDist, 0.0, accuracy: 0.001)
        
        // Dot product of vectorA with itself
        let selfDot = metric.distance(vectorA, vectorA)
        let expectedSelfDot = -(1+4+9+16+25+36+49+64) // -204
        XCTAssertEqual(selfDot, Float(expectedSelfDot), accuracy: 0.001)
        
        // Dot product of vectorA with vectorB (2*vectorA)
        let scaledDot = metric.distance(vectorA, vectorB)
        let expectedScaledDot = -2 * 204 // -408
        XCTAssertEqual(scaledDot, Float(expectedScaledDot), accuracy: 0.001)
    }
    
    // MARK: - Manhattan Distance Tests
    
    func testManhattanDistance() {
        let metric = ManhattanDistance()
        
        // Same vector
        let sameDist = metric.distance(vectorA, vectorA)
        XCTAssertEqual(sameDist, 0.0, accuracy: 0.001)
        
        // Known calculation
        let dist = metric.distance(vectorA, vectorB)
        // |2-1| + |4-2| + |6-3| + |8-4| + |10-5| + |12-6| + |14-7| + |16-8|
        // = 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 = 36
        XCTAssertEqual(dist, 36.0, accuracy: 0.001)
        
        // Binary vectors
        let binaryDist = metric.distance(vectorC, vectorD)
        XCTAssertEqual(binaryDist, 8.0, accuracy: 0.001) // All positions differ by 1
    }
    
    // MARK: - Hamming Distance Tests
    
    func testHammingDistance() {
        let metric = HammingDistance()
        
        // Same vector
        let sameDist = metric.distance(vectorA, vectorA)
        XCTAssertEqual(sameDist, 0.0)
        
        // All different
        let dist = metric.distance(vectorC, vectorD)
        XCTAssertEqual(dist, 8.0) // All 8 positions are different
        
        // Some different
        let partial1 = SIMD4<Float>(1, 2, 3, 4)
        let partial2 = SIMD4<Float>(1, 2, 0, 0)
        let partialDist = metric.distance(partial1, partial2)
        XCTAssertEqual(partialDist, 2.0) // 2 positions differ
    }
    
    // MARK: - Chebyshev Distance Tests
    
    func testChebyshevDistance() {
        let metric = ChebyshevDistance()
        
        // Same vector
        let sameDist = metric.distance(vectorA, vectorA)
        XCTAssertEqual(sameDist, 0.0)
        
        // Maximum coordinate difference
        let dist = metric.distance(vectorA, vectorB)
        // Max difference is at position 7: |16-8| = 8
        XCTAssertEqual(dist, 8.0, accuracy: 0.001)
        
        // Test with specific max difference
        let v1 = SIMD4<Float>(1, 2, 3, 4)
        let v2 = SIMD4<Float>(1, 2, 3, 14) // Max diff = 10
        let maxDist = metric.distance(v1, v2)
        XCTAssertEqual(maxDist, 10.0, accuracy: 0.001)
    }
    
    // MARK: - Minkowski Distance Tests
    
    func testMinkowskiDistance() {
        // Test p=1 (Manhattan)
        let metric1 = MinkowskiDistance(p: 1)
        let dist1 = metric1.distance(vectorA, vectorB)
        let manhattan = ManhattanDistance().distance(vectorA, vectorB)
        XCTAssertEqual(dist1, manhattan, accuracy: 0.001)
        
        // Test p=2 (Euclidean)
        let metric2 = MinkowskiDistance(p: 2)
        let dist2 = metric2.distance(vectorA, vectorB)
        let euclidean = EuclideanDistance().distance(vectorA, vectorB)
        XCTAssertEqual(dist2, euclidean, accuracy: 0.001)
        
        // Test p=infinity (Chebyshev)
        let metricInf = MinkowskiDistance(p: Float.infinity)
        let distInf = metricInf.distance(vectorA, vectorB)
        let chebyshev = ChebyshevDistance().distance(vectorA, vectorB)
        XCTAssertEqual(distInf, chebyshev, accuracy: 0.001)
        
        // Test p=3
        let metric3 = MinkowskiDistance(p: 3)
        let dist3 = metric3.distance(vectorC, vectorD)
        // All 8 positions differ by 1, so sum = 8, result = 8^(1/3) = 2
        XCTAssertEqual(dist3, 2.0, accuracy: 0.001)
    }
    
    // MARK: - Jaccard Distance Tests
    
    func testJaccardDistance() {
        let metric = JaccardDistance()
        
        // Same vector
        let sameDist = metric.distance(vectorC, vectorC)
        XCTAssertEqual(sameDist, 0.0, accuracy: 0.001)
        
        // Completely different binary vectors
        let dist = metric.distance(vectorC, vectorD)
        XCTAssertEqual(dist, 1.0, accuracy: 0.001) // No intersection
        
        // Partial overlap
        let v1 = SIMD4<Float>(1, 1, 0, 0)
        let v2 = SIMD4<Float>(1, 0, 1, 0)
        let overlap = metric.distance(v1, v2)
        // Intersection = 1, Union = 3, Distance = 1 - 1/3 = 2/3
        XCTAssertEqual(overlap, 2.0/3.0, accuracy: 0.001)
    }
    
    // MARK: - Performance Optimization Tests
    
    func testOptimizedDistanceComputation() {
        // Test that optimized versions produce same results
        let euclidean = EuclideanDistance()
        let largeA = Vector512(repeating: 0.5)
        let largeB = Vector512(repeating: 0.7)
        
        let dist = euclidean.distance(largeA, largeB)
        let expected = sqrt(512 * 0.04) // sqrt(512 * (0.7-0.5)^2)
        XCTAssertEqual(Double(dist), expected, accuracy: 0.01)
    }
    
    func testDistanceComputationUtility() {
        let a = SIMD4<Float>(1, 2, 3, 4)
        let b = SIMD4<Float>(5, 6, 7, 8)
        
        // Test squared euclidean (avoids sqrt)
        let squared = DistanceCalculator.euclideanSquared(a, b)
        XCTAssertEqual(squared, 64.0, accuracy: 0.001) // 4^2 * 4
        
        // Test normalized cosine (assumes pre-normalized)
        let norm1 = SIMD4<Float>(1, 0, 0, 0)
        let norm2 = SIMD4<Float>(0.6, 0.8, 0, 0)
        let cosDist = DistanceCalculator.normalizedCosine(norm1, norm2)
        XCTAssertEqual(cosDist, 0.4, accuracy: 0.001) // 1 - 0.6
    }
    
    // MARK: - Edge Cases
    
    func testDistanceMetricsWithSpecialValues() {
        let normal = SIMD4<Float>(1, 2, 3, 4)
        let withNaN = SIMD4<Float>(1, 2, Float.nan, 4)
        let withInf = SIMD4<Float>(1, 2, Float.infinity, 4)
        
        // Euclidean with NaN
        let euclidean = EuclideanDistance()
        let nanDist = euclidean.distance(normal, withNaN)
        XCTAssertTrue(nanDist.isNaN)
        
        // Euclidean with Infinity
        let infDist = euclidean.distance(normal, withInf)
        XCTAssertTrue(infDist.isInfinite)
        
        // Cosine with zero vector
        let cosine = CosineDistance()
        let zeroVec = SIMD4<Float>(repeating: 0)
        let zeroDist = cosine.distance(normal, zeroVec)
        XCTAssertEqual(zeroDist, 1.0) // Maximum distance
    }
    
    // MARK: - Performance Tests
    
    func testEuclideanPerformance() {
        let metric = EuclideanDistance()
        let a = Vector512(repeating: 0.5)
        let b = Vector512(repeating: 0.7)
        
        measure {
            for _ in 0..<10000 {
                _ = metric.distance(a, b)
            }
        }
    }
    
    func testCosinePerformance() {
        let metric = CosineDistance()
        let a = Vector768(repeating: 0.3)
        let b = Vector768(repeating: 0.8)
        
        measure {
            for _ in 0..<10000 {
                _ = metric.distance(a, b)
            }
        }
    }
    
    func testBatchDistancePerformance() {
        let metric = EuclideanDistance()
        let query = Vector256(repeating: 0.5)
        let candidates = (0..<100).map { _ in Vector256(repeating: Float.random(in: 0...1)) }
        
        measure {
            _ = metric.batchDistance(query: query, candidates: candidates)
        }
    }
}