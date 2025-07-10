import XCTest
@testable import VectorCore

final class PropertyBasedTests: XCTestCase {
    
    // MARK: - Test Configuration
    
    let iterations = 100  // Number of random test cases per property
    let dimensions = [32, 64, 128, 256, 512, 768, 1536]
    
    // MARK: - Vector Magnitude Properties
    
    func testNormalizedVectorMagnitudeProperty() {
        for dim in dimensions {
            for _ in 0..<iterations {
                let vector = createRandomVector(dimension: dim)
                let normalized = vector.normalized()
                
                // Property: ||normalize(v)|| = 1 (for non-zero vectors)
                if vector.magnitude > 1e-6 {
                    XCTAssertEqual(normalized.magnitude, 1.0, accuracy: 1e-5,
                                 "Normalized vector magnitude should be 1")
                }
            }
        }
    }
    
    func testZeroVectorNormalizationProperty() {
        for dim in dimensions {
            let zero = createZeroVector(dimension: dim)
            let normalized = zero.normalized()
            
            // Property: normalize(0) = 0
            XCTAssertEqual(normalized.magnitude, 0, accuracy: 1e-7,
                         "Zero vector normalization should remain zero")
        }
    }
    
    // MARK: - Distance Metric Properties
    
    func testDistanceSymmetryProperty() {
        for dim in dimensions {
            for _ in 0..<iterations {
                let v1 = createRandomVector(dimension: dim)
                let v2 = createRandomVector(dimension: dim)
                
                // Property: d(a,b) = d(b,a)
                let d1 = v1.distance(to: v2)
                let d2 = v2.distance(to: v1)
                
                XCTAssertEqual(d1, d2, accuracy: 1e-5,
                             "Distance should be symmetric")
            }
        }
    }
    
    func testDistanceTriangleInequalityProperty() {
        for dim in dimensions {
            for _ in 0..<iterations {
                let a = createRandomVector(dimension: dim)
                let b = createRandomVector(dimension: dim)
                let c = createRandomVector(dimension: dim)
                
                // Property: d(a,c) ≤ d(a,b) + d(b,c)
                let dac = a.distance(to: c)
                let dab = a.distance(to: b)
                let dbc = b.distance(to: c)
                
                XCTAssertLessThanOrEqual(dac, dab + dbc + 1e-5,
                                       "Triangle inequality should hold")
            }
        }
    }
    
    func testDistanceNonNegativityProperty() {
        for dim in dimensions {
            for _ in 0..<iterations {
                let v1 = createRandomVector(dimension: dim)
                let v2 = createRandomVector(dimension: dim)
                
                // Property: d(a,b) ≥ 0
                let distance = v1.distance(to: v2)
                XCTAssertGreaterThanOrEqual(distance, 0,
                                          "Distance should be non-negative")
                
                // Property: d(a,a) = 0
                let selfDistance = v1.distance(to: v1)
                XCTAssertEqual(selfDistance, 0, accuracy: 1e-7,
                             "Self-distance should be zero")
            }
        }
    }
    
    // MARK: - Dot Product Properties
    
    func testDotProductCommutativityProperty() {
        for dim in dimensions {
            for _ in 0..<iterations {
                let v1 = createRandomVector(dimension: dim)
                let v2 = createRandomVector(dimension: dim)
                
                // Property: a·b = b·a
                let dot1 = v1.dotProduct(v2)
                let dot2 = v2.dotProduct(v1)
                
                XCTAssertEqual(dot1, dot2, accuracy: 1e-5,
                             "Dot product should be commutative")
            }
        }
    }
    
    func testDotProductDistributivityProperty() {
        for dim in dimensions {
            for _ in 0..<iterations {
                let a = createRandomVector(dimension: dim)
                let b = createRandomVector(dimension: dim)
                let c = createRandomVector(dimension: dim)
                
                // Property: a·(b+c) = a·b + a·c
                let bc = b + c
                let left = a.dotProduct(bc)
                let right = a.dotProduct(b) + a.dotProduct(c)
                
                XCTAssertEqual(left, right, accuracy: 1e-4,
                             "Dot product should be distributive")
            }
        }
    }
    
    func testDotProductScalarMultiplicationProperty() {
        for dim in dimensions {
            for _ in 0..<iterations {
                let v1 = createRandomVector(dimension: dim)
                let v2 = createRandomVector(dimension: dim)
                let scalar = Float.random(in: -10...10)
                
                // Property: (k*a)·b = k*(a·b)
                let scaledV1 = v1 * scalar
                let left = scaledV1.dotProduct(v2)
                let right = scalar * v1.dotProduct(v2)
                
                XCTAssertEqual(left, right, accuracy: 1e-4,
                             "Scalar multiplication should factor out of dot product")
            }
        }
    }
    
    // MARK: - Vector Addition Properties
    
    func testVectorAdditionCommutativityProperty() {
        for dim in dimensions {
            for _ in 0..<iterations {
                let v1 = createRandomVector(dimension: dim)
                let v2 = createRandomVector(dimension: dim)
                
                // Property: a + b = b + a
                let sum1 = v1 + v2
                let sum2 = v2 + v1
                
                assertVectorsEqual(sum1, sum2, accuracy: 1e-5)
            }
        }
    }
    
    func testVectorAdditionAssociativityProperty() {
        for dim in dimensions {
            for _ in 0..<iterations {
                let a = createRandomVector(dimension: dim)
                let b = createRandomVector(dimension: dim)
                let c = createRandomVector(dimension: dim)
                
                // Property: (a + b) + c = a + (b + c)
                let left = (a + b) + c
                let right = a + (b + c)
                
                assertVectorsEqual(left, right, accuracy: 1e-5)
            }
        }
    }
    
    func testVectorAdditionIdentityProperty() {
        for dim in dimensions {
            let v = createRandomVector(dimension: dim)
            let zero = createZeroVector(dimension: dim)
            
            // Property: v + 0 = v
            let sum = v + zero
            assertVectorsEqual(sum, v, accuracy: 1e-6)
        }
    }
    
    // MARK: - Cosine Similarity Properties
    
    func testCosineSimilarityBoundsProperty() {
        for dim in dimensions {
            for _ in 0..<iterations {
                let v1 = createRandomVector(dimension: dim)
                let v2 = createRandomVector(dimension: dim)
                
                let similarity = v1.cosineSimilarity(to: v2)
                
                // Property: -1 ≤ cos(θ) ≤ 1
                XCTAssertGreaterThanOrEqual(similarity, -1.0 - 1e-5)
                XCTAssertLessThanOrEqual(similarity, 1.0 + 1e-5)
            }
        }
    }
    
    func testCosineSimilaritySelfProperty() {
        for dim in dimensions {
            for _ in 0..<iterations {
                let v = createRandomVector(dimension: dim)
                
                if v.magnitude > 1e-6 {
                    // Property: cos(v, v) = 1
                    let similarity = v.cosineSimilarity(to: v)
                    XCTAssertEqual(similarity, 1.0, accuracy: 1e-5)
                }
            }
        }
    }
    
    // MARK: - Numerical Stability Properties
    
    func testNumericalStabilityProperty() {
        for dim in dimensions {
            // Test with extreme values
            let small = createRandomVector(dimension: dim, range: 1e-10...1e-8)
            let large = createRandomVector(dimension: dim, range: 1e8...1e10)
            
            // Operations should not produce NaN or Inf
            let operations: [(any VectorType) -> Float] = [
                { $0.magnitude },
                { $0.normalized().magnitude },
                { $0.dotProduct($0) }
            ]
            
            for op in operations {
                let resultSmall = op(small)
                let resultLarge = op(large)
                
                XCTAssertFalse(resultSmall.isNaN, "Operation produced NaN")
                XCTAssertFalse(resultSmall.isInfinite, "Operation produced Infinity")
                XCTAssertFalse(resultLarge.isNaN, "Operation produced NaN")
                XCTAssertFalse(resultLarge.isInfinite, "Operation produced Infinity")
            }
        }
    }
    
    // MARK: - Helper Methods
    
    private func createRandomVector(dimension: Int, range: ClosedRange<Float> = -1...1) -> any VectorType {
        VectorFactory.random(dimension: dimension, range: range)
    }
    
    private func createZeroVector(dimension: Int) -> any VectorType {
        VectorFactory.zeros(dimension: dimension)
    }
    
    private func assertVectorsEqual(_ v1: any VectorType, _ v2: any VectorType, accuracy: Float) {
        XCTAssertEqual(v1.scalarCount, v2.scalarCount)
        let a1 = v1.toArray()
        let a2 = v2.toArray()
        for i in 0..<a1.count {
            XCTAssertEqual(a1[i], a2[i], accuracy: accuracy)
        }
    }
}