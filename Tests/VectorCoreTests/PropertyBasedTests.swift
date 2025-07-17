import XCTest
@testable import VectorCore

/// Property-based tests for vector operations using concrete types
///
/// These tests verify mathematical properties that should hold for all vector implementations.
/// Each property is tested with random inputs across multiple iterations to increase confidence.
final class PropertyBasedTests: XCTestCase {
    
    // MARK: - Test Configuration
    
    let iterations = 100  // Number of random test cases per property
    let accuracy: Float = 1e-5  // Floating point comparison tolerance
    
    // MARK: - Generic Property Test Functions
    
    /// Test dot product commutativity: a·b = b·a
    private func testDotProductCommutativityProperty<V: ExtendedVectorProtocol>(
        vectorType: V.Type,
        dimension: Int,
        iterations: Int
    ) where V: Equatable {
        for _ in 0..<iterations {
            let v1 = createRandomVector(type: vectorType, dimension: dimension)
            let v2 = createRandomVector(type: vectorType, dimension: dimension)
            
            let dot1 = v1.dotProduct(v2)
            let dot2 = v2.dotProduct(v1)
            
            XCTAssertEqual(dot1, dot2, accuracy: accuracy,
                          "Dot product should be commutative for \(vectorType)")
        }
    }
    
    /// Test distance symmetry: d(a,b) = d(b,a)
    private func testDistanceSymmetryProperty<V: ExtendedVectorProtocol>(
        vectorType: V.Type,
        dimension: Int,
        iterations: Int
    ) where V: Equatable {
        for _ in 0..<iterations {
            let v1 = createRandomVector(type: vectorType, dimension: dimension)
            let v2 = createRandomVector(type: vectorType, dimension: dimension)
            
            let d1 = v1.distance(to: v2)
            let d2 = v2.distance(to: v1)
            
            XCTAssertEqual(d1, d2, accuracy: accuracy,
                          "Distance should be symmetric for \(vectorType)")
        }
    }
    
    /// Test triangle inequality: d(a,c) ≤ d(a,b) + d(b,c)
    private func testTriangleInequalityProperty<V: ExtendedVectorProtocol>(
        vectorType: V.Type,
        dimension: Int,
        iterations: Int
    ) where V: Equatable {
        for _ in 0..<iterations {
            let a = createRandomVector(type: vectorType, dimension: dimension)
            let b = createRandomVector(type: vectorType, dimension: dimension)
            let c = createRandomVector(type: vectorType, dimension: dimension)
            
            let dac = a.distance(to: c)
            let dab = a.distance(to: b)
            let dbc = b.distance(to: c)
            
            XCTAssertLessThanOrEqual(dac, dab + dbc + accuracy,
                                    "Triangle inequality should hold for \(vectorType)")
        }
    }
    
    /// Test normalized vector magnitude: ||normalize(v)|| = 1 (for non-zero vectors)
    private func testNormalizedMagnitudeProperty<V: ExtendedVectorProtocol>(
        vectorType: V.Type,
        dimension: Int,
        iterations: Int
    ) where V: Equatable {
        for _ in 0..<iterations {
            let vector = createRandomVector(type: vectorType, dimension: dimension)
            
            if vector.magnitude > 1e-6 {
                let normalized = vector.normalized()
                XCTAssertEqual(normalized.magnitude, 1.0, accuracy: accuracy,
                              "Normalized vector magnitude should be 1 for \(vectorType)")
            }
        }
    }
    
    /// Test cosine similarity bounds: -1 ≤ cos(θ) ≤ 1
    private func testCosineSimilarityBoundsProperty<V: ExtendedVectorProtocol>(
        vectorType: V.Type,
        dimension: Int,
        iterations: Int
    ) where V: Equatable {
        for _ in 0..<iterations {
            let v1 = createRandomVector(type: vectorType, dimension: dimension)
            let v2 = createRandomVector(type: vectorType, dimension: dimension)
            
            if v1.magnitude > 1e-6 && v2.magnitude > 1e-6 {
                let similarity = v1.cosineSimilarity(to: v2)
                XCTAssertGreaterThanOrEqual(similarity, -1.0 - accuracy,
                                          "Cosine similarity should be >= -1 for \(vectorType)")
                XCTAssertLessThanOrEqual(similarity, 1.0 + accuracy,
                                        "Cosine similarity should be <= 1 for \(vectorType)")
            }
        }
    }
    
    /// Test self cosine similarity: cos(v, v) = 1
    private func testSelfCosineSimilarityProperty<V: ExtendedVectorProtocol>(
        vectorType: V.Type,
        dimension: Int,
        iterations: Int
    ) where V: Equatable {
        for _ in 0..<iterations {
            let v = createRandomVector(type: vectorType, dimension: dimension)
            
            if v.magnitude > 1e-6 {
                let similarity = v.cosineSimilarity(to: v)
                XCTAssertEqual(similarity, 1.0, accuracy: accuracy,
                              "Self cosine similarity should be 1 for \(vectorType)")
            }
        }
    }
    
    /// Test distance non-negativity: d(a,b) ≥ 0 and d(a,a) = 0
    private func testDistanceNonNegativityProperty<V: ExtendedVectorProtocol>(
        vectorType: V.Type,
        dimension: Int,
        iterations: Int
    ) where V: Equatable {
        for _ in 0..<iterations {
            let v1 = createRandomVector(type: vectorType, dimension: dimension)
            let v2 = createRandomVector(type: vectorType, dimension: dimension)
            
            let distance = v1.distance(to: v2)
            XCTAssertGreaterThanOrEqual(distance, 0,
                                      "Distance should be non-negative for \(vectorType)")
            
            let selfDistance = v1.distance(to: v1)
            XCTAssertEqual(selfDistance, 0, accuracy: accuracy,
                          "Self-distance should be zero for \(vectorType)")
        }
    }
    
    // MARK: - Concrete Type Test Methods
    
    func testVector32Properties() {
        let type = Vector<Dim32>.self
        let dim = 32
        
        testDotProductCommutativityProperty(vectorType: type, dimension: dim, iterations: iterations)
        testDistanceSymmetryProperty(vectorType: type, dimension: dim, iterations: iterations)
        testTriangleInequalityProperty(vectorType: type, dimension: dim, iterations: iterations)
        testNormalizedMagnitudeProperty(vectorType: type, dimension: dim, iterations: iterations)
        testCosineSimilarityBoundsProperty(vectorType: type, dimension: dim, iterations: iterations)
        testSelfCosineSimilarityProperty(vectorType: type, dimension: dim, iterations: iterations)
        testDistanceNonNegativityProperty(vectorType: type, dimension: dim, iterations: iterations)
    }
    
    func testVector64Properties() {
        let type = Vector<Dim64>.self
        let dim = 64
        
        testDotProductCommutativityProperty(vectorType: type, dimension: dim, iterations: iterations)
        testDistanceSymmetryProperty(vectorType: type, dimension: dim, iterations: iterations)
        testTriangleInequalityProperty(vectorType: type, dimension: dim, iterations: iterations)
        testNormalizedMagnitudeProperty(vectorType: type, dimension: dim, iterations: iterations)
        testCosineSimilarityBoundsProperty(vectorType: type, dimension: dim, iterations: iterations)
        testSelfCosineSimilarityProperty(vectorType: type, dimension: dim, iterations: iterations)
        testDistanceNonNegativityProperty(vectorType: type, dimension: dim, iterations: iterations)
    }
    
    func testVector128Properties() {
        let type = Vector<Dim128>.self
        let dim = 128
        
        testDotProductCommutativityProperty(vectorType: type, dimension: dim, iterations: iterations)
        testDistanceSymmetryProperty(vectorType: type, dimension: dim, iterations: iterations)
        testTriangleInequalityProperty(vectorType: type, dimension: dim, iterations: iterations)
        testNormalizedMagnitudeProperty(vectorType: type, dimension: dim, iterations: iterations)
        testCosineSimilarityBoundsProperty(vectorType: type, dimension: dim, iterations: iterations)
        testSelfCosineSimilarityProperty(vectorType: type, dimension: dim, iterations: iterations)
        testDistanceNonNegativityProperty(vectorType: type, dimension: dim, iterations: iterations)
    }
    
    func testVector256Properties() {
        let type = Vector<Dim256>.self
        let dim = 256
        
        testDotProductCommutativityProperty(vectorType: type, dimension: dim, iterations: iterations)
        testDistanceSymmetryProperty(vectorType: type, dimension: dim, iterations: iterations)
        testTriangleInequalityProperty(vectorType: type, dimension: dim, iterations: iterations)
        testNormalizedMagnitudeProperty(vectorType: type, dimension: dim, iterations: iterations)
        testCosineSimilarityBoundsProperty(vectorType: type, dimension: dim, iterations: iterations)
        testSelfCosineSimilarityProperty(vectorType: type, dimension: dim, iterations: iterations)
        testDistanceNonNegativityProperty(vectorType: type, dimension: dim, iterations: iterations)
    }
    
    func testVector512Properties() {
        let type = Vector<Dim512>.self
        let dim = 512
        
        testDotProductCommutativityProperty(vectorType: type, dimension: dim, iterations: iterations)
        testDistanceSymmetryProperty(vectorType: type, dimension: dim, iterations: iterations)
        testTriangleInequalityProperty(vectorType: type, dimension: dim, iterations: iterations)
        testNormalizedMagnitudeProperty(vectorType: type, dimension: dim, iterations: iterations)
        testCosineSimilarityBoundsProperty(vectorType: type, dimension: dim, iterations: iterations)
        testSelfCosineSimilarityProperty(vectorType: type, dimension: dim, iterations: iterations)
        testDistanceNonNegativityProperty(vectorType: type, dimension: dim, iterations: iterations)
    }
    
    func testDynamicVectorProperties() {
        let dimensions = [32, 64, 128, 256, 512, 768, 1536]
        
        for dim in dimensions {
            // Test with dimension-specific functions
            testDynamicVectorDotProductCommutativity(dimension: dim)
            testDynamicVectorDistanceSymmetry(dimension: dim)
            testDynamicVectorTriangleInequality(dimension: dim)
            testDynamicVectorNormalizedMagnitude(dimension: dim)
            testDynamicVectorCosineSimilarityBounds(dimension: dim)
            testDynamicVectorSelfCosineSimilarity(dimension: dim)
            testDynamicVectorDistanceNonNegativity(dimension: dim)
        }
    }
    
    // MARK: - DynamicVector Specific Tests
    
    private func testDynamicVectorDotProductCommutativity(dimension: Int) {
        for _ in 0..<iterations {
            let v1 = DynamicVector.random(dimension: dimension, in: -1...1)
            let v2 = DynamicVector.random(dimension: dimension, in: -1...1)
            
            let dot1 = v1.dotProduct(v2)
            let dot2 = v2.dotProduct(v1)
            
            XCTAssertEqual(dot1, dot2, accuracy: accuracy,
                          "Dot product should be commutative for DynamicVector[\(dimension)]")
        }
    }
    
    private func testDynamicVectorDistanceSymmetry(dimension: Int) {
        for _ in 0..<iterations {
            let v1 = DynamicVector.random(dimension: dimension, in: -1...1)
            let v2 = DynamicVector.random(dimension: dimension, in: -1...1)
            
            let d1 = v1.distance(to: v2)
            let d2 = v2.distance(to: v1)
            
            XCTAssertEqual(d1, d2, accuracy: accuracy,
                          "Distance should be symmetric for DynamicVector[\(dimension)]")
        }
    }
    
    private func testDynamicVectorTriangleInequality(dimension: Int) {
        for _ in 0..<iterations {
            let a = DynamicVector.random(dimension: dimension, in: -1...1)
            let b = DynamicVector.random(dimension: dimension, in: -1...1)
            let c = DynamicVector.random(dimension: dimension, in: -1...1)
            
            let dac = a.distance(to: c)
            let dab = a.distance(to: b)
            let dbc = b.distance(to: c)
            
            XCTAssertLessThanOrEqual(dac, dab + dbc + accuracy,
                                    "Triangle inequality should hold for DynamicVector[\(dimension)]")
        }
    }
    
    private func testDynamicVectorNormalizedMagnitude(dimension: Int) {
        for _ in 0..<iterations {
            let vector = DynamicVector.random(dimension: dimension, in: -1...1)
            
            if vector.magnitude > 1e-6 {
                let normalized = vector.normalized()
                XCTAssertEqual(normalized.magnitude, 1.0, accuracy: accuracy,
                              "Normalized vector magnitude should be 1 for DynamicVector[\(dimension)]")
            }
        }
    }
    
    private func testDynamicVectorCosineSimilarityBounds(dimension: Int) {
        for _ in 0..<iterations {
            let v1 = DynamicVector.random(dimension: dimension, in: -1...1)
            let v2 = DynamicVector.random(dimension: dimension, in: -1...1)
            
            if v1.magnitude > 1e-6 && v2.magnitude > 1e-6 {
                let similarity = v1.cosineSimilarity(to: v2)
                XCTAssertGreaterThanOrEqual(similarity, -1.0 - accuracy,
                                          "Cosine similarity should be >= -1 for DynamicVector[\(dimension)]")
                XCTAssertLessThanOrEqual(similarity, 1.0 + accuracy,
                                        "Cosine similarity should be <= 1 for DynamicVector[\(dimension)]")
            }
        }
    }
    
    private func testDynamicVectorSelfCosineSimilarity(dimension: Int) {
        for _ in 0..<iterations {
            let v = DynamicVector.random(dimension: dimension, in: -1...1)
            
            if v.magnitude > 1e-6 {
                let similarity = v.cosineSimilarity(to: v)
                XCTAssertEqual(similarity, 1.0, accuracy: accuracy,
                              "Self cosine similarity should be 1 for DynamicVector[\(dimension)]")
            }
        }
    }
    
    private func testDynamicVectorDistanceNonNegativity(dimension: Int) {
        for _ in 0..<iterations {
            let v1 = DynamicVector.random(dimension: dimension, in: -1...1)
            let v2 = DynamicVector.random(dimension: dimension, in: -1...1)
            
            let distance = v1.distance(to: v2)
            XCTAssertGreaterThanOrEqual(distance, 0,
                                      "Distance should be non-negative for DynamicVector[\(dimension)]")
            
            let selfDistance = v1.distance(to: v1)
            XCTAssertEqual(selfDistance, 0, accuracy: accuracy,
                          "Self-distance should be zero for DynamicVector[\(dimension)]")
        }
    }
    
    // MARK: - Numerical Stability Tests
    
    func testNumericalStabilityWithExtremeValues() {
        // Test with very small values
        testNumericalStability(range: 1e-10...1e-8, description: "small")
        
        // Test with very large values
        testNumericalStability(range: 1e8...1e10, description: "large")
        
        // Test with mixed magnitudes
        testMixedMagnitudeStability()
    }
    
    private func testNumericalStability(range: ClosedRange<Float>, description: String) {
        let dimensions = [32, 128, 512]
        
        for dim in dimensions {
            // Test Vector types
            if dim == 32 {
                let v = createRandomVector(type: Vector<Dim32>.self, dimension: dim, range: range)
                validateNumericalStability(v, description: "\(description) Vector<Dim32>")
            } else if dim == 128 {
                let v = createRandomVector(type: Vector<Dim128>.self, dimension: dim, range: range)
                validateNumericalStability(v, description: "\(description) Vector<Dim128>")
            } else if dim == 512 {
                let v = createRandomVector(type: Vector<Dim512>.self, dimension: dim, range: range)
                validateNumericalStability(v, description: "\(description) Vector<Dim512>")
            }
            
            // Test DynamicVector
            let dv = createRandomDynamicVector(dimension: dim, range: range)
            validateNumericalStability(dv, description: "\(description) DynamicVector[\(dim)]")
        }
    }
    
    private func testMixedMagnitudeStability() {
        let dim = 128
        
        // Create vectors with mixed magnitudes
        var values = [Float]()
        for i in 0..<dim {
            if i % 2 == 0 {
                values.append(Float.random(in: 1e-10...1e-8))
            } else {
                values.append(Float.random(in: 1e8...1e10))
            }
        }
        
        let v1 = Vector<Dim128>(values)
        let v2 = DynamicVector(values)
        
        validateNumericalStability(v1, description: "mixed magnitude Vector<Dim128>")
        validateNumericalStability(v2, description: "mixed magnitude DynamicVector")
    }
    
    private func validateNumericalStability<V: ExtendedVectorProtocol>(_ vector: V, description: String) {
        let magnitude = vector.magnitude
        XCTAssertFalse(magnitude.isNaN, "\(description) magnitude produced NaN")
        XCTAssertFalse(magnitude.isInfinite, "\(description) magnitude produced Infinity")
        
        if magnitude > 1e-6 {
            let normalized = vector.normalized()
            let normMag = normalized.magnitude
            XCTAssertFalse(normMag.isNaN, "\(description) normalized magnitude produced NaN")
            XCTAssertFalse(normMag.isInfinite, "\(description) normalized magnitude produced Infinity")
        }
    }
    
    // MARK: - Helper Methods
    
    private func createRandomVector<V: ExtendedVectorProtocol>(
        type: V.Type,
        dimension: Int,
        range: ClosedRange<Float> = -1...1
    ) -> V where V: Equatable {
        let values = (0..<dimension).map { _ in Float.random(in: range) }
        return V(from: values)
    }
    
    private func createRandomDynamicVector(
        dimension: Int,
        range: ClosedRange<Float> = -1...1
    ) -> DynamicVector {
        return DynamicVector.random(dimension: dimension, in: range)
    }
}