// VectorCore: Generic Vector Tests
//
// Tests for the new generic vector implementation
//

import XCTest
@testable import VectorCore

final class GenericVectorTests: XCTestCase {
    
    // MARK: - Basic Tests
    
    func testGenericVectorInitialization() {
        // Test Vector128
        let v128 = Vector128()
        XCTAssertEqual(v128.scalarCount, 128)
        
        // Test Vector256
        let v256 = Vector256(repeating: 1.0)
        XCTAssertEqual(v256.scalarCount, 256)
        for i in 0..<256 {
            XCTAssertEqual(v256[i], 1.0)
        }
        
        // Test Vector512
        let values = (0..<512).map { Float($0) }
        let v512 = Vector512(values)
        XCTAssertEqual(v512.scalarCount, 512)
        for i in 0..<512 {
            XCTAssertEqual(v512[i], Float(i))
        }
    }
    
    func testDotProduct() {
        let a = Vector128(repeating: 1.0)
        let b = Vector128(repeating: 2.0)
        
        let dot = a.dotProduct(b)
        XCTAssertEqual(dot, 256.0, accuracy: 0.001) // 128 * 1.0 * 2.0
    }
    
    func testVectorArithmetic() {
        let a = Vector256(repeating: 10.0)
        let b = Vector256(repeating: 5.0)
        
        // Addition
        let sum = a + b
        XCTAssertEqual(sum[0], 15.0)
        XCTAssertEqual(sum[255], 15.0)
        
        // Subtraction
        let diff = a - b
        XCTAssertEqual(diff[0], 5.0)
        XCTAssertEqual(diff[255], 5.0)
        
        // Scalar multiplication
        let scaled = a * 0.5
        XCTAssertEqual(scaled[0], 5.0)
        XCTAssertEqual(scaled[255], 5.0)
        
        // Scalar division
        let divided = a / 2.0
        XCTAssertEqual(divided[0], 5.0)
        XCTAssertEqual(divided[255], 5.0)
    }
    
    func testVectorNormalization() {
        var values = [Float](repeating: 0, count: 128)
        values[0] = 3.0
        values[1] = 4.0
        
        let v = Vector128(values)
        let normalized = v.normalized()
        
        XCTAssertEqual(normalized.magnitude, 1.0, accuracy: 0.001)
        XCTAssertEqual(normalized[0], 0.6, accuracy: 0.001)
        XCTAssertEqual(normalized[1], 0.8, accuracy: 0.001)
    }
    
    func testDynamicVector() {
        // Test with non-standard dimension
        let v = DynamicVector(dimension: 100, repeating: 1.0)
        XCTAssertEqual(v.dimension, 100)
        XCTAssertEqual(v.scalarCount, 100)
        
        // Test dot product
        let v2 = DynamicVector(dimension: 100, repeating: 2.0)
        let dot = v.dotProduct(v2)
        XCTAssertEqual(dot, 200.0, accuracy: 0.001) // 100 * 1.0 * 2.0
        
        // Test normalization
        var values = [Float](repeating: 0, count: 100)
        values[0] = 3.0
        values[1] = 4.0
        
        let v3 = DynamicVector(dimension: 100, values: values)
        let normalized = v3.normalized()
        XCTAssertEqual(normalized.magnitude, 1.0, accuracy: 0.001)
    }
    
    func testVectorFactory() throws {
        // Test factory with standard dimensions
        let v128 = try VectorFactory.vector(of: 128, from: Array(repeating: 1.0, count: 128))
        XCTAssertEqual(v128.scalarCount, 128)
        
        // Test factory with non-standard dimension
        let v100 = try VectorFactory.vector(of: 100, from: Array(repeating: 1.0, count: 100))
        XCTAssertEqual(v100.scalarCount, 100)
        
        // Test optimal dimension selection
        XCTAssertEqual(VectorFactory.optimalDimension(for: 120), 128)
        XCTAssertEqual(VectorFactory.optimalDimension(for: 200), 256)
        XCTAssertEqual(VectorFactory.optimalDimension(for: 1000), 768)
    }
    
    func testDistanceMetrics() {
        let a = Vector128(repeating: 1.0)
        let b = Vector128(repeating: 2.0)
        
        // Euclidean distance
        let euclidean = EuclideanDistance()
        let eucDist = euclidean.distance(a, b)
        XCTAssertEqual(eucDist, sqrt(128.0), accuracy: 0.001) // sqrt(128 * 1^2)
        
        // Cosine distance
        let cosine = CosineDistance()
        let cosDist = cosine.distance(a, b)
        XCTAssertEqual(cosDist, 0.0, accuracy: 0.001) // Same direction
    }
    
    func testCodable() throws {
        let original = Vector128(repeating: 3.14)
        
        // Encode
        let encoder = JSONEncoder()
        let data = try encoder.encode(original)
        
        // Decode
        let decoder = JSONDecoder()
        let decoded = try decoder.decode(Vector128.self, from: data)
        
        // Verify
        for i in 0..<128 {
            XCTAssertEqual(decoded[i], original[i])
        }
    }
}