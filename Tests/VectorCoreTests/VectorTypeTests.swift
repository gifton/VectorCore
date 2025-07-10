// VectorCoreTests: Vector Type Tests
//
// Comprehensive tests for vector types
//

import XCTest
@testable import VectorCore

final class VectorTypeTests: XCTestCase {
    
    // MARK: - Vector128 Tests
    
    func testVector128Initialization() {
        // Test zero initialization
        let zeroVector = Vector128()
        for i in 0..<128 {
            XCTAssertEqual(zeroVector[i], 0)
        }
        
        // Test repeating value initialization
        let constVector = Vector128(repeating: 3.14)
        for i in 0..<128 {
            XCTAssertEqual(constVector[i], 3.14)
        }
        
        // Test array initialization
        let values = (0..<128).map { Float($0) }
        let arrayVector = Vector128(values)
        for i in 0..<128 {
            XCTAssertEqual(arrayVector[i], Float(i))
        }
    }
    
    func testVector128DotProduct() {
        let a = Vector128(repeating: 1.0)
        let b = Vector128(repeating: 2.0)
        
        let dot = a.dotProduct(b)
        XCTAssertEqual(dot, 256.0, accuracy: 0.001) // 128 * 1.0 * 2.0
    }
    
    func testVector128Arithmetic() {
        let values1 = (0..<128).map { Float($0) }
        let values2 = (0..<128).map { Float($0 * 2) }
        
        let a = Vector128(values1)
        let b = Vector128(values2)
        
        // Test element access
        XCTAssertEqual(a[0], 0)
        XCTAssertEqual(a[127], 127)
        XCTAssertEqual(b[0], 0)
        XCTAssertEqual(b[127], 254)
    }
    
    // MARK: - Vector256 Tests
    
    func testVector256Initialization() {
        // Test zero initialization
        let zeroVector = Vector256()
        for i in 0..<256 {
            XCTAssertEqual(zeroVector[i], 0)
        }
        
        // Test array initialization with edge cases
        var values = [Float](repeating: 0, count: 256)
        values[0] = -Float.infinity
        values[255] = Float.infinity
        values[128] = Float.nan
        
        let vector = Vector256(values)
        XCTAssertEqual(vector[0], -Float.infinity)
        XCTAssertEqual(vector[255], Float.infinity)
        XCTAssertTrue(vector[128].isNaN)
    }
    
    func testVector256DotProduct() {
        // Test orthogonal vectors
        var a_values = [Float](repeating: 0, count: 256)
        var b_values = [Float](repeating: 0, count: 256)
        a_values[0] = 1.0
        b_values[1] = 1.0
        
        let a = Vector256(a_values)
        let b = Vector256(b_values)
        
        let dot = a.dotProduct(b)
        XCTAssertEqual(dot, 0.0, accuracy: 0.001)
    }
    
    // MARK: - Vector512 Tests
    
    func testVector512Initialization() {
        // Test initializer with unsafe buffer
        let vector = Vector512(unsafeUninitializedCapacity: 512) { buffer in
            for i in 0..<512 {
                buffer[i] = Float(i) * 0.5
            }
        }
        
        for i in 0..<512 {
            XCTAssertEqual(vector[i], Float(i) * 0.5, accuracy: 0.001)
        }
    }
    
    func testVector512Normalization() {
        let values = (0..<512).map { Float($0 + 1) } // Avoid zero
        var vector = Vector512(values)
        
        let magnitudeBefore = vector.magnitude
        XCTAssertGreaterThan(magnitudeBefore, 0)
        
        vector.normalize()
        let magnitudeAfter = vector.magnitude
        XCTAssertEqual(magnitudeAfter, 1.0, accuracy: 0.001)
    }
    
    func testVector512Distance() {
        let a = Vector512(repeating: 0)
        let b = Vector512(repeating: 3)
        
        let distance = a.distance(to: b)
        let expected = sqrt(512 * 9) // sqrt(512 * 3^2)
        XCTAssertEqual(Double(distance), expected, accuracy: 0.001)
    }
    
    func testVector512BatchCreation() {
        let flatArray = (0..<1024).map { Float($0) }
        let vectors = Vector512.createBatch(from: flatArray)
        
        XCTAssertEqual(vectors.count, 2)
        
        // Check first vector
        for i in 0..<512 {
            XCTAssertEqual(vectors[0][i], Float(i))
        }
        
        // Check second vector
        for i in 0..<512 {
            XCTAssertEqual(vectors[1][i], Float(i + 512))
        }
    }
    
    // MARK: - Vector768 Tests
    
    func testVector768TextEmbeddingSize() {
        // Vector768 is designed for BERT embeddings
        let vector = Vector768(repeating: 0.1)
        XCTAssertEqual(vector.scalarCount, 768)
        
        // Test typical embedding values
        let embedding = (0..<768).map { sin(Float($0) * 0.01) }
        let embVector = Vector768(embedding)
        
        // Embeddings are typically normalized
        let magnitude = embVector.cosineSimilarity(to: embVector)
        XCTAssertLessThanOrEqual(magnitude, 1.0)
    }
    
    // MARK: - Vector1536 Tests
    
    func testVector1536LargeModelSize() {
        // Vector1536 is designed for large model embeddings
        let vector = Vector1536(repeating: 0.01)
        XCTAssertEqual(vector.scalarCount, 1536)
        
        // Test memory efficiency of dot product
        let a = Vector1536(repeating: 0.5)
        let b = Vector1536(repeating: 0.5)
        
        let dot = a.dotProduct(b)
        XCTAssertEqual(dot, 384.0, accuracy: 0.1) // 1536 * 0.5 * 0.5
    }
    
    // MARK: - VectorProtocol Conformance Tests
    
    func testVectorProtocolConformance() {
        // Test that all vector types conform to VectorProtocol
        XCTAssertEqual(Vector128.dimensions, 128)
        XCTAssertEqual(Vector256.dimensions, 256)
        XCTAssertEqual(Vector512.dimensions, 512)
        XCTAssertEqual(Vector768.dimensions, 768)
        XCTAssertEqual(Vector1536.dimensions, 1536)
        
        // SIMD types don't directly conform to our protocols
        // They are used internally for storage optimization
    }
    
    func testVectorProtocolHelpers() {
        // Test validation
        XCTAssertTrue(Vector256.validate([Float](repeating: 0, count: 256)))
        XCTAssertFalse(Vector256.validate([Float](repeating: 0, count: 255)))
        
        // Test optional creation
        let valid = Vector256.create(from: [Float](repeating: 1, count: 256))
        XCTAssertNotNil(valid)
        
        let invalid = Vector256.create(from: [Float](repeating: 1, count: 100))
        XCTAssertNil(invalid)
    }
    
    // MARK: - Collection Conformance Tests
    
    // Removed: testVector512Collection - Vector512 doesn't conform to Collection
    
    // MARK: - Edge Cases
    
    func testEdgeCases() {
        // Test with NaN values
        var nanValues = [Float](repeating: 0, count: 512)
        nanValues[100] = Float.nan
        let nanVector = Vector512(nanValues)
        
        // Dot product with NaN should propagate
        let normalVector = Vector512(repeating: 1)
        let dotWithNaN = nanVector.dotProduct(normalVector)
        XCTAssertTrue(dotWithNaN.isNaN)
        
        // Test with infinity
        var infValues = [Float](repeating: 0, count: 512)
        infValues[200] = Float.infinity
        let infVector = Vector512(infValues)
        
        let magnitude = infVector.magnitude
        XCTAssertTrue(magnitude.isInfinite)
    }
    
    // MARK: - Performance Tests
    
    func testDotProductPerformance() {
        let a = Vector512(repeating: 0.5)
        let b = Vector512(repeating: 0.7)
        
        measure {
            for _ in 0..<10000 {
                _ = a.dotProduct(b)
            }
        }
    }
    
    func testBatchCreationPerformance() {
        let data = (0..<51200).map { Float($0) } // 100 Vector512s
        
        measure {
            _ = Vector512.createBatch(from: data)
        }
    }
}