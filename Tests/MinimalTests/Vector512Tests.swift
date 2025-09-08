import XCTest
@testable import VectorCore

final class Vector512Tests: XCTestCase {
    
    // MARK: - Basic Tests
    
    func testInitialization() {
        // Test zero initialization
        let zero = Vector512Optimized()
        XCTAssertEqual(zero[0], 0.0)
        XCTAssertEqual(zero[511], 0.0)
        
        // Test repeating value
        let ones = Vector512Optimized(repeating: 1.0)
        XCTAssertEqual(ones[0], 1.0)
        XCTAssertEqual(ones[255], 1.0)
        XCTAssertEqual(ones[511], 1.0)
    }
    
    func testArrayInitialization() throws {
        let array = Array(repeating: Float(2.5), count: 512)
        let vector = try Vector512Optimized(array)
        
        XCTAssertEqual(vector[0], 2.5)
        XCTAssertEqual(vector[255], 2.5)
        XCTAssertEqual(vector[511], 2.5)
    }
    
    func testSubscriptAccess() {
        var vector = Vector512Optimized(repeating: 0.0)
        
        // Test setting values
        vector[0] = 1.0
        vector[255] = 2.0
        vector[511] = 3.0
        
        // Test getting values
        XCTAssertEqual(vector[0], 1.0)
        XCTAssertEqual(vector[255], 2.0)
        XCTAssertEqual(vector[511], 3.0)
    }
    
    // MARK: - Arithmetic Operations
    
    func testAddition() {
        let a = Vector512Optimized(repeating: 1.0)
        let b = Vector512Optimized(repeating: 2.0)
        let c = a + b
        
        XCTAssertEqual(c[0], 3.0)
        XCTAssertEqual(c[255], 3.0)
        XCTAssertEqual(c[511], 3.0)
    }
    
    func testSubtraction() {
        let a = Vector512Optimized(repeating: 5.0)
        let b = Vector512Optimized(repeating: 2.0)
        let c = a - b
        
        XCTAssertEqual(c[0], 3.0)
        XCTAssertEqual(c[511], 3.0)
    }
    
    func testScalarMultiplication() {
        let vector = Vector512Optimized(repeating: 2.0)
        let scaled = vector * 3.0
        
        XCTAssertEqual(scaled[0], 6.0)
        XCTAssertEqual(scaled[511], 6.0)
    }
    
    // MARK: - SIMD Operations
    
    func testDotProduct() {
        let a = Vector512Optimized(repeating: 2.0)
        let b = Vector512Optimized(repeating: 3.0)
        
        let dot = a.dotProduct(b)
        let expected: Float = 2.0 * 3.0 * 512
        
        XCTAssertEqual(dot, expected, accuracy: 0.001)
    }
    
    func testMagnitude() {
        let vector = Vector512Optimized(repeating: 1.0)
        let magnitude = vector.magnitude
        let expected = sqrt(Float(512))
        
        XCTAssertEqual(magnitude, expected, accuracy: 0.001)
    }
    
    func testEuclideanDistance() {
        let a = Vector512Optimized(repeating: 0.0)
        let b = Vector512Optimized(repeating: 1.0)
        
        let distance = a.euclideanDistance(to: b)
        let expected = sqrt(Float(512))
        
        XCTAssertEqual(distance, expected, accuracy: 0.001)
    }
    
    func testNormalization() throws {
        let vector = Vector512Optimized(repeating: 2.0)
        let normalized = try vector.normalizedThrowing()
        
        let magnitude = normalized.magnitude
        XCTAssertEqual(magnitude, 1.0, accuracy: 0.0001)
    }
    
    func testCosineSimilarity() {
        let a = Vector512Optimized(repeating: 1.0)
        let b = Vector512Optimized(repeating: 1.0)
        
        let similarity = a.cosineSimilarity(to: b)
        XCTAssertEqual(similarity, 1.0, accuracy: 0.0001)
    }
    
    // MARK: - Performance Tests
    
    func testDotProductPerformance() {
        let a = Vector512Optimized(repeating: 1.0)
        let b = Vector512Optimized(repeating: 2.0)
        
        measure {
            for _ in 0..<10000 {
                _ = a.dotProduct(b)
            }
        }
    }
    
    func testDistancePerformance() {
        let a = Vector512Optimized(repeating: 1.0)
        let b = Vector512Optimized(repeating: 2.0)
        
        measure {
            for _ in 0..<10000 {
                _ = a.euclideanDistanceSquared(to: b)
            }
        }
    }
    
    func testNormalizationPerformance() throws {
        let vector = Vector512Optimized(repeating: 2.0)
        
        measure {
            for _ in 0..<10000 {
                _ = try? vector.normalizedThrowing()
            }
        }
    }
}
