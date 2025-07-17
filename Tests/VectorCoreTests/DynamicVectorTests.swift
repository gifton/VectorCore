// VectorCore: Dynamic Vector Tests
//
// Comprehensive tests for DynamicVector
//

import XCTest
@testable import VectorCore

final class DynamicVectorTests: XCTestCase {
    
    // MARK: - Initialization Tests
    
    func testInitialization() {
        // Test zero initialization
        let zero = DynamicVector(dimension: 100)
        XCTAssertEqual(zero.dimension, 100)
        XCTAssertEqual(zero.scalarCount, 100)
        for i in 0..<100 {
            XCTAssertEqual(zero[i], 0.0)
        }
        
        // Test repeating value initialization
        let repeating = DynamicVector(dimension: 50, repeating: 3.14)
        XCTAssertEqual(repeating.dimension, 50)
        for i in 0..<50 {
            XCTAssertEqual(repeating[i], 3.14)
        }
        
        // Test array initialization with dimension
        let values = (0..<75).map { Float($0) }
        let fromArray = DynamicVector(dimension: 75, values: values)
        XCTAssertEqual(fromArray.dimension, 75)
        for i in 0..<75 {
            XCTAssertEqual(fromArray[i], Float(i))
        }
        
        // Test array initialization (inferred dimension)
        let inferred = DynamicVector(values)
        XCTAssertEqual(inferred.dimension, values.count)
        XCTAssertEqual(inferred.toArray(), values)
        
        // Test initialization from buffer by converting to array
        let fromBuffer = values.withUnsafeBufferPointer { buffer in
            DynamicVector(Array(buffer))
        }
        XCTAssertEqual(fromBuffer.dimension, values.count)
        XCTAssertEqual(fromBuffer.toArray(), values)
    }
    
    func testInitializationErrors() {
        // Test dimension mismatch
        let _ = [Float](repeating: 1.0, count: 10)
        
        // This should trigger precondition failure in debug mode
        // In release mode, we can't test preconditions
        #if DEBUG
        // We can't directly test precondition failures in XCTest
        // but we document the expected behavior
        #endif
    }
    
    // MARK: - Subscript Access Tests
    
    func testSubscriptAccess() {
        let values = (0..<100).map { Float($0) }
        var vector = DynamicVector(values)
        
        // Test reading
        for i in 0..<100 {
            XCTAssertEqual(vector[i], Float(i))
        }
        
        // Test writing
        vector[50] = 999.0
        XCTAssertEqual(vector[50], 999.0)
        
        // Test multiple writes
        for i in 0..<10 {
            vector[i] = Float(i * i)
        }
        for i in 0..<10 {
            XCTAssertEqual(vector[i], Float(i * i))
        }
    }
    
    // MARK: - Array Conversion Tests
    
    func testToArray() {
        let originalValues = (0..<200).map { Float($0) * 0.5 }
        let vector = DynamicVector(originalValues)
        
        let arrayValues = vector.toArray()
        XCTAssertEqual(arrayValues, originalValues)
        
        // Test empty vector
        let emptyVector = DynamicVector(dimension: 0)
        XCTAssertEqual(emptyVector.toArray(), [])
        
        // Test single element
        let singleVector = DynamicVector([42.0])
        XCTAssertEqual(singleVector.toArray(), [42.0])
    }
    
    // MARK: - Mathematical Operations Tests
    
    func testDotProduct() {
        let a = DynamicVector([1, 2, 3, 4])
        let b = DynamicVector([5, 6, 7, 8])
        
        let dot = a.dotProduct(b)
        // Break down the calculation to help type inference
        let expected: Float = Float(1*5) + Float(2*6) + Float(3*7) + Float(4*8) // 5 + 12 + 21 + 32 = 70
        XCTAssertEqual(dot, expected)
        
        // Test with zero vector
        let zero = DynamicVector(dimension: 4)
        XCTAssertEqual(a.dotProduct(zero), 0.0)
        
        // Test with self
        let selfDot = a.dotProduct(a)
        let expectedSelf: Float = 1 + 4 + 9 + 16 // 30
        XCTAssertEqual(selfDot, expectedSelf)
    }
    
    func testMagnitude() {
        // Test simple case
        let v1 = DynamicVector([3, 4]) // 3-4-5 triangle
        XCTAssertEqual(v1.magnitude, 5.0, accuracy: 1e-6)
        XCTAssertEqual(v1.magnitudeSquared, 25.0)
        
        // Test zero vector
        let zero = DynamicVector(dimension: 10)
        XCTAssertEqual(zero.magnitude, 0.0)
        XCTAssertEqual(zero.magnitudeSquared, 0.0)
        
        // Test unit vector
        let unit = DynamicVector([1, 0, 0, 0])
        XCTAssertEqual(unit.magnitude, 1.0)
        
        // Test larger vector
        let v2 = DynamicVector([1, 1, 1, 1, 1, 1]) // sqrt(6)
        XCTAssertEqual(v2.magnitude, sqrt(6.0), accuracy: 1e-6)
    }
    
    func testNormalization() {
        // Test in-place normalization
        var v1 = DynamicVector([3, 4])
        v1.normalize()
        XCTAssertEqual(v1.magnitude, 1.0, accuracy: 1e-6)
        XCTAssertEqual(v1[0], 0.6, accuracy: 1e-6)
        XCTAssertEqual(v1[1], 0.8, accuracy: 1e-6)
        
        // Test normalized copy
        let v2 = DynamicVector([3, 4])
        let normalized = v2.normalized()
        XCTAssertEqual(normalized.magnitude, 1.0, accuracy: 1e-6)
        XCTAssertEqual(v2[0], 3.0) // Original unchanged
        
        // Test zero vector normalization (should remain zero)
        var zero = DynamicVector(dimension: 5)
        zero.normalize()
        XCTAssertEqual(zero.magnitude, 0.0)
        
        let zeroNormalized = zero.normalized()
        XCTAssertEqual(zeroNormalized.magnitude, 0.0)
    }
    
    func testDistance() {
        let v1 = DynamicVector([1, 2, 3])
        let v2 = DynamicVector([4, 6, 8])
        
        let distance = v1.distance(to: v2)
        // sqrt((4-1)² + (6-2)² + (8-3)²) = sqrt(9 + 16 + 25) = sqrt(50)
        XCTAssertEqual(distance, sqrt(50.0), accuracy: 1e-6)
        
        // Distance to self should be 0
        XCTAssertEqual(v1.distance(to: v1), 0.0)
        
        // Distance is symmetric
        XCTAssertEqual(v1.distance(to: v2), v2.distance(to: v1))
    }
    
    func testCosineSimilarity() {
        // Test parallel vectors
        let v1 = DynamicVector([1, 2, 3])
        let v2 = DynamicVector([2, 4, 6])
        XCTAssertEqual(v1.cosineSimilarity(to: v2), 1.0, accuracy: 1e-6)
        
        // Test orthogonal vectors
        let v3 = DynamicVector([1, 0, 0])
        let v4 = DynamicVector([0, 1, 0])
        XCTAssertEqual(v3.cosineSimilarity(to: v4), 0.0, accuracy: 1e-6)
        
        // Test opposite vectors
        let v5 = DynamicVector([1, 1, 1])
        let v6 = DynamicVector([-1, -1, -1])
        XCTAssertEqual(v5.cosineSimilarity(to: v6), -1.0, accuracy: 1e-6)
        
        // Test with zero vector
        let zero = DynamicVector(dimension: 3)
        XCTAssertEqual(v1.cosineSimilarity(to: zero), 0.0)
    }
    
    // MARK: - Protocol Conformance Tests
    
    func testBaseVectorProtocolConformance() {
        let vector = DynamicVector([1, 2, 3, 4, 5])
        
        // Test scalarCount
        XCTAssertEqual(vector.scalarCount, 5)
        
        // Test toArray
        XCTAssertEqual(vector.toArray(), [1, 2, 3, 4, 5])
        
        // Test subscript
        for i in 0..<5 {
            XCTAssertEqual(vector[i], Float(i + 1))
        }
    }
    
    func testExtendedVectorProtocolConformance() {
        let v1 = DynamicVector([3, 4])
        let v2 = DynamicVector([1, 2])
        
        // Test all protocol methods
        XCTAssertEqual(v1.dotProduct(v2), 11.0) // 3*1 + 4*2
        XCTAssertEqual(v1.magnitude, 5.0, accuracy: 1e-6)
        XCTAssertEqual(v1.normalized().magnitude, 1.0, accuracy: 1e-6)
        XCTAssertEqual(v1.distance(to: v2), sqrt(8.0), accuracy: 1e-6)
        XCTAssertGreaterThan(v1.cosineSimilarity(to: v2), 0.9) // Similar direction
    }
    
    func testVectorTypeProtocolConformance() {
        let vector = DynamicVector([1, 2, 3])
        
        // Test VectorType protocol requirements
        XCTAssertEqual(vector.scalarCount, 3)
        XCTAssertEqual(vector.toArray(), [1, 2, 3])
        XCTAssertEqual(vector.dotProduct(vector), 14.0) // 1 + 4 + 9
        XCTAssertEqual(vector.magnitude, sqrt(14.0), accuracy: 1e-6)
        XCTAssertEqual(vector.normalized().magnitude, 1.0, accuracy: 1e-6)
    }
    
    // MARK: - Binary Serialization Tests
    
    func testBinarySerializaton() throws {
        let original = DynamicVector([1.0, 2.0, 3.0, 4.0, 5.0])
        
        // Test encoding
        let encoded = try original.encodeBinary()
        XCTAssertGreaterThan(encoded.count, 0)
        
        // Test decoding
        let decoded = try DynamicVector.decodeBinary(from: encoded)
        XCTAssertEqual(decoded.dimension, original.dimension)
        XCTAssertEqual(decoded.toArray(), original.toArray())
        
        // Test round-trip with various sizes
        let sizes = [1, 10, 100, 1000]
        for size in sizes {
            let vector = DynamicVector((0..<size).map { Float($0) })
            let data = try vector.encodeBinary()
            let restored = try DynamicVector.decodeBinary(from: data)
            XCTAssertEqual(restored.toArray(), vector.toArray())
        }
    }
    
    // MARK: - Thread Safety Tests
    
    func testThreadSafety() async {
        let vector = DynamicVector((0..<1000).map { Float($0) })
        let iterations = 100
        
        // Concurrent reads should be safe
        await withTaskGroup(of: Float.self) { group in
            for _ in 0..<10 {
                group.addTask {
                    var sum: Float = 0
                    for _ in 0..<iterations {
                        // Read operations
                        sum += vector.magnitude
                        sum += vector.dotProduct(vector)
                        sum += vector.toArray().reduce(0, +)
                    }
                    return sum
                }
            }
            
            // All tasks should complete without issues
            var results: [Float] = []
            for await result in group {
                results.append(result)
            }
            
            // All results should be the same (deterministic operations)
            let first = results.first!
            for result in results {
                XCTAssertEqual(result, first, accuracy: 1e-3)
            }
        }
    }
    
    // MARK: - Memory Management Tests
    
    func testMemoryManagement() {
        // Test for memory leaks with large vectors
        autoreleasepool {
            // Create and destroy many large vectors
            for _ in 0..<100 {
                let largeVector = DynamicVector(dimension: 10000, repeating: 1.0)
                _ = largeVector.magnitude
                _ = largeVector.toArray()
            }
        }
        
        // Test copy-on-write behavior
        let original = DynamicVector((0..<1000).map { Float($0) })
        var copy = original
        
        // Should share storage until mutation
        copy[0] = 999.0
        
        // Verify original is unchanged
        XCTAssertEqual(original[0], 0.0)
        XCTAssertEqual(copy[0], 999.0)
    }
    
    // MARK: - Performance Tests
    
    func testPerformanceVsFixedVector() {
        let dimension = 512
        let dynamicVector = DynamicVector((0..<dimension).map { Float($0) })
        let _ = Vector512((0..<dimension).map { Float($0) })
        
        // Compare dot product performance
        measure {
            for _ in 0..<1000 {
                _ = dynamicVector.dotProduct(dynamicVector)
            }
        }
        
        // Note: In practice, fixed-size vectors should be faster due to
        // compile-time optimizations and SIMD usage
    }
    
    func testLargeVectorPerformance() {
        let largeVector = DynamicVector((0..<10000).map { Float($0) })
        
        measure {
            _ = largeVector.magnitude
            _ = largeVector.normalized()
        }
    }
    
    // MARK: - Edge Cases
    
    func testEdgeCases() {
        // Test with special float values
        let nanVector = DynamicVector([Float.nan, 1.0, 2.0])
        XCTAssertTrue(nanVector.magnitude.isNaN)
        XCTAssertTrue(nanVector.dotProduct(nanVector).isNaN)
        
        let infVector = DynamicVector([Float.infinity, 1.0, 2.0])
        XCTAssertTrue(infVector.magnitude.isInfinite)
        
        let negInfVector = DynamicVector([-Float.infinity, 1.0, 2.0])
        XCTAssertTrue(negInfVector.magnitude.isInfinite)
        
        // Test very small and very large dimensions
        let tiny = DynamicVector(dimension: 1, repeating: 42.0)
        XCTAssertEqual(tiny.dimension, 1)
        XCTAssertEqual(tiny[0], 42.0)
        
        let huge = DynamicVector(dimension: 100000, repeating: 0.1)
        XCTAssertEqual(huge.dimension, 100000)
        XCTAssertEqual(huge[50000], 0.1)
    }
    
    // MARK: - Arithmetic Operations Tests
    
    func testArithmeticOperations() {
        let a = DynamicVector([1, 2, 3])
        let b = DynamicVector([4, 5, 6])
        
        // Test addition
        let sum = a + b
        XCTAssertEqual(sum.toArray(), [5, 7, 9])
        
        // Test subtraction
        let diff = b - a
        XCTAssertEqual(diff.toArray(), [3, 3, 3])
        
        // Test scalar multiplication
        let scaled = a * 2.0
        XCTAssertEqual(scaled.toArray(), [2, 4, 6])
        
        // Test scalar division
        let divided = b / 2.0
        XCTAssertEqual(divided.toArray(), [2, 2.5, 3])
        
        // Test negation
        let negated = -a
        XCTAssertEqual(negated.toArray(), [-1, -2, -3])
    }
}