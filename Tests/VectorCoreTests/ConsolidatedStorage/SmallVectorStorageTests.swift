// SmallVectorStorage Tests
//
// Comprehensive tests for SmallVectorStorage (1-64 dimensions)
//

import XCTest
@testable import VectorCore

final class SmallVectorStorageTests: XCTestCase {
    
    // MARK: - Initialization Tests
    
    func testDefaultInitialization() {
        let storage = SmallVectorStorage()
        XCTAssertEqual(storage.count, 64)
        
        // Verify all elements are zero
        for i in 0..<64 {
            XCTAssertEqual(storage[i], 0.0)
        }
    }
    
    func testInitWithCount() {
        // Test various sizes
        let sizes = [1, 16, 32, 50, 64]
        
        for size in sizes {
            let storage = SmallVectorStorage(count: size)
            XCTAssertEqual(storage.count, size)
            
            // Verify accessible elements are zero
            for i in 0..<size {
                XCTAssertEqual(storage[i], 0.0)
            }
        }
    }
    
    func testInitWithRepeatingValue() {
        let storage = SmallVectorStorage(repeating: 3.14)
        XCTAssertEqual(storage.count, 64)
        
        for i in 0..<64 {
            XCTAssertEqual(storage[i], 3.14)
        }
    }
    
    func testInitWithCountAndRepeatingValue() {
        let storage = SmallVectorStorage(count: 20, repeating: 2.5)
        XCTAssertEqual(storage.count, 20)
        
        for i in 0..<20 {
            XCTAssertEqual(storage[i], 2.5)
        }
    }
    
    func testInitFromArray() {
        let values: [Float] = Array(1...30).map { Float($0) }
        let storage = SmallVectorStorage(from: values)
        
        XCTAssertEqual(storage.count, 30)
        
        for i in 0..<30 {
            XCTAssertEqual(storage[i], Float(i + 1))
        }
    }
    
    // MARK: - Bounds Checking Tests
    
    func testBoundsChecking() {
        let storage = SmallVectorStorage(count: 10)
        
        // Valid access
        XCTAssertNoThrow(_ = storage[0])
        XCTAssertNoThrow(_ = storage[9])
        
        // Invalid access should trap (in debug mode)
        // Note: These would cause runtime errors, so we can't directly test them
        // Instead, we document that bounds checking is enforced
    }
    
    // MARK: - Value Semantics Tests
    
    func testValueSemantics() {
        var storage1 = SmallVectorStorage(count: 10, repeating: 1.0)
        let storage2 = storage1  // Copy
        
        // Modify storage1
        storage1[0] = 42.0
        
        // Verify storage2 is unchanged (value semantics)
        XCTAssertEqual(storage1[0], 42.0)
        XCTAssertEqual(storage2[0], 1.0)
    }
    
    func testMutatingOperations() {
        var storage = SmallVectorStorage(count: 5, repeating: 1.0)
        
        // Test subscript setter
        storage[2] = 3.14
        XCTAssertEqual(storage[2], 3.14)
        
        // Test withUnsafeMutableBufferPointer
        storage.withUnsafeMutableBufferPointer { buffer in
            buffer[3] = 2.71
        }
        XCTAssertEqual(storage[3], 2.71)
    }
    
    // MARK: - Buffer Access Tests
    
    func testUnsafeBufferPointerAccess() {
        let values: [Float] = [1, 2, 3, 4, 5]
        let storage = SmallVectorStorage(from: values)
        
        storage.withUnsafeBufferPointer { buffer in
            XCTAssertEqual(buffer.count, 5)
            XCTAssertEqual(Array(buffer), values)
        }
    }
    
    func testUnsafeMutableBufferPointerAccess() {
        var storage = SmallVectorStorage(count: 3)
        
        storage.withUnsafeMutableBufferPointer { buffer in
            XCTAssertEqual(buffer.count, 3)
            buffer[0] = 1.0
            buffer[1] = 2.0
            buffer[2] = 3.0
        }
        
        XCTAssertEqual(storage[0], 1.0)
        XCTAssertEqual(storage[1], 2.0)
        XCTAssertEqual(storage[2], 3.0)
    }
    
    // MARK: - Dot Product Tests
    
    func testDotProduct() {
        let values1: [Float] = [1, 2, 3, 4]
        let values2: [Float] = [5, 6, 7, 8]
        
        let storage1 = SmallVectorStorage(from: values1)
        let storage2 = SmallVectorStorage(from: values2)
        
        let result = storage1.dotProduct(storage2)
        
        // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        XCTAssertEqual(result, 70.0)
    }
    
    func testDotProductWithPartialUsage() {
        // Test that dot product only uses actualCount elements
        let storage1 = SmallVectorStorage(count: 10, repeating: 1.0)
        let storage2 = SmallVectorStorage(count: 10, repeating: 2.0)
        
        let result = storage1.dotProduct(storage2)
        
        // 10 * (1 * 2) = 20
        XCTAssertEqual(result, 20.0)
    }
    
    // MARK: - Arithmetic Operations Tests
    
    func testAddition() {
        let values1: [Float] = [1, 2, 3]
        let values2: [Float] = [4, 5, 6]
        
        let storage1 = SmallVectorStorage(from: values1)
        let storage2 = SmallVectorStorage(from: values2)
        
        let result = storage1 + storage2
        
        XCTAssertEqual(result.count, 3)
        XCTAssertEqual(result[0], 5.0)
        XCTAssertEqual(result[1], 7.0)
        XCTAssertEqual(result[2], 9.0)
    }
    
    func testSubtraction() {
        let values1: [Float] = [10, 20, 30]
        let values2: [Float] = [3, 5, 7]
        
        let storage1 = SmallVectorStorage(from: values1)
        let storage2 = SmallVectorStorage(from: values2)
        
        let result = storage1 - storage2
        
        XCTAssertEqual(result.count, 3)
        XCTAssertEqual(result[0], 7.0)
        XCTAssertEqual(result[1], 15.0)
        XCTAssertEqual(result[2], 23.0)
    }
    
    func testScalarMultiplication() {
        let values: [Float] = [1, 2, 3, 4]
        let storage = SmallVectorStorage(from: values)
        
        let result = storage * 2.5
        
        XCTAssertEqual(result.count, 4)
        XCTAssertEqual(result[0], 2.5)
        XCTAssertEqual(result[1], 5.0)
        XCTAssertEqual(result[2], 7.5)
        XCTAssertEqual(result[3], 10.0)
    }
    
    // MARK: - Factory Methods Tests
    
    func testZerosFactory() {
        let storage = SmallVectorStorage.zeros(count: 15)
        
        XCTAssertEqual(storage.count, 15)
        for i in 0..<15 {
            XCTAssertEqual(storage[i], 0.0)
        }
    }
    
    func testOnesFactory() {
        let storage = SmallVectorStorage.ones(count: 25)
        
        XCTAssertEqual(storage.count, 25)
        for i in 0..<25 {
            XCTAssertEqual(storage[i], 1.0)
        }
    }
    
    // MARK: - Edge Cases Tests
    
    func testMinimumSize() {
        let storage = SmallVectorStorage(count: 1)
        XCTAssertEqual(storage.count, 1)
        XCTAssertEqual(storage[0], 0.0)
    }
    
    func testMaximumSize() {
        let storage = SmallVectorStorage(count: 64)
        XCTAssertEqual(storage.count, 64)
        
        // Test we can access all elements
        for i in 0..<64 {
            XCTAssertEqual(storage[i], 0.0)
        }
    }
    
    func testCommonSizes() {
        // Test common SIMD-friendly sizes
        let sizes = [8, 16, 32, 64]
        
        for size in sizes {
            let storage = SmallVectorStorage(count: size, repeating: Float(size))
            XCTAssertEqual(storage.count, size)
            
            // Verify all elements
            for i in 0..<size {
                XCTAssertEqual(storage[i], Float(size))
            }
        }
    }
    
    // MARK: - Performance Characteristics Tests
    
    func testStackAllocation() {
        // SmallVectorStorage should be stack-allocated
        // We can't directly test this, but we can verify it's a value type
        let storage1 = SmallVectorStorage(count: 10)
        var storage2 = storage1
        
        // Modifying storage2 shouldn't affect storage1
        storage2.withUnsafeMutableBufferPointer { buffer in
            buffer[0] = 42.0
        }
        
        XCTAssertEqual(storage1[0], 0.0)
        XCTAssertEqual(storage2[0], 42.0)
    }
    
    func testMemoryLayout() {
        // Verify the struct has expected size
        let size = MemoryLayout<SmallVectorStorage>.size
        let expectedSize = MemoryLayout<SIMD64<Float>>.size + MemoryLayout<Int>.size
        
        // Should be close to expected (SIMD64 + actualCount)
        XCTAssertLessThanOrEqual(size, expectedSize + 16) // Allow some padding
    }
}