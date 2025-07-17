// LargeVectorStorage Tests
//
// Comprehensive tests for LargeVectorStorage (513+ dimensions)
//

import XCTest
@testable import VectorCore

final class LargeVectorStorageTests: XCTestCase {
    
    // MARK: - Initialization Tests
    
    func testDefaultInitialization() {
        let storage = LargeVectorStorage()
        XCTAssertEqual(storage.count, 1024)
        
        // Verify zeroes (spot check)
        XCTAssertEqual(storage[0], 0.0)
        XCTAssertEqual(storage[512], 0.0)
        XCTAssertEqual(storage[1023], 0.0)
    }
    
    func testInitWithCount() {
        // Test various large sizes
        let sizes = [513, 768, 1024, 1536, 2048, 3072, 4096]
        
        for size in sizes {
            let storage = LargeVectorStorage(count: size)
            XCTAssertEqual(storage.count, size)
            
            // Verify first and last elements are zero
            XCTAssertEqual(storage[0], 0.0)
            XCTAssertEqual(storage[size - 1], 0.0)
        }
    }
    
    func testInitWithRepeatingValue() {
        let storage = LargeVectorStorage(repeating: 4.2)
        XCTAssertEqual(storage.count, 1024)
        
        // Check several positions
        XCTAssertEqual(storage[0], 4.2)
        XCTAssertEqual(storage[500], 4.2)
        XCTAssertEqual(storage[1023], 4.2)
    }
    
    func testInitWithCountAndRepeatingValue() {
        let storage = LargeVectorStorage(count: 2000, repeating: 1.5)
        XCTAssertEqual(storage.count, 2000)
        
        // Verify values at various positions
        for i in [0, 500, 1000, 1500, 1999] {
            XCTAssertEqual(storage[i], 1.5)
        }
    }
    
    func testInitFromArray() {
        let values: [Float] = Array(1...1000).map { Float($0) }
        let storage = LargeVectorStorage(from: values)
        
        XCTAssertEqual(storage.count, 1000)
        
        // Check some values
        XCTAssertEqual(storage[0], 1.0)
        XCTAssertEqual(storage[499], 500.0)
        XCTAssertEqual(storage[999], 1000.0)
    }
    
    func testInitWithCommonSize() {
        // Test common size enums
        let testCases: [(LargeVectorStorage.CommonSize, Int)] = [
            (.dim768, 768),
            (.dim1024, 1024),
            (.dim1536, 1536),
            (.dim2048, 2048),
            (.dim3072, 3072),
            (.dim4096, 4096)
        ]
        
        for (commonSize, expectedCount) in testCases {
            let storage = LargeVectorStorage(commonSize: commonSize)
            XCTAssertEqual(storage.count, expectedCount)
            XCTAssertEqual(storage[0], 0.0)
        }
    }
    
    func testInitWithCommonSizeAndRepeatingValue() {
        let storage = LargeVectorStorage(commonSize: .dim1536, repeating: 2.5)
        XCTAssertEqual(storage.count, 1536)
        XCTAssertEqual(storage[0], 2.5)
        XCTAssertEqual(storage[1535], 2.5)
    }
    
    // MARK: - Value Semantics Tests
    
    func testValueSemantics() {
        var storage1 = LargeVectorStorage(count: 1000, repeating: 1.0)
        let storage2 = storage1  // Copy
        
        // Modify storage1
        storage1[0] = 42.0
        
        // Verify storage2 is unchanged (value semantics via COW)
        XCTAssertEqual(storage1[0], 42.0)
        XCTAssertEqual(storage2[0], 1.0)
    }
    
    func testCOWBehavior() {
        // Test that COW is working through COWDynamicStorage
        let original = LargeVectorStorage(count: 2048, repeating: 5.0)
        var copy1 = original
        var copy2 = original
        
        // No mutations yet
        XCTAssertEqual(original[0], 5.0)
        XCTAssertEqual(copy1[0], 5.0)
        XCTAssertEqual(copy2[0], 5.0)
        
        // Mutate copy1
        copy1[0] = 10.0
        
        // Original and copy2 should be unchanged
        XCTAssertEqual(original[0], 5.0)
        XCTAssertEqual(copy1[0], 10.0)
        XCTAssertEqual(copy2[0], 5.0)
        
        // Mutate copy2
        copy2[1] = 20.0
        
        // All should be independent
        XCTAssertEqual(original[1], 5.0)
        XCTAssertEqual(copy1[1], 5.0)
        XCTAssertEqual(copy2[1], 20.0)
    }
    
    // MARK: - Buffer Access Tests
    
    func testUnsafeBufferPointerAccess() {
        let values: [Float] = Array(1...1000).map { Float($0) }
        let storage = LargeVectorStorage(from: values)
        
        storage.withUnsafeBufferPointer { buffer in
            XCTAssertEqual(buffer.count, 1000)
            
            // Verify some values
            XCTAssertEqual(buffer[0], 1.0)
            XCTAssertEqual(buffer[999], 1000.0)
        }
    }
    
    func testUnsafeMutableBufferPointerAccess() {
        var storage = LargeVectorStorage(count: 1500)
        
        storage.withUnsafeMutableBufferPointer { buffer in
            XCTAssertEqual(buffer.count, 1500)
            
            // Modify through buffer
            buffer[0] = 10.0
            buffer[749] = 20.0
            buffer[1499] = 30.0
        }
        
        // Verify changes persisted
        XCTAssertEqual(storage[0], 10.0)
        XCTAssertEqual(storage[749], 20.0)
        XCTAssertEqual(storage[1499], 30.0)
    }
    
    // MARK: - Dot Product Tests
    
    func testDotProduct() {
        let storage1 = LargeVectorStorage(count: 1000, repeating: 2.0)
        let storage2 = LargeVectorStorage(count: 1000, repeating: 3.0)
        
        let result = storage1.dotProduct(storage2)
        
        // 1000 * (2 * 3) = 6000
        XCTAssertEqual(result, 6000.0, accuracy: 0.01)
    }
    
    func testChunkedDotProduct() {
        let storage1 = LargeVectorStorage(count: 5000, repeating: 2.0)
        let storage2 = LargeVectorStorage(count: 5000, repeating: 3.0)
        
        // Test chunked version
        let chunkedResult = storage1.chunkedDotProduct(storage2, chunkSize: 1024)
        
        // Should be same as regular dot product
        let regularResult = storage1.dotProduct(storage2)
        
        XCTAssertEqual(chunkedResult, regularResult, accuracy: 0.01)
        XCTAssertEqual(chunkedResult, 30000.0, accuracy: 0.01) // 5000 * 6
    }
    
    func testChunkedDotProductVariousSizes() {
        let values1: [Float] = Array(1...2000).map { Float($0) }
        let values2: [Float] = Array(1...2000).map { Float($0) }
        
        let storage1 = LargeVectorStorage(from: values1)
        let storage2 = LargeVectorStorage(from: values2)
        
        let regularResult = storage1.dotProduct(storage2)
        
        // Test different chunk sizes
        let chunkSizes = [256, 512, 1024, 1500]
        
        for chunkSize in chunkSizes {
            let chunkedResult = storage1.chunkedDotProduct(storage2, chunkSize: chunkSize)
            // Use relative tolerance for large numbers
            let tolerance = abs(regularResult) * 0.00001 // 0.001% tolerance
            XCTAssertEqual(chunkedResult, regularResult, accuracy: tolerance,
                          "Chunked result with chunk size \(chunkSize) doesn't match")
        }
    }
    
    // MARK: - Factory Methods Tests
    
    func testVector768Factory() {
        let values: [Float] = Array(repeating: 1.5, count: 768)
        let storage = LargeVectorStorage.vector768(from: values)
        
        XCTAssertEqual(storage.count, 768)
        XCTAssertEqual(storage[0], 1.5)
        XCTAssertEqual(storage[767], 1.5)
    }
    
    func testVector1536Factory() {
        let values: [Float] = Array(1...1536).map { Float($0) }
        let storage = LargeVectorStorage.vector1536(from: values)
        
        XCTAssertEqual(storage.count, 1536)
        XCTAssertEqual(storage[0], 1.0)
        XCTAssertEqual(storage[1535], 1536.0)
    }
    
    func testVector3072Factory() {
        let values: [Float] = Array(repeating: 7.0, count: 3072)
        let storage = LargeVectorStorage.vector3072(from: values)
        
        XCTAssertEqual(storage.count, 3072)
        XCTAssertEqual(storage[0], 7.0)
        XCTAssertEqual(storage[3071], 7.0)
    }
    
    func testZerosFactory() {
        let storage = LargeVectorStorage.zeros(count: 2000)
        
        XCTAssertEqual(storage.count, 2000)
        for i in [0, 1000, 1999] {
            XCTAssertEqual(storage[i], 0.0)
        }
    }
    
    func testOnesFactory() {
        let storage = LargeVectorStorage.ones(count: 3000)
        
        XCTAssertEqual(storage.count, 3000)
        for i in [0, 1500, 2999] {
            XCTAssertEqual(storage[i], 1.0)
        }
    }
    
    // MARK: - Arithmetic Operations Tests
    
    func testAddition() {
        let storage1 = LargeVectorStorage(count: 1024, repeating: 10.0)
        let storage2 = LargeVectorStorage(count: 1024, repeating: 5.0)
        
        let result = storage1 + storage2
        
        XCTAssertEqual(result.count, 1024)
        XCTAssertEqual(result[0], 15.0)
        XCTAssertEqual(result[1023], 15.0)
    }
    
    // MARK: - Edge Cases Tests
    
    func testMinimumSize() {
        let storage = LargeVectorStorage(count: 513)
        XCTAssertEqual(storage.count, 513)
    }
    
    func testVeryLargeSize() {
        // Test with a very large vector
        let storage = LargeVectorStorage(count: 10000)
        XCTAssertEqual(storage.count, 10000)
        
        // Should be able to access all elements
        XCTAssertEqual(storage[0], 0.0)
        XCTAssertEqual(storage[9999], 0.0)
    }
    
    func testDynamicGrowth() {
        // LargeVectorStorage can handle arbitrary sizes efficiently
        let sizes = [1000, 5000, 10000, 20000]
        
        for size in sizes {
            let storage = LargeVectorStorage(count: size, repeating: Float(size))
            XCTAssertEqual(storage.count, size)
            XCTAssertEqual(storage[0], Float(size))
            XCTAssertEqual(storage[size - 1], Float(size))
        }
    }
    
    // MARK: - Performance Characteristics Tests
    
    func testDynamicAllocation() {
        // LargeVectorStorage uses dynamic allocation
        // Verify it scales to exact size needed
        let storage1 = LargeVectorStorage(count: 1000)
        let storage2 = LargeVectorStorage(count: 10000)
        
        XCTAssertEqual(storage1.count, 1000)
        XCTAssertEqual(storage2.count, 10000)
        
        // Both should have value semantics
        var copy1 = storage1
        copy1[0] = 42.0
        
        XCTAssertEqual(storage1[0], 0.0)
        XCTAssertEqual(copy1[0], 42.0)
    }
}