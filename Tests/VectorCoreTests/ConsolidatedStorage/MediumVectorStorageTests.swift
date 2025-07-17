// MediumVectorStorage Tests
//
// Comprehensive tests for MediumVectorStorage (65-512 dimensions)
//

import XCTest
@testable import VectorCore

final class MediumVectorStorageTests: XCTestCase {
    
    // MARK: - Initialization Tests
    
    func testDefaultInitialization() {
        let storage = MediumVectorStorage()
        XCTAssertEqual(storage.count, 512)
        
        // Verify zeroes (spot check)
        XCTAssertEqual(storage[0], 0.0)
        XCTAssertEqual(storage[255], 0.0)
        XCTAssertEqual(storage[511], 0.0)
    }
    
    func testInitWithCount() {
        // Test various sizes in the medium range
        let sizes = [65, 100, 128, 256, 384, 512]
        
        for size in sizes {
            let storage = MediumVectorStorage(count: size)
            XCTAssertEqual(storage.count, size)
            
            // Verify first and last elements are zero
            XCTAssertEqual(storage[0], 0.0)
            XCTAssertEqual(storage[size - 1], 0.0)
        }
    }
    
    func testInitWithRepeatingValue() {
        let storage = MediumVectorStorage(repeating: 2.5)
        XCTAssertEqual(storage.count, 512)
        
        // Check several positions
        XCTAssertEqual(storage[0], 2.5)
        XCTAssertEqual(storage[100], 2.5)
        XCTAssertEqual(storage[511], 2.5)
    }
    
    func testInitWithCountAndRepeatingValue() {
        let storage = MediumVectorStorage(count: 200, repeating: 3.14)
        XCTAssertEqual(storage.count, 200)
        
        // Verify values
        for i in [0, 50, 100, 150, 199] {
            XCTAssertEqual(storage[i], 3.14)
        }
    }
    
    func testInitFromArray() {
        let values: [Float] = Array(1...300).map { Float($0) }
        let storage = MediumVectorStorage(from: values)
        
        XCTAssertEqual(storage.count, 300)
        
        // Check some values
        XCTAssertEqual(storage[0], 1.0)
        XCTAssertEqual(storage[149], 150.0)
        XCTAssertEqual(storage[299], 300.0)
    }
    
    // MARK: - Bounds Checking Tests
    
    func testBoundsChecking() {
        let storage = MediumVectorStorage(count: 100)
        
        // Valid access
        XCTAssertNoThrow(_ = storage[0])
        XCTAssertNoThrow(_ = storage[99])
        
        // Document that out-of-bounds access would trap
        // (Can't test directly as it would crash)
    }
    
    func testPaddingNotAccessible() {
        // Verify that padding elements (beyond actualCount) are not accessible
        let storage = MediumVectorStorage(count: 100)
        
        // Can access up to actualCount
        XCTAssertEqual(storage.count, 100)
        
        // Buffer access should only expose actualCount elements
        storage.withUnsafeBufferPointer { buffer in
            XCTAssertEqual(buffer.count, 100) // Not 512
        }
    }
    
    // MARK: - Value Semantics Tests
    
    func testValueSemantics() {
        var storage1 = MediumVectorStorage(count: 128, repeating: 1.0)
        let storage2 = storage1  // Copy
        
        // Modify storage1
        storage1[0] = 42.0
        
        // Verify storage2 is unchanged (value semantics via COW)
        XCTAssertEqual(storage1[0], 42.0)
        XCTAssertEqual(storage2[0], 1.0)
    }
    
    func testCOWBehavior() {
        // Test that COW is working properly
        let original = MediumVectorStorage(count: 256, repeating: 5.0)
        var copy1 = original
        var copy2 = original
        
        // No mutations yet - all should share storage
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
        let values: [Float] = Array(1...100).map { Float($0) }
        let storage = MediumVectorStorage(from: values)
        
        storage.withUnsafeBufferPointer { buffer in
            XCTAssertEqual(buffer.count, 100) // Only actualCount, not 512
            
            // Verify some values
            XCTAssertEqual(buffer[0], 1.0)
            XCTAssertEqual(buffer[99], 100.0)
        }
    }
    
    func testUnsafeMutableBufferPointerAccess() {
        var storage = MediumVectorStorage(count: 150)
        
        storage.withUnsafeMutableBufferPointer { buffer in
            XCTAssertEqual(buffer.count, 150)
            
            // Modify through buffer
            buffer[0] = 10.0
            buffer[74] = 20.0
            buffer[149] = 30.0
        }
        
        // Verify changes persisted
        XCTAssertEqual(storage[0], 10.0)
        XCTAssertEqual(storage[74], 20.0)
        XCTAssertEqual(storage[149], 30.0)
    }
    
    // MARK: - Dot Product Tests
    
    func testDotProduct() {
        let storage1 = MediumVectorStorage(count: 100, repeating: 2.0)
        let storage2 = MediumVectorStorage(count: 100, repeating: 3.0)
        
        let result = storage1.dotProduct(storage2)
        
        // 100 * (2 * 3) = 600
        XCTAssertEqual(result, 600.0)
    }
    
    func testDotProductWithDifferentValues() {
        let values1: [Float] = Array(1...200).map { Float($0) }
        let values2: [Float] = Array(1...200).map { Float($0 * 2) }
        
        let storage1 = MediumVectorStorage(from: values1)
        let storage2 = MediumVectorStorage(from: values2)
        
        let result = storage1.dotProduct(storage2)
        
        // Sum of i * (2i) for i from 1 to 200
        // = 2 * sum(i^2) = 2 * (200 * 201 * 401 / 6)
        let expected: Float = 2 * Float(200 * 201 * 401) / 6.0
        XCTAssertEqual(result, expected, accuracy: 1.0)
    }
    
    // MARK: - Factory Methods Tests
    
    func testVector128Factory() {
        let values: [Float] = Array(repeating: 1.5, count: 128)
        let storage = MediumVectorStorage.vector128(from: values)
        
        XCTAssertEqual(storage.count, 128)
        XCTAssertEqual(storage[0], 1.5)
        XCTAssertEqual(storage[127], 1.5)
    }
    
    func testVector256Factory() {
        let values: [Float] = Array(1...256).map { Float($0) }
        let storage = MediumVectorStorage.vector256(from: values)
        
        XCTAssertEqual(storage.count, 256)
        XCTAssertEqual(storage[0], 1.0)
        XCTAssertEqual(storage[255], 256.0)
    }
    
    func testVector512Factory() {
        let values: [Float] = Array(repeating: 7.0, count: 512)
        let storage = MediumVectorStorage.vector512(from: values)
        
        XCTAssertEqual(storage.count, 512)
        XCTAssertEqual(storage[0], 7.0)
        XCTAssertEqual(storage[511], 7.0)
    }
    
    func testZerosFactory() {
        let storage = MediumVectorStorage.zeros(count: 300)
        
        XCTAssertEqual(storage.count, 300)
        for i in [0, 150, 299] {
            XCTAssertEqual(storage[i], 0.0)
        }
    }
    
    func testOnesFactory() {
        let storage = MediumVectorStorage.ones(count: 400)
        
        XCTAssertEqual(storage.count, 400)
        for i in [0, 200, 399] {
            XCTAssertEqual(storage[i], 1.0)
        }
    }
    
    // MARK: - Arithmetic Operations Tests
    
    func testAddition() {
        let storage1 = MediumVectorStorage(count: 128, repeating: 10.0)
        let storage2 = MediumVectorStorage(count: 128, repeating: 5.0)
        
        let result = storage1 + storage2
        
        XCTAssertEqual(result.count, 128)
        XCTAssertEqual(result[0], 15.0)
        XCTAssertEqual(result[127], 15.0)
    }
    
    // MARK: - Alignment Tests
    
    func testMemoryAlignment() {
        let storage = MediumVectorStorage(count: 256)
        
        let alignment = storage.verifyAlignment()
        
        // Should have 64-byte alignment from AlignedValueStorage
        XCTAssertEqual(alignment, 64)
    }
    
    // MARK: - Edge Cases Tests
    
    func testMinimumSize() {
        let storage = MediumVectorStorage(count: 65)
        XCTAssertEqual(storage.count, 65)
    }
    
    func testMaximumSize() {
        let storage = MediumVectorStorage(count: 512)
        XCTAssertEqual(storage.count, 512)
        
        // Should be able to use full capacity
        storage.withUnsafeBufferPointer { buffer in
            XCTAssertEqual(buffer.count, 512)
        }
    }
    
    func testCommonEmbeddingSizes() {
        // Test common embedding dimensions
        let sizes = [128, 256, 384, 512]
        
        for size in sizes {
            let storage = MediumVectorStorage(count: size, repeating: Float(size))
            XCTAssertEqual(storage.count, size)
            XCTAssertEqual(storage[0], Float(size))
            XCTAssertEqual(storage[size - 1], Float(size))
        }
    }
    
    // MARK: - Performance Characteristics Tests
    
    func testHeapAllocation() {
        // MediumVectorStorage uses heap allocation via AlignedValueStorage
        // We verify it has value semantics despite heap backing
        let storage1 = MediumVectorStorage(count: 256, repeating: 1.0)
        var storage2 = storage1
        
        // Mutation should trigger COW
        storage2[0] = 42.0
        
        // Verify independence
        XCTAssertEqual(storage1[0], 1.0)
        XCTAssertEqual(storage2[0], 42.0)
    }
}