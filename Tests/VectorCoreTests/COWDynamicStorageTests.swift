// VectorCore: COW Dynamic Storage Tests
//
// Test Copy-on-Write behavior for dynamic storage
//

import XCTest
@testable import VectorCore

final class COWDynamicStorageTests: XCTestCase {
    
    func testInitialization() {
        // Test dimension initialization
        let storage1 = COWDynamicStorage(dimension: 100)
        XCTAssertEqual(storage1.count, 100)
        for i in 0..<100 {
            XCTAssertEqual(storage1[i], 0.0)
        }
        
        // Test repeating value
        let storage2 = COWDynamicStorage(dimension: 50, repeating: 3.14)
        XCTAssertEqual(storage2.count, 50)
        for i in 0..<50 {
            XCTAssertEqual(storage2[i], 3.14)
        }
        
        // Test from array
        let values = (0..<75).map { Float($0) }
        let storage3 = COWDynamicStorage(from: values)
        XCTAssertEqual(storage3.count, 75)
        for i in 0..<75 {
            XCTAssertEqual(storage3[i], Float(i))
        }
    }
    
    func testValueSemantics() {
        // Test Copy-on-Write behavior
        var storage1 = COWDynamicStorage(dimension: 100, repeating: 1.0)
        let storage2 = storage1
        
        // Before mutation, they should behave identically
        XCTAssertEqual(storage1[0], 1.0)
        XCTAssertEqual(storage2[0], 1.0)
        
        // Mutate storage1
        storage1[0] = 42.0
        
        // storage1 should be modified, storage2 should be unchanged
        XCTAssertEqual(storage1[0], 42.0)
        XCTAssertEqual(storage2[0], 1.0, "Copy should maintain original value")
    }
    
    func testMutableBufferCOW() {
        var storage1 = COWDynamicStorage(dimension: 100, repeating: 2.0)
        let storage2 = storage1
        
        // Modify through mutable buffer
        storage1.withUnsafeMutableBufferPointer { buffer in
            for i in 0..<10 {
                buffer[i] = Float(i) * 10.0
            }
        }
        
        // Verify COW triggered
        for i in 0..<10 {
            XCTAssertEqual(storage1[i], Float(i) * 10.0)
            XCTAssertEqual(storage2[i], 2.0)
        }
    }
    
    func testDotProduct() {
        let storage1 = COWDynamicStorage(dimension: 100, repeating: 2.0)
        let storage2 = COWDynamicStorage(dimension: 100, repeating: 3.0)
        
        let result = storage1.dotProduct(storage2)
        XCTAssertEqual(result, 600.0) // 2.0 * 3.0 * 100
    }
    
    func testMultipleCopies() {
        var original = COWDynamicStorage(dimension: 50, repeating: 10.0)
        var copy1 = original
        var copy2 = original
        var copy3 = copy1
        
        // Modify each independently
        original[0] = 100.0
        copy1[1] = 200.0
        copy2[2] = 300.0
        copy3[3] = 400.0
        
        // Verify independence
        XCTAssertEqual(original[0], 100.0)
        XCTAssertEqual(original[1], 10.0)
        XCTAssertEqual(original[2], 10.0)
        XCTAssertEqual(original[3], 10.0)
        
        XCTAssertEqual(copy1[0], 10.0)
        XCTAssertEqual(copy1[1], 200.0)
        XCTAssertEqual(copy1[2], 10.0)
        XCTAssertEqual(copy1[3], 10.0)
        
        XCTAssertEqual(copy2[0], 10.0)
        XCTAssertEqual(copy2[1], 10.0)
        XCTAssertEqual(copy2[2], 300.0)
        XCTAssertEqual(copy2[3], 10.0)
        
        XCTAssertEqual(copy3[0], 10.0)
        XCTAssertEqual(copy3[1], 10.0)
        XCTAssertEqual(copy3[2], 10.0)
        XCTAssertEqual(copy3[3], 400.0)
    }
    
    func testReadDoesNotTriggerCOW() {
        let storage1 = COWDynamicStorage(dimension: 1000, repeating: 5.0)
        let storage2 = storage1
        
        // Reading should not trigger COW
        _ = storage1[500]
        _ = storage1.withUnsafeBufferPointer { buffer in
            buffer.reduce(0, +)
        }
        
        // Both should still have same values
        XCTAssertEqual(storage1[0], 5.0)
        XCTAssertEqual(storage2[0], 5.0)
    }
    
    func testLargeDimensionSupport() {
        // Test with various sizes
        let sizes = [100, 1000, 10000, 100000]
        
        for size in sizes {
            let storage = COWDynamicStorage(dimension: size, repeating: 1.0)
            XCTAssertEqual(storage.count, size)
            XCTAssertEqual(storage[size - 1], 1.0)
        }
    }
    
    func testCOWPerformance() {
        let storage = COWDynamicStorage(dimension: 10000, repeating: 1.0)
        
        measure {
            // Many copies without mutation should be fast
            for _ in 0..<1000 {
                let _ = storage
            }
        }
    }
}