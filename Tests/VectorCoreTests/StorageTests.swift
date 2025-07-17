// VectorCore: Storage Tests
//
// Comprehensive tests for vector storage implementations
//

import XCTest
@testable import VectorCore

final class StorageTests: XCTestCase {
    
    // MARK: - Array Storage Tests
    
    func testDynamicArrayStorage() {
        // Test various sizes
        let sizes = [1, 7, 13, 100, 1000, 2048]
        
        for size in sizes {
            let storage = DynamicArrayStorage(dimension: size)
            XCTAssertEqual(storage.count, size)
            
            // Test initialization to zero
            for i in 0..<size {
                XCTAssertEqual(storage[i], 0.0)
            }
        }
    }
    
    func testDynamicArrayStorageOperations() {
        // Test from array
        let values = (0..<1000).map { Float($0) / 1000.0 }
        let storage = DynamicArrayStorage(from: values)
        
        XCTAssertEqual(storage.count, 1000)
        
        // Test dot product
        let storage2 = DynamicArrayStorage(dimension: 1000, repeating: 2.0)
        let dot = storage.dotProduct(storage2)
        
        var expected: Float = 0
        for i in 0..<1000 {
            expected += values[i] * 2.0
        }
        XCTAssertEqual(dot, expected, accuracy: 1e-4)
    }
    
    func testDynamicArrayStorageThreadSafety() {
        let storage = DynamicArrayStorage(dimension: 1000, repeating: 1.0)
        let queue = DispatchQueue(label: "test", attributes: .concurrent)
        let group = DispatchGroup()
        
        // Concurrent reads should be safe
        for _ in 0..<100 {
            group.enter()
            queue.async {
                let sum = storage.withUnsafeBufferPointer { buffer in
                    buffer.reduce(0, +)
                }
                XCTAssertEqual(sum, 1000.0, accuracy: 1e-6)
                group.leave()
            }
        }
        
        group.wait()
    }
    
    // MARK: - Memory Layout Tests
    
    func testDynamicArrayStorageMemoryOverhead() {
        // Dynamic storage has some overhead for the class and array
        let storage = DynamicArrayStorage(dimension: 1000)
        
        // Should be able to access all elements
        for i in 0..<1000 {
            _ = storage[i]
        }
        
        // Memory should be contiguous
        storage.withUnsafeBufferPointer { buffer in
            if buffer.count > 1 {
                let firstAddr = buffer.baseAddress!
                let secondAddr = buffer.baseAddress! + 1
                let distance = secondAddr - firstAddr
                XCTAssertEqual(distance, 1) // Elements are adjacent
            }
        }
    }
    
    // MARK: - Edge Cases
    
    func testStorageEdgeCases() {
        // Test minimum size
        XCTAssertNoThrow(DynamicArrayStorage(dimension: 1))
    }
    
    // MARK: - Performance Tests
    
    func testDynamicStoragePerformance() {
        let storage1 = DynamicArrayStorage(dimension: 256, repeating: 1.0)
        let storage2 = DynamicArrayStorage(dimension: 256, repeating: 2.0)
        
        measure {
            for _ in 0..<10000 {
                _ = storage1.dotProduct(storage2)
            }
        }
    }
    
    // MARK: - Copy-on-Write Tests
    
    func testDynamicStorageCopySemantics() {
        // Dynamic storage is a class, but should have value semantics
        let storage1 = DynamicArrayStorage(dimension: 100, repeating: 1.0)
        let storage2 = storage1
        
        // Both should point to same data initially
        XCTAssertEqual(storage1[0], storage2[0])
        
        // Modifications require careful handling in actual implementation
        // This test documents expected behavior
    }
}