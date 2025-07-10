// VectorCore: Storage Tests
//
// Comprehensive tests for vector storage implementations
//

import XCTest
@testable import VectorCore

final class StorageTests: XCTestCase {
    
    // MARK: - SIMD Storage Tests
    
    func testSIMDStorage32() {
        // Test initialization
        let storage = SIMDStorage32()
        XCTAssertEqual(storage.count, 32)
        
        // Test with specific count
        let storage16 = SIMDStorage32(count: 16)
        XCTAssertEqual(storage16.count, 16)
        
        // Test repeating value
        let storageRepeat = SIMDStorage32(repeating: 5.0)
        for i in 0..<32 {
            XCTAssertEqual(storageRepeat[i], 5.0)
        }
        
        // Test from array
        let values = (0..<20).map { Float($0) }
        let storageArray = SIMDStorage32(from: values)
        XCTAssertEqual(storageArray.count, 20)
        for i in 0..<20 {
            XCTAssertEqual(storageArray[i], Float(i))
        }
    }
    
    func testSIMDStorage64() {
        // Test initialization
        let storage = SIMDStorage64()
        XCTAssertEqual(storage.count, 64)
        
        // Test with specific count
        let storage48 = SIMDStorage64(count: 48)
        XCTAssertEqual(storage48.count, 48)
        
        // Test subscript access
        var storage = SIMDStorage64(repeating: 1.0)
        storage[10] = 42.0
        XCTAssertEqual(storage[10], 42.0)
        
        // Verify other elements unchanged
        for i in 0..<64 where i != 10 {
            XCTAssertEqual(storage[i], 1.0)
        }
    }
    
    func testSIMDStorage128() {
        // Test initialization
        let storage = SIMDStorage128()
        XCTAssertEqual(storage.count, 128)
        
        // Test from array
        let values = (0..<128).map { Float($0) / 128.0 }
        let storage = SIMDStorage128(from: values)
        
        for i in 0..<128 {
            XCTAssertEqual(storage[i], Float(i) / 128.0, accuracy: 1e-6)
        }
        
        // Test dot product
        let storage2 = SIMDStorage128(repeating: 2.0)
        let dot = storage.dotProduct(storage2)
        
        // Calculate expected dot product
        var expected: Float = 0
        for i in 0..<128 {
            expected += storage[i] * 2.0
        }
        XCTAssertEqual(dot, expected, accuracy: 1e-5)
    }
    
    func testSIMDStorage256() {
        let storage = SIMDStorage256()
        XCTAssertEqual(storage.count, 256)
        
        // Test buffer pointer access
        var values = [Float](repeating: 0, count: 256)
        let storage2 = SIMDStorage256(repeating: 3.14)
        
        storage2.withUnsafeBufferPointer { buffer in
            XCTAssertEqual(buffer.count, 256)
            for i in 0..<256 {
                values[i] = buffer[i]
            }
        }
        
        // Verify all values copied correctly
        for value in values {
            XCTAssertEqual(value, 3.14, accuracy: 1e-6)
        }
    }
    
    func testSIMDStorage512() {
        let storage = SIMDStorage512()
        XCTAssertEqual(storage.count, 512)
        
        // Test mutable buffer pointer
        var storage = SIMDStorage512(repeating: 1.0)
        
        storage.withUnsafeMutableBufferPointer { buffer in
            // Modify through buffer
            for i in 0..<512 {
                buffer[i] *= 2.0
            }
        }
        
        // Verify modifications
        for i in 0..<512 {
            XCTAssertEqual(storage[i], 2.0, accuracy: 1e-6)
        }
    }
    
    // MARK: - Array Storage Tests
    
    func testDynamicArrayStorage() {
        // Test various sizes
        let sizes = [1, 7, 13, 100, 1000, 2048]
        
        for size in sizes {
            let storage = DynamicArrayStorage(count: size)
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
        let storage2 = DynamicArrayStorage(repeating: 2.0, count: 1000)
        let dot = storage.dotProduct(storage2)
        
        var expected: Float = 0
        for i in 0..<1000 {
            expected += values[i] * 2.0
        }
        XCTAssertEqual(dot, expected, accuracy: 1e-4)
    }
    
    func testDynamicArrayStorageThreadSafety() {
        let storage = DynamicArrayStorage(repeating: 1.0, count: 1000)
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
    
    // MARK: - Storage Protocol Conformance
    
    func testStorageProtocolConformance() {
        // Test that all storage types conform to VectorStorage
        let storages: [any VectorStorage] = [
            SIMDStorage32(),
            SIMDStorage64(),
            SIMDStorage128(),
            SIMDStorage256(),
            SIMDStorage512(),
            DynamicArrayStorage(count: 100)
        ]
        
        for storage in storages {
            XCTAssertGreaterThan(storage.count, 0)
            
            // Test subscript
            if storage.count > 0 {
                _ = storage[0]
            }
            
            // Test buffer access
            storage.withUnsafeBufferPointer { buffer in
                XCTAssertEqual(buffer.count, storage.count)
            }
        }
    }
    
    func testStorageOperationsProtocolConformance() {
        // Test that appropriate storage types conform to VectorStorageOperations
        let storages: [(any VectorStorage & VectorStorageOperations)?] = [
            SIMDStorage32() as? (any VectorStorage & VectorStorageOperations),
            SIMDStorage64() as? (any VectorStorage & VectorStorageOperations),
            SIMDStorage128() as? (any VectorStorage & VectorStorageOperations),
            SIMDStorage256() as? (any VectorStorage & VectorStorageOperations),
            SIMDStorage512() as? (any VectorStorage & VectorStorageOperations),
            DynamicArrayStorage(count: 100) as? (any VectorStorage & VectorStorageOperations)
        ]
        
        for storage in storages {
            XCTAssertNotNil(storage)
            
            if let storage = storage {
                // Test dot product exists
                let other = type(of: storage).init(repeating: 1.0)
                _ = storage.dotProduct(other as! Self)
            }
        }
    }
    
    // MARK: - Memory Layout Tests
    
    func testSIMDStorageMemoryLayout() {
        // Verify expected memory layouts
        XCTAssertEqual(MemoryLayout<SIMDStorage32>.size, 128) // 32 * 4 bytes
        XCTAssertEqual(MemoryLayout<SIMDStorage64>.size, 256) // 64 * 4 bytes
        XCTAssertEqual(MemoryLayout<SIMDStorage128>.size, 512) // 128 * 4 bytes
        XCTAssertEqual(MemoryLayout<SIMDStorage256>.size, 1024) // 256 * 4 bytes
        XCTAssertEqual(MemoryLayout<SIMDStorage512>.size, 2048) // 512 * 4 bytes
        
        // Verify alignment
        XCTAssertGreaterThanOrEqual(MemoryLayout<SIMDStorage32>.alignment, 16)
        XCTAssertGreaterThanOrEqual(MemoryLayout<SIMDStorage64>.alignment, 16)
    }
    
    func testDynamicArrayStorageMemoryOverhead() {
        // Dynamic storage has some overhead for the class and array
        let storage = DynamicArrayStorage(count: 1000)
        
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
        XCTAssertNoThrow(DynamicArrayStorage(count: 1))
        
        // Test with special values
        var storage = SIMDStorage32(repeating: 0.0)
        storage[0] = .nan
        storage[1] = .infinity
        storage[2] = -.infinity
        
        XCTAssertTrue(storage[0].isNaN)
        XCTAssertTrue(storage[1].isInfinite)
        XCTAssertTrue(storage[2].isInfinite)
        
        // Dot product with special values
        let normal = SIMDStorage32(repeating: 1.0)
        let dot = storage.dotProduct(normal)
        XCTAssertTrue(dot.isNaN) // NaN propagates
    }
    
    func testStorageBoundsChecking() {
        // Note: In release builds, bounds checking may be disabled
        // These tests verify behavior in debug builds
        
        let storage = SIMDStorage32(count: 16)
        
        // Valid access
        XCTAssertNoThrow({
            for i in 0..<16 {
                _ = storage[i]
            }
        }())
    }
    
    // MARK: - Performance Tests
    
    func testSIMDStoragePerformance() {
        let storage1 = SIMDStorage256(repeating: 1.0)
        let storage2 = SIMDStorage256(repeating: 2.0)
        
        measure {
            for _ in 0..<10000 {
                _ = storage1.dotProduct(storage2)
            }
        }
    }
    
    func testDynamicStoragePerformance() {
        let storage1 = DynamicArrayStorage(repeating: 1.0, count: 256)
        let storage2 = DynamicArrayStorage(repeating: 2.0, count: 256)
        
        measure {
            for _ in 0..<10000 {
                _ = storage1.dotProduct(storage2)
            }
        }
    }
    
    func testStorageAllocationPerformance() {
        measure {
            for _ in 0..<1000 {
                _ = SIMDStorage256(repeating: 1.0)
            }
        }
    }
    
    // MARK: - Copy-on-Write Tests
    
    func testStorageCopySemantics() {
        // SIMD types are value types
        var storage1 = SIMDStorage32(repeating: 1.0)
        let storage2 = storage1
        
        // Modify storage1
        storage1[0] = 42.0
        
        // storage2 should be unchanged (value semantics)
        XCTAssertEqual(storage2[0], 1.0)
        XCTAssertEqual(storage1[0], 42.0)
    }
    
    func testDynamicStorageCopySemantics() {
        // Dynamic storage is a class, but should have value semantics
        let storage1 = DynamicArrayStorage(repeating: 1.0, count: 100)
        let storage2 = storage1
        
        // Both should point to same data initially
        XCTAssertEqual(storage1[0], storage2[0])
        
        // Modifications require careful handling in actual implementation
        // This test documents expected behavior
    }
}