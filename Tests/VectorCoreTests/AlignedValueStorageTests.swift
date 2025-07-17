// VectorCore: Aligned Value Storage Tests
//
// Test value semantics and alignment for AlignedValueStorage
//

import XCTest
@testable import VectorCore

final class AlignedValueStorageTests: XCTestCase {
    
    func testInitialization() {
        // Test zero initialization
        let storage1 = AlignedValueStorage(count: 100)
        XCTAssertEqual(storage1.count, 100)
        for i in 0..<100 {
            XCTAssertEqual(storage1[i], 0.0)
        }
        
        // Test repeating value
        let storage2 = AlignedValueStorage(count: 50, repeating: 3.14)
        XCTAssertEqual(storage2.count, 50)
        for i in 0..<50 {
            XCTAssertEqual(storage2[i], 3.14)
        }
        
        // Test from array
        let values = (0..<75).map { Float($0) }
        let storage3 = AlignedValueStorage(count: 75, from: values)
        XCTAssertEqual(storage3.count, 75)
        for i in 0..<75 {
            XCTAssertEqual(storage3[i], Float(i))
        }
    }
    
    func testAlignment() {
        // Test default 64-byte alignment
        let storage = AlignedValueStorage(count: 512)
        storage.withUnsafeBufferPointer { buffer in
            let address = UInt(bitPattern: buffer.baseAddress!)
            XCTAssertEqual(address % 64, 0, "Storage should be 64-byte aligned")
        }
        
        // Test custom alignment
        let storage16 = AlignedValueStorage(count: 128, alignment: 16)
        storage16.withUnsafeBufferPointer { buffer in
            let address = UInt(bitPattern: buffer.baseAddress!)
            XCTAssertEqual(address % 16, 0, "Storage should be 16-byte aligned")
        }
    }
    
    func testValueSemantics() {
        // Test Copy-on-Write behavior
        var storage1 = AlignedValueStorage(count: 100, repeating: 1.0)
        let storage2 = storage1
        
        // Before mutation, they should share storage
        XCTAssertEqual(storage1[0], 1.0)
        XCTAssertEqual(storage2[0], 1.0)
        
        // Mutate storage1
        storage1[0] = 42.0
        
        // storage1 should be modified, storage2 should be unchanged
        XCTAssertEqual(storage1[0], 42.0)
        XCTAssertEqual(storage2[0], 1.0, "Copy should maintain original value")
    }
    
    func testCOWOptimization() {
        // Test that COW only triggers on actual mutation
        var storage = AlignedValueStorage(count: 1000, repeating: 1.0)
        
        // Reading should not trigger COW
        _ = storage[0]
        _ = storage.withUnsafeBufferPointer { buffer in
            buffer[0]
        }
        
        // Only mutation should trigger COW
        let copy = storage
        storage[0] = 2.0  // This should trigger COW
        
        XCTAssertEqual(storage[0], 2.0)
        XCTAssertEqual(copy[0], 1.0)
    }
    
    func testMutableBufferPointer() {
        var storage = AlignedValueStorage(count: 100, repeating: 1.0)
        let copy = storage
        
        // Mutate through buffer pointer
        storage.withUnsafeMutableBufferPointer { buffer in
            for i in 0..<100 {
                buffer[i] = Float(i)
            }
        }
        
        // Verify mutation
        for i in 0..<100 {
            XCTAssertEqual(storage[i], Float(i))
        }
        
        // Verify copy is unchanged
        for i in 0..<100 {
            XCTAssertEqual(copy[i], 1.0)
        }
    }
    
    func testDotProduct() {
        let values1 = [Float](repeating: 1.0, count: 512)
        let values2 = [Float](repeating: 2.0, count: 512)
        
        let storage1 = AlignedValueStorage(count: 512, from: values1)
        let storage2 = AlignedValueStorage(count: 512, from: values2)
        
        let result = storage1.dotProduct(storage2)
        XCTAssertEqual(result, 1024.0) // 1.0 * 2.0 * 512
    }
    
    func testFactoryMethods() {
        let storage512 = AlignedValueStorage.storage512()
        XCTAssertEqual(storage512.count, 512)
        
        let storage768 = AlignedValueStorage.storage768()
        XCTAssertEqual(storage768.count, 768)
        
        let storage1536 = AlignedValueStorage.storage1536()
        XCTAssertEqual(storage1536.count, 1536)
        
        let storage3072 = AlignedValueStorage.storage3072()
        XCTAssertEqual(storage3072.count, 3072)
    }
    
    func testThreadSafety() {
        // Test that COW makes concurrent reads safe
        let storage = AlignedValueStorage(count: 1000, repeating: 1.0)
        let queue = DispatchQueue(label: "test", attributes: .concurrent)
        let group = DispatchGroup()
        
        // Many concurrent reads should be safe
        for _ in 0..<100 {
            group.enter()
            queue.async {
                _ = storage.withUnsafeBufferPointer { buffer in
                    buffer.reduce(0, +)
                }
                group.leave()
            }
        }
        
        group.wait()
        
        // Verify storage is unchanged
        for i in 0..<1000 {
            XCTAssertEqual(storage[i], 1.0)
        }
    }
}