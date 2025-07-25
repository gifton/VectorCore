// VectorCore: Alignment Verification Tests
//
// Verify memory alignment for SIMD operations
//

import XCTest
@testable import VectorCore

final class AlignmentVerificationTests: XCTestCase {
    
    func testConsolidatedStorageAlignment() {
        // Test SmallVectorStorage (uses SIMD64 internally)
        let small32 = SmallVectorStorage(count: 32)
        small32.withUnsafeBufferPointer { buffer in
            let address = UInt(bitPattern: buffer.baseAddress!)
            XCTAssertEqual(address % 16, 0, "SmallVectorStorage should be 16-byte aligned")
        }
        
        let small64 = SmallVectorStorage(count: 64)
        small64.withUnsafeBufferPointer { buffer in
            let address = UInt(bitPattern: buffer.baseAddress!)
            XCTAssertEqual(address % 16, 0, "SmallVectorStorage should be 16-byte aligned")
        }
        
        // Test MediumVectorStorage (uses AlignedValueStorage)
        let medium128 = MediumVectorStorage(count: 128)
        medium128.withUnsafeBufferPointer { buffer in
            let address = UInt(bitPattern: buffer.baseAddress!)
            XCTAssertEqual(address % 64, 0, "MediumVectorStorage should be 64-byte aligned")
        }
        
        let medium512 = MediumVectorStorage(count: 512)
        medium512.withUnsafeBufferPointer { buffer in
            let address = UInt(bitPattern: buffer.baseAddress!)
            XCTAssertEqual(address % 64, 0, "MediumVectorStorage should be 64-byte aligned")
        }
    }
    
    func testLargeStorageAlignment() {
        // Test LargeVectorStorage (uses COWDynamicStorage)
        let large768 = LargeVectorStorage(count: 768)
        large768.withUnsafeBufferPointer { buffer in
            let address = UInt(bitPattern: buffer.baseAddress!)
            // Dynamic storage doesn't guarantee specific alignment beyond platform requirements
            XCTAssertEqual(address % UInt(MemoryLayout<Float>.alignment), 0, "LargeVectorStorage should be properly aligned")
        }
        
        let large3072 = LargeVectorStorage(count: 3072)
        large3072.withUnsafeBufferPointer { buffer in
            let address = UInt(bitPattern: buffer.baseAddress!)
            XCTAssertEqual(address % UInt(MemoryLayout<Float>.alignment), 0, "LargeVectorStorage should be properly aligned")
        }
    }
    
    func testValueSemanticsBehavior() {
        // Test value semantics for all consolidated storage types
        var storage1 = MediumVectorStorage(count: 512, repeating: 1.0)
        let storage2 = storage1
        
        // Modify storage1
        storage1[0] = 42.0
        
        // With value semantics, storage2[0] should still be 1.0
        XCTAssertEqual(storage2[0], 1.0, "MediumVectorStorage should have value semantics")
        XCTAssertEqual(storage1[0], 42.0, "Modified storage should have new value")
    }
    
    func testDynamicVectorValueSemantics() {
        // Test current DynamicVector behavior
        let original = DynamicVector([1, 2, 3, 4, 5])
        var copy = original
        
        // Modify copy
        copy[0] = 99.0
        
        // With proper value semantics, original should be unchanged
        XCTAssertEqual(original[0], 1.0, "DynamicVector should have value semantics")
        XCTAssertEqual(copy[0], 99.0, "Modified copy should have new value")
    }
}