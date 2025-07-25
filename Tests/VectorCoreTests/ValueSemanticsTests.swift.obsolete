// VectorCore: Value Semantics Tests
//
// Verify all storage types have proper value semantics
//

import XCTest
@testable import VectorCore

final class ValueSemanticsTests: XCTestCase {
    
    func testMediumStorageValueSemantics() {
        var storage1 = MediumVectorStorage(count: 512, repeating: 1.0)
        let storage2 = storage1
        
        // Modify storage1
        storage1[0] = 42.0
        
        // Verify value semantics
        XCTAssertEqual(storage1[0], 42.0)
        XCTAssertEqual(storage2[0], 1.0, "Copy should maintain original value")
    }
    
    func testLargeStorageValueSemantics() {
        var storage1 = LargeVectorStorage(count: 768, repeating: 2.0)
        let storage2 = storage1
        
        // Modify storage1
        storage1[100] = 99.0
        
        // Verify value semantics
        XCTAssertEqual(storage1[100], 99.0)
        XCTAssertEqual(storage2[100], 2.0, "Copy should maintain original value")
    }
    
    func testExtraLargeStorageValueSemantics() {
        var storage1 = LargeVectorStorage(count: 1536, repeating: 3.0)
        let storage2 = storage1
        
        // Modify storage1
        storage1[1000] = 123.0
        
        // Verify value semantics
        XCTAssertEqual(storage1[1000], 123.0)
        XCTAssertEqual(storage2[1000], 3.0, "Copy should maintain original value")
    }
    
    func testVeryLargeStorageValueSemantics() {
        var storage1 = LargeVectorStorage(count: 3072, repeating: 4.0)
        let storage2 = storage1
        
        // Modify storage1
        storage1[2000] = 456.0
        
        // Verify value semantics
        XCTAssertEqual(storage1[2000], 456.0)
        XCTAssertEqual(storage2[2000], 4.0, "Copy should maintain original value")
    }
    
    func testMutableBufferValueSemantics() {
        // Test that mutable buffer operations trigger COW
        var storage1 = MediumVectorStorage(count: 512, repeating: 5.0)
        let storage2 = storage1
        
        // Modify through mutable buffer
        storage1.withUnsafeMutableBufferPointer { buffer in
            buffer[10] = 88.0
        }
        
        // Verify COW triggered
        XCTAssertEqual(storage1[10], 88.0)
        XCTAssertEqual(storage2[10], 5.0, "Copy should be unaffected by buffer mutation")
    }
    
    func testDotProductDoesNotAffectCopies() {
        let storage1 = MediumVectorStorage(count: 512, repeating: 2.0)
        let storage2 = storage1
        let storage3 = MediumVectorStorage(count: 512, repeating: 3.0)
        
        // Compute dot product
        let result = storage1.dotProduct(storage3)
        XCTAssertEqual(result, 2.0 * 3.0 * 512) // 3072
        
        // Verify neither storage was modified
        XCTAssertEqual(storage1[0], 2.0)
        XCTAssertEqual(storage2[0], 2.0)
        XCTAssertEqual(storage3[0], 3.0)
    }
    
    func testMultipleCopiesIndependent() {
        var original = LargeVectorStorage(count: 768, repeating: 10.0)
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
    
    func testInitFromArrayValueSemantics() {
        let values = (0..<1536).map { Float($0) }
        var storage1 = LargeVectorStorage(from: values)
        let storage2 = storage1
        
        // Modify storage1
        storage1[500] = 9999.0
        
        // Verify value semantics
        XCTAssertEqual(storage1[500], 9999.0)
        XCTAssertEqual(storage2[500], 500.0, "Copy should maintain original value")
    }
    
    // Performance test to ensure COW doesn't degrade performance
    func testCOWPerformance() {
        let storage1 = LargeVectorStorage(count: 3072, repeating: 1.0)
        
        measure {
            // Many copies without mutation should be fast
            for _ in 0..<1000 {
                let _ = storage1
            }
        }
    }
    
    func testMutationPerformance() {
        measure {
            var storage = LargeVectorStorage(count: 3072, repeating: 1.0)
            // Single mutation should trigger copy
            for i in 0..<100 {
                storage[i] = Float(i)
            }
        }
    }
}