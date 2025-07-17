import XCTest
@testable import VectorCore

final class ScalarOperationCOWTests: XCTestCase {
    
    func testMultiplyByOneMaintainsCOW() {
        let v1 = Vector<Dim128>.random(in: -1...1)
        let v2 = v1 * 1  // Should share storage via COW
        
        // Both vectors should be equal
        XCTAssertEqual(v1, v2)
        
        // Modifying v1 should not affect v2
        var v1Copy = v1
        v1Copy[0] = 999
        XCTAssertNotEqual(v1Copy, v2)
        XCTAssertEqual(v2, v1)  // v2 should still equal original v1
    }
    
    func testDivideByOneMaintainsCOW() {
        let v1 = Vector<Dim128>.random(in: -1...1)
        let v2 = v1 / 1  // Should share storage via COW
        
        // Both vectors should be equal
        XCTAssertEqual(v1, v2)
        
        // Modifying v1 should not affect v2
        var v1Copy = v1
        v1Copy[0] = 999
        XCTAssertNotEqual(v1Copy, v2)
        XCTAssertEqual(v2, v1)  // v2 should still equal original v1
    }
    
    func testMultiplyByZeroCreatesNewVector() {
        let v1 = Vector<Dim128>.random(in: 1...10)
        let v2 = v1 * 0  // Should create new zero vector
        
        // v2 should be zero
        XCTAssertEqual(v2.magnitude, 0)
        
        // v1 should be unchanged
        XCTAssertGreaterThan(v1.magnitude, 0)
    }
    
    func testMutatingOperatorsCOWBehavior() {
        // Test that *= 1 doesn't trigger unnecessary COW
        let original = Vector<Dim256>.random(in: -1...1)
        var v1 = original
        var v2 = original  // v1 and v2 share storage
        
        // This should be a no-op and not trigger COW for v1
        v1 *= 1
        XCTAssertEqual(v1, original)
        
        // Modifying v2 should not affect v1
        v2[0] = 999
        XCTAssertEqual(v1, original)
        XCTAssertNotEqual(v2, original)
    }
    
    func testChainedOperations() {
        let v = Vector<Dim128>.random(in: -1...1)
        
        // Chain of identity operations should be efficient
        let result = v * 1 / 1 * 1 / 1
        XCTAssertEqual(result, v)
        
        // Mixed operations
        let result2 = v * 2 / 2 * 1
        XCTAssertTrue(result2.isApproximatelyEqual(to: v, tolerance: 1e-6))
    }
}