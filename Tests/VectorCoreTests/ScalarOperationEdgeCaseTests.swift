import XCTest
@testable import VectorCore

final class ScalarOperationEdgeCaseTests: XCTestCase {
    
    func testSpecialFloatingPointValues() {
        let v = Vector<Dim128>([1, 2, 3, 4] + Array(repeating: 0, count: 124))
        
        // Test NaN
        let vNaN = v * Float.nan
        XCTAssertTrue(vNaN[0].isNaN)
        XCTAssertTrue(vNaN[1].isNaN)
        
        // Test infinity
        let vInf = v * Float.infinity
        XCTAssertTrue(vInf[0].isInfinite)
        XCTAssertEqual(vInf[0], Float.infinity)
        XCTAssertEqual(vInf[1], Float.infinity * 2)
        
        // Test -infinity
        let vNegInf = v * -Float.infinity
        XCTAssertTrue(vNegInf[0].isInfinite)
        XCTAssertEqual(vNegInf[0], -Float.infinity)
        
        // Test division by zero
        let vDivZero = v / 0
        XCTAssertTrue(vDivZero[0].isInfinite)
        
        // Test division by infinity
        let vDivInf = v / Float.infinity
        XCTAssertEqual(vDivInf[0], 0)
        XCTAssertEqual(vDivInf[1], 0)
    }
    
    func testNegativeZero() {
        let v = Vector<Dim32>(repeating: 0)
        
        // Multiplication by -1 should preserve negative zero
        let vNeg = v * -1
        XCTAssertEqual(vNeg[0], -0.0)
        XCTAssertTrue(vNeg[0].sign == .minus)
        
        // Using unary minus should also work
        let vNeg2 = -v
        XCTAssertEqual(vNeg2[0], -0.0)
        XCTAssertTrue(vNeg2[0].sign == .minus)
    }
    
    func testMutatingOperatorsSideEffects() {
        // Test that *= 0 properly sets to zero vector
        var v1 = Vector<Dim64>.random(in: 1...10)
        v1 *= 0
        XCTAssertEqual(v1.magnitude, 0)
        for i in 0..<64 {
            XCTAssertEqual(v1[i], 0)
        }
        
        // Test that *= 1 is truly no-op
        var v2 = Vector<Dim64>.random(in: 1...10)
        let original = v2
        v2 *= 1
        XCTAssertEqual(v2, original)
        
        // Test that /= 1 is truly no-op
        var v3 = Vector<Dim64>.random(in: 1...10)
        let original3 = v3
        v3 /= 1
        XCTAssertEqual(v3, original3)
    }
    
    func testScalarOperationConsistency() {
        let v = Vector<Dim128>.random(in: -5...5)
        
        // Test that (v * a) * b == v * (a * b)
        let a: Float = 2.5
        let b: Float = 3.7
        let v1 = (v * a) * b
        let v2 = v * (a * b)
        XCTAssertTrue(v1.isApproximatelyEqual(to: v2, tolerance: 1e-5))
        
        // Test that v * 0.5 == v / 2
        let vHalf1 = v * 0.5
        let vHalf2 = v / 2
        XCTAssertTrue(vHalf1.isApproximatelyEqual(to: vHalf2, tolerance: 1e-6))
        
        // Test commutativity: a * v == v * a
        let scalar: Float = 3.14
        let v3 = scalar * v
        let v4 = v * scalar
        XCTAssertEqual(v3, v4)
    }
    
    func testFastPathCorrectness() {
        let v = Vector<Dim256>.random(in: -10...10)
        
        // Verify multiply by 0 returns actual zero vector
        let v0 = v * 0
        XCTAssertEqual(v0, Vector<Dim256>())
        
        // Verify multiply by 1 returns exact copy
        let v1 = v * 1
        XCTAssertEqual(v1, v)
        
        // Verify multiply by -1 equals unary minus
        let vNeg1 = v * -1
        let vNeg2 = -v
        XCTAssertEqual(vNeg1, vNeg2)
        
        // Verify divide by 1 returns exact copy
        let vDiv1 = v / 1
        XCTAssertEqual(vDiv1, v)
    }
    
    func testPerformanceDoesNotDegrade() {
        // Test that non-fast-path operations still perform well
        let v = Vector<Dim512>.random(in: -1...1)
        let iterations = 1000
        
        // Measure a non-fast-path scalar
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            _ = v * 2.718281828  // e
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        
        // Should still be reasonably fast (less than 10ms for 1000 iterations)
        XCTAssertLessThan(elapsed, 0.01, "Non-fast-path scalar multiplication is too slow")
    }
}