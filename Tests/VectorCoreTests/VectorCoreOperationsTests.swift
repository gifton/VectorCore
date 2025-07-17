// VectorCore: Core Operations Tests
//
// Tests for element-wise operations, clamping, and interpolation
//

import XCTest
import Foundation
@testable import VectorCore

final class VectorCoreOperationsTests: XCTestCase {
    
    // MARK: - Element-wise Min/Max Tests
    
    func testElementWiseMin() {
        let v1 = Vector128([1, 5, 3, 7] + [Float](repeating: 0, count: 124))
        let v2 = Vector128([2, 4, 6, 1] + [Float](repeating: 0, count: 124))
        
        let result = v1.min(v2)
        
        XCTAssertEqual(result[0], 1)
        XCTAssertEqual(result[1], 4)
        XCTAssertEqual(result[2], 3)
        XCTAssertEqual(result[3], 1)
    }
    
    func testElementWiseMax() {
        let v1 = Vector128([1, 5, 3, 7] + [Float](repeating: 0, count: 124))
        let v2 = Vector128([2, 4, 6, 1] + [Float](repeating: 0, count: 124))
        
        let result = v1.max(v2)
        
        XCTAssertEqual(result[0], 2)
        XCTAssertEqual(result[1], 5)
        XCTAssertEqual(result[2], 6)
        XCTAssertEqual(result[3], 7)
    }
    
    func testMinMaxOperators() {
        let v1 = Vector256.random(in: -10...10)
        let v2 = Vector256.random(in: -10...10)
        
        let minResult = v1 .< v2
        let maxResult = v1 .> v2
        
        // Verify operations
        for i in 0..<256 {
            XCTAssertEqual(minResult[i], Swift.min(v1[i], v2[i]))
            XCTAssertEqual(maxResult[i], Swift.max(v1[i], v2[i]))
        }
    }
    
    func testMinMaxElement() {
        let values: [Float] = [3, 1, 4, 1, 5, 9, 2, 6] + [Float](repeating: 0, count: 24)
        let v = Vector<Dim32>(values)
        
        let (minVal, minIdx) = v.minElement()
        let (maxVal, maxIdx) = v.maxElement()
        
        // The zeros at the end are the minimum values
        XCTAssertEqual(minVal, 0)
        XCTAssertEqual(minIdx, 8) // First zero is at index 8
        XCTAssertEqual(maxVal, 9)
        XCTAssertEqual(maxIdx, 5)
    }
    
    // MARK: - Clamp Tests
    
    func testClamp() {
        let v = Vector128([Float](repeating: 0, count: 128).enumerated().map { Float($0.offset - 64) })
        let clamped = v.clamped(to: -10...10)
        
        // Check boundaries
        XCTAssertEqual(clamped[0], -10)  // Was -64
        XCTAssertEqual(clamped[54], -10) // Was -10
        XCTAssertEqual(clamped[64], 0)   // Was 0
        XCTAssertEqual(clamped[74], 10)  // Was 10
        XCTAssertEqual(clamped[127], 10) // Was 63
        
        // Check middle values
        XCTAssertEqual(clamped[60], -4)  // Was -4
        XCTAssertEqual(clamped[70], 6)   // Was 6
    }
    
    func testClampInPlace() {
        var v = Vector128([Float](repeating: 0, count: 128).enumerated().map { Float($0.offset - 64) })
        v.clamp(to: -5...5)
        
        // All values should be in [-5, 5]
        for i in 0..<128 {
            XCTAssertGreaterThanOrEqual(v[i], -5)
            XCTAssertLessThanOrEqual(v[i], 5)
        }
    }
    
    func testClampEdgeCases() {
        let v = Vector<Dim32>([Float.nan, .infinity, -.infinity, 0.0] + [Float](repeating: 0, count: 28))
        let clamped = v.clamped(to: -1...1)
        
        // NaN remains NaN
        XCTAssertTrue(clamped[0].isNaN)
        // Infinity is clamped
        XCTAssertEqual(clamped[1], 1)
        XCTAssertEqual(clamped[2], -1)
        XCTAssertEqual(clamped[3], 0)
    }
    
    // MARK: - Linear Interpolation Tests
    
    func testLerp() {
        let v1 = Vector128(repeating: 0)
        let v2 = Vector128(repeating: 10)
        
        // Test various t values
        let lerp0 = v1.lerp(to: v2, t: 0)
        let lerp25 = v1.lerp(to: v2, t: 0.25)
        let lerp50 = v1.lerp(to: v2, t: 0.5)
        let lerp75 = v1.lerp(to: v2, t: 0.75)
        let lerp100 = v1.lerp(to: v2, t: 1)
        
        XCTAssertEqual(lerp0[0], 0)
        XCTAssertEqual(lerp25[0], 2.5)
        XCTAssertEqual(lerp50[0], 5)
        XCTAssertEqual(lerp75[0], 7.5)
        XCTAssertEqual(lerp100[0], 10)
    }
    
    func testLerpClamping() {
        let v1 = Vector<Dim32>([0.0, 0.0] + [Float](repeating: 0, count: 30))
        let v2 = Vector<Dim32>([10.0, 10.0] + [Float](repeating: 0, count: 30))
        
        // Test clamping
        let lerpNeg = v1.lerp(to: v2, t: -0.5)
        let lerpOver = v1.lerp(to: v2, t: 1.5)
        
        XCTAssertEqual(lerpNeg[0], 0)  // Clamped to t=0
        XCTAssertEqual(lerpOver[0], 10) // Clamped to t=1
    }
    
    func testLerpUnclamped() {
        let v1 = Vector<Dim32>([0.0, 0.0] + [Float](repeating: 0, count: 30))
        let v2 = Vector<Dim32>([10.0, 10.0] + [Float](repeating: 0, count: 30))
        
        // Test extrapolation
        let lerpNeg = v1.lerpUnclamped(to: v2, t: -0.5)
        let lerpOver = v1.lerpUnclamped(to: v2, t: 1.5)
        
        XCTAssertEqual(lerpNeg[0], -5)  // Extrapolated
        XCTAssertEqual(lerpOver[0], 15) // Extrapolated
    }
    
    func testSmoothstep() {
        let v1 = Vector<Dim32>([0.0, 0.0] + [Float](repeating: 0, count: 30))
        let v2 = Vector<Dim32>([10.0, 10.0] + [Float](repeating: 0, count: 30))
        
        // Smoothstep should ease in/out
        let smooth0 = v1.smoothstep(to: v2, t: 0)
        let smooth25 = v1.smoothstep(to: v2, t: 0.25)
        let smooth50 = v1.smoothstep(to: v2, t: 0.5)
        let smooth75 = v1.smoothstep(to: v2, t: 0.75)
        let smooth100 = v1.smoothstep(to: v2, t: 1)
        
        XCTAssertEqual(smooth0[0], 0)
        XCTAssertEqual(smooth50[0], 5) // Same as linear at midpoint
        XCTAssertEqual(smooth100[0], 10)
        
        // Smoothstep should be slower at start/end
        XCTAssertLessThan(smooth25[0], 2.5) // Less than linear
        XCTAssertGreaterThan(smooth75[0], 7.5) // Greater than linear
    }
    
    // MARK: - Additional Operations Tests
    
    func testAbsoluteValue() {
        let v = Vector<Dim32>([-1.0, 2.0, -3.0, 0.0] + [Float](repeating: 0, count: 28))
        let absV = v.absoluteValue()
        
        XCTAssertEqual(absV[0], 1)
        XCTAssertEqual(absV[1], 2)
        XCTAssertEqual(absV[2], 3)
        XCTAssertEqual(absV[3], 0)
    }
    
    func testSquareRoot() {
        let v = Vector<Dim32>([0.0, 1.0, 4.0, 9.0] + [Float](repeating: 0, count: 28))
        let sqrtV = v.squareRoot()
        
        XCTAssertEqual(sqrtV[0], 0)
        XCTAssertEqual(sqrtV[1], 1)
        XCTAssertEqual(sqrtV[2], 2)
        XCTAssertEqual(sqrtV[3], 3)
    }
    
    func testSquareRootNegative() {
        let v = Vector<Dim32>([-1.0, 4.0] + [Float](repeating: 0, count: 30))
        let sqrtV = v.squareRoot()
        
        XCTAssertTrue(sqrtV[0].isNaN)
        XCTAssertEqual(sqrtV[1], 2)
    }
    
    // MARK: - DynamicVector Tests
    
    func testDynamicVectorMinMax() throws {
        let v1 = DynamicVector(dimension: 4, values: [1, 5, 3, 7])
        let v2 = DynamicVector(dimension: 4, values: [2, 4, 6, 1])
        let v3 = DynamicVector(dimension: 3, values: [1, 2, 3])
        
        // Valid operations
        let minResult = try v1.min(v2)
        let maxResult = try v1.max(v2)
        
        XCTAssertEqual(minResult[0], 1)
        XCTAssertEqual(minResult[1], 4)
        XCTAssertEqual(maxResult[0], 2)
        XCTAssertEqual(maxResult[3], 7)
        
        // Dimension mismatch
        XCTAssertThrowsError(try v1.min(v3))
        XCTAssertThrowsError(try v1.max(v3))
    }
    
    func testDynamicVectorClamp() {
        let v = DynamicVector(dimension: 5, values: [-10, -1, 0, 1, 10])
        let clamped = v.clamped(to: -5...5)
        
        XCTAssertEqual(clamped[0], -5)
        XCTAssertEqual(clamped[1], -1)
        XCTAssertEqual(clamped[2], 0)
        XCTAssertEqual(clamped[3], 1)
        XCTAssertEqual(clamped[4], 5)
    }
    
    func testDynamicVectorLerp() throws {
        let v1 = DynamicVector(dimension: 3, values: [0, 0, 0])
        let v2 = DynamicVector(dimension: 3, values: [10, 20, 30])
        let v3 = DynamicVector(dimension: 2, values: [1, 2])
        
        // Valid interpolation
        let lerped = try v1.lerp(to: v2, t: 0.5)
        XCTAssertEqual(lerped[0], 5)
        XCTAssertEqual(lerped[1], 10)
        XCTAssertEqual(lerped[2], 15)
        
        // Dimension mismatch
        XCTAssertThrowsError(try v1.lerp(to: v3, t: 0.5))
    }
    
    // MARK: - Performance Tests
    
    func testMinMaxPerformance() {
        let v1 = Vector1536.random(in: -100...100)
        let v2 = Vector1536.random(in: -100...100)
        
        measure {
            for _ in 0..<1000 {
                _ = v1.min(v2)
                _ = v1.max(v2)
            }
        }
    }
    
    func testClampPerformance() {
        let v = Vector1536.random(in: -100...100)
        let range: ClosedRange<Float> = -10...10
        
        measure {
            for _ in 0..<1000 {
                _ = v.clamped(to: range)
            }
        }
    }
    
    func testLerpPerformance() {
        let v1 = Vector1536.random(in: -100...100)
        let v2 = Vector1536.random(in: -100...100)
        
        measure {
            for _ in 0..<1000 {
                _ = v1.lerp(to: v2, t: 0.5)
            }
        }
    }
}

