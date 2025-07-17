import XCTest
@testable import VectorCore

final class ScalarOperationTests: XCTestCase {
    
    func testScalarMultiplicationPerformance() {
        let iterations = 10000
        let v = Vector<Dim256>.random(in: -1...1)
        
        print("\n=== Scalar Multiplication Performance ===")
        
        // Test multiplication by 0
        let mult0Time = measureTime {
            for _ in 0..<iterations {
                _ = v * 0
            }
        }
        print("Multiply by 0: \((Double(iterations)/mult0Time/1000000).formatted()) M ops/sec")
        
        // Test multiplication by 1
        let mult1Time = measureTime {
            for _ in 0..<iterations {
                _ = v * 1
            }
        }
        print("Multiply by 1: \((Double(iterations)/mult1Time/1000000).formatted()) M ops/sec")
        
        // Test multiplication by -1
        let multNeg1Time = measureTime {
            for _ in 0..<iterations {
                _ = v * -1
            }
        }
        print("Multiply by -1: \((Double(iterations)/multNeg1Time/1000000).formatted()) M ops/sec")
        
        // Test multiplication by arbitrary scalar
        let mult2_5Time = measureTime {
            for _ in 0..<iterations {
                _ = v * 2.5
            }
        }
        print("Multiply by 2.5: \((Double(iterations)/mult2_5Time/1000000).formatted()) M ops/sec")
        
        // Test division by 1
        let div1Time = measureTime {
            for _ in 0..<iterations {
                _ = v / 1
            }
        }
        print("\nDivide by 1: \((Double(iterations)/div1Time/1000000).formatted()) M ops/sec")
        
        // Test division by arbitrary scalar
        let div2_5Time = measureTime {
            for _ in 0..<iterations {
                _ = v / 2.5
            }
        }
        print("Divide by 2.5: \((Double(iterations)/div2_5Time/1000000).formatted()) M ops/sec")
    }
    
    func testScalarOperationCorrectness() {
        let v = Vector<Dim128>([1, 2, 3, 4, 5, 6, 7, 8] + Array(repeating: 0, count: 120))
        
        // Test multiplication by 0
        let v0 = v * 0
        XCTAssertEqual(v0[0], 0)
        XCTAssertEqual(v0[1], 0)
        XCTAssertEqual(v0.magnitude, 0)
        
        // Test multiplication by 1
        let v1 = v * 1
        XCTAssertEqual(v1[0], 1)
        XCTAssertEqual(v1[1], 2)
        XCTAssertEqual(v1, v)
        
        // Test multiplication by -1
        let vNeg = v * -1
        XCTAssertEqual(vNeg[0], -1)
        XCTAssertEqual(vNeg[1], -2)
        XCTAssertEqual(vNeg, -v)
        
        // Test division by 1
        let vDiv1 = v / 1
        XCTAssertEqual(vDiv1[0], 1)
        XCTAssertEqual(vDiv1[1], 2)
        XCTAssertEqual(vDiv1, v)
        
        // Test special cases
        let vNaN = v * Float.nan
        XCTAssertTrue(vNaN[0].isNaN)
        
        let vInf = v * Float.infinity
        XCTAssertTrue(vInf[0].isInfinite)
    }
    
    private func measureTime(block: () -> Void) -> TimeInterval {
        let start = CFAbsoluteTimeGetCurrent()
        block()
        let end = CFAbsoluteTimeGetCurrent()
        return end - start
    }
}