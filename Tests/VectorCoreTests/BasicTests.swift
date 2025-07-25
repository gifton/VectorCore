import XCTest
@testable import VectorCore

final class BasicTests: XCTestCase {
    func testVectorCreation() {
        let v = Vector768.zeros()
        XCTAssertEqual(v.scalarCount, 768)
        XCTAssertEqual(v.magnitude, 0, accuracy: 1e-7)
    }
    
    func testVectorAddition() {
        let v1 = Vector768(repeating: 1.0)
        let v2 = Vector768(repeating: 2.0)
        let v3 = v1 + v2
        
        for i in 0..<v3.scalarCount {
            XCTAssertEqual(v3[i], 3.0, accuracy: 1e-7)
        }
    }
    
    func testPropertyTestsCompile() {
        // Just verify PropertyTest types are available
        let config = PropertyTest.Config.default
        XCTAssertEqual(config.iterations, 100)
    }
}