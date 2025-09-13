import XCTest
@testable import VectorCore

/// Minimal baseline test suite to verify core VectorCore functionality
final class BaselineTests: XCTestCase {

    // MARK: - Basic Vector Creation

    func testBasicVectorCreation() {
        // Test zero initialization
        let zeroVector = Vector<Dim32>()
        XCTAssertEqual(zeroVector.scalarCount, 32)

        // Test repeating value initialization
        let onesVector = Vector<Dim32>(repeating: 1.0)
        XCTAssertEqual(onesVector.scalarCount, 32)
    }

    func testVectorSubscript() {
        var vector = Vector<Dim32>(repeating: 0.0)

        // Test setting values
        vector[0] = 1.0
        vector[31] = 2.0

        // Test getting values
        XCTAssertEqual(vector[0], 1.0)
        XCTAssertEqual(vector[31], 2.0)
    }

    func testStaticProperties() {
        // Test zero vector
        let zero = Vector<Dim32>.zero
        XCTAssertEqual(zero[0], 0.0)

        // Test ones vector - using property not function call
        let ones = Vector<Dim32>.ones
        XCTAssertEqual(ones[0], 1.0)
    }

    // MARK: - Basic Operations

    func testVectorAddition() {
        let v1 = Vector<Dim32>(repeating: 1.0)
        let v2 = Vector<Dim32>(repeating: 2.0)

        let result = v1 + v2
        XCTAssertEqual(result[0], 3.0)
    }

    func testVectorSubtraction() {
        let v1 = Vector<Dim32>(repeating: 5.0)
        let v2 = Vector<Dim32>(repeating: 2.0)

        let result = v1 - v2
        XCTAssertEqual(result[0], 3.0)
    }

    func testScalarMultiplication() {
        let vector = Vector<Dim32>(repeating: 2.0)
        let result = vector * 3.0
        XCTAssertEqual(result[0], 6.0)
    }

    func testScalarDivision() {
        let vector = Vector<Dim32>(repeating: 6.0)
        let result = vector / 2.0
        XCTAssertEqual(result[0], 3.0)
    }

    // MARK: - Collection Conformance

    func testCollectionBasics() {
        let vector = Vector<Dim32>(repeating: 1.5)

        // Test count
        XCTAssertEqual(vector.count, 32)

        // Test indices
        XCTAssertEqual(vector.startIndex, 0)
        XCTAssertEqual(vector.endIndex, 32)
    }

    // MARK: - Equality

    func testEquality() {
        let v1 = Vector<Dim32>(repeating: 1.0)
        let v2 = Vector<Dim32>(repeating: 1.0)
        let v3 = Vector<Dim32>(repeating: 2.0)

        XCTAssertEqual(v1, v2)
        XCTAssertNotEqual(v1, v3)
    }
}
