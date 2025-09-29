//
//  ContiguousArrayVectorTests.swift
//  VectorCore
//
//  Tests for ContiguousArray vector operations with vDSP
//

import XCTest
@testable import VectorCore

final class ContiguousArrayVectorTests: XCTestCase {

    // MARK: - Test Data

    func generateTestVector(dimension: Int, seed: Int = 42) -> ContiguousArray<Float> {
        var values = ContiguousArray<Float>()
        values.reserveCapacity(dimension)
        for i in 0..<dimension {
            values.append(Float(i + seed) / Float(dimension))
        }
        return values
    }

    // MARK: - Basic Operations Tests

    func testElementWiseMultiplication() {
        let a = ContiguousArray<Float>([1.0, 2.0, 3.0, 4.0])
        let b = ContiguousArray<Float>([2.0, 3.0, 4.0, 5.0])

        // Test using operator
        let result1 = a * b
        XCTAssertEqual(result1[0], 2.0, accuracy: 0.001)
        XCTAssertEqual(result1[1], 6.0, accuracy: 0.001)
        XCTAssertEqual(result1[2], 12.0, accuracy: 0.001)
        XCTAssertEqual(result1[3], 20.0, accuracy: 0.001)

        // Test using method
        let result2 = a.elementWiseMultiply(b)
        XCTAssertEqual(result1, result2)
    }

    func testSum() {
        let vector = ContiguousArray<Float>([1.0, 2.0, 3.0, 4.0, 5.0])
        let sum = vector.sum()
        XCTAssertEqual(sum, 15.0, accuracy: 0.001)
    }

    func testAbs() {
        let vector = ContiguousArray<Float>([-1.0, 2.0, -3.0, 4.0, -5.0])
        let absVector = vector.abs()

        XCTAssertEqual(absVector[0], 1.0, accuracy: 0.001)
        XCTAssertEqual(absVector[1], 2.0, accuracy: 0.001)
        XCTAssertEqual(absVector[2], 3.0, accuracy: 0.001)
        XCTAssertEqual(absVector[3], 4.0, accuracy: 0.001)
        XCTAssertEqual(absVector[4], 5.0, accuracy: 0.001)
    }

    func testAddition() {
        let a = ContiguousArray<Float>([1.0, 2.0, 3.0])
        let b = ContiguousArray<Float>([4.0, 5.0, 6.0])

        let result = a + b
        XCTAssertEqual(result[0], 5.0, accuracy: 0.001)
        XCTAssertEqual(result[1], 7.0, accuracy: 0.001)
        XCTAssertEqual(result[2], 9.0, accuracy: 0.001)
    }

    func testSubtraction() {
        let a = ContiguousArray<Float>([5.0, 7.0, 9.0])
        let b = ContiguousArray<Float>([1.0, 2.0, 3.0])

        let result = a - b
        XCTAssertEqual(result[0], 4.0, accuracy: 0.001)
        XCTAssertEqual(result[1], 5.0, accuracy: 0.001)
        XCTAssertEqual(result[2], 6.0, accuracy: 0.001)
    }

    // MARK: - GraphVector Protocol Tests

    func testDotProduct() {
        let a = ContiguousArray<Float>([1.0, 2.0, 3.0])
        let b = ContiguousArray<Float>([4.0, 5.0, 6.0])

        let dot = a.dotProduct(b)
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        XCTAssertEqual(dot, 32.0, accuracy: 0.001)
    }

    func testMagnitude() {
        let vector = ContiguousArray<Float>([3.0, 4.0, 0.0])
        let mag = vector.magnitude
        // sqrt(9 + 16 + 0) = sqrt(25) = 5
        XCTAssertEqual(mag, 5.0, accuracy: 0.001)
    }

    // MARK: - Complex Operations Tests

    func testVectorNormalization() {
        let vector = ContiguousArray<Float>([3.0, 4.0, 0.0])
        let mag = vector.magnitude

        // Normalize manually using our operations
        let normalized = vector.map { $0 / mag }

        // Check that normalized vector has magnitude 1
        let normalizedArray = ContiguousArray(normalized)
        let newMag = normalizedArray.magnitude
        XCTAssertEqual(newMag, 1.0, accuracy: 0.001)
    }

    func testComplexComputation() {
        // Test a complex computation: ||a * b + c||
        let a = ContiguousArray<Float>([1.0, 2.0, 3.0])
        let b = ContiguousArray<Float>([2.0, 3.0, 4.0])
        let c = ContiguousArray<Float>([1.0, 1.0, 1.0])

        let product = a * b  // [2.0, 6.0, 12.0]
        let sum = product + c  // [3.0, 7.0, 13.0]
        let mag = sum.magnitude  // sqrt(9 + 49 + 169) = sqrt(227) ≈ 15.066

        XCTAssertEqual(mag, 15.066, accuracy: 0.01)
    }

    // MARK: - Performance Tests

    func testLargeVectorPerformance() {
        let dimension = 10000
        let a = generateTestVector(dimension: dimension, seed: 1)
        let b = generateTestVector(dimension: dimension, seed: 2)

        measure {
            _ = a * b
            _ = a + b
            _ = a - b
            _ = a.sum()
            _ = a.abs()
        }
    }

    func testOperatorVsMethodPerformance() {
        let dimension = 5000
        let a = generateTestVector(dimension: dimension, seed: 1)
        let b = generateTestVector(dimension: dimension, seed: 2)

        // Measure operator performance
        let operatorTime = measureTime {
            for _ in 0..<100 {
                _ = a * b
            }
        }

        // Measure method performance
        let methodTime = measureTime {
            for _ in 0..<100 {
                _ = a.elementWiseMultiply(b)
            }
        }

        // They should be approximately equal since operator calls the method
        print("Operator time: \(operatorTime), Method time: \(methodTime)")
        XCTAssertEqual(operatorTime, methodTime, accuracy: operatorTime * 0.2) // Allow 20% variance
    }

    private func measureTime(block: () -> Void) -> TimeInterval {
        let start = CFAbsoluteTimeGetCurrent()
        block()
        return CFAbsoluteTimeGetCurrent() - start
    }

    // MARK: - Edge Cases

    func testEmptyVectors() {
        let empty = ContiguousArray<Float>()
        XCTAssertEqual(empty.sum(), 0.0)
        XCTAssertEqual(empty.magnitude, 0.0)
        XCTAssertEqual(empty.abs(), ContiguousArray<Float>())
    }

    func testSingleElementVector() {
        let single = ContiguousArray<Float>([-5.0])
        XCTAssertEqual(single.sum(), -5.0)
        XCTAssertEqual(single.abs()[0], 5.0)
        XCTAssertEqual(single.magnitude, 5.0, accuracy: 0.001)
    }

    func testDimensionMismatch() {
        let a = ContiguousArray<Float>([1.0, 2.0])
        let b = ContiguousArray<Float>([1.0, 2.0, 3.0])

        // These should trigger precondition failures in debug mode
        // In release mode they would cause undefined behavior
        // We can't test preconditions directly in XCTest
    }
}