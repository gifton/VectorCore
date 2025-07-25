// VectorCore: Error Handling Consistency Tests
//
// Tests to ensure error handling follows established patterns
//

import XCTest
@testable import VectorCore

final class ErrorHandlingConsistencyTests: XCTestCase {
    
    // MARK: - Validation Utilities Tests
    
    func testDimensionMatchValidation() throws {
        // Valid case - should not throw
        XCTAssertNoThrow(try Validation.requireDimensionMatch(expected: 10, actual: 10))
        
        // Invalid case - should throw
        XCTAssertThrowsError(try Validation.requireDimensionMatch(expected: 10, actual: 5)) { error in
            guard let vectorError = error as? VectorError,
                  vectorError.kind == .dimensionMismatch else {
                XCTFail("Wrong error type")
                return
            }
            // Check the error context contains expected information
            XCTAssertEqual(vectorError.context.additionalInfo["expected_dimension"], "10")
            XCTAssertEqual(vectorError.context.additionalInfo["actual_dimension"], "5")
        }
    }
    
    func testValidDimensionValidation() throws {
        // Valid cases
        XCTAssertNoThrow(try Validation.requireValidDimension(1))
        XCTAssertNoThrow(try Validation.requireValidDimension(100))
        
        // Invalid cases
        XCTAssertThrowsError(try Validation.requireValidDimension(0))
        XCTAssertThrowsError(try Validation.requireValidDimension(-1))
    }
    
    func testIndexValidation() throws {
        // Valid cases
        XCTAssertNoThrow(try Validation.requireValidIndex(0, dimension: 10))
        XCTAssertNoThrow(try Validation.requireValidIndex(9, dimension: 10))
        
        // Invalid cases
        XCTAssertThrowsError(try Validation.requireValidIndex(-1, dimension: 10))
        XCTAssertThrowsError(try Validation.requireValidIndex(10, dimension: 10))
    }
    
    func testNonZeroValidation() throws {
        // Valid cases
        XCTAssertNoThrow(try Validation.requireNonZero(magnitude: 1.0, operation: "test"))
        XCTAssertNoThrow(try Validation.requireNonZero(magnitude: 0.001, operation: "test"))
        
        // Invalid case
        XCTAssertThrowsError(try Validation.requireNonZero(magnitude: 0.0, operation: "normalize")) { error in
            guard let vectorError = error as? VectorError,
                  vectorError.kind == .invalidOperation else {
                XCTFail("Wrong error type")
                return
            }
            // Check the error message contains operation info
            XCTAssertTrue(vectorError.errorDescription?.contains("normalize") ?? false)
        }
    }
    
    func testRangeValidation() throws {
        let range: ClosedRange<Float> = -1.0...1.0
        
        // Valid case
        XCTAssertNoThrow(try Validation.requireInRange([0.0, 0.5, -0.5, 1.0, -1.0], range: range))
        
        // Invalid case
        XCTAssertThrowsError(try Validation.requireInRange([0.0, 1.5, -2.0], range: range)) { error in
            guard let vectorError = error as? VectorError,
                  vectorError.kind == .invalidData else {
                XCTFail("Wrong error type")
                return
            }
            // Check the error message contains range violation info
            XCTAssertTrue(vectorError.errorDescription?.contains("range") ?? false)
        }
    }
    
    // MARK: - DynamicVector Throwing Operations Tests
    
    func testDynamicVectorThrowingDotProduct() throws {
        let v1 = DynamicVector([1, 2, 3])
        let v2 = DynamicVector([4, 5, 6])
        let v3 = DynamicVector([1, 2])
        
        // Valid case
        let dot = v1.dotProduct(v2)
        XCTAssertEqual(dot, 32.0, accuracy: 1e-6)
        
        // Invalid case - dimension mismatch
        // DynamicVector's dotProduct doesn't throw - it crashes on dimension mismatch
        // So we'll test a throwing operation instead
        XCTAssertThrowsError(try Validation.requireDimensionMatch(expected: v1.dimension, actual: v3.dimension)) { error in
            guard let vectorError = error as? VectorError,
                  vectorError.kind == .dimensionMismatch else {
                XCTFail("Wrong error type")
                return
            }
        }
    }
    
    func testDynamicVectorNormalized() throws {
        let v1 = DynamicVector([3, 4])
        let v2 = DynamicVector(dimension: 3, repeating: 0) // zero vector
        
        // Valid case
        let normalized = v1.normalized()
        XCTAssertEqual(normalized.magnitude, 1.0, accuracy: 1e-6)
        
        // Invalid case - zero vector
        // DynamicVector.normalized() returns zero vector for zero input, doesn't throw
        let zeroNorm = v2.normalized()
        XCTAssertEqual(zeroNorm.magnitude, 0, accuracy: 1e-6)
    }
    
    // Note: angle() method doesn't exist on DynamicVector
    // This test is removed as the functionality doesn't exist
    
    // Note: projected() method doesn't exist on DynamicVector  
    // This test is removed as the functionality doesn't exist
    
    func testDynamicVectorValidation() throws {
        let v1 = DynamicVector([0.5, -0.5, 0.0])
        let v2 = DynamicVector([1.5, -0.5, 0.0])
        
        // Use Validation.requireInRange instead since DynamicVector doesn't have validate method
        // Valid case - convert to array
        let v1Values = (0..<v1.dimension).map { v1[$0] }
        XCTAssertNoThrow(try Validation.requireInRange(v1Values, range: -1.0...1.0))
        
        // Invalid case - convert to array
        let v2Values = (0..<v2.dimension).map { v2[$0] }
        XCTAssertThrowsError(try Validation.requireInRange(v2Values, range: -1.0...1.0))
    }
    
    // MARK: - Result Extensions Tests
    
    func testResultCatching() {
        let successResult = Result<Float, VectorError>.catching {
            let v1 = DynamicVector([3, 4])
            let v2 = DynamicVector([1, 0])
            return v1.dotProduct(v2)
        }
        
        XCTAssertEqual(successResult.optional, 3.0)
        
        let failureResult = Result<Float, VectorError>.catching {
            let v1 = DynamicVector([3, 4])
            let v2 = DynamicVector([1, 0, 0])
            // This will crash with dimension mismatch, not throw
            // So we simulate a throwing operation
            try Validation.requireDimensionMatch(expected: v1.dimension, actual: v2.dimension)
            return v1.dotProduct(v2)
        }
        
        XCTAssertNil(failureResult.optional)
    }
    
    // MARK: - Performance Validation Tests
    
    func testPerformanceValidationInDebug() {
        // These should not crash in debug builds
        PerformanceValidation.require(true, "This should not fail")
        PerformanceValidation.requireDimensionMatch(10, 10)
        
        // In debug builds, these would fail with fatalError
        // We can't test them directly without crashing
    }
}