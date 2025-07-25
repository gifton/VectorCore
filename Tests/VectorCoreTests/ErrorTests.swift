// VectorCoreTests: Error Tests
//
// Tests for error handling
//

import XCTest
@testable import VectorCore

final class ErrorTests: XCTestCase {
    
    // MARK: - VectorError Tests
    
    func testDimensionMismatchError() {
        let error = VectorError.dimensionMismatch(expected: 512, actual: 256)
        
        XCTAssertEqual(error.kind, .dimensionMismatch)
        XCTAssertTrue(error.description.contains("Expected dimension 512, but got 256"))
        XCTAssertEqual(error.context.additionalInfo["expected_dimension"], "512")
        XCTAssertEqual(error.context.additionalInfo["actual_dimension"], "256")
    }
    
    func testInvalidDimensionError() {
        let error = VectorError.invalidDimension(100, reason: "must be power of 2")
        
        XCTAssertEqual(error.kind, .invalidDimension)
        XCTAssertTrue(error.description.contains("Invalid dimension 100: must be power of 2"))
    }
    
    func testIndexOutOfBoundsError() {
        let error = VectorError.indexOutOfBounds(index: 300, dimension: 256)
        
        XCTAssertEqual(error.kind, .indexOutOfBounds)
        XCTAssertTrue(error.description.contains("Index 300 is out of bounds for dimension 256"))
        XCTAssertEqual(error.context.additionalInfo["index"], "300")
        XCTAssertEqual(error.context.additionalInfo["max_index"], "255")
    }
    
    func testInvalidValuesError() {
        let error = VectorError.invalidValues(indices: [5, 10, 15], reason: "NaN values detected")
        
        XCTAssertEqual(error.kind, .invalidData)
        XCTAssertTrue(error.description.contains("Invalid values at indices [5, 10, 15]: NaN values detected"))
        XCTAssertEqual(error.context.additionalInfo["indices"], "5,10,15")
        XCTAssertEqual(error.context.additionalInfo["reason"], "NaN values detected")
    }
    
    func testInvalidOperationError() {
        let error = VectorError.invalidOperation("SVD decomposition", reason: "Numerical instability")
        
        XCTAssertEqual(error.kind, .invalidOperation)
        XCTAssertTrue(error.description.contains("SVD decomposition failed: Numerical instability"))
    }
    
    func testDivisionByZeroError() {
        let error = VectorError.divisionByZero(operation: "normalization")
        
        XCTAssertEqual(error.kind, .invalidOperation)
        XCTAssertEqual(error.errorDescription, "Division by zero in normalization")
        XCTAssertEqual(error.recoverySuggestion, "Check that divisor is not zero before performing division")
    }
    
    func testZeroVectorError() {
        let error = VectorError.zeroVectorError(operation: "angle calculation")
        
        XCTAssertEqual(error.kind, .invalidOperation)
        XCTAssertTrue(error.errorDescription?.contains("angle calculation") ?? false)
        XCTAssertTrue(error.errorDescription?.contains("zero vector") ?? false)
    }
    
    func testValidationFailedError() {
        let error = VectorError.invalidData("vector contains invalid values")
        
        XCTAssertEqual(error.kind, .invalidData)
        XCTAssertTrue(error.errorDescription?.contains("vector contains invalid values") ?? false)
    }
    
    func testNotNormalizedError() {
        let error = VectorError(.invalidOperation, message: "Vector is not normalized (magnitude: 2.5)")
        
        XCTAssertEqual(error.kind, .invalidOperation)
        XCTAssertTrue(error.description.contains("Vector is not normalized (magnitude: 2.5)"))
    }
    
    func testOutOfRangeError() {
        let error = VectorError.invalidValues(indices: [0, 5], reason: "Values outside range -1.0...1.0")
        
        XCTAssertEqual(error.kind, .invalidData)
        XCTAssertTrue(error.description.contains("Invalid values at indices [0, 5]"))
        XCTAssertTrue(error.description.contains("Values outside range -1.0...1.0"))
    }
    
    func testSerializationFailedError() {
        let error = VectorError(.operationFailed, message: "Serialization failed: invalid JSON format")
        
        XCTAssertEqual(error.kind, .operationFailed)
        XCTAssertTrue(error.description.contains("Serialization failed: invalid JSON format"))
    }
    
    func testInvalidDataFormatError() {
        let error = VectorError.invalidDataFormat(expected: "Base64", actual: "Hex")
        
        XCTAssertEqual(error.kind, .invalidData)
        XCTAssertTrue(error.description.contains("Invalid data format: expected Base64, got Hex"))
    }
    
    func testInsufficientDataError() {
        let error = VectorError.insufficientData(expected: 1024, actual: 512)
        
        XCTAssertEqual(error.kind, .insufficientData)
        XCTAssertTrue(error.description.contains("Insufficient data: expected 1024 bytes, got 512"))
    }
    
    // Operation failed is not a current error case
    
    // Not implemented is not a current error case
    
    // Invalid parameter is not a current error case
    
    // Invalid state is not a current error case
    
    // MARK: - Convenience Factory Method Tests
    
    // NaN values factory is not a current error case
    // Use VectorError.invalidValues directly
    
    // Infinity values factory is not a current error case
    // Use VectorError.invalidValues directly
    
    // Unsupported dimension factory is not a current error case
    // Use VectorError.invalidDimension directly
    
    // Normalization failed factory is not a current error case
    // Use VectorError.zeroVectorError or VectorError.numericalInstability directly
    
    // MARK: - Equatable Tests
    
    func testVectorErrorEquality() {
        // VectorError is not Equatable, test code properties instead
        let error1 = VectorError.dimensionMismatch(expected: 128, actual: 256)
        let error2 = VectorError.dimensionMismatch(expected: 128, actual: 256)
        let error3 = VectorError.dimensionMismatch(expected: 256, actual: 512)
        let error4 = VectorError.invalidDimension(128, reason: "test")
        
        // Test same errors have same properties
        XCTAssertEqual(error1.kind, error2.kind)
        XCTAssertEqual(error1.errorDescription, error2.errorDescription)
        
        // Test different errors have different properties
        XCTAssertNotEqual(error1.errorDescription, error3.errorDescription)
        XCTAssertNotEqual(error1.kind, error4.kind)
    }
    
    // MARK: - Hashable Tests
    
    // VectorError is not Hashable - test removed
    
    // MARK: - Error Throwing Tests
    
    func testThrowingVectorError() {
        func problematicFunction() throws {
            throw VectorError.invalidData("intentional failure")
        }
        
        XCTAssertThrowsError(try problematicFunction()) { error in
            guard let coreError = error as? VectorError else {
                XCTFail("Expected VectorError")
                return
            }
            
            XCTAssertEqual(coreError.kind, .invalidData)
            XCTAssertTrue(coreError.errorDescription?.contains("intentional failure") ?? false)
        }
    }
    
    // MARK: - Error Propagation Tests
    
    func testErrorPropagation() {
        func innerFunction() throws -> Vector256 {
            throw VectorError.invalidDimension(256, reason: "must be positive")
        }
        
        func outerFunction() throws -> Vector256 {
            do {
                return try innerFunction()
            } catch {
                throw VectorError(.operationFailed, message: "inner function failed")
            }
        }
        
        do {
            _ = try outerFunction()
            XCTFail("Expected error to be thrown")
        } catch let error as VectorError {
            XCTAssertEqual(error.kind, .operationFailed)
            XCTAssertTrue(error.errorDescription?.contains("inner function failed") ?? false)
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }
    
    // MARK: - Sendable Conformance
    
    func testVectorErrorSendable() async {
        let error = VectorError.dimensionMismatch(expected: 128, actual: 256)
        
        // Test that error can be sent across async boundaries
        let task = Task {
            return error
        }
        
        let receivedError = await task.value
        XCTAssertEqual(receivedError.kind, .dimensionMismatch)
    }
}