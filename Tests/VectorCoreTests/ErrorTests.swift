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
        
        XCTAssertEqual(error.code, "DIMENSION_MISMATCH")
        XCTAssertEqual(error.errorDescription, "Vector dimension mismatch: expected 512, got 256")
        XCTAssertEqual(error.recoverySuggestion, "Ensure all vectors have the same number of dimensions")
    }
    
    func testInvalidDimensionError() {
        let error = VectorError.invalidDimension(100, reason: "must be power of 2")
        
        XCTAssertEqual(error.code, "INVALID_DIMENSION")
        XCTAssertEqual(error.errorDescription, "Invalid dimension 100: must be power of 2")
    }
    
    func testIndexOutOfBoundsError() {
        let error = VectorError.indexOutOfBounds(index: 300, dimension: 256)
        
        XCTAssertEqual(error.code, "INDEX_OUT_OF_BOUNDS")
        XCTAssertEqual(error.errorDescription, "Index 300 out of bounds for vector of dimension 256")
        XCTAssertEqual(error.recoverySuggestion, "Use an index between 0 and dimension-1")
    }
    
    func testInvalidValuesError() {
        let error = VectorError.invalidValues(indices: [5, 10, 15], reason: "NaN values detected")
        
        XCTAssertEqual(error.code, "INVALID_VALUES")
        XCTAssertEqual(error.errorDescription, "Invalid values at indices [5, 10, 15]: NaN values detected")
        XCTAssertEqual(error.recoverySuggestion, "Check for division by zero or invalid mathematical operations")
    }
    
    func testNumericalInstabilityError() {
        let error = VectorError.numericalInstability(operation: "SVD decomposition")
        
        XCTAssertEqual(error.code, "NUMERICAL_INSTABILITY")
        XCTAssertEqual(error.errorDescription, "Numerical instability in SVD decomposition")
        XCTAssertEqual(error.recoverySuggestion, "Consider normalizing inputs or using a more numerically stable algorithm")
    }
    
    func testDivisionByZeroError() {
        let error = VectorError.divisionByZero(operation: "normalization")
        
        XCTAssertEqual(error.code, "DIVISION_BY_ZERO")
        XCTAssertEqual(error.errorDescription, "Division by zero in normalization")
        XCTAssertEqual(error.recoverySuggestion, "Check that divisor is not zero before performing division")
    }
    
    func testZeroVectorError() {
        let error = VectorError.zeroVectorError(operation: "angle calculation")
        
        XCTAssertEqual(error.code, "ZERO_VECTOR_ERROR")
        XCTAssertEqual(error.errorDescription, "Cannot perform angle calculation on zero vector")
        XCTAssertEqual(error.recoverySuggestion, "Ensure vector has non-zero magnitude before this operation")
    }
    
    func testValidationFailedError() {
        let error = VectorError.validationFailed(reason: "vector contains invalid values")
        
        XCTAssertEqual(error.code, "VALIDATION_FAILED")
        XCTAssertEqual(error.errorDescription, "Validation failed: vector contains invalid values")
    }
    
    func testNotNormalizedError() {
        let error = VectorError.notNormalized(magnitude: 2.5)
        
        XCTAssertEqual(error.code, "NOT_NORMALIZED")
        XCTAssertEqual(error.errorDescription, "Vector is not normalized (magnitude: 2.5)")
        XCTAssertEqual(error.recoverySuggestion, "Normalize the vector before this operation")
    }
    
    func testOutOfRangeError() {
        let error = VectorError.outOfRange(indices: [0, 5], range: -1.0...1.0)
        
        XCTAssertEqual(error.code, "OUT_OF_RANGE")
        XCTAssertEqual(error.errorDescription, "Values at indices [0, 5] are outside range -1.0...1.0")
        XCTAssertEqual(error.recoverySuggestion, "Clamp or scale values to the expected range")
    }
    
    func testSerializationFailedError() {
        let error = VectorError.serializationFailed(reason: "invalid JSON format")
        
        XCTAssertEqual(error.code, "SERIALIZATION_FAILED")
        XCTAssertEqual(error.errorDescription, "Serialization failed: invalid JSON format")
    }
    
    func testInvalidDataFormatError() {
        let error = VectorError.invalidDataFormat(expected: "Base64", actual: "Hex")
        
        XCTAssertEqual(error.code, "INVALID_DATA_FORMAT")
        XCTAssertEqual(error.errorDescription, "Invalid data format: expected Base64, got Hex")
        XCTAssertEqual(error.recoverySuggestion, "Convert the data to the expected format before deserializing")
    }
    
    func testInvalidDataSizeError() {
        let error = VectorError.invalidDataSize(expected: 1024, actual: 512)
        
        XCTAssertEqual(error.code, "INVALID_DATA_SIZE")
        XCTAssertEqual(error.errorDescription, "Invalid data size: expected 1024 bytes, got 512")
        XCTAssertEqual(error.recoverySuggestion, "Ensure the data size matches the vector dimension")
    }
    
    func testOperationFailedError() {
        let error = VectorError.operationFailed(operation: "matrix multiplication", reason: "incompatible dimensions")
        
        XCTAssertEqual(error.code, "OPERATION_FAILED")
        XCTAssertEqual(error.errorDescription, "Operation 'matrix multiplication' failed: incompatible dimensions")
    }
    
    func testNotImplementedError() {
        let error = VectorError.notImplemented(feature: "sparse vector support")
        
        XCTAssertEqual(error.code, "NOT_IMPLEMENTED")
        XCTAssertEqual(error.errorDescription, "Feature not implemented: sparse vector support")
        XCTAssertEqual(error.recoverySuggestion, "This feature is not yet available")
    }
    
    func testInvalidParameterError() {
        let error = VectorError.invalidParameter(name: "k", reason: "must be positive")
        
        XCTAssertEqual(error.code, "INVALID_PARAMETER")
        XCTAssertEqual(error.errorDescription, "Invalid parameter 'k': must be positive")
    }
    
    func testInvalidStateError() {
        let error = VectorError.invalidState(reason: "index not built")
        
        XCTAssertEqual(error.code, "INVALID_STATE")
        XCTAssertEqual(error.errorDescription, "Invalid state: index not built")
    }
    
    // MARK: - Convenience Factory Method Tests
    
    func testNanValuesFactory() {
        let error = VectorError.nanValues(at: [1, 2, 3])
        
        XCTAssertEqual(error.code, "INVALID_VALUES")
        if case .invalidValues(let indices, let reason) = error {
            XCTAssertEqual(indices, [1, 2, 3])
            XCTAssertEqual(reason, "NaN values detected")
        } else {
            XCTFail("Expected invalidValues case")
        }
    }
    
    func testInfinityValuesFactory() {
        let error = VectorError.infinityValues(at: [5, 10])
        
        XCTAssertEqual(error.code, "INVALID_VALUES")
        if case .invalidValues(let indices, let reason) = error {
            XCTAssertEqual(indices, [5, 10])
            XCTAssertEqual(reason, "Infinity values detected")
        } else {
            XCTFail("Expected invalidValues case")
        }
    }
    
    func testUnsupportedDimensionFactory() {
        let error = VectorError.unsupportedDimension(1000)
        
        XCTAssertEqual(error.code, "INVALID_DIMENSION")
        if case .invalidDimension(let dimension, let reason) = error {
            XCTAssertEqual(dimension, 1000)
            XCTAssertEqual(reason, "Supported dimensions are 128, 256, 512, 768, 1536")
        } else {
            XCTFail("Expected invalidDimension case")
        }
    }
    
    func testNormalizationFailedFactory() {
        // Test with zero magnitude
        let zeroError = VectorError.normalizationFailed(magnitude: 0)
        XCTAssertEqual(zeroError.code, "ZERO_VECTOR_ERROR")
        
        // Test with non-zero magnitude
        let instabilityError = VectorError.normalizationFailed(magnitude: 1e-20)
        XCTAssertEqual(instabilityError.code, "NUMERICAL_INSTABILITY")
    }
    
    // MARK: - Equatable Tests
    
    func testVectorErrorEquality() {
        let error1 = VectorError.dimensionMismatch(expected: 128, actual: 256)
        let error2 = VectorError.dimensionMismatch(expected: 128, actual: 256)
        let error3 = VectorError.dimensionMismatch(expected: 256, actual: 512)
        let error4 = VectorError.invalidDimension(128, reason: "test")
        
        XCTAssertEqual(error1, error2)
        XCTAssertNotEqual(error1, error3)
        XCTAssertNotEqual(error1, error4)
    }
    
    // MARK: - Hashable Tests
    
    func testVectorErrorHashable() {
        let error1 = VectorError.operationFailed(operation: "test", reason: "failure")
        let error2 = VectorError.operationFailed(operation: "test", reason: "failure")
        
        // Can be used in sets
        var errorSet = Set<VectorError>()
        errorSet.insert(error1)
        errorSet.insert(error2)
        XCTAssertEqual(errorSet.count, 1) // Should be deduplicated
        
        // Can be used as dictionary keys
        var errorDict = [VectorError: String]()
        errorDict[error1] = "First"
        errorDict[error2] = "Second"
        XCTAssertEqual(errorDict.count, 1)
        XCTAssertEqual(errorDict[error1], "Second") // Should be overwritten
    }
    
    // MARK: - Error Throwing Tests
    
    func testThrowingVectorError() {
        func problematicFunction() throws {
            throw VectorError.operationFailed(operation: "test", reason: "intentional failure")
        }
        
        XCTAssertThrowsError(try problematicFunction()) { error in
            guard let coreError = error as? VectorError else {
                XCTFail("Expected VectorError")
                return
            }
            
            XCTAssertEqual(coreError.code, "OPERATION_FAILED")
            XCTAssertTrue(coreError.errorDescription?.contains("intentional failure") ?? false)
        }
    }
    
    // MARK: - Error Propagation Tests
    
    func testErrorPropagation() {
        func innerFunction() throws -> Vector256 {
            throw VectorError.invalidParameter(name: "dimensions", reason: "must be positive")
        }
        
        func outerFunction() throws -> Vector256 {
            do {
                return try innerFunction()
            } catch {
                throw VectorError.operationFailed(
                    operation: "createVector",
                    reason: "inner function failed"
                )
            }
        }
        
        do {
            _ = try outerFunction()
            XCTFail("Expected error to be thrown")
        } catch let error as VectorError {
            XCTAssertEqual(error.code, "OPERATION_FAILED")
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
        XCTAssertEqual(receivedError.code, "DIMENSION_MISMATCH")
    }
}