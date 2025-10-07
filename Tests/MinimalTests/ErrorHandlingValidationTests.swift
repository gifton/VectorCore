//
//  ErrorHandlingValidationTests.swift
//  VectorCore
//
//  Minimal error handling validation tests
//

import XCTest
@testable import VectorCore

/// Minimal test class to validate error handling functionality
final class ErrorHandlingValidationTests: XCTestCase {

    func testBasicErrorCreation() {
        let error = VectorError(.dimensionMismatch)

        XCTAssertEqual(error.kind, .dimensionMismatch)
        XCTAssertNil(error.underlyingError)
        XCTAssertTrue(error.errorChain.isEmpty)
        XCTAssertTrue(error.context.timestamp.timeIntervalSince1970 > 0)
        XCTAssertTrue(error.context.additionalInfo.isEmpty)
    }

    func testErrorBuilderBasic() {
        let error = ErrorBuilder(.dimensionMismatch)
            .message("Test dimension mismatch")
            .dimension(expected: 512, actual: 256)
            .build()

        XCTAssertEqual(error.kind, .dimensionMismatch)
        XCTAssertEqual(error.context.additionalInfo["message"], "Test dimension mismatch")
        XCTAssertEqual(error.context.additionalInfo["expected_dimension"], "512")
        XCTAssertEqual(error.context.additionalInfo["actual_dimension"], "256")
    }

    func testErrorSeverityAndCategory() {
        XCTAssertEqual(VectorError.ErrorKind.dataCorruption.severity, .critical)
        XCTAssertEqual(VectorError.ErrorKind.dimensionMismatch.severity, .medium)
        XCTAssertEqual(VectorError.ErrorKind.dimensionMismatch.category, .dimension)
        XCTAssertEqual(VectorError.ErrorKind.indexOutOfBounds.category, .bounds)
    }

    func testValidationUtilitySuccess() {
        // These should not throw
        XCTAssertNoThrow(try Validation.requireDimensionMatch(expected: 512, actual: 512))
        XCTAssertNoThrow(try Validation.requireValidDimension(128))
        XCTAssertNoThrow(try Validation.requireValidIndex(100, dimension: 512))
        XCTAssertNoThrow(try Validation.requireNonZero(magnitude: 1.0, operation: "normalize"))
        XCTAssertNoThrow(try Validation.requireInRange([0.5, 0.8], range: 0.0...1.0))
    }

    func testValidationUtilityFailures() {
        // Dimension mismatch
        XCTAssertThrowsError(try Validation.requireDimensionMatch(expected: 512, actual: 256)) { error in
            XCTAssertTrue(error is VectorError)
            let vectorError = error as! VectorError
            XCTAssertEqual(vectorError.kind, .dimensionMismatch)
        }

        // Invalid dimension
        XCTAssertThrowsError(try Validation.requireValidDimension(0)) { error in
            XCTAssertTrue(error is VectorError)
            let vectorError = error as! VectorError
            XCTAssertEqual(vectorError.kind, .invalidDimension)
        }

        // Index out of bounds
        XCTAssertThrowsError(try Validation.requireValidIndex(-1, dimension: 512)) { error in
            XCTAssertTrue(error is VectorError)
            let vectorError = error as! VectorError
            XCTAssertEqual(vectorError.kind, .indexOutOfBounds)
        }

        // Zero magnitude
        XCTAssertThrowsError(try Validation.requireNonZero(magnitude: 0.0, operation: "normalize")) { error in
            XCTAssertTrue(error is VectorError)
            let vectorError = error as! VectorError
            XCTAssertEqual(vectorError.kind, .invalidOperation)
        }

        // Range validation
        XCTAssertThrowsError(try Validation.requireInRange([0.5, 1.5], range: 0.0...1.0)) { error in
            XCTAssertTrue(error is VectorError)
            let vectorError = error as! VectorError
            XCTAssertEqual(vectorError.kind, .invalidData)
        }
    }

    func testConvenienceFactoryMethods() {
        let dimError = VectorError.dimensionMismatch(expected: 512, actual: 256)
        XCTAssertEqual(dimError.kind, .dimensionMismatch)
        XCTAssertEqual(dimError.context.additionalInfo["expected_dimension"], "512")

        let indexError = VectorError.indexOutOfBounds(index: 100, dimension: 50)
        XCTAssertEqual(indexError.kind, .indexOutOfBounds)
        XCTAssertEqual(indexError.context.additionalInfo["index"], "100")

        let allocError = VectorError.allocationFailed(size: 1024)
        XCTAssertEqual(allocError.kind, .allocationFailed)
        XCTAssertEqual(allocError.context.additionalInfo["requested_size"], "1024")
    }

    func testErrorChaining() {
        let rootError = VectorError(.invalidData, message: "Root cause")
        let chainedError = VectorError(.operationFailed, message: "Operation failed").chain(with: rootError)

        XCTAssertEqual(chainedError.errorChain.count, 1)
        XCTAssertEqual(chainedError.errorChain[0].kind, .invalidData)
        XCTAssertTrue(chainedError.description.contains("Error chain:"))
    }

    func testVectorMathSafeDivision() {
        let numerator = Vector<Dim32>(repeating: 1.0)
        let validDenominator = Vector<Dim32>(repeating: 2.0)
        let invalidDenominator = Vector<Dim32>(repeating: 0.0)

        // Valid division should succeed
        XCTAssertNoThrow({
            let result = try Vector.safeDivide(numerator, by: validDenominator)
            XCTAssertEqual(result[0], 0.5, accuracy: 1e-6)
        })

        // Invalid division should throw
        XCTAssertThrowsError(try Vector.safeDivide(numerator, by: invalidDenominator)) { error in
            XCTAssertTrue(error is VectorError)
            let vectorError = error as! VectorError
            XCTAssertEqual(vectorError.kind, .invalidOperation)
            // Check for division-by-zero error message (case-insensitive for robustness across build configs)
            let description = vectorError.description.lowercased()
            XCTAssertTrue(description.contains("division") && description.contains("zero"),
                         "Expected division-by-zero error but got: \(vectorError.description)")
        }
    }

    func testAllErrorKindsCoverage() {
        // Ensure all error kinds can be created and have valid properties
        for errorKind in VectorError.ErrorKind.allCases {
            let error = VectorError(errorKind, message: "Test error for \(errorKind)")
            XCTAssertEqual(error.kind, errorKind)

            // All should have valid severity and category
            let severity = errorKind.severity
            let category = errorKind.category
            XCTAssertTrue([.critical, .high, .medium, .low, .info].contains(severity))
            XCTAssertTrue([.dimension, .bounds, .data, .operation, .resource, .configuration, .system].contains(category))
        }
    }
}
