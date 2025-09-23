//
//  StandaloneErrorHandlingTests.swift
//  VectorCore
//
//  Standalone error handling tests to validate error system functionality
//

import Testing
import Foundation
@testable import VectorCore

/// Standalone test suite for VectorCore error handling system
@Suite("Standalone Error Handling Tests")
struct StandaloneErrorHandlingTests {

    @Test("Basic VectorError creation")
    func testBasicVectorErrorCreation() async throws {
        let error = VectorError(.dimensionMismatch)

        #expect(error.kind == .dimensionMismatch)
        #expect(error.underlyingError == nil)
        #expect(error.errorChain.isEmpty)
        #expect(error.context.timestamp.timeIntervalSince1970 > 0)
        #expect(error.context.additionalInfo.isEmpty)
    }

    @Test("ErrorBuilder functionality")
    func testErrorBuilderFunctionality() async throws {
        let error = ErrorBuilder(.dimensionMismatch)
            .message("Test dimension mismatch")
            .dimension(expected: 512, actual: 256)
            .build()

        #expect(error.kind == .dimensionMismatch)
        #expect(error.context.additionalInfo["message"] == "Test dimension mismatch")
        #expect(error.context.additionalInfo["expected_dimension"] == "512")
        #expect(error.context.additionalInfo["actual_dimension"] == "256")
    }

    @Test("Error severity mapping")
    func testErrorSeverityMapping() async throws {
        #expect(VectorError.ErrorKind.dataCorruption.severity == .critical)
        #expect(VectorError.ErrorKind.systemError.severity == .critical)
        #expect(VectorError.ErrorKind.allocationFailed.severity == .high)
        #expect(VectorError.ErrorKind.dimensionMismatch.severity == .medium)
        #expect(VectorError.ErrorKind.invalidConfiguration.severity == .low)
    }

    @Test("Error category mapping")
    func testErrorCategoryMapping() async throws {
        #expect(VectorError.ErrorKind.dimensionMismatch.category == .dimension)
        #expect(VectorError.ErrorKind.indexOutOfBounds.category == .bounds)
        #expect(VectorError.ErrorKind.invalidData.category == .data)
        #expect(VectorError.ErrorKind.operationFailed.category == .operation)
        #expect(VectorError.ErrorKind.allocationFailed.category == .resource)
        #expect(VectorError.ErrorKind.invalidConfiguration.category == .configuration)
        #expect(VectorError.ErrorKind.systemError.category == .system)
    }

    @Test("Validation utility - dimension match success")
    func testValidationDimensionMatchSuccess() async throws {
        try Validation.requireDimensionMatch(expected: 512, actual: 512)
        try Validation.requireDimensionMatch(expected: 1, actual: 1)
    }

    @Test("Validation utility - dimension match failure")
    func testValidationDimensionMatchFailure() async throws {
        do {
            try Validation.requireDimensionMatch(expected: 512, actual: 256)
            #expect(Bool(false), "Should have thrown dimension mismatch error")
        } catch let error as VectorError {
            #expect(error.kind == .dimensionMismatch)
            #expect(error.context.additionalInfo["expected_dimension"] == "512")
            #expect(error.context.additionalInfo["actual_dimension"] == "256")
        }
    }

    @Test("Validation utility - valid dimension")
    func testValidationValidDimension() async throws {
        try Validation.requireValidDimension(1)
        try Validation.requireValidDimension(512)

        do {
            try Validation.requireValidDimension(0)
            #expect(Bool(false), "Should have thrown invalid dimension error")
        } catch let error as VectorError {
            #expect(error.kind == .invalidDimension)
            #expect(error.context.additionalInfo["dimension"] == "0")
        }
    }

    @Test("Validation utility - valid index")
    func testValidationValidIndex() async throws {
        try Validation.requireValidIndex(0, dimension: 1)
        try Validation.requireValidIndex(511, dimension: 512)

        do {
            try Validation.requireValidIndex(-1, dimension: 512)
            #expect(Bool(false), "Should have thrown index out of bounds error")
        } catch let error as VectorError {
            #expect(error.kind == .indexOutOfBounds)
            #expect(error.context.additionalInfo["index"] == "-1")
        }
    }

    @Test("Validation utility - non-zero magnitude")
    func testValidationNonZeroMagnitude() async throws {
        try Validation.requireNonZero(magnitude: 1.0, operation: "normalize")

        do {
            try Validation.requireNonZero(magnitude: 0.0, operation: "normalize")
            #expect(Bool(false), "Should have thrown zero vector error")
        } catch let error as VectorError {
            #expect(error.kind == .invalidOperation)
            #expect(error.context.additionalInfo["message"]?.contains("Cannot perform operation on zero vector") == true)
        }
    }

    @Test("Validation utility - range validation")
    func testValidationRangeValidation() async throws {
        try Validation.requireInRange([0.5, 0.8, 1.0], range: 0.0...1.0)

        do {
            try Validation.requireInRange([0.5, 1.5, 0.8], range: 0.0...1.0)
            #expect(Bool(false), "Should have thrown invalid values error")
        } catch let error as VectorError {
            #expect(error.kind == .invalidData)
            #expect(error.context.additionalInfo["indices"] == "1")
        }
    }

    @Test("Vector math - safe division")
    func testVectorMathSafeDivision() async throws {
        let numerator = Vector<Dim32>(repeating: 1.0)
        let validDenominator = Vector<Dim32>(repeating: 2.0)
        let invalidDenominator = Vector<Dim32>(repeating: 0.0)

        // Valid division should succeed
        let validResult = try Vector.safeDivide(numerator, by: validDenominator)
        #expect(validResult[0] == 0.5)

        // Invalid division should throw
        do {
            _ = try Vector.safeDivide(numerator, by: invalidDenominator)
            #expect(Bool(false), "Should have thrown division by zero error")
        } catch let error as VectorError {
            #expect(error.kind == .invalidOperation)
            #expect(error.description.contains("Division by zero"))
        }
    }

    @Test("Error chaining functionality")
    func testErrorChaining() async throws {
        let rootError = VectorError(.invalidData, message: "Root cause")
        let chainedError = VectorError(.operationFailed, message: "Operation failed").chain(with: rootError)

        #expect(chainedError.errorChain.count == 1)
        #expect(chainedError.errorChain[0].kind == .invalidData)
        #expect(chainedError.description.contains("Error chain:"))
    }

    @Test("Convenience factory methods")
    func testConvenienceFactoryMethods() async throws {
        let dimError = VectorError.dimensionMismatch(expected: 512, actual: 256)
        #expect(dimError.kind == .dimensionMismatch)
        #expect(dimError.context.additionalInfo["expected_dimension"] == "512")

        let indexError = VectorError.indexOutOfBounds(index: 100, dimension: 50)
        #expect(indexError.kind == .indexOutOfBounds)
        #expect(indexError.context.additionalInfo["index"] == "100")

        let allocError = VectorError.allocationFailed(size: 1024)
        #expect(allocError.kind == .allocationFailed)
        #expect(allocError.context.additionalInfo["requested_size"] == "1024")
    }

    @Test("Error string representation")
    func testErrorStringRepresentation() async throws {
        let error = VectorError(.dimensionMismatch, message: "Test error")
        let description = error.description

        #expect(description.contains("dimensionMismatch"))
        #expect(description.contains("Test error"))

        // Debug description should have more detail
        let debugDescription = error.debugDescription
        #expect(debugDescription.contains("Context:"))
        #expect(debugDescription.contains("Timestamp:"))
    }

    @Test("All error kinds coverage")
    func testAllErrorKindsCoverage() async throws {
        // Ensure all error kinds can be created and have valid properties
        for errorKind in VectorError.ErrorKind.allCases {
            let error = VectorError(errorKind, message: "Test error for \(errorKind)")
            #expect(error.kind == errorKind)

            // All should have valid severity and category
            let severity = errorKind.severity
            let category = errorKind.category
            #expect([.critical, .high, .medium, .low, .info].contains(severity))
            #expect([.dimension, .bounds, .data, .operation, .resource, .configuration, .system].contains(category))
        }
    }
}

// Mock error type for testing
struct TestMockError: Error, CustomStringConvertible {
    let message: String
    var description: String { "TestMockError: \(message)" }
}