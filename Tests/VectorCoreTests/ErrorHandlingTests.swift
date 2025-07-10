// VectorCore: Error Handling Tests
//
// Tests for modern error system with context and telemetry
//

import XCTest
@testable import VectorCore

final class ErrorHandlingTests: XCTestCase {
    
    override func setUp() async throws {
        // Setup for each test
    }
    
    // MARK: - Error Creation Tests
    
    func testBasicErrorCreation() {
        let error = VectorError(.dimensionMismatch, message: "Test error")
        
        XCTAssertEqual(error.kind, .dimensionMismatch)
        XCTAssertEqual(error.context.additionalInfo["message"], "Test error")
        XCTAssertNil(error.underlyingError)
        XCTAssertTrue(error.errorChain.isEmpty)
    }
    
    func testErrorFactoryMethods() {
        let dimError = VectorError.dimensionMismatch(expected: 512, actual: 256)
        XCTAssertEqual(dimError.kind, .dimensionMismatch)
        XCTAssertTrue(dimError.description.contains("Expected dimension 512"))
        XCTAssertTrue(dimError.description.contains("got 256"))
        
        let indexError = VectorError.indexOutOfBounds(index: 10, dimension: 5)
        XCTAssertEqual(indexError.kind, .indexOutOfBounds)
        XCTAssertTrue(indexError.description.contains("Index 10"))
        XCTAssertTrue(indexError.description.contains("dimension 5"))
        
        let opError = VectorError.invalidOperation("normalize", reason: "zero vector")
        XCTAssertEqual(opError.kind, .invalidOperation)
        XCTAssertTrue(opError.description.contains("normalize failed"))
        XCTAssertTrue(opError.description.contains("zero vector"))
    }
    
    func testErrorChaining() {
        let rootCause = VectorError(.dataCorruption, message: "Invalid input data")
        let midError = VectorError(.operationFailed, message: "Processing failed")
        let topError = VectorError(.unknown, message: "System error")
        
        let chainedError = topError
            .chain(with: midError)
            .chain(with: rootCause)
        
        XCTAssertEqual(chainedError.errorChain.count, 2)
        XCTAssertEqual(chainedError.errorChain[0].kind, .operationFailed)
        XCTAssertEqual(chainedError.errorChain[1].kind, .dataCorruption)
    }
    
    func testErrorBuilder() {
        let error = ErrorBuilder(.allocationFailed)
            .message("Failed to allocate 1GB memory")
            .parameter("size", value: "1073741824")
            .parameter("available", value: "524288000")
            .build()
        
        XCTAssertEqual(error.kind, .allocationFailed)
        XCTAssertEqual(error.context.additionalInfo["size"], "1073741824")
        XCTAssertEqual(error.context.additionalInfo["available"], "524288000")
    }
    
    func testUnderlyingError() {
        let nsError = NSError(domain: "TestDomain", code: 42, userInfo: nil)
        let error = VectorError(.systemError, message: "Wrapped error", underlying: nsError)
        
        XCTAssertNotNil(error.underlyingError)
        XCTAssertTrue(error.description.contains("Wrapped error"))
        XCTAssertTrue(error.description.contains("TestDomain"))
    }
    
    // MARK: - Context Tests
    
    func testContextCapture() {
        #if DEBUG
        let error = VectorError(.invalidData, message: "Test")
        
        // In debug builds, context should be captured
        XCTAssertNotEqual(error.context.file, "")
        XCTAssertNotEqual(error.context.line, 0)
        XCTAssertNotEqual(error.context.function, "")
        XCTAssertTrue(error.description.contains("ErrorHandlingTests.swift"))
        #else
        let error = VectorError(.invalidData, message: "Test")
        
        // In release builds, no context
        XCTAssertEqual(error.context.file, "")
        XCTAssertEqual(error.context.line, 0)
        XCTAssertEqual(error.context.function, "")
        #endif
    }
    
    // MARK: - Categorization Tests
    
    func testErrorSeverity() {
        XCTAssertEqual(VectorError.ErrorKind.dataCorruption.severity, .critical)
        XCTAssertEqual(VectorError.ErrorKind.systemError.severity, .critical)
        XCTAssertEqual(VectorError.ErrorKind.allocationFailed.severity, .high)
        XCTAssertEqual(VectorError.ErrorKind.dimensionMismatch.severity, .medium)
        XCTAssertEqual(VectorError.ErrorKind.invalidConfiguration.severity, .low)
    }
    
    func testErrorCategory() {
        XCTAssertEqual(VectorError.ErrorKind.dimensionMismatch.category, .dimension)
        XCTAssertEqual(VectorError.ErrorKind.indexOutOfBounds.category, .bounds)
        XCTAssertEqual(VectorError.ErrorKind.dataCorruption.category, .data)
        XCTAssertEqual(VectorError.ErrorKind.operationFailed.category, .operation)
        XCTAssertEqual(VectorError.ErrorKind.allocationFailed.category, .resource)
    }
    
    // MARK: - Result Extensions Tests
    
    func testResultExtensions() {
        let success: Result<Int, VectorError> = .success(42)
        let failure: Result<Int, VectorError> = .failure(
            VectorError(.operationFailed, message: "Initial error")
        )
        
        // Test mapErrorContext
        let mapped = failure.mapErrorContext { error in
            ErrorBuilder(error.kind)
                .message("Mapped: \(error.context.additionalInfo["message"] ?? "")")
                .build()
        }
        
        switch mapped {
        case .failure(let error):
            XCTAssertTrue(error.description.contains("Mapped: Initial error"))
        case .success:
            XCTFail("Should be failure")
        }
        
        // Test chainError
        let chainError = VectorError(.unknown, message: "Additional context")
        let chained = failure.chainError(chainError)
        
        switch chained {
        case .failure(let error):
            XCTAssertEqual(error.errorChain.count, 1)
            XCTAssertEqual(error.errorChain[0].kind, .unknown)
        case .success:
            XCTFail("Should be failure")
        }
    }
    
    // MARK: - Integration Tests
    
    func testVectorOperationErrors() throws {
        // Test dimension mismatch in factory
        do {
            _ = try VectorFactory.vector(of: 128, from: [1, 2, 3])
            XCTFail("Should throw dimension mismatch")
        } catch let error as VectorError {
            XCTAssertEqual(error.kind, .dimensionMismatch)
        }
        
        // Test index out of bounds
        do {
            _ = try VectorFactory.basis(dimension: 10, index: 20)
            XCTFail("Should throw index out of bounds")
        } catch let error as VectorError {
            XCTAssertEqual(error.kind, .indexOutOfBounds)
        }
    }
}