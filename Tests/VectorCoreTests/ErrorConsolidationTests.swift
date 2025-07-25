// VectorCore: Error Consolidation Tests
//
// Tests to verify that VectorError was successfully consolidated to VectorError
//

import XCTest
@testable import VectorCore

final class ErrorConsolidationTests: XCTestCase {
    
    func testDimensionMismatchError() throws {
        // Test that dimension mismatch errors work correctly
        do {
            _ = try VectorFactory.create(Dim128.self, from: Array(repeating: 0.5, count: 256))
            XCTFail("Should have thrown dimension mismatch error")
        } catch let error as VectorError {
            XCTAssertEqual(error.kind, .dimensionMismatch)
            XCTAssertTrue(error.description.contains("Expected dimension 128"))
            XCTAssertTrue(error.description.contains("but got 256"))
        }
    }
    
    func testIndexOutOfBoundsError() throws {
        // Test that index out of bounds errors work correctly
        do {
            _ = try VectorFactory.basis(dimension: 128, index: 200)
            XCTFail("Should have thrown index out of bounds error")
        } catch let error as VectorError {
            XCTAssertEqual(error.kind, .indexOutOfBounds)
            XCTAssertTrue(error.description.contains("Index 200"))
            XCTAssertTrue(error.description.contains("out of bounds"))
        }
    }
    
    func testInvalidValuesError() throws {
        // Test that invalid values errors work correctly
        do {
            _ = try VectorFactory.batch(dimension: 128, from: Array(repeating: 0.5, count: 129))
            XCTFail("Should have thrown invalid values error")
        } catch let error as VectorError {
            XCTAssertEqual(error.kind, .invalidData)
            XCTAssertTrue(error.description.contains("Values count"))
            XCTAssertTrue(error.description.contains("must be multiple"))
        }
    }
    
    func testBinaryFormatErrors() throws {
        // Test insufficient data error
        let shortData = Data([0x01, 0x02])
        do {
            _ = try BinaryFormat.readUInt32(from: shortData, at: 0)
            XCTFail("Should have thrown insufficient data error")
        } catch let error as VectorError {
            XCTAssertEqual(error.kind, .insufficientData)
            XCTAssertTrue(error.description.contains("Insufficient data"))
        }
    }
    
    func testZeroVectorError() throws {
        // Test zero vector error in safe operations
        let v1 = Vector<Dim128>()
        let _ = Vector<Dim128>()
        
        // Zero vector normalization returns zero vector (not an error)
        let normalized = v1.normalized()
        XCTAssertEqual(normalized.magnitude, 0, accuracy: 1e-7)
        
        // All components should be zero
        for i in 0..<normalized.scalarCount {
            XCTAssertEqual(normalized[i], 0, accuracy: 1e-7)
        }
    }
    
    func testDivisionByZeroError() throws {
        // Test division by zero error
        let v1 = Vector<Dim128>(repeating: 1.0)
        let v2 = Vector<Dim128>() // zero vector
        
        // Element-wise division by zero vector components results in inf/nan
        let result = v1 ./ v2  // This will produce inf since v2 components are 0
        
        // All components should be infinity
        for i in 0..<result.scalarCount {
            XCTAssertTrue(result[i].isInfinite)
        }
    }
    
    func testErrorCategorization() {
        // Test error severity and categorization
        let dimError = VectorError.dimensionMismatch(expected: 128, actual: 256)
        XCTAssertEqual(dimError.kind.severity, .medium)
        XCTAssertEqual(dimError.kind.category, .dimension)
        
        let corruptionError = VectorError.dataCorruption(reason: "Invalid CRC")
        XCTAssertEqual(corruptionError.kind.severity, .critical)
        XCTAssertEqual(corruptionError.kind.category, .data)
    }
    
    func testErrorChaining() {
        // Test error chaining functionality
        let rootError = VectorError.invalidData("File corrupted")
        let chainedError = VectorError(.operationFailed, message: "Load failed")
            .chain(with: rootError)
        
        XCTAssertEqual(chainedError.errorChain.count, 1)
        XCTAssertEqual(chainedError.errorChain.first?.kind, .invalidData)
        XCTAssertTrue(chainedError.description.contains("Error chain"))
    }
    
    func testResultExtensions() {
        // Test Result extensions work with VectorError
        let result: Result<Float, VectorError> = .failure(.dimensionMismatch(expected: 128, actual: 256))
        
        let chainedResult = result.chainError(.invalidData("Additional context"))
        
        switch chainedResult {
        case .success:
            XCTFail("Should be failure")
        case .failure(let error):
            XCTAssertEqual(error.errorChain.count, 1)
        }
    }
    
    func testLoggerIntegration() {
        // Test that logger works with VectorError
        let error = VectorError.dimensionMismatch(expected: 128, actual: 256)
        
        // This should compile and not crash
        coreLogger.error(error)
        error.log(to: coreLogger)
    }
}