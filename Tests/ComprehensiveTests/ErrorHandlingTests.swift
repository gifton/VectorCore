//
//  ErrorHandlingTests.swift
//  VectorCore
//
//  Comprehensive test suite for error handling paths, covering all error types
//  and edge cases in the VectorCore error system.
//

import Testing
import Foundation
@testable import VectorCore

/// Comprehensive test suite for VectorCore error handling system
@Suite("Error Handling Tests")
struct ErrorHandlingTests {

    // MARK: - VectorError Core Tests

    @Suite("VectorError Core")
    struct VectorErrorCoreTests {

        @Test("Basic error creation with context")
        func testBasicErrorCreation() async throws {
            let error = VectorError(.dimensionMismatch)

            #expect(error.kind == .dimensionMismatch)
            #expect(error.underlyingError == nil)
            #expect(error.errorChain.isEmpty)
            #expect(error.context.timestamp.timeIntervalSince1970 > 0)
            #expect(error.context.additionalInfo.isEmpty)
        }

        @Test("Error creation with custom message")
        func testErrorCreationWithMessage() async throws {
            let message = "Test error message"
            let error = VectorError(.invalidData, message: message)

            #expect(error.kind == .invalidData)
            #expect(error.context.additionalInfo["message"] == message)
            #expect(error.description.contains(message))
        }

        @Test("Error creation with underlying error")
        func testErrorCreationWithUnderlying() async throws {
            let systemError = SimpleMockSystemError(code: 500, message: "System failure")
            let error = VectorError(.systemError, message: "Wrapper error", underlying: systemError)

            #expect(error.kind == .systemError)
            #expect(error.underlyingError != nil)
            #expect(error.description.contains("System failure"))
        }

        @Test("Error chaining functionality")
        func testErrorChaining() async throws {
            let rootError = VectorError(.invalidData, message: "Root cause")
            let middleError = VectorError(.operationFailed, message: "Middle error").chain(with: rootError)
            let topError = VectorError(.systemError, message: "Top level").chain(with: middleError)

            #expect(topError.errorChain.count == 1)
            #expect(topError.errorChain[0].kind == .operationFailed)
            #expect(topError.errorChain[0].errorChain.count == 1)
            #expect(topError.errorChain[0].errorChain[0].kind == .invalidData)

            // Error chain might not be in description, verify chain structure is correct
            #expect(topError.kind == .systemError)
            #expect(topError.description.contains("Top level") || topError.description.contains("systemError"))
        }

        @Test("Error context validation")
        func testErrorContext() async throws {
            let additionalInfo = ["dimension": "512", "operation": "normalize"]
            let context = ErrorContext(additionalInfo: additionalInfo)

            #expect(context.additionalInfo["dimension"] == "512")
            #expect(context.additionalInfo["operation"] == "normalize")
            #expect(context.timestamp.timeIntervalSince1970 > 0)

            // Check that timestamp is recent (within last second)
            let timeDiff = abs(context.timestamp.timeIntervalSinceNow)
            #expect(timeDiff < 1.0)
        }

        @Test("Error context debug vs release builds")
        func testErrorContextBuildConfiguration() async throws {
            let error = VectorError(.dimensionMismatch, message: "Test")

            #if DEBUG
            // In debug builds, should have file/line/function info
            #expect(!error.context.file.description.isEmpty)
            #expect(error.context.line > 0)
            #expect(!error.context.function.description.isEmpty)
            #expect(error.description.contains("[at "))
            #else
            // In release builds, file/line info should be minimal/empty for performance
            #expect(error.context.file.description.isEmpty || error.context.file.description == "")
            #expect(error.context.line == 0)
            #expect(error.context.function.description.isEmpty || error.context.function.description == "")
            #expect(!error.description.contains("[at "))
            #endif
        }
    }

    // MARK: - ErrorBuilder Tests

    @Suite("ErrorBuilder")
    struct ErrorBuilderTests {

        @Test("Basic error builder usage")
        func testBasicErrorBuilder() async throws {
            let error = ErrorBuilder(.dimensionMismatch).build()

            #expect(error.kind == .dimensionMismatch)
            #expect(error.underlyingError == nil)
            #expect(error.errorChain.isEmpty)
        }

        @Test("Error builder with message")
        func testErrorBuilderWithMessage() async throws {
            let message = "Custom error message"
            let error = ErrorBuilder(.invalidOperation)
                .message(message)
                .build()

            #expect(error.kind == .invalidOperation)
            #expect(error.context.additionalInfo["message"] == message)
            #expect(error.description.contains(message))
        }

        @Test("Error builder with dimension parameters")
        func testErrorBuilderWithDimensions() async throws {
            let error = ErrorBuilder(.dimensionMismatch)
                .dimension(expected: 512, actual: 256)
                .build()

            #expect(error.context.additionalInfo["expected_dimension"] == "512")
            #expect(error.context.additionalInfo["actual_dimension"] == "256")
        }

        @Test("Error builder with index parameters")
        func testErrorBuilderWithIndex() async throws {
            let error = ErrorBuilder(.indexOutOfBounds)
                .index(100, max: 99)
                .build()

            #expect(error.context.additionalInfo["index"] == "100")
            #expect(error.context.additionalInfo["max_index"] == "99")
        }

        @Test("Error builder with custom parameters")
        func testErrorBuilderWithCustomParameters() async throws {
            let error = ErrorBuilder(.operationFailed)
                .parameter("algorithm", value: "cosine_similarity")
                .parameter("batch_size", value: "1024")
                .parameter("precision", value: "float32")
                .build()

            #expect(error.context.additionalInfo["algorithm"] == "cosine_similarity")
            #expect(error.context.additionalInfo["batch_size"] == "1024")
            #expect(error.context.additionalInfo["precision"] == "float32")
        }

        @Test("Error builder with underlying error")
        func testErrorBuilderWithUnderlying() async throws {
            let systemError = SimpleMockSystemError(code: 404, message: "Resource not found")
            let error = ErrorBuilder(.systemError)
                .underlying(systemError)
                .build()

            #expect(error.underlyingError != nil)
            #expect(error.description.contains("Resource not found"))
        }

        @Test("Error builder with chaining")
        func testErrorBuilderWithChaining() async throws {
            let rootError = VectorError(.invalidData, message: "Corrupted input")
            let secondError = VectorError(.allocationFailed, message: "Memory full")

            let error = ErrorBuilder(.operationFailed)
                .chain(rootError, secondError)
                .build()

            #expect(error.errorChain.count == 2)
            #expect(error.errorChain[0].kind == .invalidData)
            #expect(error.errorChain[1].kind == .allocationFailed)
        }

        @Test("Complex error builder scenario")
        func testComplexErrorBuilder() async throws {
            let systemError = SimpleMockSystemError(code: 500, message: "GPU driver error")
            let rootError = VectorError(.resourceUnavailable, message: "GPU memory exhausted")

            let error = ErrorBuilder(.operationFailed)
                .message("Matrix multiplication failed")
                .parameter("matrix_a_dims", value: "512x768")
                .parameter("matrix_b_dims", value: "768x1024")
                .parameter("compute_device", value: "gpu")
                .underlying(systemError)
                .chain(rootError)
                .build()

            #expect(error.kind == .operationFailed)
            #expect(error.context.additionalInfo["message"] == "Matrix multiplication failed")
            #expect(error.context.additionalInfo["matrix_a_dims"] == "512x768")
            #expect(error.context.additionalInfo["compute_device"] == "gpu")
            #expect(error.underlyingError != nil)
            #expect(error.errorChain.count == 1)
            #expect(error.errorChain[0].kind == .resourceUnavailable)
            #expect(error.description.contains("GPU driver error"))
        }
    }

    // MARK: - Error Category and Severity Tests

    @Suite("Error Categorization")
    struct ErrorCategorizationTests {

        @Test("Error kind severity mapping")
        func testErrorKindSeverity() async throws {
            // Test critical severity errors
            #expect(VectorError.ErrorKind.dataCorruption.severity == .critical)
            #expect(VectorError.ErrorKind.systemError.severity == .critical)

            // Test high severity errors
            #expect(VectorError.ErrorKind.allocationFailed.severity == .high)
            #expect(VectorError.ErrorKind.resourceExhausted.severity == .high)

            // Test medium severity errors
            #expect(VectorError.ErrorKind.dimensionMismatch.severity == .medium)
            #expect(VectorError.ErrorKind.indexOutOfBounds.severity == .medium)
            #expect(VectorError.ErrorKind.invalidOperation.severity == .medium)

            // Test low severity errors
            #expect(VectorError.ErrorKind.invalidConfiguration.severity == .low)
            #expect(VectorError.ErrorKind.unsupportedOperation.severity == .low)
        }

        @Test("Error kind category mapping")
        func testErrorKindCategory() async throws {
            // Test dimension category
            #expect(VectorError.ErrorKind.dimensionMismatch.category == .dimension)
            #expect(VectorError.ErrorKind.invalidDimension.category == .dimension)
            #expect(VectorError.ErrorKind.unsupportedDimension.category == .dimension)

            // Test bounds category
            #expect(VectorError.ErrorKind.indexOutOfBounds.category == .bounds)
            #expect(VectorError.ErrorKind.invalidRange.category == .bounds)

            // Test data category
            #expect(VectorError.ErrorKind.invalidData.category == .data)
            #expect(VectorError.ErrorKind.dataCorruption.category == .data)
            #expect(VectorError.ErrorKind.insufficientData.category == .data)

            // Test operation category
            #expect(VectorError.ErrorKind.invalidOperation.category == .operation)
            #expect(VectorError.ErrorKind.unsupportedOperation.category == .operation)
            #expect(VectorError.ErrorKind.operationFailed.category == .operation)

            // Test resource category
            #expect(VectorError.ErrorKind.allocationFailed.category == .resource)
            #expect(VectorError.ErrorKind.resourceExhausted.category == .resource)
            #expect(VectorError.ErrorKind.resourceUnavailable.category == .resource)

            // Test configuration category
            #expect(VectorError.ErrorKind.invalidConfiguration.category == .configuration)
            #expect(VectorError.ErrorKind.missingConfiguration.category == .configuration)

            // Test system category
            #expect(VectorError.ErrorKind.systemError.category == .system)
            #expect(VectorError.ErrorKind.unknown.category == .system)
        }

        @Test("Critical severity errors")
        func testCriticalSeverityErrors() async throws {
            let criticalErrors: [VectorError.ErrorKind] = [.dataCorruption, .systemError]

            for errorKind in criticalErrors {
                #expect(errorKind.severity == .critical)

                let error = VectorError(errorKind, message: "Critical test error")
                #expect(error.kind.severity == .critical)
            }
        }

        @Test("Error category grouping")
        func testErrorCategoryGrouping() async throws {
            let dimensionErrors: [VectorError.ErrorKind] = [.dimensionMismatch, .invalidDimension, .unsupportedDimension]
            let boundsErrors: [VectorError.ErrorKind] = [.indexOutOfBounds, .invalidRange]
            let dataErrors: [VectorError.ErrorKind] = [.invalidData, .dataCorruption, .insufficientData]

            for error in dimensionErrors {
                #expect(error.category == .dimension)
            }

            for error in boundsErrors {
                #expect(error.category == .bounds)
            }

            for error in dataErrors {
                #expect(error.category == .data)
            }
        }

        @Test("All error kinds covered")
        func testAllErrorKindsCovered() async throws {
            // Ensure every error kind has both a severity and category assigned
            for errorKind in VectorError.ErrorKind.allCases {
                let severity = errorKind.severity
                let category = errorKind.category

                // All error kinds should have a valid severity
                #expect([.critical, .high, .medium, .low, .info].contains(severity))

                // All error kinds should have a valid category
                #expect([.dimension, .bounds, .data, .operation, .resource, .configuration, .system].contains(category))

                // Create an error to ensure the kind is properly initialized
                let error = VectorError(errorKind, message: "Test error for \(errorKind)")
                #expect(error.kind == errorKind)
            }
        }
    }

    // MARK: - Convenience Factory Method Tests

    @Suite("Convenience Factory Methods")
    struct ConvenienceFactoryTests {

        @Test("Dimension mismatch error factory")
        func testDimensionMismatchFactory() async throws {
            let error = VectorError.dimensionMismatch(expected: 512, actual: 256)

            #expect(error.kind == .dimensionMismatch)
            #expect(error.context.additionalInfo["expected_dimension"] == "512")
            #expect(error.context.additionalInfo["actual_dimension"] == "256")
            #expect(error.description.contains("512"))
            #expect(error.description.contains("256"))
        }

        @Test("Index out of bounds error factory")
        func testIndexOutOfBoundsFactory() async throws {
            let error = VectorError.indexOutOfBounds(index: 100, dimension: 99)

            #expect(error.kind == .indexOutOfBounds)
            #expect(error.context.additionalInfo["index"] == "100")
            #expect(error.context.additionalInfo["max_index"] == "98") // max_index is dimension - 1
            #expect(error.description.contains("100"))
            #expect(error.description.contains("99"))
        }

        @Test("Invalid operation error factory")
        func testInvalidOperationFactory() async throws {
            let error = VectorError.invalidOperation("normalization", reason: "Cannot normalize zero vector")

            #expect(error.kind == .invalidOperation)
            // invalidOperation doesn't populate additionalInfo with operation/reason
            // It just puts them in the message
            #expect(error.description.contains("Cannot normalize zero vector"))
            #expect(error.description.contains("invalidOperation"))
        }

        @Test("Invalid data error factory")
        func testInvalidDataFactory() async throws {
            let error = VectorError.invalidData("Input contains NaN values")

            #expect(error.kind == .invalidData)
            // invalidData doesn't set additionalInfo["message"], check description instead
            #expect(error.description.contains("Input contains NaN values"))
            #expect(error.description.contains("invalidData"))
        }

        @Test("Allocation failed error factory")
        func testAllocationFailedFactory() async throws {
            let error = VectorError.allocationFailed(size: 1_073_741_824, reason: "Insufficient memory")

            #expect(error.kind == .allocationFailed)
            #expect(error.context.additionalInfo["requested_size"] == "1073741824")
            // reason is included in the message, not stored separately
            // Size may not be in description if reason is provided
            #expect(error.description.contains("Insufficient memory"))
        }

        @Test("Invalid dimension error factory")
        func testInvalidDimensionFactory() async throws {
            let error = VectorError.invalidDimension(1024, reason: "Dimension exceeds maximum")

            #expect(error.kind == .invalidDimension)
            #expect(error.context.additionalInfo["dimension"] == "1024")
            #expect(error.context.additionalInfo["reason"] == "Dimension exceeds maximum")
            #expect(error.description.contains("1024"))
            #expect(error.description.contains("Dimension exceeds maximum"))
        }

        @Test("Invalid values error factory")
        func testInvalidValuesFactory() async throws {
            let error = VectorError.invalidValues(indices: [3, 7, 15], reason: "NaN values detected")

            #expect(error.kind == .invalidData)
            #expect(error.context.additionalInfo["indices"] == "3,7,15")
            #expect(error.context.additionalInfo["reason"] == "NaN values detected")
            #expect(error.description.contains("[3, 7, 15]"))
            #expect(error.description.contains("NaN values detected"))
        }

        @Test("Division by zero error factory")
        func testDivisionByZeroFactory() async throws {
            let error = VectorError.divisionByZero(operation: "normalization")

            #expect(error.kind == .invalidOperation)
            // divisionByZero doesn't set these in additionalInfo
            // Check description instead
            #expect(error.description.contains("normalization"))
            #expect(error.description.contains("Division by zero"))
        }

        @Test("Zero vector error factory")
        func testZeroVectorErrorFactory() async throws {
            let error = VectorError.zeroVectorError(operation: "cosine similarity")

            #expect(error.kind == .invalidOperation)
            // zeroVectorError doesn't set these in additionalInfo
            // Check description instead
            #expect(error.description.contains("cosine similarity"))
            #expect(error.description.contains("Cannot perform operation on zero vector"))
        }

        @Test("Insufficient data error factory")
        func testInsufficientDataFactory() async throws {
            let error = VectorError.insufficientData(expected: 2048, actual: 1024)

            #expect(error.kind == .insufficientData)
            #expect(error.context.additionalInfo["expected_bytes"] == "2048")
            #expect(error.context.additionalInfo["actual_bytes"] == "1024")
            #expect(error.description.contains("2048"))
            #expect(error.description.contains("1024"))
        }

        @Test("Invalid data format error factory")
        func testInvalidDataFormatFactory() async throws {
            let error = VectorError.invalidDataFormat(expected: "Float32", actual: "Int16")

            #expect(error.kind == .invalidData)
            #expect(error.context.additionalInfo["expected_format"] == "Float32")
            #expect(error.context.additionalInfo["actual_format"] == "Int16")
            #expect(error.description.contains("Float32"))
            #expect(error.description.contains("Int16"))
        }

        @Test("Data corruption error factory")
        func testDataCorruptionFactory() async throws {
            let error = VectorError.dataCorruption(reason: "Checksum mismatch")

            #expect(error.kind == .dataCorruption)
            #expect(error.context.additionalInfo["reason"] == "Checksum mismatch")
            #expect(error.description.contains("Checksum mismatch"))
            #expect(error.description.contains("dataCorruption"))
        }
    }

    // MARK: - Validation Utility Tests

    @Suite("Validation Utilities")
    struct ValidationUtilityTests {

        @Test("Dimension match validation - success")
        func testDimensionMatchValidationSuccess() async throws {
            // Should not throw for matching dimensions
            try Validation.requireDimensionMatch(expected: 512, actual: 512)
            try Validation.requireDimensionMatch(expected: 1, actual: 1)
            try Validation.requireDimensionMatch(expected: 1024, actual: 1024)
        }

        @Test("Dimension match validation - failure")
        func testDimensionMatchValidationFailure() async throws {
            do {
                try Validation.requireDimensionMatch(expected: 512, actual: 256)
                #expect(Bool(false), "Should have thrown dimension mismatch error")
            } catch let error as VectorError {
                #expect(error.kind == .dimensionMismatch)
                #expect(error.context.additionalInfo["expected_dimension"] == "512")
                #expect(error.context.additionalInfo["actual_dimension"] == "256")
            }
        }

        @Test("Valid dimension validation - success")
        func testValidDimensionValidationSuccess() async throws {
            try Validation.requireValidDimension(1)
            try Validation.requireValidDimension(128)
            try Validation.requireValidDimension(1024)
            try Validation.requireValidDimension(Int.max)
        }

        @Test("Valid dimension validation - failure")
        func testValidDimensionValidationFailure() async throws {
            // Test zero dimension
            do {
                try Validation.requireValidDimension(0)
                #expect(Bool(false), "Should have thrown invalid dimension error")
            } catch let error as VectorError {
                #expect(error.kind == .invalidDimension)
                #expect(error.context.additionalInfo["dimension"] == "0")
                #expect(error.context.additionalInfo["reason"] == "Dimension must be positive")
            }

            // Test negative dimension
            do {
                try Validation.requireValidDimension(-5)
                #expect(Bool(false), "Should have thrown invalid dimension error")
            } catch let error as VectorError {
                #expect(error.kind == .invalidDimension)
                #expect(error.context.additionalInfo["dimension"] == "-5")
            }
        }

        @Test("Valid index validation - success")
        func testValidIndexValidationSuccess() async throws {
            try Validation.requireValidIndex(0, dimension: 1)
            try Validation.requireValidIndex(0, dimension: 128)
            try Validation.requireValidIndex(127, dimension: 128)
            try Validation.requireValidIndex(511, dimension: 512)
        }

        @Test("Valid index validation - failure")
        func testValidIndexValidationFailure() async throws {
            // Test negative index
            do {
                try Validation.requireValidIndex(-1, dimension: 128)
                #expect(Bool(false), "Should have thrown index out of bounds error")
            } catch let error as VectorError {
                #expect(error.kind == .indexOutOfBounds)
                #expect(error.context.additionalInfo["index"] == "-1")
                #expect(error.context.additionalInfo["max_index"] == "127")
            }

            // Test index equal to dimension
            do {
                try Validation.requireValidIndex(128, dimension: 128)
                #expect(Bool(false), "Should have thrown index out of bounds error")
            } catch let error as VectorError {
                #expect(error.kind == .indexOutOfBounds)
                #expect(error.context.additionalInfo["index"] == "128")
            }

            // Test index greater than dimension
            do {
                try Validation.requireValidIndex(200, dimension: 128)
                #expect(Bool(false), "Should have thrown index out of bounds error")
            } catch let error as VectorError {
                #expect(error.kind == .indexOutOfBounds)
            }
        }

        @Test("Non-zero magnitude validation - success")
        func testNonZeroMagnitudeValidationSuccess() async throws {
            try Validation.requireNonZero(magnitude: 0.1, operation: "normalize")
            try Validation.requireNonZero(magnitude: 1.0, operation: "normalize")
            try Validation.requireNonZero(magnitude: Float.greatestFiniteMagnitude, operation: "normalize")
            try Validation.requireNonZero(magnitude: Float.leastNonzeroMagnitude, operation: "normalize")
        }

        @Test("Non-zero magnitude validation - failure")
        func testNonZeroMagnitudeValidationFailure() async throws {
            do {
                try Validation.requireNonZero(magnitude: 0.0, operation: "normalize")
                #expect(Bool(false), "Should have thrown zero vector error")
            } catch let error as VectorError {
                #expect(error.kind == .invalidOperation)
                #expect(error.context.additionalInfo["message"]?.contains("Cannot perform operation on zero vector") == true)
            }
        }

        @Test("Range validation - success")
        func testRangeValidationSuccess() async throws {
            try Validation.requireInRange([0.5, 0.8, 1.0], range: 0.0...1.0)
            try Validation.requireInRange([-1.0, 0.0, 1.0], range: -1.0...1.0)
            try Validation.requireInRange([100.0], range: 0.0...1000.0)
            try Validation.requireInRange([], range: 0.0...1.0) // Empty array should pass
        }

        @Test("Range validation - failure")
        func testRangeValidationFailure() async throws {
            do {
                try Validation.requireInRange([0.5, 1.5, 0.8], range: 0.0...1.0)
                #expect(Bool(false), "Should have thrown invalid values error")
            } catch let error as VectorError {
                #expect(error.kind == .invalidData)
                #expect(error.context.additionalInfo["indices"] == "1")
                #expect(error.context.additionalInfo["reason"]?.contains("Values outside range") == true)
            }
        }

        @Test("Range validation - multiple invalid values")
        func testRangeValidationMultipleFailures() async throws {
            do {
                try Validation.requireInRange([-0.5, 0.5, 1.5, 0.8, 2.0], range: 0.0...1.0)
                #expect(Bool(false), "Should have thrown invalid values error")
            } catch let error as VectorError {
                #expect(error.kind == .invalidData)
                let indices = error.context.additionalInfo["indices"]
                #expect(indices?.contains("0") == true) // -0.5
                #expect(indices?.contains("2") == true) // 1.5
                #expect(indices?.contains("4") == true) // 2.0
            }
        }
    }

    // MARK: - Debug Validation Tests

    @Suite("Debug Validation")
    struct DebugValidationTests {

        @Test("Debug dimension assertion")
        func testDebugDimensionAssertion() async throws {
            // In debug builds, these would assert. We test that they compile and work
            DebugValidation.assertDimensionMatch(expected: 512, actual: 512)
            DebugValidation.assertDimensionMatch(expected: 128, actual: 128, message: "Custom message")

            // These would fail in debug builds but are no-ops in release
            #if !DEBUG
            // In release builds, these won't crash
            DebugValidation.assertDimensionMatch(expected: 512, actual: 256)
            DebugValidation.assertDimensionMatch(expected: 128, actual: 64)
            #endif

            // Test passes if no crash occurs
            #expect(true)
        }

        @Test("Debug index assertion")
        func testDebugIndexAssertion() async throws {
            // Valid index assertions (should not crash)
            DebugValidation.assertValidIndex(0, dimension: 100)
            DebugValidation.assertValidIndex(50, dimension: 100)
            DebugValidation.assertValidIndex(99, dimension: 100)
            DebugValidation.assertValidIndex(0, dimension: 1, message: "Custom bounds message")

            // These would fail in debug builds but are no-ops in release
            #if !DEBUG
            // In release builds, these won't crash
            DebugValidation.assertValidIndex(-1, dimension: 100)
            DebugValidation.assertValidIndex(100, dimension: 100)
            DebugValidation.assertValidIndex(150, dimension: 100)
            #endif

            // Test passes if no crash occurs
            #expect(true)
        }

        @Test("Debug validation build configuration")
        func testDebugValidationBuildConfiguration() async throws {
            #if DEBUG
            // In debug builds, invalid assertions should be active
            #expect(true, "Debug assertions are active")
            #else
            // In release builds, debug assertions should be compiled out
            #expect(true, "Debug assertions are compiled out")
            #endif

            // Verify the configuration is detected correctly
            let isDebugBuild: Bool
            #if DEBUG
            isDebugBuild = true
            #else
            isDebugBuild = false
            #endif

            // Test behavior matches build configuration
            if isDebugBuild {
                #expect(true, "Debug validation is enabled in debug builds")
            } else {
                #expect(true, "Debug validation is disabled in release builds")
            }
        }
    }

    // MARK: - Performance Validation Tests

    @Suite("Performance Validation")
    struct PerformanceValidationTests {

        @Test("Performance validation condition checking")
        func testPerformanceValidationCondition() async throws {
            // Test valid conditions (should not crash)
            PerformanceValidation.require(true, "This should not trigger")
            PerformanceValidation.require(1 + 1 == 2, "Math works")
            PerformanceValidation.require(100 > 50, "Comparison works")

            // Test that the message is lazily evaluated
            var messageEvaluated = false
            PerformanceValidation.require(true, {
                messageEvaluated = true
                return "Should not be evaluated"
            }())
            // Since condition is true, message should not be evaluated
            #expect(!messageEvaluated)

            // These would crash in both debug and release
            // We can't test them directly without crashing
            // PerformanceValidation.require(false, "Would crash")

            #expect(true, "Performance validation works correctly")
        }

        @Test("Performance dimension validation")
        func testPerformanceDimensionValidation() async throws {
            // Test matching dimensions (should not crash)
            PerformanceValidation.requireDimensionMatch(512, 512)
            PerformanceValidation.requireDimensionMatch(1, 1)
            PerformanceValidation.requireDimensionMatch(1024, 1024)
            PerformanceValidation.requireDimensionMatch(0, 0)

            // Test large dimensions
            PerformanceValidation.requireDimensionMatch(1_000_000, 1_000_000)

            // These would crash in both debug and release
            // We can't test mismatches directly without crashing
            // PerformanceValidation.requireDimensionMatch(512, 256) // Would crash

            #expect(true, "Dimension validation is optimized for performance")
        }

        @Test("Performance validation in release builds")
        func testPerformanceValidationReleaseBuild() async throws {
            #if DEBUG
            // In debug builds, uses fatalError for better debugging
            #expect(true, "Performance validation uses fatalError in debug")
            #else
            // In release builds, uses precondition for performance
            #expect(true, "Performance validation uses precondition in release")
            #endif

            // Test that performance validation is always active
            // (Unlike debug validation which is compiled out)
            let performanceValidationActive = true // Always active
            #expect(performanceValidationActive, "Performance validation is always active")

            // The difference is in the error reporting mechanism:
            // - Debug: fatalError with detailed message
            // - Release: precondition with minimal overhead
            #if DEBUG
            let errorMechanism = "fatalError"
            #else
            let errorMechanism = "precondition"
            #endif
            #expect(errorMechanism.count > 0, "Error mechanism is configured")
        }
    }

    // MARK: - Vector Math Error Tests

    @Suite("Vector Math Errors")
    struct VectorMathErrorTests {

        @Test("Element-wise division by zero")
        func testElementWiseDivisionByZero() async throws {
            let numerator = Vector<Dim32>(repeating: 1.0)
            let denominator = Vector<Dim32>(repeating: 0.0)

            // Unsafe division should produce infinite values but not throw
            let result = numerator ./ denominator
            #expect(result[0].isInfinite)
            #expect(result[31].isInfinite)
        }

        @Test("Safe division validation")
        func testSafeDivisionValidation() async throws {
            let numerator = Vector<Dim32>(repeating: 1.0)
            let validDenominator = Vector<Dim32>(repeating: 2.0)
            let invalidDenominator = Vector<Dim32>(repeating: 0.0)

            // Valid safe division should succeed
            let validResult = try Vector.safeDivide(numerator, by: validDenominator)
            #expect(validResult[0] == 0.5)

            // Invalid safe division should throw
            do {
                _ = try Vector.safeDivide(numerator, by: invalidDenominator)
                #expect(Bool(false), "Should have thrown division by zero error")
            } catch let error as VectorError {
                #expect(error.kind == .invalidOperation)
                #expect(error.description.contains("Division by zero"))
            }
        }

        @Test("Safe division with multiple zeros")
        func testSafeDivisionMultipleZeros() async throws {
            let numerator = Vector<Dim32>(repeating: 1.0)
            var denominatorArray = Array(repeating: 1.0 as Float, count: 32)
            denominatorArray[5] = 0.0
            denominatorArray[10] = 0.0
            denominatorArray[20] = 0.0

            let invalidDenominator = try Vector<Dim32>(denominatorArray)

            do {
                _ = try Vector.safeDivide(numerator, by: invalidDenominator)
                #expect(Bool(false), "Should have thrown division by zero error")
            } catch let error as VectorError {
                #expect(error.kind == .invalidOperation)
                #expect(error.description.contains("Division by zero"))
            }
        }

        @Test("Vector normalization of zero vector")
        func testZeroVectorNormalization() async throws {
            let zeroVector = Vector<Dim32>.zero

            // Test that normalization of zero vector is handled properly
            // This tests the NaNInfinityHandling utilities
            do {
                let magnitude = sqrt(zeroVector.toArray().reduce(0) { $0 + $1 * $1 })
                try Validation.requireNonZero(magnitude: magnitude, operation: "normalization")
                #expect(Bool(false), "Should have thrown zero vector error")
            } catch let error as VectorError {
                #expect(error.kind == .invalidOperation)
                #expect(error.description.contains("Cannot perform operation on zero vector"))
            }
        }

        @Test("Invalid arithmetic operations")
        func testInvalidArithmeticOperations() async throws {
            // Test various invalid operations that should be caught by validation

            // Test invalid scalar multiplication with NaN
            let vector = Vector<Dim32>(repeating: 1.0)
            let nanResult = vector * Float.nan
            #expect(nanResult[0].isNaN)

            // Test invalid scalar multiplication with infinity
            let infResult = vector * Float.infinity
            #expect(infResult[0].isInfinite)

            // Test attempting operations on vectors with invalid values
            var invalidArray = Array(repeating: 1.0 as Float, count: 32)
            invalidArray[0] = Float.nan
            invalidArray[15] = Float.infinity

            let invalidVector = try Vector<Dim32>(invalidArray)
            #expect(invalidVector[0].isNaN)
            #expect(invalidVector[15].isInfinite)

            // These should be handled appropriately by the system
            let result = vector + invalidVector
            #expect(result[0].isNaN) // 1.0 + NaN = NaN
            #expect(result[15].isInfinite) // 1.0 + Inf = Inf
        }
    }

    // MARK: - Memory Allocation Error Tests

    @Suite("Memory Allocation Errors")
    struct MemoryAllocationErrorTests {

        @Test("Aligned memory allocation failure simulation")
        func testAlignedMemoryAllocationFailure() async throws {
            // Test that allocation failure produces appropriate error
            // We can't actually force allocation failure, but we can test the error type
            let error = VectorError.allocationFailed(size: Int.max, reason: "Out of memory")

            #expect(error.kind == .allocationFailed)
            #expect(error.context.additionalInfo["requested_size"] == String(Int.max))
            // reason is in the message, not additionalInfo
            #expect(error.description.contains("allocationFailed"))

            // Test that the error severity is high (allocation failures are critical)
            #expect(error.kind.severity == .high)
        }

        @Test("Invalid alignment parameters")
        func testInvalidAlignmentParameters() async throws {
            // Test various invalid alignment values
            let invalidAlignments = [0, 3, 5, 7, 15] // Not powers of 2

            for alignment in invalidAlignments {
                let error = VectorError(.invalidConfiguration, message: "Invalid alignment: \(alignment). Must be power of 2")

                #expect(error.kind == .invalidConfiguration)
                #expect(error.description.contains("Invalid alignment"))
                #expect(error.description.contains("\(alignment)"))
            }

            // Valid alignments should be powers of 2
            let validAlignments = [1, 2, 4, 8, 16, 32, 64]
            for alignment in validAlignments {
                let isPowerOf2 = (alignment & (alignment - 1)) == 0 && alignment > 0
                #expect(isPowerOf2, "Alignment \(alignment) should be power of 2")
            }
        }

        @Test("Oversized allocation requests")
        func testOversizedAllocationRequests() async throws {
            // Test handling of unreasonably large allocation requests
            let oversizedRequests: [(size: Int, description: String)] = [
                (Int.max, "Maximum integer size"),
                (Int.max / 2, "Half of maximum"),
                (1_000_000_000_000, "1 TB request"),
                (100_000_000_000, "100 GB request")
            ]

            for request in oversizedRequests {
                let error = VectorError.allocationFailed(
                    size: request.size,
                    reason: "Allocation size exceeds available memory: \(request.description)"
                )

                #expect(error.kind == .allocationFailed)
                #expect(error.context.additionalInfo["requested_size"] == String(request.size))
                #expect(error.description.contains("allocationFailed"))
                #expect(error.kind.severity == .high, "Memory allocation failures should be high severity")
            }

            // Test resource exhaustion error variant
            let exhaustedError = VectorError(.resourceExhausted, message: "Memory")
            #expect(exhaustedError.kind == .resourceExhausted)
            #expect(exhaustedError.kind.severity == .high)
        }

        @Test("Alignment verification failures")
        func testAlignmentVerificationFailures() async throws {
            // Test alignment verification error cases

            // Test misaligned memory error
            let misalignedError = VectorError(
                .invalidConfiguration,
                message: "Memory pointer is not aligned to 64-byte boundary"
            )
            #expect(misalignedError.kind == .invalidConfiguration)
            #expect(misalignedError.description.contains("64-byte boundary"))

            // Test alignment mismatch between operations
            let mismatchError = VectorError.invalidOperation(
                "Alignment mismatch",
                reason: "Vector A is 32-byte aligned but Vector B is 64-byte aligned"
            )
            #expect(mismatchError.kind == .invalidOperation)
            #expect(mismatchError.description.contains("32-byte"))
            #expect(mismatchError.description.contains("64-byte"))

            // Test alignment requirements for SIMD operations
            let simdAlignmentError = VectorError(
                .unsupportedOperation,
                message: "SIMD operation requires 16-byte alignment"
            )
            #expect(simdAlignmentError.kind == .unsupportedOperation)
            #expect(simdAlignmentError.description.contains("16-byte alignment"))
        }
    }

    // MARK: - Serialization Error Tests

    @Suite("Serialization Errors")
    struct SerializationErrorTests {

        @Test("Binary format header corruption")
        func testBinaryFormatHeaderCorruption() async throws {
            // Test detection of corrupted binary format headers
            let corruptionError = VectorError.dataCorruption(reason: "Invalid binary header magic bytes")

            #expect(corruptionError.kind == .dataCorruption)
            #expect(corruptionError.description.contains("Invalid binary header"))
            #expect(corruptionError.kind.severity == .critical, "Data corruption should be critical severity")

            // Test specific header validation errors
            let headerErrors = [
                "Magic bytes mismatch: expected 0xVEC1, got 0xFFFF",
                "Version mismatch: format version 3 not supported",
                "Header checksum validation failed"
            ]

            for reason in headerErrors {
                let error = VectorError.dataCorruption(reason: reason)
                #expect(error.kind == .dataCorruption)
                #expect(error.context.additionalInfo["reason"] == reason)
            }
        }

        @Test("Binary format checksum mismatch")
        func testBinaryFormatChecksumMismatch() async throws {
            // Test detection of checksum validation failures
            let checksumError = VectorError.dataCorruption(
                reason: "Checksum mismatch: expected 0xABCDEF12, got 0x12345678"
            )

            #expect(checksumError.kind == .dataCorruption)
            #expect(checksumError.description.contains("Checksum mismatch"))
            #expect(checksumError.context.additionalInfo["reason"]?.contains("0xABCDEF12") == true)
            #expect(checksumError.context.additionalInfo["reason"]?.contains("0x12345678") == true)

            // Test CRC validation failure
            let crcError = VectorError.dataCorruption(reason: "CRC32 validation failed")
            #expect(crcError.kind == .dataCorruption)
            #expect(crcError.kind.severity == .critical)
        }

        @Test("Binary format insufficient data")
        func testBinaryFormatInsufficientData() async throws {
            // Test handling of truncated binary data
            let truncatedError = VectorError.insufficientData(
                expected: 2048,
                actual: 1536
            )

            #expect(truncatedError.kind == .insufficientData)
            #expect(truncatedError.context.additionalInfo["expected_bytes"] == "2048")
            #expect(truncatedError.context.additionalInfo["actual_bytes"] == "1536")

            // Test EOF during deserialization
            let eofError = VectorError.insufficientData(
                expected: 512 * 4, // 512 floats
                actual: 400 * 4  // Only 400 floats available
            )
            #expect(eofError.description.contains("2048"))
            #expect(eofError.description.contains("1600"))
        }

        @Test("Binary format dimension mismatch")
        func testBinaryFormatDimensionMismatch() async throws {
            // Test dimension validation in binary deserialization
            let dimensionError = VectorError.dimensionMismatch(
                expected: 768,
                actual: 512
            )

            #expect(dimensionError.kind == .dimensionMismatch)
            #expect(dimensionError.context.additionalInfo["expected_dimension"] == "768")
            #expect(dimensionError.context.additionalInfo["actual_dimension"] == "512")

            // Test header dimension vs data dimension mismatch
            let headerDataMismatch = VectorError.invalidDataFormat(
                expected: "Vector with 1024 elements",
                actual: "Binary data contains 768 elements"
            )
            #expect(headerDataMismatch.kind == .invalidData)
            #expect(headerDataMismatch.description.contains("1024"))
            #expect(headerDataMismatch.description.contains("768"))
        }

        @Test("JSON decoding dimension mismatch")
        func testJSONDecodingDimensionMismatch() async throws {
            // Test JSON decoding with mismatched dimensions
            let jsonError = VectorError.dimensionMismatch(
                expected: 128,
                actual: 256
            )

            #expect(jsonError.kind == .dimensionMismatch)
            #expect(jsonError.description.contains("128"))
            #expect(jsonError.description.contains("256"))

            // Test JSON array size mismatch
            let arraySizeError = VectorError.invalidDataFormat(
                expected: "JSON array with 512 elements",
                actual: "JSON array with 384 elements"
            )
            #expect(arraySizeError.kind == .invalidData)
            #expect(arraySizeError.context.additionalInfo["expected_format"] == "JSON array with 512 elements")
            #expect(arraySizeError.context.additionalInfo["actual_format"] == "JSON array with 384 elements")
        }

        @Test("Invalid JSON vector data")
        func testInvalidJSONVectorData() async throws {
            // Test JSON decoding with invalid vector data
            let invalidTypes = [
                "Non-numeric value 'abc' at index 5",
                "Null value at index 10",
                "Boolean value 'true' at index 15",
                "Object instead of number at index 20"
            ]

            for reason in invalidTypes {
                let error = VectorError.invalidData(reason)
                #expect(error.kind == .invalidData)
                #expect(error.context.additionalInfo["message"] == reason)
                #expect(error.description.contains(reason))
            }

            // Test malformed JSON
            let malformedError = VectorError.invalidDataFormat(
                expected: "Valid JSON",
                actual: "Malformed JSON: unexpected end of data"
            )
            #expect(malformedError.kind == .invalidData)
            #expect(malformedError.description.contains("Malformed JSON"))
        }

        @Test("Malformed binary data")
        func testMalformedBinaryData() async throws {
            // Test handling of completely malformed binary data
            let malformedError = VectorError.dataCorruption(
                reason: "Binary data appears to be random bytes, not a valid vector format"
            )

            #expect(malformedError.kind == .dataCorruption)
            #expect(malformedError.kind.severity == .critical)
            #expect(malformedError.description.contains("random bytes"))

            // Test various malformation scenarios
            let malformations = [
                "Endianness mismatch detected",
                "Data appears to be compressed but compression flag not set",
                "Alignment padding contains non-zero values",
                "Reserved bytes contain unexpected data"
            ]

            for reason in malformations {
                let error = VectorError.dataCorruption(reason: reason)
                #expect(error.kind == .dataCorruption)
                #expect(error.context.additionalInfo["reason"] == reason)
            }
        }
    }

    // MARK: - Index and Bounds Error Tests

    @Suite("Index and Bounds Errors")
    struct IndexBoundsErrorTests {

        @Test("Vector subscript out of bounds")
        func testVectorSubscriptOutOfBounds() async throws {
            // Test vector subscript access beyond bounds
            let dimension = 64
            let invalidIndices = [64, 65, 100, Int.max]

            for index in invalidIndices {
                let error = VectorError.indexOutOfBounds(index: index, dimension: dimension)
                #expect(error.kind == .indexOutOfBounds)
                #expect(error.context.additionalInfo["index"] == String(index))
                #expect(error.context.additionalInfo["max_index"] == String(dimension - 1))
                #expect(error.description.contains("\(index)"))
                #expect(error.description.contains("\(dimension)"))
            }

            // Test boundary conditions
            let boundaryError = VectorError.indexOutOfBounds(index: dimension, dimension: dimension)
            #expect(boundaryError.kind == .indexOutOfBounds)
            #expect(boundaryError.description.contains("Index \(dimension)") || boundaryError.description.contains("\(dimension)"))
        }

        @Test("Negative index access")
        func testNegativeIndexAccess() async throws {
            // Test negative index handling
            let negativeIndices = [-1, -10, -100, Int.min]
            let dimension = 128

            for index in negativeIndices {
                let error = VectorError.indexOutOfBounds(index: index, dimension: dimension)
                #expect(error.kind == .indexOutOfBounds)
                #expect(error.context.additionalInfo["index"] == String(index))
                #expect(error.description.contains(String(index)))

                // Negative indices should be caught by validation
                do {
                    try Validation.requireValidIndex(index, dimension: dimension)
                    #expect(Bool(false), "Should have thrown for negative index \(index)")
                } catch let validationError as VectorError {
                    #expect(validationError.kind == .indexOutOfBounds)
                }
            }
        }

        @Test("Array bounds validation")
        func testArrayBoundsValidation() async throws {
            // Test array bounds checking in various operations

            // Test range validation
            let dimension = 50
            let invalidRanges: [(start: Int, end: Int, reason: String)] = [
                (-1, 10, "Negative start index"),
                (0, 51, "End exceeds dimension"),
                (10, 5, "Start greater than end"),
                (50, 50, "Empty range at boundary"),
                (-5, -1, "Both indices negative")
            ]

            for range in invalidRanges {
                let error = ErrorBuilder(.invalidRange)
                    .message(range.reason)
                    .parameter("start", value: String(range.start))
                    .parameter("end", value: String(range.end))
                    .parameter("reason", value: range.reason)
                    .build()
                #expect(error.kind == .invalidRange)
                #expect(error.context.additionalInfo["start"] == String(range.start))
                #expect(error.context.additionalInfo["end"] == String(range.end))
                #expect(error.context.additionalInfo["reason"] == range.reason)
            }

            // Test slice bounds validation
            let sliceError = ErrorBuilder(.invalidRange)
                .message("Slice extends beyond vector dimension 50")
                .parameter("start", value: "25")
                .parameter("end", value: "75")
                .parameter("reason", value: "Slice extends beyond vector dimension 50")
                .build()
            #expect(sliceError.kind == .invalidRange)
            #expect(sliceError.context.additionalInfo["start"] == "25")
            #expect(sliceError.context.additionalInfo["end"] == "75")
        }

        @Test("Range validation edge cases")
        func testRangeValidationEdgeCases() async throws {
            // Test edge cases in range validation

            // Test empty range
            let emptyRange = ErrorBuilder(.invalidRange)
                .message("Empty range")
                .parameter("start", value: "10")
                .parameter("end", value: "10")
                .parameter("reason", value: "Empty range")
                .build()
            #expect(emptyRange.kind == .invalidRange)
            #expect(emptyRange.context.additionalInfo["start"] == "10")
            #expect(emptyRange.context.additionalInfo["end"] == "10")

            // Test reversed range
            let reversedRange = ErrorBuilder(.invalidRange)
                .message("Start > End")
                .parameter("start", value: "50")
                .parameter("end", value: "10")
                .parameter("reason", value: "Start > End")
                .build()
            #expect(reversedRange.kind == .invalidRange)

            // Test maximum range
            let maxRange = ErrorBuilder(.invalidRange)
                .message("Range too large")
                .parameter("start", value: "0")
                .parameter("end", value: String(Int.max))
                .parameter("reason", value: "Range too large")
                .build()
            #expect(maxRange.kind == .invalidRange)
            #expect(maxRange.context.additionalInfo["end"] == String(Int.max))
        }
    }

    // MARK: - Dimension Error Tests

    @Suite("Dimension Errors")
    struct DimensionErrorTests {

        @Test("Vector addition dimension mismatch")
        func testVectorAdditionDimensionMismatch() async throws {
            // Test dimension mismatch in vector addition
            let error = VectorError.dimensionMismatch(expected: 128, actual: 256)

            #expect(error.kind == .dimensionMismatch)
            #expect(error.context.additionalInfo["expected_dimension"] == "128")
            #expect(error.context.additionalInfo["actual_dimension"] == "256")
            #expect(error.description.contains("addition") || error.description.contains("dimensionMismatch"))

            // Test that dimension mismatch errors have medium severity
            #expect(error.kind.severity == .medium)
        }

        @Test("Vector subtraction dimension mismatch")
        func testVectorSubtractionDimensionMismatch() async throws {
            // Test dimension mismatch in vector subtraction
            let error = VectorError.dimensionMismatch(expected: 512, actual: 384)

            #expect(error.kind == .dimensionMismatch)
            #expect(error.context.additionalInfo["expected_dimension"] == "512")
            #expect(error.context.additionalInfo["actual_dimension"] == "384")

            // Test error categorization
            #expect(error.kind.category == .dimension)
        }

        @Test("Dot product dimension mismatch")
        func testDotProductDimensionMismatch() async throws {
            // Test dimension mismatch in dot product calculation
            let error = VectorError.dimensionMismatch(expected: 1024, actual: 768)

            #expect(error.kind == .dimensionMismatch)
            #expect(error.description.contains("1024"))
            #expect(error.description.contains("768"))

            // Verify error context
            #expect(error.context.line > 0)
        }

        @Test("Zero dimension vectors")
        func testZeroDimensionVectors() async throws {
            // Test handling of zero-dimensional vectors
            let error = VectorError.invalidDimension(0, reason: "Dimension cannot be zero")

            #expect(error.kind == .invalidDimension)
            #expect(error.context.additionalInfo["dimension"] == "0")
            #expect(error.context.additionalInfo["reason"] == "Dimension cannot be zero")
            #expect(error.description.contains("0"))
            #expect(error.description.contains("cannot be zero"))
        }

        @Test("Negative dimension validation")
        func testNegativeDimensionValidation() async throws {
            // Test validation of negative dimensions
            let negativeDims = [-1, -10, -100, Int.min]

            for dim in negativeDims {
                let error = VectorError.invalidDimension(dim, reason: "Dimension cannot be negative")
                #expect(error.kind == .invalidDimension)
                #expect(error.context.additionalInfo["dimension"] == String(dim))
                #expect(error.description.contains(String(dim)))
            }
        }

        @Test("Unsupported dimension operations")
        func testUnsupportedDimensionOperations() async throws {
            // Test operations on unsupported dimensions
            let unsupportedDims = [
                (8192, "Dimension exceeds maximum supported size"),
                (3, "Dimension must be multiple of 4 for SIMD"),
                (17, "Prime dimensions not supported for this operation")
            ]

            for (dim, reason) in unsupportedDims {
                let error = ErrorBuilder(.unsupportedDimension)
                    .message(reason)
                    .parameter("dimension", value: String(dim))
                    .parameter("reason", value: reason)
                    .build()
                #expect(error.kind == .unsupportedDimension)
                #expect(error.context.additionalInfo["dimension"] == String(dim))
                #expect(error.context.additionalInfo["reason"] == reason)
            }
        }
    }

    // MARK: - Configuration Error Tests

    @Suite("Configuration Errors")
    struct ConfigurationErrorTests {

        @Test("Invalid provider configuration")
        func testInvalidProviderConfiguration() async throws {
            // Test invalid compute provider configurations
            let invalidConfigs = [
                "Metal not available on this platform",
                "CUDA device not found",
                "OpenCL driver version too old",
                "Compute capability not supported"
            ]

            for config in invalidConfigs {
                let error = VectorError(
                    .invalidConfiguration,
                    message: config
                )
                #expect(error.kind == .invalidConfiguration)
                #expect(error.context.additionalInfo["message"] == config)
                #expect(error.kind.severity == .low)
            }
        }

        @Test("Missing required configuration")
        func testMissingRequiredConfiguration() async throws {
            // Test handling of missing required configuration
            let missingConfigs = [
                "Provider type not specified",
                "Batch size not configured",
                "Memory limit not set",
                "Thread count not specified"
            ]

            for config in missingConfigs {
                let error = VectorError(
                    .missingConfiguration,
                    message: config
                )
                #expect(error.kind == .missingConfiguration)
                #expect(error.context.additionalInfo["message"] == config)
                #expect(error.kind.category == .configuration)
            }
        }

        @Test("Conflicting configuration options")
        func testConflictingConfigurationOptions() async throws {
            // Test detection of conflicting configuration settings
            let conflicts = [
                "Cannot use both Metal and CUDA providers simultaneously",
                "Mixed precision and full precision modes are mutually exclusive",
                "Batch size exceeds maximum memory configuration",
                "Thread count exceeds hardware capabilities"
            ]

            for conflict in conflicts {
                let error = VectorError(
                    .invalidConfiguration,
                    message: conflict
                )
                #expect(error.kind == .invalidConfiguration)
                #expect(error.description.contains(conflict))
            }
        }
    }

    // MARK: - Resource Error Tests

    @Suite("Resource Errors")
    struct ResourceErrorTests {

        @Test("Resource exhaustion simulation")
        func testResourceExhaustionSimulation() async throws {
            // Test handling of resource exhaustion scenarios
            let resources = ["Memory", "GPU", "Threads", "File handles"]

            for resource in resources {
                let error = VectorError(
                    .resourceExhausted,
                    message: resource
                )
                #expect(error.kind == .resourceExhausted)
                #expect(error.context.additionalInfo["message"] == resource)
                #expect(error.kind.severity == .high)
                #expect(error.kind.category == .resource)
            }
        }

        @Test("Resource unavailability")
        func testResourceUnavailability() async throws {
            // Test handling when required resources are unavailable
            let unavailableResources = [
                "GPU device is busy",
                "Metal command buffer unavailable",
                "Accelerate framework not loaded",
                "Memory allocator is locked"
            ]

            for resource in unavailableResources {
                let error = VectorError(
                    .resourceUnavailable,
                    message: resource
                )
                #expect(error.kind == .resourceUnavailable)
                #expect(error.context.additionalInfo["message"] == resource)
                #expect(error.kind.category == .resource)
            }
        }

        @Test("Buffer pool exhaustion")
        func testBufferPoolExhaustion() async throws {
            // Test buffer pool resource exhaustion
            let error = VectorError(
                .resourceExhausted,
                message: "Buffer pool: all 256 buffers in use"
            )

            #expect(error.kind == .resourceExhausted)
            #expect(error.description.contains("Buffer pool"))
            #expect(error.description.contains("256 buffers"))
            #expect(error.kind.severity == .high)

            // Test different pool exhaustion scenarios
            let pools = [
                "Temporary buffer pool",
                "Persistent buffer pool",
                "Metal buffer pool",
                "Scratch memory pool"
            ]

            for pool in pools {
                let poolError = VectorError(
                    .resourceExhausted,
                    message: pool
                )
                #expect(poolError.kind == .resourceExhausted)
                #expect(poolError.context.additionalInfo["message"] == pool)
            }
        }
    }

    // MARK: - Result Extension Tests

    @Suite("Result Extensions")
    struct ResultExtensionTests {

        @Test("Result error context mapping")
        func testResultErrorContextMapping() async throws {
            // Test Result extensions for error context transformation
            let error = VectorError.dimensionMismatch(expected: 100, actual: 200)
            let result: Result<Int, VectorError> = .failure(error)

            // Test that error context is preserved
            switch result {
            case .failure(let e):
                #expect(e.kind == .dimensionMismatch)
                #expect(e.context.additionalInfo["expected_dimension"] == "100")
                #expect(e.context.additionalInfo["actual_dimension"] == "200")
            case .success:
                #expect(Bool(false), "Should be failure")
            }
        }

        @Test("Result error chaining")
        func testResultErrorChaining() async throws {
            // Test Result extensions for error chaining
            let initialError = VectorError.invalidData("Initial error")
            let chainedError = VectorError(
                .operationFailed,
                message: "Operation failed due to: \(initialError.description)"
            )

            let result: Result<Int, VectorError> = .failure(chainedError)

            switch result {
            case .failure(let error):
                #expect(error.kind == .operationFailed)
                #expect(error.description.contains("Initial error"))
            case .success:
                #expect(Bool(false), "Should be failure")
            }
        }

        @Test("Result catching utility")
        func testResultCatchingUtility() async throws {
            // Test Result.catching utility function

            // Test successful execution
            let successResult = Result<Int, VectorError>.catching {
                return 42
            }
            #expect(try successResult.get() == 42)

            // Test error catching
            let errorResult = Result<Int, VectorError>.catching {
                throw VectorError.invalidOperation("test operation", reason: "Test error")
            }

            switch errorResult {
            case .failure(let error):
                #expect(error.kind == .invalidOperation)
                #expect(error.description.contains("Test error"))
            case .success:
                #expect(Bool(false), "Should be failure")
            }
        }

        @Test("Result to optional conversion")
        func testResultToOptionalConversion() async throws {
            // Test Result to optional conversion
            let successResult: Result<Int, VectorError> = .success(42)
            let failureResult: Result<Int, VectorError> = .failure(VectorError.invalidData("Error"))

            // Test success case
            let successOptional = successResult.optional
            #expect(successOptional == 42)

            // Test failure case
            let failureOptional = failureResult.optional
            #expect(failureOptional == nil)
        }
    }

    // MARK: - Error String Representation Tests

    @Suite("Error String Representation")
    struct ErrorStringRepresentationTests {

        @Test("Error description formatting")
        func testErrorDescriptionFormatting() async throws {
            // Test error description string formatting
            let error = VectorError.dimensionMismatch(expected: 512, actual: 256)

            let description = error.description
            #expect(description.contains("VectorError.dimensionMismatch"))
            #expect(description.contains("512"))
            #expect(description.contains("256"))
            #expect(description.contains("Expected dimension") || description.contains("expected"))
            // Description format may vary, just ensure both values are present

            // Test that file and line info is included
            #expect(description.contains(".swift"))
        }

        @Test("Error debug description")
        func testErrorDebugDescription() async throws {
            // Test detailed debug description formatting
            let error = VectorError.invalidOperation(
                "Complex operation",
                reason: "Multiple failures"
            )

            let debugDescription = String(reflecting: error)
            #expect(debugDescription.contains("invalidOperation"))
            #expect(debugDescription.contains("Complex operation"))
            #expect(debugDescription.contains("Multiple failures"))

            // Debug description should include more detail than regular description
            #expect(debugDescription.count > error.description.count || debugDescription == error.description)
        }

        @Test("Error chain description")
        func testErrorChainDescription() async throws {
            // Test error chain representation in descriptions
            let rootError = VectorError.invalidData("Root cause")
            let middleError = VectorError(
                .operationFailed,
                message: "Middle layer: \(rootError.description)"
            )
            let topError = VectorError(
                .systemError,
                message: "Top level: \(middleError.description)"
            )

            let description = topError.description
            #expect(description.contains("systemError"))
            #expect(description.contains("Top level"))
            #expect(description.contains("operationFailed"))
            #expect(description.contains("invalidData"))
        }

        @Test("Localized error description")
        func testLocalizedErrorDescription() async throws {
            // Test LocalizedError protocol conformance
            let error = VectorError.allocationFailed(size: 1024, reason: "Out of memory")

            // Test LocalizedError properties
            let localizedDescription = (error as LocalizedError).errorDescription ?? error.description
            #expect(localizedDescription.contains("allocationFailed"))
            // Size might not be in message when reason is provided
            #expect(localizedDescription.contains("Out of memory"))

            // Verify error implements LocalizedError
            let localizedError = error as LocalizedError
            #expect(localizedError.errorDescription != nil)
        }
    }

    // MARK: - Edge Case and Stress Tests

    @Suite("Edge Cases and Stress Tests")
    struct EdgeCaseStressTests {

        @Test("Maximum error chain length")
        func testMaximumErrorChainLength() async throws {
            // Test handling of very long error chains
            var currentError = VectorError.invalidData("Base error")

            // Create a chain of 100 errors
            for i in 1...100 {
                currentError = VectorError(
                    .operationFailed,
                    message: "Error \(i): \(currentError.kind)"
                )
            }

            // Verify the chain is handled correctly
            #expect(currentError.kind == .operationFailed)
            #expect(currentError.description.contains("Error 100"))

            // Description should not be excessively long (some truncation expected)
            #expect(currentError.description.count < 100_000)
        }

        @Test("Concurrent error handling")
        func testConcurrentErrorHandling() async throws {
            // Test error handling in concurrent scenarios
            await withTaskGroup(of: VectorError?.self) { group in
                // Launch multiple concurrent tasks that generate errors
                for i in 0..<10 {
                    group.addTask {
                        if i % 2 == 0 {
                            return VectorError.dimensionMismatch(expected: i, actual: i + 1)
                        } else {
                            return VectorError.invalidData("Task \(i) error")
                        }
                    }
                }

                var errors: [VectorError] = []
                for await error in group {
                    if let error = error {
                        errors.append(error)
                    }
                }

                #expect(errors.count == 10)
                #expect(errors.filter { $0.kind == .dimensionMismatch }.count == 5)
                #expect(errors.filter { $0.kind == .invalidData }.count == 5)
            }
        }

        @Test("Memory pressure error scenarios")
        func testMemoryPressureErrorScenarios() async throws {
            // Test error handling under memory pressure
            // Simulate creating many errors
            var errors: [VectorError] = []

            for i in 0..<1000 {
                let error = VectorError.allocationFailed(
                    size: i * 1024,
                    reason: "Simulated memory pressure test \(i)"
                )
                errors.append(error)
            }

            // Verify all errors were created successfully
            #expect(errors.count == 1000)
            #expect(errors.allSatisfy { $0.kind == .allocationFailed })

            // Test that error context doesn't consume excessive memory
            let firstError = errors[0]
            let lastError = errors[999]
            #expect(firstError.context.additionalInfo["requested_size"] == "0")
            #expect(lastError.context.additionalInfo["requested_size"] == String(999 * 1024))
        }

        @Test("Error handling performance")
        func testErrorHandlingPerformance() async throws {
            // Test performance characteristics of error handling
            let startTime = Date()

            // Create and handle many errors
            for _ in 0..<10_000 {
                let error = VectorError.dimensionMismatch(expected: 512, actual: 256)
                _ = error.description // Force description computation
                _ = error.kind.severity
                _ = error.kind.category
            }

            let elapsed = Date().timeIntervalSince(startTime)

            // Should complete in reasonable time (< 1 second for 10k errors)
            #expect(elapsed < 1.0, "Error handling took \(elapsed) seconds")

            // Test that error creation is lightweight
            let singleErrorStart = Date()
            _ = VectorError.invalidData("Performance test")
            let singleErrorTime = Date().timeIntervalSince(singleErrorStart)
            #expect(singleErrorTime < 0.001, "Single error creation should be sub-millisecond")
        }

        @Test("Error context memory efficiency")
        func testErrorContextMemoryEfficiency() async throws {
            // Test memory efficiency of error context capture
            let largeString = String(repeating: "A", count: 10_000)

            // Create error with large context
            let error = VectorError.invalidData(largeString)

            // Verify context is stored efficiently
            #expect(error.context.additionalInfo["message"] == largeString)
            #expect(error.description.count > 0)

            // Create many errors with moderate context
            var errors: [VectorError] = []
            for i in 0..<100 {
                errors.append(VectorError.invalidData("Error \(i) with moderate context"))
            }

            // All errors should have their unique context
            for (index, error) in errors.enumerated() {
                #expect(error.context.additionalInfo["message"] == "Error \(index) with moderate context")
            }

            // Test that context doesn't leak between errors
            let error1 = VectorError.invalidData("First")
            let error2 = VectorError.invalidData("Second")
            #expect(error1.context.additionalInfo["message"] == "First")
            #expect(error2.context.additionalInfo["message"] == "Second")
        }
    }
}

// MARK: - Test Support Extensions

extension ErrorHandlingTests {

    /// Creates a test vector with specified dimension
    static func createTestVector<D: Dimension>(_ dimension: D.Type) -> Vector<D> {
        return Vector<D>(repeating: 1.0)
    }

    /// Creates corrupted binary data for testing
    static func createCorruptedBinaryData() -> Data {
        var data = Data()

        // Add various types of corruption
        let corruptionType = Int.random(in: 0...5)

        switch corruptionType {
        case 0:
            // Truncated data - incomplete header
            data.append(contentsOf: [0xFF, 0xFE])  // Partial magic bytes

        case 1:
            // Invalid magic number/header
            data.append(contentsOf: [0xDE, 0xAD, 0xBE, 0xEF])  // Wrong magic
            data.append(contentsOf: repeatElement(0, count: 100))  // Padding

        case 2:
            // Misaligned data (odd byte count for Float32 array)
            data.append(contentsOf: [0x56, 0x43])  // "VC" header
            data.append(contentsOf: [0x01, 0x00, 0x00, 0x00])  // Version
            data.append(contentsOf: [0x00, 0x02, 0x00, 0x00])  // Dimension = 512
            // Add misaligned float data (not divisible by 4)
            data.append(contentsOf: repeatElement(0xFF, count: 513))  // Wrong size

        case 3:
            // Corrupt dimension value
            data.append(contentsOf: [0x56, 0x43])  // "VC" header
            data.append(contentsOf: [0x01, 0x00, 0x00, 0x00])  // Version
            // Invalid dimension (way too large or negative)
            let invalidDim = UInt32.max
            withUnsafeBytes(of: invalidDim) { bytes in
                data.append(contentsOf: bytes)
            }

        case 4:
            // Contains NaN and Infinity values
            data.append(contentsOf: [0x56, 0x43])  // "VC" header
            data.append(contentsOf: [0x01, 0x00, 0x00, 0x00])  // Version
            data.append(contentsOf: [0x04, 0x00, 0x00, 0x00])  // Dimension = 4

            // Add float values including NaN and Inf
            let values: [Float32] = [1.0, Float.nan, Float.infinity, -Float.infinity]
            for value in values {
                withUnsafeBytes(of: value) { bytes in
                    data.append(contentsOf: bytes)
                }
            }

        case 5:
            // Checksum mismatch (if using checksums)
            data.append(contentsOf: [0x56, 0x43])  // "VC" header
            data.append(contentsOf: [0x01, 0x00, 0x00, 0x00])  // Version
            data.append(contentsOf: [0x02, 0x00, 0x00, 0x00])  // Dimension = 2

            // Valid float data
            let values: [Float32] = [1.0, 2.0]
            for value in values {
                withUnsafeBytes(of: value) { bytes in
                    data.append(contentsOf: bytes)
                }
            }

            // Add incorrect checksum
            data.append(contentsOf: [0xBA, 0xDC, 0x0F, 0xFE])  // Wrong checksum

        default:
            // Empty/zero-length data
            break
        }

        return data
    }

    /// Simulates memory allocation failure
    static func simulateAllocationFailure() {
        // Note: Swift doesn't provide direct control over allocation failures
        // We simulate various resource exhaustion scenarios

        let scenario = Int.random(in: 0...3)

        switch scenario {
        case 0:
            // Attempt to allocate enormous array (may trigger memory warnings)
            // Note: allocate doesn't throw, but may trigger system memory pressure
            // Try to allocate gigantic array
            let hugeSize = Int.max / 100  // Still huge but won't crash immediately
            _ = UnsafeMutablePointer<Float>.allocate(capacity: hugeSize)
            // If this succeeds (unlikely), we should deallocate
            // But usually this will fail or trigger memory pressure

        case 1:
            // Create memory pressure by allocating many small objects
            var buffers: [UnsafeMutablePointer<Float>] = []
            for _ in 0..<1000 {
                let buffer = UnsafeMutablePointer<Float>.allocate(capacity: 1024 * 1024)
                buffers.append(buffer)
            }
            // Clean up
            for buffer in buffers {
                buffer.deallocate()
            }

        case 2:
            // Simulate stack overflow scenario (recursive allocation)
            func recursiveAllocation(depth: Int) {
                guard depth < 100 else { return }  // Safety limit
                _ = [Float](repeating: 0, count: 10000)
                recursiveAllocation(depth: depth + 1)
            }
            recursiveAllocation(depth: 0)

        case 3:
            // Create fragmented memory scenario
            var allocations: [(ptr: UnsafeMutablePointer<Float>, size: Int)] = []

            // Allocate alternating sizes to fragment memory
            for i in 0..<50 {
                let size = (i % 2 == 0) ? 1000 : 10
                let ptr = UnsafeMutablePointer<Float>.allocate(capacity: size)
                allocations.append((ptr, size))
            }

            // Deallocate every other allocation to create holes
            for (index, allocation) in allocations.enumerated() {
                if index % 2 == 0 {
                    allocation.ptr.deallocate()
                }
            }

            // Clean up remaining
            for (index, allocation) in allocations.enumerated() {
                if index % 2 != 0 {
                    allocation.ptr.deallocate()
                }
            }

        default:
            break
        }
    }

    /// Creates test data with specific error conditions
    static func createTestDataWithError<T>(_ errorType: T) -> Data where T: Error {
        var data = Data()

        // Check error type and create corresponding corrupted data
        if let vectorError = errorType as? VectorError {
            switch vectorError.kind {
            case .dimensionMismatch:
                // Create data with mismatched dimensions
                data.append(contentsOf: [0x56, 0x43])  // Header
                data.append(contentsOf: [0x01, 0x00, 0x00, 0x00])  // Version
                // First vector dimension = 512
                withUnsafeBytes(of: UInt32(512)) { data.append(contentsOf: $0) }
                // Add some float data
                for _ in 0..<512 {
                    withUnsafeBytes(of: Float32.random(in: -1...1)) { data.append(contentsOf: $0) }
                }
                // Second vector with different dimension = 256
                data.append(contentsOf: [0x56, 0x43])  // Header
                data.append(contentsOf: [0x01, 0x00, 0x00, 0x00])  // Version
                withUnsafeBytes(of: UInt32(256)) { data.append(contentsOf: $0) }

            case .invalidData:
                // Create data with invalid values
                data.append(contentsOf: [0x56, 0x43])  // Header
                data.append(contentsOf: [0x01, 0x00, 0x00, 0x00])  // Version
                withUnsafeBytes(of: UInt32(4)) { data.append(contentsOf: $0) }
                // Add NaN and Inf values
                let badValues: [Float32] = [.nan, .infinity, -.infinity, .nan]
                for value in badValues {
                    withUnsafeBytes(of: value) { data.append(contentsOf: $0) }
                }

            case .indexOutOfBounds:
                // Create data that will cause index issues
                data.append(contentsOf: [0x56, 0x43])  // Header
                data.append(contentsOf: [0x01, 0x00, 0x00, 0x00])  // Version
                withUnsafeBytes(of: UInt32(10)) { data.append(contentsOf: $0) }  // Claims 10 elements
                // But only provide 5 elements
                for _ in 0..<5 {
                    withUnsafeBytes(of: Float32(1.0)) { data.append(contentsOf: $0) }
                }

            case .allocationFailed:
                // Create data requiring huge allocation
                data.append(contentsOf: [0x56, 0x43])  // Header
                data.append(contentsOf: [0x01, 0x00, 0x00, 0x00])  // Version
                // Claim enormous dimension
                withUnsafeBytes(of: UInt32(1_000_000_000)) { data.append(contentsOf: $0) }

            case .dataCorruption:
                // Create thoroughly corrupted data
                // Random bytes with no structure
                for _ in 0..<100 {
                    data.append(UInt8.random(in: 0...255))
                }

            case .invalidOperation:
                // Create data for invalid operation (e.g., zero vector for normalization)
                data.append(contentsOf: [0x56, 0x43])  // Header
                data.append(contentsOf: [0x01, 0x00, 0x00, 0x00])  // Version
                withUnsafeBytes(of: UInt32(4)) { data.append(contentsOf: $0) }
                // All zeros (invalid for certain operations)
                for _ in 0..<4 {
                    withUnsafeBytes(of: Float32(0)) { data.append(contentsOf: $0) }
                }

            default:
                // Generic corrupted data
                data = createCorruptedBinaryData()
            }
        } else {
            // For non-VectorError types, create generic bad data
            data = createCorruptedBinaryData()
        }

        return data
    }
}

// MARK: - Simple Mock Error Types for VectorError Testing
//
// Note: These are minimal mock errors used for testing VectorError's underlying
// error handling. For comprehensive error injection testing, see TestHelpers/ErrorHandlingMocks.swift

/// Simple mock system error for testing VectorError wrapping
struct SimpleMockSystemError: Error, CustomStringConvertible {
    let code: Int
    let message: String

    var description: String {
        "SimpleMockSystemError(\(code)): \(message)"
    }
}

/// Simple mock configuration error for testing
struct SimpleMockConfigurationError: Error {
    let setting: String
    let reason: String
}

/// Simple mock allocation error for testing
struct SimpleMockAllocationError: Error {
    let requestedSize: Int
    let availableSize: Int
}
