//
//  MockErrorTestExtensions.swift
//  VectorCore
//
//  Test helper extensions for working with MockError types in both
//  XCTest and Swift Testing frameworks.
//

import Foundation
@testable import VectorCore

#if canImport(XCTest)
import XCTest

// =============================================================================
// MARK: - XCTest Integration
// =============================================================================

extension XCTestCase {

    /// Assert that code throws a specific mock error type
    ///
    /// - Parameters:
    ///   - expectedType: Expected error type
    ///   - expression: Code to execute
    ///   - message: Failure message
    ///   - file: Source file (auto-captured)
    ///   - line: Source line (auto-captured)
    ///
    /// ## Example
    /// ```swift
    /// assertThrowsMockError(MockSystemError.self) {
    ///     try await ErrorInjector.shared.checkInjectionPoint("test")
    /// }
    /// ```
    public func assertThrowsMockError<T: MockError>(
        _ expectedType: T.Type,
        _ expression: @autoclosure () throws -> Void,
        _ message: String = "",
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let finalMessage = message.isEmpty ? "Expected to throw \(T.self)" : message

        XCTAssertThrowsError(try expression(), finalMessage, file: file, line: line) { error in
            XCTAssertTrue(
                error is T,
                "Expected \(T.self), got \(type(of: error))",
                file: file,
                line: line
            )
        }
    }

    /// Assert that code throws a specific mock error type (async version)
    public func assertThrowsMockError<T: MockError>(
        _ expectedType: T.Type,
        _ message: String = "",
        file: StaticString = #filePath,
        line: UInt = #line,
        _ expression: @escaping () async throws -> Void
    ) async {
        let finalMessage = message.isEmpty ? "Expected to throw \(T.self)" : message

        do {
            try await expression()
            XCTFail(finalMessage, file: file, line: line)
        } catch {
            XCTAssertTrue(
                error is T,
                "Expected \(T.self), got \(type(of: error))",
                file: file,
                line: line
            )
        }
    }

    /// Assert that error has correct recovery strategy
    ///
    /// - Parameters:
    ///   - error: Mock error to check
    ///   - expected: Expected recovery strategy
    ///   - file: Source file (auto-captured)
    ///   - line: Source line (auto-captured)
    ///
    /// ## Example
    /// ```swift
    /// let error = MockSystemError.simulateOOM()
    /// assertRecoveryStrategy(error, equals: .propagateError)
    /// ```
    public func assertRecoveryStrategy(
        _ error: any MockError,
        equals expected: RecoveryStrategy,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(
            error.suggestedRecovery,
            expected,
            "Expected recovery strategy \(expected), got \(error.suggestedRecovery)",
            file: file,
            line: line
        )
    }

    /// Assert that error is retryable
    public func assertRetryable(
        _ error: any MockError,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertTrue(
            error.isRetryable,
            "Expected error to be retryable: \(error.debugDescription)",
            file: file,
            line: line
        )
    }

    /// Assert that error is not retryable
    public func assertNotRetryable(
        _ error: any MockError,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertFalse(
            error.isRetryable,
            "Expected error to not be retryable: \(error.debugDescription)",
            file: file,
            line: line
        )
    }
}
#endif

// =============================================================================
// MARK: - Swift Testing Integration
// =============================================================================

#if canImport(Testing)
import Testing

/// Verify that an error is a specific mock error type (Swift Testing)
///
/// ## Example
/// ```swift
/// do {
///     try await ErrorInjector.shared.checkInjectionPoint("test")
///     Issue.record("Expected error to be thrown")
/// } catch {
///     expectMockError(error, toBe: MockSystemError.self)
/// }
/// ```
public func expectMockError<T: MockError>(
    _ error: any Error,
    toBe expectedType: T.Type,
    sourceLocation: SourceLocation = #_sourceLocation
) {
    #expect(
        error is T,
        "Expected \(T.self), got \(type(of: error))",
        sourceLocation: sourceLocation
    )
}

/// Verify recovery strategy (Swift Testing)
public func expectRecoveryStrategy(
    _ error: any MockError,
    equals expected: RecoveryStrategy,
    sourceLocation: SourceLocation = #_sourceLocation
) {
    #expect(
        error.suggestedRecovery == expected,
        "Expected recovery strategy \(expected), got \(error.suggestedRecovery)",
        sourceLocation: sourceLocation
    )
}

/// Verify error is retryable (Swift Testing)
public func expectRetryable(
    _ error: any MockError,
    sourceLocation: SourceLocation = #_sourceLocation
) {
    #expect(
        error.isRetryable,
        "Expected error to be retryable: \(error.debugDescription)",
        sourceLocation: sourceLocation
    )
}

/// Verify error is not retryable (Swift Testing)
public func expectNotRetryable(
    _ error: any MockError,
    sourceLocation: SourceLocation = #_sourceLocation
) {
    #expect(
        !error.isRetryable,
        "Expected error to not be retryable: \(error.debugDescription)",
        sourceLocation: sourceLocation
    )
}
#endif
