//
//  ErrorHandlingMocksTests.swift
//  VectorCore
//
//  Comprehensive test suite for error handling mock types and error injection framework.
//  Validates Kernel Spec #31 implementation.
//

import Testing
import Foundation
@testable import VectorCore

// =============================================================================
// MARK: - Error Handling Mocks Test Suite
// =============================================================================

@Suite("Error Handling Mocks")
struct ErrorHandlingMocksTests {

    // Reset error injector before each test
    init() async {
        await ErrorInjector.shared.reset()
    }

    // MARK: - Mock System Error Tests

    @Suite("System Error Mocks")
    struct SystemErrorTests {

        @Test("Out-of-memory error creation and properties")
        func testOutOfMemoryError() {
            let oom = MockSystemError.simulateOOM()

            #expect(!oom.isRetryable)
            #expect(oom.suggestedRecovery == .propagateError)

            if case .outOfMemory(let attempted, let available) = oom {
                #expect(attempted > available, "Attempted allocation should exceed available memory")
                #expect(attempted == 1_000_000_000)
            } else {
                Issue.record("Expected outOfMemory case")
            }
        }

        @Test("Disk full error creation")
        func testDiskFullError() {
            let diskFull = MockSystemError.simulateDiskFull()

            #expect(!diskFull.isRetryable)
            #expect(diskFull.suggestedRecovery == .propagateError)

            if case .diskFull(let required, let available) = diskFull {
                #expect(required > available, "Required space should exceed available")
            } else {
                Issue.record("Expected diskFull case")
            }
        }

        @Test("Resource lock error is retryable")
        func testResourceLockError() {
            let lock = MockSystemError.simulateResourceLock()

            #expect(lock.isRetryable)
            #expect(lock.suggestedRecovery == .retry)

            if case .resourceLocked(let resource, let owner) = lock {
                #expect(resource == "vector_cache")
                #expect(owner.hasPrefix("Thread-"))
            } else {
                Issue.record("Expected resourceLocked case")
            }
        }

        @Test("Timeout error is retryable")
        func testTimeoutError() {
            let timeout = MockSystemError.timeoutExpired(operation: "clustering", timeout: 30.0)

            #expect(timeout.isRetryable)
            #expect(timeout.suggestedRecovery == .retry)
        }

        @Test("Permission denied error")
        func testPermissionDeniedError() {
            let permission = MockSystemError.permissionDenied(resource: "/data/vectors", operation: "write")

            #expect(!permission.isRetryable)
            #expect(permission.suggestedRecovery == .propagateError)
        }

        @Test("Process limit exceeded error")
        func testProcessLimitError() {
            let limit = MockSystemError.processLimitExceeded(limit: 100, current: 150)

            #expect(!limit.isRetryable)
            #expect(limit.suggestedRecovery == .fallbackToDefault)
        }

        @Test("Debug description contains useful information")
        func testDebugDescription() {
            let oom = MockSystemError.outOfMemory(attemptedAllocation: 1000, availableMemory: 500)

            let description = oom.debugDescription
            #expect(description.contains("1000"))
            #expect(description.contains("500"))
            #expect(description.contains("MockSystemError.outOfMemory"))
        }
    }

    // MARK: - Mock Configuration Error Tests

    @Suite("Configuration Error Mocks")
    struct ConfigurationErrorTests {

        @Test("Dimension mismatch error")
        func testDimensionMismatch() {
            let dimError = MockConfigurationError.simulateDimensionMismatch(expected: 512, got: 768)

            #expect(!dimError.isRetryable)
            #expect(dimError.suggestedRecovery == .propagateError)

            if case .invalidDimension(let expected, let got) = dimError {
                #expect(expected == 512)
                #expect(got == 768)
            } else {
                Issue.record("Expected invalidDimension case")
            }
        }

        @Test("Invalid cluster count error")
        func testInvalidClusterCount() {
            let clusterError = MockConfigurationError.simulateInvalidClusterCount(k: 0)

            #expect(!clusterError.isRetryable)
            #expect(clusterError.suggestedRecovery == .fallbackToDefault)

            if case .invalidParameter(let param, let value, _) = clusterError {
                #expect(param == "k")
                #expect(value == "0")
            } else {
                Issue.record("Expected invalidParameter case")
            }
        }

        @Test("Incompatible GPU quantization error")
        func testIncompatibleGPUQuantization() {
            let incompatible = MockConfigurationError.simulateIncompatibleGPUQuantization()

            #expect(!incompatible.isRetryable)
            #expect(incompatible.suggestedRecovery == .propagateError)

            if case .incompatibleOptions(let opt1, let opt2, let reason) = incompatible {
                #expect(opt1 == "gpuAcceleration")
                #expect(opt2 == "int8Quantization")
                #expect(reason.contains("not supported"))
            } else {
                Issue.record("Expected incompatibleOptions case")
            }
        }

        @Test("Missing required setting error")
        func testMissingRequiredSetting() {
            let missing = MockConfigurationError.missingRequiredSetting(setting: "api_key")

            #expect(!missing.isRetryable)
            #expect(missing.suggestedRecovery == .fallbackToDefault)
        }

        @Test("Unsupported version error")
        func testUnsupportedVersion() {
            let version = MockConfigurationError.unsupportedVersion(
                component: "VectorCore",
                required: "2.0",
                found: "1.5"
            )

            #expect(!version.isRetryable)
            #expect(version.suggestedRecovery == .propagateError)
        }

        @Test("Out of range error")
        func testOutOfRange() {
            let outOfRange = MockConfigurationError.outOfRange(
                parameter: "alpha",
                value: 1.5,
                range: 0.0...1.0
            )

            #expect(!outOfRange.isRetryable)
            #expect(outOfRange.suggestedRecovery == .propagateError)
        }
    }

    // MARK: - Mock Allocation Error Tests

    @Suite("Allocation Error Mocks")
    struct AllocationErrorTests {

        @Test("Large vector allocation failure")
        func testLargeVectorAllocation() {
            let allocError = MockAllocationError.simulateLargeVectorAllocation(dimension: 1536, count: 1_000_000)

            #expect(!allocError.isRetryable)
            #expect(allocError.suggestedRecovery == .propagateError)

            if case .insufficientMemory(let requested, let available, let context) = allocError {
                #expect(requested > available)
                #expect(context == "batch vector allocation")
                #expect(requested == 1536 * 1_000_000 * MemoryLayout<Float>.size)
            } else {
                Issue.record("Expected insufficientMemory case")
            }
        }

        @Test("Pool exhaustion is retryable")
        func testPoolExhaustion() {
            let poolError = MockAllocationError.simulatePoolExhaustion()

            #expect(poolError.isRetryable)
            #expect(poolError.suggestedRecovery == .fallbackToDefault)

            if case .poolExhausted(let poolName, let capacity) = poolError {
                #expect(poolName == "vector_pool")
                #expect(capacity == 1024 * 1024 * 512)
            } else {
                Issue.record("Expected poolExhausted case")
            }
        }

        @Test("Fragmented memory is retryable")
        func testFragmentedMemory() {
            let fragError = MockAllocationError.simulateFragmentation()

            #expect(fragError.isRetryable)
            #expect(fragError.suggestedRecovery == .retry)

            if case .fragmentedMemory(let largest, let total) = fragError {
                #expect(largest < total, "Largest block should be smaller than total available")
                #expect(largest == 100_000_000)
                #expect(total == 500_000_000)
            } else {
                Issue.record("Expected fragmentedMemory case")
            }
        }

        @Test("Allocation limit reached")
        func testAllocationLimitReached() {
            let limitError = MockAllocationError.allocationLimitReached(limit: 1000, allocated: 1000)

            #expect(!limitError.isRetryable)
            #expect(limitError.suggestedRecovery == .propagateError)
        }

        @Test("Vector allocation failed")
        func testVectorAllocationFailed() {
            let vecError = MockAllocationError.vectorAllocationFailed(dimension: 512, count: 1000)

            #expect(!vecError.isRetryable)
            #expect(vecError.suggestedRecovery == .propagateError)
        }
    }

    // MARK: - Error Injection Framework Tests

    @Suite("Error Injection Framework")
    struct ErrorInjectionTests {

        @Test("Deterministic error injection")
        func testDeterministicInjection() async throws {
            await ErrorInjector.shared.reset()

            // Inject error
            await ErrorInjector.shared.injectError(
                MockSystemError.simulateOOM(),
                at: "test_point"
            )

            // Should throw
            do {
                try await ErrorInjector.shared.checkInjectionPoint("test_point")
                Issue.record("Expected error to be thrown")
            } catch {
                expectMockError(error, toBe: MockSystemError.self)
            }

            // Verify counter incremented
            let count = await ErrorInjector.shared.getErrorCount(at: "test_point")
            #expect(count == 1)

            await ErrorInjector.shared.reset()
        }

        @Test("Multiple errors are consumed in FIFO order")
        func testFIFOOrder() async throws {
            await ErrorInjector.shared.reset()

            // Inject multiple errors
            await ErrorInjector.shared.injectError(
                MockSystemError.simulateOOM(),
                at: "test_point"
            )
            await ErrorInjector.shared.injectError(
                MockSystemError.simulateDiskFull(),
                at: "test_point"
            )

            // First error should be OOM
            do {
                try await ErrorInjector.shared.checkInjectionPoint("test_point")
                Issue.record("Expected first error")
            } catch {
                if case MockSystemError.outOfMemory = error as! MockSystemError {
                    // Expected
                } else {
                    Issue.record("Expected OOM error first")
                }
            }

            // Second error should be disk full
            do {
                try await ErrorInjector.shared.checkInjectionPoint("test_point")
                Issue.record("Expected second error")
            } catch {
                if case MockSystemError.diskFull = error as! MockSystemError {
                    // Expected
                } else {
                    Issue.record("Expected diskFull error second")
                }
            }

            // No more errors
            try? await ErrorInjector.shared.checkInjectionPoint("test_point")
            let count = await ErrorInjector.shared.getErrorCount(at: "test_point")
            #expect(count == 2)

            await ErrorInjector.shared.reset()
        }

        @Test("Probabilistic error injection with 0.0 probability never throws")
        func testZeroProbability() async throws {
            await ErrorInjector.shared.reset()

            await ErrorInjector.shared.setErrorProbability(0.0, at: "test_point") {
                MockSystemError.simulateOOM()
            }

            // Try 100 times - should never throw
            for _ in 0..<100 {
                try? await ErrorInjector.shared.checkInjectionPoint("test_point")
            }

            let count = await ErrorInjector.shared.getErrorCount(at: "test_point")
            #expect(count == 0, "No errors should be thrown with 0.0 probability")

            await ErrorInjector.shared.reset()
        }

        @Test("Probabilistic error injection with 1.0 probability always throws")
        func testFullProbability() async throws {
            await ErrorInjector.shared.reset()

            await ErrorInjector.shared.setErrorProbability(1.0, at: "test_point") {
                MockSystemError.simulateOOM()
            }

            // Should always throw
            do {
                try await ErrorInjector.shared.checkInjectionPoint("test_point")
                Issue.record("Expected error with 1.0 probability")
            } catch {
                expectMockError(error, toBe: MockSystemError.self)
            }

            let count = await ErrorInjector.shared.getErrorCount(at: "test_point")
            #expect(count == 1)

            await ErrorInjector.shared.reset()
        }

        @Test("Probabilistic error injection with custom generator")
        func testCustomGenerator() async throws {
            await ErrorInjector.shared.reset()

            await ErrorInjector.shared.setErrorProbability(1.0, at: "test_point") {
                MockConfigurationError.simulateDimensionMismatch()
            }

            do {
                try await ErrorInjector.shared.checkInjectionPoint("test_point")
                Issue.record("Expected custom error")
            } catch {
                expectMockError(error, toBe: MockConfigurationError.self)
            }

            await ErrorInjector.shared.reset()
        }

        @Test("Different injection points are independent")
        func testIndependentInjectionPoints() async throws {
            await ErrorInjector.shared.reset()

            await ErrorInjector.shared.injectError(
                MockSystemError.simulateOOM(),
                at: "point_a"
            )
            await ErrorInjector.shared.injectError(
                MockConfigurationError.simulateDimensionMismatch(),
                at: "point_b"
            )

            // Point A should throw system error
            do {
                try await ErrorInjector.shared.checkInjectionPoint("point_a")
                Issue.record("Expected error at point_a")
            } catch {
                expectMockError(error, toBe: MockSystemError.self)
            }

            // Point B should throw config error
            do {
                try await ErrorInjector.shared.checkInjectionPoint("point_b")
                Issue.record("Expected error at point_b")
            } catch {
                expectMockError(error, toBe: MockConfigurationError.self)
            }

            await ErrorInjector.shared.reset()
        }

        @Test("Reset clears all state")
        func testReset() async throws {
            await ErrorInjector.shared.reset()

            // Set up errors
            await ErrorInjector.shared.injectError(
                MockSystemError.simulateOOM(),
                at: "test_point"
            )
            await ErrorInjector.shared.setErrorProbability(1.0, at: "test_point2")

            // Trigger one error
            try? await ErrorInjector.shared.checkInjectionPoint("test_point")

            // Reset
            await ErrorInjector.shared.reset()

            // Should not throw anymore
            try? await ErrorInjector.shared.checkInjectionPoint("test_point")
            try? await ErrorInjector.shared.checkInjectionPoint("test_point2")

            // Counters should be cleared
            let count1 = await ErrorInjector.shared.getErrorCount(at: "test_point")
            let count2 = await ErrorInjector.shared.getErrorCount(at: "test_point2")
            #expect(count1 == 0)
            #expect(count2 == 0)
        }

        @Test("Get statistics returns all counters")
        func testGetStatistics() async throws {
            await ErrorInjector.shared.reset()

            await ErrorInjector.shared.injectError(MockSystemError.simulateOOM(), at: "point_a")
            await ErrorInjector.shared.injectError(MockSystemError.simulateOOM(), at: "point_a")
            await ErrorInjector.shared.injectError(MockSystemError.simulateOOM(), at: "point_b")

            // Trigger errors
            try? await ErrorInjector.shared.checkInjectionPoint("point_a")
            try? await ErrorInjector.shared.checkInjectionPoint("point_a")
            try? await ErrorInjector.shared.checkInjectionPoint("point_b")

            let stats = await ErrorInjector.shared.getStatistics()
            #expect(stats["point_a"] == 2)
            #expect(stats["point_b"] == 1)

            await ErrorInjector.shared.reset()
        }
    }

    // MARK: - Recovery Strategy Tests

    @Suite("Recovery Strategy Validation")
    struct RecoveryStrategyTests {

        @Test("All error types have appropriate recovery strategies")
        func testRecoveryStrategies() {
            let testCases: [(any MockError, RecoveryStrategy, Bool)] = [
                // (error, expected recovery, is retryable)
                (MockSystemError.simulateOOM(), .propagateError, false),
                (MockSystemError.simulateResourceLock(), .retry, true),
                (MockSystemError.simulateDiskFull(), .propagateError, false),
                (MockConfigurationError.simulateInvalidClusterCount(k: 0), .fallbackToDefault, false),
                (MockConfigurationError.simulateDimensionMismatch(), .propagateError, false),
                (MockAllocationError.simulatePoolExhaustion(), .fallbackToDefault, true),
                (MockAllocationError.simulateFragmentation(), .retry, true),
                (MockAllocationError.simulateLargeVectorAllocation(), .propagateError, false),
            ]

            for (error, expectedRecovery, expectedRetryable) in testCases {
                expectRecoveryStrategy(error, equals: expectedRecovery)
                #expect(error.isRetryable == expectedRetryable,
                       "\(error.debugDescription) retryable mismatch")
            }
        }

        @Test("Retryable errors suggest retry or fallback strategies")
        func testRetryableErrorStrategies() {
            let retryableErrors: [any MockError] = [
                MockSystemError.simulateResourceLock(),
                MockSystemError.timeoutExpired(operation: "test", timeout: 1.0),
                MockAllocationError.simulatePoolExhaustion(),
                MockAllocationError.simulateFragmentation(),
            ]

            for error in retryableErrors {
                #expect(error.isRetryable, "\(error.debugDescription) should be retryable")
                #expect(
                    error.suggestedRecovery == .retry || error.suggestedRecovery == .fallbackToDefault,
                    "\(error.debugDescription) should suggest retry or fallback"
                )
            }
        }

        @Test("Non-retryable errors suggest propagate or fallback")
        func testNonRetryableErrorStrategies() {
            let nonRetryableErrors: [any MockError] = [
                MockSystemError.simulateOOM(),
                MockSystemError.simulateDiskFull(),
                MockConfigurationError.simulateDimensionMismatch(),
                MockAllocationError.simulateLargeVectorAllocation(),
            ]

            for error in nonRetryableErrors {
                #expect(!error.isRetryable, "\(error.debugDescription) should not be retryable")
                #expect(
                    error.suggestedRecovery == .propagateError || error.suggestedRecovery == .fallbackToDefault,
                    "\(error.debugDescription) should suggest propagate or fallback"
                )
            }
        }
    }
}
