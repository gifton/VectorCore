//
//  ErrorHandlingMocks.swift
//  VectorCore
//
//  Mock error types for systematic testing of error handling, recovery strategies,
//  and edge case validation. Implements Kernel Spec #31.
//
//  Integration Notes:
//  - Works with VectorCore's existing VectorError system
//  - Provides recovery strategy testing framework
//  - Supports both deterministic and probabilistic error injection
//

import Foundation
@testable import VectorCore

// =============================================================================
// MARK: - Recovery Strategy (Testing Concept)
// =============================================================================

/// Recovery strategy for mock errors (testing framework concept)
///
/// This enum represents different recovery approaches that error handlers
/// might take. It's separate from VectorCore's error system and used
/// specifically for testing error recovery logic.
public enum RecoveryStrategy: Sendable, Equatable {
    case retry
    case fallbackToDefault
    case propagateError
    case ignore
}

// =============================================================================
// MARK: - Mock Error Protocol
// =============================================================================

/// Protocol for mock errors used in testing
///
/// Mock errors simulate real-world failures in a controlled, reproducible way.
/// They support injection into error handling code paths for verification.
///
/// Note: Unlike the spec, we don't include errorID/timestamp as stored properties
/// because Swift enums can't have stored properties. Instead, we focus on the
/// essential testing concerns: error type, retryability, and recovery strategy.
public protocol MockError: Error, Sendable, CustomDebugStringConvertible {
    /// Whether this error should be retryable
    var isRetryable: Bool { get }

    /// Simulated recovery strategy for testing
    var suggestedRecovery: RecoveryStrategy { get }

    /// Custom error message for debugging
    var debugDescription: String { get }
}

/// Default implementations for MockError
extension MockError {
    public var isRetryable: Bool { false }
    public var suggestedRecovery: RecoveryStrategy { .propagateError }
}

// =============================================================================
// MARK: - System Error Mocks
// =============================================================================

/// Mock system-level errors (memory, disk, permissions)
///
/// These simulate OS-level and system resource failures for testing
/// error handling paths in resource-constrained scenarios.
public enum MockSystemError: MockError {
    case outOfMemory(attemptedAllocation: Int, availableMemory: Int)
    case diskFull(requiredSpace: Int, availableSpace: Int)
    case permissionDenied(resource: String, operation: String)
    case resourceLocked(resource: String, owner: String)
    case processLimitExceeded(limit: Int, current: Int)
    case timeoutExpired(operation: String, timeout: TimeInterval)

    // MARK: - MockError Conformance

    public var isRetryable: Bool {
        switch self {
        case .outOfMemory, .diskFull, .permissionDenied, .processLimitExceeded:
            return false  // Fatal or configuration issues
        case .resourceLocked, .timeoutExpired:
            return true   // Transient failures
        }
    }

    public var suggestedRecovery: RecoveryStrategy {
        switch self {
        case .outOfMemory, .diskFull:
            return .propagateError  // Cannot recover automatically
        case .permissionDenied:
            return .propagateError  // Requires user intervention
        case .processLimitExceeded:
            return .fallbackToDefault  // Use less intensive mode
        case .resourceLocked, .timeoutExpired:
            return .retry  // May succeed on retry
        }
    }

    public var debugDescription: String {
        switch self {
        case .outOfMemory(let attempted, let available):
            return "MockSystemError.outOfMemory: Attempted to allocate \(attempted) bytes, only \(available) available"
        case .diskFull(let required, let available):
            return "MockSystemError.diskFull: Required \(required) bytes, only \(available) available"
        case .permissionDenied(let resource, let operation):
            return "MockSystemError.permissionDenied: Cannot perform '\(operation)' on '\(resource)'"
        case .resourceLocked(let resource, let owner):
            return "MockSystemError.resourceLocked: Resource '\(resource)' locked by '\(owner)'"
        case .processLimitExceeded(let limit, let current):
            return "MockSystemError.processLimitExceeded: Limit \(limit), current \(current)"
        case .timeoutExpired(let operation, let timeout):
            return "MockSystemError.timeoutExpired: Operation '\(operation)' exceeded \(timeout)s timeout"
        }
    }
}

// MARK: - System Error Factory

extension MockSystemError {
    /// Create a realistic out-of-memory error
    ///
    /// Simulates attempting to allocate a large amount of memory with
    /// insufficient system resources available.
    public static func simulateOOM(attemptedAllocation: Int = 1_000_000_000) -> MockSystemError {
        // Ensure available memory is less than attempted allocation for realism
        let maxAvailable = attemptedAllocation > 0 ? min(100_000_000, attemptedAllocation - 1) : 100_000_000
        let minAvailable = 10_000_000
        let availableMemory = Int.random(in: minAvailable...max(minAvailable, maxAvailable))
        return .outOfMemory(attemptedAllocation: attemptedAllocation, availableMemory: availableMemory)
    }

    /// Create a realistic disk full error
    public static func simulateDiskFull(requiredSpace: Int = 50_000_000) -> MockSystemError {
        let maxAvailable = requiredSpace > 0 ? min(10_000_000, requiredSpace - 1) : 10_000_000
        let availableSpace = Int.random(in: 0...max(0, maxAvailable))
        return .diskFull(requiredSpace: requiredSpace, availableSpace: availableSpace)
    }

    /// Create a transient resource lock error (retryable)
    public static func simulateResourceLock(resource: String = "vector_cache") -> MockSystemError {
        return .resourceLocked(resource: resource, owner: "Thread-\(Int.random(in: 1...100))")
    }
}

// =============================================================================
// MARK: - Configuration Error Mocks
// =============================================================================

/// Mock configuration and validation errors
///
/// These simulate invalid parameters, incompatible options, and configuration
/// problems for testing validation and error reporting logic.
public enum MockConfigurationError: MockError {
    // Note: Using String for value description instead of Any for Sendable conformance
    case invalidParameter(parameter: String, valueDescription: String, expected: String)
    case incompatibleOptions(option1: String, option2: String, reason: String)
    case missingRequiredSetting(setting: String)
    case unsupportedVersion(component: String, required: String, found: String)
    case invalidDimension(expected: Int, got: Int)
    case outOfRange(parameter: String, value: Double, range: ClosedRange<Double>)

    // MARK: - MockError Conformance

    public var suggestedRecovery: RecoveryStrategy {
        switch self {
        case .missingRequiredSetting, .invalidParameter:
            return .fallbackToDefault
        case .invalidDimension, .outOfRange, .incompatibleOptions, .unsupportedVersion:
            return .propagateError  // Cannot recover automatically
        }
    }

    public var debugDescription: String {
        switch self {
        case .invalidParameter(let param, let valueDesc, let expected):
            return "MockConfigurationError.invalidParameter: '\(param)' = \(valueDesc), expected \(expected)"
        case .incompatibleOptions(let opt1, let opt2, let reason):
            return "MockConfigurationError.incompatibleOptions: '\(opt1)' and '\(opt2)' cannot be used together: \(reason)"
        case .missingRequiredSetting(let setting):
            return "MockConfigurationError.missingRequiredSetting: '\(setting)' must be specified"
        case .unsupportedVersion(let component, let required, let found):
            return "MockConfigurationError.unsupportedVersion: \(component) requires \(required), found \(found)"
        case .invalidDimension(let expected, let got):
            return "MockConfigurationError.invalidDimension: Expected \(expected), got \(got)"
        case .outOfRange(let param, let value, let range):
            return "MockConfigurationError.outOfRange: '\(param)' = \(value), must be in \(range)"
        }
    }
}

// MARK: - Configuration Error Factory

extension MockConfigurationError {
    /// Create dimension mismatch error (common in vector operations)
    public static func simulateDimensionMismatch(expected: Int = 512, got: Int = 768) -> MockConfigurationError {
        return .invalidDimension(expected: expected, got: got)
    }

    /// Create invalid K value for clustering
    public static func simulateInvalidClusterCount(k: Int) -> MockConfigurationError {
        return .invalidParameter(
            parameter: "k",
            valueDescription: String(k),
            expected: "positive integer > 0"
        )
    }

    /// Create incompatible GPU + quantization error
    public static func simulateIncompatibleGPUQuantization() -> MockConfigurationError {
        return .incompatibleOptions(
            option1: "gpuAcceleration",
            option2: "int8Quantization",
            reason: "INT8 quantization not supported on GPU"
        )
    }
}

// =============================================================================
// MARK: - Allocation Error Mocks
// =============================================================================

/// Mock memory allocation failures
///
/// These simulate various memory allocation failure scenarios for testing
/// resource exhaustion handling and graceful degradation.
public enum MockAllocationError: MockError {
    case insufficientMemory(requested: Int, available: Int, context: String)
    case fragmentedMemory(largestBlock: Int, totalAvailable: Int)
    case allocationLimitReached(limit: Int, allocated: Int)
    case poolExhausted(poolName: String, capacity: Int)
    case vectorAllocationFailed(dimension: Int, count: Int)

    // MARK: - MockError Conformance

    public var isRetryable: Bool {
        switch self {
        case .fragmentedMemory, .poolExhausted:
            return true   // May succeed after GC or pool refresh
        case .insufficientMemory, .allocationLimitReached, .vectorAllocationFailed:
            return false  // Fatal memory exhaustion
        }
    }

    public var suggestedRecovery: RecoveryStrategy {
        switch self {
        case .fragmentedMemory:
            return .retry  // May succeed after compaction
        case .poolExhausted:
            return .fallbackToDefault  // Use system allocator
        case .insufficientMemory, .allocationLimitReached, .vectorAllocationFailed:
            return .propagateError
        }
    }

    public var debugDescription: String {
        switch self {
        case .insufficientMemory(let requested, let available, let context):
            return "MockAllocationError.insufficientMemory: Requested \(requested) bytes in '\(context)', only \(available) available"
        case .fragmentedMemory(let largest, let total):
            return "MockAllocationError.fragmentedMemory: Largest contiguous block \(largest) bytes, total available \(total) bytes"
        case .allocationLimitReached(let limit, let allocated):
            return "MockAllocationError.allocationLimitReached: Limit \(limit) bytes, currently allocated \(allocated) bytes"
        case .poolExhausted(let pool, let capacity):
            return "MockAllocationError.poolExhausted: Pool '\(pool)' exhausted (capacity: \(capacity))"
        case .vectorAllocationFailed(let dim, let count):
            return "MockAllocationError.vectorAllocationFailed: Cannot allocate \(count) vectors of dimension \(dim)"
        }
    }
}

// MARK: - Allocation Error Factory

extension MockAllocationError {
    /// Simulate large vector allocation failure
    public static func simulateLargeVectorAllocation(dimension: Int = 1536, count: Int = 1_000_000) -> MockAllocationError {
        // Calculate requested memory (assuming FP32 vectors)
        let requested = dimension * count * MemoryLayout<Float>.size
        let available = requested / 2  // Simulate having only half the required memory
        return .insufficientMemory(requested: requested, available: available, context: "batch vector allocation")
    }

    /// Simulate memory pool exhaustion (retryable)
    public static func simulatePoolExhaustion(poolName: String = "vector_pool") -> MockAllocationError {
        return .poolExhausted(poolName: poolName, capacity: 1024 * 1024 * 512)  // 512 MB pool
    }

    /// Simulate fragmented memory (retryable after compaction)
    public static func simulateFragmentation() -> MockAllocationError {
        return .fragmentedMemory(largestBlock: 100_000_000, totalAvailable: 500_000_000)
    }
}

// =============================================================================
// MARK: - Error Injection Framework
// =============================================================================

/// Framework for injecting errors into code paths
///
/// `ErrorInjector` provides a thread-safe actor-based system for injecting
/// mock errors at specific points in your code. It supports both deterministic
/// (queue-based) and probabilistic (random) injection modes.
///
/// ## Example Usage
///
/// ### Deterministic Injection
/// ```swift
/// await ErrorInjector.shared.injectError(
///     MockSystemError.simulateOOM(),
///     at: "vector_allocation"
/// )
///
/// try await ErrorInjector.shared.checkInjectionPoint("vector_allocation")
/// // Throws the OOM error
/// ```
///
/// ### Probabilistic Injection
/// ```swift
/// await ErrorInjector.shared.setErrorProbability(
///     0.1,  // 10% chance
///     at: "cache_access"
/// ) {
///     MockSystemError.simulateResourceLock()
/// }
///
/// // 10% of the time, this will throw
/// try await ErrorInjector.shared.checkInjectionPoint("cache_access")
/// ```
public actor ErrorInjector {

    // MARK: - Singleton

    public static let shared = ErrorInjector()

    // MARK: - State

    /// Queue of errors to inject deterministically (FIFO)
    private var injectedErrors: [String: [any MockError]] = [:]

    /// Probabilistic error configuration
    private var errorProbabilities: [String: (probability: Double, generator: @Sendable () -> any MockError)] = [:]

    /// Counters for verification and statistics
    private var errorCounters: [String: Int] = [:]

    private init() {}

    // MARK: - Configuration

    /// Register an error to be thrown at a specific injection point (deterministic)
    ///
    /// Errors are thrown in FIFO order. Each call to `checkInjectionPoint`
    /// consumes one error from the queue.
    ///
    /// - Parameters:
    ///   - error: Mock error to inject
    ///   - point: Injection point identifier (e.g., "vector_allocation", "graph_traversal")
    public func injectError(_ error: any MockError, at point: String) {
        injectedErrors[point, default: []].append(error)
    }

    /// Set probability of error occurring at injection point (probabilistic)
    ///
    /// - Parameters:
    ///   - probability: Chance of error (0.0 = never, 1.0 = always)
    ///   - point: Injection point identifier
    ///   - errorGenerator: Sendable closure that generates the error to throw if triggered
    public func setErrorProbability(
        _ probability: Double,
        at point: String,
        errorGenerator: @escaping @Sendable () -> any MockError = { MockSystemError.simulateOOM() }
    ) {
        precondition(probability >= 0.0 && probability <= 1.0, "Probability must be in range [0.0, 1.0]")
        errorProbabilities[point] = (probability, errorGenerator)
    }

    /// Clear all injected errors and configurations
    public func reset() {
        injectedErrors.removeAll()
        errorProbabilities.removeAll()
        errorCounters.removeAll()
    }

    // MARK: - Error Triggering

    /// Check if error should be thrown at injection point
    ///
    /// This method should be called at strategic points in your code where
    /// you want to test error handling. It will throw an error if:
    /// 1. A deterministic error is queued for this point (FIFO), OR
    /// 2. A probabilistic error is configured and the random roll succeeds
    ///
    /// - Parameter point: Injection point identifier
    /// - Throws: MockError if one is registered and triggered for this point
    public func checkInjectionPoint(_ point: String) throws {
        if let error = determineErrorToThrow(at: point) {
            // Increment counter
            errorCounters[point, default: 0] += 1
            throw error
        }
    }

    /// Internal logic to determine if an error should be thrown
    private func determineErrorToThrow(at point: String) -> (any MockError)? {
        // Priority 1: Deterministic sequential errors
        if var errors = injectedErrors[point], !errors.isEmpty {
            let error = errors.removeFirst()
            // Update the queue, removing the entry if empty
            if errors.isEmpty {
                injectedErrors.removeValue(forKey: point)
            } else {
                injectedErrors[point] = errors
            }
            return error
        }

        // Priority 2: Probabilistic errors
        if let (probability, generator) = errorProbabilities[point] {
            if Double.random(in: 0.0...1.0) < probability {
                // Generate a fresh error instance using the generator
                return generator()
            }
        }

        return nil
    }

    // MARK: - Verification

    /// Get the number of times an error was thrown at a specific injection point
    ///
    /// Useful for verifying that error injection actually occurred during tests.
    public func getErrorCount(at point: String) -> Int {
        return errorCounters[point] ?? 0
    }

    /// Get error statistics for all injection points
    ///
    /// Returns a dictionary mapping injection point names to the number of
    /// times errors were thrown at that point.
    public func getStatistics() -> [String: Int] {
        return errorCounters
    }
}
