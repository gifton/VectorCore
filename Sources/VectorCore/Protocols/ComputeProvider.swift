// VectorCore: Compute Provider Protocol
//
// Clean abstraction for execution strategies without platform dependencies
//

import Foundation

/// Protocol for compute execution providers
///
/// ComputeProvider abstracts execution strategies (CPU, GPU, Neural) without
/// exposing platform-specific details. Implementations can live in platform-specific
/// packages while VectorCore remains dependency-free.
///
/// ## Design Principles
/// - No platform-specific types in protocol
/// - Async-first API design
/// - Supports both single and parallel execution
/// - Conformers are `Sendable`, so the compiler checks values that cross concurrency domains
///
/// ## Example Implementation
/// ```swift
/// struct CPUComputeProvider: ComputeProvider {
///     let device = ComputeDevice.cpu
///     var maxConcurrency: Int { ProcessInfo.processInfo.activeProcessorCount }
///
///     func execute<T>(_ work: () async throws -> T) async throws -> T {
///         try await work()
///     }
/// }
/// ```
public protocol ComputeProvider: Sendable {
    /// The compute device type this provider uses
    var device: ComputeDevice { get }

    /// Positive maximum worker count per default scheduling phase. Int.max is valid.
    /// Defaults reject nonpositive limits before callbacks, including empty input.
    /// Nested library calls execute inline; user-created tasks and independent root
    /// invocations are outside this bound. Custom scheduling witnesses own this obligation.
    var maxConcurrency: Int { get }

    /// Execute work on the compute device
    ///
    /// - Parameter work: The work to execute
    /// - Returns: The result of the work
    /// Conformers must check cancellation before and after the async body.
    /// Cancellation cannot interrupt a body that ignores it.
    /// - Throws: Any error from the work closure or cancellation.
    func execute<T: Sendable>(
        _ work: @Sendable @escaping () async throws -> T
    ) async throws -> T

    /// Execute work across bounded contiguous worker ranges, returning input order.
    /// Throwing defaults check cancellation before submission, before and after each
    /// work unit, and before success. Errors cancel siblings and await cleanup.
    /// User bodies must cooperate within a work unit; no cancellation deadline is promised.
    ///
    /// - Parameters:
    ///   - items: Range of items to process
    ///   - work: Work to perform for each item
    /// - Returns: Array of results in order
    /// - Throws: `.invalidRange` if the mathematical range length exceeds `Int.max`,
    ///   or any error from the work closures. Empty ranges do not invoke work.
    func parallelExecute<T: Sendable>(
        items: Range<Int>,
        _ work: @Sendable @escaping (Int) async throws -> T
    ) async throws -> [T]

    /// Execute side-effecting work in parallel across items (no result array).
    /// Chunking/scheduling is provider-defined. The mathematical range length must
    /// fit in `Int`, otherwise throws `.invalidRange`. Empty ranges do not invoke body.
    func parallelForEach(
        items: Range<Int>,
        _ body: @Sendable @escaping (Int) async throws -> Void
    ) async throws

    /// Execute side-effecting work in parallel with a minimum chunk size hint.
    /// Requires a representable range length and `minChunk > 0`, including empty ranges.
    /// Throws `.invalidRange` or `.invalidConfiguration` for these invalid inputs.
    /// The default validates both and then calls `parallelForEach(items:_:)`, so a
    /// conformer that implements only the hint-less requirement is reached by every
    /// hinted call (`ComputeProviderForwardingTests.hintedDefaultsReachHintlessWitnesses`);
    /// override this to honour the hint.
    func parallelForEach(
        items: Range<Int>,
        minChunk: Int,
        _ body: @Sendable @escaping (Int) async throws -> Void
    ) async throws

    /// Partition reduction: rangeWork aggregates its whole range, combine must be
    /// associative and initial must be its identity. Commutativity is not required:
    /// partials combine in ascending range order with initial exactly once.
    /// Floating-point addition is not associative; partition/mode bitwise equality
    /// is not promised. This is not an arbitrary sequential left fold.
    /// Providers should implement this with coarse-grained chunking to minimize overhead.
    /// A range length exceeding `Int.max` throws `.invalidRange`. Empty ranges return
    /// `initial` without invoking rangeWork or combine.
    /// Contract: rangeWork and combine run inside the library callback region, with
    /// cancellation checked before and after each callback, including parent combination.
    func parallelReduce<R: Sendable>(
        items: Range<Int>,
        initial: R,
        _ rangeWork: @Sendable @escaping (Range<Int>) async throws -> R,
        _ combine: @Sendable @escaping (R, R) -> R
    ) async throws -> R

    /// Execute a parallel reduction with a minimum chunk size hint.
    /// Requires a representable range length and `minChunk > 0`, including empty ranges.
    /// Throws `.invalidRange` or `.invalidConfiguration` for these invalid inputs.
    /// The default validates both and then calls `parallelReduce(items:initial:_:_:)`, so a
    /// conformer that implements only the hint-less requirement is reached by every
    /// optimized search route (`ComputeProviderForwardingTests.hintedDefaultsReachHintlessWitnesses`);
    /// override this to honour the hint.
    func parallelReduce<R: Sendable>(
        items: Range<Int>,
        initial: R,
        minChunk: Int,
        _ rangeWork: @Sendable @escaping (Range<Int>) async throws -> R,
        _ combine: @Sendable @escaping (R, R) -> R
    ) async throws -> R

    /// Information about the compute device
    var deviceInfo: ComputeDeviceInfo { get }
}

/// Information about a compute device
public struct ComputeDeviceInfo: Sendable {
    /// Human-readable device name
    public let name: String

    /// Available memory in bytes (if known)
    public let availableMemory: Int?

    /// Maximum threads/cores available
    public let maxThreads: Int

    /// Preferred chunk size for operations
    public let preferredChunkSize: Int

    public init(
        name: String,
        availableMemory: Int? = nil,
        maxThreads: Int,
        preferredChunkSize: Int
    ) {
        self.name = name
        self.availableMemory = availableMemory
        self.maxThreads = maxThreads
        self.preferredChunkSize = preferredChunkSize
    }
}

// MARK: - Default Implementation

public extension ComputeProvider {
    func parallelExecute<T: Sendable>(items: Range<Int>, _ work: @Sendable @escaping (Int) async throws -> T) async throws -> [T] {
        try await BoundedExecution.map(items, limit: maxConcurrency, phase: "default.execute", body: work)
    }

    func parallelForEach(items: Range<Int>, _ body: @Sendable @escaping (Int) async throws -> Void) async throws {
        try await BoundedExecution.forEach(items, limit: maxConcurrency, phase: "default.forEach", body: body)
    }

    // The hinted defaults validate the range and the hint, then call the hint-less requirements (the 0.3.3 direction),
    // so a conformer that implements only those is reached; `.invalidRange` keeps precedence over `.invalidConfiguration`.
    func parallelForEach(items: Range<Int>, minChunk: Int, _ body: @Sendable @escaping (Int) async throws -> Void) async throws {
        _ = try ComputeRangeBoundary.count(items)
        try ComputeRangeBoundary.validate(minChunk: minChunk)
        try await parallelForEach(items: items, body)
    }

    func parallelReduce<R: Sendable>(items: Range<Int>, initial: R,
        _ rangeWork: @Sendable @escaping (Range<Int>) async throws -> R,
        _ combine: @Sendable @escaping (R, R) -> R
    ) async throws -> R {
        let parts = try await BoundedExecution.ranges(items, limit: maxConcurrency, phase: "default.reduce", body: rangeWork)
        return try BoundedExecution.combinePartials(parts, initial: initial, combine)
    }

    func parallelReduce<R: Sendable>(items: Range<Int>, initial: R, minChunk: Int,
        _ rangeWork: @Sendable @escaping (Range<Int>) async throws -> R,
        _ combine: @Sendable @escaping (R, R) -> R
    ) async throws -> R {
        _ = try ComputeRangeBoundary.count(items)
        try ComputeRangeBoundary.validate(minChunk: minChunk)
        return try await parallelReduce(items: items, initial: initial, rangeWork, combine)
    }
}

// Shared arithmetic for the five throwing range helpers and CPU overrides.
internal enum ComputeRangeBoundary {
    static func count(_ items: Range<Int>) throws -> Int {
        let (count, overflow) = items.upperBound.subtractingReportingOverflow(items.lowerBound)
        guard !overflow else {
            throw VectorError(.invalidRange, message: "items range length must be representable as Int")
        }
        return count
    }

    static func validate(minChunk: Int) throws {
        guard minChunk > 0 else {
            throw VectorError(.invalidConfiguration, message: "minChunk must be positive")
        }
    }

    // Callers supply a nonnegative count and positive task count.
    static func ceilingDivide(_ count: Int, by tasks: Int) -> Int {
        count / tasks + (count % tasks == 0 ? 0 : 1)
    }
}
