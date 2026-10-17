// VectorCore: CPU Compute Provider
//
// Pure Swift CPU execution without platform dependencies
//

import Foundation
import Dispatch

/// CPU-based compute provider
///
/// Provides parallel and sequential execution on CPU cores using
/// structured Swift concurrency. No platform-specific dependencies.
///
/// ## Execution Modes
/// - Sequential: Inline execution in the calling task
/// - Parallel: Bounded child workers using the processor-count limit
/// - Automatic: Chooses based on workload size
public struct CPUComputeProvider: ComputeProvider {

    /// Execution mode for CPU operations
    public enum Mode: String, Sendable {
        case sequential
        case parallel
        case automatic
    }

    public let device: ComputeDevice
    public let mode: Mode
    private let processorCount: Int
    // Optional explicit overrides for automatic mode

    /// Threshold for automatic parallelization
    private let parallelizationThreshold: Int

    /// Legacy internal suppression control, additional to the shared library marker.
    /// ExecutionSchedulingContractTests.nestedProvidersAndLegacySuppression covers it.
    @TaskLocal internal static var isInsideParallelRegion: Bool = false

    public init(mode: Mode = .automatic, parallelizationThreshold: Int = 50_000) {
        self.device = .cpu
        self.mode = mode
        self.processorCount = ProcessInfo.processInfo.activeProcessorCount
        self.parallelizationThreshold = parallelizationThreshold
    }

    /// The initializer minChunk is a compatibility argument with no effect.
    /// Per-method minChunk overloads supply the operative scheduling hint.
    public init(mode: Mode = .automatic, parallelizationThreshold: Int? = nil, minChunk: Int? = nil) {
        self.device = .cpu
        self.mode = mode
        self.processorCount = ProcessInfo.processInfo.activeProcessorCount
        self.parallelizationThreshold = parallelizationThreshold ?? 50_000
    }

    // MARK: - ComputeProvider Implementation

    public var maxConcurrency: Int {
        switch mode {
        case .sequential:
            return 1
        case .parallel, .automatic:
            return processorCount
        }
    }

    public var deviceInfo: ComputeDeviceInfo {
        ComputeDeviceInfo(
            name: "CPU (\(processorCount) cores)",
            availableMemory: nil, // CPU doesn't have dedicated memory
            maxThreads: processorCount,
            preferredChunkSize: preferredChunkSize
        )
    }

    /// Platform-optimized chunk size
    private var preferredChunkSize: Int {
        #if arch(arm64)
        return 16384 // 16KB for Apple Silicon
        #else
        return 8192  // 8KB for Intel
        #endif
    }

    public func execute<T: Sendable>(_ work: @Sendable @escaping () async throws -> T) async throws -> T {
        try Task.checkCancellation()
        let result = try await BoundedExecution.$isInsideRegion.withValue(true) { try await work() }
        try Task.checkCancellation()
        return result
    }

    private func parallel(_ count: Int) -> Bool {
        switch mode {
        case .sequential: return false
        case .parallel: return true
        case .automatic: return count >= parallelizationThreshold
        }
    }

    public func parallelExecute<T: Sendable>(items: Range<Int>, _ work: @Sendable @escaping (Int) async throws -> T) async throws -> [T] {
        let count = try ComputeRangeBoundary.count(items)
        return try await BoundedExecution.map(items, limit: maxConcurrency, parallel: parallel(count), minimum: 256,
            phase: "cpu.execute", body: work)
    }

    public func parallelForEach(items: Range<Int>, _ body: @Sendable @escaping (Int) async throws -> Void) async throws {
        try await parallelForEach(items: items, minChunk: 1, body)
    }

    public func parallelForEach(items: Range<Int>, minChunk: Int, _ body: @Sendable @escaping (Int) async throws -> Void) async throws {
        let count = try ComputeRangeBoundary.count(items)
        try ComputeRangeBoundary.validate(minChunk: minChunk)
        try await BoundedExecution.forEach(items, limit: maxConcurrency, parallel: parallel(count), minimum: max(256, minChunk),
            phase: "cpu.forEach", body: body)
    }

    public func parallelReduce<R: Sendable>(items: Range<Int>, initial: R,
        _ rangeWork: @Sendable @escaping (Range<Int>) async throws -> R,
        _ combine: @Sendable @escaping (R, R) -> R
    ) async throws -> R {
        try await parallelReduce(items: items, initial: initial, minChunk: 1, rangeWork, combine)
    }

    public func parallelReduce<R: Sendable>(items: Range<Int>, initial: R, minChunk: Int,
        _ rangeWork: @Sendable @escaping (Range<Int>) async throws -> R,
        _ combine: @Sendable @escaping (R, R) -> R
    ) async throws -> R {
        let count = try ComputeRangeBoundary.count(items)
        try ComputeRangeBoundary.validate(minChunk: minChunk)
        let parts = try await BoundedExecution.ranges(items, limit: maxConcurrency, parallel: parallel(count), minimum: max(256, minChunk),
            phase: "cpu.reduce", body: rangeWork)
        return try BoundedExecution.combinePartials(parts, initial: initial, combine)
    }
}

public extension CPUComputeProvider {
    /// Synchronous single full-range invocation for nonempty candidates; empty input
    /// invokes no kernel. Mode, threshold and initializer minChunk do not schedule
    /// this helper. There is no autotuning or concurrent output-pointer access.
    /// The kernel must write only the logical candidate prefix; an oversized suffix
    /// is outside that output. ExecutionSchedulingContractTests exercises this policy.
    func executeBatch<V: Sendable, R: Sendable>(
        query: V, candidates: [V], kernelKind: KernelKind, dimension: Int,
        results: UnsafeMutableBufferPointer<R>,
        kernel: @Sendable @escaping (V, [V], Range<Int>, UnsafeMutableBufferPointer<R>) -> Void
    ) {
        let n = candidates.count
        guard results.count >= n else { fatalError("executeBatch output capacity must cover candidates") }
        guard n > 0 else { return }
        kernel(query, candidates, 0..<n, results)
    }
}

// MARK: - Factory Methods

public extension CPUComputeProvider {
    /// Sequential execution on single core
    static let sequential = CPUComputeProvider(mode: .sequential)

    /// Parallel execution using all cores
    static let parallel = CPUComputeProvider(mode: .parallel)

    /// Automatic mode selection based on workload
    static let automatic = CPUComputeProvider(mode: .automatic)

    /// Performance-optimized settings
    static let performance = CPUComputeProvider(
        mode: .parallel,
        parallelizationThreshold: 100
    )

    /// Efficiency-optimized settings
    static let efficiency = CPUComputeProvider(
        mode: .automatic,
        parallelizationThreshold: 5000
    )
}
