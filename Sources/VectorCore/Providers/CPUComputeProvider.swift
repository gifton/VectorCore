// VectorCore: CPU Compute Provider
//
// Pure Swift CPU execution without platform dependencies
//

import Foundation
import Dispatch

/// CPU-based compute provider
///
/// Provides parallel and sequential execution on CPU cores using
/// Swift concurrency and GCD. No platform-specific dependencies.
///
/// ## Execution Modes
/// - Sequential: Single-threaded execution
/// - Parallel: Multi-threaded using available cores
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

    /// Threshold for automatic parallelization
    private let parallelizationThreshold: Int

    public init(mode: Mode = .automatic, parallelizationThreshold: Int = 1000) {
        self.device = .cpu
        self.mode = mode
        self.processorCount = ProcessInfo.processInfo.activeProcessorCount
        self.parallelizationThreshold = parallelizationThreshold
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

    public func execute<T: Sendable>(
        _ work: @Sendable @escaping () async throws -> T
    ) async throws -> T {
        // Simple async execution
        try await work()
    }

    public func parallelExecute<T: Sendable>(
        items: Range<Int>,
        _ work: @Sendable @escaping (Int) async throws -> T
    ) async throws -> [T] {
        let count = items.count

        // Determine execution strategy
        let shouldParallelize: Bool = {
            switch mode {
            case .sequential:
                return false
            case .parallel:
                return true
            case .automatic:
                return count >= parallelizationThreshold
            }
        }()

        if shouldParallelize {
            // Parallel execution using TaskGroup
            return try await withThrowingTaskGroup(of: (Int, T).self) { group in
                // Create tasks for each item
                for i in items {
                    group.addTask {
                        let result = try await work(i)
                        return (i, result)
                    }
                }

                // Collect results
                var allResults: [(Int, T)] = []
                for try await result in group {
                    allResults.append(result)
                }

                // Sort by index to maintain order
                allResults.sort { $0.0 < $1.0 }
                return allResults.map { $0.1 }
            }
        } else {
            // Sequential execution
            var results: [T] = []
            results.reserveCapacity(count)

            for i in items {
                let result = try await work(i)
                results.append(result)
            }

            return results
        }
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
