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

    public init(mode: Mode = .automatic, parallelizationThreshold: Int = 50_000) {
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
            // Parallel execution using chunked TaskGroup to reduce scheduling overhead
            return try await withThrowingTaskGroup(of: [(Int, T)].self) { group in
                let total = count
                let cores = max(1, processorCount)
                let targetTasks = max(1, min(total, cores))
                let chunk = max(256, (total + targetTasks - 1) / targetTasks) // coarser chunks

                var start = items.lowerBound
                while start < items.upperBound {
                    let end = min(start + chunk, items.upperBound)
                    let range = start..<end
                    group.addTask {
                        var local: [(Int, T)] = []
                        local.reserveCapacity(range.count)
                        for i in range {
                            let result = try await work(i)
                            local.append((i, result))
                        }
                        return local
                    }
                    start = end
                }

                var allResults: [(Int, T)] = []
                allResults.reserveCapacity(total)
                for try await partial in group {
                    allResults.append(contentsOf: partial)
                }

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

// MARK: - Chunked helpers (override protocol defaults)

public extension CPUComputeProvider {
    func parallelForEach(
        items: Range<Int>,
        _ body: @Sendable @escaping (Int) async throws -> Void
    ) async throws {
        let count = items.count
        let shouldParallelize: Bool = {
            switch mode {
            case .sequential: return false
            case .parallel: return true
            case .automatic: return count >= parallelizationThreshold
            }
        }()

        if shouldParallelize {
            let total = count
            let cores = max(1, processorCount)
            let targetTasks = max(1, min(total, cores))
            let chunk = max(256, (total + targetTasks - 1) / targetTasks)
            try await withThrowingTaskGroup(of: Void.self) { group in
                var start = items.lowerBound
                while start < items.upperBound {
                    let end = min(start + chunk, items.upperBound)
                    let range = start..<end
                    group.addTask {
                        for i in range { try await body(i) }
                    }
                    start = end
                }
                for try await _ in group { _ = () }
            }
        } else {
            for i in items { try await body(i) }
        }
    }

    func parallelReduce<R: Sendable>(
        items: Range<Int>,
        initial: R,
        _ rangeWork: @Sendable @escaping (Range<Int>) async throws -> R,
        _ combine: @Sendable @escaping (R, R) -> R
    ) async throws -> R {
        let count = items.count
        if count == 0 { return initial }
        let shouldParallelize: Bool = {
            switch mode {
            case .sequential: return false
            case .parallel: return true
            case .automatic: return count >= parallelizationThreshold
            }
        }()

        if shouldParallelize {
            let total = count
            let cores = max(1, processorCount)
            let targetTasks = max(1, min(total, cores))
            let chunk = max(256, (total + targetTasks - 1) / targetTasks)
            return try await withThrowingTaskGroup(of: R.self) { group in
                var start = items.lowerBound
                while start < items.upperBound {
                    let end = min(start + chunk, items.upperBound)
                    let range = start..<end
                    group.addTask { try await rangeWork(range) }
                    start = end
                }
                var acc = initial
                for try await part in group { acc = combine(acc, part) }
                return acc
            }
        } else {
            // Sequential accumulation by invoking rangeWork on whole range
            return try await rangeWork(items)
        }
    }

    // Min-chunk hint variants
    func parallelForEach(
        items: Range<Int>,
        minChunk: Int,
        _ body: @Sendable @escaping (Int) async throws -> Void
    ) async throws {
        let count = items.count
        let shouldParallelize: Bool = {
            switch mode {
            case .sequential: return false
            case .parallel: return true
            case .automatic: return count >= parallelizationThreshold
            }
        }()

        if shouldParallelize {
            let total = count
            let cores = max(1, processorCount)
            let targetTasks = max(1, min(total, cores))
            let base = max(256, (total + targetTasks - 1) / targetTasks)
            let chunk = max(minChunk, base)
            try await withThrowingTaskGroup(of: Void.self) { group in
                var start = items.lowerBound
                while start < items.upperBound {
                    let end = min(start + chunk, items.upperBound)
                    let range = start..<end
                    group.addTask { for i in range { try await body(i) } }
                    start = end
                }
                for try await _ in group { _ = () }
            }
        } else {
            for i in items { try await body(i) }
        }
    }

    func parallelReduce<R: Sendable>(
        items: Range<Int>,
        initial: R,
        minChunk: Int,
        _ rangeWork: @Sendable @escaping (Range<Int>) async throws -> R,
        _ combine: @Sendable @escaping (R, R) -> R
    ) async throws -> R {
        let count = items.count
        if count == 0 { return initial }
        let shouldParallelize: Bool = {
            switch mode {
            case .sequential: return false
            case .parallel: return true
            case .automatic: return count >= parallelizationThreshold
            }
        }()

        if shouldParallelize {
            let total = count
            let cores = max(1, processorCount)
            let targetTasks = max(1, min(total, cores))
            let base = max(256, (total + targetTasks - 1) / targetTasks)
            let chunk = max(minChunk, base)
            return try await withThrowingTaskGroup(of: R.self) { group in
                var start = items.lowerBound
                while start < items.upperBound {
                    let end = min(start + chunk, items.upperBound)
                    let range = start..<end
                    group.addTask { try await rangeWork(range) }
                    start = end
                }
                var acc = initial
                for try await part in group { acc = combine(acc, part) }
                return acc
            }
        } else {
            return try await rangeWork(items)
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
