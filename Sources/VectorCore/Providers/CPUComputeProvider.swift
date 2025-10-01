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
    // Optional explicit overrides for automatic mode
    private let explicitThreshold: Int?
    private let explicitMinChunk: Int?

    /// Threshold for automatic parallelization
    private let parallelizationThreshold: Int

    public init(mode: Mode = .automatic, parallelizationThreshold: Int = 50_000) {
        self.device = .cpu
        self.mode = mode
        self.processorCount = ProcessInfo.processInfo.activeProcessorCount
        self.parallelizationThreshold = parallelizationThreshold
        self.explicitThreshold = nil
        self.explicitMinChunk = nil
    }

    // Overload supporting optional threshold and minChunk (non-breaking addition)
    public init(mode: Mode = .automatic, parallelizationThreshold: Int? = nil, minChunk: Int? = nil) {
        self.device = .cpu
        self.mode = mode
        self.processorCount = ProcessInfo.processInfo.activeProcessorCount
        self.parallelizationThreshold = parallelizationThreshold ?? 50_000
        self.explicitThreshold = parallelizationThreshold
        self.explicitMinChunk = minChunk
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
        if isEmpty { return initial }
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
        if isEmpty { return initial }
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

// MARK: - Auto-tuned batch execution helper

public extension CPUComputeProvider {
    /// Execute a batch kernel over candidates with auto-tuned mode and chunking when in automatic mode.
    /// - Parameters:
    ///   - query: The query object passed to the kernel
    ///   - candidates: The candidates array
    ///   - kernelKind: The kernel kind for tuning cache key
    ///   - dimension: Dimensionality (e.g., 512/768/1536)
    ///   - results: Output buffer with capacity = candidates.count
    ///   - kernel: Range kernel to execute (e.g., BatchKernels.range_*).
    func executeBatch<V: Sendable, R: Sendable>(
        query: V,
        candidates: [V],
        kernelKind: KernelKind,
        dimension: Int,
        results: UnsafeMutableBufferPointer<R>,
        kernel: @Sendable @escaping (V, [V], Range<Int>, UnsafeMutableBufferPointer<R>) -> Void
    ) {
        let n = candidates.count
        guard n > 0 else { return }
        // For now, avoid concurrent writes into non-Sendable pointers under Strict Concurrency.
        // Execute sequentially; callers seeking parallelism should use async parallelReduce helpers.
        kernel(query, candidates, 0..<n, results)
    }
}

private extension CPUComputeProvider {
    /// Measure probes for auto-tuning using the provided range kernel.
    func measureProbes<V: Sendable, R: Sendable>(
        dim: Int,
        kernel: @Sendable @escaping (V, [V], Range<Int>, UnsafeMutableBufferPointer<R>) -> Void,
        query: V,
        candidates: [V],
        results: UnsafeMutableBufferPointer<R>
    ) -> CalibrationProbes {
        // Selection of M per dim (bounded)
        let M: Int = {
            if dim >= 1536 { return 256 }
            if dim >= 768 { return 512 }
            return 1024
        }()

        let mN = Swift.min(candidates.count, M)
        guard mN > 64, let template = results.first else {
            let peff = Double(Swift.min(processorCount, 8)) * 0.7
            return CalibrationProbes(nsPerCandidateSeq: 200.0, parallelOverheadNs: 50_000.0, effectiveParallelFactor: peff)
        }

        // Prepare temporary output buffer
        var tmp = [R](repeating: template, count: mN)
        let clock = ContinuousClock()

        // Warm-up
        tmp.withUnsafeMutableBufferPointer { out in
            let warm = Swift.min(mN / 4, 128)
            if warm > 0 { kernel(query, candidates, 0..<warm, out) }
        }

        // Sequential measurement
        let tSeq = clock.measure {
            tmp.withUnsafeMutableBufferPointer { out in
                kernel(query, candidates, 0..<mN, out)
            }
        }.nanoseconds
        let a = tSeq / Double(mN)

        // Crude parallel model without spawning threads (to satisfy concurrency safety in sync context)
        // Assume effective factor based on cores; overhead estimated from heuristic.
        let peff = Double(min(processorCount, 8)) * 0.7
        let Tp = ParallelHeuristic.parallelOverheadNs(items: mN)
        return CalibrationProbes(nsPerCandidateSeq: a, parallelOverheadNs: Tp, effectiveParallelFactor: peff)
    }
}

fileprivate extension UnsafeMutableBufferPointer {
    var first: Element? { !isEmpty ? self[0] : nil }
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
