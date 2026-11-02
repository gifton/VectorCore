//
//  CPUContext.swift
//  VectorCore
//
//

import Foundation
import Dispatch

// MARK: - CPU Context

/// CPU-based execution context with configurable parallelism
///
/// - Note: This API is internal. ExecutionContext is an internal concept; use Operations/BatchOperations for public APIs.
internal struct CPUContext: ExecutionContext {
    internal let device = ComputeDevice.cpu
    internal let maxThreadCount: Int
    internal let preferredChunkSize: Int

    private let queue: DispatchQueue?
    private let qosClass: DispatchQoS.QoSClass

    // MARK: - Preset Configurations

    /// Sequential execution on a single thread
    internal static let sequential = CPUContext(threadCount: 1)

    /// Automatic parallelization using all available cores
    internal static let automatic = CPUContext(
        threadCount: ProcessInfo.processInfo.activeProcessorCount
    )

    /// All active cores at `.userInteractive` quality of service
    internal static let highPerformance = CPUContext(
        threadCount: ProcessInfo.processInfo.activeProcessorCount,
        qos: .userInteractive
    )

    /// Background processing configuration
    internal static let background = CPUContext(
        threadCount: max(1, ProcessInfo.processInfo.activeProcessorCount / 2),
        qos: .background
    )

    // MARK: - Initialization

    /// Initialize CPU context with custom configuration
    /// - Parameters:
    ///   - threadCount: Number of threads to use (nil for automatic)
    ///   - queue: Custom dispatch queue (nil to create one)
    ///   - qos: Quality of service class
    internal init(
        threadCount: Int? = nil,
        queue: DispatchQueue? = nil,
        qos: DispatchQoS.QoSClass = .default
    ) {
        self.maxThreadCount = threadCount ?? ProcessInfo.processInfo.activeProcessorCount
        self.qosClass = qos

        // Create queue if not provided
        if let queue = queue {
            self.queue = queue
        } else if maxThreadCount == 1 {
            // Serial queue for sequential execution
            self.queue = DispatchQueue(
                label: "com.vectorcore.cpu.sequential",
                qos: DispatchQoS(qosClass: qos, relativePriority: 0)
            )
        } else {
            // Concurrent queue for parallel execution
            self.queue = DispatchQueue(
                label: "com.vectorcore.cpu.parallel",
                qos: DispatchQoS(qosClass: qos, relativePriority: 0),
                attributes: .concurrent
            )
        }

        // Optimize chunk size for cache efficiency
        #if os(iOS) || arch(arm64)
        // Apple Silicon: larger L1 cache, optimize for it
        self.preferredChunkSize = 16384 / MemoryLayout<Float>.size  // 16KB chunks
        #else
        // Intel: standard cache line optimization
        self.preferredChunkSize = 8192 / MemoryLayout<Float>.size   // 8KB chunks
        #endif
    }

    // MARK: - Execution

    /// Execute work within the CPU context
    internal func execute<T>(_ work: @Sendable @escaping () throws -> T) async throws -> T where T: Sendable {
        if let queue = queue {
            return try await withCheckedThrowingContinuation { continuation in
                queue.async {
                    do {
                        let result = try work()
                        continuation.resume(returning: result)
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
            }
        } else {
            // Direct execution on current thread
            return try work()
        }
    }

    /// Execute work with specific priority
    internal func execute<T>(
        priority: TaskPriority?,
        _ work: @Sendable @escaping () throws -> T
    ) async throws -> T where T: Sendable {
        if let priority = priority {
            return try await Task(priority: priority) {
                try await execute(work)
            }.value
        } else {
            return try await execute(work)
        }
    }
}

// MARK: - Performance Hints

extension CPUContext {
    /// Provides hints about expected performance characteristics
    internal struct PerformanceHints {
        /// Whether operations will be parallelized
        internal let isParallel: Bool

        /// Expected speedup factor from parallelization
        internal let expectedSpeedup: Double

        /// Recommended minimum workload size for parallelization
        internal let parallelizationThreshold: Int
    }

    /// Get performance hints for this context
    internal var performanceHints: PerformanceHints {
        PerformanceHints(
            isParallel: maxThreadCount > 1,
            expectedSpeedup: Double(maxThreadCount) * 0.8, // Account for overhead
            parallelizationThreshold: preferredChunkSize * 2
        )
    }
}
