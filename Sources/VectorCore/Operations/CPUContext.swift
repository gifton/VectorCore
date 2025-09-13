//
//  CPUContext.swift
//  VectorCore
//
//

import Foundation
import Dispatch

// MARK: - CPU Context

/// CPU-based execution context with configurable parallelism
public struct CPUContext: ExecutionContext {
    public let device = ComputeDevice.cpu
    public let maxThreadCount: Int
    public let preferredChunkSize: Int
    
    private let queue: DispatchQueue?
    private let qosClass: DispatchQoS.QoSClass
    
    // MARK: - Preset Configurations
    
    /// Sequential execution on a single thread
    public static let sequential = CPUContext(threadCount: 1)
    
    /// Automatic parallelization using all available cores
    public static let automatic = CPUContext(
        threadCount: ProcessInfo.processInfo.activeProcessorCount
    )
    
    /// High-performance configuration with user-interactive priority
    public static let highPerformance = CPUContext(
        threadCount: ProcessInfo.processInfo.activeProcessorCount,
        qos: .userInteractive
    )
    
    /// Background processing configuration
    public static let background = CPUContext(
        threadCount: max(1, ProcessInfo.processInfo.activeProcessorCount / 2),
        qos: .background
    )
    
    // MARK: - Initialization
    
    /// Initialize CPU context with custom configuration
    /// - Parameters:
    ///   - threadCount: Number of threads to use (nil for automatic)
    ///   - queue: Custom dispatch queue (nil to create one)
    ///   - qos: Quality of service class
    public init(
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
    public func execute<T>(_ work: @Sendable @escaping () throws -> T) async throws -> T where T: Sendable {
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
    public func execute<T>(
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

// MARK: - Helpers

extension CPUContext {
    /// Calculate optimal chunk size for a given workload
    public func optimalChunkSize(for itemCount: Int) -> Int {
        guard itemCount > 0 else { return 1 }
        
        // For small workloads, use a single chunk
        if itemCount <= preferredChunkSize {
            return itemCount
        }
        
        // For large workloads, divide evenly among threads
        let idealChunkCount = maxThreadCount
        let baseChunkSize = itemCount / idealChunkCount
        
        // Round up to preferred chunk size for cache efficiency
        if baseChunkSize < preferredChunkSize {
            return preferredChunkSize
        }
        
        // Ensure chunks are reasonably sized
        return min(baseChunkSize, preferredChunkSize * 4)
    }
    
    /// Create task groups for parallel execution
    public func withParallelExecution<T>(
        of items: Range<Int>,
        _ operation: @Sendable @escaping (Int) async throws -> T
    ) async throws -> [T] where T: Sendable {
        let chunkSize = optimalChunkSize(for: items.count)
        
        return try await withThrowingTaskGroup(of: (Int, T).self) { group in
            // Add tasks for each chunk
            for chunkStart in stride(from: items.lowerBound, to: items.upperBound, by: chunkSize) {
                let chunkEnd = min(chunkStart + chunkSize, items.upperBound)
                
                for index in chunkStart..<chunkEnd {
                    group.addTask {
                        let result = try await operation(index)
                        return (index, result)
                    }
                }
            }
            
            // Collect results in order
            var results = Array<T?>(repeating: nil, count: items.count)
            for try await (index, result) in group {
                results[index - items.lowerBound] = result
            }
            
            return results.compactMap { $0 }
        }
    }
}

// MARK: - Performance Hints

extension CPUContext {
    /// Provides hints about expected performance characteristics
    public struct PerformanceHints {
        /// Whether operations will be parallelized
        public let isParallel: Bool
        
        /// Expected speedup factor from parallelization
        public let expectedSpeedup: Double
        
        /// Recommended minimum workload size for parallelization
        public let parallelizationThreshold: Int
    }
    
    /// Get performance hints for this context
    public var performanceHints: PerformanceHints {
        PerformanceHints(
            isParallel: maxThreadCount > 1,
            expectedSpeedup: Double(maxThreadCount) * 0.8, // Account for overhead
            parallelizationThreshold: preferredChunkSize * 2
        )
    }
}