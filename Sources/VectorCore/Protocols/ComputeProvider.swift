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
/// - Thread-safe by requiring Sendable conformance
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
    
    /// Maximum concurrency level supported
    var maxConcurrency: Int { get }
    
    /// Execute work on the compute device
    ///
    /// - Parameter work: The work to execute
    /// - Returns: The result of the work
    /// - Throws: Any error from the work closure
    func execute<T: Sendable>(
        _ work: @Sendable @escaping () async throws -> T
    ) async throws -> T
    
    /// Execute work in parallel across items
    ///
    /// - Parameters:
    ///   - items: Range of items to process
    ///   - work: Work to perform for each item
    /// - Returns: Array of results in order
    /// - Throws: Any error from the work closures
    func parallelExecute<T: Sendable>(
        items: Range<Int>,
        _ work: @Sendable @escaping (Int) async throws -> T
    ) async throws -> [T]
    
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
    /// Default parallel execution using concurrent tasks
    func parallelExecute<T: Sendable>(
        items: Range<Int>,
        _ work: @Sendable @escaping (Int) async throws -> T
    ) async throws -> [T] {
        try await withThrowingTaskGroup(of: (Int, T).self) { group in
            for i in items {
                group.addTask {
                    (i, try await work(i))
                }
            }
            
            var results = [(Int, T)]()
            for try await result in group {
                results.append(result)
            }
            
            // Sort by index to maintain order
            results.sort { $0.0 < $1.0 }
            return results.map { $0.1 }
        }
    }
}