//
//  ExecutionContext.swift
//  VectorCore
//
//

import Foundation

// MARK: - Compute Device
// ComputeDevice is now defined in Execution/ComputeDevice.swift
// Import it to use here

// MARK: - Execution Context Protocol

/// Protocol for execution contexts that manage how vector operations are performed
public protocol ExecutionContext: Sendable {
    /// The compute device used by this context
    var device: ComputeDevice { get }
    
    /// Maximum number of concurrent threads/operations
    var maxThreadCount: Int { get }
    
    /// Preferred chunk size for parallel operations
    var preferredChunkSize: Int { get }
    
    /// Execute work within this context
    func execute<T>(_ work: @Sendable @escaping () throws -> T) async throws -> T where T: Sendable
    
    /// Execute work with a specific priority
    func execute<T>(
        priority: TaskPriority?,
        _ work: @Sendable @escaping () throws -> T
    ) async throws -> T where T: Sendable
}

// MARK: - Default Implementation

extension ExecutionContext {
    /// Default implementation with nil priority
    public func execute<T>(
        priority: TaskPriority? = nil,
        _ work: @Sendable @escaping () throws -> T
    ) async throws -> T where T: Sendable {
        try await execute(work)
    }
}

// MARK: - Default Context

extension ExecutionContext where Self == CPUContext {
    /// Default execution context - automatically uses available CPU cores
    public static var `default`: any ExecutionContext { 
        CPUContext.automatic 
    }
}

// MARK: - Error Types

extension VectorError {
    /// Create unsupported device error
    public static func unsupportedDevice(_ message: String) -> VectorError {
        VectorError.invalidDimension(0, reason: "Unsupported device: \(message)")
    }
}