// VectorCore: Buffer Provider Protocol
//
// Memory management abstraction without platform dependencies
//

import Foundation

/// Handle to a managed buffer
public struct BufferHandle: Sendable, Hashable {
    /// Unique identifier for this buffer
    public let id: UUID
    
    /// Size of the buffer in bytes
    public let size: Int
    
    /// Pointer to the buffer memory (as integer for Sendable)
    private let pointerAddress: Int
    
    /// Access the pointer
    /// - Warning: Only valid while handle is active
    public var pointer: UnsafeMutableRawPointer {
        UnsafeMutableRawPointer(bitPattern: pointerAddress)!
    }
    
    public init(id: UUID = UUID(), size: Int, pointer: UnsafeMutableRawPointer) {
        self.id = id
        self.size = size
        self.pointerAddress = Int(bitPattern: pointer)
    }
}

/// Statistics about buffer pool usage
public struct BufferStatistics: Sendable {
    /// Total number of allocations made
    public let totalAllocations: Int
    
    /// Number of times a buffer was reused
    public let reusedBuffers: Int
    
    /// Current memory usage in bytes
    public let currentUsageBytes: Int
    
    /// Peak memory usage in bytes
    public let peakUsageBytes: Int
    
    /// Hit rate (reused / total requests)
    public var hitRate: Double {
        let total = totalAllocations + reusedBuffers
        guard total > 0 else { return 0 }
        return Double(reusedBuffers) / Double(total)
    }
    
    public init(
        totalAllocations: Int,
        reusedBuffers: Int,
        currentUsageBytes: Int,
        peakUsageBytes: Int
    ) {
        self.totalAllocations = totalAllocations
        self.reusedBuffers = reusedBuffers
        self.currentUsageBytes = currentUsageBytes
        self.peakUsageBytes = peakUsageBytes
    }
}

/// Protocol for managed memory buffers
///
/// BufferProvider abstracts memory management for vector operations,
/// enabling efficient buffer reuse and platform-specific optimizations.
///
/// ## Design Principles
/// - Async-safe with actor isolation
/// - Supports buffer pooling for efficiency
/// - Platform-agnostic interface
/// - Statistics for performance monitoring
///
/// ## Usage Pattern
/// ```swift
/// let handle = try await provider.acquire(size: 1024)
/// defer { await provider.release(handle) }
/// // Use handle.pointer for operations
/// ```
public protocol BufferProvider: Sendable {
    
    /// Acquire a buffer of at least the specified size
    ///
    /// - Parameter size: Minimum size needed in bytes
    /// - Returns: Handle to the buffer
    /// - Throws: If allocation fails
    /// - Note: Actual buffer may be larger than requested
    func acquire(size: Int) async throws -> BufferHandle
    
    /// Release a buffer back to the pool
    ///
    /// - Parameter handle: Buffer to release
    /// - Note: Buffer contents are undefined after release
    func release(_ handle: BufferHandle) async
    
    /// Get current statistics
    func statistics() async -> BufferStatistics
    
    /// Clear all cached buffers
    ///
    /// - Note: Active handles remain valid
    func clear() async
    
    /// Preferred alignment for buffers
    var alignment: Int { get }
}

// MARK: - Default Implementation

public extension BufferProvider {
    /// Default alignment (64 bytes for cache lines)
    var alignment: Int { 64 }
}

/// Errors that can occur in buffer operations
public enum BufferError: Error, Sendable {
    case allocationFailed(size: Int)
    case invalidHandle
    case outOfMemory
}

// MARK: - Helper Types

/// Configuration for buffer providers
public struct BufferConfiguration: Sendable {
    /// Maximum memory to use for pooling
    public let maxPoolSize: Int
    
    /// Maximum size of individual buffers to pool
    public let maxBufferSize: Int
    
    /// Alignment requirement
    public let alignment: Int
    
    /// Enable statistics tracking
    public let trackStatistics: Bool
    
    public init(
        maxPoolSize: Int = 100_000_000, // 100MB
        maxBufferSize: Int = 10_000_000, // 10MB
        alignment: Int = 64,
        trackStatistics: Bool = true
    ) {
        self.maxPoolSize = maxPoolSize
        self.maxBufferSize = maxBufferSize
        self.alignment = alignment
        self.trackStatistics = trackStatistics
    }
    
    /// Default configuration
    public static let `default` = BufferConfiguration()
    
    /// Configuration for minimal memory usage
    public static let minimal = BufferConfiguration(
        maxPoolSize: 10_000_000,  // 10MB
        maxBufferSize: 1_000_000, // 1MB
        trackStatistics: false
    )
}