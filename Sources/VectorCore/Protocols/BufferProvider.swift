// VectorCore: Buffer Provider Protocol
//
// Memory management abstraction without platform dependencies
//

import Foundation

/// Immutable borrowed buffer metadata. Retain its provider until release.
/// Sending metadata does not synchronize access to its pointee.
public struct BufferHandle: Sendable, Hashable {
    /// Unique identifier for this buffer
    public let id: UUID

    /// Size of the buffer in bytes
    public let size: Int

    /// Pointer to the buffer memory (as integer for Sendable)
    private let pointerAddress: Int

    /// Access the pointer
    /// - Warning: Only valid while the provider is alive and the handle is active.
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
        guard totalAllocations >= 0, reusedBuffers >= 0 else { return 0 }
        let total = Double(totalAllocations) + Double(reusedBuffers)
        guard total > 0 else { return 0 }
        return Double(reusedBuffers) / total
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
/// BufferProvider abstracts buffer acquisition and release for vector operations,
/// so a conformer can reuse buffers or use platform-specific allocation.
///
/// ## Design Principles
/// - Async-safe with actor isolation
/// - Supports buffer pooling for efficiency
/// - Platform-agnostic interface
/// - Statistics for performance monitoring
///
/// ## Usage Pattern
/// ```swift
/// let address = try await provider.withBuffer(size: 1024) { handle in
///     UInt(bitPattern: handle.pointer)
/// }
/// // The returned address is diagnostic metadata; its lease has ended.
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
    /// Acquires uninitialized storage, retains the provider through the body, and
    /// awaits release before returning or rethrowing the body's error. Cancellation
    /// does not release storage until the body finishes or throws. The caller must
    /// deinitialize every initialized element before the body exits and must not
    /// let a usable pointer escape its lease.
    func withBuffer<R: Sendable>(
        size: Int, _ body: @Sendable (BufferHandle) async throws -> R
    ) async throws -> R {
        let handle = try await acquire(size: size)
        do {
            let result = try await body(handle)
            await release(handle)
            return result
        } catch {
            await release(handle)
            throw error
        }
    }

    /// Default alignment (64 bytes for cache lines)
    var alignment: Int { 64 }
}

/// Errors that can occur in buffer operations
public enum BufferError: Error, Sendable {
    /// Negative sizes or unrepresentable physical bucket sizes.
    case invalidRequest(size: Int)
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
        guard maxPoolSize >= 0, maxBufferSize >= 0,
              PoolRequest.validAlignment(alignment) else {
            fatalError("BufferConfiguration requires nonnegative budgets and power-of-two alignment >= 16")
        }
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
