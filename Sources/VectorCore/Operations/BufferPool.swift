//
//  BufferPool.swift
//  VectorCore
//
//

import Foundation

// MARK: - Buffer Wrapper

/// Wrapper for unsafe buffer pointers to make them Sendable
internal struct SendableBuffer: @unchecked Sendable {
    public let buffer: UnsafeMutableBufferPointer<Float>

    init(_ buffer: UnsafeMutableBufferPointer<Float>) {
        self.buffer = buffer
    }
}

// MARK: - Buffer Pool

/// Thread-safe pool for reusing memory buffers
internal actor BufferPool {
    /// Available buffers organized by size
    private var available: [Int: [UnsafeMutableBufferPointer<Float>]] = [:]

    /// Statistics for monitoring pool performance
    private var stats = Statistics()

    /// Maximum number of buffers to keep per size
    private let maxBuffersPerSize: Int

    /// Total memory limit in bytes
    private let memoryLimit: Int

    /// Current memory usage in bytes
    private var currentMemoryUsage: Int = 0

    // MARK: - Statistics

    public struct Statistics {
        var acquisitions: Int = 0
        var releases: Int = 0
        var allocations: Int = 0
        var reuses: Int = 0
        var evictions: Int = 0

        var reuseRate: Double {
            guard acquisitions > 0 else { return 0 }
            return Double(reuses) / Double(acquisitions)
        }
    }

    // MARK: - Initialization

    /// Initialize buffer pool with memory limits
    /// - Parameters:
    ///   - maxBuffersPerSize: Maximum buffers to keep for each size
    ///   - memoryLimit: Total memory limit in MB (default 256MB)
    internal init(maxBuffersPerSize: Int = 10, memoryLimitMB: Int = 256) {
        self.maxBuffersPerSize = maxBuffersPerSize
        self.memoryLimit = memoryLimitMB * 1024 * 1024
    }

    // MARK: - Buffer Management

    /// Acquire a buffer of specified size
    internal func acquire(count: Int) -> SendableBuffer {
        stats.acquisitions += 1

        // Check if we have an available buffer
        if let buffer = available[count]?.popLast() {
            stats.reuses += 1
            return SendableBuffer(buffer)
        }

        // Allocate new buffer
        stats.allocations += 1
        let memory = UnsafeMutablePointer<Float>.allocate(capacity: count)
        let buffer = UnsafeMutableBufferPointer(start: memory, count: count)

        // Update memory usage
        currentMemoryUsage += count * MemoryLayout<Float>.size

        // Evict old buffers if over memory limit
        if currentMemoryUsage > memoryLimit {
            evictOldBuffers()
        }

        return SendableBuffer(buffer)
    }

    /// Release a buffer back to the pool
    internal func release(_ sendableBuffer: SendableBuffer) {
        let buffer = sendableBuffer.buffer
        stats.releases += 1

        let count = buffer.count

        // Check if we should keep this buffer
        let buffersForSize = available[count]?.count ?? 0

        if buffersForSize < maxBuffersPerSize {
            // Add to pool
            available[count, default: []].append(buffer)
        } else {
            // Deallocate
            deallocateBuffer(buffer)
        }
    }

    /// Clear all buffers from the pool
    internal func clear() {
        for (_, buffers) in available {
            for buffer in buffers {
                deallocateBuffer(buffer)
            }
        }
        available.removeAll()
        currentMemoryUsage = 0
    }

    /// Get current statistics
    internal func getStatistics() -> Statistics {
        stats
    }

    // MARK: - Private Methods

    /// Deallocate a buffer and update memory usage
    private func deallocateBuffer(_ buffer: UnsafeMutableBufferPointer<Float>) {
        buffer.baseAddress?.deallocate()
        currentMemoryUsage -= buffer.count * MemoryLayout<Float>.size
    }

    /// Evict old buffers when memory limit is exceeded
    private func evictOldBuffers() {
        // Simple strategy: remove largest buffers first
        let sortedSizes = available.keys.sorted(by: >)

        for size in sortedSizes {
            guard currentMemoryUsage > memoryLimit else { break }

            if var buffers = available[size], !buffers.isEmpty {
                let buffer = buffers.removeLast()
                deallocateBuffer(buffer)
                stats.evictions += 1

                if buffers.isEmpty {
                    available.removeValue(forKey: size)
                } else {
                    available[size] = buffers
                }
            }
        }
    }

    /// Clean up all buffers - call this before deinit
    internal func cleanup() {
        // Clean up all remaining buffers
        for (_, buffers) in available {
            for buffer in buffers {
                buffer.baseAddress?.deallocate()
            }
        }
        available.removeAll()
        currentMemoryUsage = 0
    }
}

// MARK: - Typed Buffer Pool

/// Generic buffer pool for any type
public actor TypedBufferPool<Element> {
    private var available: [Int: [UnsafeMutableBufferPointer<Element>]] = [:]
    private let maxBuffersPerSize: Int

    public init(maxBuffersPerSize: Int = 10) {
        self.maxBuffersPerSize = maxBuffersPerSize
    }

    public func acquire(count: Int) -> UnsafeMutableBufferPointer<Element> {
        if let buffer = available[count]?.popLast() {
            return buffer
        }

        let memory = UnsafeMutablePointer<Element>.allocate(capacity: count)
        return UnsafeMutableBufferPointer(start: memory, count: count)
    }

    public func release(_ buffer: UnsafeMutableBufferPointer<Element>) {
        let count = buffer.count
        let buffersForSize = available[count]?.count ?? 0

        if buffersForSize < maxBuffersPerSize {
            available[count, default: []].append(buffer)
        } else {
            buffer.baseAddress?.deallocate()
        }
    }
}

// MARK: - Global Buffer Pool

/// Shared buffer pool for vector operations (internal)
internal let globalBufferPool = BufferPool()

// MARK: - Buffer Pool Context

/// Context for using buffer pool in operations
internal struct BufferPoolContext {
    private let pool: BufferPool

    internal init(pool: BufferPool = globalBufferPool) {
        self.pool = pool
    }

    /// Use a temporary buffer for an operation
    internal func withTemporaryBuffer<T>(
        count: Int,
        _ body: (UnsafeMutableBufferPointer<Float>) async throws -> T
    ) async rethrows -> T where T: Sendable {
        let sendableBuffer = await pool.acquire(count: count)
        do {
            let result = try await body(sendableBuffer.buffer)
            await pool.release(sendableBuffer)
            return result
        } catch {
            await pool.release(sendableBuffer)
            throw error
        }
    }

    /// Use multiple temporary buffers
    internal func withTemporaryBuffers<T>(
        counts: [Int],
        _ body: ([UnsafeMutableBufferPointer<Float>]) async throws -> T
    ) async rethrows -> T where T: Sendable {
        var sendableBuffers: [SendableBuffer] = []

        // Acquire all buffers
        for count in counts {
            sendableBuffers.append(await pool.acquire(count: count))
        }

        do {
            let buffers = sendableBuffers.map { $0.buffer }
            let result = try await body(buffers)
            // Release all buffers
            for sendableBuffer in sendableBuffers {
                await pool.release(sendableBuffer)
            }
            return result
        } catch {
            // Ensure cleanup even on error
            for sendableBuffer in sendableBuffers {
                await pool.release(sendableBuffer)
            }
            throw error
        }
    }
}
