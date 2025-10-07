// VectorCore: Pure Swift Buffer Pool
//
// Cross-platform memory management without dependencies
//

import Foundation

/// Pure Swift implementation of BufferProvider
///
/// Provides efficient buffer pooling using actor isolation for thread safety.
/// Implements power-of-two sizing for optimal reuse.
///
/// ## Design Features
/// - Actor-based thread safety
/// - Power-of-two buffer sizes
/// - LRU-style cleanup
/// - Statistics tracking
public actor SwiftBufferPool: BufferProvider {

    private let configuration: BufferConfiguration

    /// Pools organized by size (power of two)
    private var pools: [Int: [BufferEntry]] = [:]

    /// Active handles
    private var activeHandles: Set<UUID> = []

    /// Statistics
    private var stats = Stats()

    private struct BufferEntry {
        let buffer: UnsafeMutableRawPointer
        let size: Int
        var lastUsed: Date
    }

    private struct Stats {
        var totalAllocations: Int = 0
        var reusedBuffers: Int = 0
        var currentUsageBytes: Int = 0
        var peakUsageBytes: Int = 0
    }

    public init(configuration: BufferConfiguration = .default) {
        self.configuration = configuration
    }

    deinit {
        // Actor's deinit automatically handles cleanup
        // Buffer deallocation happens in clear() method
    }

    // MARK: - BufferProvider Implementation

    public nonisolated var alignment: Int { configuration.alignment }

    public func acquire(size: Int) async throws -> BufferHandle {
        // Round up to power of two for better reuse
        let roundedSize = nextPowerOfTwo(size)

        // Try to reuse from pool
        if var poolForSize = pools[roundedSize], !poolForSize.isEmpty {
            var entry = poolForSize.removeLast()
            entry.lastUsed = Date()
            pools[roundedSize] = poolForSize

            let handle = BufferHandle(size: size, pointer: entry.buffer)
            activeHandles.insert(handle.id)

            stats.reusedBuffers += 1
            stats.currentUsageBytes += roundedSize
            stats.peakUsageBytes = max(stats.peakUsageBytes, stats.currentUsageBytes)

            return handle
        }

        // Allocate new buffer
        guard let buffer = allocateAligned(size: roundedSize, alignment: alignment) else {
            throw BufferError.allocationFailed(size: size)
        }

        let handle = BufferHandle(size: size, pointer: buffer)
        activeHandles.insert(handle.id)

        stats.totalAllocations += 1
        stats.currentUsageBytes += roundedSize
        stats.peakUsageBytes = max(stats.peakUsageBytes, stats.currentUsageBytes)

        return handle
    }

    public func release(_ handle: BufferHandle) async {
        guard activeHandles.remove(handle.id) != nil else {
            return // Already released
        }

        let roundedSize = nextPowerOfTwo(handle.size)
        stats.currentUsageBytes -= roundedSize

        // Return to pool if under size limit
        if roundedSize <= configuration.maxBufferSize {
            let entry = BufferEntry(
                buffer: handle.pointer,
                size: roundedSize,
                lastUsed: Date()
            )

            if pools[roundedSize] != nil {
                pools[roundedSize]!.append(entry)
            } else {
                pools[roundedSize] = [entry]
            }

            // Cleanup if pool is too large
            await cleanupIfNeeded()
        } else {
            // Too large for pooling, deallocate immediately
            AlignedMemory.deallocate(handle.pointer)
        }
    }

    public func statistics() async -> BufferStatistics {
        BufferStatistics(
            totalAllocations: stats.totalAllocations,
            reusedBuffers: stats.reusedBuffers,
            currentUsageBytes: stats.currentUsageBytes,
            peakUsageBytes: stats.peakUsageBytes
        )
    }

    public func clear() async {
        // Deallocate all pooled buffers
        for (_, entries) in pools {
            for entry in entries {
                AlignedMemory.deallocate(entry.buffer)
            }
        }
        pools.removeAll()
    }

    // MARK: - Private Helpers

    private func nextPowerOfTwo(_ n: Int) -> Int {
        guard n > 0 else { return 1 }
        var power = 1
        while power < n {
            power *= 2
        }
        return power
    }

    private func allocateAligned(size: Int, alignment: Int) -> UnsafeMutableRawPointer? {
        #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
        // Use posix_memalign on Darwin platforms
        var pointer: UnsafeMutableRawPointer?
        let result = posix_memalign(&pointer, alignment, size)
        return result == 0 ? pointer : nil
        #else
        // Use aligned_alloc on Linux (C11 standard)
        return UnsafeMutableRawPointer(aligned_alloc(alignment, size))
        #endif
    }

    private func cleanupIfNeeded() async {
        // Calculate total pooled memory
        var totalPooledBytes = 0
        for (size, entries) in pools {
            totalPooledBytes += size * entries.count
        }

        // Clean up if over limit
        if totalPooledBytes > configuration.maxPoolSize {
            let cutoffDate = Date().addingTimeInterval(-60) // 1 minute ago

            // Remove old entries
            for (size, entries) in pools {
                let filtered = entries.filter { $0.lastUsed > cutoffDate }
                let removed = entries.filter { $0.lastUsed <= cutoffDate }

                // Deallocate removed buffers
                for entry in removed {
                    AlignedMemory.deallocate(entry.buffer)
                }

                if filtered.isEmpty {
                    pools.removeValue(forKey: size)
                } else {
                    pools[size] = filtered
                }
            }
        }
    }
}

// MARK: - Singleton Instance

public extension SwiftBufferPool {
    /// Shared buffer pool instance
    static let shared = SwiftBufferPool()
}
