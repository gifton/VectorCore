//
//  MemoryPool.swift
//  VectorCore
//
//  Thread-safe memory pool for reusing temporary buffers to reduce allocation overhead
//

import Foundation

/// Thread-safe memory pool that reduces allocation overhead by reusing memory buffers
/// for temporary calculations in vector operations.
internal final class MemoryPool: @unchecked Sendable {
    // MARK: - Singleton

    /// Shared memory pool instance for application-wide buffer reuse
    public static let shared = MemoryPool()

    // MARK: - Configuration

    /// Configuration parameters for memory pool behavior
    internal struct Configuration {
        /// Maximum number of buffers to keep per size category
        public var maxBuffersPerSize: Int = 10
        /// Maximum total memory to use for pooled buffers (in bytes)
        public var maxTotalMemory: Int = 100_000_000  // 100MB
        /// Interval for automatic cleanup of unused buffers
        public var cleanupInterval: TimeInterval = 30.0

        public init() {}
    }

    // MARK: - Buffer Handle

    /// Handle to a borrowed buffer that automatically returns to pool on deallocation
    internal final class BufferHandle<T> {
        fileprivate let buffer: UnsafeMutableBufferPointer<T>
        fileprivate weak var pool: MemoryPool?
        fileprivate let sizeKey: Int
        fileprivate let alignment: Int

        /// Pointer to the buffer's base address
        public var pointer: UnsafeMutablePointer<T> {
            buffer.baseAddress!
        }

        /// Number of elements in the buffer
        public var count: Int {
            buffer.count
        }

        fileprivate init(
            buffer: UnsafeMutableBufferPointer<T>,
            pool: MemoryPool,
            sizeKey: Int,
            alignment: Int
        ) {
            self.buffer = buffer
            self.pool = pool
            self.sizeKey = sizeKey
            self.alignment = alignment
        }

        deinit {
            // Return buffer to pool
            pool?.returnBuffer(buffer, sizeKey: sizeKey, alignment: alignment)
        }
    }

    // MARK: - Pool Statistics

    /// Statistics about pool usage for monitoring and optimization
    internal struct PoolStatistics {
        public let totalAllocated: Int
        public let totalInUse: Int
        public let hitRate: Double
        public let bufferCountByType: [String: Int]
    }

    // MARK: - Private State

    /// Configuration
    private var configuration: Configuration

    /// Thread-safe access queue
    private let queue = DispatchQueue(label: "com.vectorcore.memorypool", attributes: .concurrent)

    /// Pool storage: [TypeID: [SizeKey: [BufferEntry]]]
    private var pools: [ObjectIdentifier: [Int: [BufferEntry]]] = [:]

    /// Statistics tracking
    private var stats = Statistics()

    /// Cleanup timer
    private var cleanupTimer: (any DispatchSourceTimer)?

    // MARK: - Internal Types

    private struct BufferEntry {
        let pointer: UnsafeMutableRawPointer
        let capacity: Int
        let alignment: Int
        var lastAccessed: Date
    }

    private struct Statistics {
        var totalAllocated: Int = 0
        var totalInUse: Int = 0
        var hits: Int = 0
        var misses: Int = 0
    }

    // MARK: - Initialization

    /// Initialize with optional configuration
    internal init(configuration: Configuration = Configuration()) {
        self.configuration = configuration
        setupCleanupTimer()
    }

    // MARK: - Public API

    /// Acquire a buffer from the pool
    internal func acquire<T>(
        type: T.Type,
        count: Int,
        alignment: Int = 16
    ) -> BufferHandle<T>? {
        let typeID = ObjectIdentifier(T.self)
        let sizeKey = roundUpToPowerOfTwo(count)

        // Try to get from pool
        var buffer: UnsafeMutableBufferPointer<T>?

        queue.sync(flags: .barrier) {
            if let typePool = pools[typeID],
               let sizePool = typePool[sizeKey],
               let index = sizePool.firstIndex(where: { $0.alignment >= alignment }) {

                // Found suitable buffer
                var entry = sizePool[index]
                entry.lastAccessed = Date()

                let typedPointer = entry.pointer.bindMemory(to: T.self, capacity: entry.capacity)
                buffer = UnsafeMutableBufferPointer(start: typedPointer, count: count)

                // Remove from pool
                pools[typeID]![sizeKey]!.remove(at: index)
                stats.hits += 1
                stats.totalInUse += 1
            } else {
                stats.misses += 1
            }
        }

        // Allocate new if not found
        if buffer == nil {
            let capacity = sizeKey
            // Try aligned allocation; if it fails, return nil (caller will handle)
            if let pointer = try? AlignedMemory.allocateAligned(type: T.self, count: capacity, alignment: alignment) {
                buffer = UnsafeMutableBufferPointer(start: pointer, count: count)
                queue.async(flags: .barrier) {
                    self.stats.totalAllocated += capacity * MemoryLayout<T>.stride
                    self.stats.totalInUse += 1
                }
            } else {
                return nil
            }
        }

        return BufferHandle(
            buffer: buffer!,
            pool: self,
            sizeKey: sizeKey,
            alignment: alignment
        )
    }

    /// Use a buffer temporarily with automatic return to pool
    internal func withBuffer<T, R>(
        type: T.Type,
        count: Int,
        alignment: Int = 16,
        _ body: (UnsafeMutableBufferPointer<T>) throws -> R
    ) rethrows -> R {
        guard let handle = acquire(type: type, count: count, alignment: alignment) else {
            // Fallback to direct aligned allocation; if that fails, rethrow allocation error up as VectorError via body
            if let pointer = try? AlignedMemory.allocateAligned(type: T.self, count: count, alignment: alignment) {
                // posix_memalign → must free with `free()`, not `.deallocate()`
                defer { AlignedMemory.deallocate(pointer) }
                let buffer = UnsafeMutableBufferPointer(start: pointer, count: count)
                return try body(buffer)
            } else {
                // As last resort, allocate with natural alignment to avoid immediate crash
                let raw = UnsafeMutableRawPointer.allocate(byteCount: count * MemoryLayout<T>.stride,
                                                           alignment: MemoryLayout<T>.alignment)
                let pointer = raw.assumingMemoryBound(to: T.self)
                defer { raw.deallocate() }
                let buffer = UnsafeMutableBufferPointer(start: pointer, count: count)
                return try body(buffer)
            }
        }

        return try body(handle.buffer)
    }

    /// Get current statistics
    internal var statistics: PoolStatistics {
        queue.sync {
            let hitRate = stats.hits + stats.misses > 0
                ? Double(stats.hits) / Double(stats.hits + stats.misses)
                : 0.0

            var bufferCounts: [String: Int] = [:]
            for (typeID, typePool) in pools {
                let typeName = String(describing: typeID)
                bufferCounts[typeName] = typePool.values.reduce(0) { $0 + $1.count }
            }

            return PoolStatistics(
                totalAllocated: stats.totalAllocated,
                totalInUse: stats.totalInUse,
                hitRate: hitRate,
                bufferCountByType: bufferCounts
            )
        }
    }

    /// Synchronize with the internal queue to ensure all pending
    /// pool operations (returns, stats updates, cleanups) have completed.
    /// Useful for tests to avoid arbitrary sleeps.
    internal func quiesce() {
        queue.sync(flags: .barrier) { }
    }

    /// Manually trigger cleanup
    internal func cleanup() {
        let cutoffDate = Date().addingTimeInterval(-configuration.cleanupInterval)

        queue.async(flags: .barrier) {
            var totalFreed = 0

            for (typeID, var typePool) in self.pools {
                for (sizeKey, var sizePool) in typePool {
                    // Remove old entries
                    sizePool.removeAll { entry in
                        if entry.lastAccessed < cutoffDate {
                            // Pooled entries originate from posix_memalign-backed allocations
                            AlignedMemory.deallocate(entry.pointer)
                            totalFreed += entry.capacity
                            return true
                        }
                        return false
                    }

                    // Remove empty size pools
                    if sizePool.isEmpty {
                        typePool.removeValue(forKey: sizeKey)
                    } else {
                        typePool[sizeKey] = sizePool
                    }
                }

                // Remove empty type pools
                if typePool.isEmpty {
                    self.pools.removeValue(forKey: typeID)
                } else {
                    self.pools[typeID] = typePool
                }
            }

            self.stats.totalAllocated -= totalFreed
        }
    }

    // MARK: - Private Methods

    /// Return buffer to pool
    fileprivate func returnBuffer<T>(
        _ buffer: UnsafeMutableBufferPointer<T>,
        sizeKey: Int,
        alignment: Int
    ) {
        guard let pointer = buffer.baseAddress else { return }

        let typeID = ObjectIdentifier(T.self)
        let rawPointer = UnsafeMutableRawPointer(pointer)

        // Capture pointer address as Int to make it Sendable
        let pointerAddress = Int(bitPattern: rawPointer)

        queue.async(flags: .barrier) { [weak self] in
            guard let self = self else { return }
            // Reconstruct pointer from address
            let rawPointer = UnsafeMutableRawPointer(bitPattern: pointerAddress)!
            // Initialize type pool if needed
            if self.pools[typeID] == nil {
                self.pools[typeID] = [:]
            }

            // Initialize size pool if needed
            if self.pools[typeID]![sizeKey] == nil {
                self.pools[typeID]![sizeKey] = []
            }

            // Check pool limits
            let currentCount = self.pools[typeID]![sizeKey]!.count
            let totalMemory = self.pools.values.flatMap { $0.values }.flatMap { $0 }.reduce(0) { $0 + $1.capacity }

            if currentCount < self.configuration.maxBuffersPerSize &&
                totalMemory < self.configuration.maxTotalMemory {
                // Add to pool
                let entry = BufferEntry(
                    pointer: rawPointer,
                    capacity: sizeKey,
                    alignment: alignment,
                    lastAccessed: Date()
                )
                self.pools[typeID]![sizeKey]!.append(entry)
            } else {
                // Pool full, deallocate (posix_memalign-backed pointer)
                AlignedMemory.deallocate(rawPointer)
                self.stats.totalAllocated -= sizeKey * MemoryLayout<T>.stride
            }

            self.stats.totalInUse -= 1
        }
    }

    /// Round up to next power of two
    private func roundUpToPowerOfTwo(_ n: Int) -> Int {
        guard n > 0 else { return 1 }

        var power = 1
        while power < n && power <= Int.max / 2 {
            power *= 2
        }
        return power
    }

    /// Setup cleanup timer
    private func setupCleanupTimer() {
        // Only create timer if cleanup interval is finite
        guard configuration.cleanupInterval.isFinite && configuration.cleanupInterval > 0 else {
            return
        }

        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(
            deadline: .now() + configuration.cleanupInterval,
            repeating: configuration.cleanupInterval
        )
        timer.setEventHandler { [weak self] in
            self?.cleanup()
        }
        timer.resume()
        cleanupTimer = timer
    }

    deinit {
        cleanupTimer?.cancel()

        // Deallocate all buffers
        for typePool in pools.values {
            for sizePool in typePool.values {
                for entry in sizePool {
                    // Pooled entries are aligned allocations
                    AlignedMemory.deallocate(entry.pointer)
                }
            }
        }
    }
}
