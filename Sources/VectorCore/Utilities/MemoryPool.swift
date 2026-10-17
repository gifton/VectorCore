import Foundation

/// Reuses uninitialized storage. Pool bookkeeping is synchronized by stateLock;
/// handles and their pointees are not Sendable. The standalone regression driver
/// exercises concurrent acquire/return/quiesce with an external deadline.
public final class MemoryPool: @unchecked Sendable {
    public static let shared = MemoryPool()

    public struct Configuration {
        public var maxBuffersPerSize: Int = 10
        /// Strict cap on cached physical payload bytes, excluding live handles.
        public var maxTotalMemory: Int = 100_000_000
        /// Nonpositive, nonfinite or beyond the dispatch horizon disables the timer.
        /// Explicit cleanup still uses this interval as its age cutoff.
        public var cleanupInterval: TimeInterval = 30
        public init() {}
    }

    /// Owns its allocation even after the pool is destroyed. Initialize and
    /// deinitialize exactly the used elements before releasing the handle.
    public final class BufferHandle<T> {
        fileprivate let allocation: PoolAllocation
        fileprivate weak var pool: MemoryPool?
        public let count: Int
        public var pointer: UnsafeMutablePointer<T> { allocation.pointer.assumingMemoryBound(to: T.self) }
        fileprivate var buffer: UnsafeMutableBufferPointer<T> { UnsafeMutableBufferPointer(start: pointer, count: count) }

        fileprivate init(allocation: PoolAllocation, count: Int, pool: MemoryPool) {
            self.allocation = allocation
            self.count = count
            self.pool = pool
        }
        deinit { pool?.returnBuffer(allocation, type: T.self) }
    }

    internal struct PoolStatistics {
        let totalAllocated: Int
        let totalInUse: Int
        let cachedBytes: Int
        let hitRate: Double
        let bufferCountByType: [String: Int]
    }

    private let configuration: Configuration
    private let allocationBoundary: any PoolAllocationBoundary
    private let stateLock = NSLock()
    private var pools: [ObjectIdentifier: [Int: [BufferEntry]]] = [:]
    private var stats = Statistics()
    private var cachedByteCount = 0
    private var cleanupTimer: (any DispatchSourceTimer)?

    private struct BufferEntry {
        let allocation: PoolAllocation
        let lastAccessed: Date
    }
    private struct Statistics {
        var totalAllocated = 0
        var totalInUse = 0
        var hits = 0
        var misses = 0
    }
    private struct Request {
        let capacity: Int
        let byteCount: Int
        let alignment: Int
    }

    internal init(configuration: Configuration = Configuration(),
                  allocationBoundary: any PoolAllocationBoundary = SystemPoolAllocationBoundary()) {
        guard configuration.maxBuffersPerSize >= 0, configuration.maxTotalMemory >= 0 else {
            fatalError("MemoryPool requires nonnegative cache limits")
        }
        self.configuration = configuration
        self.allocationBoundary = allocationBoundary
        setupCleanupTimer()
    }

    private func request<T>(type: T.Type, count: Int, alignment: Int) -> Request? {
        guard PoolRequest.validAlignment(alignment, natural: MemoryLayout<T>.alignment),
              let capacity = PoolRequest.capacity(count, powerOfTwo: true),
              let bytes = PoolRequest.byteCount(capacity: capacity, stride: MemoryLayout<T>.stride) else { return nil }
        return Request(capacity: capacity, byteCount: bytes, alignment: alignment)
    }

    /// Invalid, unrepresentable or failed requests return nil without a live handle.
    /// Zero count has zero logical elements and a minimum one-element allocation.
    public func acquire<T>(type: T.Type, count: Int, alignment: Int = 16) -> BufferHandle<T>? {
        guard let request = request(type: type, count: count, alignment: alignment) else { return nil }
        return acquire(type: type, count: count, request: request)
    }

    private func acquire<T>(type: T.Type, count: Int, request: Request) -> BufferHandle<T>? {
        let typeID = ObjectIdentifier(type)
        let prepared: (cached: PoolAllocation?, mayAllocate: Bool) = stateLock.withLock {
            guard stats.totalInUse < Int.max else { return (nil, false) }
            if let index = pools[typeID]?[request.capacity]?.firstIndex(where: { $0.allocation.alignment >= request.alignment }),
               let entry = pools[typeID]?[request.capacity]?.remove(at: index) {
                cachedByteCount -= entry.allocation.byteCount
                stats.totalInUse += 1
                PoolRequest.increment(&stats.hits)
                return (entry.allocation, false)
            }
            guard !stats.totalAllocated.addingReportingOverflow(request.byteCount).overflow else { return (nil, false) }
            PoolRequest.increment(&stats.misses)
            return (nil, true)
        }
        if let cached = prepared.cached { return BufferHandle(allocation: cached, count: count, pool: self) }
        guard prepared.mayAllocate else { return nil }
        // Allocation runs outside the lock. Recheck the accounting commit because
        // another acquisition may have advanced it while the allocator ran.
        guard let allocation = PoolAllocation(capacity: request.capacity, byteCount: request.byteCount,
            alignment: request.alignment, boundary: allocationBoundary) else { return nil }
        let committed = stateLock.withLock {
            let (total, overflow) = stats.totalAllocated.addingReportingOverflow(request.byteCount)
            guard !overflow, stats.totalInUse < Int.max else { return false }
            stats.totalAllocated = total
            stats.totalInUse += 1
            return true
        }
        guard committed else { return nil }
        _ = allocation.pointer.bindMemory(to: T.self, capacity: request.capacity)
        return BufferHandle(allocation: allocation, count: count, pool: self)
    }

    /// The caller must deinitialize initialized elements on every exit. Allocation
    /// failure tries aligned direct storage, then a best-effort Swift raw allocation
    /// with the same alignment; real exhaustion may terminate in the final stage.
    internal func withBuffer<T, R>(type: T.Type, count: Int, alignment: Int = 16,
        _ body: (UnsafeMutableBufferPointer<T>) throws -> R
    ) rethrows -> R {
        guard let request = request(type: type, count: count, alignment: alignment) else {
            fatalError("MemoryPool.withBuffer requires a nonnegative representable count and valid alignment")
        }
        if let handle = acquire(type: type, count: count, request: request) {
            return try withExtendedLifetime(handle) { try body(handle.buffer) }
        }
        let allocation = PoolAllocation(capacity: request.capacity, byteCount: request.byteCount,
            alignment: alignment, stage: .direct, boundary: allocationBoundary)
            ?? PoolAllocation(capacity: request.capacity, byteCount: request.byteCount,
                alignment: alignment, family: .swiftRaw, stage: .finalFallback, boundary: allocationBoundary)
        guard let allocation else { fatalError("MemoryPool.withBuffer final allocation failed") }
        let pointer = allocation.pointer.bindMemory(to: T.self, capacity: request.capacity)
        return try withExtendedLifetime(allocation) {
            try body(UnsafeMutableBufferPointer(start: pointer, count: count))
        }
    }

    internal var statistics: PoolStatistics {
        stateLock.withLock {
            let requests = Double(stats.hits) + Double(stats.misses)
            var counts: [String: Int] = [:]
            for (type, buckets) in pools {
                counts[String(describing: type)] = buckets.values.reduce(0) { $0 + $1.count }
            }
            return PoolStatistics(totalAllocated: stats.totalAllocated, totalInUse: stats.totalInUse,
                cachedBytes: cachedByteCount, hitRate: requests > 0 ? Double(stats.hits) / requests : 0,
                bufferCountByType: counts)
        }
    }

    /// Independent bucket scan for invariant tests; the return path never scans.
    internal var independentlyCountedCachedBytes: Int {
        stateLock.withLock {
            pools.values.reduce(0) { total, buckets in
                total + buckets.values.reduce(0) { subtotal, entries in
                    subtotal + entries.reduce(0) { $0 + $1.allocation.byteCount }
                }
            }
        }
    }
    internal var automaticCleanupEnabled: Bool { cleanupTimer != nil }

    public func quiesce() { stateLock.withLock {} }

    public func cleanup() {
        let cutoff = Date().addingTimeInterval(-configuration.cleanupInterval)
        stateLock.withLock {
            for (type, var buckets) in pools {
                for (size, var entries) in buckets {
                    entries.removeAll { entry in
                        guard entry.lastAccessed < cutoff else { return false }
                        cachedByteCount -= entry.allocation.byteCount
                        stats.totalAllocated -= entry.allocation.byteCount
                        return true
                    }
                    if entries.isEmpty { buckets.removeValue(forKey: size) }
                    else { buckets[size] = entries }
                }
                if buckets.isEmpty { pools.removeValue(forKey: type) }
                else { pools[type] = buckets }
            }
        }
    }

    private func returnBuffer<T>(_ allocation: PoolAllocation, type: T.Type) {
        stateLock.withLock {
            let typeID = ObjectIdentifier(type)
            if (pools[typeID]?[allocation.capacity]?.count ?? 0) < configuration.maxBuffersPerSize,
               allocation.byteCount <= configuration.maxTotalMemory - cachedByteCount {
                pools[typeID, default: [:]][allocation.capacity, default: []].append(
                    BufferEntry(allocation: allocation, lastAccessed: Date()))
                cachedByteCount += allocation.byteCount
            } else { stats.totalAllocated -= allocation.byteCount }
            stats.totalInUse -= 1
        }
    }

    private func setupCleanupTimer() {
        let seconds = configuration.cleanupInterval
        // Strict bound avoids Double(Int.max) rounding upward to an invalid Int.
        guard seconds.isFinite, seconds > 0, seconds < Double(Int.max) / 1_000_000_000 else { return }
        let nanoseconds = max(1, UInt64(seconds * 1_000_000_000))
        let now = DispatchTime.now().uptimeNanoseconds
        guard nanoseconds < UInt64(Int.max), now <= UInt64.max - nanoseconds else { return }
        let timer = DispatchSource.makeTimerSource(queue: .global(qos: .utility))
        timer.schedule(deadline: DispatchTime(uptimeNanoseconds: now + nanoseconds),
            repeating: .nanoseconds(Int(nanoseconds)))
        timer.setEventHandler { [weak self] in self?.cleanup() }
        timer.resume()
        cleanupTimer = timer
    }

    deinit { cleanupTimer?.cancel() }
}
