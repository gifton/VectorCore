import Foundation

/// Borrowed metadata only; retain the pool and synchronize pointee access.
internal struct SendableBuffer: Sendable {
    fileprivate let id: UUID
    fileprivate let address: UInt
    fileprivate let count: Int
    var buffer: UnsafeMutableBufferPointer<Float> {
        UnsafeMutableBufferPointer(start: UnsafeMutablePointer<Float>(bitPattern: address)!, count: count)
    }
}

/// Pool-owned uninitialized Float storage. Clear affects cached allocations only.
internal actor BufferPool {
    private var storage: BorrowedPoolStorage
    private var stats = Statistics()

    struct Statistics: Sendable {
        var acquisitions = 0
        var releases = 0
        var allocations = 0
        var reuses = 0
        var evictions = 0
        var reuseRate: Double { acquisitions > 0 ? Double(reuses) / Double(acquisitions) : 0 }
    }

    init(maxBuffersPerSize: Int = 10, memoryLimitMB: Int = 256,
         allocationBoundary: any PoolAllocationBoundary = SystemPoolAllocationBoundary()) {
        let (bytes, overflow) = memoryLimitMB.multipliedReportingOverflow(by: 1024 * 1024)
        guard maxBuffersPerSize >= 0, memoryLimitMB >= 0, !overflow else {
            fatalError("BufferPool requires nonnegative, representable cache limits")
        }
        storage = BorrowedPoolStorage(maxCachedBytes: bytes, maxBuffersPerSize: maxBuffersPerSize,
            maxBufferBytes: Int.max, boundary: allocationBoundary)
    }

    deinit { storage.leakActiveLeases() }

    var cachedBytes: Int { storage.cachedBytes }
    var activeBytes: Int { storage.activeBytes }

    func acquire(count: Int) -> SendableBuffer {
        guard let capacity = PoolRequest.capacity(count, powerOfTwo: false),
              let bytes = PoolRequest.byteCount(capacity: capacity, stride: MemoryLayout<Float>.stride) else {
            fatalError("BufferPool.acquire requires a nonnegative, representable count")
        }
        guard let (lease, reused) = storage.acquire(count: count, capacity: capacity, byteCount: bytes, alignment: 16) else {
            fatalError("BufferPool.acquire allocation or active byte accounting failed")
        }
        _ = lease.allocation.pointer.bindMemory(to: Float.self, capacity: capacity)
        PoolRequest.increment(&stats.acquisitions)
        if reused { PoolRequest.increment(&stats.reuses) } else { PoolRequest.increment(&stats.allocations) }
        return SendableBuffer(id: lease.id, address: lease.allocation.address, count: count)
    }

    func release(_ handle: SendableBuffer) {
        let previous = storage.cachedBytes
        guard storage.release(id: handle.id, count: handle.count, address: handle.address) else { return }
        PoolRequest.increment(&stats.releases)
        if storage.cachedBytes == previous { PoolRequest.increment(&stats.evictions) }
    }

    func clear() { storage.clear() }
    func cleanup() { storage.clear() }
    func getStatistics() -> Statistics { stats }
}

/// Pool-owned uninitialized storage. A handle borrows memory: release it before the
/// pool is destroyed. A lease still active at destruction is leaked, not freed, so an
/// escaped handle never points at freed memory
/// (`BufferPoolOwnershipTests.internalAndTypedDestructionOwnCachedAndLeakActive`).
/// Initialize/deinitialize exactly the elements used before release; sending
/// metadata alone does not make pointee access safe.
public actor TypedBufferPool<Element> {
    public struct Handle: Sendable {
        fileprivate let id: UUID
        fileprivate let address: UInt
        public let count: Int
        public var pointer: UnsafeMutablePointer<Element> { UnsafeMutablePointer(bitPattern: address)! }
        public var buffer: UnsafeMutableBufferPointer<Element> {
            UnsafeMutableBufferPointer(start: pointer, count: count)
        }
    }

    private var storage: BorrowedPoolStorage

    public init(maxCachedBytes: Int, maxBuffersPerSize: Int = 10) {
        guard maxCachedBytes >= 0, maxBuffersPerSize >= 0 else {
            fatalError("TypedBufferPool requires nonnegative cache limits")
        }
        storage = BorrowedPoolStorage(maxCachedBytes: maxCachedBytes, maxBuffersPerSize: maxBuffersPerSize,
            maxBufferBytes: Int.max, boundary: SystemPoolAllocationBoundary())
    }

    internal init(maxCachedBytes: Int, maxBuffersPerSize: Int = 10, allocationBoundary: any PoolAllocationBoundary) {
        guard maxCachedBytes >= 0, maxBuffersPerSize >= 0 else {
            fatalError("TypedBufferPool requires nonnegative cache limits")
        }
        storage = BorrowedPoolStorage(maxCachedBytes: maxCachedBytes, maxBuffersPerSize: maxBuffersPerSize,
            maxBufferBytes: Int.max, boundary: allocationBoundary)
    }

    deinit { storage.leakActiveLeases() }

    public var cachedBytes: Int { storage.cachedBytes }
    public var activeBytes: Int { storage.activeBytes }

    public func acquire(count: Int) -> Handle {
        guard let capacity = PoolRequest.capacity(count, powerOfTwo: false),
              let bytes = PoolRequest.byteCount(capacity: capacity, stride: MemoryLayout<Element>.stride) else {
            fatalError("TypedBufferPool.acquire requires a nonnegative, representable count")
        }
        let alignment = max(16, MemoryLayout<Element>.alignment)
        guard let (lease, _) = storage.acquire(count: count, capacity: capacity, byteCount: bytes, alignment: alignment) else {
            fatalError("TypedBufferPool.acquire allocation or active byte accounting failed")
        }
        _ = lease.allocation.pointer.bindMemory(to: Element.self, capacity: capacity)
        return Handle(id: lease.id, address: lease.allocation.address, count: count)
    }

    public func release(_ handle: Handle) {
        _ = storage.release(id: handle.id, count: handle.count, address: handle.address)
    }

    public func clear() { storage.clear() }

    /// The synchronous body runs on this actor. It must not escape its pointer and
    /// must deinitialize all initialized elements before returning or throwing.
    public func withBuffer<R: Sendable>(count: Int,
        _ body: @Sendable (UnsafeMutableBufferPointer<Element>) throws -> R
    ) rethrows -> R {
        let handle = acquire(count: count)
        defer { release(handle) }
        return try body(handle.buffer)
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
