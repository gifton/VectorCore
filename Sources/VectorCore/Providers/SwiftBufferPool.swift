import Foundation

/// Actor-isolated pool of uninitialized raw buffers in power-of-two buckets.
/// Handles borrow storage: retain this provider until release and do not access a
/// pointer after release. A lease still active when the provider is destroyed is
/// leaked, not freed (`BufferPoolOwnershipTests.swiftPoolDestructionLeaksActiveLease`).
/// Cache admission is bounded by physical payload bytes.
public actor SwiftBufferPool: BufferProvider {
    private let configuration: BufferConfiguration
    private var storage: BorrowedPoolStorage
    private var allocations = 0
    private var reuses = 0
    private var peak = 0

    public init(configuration: BufferConfiguration = .default) {
        self.configuration = configuration
        self.storage = BorrowedPoolStorage(maxCachedBytes: configuration.maxPoolSize,
            maxBuffersPerSize: Int.max, maxBufferBytes: configuration.maxBufferSize,
            boundary: SystemPoolAllocationBoundary())
    }

    internal init(configuration: BufferConfiguration = .default, allocationBoundary: any PoolAllocationBoundary) {
        self.configuration = configuration
        self.storage = BorrowedPoolStorage(maxCachedBytes: configuration.maxPoolSize,
            maxBuffersPerSize: Int.max, maxBufferBytes: configuration.maxBufferSize,
            boundary: allocationBoundary)
    }

    deinit { storage.leakActiveLeases() }

    public nonisolated var alignment: Int { configuration.alignment }
    /// Physical payload bytes currently retained for reuse, excluding active leases.
    public var cachedBytes: Int { storage.cachedBytes }

    public func acquire(size: Int) async throws -> BufferHandle {
        guard let capacity = PoolRequest.capacity(size, powerOfTwo: true) else {
            throw BufferError.invalidRequest(size: size)
        }
        guard let (lease, reused) = storage.acquire(count: size, capacity: capacity,
            byteCount: capacity, alignment: alignment) else {
            throw BufferError.allocationFailed(size: size)
        }
        if configuration.trackStatistics {
            if reused { PoolRequest.increment(&reuses) } else { PoolRequest.increment(&allocations) }
            peak = max(peak, storage.activeBytes)
        }
        return BufferHandle(id: lease.id, size: size, pointer: lease.allocation.pointer)
    }

    /// Invalid, stale, duplicate or foreign metadata does not consume a live lease.
    public func release(_ handle: BufferHandle) async {
        _ = storage.release(id: handle.id, count: handle.size, address: UInt(bitPattern: handle.pointer))
    }

    public func statistics() async -> BufferStatistics {
        BufferStatistics(totalAllocations: allocations, reusedBuffers: reuses,
            currentUsageBytes: storage.activeBytes, peakUsageBytes: peak)
    }

    /// Releases cached storage; active borrowed leases remain valid.
    public func clear() async { storage.clear() }
}

public extension SwiftBufferPool {
    static let shared = SwiftBufferPool()
}
