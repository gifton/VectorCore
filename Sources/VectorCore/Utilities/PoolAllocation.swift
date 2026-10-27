import Foundation

internal enum PoolAllocationFamily: Sendable { case aligned, swiftRaw }
internal enum PoolAllocationStage: Sendable { case pooled, direct, finalFallback }

/// The boundary is per pool instance. Implementations must synchronize any mutable
/// diagnostic state. Returning nil injects or reports an allocation failure.
internal protocol PoolAllocationBoundary: Sendable {
    func allocate(byteCount: Int, alignment: Int, family: PoolAllocationFamily, stage: PoolAllocationStage) -> UnsafeMutableRawPointer?
    func deallocate(address: UInt, family: PoolAllocationFamily)
}

internal struct SystemPoolAllocationBoundary: PoolAllocationBoundary {
    func allocate(byteCount: Int, alignment: Int, family: PoolAllocationFamily, stage: PoolAllocationStage) -> UnsafeMutableRawPointer? {
        switch family {
        case .aligned:
            var pointer: UnsafeMutableRawPointer?
            guard posix_memalign(&pointer, alignment, byteCount) == 0 else { return nil }
            return pointer
        case .swiftRaw:
            return UnsafeMutableRawPointer.allocate(byteCount: byteCount, alignment: alignment)
        }
    }

    func deallocate(address: UInt, family: PoolAllocationFamily) {
        let pointer = UnsafeMutableRawPointer(bitPattern: address)!
        switch family {
        case .aligned: free(pointer)
        case .swiftRaw: pointer.deallocate()
        }
    }
}

/// Owns raw allocation lifetime, not initialized elements. Immutable address
/// metadata is Sendable; this does not synchronize or make pointee access Sendable.
/// The final owner releases through the same boundary and allocation family.
internal final class PoolAllocation: Sendable {
    let address: UInt
    let capacity: Int
    let byteCount: Int
    let alignment: Int
    private let family: PoolAllocationFamily
    private let boundary: any PoolAllocationBoundary
    var pointer: UnsafeMutableRawPointer { UnsafeMutableRawPointer(bitPattern: address)! }

    init?(capacity: Int, byteCount: Int, alignment: Int, family: PoolAllocationFamily = .aligned,
          stage: PoolAllocationStage = .pooled, boundary: any PoolAllocationBoundary) {
        guard let pointer = boundary.allocate(byteCount: byteCount, alignment: alignment, family: family, stage: stage) else { return nil }
        self.address = UInt(bitPattern: pointer)
        self.capacity = capacity
        self.byteCount = byteCount
        self.alignment = alignment
        self.family = family
        self.boundary = boundary
    }

    deinit { boundary.deallocate(address: address, family: family) }
}

internal enum PoolRequest {
    static func validAlignment(_ alignment: Int, natural: Int = 1) -> Bool {
        alignment >= max(16, natural) && (alignment & (alignment - 1)) == 0
    }

    static func capacity(_ count: Int, powerOfTwo: Bool) -> Int? {
        guard count >= 0 else { return nil }
        let minimum = max(1, count)
        guard powerOfTwo else { return minimum }
        var result = 1
        while result < minimum {
            let (next, overflow) = result.multipliedReportingOverflow(by: 2)
            guard !overflow else { return nil }
            result = next
        }
        return result
    }

    static func byteCount(capacity: Int, stride: Int) -> Int? {
        let (bytes, overflow) = capacity.multipliedReportingOverflow(by: stride)
        return overflow ? nil : bytes
    }

    /// Diagnostic counters saturate; operational byte accounting never wraps.
    static func increment(_ value: inout Int) { if value < Int.max { value += 1 } }
}

internal struct PoolLease: Sendable {
    let id: UUID
    let count: Int
    let allocation: PoolAllocation
}

/// Actor-isolated value state shared by the three borrowed-lease pools. References in the active and cached
/// dictionaries own all allocations; handles returned to callers contain no owner references. The owning actor's
/// `deinit` calls `leakActiveLeases()` first, so on destruction cached allocations are freed and active leases are
/// leaked rather than freed under a live handle
/// (`BufferPoolOwnershipTests.internalAndTypedDestructionOwnCachedAndLeakActive`).
internal struct BorrowedPoolStorage: Sendable {
    private var cached: [Int: [PoolAllocation]] = [:]
    private var active: [UUID: PoolLease] = [:]
    private(set) var activeBytes = 0
    private(set) var cachedBytes = 0
    let maxCachedBytes: Int
    let maxBuffersPerSize: Int
    let maxBufferBytes: Int
    let boundary: any PoolAllocationBoundary

    init(maxCachedBytes: Int, maxBuffersPerSize: Int, maxBufferBytes: Int,
         boundary: any PoolAllocationBoundary) {
        self.maxCachedBytes = maxCachedBytes
        self.maxBuffersPerSize = maxBuffersPerSize
        self.maxBufferBytes = maxBufferBytes
        self.boundary = boundary
    }

    mutating func acquire(count: Int, capacity: Int, byteCount: Int, alignment: Int) -> (PoolLease, reused: Bool)? {
        let (nextActive, overflow) = activeBytes.addingReportingOverflow(byteCount)
        guard !overflow else { return nil }
        let allocation: PoolAllocation
        let reused: Bool
        if let existing = cached[capacity]?.popLast() {
            allocation = existing
            cachedBytes -= existing.byteCount
            reused = true
        } else {
            guard let fresh = PoolAllocation(capacity: capacity, byteCount: byteCount, alignment: alignment, boundary: boundary) else { return nil }
            allocation = fresh
            reused = false
        }
        let lease = PoolLease(id: UUID(), count: count, allocation: allocation)
        active[lease.id] = lease
        activeBytes = nextActive
        return (lease, reused)
    }

    /// Compare every caller-controlled field before consuming the live lease.
    mutating func release(id: UUID, count: Int, address: UInt) -> Bool {
        guard let lease = active[id], lease.count == count, lease.allocation.address == address else { return false }
        active.removeValue(forKey: id)
        let allocation = lease.allocation
        activeBytes -= allocation.byteCount
        if allocation.byteCount <= maxBufferBytes,
           allocation.byteCount <= maxCachedBytes - cachedBytes,
           (cached[allocation.capacity]?.count ?? 0) < maxBuffersPerSize {
            cached[allocation.capacity, default: []].append(allocation)
            cachedBytes += allocation.byteCount
        }
        return true
    }

    mutating func clear() {
        cached.removeAll()
        cachedBytes = 0
    }

    /// Retains every active lease's allocation once more, for good. Called from the owning actor's `deinit`, it turns
    /// a lease that outlived its pool into a leak instead of a use-after-free through the escaped handle (0.3.3 leaked
    /// too). Cached allocations are untouched and are freed with the storage.
    func leakActiveLeases() {
        for lease in active.values { _ = Unmanaged.passRetained(lease.allocation) }
    }
}
