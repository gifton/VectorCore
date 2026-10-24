import Foundation

/// Instance-owned ledger around the same real allocation families as production.
final class ProbeAllocationBoundary: PoolAllocationBoundary, @unchecked Sendable {
    private let lock = NSLock()
    private var allocations: [UInt: PoolAllocationFamily] = [:]
    deinit { precondition(allocations.isEmpty, "pool teardown leaked allocations") }
    var outstandingAllocations: Int { lock.withLock { allocations.count } }

    func allocate(byteCount: Int, alignment: Int, family: PoolAllocationFamily, stage: PoolAllocationStage) -> UnsafeMutableRawPointer? {
        guard let pointer = SystemPoolAllocationBoundary().allocate(byteCount: byteCount, alignment: alignment, family: family, stage: stage) else { return nil }
        lock.withLock {
            precondition(allocations.updateValue(family, forKey: UInt(bitPattern: pointer)) == nil)
        }
        return pointer
    }

    func deallocate(address: UInt, family: PoolAllocationFamily) {
        lock.withLock { precondition(allocations.removeValue(forKey: address) == family, "duplicate, foreign or mismatched free") }
        SystemPoolAllocationBoundary().deallocate(address: address, family: family)
    }
}
