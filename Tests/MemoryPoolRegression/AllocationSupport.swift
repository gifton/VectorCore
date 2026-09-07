// Allocation boundary for the standalone MemoryPool regression executable.
// Real aligned allocations are counted to detect a missing deallocation call.
import Foundation

private final class AllocationLedger: @unchecked Sendable {
    private let lock = NSLock()
    private var addresses: Set<UInt> = []

    func insert(_ address: UInt) {
        lock.withLock { precondition(addresses.insert(address).inserted) }
    }

    func remove(_ address: UInt) {
        lock.withLock { precondition(addresses.remove(address) != nil, "duplicate or foreign free") }
    }

    var count: Int { lock.withLock { addresses.count } }
}

enum ProbeAllocationError: Error { case allocationFailed }

// Matches the allocator API consumed by the actual MemoryPool.swift source.
// This fixture owns allocation accounting; production code has no test hooks.
enum AlignedMemory {
    private static let ledger = AllocationLedger()

    static var outstandingAllocations: Int { ledger.count }

    static func allocateAligned<T>(type: T.Type, count: Int, alignment: Int) throws -> UnsafeMutablePointer<T> {
        var allocation: UnsafeMutableRawPointer?
        guard posix_memalign(&allocation, alignment, count * MemoryLayout<T>.stride) == 0,
              let allocation else { throw ProbeAllocationError.allocationFailed }
        ledger.insert(UInt(bitPattern: allocation))
        return allocation.assumingMemoryBound(to: T.self)
    }

    static func deallocate<T>(_ pointer: UnsafeMutablePointer<T>) {
        deallocate(UnsafeMutableRawPointer(pointer))
    }

    static func deallocate(_ pointer: UnsafeMutableRawPointer) {
        ledger.remove(UInt(bitPattern: pointer))
        free(pointer)
    }
}
