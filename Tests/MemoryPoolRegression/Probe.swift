import Foundation

@main
struct MemoryPoolRegressionProbe {
    private static func require(_ condition: Bool, _ message: String) {
        if !condition {
            FileHandle.standardError.write(Data((message + "\n").utf8))
            exit(1)
        }
    }

    private static func configuration() -> MemoryPool.Configuration {
        var configuration = MemoryPool.Configuration()
        configuration.cleanupInterval = .infinity
        return configuration
    }

    private static func useBuffer(_ pool: MemoryPool) {
        let handle = pool.acquire(type: Float.self, count: 37, alignment: 64)!
        handle.pointer.initialize(repeating: 0, count: handle.count)
        handle.pointer.deinitialize(count: handle.count)
    }

    private static func saturateExecutor() async {
        let ledger = ProbeAllocationBoundary()
        let pool = MemoryPool(configuration: configuration(), allocationBoundary: ledger)
        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<256 {
                group.addTask {
                    for _ in 0..<1000 {
                        useBuffer(pool)
                        pool.quiesce()
                    }
                }
            }
        }
        require(pool.statistics.totalInUse == 0, "borrowed buffers remain after task completion")
        require(pool.statistics.cachedBytes == pool.independentlyCountedCachedBytes, "concurrent cache accounting diverged")
    }

    private static func releaseAfterPoolDeallocation() {
        let ledger = ProbeAllocationBoundary()
        var pool: MemoryPool? = MemoryPool(configuration: configuration(), allocationBoundary: ledger)
        weak var retiredPool: MemoryPool?
        retiredPool = pool
        var handles: [MemoryPool.BufferHandle<Float>] = []
        for _ in 0..<8 {
            handles.append(pool!.acquire(type: Float.self, count: 64, alignment: 64)!)
        }
        pool!.quiesce()
        pool = nil
        require(retiredPool == nil, "fixture must destroy pool before releasing handles")
        require(ledger.outstandingAllocations == 8, "handles must retain their eight allocations")
        handles.removeAll()
        require(ledger.outstandingAllocations == 0, "late handles leaked their allocations")
    }

    private static func retainedByteLimit<T>(type: T.Type, count: Int, oversizedCount: Int) {
        var configuration = configuration()
        configuration.maxTotalMemory = 128
        let ledger = ProbeAllocationBoundary()
        let pool = MemoryPool(configuration: configuration, allocationBoundary: ledger)
        var first = pool.acquire(type: type, count: count)
        var second = pool.acquire(type: type, count: count)
        withExtendedLifetime((first, second)) {
            require(pool.statistics.totalAllocated == 256, "fixture must hold two 128-byte allocations")
        }
        first = nil
        second = nil
        require(pool.statistics.totalAllocated == 128, "retained buffers exceeded the 128-byte budget")
        let oversizedLedger = ProbeAllocationBoundary()
        let oversizedPool = MemoryPool(configuration: configuration, allocationBoundary: oversizedLedger)
        oversizedPool.withBuffer(type: type, count: oversizedCount) { _ in }
        require(oversizedPool.statistics.totalAllocated == 0, "oversized buffer was retained")
        require(oversizedLedger.outstandingAllocations == 0, "oversized allocation was not freed")
        require(pool.statistics.cachedBytes == pool.independentlyCountedCachedBytes, "maintained cache bytes diverged")
    }

    private static func cleanupByteAccounting() {
        var configuration = configuration()
        configuration.cleanupInterval = -1
        let ledger = ProbeAllocationBoundary()
        let pool = MemoryPool(configuration: configuration, allocationBoundary: ledger)
        pool.withBuffer(type: Float.self, count: 17) { _ in }
        pool.withBuffer(type: Double.self, count: 9) { _ in }
        pool.withBuffer(type: UInt8.self, count: 65) { _ in }
        require(pool.statistics.totalAllocated == 384, "fixture must retain three 128-byte allocations")
        pool.cleanup()
        require(pool.statistics.totalAllocated == 0, "cleanup did not subtract allocation bytes")
        require(ledger.outstandingAllocations == 0, "cleanup did not free all allocations")
    }

    static func main() async {
        switch CommandLine.arguments.dropFirst().first {
        case "saturation":
            await saturateExecutor()
        case "lifetime":
            releaseAfterPoolDeallocation()
        case "retention":
            retainedByteLimit(type: Float.self, count: 17, oversizedCount: 33)
            retainedByteLimit(type: Double.self, count: 9, oversizedCount: 17)
            retainedByteLimit(type: UInt8.self, count: 65, oversizedCount: 129)
        case "cleanup":
            cleanupByteAccounting()
        default:
            require(false, "expected saturation, lifetime, retention, or cleanup mode")
        }
        print("MemoryPool regression passed:", CommandLine.arguments[1])
    }
}
