import Foundation

// Instrument only the deallocator dependency; SwiftBufferPool and BufferProvider
// are compiled directly from the reviewed repository, unchanged.
enum AlignedMemory {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var deallocations = 0
    static var frees: Int { lock.withLock { deallocations } }
    static func deallocate(_ pointer: UnsafeMutableRawPointer) {
        lock.withLock { deallocations += 1 }
        free(pointer)
    }
}

@main struct Probe {
    static func main() async throws {
        let pool = SwiftBufferPool(configuration: .init(maxPoolSize: 1024, maxBufferSize: 1024))
        var first: [BufferHandle] = []
        for _ in 0..<8 { first.append(try await pool.acquire(size: 1024)) }
        for handle in first { await pool.release(handle) }
        let before = await pool.statistics()
        var second: [BufferHandle] = []
        for _ in 0..<8 { second.append(try await pool.acquire(size: 1024)) }
        let after = await pool.statistics()
        print("cap=1024 returned=8192 reused=\(after.reusedBuffers - before.reusedBuffers) frees=\(AlignedMemory.frees)")
        for handle in second { await pool.release(handle) }
        await pool.clear()

        let freesBefore = AlignedMemory.frees
        var localPool: SwiftBufferPool? = SwiftBufferPool()
        weak var retiredPool = localPool
        let handle = try await localPool!.acquire(size: 1024)
        await localPool!.release(handle)
        localPool = nil
        print("poolDestroyed=\(retiredPool == nil) cachedBuffersFreed=\(AlignedMemory.frees - freesBefore)")
        // Reclaim the leaked allocation after proving pool destruction freed none.
        free(handle.pointer)
    }
}
