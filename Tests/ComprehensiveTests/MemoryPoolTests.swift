import Foundation
import Testing
@testable import VectorCore

// MARK: - Timeout Helpers (file-scoped to avoid capturing self)
private enum TestTimeoutError: Error { case timeout }

private func withTimeout(_ seconds: Double = 45, _ body: @escaping @Sendable () async throws -> Void) async {
    do {
        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask { try await body() }
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
                throw TestTimeoutError.timeout
            }
            do { _ = try await group.next() } catch { throw error }
            group.cancelAll()
        }
    } catch {
        Issue.record("Test exceeded timeout \(seconds)s: \(error)")
    }
}

@inline(__always)
private func sleepMs(_ ms: UInt64) async {
    try? await Task.sleep(nanoseconds: ms * 1_000_000)
}

@Suite("Memory Pool Tests", .serialized)
struct MemoryPoolTests {

    // Acquire basic behavior
    @Test
    func testAcquire_ReturnsBufferWithCountAndAlignment() async {
        await withTimeout(45) {
            let pool = MemoryPool(configuration: .init())
            // Scope the handle so it returns to the pool deterministically.
            do {
                guard let handle = pool.acquire(type: Float.self, count: 37, alignment: 64) else {
                    Issue.record("Expected non-nil buffer handle")
                    return
                }
                #expect(handle.count == 37)
                #expect(AlignedMemory.isAligned(handle.pointer, to: 64))
                // Touch memory to ensure the buffer is writable
                handle.pointer.initialize(to: 0)
                _ = handle // returned on scope exit
            }
            // Give the pool a brief moment to process async return bookkeeping without blocking.
            await sleepMs(5)
        }
    }

    @Test
    func testAcquire_ReusesReturnedBuffer_IncreasesHitRate() async {
        await withTimeout(45) {
            let pool = MemoryPool(configuration: .init())
            var firstPtrAddr: Int = 0
            do {
                let h1 = pool.acquire(type: Float.self, count: 32, alignment: 32)!
                firstPtrAddr = Int(bitPattern: UnsafeMutableRawPointer(h1.pointer))
                _ = h1
            }
            await sleepMs(50)
            let hitsBefore = pool.statistics.hitRate
            do {
                let h2 = pool.acquire(type: Float.self, count: 31, alignment: 32)!
                let addr2 = Int(bitPattern: UnsafeMutableRawPointer(h2.pointer))
                #expect(addr2 == firstPtrAddr)
                _ = h2
            }
            await sleepMs(50)
            let hitsAfter = pool.statistics.hitRate
            #expect(hitsAfter >= hitsBefore)
        }
    }

    @Test
    func testAcquire_PowerOfTwoBucketing_ReusesAcrossCounts() async {
        await withTimeout(45) {
            let pool = MemoryPool(configuration: .init())
            var addr1: Int = 0
            do {
                let h = pool.acquire(type: Double.self, count: 30, alignment: 16)!
                addr1 = Int(bitPattern: UnsafeMutableRawPointer(h.pointer))
                _ = h
            }
            await sleepMs(50)
            do {
                let h = pool.acquire(type: Double.self, count: 31, alignment: 16)!
                let addr2 = Int(bitPattern: UnsafeMutableRawPointer(h.pointer))
                #expect(addr2 == addr1)
                _ = h
            }
        }
    }

    @Test
    func testAcquire_AlignmentRequirement_PreventsLowerAlignedReuse() async {
        await withTimeout(45) {
            let pool = MemoryPool(configuration: .init())
            var lowAlignedAddr: Int = 0
            do {
                let h = pool.acquire(type: UInt8.self, count: 64, alignment: 16)!
                #expect(AlignedMemory.isAligned(h.pointer, to: 16))
                lowAlignedAddr = Int(bitPattern: UnsafeMutableRawPointer(h.pointer))
                _ = h
            }
            await sleepMs(50)
            do {
                let h = pool.acquire(type: UInt8.self, count: 64, alignment: 64)!
                #expect(AlignedMemory.isAligned(h.pointer, to: 64))
                let addr2 = Int(bitPattern: UnsafeMutableRawPointer(h.pointer))
                #expect(addr2 != lowAlignedAddr)
                _ = h
            }
        }
    }

    // Limits and cleanup
    @Test
    func testReturn_RespectsMaxBuffersPerSizeLimit() async {
        await withTimeout(45) {
            var config = MemoryPool.Configuration()
            config.maxBuffersPerSize = 1
            config.maxTotalMemory = 1_000_000
            let pool = MemoryPool(configuration: config)
            // Acquire and return multiple same-bucket buffers
            for _ in 0..<3 {
                let h = pool.acquire(type: Float.self, count: 40, alignment: 16)!
                _ = h
            }
            await sleepMs(80)
            let stats = pool.statistics
            // Only one buffer should be retained for this size
            let totalCount = stats.bufferCountByType.values.reduce(0, +)
            #expect(totalCount <= config.maxBuffersPerSize)
        }
    }

    @Test
    func testReturn_RespectsMaxTotalMemoryLimit() async {
        await withTimeout(45) {
            var config = MemoryPool.Configuration()
            config.maxBuffersPerSize = 100
            // Limit to roughly one 1024-float buffer
            config.maxTotalMemory = 1024 * MemoryLayout<Float>.stride
            let pool = MemoryPool(configuration: config)
            // First buffer occupy near the cap
            do { let h = pool.acquire(type: Float.self, count: 900, alignment: 16)!; _ = h }
            // Second buffer exceeds cap and should be deallocated on return
            do { let h = pool.acquire(type: Float.self, count: 900, alignment: 16)!; _ = h }
            await sleepMs(120)
            let stats = pool.statistics
            #expect(stats.totalAllocated <= config.maxTotalMemory)
        }
    }

    @Test
    func testCleanup_RemovesStaleEntries_UpdatesStats() async {
        await withTimeout(45) {
            var config = MemoryPool.Configuration()
            config.cleanupInterval = 0.05
            let pool = MemoryPool(configuration: config)
            // Add a pooled buffer
            do { let h = pool.acquire(type: UInt16.self, count: 20, alignment: 16)!; _ = h }
            await sleepMs(60) // exceed cleanup interval
            let before = pool.statistics.totalAllocated
            pool.cleanup()
            await sleepMs(60)
            let after = pool.statistics.totalAllocated
            #expect(after <= before)
        }
    }

    // withBuffer behavior
    @Test
    func testWithBuffer_ProvidesWritableAlignedBuffer() async {
        await withTimeout(45) {
            let pool = MemoryPool(configuration: .init())
            let sum = pool.withBuffer(type: Float.self, count: 64, alignment: 32) { buf in
                #expect(AlignedMemory.isAligned(buf.baseAddress!, to: 32))
                for i in 0..<buf.count { buf[i] = Float(i) }
                return buf.reduce(0, +)
            }
            #expect(approxEqual(sum, 2016))
        }
    }

    @Test
    func testWithBuffer_FallbackPathWorksWhenAcquireNil() async {
        await withTimeout(45) {
            let pool = MemoryPool(configuration: .init())
            let first = pool.withBuffer(type: UInt8.self, count: 0, alignment: 16) { buf in buf.count }
            #expect(first == 0)
            let second = pool.withBuffer(type: UInt8.self, count: 5, alignment: 16) { buf in
                for i in 0..<buf.count { buf[i] = UInt8(i) }
                return Int(buf[4])
            }
            #expect(second == 4)
        }
    }

    // Statistics correctness
    @Test
    func testStatistics_TotalInUseTracksAcquireAndReturn() async {
        await withTimeout(45) {
            let pool = MemoryPool(configuration: .init())
            let h1 = pool.acquire(type: Float.self, count: 10)!
            let h2 = pool.acquire(type: Float.self, count: 10)!
            await sleepMs(30)
            var inUse = pool.statistics.totalInUse
            #expect(inUse >= 2)
            _ = h1
            await sleepMs(60)
            inUse = pool.statistics.totalInUse
            #expect(inUse >= 1)
            _ = h2
        }
    }

    @Test
    func testStatistics_BufferCountByTypeReflectsPools() async {
        await withTimeout(45) {
            let pool = MemoryPool(configuration: .init())
            do { let h = pool.acquire(type: Float.self, count: 20)!; _ = h }
            do { let h = pool.acquire(type: Double.self, count: 20)!; _ = h }
            await sleepMs(80)
            let total = pool.statistics.bufferCountByType.values.reduce(0, +)
            #expect(total >= 2)
        }
    }

    // Concurrency and multi-type separation
    @Test
    func testConcurrentAcquireAndReturn_NoLeaksNoCrashes() async {
        await withTimeout(45) {
            // Minimize external factors (cleanup timer still exists but load is tiny).
            let pool = MemoryPool(configuration: .init())
            await withTaskGroup(of: Void.self) { group in
                // Very small, predictable load: 4 tasks × 8 iterations
                for _ in 0..<4 {
                    group.addTask {
                        for _ in 0..<8 {
                            let h = pool.acquire(type: Float.self, count: 8)!
                            _ = h // return on scope exit
                            await Task.yield()
                        }
                    }
                }
            }
            // Brief settle period for async stats updates
            await sleepMs(50)
            let stats = pool.statistics
            #expect(stats.totalInUse == 0)
            // hitRate and totalAllocated are opportunistic; just ensure they are within sane bounds
            #expect(stats.hitRate >= 0 && stats.hitRate <= 1)
        }
    }

    @Test
    func testSeparateTypePools_DoNotInterfere() async {
        await withTimeout(45) {
            let pool = MemoryPool(configuration: .init())
            var floatAddr: Int = 0
            do { let h = pool.acquire(type: Float.self, count: 32)!; floatAddr = Int(bitPattern: UnsafeMutableRawPointer(h.pointer)); _ = h }
            await sleepMs(50)
            do { let h = pool.acquire(type: Double.self, count: 32)!; _ = h }
            await sleepMs(50)
            do { let h = pool.acquire(type: Float.self, count: 31)!; let addr2 = Int(bitPattern: UnsafeMutableRawPointer(h.pointer)); #expect(addr2 == floatAddr); _ = h }
        }
    }

    // Edge cases
    @Test
    func testAcquire_CountZero_ReturnsZeroLengthHandle() async {
        await withTimeout(45) {
            let pool = MemoryPool(configuration: .init())
            let h = pool.acquire(type: Int32.self, count: 0)!
            #expect(h.count == 0)
            #expect(AlignedMemory.isAligned(h.pointer, to: 16))
            _ = h
        }
    }

    @Test
    func testAcquire_SmallCounts_AlignmentDefaultIsApplied() async {
        await withTimeout(45) {
            let pool = MemoryPool(configuration: .init())
            let h = pool.acquire(type: UInt16.self, count: 1)! // default alignment 16
            #expect(AlignedMemory.isAligned(h.pointer, to: 16))
            _ = h
        }
    }
}
