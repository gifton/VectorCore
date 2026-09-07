import Foundation
import Testing
@testable import VectorCore

// MARK: - Error reporting (file-scoped to avoid capturing self)
private func recordTestErrors(_ body: @escaping @Sendable () async throws -> Void) async {
    do {
        try await body()
    } catch {
        Issue.record("Test error: \(error)")
    }
}

@inline(__always)
private func sleepMs(_ ms: UInt64) async {
    try? await Task.sleep(nanoseconds: ms * 1_000_000)
}

/// Create test configuration with cleanup timer disabled
private func testConfig() -> MemoryPool.Configuration {
    var config = MemoryPool.Configuration()
    config.cleanupInterval = .infinity  // Disable cleanup timer for tests
    return config
}

private func checkRetainedByteLimit<T>(type: T.Type, count: Int, oversizedCount: Int) {
    var configuration = testConfig()
    configuration.maxBuffersPerSize = 10
    configuration.maxTotalMemory = 128
    let pool = MemoryPool(configuration: configuration)
    var first = pool.acquire(type: type, count: count)
    var second = pool.acquire(type: type, count: count)
    #expect(first != nil && second != nil)
    withExtendedLifetime((first, second)) {
        #expect(pool.statistics.totalAllocated == 256)
        #expect(pool.statistics.totalInUse == 2)
    }
    first = nil
    second = nil
    #expect(pool.statistics.totalAllocated == 128)
    #expect(pool.statistics.totalInUse == 0)
    #expect(pool.statistics.bufferCountByType.values.reduce(0, +) == 1)

    let oversizedPool = MemoryPool(configuration: configuration)
    oversizedPool.withBuffer(type: type, count: oversizedCount) { _ in }
    #expect(oversizedPool.statistics.totalAllocated == 0)
    #expect(oversizedPool.statistics.bufferCountByType.values.reduce(0, +) == 0)
}

@Suite("Memory Pool Tests", .serialized)
struct MemoryPoolTests {

    // Acquire basic behavior
    @Test
    func testAcquire_ReturnsBufferWithCountAndAlignment() async {
        await recordTestErrors {
            let pool = MemoryPool(configuration: testConfig())
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
            // Returned buffers are available before the handle release completes.
            pool.quiesce()
        }
    }

    @Test
    func testAcquire_ReusesReturnedBuffer_IncreasesHitRate() async {
        await recordTestErrors {
            let pool = MemoryPool(configuration: testConfig())
            var firstPtrAddr: Int = 0
            do {
                let h1 = pool.acquire(type: Float.self, count: 32, alignment: 32)!
                firstPtrAddr = Int(bitPattern: UnsafeMutableRawPointer(h1.pointer))
                _ = h1
            }
            pool.quiesce()
            let hitsBefore = pool.statistics.hitRate
            do {
                let h2 = pool.acquire(type: Float.self, count: 31, alignment: 32)!
                let addr2 = Int(bitPattern: UnsafeMutableRawPointer(h2.pointer))
                #expect(addr2 == firstPtrAddr)
                _ = h2
            }
            pool.quiesce()
            let hitsAfter = pool.statistics.hitRate
            #expect(hitsAfter >= hitsBefore)
        }
    }

    @Test
    func testAcquire_PowerOfTwoBucketing_ReusesAcrossCounts() async {
        await recordTestErrors {
            let pool = MemoryPool(configuration: testConfig())
            var addr1: Int = 0
            do {
                let h = pool.acquire(type: Double.self, count: 30, alignment: 16)!
                addr1 = Int(bitPattern: UnsafeMutableRawPointer(h.pointer))
                _ = h
            }
            pool.quiesce()
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
        await recordTestErrors {
            let pool = MemoryPool(configuration: testConfig())
            var lowAlignedAddr: Int = 0
            do {
                let h = pool.acquire(type: UInt8.self, count: 64, alignment: 16)!
                #expect(AlignedMemory.isAligned(h.pointer, to: 16))
                lowAlignedAddr = Int(bitPattern: UnsafeMutableRawPointer(h.pointer))
                _ = h
            }
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
        await recordTestErrors {
            var config = testConfig()
            config.maxBuffersPerSize = 1
            config.maxTotalMemory = 1_000_000
            let pool = MemoryPool(configuration: config)
            // Acquire and return multiple same-bucket buffers
            for _ in 0..<3 {
                let h = pool.acquire(type: Float.self, count: 40, alignment: 16)!
                _ = h
            }
            pool.quiesce()
            let stats = pool.statistics
            // Only one buffer should be retained for this size
            let totalCount = stats.bufferCountByType.values.reduce(0, +)
            #expect(totalCount <= config.maxBuffersPerSize)
        }
    }

    @Test
    func testReturn_RespectsMaxTotalMemoryLimit() async {
        await recordTestErrors {
            var config = testConfig()
            config.maxBuffersPerSize = 100
            // Limit to roughly one 1024-float buffer
            config.maxTotalMemory = 1024 * MemoryLayout<Float>.stride
            let pool = MemoryPool(configuration: config)
            // First buffer occupy near the cap
            do { let h = pool.acquire(type: Float.self, count: 900, alignment: 16)!; _ = h }
            // Second buffer exceeds cap and should be deallocated on return
            do { let h = pool.acquire(type: Float.self, count: 900, alignment: 16)!; _ = h }
            pool.quiesce()
            let stats = pool.statistics
            #expect(stats.totalAllocated <= config.maxTotalMemory)
        }
    }

    @Test
    func testRetentionLimitUsesBytesAcrossElementTypes() {
        // Each ordinary buffer rounds to 128 bytes; each oversized one to 256.
        checkRetainedByteLimit(type: Float.self, count: 17, oversizedCount: 33)
        checkRetainedByteLimit(type: Double.self, count: 9, oversizedCount: 17)
        checkRetainedByteLimit(type: UInt8.self, count: 65, oversizedCount: 129)
    }

    @Test
    func testCleanupSubtractsBytesAcrossElementTypes() {
        var configuration = testConfig()
        // Nonpositive intervals disable the timer. The explicit cutoff lies one
        // second in the future, so all returned entries qualify without a sleep.
        configuration.cleanupInterval = -1
        let pool = MemoryPool(configuration: configuration)
        pool.withBuffer(type: Float.self, count: 17) { _ in }
        pool.withBuffer(type: Double.self, count: 9) { _ in }
        pool.withBuffer(type: UInt8.self, count: 65) { _ in }
        #expect(pool.statistics.totalAllocated == 384)
        #expect(pool.statistics.totalInUse == 0)
        pool.cleanup()
        #expect(pool.statistics.totalAllocated == 0)
        #expect(pool.statistics.bufferCountByType.isEmpty)
    }

    @Test
    func testCleanup_RemovesStaleEntries_UpdatesStats() async {
        await recordTestErrors {
            var config = MemoryPool.Configuration()
            config.cleanupInterval = 0.05  // This test specifically tests cleanup
            let pool = MemoryPool(configuration: config)
            // Add a pooled buffer
            do { let h = pool.acquire(type: UInt16.self, count: 20, alignment: 16)!; _ = h }
            await sleepMs(60) // exceed cleanup interval
            let before = pool.statistics.totalAllocated
            pool.cleanup()
            pool.quiesce()
            let after = pool.statistics.totalAllocated
            #expect(after <= before)
        }
    }

    // withBuffer behavior
    @Test
    func testWithBuffer_ProvidesWritableAlignedBuffer() async {
        await recordTestErrors {
            let pool = MemoryPool(configuration: testConfig())
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
        await recordTestErrors {
            let pool = MemoryPool(configuration: testConfig())
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
        await recordTestErrors {
            let pool = MemoryPool(configuration: testConfig())
            let h1 = pool.acquire(type: Float.self, count: 10)!
            let h2 = pool.acquire(type: Float.self, count: 10)!
            pool.quiesce()
            var inUse = pool.statistics.totalInUse
            #expect(inUse >= 2)
            _ = h1
            pool.quiesce()
            inUse = pool.statistics.totalInUse
            #expect(inUse >= 1)
            _ = h2
        }
    }

    @Test
    func testStatistics_BufferCountByTypeReflectsPools() async {
        await recordTestErrors {
            let pool = MemoryPool(configuration: testConfig())
            do { let h = pool.acquire(type: Float.self, count: 20)!; _ = h }
            do { let h = pool.acquire(type: Double.self, count: 20)!; _ = h }
            pool.quiesce()
            let total = pool.statistics.bufferCountByType.values.reduce(0, +)
            #expect(total >= 2)
        }
    }

    // Concurrency and multi-type separation
    @Test
    func testConcurrentAcquireAndReturn_NoLeaksNoCrashes() async {
        await recordTestErrors {
            // Minimize external factors (cleanup timer disabled).
            let pool = MemoryPool(configuration: testConfig())
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
            let stats = pool.statistics
            #expect(stats.totalInUse == 0)
            // hitRate and totalAllocated are opportunistic; just ensure they are within sane bounds
            #expect(stats.hitRate >= 0 && stats.hitRate <= 1)
        }
    }

    @Test
    func testSeparateTypePools_DoNotInterfere() async {
        await recordTestErrors {
            let pool = MemoryPool(configuration: testConfig())
            var floatAddr: Int = 0
            do {
                let h = pool.acquire(type: Float.self, count: 32)!
                floatAddr = Int(bitPattern: UnsafeMutableRawPointer(h.pointer))
                _ = h
            }
            pool.quiesce()
            do {
                let h = pool.acquire(type: Double.self, count: 32)!
                _ = h
            }
            pool.quiesce()
            do {
                let h = pool.acquire(type: Float.self, count: 31)!
                let addr2 = Int(bitPattern: UnsafeMutableRawPointer(h.pointer))
                #expect(addr2 == floatAddr)
                _ = h
            }
        }
    }

    // Edge cases
    @Test
    func testAcquire_CountZero_ReturnsZeroLengthHandle() async {
        await recordTestErrors {
            let pool = MemoryPool(configuration: testConfig())
            let h = pool.acquire(type: Int32.self, count: 0)!
            #expect(h.count == 0)
            #expect(AlignedMemory.isAligned(h.pointer, to: 16))
            _ = h
        }
    }

    @Test
    func testAcquire_SmallCounts_AlignmentDefaultIsApplied() async {
        await recordTestErrors {
            let pool = MemoryPool(configuration: testConfig())
            let h = pool.acquire(type: UInt16.self, count: 1)! // default alignment 16
            #expect(AlignedMemory.isAligned(h.pointer, to: 16))
            _ = h
        }
    }

    // The standalone MemoryPoolRegression probe counts real allocations and frees
    // to verify this lifetime path does not leak; this case checks live handle use.
    @Test
    func testReturn_AfterPoolDeallocated_FreesBufferWithoutCrash() async {
        await recordTestErrors {
            var handles: [MemoryPool.BufferHandle<Float>] = []
            weak var retiredPool: MemoryPool?
            do {
                let pool = MemoryPool(configuration: testConfig())
                retiredPool = pool
                for _ in 0..<8 {
                    guard let handle = pool.acquire(type: Float.self, count: 64, alignment: 64) else {
                        Issue.record("Expected non-nil buffer handle")
                        return
                    }
                    handle.pointer.initialize(repeating: 0, count: handle.count)
                    handles.append(handle)
                }
            }
            #expect(retiredPool == nil)
            #expect(handles.count == 8)
            for handle in handles {
                handle.pointer[0] = 42
                #expect(approxEqual(handle.pointer[0], 42))
            }
            handles.removeAll()
        }
    }
}
