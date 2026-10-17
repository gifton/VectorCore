import Foundation
import Testing
@testable import VectorCore

@Suite("E04 MemoryPool boundaries")
struct MemoryPoolBoundaryTests {
    private func configuration(bytes: Int = 128) -> MemoryPool.Configuration {
        var configuration = MemoryPool.Configuration()
        configuration.cleanupInterval = 0
        configuration.maxTotalMemory = bytes
        return configuration
    }

    @Test func lateHandleOwnsAllocationAndAlignmentMetadataSurvivesReuse() {
        let ledger = E04AllocationLedger()
        var pool: MemoryPool? = MemoryPool(configuration: configuration(), allocationBoundary: ledger)
        var first = pool!.acquire(type: UInt8.self, count: 64, alignment: 64)
        let address = UInt(bitPattern: first!.pointer)
        first = nil
        var lower = pool!.acquire(type: UInt8.self, count: 64, alignment: 16)
        #expect(UInt(bitPattern: lower!.pointer) == address)
        lower = nil
        var late = pool!.acquire(type: UInt8.self, count: 64, alignment: 64)
        #expect(UInt(bitPattern: late!.pointer) == address)
        #expect(ledger.counts.allocated == 1)
        pool = nil
        #expect(ledger.bytes == 64)
        late!.pointer.initialize(to: 7)
        #expect(late!.pointer.pointee == 7)
        late!.pointer.deinitialize(count: 1)
        late = nil
        #expect(ledger.bytes == 0)
    }

    @Test func forcedFallbacksPreserveAlignmentBindingAndFreeFamily() throws {
        for failures: Set<PoolAllocationStage> in [[.pooled], [.pooled, .direct]] {
            let ledger = E04AllocationLedger()
            ledger.fail(failures)
            let pool = MemoryPool(configuration: configuration(), allocationBoundary: ledger)
            let result = pool.withBuffer(type: String.self, count: 3, alignment: 64) { buffer in
                #expect(UInt(bitPattern: buffer.baseAddress!) % 64 == 0)
                buffer.baseAddress!.initialize(repeating: "initialized", count: 3)
                defer { buffer.baseAddress!.deinitialize(count: 3) }
                return buffer[2]
            }
            #expect(result == "initialized")
            #expect(ledger.bytes == 0)
            #expect(ledger.counts.allocated == 1)
            #expect(ledger.counts.freed == 1)
            #expect(ledger.stages == (failures.count == 1 ? [.pooled, .direct] : [.pooled, .direct, .finalFallback]))
        }
    }
}

extension MemoryPoolBoundaryTests {
    @Test func invalidRequestsStopBeforeAllocatorAndRepresentableLimitsCanFail() {
        let ledger = E04AllocationLedger()
        let pool = MemoryPool(configuration: configuration(), allocationBoundary: ledger)
        for count in [Int.min, -1, Int.max, Int.max / 2 + 2] {
            #expect(pool.acquire(type: UInt8.self, count: count) == nil)
        }
        for alignment in [Int.min, -1, 0, 1, 3, 8, 15, 17, Int.max] {
            #expect(pool.acquire(type: UInt8.self, count: 1, alignment: alignment) == nil)
        }
        for count in [Int.max / 8, Int.max / 8 + 1, Int.max / 2 + 1] {
            #expect(pool.acquire(type: Double.self, count: count) == nil)
        }
        #expect(ledger.stages.isEmpty)
        ledger.fail([.pooled])
        #expect(pool.acquire(type: UInt8.self, count: Int.max / 2 + 1) == nil)
        #expect(pool.acquire(type: UInt8.self, count: 1, alignment: Int.max / 2 + 1) == nil)
        #expect(ledger.stages == [.pooled, .pooled])
        #expect(ledger.bytes == 0)
        #expect(pool.statistics.totalAllocated == 0)
    }

    @Test(arguments: [0, 1, 31, 32, 33, 64, 128])
    func mixedPhysicalByteAccountingAndZeroCount(budget: Int) {
        let ledger = E04AllocationLedger()
        var config = configuration(bytes: budget)
        config.cleanupInterval = -1
        config.maxBuffersPerSize = 1
        var pool: MemoryPool? = MemoryPool(configuration: config, allocationBoundary: ledger)
        for _ in 0..<3 {
            var bytes = pool!.acquire(type: UInt8.self, count: 17)
            var doubles = pool!.acquire(type: Double.self, count: 3)
            var zero = pool!.acquire(type: Double.self, count: 0)
            #expect(bytes!.count == 17 && doubles!.count == 3 && zero!.count == 0)
            #expect(pool!.statistics.totalInUse == 3)
            #expect(pool!.statistics.totalAllocated == 72)
            #expect(ledger.bytes == 72)
            pool!.cleanup()
            #expect(ledger.bytes == 72)
            bytes = nil
            #expect(pool!.statistics.cachedBytes == pool!.independentlyCountedCachedBytes)
            doubles = nil
            #expect(pool!.statistics.cachedBytes == pool!.independentlyCountedCachedBytes)
            zero = nil
            let stats = pool!.statistics
            #expect(stats.totalInUse == 0)
            #expect(stats.cachedBytes <= budget)
            #expect(stats.cachedBytes == pool!.independentlyCountedCachedBytes)
            #expect(stats.totalAllocated == stats.cachedBytes && ledger.bytes == stats.cachedBytes)
            pool!.cleanup(); pool!.cleanup()
            #expect(pool!.statistics.cachedBytes == 0 && ledger.bytes == 0)
        }
        pool = nil
        #expect(ledger.counts.allocated == ledger.counts.freed)
    }

    @Test func timerDomainAndCachedDestruction() {
        for interval in [Double.nan, .infinity, -.infinity, -1, 0, Double.greatestFiniteMagnitude, Double(Int.max) / 1e9] {
            let ledger = E04AllocationLedger()
            var config = configuration()
            config.cleanupInterval = interval
            var pool: MemoryPool? = MemoryPool(configuration: config, allocationBoundary: ledger)
            #expect(!pool!.automaticCleanupEnabled)
            pool!.withBuffer(type: UInt8.self, count: 8) { _ in }
            #expect(ledger.bytes == 8)
            pool = nil
            #expect(ledger.bytes == 0)
        }
        var config = configuration()
        config.cleanupInterval = 30
        let pool = MemoryPool(configuration: config)
        #expect(pool.automaticCleanupEnabled)
    }

    @Test func scopedThrowAndReferenceDestructionAcrossAllStages() {
        final class Payload { }
        enum Expected: Error { case body }
        for failures: Set<PoolAllocationStage> in [[], [.pooled], [.pooled, .direct]] {
            let ledger = E04AllocationLedger()
            ledger.fail(failures)
            let pool = MemoryPool(configuration: configuration(bytes: 0), allocationBoundary: ledger)
            do {
                let _: Int = try pool.withBuffer(type: Payload.self, count: 1, alignment: 64) { buffer in
                    var payload: Payload? = Payload()
                    weak var witness = payload
                    buffer.baseAddress!.initialize(to: payload!)
                    payload = nil
                    #expect(witness != nil)
                    defer {
                        buffer.baseAddress!.deinitialize(count: 1)
                        #expect(witness == nil)
                    }
                    throw Expected.body
                }
                Issue.record("expected body throw")
            } catch Expected.body { } catch { Issue.record("wrong error") }
            #expect(ledger.bytes == 0)
            #expect(pool.statistics.totalInUse == 0)
        }
    }
}

private struct E04TrapBoundary: PoolAllocationBoundary {
    func allocate(byteCount: Int, alignment: Int, family: PoolAllocationFamily, stage: PoolAllocationStage) -> UnsafeMutableRawPointer? {
        FileHandle.standardOutput.write(Data("E04_ALLOCATOR_CALLED\n".utf8))
        return nil
    }
    func deallocate(address: UInt, family: PoolAllocationFamily) { }
}

extension MemoryPoolBoundaryTests {
    @Test func guardChildProbe() async {
        guard let mode = ProcessInfo.processInfo.environment["E04_GUARD_MODE"] else { return }
        FileHandle.standardOutput.write(Data("E04_CHILD_STARTED:\(mode)\n".utf8))
        let boundary = E04TrapBoundary()
        switch mode {
        case "configuration-negative": _ = BufferConfiguration(maxPoolSize: -1)
        case "configuration-buffer-negative": _ = BufferConfiguration(maxBufferSize: -1)
        case "configuration-alignment": _ = BufferConfiguration(alignment: 3)
        case "internal-negative-budget": _ = BufferPool(memoryLimitMB: -1, allocationBoundary: boundary)
        case "internal-overflow-budget": _ = BufferPool(memoryLimitMB: Int.max, allocationBoundary: boundary)
        case "internal-negative-per-size": _ = BufferPool(maxBuffersPerSize: -1, allocationBoundary: boundary)
        case "typed-negative-budget": _ = TypedBufferPool<Float>(maxCachedBytes: -1, allocationBoundary: boundary)
        case "typed-negative-per-size": _ = TypedBufferPool<Float>(maxCachedBytes: 0, maxBuffersPerSize: -1, allocationBoundary: boundary)
        case "memory-negative-budget": var c = configuration(); c.maxTotalMemory = -1; _ = MemoryPool(configuration: c, allocationBoundary: boundary)
        case "memory-negative-per-size": var c = configuration(); c.maxBuffersPerSize = -1; _ = MemoryPool(configuration: c, allocationBoundary: boundary)
        case "internal-negative-count": _ = await BufferPool(allocationBoundary: boundary).acquire(count: -1)
        case "internal-overflow-count": _ = await BufferPool(allocationBoundary: boundary).acquire(count: Int.max)
        case "typed-negative-count": _ = await TypedBufferPool<Double>(maxCachedBytes: 0, allocationBoundary: boundary).acquire(count: Int.min)
        case "typed-overflow-count": _ = await TypedBufferPool<Double>(maxCachedBytes: 0, allocationBoundary: boundary).acquire(count: Int.max / 8 + 1)
        case "internal-allocation-failure": _ = await BufferPool(allocationBoundary: boundary).acquire(count: 1)
        case "typed-allocation-failure": _ = await TypedBufferPool<Float>(maxCachedBytes: 0, allocationBoundary: boundary).acquire(count: 1)
        case "scoped-negative-count": MemoryPool(allocationBoundary: boundary).withBuffer(type: Float.self, count: -1) { _ in print("E04_BODY_CALLED") }
        case "scoped-overflow-count": MemoryPool(allocationBoundary: boundary).withBuffer(type: Double.self, count: Int.max) { _ in print("E04_BODY_CALLED") }
        case "scoped-invalid-alignment": MemoryPool(allocationBoundary: boundary).withBuffer(type: Float.self, count: 1, alignment: 8) { _ in print("E04_BODY_CALLED") }
        case "scoped-final-failure": MemoryPool(allocationBoundary: boundary).withBuffer(type: Float.self, count: 1) { _ in print("E04_BODY_CALLED") }
        default: Issue.record("unknown guard case")
        }
        FileHandle.standardOutput.write(Data("E04_UNEXPECTED_RETURN\n".utf8))
    }

    @Test(.serialized, arguments: [
        "configuration-negative", "configuration-buffer-negative", "configuration-alignment",
        "internal-negative-budget", "internal-overflow-budget", "internal-negative-per-size",
        "typed-negative-budget", "typed-negative-per-size", "memory-negative-budget", "memory-negative-per-size",
        "internal-negative-count", "internal-overflow-count", "typed-negative-count", "typed-overflow-count",
        "internal-allocation-failure", "typed-allocation-failure", "scoped-negative-count", "scoped-overflow-count",
        "scoped-invalid-alignment", "scoped-final-failure"
    ])
    func namedGuardsRejectBeforeBodyOrInvalidAllocation(mode: String) throws {
        if ProcessInfo.processInfo.environment["E04_SKIP_GUARDS"] == "1" { return }
        var environment = ProcessInfo.processInfo.environment
        environment["E04_GUARD_MODE"] = mode
        let directory = environment["E04_CAPTURE_DIRECTORY"].map { URL(fileURLWithPath: $0) } ?? FileManager.default.temporaryDirectory
        let url = directory.appendingPathComponent("E04-guard-\(mode)-\(UUID().uuidString).log")
        _ = FileManager.default.createFile(atPath: url.path, contents: nil)
        let output = try FileHandle(forWritingTo: url)
        let process = try launchChildProbe(.swiftTesting(filter: "MemoryPoolBoundaryTests/guardChildProbe"),
                                           environment: environment, output: output, deadline: 15,
                                           cleanup: .terminateThenKill(grace: 1, seconds: 1))
        let timeout = process.timedOut
        try output.close()
        let text = try String(contentsOf: url, encoding: .utf8)
        print("E04_CHILD_CAPTURE \(mode) \(url.path)")
        if timeout {
            throw NSError(domain: "E04.childTimeout", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "child deadline: \(url.path)"])
        }
        #expect(!process.isRunning)
        #expect(process.terminationStatus != nil && process.terminationStatus != 0)
        #expect(text.contains("E04_CHILD_STARTED:\(mode)"))
        #expect(!text.contains("E04_BODY_CALLED") && !text.contains("E04_UNEXPECTED_RETURN"))
        let expected: String
        if mode.hasPrefix("configuration") { expected = "BufferConfiguration requires" }
        else if mode.hasPrefix("internal-") { expected = mode.contains("count") || mode.contains("failure") ? "BufferPool.acquire" : "BufferPool requires" }
        else if mode.hasPrefix("typed-") { expected = mode.contains("count") || mode.contains("failure") ? "TypedBufferPool.acquire" : "TypedBufferPool requires" }
        else if mode.hasPrefix("memory-") { expected = "MemoryPool requires" }
        else { expected = "MemoryPool.withBuffer" }
        #expect(text.contains(expected))
        #expect(text.contains("E04_ALLOCATOR_CALLED") == mode.contains("failure"))
    }
}

extension MemoryPoolBoundaryTests {
    @Test func synchronousScopedBodyObservesCancellationAndReleasesBeforeThrow() async {
        let ledger = E04AllocationLedger()
        let pool = MemoryPool(configuration: configuration(bytes: 0), allocationBoundary: ledger)
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try pool.withBuffer(type: String.self, count: 1) { buffer in
                buffer.baseAddress!.initialize(to: "cancelled")
                defer { buffer.baseAddress!.deinitialize(count: 1) }
                try Task.checkCancellation()
                return 1
            }
        }
        do { _ = try await task.value; Issue.record("expected cancellation") }
        catch { #expect(error is CancellationError) }
        #expect(ledger.bytes == 0)
        #expect(pool.statistics.totalInUse == 0)
    }
}
