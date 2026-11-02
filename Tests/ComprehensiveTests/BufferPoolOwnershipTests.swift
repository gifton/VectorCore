import Foundation
import Testing
@testable import VectorCore

/// Each test owns a separate boundary; all successful requests use real allocation.
final class E04AllocationLedger: PoolAllocationBoundary, @unchecked Sendable {
    struct Record: Equatable { let bytes: Int; let alignment: Int; let family: PoolAllocationFamily }
    private let lock = NSLock()
    private var live: [UInt: Record] = [:]
    private var allocationCount = 0
    private var freeCount = 0
    private var failures: Set<PoolAllocationStage> = []
    private var requests: [PoolAllocationStage] = []
    func fail(_ stages: Set<PoolAllocationStage>) { lock.withLock { failures = stages } }
    func allocate(byteCount: Int, alignment: Int, family: PoolAllocationFamily, stage: PoolAllocationStage) -> UnsafeMutableRawPointer? {
        let fail = lock.withLock { requests.append(stage); return failures.contains(stage) }
        guard !fail, let pointer = SystemPoolAllocationBoundary().allocate(byteCount: byteCount, alignment: alignment, family: family, stage: stage) else { return nil }
        lock.withLock {
            #expect(live.updateValue(Record(bytes: byteCount, alignment: alignment, family: family), forKey: UInt(bitPattern: pointer)) == nil)
            allocationCount += 1
        }
        return pointer
    }
    func deallocate(address: UInt, family: PoolAllocationFamily) {
        lock.withLock {
            let record = live.removeValue(forKey: address)
            #expect(record?.family == family)
            freeCount += 1
        }
        SystemPoolAllocationBoundary().deallocate(address: address, family: family)
    }
    var bytes: Int { lock.withLock { live.values.reduce(0) { $0 + $1.bytes } } }
    var counts: (allocated: Int, freed: Int) { lock.withLock { (allocationCount, freeCount) } }
    var stages: [PoolAllocationStage] { lock.withLock { requests } }
}

@Suite("E04 buffer pool ownership")
struct BufferPoolOwnershipTests {
    @Test func swiftPoolDestructionAndFreshReturnCap() async throws {
        let ledger = E04AllocationLedger()
        var pool: SwiftBufferPool? = SwiftBufferPool(configuration: BufferConfiguration(maxPoolSize: 64, maxBufferSize: 64), allocationBoundary: ledger)
        weak var owner = pool
        var handles: [BufferHandle] = []
        for _ in 0..<8 { handles.append(try await pool!.acquire(size: 64)) }
        #expect(ledger.bytes == 512)
        for handle in handles {
            await pool!.release(handle)
            #expect(await pool!.cachedBytes <= 64)
        }
        #expect(ledger.bytes == 64)
        pool = nil
        #expect(owner == nil)
        #expect(ledger.bytes == 0)
        #expect(ledger.counts.allocated == ledger.counts.freed)
    }

    /// Destroying a pool frees its cached allocations and leaks (never frees) its active leases, so a handle that
    /// outlived its pool still points at live memory (PR #56 review AM1; BASE leaked too).
    @Test func internalAndTypedDestructionOwnCachedAndLeakActive() async {
        let first = E04AllocationLedger()
        var pool: BufferPool? = BufferPool(allocationBoundary: first)
        let active = await pool!.acquire(count: 16)
        let returned = await pool!.acquire(count: 16)
        await pool!.release(returned)
        await pool!.clear()
        #expect(first.bytes == 64)
        pool = nil
        #expect(first.bytes == 64, "the active 16-Float lease was freed under a live handle")
        active.buffer.initialize(repeating: 1)      // safe: the lease is still allocated
        withExtendedLifetime(active) {}
        #expect(first.bytes == 64, "an escaped lease is leaked, not freed, after its handle drops")

        let second = E04AllocationLedger()
        var typed: TypedBufferPool<String>? = TypedBufferPool(maxCachedBytes: 128, allocationBoundary: second)
        let borrowed = await typed!.acquire(count: 2)
        let cached = await typed!.acquire(count: 2)
        await typed!.release(cached)
        typed = nil
        #expect(second.bytes == 2 * MemoryLayout<String>.stride, "the active 2-element lease was freed under a live handle")
        borrowed.buffer.initialize(repeating: "live")
        borrowed.buffer.deinitialize()
        withExtendedLifetime(borrowed) {}
    }

    /// `SwiftBufferPool`: the same teardown rule for its public `BufferHandle`.
    @Test func swiftPoolDestructionLeaksActiveLease() async throws {
        let ledger = E04AllocationLedger()
        let configuration = BufferConfiguration(maxPoolSize: 64, maxBufferSize: 64)
        var pool: SwiftBufferPool? = SwiftBufferPool(configuration: configuration, allocationBoundary: ledger)
        let active = try await pool!.acquire(size: 48)
        let returned = try await pool!.acquire(size: 48)
        await pool!.release(returned)
        #expect(ledger.bytes == 128)
        pool = nil
        #expect(ledger.bytes == 64, "the active lease was freed under a live handle")
        active.pointer.storeBytes(of: UInt64.max, as: UInt64.self)   // safe: the lease is still allocated
        withExtendedLifetime(active) {}
        #expect(ledger.bytes == 64)
        #expect(ledger.counts.allocated == ledger.counts.freed + 1)
    }
}

private final class E04BodyError: Error { let identity = UUID() }
private actor E04Gate {
    private var open = false
    private var waiters: [CheckedContinuation<Void, Never>] = []
    func wait() async {
        if open { return }
        await withCheckedContinuation { waiters.append($0) }
    }
    func signal() { open = true; let pending = waiters; waiters.removeAll(); for waiter in pending { waiter.resume() } }
}

extension BufferPoolOwnershipTests {
    @Test(arguments: [0, 1, 63, 64, 65, 128])
    func swiftLifecycleBudgetMatrix(budget: Int) async throws {
        let ledger = E04AllocationLedger()
        var pool: SwiftBufferPool? = SwiftBufferPool(configuration: BufferConfiguration(maxPoolSize: budget, maxBufferSize: 64), allocationBoundary: ledger)
        #expect(ledger.bytes == 0)
        await pool!.clear(); await pool!.clear()
        for _ in 0..<3 {
            let zero = try await pool!.acquire(size: 0)
            let one = try await pool!.acquire(size: 1)
            let large = try await pool!.acquire(size: 33)
            #expect((await pool!.statistics()).currentUsageBytes == 66)
            await pool!.clear()
            #expect(ledger.bytes == 66)
            for handle in [large, zero, one] {
                await pool!.release(handle)
                let stats = await pool!.statistics()
                let cached = await pool!.cachedBytes
                #expect(cached <= budget)
                #expect(ledger.bytes == stats.currentUsageBytes + cached)
            }
            await pool!.clear()
            #expect(ledger.bytes == 0)
        }
        let active = try await pool!.acquire(size: 7)
        pool = nil
        #expect(ledger.bytes == 8, "the active lease (capacity 8) was freed under a live handle")
        withExtendedLifetime(active) {}
        #expect(ledger.counts.allocated == ledger.counts.freed + 1)
    }

    @Test func swiftRejectsAllForgedFieldsBeforeConsumingLease() async throws {
        let ledger = E04AllocationLedger()
        let pool = SwiftBufferPool(allocationBoundary: ledger)
        let foreign = SwiftBufferPool()
        let first = try await pool.acquire(size: 32)
        await foreign.release(first)
        await pool.release(BufferHandle(id: first.id, size: 31, pointer: first.pointer))
        let other = try await pool.acquire(size: 32)
        await pool.release(BufferHandle(id: first.id, size: first.size, pointer: other.pointer))
        await pool.release(BufferHandle(size: first.size, pointer: first.pointer))
        #expect((await pool.statistics()).currentUsageBytes == 64)
        #expect(await pool.cachedBytes == 0)
        await pool.release(first)
        let reused = try await pool.acquire(size: 32)
        #expect(reused.pointer == first.pointer)
        #expect(reused.id != first.id)
        await pool.release(first)
        #expect((await pool.statistics()).currentUsageBytes == 64)
        await pool.release(reused); await pool.release(reused); await pool.release(other)
        #expect(await pool.cachedBytes == 64)
        await pool.clear()
        #expect(ledger.bytes == 0)
    }

    @Test func statisticsAndRequestFailures() async throws {
        for tracking in [false, true] {
            let ledger = E04AllocationLedger()
            let pool = SwiftBufferPool(configuration: BufferConfiguration(trackStatistics: tracking), allocationBoundary: ledger)
            for invalid in [Int.min, -1, Int.max, (Int.max / 2) + 2] {
                do { _ = try await pool.acquire(size: invalid); Issue.record("accepted invalid request") }
                catch BufferError.invalidRequest(let size) { #expect(size == invalid) }
                catch { Issue.record("wrong error: \(error)") }
            }
            #expect(ledger.stages.isEmpty)
            let first = try await pool.acquire(size: 17)
            await pool.release(first)
            let second = try await pool.acquire(size: 17)
            await pool.clear()
            let stats = await pool.statistics()
            #expect(stats.currentUsageBytes == 32)
            #expect(stats.totalAllocations == (tracking ? 1 : 0))
            #expect(stats.reusedBuffers == (tracking ? 1 : 0))
            #expect(stats.peakUsageBytes == (tracking ? 32 : 0))
            await pool.release(second); await pool.clear()
            #expect((await pool.statistics()).totalAllocations == stats.totalAllocations)
            ledger.fail([.pooled])
            do { _ = try await pool.acquire(size: 64); Issue.record("expected allocation failure") }
            catch BufferError.allocationFailed(let size) { #expect(size == 64) }
            catch { Issue.record("wrong error") }
        }
        #expect(BufferStatistics(totalAllocations: Int.max, reusedBuffers: Int.max, currentUsageBytes: 0, peakUsageBytes: 0).hitRate == 0.5)
        #expect(BufferStatistics(totalAllocations: -1, reusedBuffers: 1, currentUsageBytes: 0, peakUsageBytes: 0).hitRate == 0)
    }

    @Test(arguments: [0, 1, 31, 32, 33, 64])
    func typedLifecycleBudgetAndFreshLease(budget: Int) async {
        let ledger = E04AllocationLedger()
        var pool: TypedBufferPool<Double>? = TypedBufferPool(maxCachedBytes: budget, maxBuffersPerSize: 1, allocationBoundary: ledger)
        let foreign = TypedBufferPool<Double>(maxCachedBytes: 64)
        await pool!.clear(); await pool!.clear()
        for _ in 0..<3 {
            let a = await pool!.acquire(count: 4)
            let b = await pool!.acquire(count: 4)
            let zero = await pool!.acquire(count: 0)
            #expect(ledger.bytes == 72)
            await foreign.release(a)
            await pool!.clear()
            #expect(await pool!.activeBytes == 72)
            await pool!.release(a)
            let reused = await pool!.acquire(count: 4)
            await pool!.release(a)
            #expect(await pool!.activeBytes == 72)
            for h in [b, zero, reused] {
                await pool!.release(h); await pool!.release(h)
                let cached = await pool!.cachedBytes
                let active = await pool!.activeBytes
                #expect(cached <= budget)
                #expect(ledger.bytes == active + cached)
            }
            await pool!.clear()
            #expect(ledger.bytes == 0)
        }
        pool = nil
        #expect(ledger.bytes == 0)
    }

    @Test(arguments: [0, 1])
    func internalBudgetAndReleaseIdentity(megabytes: Int) async {
        let ledger = E04AllocationLedger()
        var pool: BufferPool? = BufferPool(maxBuffersPerSize: 1, memoryLimitMB: megabytes, allocationBoundary: ledger)
        let other = BufferPool()
        await pool!.cleanup(); await pool!.clear()
        let count = 262_144
        let first = await pool!.acquire(count: count)
        let second = await pool!.acquire(count: count)
        await other.release(first)
        await pool!.clear()
        #expect(ledger.bytes == 2 * 1_048_576)
        await pool!.release(first)
        let next = await pool!.acquire(count: count)
        await pool!.release(first)
        #expect(await pool!.activeBytes == 2 * 1_048_576)
        await pool!.release(next); await pool!.release(second); await pool!.release(second)
        #expect(await pool!.cachedBytes == megabytes * 1_048_576)
        #expect(ledger.bytes == megabytes * 1_048_576)
        #expect((await pool!.getStatistics()).releases == 3)
        await pool!.cleanup(); await pool!.cleanup()
        #expect(ledger.bytes == 0)
        pool = nil
    }

    @Test func scopedProviderSuccessThrowCancellationAndAcquireFailure() async throws {
        let ledger = E04AllocationLedger()
        let pool = SwiftBufferPool(configuration: BufferConfiguration(maxPoolSize: 0), allocationBoundary: ledger)
        let result = try await pool.withBuffer(size: 16) { _ in 42 }
        #expect(result == 42 && ledger.bytes == 0)
        let exact = E04BodyError()
        do { let _: Int = try await pool.withBuffer(size: 16) { _ in throw exact }; Issue.record("expected throw") }
        catch { #expect((error as? E04BodyError) === exact) }
        #expect(ledger.bytes == 0)
        let entered = E04Gate()
        let task = Task {
            try await pool.withBuffer(size: 16) { _ in
                await entered.signal()
                try await Task.sleep(for: .seconds(30))
                return 0
            }
        }
        await entered.wait(); task.cancel()
        do { _ = try await task.value; Issue.record("expected cancellation") }
        catch { #expect(error is CancellationError) }
        #expect(ledger.bytes == 0)
        ledger.fail([.pooled])
        do { let _: Int = try await pool.withBuffer(size: 16) { _ in Issue.record("body ran after allocation failure"); return 0 } }
        catch BufferError.allocationFailed { }
        #expect(ledger.counts.allocated == ledger.counts.freed)
    }

    @Test func scopedProviderRetainedDuringCancellationIgnoringBody() async throws {
        let ledger = E04AllocationLedger()
        var pool: SwiftBufferPool? = SwiftBufferPool(configuration: BufferConfiguration(maxPoolSize: 0), allocationBoundary: ledger)
        weak var weakPool = pool
        let entered = E04Gate(), finish = E04Gate()
        let task = Task { [provider = pool!] in
            try await provider.withBuffer(size: 16) { _ in
                await entered.signal(); await finish.wait()
                return Task.isCancelled
            }
        }
        await entered.wait(); pool = nil; task.cancel()
        #expect(weakPool != nil && ledger.bytes == 16)
        await weakPool!.clear()
        #expect(ledger.bytes == 16)
        await finish.signal()
        #expect(try await task.value)
        #expect(ledger.bytes == 0)
    }

    @Test func scopedInternalContextsAwaitReleaseAndTypedReferenceLifetime() async throws {
        let ledger = E04AllocationLedger()
        let pool = BufferPool(memoryLimitMB: 0, allocationBoundary: ledger)
        let context = BufferPoolContext(pool: pool)
        let operation = OperationContext(bufferPool: pool)
        let exact = E04BodyError()
        #expect(await context.withTemporaryBuffer(count: 3) { $0.count } == 3)
        #expect(ledger.bytes == 0)
        #expect(await operation.withTemporaryBuffer(count: 3) { $0.count } == 3)
        #expect(ledger.bytes == 0)
        for useOperation in [false, true] {
            do {
                if useOperation { _ = try await operation.withTemporaryBuffer(count: 3) { _ -> Int in throw exact } }
                else { _ = try await context.withTemporaryBuffer(count: 3) { _ -> Int in throw exact } }
                Issue.record("expected throw")
            } catch { #expect((error as? E04BodyError) === exact) }
            #expect(ledger.bytes == 0)
            let entered = E04Gate()
            let task = Task {
                if useOperation {
                    return try await operation.withTemporaryBuffer(count: 3) { _ in await entered.signal(); try await Task.sleep(for: .seconds(30)); return 0 }
                }
                return try await context.withTemporaryBuffer(count: 3) { _ in await entered.signal(); try await Task.sleep(for: .seconds(30)); return 0 }
            }
            await entered.wait(); task.cancel()
            do { _ = try await task.value; Issue.record("expected cancellation") } catch { #expect(error is CancellationError) }
            #expect(ledger.bytes == 0)
        }
        let typed = TypedBufferPool<E04BodyError>(maxCachedBytes: 0, allocationBoundary: ledger)
        let referenceReleased = await typed.withBuffer(count: 1) { buffer in
            var reference: E04BodyError? = E04BodyError()
            weak var weakReference = reference
            buffer.baseAddress!.initialize(to: reference!)
            reference = nil
            #expect(weakReference != nil)
            buffer.baseAddress!.deinitialize(count: 1)
            return weakReference == nil
        }
        #expect(referenceReleased)
        do { let _: Int = try await typed.withBuffer(count: 1) { _ in throw exact } } catch { #expect((error as? E04BodyError) === exact) }
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await typed.withBuffer(count: 1) { _ in try Task.checkCancellation(); return 0 }
        }
        do { _ = try await task.value; Issue.record("expected cancellation") } catch { #expect(error is CancellationError) }
        #expect(ledger.bytes == 0)
    }
}

extension BufferPoolOwnershipTests {
    @Test func emptyOwnersOversizedReturnsAndInternalZeroCount() async throws {
        let ledger = E04AllocationLedger()
        var swift: SwiftBufferPool? = SwiftBufferPool(configuration: BufferConfiguration(maxPoolSize: 100, maxBufferSize: 32), allocationBoundary: ledger)
        var typed: TypedBufferPool<Float>? = TypedBufferPool(maxCachedBytes: 0, allocationBoundary: ledger)
        var internalPool: BufferPool? = BufferPool(memoryLimitMB: 0, allocationBoundary: ledger)
        #expect(await swift!.cachedBytes == 0)
        #expect(await typed!.cachedBytes == 0)
        #expect(await internalPool!.cachedBytes == 0)
        swift = nil; typed = nil; internalPool = nil
        #expect(ledger.counts.allocated == 0 && ledger.counts.freed == 0)
        let pool = SwiftBufferPool(configuration: BufferConfiguration(maxPoolSize: 100, maxBufferSize: 32), allocationBoundary: ledger)
        let oversized = try await pool.acquire(size: 33)
        await pool.release(oversized)
        #expect(ledger.bytes == 0)
        let floats = BufferPool(memoryLimitMB: 0, allocationBoundary: ledger)
        let zero = await floats.acquire(count: 0)
        #expect(zero.buffer.count == 0 && ledger.bytes == 4)
        await floats.release(zero)
        #expect(ledger.bytes == 0)
        let context = BufferPoolContext(pool: floats)
        #expect(await context.withTemporaryBuffers(counts: [0, 1, 2]) { $0.map(\.count) } == [0, 1, 2])
        #expect(ledger.bytes == 0)
    }
}

private actor E04RecordingProvider: BufferProvider {
    enum Failure: Error { case acquisition }
    let pool = SwiftBufferPool(configuration: BufferConfiguration(maxPoolSize: 0))
    var acquisitions = 0
    var releases = 0
    let fail: Bool
    init(fail: Bool = false) { self.fail = fail }
    func acquire(size: Int) async throws -> BufferHandle {
        acquisitions += 1
        if fail { throw Failure.acquisition }
        return try await pool.acquire(size: size)
    }
    func release(_ handle: BufferHandle) async { releases += 1; await pool.release(handle) }
    func statistics() async -> BufferStatistics { await pool.statistics() }
    func clear() async { await pool.clear() }
}

extension BufferPoolOwnershipTests {
    @Test func protocolScopedDefaultInvokesReleaseExactlyOncePerExit() async throws {
        let provider = E04RecordingProvider()
        _ = try await provider.withBuffer(size: 1) { _ in 1 }
        #expect(await provider.acquisitions == 1)
        #expect(await provider.releases == 1)
        let exact = E04BodyError()
        do { let _: Int = try await provider.withBuffer(size: 1) { _ in throw exact } }
        catch { #expect((error as? E04BodyError) === exact) }
        #expect(await provider.releases == 2)
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await provider.withBuffer(size: 1) { _ in try Task.checkCancellation(); return 1 }
        }
        do { _ = try await task.value; Issue.record("expected cancellation") }
        catch { #expect(error is CancellationError) }
        #expect(await provider.acquisitions == 3)
        #expect(await provider.releases == 3)
        let failing = E04RecordingProvider(fail: true)
        do { let _: Int = try await failing.withBuffer(size: 1) { _ in Issue.record("body ran after acquisition failure"); return 1 } }
        catch E04RecordingProvider.Failure.acquisition { }
        #expect(await failing.acquisitions == 1)
        #expect(await failing.releases == 0)
    }
}

extension BufferPoolOwnershipTests {
    @Test func internalCachedAllocationIsFreedByDestructionWithoutClear() async {
        let ledger = E04AllocationLedger()
        var pool: BufferPool? = BufferPool(allocationBoundary: ledger)
        weak var owner = pool
        let handle = await pool!.acquire(count: 16)
        await pool!.release(handle)
        #expect(ledger.bytes == 64)
        #expect(await pool!.cachedBytes == 64)
        pool = nil
        #expect(owner == nil)
        #expect(ledger.bytes == 0)
        #expect(ledger.counts.allocated == 1 && ledger.counts.freed == 1)
        withExtendedLifetime(handle) {}
    }
}
