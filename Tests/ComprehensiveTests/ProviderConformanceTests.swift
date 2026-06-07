//
//  ProviderConformanceTests.swift
//  VectorCore
//
//  Executable conformance suite for the provider protocols (beta-evolution-4,
//  DOCUMENT-4 S4). Encodes the semantic laws from
//  DOCUMENT-4_S4_Provider_Conformance.md and runs them against Core's conformers
//  (CPUComputeProvider, SwiftBufferPool) and minimal mocks. Downstream packages
//  (VectorAccelerate) should mirror these checkers against their own conformers.
//

import Testing
import Foundation
@testable import VectorCore

// MARK: - Helpers

private enum ProbeError: Error { case boom }

/// Actor that records how many times each index was visited (for completeness laws).
private actor IndexCounter {
    private var counts: [Int]
    init(_ n: Int) { counts = Array(repeating: 0, count: n) }
    func inc(_ i: Int) { counts[i] += 1 }
    func snapshot() -> [Int] { counts }
}

// MARK: - ComputeProvider conformance checker (laws C1–C8)

private func checkComputeProvider(_ p: any ComputeProvider, _ label: String) async throws {
    // C7 — metadata sanity
    #expect(p.maxConcurrency >= 1, "\(label): maxConcurrency")
    #expect(p.deviceInfo.maxThreads >= 1, "\(label): maxThreads")
    #expect(p.deviceInfo.preferredChunkSize >= 1, "\(label): preferredChunkSize")

    // C1 — execute fidelity
    let e = try await p.execute { 42 }
    #expect(e == 42, "\(label): execute returns work result")

    // C8 — error propagation
    var executeThrew = false
    do { _ = try await p.execute { throw ProbeError.boom } } catch { executeThrew = true }
    #expect(executeThrew, "\(label): execute propagates errors")

    let n = 1000

    // C2 — parallelExecute order & completeness
    let mapped = try await p.parallelExecute(items: 0..<n) { $0 * 2 }
    #expect(mapped.count == n, "\(label): parallelExecute length")
    #expect(mapped == (0..<n).map { $0 * 2 }, "\(label): parallelExecute preserves index order")

    // C6 — empty range
    let emptyMap = try await p.parallelExecute(items: 0..<0) { $0 }
    #expect(emptyMap.isEmpty, "\(label): parallelExecute empty → []")

    // C3 — parallelForEach completeness (exactly once per item)
    let counter = IndexCounter(n)
    try await p.parallelForEach(items: 0..<n) { await counter.inc($0) }
    let visits = await counter.snapshot()
    #expect(visits.allSatisfy { $0 == 1 }, "\(label): parallelForEach visits each item exactly once")

    // C4/C5 — parallelReduce partition with identity initial (0 is identity for +)
    let sum = try await p.parallelReduce(
        items: 0..<n, initial: 0,
        { range in range.reduce(0, +) },
        { $0 + $1 }
    )
    #expect(sum == (0..<n).reduce(0, +), "\(label): parallelReduce sums via partition")

    // C6 — reduce over empty range → initial
    let reduceEmpty = try await p.parallelReduce(
        items: 0..<0, initial: 7,
        { _ in 999 },
        { $0 + $1 }
    )
    #expect(reduceEmpty == 7, "\(label): parallelReduce empty → initial")
}

// MARK: - BufferProvider conformance checker (laws B1–B7)

private func checkBufferProvider(_ p: any BufferProvider, _ label: String) async throws {
    // B2 — alignment is a power of two ≥ 1
    #expect(p.alignment >= 1, "\(label): alignment ≥ 1")
    #expect(p.alignment.nonzeroBitCount == 1, "\(label): alignment is a power of two")

    let size = 1024
    let h1 = try await p.acquire(size: size)

    // B1 — size
    #expect(h1.size >= size, "\(label): acquired size ≥ requested")
    // B2 — pointer aligned
    #expect(Int(bitPattern: h1.pointer) % p.alignment == 0, "\(label): pointer aligned")

    // B3 — writability (first `size` bytes round-trip)
    let bytes = h1.pointer.assumingMemoryBound(to: UInt8.self)
    for i in 0..<size { bytes[i] = UInt8(truncatingIfNeeded: i) }
    var roundTripOK = true
    for i in 0..<size where bytes[i] != UInt8(truncatingIfNeeded: i) { roundTripOK = false; break }
    #expect(roundTripOK, "\(label): buffer is writable and reads back")

    // B4 — two simultaneously-active buffers do not alias
    let h2 = try await p.acquire(size: size)
    #expect(h2.id != h1.id, "\(label): distinct handle ids")
    #expect(h2.pointer != h1.pointer, "\(label): active buffers do not alias")
    await p.release(h2)
    await p.release(h1)

    // B7 — statistics sanity
    let stats = await p.statistics()
    #expect(stats.totalAllocations >= 1, "\(label): recorded an allocation")
    #expect(stats.hitRate >= 0 && stats.hitRate <= 1, "\(label): hitRate ∈ [0,1]")

    // B6 — clear keeps active handles valid
    let h3 = try await p.acquire(size: size)
    await p.clear()
    let live = h3.pointer.assumingMemoryBound(to: UInt8.self)
    live[0] = 7
    #expect(live[0] == 7, "\(label): active handle survives clear()")
    await p.release(h3)
}

// MARK: - Mocks

/// Minimal ComputeProvider implementing only `execute`; inherits parallel* defaults.
/// Verifies that the protocol's default implementations are themselves conformant.
private struct MockComputeProvider: ComputeProvider {
    let device: ComputeDevice = .cpu
    var maxConcurrency: Int { 4 }
    var deviceInfo: ComputeDeviceInfo {
        ComputeDeviceInfo(name: "mock", availableMemory: nil, maxThreads: 4, preferredChunkSize: 256)
    }
    func execute<T: Sendable>(_ work: @Sendable @escaping () async throws -> T) async throws -> T {
        try await work()
    }
}

/// Minimal BufferProvider backed by per-acquire aligned allocations.
private actor MockBufferProvider: BufferProvider {
    nonisolated let alignment: Int = 64
    private var active: [UUID: UnsafeMutableRawPointer] = [:]
    private var allocations = 0

    func acquire(size: Int) async throws -> BufferHandle {
        let raw = try AlignedMemory.allocateAligned(type: UInt8.self, count: max(1, size), alignment: alignment)
        let ptr = UnsafeMutableRawPointer(raw)
        let handle = BufferHandle(size: size, pointer: ptr)
        active[handle.id] = ptr
        allocations += 1
        return handle
    }
    func release(_ handle: BufferHandle) async {
        if let ptr = active.removeValue(forKey: handle.id) { AlignedMemory.deallocate(ptr) }
    }
    func statistics() async -> BufferStatistics {
        BufferStatistics(totalAllocations: allocations, reusedBuffers: 0, currentUsageBytes: 0, peakUsageBytes: 0)
    }
    func clear() async { /* nothing pooled; active buffers intentionally retained (B6) */ }
}

private enum MockAccelError: Error { case unsupported }

/// Minimal AccelerationProvider (a PAT) for A1/A3/A4 + type preservation.
private struct MockAccelerator: AccelerationProvider {
    struct Config: Sendable { let supported: Set<AcceleratedOperation> }
    let supported: Set<AcceleratedOperation>
    init(configuration: Config) async throws { self.supported = configuration.supported }
    func isSupported(for operation: AcceleratedOperation) -> Bool { supported.contains(operation) }
    func accelerate<T>(_ operation: AcceleratedOperation, input: T) async throws -> T {
        guard supported.contains(operation) else { throw MockAccelError.unsupported }
        return input
    }
}

// MARK: - Tests

@Suite("Provider conformance")
struct ProviderConformanceTests {

    @Test("CPUComputeProvider .sequential conforms")
    func cpuSequential() async throws {
        try await checkComputeProvider(CPUComputeProvider.sequential, "cpu.sequential")
    }

    @Test("CPUComputeProvider .parallel conforms")
    func cpuParallel() async throws {
        try await checkComputeProvider(CPUComputeProvider.parallel, "cpu.parallel")
    }

    @Test("CPUComputeProvider .automatic (parallel path) conforms")
    func cpuAutomatic() async throws {
        // Threshold 1 forces the parallel branch for n ≥ 1, exercising chunked execution.
        try await checkComputeProvider(
            CPUComputeProvider(mode: .automatic, parallelizationThreshold: 1), "cpu.automatic")
    }

    @Test("ComputeProvider default implementations conform")
    func defaultsConform() async throws {
        try await checkComputeProvider(MockComputeProvider(), "mock-defaults")
    }

    @Test("parallelReduce is mode-invariant for an identity initial (C5)")
    func reduceIdentityInvariant() async throws {
        let n = 500
        let seq = try await CPUComputeProvider.sequential.parallelReduce(
            items: 0..<n, initial: 0, { $0.reduce(0, +) }, { $0 + $1 })
        let par = try await CPUComputeProvider.parallel.parallelReduce(
            items: 0..<n, initial: 0, { $0.reduce(0, +) }, { $0 + $1 })
        #expect(seq == par)
        #expect(seq == (0..<n).reduce(0, +))
    }

    @Test("SwiftBufferPool conforms")
    func swiftBufferPool() async throws {
        try await checkBufferProvider(SwiftBufferPool(), "swift-pool")
    }

    @Test("MockBufferProvider conforms")
    func mockBufferPool() async throws {
        try await checkBufferProvider(MockBufferProvider(), "mock-pool")
    }

    @Test("AccelerationProvider laws A1/A3/A4 + type preservation")
    func accelerationProvider() async throws {
        let p = try await MockAccelerator(configuration: .init(supported: [.distanceComputation]))

        // A1 — isSupported deterministic
        #expect(p.isSupported(for: .distanceComputation))
        #expect(p.isSupported(for: .distanceComputation))
        #expect(!p.isSupported(for: .matrixMultiplication))

        // Type preservation for a supported op
        let out: [Float] = try await p.accelerate(.distanceComputation, input: [1, 2, 3])
        #expect(out == [1, 2, 3])

        // A3 — unsupported throws
        var threw = false
        do { _ = try await p.accelerate(.matrixMultiplication, input: [Float]()) } catch { threw = true }
        #expect(threw, "unsupported operation must throw, not silently mis-compute")
    }
}
