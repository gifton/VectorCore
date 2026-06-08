//
//  BatchKernelProviderTests.swift
//  VectorCore
//
//  R4: a ComputeProvider sub-protocol that supplies real batch kernels, so an
//  installed GPU provider transparently services Operations.findNearest.
//

import Testing
import Foundation
@testable import VectorCore

/// A mock GPU provider that returns an impossible-from-CPU sentinel, so tests can
/// detect that dispatch was delegated to it.
private struct MockGPUProvider: BatchKernelProvider {
    let device: ComputeDevice = .cpu
    var maxConcurrency: Int { 1 }
    var deviceInfo: ComputeDeviceInfo {
        ComputeDeviceInfo(name: "mock-gpu", availableMemory: nil, maxThreads: 1, preferredChunkSize: 1)
    }
    func execute<T: Sendable>(_ work: @Sendable @escaping () async throws -> T) async throws -> T {
        try await work()
    }
    func batchDistance<V: VectorProtocol>(query: V, candidates: [V], metric: any DistanceMetric)
        async throws -> [Float] where V.Scalar == Float {
        Array(repeating: Float(-1), count: candidates.count)
    }
    func findNearest<V: VectorProtocol>(query: V, candidates: [V], k: Int, metric: any DistanceMetric)
        async throws -> [(index: Int, distance: Float)] where V.Scalar == Float {
        [(index: 999, distance: -42)]   // impossible from the CPU path
    }
}

@Suite("BatchKernelProvider GPU dispatch (R4)")
struct BatchKernelProviderTests {

    @Test("A BatchKernelProvider is usable as a ComputeProvider")
    func conformsToComputeProvider() async throws {
        let p: any ComputeProvider = MockGPUProvider()
        let mapped = try await p.parallelExecute(items: 0..<4) { $0 * 2 }
        #expect(mapped == [0, 2, 4, 6])   // inherits the ComputeProvider default
    }
}
