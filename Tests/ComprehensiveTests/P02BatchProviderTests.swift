import Foundation
import Testing
@testable import VectorCore

// P02 phase A provider tests for `Operations.findNearestBatch` (API-41; phase 1 check map rows P5, P6, P8, E20-E22),
// written before any P02 production edit, at Q = 8 and N = 256 with eligible optimized types (512/768/1536) and
// Euclidean and cosine metrics: the inputs that reach the implicit Gram (GEMM) route on the unchanged tree when no
// BatchKernelProvider is installed. The earlier Q4/Q5 provider tests never reach that gate.
//
// - Provider precedence (P5/P6/E20-E22): an installed BatchKernelProvider receives the original whole batch, k
//   (including k > N) and metric once, ahead of any CPU route, and its rows and errors come back unchanged. These
//   already hold on the unchanged tree (route 1 precedes the GEMM gate) and must keep holding after the default
//   changes. Phase C added `providerPrecedesInvalidAndInfeasibleCPUOptions`: the same precedence with invalid and
//   infeasible CPU-only `cpuStrategy` options, which the provider path never validates or plans; the phase C review
//   added `providerPrecedesFeasibleGram`, the same precedence over a `.gram` request the CPU route would run.
// - CPU-policy control (P8): a plain sequential `ComputeProvider` (no BatchKernelProvider) must schedule the
//   per-query fallback: one `parallelExecute(0..<8)` on the installed provider, no submitted worker tasks, and rows
//   equal to `findNearest`. These FAIL on the unchanged tree, which takes the GEMM route instead.

// MARK: - Spies

struct P02ProviderError: Error, Equatable, Sendable { let token: Int }

final class P02ProviderLog: @unchecked Sendable {
    struct Call: Equatable, Sendable, CustomStringConvertible {
        let queries: Int
        let candidates: Int
        let k: Int
        let metric: String
        var description: String { "(\(queries)x\(candidates) k\(k) \(metric))" }
    }
    private let lock = NSLock()
    private var batch: [Call] = []
    private var single: [Call] = []
    private var executes: [Range<Int>] = []
    func noteBatch(_ call: Call) { lock.lock(); defer { lock.unlock() }; batch.append(call) }
    func noteSingle(_ call: Call) { lock.lock(); defer { lock.unlock() }; single.append(call) }
    func noteExecute(_ range: Range<Int>) { lock.lock(); defer { lock.unlock() }; executes.append(range) }
    var batchCalls: [Call] { lock.lock(); defer { lock.unlock() }; return batch }
    var singleCalls: [Call] { lock.lock(); defer { lock.unlock() }; return single }
    var executeRanges: [Range<Int>] { lock.lock(); defer { lock.unlock() }; return executes }
}

private func p02MetricName(_ metric: any DistanceMetric) -> String { String(describing: type(of: metric)) }

/// A distinguishable per-query reply: one entry whose index is 1000 + the bit pattern's low byte of the query's
/// first scalar and whose distance is that scalar, so each row identifies the query it answered.
private func p02ReplyFor<V: VectorProtocol>(_ query: V) -> [(index: Int, distance: Float)] where V.Scalar == Float {
    let first = query.withUnsafeBufferPointer { $0[0] }
    return [(index: 1000 + Int(first.bitPattern & 0xFF), distance: first)]
}

/// A BatchKernelProvider WITH a `findNearestBatch` override that returns `batchReply` verbatim (or throws).
struct P02BatchOverrideSpy: BatchKernelProvider {
    let log: P02ProviderLog
    var batchReply: [[(index: Int, distance: Float)]]
    var failure: P02ProviderError?

    var device: ComputeDevice { .cpu }
    var maxConcurrency: Int { 2 }
    var deviceInfo: ComputeDeviceInfo {
        ComputeDeviceInfo(name: "p02-batch-override-spy", availableMemory: nil, maxThreads: 2, preferredChunkSize: 1)
    }
    func execute<T: Sendable>(_ work: @Sendable @escaping () async throws -> T) async throws -> T { try await work() }
    func batchDistance<V: VectorProtocol>(query: V, candidates: [V], metric: any DistanceMetric)
        async throws -> [Float] where V.Scalar == Float { [] }
    func findNearest<V: VectorProtocol>(query: V, candidates: [V], k: Int, metric: any DistanceMetric)
        async throws -> [(index: Int, distance: Float)] where V.Scalar == Float {
        log.noteSingle(.init(queries: 1, candidates: candidates.count, k: k, metric: p02MetricName(metric)))
        return []
    }
    func findNearestBatch<V: VectorProtocol>(queries: [V], candidates: [V], k: Int, metric: any DistanceMetric)
        async throws -> [[(index: Int, distance: Float)]] where V.Scalar == Float {
        log.noteBatch(.init(queries: queries.count, candidates: candidates.count, k: k, metric: p02MetricName(metric)))
        if let failure { throw failure }
        return batchReply
    }
}

/// A BatchKernelProvider WITHOUT a `findNearestBatch` override: the protocol default runs `findNearest` per query
/// through this provider's `parallelExecute`. Each per-query reply is `p02ReplyFor(query)` (or a throw).
struct P02InheritedBatchSpy: BatchKernelProvider {
    let log: P02ProviderLog
    var failure: P02ProviderError?

    var device: ComputeDevice { .cpu }
    var maxConcurrency: Int { 2 }
    var deviceInfo: ComputeDeviceInfo {
        ComputeDeviceInfo(name: "p02-inherited-batch-spy", availableMemory: nil, maxThreads: 2, preferredChunkSize: 1)
    }
    func execute<T: Sendable>(_ work: @Sendable @escaping () async throws -> T) async throws -> T { try await work() }
    func batchDistance<V: VectorProtocol>(query: V, candidates: [V], metric: any DistanceMetric)
        async throws -> [Float] where V.Scalar == Float { [] }
    func findNearest<V: VectorProtocol>(query: V, candidates: [V], k: Int, metric: any DistanceMetric)
        async throws -> [(index: Int, distance: Float)] where V.Scalar == Float {
        log.noteSingle(.init(queries: 1, candidates: candidates.count, k: k, metric: p02MetricName(metric)))
        if let failure { throw failure }
        return p02ReplyFor(query)
    }
}

/// A scheduling-only ComputeProvider (not a BatchKernelProvider) that delegates every requirement to
/// `CPUComputeProvider.sequential` and records the item range of each `parallelExecute` call.
struct P02ExecuteRecorder: ComputeProvider {
    let log: P02ProviderLog
    let inner: any ComputeProvider = CPUComputeProvider(mode: .sequential)

    var device: ComputeDevice { inner.device }
    var maxConcurrency: Int { inner.maxConcurrency }
    var deviceInfo: ComputeDeviceInfo { inner.deviceInfo }
    func execute<T: Sendable>(_ work: @Sendable @escaping () async throws -> T) async throws -> T {
        try await inner.execute(work)
    }
    func parallelExecute<T: Sendable>(items: Range<Int>, _ work: @Sendable @escaping (Int) async throws -> T) async throws -> [T] {
        log.noteExecute(items)
        return try await inner.parallelExecute(items: items, work)
    }
    func parallelForEach(items: Range<Int>, _ body: @Sendable @escaping (Int) async throws -> Void) async throws {
        try await inner.parallelForEach(items: items, body)
    }
    func parallelForEach(items: Range<Int>, minChunk: Int, _ body: @Sendable @escaping (Int) async throws -> Void) async throws {
        try await inner.parallelForEach(items: items, minChunk: minChunk, body)
    }
    func parallelReduce<R: Sendable>(items: Range<Int>, initial: R,
                                     _ rangeWork: @Sendable @escaping (Range<Int>) async throws -> R,
                                     _ combine: @Sendable @escaping (R, R) -> R) async throws -> R {
        try await inner.parallelReduce(items: items, initial: initial, rangeWork, combine)
    }
    func parallelReduce<R: Sendable>(items: Range<Int>, initial: R, minChunk: Int,
                                     _ rangeWork: @Sendable @escaping (Range<Int>) async throws -> R,
                                     _ combine: @Sendable @escaping (R, R) -> R) async throws -> R {
        try await inner.parallelReduce(items: items, initial: initial, minChunk: minChunk, rangeWork, combine)
    }
}

final class P02EventLog: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [BoundedExecution.Event] = []
    func append(_ event: BoundedExecution.Event) { lock.lock(); defer { lock.unlock() }; events.append(event) }
    var snapshot: [BoundedExecution.Event] { lock.lock(); defer { lock.unlock() }; return events }
}

// MARK: - Tests

@Suite("P02 batch provider precedence at Q8/N256 (API-41)")
struct P02BatchProviderTests {

    /// Distinguishable reply shape: fewer rows than queries, an empty row, a repeated index with NaN payloads.
    static let batchReply: [[(index: Int, distance: Float)]] = [
        [(index: 5, distance: 2)], [], [(index: 1, distance: .nan), (index: 1, distance: Float(bitPattern: 0x7FC0_0042))]
    ]
    static var batchReplyEntries: [[P01OpsEntry]] {
        batchReply.map { $0.map { P01OpsEntry(index: $0.index, bits: $0.distance.bitPattern) } }
    }
    static let ks = [1, 10, 256, 257, Int.max]

    /// P5/P6/E20: the override is called once with the original whole batch, k (including k > N and Int.max) and
    /// metric; its reply comes back verbatim and no per-query witness or CPU route runs.
    @Test func batchOverrideReceivesWholeBatchOnce() async throws {
        try await override(Vector512Optimized.self, width: 512)
        try await override(Vector768Optimized.self, width: 768)
        try await override(Vector1536Optimized.self, width: 1536)
    }

    private func override<V: VectorProtocol & Sendable>(_: V.Type, width: Int) async throws where V.Scalar == Float {
        let all = try p02Vectors(p02Rows(.near, width: width), as: V.self)
        let queries = Array(all.prefix(8))
        for k in Self.ks {
            try await overrideCase(queries, all, k: k, metric: EuclideanDistance(), width: width)
            try await overrideCase(queries, all, k: k, metric: CosineDistance(), width: width)
        }
    }

    private func overrideCase<V: VectorProtocol & Sendable, M: DistanceMetric>(
        _ queries: [V], _ candidates: [V], k: Int, metric: M, width: Int
    ) async throws where V.Scalar == Float, M.Scalar == Float {
        let name = String(describing: M.self)
        let log = P02ProviderLog()
        let spy = P02BatchOverrideSpy(log: log, batchReply: Self.batchReply)
        let result = try await Operations.$computeProvider.withValue(spy) {
            try await Operations.findNearestBatch(queries: queries, in: candidates, k: k, metric: metric)
        }
        #expect(result.map(p01Entries) == Self.batchReplyEntries, "w\(width) \(name) k\(k)")
        #expect(log.batchCalls == [.init(queries: 8, candidates: 256, k: k, metric: name)], "w\(width) \(name) k\(k)")
        #expect(log.singleCalls.isEmpty, "w\(width) \(name) k\(k)")
    }

    /// E21: without an override, the protocol default calls the provider's `findNearest` once per query with the
    /// original k and metric; each returned row is that query's own reply.
    @Test func inheritedBatchDefaultCallsPerQueryWitness() async throws {
        try await inherited(Vector512Optimized.self, width: 512)
        try await inherited(Vector768Optimized.self, width: 768)
        try await inherited(Vector1536Optimized.self, width: 1536)
    }

    private func inherited<V: VectorProtocol & Sendable>(_: V.Type, width: Int) async throws where V.Scalar == Float {
        let all = try p02Vectors(p02Rows(.near, width: width), as: V.self)
        let queries = Array(all.prefix(8))
        let expected = queries.map { p02ReplyFor($0).map { P01OpsEntry(index: $0.index, bits: $0.distance.bitPattern) } }
        for k in [10, 257] {
            try await inheritedCase((queries, all, expected), k: k, metric: EuclideanDistance(), width: width)
            try await inheritedCase((queries, all, expected), k: k, metric: CosineDistance(), width: width)
        }
    }

    private func inheritedCase<V: VectorProtocol & Sendable, M: DistanceMetric>(
        _ inputs: (queries: [V], candidates: [V], expected: [[P01OpsEntry]]), k: Int, metric: M, width: Int
    ) async throws where V.Scalar == Float, M.Scalar == Float {
        let (queries, candidates, expected) = inputs
        let name = String(describing: M.self)
        let log = P02ProviderLog()
        let result = try await Operations.$computeProvider.withValue(P02InheritedBatchSpy(log: log)) {
            try await Operations.findNearestBatch(queries: queries, in: candidates, k: k, metric: metric)
        }
        #expect(result.map(p01Entries) == expected, "w\(width) \(name) k\(k)")
        #expect(log.singleCalls == Array(repeating: .init(queries: 1, candidates: 256, k: k, metric: name), count: 8),
                "w\(width) \(name) k\(k)")
    }

    /// E22: the original provider error comes back unchanged, from the override and from the inherited default.
    @Test func providerErrorsPassThroughUnchanged() async throws {
        let all = try p02Vectors(p02Rows(.near, width: 512), as: Vector512Optimized.self)
        let queries = Array(all.prefix(8))
        try await errorCase(queries, all, metric: EuclideanDistance())
        try await errorCase(queries, all, metric: CosineDistance())
    }

    private func errorCase<M: DistanceMetric>(
        _ queries: [Vector512Optimized], _ candidates: [Vector512Optimized], metric: M
    ) async throws where M.Scalar == Float {
        let failingOverride = P02BatchOverrideSpy(log: P02ProviderLog(), batchReply: [], failure: P02ProviderError(token: 7))
        await #expect(throws: P02ProviderError(token: 7)) {
            _ = try await Operations.$computeProvider.withValue(failingOverride) {
                try await Operations.findNearestBatch(queries: queries, in: candidates, k: 257, metric: metric)
            }
        }
        let failingInherited = P02InheritedBatchSpy(log: P02ProviderLog(), failure: P02ProviderError(token: 8))
        await #expect(throws: P02ProviderError(token: 8)) {
            _ = try await Operations.$computeProvider.withValue(failingInherited) {
                try await Operations.findNearestBatch(queries: queries, in: candidates, k: 10, metric: metric)
            }
        }
    }

    /// P5, P7 (phase C): with an installed BatchKernelProvider, CPU-only options that would be rejected on the CPU
    /// route (`.gram(0, 0, -1)`: invalid fields; `.gram(8, 256, 0)`: a zero payload limit, infeasible for any plan;
    /// `.gram` with Manhattan: unsupported) do not reach validation or planning. The override is called once with the
    /// original batch, k (also > N) and metric and its reply comes back verbatim; the inherited default calls the
    /// per-query witness 8 times; no Gram event is traced.
    @Test func providerPrecedesInvalidAndInfeasibleCPUOptions() async throws {
        let all = try p02Vectors(p02Rows(.near, width: 512), as: Vector512Optimized.self)
        let queries = Array(all.prefix(8))
        let options: [BatchSearchCPUStrategy] = [
            .gram(queryTileRows: 0, candidateTileRows: 0, maximumTemporaryPayloadBytes: -1),
            .gram(queryTileRows: 8, candidateTileRows: 256, maximumTemporaryPayloadBytes: 0)
        ]
        for option in options {
            for k in [10, 257] {
                for euclidean in [true, false] {
                    let label = "\(option) k\(k) \(euclidean)"
                    let name = euclidean ? "EuclideanDistance" : "CosineDistance"
                    let log = P02ProviderLog()
                    let traces = P02GramTraceLog()
                    let spy = P02BatchOverrideSpy(log: log, batchReply: Self.batchReply)
                    let result = try await Operations.$gramTrace.withValue({ traces.append($0) }, operation: {
                        try await Operations.$computeProvider.withValue(spy) {
                            try await p02Batch(queries, all, k: k, euclidean: euclidean, strategy: option)
                        }
                    })
                    #expect(result.map(p01Entries) == Self.batchReplyEntries, "\(label)")
                    #expect(log.batchCalls == [.init(queries: 8, candidates: 256, k: k, metric: name)], "\(label)")
                    let inheritedLog = P02ProviderLog()
                    let inherited = try await Operations.$gramTrace.withValue({ traces.append($0) }, operation: {
                        try await Operations.$computeProvider.withValue(P02InheritedBatchSpy(log: inheritedLog)) {
                            try await p02Batch(queries, all, k: k, euclidean: euclidean, strategy: option)
                        }
                    })
                    #expect(inherited.count == 8, "\(label)")
                    #expect(inheritedLog.singleCalls == Array(repeating: .init(queries: 1, candidates: 256, k: k, metric: name),
                                                              count: 8), "\(label)")
                    #expect(traces.snapshot.isEmpty, "\(label)")
                }
            }
        }
        // An unsupported `.gram` metric with an infeasible limit: still the provider's reply.
        let log = P02ProviderLog()
        let reply = try await Operations.$computeProvider.withValue(P02BatchOverrideSpy(log: log, batchReply: Self.batchReply)) {
            try await Operations.findNearestBatch(queries: queries, in: all, k: 3, metric: ManhattanDistance(),
                                                  cpuStrategy: .gram(queryTileRows: 1, candidateTileRows: 1,
                                                                     maximumTemporaryPayloadBytes: 0))
        }
        #expect(reply.map(p01Entries) == Self.batchReplyEntries)
        #expect(log.batchCalls == [.init(queries: 8, candidates: 256, k: 3, metric: "ManhattanDistance")])
    }

    /// P5 over a FEASIBLE `.gram` (phase C review m4): `.gram(8, 256, Int.max)` on Vector512Optimized at Q8/N256 with
    /// Euclidean and cosine distance is a request the CPU route would run. With the override installed it is still
    /// called once with the original batch, k (also > N) and metric, its reply comes back verbatim, no per-query
    /// witness runs and the Gram seam traces no event. Control: the same request with no provider installed traces one
    /// Gram tile, so the zero-event check can fail (a route that tried `.gram` first would trace events here).
    @Test func providerPrecedesFeasibleGram() async throws {
        let all = try p02Vectors(p02Rows(.near, width: 512), as: Vector512Optimized.self)
        let queries = Array(all.prefix(8))
        let feasible = BatchSearchCPUStrategy.gram(queryTileRows: 8, candidateTileRows: 256,
                                                   maximumTemporaryPayloadBytes: Int.max)
        for k in [10, 257] {
            for euclidean in [true, false] {
                let label = "k\(k) \(euclidean ? "euclidean" : "cosine")"
                let name = euclidean ? "EuclideanDistance" : "CosineDistance"
                let log = P02ProviderLog()
                let traces = P02GramTraceLog()
                let spy = P02BatchOverrideSpy(log: log, batchReply: Self.batchReply)
                let result = try await Operations.$gramTrace.withValue({ traces.append($0) }, operation: {
                    try await Operations.$computeProvider.withValue(spy) {
                        try await p02Batch(queries, all, k: k, euclidean: euclidean, strategy: feasible)
                    }
                })
                #expect(result.map(p01Entries) == Self.batchReplyEntries, "\(label)")
                #expect(log.batchCalls == [.init(queries: 8, candidates: 256, k: k, metric: name)], "\(label)")
                #expect(log.singleCalls.isEmpty, "\(label)")
                #expect(traces.snapshot.isEmpty, "\(label): \(traces.snapshot.count) Gram events")
                // Control: without a provider, the same request runs the Gram route (one 8 x 256 tile).
                let control = P02GramTraceLog()
                let rows = try await Operations.$gramTrace.withValue({ control.append($0) }, operation: {
                    try await p02Batch(queries, all, k: k, euclidean: euclidean, strategy: feasible)
                })
                #expect(rows.count == 8 && control.tiles.count == 1, "\(label): control traced \(control.tiles.count) tiles")
            }
        }
    }

    /// P8 (CPU-policy control): under a plain sequential ComputeProvider every batch row equals `findNearest` under
    /// the same provider, bit for bit (the near cluster, where the GEMM route differs on the unchanged tree).
    @Test func sequentialProviderMatchesSingleQuery() async throws {
        try await sequential(Vector512Optimized.self, width: 512)
        try await sequential(Vector768Optimized.self, width: 768)
        try await sequential(Vector1536Optimized.self, width: 1536)
    }

    private func sequential<V: VectorProtocol & Sendable>(_: V.Type, width: Int) async throws where V.Scalar == Float {
        let all = try p02Vectors(p02Rows(.near, width: width), as: V.self)
        let queries = Array(all.prefix(8))
        try await Operations.$computeProvider.withValue(CPUComputeProvider(mode: .sequential)) {
            let euclidean = try await p02BatchRowsDifferingFromSingleQuery(
                queries: queries, candidates: all, k: 10, metric: EuclideanDistance())
            #expect(euclidean.isEmpty, "w\(width) euclidean: \(euclidean.count) rows differ, first \(euclidean.prefix(1))")
            let cosine = try await p02BatchRowsDifferingFromSingleQuery(
                queries: queries, candidates: all, k: 10, metric: CosineDistance())
            #expect(cosine.isEmpty, "w\(width) cosine: \(cosine.count) rows differ, first \(cosine.prefix(1))")
        }
    }

    /// P8/P9 (CPU-policy control): the scheduling-only provider is asked for the per-query fan-out exactly once,
    /// `parallelExecute(0..<8)`, as its first `parallelExecute`; the sequential provider submits no worker task
    /// (BoundedExecution observation: no `.submitted` event, and at least one `cpu.execute` region).
    @Test func schedulingOnlyProviderSchedulesPerQueryFallback() async throws {
        let all = try p02Vectors(p02Rows(.near, width: 512), as: Vector512Optimized.self)
        let queries = Array(all.prefix(8))
        try await schedulingCase(queries, all, metric: EuclideanDistance())
        try await schedulingCase(queries, all, metric: CosineDistance())
    }

    private func schedulingCase<M: DistanceMetric>(
        _ queries: [Vector512Optimized], _ candidates: [Vector512Optimized], metric: M
    ) async throws where M.Scalar == Float {
        let name = String(describing: M.self)
        let log = P02ProviderLog()
        let events = P02EventLog()
        let rows = try await BoundedExecution.$observation.withValue({ events.append($0) }, operation: {
            try await Operations.$computeProvider.withValue(P02ExecuteRecorder(log: log)) {
                try await Operations.findNearestBatch(queries: queries, in: candidates, k: 10, metric: metric)
            }
        })
        #expect(rows.count == 8, "\(name)")
        let ranges = log.executeRanges
        #expect(ranges.first == 0..<8 && ranges.filter { $0 == 0..<8 }.count == 1, "\(name): parallelExecute ranges \(ranges)")
        let snapshot = events.snapshot
        #expect(!snapshot.contains { $0.kind == .submitted }, "\(name): submitted worker tasks")
        #expect(snapshot.contains { $0.phase == "cpu.execute" && $0.kind == .begin }, "\(name): no cpu.execute region")
    }
}
