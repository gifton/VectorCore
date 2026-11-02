import Foundation
import Testing
@testable import VectorCore

/// P01 Operations behaviour changes (phase E), observed through public routes:
/// - Operations generic small-k: exactly N metric calls in a fresh process with a cold AutoTuning key,
///   entry into the installed ComputeProvider's reduction, and no tuning publication. The large-k twin is a
///   control that keeps calibration (N + M calls, published).
/// - Operations.findNearestBatch fallback: candidate validation does not repeat per query.
/// RED on the unchanged base (phase A evidence: 200 calls for N = 100 and calibration published; candidate reads per
/// row growing with the query count). Phase E turned it green and lifted `P01RedGate.operations`, so it runs by default.
@Suite(.serialized, .enabled(if: P01RedGate.operations))
struct P01OperationsRedTests {
    static let childFilter = "P01OperationsRedTests/p01Child"

    // MARK: Operations generic small-k callbacks and calibration (phase E)

    @Test func operationsColdKeySmallKEvaluatesEachCandidateOnce() throws {
        let child = try p01RunChild("operations-cold-small-k", filter: Self.childFilter, removing: ["VC_AUTOTUNE"])
        #expect(child.run.terminationStatus == 0, "status=\(child.run.statusDescription): \(child.tail)")
        #expect(child.output.contains("P01_CHILD_COMPLETE operations-cold-small-k"), "\(child.tail)")
    }

    /// Control (green before and after P01): large k keeps computeDistances calibration on a cold key.
    @Test func operationsColdKeyLargeKKeepsCalibrationControl() throws {
        let child = try p01RunChild("operations-cold-large-k", filter: Self.childFilter, removing: ["VC_AUTOTUNE"])
        #expect(child.run.terminationStatus == 0, "status=\(child.run.statusDescription): \(child.tail)")
        #expect(child.output.contains("P01_CHILD_COMPLETE operations-cold-large-k"), "\(child.tail)")
    }

    // MARK: Batch fallback validation (phase E)

    @Test func batchFallbackCandidateValidationDoesNotRepeatPerQuery() async throws {
        func reads(queries count: Int) async throws -> [Int: Int] {
            P01CountingVector.log.reset()
            let rows = (0..<40).map { P01CountingVector(row: $0, [Float($0), 0, 0]) }
            let queries = (0..<count).map { P01CountingVector(row: 1000 + $0, [0, 0, 0]) }
            let result = try await Operations.findNearestBatch(queries: queries, in: rows, k: 1, metric: P01FirstCoordinateMetric())
            #expect(result.count == count && result.allSatisfy { $0.map(\.index) == [0] })
            return P01CountingVector.log.readsByRow()
        }
        let one = try await reads(queries: 1)
        let three = try await reads(queries: 3)
        let rowsOne = (1..<40).map { one[$0, default: 0] }
        let rowsThree = (1..<40).map { three[$0, default: 0] }
        p01Marker("P01_OBSERVED batch-validation candidateReadsPerRow Q1=\(Set(rowsOne).sorted()) Q3=\(Set(rowsThree).sorted()) "
                  + "row0 Q1=\(one[0, default: 0]) Q3=\(three[0, default: 0])")
        #expect(rowsOne.allSatisfy { $0 >= 1 }, "every candidate row is validated")
        #expect(rowsThree == rowsOne, """
            candidate scalarCount reads per row grew with the query count: \
            Q=1 \(Set(rowsOne).sorted()), Q=3 \(Set(rowsThree).sorted())
            """)

        // Control: the public single-query call still validates every candidate row.
        P01CountingVector.log.reset()
        let rows = (0..<40).map { P01CountingVector(row: $0, [Float($0), 0, 0]) }
        _ = try await Operations.findNearest(to: P01CountingVector(row: 2000, [0, 0, 0]), in: rows, k: 1,
                                             metric: P01FirstCoordinateMetric())
        let single = P01CountingVector.log.readsByRow()
        #expect((0..<40).allSatisfy { single[$0, default: 0] >= 1 })
    }

    // MARK: Child entry

    /// Returns at once unless the parent selected a scenario.
    @Test func p01Child() async throws {
        guard let scenario = ProcessInfo.processInfo.environment["P01_SCENARIO"] else { return }
        p01Marker("P01_CHILD_BEGIN \(scenario)")
        switch scenario {
        case "operations-cold-small-k": try await coldKey(scenario: scenario, dimension: 17, k: 1)
        case "operations-cold-large-k": try await coldKey(scenario: scenario, dimension: 19, k: 100)
        default:
            Issue.record("Unknown P01 scenario \(scenario)")
        }
        p01Marker("P01_CHILD_COMPLETE \(scenario)")
    }

    /// N = 100 candidates; the calibrator, when it runs, probes M = min(max(32, min(N, 256)), N) = 100 of them.
    private func coldKey(scenario: String, dimension: Int, k: Int) async throws {
        let count = 100, probes = 100
        let wasCold = AutoTuning.get(dim: dimension, kind: .euclid) == nil
        let calls = P01Log(), provider = P01SpyProvider()
        let query = DynamicVector([Float](repeating: 0, count: dimension))
        let rows = (0..<count).map { index -> DynamicVector in
            var values = [Float](repeating: 0, count: dimension); values[0] = Float(index); return DynamicVector(values)
        }
        let result = try await Operations.$computeProvider.withValue(provider) {
            try await Operations.findNearest(to: query, in: rows, k: k, metric: P01SpyMetric(calls: calls))
        }
        let published = AutoTuning.get(dim: dimension, kind: .euclid) != nil
        p01Marker("P01_OBSERVED \(scenario) cold=\(wasCold) N=\(count) k=\(k) calls=\(calls.count) "
                  + "reduceEntries=\(provider.reduceEntries.count) executeEntries=\(provider.executeEntries.count) "
                  + "published=\(published) first=\(result.prefix(3).map(\.index))")
        #expect(wasCold)
        #expect(result.map(\.index) == Array(0..<min(k, count)))
        if k < count / 10 {
            #expect(calls.count == count, "small k evaluated the metric \(calls.count) times; expected exactly \(count)")
            #expect(provider.reduceEntries.count >= 1, "small k did not enter the installed provider's parallelReduce")
            #expect(!published, "small k published an AutoTuning entry for (\(dimension), euclid)")
        } else {
            #expect(calls.count == count + probes, "large-k control: \(calls.count) calls; expected \(count + probes)")
            #expect(published, "large-k control: calibration did not publish")
        }
    }
}

/// Custom metric spy: |a0 - b0|, recording each call. It never reads scalarCount.
private struct P01SpyMetric: DistanceMetric {
    typealias Scalar = Float
    let calls: P01Log
    var name: String { "P01 spy" }
    func distance<V: VectorProtocol>(_ a: V, _ b: V) -> Float where V.Scalar == Float {
        calls.append(Int(b[0]))
        return abs(a[0] - b[0])
    }
}

private struct P01FirstCoordinateMetric: DistanceMetric {
    typealias Scalar = Float
    var name: String { "P01 first coordinate" }
    func distance<V: VectorProtocol>(_ a: V, _ b: V) -> Float where V.Scalar == Float { abs(a[0] - b[0]) }
}

/// A conforming ComputeProvider that records entry and delegates to the sequential CPU provider.
private struct P01SpyProvider: ComputeProvider {
    let inner = CPUComputeProvider(mode: .sequential)
    let reduceEntries = P01Log()
    let executeEntries = P01Log()
    var device: ComputeDevice { inner.device }
    var maxConcurrency: Int { inner.maxConcurrency }
    var deviceInfo: ComputeDeviceInfo { inner.deviceInfo }

    func execute<T: Sendable>(_ work: @Sendable @escaping () async throws -> T) async throws -> T {
        try await inner.execute(work)
    }
    func parallelExecute<T: Sendable>(items: Range<Int>, _ work: @Sendable @escaping (Int) async throws -> T) async throws -> [T] {
        executeEntries.append(items.count)
        return try await inner.parallelExecute(items: items, work)
    }
    func parallelForEach(items: Range<Int>, _ body: @Sendable @escaping (Int) async throws -> Void) async throws {
        try await inner.parallelForEach(items: items, body)
    }
    func parallelForEach(items: Range<Int>, minChunk: Int, _ body: @Sendable @escaping (Int) async throws -> Void) async throws {
        try await inner.parallelForEach(items: items, minChunk: minChunk, body)
    }
    func parallelReduce<R: Sendable>(items: Range<Int>, initial: R, _ rangeWork: @Sendable @escaping (Range<Int>) async throws -> R,
                                     _ combine: @Sendable @escaping (R, R) -> R) async throws -> R {
        reduceEntries.append(items.count)
        return try await inner.parallelReduce(items: items, initial: initial, rangeWork, combine)
    }
    func parallelReduce<R: Sendable>(items: Range<Int>, initial: R, minChunk: Int,
                                     _ rangeWork: @Sendable @escaping (Range<Int>) async throws -> R,
                                     _ combine: @Sendable @escaping (R, R) -> R) async throws -> R {
        reduceEntries.append(items.count)
        return try await inner.parallelReduce(items: items, initial: initial, minChunk: minChunk, rangeWork, combine)
    }
}

/// Lock-protected scalarCount read counts by row.
final class P01ReadLog: @unchecked Sendable {
    private let lock = NSLock()
    private var reads: [Int: Int] = [:]
    func read(_ row: Int) { lock.lock(); defer { lock.unlock() }; reads[row, default: 0] += 1 }
    func reset() { lock.lock(); defer { lock.unlock() }; reads.removeAll() }
    func readsByRow() -> [Int: Int] { lock.lock(); defer { lock.unlock() }; return reads }
}

/// A public-API vector whose `scalarCount` reads are counted per row; `endIndex` does not count.
private struct P01CountingVector: VectorProtocol {
    static let log = P01ReadLog()
    var storage: [Float]
    var row: Int
    var scalarCount: Int { Self.log.read(row); return storage.count }
    var endIndex: Int { storage.count }
    subscript(index: Int) -> Float { storage[index] }
    init() { storage = [0, 0, 0]; row = -1 }
    init(_ array: [Float]) { storage = array; row = -1 }
    init(repeating value: Float) { storage = [value, value, value]; row = -1 }
    init(row: Int, _ array: [Float]) { storage = array; self.row = row }
    func toArray() -> [Float] { storage }
    func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeBufferPointer(body)
    }
    mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeMutableBufferPointer { try body($0) }
    }
}
