import Foundation
import Testing
@testable import VectorCore

// P02 phase C: the `.perQuery` default of `Operations.findNearestBatch` (phase 1 check map S2, P10-P15, E5, E7, X10).
// Every row is compared with public single-query `findNearest` under the same k, metric and provider, by index and
// exact score bits (no tolerance).

/// A public-API vector whose `scalarCount` reads are counted per row (the P01 batch-validation probe's shape).
private struct P02CountingVector: VectorProtocol {
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

private struct P02FirstCoordinateMetric: DistanceMetric {
    typealias Scalar = Float
    var name: String { "P02 first coordinate" }
    func distance<V: VectorProtocol>(_ a: V, _ b: V) -> Float where V.Scalar == Float { abs(a[0] - b[0]) }
}

/// Rows for the route-family matrix: the exceptional rows (zero, NaN, ±inf, all-NaN, duplicates), an all -0 row (a
/// ±0 tie with the zero row for the zero query), and a root collision for the zero query (squared distances 1 and
/// 1 + 2^-23, both with root 1).
private func p02FamilyRows(_ count: Int, width: Int) -> [[Float]] {
    var rows = p02ExceptionalRows(count, width: width, seed: 0x5EED)
    rows[8] = [Float](repeating: 0, count: width)
    rows[8][0] = 1
    rows[8][1] = 1.25 * 0x1p-12
    rows[9] = [Float](repeating: 0, count: width)
    rows[9][0] = 1
    rows[10] = [Float](repeating: -0.0, count: width)
    return rows
}

@Suite("P02 per-query default route")
struct P02PerQueryRouteTests {

    /// P14, P15, E5, E7: default rows equal `findNearest` bit for bit across the route families: k in {1, N - 1, N,
    /// N + 1, Int.max} (direct optimized k < N, generic small-k stream and full-score k >= N), metrics Euclidean,
    /// cosine (fused 512 and the 768/1536 full-score route), dot product, Manhattan and a custom witness, at Q7/N255
    /// and Q8/N256 on all three widths and on DynamicVector, with NaN, ±0, duplicate and root-collision rows.
    @Test func routeFamiliesMatchSingleQuery() async throws {
        try await family(Vector512Optimized.self, width: 512)
        try await family(Vector768Optimized.self, width: 768)
        try await family(Vector1536Optimized.self, width: 1536)
        try await family(DynamicVector.self, width: 512)
    }

    private func family<V: VectorProtocol & Sendable>(_: V.Type, width: Int) async throws where V.Scalar == Float {
        let all = try p02Vectors(p02FamilyRows(256, width: width), as: V.self)
        for (q, n) in [(7, 255), (8, 256)] {
            let candidates = Array(all.prefix(n)), queries = Array(all.prefix(q))
            for k in [1, n - 1, n, n + 1, Int.max] {
                let label = "\(V.self) Q\(q) N\(n) k\(k)"
                try await expectSame(queries, candidates, k: k, EuclideanDistance(), "\(label) euclid")
                try await expectSame(queries, candidates, k: k, CosineDistance(), "\(label) cosine")
                try await expectSame(queries, candidates, k: k, DotProductDistance(), "\(label) dot")
                try await expectSame(queries, candidates, k: k, ManhattanDistance(), "\(label) manhattan")
                try await expectSame(queries, candidates, k: k, P02CountingMetric(calls: .init()), "\(label) custom")
            }
        }
    }

    private func expectSame<V: VectorProtocol & Sendable, M: DistanceMetric>(
        _ queries: [V], _ candidates: [V], k: Int, _ metric: M, _ label: String
    ) async throws where V.Scalar == Float, M.Scalar == Float {
        let differing = try await p02BatchRowsDifferingFromSingleQuery(queries: queries, candidates: candidates, k: k,
                                                                       metric: metric)
        #expect(differing.isEmpty, "\(label): \(differing.count) rows differ, first \(differing.prefix(1))")
    }

    /// S2, P10, P11: the default validates every candidate row once for the whole batch, so per-row `scalarCount`
    /// reads at Q8/N256 equal those at Q1 (a per-query public `findNearest` would add one read per row per query).
    @Test func candidateValidationDoesNotRepeatAtQ8N256() async throws {
        func reads(queries count: Int) async throws -> [Int: Int] {
            P02CountingVector.log.reset()
            let rows = (0..<256).map { P02CountingVector(row: $0, [Float($0), 0, 0]) }
            let queries = (0..<count).map { P02CountingVector(row: 1000 + $0, [0, 0, 0]) }
            let result = try await Operations.findNearestBatch(queries: queries, in: rows, k: 1,
                                                               metric: P02FirstCoordinateMetric())
            #expect(result.count == count && result.allSatisfy { $0.map(\.index) == [0] })
            return P02CountingVector.log.readsByRow()
        }
        let one = try await reads(queries: 1)
        let eight = try await reads(queries: 8)
        let rowsOne = (1..<256).map { one[$0, default: 0] }
        let rowsEight = (1..<256).map { eight[$0, default: 0] }
        #expect(rowsOne.allSatisfy { $0 >= 1 }, "every candidate row is validated")
        #expect(rowsEight == rowsOne, "reads per row grew with the query count: Q1 \(Set(rowsOne)), Q8 \(Set(rowsEight))")
    }

    /// X10: the default keeps the providers' cancellation law: in an already-cancelled task, at Q8/N256 with a custom
    /// metric, each built-in CPU provider throws `CancellationError` before any metric call.
    @Test func preCancelledDefaultThrowsBeforeAnyMetricCall() async throws {
        let all = try p02Vectors(p02Rows(.rand, width: 512), as: Vector512Optimized.self)
        let queries = Array(all.prefix(8))
        for (label, provider) in [("sequential", CPUComputeProvider(mode: .sequential)),
                                  ("parallel", CPUComputeProvider(mode: .parallel)),
                                  ("automatic", CPUComputeProvider(mode: .automatic))] {
            let calls = P02CountingMetric.Calls()
            let metric = P02CountingMetric(calls: calls)
            let error = await p02AnyErrorOf(cancelled: true) {
                _ = try await Operations.$computeProvider.withValue(provider) {
                    try await Operations.findNearestBatch(queries: queries, in: all, k: 10, metric: metric)
                }
            }
            #expect(error is CancellationError, "\(label): \(String(describing: error))")
            #expect(calls.count == 0, "\(label): \(calls.count) metric calls")
        }
    }
}
