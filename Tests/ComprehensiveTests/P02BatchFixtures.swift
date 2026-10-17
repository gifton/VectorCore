import Foundation
import Testing
@testable import VectorCore

// P02 shared fixtures for `P02DefaultRouteRedTests` and `P02BatchProviderTests` (phase A, written before any P02
// production edit).
//
// The "near" and "rand" rows reproduce P02 phase 1 probe A exactly (same generator, seeds and construction), so the
// RED transcript of these tests can be compared with that probe's retained output
// (`.stabilization-recovery/2026-10-03/P02/phase1/probes/run-route.log` and its phase A re-run). The "offset1000" rows
// are the audit fixture of API-22: query `1000 * 1`, candidates `[near, query, near, near, ...]` with
// `near = query + 0.125 * e0`.

/// SplitMix64, as in P02 phase 1 probe A.
struct P02SplitMix {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    /// Uniform in [-1, 1) from the top 53 bits, rounded to Float.
    mutating func unit() -> Float { Float(Double(next() >> 11) / Double(1 << 53) * 2 - 1) }
}

enum P02Fixture: String, CaseIterable, Sendable {
    /// A common base in [-1, 1) plus 1e-3 uniform noise per coordinate (the T14 phase 2 review cluster).
    case near
    /// Uniform [-1, 1).
    case rand
    /// The API-22 audit fixture: row 1 is `1000 * 1`; every other row is that plus 0.125 in coordinate 0.
    case offset1000
}

/// 256 rows of `width` Floats. For `near` and `rand`, row 255 is an exact copy of row 0 (so it is present only when
/// all 256 rows are used) and the seed is `0xC0FFEE + width`, as in phase 1 probe A.
func p02Rows(_ fixture: P02Fixture, width: Int) -> [[Float]] {
    let count = 256
    switch fixture {
    case .offset1000:
        let one = [Float](repeating: 1000, count: width)
        var near = one
        near[0] += 0.125
        return (0..<count).map { $0 == 1 ? one : near }
    case .near:
        var random = P02SplitMix(state: 0xC0FFEE &+ UInt64(width))
        let base = (0..<width).map { _ in random.unit() }
        var rows = (0..<count).map { _ in base.map { $0 + random.unit() * 1e-3 } }
        rows[count - 1] = rows[0]
        return rows
    case .rand:
        var random = P02SplitMix(state: 0xC0FFEE &+ UInt64(width))
        var rows = (0..<count).map { _ in (0..<width).map { _ in random.unit() } }
        rows[count - 1] = rows[0]
        return rows
    }
}

func p02Vectors<V: VectorProtocol>(_ rows: [[Float]], as _: V.Type) throws -> [V] where V.Scalar == Float {
    try rows.map { try V($0) }
}

/// One cell label for failure messages.
struct P02Cell: CustomStringConvertible, Sendable {
    let width: Int
    let fixture: P02Fixture
    let metric: String
    let q: Int
    let n: Int
    let k: Int
    var description: String { "w\(width) \(fixture.rawValue) \(metric) Q\(q) N\(n) k\(k)" }
}

/// Rows of `findNearestBatch` compared with `findNearest` of the same query, candidates, k, metric and provider
/// context (the task-local `Operations.computeProvider` of the caller), by index and exact score bits. Returns
/// one description per differing row (empty when every row matches).
func p02BatchRowsDifferingFromSingleQuery<V: VectorProtocol & Sendable, M: DistanceMetric>(
    queries: [V], candidates: [V], k: Int, metric: M
) async throws -> [String] where V.Scalar == Float, M.Scalar == Float {
    let batch = try await Operations.findNearestBatch(queries: queries, in: candidates, k: k, metric: metric)
    var differing: [String] = []
    if batch.count != queries.count { differing.append("row count \(batch.count) != \(queries.count)") }
    for i in 0..<min(batch.count, queries.count) {
        let single = try await Operations.findNearest(to: queries[i], in: candidates, k: k, metric: metric)
        let b = p01Entries(batch[i]), s = p01Entries(single)
        if b != s {
            differing.append("q\(i) batch\(Array(b.prefix(3))) single\(Array(s.prefix(3)))")
        }
    }
    return differing
}
