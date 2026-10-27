import Foundation
import Testing
@testable import VectorCore

// P02 phase A RED tests (root ruling RD2: D1-D4) for the DEFAULT CPU route of `Operations.findNearestBatch`,
// written before any P02 production edit. Every call here omits the CPU strategy argument, so it exercises whatever
// the default route is. On the unchanged tree (integration 56afcdc) that is the implicit Gram (GEMM) route at
// Q >= 8 and N >= 256, and these tests fail there; once the default is the per-query route (P02 phase C) they must
// pass unchanged.
//
// No test requires a mismatch to occur on any host: each asserts only a property of the correct answer.
// - D1: exact row parity (index and score bits) with the single-query reference `Operations.findNearest` under the
//   same k, metric and provider context.
// - D2: an analytic answer on the API-22 offset-1000 fixture, where every floating-point operation is exact.
// - D3: membership against a Double oracle wherever a derived per-pair error bound separates the k-th and (k+1)-th
//   neighbours, plus the exact-zero self and duplicate distances.
// - D4: the self (or its exact duplicate) ranked first for cosine. This is an OBSERVED property of the fixture under
//   the per-query routes (P02 phase 1 probe A and its phase A re-run: self first in every query at every width on
//   the single-query side), not a derived guarantee; D1 is the exact cosine check.
// Fixtures: P02BatchFixtures.swift.

@Suite("P02 default batch route (RED: D1-D4)")
struct P02DefaultRouteRedTests {

    // MARK: D1

    /// D1: every default-route batch row equals `findNearest` for the same query, bit for bit, at the gate
    /// neighbours Q{7,8} x N{255,256}, for three widths, two metrics and three fixtures.
    @Test func defaultRouteEqualsSingleQuery() async throws {
        try await d1(Vector512Optimized.self, width: 512)
        try await d1(Vector768Optimized.self, width: 768)
        try await d1(Vector1536Optimized.self, width: 1536)
    }

    private func d1<V: VectorProtocol & Sendable>(_: V.Type, width: Int) async throws where V.Scalar == Float {
        for fixture in P02Fixture.allCases {
            let all = try p02Vectors(p02Rows(fixture, width: width), as: V.self)
            for (q, n) in [(7, 255), (7, 256), (8, 255), (8, 256)] {
                let queries = Array(all.prefix(q)), candidates = Array(all.prefix(n))
                let euclidean = try await p02BatchRowsDifferingFromSingleQuery(
                    queries: queries, candidates: candidates, k: 10, metric: EuclideanDistance())
                let euclideanCell = P02Cell(width: width, fixture: fixture, metric: "euclidean", q: q, n: n, k: 10)
                #expect(euclidean.isEmpty, "\(euclideanCell): \(euclidean.count) rows differ, first \(euclidean.prefix(1))")
                let cosine = try await p02BatchRowsDifferingFromSingleQuery(
                    queries: queries, candidates: candidates, k: 10, metric: CosineDistance())
                let cosineCell = P02Cell(width: width, fixture: fixture, metric: "cosine", q: q, n: n, k: 10)
                #expect(cosine.isEmpty, "\(cosineCell): \(cosine.count) rows differ, first \(cosine.prefix(1))")
            }
        }
    }

    // MARK: D2

    /// D2: the API-22 audit fixture. Query `1000 * 1`; candidates `[near, query, near, ...]` with
    /// `near = query + 0.125 * e0`. Default metric (Euclidean), k = 3. Every row must be exactly
    /// `[(1, 0.0), (0, 0.125), (2, 0.125)]`.
    ///
    /// Why exact (derivable from the fixture): 1000.125 = 1000 + 2^-3 needs 13 significant bits, so it is a Float;
    /// every coordinate difference is 0 or exactly 2^-3; its square 2^-6 is exact; a sum of zeros and one 2^-6 is
    /// exact in any order; sqrt(2^-6) = 2^-3 exactly and sqrt(+0) = +0. The two candidates at 0.125 with the
    /// smallest indices are 0 and 2 (equal scores rank by the smaller index).
    @Test func offset1000AnalyticOracle() async throws {
        try await d2(Vector512Optimized.self, width: 512)
        try await d2(Vector768Optimized.self, width: 768)
        try await d2(Vector1536Optimized.self, width: 1536)
    }

    private func d2<V: VectorProtocol & Sendable>(_: V.Type, width: Int) async throws where V.Scalar == Float {
        let rows = try p02Vectors(p02Rows(.offset1000, width: width), as: V.self)
        let query = rows[1]
        let expected = [P01OpsEntry(index: 1, bits: Float(0).bitPattern),
                        P01OpsEntry(index: 0, bits: Float(0.125).bitPattern),
                        P01OpsEntry(index: 2, bits: Float(0.125).bitPattern)]
        for (q, n) in [(7, 255), (7, 256), (8, 255), (8, 256)] {
            let result = try await Operations.findNearestBatch(
                queries: Array(repeating: query, count: q), in: Array(rows.prefix(n)), k: 3)
            let wrong = result.enumerated().filter { p01Entries($0.element) != expected }
            let firstWrong = wrong.first.map { "q\($0.offset) \(p01Entries($0.element))" } ?? "-"
            #expect(result.count == q && wrong.isEmpty,
                    "w\(width) Q\(q) N\(n): \(wrong.count) of \(result.count) rows differ from \(expected); first \(firstWrong)")
        }
    }

    // MARK: D3

    /// D3: the near-duplicate cluster with an exact duplicate (row 255 = row 0), Euclidean, k = 10, N = 256,
    /// Q in {7, 8}.
    /// - Self and duplicate: query i's own row is at distance exactly +0 (every difference is 0), so the first entry
    ///   is `(i, +0)`; for query 0 the second entry is `(255, +0)` (an equal score ranks by the smaller index).
    /// - Membership: let s_j be the squared distances in Double and gamma = gamma_{D+2} = (D+2)u / (1 - (D+2)u),
    ///   u = 2^-24. The per-query Euclidean routes compute the squared score as a Float sum of squared Float
    ///   differences, so the computed score lies within gamma * s_j of s_j: each term is (d(1+e1))^2 (1+e2) with
    ///   |e| <= u, and summing D nonnegative terms in any order adds at most gamma_{D-1}; together gamma_{D+2}
    ///   (no underflow: the squares here are about 1e-6). If the 10th and 11th Double scores satisfy
    ///   s10 (1 + gamma) < s11 (1 - gamma), every true top-10 member scores below every non-member in Float too,
    ///   so the returned members must equal the Double top-10. Queries without that gap are not compared, and the
    ///   test requires the fixture to certify at least half the queries (a property of the data, computed here).
    ///   The Double reference adds a relative error near D * 2^-53, covered by the 1e-9 added to gamma.
    @Test func nearClusterEuclideanMatchesDoubleOracle() async throws {
        try await d3(Vector512Optimized.self, width: 512)
        try await d3(Vector768Optimized.self, width: 768)
        try await d3(Vector1536Optimized.self, width: 1536)
    }

    private func d3<V: VectorProtocol & Sendable>(_: V.Type, width: Int) async throws where V.Scalar == Float {
        let data = p02Rows(.near, width: width)
        let all = try p02Vectors(data, as: V.self)
        let n = 256, k = 10
        let m = Double(width + 2), u = 0x1p-24
        let gamma = m * u / (1 - m * u) + 1e-9
        for q in [7, 8] {
            let result = try await Operations.findNearestBatch(queries: Array(all.prefix(q)), in: all, k: k)
            var problems: [String] = []
            var certified = 0
            for i in 0..<q {
                let row = p01Entries(result[i])
                if row.first != P01OpsEntry(index: i, bits: Float(0).bitPattern) {
                    problems.append("q\(i) first \(row.first.map(\.description) ?? "-") (expected \(i):0.0)")
                }
                if i == 0, row.count > 1, row[1] != P01OpsEntry(index: n - 1, bits: Float(0).bitPattern) {
                    problems.append("q0 second \(row[1]) (expected \(n - 1):0.0)")
                }
                let s: [Double] = (0..<n).map { j in
                    var sum = 0.0
                    for t in 0..<width { let d = Double(data[i][t]) - Double(data[j][t]); sum += d * d }
                    return sum
                }
                let order = (0..<n).sorted { s[$0] != s[$1] ? s[$0] < s[$1] : $0 < $1 }
                if s[order[k - 1]] * (1 + gamma) < s[order[k]] * (1 - gamma) {
                    certified += 1
                    let expected = Set(order.prefix(k)), got = Set(row.map(\.index))
                    if got != expected {
                        problems.append("q\(i) members +\(got.subtracting(expected).sorted()) -\(expected.subtracting(got).sorted())")
                    }
                }
            }
            #expect(problems.isEmpty, "w\(width) Q\(q): \(problems.count) problems: \(problems.prefix(3))")
            #expect(certified * 2 >= q, "w\(width) Q\(q): the fixture certifies only \(certified) of \(q) queries")
        }
    }

    // MARK: D4

    /// D4: the same near-duplicate cluster under cosine, k = 10, N = 256, Q in {7, 8}: each query's first entry is
    /// its own row (for query 0, rows 0 and 255 are bitwise identical, so their scores are equal and index 0 ranks
    /// first). Observed for the per-query routes, not derived (see the file header).
    @Test func nearClusterCosineRanksSelfFirst() async throws {
        try await d4(Vector512Optimized.self, width: 512)
        try await d4(Vector768Optimized.self, width: 768)
        try await d4(Vector1536Optimized.self, width: 1536)
    }

    private func d4<V: VectorProtocol & Sendable>(_: V.Type, width: Int) async throws where V.Scalar == Float {
        let all = try p02Vectors(p02Rows(.near, width: width), as: V.self)
        for q in [7, 8] {
            let result = try await Operations.findNearestBatch(
                queries: Array(all.prefix(q)), in: all, k: 10, metric: CosineDistance())
            let firsts = result.map { $0.first?.index ?? -1 }
            #expect(firsts == Array(0..<q), "w\(width) Q\(q): first indices \(firsts)")
        }
    }
}
