import Foundation
import Testing
@testable import VectorCore

// P02 phase C, RD1: the derived worst-case bound of the `.gram` scores (documented on `BatchSearchCPUStrategy.gram`),
// checked entry by entry against an independent Double reference on the scores the seam copies from the actual tiles.
//
// With u = 2^-24 and γ_n = n·u / (1 − n·u) (computed in Double):
// - Euclidean squared score ĝ: |ĝ − Σ(x_k − y_k)²| ≤ γ_{D+2}·(‖x‖² + ‖y‖² + 2·Σ|x_k·y_k|) (T11's MatrixDistance count:
//   each of the three terms passes through at most D + 2 binary32 roundings; the clamp at 0 only moves toward the
//   true value).
// - Cosine distance ĉ, with η = 2^-24 + 2^-50 the per-coordinate relative error of NormalizationCore's Float result
//   (a Kahan-compensated Double norm, a Double division, then one Float rounding): |ĉ − c| ≤ 2η + η² + γ_D·(1 + η)² +
//   u·(1 + (1 + γ_D)·(1 + η)²) + D·2^-140 (the normalization error of the exact products, the inner product's γ_D,
//   the rounding of 1 − dot, and a term for subnormal coordinates and products).
// Both rest on the unverified Accelerate contract stated in the documentation (binary32 inner products in some order,
// no fast matrix multiplication), and the Euclidean one on its domain hypothesis (no binary32 overflow and no product
// underflow; every fixture below is inside it). `underflowedSquaresScoreZero` and
// `overflowedNormsGiveNaNForAnExactDuplicate` pin what happens outside that domain. The Double references add a slack
// of (D + 4)·2^-52 relative to their own magnitude.
// The largest measured error as a fraction of its bound is printed as a `P02_ACCURACY` line (one host; a record, not
// an assertion).

private let p02U = 0x1p-24
private func p02Gamma(_ n: Int) -> Double { Double(n) * p02U / (1 - Double(n) * p02U) }

@Suite("P02 Gram accuracy (RD1)")
struct P02GramAccuracyTests {
    static let geometries: [(r: Int, c: Int)] = [(1, 1), (3, 7), (8, 256), (Int.max, Int.max)]

    /// Fixtures: the near-duplicate cluster (row 255 duplicates row 0), uniform rows (row 255 duplicates row 0), the
    /// offset-1000 rows (row 1 duplicates no other; near rows are exact duplicates of each other) and unit-norm rows.
    static func fixtures(width: Int) -> [(String, [[Float]])] {
        let unit = p02RandomRows(256, width: width, seed: 0x0417).map { row -> [Float] in
            let norm = row.reduce(0.0) { $0 + Double($1) * Double($1) }.squareRoot()
            return row.map { Float(Double($0) / norm) }
        }
        return [("near", p02Rows(.near, width: width)), ("rand", p02Rows(.rand, width: width)),
                ("offset1000", p02Rows(.offset1000, width: width)), ("unit", unit)]
    }

    @Test func euclideanScoresWithinTheDerivedBound() async throws {
        try await p02ForEachWidth { type in try await accuracy(type, euclidean: true) }
    }

    @Test func cosineScoresWithinTheDerivedBound() async throws {
        try await p02ForEachWidth { type in try await accuracy(type, euclidean: false) }
    }

    private func accuracy<V: P02GramWidth>(_ type: V.Type, euclidean: Bool) async throws {
        let d = V.p02Width
        var worst = 0.0, worstLabel = ""
        var checked = 0
        for (name, values) in Self.fixtures(width: d) {
            let candidates = try V.rows(values)
            let queries = Array(candidates.prefix(8))
            let x = values.map { $0.map(Double.init) }
            // Independent Double references and bounds for the 8 x 256 pairs.
            var reference = [[Double]](), bound = [[Double]]()
            for q in 0..<8 {
                var refRow = [Double](), boundRow = [Double]()
                for c in 0..<values.count {
                    var squared = 0.0, normQ = 0.0, normC = 0.0, dot = 0.0, absDot = 0.0
                    for i in 0..<d {
                        let a = x[q][i], b = x[c][i]
                        squared += (a - b) * (a - b); normQ += a * a; normC += b * b; dot += a * b; absDot += abs(a * b)
                    }
                    let slack = Double(d + 4) * 0x1p-52
                    if euclidean {
                        refRow.append(squared)
                        boundRow.append(p02Gamma(d + 2) * (normQ + normC + 2 * absDot) + slack * (normQ + normC + 2 * absDot))
                    } else {
                        let eta = 0x1p-24 + 0x1p-50, g = p02Gamma(d)
                        refRow.append(1 - dot / (normQ * normC).squareRoot())
                        boundRow.append(2 * eta + eta * eta + g * (1 + eta) * (1 + eta)
                                        + p02U * (1 + (1 + g) * (1 + eta) * (1 + eta)) + Double(d) * 0x1p-140 + slack)
                    }
                }
                reference.append(refRow)
                bound.append(boundRow)
            }
            for (r, c) in Self.geometries {
                let (result, log) = await p02TracedMetric(queries, candidates, k: 10, euclidean: euclidean,
                                                          strategy: p02Gram(r, c))
                _ = try result.get()
                var seen = 0
                for tile in log.tiles {
                    for (i, q) in tile.queryRows.enumerated() {
                        for (j, cand) in tile.candidateRows.enumerated() {
                            let score = Double(tile.scores[i * tile.n + j])
                            let error = abs(score - reference[q][cand])
                            let label = "w\(d) \(name) \(euclidean ? "euclid" : "cosine") r\(r) c\(c) (\(q), \(cand))"
                            #expect(error <= bound[q][cand], "\(label): |\(score) - \(reference[q][cand])| > \(bound[q][cand])")
                            if error / bound[q][cand] > worst { worst = error / bound[q][cand]; worstLabel = label }
                            seen += 1
                        }
                    }
                }
                #expect(seen == 8 * values.count, "w\(d) \(name) r\(r) c\(c): \(seen) entries")
                checked += seen
            }
        }
        print("P02_ACCURACY \(euclidean ? "euclidean" : "cosine") width=\(d) entries=\(checked) "
              + "maxErrorOverBound=\(worst) at \(worstLabel)")
    }

    // MARK: Outside the bound's domain (P02 phase C review RA-2; documented on `BatchSearchCPUStrategy.gram`)

    /// Underflow: x = 512 coordinates of 1e-30 against a zero row. Every x_k² (about 1e-60) is below the smallest
    /// binary32 subnormal, so ‖x‖² rounds to 0, the cross term is 0 and the `.gram` squared score is exactly 0, while
    /// the true squared distance (Double) is about 5.1e-58: the stated bound (about 1.6e-62 here) does not hold
    /// outside its no-underflow hypothesis. The returned distance is 0.
    @Test func underflowedSquaresScoreZero() async throws {
        let x = try Vector512Optimized([Float](repeating: 1e-30, count: 512))
        let zero = try Vector512Optimized([Float](repeating: 0, count: 512))
        let (result, log) = await p02Traced([x], [zero], k: 1, metric: EuclideanDistance(), strategy: p02Gram(1, 1))
        let rows = try result.get()
        let score = try #require(log.tiles.first?.scores.first)
        let coordinate = Double(Float(1e-30))
        let truth = 512 * coordinate * coordinate
        let bound = p02Gamma(514) * truth   // ‖x‖² + ‖0‖² + 2·Σ|x_k·0| = ‖x‖² in exact arithmetic
        #expect(score == 0, "squared score \(score)")
        #expect(truth > 5e-58 && abs(Double(score) - truth) > bound, "truth \(truth), bound \(bound)")
        #expect(rows.map(p01Entries) == [[P01OpsEntry(index: 0, bits: Float(0).bitPattern)]])
    }

    /// Overflow: x = 512 coordinates of 1e20 (finite), candidates [zero row, x]. In binary32, every x_k² and x_k·x_k
    /// (1e40) overflows to +∞ in any summation order, so ‖x‖² = +∞ and the cross term −2·x·x = −∞. The zero row then
    /// scores (−0 + ∞) + 0 = +∞, and the exact duplicate (−∞ + ∞) + ∞ = NaN, which the NaN-last order ranks LAST.
    /// `.perQuery` (the per-pair difference route) returns 0 for the duplicate and ranks it first.
    @Test func overflowedNormsGiveNaNForAnExactDuplicate() async throws {
        let x = try Vector512Optimized([Float](repeating: 1e20, count: 512))
        let zero = try Vector512Optimized([Float](repeating: 0, count: 512))
        let (gram, log) = await p02Traced([x], [zero, x], k: 2, metric: EuclideanDistance(), strategy: p02Gram(1, 2))
        let gramRows = try gram.get()
        let gramRow = try #require(gramRows.first)
        let tile = try #require(log.tiles.first)
        #expect(tile.scores.count == 2 && tile.scores[0] == .infinity && tile.scores[1].isNaN, "tile \(tile.scores)")
        #expect(gramRow.map(\.index) == [0, 1], "gram \(gramRow)")
        #expect(gramRow.first?.distance == .infinity && gramRow.last?.distance.isNaN == true, "gram \(gramRow)")
        let perQuery = try await Operations.findNearestBatch(queries: [x], in: [zero, x], k: 2)
        #expect(perQuery.first?.map(\.index) == [1, 0], "perQuery \(perQuery)")
        #expect(perQuery.first?.first?.distance == 0 && perQuery.first?.last?.distance == .infinity, "perQuery \(perQuery)")
    }
}
