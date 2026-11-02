import Foundation
import Testing
@testable import VectorCore

// T19 (L04 STOP S1): exact PCA and LAPACK svdThin threw "sgesdd returned an invalid workspace query" on sgesdd path 5
// (5t when wide): R < floor(11r/6) with r = min(m, n), R = max(m, n). vc_la_query rejected Accelerate's positive query
// answer because it was below the REFERENCE sufficient bound 4r^2 + 7r; after T19 that bound is an allocation floor
// only. The sweeps cross the path-5 boundary on both sides, tall and wide, through the public routes (the default
// provider's svdThin, and PCAModel.fit with oversampling Int.max), plus L04's reported cells, two sketch-path PCA
// fits whose wide QᵀA panel is on path 5t, and near-boundary shapes of the other vc_la_query callers
// (symmetricEigen: ssyevd; qrThin: sgeqrf and sorgqr).
//
// Reference: every SVD and eigen input has a designed spectrum, A = H1·S·H2 built in Double from Householder
// reflections, so expected values never come from a provider. Tolerance per case: min(2e-3, gap/4), gap = the
// designed spacing (1/r for singular values, 1/n for eigenvalues), so a value shifted by one index fails; designed
// values lie in [1, 2] and the smallest tolerance is 2.5e-4 (r = 1000). Rounding A to Float moves each value by at
// most ||dA||_F <= 2^-24·||A||_F <= 2^-24·2·sqrt(1000) < 4e-6 (Weyl); the rest is the backend's backward error,
// which the tolerance bounds for these shapes rather than derives. Each test prints one
// `T19CELLS<TAB>route<TAB>failing/total<TAB>cells<TAB>worst=<largest error/tolerance of a passing cell>` line, so a run
// log is the per-cell RED/GREEN record; a cell id ends in `p5` when the shape is on sgesdd path 5. Synchronous @Test
// functions only (Xcode 27 drops failures of nonisolated async XCTest methods).

/// One cell: the first failure, or nil and the worst |computed - designed| / tolerance (QR: residual ratio).
struct T19Outcome {
    var failure: String?
    var worst = 0.0
    init(failure: String) { self.failure = failure }
    init(worst: Double) { self.worst = worst }
}

enum T19Design {
    /// min(2e-3, gap/4) with gap the smallest spacing of `designed` (2e-3 for a single value).
    static func tolerance(_ designed: [Double]) -> Double {
        var gap = Double.infinity
        for i in designed.indices.dropLast() { gap = min(gap, abs(designed[i] - designed[i + 1])) }
        return min(2e-3, gap / 4)
    }

    /// Designed singular values: 2 - i/r for i < nonzero (distinct, descending, in (1, 2]), then 0.
    static func sigma(_ r: Int, nonzero: Int) -> [Double] {
        (0..<r).map { $0 < nonzero ? 2 - Double($0) / Double(r) : 0 }
    }

    /// Column-major m×n A = H1·S·H2, S = diag(sigma). H1 reflects along v = 1 - sqrt(m)·e_(m-1), which maps the
    /// all-ones vector to sqrt(m)·e_(m-1): when sigma is 0 from index m - 1 on, every column of A sums to 0 (in real
    /// arithmetic), so PCA's centering keeps the designed spectrum. H2 reflects along a seeded Gaussian vector.
    static func matrix(rows m: Int, columns n: Int, sigma: [Double], seed: UInt64) -> [Float] {
        precondition(m >= 2 && n >= 1 && sigma.count == min(m, n))
        var gaussian = GaussianSource(seed: seed)
        let w = (0..<n).map { _ in Double(gaussian.next()) }
        let ww = w.reduce(0) { $0 + $1 * $1 }
        var b = [Double](repeating: 0, count: m * n)   // B = S·H2, column-major; rows >= min(m, n) stay 0
        for i in sigma.indices where sigma[i] != 0 {
            for j in 0..<n { b[j * m + i] = sigma[i] * ((i == j ? 1 : 0) - 2 * w[i] * w[j] / ww) }
        }
        let root = Double(m).squareRoot()
        let vv = 2 * (Double(m) - root)                // vᵀv = (m - 1) + (1 - sqrt(m))^2
        var a = [Float](repeating: 0, count: m * n)
        for j in 0..<n {
            var dot = 0.0
            for i in 0..<m { dot += (i == m - 1 ? 1 - root : 1) * b[j * m + i] }
            let scale = 2 * dot / vv                   // A = B - v·(2·vᵀB / vᵀv)
            for i in 0..<m { a[j * m + i] = Float(b[j * m + i] - scale * (i == m - 1 ? 1 - root : 1)) }
        }
        return a
    }

    /// Symmetric n×n H·diag(lambda)·H, H = I - c·u·uᵀ (c = 2/uᵀu, u seeded Gaussian); lambda_i = 1 + (i + 1)/n,
    /// ascending and distinct in (1, 2]. Entry (i, j): lambda_i·δij - c·u_i·u_j·(lambda_i + lambda_j) + c²·s·u_i·u_j
    /// with s = Σ lambda_k·u_k².
    static func symmetric(_ n: Int, seed: UInt64) -> (a: [Float], lambda: [Double]) {
        var gaussian = GaussianSource(seed: seed)
        let u = (0..<n).map { _ in Double(gaussian.next()) }
        let lambda = (0..<n).map { 1 + Double($0 + 1) / Double(n) }
        let c = 2 / u.reduce(0) { $0 + $1 * $1 }
        let s = zip(lambda, u).reduce(0) { $0 + $1.0 * $1.1 * $1.1 }
        var a = [Float](repeating: 0, count: n * n)
        for j in 0..<n {
            for i in 0..<n {
                let diagonal = i == j ? lambda[i] : 0
                a[j * n + i] = Float(diagonal - c * u[i] * u[j] * (lambda[i] + lambda[j]) + c * c * s * u[i] * u[j])
            }
        }
        return (a, lambda)
    }

    /// The first failing value (`value[i] got vs designed (tolerance t)`), or the worst error / tolerance. NaN fails.
    static func compare(_ computed: [Float], _ designed: [Double]) -> T19Outcome {
        guard computed.count >= designed.count else {
            return T19Outcome(failure: "count \(computed.count) < \(designed.count)")
        }
        let limit = tolerance(designed)
        var worst = 0.0
        for i in designed.indices {
            let error = abs(Double(computed[i]) - designed[i])
            guard error <= limit else {
                return T19Outcome(failure: "value[\(i)] \(computed[i]) vs \(designed[i]) (tolerance \(limit))")
            }
            worst = max(worst, error / limit)
        }
        return T19Outcome(worst: worst)
    }
}

enum T19Cells {
    /// For r in {8, 16, 64}, every R in r...2r, tall (R×r) and wide (r×R). floor(11r/6) is 14, 29 and 117, inside
    /// each range, so every r has path-5 cells and QR-first cells in both orientations (179 cells).
    static let sweep: [(rows: Int, columns: Int)] = {
        var shapes: [(rows: Int, columns: Int)] = []
        for r in [8, 16, 64] {
            shapes.append((rows: r, columns: r))
            for big in (r + 1)...(2 * r) { shapes += [(rows: big, columns: r), (rows: r, columns: big)] }
        }
        return shapes
    }()

    /// L04 S1's reported PCA cells (samples × dimension); all three are on path 5.
    static let reported: [(rows: Int, columns: Int)] = [(500, 384), (500, 768), (1000, 1536)]

    static func id(_ rows: Int, _ columns: Int) -> String {
        let r = min(rows, columns), big = max(rows, columns)
        return "\(rows)x\(columns)\(big < 11 * r / 6 ? "p5" : "")"
    }

    static func seed(_ rows: Int, _ columns: Int) -> UInt64 { UInt64(rows) &* 10_007 &+ UInt64(columns) }

    /// svdThin on the default provider: the designed full-rank spectrum, failure nil on success.
    static func svd(_ rows: Int, _ columns: Int) -> T19Outcome {
        let designed = T19Design.sigma(min(rows, columns), nonzero: min(rows, columns))
        let a = T19Design.matrix(rows: rows, columns: columns, sigma: designed, seed: seed(rows, columns))
        do {
            let svd = try Operations.linearAlgebraProvider.svdThin(a, rows: rows, columns: columns)
            return T19Design.compare(svd.singularValues, designed)
        } catch {
            return T19Outcome(failure: "threw \(error)")
        }
    }

    /// Exact PCA (oversampling Int.max) of n samples in dimension d, k = min(n - 1, d) components; each singular
    /// value is recovered as sqrt(explainedVariance · (n - 1)). failure nil on success.
    static func pca(_ n: Int, _ d: Int) -> T19Outcome {
        let k = min(n - 1, d)
        let designed = T19Design.sigma(min(n, d), nonzero: k)
        let flat = T19Design.matrix(rows: n, columns: d, sigma: designed, seed: seed(n, d))
        let vectors = (0..<n).map { i in DynamicVector((0..<d).map { j in flat[j * n + i] }) }
        do {
            let model = try PCAModel.fit(vectors, components: k, config: PCAConfig(oversampling: .max))
            let recovered = model.explainedVariance.map { Float((Double($0) * Double(n - 1)).squareRoot()) }
            return T19Design.compare(recovered, Array(designed.prefix(k)))
        } catch {
            return T19Outcome(failure: "threw \(error)")
        }
    }

    /// Sketch-path PCA (oversampling p, default power iterations): the final SVD is on the ℓ×d panel QᵀA, ℓ = k + p.
    /// The data has rank ℓ - 5 < ℓ, so in real arithmetic range(Y) = range(A), QQᵀA = A and the panel keeps A's
    /// nonzero singular values; the top k are the designed ones. failure nil on success.
    static func pcaSketch(_ n: Int, _ d: Int, components k: Int, oversampling p: Int) -> T19Outcome {
        let designed = T19Design.sigma(min(n, d), nonzero: k + p - 5)
        let flat = T19Design.matrix(rows: n, columns: d, sigma: designed, seed: seed(n, d))
        let vectors = (0..<n).map { i in DynamicVector((0..<d).map { j in flat[j * n + i] }) }
        do {
            let model = try PCAModel.fit(vectors, components: k, config: PCAConfig(oversampling: p))
            let recovered = model.explainedVariance.map { Float((Double($0) * Double(n - 1)).squareRoot()) }
            return T19Design.compare(recovered, Array(designed.prefix(k)))
        } catch {
            return T19Outcome(failure: "threw \(error)")
        }
    }

    /// Runs `cell` over `shapes`; prints the T19CELLS line and returns the failures (`id reason`).
    static func run(_ route: String, _ shapes: [(rows: Int, columns: Int)],
                    _ cell: (Int, Int) -> T19Outcome) -> [String] {
        var failing: [String] = []
        var worst = 0.0
        for shape in shapes {
            let outcome = cell(shape.rows, shape.columns)
            if let failure = outcome.failure {
                failing.append("\(id(shape.rows, shape.columns)) \(failure)")
            } else {
                worst = max(worst, outcome.worst)
            }
        }
        report(route, failing, of: shapes.count, worst: worst)
        return failing
    }

    /// Prints `T19CELLS<TAB>route<TAB>failing/total<TAB>cells<TAB>worst=<largest error/tolerance of a passing cell>`.
    static func report(_ route: String, _ failing: [String], of total: Int, worst: Double) {
        let cells = failing.map { String($0.split(separator: " ", maxSplits: 1)[0]) }
        print("T19CELLS\t\(route)\t\(failing.count)/\(total)\t\(cells.joined(separator: ","))\tworst=\(worst)")
    }
}

enum T19OtherCallers {
    /// ssyevd: n = 1 is the shim's special case (both minimums 1), n = 2 the first general size; 1...34 also crosses
    /// the reference SMLSIZ (25) and SYTRD crossover (32) defaults, plus the sizes around 64 and 128.
    static let eigenSizes = Array(1...34) + [63, 64, 65, 127, 128, 129]

    /// symmetricEigen on the default provider against the designed eigenvalues; failure nil on success.
    static func eigen(_ n: Int, vectors: Bool) -> T19Outcome {
        let (a, lambda) = T19Design.symmetric(n, seed: UInt64(n) &* 7_919)
        do {
            let result = try Operations.linearAlgebraProvider.symmetricEigen(a, dimension: n, computeEigenvectors: vectors)
            return T19Design.compare(result.eigenvalues, lambda)
        } catch {
            return T19Outcome(failure: "threw \(error)")
        }
    }

    /// sgeqrf/sorgqr: n = 1 meets the shim's max(n, 1) floor; 32/33 and 128/129 straddle the reference blocking
    /// defaults (NB 32, NX 128); m in {n, n + 1, 2n}.
    static let qrShapes: [(rows: Int, columns: Int)] = {
        var shapes: [(rows: Int, columns: Int)] = []
        for n in [1, 2, 3, 31, 32, 33, 127, 128, 129] {
            for m in Set([n, n + 1, 2 * n]).sorted() { shapes.append((rows: m, columns: n)) }
        }
        return shapes
    }()

    /// qrThin on the default provider, checked in Double: max|QR - A| <= 1e-4·max|A|, max|QᵀQ - I| <= 1e-4 and R
    /// upper triangular. failure nil on success.
    static func qr(_ m: Int, _ n: Int) -> T19Outcome {
        var gaussian = GaussianSource(seed: UInt64(m) &* 104_729 &+ UInt64(n))
        let a = (0..<(m * n)).map { _ in gaussian.next() }
        let scale = Double(a.map { abs($0) }.max() ?? 1)
        do {
            let qr = try Operations.linearAlgebraProvider.qrThin(a, rows: m, columns: n)
            var residual = 0.0, orthogonality = 0.0, lower = 0.0
            for j in 0..<n {
                for i in 0..<m {
                    var sum = 0.0
                    for p in 0...j { sum += Double(qr.q[p * m + i]) * Double(qr.r[j * n + p]) }
                    residual = max(residual, abs(sum - Double(a[j * m + i])))
                }
                for i in 0..<n {
                    var dot = 0.0
                    for p in 0..<m { dot += Double(qr.q[i * m + p]) * Double(qr.q[j * m + p]) }
                    orthogonality = max(orthogonality, abs(dot - (i == j ? 1 : 0)))
                    if i > j { lower = max(lower, abs(Double(qr.r[j * n + i]))) }
                }
            }
            // Swift's max drops a NaN operand, so finiteness is checked separately.
            guard qr.q.allSatisfy(\.isFinite), qr.r.allSatisfy(\.isFinite),
                  residual <= 1e-4 * scale, orthogonality <= 1e-4, lower == 0 else {
                return T19Outcome(failure: "residual \(residual) orthogonality \(orthogonality) lower \(lower)")
            }
            return T19Outcome(worst: max(residual / (1e-4 * scale), orthogonality / 1e-4))
        } catch {
            return T19Outcome(failure: "threw \(error)")
        }
    }
}

@Suite("T19 LAPACK workspace query: the reference minimum is a floor, not a validity threshold",
       .enabled(if: LAPACKLinearAlgebraProvider.isAvailable))
struct T19LAPACKWorkspaceQueryTests {

    /// The sweeps below must reach the shim, so the default provider must be the LAPACK one.
    @Test func defaultProviderIsLAPACK() {
        #expect(Operations.linearAlgebraProvider is LAPACKLinearAlgebraProvider)
    }

    @Test func svdThinSweepCrossesPathFive() {
        let failing = T19Cells.run("svdThin-sweep", T19Cells.sweep, T19Cells.svd)
        #expect(failing.isEmpty, "\(failing.count) cells: \(failing.prefix(6).joined(separator: "; "))")
    }

    @Test func exactPCASweepCrossesPathFive() {
        let failing = T19Cells.run("pca-exact-sweep", T19Cells.sweep, T19Cells.pca)
        #expect(failing.isEmpty, "\(failing.count) cells: \(failing.prefix(6).joined(separator: "; "))")
    }

    /// 500×384, 500×768 and 1000×1536 through PCA, and through svdThin as given and transposed.
    @Test func reportedCellsSucceed() {
        let transposed = T19Cells.reported.map { (rows: $0.columns, columns: $0.rows) }
        var failing = T19Cells.run("pca-exact-reported", T19Cells.reported, T19Cells.pca)
        failing += T19Cells.run("svdThin-reported", T19Cells.reported + transposed, T19Cells.svd)
        #expect(failing.isEmpty, "\(failing.joined(separator: "; "))")
    }

    /// The sketch path's final SVD on wide panels past the boundary: ℓ×d = 260×384 and 108×128 (floor(11ℓ/6) is 476
    /// and 198, so both are on path 5t), with ℓ < min(n, d) so the randomized branch runs.
    @Test func randomizedPCAWidePanelsCrossPathFive() {
        var failing: [String] = []
        var worst = 0.0
        for (n, d, k, p) in [(500, 384, 250, 10), (300, 128, 100, 8)] {
            let outcome = T19Cells.pcaSketch(n, d, components: k, oversampling: p)
            if let reason = outcome.failure {
                failing.append("\(n)x\(d)k\(k)p\(p) \(reason)")
            } else {
                worst = max(worst, outcome.worst)
            }
        }
        T19Cells.report("pca-sketch", failing, of: 2, worst: worst)
        #expect(failing.isEmpty, "\(failing.joined(separator: "; "))")
    }

    @Test func symmetricEigenNearBoundaryShapes() {
        var failing: [String] = []
        var worst = 0.0
        for vectors in [true, false] {
            for n in T19OtherCallers.eigenSizes {
                let outcome = T19OtherCallers.eigen(n, vectors: vectors)
                if let reason = outcome.failure {
                    failing.append("\(n)\(vectors ? "V" : "N") \(reason)")
                } else {
                    worst = max(worst, outcome.worst)
                }
            }
        }
        T19Cells.report("symmetricEigen", failing, of: 2 * T19OtherCallers.eigenSizes.count, worst: worst)
        #expect(failing.isEmpty, "\(failing.prefix(6).joined(separator: "; "))")
    }

    @Test func qrThinNearBoundaryShapes() {
        let failing = T19Cells.run("qrThin", T19OtherCallers.qrShapes, T19OtherCallers.qr)
        #expect(failing.isEmpty, "\(failing.prefix(6).joined(separator: "; "))")
    }
}
