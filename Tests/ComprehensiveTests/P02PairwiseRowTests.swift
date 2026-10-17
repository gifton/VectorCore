import Foundation
import Testing
@testable import VectorCore

// P02 phase D (RD10; phase 1 check map R2-R4, R8-R10, X11/R13, E24): the GEMM route of
// `BatchOperations.pairwiseDistances` prepares the candidates once and computes each query row through MatrixDistance's
// prepared/into path straight into its final row, formatting Euclidean rows in place (`v > 0 ? sqrt(v) : 0`). Every
// test runs at an eligible size under the default configuration (`enableMatrixRouting`, `matrixRoutingMinN` 256;
// E06 changes it only in child processes) and observes the `batch.pairwise.gemm` phase, so it exercises that route.
// The source shape (no flat N·N buffer, no allocating matrix call) is pinned lexically by
// Scripts/ci/tests/test_gram_route_lexical.py C7.

/// The independent per-row reference: `MatrixDistance.prepare` once, then the public prepared/into entry for each query
/// row `[rows[i]]`, Euclidean rows formatted as the route documents (a NaN or nonpositive squared score gives 0).
func p02PreparedRows<V: UnifiedVectorBuffer>(_ rows: [V], euclidean: Bool) -> [[Float]] {
    let n = rows.count
    let prepared = MatrixDistance.prepare(rows, normalized: !euclidean)
    return (0..<n).map { i in
        var row = [Float](repeating: 0, count: n)   // swiftlang/swift#92562: a non-literal initializer
        if euclidean {
            MatrixDistance.euclideanSquaredMatrix(queries: [rows[i]], prepared: prepared, into: &row)
            return row.map { $0 > 0 ? $0.squareRoot() : 0 }
        }
        MatrixDistance.cosineDistanceMatrix(queries: [rows[i]], prepared: prepared, into: &row)
        return row
    }
}

/// Runs `pairwiseDistances` under a BoundedExecution observer; returns the matrix and whether the GEMM phase began and
/// ended.
func p02ObservedPairwise<V: VectorProtocol & Sendable>(
    _ rows: [V], euclidean: Bool
) async -> (matrix: [[Float]], gemmRoute: Bool) where V.Scalar == Float {
    let events = P02EventLog()
    let matrix = await BoundedExecution.$observation.withValue({ events.append($0) }, operation: {
        if euclidean { return await BatchOperations.pairwiseDistances(rows, metric: EuclideanDistance()) }
        return await BatchOperations.pairwiseDistances(rows, metric: CosineDistance())
    })
    let gemm = events.snapshot.filter { $0.phase == "batch.pairwise.gemm" }
    return (matrix, gemm.contains { $0.kind == .begin } && gemm.contains { $0.kind == .end })
}

@Suite("P02 pairwise GEMM rows (prepared/into per row)")
struct P02PairwiseRowTests {

    /// Rows for an eligible pairwise call: the near-duplicate cluster (row 255 duplicates row 0), plus seeded uniform
    /// rows past 256.
    static func rows(count: Int, width: Int) -> [[Float]] {
        let base = p02Rows(.near, width: width)
        return Array(base.prefix(count)) + p02RandomRows(max(0, count - base.count), width: width, seed: 0x5EED)
    }

    // MARK: R2, R3, R4, R8, R10: row bits are the per-row prepared/into bits; the diagonal is the metric

    /// At N = 256 and 257, every width, Euclidean and cosine: each off-diagonal entry equals, bit for bit, the public
    /// prepared/into entry run on that query row (`p02PreparedRows`), and each diagonal entry equals
    /// `metric.distance(v, v)` bit for bit. The number of off-diagonal entries where a single N×N multiply (the former
    /// route) gives other bits is printed as a `P02_PAIRWISE_GEOMETRY` line: a one-host record, not an assertion.
    @Test func rowsEqualPreparedIntoPerRow() async throws {
        try await p02ForEachWidth { type in try await perRow(type) }
    }

    private func perRow<V: P02GramWidth>(_ type: V.Type) async throws {
        let w = V.p02Width
        for n in [256, 257] {
            let rows = try V.rows(Self.rows(count: n, width: w))
            for euclidean in [true, false] {
                let label = "w\(w) N\(n) \(euclidean ? "euclidean" : "cosine")"
                let (matrix, gemmRoute) = await p02ObservedPairwise(rows, euclidean: euclidean)
                #expect(gemmRoute, "\(label): no batch.pairwise.gemm phase")
                let reference = p02PreparedRows(rows, euclidean: euclidean)
                let full = euclidean
                    ? MatrixDistance.euclideanSquaredMatrix(queries: rows, candidates: rows).map { $0 > 0 ? $0.squareRoot() : 0 }
                    : MatrixDistance.cosineDistanceMatrix(queries: rows, candidates: rows)
                var mismatches = 0, geometryDifferences = 0
                #expect(matrix.count == n && matrix.allSatisfy { $0.count == n }, "\(label)")
                for i in 0..<n {
                    let diagonal = euclidean ? EuclideanDistance().distance(rows[i], rows[i])
                                             : CosineDistance().distance(rows[i], rows[i])
                    #expect(matrix[i][i].bitPattern == diagonal.bitPattern, "\(label) diagonal \(i)")
                    for j in 0..<n where j != i {
                        if matrix[i][j].bitPattern != reference[i][j].bitPattern { mismatches += 1 }
                        if full[i * n + j].bitPattern != reference[i][j].bitPattern { geometryDifferences += 1 }
                    }
                }
                #expect(mismatches == 0, "\(label): \(mismatches) off-diagonal entries differ from prepared/into per row")
                print("P02_PAIRWISE_GEOMETRY \(label) offDiagonal=\(n * (n - 1)) "
                      + "differFromSingleNxNMultiply=\(geometryDifferences)")
            }
        }
    }

    // MARK: R9, R10: exceptional rows at an eligible N

    /// N = 256 with a NaN coordinate in row 3 and an all-NaN row 4. Euclidean: the per-row squared scores of those rows
    /// are NaN and the route formats them to 0 off the diagonal (the documented pairwise Euclidean NaN-to-zero
    /// formatting), in their rows and columns; the diagonal stays `metric.distance(v, v)` (NaN). Cosine: no formatting,
    /// so those off-diagonal entries stay NaN. Every entry still equals the per-row prepared/into reference bits.
    @Test func nanRowsAtAnEligibleN() async throws {
        try await p02ForEachWidth { type in try await nanRows(type) }
    }

    private func nanRows<V: P02GramWidth>(_ type: V.Type) async throws {
        let w = V.p02Width
        var values = Self.rows(count: 256, width: w)
        values[3][7] = .nan
        values[4] = [Float](repeating: .nan, count: w)
        let rows = try V.rows(values)
        for euclidean in [true, false] {
            let label = "w\(w) \(euclidean ? "euclidean" : "cosine")"
            let (matrix, gemmRoute) = await p02ObservedPairwise(rows, euclidean: euclidean)
            #expect(gemmRoute, "\(label): no batch.pairwise.gemm phase")
            let reference = p02PreparedRows(rows, euclidean: euclidean)
            var squared = [Float](repeating: 0, count: 256)   // swiftlang/swift#92562: a non-literal initializer
            MatrixDistance.euclideanSquaredMatrix(queries: [rows[3]], prepared: MatrixDistance.prepare(rows), into: &squared)
            #expect(squared[10].isNaN, "\(label): the unformatted squared score of row 3 is \(squared[10])")
            for i in 0..<256 {
                for j in 0..<256 where j != i {
                    #expect(matrix[i][j].bitPattern == reference[i][j].bitPattern, "\(label) (\(i), \(j))")
                    if [3, 4].contains(i) || [3, 4].contains(j) {
                        if euclidean {
                            #expect(matrix[i][j] == 0 && matrix[i][j].sign == .plus, "\(label) (\(i), \(j)) = \(matrix[i][j])")
                        } else {
                            #expect(matrix[i][j].isNaN, "\(label) (\(i), \(j)) = \(matrix[i][j])")
                        }
                    }
                }
            }
            for i in [3, 4] {
                let direct = euclidean ? EuclideanDistance().distance(rows[i], rows[i])
                                       : CosineDistance().distance(rows[i], rows[i])
                #expect(matrix[i][i].bitPattern == direct.bitPattern, "\(label) diagonal \(i)")
            }
        }
    }

    // MARK: Guard order (phase E): the identity checks precede the GEMM observation

    /// At N = 256 with Euclidean and cosine distance (an eligible size and metric), vector types the GEMM route does not
    /// take (`DynamicVector`, `Vector384Optimized`, `Vector<Dim512>`) open no `batch.pairwise.gemm` phase: the
    /// element-type identity checks (`exactArray(...) != nil`) run before that observation begins. Each diagonal is
    /// still the metric's own self-distance. Control: `Vector512Optimized` at the same size opens the phase.
    @Test func ineligibleTypesOpenNoGemmPhase() async throws {
        let values = Self.rows(count: 256, width: 512)
        func check<V: VectorProtocol & Sendable>(_ rows: [V], expectGemm: Bool) async where V.Scalar == Float {
            for euclidean in [true, false] {
                let (matrix, gemmRoute) = await p02ObservedPairwise(rows, euclidean: euclidean)
                #expect(gemmRoute == expectGemm, "\(V.self) \(euclidean ? "euclidean" : "cosine")")
                #expect(matrix.count == rows.count)
                let diagonal = euclidean ? EuclideanDistance().distance(rows[0], rows[0])
                                         : CosineDistance().distance(rows[0], rows[0])
                #expect(matrix.first?.first?.bitPattern == diagonal.bitPattern, "\(V.self)")
            }
        }
        await check(values.map { DynamicVector($0) }, expectGemm: false)
        await check(try values.map { try Vector384Optimized(Array($0.prefix(384))) }, expectGemm: false)
        await check(try values.map { try Vector<Dim512>($0) }, expectGemm: false)
        await check(try p02Vectors(values, as: Vector512Optimized.self), expectGemm: true)
    }

    // MARK: X11, R13: complete result under cancellation at an eligible N

    /// `pairwiseDistances` is nonthrowing and returns its complete result when the calling task is already cancelled:
    /// at N = 256, Vector512Optimized, Euclidean and cosine, the cancelled call takes the GEMM route and returns the
    /// same N×N bits as an uncancelled call (the ExecutionScheduling/P01 cancelled-matrix law at a GEMM-eligible size).
    @Test func completeResultUnderCancellation() async throws {
        let rows = try p02Vectors(Self.rows(count: 256, width: 512), as: Vector512Optimized.self)
        for euclidean in [true, false] {
            let label = euclidean ? "euclidean" : "cosine"
            let (expected, uncancelledRoute) = await p02ObservedPairwise(rows, euclidean: euclidean)
            let (cancelled, sawCancellation, cancelledRoute) = await Task {
                withUnsafeCurrentTask { $0?.cancel() }
                let observed = await p02ObservedPairwise(rows, euclidean: euclidean)
                return (observed.matrix, Task.isCancelled, observed.gemmRoute)
            }.value
            #expect(sawCancellation, "\(label): the task was not cancelled")
            #expect(uncancelledRoute && cancelledRoute, "\(label): no batch.pairwise.gemm phase")
            #expect(cancelled.count == 256 && cancelled.allSatisfy { $0.count == 256 }, "\(label)")
            #expect(cancelled.map { $0.map(\.bitPattern) } == expected.map { $0.map(\.bitPattern) }, "\(label)")
        }
    }
}
