// VectorCore: Matrix (GEMM) Batch Distance
//
// beta-evolution-4, DOCUMENT-2. CPU batch-distance via the identity
//   ‖x − y‖² = ‖x‖² + ‖y‖² − 2⟨x, y⟩
// computed with Accelerate's cblas_sgemm. Which hardware unit Accelerate uses is
// not specified by this repository, and whether this path is faster than the
// per-pair SIMD kernels (BatchKernels) at a given size is not verified here.
//
// CPU-only. The GPU GEMM lives in VectorAccelerate.

import Foundation
import Accelerate

/// GEMM-based batch distance over contiguous `Float32` vectors.
///
/// Packs queries (`q × d`) and candidates (`n × d`) into row-major matrices and
/// computes the full `q × n` distance matrix with a single matrix multiply. Any
/// `UnifiedVectorBuffer` works as input — the optimized `Vector*Optimized` types and
/// `DynamicVector` all conform — so one implementation serves every dimension.
///
/// Results are written row-major: `out[i*n + j]` is the distance from `queries[i]`
/// to `candidates[j]`. The `into:` forms are the hot path (caller owns `out`); the
/// returning forms are convenience wrappers that allocate.
///
/// For **repeated** search against a fixed candidate set, pack the candidates once
/// with ``prepare(_:normalized:)`` and reuse the ``PreparedCandidates`` across many
/// query batches — this avoids re-packing and re-norming the (large) candidate side
/// on every call.
public enum MatrixDistance {

    // MARK: - Prepared (reusable) candidates

    /// Candidates packed into a contiguous row-major matrix with their norms
    /// precomputed, ready to be reused across many query batches.
    ///
    /// Build with ``MatrixDistance/prepare(_:normalized:)``. Use `normalized: false`
    /// for Euclidean (stores raw rows + ‖·‖²) and `normalized: true` for Cosine
    /// (stores unit-normalized rows).
    public struct PreparedCandidates: Sendable {
        @usableFromInline let packed: [Float]   // n × d, row-major
        @usableFromInline let norms: [Float]    // ‖row‖² per candidate (empty when normalized)
        @usableFromInline let n: Int
        @usableFromInline let d: Int
        @usableFromInline let normalized: Bool

        @usableFromInline
        init(packed: [Float], norms: [Float], n: Int, d: Int, normalized: Bool) {
            self.packed = packed; self.norms = norms; self.n = n; self.d = d; self.normalized = normalized
        }

        /// Number of candidates.
        public var count: Int { n }
        /// Vector dimension.
        public var dimension: Int { d }
    }

    /// Pack candidates once for reuse. `normalized: false` prepares for Euclidean
    /// (raw rows + squared norms); `normalized: true` prepares for Cosine
    /// (L2-normalized rows). Every row must have the first row's `elementCount`; a
    /// shorter or longer row stops with a named `fatalError` in every build, before
    /// any row is read past its end (`ReleaseSizeGuardTests.matrixRowsOfUnequalLengthTrap`).
    /// The same check covers the query rows of every matrix entry.
    public static func prepare<V: UnifiedVectorBuffer>(
        _ candidates: [V], normalized: Bool = false
    ) -> PreparedCandidates {
        let n = candidates.count
        let d = n > 0 ? candidates[0].elementCount : 0
        var packed = packRows(candidates, d: d)
        if normalized {
            normalizeRows(&packed, rows: n, d: d)
            return PreparedCandidates(packed: packed, norms: [], n: n, d: d, normalized: true)
        } else {
            // swiftlang/swift#92562 (P02 R-LS4): a non-literal initializer.
            var norms = [Float](repeating: 0, count: n)
            rowSumOfSquares(packed, rows: n, d: d, into: &norms)
            return PreparedCandidates(packed: packed, norms: norms, n: n, d: d, normalized: false)
        }
    }

    // MARK: - Euclidean (squared)

    /// Squared-Euclidean distance matrix via GEMM. `out[i*n + j] = ‖q_i − c_j‖²`,
    /// clamped at 0 (the identity can round slightly negative when `q_i ≈ c_j`).
    ///
    /// - Precondition: `out.count == queries.count * candidates.count`, and all
    ///   inputs share a dimension.
    public static func euclideanSquaredMatrix<V: UnifiedVectorBuffer>(
        queries: [V],
        candidates: [V],
        into out: inout [Float]
    ) {
        guard !queries.isEmpty, !candidates.isEmpty else { return }
        euclideanSquaredMatrix(queries: queries, prepared: prepare(candidates, normalized: false), into: &out)
    }

    /// Squared-Euclidean distance matrix against pre-packed candidates (reusable path).
    public static func euclideanSquaredMatrix<V: UnifiedVectorBuffer>(
        queries: [V],
        prepared: PreparedCandidates,
        into out: inout [Float]
    ) {
        let q = queries.count
        let n = prepared.n
        guard q > 0, n > 0 else { return }
        precondition(!prepared.normalized, "PreparedCandidates for Euclidean must be built with normalized: false")
        let d = queries[0].elementCount
        precondition(prepared.d == d, "queries and candidates must share a dimension (\(d) vs \(prepared.d))")
        precondition(out.count == q * n, "out must have q*n elements (got \(out.count), expected \(q * n))")

        let X = packRows(queries, d: d)
        // swiftlang/swift#92562 (P02 R-LS4): a non-literal initializer.
        var xNorm = [Float](repeating: 0, count: q)
        rowSumOfSquares(X, rows: q, d: d, into: &xNorm)

        gemmCrossTerm(X: X, Y: prepared.packed, n: n, d: d, alpha: -2, into: &out)

        let yNorm = prepared.norms
        out.withUnsafeMutableBufferPointer { ob in
            xNorm.withUnsafeBufferPointer { xb in
                yNorm.withUnsafeBufferPointer { yb in
                    euclideanPostprocess(ob.baseAddress!, rows: q, columns: n, ldc: n,
                                         xNorm: xb.baseAddress!, yNorm: yb.baseAddress!)
                }
            }
        }
    }

    /// Allocating convenience overload.
    public static func euclideanSquaredMatrix<V: UnifiedVectorBuffer>(
        queries: [V], candidates: [V]
    ) -> [Float] {
        var out = [Float](repeating: 0, count: queries.count * candidates.count)
        euclideanSquaredMatrix(queries: queries, candidates: candidates, into: &out)
        return out
    }

    // MARK: - Cosine distance (1 − cosine similarity)

    /// Cosine *distance* matrix via GEMM on L2-normalized rows.
    /// `out[i*n + j] = 1 − cos(q_i, c_j)`, clamped to `[0, 2]`.
    public static func cosineDistanceMatrix<V: UnifiedVectorBuffer>(
        queries: [V],
        candidates: [V],
        into out: inout [Float]
    ) {
        guard !queries.isEmpty, !candidates.isEmpty else { return }
        cosineDistanceMatrix(queries: queries, prepared: prepare(candidates, normalized: true), into: &out)
    }

    /// Cosine distance matrix against pre-normalized candidates (reusable path).
    public static func cosineDistanceMatrix<V: UnifiedVectorBuffer>(
        queries: [V],
        prepared: PreparedCandidates,
        into out: inout [Float]
    ) {
        let q = queries.count
        let n = prepared.n
        guard q > 0, n > 0 else { return }
        precondition(prepared.normalized, "PreparedCandidates for Cosine must be built with normalized: true")
        let d = queries[0].elementCount
        precondition(prepared.d == d, "queries and candidates must share a dimension (\(d) vs \(prepared.d))")
        precondition(out.count == q * n, "out must have q*n elements (got \(out.count), expected \(q * n))")

        var X = packRows(queries, d: d)
        normalizeRows(&X, rows: q, d: d)

        gemmCrossTerm(X: X, Y: prepared.packed, n: n, d: d, alpha: 1, into: &out)

        out.withUnsafeMutableBufferPointer { ob in
            cosinePostprocess(ob.baseAddress!, rows: q, columns: n, ldc: n)
        }
    }

    /// Allocating convenience overload.
    public static func cosineDistanceMatrix<V: UnifiedVectorBuffer>(
        queries: [V], candidates: [V]
    ) -> [Float] {
        var out = [Float](repeating: 0, count: queries.count * candidates.count)
        cosineDistanceMatrix(queries: queries, candidates: candidates, into: &out)
        return out
    }

    // MARK: - Private helpers (the array forms used by the public entries above)

    /// Pack vectors into a contiguous row-major `[rows × d]` Float buffer.
    private static func packRows<V: UnifiedVectorBuffer>(_ vs: [V], d: Int) -> [Float] {
        // swiftlang/swift#92562 (P02 R-LS4): a non-literal initializer.
        var buf = [Float](repeating: 0, count: vs.count * d)
        buf.withUnsafeMutableBufferPointer { dst in
            packRows(vs, rows: 0..<vs.count, d: d, into: dst.baseAddress!)
        }
        return buf
    }

    /// Per-row sum of squares (‖row‖²) via vDSP.
    private static func rowSumOfSquares(_ buf: [Float], rows: Int, d: Int, into norms: inout [Float]) {
        buf.withUnsafeBufferPointer { bp in
            norms.withUnsafeMutableBufferPointer { nb in
                rowSumOfSquares(bp.baseAddress!, rows: rows, d: d, into: nb.baseAddress!)
            }
        }
    }

    /// L2-normalize each row in place (no-op for zero rows).
    private static func normalizeRows(_ buf: inout [Float], rows: Int, d: Int) {
        buf.withUnsafeMutableBufferPointer { bp in
            normalizeRows(bp.baseAddress!, rows: rows, d: d)
        }
    }

    /// `out (q × n) = alpha · X (q × d) · Y (n × d)ᵀ` via cblas_sgemm.
    /// The row count `q` is derived from `X.count / d` (X is row-major `q × d`).
    private static func gemmCrossTerm(
        X: [Float], Y: [Float], n: Int, d: Int, alpha: Float, into out: inout [Float]
    ) {
        let q = X.count / d
        X.withUnsafeBufferPointer { xb in
            Y.withUnsafeBufferPointer { yb in
                out.withUnsafeMutableBufferPointer { ob in
                    gemmCrossTerm(x: xb.baseAddress!, y: yb.baseAddress!,
                                  arguments: GemmArguments(rows: q, columns: n, d: d, ldc: n), alpha: alpha,
                                  into: ob.baseAddress!)
                }
            }
        }
    }

    // MARK: - Active-row helpers (internal; also the arithmetic of the explicit Gram route)
    //
    // P02 (RD9): the public entries above and `Operations.findNearestBatch` with `BatchSearchCPUStrategy.gram` share
    // these helpers, so a Gram tile computes the same arithmetic as the public prepared/into entries on the same rows
    // (`P02GramRouteTests.tileScoresMatchPublicPreparedEntries`). Every row count is an ACTIVE count, never a scratch
    // capacity. Pointers are passed by value, never `inout` (P02 R-LS2, swiftlang/swift#92562), and every caller keeps
    // the borrow that produced them synchronous. Unchecked caller preconditions, as before this refactor: every
    // vector has at least `d` Floats, every pointer covers the rows and columns it is given, and `rows`, `columns`,
    // `d` and `ldc` fit `Int32` (`Int32(_:)` traps otherwise); the Gram route checks the last with `GramPlan`.

    /// Copies rows `rows` of `vs`, `d` Floats each, to consecutive rows of `dst`: row `rows.lowerBound + i` goes to
    /// `dst + i * d`.
    internal static func packRows<V: UnifiedVectorBuffer>(
        _ vs: [V], rows: Range<Int>, d: Int, into dst: UnsafeMutablePointer<Float>
    ) {
        let bytes = d * MemoryLayout<Float>.stride
        for (offset, i) in rows.enumerated() {
            // A named guard: a `precondition` message is dropped at -O, and the Release row asserts this diagnostic.
            guard vs[i].elementCount == d else {
                fatalError("MatrixDistance rows must share a dimension: row \(i) has \(vs[i].elementCount) elements, expected \(d)")
            }
            let target = dst + offset * d
            vs[i].withUnsafeContiguousBytes { src in
                _ = memcpy(target, src.baseAddress!, bytes)
            }
        }
    }

    /// `norms[i] = ‖row i‖²` (`vDSP_svesq`) for `rows` rows of `d` Floats at `src`.
    internal static func rowSumOfSquares(
        _ src: UnsafePointer<Float>, rows: Int, d: Int, into norms: UnsafeMutablePointer<Float>
    ) {
        for i in 0..<rows {
            var ss: Float = 0
            vDSP_svesq(src + i * d, 1, &ss, vDSP_Length(d))
            norms[i] = ss
        }
    }

    /// L2-normalizes `rows` rows of `d` Floats at `base` in place with `NormalizationCore` (N01): a zero row stays
    /// unchanged and a row with a nonfinite coordinate becomes all NaN.
    internal static func normalizeRows(_ base: UnsafeMutablePointer<Float>, rows: Int, d: Int) {
        for i in 0..<rows {
            let row = base + i * d
            switch NormalizationCore.plan(count: d, get: { row[$0] }) {
            case .success(let plan):
                for j in 0..<d { row[j] = plan.coordinate(row[j]) }
            case .failure:
                if (0..<d).contains(where: { !row[$0].isFinite }) {
                    for j in 0..<d { row[j] = .nan }
                }
            }
        }
    }

    /// The integer arguments of one `gemmCrossTerm` call. `gemmCrossTerm` passes exactly these fields to `cblas_sgemm`
    /// (lexical pin C8 in `Scripts/ci/tests/test_gram_route_lexical.py`), so a caller that keeps this value (the Gram
    /// route's trace seam, P02 phase C review RA-1) holds the arguments BLAS received rather than a recomputation of them.
    internal struct GemmArguments: Sendable, Equatable {
        /// Active rows of `x` (M), active rows of `y` (N), the dimension (K), and the leading dimensions of `x`, `y`
        /// and `out`.
        let m: Int32, n: Int32, k: Int32, lda: Int32, ldb: Int32, ldc: Int32

        /// A `rows × columns` output block of `d`-long rows with row stride `ldc`; `lda = ldb = d`. `Int32(_:)` traps
        /// when a value does not fit. Its one caller is the public entries' array wrapper, which checks nothing first;
        /// the Gram route does not use it (it takes `GramPlan.gemmArguments`, built from the plan's checked fields).
        init(rows: Int, columns: Int, d: Int, ldc: Int) {
            m = Int32(rows)
            n = Int32(columns)
            k = Int32(d)
            lda = Int32(d)
            ldb = Int32(d)
            self.ldc = Int32(ldc)
        }

        /// The six fields as given, already `Int32`: nothing is converted here. The Gram route builds its value with
        /// `GramPlan.gemmArguments(activeQueryRows:activeCandidateRows:)` from the plan's checked fields (P02 final
        /// review C-M3).
        init(m: Int32, n: Int32, k: Int32, lda: Int32, ldb: Int32, ldc: Int32) {
            self.m = m
            self.n = n
            self.k = k
            self.lda = lda
            self.ldb = ldb
            self.ldc = ldc
        }
    }

    /// `out[i * ldc + j] = alpha · Σ_k x[i * d + k] · y[j * d + k]` for `i < m`, `j < n` (the fields of `arguments`):
    /// one `cblas_sgemm` (row-major, `x` not transposed, `y` transposed, `beta = 0`) that receives the six integer
    /// fields of `arguments` unchanged. Only the active `m × n` block of `out` is passed to BLAS; with `beta = 0`,
    /// BLAS does not read `out`.
    internal static func gemmCrossTerm(
        x: UnsafePointer<Float>, y: UnsafePointer<Float>, arguments: GemmArguments, alpha: Float,
        into out: UnsafeMutablePointer<Float>
    ) {
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            arguments.m, arguments.n, arguments.k,
            alpha, x, arguments.lda,
            y, arguments.ldb,
            0, out, arguments.ldc
        )
    }

    /// Squared-Euclidean postprocess of the active `rows × columns` block of `out` (row stride `ldc`), which holds
    /// `−2·x·y`: `v = (out + xNorm[i]) + yNorm[j]`, stored as `v < 0 ? 0 : v` (a NaN stays NaN).
    internal static func euclideanPostprocess( // swiftlint:disable:this function_parameter_count
        _ out: UnsafeMutablePointer<Float>, rows: Int, columns: Int, ldc: Int,
        xNorm: UnsafePointer<Float>, yNorm: UnsafePointer<Float>
    ) {
        for i in 0..<rows {
            let xi = xNorm[i]
            let base = i * ldc
            for j in 0..<columns {
                let v = out[base + j] + xi + yNorm[j]
                out[base + j] = v < 0 ? 0 : v
            }
        }
    }

    /// Cosine-distance postprocess of the active `rows × columns` block of `out` (row stride `ldc`), which holds the
    /// dot products of normalized rows: `1 − dot`, clamped to `[0, 2]` by comparisons (a NaN stays NaN).
    internal static func cosinePostprocess(_ out: UnsafeMutablePointer<Float>, rows: Int, columns: Int, ldc: Int) {
        for i in 0..<rows {
            let base = i * ldc
            for j in 0..<columns {
                let dist = 1 - out[base + j]
                out[base + j] = dist < 0 ? 0 : (dist > 2 ? 2 : dist)
            }
        }
    }
}
