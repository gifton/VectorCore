// VectorCore: Matrix (GEMM) Batch Distance
//
// beta-evolution-4, DOCUMENT-2. CPU batch-distance via the identity
//   ‖x − y‖² = ‖x‖² + ‖y‖² − 2⟨x, y⟩
// computed with Accelerate's cblas_sgemm, which dispatches to the AMX coprocessor
// on Apple Silicon. This is the high-throughput path for large query × candidate
// matrices; the per-pair SIMD kernels (BatchKernels) remain the small-batch path.
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
public enum MatrixDistance {

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
        let q = queries.count
        let n = candidates.count
        guard q > 0, n > 0 else { return }
        let d = queries[0].elementCount
        precondition(candidates[0].elementCount == d,
                     "queries and candidates must share a dimension (\(d) vs \(candidates[0].elementCount))")
        precondition(out.count == q * n,
                     "out must have q*n elements (got \(out.count), expected \(q * n))")

        let X = packRows(queries, d: d)
        let Y = packRows(candidates, d: d)

        var xNorm = [Float](repeating: 0, count: q)
        var yNorm = [Float](repeating: 0, count: n)
        rowSumOfSquares(X, rows: q, d: d, into: &xNorm)
        rowSumOfSquares(Y, rows: n, d: d, into: &yNorm)

        // out = -2 · X · Yᵀ
        gemmCrossTerm(X: X, Y: Y, q: q, n: n, d: d, alpha: -2, into: &out)

        // out[i,j] += ‖x_i‖² + ‖y_j‖² ; clamp ≥ 0 (cancellation guard before any sqrt).
        out.withUnsafeMutableBufferPointer { ob in
            let o = ob.baseAddress!
            for i in 0..<q {
                let xi = xNorm[i]
                let base = i * n
                for j in 0..<n {
                    let v = o[base + j] + xi + yNorm[j]
                    o[base + j] = v < 0 ? 0 : v
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
        let q = queries.count
        let n = candidates.count
        guard q > 0, n > 0 else { return }
        let d = queries[0].elementCount
        precondition(candidates[0].elementCount == d,
                     "queries and candidates must share a dimension (\(d) vs \(candidates[0].elementCount))")
        precondition(out.count == q * n,
                     "out must have q*n elements (got \(out.count), expected \(q * n))")

        var X = packRows(queries, d: d)
        var Y = packRows(candidates, d: d)
        normalizeRows(&X, rows: q, d: d)
        normalizeRows(&Y, rows: n, d: d)

        // out = X · Yᵀ  (cosine similarities, since rows are unit-norm)
        gemmCrossTerm(X: X, Y: Y, q: q, n: n, d: d, alpha: 1, into: &out)

        // distance = 1 − similarity, clamped to [0, 2].
        out.withUnsafeMutableBufferPointer { ob in
            let o = ob.baseAddress!
            for k in 0..<(q * n) {
                let dist = 1 - o[k]
                o[k] = dist < 0 ? 0 : (dist > 2 ? 2 : dist)
            }
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

    // MARK: - Private helpers

    /// Pack vectors into a contiguous row-major `[rows × d]` Float buffer.
    private static func packRows<V: UnifiedVectorBuffer>(_ vs: [V], d: Int) -> [Float] {
        let stride = MemoryLayout<Float>.stride
        var buf = [Float](repeating: 0, count: vs.count * d)
        buf.withUnsafeMutableBytes { dst in
            let dstBase = dst.baseAddress!
            for (i, v) in vs.enumerated() {
                v.withUnsafeContiguousBytes { src in
                    memcpy(dstBase.advanced(by: i * d * stride), src.baseAddress!, d * stride)
                }
            }
        }
        return buf
    }

    /// Per-row sum of squares (‖row‖²) via vDSP.
    private static func rowSumOfSquares(_ buf: [Float], rows: Int, d: Int, into norms: inout [Float]) {
        buf.withUnsafeBufferPointer { bp in
            let base = bp.baseAddress!
            for i in 0..<rows {
                var ss: Float = 0
                vDSP_svesq(base + i * d, 1, &ss, vDSP_Length(d))
                norms[i] = ss
            }
        }
    }

    /// L2-normalize each row in place (no-op for zero rows).
    private static func normalizeRows(_ buf: inout [Float], rows: Int, d: Int) {
        buf.withUnsafeMutableBufferPointer { bp in
            let base = bp.baseAddress!
            for i in 0..<rows {
                var ss: Float = 0
                vDSP_svesq(base + i * d, 1, &ss, vDSP_Length(d))
                if ss > 0 {
                    var inv = 1 / sqrtf(ss)
                    vDSP_vsmul(base + i * d, 1, &inv, base + i * d, 1, vDSP_Length(d))
                }
            }
        }
    }

    /// `out (q × n) = alpha · X (q × d) · Y (n × d)ᵀ` via cblas_sgemm (AMX on Apple Silicon).
    private static func gemmCrossTerm(
        X: [Float], Y: [Float], q: Int, n: Int, d: Int, alpha: Float, into out: inout [Float]
    ) {
        X.withUnsafeBufferPointer { xb in
            Y.withUnsafeBufferPointer { yb in
                out.withUnsafeMutableBufferPointer { ob in
                    cblas_sgemm(
                        CblasRowMajor, CblasNoTrans, CblasTrans,
                        Int32(q), Int32(n), Int32(d),
                        alpha, xb.baseAddress, Int32(d),
                        yb.baseAddress, Int32(d),
                        0, ob.baseAddress, Int32(n)
                    )
                }
            }
        }
    }
}
