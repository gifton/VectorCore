//
//  MatrixMultiply.swift
//  VectorCore
//
//  Internal GEMM utility for the LinearAlgebra layer.
//
//  COLUMN-MAJOR, matching the LinearAlgebraProvider seam, so panels flow
//  between GEMM passes and qrThin/svdThin with no physical transposes.
//  Apple platforms route to cblas_sgemm (AMX on Apple Silicon, same path
//  as Operations/MatrixDistance.swift); elsewhere a plain column-oriented
//  Swift kernel runs. The Swift kernel compiles on all platforms (and is
//  parity-tested against cblas) so the cross-platform path is never
//  dead code on CI. Not a public seam: PCA and future projection
//  primitives are the only intended callers, and they validate shapes
//  before calling — this layer traps on programmer error instead of
//  throwing.
//

#if canImport(Accelerate)
import Accelerate
#endif

internal enum LAMatMul {

    /// `C (m×n) = op(A) (m×k) · op(B) (k×n)`, all column-major flat buffers.
    ///
    /// `transposeA` means `a` is stored k×m and op(A) = aᵀ (likewise for B);
    /// storage dimensions are derived, so no leading-dimension bookkeeping
    /// leaks to callers.
    static func multiply(
        _ a: [Float], _ b: [Float],
        m: Int, n: Int, k: Int,
        transposeA: Bool = false,
        transposeB: Bool = false
    ) -> [Float] {
        validate(a, b, m: m, n: n, k: k)
        #if canImport(Accelerate)
        var c = [Float](repeating: 0, count: m * n)
        let lda = Int32(transposeA ? k : m)
        let ldb = Int32(transposeB ? n : k)
        a.withUnsafeBufferPointer { ap in
            b.withUnsafeBufferPointer { bp in
                c.withUnsafeMutableBufferPointer { cp in
                    cblas_sgemm(
                        CblasColMajor,
                        transposeA ? CblasTrans : CblasNoTrans,
                        transposeB ? CblasTrans : CblasNoTrans,
                        Int32(m), Int32(n), Int32(k),
                        1, ap.baseAddress, lda,
                        bp.baseAddress, ldb,
                        0, cp.baseAddress, Int32(m)
                    )
                }
            }
        }
        return c
        #else
        return multiplySwift(a, b, m: m, n: n, k: k, transposeA: transposeA, transposeB: transposeB)
        #endif
    }

    /// Pure-Swift kernel — the non-Accelerate path, kept unconditionally
    /// compiled so it can be parity-tested against cblas on Apple platforms.
    ///
    /// Column-oriented update: `C[:, j] += B(l, j) · op(A)[:, l]`. The inner
    /// loop is contiguous in C; op(A) columns are contiguous when A is
    /// untransposed (the dominant case in PCA's panels).
    internal static func multiplySwift(
        _ a: [Float], _ b: [Float],
        m: Int, n: Int, k: Int,
        transposeA: Bool = false,
        transposeB: Bool = false
    ) -> [Float] {
        validate(a, b, m: m, n: n, k: k)
        var c = [Float](repeating: 0, count: m * n)
        a.withUnsafeBufferPointer { ap in
            b.withUnsafeBufferPointer { bp in
                c.withUnsafeMutableBufferPointer { cp in
                    for j in 0..<n {
                        for l in 0..<k {
                            // Storage of b: untransposed is k×n (col-major:
                            // (l,j) at j*k+l); transposed is n×k ((j,l) at l*n+j).
                            let scale = transposeB ? bp[l * n + j] : bp[j * k + l]
                            if scale == 0 { continue }
                            if transposeA {
                                // a stored k×m: op(A)(i,l) = a(l,i) at i*k + l
                                for i in 0..<m {
                                    cp[j * m + i] += scale * ap[i * k + l]
                                }
                            } else {
                                // a stored m×k: op(A)(:,l) contiguous at l*m
                                let base = l * m
                                for i in 0..<m {
                                    cp[j * m + i] += scale * ap[base + i]
                                }
                            }
                        }
                    }
                }
            }
        }
        return c
    }

    private static func validate(_ a: [Float], _ b: [Float], m: Int, n: Int, k: Int) {
        precondition(m >= 1 && n >= 1 && k >= 1, "GEMM dimensions must be positive")
        precondition(a.count == m * k, "A element count \(a.count) != \(m)×\(k)")
        precondition(b.count == k * n, "B element count \(b.count) != \(k)×\(n)")
    }
}
