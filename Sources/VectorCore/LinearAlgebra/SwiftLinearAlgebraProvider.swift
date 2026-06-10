//
//  SwiftLinearAlgebraProvider.swift
//  VectorCore
//
//  Pure-Swift LinearAlgebraProvider fallback. Algorithm choices favor
//  implementation transparency and unconditional numerical robustness over
//  speed — this is the portability/parity path, not the fast path:
//
//  - QR:    Householder reflections (LAPACK-style storage: reflectors below
//           the diagonal, accumulated backward to form thin Q).
//  - Eigen: cyclic Jacobi rotations. Unconditionally convergent for
//           symmetric matrices, quadratic in the tail. O(n³·sweeps).
//  - SVD:   Hestenes one-sided Jacobi — orthogonalizes column pairs of A
//           directly, accumulating V; singular values fall out as column
//           norms. Avoids forming AᵀA (which would square the condition
//           number — significant in Float32).
//
//  Caveat (documented, deliberate): for exactly rank-deficient inputs the
//  SVD's U columns paired with zero singular values are not completed to an
//  orthonormal basis (LAPACK completes them). Downstream PCA/randomized-SVD
//  consumers truncate at k ≪ rank, so this is unobservable there.
//
//  All matrices column-major; see LinearAlgebraProvider layout contract.
//

import Foundation

public struct SwiftLinearAlgebraProvider: LinearAlgebraProvider {

    /// Hard cap on Jacobi sweeps (eigen and SVD). Jacobi converges
    /// quadratically once rotations get small; well-posed Float32 problems
    /// finish in < 15 sweeps. Exhaustion ⇒ throw rather than return junk.
    private static let maxSweeps = 60

    public init() {}

    // MARK: - QR (Householder)

    public func qrThin(_ a: [Float], rows m: Int, columns n: Int) throws -> QRFactorization {
        try LinearAlgebraValidation.validateQR(count: a.count, rows: m, columns: n)

        var fac = a                                  // factored form: R upper, reflectors below
        var tau = [Float](repeating: 0, count: n)

        for j in 0..<n {
            // Householder vector for column j, rows j..<m.
            var normSq: Float = 0
            for i in j..<m { normSq += fac[j * m + i] * fac[j * m + i] }
            let norm = normSq.squareRoot()
            if norm == 0 { tau[j] = 0; continue }    // zero column: identity reflector

            let alpha = fac[j * m + j]
            let beta: Float = alpha >= 0 ? -norm : norm
            tau[j] = (beta - alpha) / beta
            let invScale = 1 / (alpha - beta)        // v = [1, x_tail/(alpha-beta)]
            for i in (j + 1)..<m { fac[j * m + i] *= invScale }
            fac[j * m + j] = beta                    // R diagonal entry

            // Apply H_j = I − τ v vᵀ to trailing columns.
            for c in (j + 1)..<n {
                var w = fac[c * m + j]               // v_j = 1 implicit
                for i in (j + 1)..<m { w += fac[j * m + i] * fac[c * m + i] }
                w *= tau[j]
                fac[c * m + j] -= w
                for i in (j + 1)..<m { fac[c * m + i] -= w * fac[j * m + i] }
            }
        }

        // Extract R (n×n upper triangle).
        var r = [Float](repeating: 0, count: n * n)
        for j in 0..<n {
            for i in 0...j { r[j * n + i] = fac[j * m + i] }
        }

        // Accumulate thin Q: apply H_{n-1}…H_0 to the first n identity columns,
        // backward so each reflector touches only rows/cols it acts on.
        var q = [Float](repeating: 0, count: m * n)
        for j in 0..<n { q[j * m + j] = 1 }
        for j in stride(from: n - 1, through: 0, by: -1) {
            let t = tau[j]
            if t == 0 { continue }
            for c in j..<n {
                var w = q[c * m + j]
                for i in (j + 1)..<m { w += fac[j * m + i] * q[c * m + i] }
                w *= t
                q[c * m + j] -= w
                for i in (j + 1)..<m { q[c * m + i] -= w * fac[j * m + i] }
            }
        }

        return QRFactorization(q: q, r: r, rows: m, columns: n)
    }

    // MARK: - Symmetric eigen (cyclic Jacobi)

    public func symmetricEigen(
        _ a: [Float],
        dimension n: Int,
        computeEigenvectors: Bool
    ) throws -> SymmetricEigenDecomposition {
        try LinearAlgebraValidation.validateSquare(count: a.count, dimension: n)

        // Symmetrize from the LOWER triangle (matches LAPACK uplo='L' contract).
        var mat = a
        for j in 0..<n {
            for i in (j + 1)..<n { mat[i * n + j] = mat[j * n + i] }
        }

        var vecs: [Float] = []
        if computeEigenvectors {
            vecs = [Float](repeating: 0, count: n * n)
            for j in 0..<n { vecs[j * n + j] = 1 }
        }

        var frobSq: Float = 0
        for v in mat { frobSq += v * v }
        let negligible = Float.ulpOfOne * frobSq.squareRoot()

        var converged = (n == 1)
        var sweep = 0
        while !converged && sweep < Self.maxSweeps {
            sweep += 1
            converged = true
            for p in 0..<(n - 1) {
                for q in (p + 1)..<n {
                    let apq = mat[q * n + p]
                    if abs(apq) <= negligible { continue }
                    converged = false

                    let app = mat[p * n + p]
                    let aqq = mat[q * n + q]
                    let theta = (aqq - app) / (2 * apq)
                    // Stable smaller root of t² + 2θt − 1 = 0.
                    let t: Float = (theta >= 0 ? 1 : -1) / (abs(theta) + (theta * theta + 1).squareRoot())
                    let c = 1 / (t * t + 1).squareRoot()
                    let s = t * c

                    for i in 0..<n where i != p && i != q {
                        let aip = mat[p * n + i]
                        let aiq = mat[q * n + i]
                        let newIP = c * aip - s * aiq
                        let newIQ = s * aip + c * aiq
                        mat[p * n + i] = newIP; mat[i * n + p] = newIP
                        mat[q * n + i] = newIQ; mat[i * n + q] = newIQ
                    }
                    mat[p * n + p] = app - t * apq
                    mat[q * n + q] = aqq + t * apq
                    mat[q * n + p] = 0
                    mat[p * n + q] = 0

                    if computeEigenvectors {
                        for i in 0..<n {
                            let vip = vecs[p * n + i]
                            let viq = vecs[q * n + i]
                            vecs[p * n + i] = c * vip - s * viq
                            vecs[q * n + i] = s * vip + c * viq
                        }
                    }
                }
            }
        }
        guard converged else {
            throw ErrorBuilder(.operationFailed)
                .message("Jacobi eigendecomposition did not converge in \(Self.maxSweeps) sweeps (n=\(n))")
                .build()
        }

        // Sort ascending (ssyevd convention), permuting eigenvector columns.
        var order = Array(0..<n)
        order.sort { mat[$0 * n + $0] < mat[$1 * n + $1] }

        var eigenvalues = [Float](repeating: 0, count: n)
        for (dst, src) in order.enumerated() { eigenvalues[dst] = mat[src * n + src] }

        var sortedVecs: [Float]?
        if computeEigenvectors {
            var sorted = [Float](repeating: 0, count: n * n)
            for (dst, src) in order.enumerated() {
                for i in 0..<n { sorted[dst * n + i] = vecs[src * n + i] }
            }
            sortedVecs = sorted
        }

        return SymmetricEigenDecomposition(
            eigenvalues: eigenvalues, eigenvectors: sortedVecs, dimension: n)
    }

    // MARK: - SVD (Hestenes one-sided Jacobi)

    public func svdThin(_ a: [Float], rows m: Int, columns n: Int) throws -> SingularValueDecomposition {
        try LinearAlgebraValidation.validateRect(count: a.count, rows: m, columns: n)

        // Hestenes wants m ≥ n (orthogonalizes columns). For wide matrices,
        // decompose Aᵀ and swap factors: A = UΣVᵀ ⇔ Aᵀ = VΣUᵀ.
        if m < n {
            let t = Self.transpose(a, rows: m, columns: n)        // n×m
            let tDec = try svdThin(t, rows: n, columns: m)
            return SingularValueDecomposition(
                u: Self.transpose(tDec.vt, rows: tDec.rank, columns: m),   // (k×m)ᵀ = m×k
                singularValues: tDec.singularValues,
                vt: Self.transpose(tDec.u, rows: n, columns: tDec.rank),   // (n×k)ᵀ = k×n
                rows: m, columns: n
            )
        }

        var u = a                                    // m×n, columns rotated toward orthogonality
        var v = [Float](repeating: 0, count: n * n)  // accumulates right rotations
        for j in 0..<n { v[j * n + j] = 1 }

        // Rotate while any column pair has |⟨u_p,u_q⟩| > ε‖u_p‖‖u_q‖.
        let eps: Float = 1e-6                        // relative orthogonality target (Float32)
        var converged = (n == 1)
        var sweep = 0
        while !converged && sweep < Self.maxSweeps {
            sweep += 1
            converged = true
            for p in 0..<(n - 1) {
                for q in (p + 1)..<n {
                    var alpha: Float = 0, beta: Float = 0, gamma: Float = 0
                    for i in 0..<m {
                        let up = u[p * m + i], uq = u[q * m + i]
                        alpha += up * up
                        beta += uq * uq
                        gamma += up * uq
                    }
                    if gamma == 0 || gamma * gamma <= eps * eps * alpha * beta { continue }
                    converged = false

                    let zeta = (beta - alpha) / (2 * gamma)
                    let t: Float = (zeta >= 0 ? 1 : -1) / (abs(zeta) + (zeta * zeta + 1).squareRoot())
                    let c = 1 / (t * t + 1).squareRoot()
                    let s = t * c

                    for i in 0..<m {
                        let up = u[p * m + i], uq = u[q * m + i]
                        u[p * m + i] = c * up - s * uq
                        u[q * m + i] = s * up + c * uq
                    }
                    for i in 0..<n {
                        let vp = v[p * n + i], vq = v[q * n + i]
                        v[p * n + i] = c * vp - s * vq
                        v[q * n + i] = s * vp + c * vq
                    }
                }
            }
        }
        guard converged else {
            throw ErrorBuilder(.operationFailed)
                .message("Hestenes SVD did not converge in \(Self.maxSweeps) sweeps (m=\(m), n=\(n))")
                .build()
        }

        // Singular values = column norms; normalize U columns.
        var sv = [Float](repeating: 0, count: n)
        for j in 0..<n {
            var normSq: Float = 0
            for i in 0..<m { normSq += u[j * m + i] * u[j * m + i] }
            let norm = normSq.squareRoot()
            sv[j] = norm
            if norm > 0 {
                let inv = 1 / norm
                for i in 0..<m { u[j * m + i] *= inv }
            }
        }

        // Sort descending (LAPACK convention), permuting U and V columns.
        var order = Array(0..<n)
        order.sort { sv[$0] > sv[$1] }

        var sortedS = [Float](repeating: 0, count: n)
        var sortedU = [Float](repeating: 0, count: m * n)
        var vt = [Float](repeating: 0, count: n * n)
        for (dst, src) in order.enumerated() {
            sortedS[dst] = sv[src]
            for i in 0..<m { sortedU[dst * m + i] = u[src * m + i] }
            // vt row `dst` = V column `src`  ⇒ vt[col j, row dst] = v[src*n + j]
            for j in 0..<n { vt[j * n + dst] = v[src * n + j] }
        }

        return SingularValueDecomposition(
            u: sortedU, singularValues: sortedS, vt: vt, rows: m, columns: n)
    }

    // MARK: - Helpers

    /// Column-major transpose: (rows×columns) → (columns×rows).
    private static func transpose(_ a: [Float], rows: Int, columns: Int) -> [Float] {
        var out = [Float](repeating: 0, count: a.count)
        for j in 0..<columns {
            for i in 0..<rows {
                out[i * columns + j] = a[j * rows + i]
            }
        }
        return out
    }
}
