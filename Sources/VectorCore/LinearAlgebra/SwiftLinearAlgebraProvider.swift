// Householder QR, cyclic symmetric Jacobi eigen, and one-sided Jacobi SVD.
// Double working storage preserves the Float input exponent range through
// normalization. LinearAlgebraRangeTests.sharedExponentSweep exercises the published Float boundary.
// Exact zero SVD working columns do not receive a completed null-space basis:
// this affects U for tall/square inputs and Vt rows for wide inputs.
import Foundation

public struct SwiftLinearAlgebraProvider: LinearAlgebraProvider {

    // Exhaustion throws; convergence for arbitrary conditioning is not promised.
    private static let maxSweeps = 60

    public init() {}

    // MARK: - QR (Householder)

    public func qrThin(_ a: [Float], rows m: Int, columns n: Int) throws -> QRFactorization {
        try LinearAlgebraValidation.validateQR(count: a.count, rows: m, columns: n)
        try LinearAlgebraValidation.validateFinite(a, operation: "qrThin")

        var fac = a.map(Double.init)                                  // factored form: R upper, reflectors below
        var tau = [Double](repeating: 0, count: n)

        for j in 0..<n {
            // Householder vector for column j, rows j..<m.
            let norm = try Self.norm(fac[(j * m + j)..<((j + 1) * m)], operation: "qrThin")
            if norm == 0 { tau[j] = 0; continue }    // zero column: identity reflector

            let alpha = fac[j * m + j]
            let beta: Double = alpha >= 0 ? -norm : norm
            tau[j] = (beta - alpha) / beta
            let denominator = alpha - beta
            for i in (j + 1)..<m { fac[j * m + i] /= denominator }
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
        var r = [Double](repeating: 0, count: n * n)
        for j in 0..<n {
            for i in 0...j { r[j * n + i] = fac[j * m + i] }
        }

        // Accumulate thin Q: apply H_{n-1}…H_0 to the first n identity columns,
        // backward so each reflector touches only rows/cols it acts on.
        var q = [Double](repeating: 0, count: m * n)
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

        return QRFactorization(q: try Self.output(q, operation: "qrThin"),
                               r: try Self.output(r, operation: "qrThin"), rows: m, columns: n)
    }

    // MARK: - Symmetric eigen (cyclic Jacobi)

    // swiftlint:disable:next cyclomatic_complexity
    public func symmetricEigen(
        _ a: [Float],
        dimension n: Int,
        computeEigenvectors: Bool
    ) throws -> SymmetricEigenDecomposition {
        try LinearAlgebraValidation.validateSquare(count: a.count, dimension: n)

        try LinearAlgebraValidation.validateLowerFinite(a, dimension: n)
        // Never copy ignored upper-triangle values into the working matrix.
        var mat = [Double](repeating: 0, count: a.count)
        for j in 0..<n {
            for i in j..<n {
                mat[j * n + i] = Double(a[j * n + i])
                mat[i * n + j] = Double(a[j * n + i])
            }
        }

        var vecs: [Double] = []
        if computeEigenvectors {
            vecs = [Double](repeating: 0, count: n * n)
            for j in 0..<n { vecs[j * n + j] = 1 }
        }

        let negligible = Self.epsilon(n) * (try Self.norm(mat[...], operation: "symmetricEigen"))
        var converged = Self.offDiagonalMaximum(mat, n: n) <= negligible
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
                    let t = try Self.rotation(delta: aqq - app, twiceOffDiagonal: 2 * apq,
                                              operation: "symmetricEigen")
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
        try Self.finite(mat, operation: "symmetricEigen")
        converged = Self.offDiagonalMaximum(mat, n: n) <= negligible
        guard converged else {
            throw ErrorBuilder(.operationFailed)
                .message("Jacobi eigendecomposition did not converge in \(Self.maxSweeps) sweeps (n=\(n))")
                .build()
        }

        // Sort ascending (ssyevd convention), permuting eigenvector columns.
        var order = Array(0..<n)
        order.sort { mat[$0 * n + $0] < mat[$1 * n + $1] }

        var eigenvalues = [Double](repeating: 0, count: n)
        for (dst, src) in order.enumerated() { eigenvalues[dst] = mat[src * n + src] }

        var sortedVecs: [Float]?
        if computeEigenvectors {
            var sorted = [Double](repeating: 0, count: n * n)
            for (dst, src) in order.enumerated() {
                for i in 0..<n { sorted[dst * n + i] = vecs[src * n + i] }
            }
            sortedVecs = try Self.output(sorted, operation: "symmetricEigen")
        }

        return SymmetricEigenDecomposition(
            eigenvalues: try Self.output(eigenvalues, operation: "symmetricEigen"), eigenvectors: sortedVecs, dimension: n)
    }

    // MARK: - SVD (Hestenes one-sided Jacobi)

    // swiftlint:disable:next cyclomatic_complexity
    public func svdThin(_ a: [Float], rows m: Int, columns n: Int) throws -> SingularValueDecomposition {
        try LinearAlgebraValidation.validateRect(count: a.count, rows: m, columns: n)

        try LinearAlgebraValidation.validateFinite(a, operation: "svdThin")
        let result = try Self.svdWide(a.map(Double.init), rows: m, columns: n)
        return SingularValueDecomposition(
            u: try Self.output(result.u, operation: "svdThin"),
            singularValues: try Self.output(result.s, operation: "svdThin"),
            vt: try Self.output(result.vt, operation: "svdThin"), rows: m, columns: n)
    }

    private static func svdWide(_ a: [Double], rows m: Int, columns n: Int)
        throws -> (u: [Double], s: [Double], vt: [Double]) {
        // Hestenes wants m ≥ n (orthogonalizes columns). For wide matrices,
        // decompose Aᵀ and swap factors: A = UΣVᵀ ⇔ Aᵀ = VΣUᵀ.
        if m < n {
            let t = Self.transpose(a, rows: m, columns: n)        // n×m
            let tDec = try svdWide(t, rows: n, columns: m)
            let k = tDec.s.count                     // = min(m, n)
            return (Self.transpose(tDec.vt, rows: k, columns: m), tDec.s,
                    Self.transpose(tDec.u, rows: n, columns: k))
        }

        var u = a                                    // m×n, columns rotated toward orthogonality
        var v = [Double](repeating: 0, count: n * n)  // accumulates right rotations
        for j in 0..<n { v[j * n + j] = 1 }

        let eps = Self.epsilon(m)
        var converged = (n == 1)
        var sweep = 0
        while !converged && sweep < Self.maxSweeps {
            sweep += 1
            converged = true
            for p in 0..<(n - 1) {
                for q in (p + 1)..<n {
                    var alpha: Double = 0, beta: Double = 0, gamma: Double = 0
                    for i in 0..<m {
                        let up = u[p * m + i], uq = u[q * m + i]
                        alpha += up * up
                        beta += uq * uq
                        gamma += up * uq
                    }
                    guard alpha.isFinite, beta.isFinite, gamma.isFinite else { throw Self.failure("svdThin") }
                    if alpha == 0 || beta == 0 || abs(gamma) / sqrt(alpha) / sqrt(beta) <= eps { continue }
                    converged = false

                    let t = try Self.rotation(delta: beta - alpha, twiceOffDiagonal: 2 * gamma,
                                              operation: "svdThin")
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
        try Self.finite(u, operation: "svdThin")
        guard converged else {
            throw ErrorBuilder(.operationFailed)
                .message("Hestenes SVD did not converge in \(Self.maxSweeps) sweeps (m=\(m), n=\(n))")
                .build()
        }

        // Singular values = column norms; normalize U columns.
        var sv = [Double](repeating: 0, count: n)
        for j in 0..<n {
            let norm = try Self.norm(u[(j * m)..<((j + 1) * m)], operation: "svdThin")
            sv[j] = norm
            if norm > 0 {
                for i in 0..<m { u[j * m + i] /= norm }
            }
        }

        // Sort descending (LAPACK convention), permuting U and V columns.
        var order = Array(0..<n)
        order.sort { sv[$0] > sv[$1] }

        var sortedS = [Double](repeating: 0, count: n)
        var sortedU = [Double](repeating: 0, count: m * n)
        var vt = [Double](repeating: 0, count: n * n)
        for (dst, src) in order.enumerated() {
            sortedS[dst] = sv[src]
            for i in 0..<m { sortedU[dst * m + i] = u[src * m + i] }
            // vt row `dst` = V column `src`  ⇒ vt[col j, row dst] = v[src*n + j]
            for j in 0..<n { vt[j * n + dst] = v[src * n + j] }
        }

        return (sortedU, sortedS, vt)
    }

    // MARK: - Helpers

    /// Column-major transpose: (rows×columns) → (columns×rows).
    private static func transpose(_ a: [Double], rows: Int, columns: Int) -> [Double] {
        var out = [Double](repeating: 0, count: a.count)
        for j in 0..<columns {
            for i in 0..<rows {
                out[i * columns + j] = a[j * rows + i]
            }
        }
        return out
    }

    private static func epsilon(_ dimension: Int) -> Double {
        let du = Double(dimension) * (Double.ulpOfOne / 2)
        return max(1e-12, 8 * du / (1 - du))
    }

    private static func failure(_ operation: String) -> VectorError {
        ErrorBuilder(.operationFailed).message("\(operation) produced a nonfinite working value or an unrepresentable Float result").build()
    }

    private static func finite(_ values: [Double], operation: String) throws {
        guard values.allSatisfy(\.isFinite) else { throw failure(operation) }
    }

    private static func output(_ values: [Double], operation: String) throws -> [Float] {
        try finite(values, operation: operation)
        let result = values.map(Float.init)
        try LinearAlgebraValidation.validateOutput(result, operation: operation)
        return result
    }

    private static func norm(_ values: ArraySlice<Double>, operation: String) throws -> Double {
        var scale = 0.0, sum = 1.0
        for value in values {
            guard value.isFinite else { throw failure(operation) }
            let magnitude = abs(value)
            if magnitude == 0 { continue }
            if scale < magnitude {
                let ratio = scale / magnitude
                sum = 1 + sum * ratio * ratio
                scale = magnitude
            } else {
                let ratio = magnitude / scale
                sum += ratio * ratio
            }
        }
        let result = scale * sqrt(sum)
        guard result.isFinite else { throw failure(operation) }
        return result
    }

    // The smaller root, evaluated after scaling both arguments, avoids an
    // unbounded theta or theta². delta==0 chooses the positive unit root.
    private static func rotation(delta: Double, twiceOffDiagonal off: Double,
                                 operation: String) throws -> Double {
        guard delta.isFinite, off.isFinite, off != 0 else { throw failure(operation) }
        if delta == 0 { return 1 }
        let scale = max(abs(delta), abs(off))
        let d = delta / scale, b = off / scale
        let magnitude = abs(b) / (abs(d) + hypot(d, b))
        return (delta.sign == off.sign ? 1 : -1) * magnitude
    }

    private static func offDiagonalMaximum(_ matrix: [Double], n: Int) -> Double {
        var result = 0.0
        for j in 0..<n {
            for i in 0..<j {
                let value = abs(matrix[j * n + i])
                if !value.isFinite { return .infinity }
                result = max(result, value)
            }
        }
        return result
    }
}
