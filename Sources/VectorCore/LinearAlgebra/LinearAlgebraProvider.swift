//
//  LinearAlgebraProvider.swift
//  VectorCore
//
//  Dense linear-algebra seam: QR, thin SVD, symmetric eigendecomposition.
//
//  This is the foundational enabler for projection primitives (PCA /
//  randomized SVD; see Docs/gap-analysis-hn-semantic-search.md §3.0–3.1).
//  It mirrors the SIMDProvider pattern: a protocol seam with a LAPACK-backed
//  implementation on Apple platforms (via the VectorCoreC vc_lapack shim)
//  and a pure-Swift fallback everywhere else.
//
//  ## Layout contract — COLUMN-MAJOR
//
//  All matrices cross this seam as flat `[Float]` in COLUMN-MAJOR order
//  (element (i,j) of an m×n matrix lives at index `j*m + i`), matching
//  LAPACK's native layout so the fast path is copy-free in orientation.
//  This is an internal-facing seam; higher-level APIs (e.g. the future
//  `Operations.pca`) own the translation from row-major vector collections.
//
//  ## Scale contract
//
//  These are SMALL-PANEL routines: covariance/Gram matrices (d ≤ a few
//  thousand) and randomized-SVD sketch panels (n × (k+p), k+p ≤ ~100).
//  They are deliberately not streamed or tiled — the streaming happens a
//  level above, in the GEMM passes that build the panels.
//

// MARK: - Result types

/// Thin QR factorization: `A (m×n, m ≥ n) = Q (m×n) · R (n×n)`.
public struct QRFactorization: Sendable {
    /// Orthonormal factor, m×n, column-major. Columns satisfy QᵀQ = I.
    public let q: [Float]
    /// Upper-triangular factor, n×n, column-major.
    public let r: [Float]
    /// Row count of A (and Q).
    public let rows: Int
    /// Column count of A (and Q; also the dimension of R).
    public let columns: Int

    public init(q: [Float], r: [Float], rows: Int, columns: Int) {
        self.q = q
        self.r = r
        self.rows = rows
        self.columns = columns
    }
}

/// Thin singular value decomposition: `A (m×n) = U (m×k) · diag(s) · Vᵀ (k×n)`
/// with `k = min(m, n)`.
public struct SingularValueDecomposition: Sendable {
    /// Left singular vectors, m×k, column-major, orthonormal columns.
    ///
    /// Caveat: for exactly rank-deficient inputs, the pure-Swift fallback
    /// does not complete U columns paired with zero singular values to an
    /// orthonormal basis (LAPACK does). See SwiftLinearAlgebraProvider.
    public let u: [Float]
    /// Singular values, length k, in DESCENDING order (LAPACK convention).
    public let singularValues: [Float]
    /// Right singular vectors transposed, k×n, column-major, orthonormal rows.
    public let vt: [Float]
    /// Row count of A.
    public let rows: Int
    /// Column count of A.
    public let columns: Int

    public init(u: [Float], singularValues: [Float], vt: [Float], rows: Int, columns: Int) {
        self.u = u
        self.singularValues = singularValues
        self.vt = vt
        self.rows = rows
        self.columns = columns
    }
}

/// Symmetric eigendecomposition: `A (n×n, symmetric) = V · diag(λ) · Vᵀ`.
public struct SymmetricEigenDecomposition: Sendable {
    /// Eigenvalues, length n, in ASCENDING order (LAPACK ssyevd convention).
    /// Note this is the opposite of `SingularValueDecomposition.singularValues`.
    public let eigenvalues: [Float]
    /// Eigenvectors, n×n column-major (column j pairs with eigenvalues[j]),
    /// or `nil` when vectors were not requested.
    public let eigenvectors: [Float]?
    /// Matrix dimension.
    public let dimension: Int

    public init(eigenvalues: [Float], eigenvectors: [Float]?, dimension: Int) {
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.dimension = dimension
    }
}

// MARK: - Provider protocol

/// Seam for dense factorizations consumed by projection primitives.
///
/// Implementations: `LAPACKLinearAlgebraProvider` (Accelerate's modern LAPACK
/// via the VectorCoreC shim; Apple platforms) and
/// `SwiftLinearAlgebraProvider` (pure Swift Householder/Jacobi; everywhere).
///
/// All inputs/outputs are column-major flat buffers — see the layout contract
/// in this file's header.
public protocol LinearAlgebraProvider: Sendable {

    /// Thin QR of an m×n matrix with m ≥ n.
    ///
    /// - Parameters:
    ///   - a: m×n matrix, column-major, `a.count == rows * columns`.
    ///   - rows: m.
    ///   - columns: n. Requires `rows >= columns >= 1`.
    /// - Throws: `VectorError` — `.invalidOperation` for wide matrices,
    ///   `.dimensionMismatch`/`.invalidDimension` on bad shape,
    ///   `.operationFailed` on backend failure.
    func qrThin(_ a: [Float], rows: Int, columns: Int) throws -> QRFactorization

    /// Thin SVD of an arbitrary m×n matrix.
    ///
    /// - Throws: `VectorError` (.operationFailed if the algorithm fails to
    ///   converge — LAPACK info > 0, or Jacobi sweep exhaustion in fallback).
    func svdThin(_ a: [Float], rows: Int, columns: Int) throws -> SingularValueDecomposition

    /// Eigendecomposition of a symmetric n×n matrix.
    ///
    /// Only the LOWER triangle of `a` is required to be populated (both
    /// implementations reference uplo='L'); the strict upper triangle is
    /// ignored. Eigenvalues return ascending.
    func symmetricEigen(_ a: [Float], dimension: Int, computeEigenvectors: Bool) throws -> SymmetricEigenDecomposition
}

// MARK: - Operations integration

extension Operations {
    /// The active dense linear-algebra provider, mirroring `simdProvider`.
    ///
    /// Defaults to the LAPACK-backed provider on Apple platforms and the
    /// pure-Swift provider elsewhere. Override per-scope via:
    /// ```swift
    /// Operations.$linearAlgebraProvider.withValue(SwiftLinearAlgebraProvider()) {
    ///     // deterministic cross-platform path, e.g. in parity tests
    /// }
    /// ```
    #if canImport(Accelerate)
    @TaskLocal public static var linearAlgebraProvider: any LinearAlgebraProvider = LAPACKLinearAlgebraProvider()
    #else
    @TaskLocal public static var linearAlgebraProvider: any LinearAlgebraProvider = SwiftLinearAlgebraProvider()
    #endif
}

// MARK: - Shared validation

internal enum LinearAlgebraValidation {
    /// Both backends use 32-bit LAPACK indices (see vc_lapack.h's integer
    /// model note); reject out-of-range dims with a typed error rather than
    /// letting `Int32(_:)` trap in the provider.
    private static let maxLAPACKDimension = Int(Int32.max)

    private static func validatePositiveAndRepresentable(_ rows: Int, _ columns: Int) throws {
        guard rows >= 1, columns >= 1 else {
            throw VectorError.invalidDimension(rows <= 0 ? rows : columns, reason: "matrix dimensions must be positive")
        }
        guard rows <= maxLAPACKDimension, columns <= maxLAPACKDimension else {
            throw VectorError.invalidDimension(
                Swift.max(rows, columns),
                reason: "exceeds the 32-bit LAPACK index model (see vc_lapack.h)")
        }
    }

    static func validateQR(count: Int, rows: Int, columns: Int) throws {
        try validatePositiveAndRepresentable(rows, columns)
        guard rows >= columns else {
            throw VectorError.invalidOperation(
                "qrThin", reason: "thin QR requires rows (\(rows)) >= columns (\(columns)); transpose or use svdThin")
        }
        guard count == rows * columns else {
            throw VectorError.dimensionMismatch(expected: rows * columns, actual: count)
        }
    }

    static func validateRect(count: Int, rows: Int, columns: Int) throws {
        try validatePositiveAndRepresentable(rows, columns)
        guard count == rows * columns else {
            throw VectorError.dimensionMismatch(expected: rows * columns, actual: count)
        }
    }

    static func validateSquare(count: Int, dimension: Int) throws {
        try validatePositiveAndRepresentable(dimension, dimension)
        guard count == dimension * dimension else {
            throw VectorError.dimensionMismatch(expected: dimension * dimension, actual: count)
        }
    }
}
