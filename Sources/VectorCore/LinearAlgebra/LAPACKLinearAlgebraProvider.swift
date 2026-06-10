//
//  LAPACKLinearAlgebraProvider.swift
//  VectorCore
//
//  LAPACK-backed LinearAlgebraProvider, routed through the VectorCoreC
//  vc_lapack shim (Accelerate's modern LAPACK, ACCELERATE_NEW_LAPACK).
//
//  Why the shim exists (vs. importing Accelerate here): the modern LAPACK
//  interface is macro-gated at C-header level; defining it from Swift would
//  require `unsafeFlags`. See Sources/VectorCoreC/include/vc_lapack.h.
//
//  Availability is a runtime probe (`vc_lapack_available()`), so this type
//  compiles on every platform and fails with a typed error rather than a
//  missing symbol on platforms without a backend. Use
//  `LAPACKLinearAlgebraProvider.isAvailable` to branch ahead of time.
//

import VectorCoreC

public struct LAPACKLinearAlgebraProvider: LinearAlgebraProvider {

    public init() {}

    /// Whether a LAPACK backend is compiled in on this platform.
    public static var isAvailable: Bool { vc_lapack_available() != 0 }

    // MARK: - QR

    public func qrThin(_ a: [Float], rows: Int, columns: Int) throws -> QRFactorization {
        try LinearAlgebraValidation.validateQR(count: a.count, rows: rows, columns: columns)
        try Self.ensureAvailable()

        let m = Int32(rows), n = Int32(columns)
        var work = a                       // sgeqrf factors in place
        var tau = [Float](repeating: 0, count: columns)

        let geqrfInfo = work.withUnsafeMutableBufferPointer { wp in
            tau.withUnsafeMutableBufferPointer { tp in
                vc_lapack_sgeqrf(m, n, wp.baseAddress!, m, tp.baseAddress!)
            }
        }
        try Self.check(geqrfInfo, routine: "sgeqrf")

        // Extract R (n×n upper triangle) before sorgqr overwrites `work` with Q.
        var r = [Float](repeating: 0, count: columns * columns)
        for j in 0..<columns {
            for i in 0...j {
                r[j * columns + i] = work[j * rows + i]
            }
        }

        // Form explicit thin Q (m×n) from the n reflectors.
        let orgqrInfo = work.withUnsafeMutableBufferPointer { wp in
            tau.withUnsafeBufferPointer { tp in
                vc_lapack_sorgqr(m, n, n, wp.baseAddress!, m, tp.baseAddress!)
            }
        }
        try Self.check(orgqrInfo, routine: "sorgqr")

        return QRFactorization(q: work, r: r, rows: rows, columns: columns)
    }

    // MARK: - SVD

    public func svdThin(_ a: [Float], rows: Int, columns: Int) throws -> SingularValueDecomposition {
        try LinearAlgebraValidation.validateRect(count: a.count, rows: rows, columns: columns)
        try Self.ensureAvailable()

        let m = Int32(rows), n = Int32(columns)
        let k = Swift.min(rows, columns)
        var work = a                       // sgesdd destroys A
        var s = [Float](repeating: 0, count: k)
        var u = [Float](repeating: 0, count: rows * k)
        var vt = [Float](repeating: 0, count: k * columns)

        let info = work.withUnsafeMutableBufferPointer { ap in
            s.withUnsafeMutableBufferPointer { sp in
                u.withUnsafeMutableBufferPointer { up in
                    vt.withUnsafeMutableBufferPointer { vtp in
                        vc_lapack_sgesdd(
                            m, n,
                            ap.baseAddress!, m,
                            sp.baseAddress!,
                            up.baseAddress!, m,
                            vtp.baseAddress!, Int32(k)
                        )
                    }
                }
            }
        }
        try Self.check(info, routine: "sgesdd")

        return SingularValueDecomposition(u: u, singularValues: s, vt: vt, rows: rows, columns: columns)
    }

    // MARK: - Symmetric eigen

    public func symmetricEigen(
        _ a: [Float],
        dimension: Int,
        computeEigenvectors: Bool
    ) throws -> SymmetricEigenDecomposition {
        try LinearAlgebraValidation.validateSquare(count: a.count, dimension: dimension)
        try Self.ensureAvailable()

        let n = Int32(dimension)
        var work = a                       // ssyevd overwrites A with eigenvectors (jobz='V')
        var w = [Float](repeating: 0, count: dimension)
        let jobz: CChar = computeEigenvectors ? 86 /* 'V' */ : 78 /* 'N' */

        let info = work.withUnsafeMutableBufferPointer { ap in
            w.withUnsafeMutableBufferPointer { wp in
                vc_lapack_ssyevd(jobz, 76 /* 'L' */, n, ap.baseAddress!, n, wp.baseAddress!)
            }
        }
        try Self.check(info, routine: "ssyevd")

        return SymmetricEigenDecomposition(
            eigenvalues: w,
            eigenvectors: computeEigenvectors ? work : nil,
            dimension: dimension
        )
    }

    // MARK: - Error mapping

    private static func ensureAvailable() throws {
        guard isAvailable else {
            throw ErrorBuilder(.resourceUnavailable)
                .message("No LAPACK backend on this platform; use SwiftLinearAlgebraProvider")
                .build()
        }
    }

    /// Mirrors VC_LAPACK_UNAVAILABLE in vc_lapack.h. Re-declared here because
    /// parenthesized negative macros don't reliably cross the Clang importer.
    private static let unavailableCode: Int32 = -1000

    private static func check(_ info: Int32, routine: String) throws {
        guard info != 0 else { return }
        if info == Self.unavailableCode {
            throw ErrorBuilder(.resourceUnavailable)
                .message("LAPACK shim unavailable or workspace allocation failed in \(routine)")
                .build()
        }
        if info < 0 {
            // Illegal argument — indicates a bug in this provider, not user input.
            throw VectorError.invalidOperation(
                routine, reason: "illegal argument at position \(-info) (provider bug)")
        }
        // info > 0: routine-specific algorithmic failure (e.g. D&C non-convergence).
        throw ErrorBuilder(.operationFailed)
            .message("\(routine) failed to converge (info=\(info))")
            .build()
    }
}
