//
//  LinearAlgebraProviderTests.swift
//  VectorCore
//
//  Correctness + cross-provider parity tests for the LinearAlgebraProvider
//  seam (LAPACK shim and pure-Swift fallback).
//
//  Conventions under test (see LinearAlgebraProvider.swift):
//  - column-major flat [Float] buffers
//  - SVD singular values DESCENDING, eigen eigenvalues ASCENDING
//  - symmetricEigen reads only the LOWER triangle (uplo='L')
//
//  Sign conventions (Q column signs, singular/eigenvector signs) are NOT
//  pinned by either backend, so tests assert backend-free invariants:
//  reconstruction, orthonormality, triangularity, and spectra (sign-free).
//

import Testing
import Foundation
@testable import VectorCore

// MARK: - Deterministic fixtures & dense helpers (column-major)

private enum LA {
    /// SplitMix64 → uniform Float in [-1, 1]; deterministic across platforms.
    static func matrix(seed: UInt64, _ rows: Int, _ columns: Int) -> [Float] {
        var state = seed &+ 0x9E3779B97F4A7C15
        func next() -> Float {
            state &+= 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            z ^= z >> 31
            return Float(z >> 40) / Float(1 << 24) * 2 - 1
        }
        return (0..<(rows * columns)).map { _ in next() }
    }

    static func symmetric(seed: UInt64, _ n: Int) -> [Float] {
        let b = matrix(seed: seed, n, n)
        var a = [Float](repeating: 0, count: n * n)
        for j in 0..<n {
            for i in 0..<n {
                a[j * n + i] = (b[j * n + i] + b[i * n + j]) / 2
            }
        }
        return a
    }

    /// C = A (m×k) · B (k×n), all column-major.
    static func matmul(_ a: [Float], _ b: [Float], m: Int, k: Int, n: Int) -> [Float] {
        var c = [Float](repeating: 0, count: m * n)
        for j in 0..<n {
            for l in 0..<k {
                let blj = b[j * k + l]
                if blj == 0 { continue }
                for i in 0..<m {
                    c[j * m + i] += a[l * m + i] * blj
                }
            }
        }
        return c
    }

    static func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).reduce(0) { Swift.max($0, abs($1.0 - $1.1)) }
    }

    /// Runs `body`, returning the thrown VectorError kind (nil = didn't
    /// throw / wrong error type) — lets tests pin the exact error contract.
    static func thrownKind(_ body: () throws -> Void) -> VectorError.ErrorKind? {
        do {
            try body()
            return nil
        } catch let error as VectorError {
            return error.kind
        } catch {
            return nil
        }
    }

    /// max |GᵀG − I| over a column-major m×n G — orthonormality defect.
    static func orthoDefect(_ g: [Float], rows m: Int, columns n: Int) -> Float {
        var worst: Float = 0
        for j1 in 0..<n {
            for j2 in j1..<n {
                var dot: Float = 0
                for i in 0..<m { dot += g[j1 * m + i] * g[j2 * m + i] }
                let target: Float = j1 == j2 ? 1 : 0
                worst = Swift.max(worst, abs(dot - target))
            }
        }
        return worst
    }

    /// Providers to exercise. The Swift fallback always runs; LAPACK runs
    /// where a backend is compiled in (all Apple platforms).
    static var providers: [(name: String, provider: any LinearAlgebraProvider)] {
        var list: [(String, any LinearAlgebraProvider)] = [("Swift", SwiftLinearAlgebraProvider())]
        if LAPACKLinearAlgebraProvider.isAvailable {
            list.append(("LAPACK", LAPACKLinearAlgebraProvider()))
        }
        return list
    }
}

// MARK: - QR

@Suite("LinearAlgebra - QR")
struct QRThinSuite {

    @Test("Reconstruction, orthonormality, triangularity across shapes")
    func qrInvariants() throws {
        for (name, p) in LA.providers {
            for (m, n) in [(8, 8), (20, 5), (50, 30), (7, 1), (1, 1)] {
                let a = LA.matrix(seed: UInt64(m * 31 + n), m, n)
                let f = try p.qrThin(a, rows: m, columns: n)

                #expect(f.q.count == m * n, "\(name) \(m)x\(n): Q shape")
                #expect(f.r.count == n * n, "\(name) \(m)x\(n): R shape")

                let recon = LA.matmul(f.q, f.r, m: m, k: n, n: n)
                #expect(LA.maxAbsDiff(recon, a) < 1e-4, "\(name) \(m)x\(n): QR ≠ A")
                #expect(LA.orthoDefect(f.q, rows: m, columns: n) < 1e-4, "\(name) \(m)x\(n): QᵀQ ≠ I")

                // R strictly lower triangle must be exactly zero (we construct it).
                for j in 0..<n {
                    for i in (j + 1)..<n {
                        #expect(f.r[j * n + i] == 0, "\(name) \(m)x\(n): R not upper-triangular")
                    }
                }
            }
        }
    }

    @Test("Rank-deficient input still reconstructs")
    func qrRankDeficient() throws {
        let m = 10, n = 4
        var a = LA.matrix(seed: 7, m, n)
        for i in 0..<m { a[2 * m + i] = a[1 * m + i] }   // duplicate column

        for (name, p) in LA.providers {
            let f = try p.qrThin(a, rows: m, columns: n)
            let recon = LA.matmul(f.q, f.r, m: m, k: n, n: n)
            #expect(LA.maxAbsDiff(recon, a) < 1e-4, "\(name): rank-deficient QR ≠ A")
        }
    }

    @Test("Wide matrix (m < n) throws .invalidOperation")
    func qrWideThrows() {
        for (name, p) in LA.providers {
            #expect(LA.thrownKind { _ = try p.qrThin(LA.matrix(seed: 1, 3, 5), rows: 3, columns: 5) }
                == .invalidOperation, "\(name)")
        }
    }

    @Test("Element-count mismatch throws .dimensionMismatch")
    func qrShapeMismatchThrows() {
        for (name, p) in LA.providers {
            #expect(LA.thrownKind { _ = try p.qrThin([1, 2, 3], rows: 4, columns: 2) }
                == .dimensionMismatch, "\(name)")
        }
    }
}

// MARK: - Symmetric eigendecomposition

@Suite("LinearAlgebra - SymmetricEigen")
struct SymmetricEigenSuite {

    @Test("Known 2x2 spectrum: [[2,1],[1,2]] → {1, 3}")
    func knownSpectrum() throws {
        let a: [Float] = [2, 1, 1, 2]   // column-major (symmetric, layout-agnostic)
        for (name, p) in LA.providers {
            let e = try p.symmetricEigen(a, dimension: 2, computeEigenvectors: true)
            #expect(abs(e.eigenvalues[0] - 1) < 1e-5, "\(name): λ₀")
            #expect(abs(e.eigenvalues[1] - 3) < 1e-5, "\(name): λ₁")
        }
    }

    @Test("Random symmetric: AV = VΛ, VᵀV = I, λ ascending")
    func eigenInvariants() throws {
        for (name, p) in LA.providers {
            for n in [2, 10, 50] {
                let a = LA.symmetric(seed: UInt64(n), n)
                let e = try p.symmetricEigen(a, dimension: n, computeEigenvectors: true)
                let v = try #require(e.eigenvectors, "\(name) n=\(n): vectors missing")

                #expect(LA.orthoDefect(v, rows: n, columns: n) < 1e-4, "\(name) n=\(n): VᵀV ≠ I")

                for j in 1..<n {
                    #expect(e.eigenvalues[j - 1] <= e.eigenvalues[j] + 1e-6,
                            "\(name) n=\(n): eigenvalues not ascending")
                }

                // residual max |AV − VΛ|
                let av = LA.matmul(a, v, m: n, k: n, n: n)
                var worst: Float = 0
                for j in 0..<n {
                    for i in 0..<n {
                        worst = max(worst, abs(av[j * n + i] - v[j * n + i] * e.eigenvalues[j]))
                    }
                }
                #expect(worst < 1e-3, "\(name) n=\(n): |AV − VΛ| = \(worst)")
            }
        }
    }

    @Test("Only the lower triangle is read (uplo='L' contract)")
    func lowerTriangleContract() throws {
        let n = 8
        let a = LA.symmetric(seed: 99, n)
        var poisoned = a
        for j in 0..<n {
            for i in 0..<j { poisoned[j * n + i] = 999 }   // strict upper, col-major
        }
        for (name, p) in LA.providers {
            let clean = try p.symmetricEigen(a, dimension: n, computeEigenvectors: false)
            let dirty = try p.symmetricEigen(poisoned, dimension: n, computeEigenvectors: false)
            #expect(LA.maxAbsDiff(clean.eigenvalues, dirty.eigenvalues) < 1e-5,
                    "\(name): upper triangle leaked into result")
        }
    }

    @Test("computeEigenvectors: false returns nil vectors")
    func valuesOnly() throws {
        for (_, p) in LA.providers {
            let e = try p.symmetricEigen(LA.symmetric(seed: 3, 6), dimension: 6, computeEigenvectors: false)
            #expect(e.eigenvectors == nil)
            #expect(e.eigenvalues.count == 6)
        }
    }
}

// MARK: - SVD

@Suite("LinearAlgebra - SVD")
struct SVDThinSuite {

    @Test("Reconstruction + orthonormality across shapes (incl. wide)")
    func svdInvariants() throws {
        for (name, p) in LA.providers {
            for (m, n) in [(8, 8), (30, 10), (10, 30), (5, 1), (1, 5)] {
                let k = min(m, n)
                let a = LA.matrix(seed: UInt64(m * 131 + n), m, n)
                let d = try p.svdThin(a, rows: m, columns: n)

                #expect(d.u.count == m * k && d.vt.count == k * n && d.singularValues.count == k,
                        "\(name) \(m)x\(n): factor shapes")

                for j in 1..<k {
                    #expect(d.singularValues[j - 1] >= d.singularValues[j] - 1e-6,
                            "\(name) \(m)x\(n): singular values not descending")
                }
                #expect(d.singularValues.allSatisfy { $0 >= 0 }, "\(name) \(m)x\(n): negative σ")

                #expect(LA.orthoDefect(d.u, rows: m, columns: k) < 1e-4, "\(name) \(m)x\(n): UᵀU ≠ I")
                // Vᵀ has orthonormal ROWS ⇒ V = (Vᵀ)ᵀ has orthonormal columns;
                // check via Vᵀ·V structure using matmul on (k×n)(n×k).
                var v = [Float](repeating: 0, count: n * k)
                for j in 0..<n {
                    for i in 0..<k { v[i * n + j] = d.vt[j * k + i] }
                }
                #expect(LA.orthoDefect(v, rows: n, columns: k) < 1e-4, "\(name) \(m)x\(n): VVᵀ ≠ I")

                // U · diag(σ) · Vᵀ = A
                var us = d.u
                for j in 0..<k {
                    for i in 0..<m { us[j * m + i] *= d.singularValues[j] }
                }
                let recon = LA.matmul(us, d.vt, m: m, k: k, n: n)
                #expect(LA.maxAbsDiff(recon, a) < 1e-3, "\(name) \(m)x\(n): UΣVᵀ ≠ A")
            }
        }
    }

    @Test("Known diagonal: σ(diag(3,2)) padded = {3, 2}")
    func knownSingularValues() throws {
        // 4×2 column-major: columns [3,0,0,0], [0,2,0,0]
        let a: [Float] = [
            3, 0, 0, 0, // column 0
            0, 2, 0, 0  // column 1
        ]
        for (name, p) in LA.providers {
            let d = try p.svdThin(a, rows: 4, columns: 2)
            #expect(abs(d.singularValues[0] - 3) < 1e-5, "\(name): σ₀")
            #expect(abs(d.singularValues[1] - 2) < 1e-5, "\(name): σ₁")
        }
    }

    @Test("Error contract: shape mismatch and bad dimensions")
    func svdErrorPaths() {
        for (name, p) in LA.providers {
            #expect(LA.thrownKind { _ = try p.svdThin([1, 2, 3], rows: 2, columns: 2) }
                == .dimensionMismatch, "\(name): count mismatch")
            #expect(LA.thrownKind { _ = try p.svdThin([], rows: 0, columns: 3) }
                == .invalidDimension, "\(name): zero rows")
            #expect(LA.thrownKind {
                _ = try p.symmetricEigen([1, 2, 3], dimension: 2, computeEigenvectors: false)
            } == .dimensionMismatch, "\(name): eigen count mismatch")
            #expect(LA.thrownKind {
                _ = try p.symmetricEigen([], dimension: -1, computeEigenvectors: false)
            } == .invalidDimension, "\(name): eigen bad dimension")
        }
    }
}

// MARK: - Cross-provider parity + integration

@Suite("LinearAlgebra - Parity & Integration")
struct LinearAlgebraParitySuite {

    @Test("LAPACK and Swift fallback agree on spectra (sign-free quantities)",
          .enabled(if: LAPACKLinearAlgebraProvider.isAvailable))
    func parity() throws {
        let lapack = LAPACKLinearAlgebraProvider()
        let swift = SwiftLinearAlgebraProvider()

        let (m, n) = (40, 12)
        let a = LA.matrix(seed: 2026, m, n)
        let sL = try lapack.svdThin(a, rows: m, columns: n).singularValues
        let sS = try swift.svdThin(a, rows: m, columns: n).singularValues
        #expect(LA.maxAbsDiff(sL, sS) < 5e-4, "singular value parity: \(LA.maxAbsDiff(sL, sS))")

        let sym = LA.symmetric(seed: 2027, 24)
        let wL = try lapack.symmetricEigen(sym, dimension: 24, computeEigenvectors: false).eigenvalues
        let wS = try swift.symmetricEigen(sym, dimension: 24, computeEigenvectors: false).eigenvalues
        #expect(LA.maxAbsDiff(wL, wS) < 5e-4, "eigenvalue parity: \(LA.maxAbsDiff(wL, wS))")
    }

    @Test("Operations.linearAlgebraProvider default works and is overridable")
    func taskLocalSeam() async throws {
        // Platform-expected default (so the override below is non-vacuous).
        #if canImport(Accelerate)
        #expect(Operations.linearAlgebraProvider is LAPACKLinearAlgebraProvider)
        #else
        #expect(Operations.linearAlgebraProvider is SwiftLinearAlgebraProvider)
        #endif

        // Default provider resolves and factors.
        let a = LA.matrix(seed: 5, 6, 3)
        let f = try Operations.linearAlgebraProvider.qrThin(a, rows: 6, columns: 3)
        #expect(f.q.count == 18)

        // Per-scope override, mirroring the simdProvider TaskLocal pattern.
        try await Operations.$linearAlgebraProvider.withValue(SwiftLinearAlgebraProvider()) {
            #expect(Operations.linearAlgebraProvider is SwiftLinearAlgebraProvider)
            let g = try Operations.linearAlgebraProvider.qrThin(a, rows: 6, columns: 3)
            #expect(LA.orthoDefect(g.q, rows: 6, columns: 3) < 1e-4)
        }
    }
}
