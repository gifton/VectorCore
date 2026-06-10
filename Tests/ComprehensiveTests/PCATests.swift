//
//  PCATests.swift
//  VectorCore
//
//  Correctness tests for PCAModel / Operations.pca (randomized SVD, gap
//  report §3.1) against the LinearAlgebraProvider seam.
//
//  Strategy: every spectral claim is checked against an independent exact
//  computation — symmetricEigen of the explicitly assembled sample
//  covariance — never against PCA's own outputs. Sign conventions ARE
//  pinned by PCAModel (largest-|coordinate| positive), so component
//  comparisons use |cosine| only where the reference's signs are free.
//

import Testing
import Foundation
@testable import VectorCore

// MARK: - Fixtures & helpers

private enum PCAFixtures {
    /// Deterministic anisotropic Gaussian sample: x = diag(scales)·z with
    /// z ~ N(0, I), plus a constant offset so centering is actually tested.
    static func sample(seed: UInt64, count n: Int, scales: [Float], offset: Float = 3) -> [DynamicVector] {
        var gaussian = GaussianSource(seed: seed)
        let d = scales.count
        return (0..<n).map { _ in
            var values = [Float](repeating: 0, count: d)
            for j in 0..<d { values[j] = scales[j] * gaussian.next() + offset }
            return DynamicVector(values)
        }
    }

    /// Sample covariance C = XcᵀXc/(n−1), d×d column-major, built directly
    /// from definition (independent of LAMatMul and the PCA code paths).
    static func covariance(_ vectors: [DynamicVector]) -> (matrix: [Float], mean: [Float]) {
        let n = vectors.count
        let d = vectors[0].scalarCount
        var mean = [Float](repeating: 0, count: d)
        for v in vectors {
            for j in 0..<d { mean[j] += v[j] }
        }
        for j in 0..<d { mean[j] /= Float(n) }
        var cov = [Float](repeating: 0, count: d * d)
        for v in vectors {
            for j in 0..<d {
                let dj = v[j] - mean[j]
                for i in 0..<d {
                    cov[j * d + i] += (v[i] - mean[i]) * dj
                }
            }
        }
        for idx in 0..<(d * d) { cov[idx] /= Float(n - 1) }
        return (cov, mean)
    }

    /// Top-k eigenpairs of the sample covariance (exact, via the seam),
    /// descending. Columns of `vectors` are the reference principal axes.
    static func referenceEigen(_ vectors: [DynamicVector], k: Int) throws -> (values: [Float], axes: [[Float]]) {
        let d = vectors[0].scalarCount
        let cov = covariance(vectors).matrix
        let eigen = try Operations.linearAlgebraProvider.symmetricEigen(
            cov, dimension: d, computeEigenvectors: true)
        let vecs = try #require(eigen.eigenvectors)
        // ssyevd convention: ascending — take the tail, reversed.
        var values = [Float]()
        var axes = [[Float]]()
        for r in 0..<k {
            let column = d - 1 - r
            values.append(eigen.eigenvalues[column])
            axes.append((0..<d).map { vecs[column * d + $0] })
        }
        return (values, axes)
    }

    static func dot(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
    }

    static func componentRow(_ model: PCAModel, _ r: Int) -> [Float] {
        let d = model.inputDimension
        return Array(model.components[(r * d)..<((r + 1) * d)])
    }
}

// MARK: - Spectral correctness

@Suite("PCA - Spectral correctness")
struct PCASpectralSuite {

    @Test("Exact path (sketch ≥ min(n,d)) matches covariance eigendecomposition")
    func exactPathMatchesEigen() throws {
        // d=8, k=3, p=8 → sketch 11 ≥ 8 forces the exact-SVD path.
        let scales: [Float] = [10, 7, 5, 1, 0.6, 0.4, 0.25, 0.1]
        let data = PCAFixtures.sample(seed: 11, count: 240, scales: scales)
        let model = try PCAModel.fit(data, components: 3)
        let reference = try PCAFixtures.referenceEigen(data, k: 3)

        for r in 0..<3 {
            let relativeError = abs(model.explainedVariance[r] - reference.values[r])
                / reference.values[r]
            #expect(relativeError < 1e-3, "λ\(r): \(model.explainedVariance[r]) vs \(reference.values[r])")
            let alignment = abs(PCAFixtures.dot(PCAFixtures.componentRow(model, r), reference.axes[r]))
            #expect(alignment > 0.999, "axis \(r) misaligned: |cos| = \(alignment)")
        }
    }

    @Test("Randomized path matches covariance eigendecomposition on a decaying spectrum")
    func randomizedPathMatchesEigen() throws {
        // d=24, k=4, p=4 → sketch 8 < 24 exercises the genuine sketch path.
        var scales = [Float](repeating: 0.2, count: 24)
        scales.replaceSubrange(0..<6, with: [12, 9, 6, 4, 0.9, 0.5])
        let data = PCAFixtures.sample(seed: 17, count: 400, scales: scales)
        let model = try PCAModel.fit(
            data, components: 4,
            config: PCAConfig(oversampling: 4, powerIterations: 2))
        let reference = try PCAFixtures.referenceEigen(data, k: 4)

        for r in 0..<4 {
            let relativeError = abs(model.explainedVariance[r] - reference.values[r])
                / reference.values[r]
            #expect(relativeError < 0.01, "λ\(r): \(model.explainedVariance[r]) vs \(reference.values[r])")
            let alignment = abs(PCAFixtures.dot(PCAFixtures.componentRow(model, r), reference.axes[r]))
            #expect(alignment > 0.99, "axis \(r) misaligned: |cos| = \(alignment)")
        }
    }

    @Test("Model invariants: orthonormal rows, descending variance, sane ratios, signs")
    func modelInvariants() throws {
        let scales: [Float] = [8, 5, 3, 2, 1.5, 1, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05]
        let data = PCAFixtures.sample(seed: 23, count: 300, scales: scales)
        let model = try PCAModel.fit(data, components: 5)
        let d = model.inputDimension

        for r in 0..<5 {
            let row = PCAFixtures.componentRow(model, r)
            for s in r..<5 {
                let gram = PCAFixtures.dot(row, PCAFixtures.componentRow(model, s))
                let expected: Float = r == s ? 1 : 0
                #expect(abs(gram - expected) < 1e-4, "WWᵀ[\(r),\(s)] = \(gram)")
            }
            // Pinned sign convention: largest-|coordinate| entry is positive.
            var maxAbs: Float = -1
            var signed: Float = 0
            for j in 0..<d where abs(row[j]) > maxAbs {
                maxAbs = abs(row[j])
                signed = row[j]
            }
            #expect(signed > 0, "row \(r) violates the sign convention")
        }
        for r in 1..<5 {
            #expect(model.explainedVariance[r] <= model.explainedVariance[r - 1])
        }
        let ratioSum = model.explainedVarianceRatio.reduce(0, +)
        #expect(ratioSum > 0 && ratioSum <= 1 + 1e-4)
        for r in 0..<5 { #expect(model.explainedVarianceRatio[r] > 0) }
        #expect(model.mean.count == d)
        // Check the offset on the lowest-noise dimension (σ = 0.05): the
        // sample mean's standard error there is 0.05/√300 ≈ 0.003.
        #expect(abs(model.mean[11] - 3) < 0.05, "offset 3 should appear in the mean")
    }

    @Test("Full-rank fit reconstructs the data (projection loses nothing at k = d)")
    func fullRankReconstruction() throws {
        let scales: [Float] = [4, 3, 2, 1, 0.5, 0.25]
        let data = PCAFixtures.sample(seed: 29, count: 100, scales: scales)
        let model = try PCAModel.fit(data, components: 6)
        let projected = try model.transform(data)

        for (vector, y) in zip(data, projected) {
            let d = model.inputDimension
            for j in 0..<d {
                // x̂ = μ + Wᵀy
                var reconstructed = model.mean[j]
                for r in 0..<model.componentCount {
                    reconstructed += model.components[r * d + j] * y[r]
                }
                #expect(abs(reconstructed - vector[j]) < 1e-3)
            }
        }
    }
}

// MARK: - GEMM parity

@Suite("PCA - GEMM parity")
struct LAMatMulParitySuite {

    /// The pure-Swift GEMM is the live path on non-Accelerate platforms but
    /// dead code on Apple CI — pin it against the cblas path here, across
    /// all four transpose combinations and non-square shapes.
    @Test("Swift GEMM matches the platform GEMM for all transpose modes")
    func swiftKernelMatchesPlatform() {
        var gaussian = GaussianSource(seed: 53)
        let (m, n, k) = (7, 5, 9)
        for (transposeA, transposeB) in [(false, false), (true, false), (false, true), (true, true)] {
            let a = gaussian.matrix(count: m * k)
            let b = gaussian.matrix(count: k * n)
            let platform = LAMatMul.multiply(
                a, b, m: m, n: n, k: k, transposeA: transposeA, transposeB: transposeB)
            let swift = LAMatMul.multiplySwift(
                a, b, m: m, n: n, k: k, transposeA: transposeA, transposeB: transposeB)
            for (x, y) in zip(platform, swift) {
                #expect(abs(x - y) < 1e-4, "tA=\(transposeA) tB=\(transposeB)")
            }
        }
    }
}

// MARK: - API behavior

@Suite("PCA - API behavior")
struct PCABehaviorSuite {

    @Test("Operations.pca one-shot == fit then transform; single == batch")
    func oneShotAndSingleConsistency() throws {
        let scales: [Float] = [6, 4, 2, 1, 0.5, 0.3, 0.2, 0.1]
        let data = PCAFixtures.sample(seed: 31, count: 150, scales: scales)

        let (projected, model) = try Operations.pca(data, components: 3)
        let viaModel = try PCAModel.fit(data, components: 3).transform(data)
        #expect(projected == viaModel)

        for i in [0, 7, 149] {
            let single = try model.transform(data[i].toArray())
            for r in 0..<3 {
                #expect(abs(single[r] - projected[i][r]) < 1e-4)
            }
        }
    }

    @Test("Same seed reproduces the model exactly; providers agree on spectra")
    func determinismAndProviderParity() throws {
        var scales = [Float](repeating: 0.3, count: 20)
        scales.replaceSubrange(0..<4, with: [9, 6, 3, 1.5])
        let data = PCAFixtures.sample(seed: 37, count: 250, scales: scales)
        let config = PCAConfig(oversampling: 4, powerIterations: 2, seed: 99)

        let first = try PCAModel.fit(data, components: 3, config: config)
        let second = try PCAModel.fit(data, components: 3, config: config)
        #expect(first.components == second.components)
        #expect(first.explainedVariance == second.explainedVariance)

        let swiftModel = try Operations.$linearAlgebraProvider.withValue(SwiftLinearAlgebraProvider()) {
            try PCAModel.fit(data, components: 3, config: config)
        }
        for r in 0..<3 {
            let difference = abs(first.explainedVariance[r] - swiftModel.explainedVariance[r])
            #expect(difference < 5e-3, "provider spectra diverge at λ\(r): \(difference)")
            let alignment = abs(PCAFixtures.dot(
                PCAFixtures.componentRow(first, r), PCAFixtures.componentRow(swiftModel, r)))
            #expect(alignment > 0.999, "provider axes diverge at \(r): |cos| = \(alignment)")
        }
    }

    @Test("center: false skips the mean and reports raw second moments")
    func uncenteredFit() throws {
        let scales: [Float] = [5, 2, 1, 0.5]
        let data = PCAFixtures.sample(seed: 41, count: 120, scales: scales, offset: 10)
        let model = try PCAModel.fit(
            data, components: 2, config: PCAConfig(center: false))
        #expect(model.mean.allSatisfy { $0 == 0 })
        // With a +10 offset on every dimension the dominant uncentered
        // direction is the offset itself, far above any scaled axis.
        #expect(model.explainedVariance[0] > 50)
    }

    @Test("Error contracts: bad k, ragged input, tiny n, transform mismatch, bad config")
    func errorContracts() throws {
        let data = PCAFixtures.sample(seed: 43, count: 10, scales: [1, 1, 1, 1])

        #expect(throws: VectorError.self) { try PCAModel.fit(data, components: 0) }
        #expect(throws: VectorError.self) { try PCAModel.fit(data, components: 5) }   // k > d
        let small = Array(data.prefix(3))
        #expect(throws: VectorError.self) { try PCAModel.fit(small, components: 3) }  // k > n−1 = 2
        #expect(throws: VectorError.self) { try PCAModel.fit([data[0]], components: 1) } // n < 2

        var ragged = data
        ragged[3] = DynamicVector([1, 2, 3])
        #expect(throws: VectorError.self) { try PCAModel.fit(ragged, components: 2) }

        let model = try PCAModel.fit(data, components: 2)
        #expect(throws: VectorError.self) { try model.transform([1, 2, 3]) }
        #expect(throws: VectorError.self) {
            try PCAModel.fit(data, components: 2, config: PCAConfig(oversampling: -1))
        }
    }
}
