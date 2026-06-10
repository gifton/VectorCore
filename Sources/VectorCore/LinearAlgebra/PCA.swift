//
//  PCA.swift
//  VectorCore
//
//  Principal Component Analysis via randomized SVD (Halko–Martinsson–Tropp,
//  "Finding Structure with Randomness", 2011). Gap analysis §3.1: the
//  384→~50 pre-reduction feeding UMAP, and the projection primitive for the
//  2-D corpus scatter.
//
//  ## Why randomized rather than full SVD
//
//  Cost is a few GEMM passes over the data — O(n·d·ℓ) with ℓ = k + p — plus
//  factorizations of small panels only (n×ℓ QR, ℓ×d SVD). No n×n or full
//  n×d factorization is ever formed, so fitting on a 100–500k sample is
//  GEMM-bound and fast. When ℓ ≥ min(n, d) the sketch cannot help; the
//  implementation falls back to one exact thin SVD of the centered sample.
//
//  ## Usage at corpus scale
//
//  `PCAModel.fit` on a representative sample, then `transform` the full
//  corpus in caller-sized batches (one GEMM per batch) — the model is an
//  immutable value and `transform` is pure, so batches parallelize freely.
//
//  ## Layout
//
//  Internally everything is COLUMN-MAJOR (the LinearAlgebraProvider/LAMatMul
//  contract). The public surface speaks vectors and row-major `components`;
//  the column-major identity (row-major k×d ≡ column-major d×k transposed)
//  makes `transform`'s GEMM copy-free over the components buffer.
//

import Foundation

// MARK: - Configuration

/// Tuning knobs for randomized-SVD PCA. The defaults follow Halko et al.'s
/// recommendations and are appropriate for embedding workloads.
public struct PCAConfig: Sendable {
    /// Sketch oversampling `p`: the sketch width is `k + p` columns.
    /// 5–10 suffices when singular values decay; raise for flat spectra.
    public var oversampling: Int

    /// Power (subspace) iterations `q`. Each costs two extra GEMM passes and
    /// sharpens accuracy on slowly decaying spectra; 1–2 is standard.
    public var powerIterations: Int

    /// Subtract the per-dimension mean before factorization (classical PCA).
    /// With `false` this computes truncated SVD of the raw data and
    /// `explainedVariance` reads as raw second moments, not variances.
    public var center: Bool

    /// Seed for the Gaussian sketch. Fits are fully deterministic for a
    /// given (data, config) pair — including across the LAPACK and Swift
    /// providers up to floating-point noise.
    public var seed: UInt64

    public init(
        oversampling: Int = 8,
        powerIterations: Int = 2,
        center: Bool = true,
        seed: UInt64 = 0x5EED_CAFE
    ) {
        self.oversampling = oversampling
        self.powerIterations = powerIterations
        self.center = center
        self.seed = seed
    }
}

// MARK: - Model

/// A fitted PCA projection: `y = W · (x − μ)` with orthonormal rows `W`.
///
/// Fit on a sample with `fit`, then apply with `transform` — see the header
/// note on corpus-scale batching.
public struct PCAModel: Sendable {
    /// Per-dimension mean μ of the fit data (zeros when fit uncentered).
    public let mean: [Float]

    /// Principal axes `W`, k×d ROW-MAJOR: `components[r*d + j]` is
    /// coordinate j of axis r. Rows are orthonormal and ordered by
    /// decreasing explained variance. Sign convention: each row's
    /// largest-magnitude coordinate is positive (first index on ties).
    public let components: [Float]

    /// Variance captured per axis, descending: σᵣ² / (n − 1).
    public let explainedVariance: [Float]

    /// `explainedVariance` as a fraction of the fit data's total variance.
    public let explainedVarianceRatio: [Float]

    /// d — dimension of the input space.
    public let inputDimension: Int

    /// k — dimension of the projected space.
    public let componentCount: Int

    // MARK: Fit

    /// Fits a k-component PCA on `vectors` (the sample) via randomized SVD.
    ///
    /// - Throws: `VectorError.invalidDimension` for k outside
    ///   `1...min(n − 1, d)` (n − 1: centering loses one rank) or invalid
    ///   config; `.dimensionMismatch` for ragged input;
    ///   `.invalidOperation` for n < 2; plus anything the active
    ///   `Operations.linearAlgebraProvider` throws.
    public static func fit<V: VectorProtocol>(
        _ vectors: [V],
        components k: Int,
        config: PCAConfig = PCAConfig()
    ) throws -> PCAModel where V.Scalar == Float {
        let n = vectors.count
        guard n >= 2 else {
            throw VectorError.invalidOperation(
                "PCAModel.fit", reason: "PCA requires at least 2 vectors, got \(n)")
        }
        let d = vectors[0].scalarCount
        let flat = try flattenColumnMajor(vectors, count: n, dimension: d)
        return try fit(columnMajor: flat, rows: n, dimension: d, components: k, config: config)
    }

    /// Core fit on an n×d column-major buffer (consumed and mutated).
    internal static func fit(
        columnMajor a: [Float],
        rows n: Int,
        dimension d: Int,
        components k: Int,
        config: PCAConfig
    ) throws -> PCAModel {
        guard config.oversampling >= 0, config.powerIterations >= 0 else {
            throw VectorError.invalidDimension(
                min(config.oversampling, config.powerIterations),
                reason: "oversampling and powerIterations must be non-negative")
        }
        let maxRank = min(config.center ? n - 1 : n, d)
        guard k >= 1, k <= maxRank else {
            throw VectorError.invalidDimension(
                k, reason: "components must be in 1...\(maxRank) (n=\(n), d=\(d), centered=\(config.center))")
        }

        var a = a

        // Center and measure total variance in one pass. Column j is
        // contiguous (length n); Double accumulation keeps the mean and the
        // variance stable at sample sizes in the 10⁵–10⁶ range.
        var mean = [Float](repeating: 0, count: d)
        var totalVariance: Double = 0
        let varianceDenominator = Double(max(n - 1, 1))
        a.withUnsafeMutableBufferPointer { ab in
            for j in 0..<d {
                let base = j * n
                if config.center {
                    var sum: Double = 0
                    for i in 0..<n { sum += Double(ab[base + i]) }
                    let mu = Float(sum / Double(n))
                    mean[j] = mu
                    for i in 0..<n { ab[base + i] -= mu }
                }
                var sumSq: Double = 0
                for i in 0..<n { sumSq += Double(ab[base + i]) * Double(ab[base + i]) }
                totalVariance += sumSq / varianceDenominator
            }
        }

        let provider = Operations.linearAlgebraProvider
        let sketchWidth = min(k + config.oversampling, min(n, d))

        // The ℓ×d panel whose SVD yields the principal axes: either the
        // projected sketch B = QᵀA (randomized path) or A itself (exact
        // path, when the sketch could not be thinner than the data).
        let panel: [Float]
        let panelRows: Int
        if sketchWidth >= min(n, d) {
            panel = a
            panelRows = n
        } else {
            var gaussian = GaussianSource(seed: config.seed)
            let omega = gaussian.matrix(count: d * sketchWidth)

            // Y = A·Ω, then q rounds of QR-stabilized subspace iteration:
            // Q = qr(Y).q;  Y ← A·qr(AᵀQ).q — re-orthonormalizing between
            // multiplications prevents the sketch from collapsing onto the
            // dominant axis (the classic power-iteration failure mode).
            var y = LAMatMul.multiply(a, omega, m: n, n: sketchWidth, k: d)
            for _ in 0..<config.powerIterations {
                let q = try provider.qrThin(y, rows: n, columns: sketchWidth).q
                let z = LAMatMul.multiply(a, q, m: d, n: sketchWidth, k: n, transposeA: true)
                let qz = try provider.qrThin(z, rows: d, columns: sketchWidth).q
                y = LAMatMul.multiply(a, qz, m: n, n: sketchWidth, k: d)
            }
            let q = try provider.qrThin(y, rows: n, columns: sketchWidth).q
            panel = LAMatMul.multiply(q, a, m: sketchWidth, n: d, k: n, transposeA: true)
            panelRows = sketchWidth
        }

        let svd = try provider.svdThin(panel, rows: panelRows, columns: d)
        let panelRank = min(panelRows, d) // rows of vt

        // Extract the top-k right singular vectors (rows of vt) into
        // row-major components, applying the sign convention.
        var componentsRowMajor = [Float](repeating: 0, count: k * d)
        for r in 0..<k {
            var maxAbs: Float = -1
            var flip: Float = 1
            for j in 0..<d {
                let value = svd.vt[j * panelRank + r] // vt is panelRank×d col-major
                let magnitude = abs(value)
                if magnitude > maxAbs {
                    maxAbs = magnitude
                    flip = value < 0 ? -1 : 1
                }
            }
            for j in 0..<d {
                componentsRowMajor[r * d + j] = flip * svd.vt[j * panelRank + r]
            }
        }

        var explained = [Float](repeating: 0, count: k)
        var ratios = [Float](repeating: 0, count: k)
        for r in 0..<k {
            let sigma = Double(svd.singularValues[r])
            let variance = sigma * sigma / varianceDenominator
            explained[r] = Float(variance)
            ratios[r] = totalVariance > 0 ? Float(variance / totalVariance) : 0
        }

        return PCAModel(
            mean: mean,
            components: componentsRowMajor,
            explainedVariance: explained,
            explainedVarianceRatio: ratios,
            inputDimension: d,
            componentCount: k)
    }

    // MARK: Transform

    /// Projects a batch: one GEMM over the whole batch. Returns one k-dim
    /// row per input vector.
    public func transform<V: VectorProtocol>(
        _ vectors: [V]
    ) throws -> [[Float]] where V.Scalar == Float {
        guard !vectors.isEmpty else { return [] }
        let n = vectors.count
        let d = inputDimension
        var centered = try Self.flattenColumnMajor(vectors, count: n, dimension: d)
        if mean.contains(where: { $0 != 0 }) {
            centered.withUnsafeMutableBufferPointer { cb in
                for j in 0..<d {
                    let mu = mean[j]
                    if mu == 0 { continue }
                    let base = j * n
                    for i in 0..<n { cb[base + i] -= mu }
                }
            }
        }

        // components is k×d row-major, which IS d×k column-major Wᵀ —
        // so P (n×k, col-major) = Xc (n×d) · Wᵀ (d×k) needs no transpose.
        let projected = LAMatMul.multiply(
            centered, components, m: n, n: componentCount, k: d)

        var out = [[Float]](repeating: [], count: n)
        for i in 0..<n {
            var row = [Float](repeating: 0, count: componentCount)
            for r in 0..<componentCount { row[r] = projected[r * n + i] }
            out[i] = row
        }
        return out
    }

    /// Projects a single raw vector — the streaming-friendly form.
    public func transform(_ vector: [Float]) throws -> [Float] {
        guard vector.count == inputDimension else {
            throw VectorError.dimensionMismatch(expected: inputDimension, actual: vector.count)
        }
        let d = inputDimension
        var out = [Float](repeating: 0, count: componentCount)
        vector.withUnsafeBufferPointer { xb in
            components.withUnsafeBufferPointer { wb in
                for r in 0..<componentCount {
                    var acc: Float = 0
                    let base = r * d
                    for j in 0..<d { acc += wb[base + j] * (xb[j] - mean[j]) }
                    out[r] = acc
                }
            }
        }
        return out
    }

    // MARK: Flattening

    /// Flattens vectors into an n×d COLUMN-MAJOR buffer, validating that
    /// every vector has dimension d. Writes for consecutive vectors land on
    /// adjacent addresses within each of the d column panels, so the pass
    /// stays cache-resident for embedding-sized d.
    private static func flattenColumnMajor<V: VectorProtocol>(
        _ vectors: [V], count n: Int, dimension d: Int
    ) throws -> [Float] where V.Scalar == Float {
        guard d >= 1 else {
            throw VectorError.invalidDimension(d, reason: "vectors must be non-empty")
        }
        var flat = [Float](repeating: 0, count: n * d)
        try flat.withUnsafeMutableBufferPointer { fb in
            for i in 0..<n {
                let v = vectors[i]
                guard v.scalarCount == d else {
                    throw VectorError.dimensionMismatch(expected: d, actual: v.scalarCount)
                }
                v.withUnsafeBufferPointer { vb in
                    for j in 0..<d { fb[j * n + i] = vb[j] }
                }
            }
        }
        return flat
    }
}

// MARK: - Operations integration

extension Operations {
    /// One-shot PCA: fit on `vectors` and project them, per the gap-report
    /// API sketch. For fit-on-sample / transform-the-corpus workflows use
    /// `PCAModel.fit` + `PCAModel.transform` directly.
    public static func pca<V: VectorProtocol>(
        _ vectors: [V],
        components: Int,
        config: PCAConfig = PCAConfig()
    ) throws -> (projected: [[Float]], model: PCAModel) where V.Scalar == Float {
        let model = try PCAModel.fit(vectors, components: components, config: config)
        let projected = try model.transform(vectors)
        return (projected, model)
    }
}

// MARK: - Seeded Gaussian source

/// Deterministic standard-normal stream: SplitMix64 → Box–Muller.
/// Cross-platform reproducible for a given seed (pure integer state plus
/// libm; no rejection sampling, so the stream position is data-independent).
internal struct GaussianSource {
    private var state: UInt64
    private var spare: Float?

    init(seed: UInt64) {
        self.state = seed
    }

    private mutating func nextUniform() -> Double {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        z ^= z >> 31
        // 53-bit mantissa → [0, 1); the +0.5 offset keeps log() off zero.
        return (Double(z >> 11) + 0.5) * 0x1p-53
    }

    mutating func next() -> Float {
        if let cached = spare {
            spare = nil
            return cached
        }
        let u1 = nextUniform()
        let u2 = nextUniform()
        let radius = (-2 * Foundation.log(u1)).squareRoot()
        let angle = 2 * Double.pi * u2
        spare = Float(radius * Foundation.sin(angle))
        return Float(radius * Foundation.cos(angle))
    }

    mutating func matrix(count: Int) -> [Float] {
        var out = [Float](repeating: 0, count: count)
        for i in 0..<count { out[i] = next() }
        return out
    }
}
