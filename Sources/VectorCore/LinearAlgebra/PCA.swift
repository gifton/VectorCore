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
//  The randomized path uses GEMM passes O(n·d·ℓ), n×ℓ QR, and ℓ×d SVD.
//  The sketch width is clamped to min(n,d) before addition. At that limit,
//  fitting uses an exact thin SVD of the full centered n×d sample (PCATests.exactPathMatchesEigen;
//  near-square samples on both sides of LAPACK sgesdd's path-5 boundary, tall and wide:
//  T19LAPACKWorkspaceQueryTests.exactPCASweepCrossesPathFive).
//  No corpus-size memory or throughput envelope is implied.
//
//  ## Usage at corpus scale
//
//  `PCAModel.fit` on a representative sample, then `transform` the full
//  corpus in caller-sized batches (one GEMM per batch) — the model is an
//  immutable value and `transform` is pure, so batches can run in parallel as far
//  as the active `Operations.linearAlgebraProvider` and GEMM are safe to call
//  concurrently; that is the provider's contract and is not verified here.
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

/// Configuration for randomized-SVD PCA; workload suitability is caller-selected.
public struct PCAConfig: Sendable {
    /// Nonnegative sketch oversampling p, clamped to min(n,d)-k before
    /// addition. Int.max is accepted and selects the exact fallback
    /// (T19LAPACKWorkspaceQueryTests.exactPCASweepCrossesPathFive and
    /// reportedCellsSucceed, default LAPACK provider).
    /// See PCACenteringContractTests.sketchBoundariesAndProviderIdentity for
    /// sketch-boundary controls.
    public var oversampling: Int

    /// Power (subspace) iterations `q`. Each adds two GEMM passes (`AᵀQ` and
    /// `A·Q'`) and two QR factorizations. Per Halko–Martinsson–Tropp (2011) they
    /// improve accuracy on slowly decaying spectra and q = 1–2 is a common
    /// choice (not verified for this implementation by a test here).
    public var powerIterations: Int

    /// Subtract the per-dimension mean before factorization (classical PCA).
    /// With `false` this computes truncated SVD of the raw data and
    /// `explainedVariance` reads as raw second moments, not variances.
    public var center: Bool

    /// Seed for the Gaussian sketch. PCATests.determinismAndProviderParity checks
    /// repeatability for its fixtures; bitwise parity across providers or
    /// hardware is not promised.
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

/// A fitted PCA projection: `y = W · (x − μ)`.
///
/// See `components` for the Swift provider's zero-singular-value basis limitation.
///
/// Fit on a sample with `fit`, then apply with `transform` — see the header
/// note on corpus-scale batching.
public struct PCAModel: Sendable {
    /// Widened per-dimension center μ used during fit and both transforms;
    /// exact Double zeros when uncentered (PCATests.uncenteredFit). Reconstruct in Double or explicitly
    /// round when migrating a caller that previously expected [Float].
    public let mean: [Double]

    /// Principal axes `W`, k×d ROW-MAJOR: `components[r*d + j]` is
    /// coordinate j of axis r. Rows are ordered by decreasing explained variance
    /// (ordering and sign convention: PCATests.modelInvariants).
    /// The PCA contract calls for orthonormal rows, but this buffer copies the
    /// SVD's Vᵀ rows: with the Swift provider, wide rank-deficient panels can
    /// yield zero rows for zero singular values. See `SingularValueDecomposition.vt`.
    /// Sign convention: each nonzero row's largest-magnitude coordinate is
    /// positive (first index on ties).
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
    /// Work and storage (from the internal fit body): the sample is flattened
    /// into an n×d column-major Float buffer and a second n×d copy is centered.
    /// When the sketch width reaches min(n, d), that full centered panel goes to
    /// the provider's thin SVD; otherwise the sketch adds n×L and d×L Float
    /// buffers, L being the sketch width.
    ///
    /// One-host time and peak memory for n up to 100,000 and d of 384, 768 and
    /// 1536 are recorded in Docs/Performance_Guide.md ("Measured Workload
    /// Envelope (One Host)"). Larger inputs are not measured.
    ///
    /// - Throws: `VectorError.invalidDimension` for k outside
    ///   `1...min(n − 1, d)` when centered, or `1...min(n, d)` when
    ///   `config.center == false`; also for invalid config.
    ///   These are structural bounds, not a measurement of the data's rank
    ///   (PCATests.errorContracts).
    ///   `.dimensionMismatch` for ragged input;
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
        let count = try validateVectorShapes(vectors, count: n, dimension: d)
        _ = try PCAShape.fit(rows: n, dimension: d, components: k, config: config, flatCount: count)
        let flat = try flattenColumnMajor(vectors, count: n, dimension: d)
        return try fit(columnMajor: flat, rows: n, dimension: d, components: k, config: config)
    }

    /// Core fit on an n×d column-major buffer. It centers a local copy
    /// (`var a = a`), so the caller's array is not modified.
    internal static func fit(
        columnMajor a: [Float],
        rows n: Int,
        dimension d: Int,
        components k: Int,
        config: PCAConfig
    ) throws -> PCAModel {
        let sketchWidth = try PCAShape.fit(rows: n, dimension: d, components: k,
                                            config: config, flatCount: a.count)
        try Self.requireFiniteInput(a)
        var a = a

        // PCACenteringContractTests exercises representable translations
        // (translatedPair, multicolumnAndRandomizedTranslations).
        // Compensated Double centers do not remove Float panel/GEMM rounding.
        var mean = [Double](repeating: 0, count: d)
        var totalVariance: Double = 0
        let varianceDenominator = Double(n - 1)
        for j in 0..<d {
            let base = j * n
            if config.center {
                var sum = 0.0, correction = 0.0
                for i in 0..<n {
                    let value = Double(a[base + i])
                    let next = sum + value
                    correction += abs(sum) >= abs(value) ? (sum - next) + value : (value - next) + sum
                    sum = next
                }
                mean[j] = (sum + correction) / Double(n)
            }
            var sumSq = 0.0
            for i in 0..<n {
                a[base + i] = try Self.centered(a[base + i], mean: mean[j])
                let residual = Double(a[base + i])
                sumSq += residual * residual
            }
            totalVariance += sumSq / varianceDenominator
        }
        guard mean.allSatisfy(\.isFinite), totalVariance.isFinite else {
            throw Self.resultFailure("centering statistics")
        }
        let provider = Operations.linearAlgebraProvider

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
            // dominant axis (the classic power-iteration failure mode; Halko et
            // al. 2011, Alg. 4.4; not verified by a test here).
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
            explained[r] = try Self.publish(variance, stage: "explained variance")
            ratios[r] = try Self.publish(totalVariance > 0 ? variance / totalVariance : 0,
                                         stage: "explained variance ratio")
        }

        try Self.requireFiniteResult(componentsRowMajor, stage: "components")
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
        _ = try Self.validateVectorShapes(vectors, count: n, dimension: d)
        try PCAShape.transform(rows: n, dimension: d, components: componentCount)
        var centered = try Self.flattenColumnMajor(vectors, count: n, dimension: d)
        try Self.requireFiniteInput(centered)
        for j in 0..<d {
            let base = j * n
            for i in 0..<n { centered[base + i] = try Self.centered(centered[base + i], mean: mean[j]) }
        }

        // components is k×d row-major, which IS d×k column-major Wᵀ —
        // so P (n×k, col-major) = Xc (n×d) · Wᵀ (d×k) needs no transpose.
        let projected = LAMatMul.multiply(
            centered, components, m: n, n: componentCount, k: d)

        try Self.requireFiniteResult(projected, stage: "batch projection")
        var out = [[Float]](repeating: [], count: n)
        for i in 0..<n {
            var row = [Float](repeating: 0, count: componentCount)
            for r in 0..<componentCount { row[r] = projected[r * n + i] }
            out[i] = row
        }
        return out
    }

    /// Projects a single raw vector.
    public func transform(_ vector: [Float]) throws -> [Float] {
        guard vector.count == inputDimension else {
            throw VectorError.dimensionMismatch(expected: inputDimension, actual: vector.count)
        }
        let d = inputDimension
        try PCAShape.transform(rows: 1, dimension: d, components: componentCount)
        try Self.requireFiniteInput(vector)
        var out = [Float](repeating: 0, count: componentCount)
        for r in 0..<componentCount {
            var acc: Float = 0
            let base = r * d
            for j in 0..<d { acc += components[base + j] * (try Self.centered(vector[j], mean: mean[j])) }
            out[r] = acc
        }
        try Self.requireFiniteResult(out, stage: "single projection")
        return out
    }

    // MARK: Flattening

    /// Flattens vectors into an n×d COLUMN-MAJOR buffer, validating that
    /// every vector has dimension d. Shared with KNNGraph.bruteForce: only
    /// general row/count/byte shape checks apply here. PCA finite-input and
    /// LP64 policy is enforced by PCA callers, not by this shared helper.
    internal static func flattenColumnMajor<V: VectorProtocol>(
        _ vectors: [V], count n: Int, dimension d: Int
    ) throws -> [Float] where V.Scalar == Float {
        let count = try validateVectorShapes(vectors, count: n, dimension: d)
        var flat = [Float](repeating: 0, count: count)
        flat.withUnsafeMutableBufferPointer { fb in
            for i in 0..<n {
                vectors[i].withUnsafeBufferPointer { vb in
                    for j in 0..<d { fb[j * n + i] = vb[j] }
                }
            }
        }
        return flat
    }

    private static func validateVectorShapes<V: VectorProtocol>(
        _ vectors: [V], count n: Int, dimension d: Int
    ) throws -> Int where V.Scalar == Float {
        let count = try PCAShape.flatten(rows: n, dimension: d, vectorCount: vectors.count)
        // Complete structural validation precedes any PCA finite scan.
        for vector in vectors where vector.scalarCount != d {
            throw VectorError.dimensionMismatch(expected: d, actual: vector.scalarCount)
        }
        return count
    }

    private static func requireFiniteInput(_ values: [Float]) throws {
        guard values.allSatisfy(\.isFinite) else { throw VectorError.invalidData("PCA requires finite input coordinates") }
    }

    private static func centered(_ value: Float, mean: Double) throws -> Float {
        try publish(Double(value) - mean, stage: "centering")
    }

    private static func publish(_ value: Double, stage: String) throws -> Float {
        let result = Float(value)
        guard value.isFinite, result.isFinite else { throw resultFailure(stage) }
        return result
    }

    private static func requireFiniteResult(_ values: [Float], stage: String) throws {
        guard values.allSatisfy(\.isFinite) else { throw resultFailure(stage) }
    }

    private static func resultFailure(_ stage: String) -> VectorError {
        ErrorBuilder(.operationFailed).message("PCA \(stage) produced a nonfinite or unrepresentable Float result").build()
    }

}

// Pure representability checks; PCACenteringContractTests.shapeAndSharedFlattenBoundaries exercises boundaries
// without allocating extreme shapes. These do not guarantee allocation success.
internal enum PCAShape {
    static func count(rows: Int, columns: Int, width: Int = MemoryLayout<Float>.stride) throws -> Int {
        guard rows >= 0, columns >= 0, width > 0 else {
            throw VectorError.invalidDimension(min(rows, columns), reason: "invalid PCA count domain")
        }
        let product = rows.multipliedReportingOverflow(by: columns)
        guard !product.overflow, !product.partialValue.multipliedReportingOverflow(by: width).overflow else {
            throw VectorError.invalidDimension(max(rows, columns), reason: "PCA element or byte count is not representable in Int")
        }
        return product.partialValue
    }

    static func flatten(rows n: Int, dimension d: Int, vectorCount: Int) throws -> Int {
        guard n >= 0, n == vectorCount else {
            throw VectorError.invalidDimension(n, reason: "flatten row count must equal vector count")
        }
        guard d >= 1 else { throw VectorError.invalidDimension(d, reason: "vectors must be non-empty") }
        return try count(rows: n, columns: d)
    }

    private static func blas(_ dimensions: Int...) throws {
        for dimension in dimensions where dimension < 1 || dimension > Int(Int32.max) {
            throw VectorError.invalidDimension(dimension, reason: "PCA dimension must fit positive LP64 BLAS indices")
        }
    }

    static func fit(rows n: Int, dimension d: Int, components k: Int,
                    config: PCAConfig, flatCount: Int) throws -> Int {
        guard n >= 2 else { throw VectorError.invalidDimension(n, reason: "internal PCA fit requires at least two rows") }
        try blas(n, d)
        guard config.oversampling >= 0, config.powerIterations >= 0 else {
            throw VectorError.invalidDimension(min(config.oversampling, config.powerIterations),
                                               reason: "oversampling and powerIterations must be non-negative")
        }
        let maxRank = min(config.center ? n - 1 : n, d)
        guard k >= 1, k <= maxRank else {
            throw VectorError.invalidDimension(k, reason: "components must be in 1...\(maxRank) (n=\(n), d=\(d), centered=\(config.center))")
        }
        let inputCount = try count(rows: n, columns: d)
        guard flatCount == inputCount else {
            throw VectorError.dimensionMismatch(expected: inputCount, actual: flatCount)
        }
        let limit = min(n, d)
        let sketchWidth = k + min(config.oversampling, limit - k)
        try blas(k, sketchWidth)
        _ = try count(rows: d, columns: 1, width: MemoryLayout<Double>.stride)
        _ = try count(rows: n, columns: sketchWidth)
        _ = try count(rows: d, columns: sketchWidth)
        _ = try count(rows: k, columns: d)
        _ = try count(rows: n, columns: k)
        return sketchWidth
    }

    static func transform(rows n: Int, dimension d: Int, components k: Int) throws {
        try blas(n, d, k)
        _ = try count(rows: n, columns: d)
        _ = try count(rows: k, columns: d)
        _ = try count(rows: n, columns: k)
        _ = try count(rows: d, columns: 1, width: MemoryLayout<Double>.stride)
        _ = try count(rows: n, columns: 1, width: MemoryLayout<[Float]>.stride)
    }
}

// MARK: - Operations integration

extension Operations {
    /// One-shot PCA: fit on `vectors` and project them, per the gap-report
    /// API sketch. For fit-on-sample / transform-the-corpus workflows use
    /// `PCAModel.fit` + `PCAModel.transform` directly.
    ///
    /// The body is `PCAModel.fit` followed by one batch `transform` of the same
    /// vectors, so it pays for both. One-host measurements of that pair of calls
    /// for n up to 100,000 and d up to 1536 are in Docs/Performance_Guide.md
    /// ("Measured Workload Envelope (One Host)"). Larger inputs are not measured.
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
/// Integer state is deterministic for a given seed. No rejection sampling
/// makes stream position data-independent; libm bits are not guaranteed
/// identical across platforms.
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
