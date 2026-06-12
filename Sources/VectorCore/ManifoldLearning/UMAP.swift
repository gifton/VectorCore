//
//  UMAP.swift
//  VectorCore
//
//  UMAP layout (gap analysis §3.3): fuzzy simplicial set → initialization →
//  SGD with negative sampling. The math follows McInnes, Healy & Melville
//  2018 and mirrors umap-learn's reference schedule (find_ab_params /
//  make_epochs_per_sample / optimize_layout_euclidean) so results are
//  comparable to the de-facto standard implementation.
//
//  ## Boundary (§1, §3.2)
//
//  VectorCore owns the math only. The kNN graph arrives as a `KNNGraph`
//  CSR value — produced by VectorIndex (or any ANN backend) at corpus
//  scale, or by `KNNGraph.bruteForce` in-Core at sample scale — so this
//  package never depends on VectorIndex; data flows Index → Core.
//
//  ## Initialization (§3.3.2)
//
//  PCA initialization is the default (reusing §3.1), per the gap report's
//  scale correction: dense spectral init is not viable at 5M nodes, and a
//  sparse Lanczos path is a stretch goal, not a dependency.
//
//  ## Determinism
//
//  The optimizer is single-threaded and fully seeded: a given (graph,
//  initial coordinates, config) triple reproduces the layout bit-for-bit.
//  The Hogwild-parallel variant trades that away and is the natural future
//  ComputeProvider/GPU candidate (§3.3.3) — not this baseline.
//

import Foundation

// MARK: - Configuration

/// Strategy for seeding the layout coordinates (vectors entry point).
public enum UMAPInitialization: Sendable {
    /// PCA of the input vectors (§3.1), rescaled to a ±10 extent with tiny
    /// seeded jitter — umap-learn's convention, and the gap report's
    /// recommended default at scale.
    case pca
    /// Seeded uniform coordinates in [−10, 10] per axis.
    case random
}

/// Tuning knobs for UMAP. Defaults match umap-learn's.
public struct UMAPConfig: Sendable {
    /// k for the in-Core brute-force kNN — used only when the vectors entry
    /// point has to build its own graph; ignored when a graph is supplied.
    public var neighbors: Int

    /// Minimum spacing between embedded points; the low-distance plateau of
    /// the output kernel. Must not exceed `spread`.
    public var minDist: Float

    /// Scale of the output kernel's decay. Together with `minDist` it
    /// determines the fitted (a, b) curve parameters.
    public var spread: Float

    /// SGD epochs; nil picks umap-learn's default (500 when n ≤ 10 000,
    /// else 200).
    public var epochs: Int?

    /// Initial SGD step size, annealed linearly to zero.
    public var learningRate: Float

    /// Repulsive samples drawn per attractive update. Zero disables
    /// repulsion (attraction-only layouts collapse; useful for debugging).
    public var negativeSampleRate: Int

    /// Coordinate seeding strategy (vectors entry point only; the graph
    /// entry point takes explicit coordinates or random).
    public var initialization: UMAPInitialization

    /// Seed for the PCA sketch, init jitter, and negative sampling.
    public var seed: UInt64

    public init(
        neighbors: Int = 15,
        minDist: Float = 0.1,
        spread: Float = 1.0,
        epochs: Int? = nil,
        learningRate: Float = 1.0,
        negativeSampleRate: Int = 5,
        initialization: UMAPInitialization = .pca,
        seed: UInt64 = 0x5EED_CAFE
    ) {
        self.neighbors = neighbors
        self.minDist = minDist
        self.spread = spread
        self.epochs = epochs
        self.learningRate = learningRate
        self.negativeSampleRate = negativeSampleRate
        self.initialization = initialization
        self.seed = seed
    }

    internal func validate() throws {
        guard neighbors >= 2 else {
            throw VectorError.invalidDimension(neighbors, reason: "neighbors must be at least 2")
        }
        guard spread > 0, minDist >= 0, minDist <= spread else {
            throw VectorError.invalidOperation(
                "UMAPConfig", reason: "require 0 ≤ minDist ≤ spread and spread > 0 "
                    + "(minDist = \(minDist), spread = \(spread))")
        }
        if let epochs {
            guard epochs >= 1 else {
                throw VectorError.invalidOperation("UMAPConfig", reason: "epochs must be ≥ 1")
            }
        }
        guard learningRate > 0 else {
            throw VectorError.invalidOperation("UMAPConfig", reason: "learningRate must be > 0")
        }
        guard negativeSampleRate >= 0 else {
            throw VectorError.invalidOperation(
                "UMAPConfig", reason: "negativeSampleRate must be ≥ 0")
        }
    }
}

// MARK: - Result

/// An embedded layout: n points in m dimensions, flat point-major storage
/// (point i occupies `coordinates[i*dimension ..< (i+1)*dimension]`).
public struct UMAPResult: Sendable {
    public let coordinates: [Float]
    public let pointCount: Int
    public let dimension: Int

    /// Copy of point i's coordinate row.
    public func point(_ i: Int) -> [Float] {
        precondition(i >= 0 && i < pointCount, "point index \(i) out of range 0..<\(pointCount)")
        let base = i * dimension
        return Array(coordinates[base..<(base + dimension)])
    }

    /// The 2-D scatter as SIMD pairs (the gap-report sketch's shape);
    /// nil unless `dimension == 2`.
    public var points2D: [SIMD2<Float>]? {
        guard dimension == 2 else { return nil }
        var out = [SIMD2<Float>]()
        out.reserveCapacity(pointCount)
        for i in 0..<pointCount {
            out.append(SIMD2(coordinates[2 * i], coordinates[2 * i + 1]))
        }
        return out
    }
}

// MARK: - Operations integration

extension Operations {
    /// Self-contained UMAP: kNN (supplied or exact in-Core brute force) →
    /// fuzzy simplicial set → PCA (or random) init → SGD layout.
    ///
    /// Pass `graph` to reuse an ANN-built kNN graph (VectorIndex, §3.2) —
    /// the vectors then only feed the PCA initialization. With `graph: nil`
    /// the O(n²·d) brute-force builder runs, which is meant for samples and
    /// tests, not the 5M corpus.
    ///
    /// - Throws: `VectorError` on invalid config/shapes, plus anything PCA
    ///   or the brute-force builder throws.
    public static func umap<V: VectorProtocol>(
        _ vectors: [V],
        dimensions: Int = 2,
        graph: KNNGraph? = nil,
        config: UMAPConfig = UMAPConfig()
    ) throws -> UMAPResult where V.Scalar == Float {
        try config.validate()
        guard dimensions >= 1 else {
            throw VectorError.invalidDimension(dimensions, reason: "output dimension must be ≥ 1")
        }
        let knn: KNNGraph
        if let graph {
            guard graph.pointCount == vectors.count else {
                throw VectorError.dimensionMismatch(
                    expected: vectors.count, actual: graph.pointCount)
            }
            knn = graph
        } else {
            knn = try KNNGraph.bruteForce(vectors, neighbors: config.neighbors)
        }
        let initial: [Float]
        switch config.initialization {
        case .pca:
            initial = try pcaInitialCoordinates(vectors, dimensions: dimensions, config: config)
        case .random:
            initial = UMAPLayout.randomCoordinates(
                count: vectors.count * dimensions, seed: config.seed)
        }
        return try umapLayout(graph: knn, dimensions: dimensions, initial: initial, config: config)
    }

    /// Graph-only UMAP for callers that no longer hold the original
    /// vectors: consumes the §3.2 CSR interchange directly.
    /// `initialCoordinates` rows (e.g. a PCA projection of the same points)
    /// seed the layout; nil falls back to seeded random in [−10, 10].
    public static func umap(
        graph: KNNGraph,
        dimensions: Int = 2,
        initialCoordinates: [[Float]]? = nil,
        config: UMAPConfig = UMAPConfig()
    ) throws -> UMAPResult {
        try config.validate()
        guard dimensions >= 1 else {
            throw VectorError.invalidDimension(dimensions, reason: "output dimension must be ≥ 1")
        }
        let n = graph.pointCount
        let initial: [Float]
        if let provided = initialCoordinates {
            guard provided.count == n else {
                throw VectorError.dimensionMismatch(expected: n, actual: provided.count)
            }
            var flat = [Float](repeating: 0, count: n * dimensions)
            let m = dimensions
            for i in 0..<n {
                guard provided[i].count == m else {
                    throw VectorError.dimensionMismatch(expected: m, actual: provided[i].count)
                }
                for c in 0..<m { flat[i * m + c] = provided[i][c] }
            }
            initial = flat
        } else {
            initial = UMAPLayout.randomCoordinates(count: n * dimensions, seed: config.seed)
        }
        return try umapLayout(graph: graph, dimensions: dimensions, initial: initial, config: config)
    }

    /// Shared tail: fuzzy set → curve fit → SGD.
    private static func umapLayout(
        graph: KNNGraph,
        dimensions: Int,
        initial: [Float],
        config: UMAPConfig
    ) throws -> UMAPResult {
        let n = graph.pointCount
        guard n >= 2 else {
            throw VectorError.invalidOperation(
                "Operations.umap", reason: "need at least 2 points, got \(n)")
        }
        let edges = FuzzySimplicialSet.build(graph)
        guard edges.count > 0 else {
            throw VectorError.invalidData("kNN graph produced no usable edges")
        }
        let curve = UMAPLayout.fitABParams(spread: config.spread, minDist: config.minDist)
        let epochs = config.epochs ?? (n <= 10_000 ? 500 : 200)
        var coordinates = initial
        UMAPLayout.optimize(
            edges: edges,
            coordinates: &coordinates,
            pointCount: n,
            dimension: dimensions,
            parameters: UMAPLayout.Parameters(
                a: curve.a, b: curve.b,
                epochs: epochs,
                learningRate: config.learningRate,
                negativeSampleRate: config.negativeSampleRate,
                seed: config.seed))
        return UMAPResult(coordinates: coordinates, pointCount: n, dimension: dimensions)
    }

    /// PCA projection rescaled to a ±10 extent plus 1e-4 seeded jitter
    /// (umap-learn's pca-init convention; the jitter breaks exact
    /// coincidences before SGD).
    private static func pcaInitialCoordinates<V: VectorProtocol>(
        _ vectors: [V],
        dimensions: Int,
        config: UMAPConfig
    ) throws -> [Float] where V.Scalar == Float {
        let model = try PCAModel.fit(
            vectors, components: dimensions, config: PCAConfig(seed: config.seed))
        let projected = try model.transform(vectors)
        let n = vectors.count
        let m = dimensions
        var flat = [Float](repeating: 0, count: n * m)
        var maxAbs: Float = 0
        for i in 0..<n {
            for c in 0..<m {
                let value = projected[i][c]
                flat[i * m + c] = value
                maxAbs = max(maxAbs, abs(value))
            }
        }
        let scale: Float = maxAbs > 0 ? 10 / maxAbs : 1
        var noise = GaussianSource(seed: config.seed ^ 0x9E37_79B9_7F4A_7C15)
        for idx in 0..<flat.count {
            flat[idx] = flat[idx] * scale + 1e-4 * noise.next()
        }
        return flat
    }
}

// MARK: - Layout engine

/// The SGD optimizer and its supporting math — umap-learn's
/// `optimize_layout_euclidean`, single-threaded for determinism.
internal enum UMAPLayout {

    internal struct Parameters {
        var a: Float
        var b: Float
        var epochs: Int
        var learningRate: Float
        var negativeSampleRate: Int
        var seed: UInt64
    }

    /// Output-kernel parameters: least-squares fit of `1/(1 + a·x^{2b})` to
    /// the piecewise target (1 below `minDist`, `exp(−(x−minDist)/spread)`
    /// above) on 300 samples of `[0, 3·spread]` — umap-learn's
    /// `find_ab_params` grid. Solved by Levenberg–Marquardt on the
    /// 2-parameter normal equations; deterministic, no RNG.
    static func fitABParams(spread: Float, minDist: Float) -> (a: Float, b: Float) {
        let samples = 300
        let spreadD = Double(spread)
        let minDistD = Double(minDist)
        var xs = [Double](repeating: 0, count: samples)
        var ys = [Double](repeating: 0, count: samples)
        for idx in 0..<samples {
            let x = 3 * spreadD * Double(idx) / Double(samples - 1)
            xs[idx] = x
            ys[idx] = x < minDistD ? 1 : Foundation.exp(-(x - minDistD) / spreadD)
        }

        var a = 1.0
        var b = 1.0
        var lambda = 1e-3
        var sse = curveSSE(xs, ys, a: a, b: b)
        for _ in 0..<200 {
            // Accumulate JᵀJ and Jᵀr for r(x) = f(x) − y(x).
            var jaa = 0.0, jab = 0.0, jbb = 0.0
            var ga = 0.0, gb = 0.0
            for s in 0..<samples where xs[s] > 0 {
                let x = xs[s]
                let xp = Foundation.pow(x, 2 * b)
                let denom = 1 + a * xp
                let residual = 1 / denom - ys[s]
                let dfda = -xp / (denom * denom)
                let dfdb = -2 * a * xp * Foundation.log(x) / (denom * denom)
                jaa += dfda * dfda
                jab += dfda * dfdb
                jbb += dfdb * dfdb
                ga += dfda * residual
                gb += dfdb * residual
            }
            // (JᵀJ + λ·diag(JᵀJ)) δ = −Jᵀr, 2×2 closed form.
            let daa = jaa * (1 + lambda)
            let dbb = jbb * (1 + lambda)
            let det = daa * dbb - jab * jab
            if abs(det) < 1e-30 { break }
            let deltaA = (-ga * dbb + gb * jab) / det
            let deltaB = (-gb * daa + ga * jab) / det
            let candidateA = a + deltaA
            let candidateB = b + deltaB
            if candidateA <= 0 || candidateB <= 0 {
                lambda *= 10
                continue
            }
            let candidateSSE = curveSSE(xs, ys, a: candidateA, b: candidateB)
            if candidateSSE < sse {
                a = candidateA
                b = candidateB
                sse = candidateSSE
                lambda = max(lambda / 3, 1e-12)
                if abs(deltaA) < 1e-10 && abs(deltaB) < 1e-10 { break }
            } else {
                lambda *= 10
                if lambda > 1e12 { break }
            }
        }
        return (Float(a), Float(b))
    }

    private static func curveSSE(_ xs: [Double], _ ys: [Double], a: Double, b: Double) -> Double {
        var total = 0.0
        for s in 0..<xs.count {
            let x = xs[s]
            let fitted = x > 0 ? 1 / (1 + a * Foundation.pow(x, 2 * b)) : 1
            let residual = fitted - ys[s]
            total += residual * residual
        }
        return total
    }

    /// Seeded uniform coordinates in [−10, 10] (umap-learn's random init).
    static func randomCoordinates(count: Int, seed: UInt64) -> [Float] {
        var rng = SplitMix64(state: seed)
        var out = [Float](repeating: 0, count: count)
        for idx in 0..<count { out[idx] = Float(rng.nextUniform() * 20 - 10) }
        return out
    }

    /// In-place SGD over the symmetric edge list, mirroring umap-learn's
    /// schedule: edges below `maxW/epochs` are pruned, edge e then fires
    /// every `maxW/wₑ` epochs; each firing attracts both endpoints along
    /// the edge and repulses the head against `negativeSampleRate` random
    /// vertices; the step anneals linearly to zero. Gradients are clamped
    /// to ±4 per coordinate (the reference's stability clip).
    static func optimize(
        edges: UMAPEdgeList,
        coordinates: inout [Float],
        pointCount n: Int,
        dimension m: Int,
        parameters: Parameters
    ) {
        var maxWeight: Float = 0
        for w in edges.weights where w > maxWeight { maxWeight = w }
        guard maxWeight > 0, parameters.epochs >= 1, n >= 1 else { return }

        let pruneBelow = parameters.epochs > 10 ? maxWeight / Float(parameters.epochs) : 0
        var heads = [Int32]()
        var tails = [Int32]()
        var cadence = [Double]() // epochs between samples of each edge
        for e in 0..<edges.count {
            let w = edges.weights[e]
            guard w > 0, w >= pruneBelow else { continue }
            heads.append(edges.heads[e])
            tails.append(edges.tails[e])
            cadence.append(Double(maxWeight / w))
        }
        guard !cadence.isEmpty else { return }

        let negativeRate = parameters.negativeSampleRate
        var nextSample = cadence
        var negativeCadence = [Double](repeating: 0, count: cadence.count)
        var nextNegative = [Double](repeating: 0, count: cadence.count)
        if negativeRate > 0 {
            for e in 0..<cadence.count {
                negativeCadence[e] = cadence[e] / Double(negativeRate)
                nextNegative[e] = negativeCadence[e]
            }
        }

        var rng = SplitMix64(state: parameters.seed ^ 0xD6E8_FEB8_6659_FD93)
        let kernel = (a: parameters.a, b: parameters.b)
        let totalEpochs = parameters.epochs

        coordinates.withUnsafeMutableBufferPointer { coords in
            for epoch in 0..<totalEpochs {
                let alpha = parameters.learningRate * (1 - Float(epoch) / Float(totalEpochs))
                let now = Double(epoch)
                for e in 0..<heads.count where nextSample[e] <= now {
                    let headBase = Int(heads[e]) * m
                    let tailBase = Int(tails[e]) * m
                    attract(coords, headBase, tailBase, m, kernel, alpha)
                    nextSample[e] += cadence[e]

                    guard negativeRate > 0 else { continue }
                    let due = Int((now - nextNegative[e]) / negativeCadence[e])
                    guard due > 0 else { continue }
                    for _ in 0..<due {
                        let otherBase = Int(rng.next() % UInt64(n)) * m
                        if otherBase == headBase { continue }
                        repulse(coords, headBase, otherBase, m, kernel, alpha)
                    }
                    nextNegative[e] += Double(due) * negativeCadence[e]
                }
            }
        }
    }

    /// Attractive update along edge (head, tail): both endpoints move by
    /// ∓clip(−2ab·d^{2(b−1)}/(1 + a·d^{2b}) · Δ) · α. Coincident endpoints
    /// are a no-op (zero gradient).
    @inline(__always)
    private static func attract(
        _ coords: UnsafeMutableBufferPointer<Float>,
        _ headBase: Int,
        _ tailBase: Int,
        _ m: Int,
        _ kernel: (a: Float, b: Float),
        _ alpha: Float
    ) {
        var squared: Float = 0
        for c in 0..<m {
            let diff = coords[headBase + c] - coords[tailBase + c]
            squared += diff * diff
        }
        guard squared > 0 else { return }
        let pd = powf32(squared, kernel.b)
        let coefficient = -2 * kernel.a * kernel.b * pd / (squared * (1 + kernel.a * pd))
        for c in 0..<m {
            let g = clip4(coefficient * (coords[headBase + c] - coords[tailBase + c]))
            coords[headBase + c] += alpha * g
            coords[tailBase + c] -= alpha * g
        }
    }

    /// Repulsive update of head against a sampled vertex:
    /// clip(2b/((0.001 + d²)(1 + a·d^{2b})) · Δ) · α; coincident pairs take
    /// the full clip bound to break the tie (umap-learn's behavior).
    @inline(__always)
    private static func repulse(
        _ coords: UnsafeMutableBufferPointer<Float>,
        _ headBase: Int,
        _ otherBase: Int,
        _ m: Int,
        _ kernel: (a: Float, b: Float),
        _ alpha: Float
    ) {
        var squared: Float = 0
        for c in 0..<m {
            let diff = coords[headBase + c] - coords[otherBase + c]
            squared += diff * diff
        }
        if squared > 0 {
            let pd = powf32(squared, kernel.b)
            let coefficient = 2 * kernel.b / ((0.001 + squared) * (1 + kernel.a * pd))
            for c in 0..<m {
                let g = clip4(coefficient * (coords[headBase + c] - coords[otherBase + c]))
                coords[headBase + c] += alpha * g
            }
        } else {
            for c in 0..<m { coords[headBase + c] += alpha * 4 }
        }
    }

    @inline(__always)
    private static func clip4(_ x: Float) -> Float {
        min(4, max(-4, x))
    }

    /// Float power via libm's Double pow — portable across the Darwin and
    /// Glibc overlays; the conversion cost is irrelevant next to pow itself.
    @inline(__always)
    private static func powf32(_ x: Float, _ y: Float) -> Float {
        Float(Foundation.pow(Double(x), Double(y)))
    }
}

// MARK: - Seeded uniform source

/// Minimal deterministic RNG (the same SplitMix64 mixer that drives
/// `GaussianSource`) for negative sampling and random initialization.
/// Not cryptographic.
internal struct SplitMix64 {
    var state: UInt64

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniform in [0, 1) with a 53-bit mantissa.
    mutating func nextUniform() -> Double {
        (Double(next() >> 11) + 0.5) * 0x1p-53
    }
}
