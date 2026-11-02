//
//  UMAP.swift
//  VectorCore
//
//  UMAP layout (gap analysis §3.3): fuzzy simplicial set → initialization →
//  SGD with negative sampling. The math follows McInnes, Healy & Melville
//  2018. Calibration and checked numerical boundaries follow the explicit
//  contracts below; no exact reference-implementation equivalence is asserted.
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
//  initial coordinates, config) triple is deterministic within a build, as
//  exercised by UMAPLayoutSuite.determinism; cross-platform bits are not promised.
//  The Hogwild-parallel variant trades that away and is the natural future
//  ComputeProvider/GPU candidate (§3.3.3) — not this baseline.
//

import Foundation

// Checked representation boundaries shared by the three UMAP/KNN files.
// UMAPRangeContractTests exercises pure limits without allocating huge buffers.
internal enum UMAPChecked {
    static let maximumScheduleCount = Int(exactly: UInt64(9_007_199_254_740_991)) ?? Int.max

    static func failure(_ stage: String) -> VectorError {
        ErrorBuilder(.operationFailed).message("UMAP numerical range or convergence failure: \(stage)").build()
    }

    static func count(_ factors: [Int], stride: Int, label: String) throws -> Int {
        guard stride > 0 else { throw VectorError.invalidDimension(stride, reason: label + " element stride") }
        var count = 1
        for factor in factors {
            guard factor >= 0 else { throw VectorError.invalidDimension(factor, reason: label + " negative count") }
            let product = count.multipliedReportingOverflow(by: factor)
            guard !product.overflow else { throw VectorError.invalidDimension(factor, reason: label + " element count overflow") }
            count = product.partialValue
        }
        guard !count.multipliedReportingOverflow(by: stride).overflow else {
            throw VectorError.invalidDimension(count, reason: label + " byte count overflow")
        }
        return count
    }

    static func increment(_ value: Int, label: String) throws -> Int {
        let result = value.addingReportingOverflow(1)
        guard value >= 0, !result.overflow else { throw VectorError.invalidDimension(value, reason: label + " count overflow") }
        return result.partialValue
    }

    static func lp64(_ value: Int, label: String) throws {
        guard value > 0, value <= Int(Int32.max) else {
            throw VectorError.invalidDimension(value, reason: label + " must fit positive LP64 BLAS Int32")
        }
    }

    static func layoutCount(points: Int, dimensions: Int) throws -> Int {
        guard points >= 2 else { throw VectorError.invalidDimension(points, reason: "UMAP needs at least 2 points") }
        guard dimensions >= 1 else { throw VectorError.invalidDimension(dimensions, reason: "output dimension must be positive") }
        return try count([points, dimensions], stride: MemoryLayout<Float>.stride, label: "UMAP coordinates")
    }

    static func schedule(epochs: Int, negativeRate: Int) throws {
        guard epochs >= 1, epochs <= maximumScheduleCount,
              negativeRate >= 0, negativeRate <= maximumScheduleCount else {
            throw VectorError.invalidOperation("UMAPConfig", reason: "epochs/rate must fit the exact-integer Double schedule domain")
        }
        let product = epochs.multipliedReportingOverflow(by: max(1, negativeRate))
        guard !product.overflow, product.partialValue <= maximumScheduleCount else {
            throw VectorError.invalidOperation("UMAPConfig", reason: "epochs * max(1, negativeSampleRate) exceeds the exact-integer Double schedule domain")
        }
    }

    static func dueCount(_ value: Double) throws -> Int {
        guard value.isFinite else { throw failure("nonfinite negative-sample count") }
        if value <= 0 { return 0 }
        let truncated = value.rounded(.towardZero)
        guard truncated <= Double(maximumScheduleCount), let count = Int(exactly: truncated) else {
            throw failure("unrepresentable negative-sample count")
        }
        return count
    }

    static func finite(_ value: Float, _ stage: String) throws -> Float {
        guard value.isFinite else { throw failure(stage) }
        return value
    }
}

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

/// Tuning knobs for UMAP. Floating options must be finite; unsupported numerical
/// coefficient or SGD ranges throw, rather than promising a finite layout for all
/// finite inputs. UMAPRangeContractTests exercises these boundaries.
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

    /// SGD epochs; nil picks the default (500 when n ≤ 10 000,
    /// else 200). Epochs and negativeSampleRate, and their product using at
    /// least rate 1, must fit 2^53-1 (or native Int.max if smaller).
    /// This is a representation domain, not an execution-time budget.
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
        guard spread.isFinite, minDist.isFinite, spread > 0, minDist >= 0, minDist <= spread else {
            throw VectorError.invalidOperation(
                "UMAPConfig", reason: "require 0 ≤ minDist ≤ spread and spread > 0 "
                    + "(minDist = \(minDist), spread = \(spread))")
        }
        if let epochs {
            guard epochs >= 1 else {
                throw VectorError.invalidOperation("UMAPConfig", reason: "epochs must be ≥ 1")
            }
        }
        guard learningRate.isFinite, learningRate > 0 else {
            throw VectorError.invalidOperation("UMAPConfig", reason: "learningRate must be finite and > 0")
        }
        guard negativeSampleRate >= 0 else {
            throw VectorError.invalidOperation(
                "UMAPConfig", reason: "negativeSampleRate must be ≥ 0")
        }
        guard negativeSampleRate <= UMAPChecked.maximumScheduleCount else {
            throw VectorError.invalidOperation("UMAPConfig", reason: "negativeSampleRate exceeds the exact-integer Double domain")
        }
        if let epochs { try UMAPChecked.schedule(epochs: epochs, negativeRate: negativeSampleRate) }
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
    /// Self-contained UMAP: kNN (supplied or exhaustive Float Gram scores) →
    /// fuzzy simplicial set → PCA (or random) init → SGD layout.
    ///
    /// Pass `graph` to reuse an ANN-built kNN graph (VectorIndex, §3.2) —
    /// the vectors then only feed the PCA initialization. With `graph: nil`
    /// the O(n²·d) brute-force builder runs, which is meant for samples and
    /// tests, not the 5M corpus.
    ///
    /// With `graph: nil` this entry runs that builder, then a PCA fit and
    /// transform for `.pca` initialization, the fuzzy set, and `epochs` passes
    /// that each scan every retained edge (see `umapLayout`). One-host
    /// measurements of this entry (default configuration, 2-D) are in
    /// Docs/Performance_Guide.md ("Measured Workload Envelope (One Host)"):
    /// 85.5 s at n = 100,000, d = 768 in High Power Mode on synthetic data (see
    /// the dated record's conditions), of which the separately measured graph
    /// build is about half. Larger n or d is not measured.
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
        let coordinateCount = try UMAPChecked.layoutCount(points: vectors.count, dimensions: dimensions)
        try UMAPChecked.schedule(epochs: config.epochs ?? (vectors.count <= 10_000 ? 500 : 200), negativeRate: config.negativeSampleRate)
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
                count: coordinateCount, seed: config.seed)
        }
        return try umapLayout(graph: knn, dimensions: dimensions, initial: initial, config: config)
    }

    /// Graph-only UMAP for callers that no longer hold the original
    /// vectors: consumes the §3.2 CSR interchange directly.
    /// `initialCoordinates` rows (e.g. a PCA projection of the same points)
    /// seed the layout; nil falls back to seeded random in [−10, 10].
    /// - Throws: `.invalidDimension` for fewer than two points (0.3.3 threw `.invalidOperation`) and for the other
    ///   count, configuration and scheduling rejections the entry validates before allocating.
    public static func umap(
        graph: KNNGraph,
        dimensions: Int = 2,
        initialCoordinates: [[Float]]? = nil,
        config: UMAPConfig = UMAPConfig()
    ) throws -> UMAPResult {
        try config.validate()
        let coordinateCount = try UMAPChecked.layoutCount(points: graph.pointCount, dimensions: dimensions)
        try UMAPChecked.schedule(epochs: config.epochs ?? (graph.pointCount <= 10_000 ? 500 : 200), negativeRate: config.negativeSampleRate)
        let n = graph.pointCount
        let initial: [Float]
        if let provided = initialCoordinates {
            guard provided.count == n else {
                throw VectorError.dimensionMismatch(expected: n, actual: provided.count)
            }
            let m = dimensions
            for row in provided {
                guard row.count == m else { throw VectorError.dimensionMismatch(expected: m, actual: row.count) }
            }
            for row in provided {
                guard row.allSatisfy({ $0.isFinite }) else { throw VectorError.invalidData("UMAP initial coordinates must be finite") }
            }
            var flat = [Float](repeating: 0, count: coordinateCount)
            for i in 0..<n {
                for c in 0..<m { flat[i * m + c] = provided[i][c] }
            }
            initial = flat
        } else {
            initial = UMAPLayout.randomCoordinates(count: coordinateCount, seed: config.seed)
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
        let coordinateCount = try UMAPChecked.layoutCount(points: n, dimensions: dimensions)
        guard initial.count == coordinateCount, initial.allSatisfy({ $0.isFinite }) else {
            throw VectorError.invalidData("UMAP initial buffer has invalid shape or nonfinite coordinates")
        }
        let edges = try FuzzySimplicialSet.build(graph)
        guard edges.count > 0 else {
            throw VectorError.invalidData("kNN graph produced no usable edges")
        }
        let curve = try UMAPLayout.fitABParams(spread: config.spread, minDist: config.minDist)
        let epochs = config.epochs ?? (n <= 10_000 ? 500 : 200)
        var coordinates = initial
        try UMAPLayout.optimize(
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
        guard coordinates.allSatisfy({ $0.isFinite }) else { throw UMAPChecked.failure("nonfinite final coordinates") }
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
        var flat = [Float](repeating: 0, count: try UMAPChecked.layoutCount(points: n, dimensions: m))
        var maxAbs: Float = 0
        for i in 0..<n {
            for c in 0..<m {
                let value = projected[i][c]
                flat[i * m + c] = value
                maxAbs = max(maxAbs, abs(value))
            }
        }
        var noise = GaussianSource(seed: config.seed ^ 0x9E37_79B9_7F4A_7C15)
        for idx in 0..<flat.count {
            let scaled = maxAbs > 0 ? Float(10 * (Double(flat[idx]) / Double(maxAbs))) : flat[idx]
            flat[idx] = try UMAPChecked.finite(scaled + 1e-4 * noise.next(), "PCA initialization")
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

    /// Dimensionless 300-point least-squares calibration in log(alpha), log(beta).
    /// Contract: convergence requires finite SSE-gradient infinity norm <= 1e-8;
    /// unsupported Float coefficient ranges throw rather than publishing a poor fit.
    /// UMAPRangeContractTests compares independent derivative-free minima and scales.
    static func fitABParams(spread: Float, minDist: Float) throws -> (a: Float, b: Float) {
        guard spread.isFinite, minDist.isFinite, spread > 0, minDist >= 0, minDist <= spread else {
            throw VectorError.invalidOperation("UMAP curve", reason: "require finite 0 <= minDist <= spread and spread > 0")
        }
        let ratio = Double(minDist) / Double(spread)
        let samples = 300
        let logs = (1..<samples).map { Foundation.log(3 * Double($0) / Double(samples - 1)) }
        let targets = (1..<samples).map { index -> Double in
            let t = 3 * Double(index) / Double(samples - 1)
            return t < ratio ? 1 : Foundation.exp(-(t - ratio))
        }
        var logAlpha = 0.0, logBeta = 0.0, lambda = 1e-3
        guard var state = evaluateCurve(logAlpha, logBeta, logs: logs, targets: targets) else {
            throw UMAPChecked.failure("curve initial objective")
        }
        for _ in 0..<200 {
            if state.gradient <= 1e-8 { break }
            let daa = state.jaa * (1 + lambda)
            let dbb = state.jbb * (1 + lambda)
            let determinant = daa * dbb - state.jab * state.jab
            var candidate: CurveEvaluation?
            var nextA = logAlpha, nextB = logBeta
            if determinant.isFinite, determinant > 0, daa.isFinite, dbb.isFinite {
                let deltaA = (-state.ga * dbb + state.gb * state.jab) / determinant
                let deltaB = (-state.gb * daa + state.ga * state.jab) / determinant
                nextA += deltaA; nextB += deltaB
                if nextA.isFinite, nextB.isFinite {
                    candidate = evaluateCurve(nextA, nextB, logs: logs, targets: targets)
                }
            }
            if let accepted = candidate,
               let change = objectiveChange(from: state, to: accepted, targets: targets), change < 0 {
                logAlpha = nextA; logBeta = nextB; state = accepted
                lambda = max(lambda / 3, 1e-12)
            } else {
                lambda *= 10
                if !lambda.isFinite || lambda > 1e12 { break }
            }
        }
        guard state.gradient.isFinite, state.gradient <= 1e-8 else {
            throw UMAPChecked.failure("curve fitting did not reach stationarity in 200 iterations or damping stalled")
        }
        let b = Float(Foundation.exp(logBeta))
        guard b.isFinite, b > 0 else { throw UMAPChecked.failure("curve exponent Float range") }
        let physicalLogAlpha = logAlpha - 2 * Double(b) * Foundation.log(Double(spread))
        let a = Float(Foundation.exp(physicalLogAlpha))
        guard a.isFinite, a > 0 else { throw UMAPChecked.failure("curve coefficient Float range") }
        let roundedLogAlpha = Foundation.log(Double(a)) + 2 * Double(b) * Foundation.log(Double(spread))
        var deviation = 0.0
        for index in logs.indices {
            let ell = roundedLogAlpha + 2 * Double(b) * logs[index]
            guard ell.isFinite else { throw UMAPChecked.failure("rounded curve evaluation range") }
            deviation = max(deviation, abs(logistic(ell) - state.values[index]))
        }
        guard deviation.isFinite, deviation <= 64 * Double(Float.ulpOfOne) else {
            throw UMAPChecked.failure("rounded curve coefficient range loses dimensionless fidelity")
        }
        return (a, b)
    }

    private struct CurveEvaluation {
        var sse = 0.0, jaa = 0.0, jab = 0.0, jbb = 0.0
        var ga = 0.0, gb = 0.0
        var values: [Double] = []
        // SSE's gradient is twice Jᵀr.
        var gradient: Double { 2 * max(abs(ga), abs(gb)) }
    }

    /// Algebraic SSE difference, compensated before comparing to zero. Comparing
    /// separately rounded totals can stall above the unchanged gradient criterion;
    /// UMAPRangeContractTests.curveOrdinaryNonDyadicScales exercises that case.
    private static func objectiveChange(from old: CurveEvaluation, to new: CurveEvaluation, targets: [Double]) -> Double? {
        var sum = 0.0, correction = 0.0
        for i in targets.indices {
            let oldResidual = old.values[i] - targets[i]
            let newResidual = new.values[i] - targets[i]
            let delta = new.values[i] - old.values[i]
            let term = delta * (newResidual + oldResidual)
            let updated = sum + term
            guard oldResidual.isFinite, newResidual.isFinite, delta.isFinite,
                  term.isFinite, updated.isFinite else { return nil }
            if abs(sum) >= abs(term) { correction += (sum - updated) + term }
            else { correction += (term - updated) + sum }
            sum = updated
        }
        let change = sum + correction
        return change.isFinite ? change : nil
    }

    private static func logistic(_ ell: Double) -> Double {
        let exponential = Foundation.exp(-abs(ell))
        return ell >= 0 ? exponential / (1 + exponential) : 1 / (1 + exponential)
    }

    private static func evaluateCurve(_ a: Double, _ b: Double, logs: [Double], targets: [Double]) -> CurveEvaluation? {
        let beta = Foundation.exp(b)
        guard a.isFinite, beta.isFinite, beta > 0 else { return nil }
        var result = CurveEvaluation()
        result.values.reserveCapacity(logs.count)
        for i in logs.indices {
            let multiplier = 2 * beta * logs[i]
            let ell = a + multiplier
            guard multiplier.isFinite, ell.isFinite else { return nil }
            let fitted = logistic(ell)
            let residual = fitted - targets[i]
            let derivative = -fitted * (1 - fitted)
            let db = multiplier * derivative
            result.sse += residual * residual
            result.jaa += derivative * derivative
            result.jab += derivative * db
            result.jbb += db * db
            result.ga += derivative * residual
            result.gb += db * residual
            result.values.append(fitted)
        }
        guard [result.sse, result.jaa, result.jab, result.jbb, result.ga, result.gb, result.gradient].allSatisfy({ $0.isFinite }) else { return nil }
        return result
    }

    /// Seeded uniform coordinates in [−10, 10] (umap-learn's random init).
    static func randomCoordinates(count: Int, seed: UInt64) -> [Float] {
        var rng = SplitMix64(state: seed)
        var out = [Float](repeating: 0, count: count)
        for idx in 0..<count { out[idx] = Float(rng.nextUniform() * 20 - 10) }
        return out
    }

    /// Checked in-place Float SGD with the existing clipping/zero-distance policy.
    /// Contract: every published coordinate is finite; intermediate overflow throws
    /// before clipping can mask it. Internal mutation on failure is not transactional.
    /// UMAPRangeContractTests covers scheduling boundaries and finite-range failures.
    static func optimize(
        edges: UMAPEdgeList,
        coordinates: inout [Float],
        pointCount n: Int,
        dimension m: Int,
        parameters: Parameters
    ) throws {
        let count = try UMAPChecked.layoutCount(points: n, dimensions: m)
        try UMAPChecked.schedule(epochs: parameters.epochs, negativeRate: parameters.negativeSampleRate)
        guard coordinates.count == count, coordinates.allSatisfy({ $0.isFinite }) else {
            throw VectorError.invalidData("UMAP optimizer coordinates have invalid shape or values")
        }
        guard parameters.learningRate.isFinite, parameters.learningRate > 0,
              parameters.a.isFinite, parameters.a > 0, parameters.b.isFinite, parameters.b > 0 else {
            throw UMAPChecked.failure("optimizer parameter range")
        }
        guard edges.heads.count == edges.tails.count, edges.heads.count == edges.weights.count else {
            throw VectorError.invalidData("UMAP edge arrays disagree")
        }
        _ = try UMAPChecked.count([edges.count], stride: MemoryLayout<Double>.stride, label: "UMAP schedule")
        _ = try UMAPChecked.count([edges.count], stride: MemoryLayout<Int32>.stride, label: "UMAP edge indices")
        var maxWeight: Float = 0
        for e in 0..<edges.count {
            let weight = edges.weights[e]
            guard weight.isFinite, weight >= 0, weight <= 1,
                  edges.heads[e] >= 0, Int(edges.heads[e]) < n,
                  edges.tails[e] >= 0, Int(edges.tails[e]) < n else {
                throw VectorError.invalidData("UMAP edge values outside checked domain")
            }
            maxWeight = max(maxWeight, weight)
        }
        guard maxWeight > 0 else { return }
        let pruneBelow = parameters.epochs > 10 ? Double(maxWeight) / Double(parameters.epochs) : 0
        var heads = [Int32](), tails = [Int32](), cadence = [Double]()
        for e in 0..<edges.count {
            let weight = edges.weights[e]
            guard weight > 0, Double(weight) >= pruneBelow else { continue }
            let interval = Double(maxWeight) / Double(weight)
            guard interval.isFinite, interval > 0 else { throw UMAPChecked.failure("edge cadence") }
            heads.append(edges.heads[e]); tails.append(edges.tails[e]); cadence.append(interval)
        }
        guard !cadence.isEmpty else { return }
        let negativeRate = parameters.negativeSampleRate
        var nextSample = cadence
        var negativeCadence = [Double](repeating: 0, count: cadence.count)
        var nextNegative = [Double](repeating: 0, count: cadence.count)
        if negativeRate > 0 {
            for e in cadence.indices {
                let interval = cadence[e] / Double(negativeRate)
                guard interval.isFinite, interval > 0 else { throw UMAPChecked.failure("negative cadence") }
                negativeCadence[e] = interval; nextNegative[e] = interval
            }
        }
        var rng = SplitMix64(state: parameters.seed ^ 0xD6E8_FEB8_6659_FD93)
        let kernel = (a: parameters.a, b: parameters.b)
        let totalEpochs = parameters.epochs
        try coordinates.withUnsafeMutableBufferPointer { coords in
            for epoch in 0..<totalEpochs {
                let fraction = 1 - Double(epoch) / Double(totalEpochs)
                let alpha = try UMAPChecked.finite(Float(Double(parameters.learningRate) * fraction), "annealed step")
                let now = Double(epoch)
                for e in heads.indices where nextSample[e] <= now {
                    let headBase = Int(heads[e]) * m // Bounded by checked n*m.
                    let tailBase = Int(tails[e]) * m
                    try attract(coords, headBase, tailBase, m, kernel, alpha)
                    let next = nextSample[e] + cadence[e]
                    guard next.isFinite else { throw UMAPChecked.failure("next edge epoch") }
                    nextSample[e] = next
                    guard negativeRate > 0 else { continue }
                    let quotient = (now - nextNegative[e]) / negativeCadence[e]
                    let due = try UMAPChecked.dueCount(quotient)
                    guard due > 0 else { continue }
                    for _ in 0..<due {
                        let otherBase = Int(rng.next() % UInt64(n)) * m
                        if otherBase != headBase { try repulse(coords, headBase, otherBase, m, kernel, alpha) }
                    }
                    let increment = Double(due) * negativeCadence[e]
                    let negativeNext = nextNegative[e] + increment
                    guard increment.isFinite, negativeNext.isFinite else { throw UMAPChecked.failure("next negative epoch") }
                    nextNegative[e] = negativeNext
                }
            }
        }
    }

    private static func squaredDistance(_ coords: UnsafeMutableBufferPointer<Float>, _ lhs: Int, _ rhs: Int, _ dimensions: Int) throws -> Float {
        var squared: Float = 0
        for c in 0..<dimensions {
            let difference = try UMAPChecked.finite(coords[lhs + c] - coords[rhs + c], "coordinate difference")
            let product = try UMAPChecked.finite(difference * difference, "squared coordinate difference")
            squared = try UMAPChecked.finite(squared + product, "squared-distance accumulation")
        }
        return squared
    }

    private static func attract(
        _ coords: UnsafeMutableBufferPointer<Float>, _ headBase: Int, _ tailBase: Int,
        _ m: Int, _ kernel: (a: Float, b: Float), _ alpha: Float
    ) throws {
        let squared = try squaredDistance(coords, headBase, tailBase, m)
        guard squared > 0 else { return }
        let power = try UMAPChecked.finite(powf32(squared, kernel.b), "attractive power")
        let weightedPower = try UMAPChecked.finite(kernel.a * power, "attractive weighted power")
        let sum = try UMAPChecked.finite(1 + weightedPower, "attractive denominator sum")
        let denominator = try UMAPChecked.finite(squared * sum, "attractive denominator")
        let ab = try UMAPChecked.finite(-2 * kernel.a, "attractive coefficient scale")
        let abb = try UMAPChecked.finite(ab * kernel.b, "attractive exponent coefficient")
        let numerator = try UMAPChecked.finite(abb * power, "attractive numerator")
        let coefficient = try UMAPChecked.finite(numerator / denominator, "attractive coefficient")
        for c in 0..<m {
            let difference = try UMAPChecked.finite(coords[headBase + c] - coords[tailBase + c], "attractive difference")
            let gradient = try UMAPChecked.finite(coefficient * difference, "unclipped attractive gradient")
            let step = try UMAPChecked.finite(alpha * clip4(gradient), "attractive step")
            let head = try UMAPChecked.finite(coords[headBase + c] + step, "attractive head update")
            let tail = try UMAPChecked.finite(coords[tailBase + c] - step, "attractive tail update")
            coords[headBase + c] = head; coords[tailBase + c] = tail
        }
    }

    private static func repulse(
        _ coords: UnsafeMutableBufferPointer<Float>, _ headBase: Int, _ otherBase: Int,
        _ m: Int, _ kernel: (a: Float, b: Float), _ alpha: Float
    ) throws {
        let squared = try squaredDistance(coords, headBase, otherBase, m)
        if squared > 0 {
            let power = try UMAPChecked.finite(powf32(squared, kernel.b), "repulsive power")
            let weighted = try UMAPChecked.finite(kernel.a * power, "repulsive weighted power")
            let sum = try UMAPChecked.finite(1 + weighted, "repulsive denominator sum")
            let regularized = try UMAPChecked.finite(0.001 + squared, "repulsive regularizer")
            let denominator = try UMAPChecked.finite(regularized * sum, "repulsive denominator")
            let numerator = try UMAPChecked.finite(2 * kernel.b, "repulsive numerator")
            let coefficient = try UMAPChecked.finite(numerator / denominator, "repulsive coefficient")
            for c in 0..<m {
                let difference = try UMAPChecked.finite(coords[headBase + c] - coords[otherBase + c], "repulsive difference")
                let gradient = try UMAPChecked.finite(coefficient * difference, "unclipped repulsive gradient")
                let step = try UMAPChecked.finite(alpha * clip4(gradient), "repulsive step")
                coords[headBase + c] = try UMAPChecked.finite(coords[headBase + c] + step, "repulsive coordinate update")
            }
        } else {
            let step = try UMAPChecked.finite(alpha * 4, "coincident repulsion step")
            for c in 0..<m {
                coords[headBase + c] = try UMAPChecked.finite(coords[headBase + c] + step, "coincident repulsion update")
            }
        }
    }

    @inline(__always)
    private static func clip4(_ x: Float) -> Float {
        min(4, max(-4, x))
    }

    /// Float projection of libm's Double power; callers check representability.
    /// UMAPRangeContractTests.finiteSGDRangeAndInputImmutability exercises overflow.
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
