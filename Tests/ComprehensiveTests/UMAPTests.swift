//
//  UMAPTests.swift
//  VectorCore
//
//  Correctness tests for the UMAP stack (gap report §3.3): the KNNGraph
//  CSR contract and exact brute-force builder, the fuzzy simplicial set
//  (ρ/σ smooth-kNN search, t-conorm symmetrization), the output-kernel
//  curve fit, and the SGD layout.
//
//  Strategy: structural pieces are checked against hand-computed values on
//  tiny graphs (the σ search against the closed-form solution of its own
//  defining equation, the curve fit against umap-learn's published
//  constants for the default config); the end-to-end layout is checked by
//  geometry — well-separated clusters in ambient space must stay separated
//  in the embedding — plus exact determinism under a fixed seed.
//

import Testing
import Foundation
@testable import VectorCore

// MARK: - Fixtures & helpers

private enum UMAPFixtures {
    /// Two isotropic Gaussian blobs in d dims: points [0, half) sit at the
    /// origin, points [half, 2·half) are offset by `separation` along
    /// dimension 0. Deterministic for a given seed.
    static func clusters(
        seed: UInt64, half: Int, dimensions d: Int, separation: Float, scale: Float
    ) -> [DynamicVector] {
        var gaussian = GaussianSource(seed: seed)
        return (0..<(2 * half)).map { i in
            var values = [Float](repeating: 0, count: d)
            for j in 0..<d { values[j] = scale * gaussian.next() }
            if i >= half { values[0] += separation }
            return DynamicVector(values)
        }
    }

    /// Inter-centroid distance over the larger mean within-cluster radius —
    /// the embedding-space "how separated did the clusters stay" score.
    static func separationRatio(_ result: UMAPResult, half: Int) -> Float {
        let m = result.dimension
        var centroidA = [Float](repeating: 0, count: m)
        var centroidB = [Float](repeating: 0, count: m)
        for i in 0..<half {
            for c in 0..<m {
                centroidA[c] += result.coordinates[i * m + c]
                centroidB[c] += result.coordinates[(half + i) * m + c]
            }
        }
        for c in 0..<m {
            centroidA[c] /= Float(half)
            centroidB[c] /= Float(half)
        }
        func radius(_ start: Int, _ centroid: [Float]) -> Float {
            var total: Float = 0
            for i in start..<(start + half) {
                var squared: Float = 0
                for c in 0..<m {
                    let diff = result.coordinates[i * m + c] - centroid[c]
                    squared += diff * diff
                }
                total += squared.squareRoot()
            }
            return total / Float(half)
        }
        var betweenSquared: Float = 0
        for c in 0..<m {
            let diff = centroidA[c] - centroidB[c]
            betweenSquared += diff * diff
        }
        let spread = max(radius(0, centroidA), radius(half, centroidB))
        return betweenSquared.squareRoot() / max(spread, .leastNormalMagnitude)
    }

    /// Direct-from-definition exact kNN (per-pair distances, full sort) —
    /// the reference for `KNNGraph.bruteForce`'s Gram-trick path.
    static func naiveNeighbors(
        _ data: [DynamicVector], k: Int
    ) -> [[(index: Int, distance: Float)]] {
        let n = data.count
        let d = data[0].scalarCount
        return (0..<n).map { i in
            var candidates: [(index: Int, distance: Float)] = []
            for j in 0..<n where j != i {
                var squared: Float = 0
                for c in 0..<d {
                    let diff = data[i][c] - data[j][c]
                    squared += diff * diff
                }
                candidates.append((j, squared.squareRoot()))
            }
            candidates.sort { ($0.distance, $0.index) < ($1.distance, $1.index) }
            return Array(candidates.prefix(k))
        }
    }

    static func edgeKey(_ head: Int32, _ tail: Int32) -> UInt64 {
        UInt64(UInt32(bitPattern: head)) << 32 | UInt64(UInt32(bitPattern: tail))
    }

    static func weightMap(_ edges: UMAPEdgeList) -> [UInt64: Float] {
        var map: [UInt64: Float] = [:]
        for e in 0..<edges.count {
            map[edgeKey(edges.heads[e], edges.tails[e])] = edges.weights[e]
        }
        return map
    }

    /// Four collinear points at x = 0, 1, 2, 4 with their exact 2-NN graph,
    /// small enough that every ρ, σ, and fused weight is hand-checkable.
    static func handGraph() throws -> KNNGraph {
        try KNNGraph(
            pointCount: 4,
            rowOffsets: [0, 2, 4, 6, 8],
            neighborIndices: [1, 2, 0, 2, 1, 3, 2, 1],
            distances: [1, 2, 1, 1, 1, 2, 2, 3])
    }
}

// MARK: - Curve fit

@Suite("UMAP - Curve fit")
struct UMAPCurveFitSuite {

    @Test("Default (minDist 0.1, spread 1.0) reproduces umap-learn's a, b")
    func defaultParameters() {
        // Reference values from umap-learn's find_ab_params on the same
        // 300-point grid: a ≈ 1.57694, b ≈ 0.89506.
        let (a, b) = UMAPLayout.fitABParams(spread: 1.0, minDist: 0.1)
        #expect(abs(a - 1.57694) < 0.02, "a = \(a)")
        #expect(abs(b - 0.89506) < 0.01, "b = \(b)")
    }

    @Test("Fitted kernel respects the plateau and the tail for other configs")
    func fitQuality() {
        for (minDist, spread) in [(Float(0), Float(1)), (0.5, 1.5), (0.25, 0.7)] {
            let (a, b) = UMAPLayout.fitABParams(spread: spread, minDist: minDist)
            #expect(a > 0 && b > 0)
            func kernel(_ x: Float) -> Float {
                x > 0 ? 1 / (1 + a * Foundation.pow(x, 2 * b)) : 1
            }
            // Plateau: still near 1 at the minimum distance.
            #expect(kernel(minDist) > 0.8, "minDist=\(minDist) spread=\(spread)")
            // Tail: mostly decayed at the grid edge.
            #expect(kernel(3 * spread) < 0.2, "minDist=\(minDist) spread=\(spread)")
            // Monotone decreasing on the fit grid.
            var previous = kernel(0)
            for s in 1...30 {
                let value = kernel(3 * spread * Float(s) / 30)
                #expect(value <= previous + 1e-6)
                previous = value
            }
        }
    }
}

// MARK: - KNNGraph

@Suite("UMAP - KNNGraph")
struct KNNGraphSuite {

    @Test("Brute force matches direct-from-definition kNN")
    func bruteForceMatchesNaive() throws {
        let data = UMAPFixtures.clusters(seed: 61, half: 20, dimensions: 6, separation: 4, scale: 1)
        let k = 5
        let graph = try KNNGraph.bruteForce(data, neighbors: k)
        let reference = UMAPFixtures.naiveNeighbors(data, k: k)

        #expect(graph.pointCount == 40)
        #expect(graph.edgeCount == 40 * k)
        for i in 0..<40 {
            let range = graph.neighborRange(of: i)
            #expect(range.count == k)
            let got = Set(range.map { Int(graph.neighborIndices[$0]) })
            let expected = Set(reference[i].map(\.index))
            #expect(got == expected, "row \(i): \(got) vs \(expected)")
            // Distances ascending and matching the direct computation.
            for (offset, idx) in range.enumerated() {
                #expect(abs(graph.distances[idx] - reference[i][offset].distance) < 2e-3)
                if idx > range.lowerBound {
                    #expect(graph.distances[idx] >= graph.distances[idx - 1])
                }
            }
        }
    }

    @Test("CSR validation rejects malformed graphs")
    func csrValidation() throws {
        // Baseline that should construct fine.
        _ = try KNNGraph(
            pointCount: 3, rowOffsets: [0, 1, 2, 2],
            neighborIndices: [1, 0], distances: [1, 1])

        #expect(throws: VectorError.self) { // offsets wrong length
            try KNNGraph(pointCount: 3, rowOffsets: [0, 1, 2],
                         neighborIndices: [1, 0], distances: [1, 1])
        }
        #expect(throws: VectorError.self) { // offsets do not start at 0
            try KNNGraph(pointCount: 2, rowOffsets: [1, 1, 2],
                         neighborIndices: [1, 0], distances: [1, 1])
        }
        #expect(throws: VectorError.self) { // decreasing offsets
            try KNNGraph(pointCount: 2, rowOffsets: [0, 2, 1],
                         neighborIndices: [1, 0], distances: [1, 1])
        }
        #expect(throws: VectorError.self) { // offsets end != nnz
            try KNNGraph(pointCount: 2, rowOffsets: [0, 1, 1],
                         neighborIndices: [1, 0], distances: [1, 1])
        }
        #expect(throws: VectorError.self) { // index out of range
            try KNNGraph(pointCount: 2, rowOffsets: [0, 1, 2],
                         neighborIndices: [2, 0], distances: [1, 1])
        }
        #expect(throws: VectorError.self) { // self-loop
            try KNNGraph(pointCount: 2, rowOffsets: [0, 1, 2],
                         neighborIndices: [0, 0], distances: [1, 1])
        }
        #expect(throws: VectorError.self) { // negative distance
            try KNNGraph(pointCount: 2, rowOffsets: [0, 1, 2],
                         neighborIndices: [1, 0], distances: [-1, 1])
        }
        #expect(throws: VectorError.self) { // non-finite distance
            try KNNGraph(pointCount: 2, rowOffsets: [0, 1, 2],
                         neighborIndices: [1, 0], distances: [.infinity, 1])
        }
        #expect(throws: VectorError.self) { // arrays disagree in length
            try KNNGraph(pointCount: 2, rowOffsets: [0, 1, 2],
                         neighborIndices: [1, 0], distances: [1])
        }
    }

    @Test("Brute-force contracts: tiny n, k out of range, ragged input")
    func bruteForceContracts() throws {
        let data = UMAPFixtures.clusters(seed: 67, half: 5, dimensions: 4, separation: 3, scale: 1)
        #expect(throws: VectorError.self) {
            try KNNGraph.bruteForce([data[0]], neighbors: 1)
        }
        #expect(throws: VectorError.self) {
            try KNNGraph.bruteForce(data, neighbors: 0)
        }
        #expect(throws: VectorError.self) {
            try KNNGraph.bruteForce(data, neighbors: 10) // k > n−1 = 9
        }
        var ragged = data
        ragged[2] = DynamicVector([1, 2])
        #expect(throws: VectorError.self) {
            try KNNGraph.bruteForce(ragged, neighbors: 3)
        }
    }
}

// MARK: - Fuzzy simplicial set

@Suite("UMAP - Fuzzy simplicial set")
struct FuzzySimplicialSetSuite {

    @Test("ρ and σ solve the smooth-kNN equation on a hand graph")
    func smoothDistancesHandGraph() throws {
        let graph = try UMAPFixtures.handGraph()
        let (rho, sigma) = FuzzySimplicialSet.smoothDistances(graph)

        // ρ is the nearest positive neighbor distance per row.
        #expect(rho == [1, 1, 1, 2])

        // Rows 0, 2, 3 share the same shifted-distance profile {0, 1}, so
        // each σ solves 1 + exp(−1/σ) = log₂(3); closed form σ ≈ 1.86495.
        let target = Float(Foundation.log2(3.0))
        for i in [0, 2, 3] {
            let psum = 1 + Foundation.exp(-1 / Double(sigma[i]))
            #expect(abs(Float(psum) - target) < 1e-4, "row \(i): psum = \(psum)")
            #expect(abs(sigma[i] - 1.86495) < 1e-3, "row \(i): σ = \(sigma[i])")
        }
        // Row 1's neighbors are both at d = ρ (psum is constantly 2 > the
        // target), so the bisection collapses and the floor takes over:
        // σ = 1e-3 × mean(1, 1).
        #expect(abs(sigma[1] - 0.001) < 1e-6)
    }

    @Test("Memberships and t-conorm symmetrization on the hand graph")
    func symmetrizationHandGraph() throws {
        let graph = try UMAPFixtures.handGraph()
        let edges = FuzzySimplicialSet.build(graph)
        let weights = UMAPFixtures.weightMap(edges)

        // 5 undirected pairs → 10 directed entries.
        #expect(edges.count == 10)

        // Every edge appears in both directions with the same weight.
        for e in 0..<edges.count {
            let reverse = weights[UMAPFixtures.edgeKey(edges.tails[e], edges.heads[e])]
            #expect(reverse == edges.weights[e])
            #expect(edges.weights[e] > 0 && edges.weights[e] <= 1)
        }

        // Hand-checked fused weights. exp(−1/σ) with σ ≈ 1.86495 is
        // w ≈ 0.58496:
        //   (0,1): 1 ∪ 1 = 1            (mutual nearest neighbors)
        //   (1,2): 1 ∪ 1 = 1
        //   (2,3): 0.58496 ∪ 1 = 1      (t-conorm with a certain edge)
        //   (0,2): 0.58496 ∪ ∅ = 0.58496 (one-directional edge survives)
        //   (1,3): ∅ ∪ 0.58496 = 0.58496
        let w = Float(0.58496)
        #expect(abs(weights[UMAPFixtures.edgeKey(0, 1)]! - 1) < 1e-4)
        #expect(abs(weights[UMAPFixtures.edgeKey(1, 2)]! - 1) < 1e-4)
        #expect(abs(weights[UMAPFixtures.edgeKey(2, 3)]! - 1) < 1e-4)
        #expect(abs(weights[UMAPFixtures.edgeKey(0, 2)]! - w) < 1e-3)
        #expect(abs(weights[UMAPFixtures.edgeKey(1, 3)]! - w) < 1e-3)
        #expect(weights[UMAPFixtures.edgeKey(0, 3)] == nil, "0 and 3 are not kNN-adjacent")
    }

    @Test("Isolated rows contribute no edges and keep σ = 0")
    func isolatedRow() throws {
        let graph = try KNNGraph(
            pointCount: 3, rowOffsets: [0, 1, 2, 2],
            neighborIndices: [1, 0], distances: [1, 1])
        let (rho, sigma) = FuzzySimplicialSet.smoothDistances(graph)
        #expect(rho[2] == 0 && sigma[2] == 0)

        let edges = FuzzySimplicialSet.build(graph)
        #expect(edges.count == 2)
        for e in 0..<edges.count {
            #expect(edges.heads[e] != 2 && edges.tails[e] != 2)
            #expect(edges.weights[e] == 1) // mutual NN at d == ρ
        }
    }

    @Test("Symmetry and weight range hold on a real brute-force graph")
    func symmetryOnRealGraph() throws {
        let data = UMAPFixtures.clusters(seed: 71, half: 25, dimensions: 8, separation: 6, scale: 0.8)
        let graph = try KNNGraph.bruteForce(data, neighbors: 6)
        let edges = FuzzySimplicialSet.build(graph)
        let weights = UMAPFixtures.weightMap(edges)

        #expect(edges.count >= graph.edgeCount, "union can only add directions")
        for e in 0..<edges.count {
            #expect(edges.weights[e] > 0 && edges.weights[e] <= 1)
            #expect(weights[UMAPFixtures.edgeKey(edges.tails[e], edges.heads[e])] == edges.weights[e])
        }
        // Every point's nearest neighbor edge has weight 1 (d == ρ ⇒
        // membership 1): the t-conorm gives 1 + w − 1·w, which is exactly 1
        // up to one Float rounding of the add/subtract pair.
        for i in 0..<graph.pointCount {
            let first = graph.neighborRange(of: i).lowerBound
            let nearest = graph.neighborIndices[first]
            let weight = weights[UMAPFixtures.edgeKey(Int32(i), nearest)] ?? 0
            #expect(abs(weight - 1) < 1e-6, "point \(i) nearest-neighbor weight: \(weight)")
        }
    }
}

// MARK: - Layout

@Suite("UMAP - Layout")
struct UMAPLayoutSuite {

    @Test("Well-separated clusters stay separated (PCA and random init)")
    func clusterSeparation() throws {
        let half = 60
        let data = UMAPFixtures.clusters(seed: 73, half: half, dimensions: 10, separation: 12, scale: 0.5)

        for initialization in [UMAPInitialization.pca, .random] {
            let config = UMAPConfig(neighbors: 10, initialization: initialization)
            let result = try Operations.umap(data, config: config)
            #expect(result.pointCount == 2 * half)
            #expect(result.dimension == 2)
            #expect(result.coordinates.count == 2 * half * 2)
            let allFinite = result.coordinates.allSatisfy { $0.isFinite }
            #expect(allFinite)
            let ratio = UMAPFixtures.separationRatio(result, half: half)
            #expect(ratio > 2, "init \(initialization): separation ratio \(ratio)")
        }
    }

    @Test("Same seed reproduces the layout exactly; a different seed does not")
    func determinism() throws {
        let data = UMAPFixtures.clusters(seed: 79, half: 30, dimensions: 8, separation: 8, scale: 0.6)
        let config = UMAPConfig(neighbors: 8, epochs: 150)

        let first = try Operations.umap(data, config: config)
        let second = try Operations.umap(data, config: config)
        #expect(first.coordinates == second.coordinates)

        var reseeded = config
        reseeded.seed = 1
        let third = try Operations.umap(data, config: reseeded)
        #expect(first.coordinates != third.coordinates)
    }

    @Test("Injected graph takes the same path as the internal brute force")
    func graphInjectionEquivalence() throws {
        let data = UMAPFixtures.clusters(seed: 83, half: 25, dimensions: 8, separation: 8, scale: 0.6)
        let config = UMAPConfig(neighbors: 6, epochs: 100)
        let graph = try KNNGraph.bruteForce(data, neighbors: config.neighbors)

        let implicit = try Operations.umap(data, config: config)
        let injected = try Operations.umap(data, graph: graph, config: config)
        #expect(implicit.coordinates == injected.coordinates)
    }

    @Test("Graph-only entry point: custom initial coordinates and random fallback")
    func graphEntryPoint() throws {
        let data = UMAPFixtures.clusters(seed: 89, half: 25, dimensions: 8, separation: 8, scale: 0.6)
        let config = UMAPConfig(neighbors: 6, epochs: 100)
        let graph = try KNNGraph.bruteForce(data, neighbors: config.neighbors)

        // PCA projection as explicit init — the corpus-scale composition.
        let projected = try Operations.pca(data, components: 2).projected
        let seeded = try Operations.umap(
            graph: graph, initialCoordinates: projected, config: config)
        let seededFinite = seeded.coordinates.allSatisfy { $0.isFinite }
        #expect(seededFinite)
        #expect(UMAPFixtures.separationRatio(seeded, half: 25) > 2)

        // Random fallback is deterministic too.
        let randomA = try Operations.umap(graph: graph, config: config)
        let randomB = try Operations.umap(graph: graph, config: config)
        #expect(randomA.coordinates == randomB.coordinates)
    }
}

// MARK: - API behavior

@Suite("UMAP - API behavior")
struct UMAPBehaviorSuite {

    @Test("Result accessors: point rows and the 2-D SIMD view")
    func resultAccessors() throws {
        let data = UMAPFixtures.clusters(seed: 97, half: 15, dimensions: 5, separation: 6, scale: 0.5)
        let result = try Operations.umap(data, config: UMAPConfig(neighbors: 5, epochs: 50))

        let points = try #require(result.points2D)
        #expect(points.count == result.pointCount)
        for i in [0, 7, result.pointCount - 1] {
            let row = result.point(i)
            #expect(row.count == 2)
            #expect(row[0] == result.coordinates[2 * i] && row[1] == result.coordinates[2 * i + 1])
            #expect(points[i] == SIMD2(row[0], row[1]))
        }

        let volume = try Operations.umap(
            data, dimensions: 3, config: UMAPConfig(neighbors: 5, epochs: 50))
        #expect(volume.points2D == nil)
        #expect(volume.dimension == 3)
        #expect(volume.coordinates.count == result.pointCount * 3)
    }

    @Test("Attraction-only runs (negativeSampleRate 0)")
    func attractionOnly() throws {
        let data = UMAPFixtures.clusters(seed: 101, half: 15, dimensions: 5, separation: 6, scale: 0.5)
        let result = try Operations.umap(
            data, config: UMAPConfig(neighbors: 5, epochs: 50, negativeSampleRate: 0))
        let allFinite = result.coordinates.allSatisfy { $0.isFinite }
        #expect(allFinite)
    }

    @Test("Error contracts: config, shapes, and initialization limits")
    func errorContracts() throws {
        let data = UMAPFixtures.clusters(seed: 103, half: 10, dimensions: 4, separation: 5, scale: 0.5)
        let graph = try KNNGraph.bruteForce(data, neighbors: 4)

        #expect(throws: VectorError.self) { // neighbors < 2
            try Operations.umap(data, config: UMAPConfig(neighbors: 1))
        }
        #expect(throws: VectorError.self) { // minDist > spread
            try Operations.umap(data, config: UMAPConfig(minDist: 2, spread: 1))
        }
        #expect(throws: VectorError.self) { // non-positive learning rate
            try Operations.umap(data, config: UMAPConfig(learningRate: 0))
        }
        #expect(throws: VectorError.self) { // epochs < 1
            try Operations.umap(data, config: UMAPConfig(epochs: 0))
        }
        #expect(throws: VectorError.self) { // negative sampling rate < 0
            try Operations.umap(data, config: UMAPConfig(negativeSampleRate: -1))
        }
        #expect(throws: VectorError.self) { // output dimension < 1
            try Operations.umap(data, dimensions: 0, config: UMAPConfig(neighbors: 4))
        }
        #expect(throws: VectorError.self) { // k > n−1 in the brute-force path
            try Operations.umap(Array(data.prefix(8)), config: UMAPConfig(neighbors: 15))
        }
        #expect(throws: VectorError.self) { // graph size != vector count
            try Operations.umap(Array(data.prefix(10)), graph: graph,
                                config: UMAPConfig(neighbors: 4))
        }
        #expect(throws: VectorError.self) { // initial coordinate count != n
            try Operations.umap(graph: graph,
                                initialCoordinates: [[0, 0], [1, 1]],
                                config: UMAPConfig(neighbors: 4))
        }
        #expect(throws: VectorError.self) { // ragged initial coordinates
            var coords = [[Float]](repeating: [0, 0], count: graph.pointCount)
            coords[3] = [1, 2, 3]
            _ = try Operations.umap(graph: graph, initialCoordinates: coords,
                                    config: UMAPConfig(neighbors: 4))
        }
        #expect(throws: VectorError.self) { // PCA init impossible: d < dimensions
            let thin = (0..<20).map { DynamicVector([Float($0) * 0.1]) }
            _ = try Operations.umap(thin, config: UMAPConfig(neighbors: 3))
        }
    }
}
