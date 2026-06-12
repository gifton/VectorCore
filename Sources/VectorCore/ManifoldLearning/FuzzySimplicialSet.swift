//
//  FuzzySimplicialSet.swift
//  VectorCore
//
//  UMAP step 1 (gap analysis §3.3.1): convert a kNN graph into a fuzzy
//  simplicial set. Per point, a binary search finds the adaptive bandwidth
//  σᵢ whose exponential memberships sum to a log₂ perplexity-like target;
//  the directed memberships are then fused into a symmetric weighted graph
//  with the probabilistic t-conorm. Pure math on the CSR graph, following
//  the reference implementation (McInnes, Healy & Melville 2018; umap-learn
//  `smooth_knn_dist` / `compute_membership_strengths` / fuzzy set union
//  with set_op_mix_ratio = 1).
//

import Foundation

/// Symmetric weighted edge list — both directions of every pair are present
/// with equal weight, sorted by (head, tail). The layout stage's input.
internal struct UMAPEdgeList {
    var heads: [Int32]
    var tails: [Int32]
    var weights: [Float]
    var count: Int { heads.count }
}

internal enum FuzzySimplicialSet {
    /// umap-learn constants: psum convergence tolerance, σ floor scale,
    /// and bisection depth.
    private static let smoothTolerance = 1e-5
    private static let minBandwidthScale = 1e-3
    private static let searchIterations = 64

    /// ρᵢ (distance to the nearest non-duplicate neighbor) and σᵢ solving
    /// `Σⱼ exp(−max(0, dᵢⱼ − ρᵢ)/σᵢ) = log₂(kᵢ + 1)` by bisection.
    ///
    /// The +1 mirrors umap-learn, whose `n_neighbors` counts the point
    /// itself (its kNN rows carry a zero-distance self entry); `KNNGraph`
    /// forbids self-loops, so kᵢ stored neighbors correspond to
    /// n_neighbors = kᵢ + 1. Rows with no entries (isolated points) keep
    /// ρ = σ = 0 and contribute no edges.
    static func smoothDistances(_ graph: KNNGraph) -> (rho: [Float], sigma: [Float]) {
        let n = graph.pointCount
        var rho = [Float](repeating: 0, count: n)
        var sigma = [Float](repeating: 0, count: n)

        var globalSum = 0.0
        for dist in graph.distances { globalSum += Double(dist) }
        let globalMean = graph.distances.isEmpty ? 0 : globalSum / Double(graph.distances.count)

        for i in 0..<n {
            let range = graph.neighborRange(of: i)
            let k = range.count
            if k == 0 { continue }

            var rowSum = 0.0
            var minPositive = Double.infinity
            for idx in range {
                let dist = Double(graph.distances[idx])
                rowSum += dist
                if dist > 0 && dist < minPositive { minPositive = dist }
            }
            let rhoI = minPositive.isFinite ? minPositive : 0
            rho[i] = Float(rhoI)

            let target = Foundation.log2(Double(k + 1))
            var low = 0.0
            var high = Double.infinity
            var mid = 1.0
            for _ in 0..<searchIterations {
                var psum = 0.0
                for idx in range {
                    let shifted = Double(graph.distances[idx]) - rhoI
                    psum += shifted > 0 ? Foundation.exp(-shifted / mid) : 1
                }
                if abs(psum - target) < smoothTolerance { break }
                if psum > target {
                    high = mid
                    mid = (low + high) / 2
                } else {
                    low = mid
                    mid = high.isInfinite ? mid * 2 : (low + high) / 2
                }
            }
            // Bandwidth floor: a fraction of the local (or, for pure
            // duplicate rows, global) mean distance keeps σ off zero.
            let reference = rhoI > 0 ? rowSum / Double(k) : globalMean
            sigma[i] = Float(max(mid, minBandwidthScale * reference))
        }
        return (rho, sigma)
    }

    /// Directed memberships `exp(−max(0, d − ρᵢ)/σᵢ)` fused into a
    /// symmetric edge list with the probabilistic t-conorm
    /// `w = w₁ + w₂ − w₁·w₂` (fuzzy set union). Duplicate directed entries
    /// merge by max before fusing; pairs whose fused weight underflows to
    /// zero are dropped.
    static func build(_ graph: KNNGraph) -> UMAPEdgeList {
        let (rho, sigma) = smoothDistances(graph)
        let n = graph.pointCount

        // Key each directed membership by its canonical (low, high) pair so
        // one sort lands both directions adjacently.
        var canonical = [(key: UInt64, forward: Float, backward: Float)]()
        canonical.reserveCapacity(graph.edgeCount)
        for i in 0..<n {
            let rhoI = Double(rho[i])
            let sigmaI = Double(sigma[i])
            for idx in graph.neighborRange(of: i) {
                let j = Int(graph.neighborIndices[idx])
                let shifted = Double(graph.distances[idx]) - rhoI
                let weight: Float = (shifted <= 0 || sigmaI == 0)
                    ? 1
                    : Float(Foundation.exp(-shifted / sigmaI))
                if i < j {
                    canonical.append((UInt64(i) << 32 | UInt64(j), weight, 0))
                } else {
                    canonical.append((UInt64(j) << 32 | UInt64(i), 0, weight))
                }
            }
        }
        canonical.sort { $0.key < $1.key }

        var fused = [(key: UInt64, weight: Float)]()
        fused.reserveCapacity(2 * canonical.count)
        var cursor = 0
        while cursor < canonical.count {
            let key = canonical[cursor].key
            var forward: Float = 0
            var backward: Float = 0
            while cursor < canonical.count && canonical[cursor].key == key {
                forward = max(forward, canonical[cursor].forward)
                backward = max(backward, canonical[cursor].backward)
                cursor += 1
            }
            let weight = forward + backward - forward * backward
            if weight > 0 {
                let low = key >> 32
                let high = key & 0xFFFF_FFFF
                fused.append((low << 32 | high, weight))
                fused.append((high << 32 | low, weight))
            }
        }
        fused.sort { $0.key < $1.key }

        var heads = [Int32]()
        var tails = [Int32]()
        var weights = [Float]()
        heads.reserveCapacity(fused.count)
        tails.reserveCapacity(fused.count)
        weights.reserveCapacity(fused.count)
        for (key, weight) in fused {
            heads.append(Int32(key >> 32))
            tails.append(Int32(key & 0xFFFF_FFFF))
            weights.append(weight)
        }
        return UMAPEdgeList(heads: heads, tails: tails, weights: weights)
    }
}
