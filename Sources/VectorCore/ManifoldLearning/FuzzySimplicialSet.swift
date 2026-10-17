//
//  FuzzySimplicialSet.swift
//  VectorCore
//
//  UMAP step 1 (gap analysis §3.3.1): convert a kNN graph into a fuzzy
//  simplicial set. Per point, a binary search finds the adaptive bandwidth
//  σᵢ whose exponential memberships sum to a log₂ perplexity-like target;
//  the directed memberships are then fused into a symmetric weighted graph
//  with the probabilistic t-conorm. Calibration uses the explicit limiting,
//  floor and checked residual policies documented below; it is not a promise
//  of bitwise equivalence to umap-learn.
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

    /// Double rho/bandwidth calibration. Stored duplicate entries count separately
    /// in k; fusion subsequently merges duplicate IDs by max. Isolated rows stay zero.
    /// Contract: infeasible/equality rows use a zero pre-floor bandwidth, and the
    /// local/global mean floor may change the solved target. UMAPRangeContractTests
    /// separates roots, ties, floor-dominated rows and common distance scales.
    static func smoothDistances(_ graph: KNNGraph) throws -> (rho: [Double], sigma: [Double]) {
        let n = try UMAPChecked.count([graph.pointCount], stride: MemoryLayout<Double>.stride, label: "smooth rows")
        var rho = [Double](repeating: 0, count: n)
        var sigma = [Double](repeating: 0, count: n)
        var globalSum = 0.0
        for distance in graph.distances { globalSum += Double(distance) }
        let globalMean = graph.distances.isEmpty ? 0 : globalSum / Double(graph.distances.count)
        guard globalMean.isFinite else { throw UMAPChecked.failure("smooth global mean") }

        for i in 0..<n {
            let range = graph.neighborRange(of: i)
            let k = range.count
            if k == 0 { continue }
            _ = try UMAPChecked.count([k], stride: MemoryLayout<Double>.stride, label: "smooth row workspace")
            var sum = 0.0, nearest = Double.infinity
            for index in range {
                let distance = Double(graph.distances[index])
                sum += distance
                if distance > 0 { nearest = min(nearest, distance) }
            }
            let localRho = nearest.isFinite ? nearest : 0
            rho[i] = localRho
            let deltas = range.map { max(0, Double(graph.distances[$0]) - localRho) }
            let zeros = deltas.reduce(0) { $0 + ($1 == 0 ? 1 : 0) }
            let target = Foundation.log2(Double(k) + 1)
            var bandwidth = 0.0
            if Double(zeros) < target {
                guard let scale = deltas.max(), scale > 0, scale.isFinite else {
                    throw UMAPChecked.failure("smooth positive scale")
                }
                let logs = deltas.filter { $0 > 0 }.map { Foundation.log($0 / scale) }
                guard let minimumLog = logs.min(), logs.allSatisfy({ $0.isFinite }) else {
                    throw UMAPChecked.failure("smooth normalized deltas")
                }
                let tolerance = smoothTolerance * max(1, target)
                let epsilon = min(tolerance / 4, (target - Double(zeros)) / 2)
                let positiveCount = Double(k - zeros)
                guard epsilon.isFinite, epsilon > 0, epsilon < positiveCount else {
                    throw UMAPChecked.failure("smooth lower bracket epsilon")
                }
                var low = minimumLog - Foundation.log(-Foundation.log(epsilon / positiveCount))
                var high = Foundation.log(2.0)
                func total(_ logTau: Double) throws -> Double {
                    var total = Double(zeros)
                    for logDelta in logs {
                        let exponent = Foundation.exp(logDelta - logTau)
                        guard exponent.isFinite else { throw UMAPChecked.failure("smooth exponential range") }
                        total += Foundation.exp(-exponent)
                    }
                    guard total.isFinite else { throw UMAPChecked.failure("smooth membership sum") }
                    return total
                }
                guard low.isFinite, high.isFinite, low <= high else { throw UMAPChecked.failure("smooth log bracket") }
                let lowValue = try total(low), highValue = try total(high)
                var accepted: Double?
                if abs(lowValue - target) <= tolerance { accepted = low }
                else if abs(highValue - target) <= tolerance { accepted = high }
                else {
                    guard lowValue < target, highValue > target else { throw UMAPChecked.failure("smooth bracket does not straddle target") }
                    for _ in 0..<searchIterations {
                        let middle = low + (high - low) / 2
                        let value = try total(middle)
                        if abs(value - target) <= tolerance { accepted = middle; break }
                        if value < target { low = middle } else { high = middle }
                    }
                }
                guard let logTau = accepted else { throw UMAPChecked.failure("smooth residual unmet after 64 log bisections") }
                bandwidth = scale * Foundation.exp(logTau)
                guard bandwidth.isFinite, bandwidth > 0 else { throw UMAPChecked.failure("smooth physical bandwidth") }
            }
            let reference = localRho > 0 ? sum / Double(k) : globalMean
            let floor = minBandwidthScale * reference
            guard floor.isFinite, floor >= 0 else { throw UMAPChecked.failure("smooth floor") }
            sigma[i] = max(bandwidth, floor)
        }
        return (rho, sigma)
    }

    /// Zero bandwidth is the limiting membership, not unconditional certainty.
    /// Exercised directly and through build by UMAPRangeContractTests.
    static func membership(distance: Double, rho: Double, sigma: Double) throws -> Double {
        guard distance.isFinite, rho.isFinite, sigma.isFinite, distance >= 0, rho >= 0, sigma >= 0 else {
            throw UMAPChecked.failure("membership input range")
        }
        let delta = max(0, distance - rho)
        let weight = delta == 0 ? 1 : (sigma == 0 ? 0 : Foundation.exp(-delta / sigma))
        guard weight.isFinite, weight >= 0, weight <= 1 else { throw UMAPChecked.failure("membership weight") }
        return weight
    }

    /// Directed memberships `exp(−max(0, d − ρᵢ)/σᵢ)` fused into a
    /// symmetric edge list with the probabilistic t-conorm
    /// `w = w₁ + w₂ − w₁·w₂` (fuzzy set union). Duplicate directed entries
    /// merge by max before fusing; pairs whose fused weight underflows to
    /// zero are dropped.
    static func build(_ graph: KNNGraph) throws -> UMAPEdgeList {
        let (rho, sigma) = try smoothDistances(graph)
        let n = graph.pointCount

        // Key each directed membership by its canonical (low, high) pair so
        // one sort lands both directions adjacently.
        var canonical = [(key: UInt64, forward: Float, backward: Float)]()
        _ = try UMAPChecked.count([graph.edgeCount], stride: MemoryLayout<(UInt64, Float, Float)>.stride, label: "canonical edges")
        let maximumFused = try UMAPChecked.count([2, graph.edgeCount], stride: MemoryLayout<(UInt64, Float)>.stride, label: "fused edges")
        _ = try UMAPChecked.count([maximumFused], stride: MemoryLayout<Float>.stride, label: "edge output")
        canonical.reserveCapacity(graph.edgeCount)
        for i in 0..<n {
            let rhoI = Double(rho[i])
            let sigmaI = Double(sigma[i])
            for idx in graph.neighborRange(of: i) {
                let j = Int(graph.neighborIndices[idx])
                let weight = Float(try membership(distance: Double(graph.distances[idx]), rho: rhoI, sigma: sigmaI))
                if i < j {
                    canonical.append((UInt64(i) << 32 | UInt64(j), weight, 0))
                } else {
                    canonical.append((UInt64(j) << 32 | UInt64(i), 0, weight))
                }
            }
        }
        canonical.sort { $0.key < $1.key }

        var fused = [(key: UInt64, weight: Float)]()
        fused.reserveCapacity(maximumFused)
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
            guard weight.isFinite, weight >= 0, weight <= 1 else { throw UMAPChecked.failure("fused weight range") }
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
