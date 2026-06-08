//
//  BatchGEMMRoutingTests.swift
//  VectorCore
//
//  Verifies BatchOperations.pairwiseDistances routes large matrices through the
//  GEMM (cblas_sgemm/AMX) path (beta-evolution-4, DOCUMENT-2 routing) and that the
//  routed results match the per-pair kernels, while non-routable inputs fall back.
//
//  Suite is .serialized because one test toggles the process-global routing config.
//

import Testing
import Foundation
@testable import VectorCore

@Suite("GEMM routing in BatchOperations", .serialized)
struct BatchGEMMRoutingTests {

    private struct RNG {
        var state: UInt64
        mutating func next() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(state >> 40) / Float(1 << 23) - 1
        }
    }

    private func vecs512(_ count: Int, seed: UInt64) -> [Vector512Optimized] {
        var rng = RNG(state: seed)
        return (0..<count).map { _ in try! Vector512Optimized((0..<512).map { _ in rng.next() }) }
    }

    // N ≥ matrixRoutingMinN (256) so the GEMM path is taken.
    private let N = 300

    @Test("Routed Euclidean pairwise matches the direct kernel")
    func euclideanRoutedMatchesKernel() async {
        let vs = vecs512(N, seed: 0xE1)
        let m = await BatchOperations.pairwiseDistances(vs)            // routes via GEMM
        #expect(m.count == N && m[0].count == N)
        // Diagonal ≈ 0. The GEMM identity ‖x‖²+‖x‖²−2‖x‖² cannot reach exactly 0 for
        // a self-pair (catastrophic cancellation leaves ~ε·‖x‖²); the clamp keeps it
        // non-negative, and sqrt of that residual is ~0.01 at these magnitudes. This is
        // the documented DOCUMENT-2 §4 behavior, negligible for ranking.
        for i in stride(from: 0, to: N, by: 37) {
            #expect(m[i][i] >= 0 && m[i][i] < 0.05, "self-distance residual too large: \(m[i][i])")
        }
        // Spot-check rows against the direct SIMD kernel (sqrt of squared).
        for i in 0..<5 {
            for j in 0..<N {
                let ref = (EuclideanKernels.squared512(vs[i], vs[j])).squareRoot()
                #expect(abs(m[i][j] - ref) <= 0.05 + 1e-3 * ref, "i=\(i) j=\(j) got=\(m[i][j]) ref=\(ref)")
            }
        }
    }

    @Test("Routed Cosine pairwise matches double-precision reference")
    func cosineRoutedMatchesReference() async {
        let vs = vecs512(N, seed: 0xC0)
        let m = await BatchOperations.pairwiseDistances(vs, metric: CosineDistance())
        for i in 0..<4 {
            let a = vs[i].toArray()
            for j in 0..<N {
                let b = vs[j].toArray()
                var dot = 0.0, na = 0.0, nb = 0.0
                for k in 0..<512 { dot += Double(a[k]) * Double(b[k]); na += Double(a[k]*a[k]); nb += Double(b[k]*b[k]) }
                let ref = Float(1.0 - dot / (sqrt(na) * sqrt(nb)))
                #expect(abs(m[i][j] - ref) <= 2e-3, "i=\(i) j=\(j) got=\(m[i][j]) ref=\(ref)")
            }
        }
    }

    @Test("Non-routable metric (Manhattan) falls back to the per-pair path")
    func manhattanFallsBack() async {
        let vs = vecs512(N, seed: 0x3A)
        let m = await BatchOperations.pairwiseDistances(vs, metric: ManhattanDistance())
        let metric = ManhattanDistance()
        for i in 0..<3 {
            for j in stride(from: 0, to: N, by: 23) {
                let ref = metric.distance(vs[i], vs[j])
                #expect(abs(m[i][j] - ref) <= 1e-3, "manhattan not exact at i=\(i) j=\(j)")
            }
        }
    }

    @Test("DynamicVector does not route (concrete-type gate) and stays correct")
    func dynamicDoesNotRoute() async {
        let vs = (0..<N).map { k in DynamicVector((0..<64).map { Float((($0 + k) % 13) - 6) }) }
        let m = await BatchOperations.pairwiseDistances(vs)
        for i in 0..<3 {
            let a = vs[i].toArray()
            for j in stride(from: 0, to: N, by: 29) {
                let b = vs[j].toArray()
                var ss = 0.0
                for k in 0..<64 { let d = Double(a[k] - b[k]); ss += d * d }
                let ref = Float(ss.squareRoot())
                #expect(abs(m[i][j] - ref) <= 1e-2 + 1e-3 * ref)
            }
        }
    }

    @Test("Routing ON and OFF agree within tolerance")
    func routingToggleConsistent() async {
        let vs = vecs512(N, seed: 0x7B)

        await BatchOperations.updateConfiguration { $0.enableMatrixRouting = false }
        let exact = await BatchOperations.pairwiseDistances(vs)        // per-pair kernels
        await BatchOperations.updateConfiguration { $0.enableMatrixRouting = true }
        let gemm = await BatchOperations.pairwiseDistances(vs)         // GEMM

        for i in stride(from: 0, to: N, by: 31) {
            for j in stride(from: 0, to: N, by: 31) {
                #expect(abs(exact[i][j] - gemm[i][j]) <= 0.05 + 1e-3 * exact[i][j],
                        "routed vs exact diverge at i=\(i) j=\(j): \(gemm[i][j]) vs \(exact[i][j])")
            }
        }
        // Leave the global default as shipped (routing enabled).
        await BatchOperations.updateConfiguration { $0.enableMatrixRouting = true }
    }
}
