//
//  BatchKNNGEMMTests.swift
//  VectorCore
//
//  Verifies Operations.findNearestBatch routes large multi-query k-NN through the
//  GEMM matrix + per-row Top-K path (beta-evolution-4, DOCUMENT-2), matching the
//  per-query reference, and falls back below the crossover sizes.
//

import Testing
import Foundation
@testable import VectorCore

@Suite("GEMM batch k-NN (findNearestBatch routing)")
struct BatchKNNGEMMTests {

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

    @Test("Routed Euclidean batch k-NN matches the per-query reference")
    func euclideanBatchMatchesPerQuery() async throws {
        let qs = vecs512(16, seed: 0xB1)          // q ≥ 8
        let cs = vecs512(300, seed: 0xB2)         // n ≥ 256 → GEMM routes
        let k = 10
        let batch = try await Operations.findNearestBatch(queries: qs, in: cs, k: k)
        #expect(batch.count == 16)
        for i in 0..<16 {
            let ref = try await Operations.findNearest(to: qs[i], in: cs, k: k)
            #expect(batch[i].count == k)
            // Nearest neighbour index is robust; compare it exactly.
            #expect(batch[i][0].index == ref[0].index, "query \(i): nearest index differs")
            // Positional distances agree within tolerance (robust to rare boundary swaps,
            // whose distances are near-equal by construction).
            for t in 0..<k {
                #expect(abs(batch[i][t].distance - ref[t].distance) <= 0.05 + 1e-3 * ref[t].distance,
                        "query \(i) rank \(t): \(batch[i][t].distance) vs \(ref[t].distance)")
            }
        }
    }

    @Test("Routed Cosine batch k-NN matches the per-query reference")
    func cosineBatchMatchesPerQuery() async throws {
        let qs = vecs512(12, seed: 0xC1)
        let cs = vecs512(280, seed: 0xC2)
        let k = 8
        let batch = try await Operations.findNearestBatch(queries: qs, in: cs, k: k, metric: CosineDistance())
        for i in 0..<12 {
            let ref = try await Operations.findNearest(to: qs[i], in: cs, k: k, metric: CosineDistance())
            #expect(batch[i].count == k)
            #expect(batch[i][0].index == ref[0].index, "query \(i): nearest index differs")
            for t in 0..<k {
                #expect(abs(batch[i][t].distance - ref[t].distance) <= 2e-3,
                        "query \(i) rank \(t): \(batch[i][t].distance) vs \(ref[t].distance)")
            }
        }
    }

    @Test("Small batch (q < crossover) falls back and stays correct")
    func smallBatchFallsBack() async throws {
        let qs = vecs512(4, seed: 0xD1)           // q < 8 → no GEMM routing
        let cs = vecs512(300, seed: 0xD2)
        let k = 5
        let batch = try await Operations.findNearestBatch(queries: qs, in: cs, k: k)
        for i in 0..<4 {
            let ref = try await Operations.findNearest(to: qs[i], in: cs, k: k)
            #expect(batch[i].map { $0.index } == ref.map { $0.index })
        }
    }
}
