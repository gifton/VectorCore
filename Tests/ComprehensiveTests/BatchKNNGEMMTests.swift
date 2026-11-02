//
//  BatchKNNGEMMTests.swift
//  VectorCore
//
//  Verifies Operations.findNearestBatch with the explicit `.gram` CPU strategy (P02; formerly
//  an implicit route at >= 8 queries and >= 256 candidates) runs the tiled GEMM route, observed
//  through the task-local tile seam, and stays near the per-query reference; and that the
//  default `.perQuery` strategy runs no GEMM tile and matches the per-query reference.
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

    /// Route observation: every query x candidate pair is scored by exactly one GEMM tile whose BLAS inner
    /// dimension is the whole 512.
    private func expectGramTiles(_ log: P02GramTraceLog, queries: Int, candidates: Int) {
        let tiles = log.tiles
        #expect(!tiles.isEmpty, "the .gram strategy ran no GEMM tile")
        #expect(tiles.allSatisfy { $0.k == 512 && $0.lda == 512 && $0.ldb == 512 })
        #expect(tiles.reduce(0) { $0 + $1.m * $1.n } == queries * candidates)
    }

    @Test("Explicit .gram Euclidean batch k-NN runs GEMM tiles and stays near the per-query reference")
    func euclideanBatchMatchesPerQuery() async throws {
        let qs = vecs512(16, seed: 0xB1)
        let cs = vecs512(300, seed: 0xB2)         // 8 x 256 tiles: 2 x 2 tiles, the last candidate tile partial
        let k = 10
        let log = P02GramTraceLog()
        let batch = try await Operations.$gramTrace.withValue({ log.append($0) }, operation: {
            try await Operations.findNearestBatch(queries: qs, in: cs, k: k, cpuStrategy: p02Gram(8, 256))
        })
        expectGramTiles(log, queries: 16, candidates: 300)
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

    @Test("Explicit .gram Cosine batch k-NN runs GEMM tiles and stays near the per-query reference")
    func cosineBatchMatchesPerQuery() async throws {
        let qs = vecs512(12, seed: 0xC1)
        let cs = vecs512(280, seed: 0xC2)
        let k = 8
        let log = P02GramTraceLog()
        let batch = try await Operations.$gramTrace.withValue({ log.append($0) }, operation: {
            try await Operations.findNearestBatch(queries: qs, in: cs, k: k, metric: CosineDistance(),
                                                  cpuStrategy: p02Gram(8, 256))
        })
        expectGramTiles(log, queries: 12, candidates: 280)
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

    @Test("The default .perQuery strategy runs no GEMM tile and matches the per-query reference")
    func smallBatchFallsBack() async throws {
        let qs = vecs512(4, seed: 0xD1)
        let cs = vecs512(300, seed: 0xD2)
        let k = 5
        let log = P02GramTraceLog()
        let batch = try await Operations.$gramTrace.withValue({ log.append($0) }, operation: {
            try await Operations.findNearestBatch(queries: qs, in: cs, k: k)
        })
        #expect(log.snapshot.isEmpty, "the default strategy ran Gram work")
        for i in 0..<4 {
            let ref = try await Operations.findNearest(to: qs[i], in: cs, k: k)
            #expect(batch[i].map { $0.index } == ref.map { $0.index })
        }
    }
}
