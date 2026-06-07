//
//  MatrixDistanceTests.swift
//  VectorCore
//
//  Parity + numerical-contract tests for the GEMM batch-distance path
//  (beta-evolution-4, DOCUMENT-2). Asserts the GEMM identity agrees with the
//  direct SIMD kernels, and that the cancellation clamp prevents negative/NaN.
//

import Testing
import Foundation
@testable import VectorCore

@Suite("MatrixDistance (GEMM batch-distance)")
struct MatrixDistanceTests {

    /// Deterministic pseudo-random Float in [-1, 1] (SplitMix64-based; no global RNG).
    private struct RNG {
        var state: UInt64
        mutating func next() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let bits = (state >> 40)               // 24 high bits
            return Float(bits) / Float(1 << 23) - 1 // → [-1, 1)
        }
    }

    private func makeVecs512(_ count: Int, seed: UInt64) -> [Vector512Optimized] {
        var rng = RNG(state: seed)
        return (0..<count).map { _ in
            try! Vector512Optimized((0..<512).map { _ in rng.next() })
        }
    }

    /// Double-precision ground-truth cosine distance.
    private func cosineRef(_ a: Vector512Optimized, _ b: Vector512Optimized) -> Float {
        let av = a.toArray(), bv = b.toArray()
        var dot = 0.0, na = 0.0, nb = 0.0
        for k in 0..<512 {
            dot += Double(av[k]) * Double(bv[k])
            na += Double(av[k]) * Double(av[k])
            nb += Double(bv[k]) * Double(bv[k])
        }
        return Float(1.0 - dot / (sqrt(na) * sqrt(nb)))
    }

    @Test("Euclidean GEMM matches the direct SIMD kernel")
    func euclideanParity512() {
        let qs = makeVecs512(4, seed: 0x1234)
        let cs = makeVecs512(10, seed: 0x9ABC)
        let out = MatrixDistance.euclideanSquaredMatrix(queries: qs, candidates: cs)
        #expect(out.count == 4 * 10)
        for i in 0..<4 {
            for j in 0..<10 {
                let ref = EuclideanKernels.squared512(qs[i], cs[j])
                let got = out[i * 10 + j]
                // Generous absolute + relative band; GEMM vs direct agree to ~1e-4 rel here.
                #expect(abs(got - ref) <= 1e-2 + 1e-3 * abs(ref),
                        "euclid mismatch i=\(i) j=\(j) got=\(got) ref=\(ref)")
            }
        }
    }

    @Test("Identical query/candidate clamps to zero (no negative, no NaN)")
    func euclideanIdentityClamp() {
        let v = makeVecs512(1, seed: 0x7777)
        let out = MatrixDistance.euclideanSquaredMatrix(queries: v, candidates: v)
        #expect(out.count == 1)
        #expect(out[0] >= 0, "must be clamped non-negative")
        #expect(out[0].isFinite, "must not be NaN/Inf")
        #expect(out[0] < 1e-1, "self-distance ≈ 0")
    }

    @Test("Cosine GEMM matches double-precision ground truth")
    func cosineParity512() {
        let qs = makeVecs512(3, seed: 0x55)
        let cs = makeVecs512(8, seed: 0xAA)
        let out = MatrixDistance.cosineDistanceMatrix(queries: qs, candidates: cs)
        for i in 0..<3 {
            for j in 0..<8 {
                let ref = cosineRef(qs[i], cs[j])
                let got = out[i * 8 + j]
                #expect(abs(got - ref) <= 2e-3, "cosine mismatch i=\(i) j=\(j) got=\(got) ref=\(ref)")
            }
        }
    }

    @Test("Self-cosine-distance is ~0")
    func cosineSelf() {
        let v = makeVecs512(1, seed: 0x42)
        let out = MatrixDistance.cosineDistanceMatrix(queries: v, candidates: v)
        #expect(out[0] >= 0)
        #expect(out[0] < 1e-3)
    }

    @Test("Works for 768 and 1536 dimensions")
    func otherDims() {
        let q7 = (0..<2).map { _ in try! Vector768Optimized((0..<768).map { Float(($0 % 7) - 3) }) }
        let c7 = (0..<3).map { k in try! Vector768Optimized((0..<768).map { Float((($0 + k) % 5) - 2) }) }
        let o7 = MatrixDistance.euclideanSquaredMatrix(queries: q7, candidates: c7)
        #expect(o7.count == 6)
        for i in 0..<2 {
            for j in 0..<3 {
                let ref = EuclideanKernels.squared768(q7[i], c7[j])
                #expect(abs(o7[i * 3 + j] - ref) <= 1e-1 + 1e-3 * abs(ref))
            }
        }
        // 1536 self-distance smoke (must clamp to ~0)
        let q15 = [try! Vector1536Optimized((0..<1536).map { Float(($0 % 11) - 5) })]
        let o15 = MatrixDistance.euclideanSquaredMatrix(queries: q15, candidates: q15)
        #expect(o15[0] >= 0 && o15[0] < 1e-1)
    }

    @Test("Empty inputs are a no-op")
    func emptyInputs() {
        let cs = makeVecs512(3, seed: 0x9)
        var out = [Float]()
        MatrixDistance.euclideanSquaredMatrix(queries: [], candidates: cs, into: &out)
        #expect(out.isEmpty)
    }

    @Test("DynamicVector works through the same generic path")
    func dynamicVectorPath() {
        let qs = [DynamicVector((0..<64).map { Float($0 % 5) })]
        let cs = (0..<3).map { k in DynamicVector((0..<64).map { Float(($0 + k) % 5) }) }
        let out = MatrixDistance.euclideanSquaredMatrix(queries: qs, candidates: cs)
        #expect(out.count == 3)
        for j in 0..<3 {
            // direct reference
            var ref = 0.0
            let a = qs[0].toArray(), b = cs[j].toArray()
            for k in 0..<64 { let d = Double(a[k] - b[k]); ref += d * d }
            #expect(abs(Double(out[j]) - ref) <= 1e-2 + 1e-3 * abs(ref))
        }
    }
}
