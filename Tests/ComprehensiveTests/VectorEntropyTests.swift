import Foundation
import Testing
@testable import VectorCore

@Suite("Vector Entropy")
struct VectorEntropySuite {
    // Helper to compute manual entropy for small vectors
    private func manualEntropy(_ values: [Float]) -> Float {
        let absVals = values.map { Swift.abs($0) }
        let sum = absVals.reduce(0, +)
        guard sum > Float.ulpOfOne else { return 0 }
        var h: Float = 0
        for a in absVals {
            let p = a / sum
            if p > Float.ulpOfOne { h -= p * Foundation.log(p) }
        }
        return h
    }

    // Baseline correctness
    @Test
    func testEntropy_ZeroVector_ReturnsZero_Small() {
        let v = Vector<Dim8>.zero
        #expect(approxEqual(v.entropy, 0, tol: 1e-7))
        #expect(approxEqual(v.entropyFast, 0, tol: 1e-7))
    }

    @Test
    func testEntropy_SingleSpike_ReturnsZero_Small() {
        var vals = Array(repeating: Float(0), count: 8)
        vals[3] = 5
        let v = try! Vector<Dim8>(vals)
        #expect(approxEqual(v.entropy, 0, tol: 1e-7))
    }

    @Test
    func testEntropy_UniformOnes_ReturnsLogN_Small() {
        let v = Vector<Dim16>.ones
        let expected = Foundation.log(Float(Dim16.value))
        #expect(approxEqual(v.entropy, expected, tol: 1e-6))
    }

    @Test
    func testEntropy_UniformMixedSigns_ReturnsLogN_Small() {
        var vals = (0..<16).map { _ in Float(1) }
        // Flip half to negative
        for i in 0..<8 { vals[i] = -1 }
        let v = try! Vector<Dim16>(vals)
        let expected = Foundation.log(Float(Dim16.value))
        #expect(approxEqual(v.entropy, expected, tol: 1e-6))
    }

    @Test
    func testEntropy_TwoCategoryEqual_ReturnsLog2_Small() {
        let v = try! Vector<Dim8>([2, -2, 0, 0, 0, 0, 0, 0])
        let expected: Float = Foundation.log(2)
        #expect(approxEqual(v.entropy, expected, tol: 1e-6))
    }

    @Test
    func testEntropy_KnownDistribution_MatchesManual_Small() {
        // Probabilities: [0.5, 0.25, 0.125, 0.125]
        let vals: [Float] = [8, 4, 2, 2, 0, 0, 0, 0]
        let v = try! Vector<Dim8>(vals)
        let expected = manualEntropy(vals)
        #expect(approxEqual(v.entropy, expected, tol: 1e-6))
    }

    // Invariance properties
    @Test
    func testEntropy_ScaleInvariance_Positive() {
        let vals: [Float] = [0.1, 0.3, 0.6, 0]
        let v = try! Vector<Dim4>(vals)
        let k: Float = 7.5
        let w = v * k
        #expect(approxEqual(v.entropy, w.entropy, tol: 1e-6))
    }

    @Test
    func testEntropy_SignInvariance() {
        let vals: [Float] = [1, -2, 3, -4]
        let v = try! Vector<Dim4>(vals)
        let u = -v
        #expect(approxEqual(v.entropy, u.entropy, tol: 1e-6))
    }

    @Test
    func testEntropy_PermutationInvariance() {
        let vals: [Float] = [1, 2, 3, 4, 0, 0, 5, 0]
        let v = try! Vector<Dim8>(vals)
        var shuffled = vals
        shuffled.shuffle()
        let u = try! Vector<Dim8>(shuffled)
        #expect(approxEqual(v.entropy, u.entropy, tol: 1e-6))
    }

    // Path parity (small vs large)
    @Test
    func testEntropy_ZeroVector_Large() {
        let v = Vector<Dim128>.zero
        #expect(approxEqual(v.entropy, 0, tol: 1e-7))
    }

    @Test
    func testEntropy_UniformOnes_Large() {
        let v = Vector<Dim256>.ones
        let expected = Foundation.log(Float(Dim256.value))
        #expect(approxEqual(v.entropy, expected, tol: 1e-5))
    }

    @Test
    func testEntropy_DistributionParity_SmallVsLarge() {
        // Same non-zero values, zeros padded in large vector should not affect entropy
        let base: [Float] = [3, 1, 0, 0, 2, 0, 0, 0]
        let vSmall = try! Vector<Dim8>(base)
        var large = Array(repeating: Float(0), count: 128)
        // place same non-zero values in the front
        for i in 0..<base.count { large[i] = base[i] }
        let vLarge = try! Vector<Dim128>(large)
        #expect(approxEqual(vSmall.entropy, vLarge.entropy, tol: 1e-6))
    }

    // Bounds and monotonicity
    @Test
    func testEntropy_Bounds_RandomVectorsWithinRange() {
        for _ in 0..<5 {
            let v = Vector<Dim32>.random(in: -1...1)
            let h = v.entropy
            #expect(h >= 0 - 1e-6)
            #expect(h <= Foundation.log(Float(Dim32.value)) + 1e-6)
        }
    }

    @Test
    func testEntropy_ConcentrationDecreasesFromUniform() {
        // Uniform over 4 entries
        let u = Vector<Dim4>.ones
        // Concentrate mass while keeping sum constant (4)
        let c = try! Vector<Dim4>([2, 1, 1, 0])
        #expect(u.entropy > c.entropy)
    }

    // Near-zero and numerical handling
    @Test
    func testEntropy_UlpThreshold_IgnoresNearZero() {
        let tiny = Float.ulpOfOne / 4
        var vals = Array(repeating: Float(0), count: 8)
        vals[0] = 1
        vals[1] = tiny
        let v = try! Vector<Dim8>(vals)
        #expect(approxEqual(v.entropy, 0, tol: 1e-6))
    }

    @Test
    func testEntropy_TinyNonZero_ProducesSmallPositive() {
        // Make the second value small but above ulp threshold relative to sum
        let vals: [Float] = [1, 1e-4, 0, 0]
        let v = try! Vector<Dim4>(vals)
        let h = v.entropy
        #expect(h > 0)
        #expect(h < Foundation.log(2) + 1e-3) // certainly less than 2-category uniform
    }

    @Test
    func testEntropy_ZerosDoNotAffectBeyondScale() {
        let a = try! Vector<Dim8>([1, 2, 3, 0, 0, 0, 0, 0])
        let b = try! Vector<Dim8>([1, 2, 3, 0, 0, 0, 0, 0])
        #expect(approxEqual(a.entropy, b.entropy, tol: 1e-7))
        // Compare to a padded version in larger dimension
        var big = Array(repeating: Float(0), count: 128)
        big[0] = 1; big[1] = 2; big[2] = 3
        let c = try! Vector<Dim128>(big)
        #expect(approxEqual(a.entropy, c.entropy, tol: 1e-6))
    }

    // Non-finite inputs
    @Test
    func testEntropy_NaNInput_ReturnsNaN_Small() {
        var vals = Array(repeating: Float(0), count: 8)
        vals[0] = .nan
        let v = try! Vector<Dim8>(vals)
        #expect(v.entropy.isNaN)
    }

    @Test
    func testEntropy_InfiniteInput_ReturnsNaN_Small() {
        var vals = Array(repeating: Float(0), count: 8)
        vals[3] = .infinity
        let v = try! Vector<Dim8>(vals)
        #expect(v.entropy.isNaN)
    }

    @Test
    func testEntropy_NonFinite_LargePath_ReturnsNaN() {
        var vals = Array(repeating: Float(1), count: 256)
        vals[100] = .nan
        let v = try! Vector<Dim256>(vals)
        #expect(v.entropy.isNaN)
    }

    // Stability across scales
    @Test
    func testEntropy_ScaleStability_LargeMagnitudesVsSmall_Large() {
        // Same distribution different scales
        var a = Array(repeating: Float(0), count: 128)
        a[0] = 1e6; a[1] = 2e6; a[2] = 3e6; a[3] = 4e6
        var b = Array(repeating: Float(0), count: 128)
        b[0] = 1e-3; b[1] = 2e-3; b[2] = 3e-3; b[3] = 4e-3
        let vA = try! Vector<Dim128>(a)
        let vB = try! Vector<Dim128>(b)
        #expect(approxEqual(vA.entropy, vB.entropy, tol: 1e-6))
    }
}
