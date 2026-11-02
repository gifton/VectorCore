import Foundation
import Testing
@testable import VectorCore

// N04: scale-invariant truncated entropy contract.
//
// Contract under test (see Sources/VectorCore/Operations/VectorEntropy.swift):
//   p_i = |x_i| / sum_j |x_j|;  H = Float( sum_{p_i > Float.ulpOfOne} -p_i ln p_i ).
// The cutoff applies to the normalized probability only, never to the input sum.
// Any NaN or infinity yields NaN; all-zero input yields +0.

private enum EDim1: StaticDimension {
    static let value = 1
}
private enum EDim63: StaticDimension {
    static let value = 63
}
private enum EDim65: StaticDimension {
    static let value = 65
}

/// Independent reference. Deliberately structured differently from production:
/// Double max-scaled magnitudes, plain (uncompensated) scalar sum of the scaled
/// magnitudes, then normalization of the scaled values.
func entropyContractReference(_ values: [Float]) -> Float {
    var maxMag = 0.0
    for v in values {
        if !v.isFinite { return .nan }
        maxMag = Swift.max(maxMag, Swift.abs(Double(v)))
    }
    if maxMag == 0 { return 0 }
    let scaled: [Double] = values.map { Swift.abs(Double($0)) / maxMag }
    var total = 0.0
    for s in scaled { total += s }
    var h = 0.0
    for s in scaled {
        let p = s / total
        if p > Double(Float.ulpOfOne) { h -= p * Foundation.log(p) }
    }
    return Float(h)
}

private let routeNames = ["entropy", "entropyFast", "entropySmall", "entropyLarge"]

private func routes<D: StaticDimension>(_: D.Type, _ values: [Float]) -> [Float] {
    let v = try! Vector<D>(values)
    return [v.entropy, v.entropyFast, v.entropySmall(), v.entropyLarge()]
}

private let testedDims = [1, 4, 8, 63, 64, 65, 128, 1536]

/// All four routes (public, entropyFast, both direct helpers) for a given width.
private func allRoutes(dim: Int, _ values: [Float]) -> [Float] {
    precondition(values.count == dim)
    switch dim {
    case 1: return routes(EDim1.self, values)
    case 4: return routes(Dim4.self, values)
    case 8: return routes(Dim8.self, values)
    case 63: return routes(EDim63.self, values)
    case 64: return routes(Dim64.self, values)
    case 65: return routes(EDim65.self, values)
    case 128: return routes(Dim128.self, values)
    case 1536: return routes(Dim1536.self, values)
    default: preconditionFailure("untested dimension \(dim)")
    }
}

private func close(_ actual: Float, _ reference: Float) -> Bool {
    guard actual.isFinite, reference.isFinite else { return false }
    return Swift.abs(actual - reference) <= 8 * Float.ulpOfOne * Swift.max(1, Swift.abs(reference))
}

private func gcd(_ a: Int, _ b: Int) -> Int { b == 0 ? a : gcd(b, a % b) }

private func isPositiveZero(_ x: Float) -> Bool { x.bitPattern == 0 }

private func expectAllClose(_ values: [Float], _ reference: Float, _ label: @autoclosure () -> String,
                            sourceLocation: SourceLocation = #_sourceLocation) {
    let out = allRoutes(dim: values.count, values)
    for (name, got) in zip(routeNames, out) {
        #expect(close(got, reference), "\(label()) \(name)=\(got) ref=\(reference)", sourceLocation: sourceLocation)
    }
}

private func expectAllPositiveZero(_ values: [Float], _ label: @autoclosure () -> String,
                                   sourceLocation: SourceLocation = #_sourceLocation) {
    let out = allRoutes(dim: values.count, values)
    for (name, got) in zip(routeNames, out) {
        #expect(isPositiveZero(got), "\(label()) \(name)=\(got)", sourceLocation: sourceLocation)
    }
}

private func expectAllNaN(_ values: [Float], _ label: @autoclosure () -> String,
                          sourceLocation: SourceLocation = #_sourceLocation) {
    let out = allRoutes(dim: values.count, values)
    for (name, got) in zip(routeNames, out) {
        #expect(got.isNaN, "\(label()) \(name)=\(got)", sourceLocation: sourceLocation)
    }
}

/// Every positive Float magnitude scale exercised by the uniform tests.
private func uniformScales() -> [Float] {
    var scales: [Float] = [
        Float.leastNonzeroMagnitude,
        Float.leastNormalMagnitude.nextDown,   // largest subnormal
        Float.leastNormalMagnitude,
        1e-10, 1, 1e10,
        Float.greatestFiniteMagnitude
    ]
    for k in -149...127 { scales.append(Float(sign: .plus, exponent: k, significand: 1)) }
    return scales
}

private func alternating(_ magnitude: Float, count: Int) -> [Float] {
    (0..<count).map { $0 % 2 == 0 ? magnitude : -magnitude }
}

@Suite("Entropy range contract")
struct EntropyRangeContractTests {
    @Test func uniformMagnitudesReturnLogD() {
        for dim in testedDims {
            let expected = Float(Foundation.log(Double(dim)))
            for s in uniformScales() {
                expectAllClose(alternating(s, count: dim), expected, "uniform D=\(dim) s=\(s)")
            }
        }
    }

    @Test func singleSpikesAreExactPositiveZero() {
        for dim in testedDims {
            let positions = Set([0, dim / 2, dim - 1])
            for s in uniformScales() {
                for pos in positions {
                    for sign: Float in [1, -1] {
                        var vals = [Float](repeating: 0, count: dim)
                        vals[pos] = sign * s
                        expectAllPositiveZero(vals, "spike D=\(dim) s=\(s) pos=\(pos)")
                    }
                }
            }
        }
    }

    @Test func rationalClosedForms() {
        let ln2 = Foundation.log(2.0)
        // [8,4,2,2] -> p = [1/2,1/4,1/8,1/8]; H = 7/4 ln 2.
        let h8422 = Float(1.75 * ln2)
        // [1,1,2] -> p = [1/4,1/4,1/2]; H = 3/2 ln 2.
        let h112 = Float(1.5 * ln2)
        // [3,1] -> p = [3/4,1/4].
        let h31 = Float(-(0.75 * Foundation.log(0.75) + 0.25 * Foundation.log(0.25)))
        let cases: [([Float], Float)] = [([8, 4, 2, 2], h8422), ([1, 1, 2], h112), ([3, 1], h31)]
        for (pattern, expected) in cases {
            #expect(close(entropyContractReference(pattern), expected))
            for dim in testedDims where dim >= pattern.count {
                for k in [-148, -100, -10, 0, 10, 100, 124] {
                    // Powers-of-two scaling is exact; the premise is asserted below. k = -148 yields subnormal
                    // components (e.g. 2 * 2^-148 = 2^-147), which are still exactly representable.
                    let scale = Float(sign: .plus, exponent: k, significand: 1)
                    var vals = [Float](repeating: 0, count: dim)
                    for (i, p) in pattern.enumerated() { vals[i] = p * scale }
                    for (i, p) in pattern.enumerated() {
                        #expect(vals[i].isFinite && vals[i] > 0 && vals[i] / scale == p, "inexact premise k=\(k) \(pattern)")
                    }
                    expectAllClose(vals, expected, "rational D=\(dim) k=\(k) \(pattern)")
                }
            }
        }
    }

    @Test func unequalFinitePatternsMatchReferenceUnderPermutationAndSign() {
        let base: [Float] = [1, 2, 3, 4, 0.5, 5, 0.125, 7]
        let scales: [Float] = [Float.leastNonzeroMagnitude * 64, 1e-10, 1, 1e10, 1e30]
        for dim in testedDims where dim >= base.count {
            for s in scales {
                var padded = [Float](repeating: 0, count: dim)
                for (i, b) in base.enumerated() { padded[i] = b * s }
                let ref = entropyContractReference(padded)
                expectAllClose(padded, ref, "pattern D=\(dim) s=\(s)")
                expectAllClose(padded.reversed(), ref, "reversed D=\(dim) s=\(s)")
                var perm = padded
                // Deterministic cyclic rotation.
                let shift = dim / 3 + 1
                perm = Array(perm[shift...] + perm[..<shift])
                expectAllClose(perm, ref, "rotated D=\(dim) s=\(s)")
                // Non-cyclic permutation: index i -> (i * 5) mod dim, a bijection iff gcd(5, dim) == 1.
                // The stride is chosen per dimension as the first of 5, 3, 2, 7 coprime to dim.
                let stride = [5, 3, 2, 7].first { gcd($0, dim) == 1 }!
                var strided = [Float](repeating: 0, count: dim)
                var seen = Set<Int>()
                for i in 0..<dim { let j = (i * stride) % dim; seen.insert(j); strided[j] = padded[i] }
                #expect(seen.count == dim, "stride permutation premise D=\(dim)")
                expectAllClose(strided, ref, "strided D=\(dim) stride=\(stride) s=\(s)")
                let flipped = padded.enumerated().map { $0.offset % 2 == 0 ? -$0.element : $0.element }
                expectAllClose(flipped, ref, "signflipped D=\(dim) s=\(s)")
            }
        }
    }

    @Test func commonPowerOfTwoScaleInvariance() {
        let base: [Float] = [3, 1, 0, 5, 2, 0, 0, 9]
        let ref = entropyContractReference(base)
        for k in stride(from: -140, through: 115, by: 5) {
            let scale = Float(sign: .plus, exponent: k, significand: 1)
            let scaled = base.map { $0 * scale }
            // Premise: scaling is exact and loses/overflows no component.
            for (a, b) in zip(base, scaled) {
                #expect(a == 0 ? b == 0 : (b.isFinite && b / scale == a))
            }
            expectAllClose(scaled, ref, "scale k=\(k)")
        }
    }

    @Test func normalizedCutoffTable() {
        // Exact premise: components scaled by 2^47 are integers summing to 2^47.
        let t47 = 1 << 47, t24 = 1 << 24
        let belowInts = [t24 - 1, t47 - t24, 1]
        let equalInts = [t24, t47 - t24, 0]
        let aboveInts = [t24 + 2, t47 - 2 * t24, t24 - 2]
        for ints in [belowInts, equalInts, aboveInts] { #expect(ints.reduce(0, +) == t47) }
        let u: Float = 1.0 / Float(t24)          // 2^-24 exactly
        let p23: Float = 2 * u                   // 2^-23
        let p47: Float = Float(sign: .plus, exponent: -47, significand: 1)
        let p46: Float = 2 * p47
        let below: [Float] = [p23 - p47, 1 - p23, p47]
        let equal: [Float] = [p23, 1 - p23, 0]
        let above: [Float] = [p23 + p46, 1 - 2 * p23, p23 - p46]
        // Float vectors equal the integer premise exactly (Double conversion is exact).
        for (f, ints) in [(below, belowInts), (equal, equalInts), (above, aboveInts)] {
            for (x, i) in zip(f, ints) { #expect(Double(x) * Double(t47) == Double(i)) }
        }
        let cutoff = Double(Float.ulpOfOne)
        func term(_ p: Double) -> Double { -p * Foundation.log(p) }
        let dominantBelow = 1.0 - cutoff           // exact in Double
        let tDomBelow = -dominantBelow * Foundation.log1p(-cutoff)
        let pAbove = cutoff + Double(p46)
        let dAbove = 1.0 - 2 * cutoff
        let tAbove = term(pAbove) + (-dAbove * Foundation.log1p(-2 * cutoff))
        let omittedTerm = term(cutoff)             // what 'equal' would add if the cutoff were inclusive
        let cases: [(String, [Float], Double)] = [
            ("below", below, tDomBelow),
            ("equal", equal, tDomBelow),
            ("above", above, tAbove)
        ]
        for dim in [4, 8, 64, 128] {
            for (name, vec, expected) in cases {
                var vals = [Float](repeating: 0, count: dim)
                for (i, x) in vec.enumerated() { vals[i] = x }
                let out = allRoutes(dim: dim, vals)
                for (route, got) in zip(routeNames, out) {
                    #expect(got > 0, "\(name) D=\(dim) \(route) dropped retained dominant term")
                    #expect(close(got, Float(expected)), "\(name) D=\(dim) \(route)=\(got) exp=\(expected)")
                }
                if name == "below" || name == "equal" {
                    // Policy assertion: the omitted term is distinguishable from the tolerance.
                    #expect(omittedTerm > 8 * Double(Float.ulpOfOne))
                    for got in out { #expect(Swift.abs(Double(got) - (expected + omittedTerm)) > 8 * cutoff) }
                }
            }
        }
    }

    @Test func tinyCoordinateStaysInDenominator() {
        // [1, 2^-25]: tiny p is omitted, dominant p = 1/(1+2^-25) is retained and nonzero.
        let tiny = Float.ulpOfOne / 4
        let q = 1.0 / (1.0 + Double(tiny))
        let expected = Float(-q * Foundation.log(q))
        #expect(expected > 0)
        for dim in [8, 128] {
            var vals = [Float](repeating: 0, count: dim)
            vals[0] = 1; vals[1] = tiny
            let out = allRoutes(dim: dim, vals)
            for (route, got) in zip(routeNames, out) {
                #expect(got > 0, "\(route)")
                #expect(close(got, expected), "\(route)=\(got) exp=\(expected)")
            }
        }
    }

    @Test func nonFinitePropagatesAtAnyPosition() {
        let bad: [Float] = [.nan, .infinity, -.infinity]
        let fillers: [Float] = [0, 1, -3, Float.greatestFiniteMagnitude, Float.leastNonzeroMagnitude]
        for dim in testedDims {
            for b in bad {
                for pos in Set([0, dim / 2, dim - 1]) {
                    for f in fillers {
                        var vals = [Float](repeating: f, count: dim)
                        vals[pos] = b
                        expectAllNaN(vals, "D=\(dim) bad=\(b) pos=\(pos) filler=\(f)")
                    }
                }
                if dim >= 3 {
                    var mixed = [Float](repeating: 2, count: dim)
                    mixed[0] = .nan; mixed[dim - 1] = b
                    expectAllNaN(mixed, "D=\(dim) nan+\(b)")
                }
            }
        }
    }

    @Test func signedZerosReturnPositiveZero() {
        for dim in testedDims {
            expectAllPositiveZero([Float](repeating: 0, count: dim), "+0 D=\(dim)")
            expectAllPositiveZero([Float](repeating: -0.0, count: dim), "-0 D=\(dim)")
            expectAllPositiveZero((0..<dim).map { $0 % 2 == 0 ? 0 : -0.0 }, "alt0 D=\(dim)")
            expectAllPositiveZero((0..<dim).map { $0 % 3 == 0 ? -0.0 : 0 }, "mix0 D=\(dim)")
        }
    }
}
