import Foundation
import VectorCore
import VectorCoreBenchmarking

// CorrectnessStats moved to VectorCoreBenchmarking library

enum Correctness {
    /// The relative-error floor. The denominator of every relative error is `max(|ref|, referenceFloor)`, so a
    /// reference smaller than the floor (zero included) is never a denominator: below it the reported relative error
    /// is the absolute error divided by the floor, a mixed rel/abs criterion whose absolute tolerance is
    /// `threshold * referenceFloor`. A relative error against a zero reference is undefined, which is why a floor is
    /// needed at all: against the pre-T11 floor of 1e-12, a matrix diagonal error of 3e-5 read as a relative error of 3e7.
    ///
    /// The value 1 is a stated convention: it is the square of the coordinate bound of the benchmark fixtures
    /// (`InputFactory.randomArray` draws from -1...1; `testFixtureCoordinatesLieWithinTheStatedBound`), so the metric
    /// turns absolute only below the magnitude of one fixture term. The floor cannot make a derived threshold unsound: a relative bound `|c - r| <= a * |r|` gives
    /// `|c - r| / max(|r|, f) <= a` for every floor `f`; and the route whose error does not scale with `|r|` (the
    /// matrix norm identity, where a zero diagonal reference still carries rounding error of the size of the summed
    /// norms) is gated on absolute error instead (`CorrectnessPolicy`). Pinned by
    /// `BenchmarkCorrectnessPolicyTests.testRelativeErrorNeverDividesByReferenceBelowFloor` (with a negative control on
    /// the pre-T11 metric) and `testOnlyCorrectnessComputesRelativeErrors`.
    static let referenceFloor: Double = 1

    /// `absError / max(|ref|, referenceFloor)`: the only relative-error computation in the benchmark sources
    /// (`testOnlyCorrectnessComputesRelativeErrors`).
    @inline(__always)
    static func relErr(absError: Double, ref: Double) -> Double {
        absError / max(abs(ref), referenceFloor)
    }

    static func scalar(current: Float, reference: Double) -> CorrectnessStats {
        guard current.isFinite && reference.isFinite else { return CorrectnessStats(invalidReason: "Nonfinite scalar observation") }
        let curr = Double(current)
        let absErr = abs(curr - reference)
        let rel = relErr(absError: absErr, ref: reference)
        return CorrectnessStats(samples: 1, maxAbsError: absErr, meanAbsError: absErr, maxRelError: rel, meanRelError: rel)
    }

    static func vector(currents: [Float], references: [Double]) -> CorrectnessStats {
        guard currents.count == references.count else { return CorrectnessStats(invalidReason: "Vector length mismatch: \(currents.count) versus \(references.count)") }
        let n = currents.count
        guard n > 0 else { return CorrectnessStats(invalidReason: "Empty comparison") }
        guard currents.allSatisfy({ $0.isFinite }) && references.allSatisfy({ $0.isFinite }) else { return CorrectnessStats(invalidReason: "Nonfinite vector observation") }
        var maxAbs = 0.0
        var sumAbs = 0.0
        var maxRel = 0.0
        var sumRel = 0.0
        for i in 0..<n {
            let c = Double(currents[i])
            let r = references[i]
            let ae = abs(c - r)
            let re = relErr(absError: ae, ref: r)
            if ae > maxAbs { maxAbs = ae }
            if re > maxRel { maxRel = re }
            sumAbs += ae
            sumRel += re
        }
        return CorrectnessStats(samples: n, maxAbsError: maxAbs, meanAbsError: sumAbs / Double(n), maxRelError: maxRel, meanRelError: sumRel / Double(n))
    }
}

// MARK: - Double-precision reference implementations

enum DoubleRef {
    static func dot(_ a: UnsafeBufferPointer<Float>, _ b: UnsafeBufferPointer<Float>) -> Double {
        let n = min(a.count, b.count)
        var s = 0.0
        for i in 0..<n { s += Double(a[i]) * Double(b[i]) }
        return s
    }

    static func euclid2(_ a: UnsafeBufferPointer<Float>, _ b: UnsafeBufferPointer<Float>) -> Double {
        let n = min(a.count, b.count)
        var s = 0.0
        for i in 0..<n { let d = Double(a[i]) - Double(b[i]); s += d * d }
        return s
    }

    static func euclid(_ a: UnsafeBufferPointer<Float>, _ b: UnsafeBufferPointer<Float>) -> Double {
        sqrt(euclid2(a, b))
    }

    static func manhattan(_ a: UnsafeBufferPointer<Float>, _ b: UnsafeBufferPointer<Float>) -> Double {
        let n = min(a.count, b.count)
        var s = 0.0
        for i in 0..<n { s += abs(Double(a[i]) - Double(b[i])) }
        return s
    }

    static func chebyshev(_ a: UnsafeBufferPointer<Float>, _ b: UnsafeBufferPointer<Float>) -> Double {
        let n = min(a.count, b.count)
        var m = 0.0
        for i in 0..<n { let d = abs(Double(a[i]) - Double(b[i])); if d > m { m = d } }
        return m
    }

    static func hamming(_ a: UnsafeBufferPointer<Float>, _ b: UnsafeBufferPointer<Float>, threshold: Double = 0.5) -> Double {
        let n = min(a.count, b.count)
        var count = 0.0
        for i in 0..<n {
            let abit = Double(a[i]) > threshold
            let bbit = Double(b[i]) > threshold
            if abit != bbit { count += 1.0 }
        }
        return count
    }

    static func minkowski(_ a: UnsafeBufferPointer<Float>, _ b: UnsafeBufferPointer<Float>, p: Double) -> Double {
        if p == 1 { return manhattan(a, b) }
        if p == 2 { return euclid(a, b) }
        if p.isInfinite { return chebyshev(a, b) }
        let n = min(a.count, b.count)
        var s = 0.0
        for i in 0..<n { s += pow(abs(Double(a[i]) - Double(b[i])), p) }
        return pow(s, 1.0 / p)
    }

    static func cosineDist(_ a: UnsafeBufferPointer<Float>, _ b: UnsafeBufferPointer<Float>) -> Double {
        let n = min(a.count, b.count)
        var dot = 0.0, sa = 0.0, sb = 0.0
        for i in 0..<n { let x = Double(a[i]); let y = Double(b[i]); dot += x*y; sa += x*x; sb += y*y }
        // The library's cosine policy (T15): a NaN or infinite coordinate on either side gives NaN, even against a
        // zero vector, and the clamp applies only to a finite similarity (T15DoubleRefCosineTests).
        guard sa.isFinite && sb.isFinite else { return .nan }
        let mag = sqrt(sa * sb)
        if mag <= Double.ulpOfOne { return 1.0 }
        let similarity = dot / mag
        guard similarity.isFinite else { return .nan }
        let cosSim = max(-1.0, min(1.0, similarity))
        return 1.0 - cosSim
    }
}

