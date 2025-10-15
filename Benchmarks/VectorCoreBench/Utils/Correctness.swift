import Foundation
import VectorCore
import VectorCoreBenchmarking

// CorrectnessStats moved to VectorCoreBenchmarking library

enum Correctness {
    @inline(__always)
    static func relErr(absError: Double, ref: Double, epsilon: Double = 1e-12) -> Double {
        absError / max(abs(ref), epsilon)
    }

    static func scalar(current: Float, reference: Double) -> CorrectnessStats {
        let curr = Double(current)
        let absErr = abs(curr - reference)
        let rel = relErr(absError: absErr, ref: reference)
        return CorrectnessStats(samples: 1, maxAbsError: absErr, meanAbsError: absErr, maxRelError: rel, meanRelError: rel)
    }

    static func vector(currents: [Float], references: [Double]) -> CorrectnessStats {
        let n = min(currents.count, references.count)
        guard n > 0 else { return CorrectnessStats(samples: 0, maxAbsError: 0, meanAbsError: 0, maxRelError: 0, meanRelError: 0) }
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
        let mag = sqrt(sa * sb)
        if mag <= Double.ulpOfOne { return 1.0 }
        let cosSim = max(-1.0, min(1.0, dot / mag))
        return 1.0 - cosSim
    }
}

