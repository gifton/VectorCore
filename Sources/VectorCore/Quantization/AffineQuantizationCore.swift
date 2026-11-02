import Foundation

/// Internal arithmetic for the bounded INT8 affine domain.
@usableFromInline
internal enum AffineQuantizationCore {
    @usableFromInline static func calibrate(minimum: Float, maximum: Float, symmetric: Bool) -> (Float, Int32) {
        guard minimum.isFinite && maximum.isFinite && minimum <= maximum else { fatalError("Quantization requires finite ordered bounds") }
        if minimum == 0 && maximum == 0 { return (1, 0) }
        let lo = min(Double(minimum), 0), hi = max(Double(maximum), 0)
        let ideal = symmetric ? max(abs(lo), abs(hi)) / 127 : (hi - lo) / 255
        let zero: Int32 = symmetric ? 0 : Int32(max(-128, min(127, (-128 - lo / ideal).rounded(.toNearestOrAwayFromZero))))
        let extent = max(abs(-128 - Int64(zero)), abs(127 - Int64(zero)))
        var cap = Float(Double(Float.greatestFiniteMagnitude) / Double(extent))
        if Double(cap) * Double(extent) > Double(Float.greatestFiniteMagnitude) { cap = cap.nextDown }
        return (min(max(Float(ideal), Float.leastNonzeroMagnitude), cap), zero)
    }

    @usableFromInline static func requireFinite(_ values: [Float]) {
        guard values.allSatisfy(\.isFinite) else { fatalError("Quantization requires finite input coordinates") }
    }

    @usableFromInline static func percentileIndex(count: Int, percentile: Float) -> Int {
        guard count > 0 else { fatalError("Quantization percentile requires a positive count") }
        guard percentile.isFinite && percentile > 0 && percentile <= 100 else { fatalError("Quantization percentile must be finite and in (0,100]") }
        if percentile == 100 { return count - 1 }
        let rank = ceil(Double(percentile) * Double(count) / 100) - 1
        if rank >= Double(count - 1) { return count - 1 }
        return Int(max(0, rank))
    }

    @usableFromInline static func product(_ code: Int8, _ params: LinearQuantizationParams) -> (Double, Double) {
        let a = Double(params.scale)
        let b = Double(Int64(code) - Int64(params.zeroPoint))
        let high = a * b
        return (high, (-high).addingProduct(a, b))
    }

    /// Knuth two-sum: operands and their sum are finite in this bounded domain.
    @usableFromInline static func twoSum(_ a: Double, _ b: Double) -> (Double, Double) {
        let sum = a + b
        let virtualB = sum - a
        return (sum, (a - (sum - virtualB)) + (b - virtualB))
    }

    /// Grow an error-free expansion of the four product terms, then round to Double.
    @usableFromInline static func difference(_ a: (Double, Double), _ b: (Double, Double)) -> Double {
        let terms = SIMD4<Double>(a.1, a.0, -b.1, -b.0)
        var expansion = SIMD4<Double>.zero
        var count = 0
        for termIndex in 0..<4 {
            var carry = terms[termIndex]
            var next = SIMD4<Double>.zero
            var nextCount = 0
            for index in 0..<count {
                let pair = twoSum(carry, expansion[index])
                if pair.1 != 0 {
                    next[nextCount] = pair.1
                    nextCount += 1
                }
                carry = pair.0
            }
            // Before this append, nextCount <= count <= termIndex <= 3.
            next[nextCount] = carry
            expansion = next
            count = nextCount + 1
        }
        var rounded: Double = 0
        for index in 0..<count { rounded += expansion[index] }
        return rounded
    }

    @usableFromInline static func difference(_ code: Int8, _ params: LinearQuantizationParams, _ value: Float) -> Double {
        let coordinate = product(code, params)
        if !value.isFinite { return coordinate.0 - Double(value) }
        return difference(coordinate, (Double(value), 0))
    }
}

/// Exact signed sums for built-in dimensions (centered products fit Int64; sums < 2^75).
@usableFromInline
internal struct AffineWideInteger {
    @usableFromInline var high: Int64 = 0
    @usableFromInline var low: UInt64 = 0
    @usableFromInline init() {}
    @usableFromInline mutating func add(_ value: Int64) {
        let result = low.addingReportingOverflow(UInt64(bitPattern: value))
        low = result.partialValue
        high += (value < 0 ? -1 : 0) + (result.overflow ? 1 : 0)
    }
    @usableFromInline var doubleValue: Double {
        let negative = high < 0
        var magnitudeHigh = UInt64(bitPattern: high)
        var magnitudeLow = low
        if negative {
            let result = (~magnitudeLow).addingReportingOverflow(1)
            magnitudeLow = result.partialValue
            magnitudeHigh = ~magnitudeHigh &+ (result.overflow ? 1 : 0)
        }
        let magnitude = Double(magnitudeHigh) * 18446744073709551616.0 + Double(magnitudeLow)
        return negative ? -magnitude : magnitude
    }
}
