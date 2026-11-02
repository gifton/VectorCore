// Internal normalization plan: validate and accumulate before any output write.
import Foundation

@usableFromInline
struct NormalizationPlan<S: BinaryFloatingPoint> {
    @usableFromInline let scale: S
    @usableFromInline let divisor: S
    @usableFromInline let widenedNorm: Double?

    @usableFromInline
    func coordinate(_ value: S) -> S {
        if let norm = widenedNorm { return S(Double(value) / norm) }
        return (value / scale) / divisor
    }

    @usableFromInline
    func isWithinTolerance(_ tolerance: Double) -> Bool {
        // This decision is used only by `Vector.normalized(tolerance:)` (and so `normalizedFast()`).
        guard let norm = widenedNorm else { return false }
        return abs(norm - 1) < tolerance
    }
}

@usableFromInline
enum NormalizationCore {
    /// Float/Float16 convert exactly to Double. Their squared components and the
    /// sum of at most Int.max such squares fit Double's exponent range.
    @usableFromInline
    static func plan<S: BinaryFloatingPoint>(count: Int, get: (Int) -> S) -> Result<NormalizationPlan<S>, VectorError> {
        guard count >= 0 else { fatalError("NormalizationCore: count must be nonnegative") }
        guard count > 0 else { return .failure(.invalidOperation("normalize", reason: "Cannot normalize empty vector")) }
        var scale: S = 0
        for i in 0..<count {
            let x = get(i)
            guard x.isFinite else { return .failure(.invalidOperation("normalize", reason: "Cannot normalize nonfinite component")) }
            scale = max(scale, abs(x))
        }
        guard scale > 0 else { return .failure(.invalidOperation("normalize", reason: "Cannot normalize zero vector")) }
        // The SDK marks Float16 unavailable on Intel macOS and Mac Catalyst x86_64; this is the same
        // guard as the Float16 overflow utilities in MixedPrecisionKernels. No S can be Float16 there.
        #if !(arch(x86_64) && (os(macOS) || targetEnvironment(macCatalyst)))
        let isNativeHalf = S.self == Float16.self
        #else
        let isNativeHalf = false
        #endif
        if S.self == Float.self || isNativeHalf {
            var sum: Double = 0
            var correction: Double = 0
            for i in 0..<count {
                let x = Double(get(i))
                let y = x * x - correction
                let next = sum + y
                correction = (next - sum) - y
                sum = next
            }
            return .success(NormalizationPlan(scale: scale, divisor: 1, widenedNorm: sum.squareRoot()))
        }
        var sum: S = 0
        var correction: S = 0
        for i in 0..<count {
            let x = get(i) / scale
            let y = x * x - correction
            let next = sum + y
            correction = (next - sum) - y
            sum = next
        }
        guard sum.isFinite && sum > 0 else {
            return .failure(.invalidOperation("normalize", reason: "Internal normalized calculation is unrepresentable"))
        }
        return .success(NormalizationPlan(scale: scale, divisor: sum.squareRoot(), widenedNorm: nil))
    }

    /// Empty, zero and nonfinite arrays are permissive bit-preserving no-ops.
    @usableFromInline
    static func permissive(_ input: [Float]) -> [Float] {
        guard case .success(let plan) = plan(count: input.count, get: { input[$0] }) else { return input }
        return input.map { plan.coordinate($0) }
    }

    @usableFromInline
    static func floatSquares(count: Int, get: (Int) -> Float) -> Double {
        var sum: Double = 0
        var correction: Double = 0
        var infinite = false
        for i in 0..<count {
            let x = Double(get(i))
            if x.isNaN { return .nan }
            if x.isInfinite { infinite = true; continue }
            let y = x * x - correction
            let next = sum + y
            correction = (next - sum) - y
            sum = next
        }
        return infinite ? .infinity : sum
    }
}
