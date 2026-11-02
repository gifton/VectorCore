import Foundation

/// Deterministic anti-elision consumer that reads every produced score.
///
/// $h_0 = \mathrm{0xcbf29ce484222325}$, $h_{i+1} = (h_i \oplus w_i)\cdot p \bmod 2^{64}$ with
/// $p = \mathrm{0x100000001b3}$ and $w_i$ the zero-extended 32-bit pattern of element $i$.
///
/// Single-position sensitivity (derivable here in one step): $p$ is odd, hence invertible mod $2^{64}$, so
/// $h \mapsto (h \oplus w)\cdot p$ is a bijection of UInt64 for every fixed $w$, and for a fixed $h$,
/// $w \mapsto (h \oplus w)\cdot p$ is injective on 32-bit $w$. Changing the bit pattern of exactly one element
/// therefore changes that step's output, and every later step is a bijection of the running value, so the
/// final value changes. Tests: `BenchmarkOutputFoldContractTests` (`testMatrixConsumerSeesEveryPositionAtAllWidthsSmallN`,
/// `testFoldSensitivityToSignedZeroNaNPayloadAndOrder`).
///
/// This is an anti-elision observation. It is not a numerical oracle, not a collision-free fingerprint for
/// changes to more than one element, and it distinguishes -0 from +0 and NaN payloads
/// (`testFoldSensitivityToSignedZeroNaNPayloadAndOrder`). The fold uses integer arithmetic only, so the consumer has no
/// floating-point cancellation.
///
/// - Complexity: O(n) time, O(1) space; one sequential pass.
///
/// `@inline(never)` asks for one out-of-line symbol (whether an optimized binary keeps it is not verified in this
/// repository), and the timed body and the consumer-only record call this same function.
enum OutputFold {
    static let offsetBasis: UInt64 = 0xcbf29ce484222325
    static let prime: UInt64 = 0x100000001b3

    @inline(never)
    static func fold(_ values: UnsafeBufferPointer<Float>) -> UInt64 {
        var h = offsetBasis
        for value in values {
            h = (h ^ UInt64(value.bitPattern)) &* prime
        }
        return h
    }
}
