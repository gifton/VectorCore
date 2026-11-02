//
//  SIMDSumSemanticsTests.swift
//  VectorCore
//
//  T14: pins VectorCore's `sum()` on `SIMD where Scalar == Float` and `Scalar == Double`
//  (Sources/VectorCore/Platform/SIMDExtensions.swift), which shadows the standard library's `SIMD.sum()` inside the
//  module: lanes added in index order, starting from +0. With `@testable import`, `v.sum()` on a concrete Float or
//  Double SIMD vector binds the module function. NaN payloads are not pinned.
//

import Testing
@testable import VectorCore

/// The standard library member: in a generic context the module's `Scalar == Float`/`Scalar == Double` extensions do
/// not apply, so this binds `SIMD.sum()` from the standard library.
@inline(never)
func t14StdlibSum<V: SIMD>(_ vector: V) -> V.Scalar where V.Scalar: FloatingPoint {
    vector.sum()
}

@Suite("SIMD sum semantics (T14)")
struct SIMDSumSemanticsTests {
    /// All lanes -0 sum to +0: the first addition is (+0) + (-0) = +0. The standard library member, which starts from
    /// -0, returns -0 here.
    @Test func allNegativeZeroLanesSumToPositiveZero() {
        #expect(SIMD2<Float>(repeating: -0.0).sum().bitPattern == 0)
        #expect(SIMD4<Float>(repeating: -0.0).sum().bitPattern == 0)
        #expect(SIMD8<Float>(repeating: -0.0).sum().bitPattern == 0)
        #expect(SIMD16<Float>(repeating: -0.0).sum().bitPattern == 0)
        #expect(SIMD4<Double>(repeating: -0.0).sum().bitPattern == 0)
        #expect(SIMD8<Double>(repeating: -0.0).sum().bitPattern == 0)
    }

    /// Order-sensitive fixtures: index order from lane 0 and a pairwise tree give different results.
    @Test func lanesAreAddedInIndexOrder() {
        // ((1e8 + 1) - 1e8) + 1 = 1, because 1e8 + 1 rounds to 1e8 (the Float spacing at 1e8 is 8);
        // the tree (1e8 + 1) + (-1e8 + 1) gives 0.
        #expect(SIMD4<Float>(1e8, 1, -1e8, 1).sum().bitPattern == Float(1).bitPattern)
        #expect(SIMD8<Float>(1e8, 1, -1e8, 1, 1e8, 1, -1e8, 1).sum().bitPattern == Float(1).bitPattern)
        // 2^24 + 1 rounds back to 2^24 (a tie, to even), three times: 0x4B800000. The tree gives 2^24 + 2 (0x4B800001)
        // and the reverse order 1 + 1 + 1 + 2^24 rounds to 2^24 + 4 (0x4B800002).
        #expect(SIMD4<Float>(16_777_216, 1, 1, 1).sum().bitPattern == 0x4B80_0000)
        var sixteen = SIMD16<Float>(repeating: 1)
        sixteen[0] = 16_777_216
        #expect(sixteen.sum().bitPattern == 0x4B80_0000)
        // Double: the spacing at 1e17 is 16, so 1e17 + 1 rounds to 1e17.
        #expect(SIMD4<Double>(1e17, 1, -1e17, 1).sum().bitPattern == Double(1).bitPattern)
        #expect(SIMD8<Double>(1e17, 1, -1e17, 1, 1e17, 1, -1e17, 1).sum().bitPattern == Double(1).bitPattern)
    }

    /// One in-library route: the SoA dot product accumulates with fused multiply-adds, and when every product
    /// underflows to -0 the accumulators hold -0 and the result is `sum()` of an all-(-0) vector: +0.
    @Test func soaDotOfUnderflowingProductsIsPositiveZero() {
        let query = Vector512Optimized(generator: { $0 % 2 == 0 ? 1e-10 : 1e-36 })
        let candidate = Vector512Optimized(generator: { $0 % 2 == 0 ? -1e-36 : -1e-10 })
        for count in [1, 5] {
            let scores = BatchKernels_SoA.batchDotProduct512(query: query, candidates: Array(repeating: candidate, count: count))
            #expect(scores.count == count)
            #expect(scores.allSatisfy { $0.bitPattern == 0 }, "n=\(count): \(scores.map { String($0.bitPattern, radix: 16) })")
        }
    }

    /// On the toolchain under test the standard library member adds in the same order and differs only by its -0 start.
    /// If a toolchain changes the stdlib member's order, this test fails while the tests above still pin the module
    /// function; update SIMDExtensions.swift's documentation and the T12 shadow-lint allowlist reason.
    @Test func stdlibMemberDiffersOnlyByItsNegativeZeroStart() {
        #expect(t14StdlibSum(SIMD4<Float>(repeating: -0.0)).bitPattern == (-0.0 as Float).bitPattern)
        #expect(t14StdlibSum(SIMD4<Double>(repeating: -0.0)).bitPattern == (-0.0 as Double).bitPattern)
        #expect(t14StdlibSum(SIMD4<Float>(1e8, 1, -1e8, 1)).bitPattern == SIMD4<Float>(1e8, 1, -1e8, 1).sum().bitPattern)
        #expect(t14StdlibSum(SIMD4<Float>(16_777_216, 1, 1, 1)).bitPattern == 0x4B80_0000)
        var random = P01OpsRandom(state: 0x5_0004)
        var differing: [String] = []
        for _ in 0..<2_000 {
            var float4 = SIMD4<Float>(), double8 = SIMD8<Double>()
            for lane in 0..<4 {
                float4[lane] = random.unit() * Float(sign: .plus, exponent: Int(random.next() % 61) - 30, significand: 1)
            }
            for lane in 0..<8 {
                double8[lane] = Double(random.unit()) * Double(sign: .plus, exponent: Int(random.next() % 121) - 60, significand: 1)
            }
            if t14StdlibSum(float4).bitPattern != float4.sum().bitPattern { differing.append("\(float4)") }
            if t14StdlibSum(double8).bitPattern != double8.sum().bitPattern { differing.append("\(double8)") }
        }
        #expect(differing.isEmpty, "\(differing.count) random vectors differ: \(differing.prefix(3))")
    }
}
