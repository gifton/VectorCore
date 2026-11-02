import Foundation
import Testing
import VectorCoreC

// D02: the public VectorCoreC contracts, called through the module's public header (no @testable access).
//
// * vc_dot_int8: the exact integer dot product reduced modulo 2^32 and read in the Int32 range
//   (VectorCoreC.h). Every fixture is compared with an independent oracle (an exact Int64 sum, then a
//   modulo mapping written here); the oracle never calls VectorCoreC, and a saturating mutant of it must
//   disagree on at least one fixture (mutation control).
// * vc_range_l2sq_fp32_512: compact output out[0 ..< end - start] independent of start, padded stride,
//   NaN guards on both sides of the written range, an untouched suffix and an empty range with nil pointers.
// * the feature queries against the compiled configuration (ARM: compile-target macros; x86_64: runtime).
//
// These run the dispatcher actually linked into this test binary. The internal scalar and architecture entries,
// the NEON fallback build, AddressSanitizer/UndefinedBehaviorSanitizer and the x86_64 cross-compile are covered
// by Scripts/ci/tests/D02CContract.py (evidence script, not a CI gate).

private enum D02Int8Oracle {
    /// Exact Int64 sum of the products, then the modulo-2^32 mapping into Int32.
    static func dot(_ a: [Int8], _ b: [Int8], saturating: Bool = false) -> Int32 {
        var exact: Int64 = 0
        for i in a.indices { exact += Int64(a[i]) * Int64(b[i]) }
        return saturating ? saturate(exact) : modulo(exact)
    }

    static func exact(_ a: [Int8], _ b: [Int8]) -> Int64 {
        var exact: Int64 = 0
        for i in a.indices { exact += Int64(a[i]) * Int64(b[i]) }
        return exact
    }

    static func modulo(_ exact: Int64) -> Int32 {
        var r = exact % 4_294_967_296
        if r < 0 { r += 4_294_967_296 }
        if r >= 2_147_483_648 { r -= 4_294_967_296 }
        return Int32(r)
    }

    /// Deliberately wrong (mutation control): saturation instead of the modulo law.
    static func saturate(_ exact: Int64) -> Int32 {
        Int32(Swift.min(Swift.max(exact, Int64(Int32.min)), Int64(Int32.max)))
    }
}

private struct D02Int8Fixture {
    let name: String
    let a: [Int8]
    let b: [Int8]
}

private func d02Int8Fixtures() -> [D02Int8Fixture] {
    var fixtures: [D02Int8Fixture] = []
    func add(_ name: String, _ lanes: Int, _ pair: (Int) -> (Int8, Int8)) {
        var a = [Int8](repeating: 0, count: lanes), b = [Int8](repeating: 0, count: lanes)
        for i in 0..<lanes { (a[i], b[i]) = pair(i) }
        fixtures.append(D02Int8Fixture(name: "\(name)/\(lanes)", a: a, b: b))
    }
    var state: UInt32 = 0x2545_F491
    func next() -> Int8 {
        state = state &* 1_664_525 &+ 1_013_904_223
        return Int8(Int(state >> 24) - 128)
    }
    let lanesTable = [1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 512, 768, 1536, 131_071, 131_072, 131_073]
    for lanes in lanesTable {
        add("posMax", lanes) { _ in (-128, -128) }
        add("negMax", lanes) { _ in (-128, 127) }
        add("pos127", lanes) { _ in (127, 127) }
        add("zero", lanes) { i in (0, Int8(i & 0x7f)) }
        add("altSign", lanes) { i in (i & 1 == 1 ? -128 : 127, i % 3 == 0 ? 127 : -128) }
        add("unequal", lanes) { _ in (next(), next()) }
    }
    // 2^20 lanes of 16384 = 2^34: crosses four modulo boundaries (result 0); a mixed bounded case beside it.
    add("posMax", 1_048_576) { _ in (-128, -128) }
    add("unequal", 1_048_579) { _ in (next(), next()) }
    // Individual SDOT accumulator lanes wrap (each of the 16 int32 lanes gets 4 * floor(lanes / 64) products):
    // (-128,-128) wraps a lane at both sizes (2^31 and 2^32 per lane); (-128,127) wraps a lane only at 2^22 + 33
    // (262144 * -16256 < -2^31; at 2^21 + 17 only the horizontal sum wraps). The fix1/c-mutants saturating-lane mutant
    // mismatches on exactly those 3 fixtures. +17/+33 add tails.
    for lanes in [(1 << 21) + 17, (1 << 22) + 33] {
        add("laneWrapPosMax", lanes) { _ in (-128, -128) }
        add("laneWrapNegMax", lanes) { _ in (-128, 127) }
    }
    // Every int8 pair once: exact sum (sum a)(sum b) = (-128)(-128) = 16384.
    add("allPairs", 65_536) { i in (Int8((i >> 8) - 128), Int8((i & 0xff) - 128)) }
    // Cancellation: the prefix 131072 * 16384 = 2^31 leaves Int32; the final sum 2^24 fits.
    add("cancellation", 262_144) { i in (-128, i < 131_072 ? -128 : 127) }
    // Tail/final witnesses: the SIMD part (a multiple of every block size) fits Int32, the tail pushes it out.
    add("tailWitnessPositive", 131_073) { i in i >= 131_071 ? (127, 127) : (-128, -128) }
    add("tailWitnessNegative", 132_105) { _ in (-128, 127) }
    return fixtures
}

private func d02DispatcherDot(_ a: [Int8], _ b: [Int8]) -> Int32 {
    a.withUnsafeBufferPointer { ap in
        b.withUnsafeBufferPointer { bp in
            vc_dot_int8(ap.baseAddress, bp.baseAddress, ap.count)
        }
    }
}

@Suite("D02 VectorCoreC INT8 dispatcher contract")
struct D02Int8DispatcherContractTests {
    @Test("vc_dot_int8 equals the exact sum reduced modulo 2^32 on every fixture")
    func dispatcherFollowsTheModuloLaw() {
        let fixtures = d02Int8Fixtures()
        #expect(fixtures.count == 16 * 6 + 6 + 4)
        for fixture in fixtures {
            let got = d02DispatcherDot(fixture.a, fixture.b)
            #expect(got == D02Int8Oracle.dot(fixture.a, fixture.b), "\(fixture.name)")
        }
    }

    @Test("Fixed widths 512/768/1536 return the full integer value, within 1536 * 16384")
    func fixedWidthsAreTheFullIntegerValue() {
        for fixture in d02Int8Fixtures() where [512, 768, 1536].contains(fixture.a.count) {
            let exact = D02Int8Oracle.exact(fixture.a, fixture.b)
            #expect(Swift.abs(exact) <= 25_165_824, "\(fixture.name)")
            #expect(Int64(d02DispatcherDot(fixture.a, fixture.b)) == exact, "\(fixture.name)")
        }
    }

    @Test("Zero lanes reads nothing: nil pointers return 0")
    func zeroLanesWithNilPointers() {
        #expect(vc_dot_int8(nil, nil, 0) == 0)
    }

    @Test("Mutation control: a saturating oracle disagrees with the dispatcher on the overflowing fixtures")
    func saturatingOracleIsCaught() {
        let disagreements = d02Int8Fixtures().filter {
            d02DispatcherDot($0.a, $0.b) != D02Int8Oracle.dot($0.a, $0.b, saturating: true)
        }
        #expect(!disagreements.isEmpty)
        #expect(disagreements.allSatisfy { Swift.abs(D02Int8Oracle.exact($0.a, $0.b)) > Int64(Int32.max) })
    }

    @Test("Oracle self-check: the modulo mapping at the Int32 boundaries")
    func oracleBoundaries() {
        #expect(D02Int8Oracle.modulo(Int64(Int32.max)) == Int32.max)
        #expect(D02Int8Oracle.modulo(Int64(Int32.max) + 1) == Int32.min)
        #expect(D02Int8Oracle.modulo(-1) == -1)
        #expect(D02Int8Oracle.modulo(1 << 34) == 0)
        #expect(D02Int8Oracle.modulo(Int64(Int32.min) - 1) == Int32.max)
    }
}

@Suite("D02 VectorCoreC range512 contract")
struct D02Range512ContractTests {
    /// gamma_n(u) = n u / (1 - n u).
    private static func gamma(_ n: Double, _ u: Double) -> Double { n * u / (1 - n * u) }

    /// Tolerance for one row: 514 = 512 accumulations plus one rounding each for the difference and the square
    /// (any summation order, with or without FMA), so |computed - S| <= gamma_514(2^-24) * S for the exact S;
    /// the Double reference adds at most gamma_514(2^-53) * S of its own (counted twice to cover using the
    /// reference in place of S), and 512 * 2^-149 covers float underflow of a square.
    private static func tolerance(_ reference: Double) -> Double {
        (gamma(514, 0x1p-24) + 2 * gamma(514, 0x1p-53)) * reference + 512 * 0x1p-149
    }

    private struct Case { let stride: Int; let start: Int; let end: Int; let suffix: Int }

    @Test("Compact output, nonzero start, padded stride, both guards and an untouched suffix")
    func compactOutputWithGuards() {
        let cases = [Case(stride: 512, start: 0, end: 3, suffix: 2), Case(stride: 512, start: 5, end: 9, suffix: 3),
                     Case(stride: 520, start: 3, end: 8, suffix: 1), Case(stride: 1031, start: 1, end: 2, suffix: 0)]
        var state: UInt32 = 12_345
        func next() -> Float {
            state = state &* 1_664_525 &+ 1_013_904_223
            return Float(Int32(state >> 8) - (1 << 23)) / Float(1 << 23)
        }
        for c in cases {
            let q = (0..<512).map { _ in next() }
            var base = [Float](repeating: .nan, count: c.end * c.stride)   // NaN padding is never read
            for row in 0..<c.end { for i in 0..<512 { base[row * c.stride + i] = next() } }
            let count = c.end - c.start
            let outStorage = [Float](repeating: .nan, count: 1 + count + c.suffix + 1)
            // Written and read inside one closure (swiftlang/swift#92562 lost-store rule).
            let observed: [Float] = {
                var out = outStorage
                return out.withUnsafeMutableBufferPointer { o in
                    q.withUnsafeBufferPointer { qp in
                        base.withUnsafeBufferPointer { bp in
                            vc_range_l2sq_fp32_512(qp.baseAddress, bp.baseAddress, c.stride, c.start, c.end,
                                                   o.baseAddress! + 1)
                        }
                    }
                    return Array(o)
                }
            }()
            #expect(observed[0].isNaN, "leading guard, stride \(c.stride) start \(c.start)")
            for k in (1 + count)..<observed.count {
                #expect(observed[k].isNaN, "suffix/trailing guard \(k), stride \(c.stride) start \(c.start)")
            }
            for j in 0..<count {
                let rowStart = (c.start + j) * c.stride
                var reference = 0.0
                for i in 0..<512 { let d = Double(q[i]) - Double(base[rowStart + i]); reference += d * d }
                let got = observed[1 + j]
                #expect(got.isFinite && Swift.abs(Double(got) - reference) <= Self.tolerance(reference),
                        "row \(c.start + j): \(got) vs \(reference)")
            }
        }
    }

    @Test("A valid empty range performs no pointer access (nil pointers)")
    func emptyRangeTouchesNothing() {
        vc_range_l2sq_fp32_512(nil, nil, 512, 7, 7, nil)
        vc_range_l2sq_fp32_512(nil, nil, 512, 0, 0, nil)
    }
}

@Suite("D02 VectorCoreC feature queries")
struct D02FeatureQueryTests {
    @Test("Feature queries report the compiled configuration (ARM macros) or the runtime CPU (x86_64)")
    func featureQueriesMatchTheCompiledConfiguration() {
        let neon = vc_has_neon(), dotprod = vc_has_dotprod(), avx2 = vc_has_avx2(), avx512f = vc_has_avx512f()
        for value in [neon, dotprod, avx2, avx512f] { #expect(value == 0 || value == 1) }
        #if arch(arm64)
        #expect(neon == 1)                 // __ARM_NEON is defined for every arm64 target
        #expect(avx2 == 0 && avx512f == 0) // x86 queries return 0 off x86_64
        #if os(macOS)
        // `clang -target arm64-apple-macos14.0 -dM` defines __ARM_FEATURE_DOTPROD (Scripts/ci/tests/D02CContract.py
        // records the macro set); the query reports that compile-target macro, not a runtime probe.
        #expect(dotprod == 1)
        #endif
        #elseif arch(x86_64)
        #expect(neon == 0 && dotprod == 0)
        // Runtime queries: compare with the kernel's own report for the executing CPU.
        func sysctlFlag(_ name: String) -> Int32 {
            var value: Int32 = 0
            var size = MemoryLayout<Int32>.size
            return sysctlbyname(name, &value, &size, nil, 0) == 0 ? value : 0
        }
        #expect(avx2 == sysctlFlag("hw.optional.avx2_0"))
        #expect(avx512f == sysctlFlag("hw.optional.avx512f"))
        #endif
    }
}
