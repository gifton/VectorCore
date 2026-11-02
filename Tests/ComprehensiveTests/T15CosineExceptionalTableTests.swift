import Foundation
import Testing
@testable import VectorCore

// T15 test T2: the registry-driven exceptional table. Every registered cosine implementation (T15CosineRegistry) is
// evaluated on one fixture table per width and must give the class the T15 policy rules for that fixture:
//   P1  a NaN or infinite coordinate on either side (F01-F12), or an undefined similarity, gives NaN distance; the
//       T15-guarded paths return `Float.nan` (0x7FC00000) exactly, so tiled, untiled, paired and tail paths agree
//       to the bit (ruling 2). Propagated payloads (GEMM, similarity quotients) are checked as NaN only.
//   P2  the [-1, 1] clamp applies only to finite similarities: the overshoot fixtures F25-F28 give exactly 0 or 2.
//   P3a two vectors in an implementation's own zero class give distance 1: F13 and F16 everywhere, and the tiny
//       fixtures a family already treats as zero (for example F18/F19 on the helper family, whose zero class is a
//       squared norm at or below Float.leastNormalMagnitude). P3b one zero vector gives 1 (F14, F15, F17).
//   P3c the tiny-magnitude classes F18-F21 are pinned per family as documented divergences (not unified): P3c keeps
//       each implementation's zero-class threshold unchanged, not the value two vectors inside it used to get.
//   P4  F22-F24 (squared norm overflows Float) are outside the documented domain; their current class is pinned. The
//       FP32 x FP16 cores (fp16Mixed rows) have a narrower domain, ||q||^2 * ||c||^2 < Float.greatestFiniteMagnitude,
//       documented at MixedPrecisionKernels.cosineMixedCore. No fixture here has two finite squared norms whose
//       product overflows: F22-F24 overflow a squared norm itself (or an FP16 conversion), so that case is not a row.
//   P5  similarity APIs give NaN for a non-finite magnitude and 0 for a zero vector.
// Bit-equality groups pin agreement where implementations share arithmetic.
// Fixture definitions are the T15 phase 1 probe's (T15/phase1/probes/gen_probe.py), so this table and the probe agree.

// MARK: - Fixtures

struct T15Fixture: Sendable {
    let name: String
    let a: [Float]
    let b: [Float]
    var code: String { String(name.prefix(3)) }
}

private struct T15TableLCG {
    var state: UInt64
    mutating func next() -> Float {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        let u = Float((state >> 40) & 0xFFFFFF) / Float(1 << 24)
        return u * 2 - 1
    }
}

private func t15TableRandom(_ d: Int, seed: UInt64) -> [Float] {
    var generator = T15TableLCG(state: seed &* 0x9E3779B97F4A7C15 &+ 17)
    return (0..<d).map { _ in generator.next() }
}

/// Overshoot fixtures (phase 1 probe `searchOvershoot`). formA: coordinates (1, k/1024) with s = 1 + k^2/2^20 exact and
/// s / (sqrt(s) * sqrt(s)) > 1, so every FP32 path that forms sqrt(A) * sqrt(B) overshoots in every summation order.
/// formB: a = (1, k/1024), b = m * a, FP16-exact, overshooting the FP16 cores' sqrt(A * B) form.
enum T15Overshoot {
    static let formA: Int = {
        for k in 1..<1024 {
            let s: Float = 1 + Float(k * k) / Float(1 << 20)
            let r = s.squareRoot()
            if r * r < s && s / (r * r) > 1 { return k }
        }
        return -1
    }()
    static let formB: (k: Int, m: Int) = {
        for m in [3, 5, 7, 9, 11, 13] {
            for k in 1..<1024 {
                let a4 = Float(k) / 1024, b4 = Float(m) * a4
                guard Float(Float16Bits.roundTrip(a4)) == a4, Float(Float16Bits.roundTrip(b4)) == b4 else { continue }
                let dot = (Float(1) * Float(m)).addingProduct(a4, b4)
                let aa = Float(1).addingProduct(a4, a4)
                let bb = (Float(m) * Float(m)).addingProduct(b4, b4)
                if dot / (aa * bb).squareRoot() > 1 { return (k, m) }
            }
        }
        return (1, 3)
    }()
}

/// FP16 representability through the library's own converter (Float16 is unavailable on Intel macOS).
enum Float16Bits {
    static func roundTrip(_ value: Float) -> Float {
        MixedPrecisionKernels.fp16ToFp32_scalar(MixedPrecisionKernels.fp32ToFp16_scalar(value))
    }
}

func t15Fixtures(_ d: Int) -> [T15Fixture] {
    let x = t15TableRandom(d, seed: 1), y = t15TableRandom(d, seed: 2)
    func with(_ v: [Float], _ edits: [(Int, Float)]) -> [Float] { var w = v; for (i, e) in edits { w[i] = e }; return w }
    func twoCoord(_ k: Int, sign: Float = 1) -> [Float] {
        var v = [Float](repeating: 0, count: d); v[0] = sign; v[4] = sign * Float(k) / 1024; return v
    }
    let zero = [Float](repeating: 0, count: d), negZero = [Float](repeating: -0.0, count: d)
    let kA = T15Overshoot.formA, (kB, mB) = T15Overshoot.formB
    return [
        T15Fixture(name: "F00.finite", a: x, b: y),
        T15Fixture(name: "F01.nanA", a: with(x, [(7, .nan)]), b: y),
        T15Fixture(name: "F02.nanB", a: x, b: with(y, [(7, .nan)])),
        T15Fixture(name: "F03.nanBoth", a: with(x, [(7, .nan)]), b: with(y, [(9, .nan)])),
        T15Fixture(name: "F04.posInfA", a: with(x, [(7, .infinity)]), b: y),
        T15Fixture(name: "F05.posInfB", a: x, b: with(y, [(7, .infinity)])),
        T15Fixture(name: "F06.negInfB", a: x, b: with(y, [(7, -.infinity)])),
        T15Fixture(name: "F07.infMixB", a: x, b: with(y, [(7, .infinity), (9, -.infinity)])),
        T15Fixture(name: "F08.allPosInfB", a: x, b: [Float](repeating: .infinity, count: d)),
        T15Fixture(name: "F09.infB.zeroCoordA", a: with(x, [(7, 0)]), b: with(y, [(7, .infinity)])),
        T15Fixture(name: "F10.nanA.zeroB", a: with(x, [(7, .nan)]), b: zero),
        T15Fixture(name: "F11.infA.zeroB", a: with(x, [(7, .infinity)]), b: zero),
        T15Fixture(name: "F12.zeroA.nanB", a: zero, b: with(y, [(7, .nan)])),
        T15Fixture(name: "F13.zeroBoth", a: zero, b: zero),
        T15Fixture(name: "F14.zeroA", a: zero, b: y),
        T15Fixture(name: "F15.zeroB", a: x, b: zero),
        T15Fixture(name: "F16.negZeroBoth", a: negZero, b: negZero),
        T15Fixture(name: "F17.negZeroA", a: negZero, b: y),
        T15Fixture(name: "F18.subnormalCoords.parallel", a: y.map { $0 * 1e-41 }, b: y.map { $0 * 1e-41 }),
        T15Fixture(name: "F19.subnormalSumSq.parallel", a: y.map { $0 * 1e-21 }, b: y.map { $0 * 1e-21 }),
        T15Fixture(name: "F20.tinyNormalSumSq.parallel", a: y.map { $0 * 3e-19 }, b: y.map { $0 * 3e-19 }),
        T15Fixture(name: "F21.subnormalSumSqA.normalB.parallel", a: y.map { $0 * 1e-21 }, b: y),
        T15Fixture(name: "F22.overflow.parallel", a: y.map { $0 * 1e20 }, b: y.map { $0 * 1e20 }),
        T15Fixture(name: "F23.overflowA.normalB.parallel", a: y.map { $0 * 1e20 }, b: y),
        T15Fixture(name: "F24.greatestFiniteCoordA", a: with(x, [(7, .greatestFiniteMagnitude)]), b: y),
        T15Fixture(name: "F25.overshootA.self", a: twoCoord(kA), b: twoCoord(kA)),
        T15Fixture(name: "F26.overshootA.anti", a: twoCoord(kA), b: twoCoord(kA, sign: -1)),
        T15Fixture(name: "F27.overshootB.scaled", a: twoCoord(kB), b: twoCoord(kB).map { $0 * Float(mB) }),
        T15Fixture(name: "F28.overshootB.antiScaled", a: twoCoord(kB), b: twoCoord(kB).map { $0 * -Float(mB) }),
        T15Fixture(name: "F29.random.self", a: y, b: y),
        T15Fixture(name: "F30.random.anti", a: y, b: y.map { -$0 }),
        T15Fixture(name: "F31.negZeroCoords.finite", a: x.map { $0 == 0 ? -0.0 : $0 }, b: with(y, [(3, -0.0), (5, -0.0)]))
    ]
}

// MARK: - Expected classes

enum T15Expect: CustomStringConvertible {
    case nanCanonical          // exactly Float.nan, 0x7FC00000
    case nanAny                // any NaN (a propagated payload is never pinned)
    case bits(UInt32)          // exact bit pattern
    case near0                 // finite, 0 <= v <= 1e-3
    case near2                 // finite, 2 - 1e-3 <= v <= 2
    case unitRange             // finite, 0 <= v <= 2
    case simFinite             // finite similarity
    case notNaN                // documented divergence: finite (NaN swallowed), routed to root

    static let zero = T15Expect.bits(0x0000_0000), one = T15Expect.bits(0x3F80_0000), two = T15Expect.bits(0x4000_0000)

    func admits(_ v: Float) -> Bool {
        switch self {
        case .nanCanonical: return v.bitPattern == 0x7FC0_0000
        case .nanAny: return v.isNaN
        case .bits(let b): return v.bitPattern == b
        case .near0: return v.isFinite && v >= 0 && v <= 1e-3
        case .near2: return v.isFinite && v >= 2 - 1e-3 && v <= 2
        case .unitRange: return v.isFinite && v >= 0 && v <= 2
        case .simFinite: return v.isFinite
        case .notNaN: return !v.isNaN
        }
    }

    var description: String {
        switch self {
        case .nanCanonical: return "NaN 0x7fc00000"
        case .nanAny: return "NaN"
        case .bits(let b): return String(format: "0x%08x", b)
        case .near0: return "finite in [0, 1e-3]"
        case .near2: return "finite in [2 - 1e-3, 2]"
        case .unitRange: return "finite in [0, 2]"
        case .simFinite: return "finite similarity"
        case .notNaN: return "not NaN (routed divergence)"
        }
    }
}

/// Implementation families: rows of one family share the expected class on every fixture.
enum T15Family: String, CaseIterable {
    case helper        // CosineKernels.calculateCosineDistance and every path built on it
    case preNorm       // 1 - clamp(dot) on unit-norm inputs (fed unnormalized fixtures here)
    case witness       // generic CosineDistance witness (vDSP sums, product guard); on Vector384Optimized and the
                       // generic and dynamic types (on 512/768/1536 it calls CosineKernels: T14 phase 3, family helper)
    case vpExt         // VectorProtocol.cosineDistance(to:) and distance(to:metric: .cosine)
    case soa           // BatchKernels_SoA fused cosine (2-way, 4-way, tiled, Safe)
    case soaPre        // BatchKernels_SoA pre-normalized cosine
    case fp16Pair      // MixedPrecisionKernels.cosineFP16Core (FP16 x FP16)
    case fp16Mixed     // MixedPrecisionKernels.cosineMixedCore (FP32 x FP16 either way)
    case mixedRange    // MixedPrecisionKernels.range_cosine_mixed_core
    case gemm          // MatrixDistance.cosineDistanceMatrix (scaled normalization)
    case similarity    // cosineSimilarity APIs of VectorProtocol and the optimized vectors
    case similarityArray // cosineSimilarity of the array SIMD providers (magnitude sqrt(sum of squares) in Float)
}

/// The ruled class for (family, fixture). F18-F24 are the pinned P3c/P4 divergences (T15 phase 1 report section 3.1,
/// with P3a applied to each implementation's own zero class).
func t15Expected(_ family: T15Family, _ code: String) -> T15Expect {
    let n = Int(code.dropFirst()) ?? -1
    if family == .similarity || family == .similarityArray {
        switch n {
        case 1...12: return .nanCanonical
        case 13...18: return .zero
        // P4 divergence: the array providers' Float sum of squares overflows for F22-F24 (finite coordinates whose
        // squared norm exceeds Float), so their magnitude is infinite and the similarity NaN; the widened (optimized)
        // and scaled (VectorProtocol) magnitudes stay finite.
        case 22: return family == .similarityArray ? .nanCanonical : .nanAny
        case 23, 24: return family == .similarityArray ? .nanCanonical : .simFinite
        default: return .simFinite
        }
    }
    switch n {
    case 1...12: return family == .gemm ? .nanAny : .nanCanonical
    case 13...17: return .one
    case 25: return .zero
    case 26: return .two
    case 27: return family == .vpExt ? .near0 : .zero
    case 28: return .two
    case 29: return .near0
    case 30: return .near2
    case 0, 31: return .unitRange
    default: break
    }
    // Pinned divergences F18-F24.
    let pins: [T15Family: [T15Expect]] = [
        // F18, F19, F20, F21, F22, F23, F24
        .helper: [.one, .one, .near0, .one, .nanCanonical, .one, .one],
        .preNorm: [.one, .one, .one, .one, .nanCanonical, .zero, .zero],
        .witness: [.one, .one, .near0, .near0, .nanCanonical, .one, .one],
        .vpExt: [.one, .near0, .near0, .near0, .nanCanonical, .near0, .one],
        .soa: [.one, .near0, .near0, .near0, .nanCanonical, .one, .one],
        .soaPre: [.one, .one, .one, .one, .nanCanonical, .zero, .zero],
        .fp16Pair: [.one, .one, .one, .one, .nanCanonical, .nanCanonical, .nanCanonical],
        .fp16Mixed: [.one, .one, .one, .one, .nanCanonical, .nanCanonical, .nanCanonical],
        .mixedRange: [.one, .one, .one, .one, .nanCanonical, .nanCanonical, .nanCanonical],
        .gemm: [.near0, .near0, .near0, .near0, .near0, .near0, .unitRange]
    ]
    return pins[family]![n - 18]
}

// MARK: - Registry rows

struct T15Row {
    let id: String
    let family: T15Family
    let eval: (T15Fixture) async throws -> Float
}

@inline(never)
func t15Witness<V: VectorProtocol>(_ a: V, _ b: V) -> Float where V.Scalar == Float {
    CosineDistance().distance(a, b)   // generic context: the CosineDistance witness, never a concrete overload
}

private func withOut(_ n: Int, _ body: (UnsafeMutableBufferPointer<Float>) throws -> Void) rethrows -> [Float] {
    var out = [Float](repeating: -12345, count: n)
    try out.withUnsafeMutableBufferPointer { try body($0) }
    return out
}

/// Per-width hooks for the optimized FP32 types.
protocol T15Width: VectorProtocol where Scalar == Float {
    static var width: Int { get }
    static func ckFused(_ a: Self, _ b: Self) -> Float
    static func ckPre(_ a: Self, _ b: Self) -> Float
    static func bkFused(_ q: Self, _ c: [Self], _ r: Range<Int>, _ o: UnsafeMutableBufferPointer<Float>)
    static func bkPre(_ q: Self, _ c: [Self], _ r: Range<Int>, _ o: UnsafeMutableBufferPointer<Float>)
    func t15Similarity(_ other: Self) -> Float
}

/// Hooks for the widths with SoA, FP16 and concrete-overload paths (512, 768, 1536).
protocol T15WideWidth: T15Width, SoACompatible {
    associatedtype FP16
    static func soaFused(_ q: Self, _ c: [Self]) -> [Float]
    static func soaPre(_ q: Self, _ s: SoA<Self>, _ o: UnsafeMutableBufferPointer<Float>)
    static func overloadDistance(_ a: Self, _ b: Self) -> Float
    static func overloadBatch(_ a: Self, _ b: [Self]) -> [Float]
    static func helpersPre(_ q: Self, _ c: [Self], _ o: UnsafeMutableBufferPointer<Float>)
    static func toFP16(_ v: Self) -> FP16
    static func fp16Pair(_ a: FP16, _ b: FP16) -> Float
    static func fp16Core(_ a: FP16, _ b: FP16) -> Float
    static func q32c16(_ q: Self, _ c: FP16) -> Float
    static func q16c32(_ q: FP16, _ c: Self) -> Float
    static func mixedRange(_ q: Self, _ c: [FP16], _ r: Range<Int>, _ o: UnsafeMutableBufferPointer<Float>)
}

extension Vector384Optimized: T15Width {
    static var width: Int { 384 }
    static func ckFused(_ a: Self, _ b: Self) -> Float { CosineKernels.distance384_fused(a, b) }
    static func ckPre(_ a: Self, _ b: Self) -> Float { CosineKernels.distance384_preNormalized(a, b) }
    static func bkFused(_ q: Self, _ c: [Self], _ r: Range<Int>, _ o: UnsafeMutableBufferPointer<Float>) {
        BatchKernels.range_cosine_fused_384(query: q, candidates: c, range: r, out: o)
    }
    static func bkPre(_ q: Self, _ c: [Self], _ r: Range<Int>, _ o: UnsafeMutableBufferPointer<Float>) {
        BatchKernels.range_cosine_preNorm_384(query: q, candidates: c, range: r, out: o)
    }
    func t15Similarity(_ other: Self) -> Float { cosineSimilarity(to: other) }
}

extension Vector512Optimized: T15WideWidth {
    typealias FP16 = MixedPrecisionKernels.Vector512FP16
    static var width: Int { 512 }
    static func ckFused(_ a: Self, _ b: Self) -> Float { CosineKernels.distance512_fused(a, b) }
    static func ckPre(_ a: Self, _ b: Self) -> Float { CosineKernels.distance512_preNormalized(a, b) }
    static func bkFused(_ q: Self, _ c: [Self], _ r: Range<Int>, _ o: UnsafeMutableBufferPointer<Float>) {
        BatchKernels.range_cosine_fused_512(query: q, candidates: c, range: r, out: o)
    }
    static func bkPre(_ q: Self, _ c: [Self], _ r: Range<Int>, _ o: UnsafeMutableBufferPointer<Float>) {
        BatchKernels.range_cosine_preNorm_512(query: q, candidates: c, range: r, out: o)
    }
    func t15Similarity(_ other: Self) -> Float { cosineSimilarity(to: other) }
    static func soaFused(_ q: Self, _ c: [Self]) -> [Float] { BatchKernels_SoA.batchCosineDistance512(query: q, candidates: c) }
    static func soaPre(_ q: Self, _ s: SoA<Self>, _ o: UnsafeMutableBufferPointer<Float>) {
        BatchKernels_SoA.batchCosineDistancePreNorm512(query: q, soa: s, out: o)
    }
    static func overloadDistance(_ a: Self, _ b: Self) -> Float { CosineDistance().distance(a, b) }
    static func overloadBatch(_ a: Self, _ b: [Self]) -> [Float] { CosineDistance().batchDistance(query: a, candidates: b) }
    static func helpersPre(_ q: Self, _ c: [Self], _ o: UnsafeMutableBufferPointer<Float>) {
        CosineBatchHelpers.batch_cosine_preNormalized_512(queryNormalized: q, candidatesNormalized: c, out: o)
    }
    static func toFP16(_ v: Self) -> FP16 { FP16(from: v) }
    static func fp16Pair(_ a: FP16, _ b: FP16) -> Float { MixedPrecisionKernels.cosine512(a, b) }
    static func fp16Core(_ a: FP16, _ b: FP16) -> Float {
        a.storage.withUnsafeBufferPointer { pa in b.storage.withUnsafeBufferPointer { pb in
            MixedPrecisionKernels.cosineFP16Core(aPtr: pa.baseAddress!, bPtr: pb.baseAddress!, count: 512) } }
    }
    static func q32c16(_ q: Self, _ c: FP16) -> Float { MixedPrecisionKernels.cosine512(query: q, candidate: c) }
    static func q16c32(_ q: FP16, _ c: Self) -> Float { MixedPrecisionKernels.cosine512(query: q, candidate: c) }
    static func mixedRange(_ q: Self, _ c: [FP16], _ r: Range<Int>, _ o: UnsafeMutableBufferPointer<Float>) {
        MixedPrecisionKernels.range_cosine_mixed_512(query: q, candidatesFP16: c, range: r, out: o)
    }
}

extension Vector768Optimized: T15WideWidth {
    typealias FP16 = MixedPrecisionKernels.Vector768FP16
    static var width: Int { 768 }
    static func ckFused(_ a: Self, _ b: Self) -> Float { CosineKernels.distance768_fused(a, b) }
    static func ckPre(_ a: Self, _ b: Self) -> Float { CosineKernels.distance768_preNormalized(a, b) }
    static func bkFused(_ q: Self, _ c: [Self], _ r: Range<Int>, _ o: UnsafeMutableBufferPointer<Float>) {
        BatchKernels.range_cosine_fused_768(query: q, candidates: c, range: r, out: o)
    }
    static func bkPre(_ q: Self, _ c: [Self], _ r: Range<Int>, _ o: UnsafeMutableBufferPointer<Float>) {
        BatchKernels.range_cosine_preNorm_768(query: q, candidates: c, range: r, out: o)
    }
    func t15Similarity(_ other: Self) -> Float { cosineSimilarity(to: other) }
    static func soaFused(_ q: Self, _ c: [Self]) -> [Float] { BatchKernels_SoA.batchCosineDistance768(query: q, candidates: c) }
    static func soaPre(_ q: Self, _ s: SoA<Self>, _ o: UnsafeMutableBufferPointer<Float>) {
        BatchKernels_SoA.batchCosineDistancePreNorm768(query: q, soa: s, out: o)
    }
    static func overloadDistance(_ a: Self, _ b: Self) -> Float { CosineDistance().distance(a, b) }
    static func overloadBatch(_ a: Self, _ b: [Self]) -> [Float] { CosineDistance().batchDistance(query: a, candidates: b) }
    static func helpersPre(_ q: Self, _ c: [Self], _ o: UnsafeMutableBufferPointer<Float>) {
        CosineBatchHelpers.batch_cosine_preNormalized_768(queryNormalized: q, candidatesNormalized: c, out: o)
    }
    static func toFP16(_ v: Self) -> FP16 { FP16(from: v) }
    static func fp16Pair(_ a: FP16, _ b: FP16) -> Float { MixedPrecisionKernels.cosine768(a, b) }
    static func fp16Core(_ a: FP16, _ b: FP16) -> Float {
        a.storage.withUnsafeBufferPointer { pa in b.storage.withUnsafeBufferPointer { pb in
            MixedPrecisionKernels.cosineFP16Core(aPtr: pa.baseAddress!, bPtr: pb.baseAddress!, count: 768) } }
    }
    static func q32c16(_ q: Self, _ c: FP16) -> Float { MixedPrecisionKernels.cosine768(query: q, candidate: c) }
    static func q16c32(_ q: FP16, _ c: Self) -> Float { MixedPrecisionKernels.cosine768(query: q, candidate: c) }
    static func mixedRange(_ q: Self, _ c: [FP16], _ r: Range<Int>, _ o: UnsafeMutableBufferPointer<Float>) {
        MixedPrecisionKernels.range_cosine_mixed_768(query: q, candidatesFP16: c, range: r, out: o)
    }
}

extension Vector1536Optimized: T15WideWidth {
    typealias FP16 = MixedPrecisionKernels.Vector1536FP16
    static var width: Int { 1536 }
    static func ckFused(_ a: Self, _ b: Self) -> Float { CosineKernels.distance1536_fused(a, b) }
    static func ckPre(_ a: Self, _ b: Self) -> Float { CosineKernels.distance1536_preNormalized(a, b) }
    static func bkFused(_ q: Self, _ c: [Self], _ r: Range<Int>, _ o: UnsafeMutableBufferPointer<Float>) {
        BatchKernels.range_cosine_fused_1536(query: q, candidates: c, range: r, out: o)
    }
    static func bkPre(_ q: Self, _ c: [Self], _ r: Range<Int>, _ o: UnsafeMutableBufferPointer<Float>) {
        BatchKernels.range_cosine_preNorm_1536(query: q, candidates: c, range: r, out: o)
    }
    func t15Similarity(_ other: Self) -> Float { cosineSimilarity(to: other) }
    static func soaFused(_ q: Self, _ c: [Self]) -> [Float] { BatchKernels_SoA.batchCosineDistance1536(query: q, candidates: c) }
    static func soaPre(_ q: Self, _ s: SoA<Self>, _ o: UnsafeMutableBufferPointer<Float>) {
        BatchKernels_SoA.batchCosineDistancePreNorm1536(query: q, soa: s, out: o)
    }
    static func overloadDistance(_ a: Self, _ b: Self) -> Float { CosineDistance().distance(a, b) }
    static func overloadBatch(_ a: Self, _ b: [Self]) -> [Float] { CosineDistance().batchDistance(query: a, candidates: b) }
    static func helpersPre(_ q: Self, _ c: [Self], _ o: UnsafeMutableBufferPointer<Float>) {
        CosineBatchHelpers.batch_cosine_preNormalized_1536(queryNormalized: q, candidatesNormalized: c, out: o)
    }
    static func toFP16(_ v: Self) -> FP16 { FP16(from: v) }
    static func fp16Pair(_ a: FP16, _ b: FP16) -> Float { MixedPrecisionKernels.cosine1536(a, b) }
    static func fp16Core(_ a: FP16, _ b: FP16) -> Float {
        a.storage.withUnsafeBufferPointer { pa in b.storage.withUnsafeBufferPointer { pb in
            MixedPrecisionKernels.cosineFP16Core(aPtr: pa.baseAddress!, bPtr: pb.baseAddress!, count: 1536) } }
    }
    static func q32c16(_ q: Self, _ c: FP16) -> Float { MixedPrecisionKernels.cosine1536(query: q, candidate: c) }
    static func q16c32(_ q: FP16, _ c: Self) -> Float { MixedPrecisionKernels.cosine1536(query: q, candidate: c) }
    static func mixedRange(_ q: Self, _ c: [FP16], _ r: Range<Int>, _ o: UnsafeMutableBufferPointer<Float>) {
        MixedPrecisionKernels.range_cosine_mixed_1536(query: q, candidatesFP16: c, range: r, out: o)
    }
}

/// Rows every optimized width has.
func t15BaseRows<V: T15Width>(_ type: V.Type) -> [T15Row] {
    let d = V.width
    let filler = try! V(t15TableRandom(d, seed: 3))
    func pair(_ f: T15Fixture) -> (V, V) { (try! V(f.a), try! V(f.b)) }
    return [
        T15Row(id: "CK.fused.\(d)", family: .helper) { f in let (a, b) = pair(f); return V.ckFused(a, b) },
        T15Row(id: "CK.pre.\(d)", family: .preNorm) { f in let (a, b) = pair(f); return V.ckPre(a, b) },
        T15Row(id: "BK.fused.\(d).pair0", family: .helper) { f in
            let (a, b) = pair(f); return withOut(2) { V.bkFused(a, [b, filler], 0..<2, $0) }[0] },
        T15Row(id: "BK.fused.\(d).pair1", family: .helper) { f in
            let (a, b) = pair(f); return withOut(2) { V.bkFused(a, [filler, b], 0..<2, $0) }[1] },
        T15Row(id: "BK.fused.\(d).tail", family: .helper) { f in
            let (a, b) = pair(f); return withOut(1) { V.bkFused(a, [b], 0..<1, $0) }[0] },
        T15Row(id: "BK.pre.\(d).pair0", family: .preNorm) { f in
            let (a, b) = pair(f); return withOut(2) { V.bkPre(a, [b, filler], 0..<2, $0) }[0] },
        T15Row(id: "BK.pre.\(d).pair1", family: .preNorm) { f in
            let (a, b) = pair(f); return withOut(2) { V.bkPre(a, [filler, b], 0..<2, $0) }[1] },
        T15Row(id: "BK.pre.\(d).tail", family: .preNorm) { f in
            let (a, b) = pair(f); return withOut(1) { V.bkPre(a, [b], 0..<1, $0) }[0] },
        // T14 phase 3: on Vector512/768/1536Optimized the witness calls CosineKernels' fused kernel (helper family).
        T15Row(id: "witness.\(d)", family: d == 384 ? .witness : .helper) { f in let (a, b) = pair(f); return t15Witness(a, b) },
        T15Row(id: "vpext.cosineDistance.\(d)", family: .vpExt) { f in let (a, b) = pair(f); return a.cosineDistance(to: b) },
        T15Row(id: "vpext.distanceMetric.\(d)", family: .vpExt) { f in
            let (a, b) = pair(f); return a.distance(to: b, metric: .cosine) },
        T15Row(id: "sim.opt.\(d)", family: .similarity) { f in let (a, b) = pair(f); return a.t15Similarity(b) }
    ]
}

func t15GemmRow<V: T15Width & UnifiedVectorBuffer>(_ type: V.Type) -> T15Row {
    T15Row(id: "gemm.\(V.width)", family: .gemm) { f in
        MatrixDistance.cosineDistanceMatrix(queries: [try V(f.a)], candidates: [try V(f.b)])[0]
    }
}

/// P02 phase C (RD13): cosine through the explicit `.gram` route of `Operations.findNearestBatch` (Q1/N1, one tile):
/// MatrixDistance's normalization, GEMM and clamp run by the Gram engine (`MatrixDistance.cosinePostprocess`).
func t15GramRow<V: T15Width & UnifiedVectorBuffer & Sendable>(_ type: V.Type) -> T15Row {
    T15Row(id: "gram.\(V.width)", family: .gemm) { f in
        try await Operations.findNearestBatch(
            queries: [try V(f.a)], in: [try V(f.b)], k: 1, metric: CosineDistance(),
            cpuStrategy: .gram(queryTileRows: 1, candidateTileRows: 1, maximumTemporaryPayloadBytes: Int.max))[0][0].distance
    }
}

/// Rows of the 512/768/1536 widths.
func t15WideRows<V: T15WideWidth>(_ type: V.Type) -> [T15Row] {
    let d = V.width
    let filler = try! V(t15TableRandom(d, seed: 3))
    let filler16 = V.toFP16(filler)
    // Above the 8 MiB SoA tiling threshold (BatchKernels_SoA.shouldTile), as in the phase 1 probe.
    let tiledN = (8 * 1024 * 1024) / (V.lanes * 16) + 2
    func pair(_ f: T15Fixture) -> (V, V) { (try! V(f.a), try! V(f.b)) }
    var rows = [
        // Concrete-typed calls: they bind the CosineDistance witness since T14 phase 3 removed the concrete overloads, so
        // a reintroduced overload that leaves CosineKernels' arithmetic or the policy fails these rows.
        T15Row(id: "concrete.distance.\(d)", family: .helper) { f in let (a, b) = pair(f); return V.overloadDistance(a, b) },
        T15Row(id: "concrete.batchDistance.\(d)", family: .helper) { f in let (a, b) = pair(f); return V.overloadBatch(a, [b])[0] },
        T15Row(id: "SoA.fused.\(d).2way.N1", family: .soa) { f in let (a, b) = pair(f); return V.soaFused(a, [b])[0] },
        T15Row(id: "SoA.fused.\(d).2way.N2", family: .soa) { f in let (a, b) = pair(f); return V.soaFused(a, [b, filler])[0] },
        T15Row(id: "SoA.fused.\(d).4way.N4", family: .soa) { f in
            let (a, b) = pair(f); return V.soaFused(a, [b, filler, filler, filler])[0] },
        T15Row(id: "SoA.fused.\(d).4way.tail.N5", family: .soa) { f in
            let (a, b) = pair(f); return V.soaFused(a, [filler, filler, filler, filler, b])[4] },
        T15Row(id: "SoA.fused.\(d).tiled", family: .soa) { f in
            let (a, b) = pair(f)
            var candidates = [V](repeating: filler, count: tiledN); candidates[0] = b
            return V.soaFused(a, candidates)[0] },
        T15Row(id: "SoA.pre.\(d).N4", family: .soaPre) { f in
            let (a, b) = pair(f); let soa = SoA<V>.build(from: [b, filler, filler, filler])
            return withOut(4) { V.soaPre(a, soa, $0) }[0] },
        T15Row(id: "SoA.pre.\(d).N1", family: .soaPre) { f in
            let (a, b) = pair(f); let soa = SoA<V>.build(from: [b]); return withOut(1) { V.soaPre(a, soa, $0) }[0] },
        T15Row(id: "helpers.pre.\(d)", family: .preNorm) { f in
            let (a, b) = pair(f); return withOut(1) { V.helpersPre(a, [b], $0) }[0] },
        T15Row(id: "fp16.pair.\(d)", family: .fp16Pair) { f in
            let (a, b) = pair(f); return V.fp16Pair(V.toFP16(a), V.toFP16(b)) },
        T15Row(id: "fp16.core.\(d)", family: .fp16Pair) { f in
            let (a, b) = pair(f); return V.fp16Core(V.toFP16(a), V.toFP16(b)) },
        T15Row(id: "fp16.q32c16.\(d)", family: .fp16Mixed) { f in let (a, b) = pair(f); return V.q32c16(a, V.toFP16(b)) },
        T15Row(id: "fp16.q16c32.\(d)", family: .fp16Mixed) { f in let (a, b) = pair(f); return V.q16c32(V.toFP16(a), b) },
        T15Row(id: "mixedRange.\(d).pair0", family: .mixedRange) { f in
            let (a, b) = pair(f); return withOut(2) { V.mixedRange(a, [V.toFP16(b), filler16], 0..<2, $0) }[0] },
        T15Row(id: "mixedRange.\(d).tail", family: .mixedRange) { f in
            let (a, b) = pair(f); return withOut(1) { V.mixedRange(a, [V.toFP16(b)], 0..<1, $0) }[0] }
    ]
    if d == 512 {
        rows.append(T15Row(id: "SoA.fused.512.safe", family: .soa) { f in
            let a = try Vector512Optimized(f.a), b = try Vector512Optimized(f.b)
            let soa = SoA<Vector512Optimized>.build(from: [b, a, b])
            return try withOut(3) { try BatchKernels_SoA.batchCosineDistance512Safe(query: a, soa: soa, out: $0) }[0]
        })
    }
    return rows
}

/// Rows reached only at width 512 (Top-K kernels, generic types, providers).
func t15Extra512Rows() -> [T15Row] {
    typealias V = Vector512Optimized
    func pair(_ f: T15Fixture) -> (V, V) { (try! V(f.a), try! V(f.b)) }
    let provider = MixedPrecisionProvider(minBatchSize: 1)
    return [
        T15Row(id: "topk.fused.512", family: .helper) { f in
            let (a, b) = pair(f); return TopKSelection.nearestCosineFused512(k: 1, query: a, candidates: [b]).distances[0] },
        T15Row(id: "topk.pre.512", family: .preNorm) { f in
            let (a, b) = pair(f); return TopKSelection.nearestCosinePreNormalized512(k: 1, query: a, candidates: [b]).distances[0] },
        T15Row(id: "topk.nearest.witness.512", family: .helper) { f in
            let (a, b) = pair(f)
            return TopKSelection.nearest(k: 1, query: a, candidates: [b], metric: CosineDistance()).distances[0] },
        T15Row(id: "topkKernel.fused.512", family: .helper) { f in
            let (a, b) = pair(f); var buffer = TopKBuffer(k: 1, isMinHeap: false)
            TopKSelectionKernels.range_topk_cosine_fused_512(query: a, candidates: [b], range: 0..<1, k: 1, localTopK: &buffer)
            return buffer.vals[0] },
        T15Row(id: "topkKernel.pre.512", family: .preNorm) { f in
            let (a, b) = pair(f); var buffer = TopKBuffer(k: 1, isMinHeap: false)
            TopKSelectionKernels.range_topk_cosine_preNorm_512(query: a, candidates: [b], range: 0..<1, k: 1, localTopK: &buffer)
            return buffer.vals[0] },
        T15Row(id: "provider.distance.512", family: .helper) { f in
            let (a, b) = pair(f); return try await provider.distance(from: a, to: b, metric: .cosine) },
        T15Row(id: "provider.batchDistance.512", family: .helper) { f in
            let (a, b) = pair(f); return try await provider.batchDistance(from: a, to: [b], metric: .cosine)[0] },
        T15Row(id: "provider.distance.dynamic", family: .vpExt) { f in
            try await provider.distance(from: DynamicVector(f.a), to: DynamicVector(f.b), metric: .cosine) },
        T15Row(id: "witness.generic512", family: .witness) { f in
            t15Witness(try Vector<Dim512>(f.a), try Vector<Dim512>(f.b)) },
        T15Row(id: "witness.dynamic", family: .witness) { f in t15Witness(DynamicVector(f.a), DynamicVector(f.b)) },
        T15Row(id: "vpext.cosineDistance.generic512", family: .vpExt) { f in
            try Vector<Dim512>(f.a).cosineDistance(to: try Vector<Dim512>(f.b)) },
        T15Row(id: "vpext.cosineDistance.dynamic", family: .vpExt) { f in
            DynamicVector(f.a).cosineDistance(to: DynamicVector(f.b)) },
        T15Row(id: "gemm.dynamic", family: .gemm) { f in
            MatrixDistance.cosineDistanceMatrix(queries: [DynamicVector(f.a)], candidates: [DynamicVector(f.b)])[0] },
        T15Row(id: "sim.vp.dynamic", family: .similarity) { f in DynamicVector(f.a).cosineSimilarity(to: DynamicVector(f.b)) },
        T15Row(id: "sim.array.default", family: .similarityArray) { f in DefaultArraySIMDProvider().cosineSimilarity(f.a, f.b) },
        T15Row(id: "sim.array.swift", family: .similarityArray) { f in SwiftSIMDProvider().cosineSimilarity(f.a, f.b) },
        T15Row(id: "sim.array.accelerate", family: .similarityArray) { f in AccelerateArraySIMDProvider().cosineSimilarity(f.a, f.b) }
    ]
}

func t15AllRows() -> [(width: Int, rows: [T15Row])] {
    [
        (384, t15BaseRows(Vector384Optimized.self) + [t15GemmRow(Vector384Optimized.self)]),
        (512, t15BaseRows(Vector512Optimized.self) + [t15GemmRow(Vector512Optimized.self), t15GramRow(Vector512Optimized.self)]
            + t15WideRows(Vector512Optimized.self) + t15Extra512Rows()),
        (768, t15BaseRows(Vector768Optimized.self) + [t15GemmRow(Vector768Optimized.self), t15GramRow(Vector768Optimized.self)]
            + t15WideRows(Vector768Optimized.self)),
        (1536, t15BaseRows(Vector1536Optimized.self) + [t15GemmRow(Vector1536Optimized.self), t15GramRow(Vector1536Optimized.self)]
            + t15WideRows(Vector1536Optimized.self))
    ]
}

/// Bit-equality groups: ids (without the width suffix where noted) that share arithmetic and must agree to the bit on
/// every fixture, exceptional ones included. Since T14 phase 3 the `BatchKernels` pair loops compute their tail kernels'
/// arithmetic, and on 512/768/1536 the witness and the Top-K kernel compute `CosineKernels.fused`'s, so the pair, tail,
/// witness and Top-K kernel rows share one group per family.
func t15BitGroups(_ d: Int) -> [[String]] {
    var groups: [[String]] = [
        ["CK.fused.\(d)", "BK.fused.\(d).tail", "BK.fused.\(d).pair0", "BK.fused.\(d).pair1"],
        ["CK.pre.\(d)", "BK.pre.\(d).tail", "BK.pre.\(d).pair0", "BK.pre.\(d).pair1"],
        ["vpext.cosineDistance.\(d)", "vpext.distanceMetric.\(d)"]
    ]
    if d != 384 {
        groups[0] += ["concrete.distance.\(d)", "concrete.batchDistance.\(d)", "witness.\(d)"]
        groups[1] += ["helpers.pre.\(d)"]
        groups.append(["SoA.fused.\(d).2way.N1", "SoA.fused.\(d).2way.N2", "SoA.fused.\(d).4way.N4",
                       "SoA.fused.\(d).4way.tail.N5", "SoA.fused.\(d).tiled"] + (d == 512 ? ["SoA.fused.512.safe"] : []))
        groups.append(["fp16.pair.\(d)", "fp16.core.\(d)"])
        // P02 phase C: the `.gram` route at Q1/N1 makes the same single-row prepare, GEMM and postprocess calls.
        groups.append(["gemm.\(d)", "gram.\(d)"])
    }
    if d == 512 {
        groups[0] += ["topk.fused.512", "provider.distance.512", "provider.batchDistance.512", "topkKernel.fused.512",
                      "topk.nearest.witness.512"]
        groups[1] += ["topk.pre.512", "topkKernel.pre.512"]
        groups.append(["vpext.cosineDistance.dynamic", "provider.distance.dynamic"])
        groups.append(["sim.array.default", "sim.array.swift"])
    }
    return groups
}

// MARK: - Tests

@Suite("T15 cosine exceptional table (registry-driven)")
struct T15CosineExceptionalTableTests {

    /// Every registered implementation, every fixture, every width: the ruled class (P1-P5 and pinned divergences),
    /// and the bit-equality groups.
    @Test func everyImplementationFollowsThePolicy() async throws {
        var mismatches: [String] = []
        var evaluated = 0
        for (width, rows) in t15AllRows() {
            for fixture in t15Fixtures(width) {
                var values: [String: Float] = [:]
                for row in rows {
                    let value = try await row.eval(fixture)
                    values[row.id] = value
                    evaluated += 1
                    let expected = t15Expected(row.family, fixture.code)
                    if !expected.admits(value) {
                        mismatches.append("\(row.id) \(fixture.name): got \(String(format: "0x%08x", value.bitPattern)) "
                                          + "(\(value)), expected \(expected)")
                    }
                }
                for group in t15BitGroups(width) {
                    let present = group.compactMap { id in values[id].map { (id, $0.bitPattern) } }
                    #expect(present.count == group.count, "bit group \(group) has unevaluated members at \(width)")
                    if Set(present.map(\.1)).count > 1 {
                        mismatches.append("bit group \(fixture.name) D=\(width): "
                                          + present.map { "\($0.0)=\(String(format: "0x%08x", $0.1))" }.joined(separator: " "))
                    }
                }
            }
        }
        #expect(evaluated > 3000, "evaluated \(evaluated) cells")
        #expect(mismatches.isEmpty, "\(mismatches.count) cells off policy:\n\(mismatches.joined(separator: "\n"))")
    }

    /// The shared helper on synthetic (dot, sumAA, sumBB) inputs (phase 1 probe helper table).
    @Test func sharedHelperScalarTable() {
        let kA = T15Overshoot.formA
        let sA: Float = 1 + Float(kA * kA) / Float(1 << 20)
        let ln = Float.leastNormalMagnitude
        let cases: [(String, Float, Float, Float, T15Expect)] = [
            ("H00.unit", 0.5, 1, 1, .bits(Float(0.5).bitPattern)),
            ("H01.dotNaN", .nan, 1, 1, .nanCanonical),
            ("H02.sumAANaN", 1, .nan, 1, .nanCanonical),
            ("H03.sumBBNaN", 1, 1, .nan, .nanCanonical),
            ("H04.allNaN", .nan, .nan, .nan, .nanCanonical),
            ("H05.dotInf.sumAAInf", .infinity, .infinity, 1, .nanCanonical),
            // A finite dot with an infinite squared norm arises only from finite coordinates whose squares overflow
            // (an infinite coordinate makes dot non-finite): P4, similarity 0, distance 1.
            ("H06.dotFinite.sumAAInf", 1, .infinity, 1, .one),
            ("H07.dotNegInf.sumAAInf", -.infinity, .infinity, 1, .nanCanonical),
            ("H08.allInf", .infinity, .infinity, .infinity, .nanCanonical),
            ("H09.dotNaN.sumAANaN.sumBBZero", .nan, .nan, 0, .nanCanonical),
            ("H10.sumAAInf.sumBBZero", .nan, .infinity, 0, .nanCanonical),
            ("H11.bothZero", 0, 0, 0, .one),
            ("H12.zeroA", 0, 0, 1, .one),
            ("H13.zeroB", 0, 1, 0, .one),
            ("H14.negZero", -0.0, -0.0, -0.0, .one),
            ("H15.subnormalBoth", 1e-40, 1e-40, 1e-40, .one),
            ("H16.leastNormalBoth", ln, ln, ln, .one),
            ("H17.justAboveLeastNormal", ln.nextUp, ln.nextUp, ln.nextUp, .near0),
            ("H18.overshootA.self", sA, sA, sA, .zero),
            ("H19.overshootA.anti", -sA, sA, sA, .two),
            ("H20.dotOverDenomHuge", 1e30, ln.nextUp, ln.nextUp, .nanCanonical),
            ("H21.inf", .infinity, .infinity, .infinity, .nanCanonical),
            ("H22.sumAANegative", 1, -1, 1, .one)
        ]
        for (name, dot, aa, bb, expected) in cases {
            let value = CosineKernels.calculateCosineDistance(dot: dot, sumAA: aa, sumBB: bb)
            #expect(expected.admits(value), "helper.H \(name): got \(String(format: "0x%08x", value.bitPattern)), expected \(expected)")
        }
    }

    /// INT8 cosine (Double arithmetic). Quantization rejects non-finite inputs, so only finite classes exist.
    @Test func quantizedCosineTable() {
        let params = LinearQuantizationParams(minValue: -127, maxValue: 127)
        func lanes(_ count: Int, _ values: [Int8]) -> ContiguousArray<SIMD4<Int8>> {
            var storage = ContiguousArray<SIMD4<Int8>>(repeating: .zero, count: count)
            for (i, v) in values.enumerated() { storage[i / 4][i % 4] = v }
            return storage
        }
        let cases: [(String, [Int8], [Int8], T15Expect)] = [
            ("Q00.s3.self", [1, 1, 1], [1, 1, 1], .zero),   // Double ratio 1.0000000000000002: finite overshoot, clamped
            ("Q01.s3.anti", [1, 1, 1], [-1, -1, -1], .two),
            ("Q02.zeroBoth", [], [], .one),
            ("Q03.zeroA", [], [3, -2, 5, 7], .one),
            ("Q04.finite", [1, 1, 1], [3, -2, 5, 7], .unitRange)
        ]
        for (name, a, b, expected) in cases {
            let d512 = Vector512INT8(storage: lanes(128, a), params: params)
                .cosineDistance(to: Vector512INT8(storage: lanes(128, b), params: params))
            let d768 = Vector768INT8(storage: lanes(192, a), params: params)
                .cosineDistance(to: Vector768INT8(storage: lanes(192, b), params: params))
            let d1536 = Vector1536INT8(storage: lanes(384, a), params: params)
                .cosineDistance(to: Vector1536INT8(storage: lanes(384, b), params: params))
            for (label, value) in [("int8.512.", d512), ("int8.768.", d768), ("int8.1536.", d1536)] {
                #expect(expected.admits(value), "\(label)\(name): got \(value), expected \(expected)")
            }
        }
    }

    /// `[Vector<Dim768>].cosineSimilarityMatrix`: finite rows in [-1, 1], zero rows 0, non-finite rows NaN (P5).
    @Test func transformSimilarityMatrix() async throws {
        let d = 768
        let a = t15TableRandom(d, seed: 11), b = t15TableRandom(d, seed: 12)
        var nanRow = t15TableRandom(d, seed: 13); nanRow[7] = .nan
        var infRow = t15TableRandom(d, seed: 14); infRow[7] = .infinity
        let rows = try [a, b, [Float](repeating: 0, count: d), nanRow, infRow].map { try Vector<Dim768>($0) }
        let matrix = try await rows.cosineSimilarityMatrix()
        for i in 0..<rows.count {
            for j in 0..<rows.count {
                let value = matrix[i][j]
                let label = "transform.768 [\(i)][\(j)] = \(value)"
                if i >= 3 || j >= 3 {
                    #expect(value.isNaN, "\(label): a non-finite row must give NaN")
                } else if i == 2 || j == 2 {
                    #expect(value == 0, "\(label): a zero row gives similarity 0")
                } else {
                    #expect(value.isFinite && value >= -1 && value <= 1, "\(label)")
                }
            }
        }
    }

    /// Registry coverage: every `.table` prefix matches an evaluated row id (or the helper/INT8/transform tables),
    /// and every `.selection` route is a T1 route.
    @Test func registryCoverage() {
        let ids = t15AllRows().flatMap { $0.rows.map(\.id) }
            + ["helper.H", "int8.512.", "int8.768.", "int8.1536.", "transform.768"]
        let routes = Set(T15SelectionRoute.allCases.map(\.rawValue))
        #expect(Set(ids).count == ids.count, "duplicate row ids")
        for (site, coverage) in T15CosineRegistry.sites {
            switch coverage {
            case .table(let prefixes):
                for prefix in prefixes {
                    #expect(ids.contains { $0.hasPrefix(prefix) }, "\(site): no row id starts with \(prefix)")
                }
            case .selection(let named):
                for route in named { #expect(routes.contains(route), "\(site): no T1 route \(route)") }
            case let .tableAndSelection(prefixes, named):
                for prefix in prefixes {
                    #expect(ids.contains { $0.hasPrefix(prefix) }, "\(site): no row id starts with \(prefix)")
                }
                for route in named { #expect(routes.contains(route), "\(site): no T1 route \(route)") }
            case .external(let reference):
                #expect(reference.contains("::"), "\(site): external reference \(reference)")
            case .exempt(let reason):
                #expect(!reason.isEmpty, "\(site): exemption without a reason")
            }
        }
        // Every row id is reached from some registered site or is a route-level row of a registered site.
        for id in ids {
            let covered = T15CosineRegistry.sites.contains { _, coverage in
                if case .table(let prefixes) = coverage { return prefixes.contains { id.hasPrefix($0) } }
                if case .tableAndSelection(let prefixes, _) = coverage { return prefixes.contains { id.hasPrefix($0) } }
                return false
            }
            let routeRows: Set<String> = ["vpext.distanceMetric.384", "vpext.distanceMetric.512", "vpext.distanceMetric.768",
                                          "vpext.distanceMetric.1536", "provider.distance.512", "provider.batchDistance.512",
                                          "provider.distance.dynamic", "topk.nearest.witness.512"]
            #expect(covered || routeRows.contains(id), "row \(id) is not tied to a registered site")
        }
    }
}

// MARK: - Every other metric's exceptional behaviour (ruling 9 and the closure table)

@Suite("T15 metric exceptional table")
struct T15MetricExceptionalTableTests {
    /// Fixtures of the phase 1 other-metrics probe plus NaN in a, a same-sign infinity at one coordinate, and zeros.
    static func fixtures() -> [(String, [Float], [Float])] {
        let d = 512
        var x = [Float](repeating: 0, count: d), y = [Float](repeating: 0, count: d)
        for i in 0..<d { x[i] = Float(i % 7) * 0.1 - 0.3; y[i] = Float(i % 5) * 0.1 - 0.2 }
        func with(_ v: [Float], _ edits: [(Int, Float)]) -> [Float] { var w = v; for (i, e) in edits { w[i] = e }; return w }
        return [
            ("finite", x, y),
            ("nanA", with(x, [(7, .nan)]), y),
            ("nanB", x, with(y, [(7, .nan)])),
            ("posInfB", x, with(y, [(7, .infinity)])),
            ("infMixB", x, with(y, [(7, .infinity), (9, -.infinity)])),
            ("posInfBoth", with(x, [(7, .infinity)]), with(y, [(7, .infinity)])),
            ("zeroBoth", [Float](repeating: 0, count: d), [Float](repeating: 0, count: d))
        ]
    }

    enum Want { case nanCanonical, nanAny, posInf, negInf, finite, zero }

    static func admits(_ want: Want, _ v: Float) -> Bool {
        switch want {
        case .nanCanonical: return v.bitPattern == 0x7FC0_0000
        case .nanAny: return v.isNaN
        case .posInf: return v == .infinity
        case .negInf: return v == -.infinity
        case .finite: return v.isFinite
        case .zero: return v == 0
        }
    }

    @Test func everyMetricPinsItsExceptionalClass() {
        // Columns: finite, nanA, nanB, posInfB, infMixB, posInfBoth, zeroBoth.
        let table: [(String, [Want], (DynamicVector, DynamicVector) -> Float)] = [
            ("euclidean.witness", [.finite, .nanAny, .nanAny, .posInf, .posInf, .nanAny, .zero],
             { t15MetricWitness(EuclideanDistance(), $0, $1) }),
            ("manhattan.witness", [.finite, .nanAny, .nanAny, .posInf, .posInf, .nanAny, .zero],
             { t15MetricWitness(ManhattanDistance(), $0, $1) }),
            ("dot.witness", [.finite, .nanAny, .nanAny, .posInf, .nanAny, .negInf, .zero],
             { t15MetricWitness(DotProductDistance(), $0, $1) }),
            ("minkowski3.witness", [.finite, .nanAny, .nanAny, .posInf, .posInf, .nanAny, .zero],
             { t15MetricWitness(MinkowskiDistance(p: 3), $0, $1) }),
            // Ruling 9: any NaN coordinate gives NaN (Float.nan); a same-sign infinity pair differs by NaN as well.
            ("chebyshev.witness", [.finite, .nanCanonical, .nanCanonical, .posInf, .posInf, .nanCanonical, .zero],
             { t15MetricWitness(ChebyshevDistance(), $0, $1) }),
            // Ruling 9: Hamming does not treat NaN as bit 0. Infinities threshold like any value.
            ("hamming.witness", [.zero, .nanCanonical, .nanCanonical, .finite, .finite, .finite, .zero],
             { t15MetricWitness(HammingDistance(), $0, $1) }),
            ("vpext.euclideanDistance", [.finite, .nanAny, .nanAny, .posInf, .posInf, .nanAny, .zero], { $0.euclideanDistance(to: $1) }),
            ("vpext.manhattanDistance", [.finite, .nanAny, .nanAny, .posInf, .posInf, .nanAny, .zero], { $0.manhattanDistance(to: $1) }),
            // Fix round 1, ruling 13: `VectorProtocol.lInfinityNorm` returns Float.nan for a NaN element, as `magnitude`
            // and `l1Norm` propagate NaN, so the extension's Chebyshev distance is NaN for a NaN coordinate and for a
            // same-sign infinity pair (whose difference is NaN).
            ("vpext.chebyshevDistance", [.finite, .nanCanonical, .nanCanonical, .posInf, .posInf, .nanCanonical, .zero],
             { $0.chebyshevDistance(to: $1) }),
            ("vpext.distance(metric:.chebyshev)", [.finite, .nanCanonical, .nanCanonical, .posInf, .posInf, .nanCanonical, .zero],
             { $0.distance(to: $1, metric: .chebyshev) })
        ]
        for (label, wants, metric) in table {
            for (column, fixture) in Self.fixtures().enumerated() {
                let value = metric(DynamicVector(fixture.1), DynamicVector(fixture.2))
                #expect(Self.admits(wants[column], value),
                        "\(label) \(fixture.0): got \(String(format: "0x%08x", value.bitPattern)) (\(value)), expected \(wants[column])")
            }
        }
        // Concrete-typed calls, outside any generic context. `Vector<D>` binds its own `chebyshevDistance(to:)` and
        // `lInfinityNorm` (VectorMath.swift); its `distance(to:metric:)` is the VectorProtocol extension. A concrete
        // `ChebyshevDistance()` call on an optimized type binds whatever overload is visible here: the witness since
        // T14 removed the concrete overloads, so a reintroduced overload that drops a NaN fails these rows. The 768 and
        // 1536 fixtures append zeros, which leave every column's class unchanged.
        let chebyshevWants: [Want] = [.finite, .nanCanonical, .nanCanonical, .posInf, .posInf, .nanCanonical, .zero]
        for (column, fixture) in Self.fixtures().enumerated() {
            let va = try! Vector<Dim512>(fixture.1), vb = try! Vector<Dim512>(fixture.2)
            let oa = try! Vector512Optimized(fixture.1), ob = try! Vector512Optimized(fixture.2)
            let pad768 = [Float](repeating: 0, count: 256), pad1536 = [Float](repeating: 0, count: 1024)
            let ma = try! Vector768Optimized(fixture.1 + pad768), mb = try! Vector768Optimized(fixture.2 + pad768)
            let la = try! Vector1536Optimized(fixture.1 + pad1536), lb = try! Vector1536Optimized(fixture.2 + pad1536)
            let rows: [(String, Float)] = [
                ("vector512.chebyshevDistance", va.chebyshevDistance(to: vb)),
                ("vector512.distance(metric:.chebyshev)", va.distance(to: vb, metric: .chebyshev)),
                ("v512opt.ChebyshevDistance.distance (concrete)", ChebyshevDistance().distance(oa, ob)),
                ("v512opt.ChebyshevDistance.batchDistance (concrete)", ChebyshevDistance().batchDistance(query: oa, candidates: [ob])[0]),
                ("v768opt.ChebyshevDistance.distance (concrete)", ChebyshevDistance().distance(ma, mb)),
                ("v768opt.ChebyshevDistance.batchDistance (concrete)", ChebyshevDistance().batchDistance(query: ma, candidates: [mb])[0]),
                ("v1536opt.ChebyshevDistance.distance (concrete)", ChebyshevDistance().distance(la, lb)),
                ("v1536opt.ChebyshevDistance.batchDistance (concrete)",
                 ChebyshevDistance().batchDistance(query: la, candidates: [lb])[0])
            ]
            for (label, value) in rows {
                let want = chebyshevWants[column]
                #expect(Self.admits(want, value),
                        "\(label) \(fixture.0): got \(String(format: "0x%08x", value.bitPattern)) (\(value)), expected \(want)")
            }
        }
        // The generic witnesses on an optimized type agree with DynamicVector (same generic bodies).
        for (name, a, b) in Self.fixtures() {
            let oa = try! Vector512Optimized(a), ob = try! Vector512Optimized(b)
            let da = DynamicVector(a), db = DynamicVector(b)
            for (label, lhs, rhs) in [
                ("chebyshev", t15MetricWitness(ChebyshevDistance(), oa, ob), t15MetricWitness(ChebyshevDistance(), da, db)),
                ("hamming", t15MetricWitness(HammingDistance(), oa, ob), t15MetricWitness(HammingDistance(), da, db))
            ] {
                #expect(lhs.bitPattern == rhs.bitPattern, "\(label) \(name): V512 \(lhs) vs Dynamic \(rhs)")
            }
        }
    }
}

@inline(never)
func t15MetricWitness<M: DistanceMetric, V: VectorProtocol>(_ metric: M, _ a: V, _ b: V) -> Float
where V.Scalar == Float, M.Scalar == Float {
    metric.distance(a, b)   // generic context: the metric's witness, never a concrete overload
}
