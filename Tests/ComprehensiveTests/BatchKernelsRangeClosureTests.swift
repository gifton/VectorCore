import Foundation
import Testing
@testable import VectorCore

// T14 phase 3 (class closure for position-dependent scores). A range helper writes `out[i - range.lowerBound]` for each
// candidate `i` of `range`, scoring candidates in pairs from `range.lowerBound` and an odd last candidate with a
// single-candidate kernel. Its score for a candidate must not depend on that position: slot 0 or slot 1 of a pair, the
// odd tail, and the same positions in a range that starts above 0 all give the bits of the single-candidate call.
// Before T14 phase 3 the eight cosine helpers of `BatchKernels` (fused and pre-normalized, 384 to 1536) failed this: their
// pair loops used other accumulators than their tail kernels, so a search score depended on the number of candidates
// and on the chunking. The pair loops now use `CosineKernels.fusedPair` and `DotKernels.dotPair`, the 2-way blocked forms
// of `CosineKernels.fused` and `DotKernels.dot`.
//
// `publicRangeHelpers` lists every public `range_*` helper of a public type in Sources; the list between the BEGIN and END
// markers is read by Scripts/ci/tests/test_range_helper_closure.py, which fails on a public helper missing from it and on
// a listed name that no longer exists. `everyListedHelperHasAnEvaluator` ties each name to its evaluator.

/// One candidate against one query, raw coordinates.
private struct ClosureFixture {
    let name: String
    let query: [Float]
    let candidate: [Float]
}

/// Random, unit-norm, near-duplicate (where the old cosine pair and tail arithmetics differed on most rows) and
/// exceptional pairs. Exceptional classes must agree too: NaN on either side, infinities, a zero candidate, tiny and
/// large magnitudes.
private func closureFixtures(_ n: Int) -> [ClosureFixture] {
    var random = P01OpsRandom(state: 0xC105_0000 &+ UInt64(n))
    func rand(_ scale: Float = 1) -> [Float] { (0..<n).map { _ in random.unit() * scale } }
    func unit(_ v: [Float]) -> [Float] {
        let norm = v.reduce(0.0) { $0 + Double($1) * Double($1) }.squareRoot()
        return v.map { Float(Double($0) / norm) }
    }
    func with(_ v: [Float], _ index: Int, _ value: Float) -> [Float] {
        var copy = v
        copy[index] = value
        return copy
    }
    var out: [ClosureFixture] = []
    for i in 0..<32 { out.append(ClosureFixture(name: "rand#\(i)", query: rand(), candidate: rand())) }
    for i in 0..<16 { out.append(ClosureFixture(name: "unit#\(i)", query: unit(rand()), candidate: unit(rand()))) }
    for i in 0..<32 {
        let a = rand()
        out.append(ClosureFixture(name: "near#\(i)", query: a, candidate: a.map { $0 + random.unit() * 1e-3 }))
    }
    let q = rand(), c = rand()
    out += [
        ClosureFixture(name: "nanCandidate", query: q, candidate: with(c, n / 2 + 1, .nan)),
        ClosureFixture(name: "nanQuery", query: with(q, 0, .nan), candidate: c),
        ClosureFixture(name: "posInfCandidate", query: q, candidate: with(c, n - 1, .infinity)),
        ClosureFixture(name: "negInfQuery", query: with(q, 7, -.infinity), candidate: c),
        ClosureFixture(name: "zeroCandidate", query: q, candidate: [Float](repeating: 0, count: n)),
        ClosureFixture(name: "zeroQuery", query: [Float](repeating: 0, count: n), candidate: c),
        ClosureFixture(name: "tinyCandidate", query: q, candidate: rand(1e-21)),
        ClosureFixture(name: "largeBoth", query: rand(1e18), candidate: rand(1e18))
    ]
    return out
}

/// One helper, evaluated for a candidate at six positions: slot 0 and slot 1 of a pair, the odd tail, and the same
/// three positions in a range that starts at 1. Every value is read inside the buffer closure and returned from it
/// (the Swift 6.4 lost-store lesson, T15/lost-store-investigation).
private struct RangeHelper {
    let width: Int
    let positions: (_ query: [Float], _ candidate: [Float], _ filler: [Float]) throws -> [Float]
}

private func rangeHelper<V, C>(
    _ width: Int, _ make: @escaping ([Float]) throws -> V, _ candidate: @escaping (V) -> C,
    run: @escaping (V, [C], Range<Int>, UnsafeMutableBufferPointer<Float>) -> Void
) -> RangeHelper {
    RangeHelper(width: width) { queryValues, candidateValues, fillerValues in
        let query = try make(queryValues), c = candidate(try make(candidateValues)), f = candidate(try make(fillerValues))
        func score(_ candidates: [C], _ range: Range<Int>, at index: Int) -> Float {
            var out = [Float](repeating: .nan, count: range.count)
            return out.withUnsafeMutableBufferPointer { buffer -> Float in
                run(query, candidates, range, buffer)
                return buffer[index - range.lowerBound]
            }
        }
        return [
            score([c], 0..<1, at: 0),                    // the single-candidate call (odd tail of a one-candidate range)
            score([c, f], 0..<2, at: 0),                 // pair slot 0
            score([f, c], 0..<2, at: 1),                 // pair slot 1
            score([f, f, c], 0..<3, at: 2),              // odd tail after a pair
            score([f, c, f], 1..<3, at: 1),              // pair slot 0 of a range starting at 1
            score([f, f, c], 1..<3, at: 2),              // pair slot 1 of a range starting at 1
            score([f, f, f, c], 1..<4, at: 3)            // odd tail of a range starting at 1
        ]
    }
}

private typealias MK = MixedPrecisionKernels

@Suite("Range helpers: a candidate's score does not depend on its position (T14)")
struct BatchKernelsRangeClosureTests {
    /// Every public `range_*` helper of a public type in Sources (Scripts/ci/tests/test_range_helper_closure.py).
    static let publicRangeHelpers: [String] = [
        // BEGIN PUBLIC RANGE HELPERS
        "BatchKernels.range_euclid2_384", "BatchKernels.range_euclid2_512", "BatchKernels.range_euclid2_768",
        "BatchKernels.range_euclid2_1536",
        "BatchKernels.range_cosine_preNorm_384", "BatchKernels.range_cosine_preNorm_512",
        "BatchKernels.range_cosine_preNorm_768", "BatchKernels.range_cosine_preNorm_1536",
        "BatchKernels.range_cosine_fused_384", "BatchKernels.range_cosine_fused_512", "BatchKernels.range_cosine_fused_768",
        "BatchKernels.range_cosine_fused_1536",
        "MixedPrecisionKernels.range_euclid2_mixed_512", "MixedPrecisionKernels.range_euclid2_mixed_768",
        "MixedPrecisionKernels.range_euclid2_mixed_1536",
        "MixedPrecisionKernels.range_cosine_mixed_512", "MixedPrecisionKernels.range_cosine_mixed_768",
        "MixedPrecisionKernels.range_cosine_mixed_1536",
        "MixedPrecisionKernels.range_dot_mixed_512", "MixedPrecisionKernels.range_dot_mixed_768",
        "MixedPrecisionKernels.range_dot_mixed_1536"
        // END PUBLIC RANGE HELPERS
    ]

    /// Internal helpers on the same pair and tail scheme (the Euclidean root helpers of the small-k routes).
    static let internalRangeHelpers: [String] = [
        "BatchKernels.range_euclid_384", "BatchKernels.range_euclid_512", "BatchKernels.range_euclid_768",
        "BatchKernels.range_euclid_1536"
    ]

    private static func batchKernels() -> [String: RangeHelper] {
        let v384: ([Float]) throws -> Vector384Optimized = { try Vector384Optimized($0) }
        let v512: ([Float]) throws -> Vector512Optimized = { try Vector512Optimized($0) }
        let v768: ([Float]) throws -> Vector768Optimized = { try Vector768Optimized($0) }
        let v1536: ([Float]) throws -> Vector1536Optimized = { try Vector1536Optimized($0) }
        return [
            "BatchKernels.range_euclid2_384": rangeHelper(384, v384, { $0 }, run: {
                BatchKernels.range_euclid2_384(query: $0, candidates: $1, range: $2, out: $3)
            }),
            "BatchKernels.range_euclid2_512": rangeHelper(512, v512, { $0 }, run: {
                BatchKernels.range_euclid2_512(query: $0, candidates: $1, range: $2, out: $3)
            }),
            "BatchKernels.range_euclid2_768": rangeHelper(768, v768, { $0 }, run: {
                BatchKernels.range_euclid2_768(query: $0, candidates: $1, range: $2, out: $3)
            }),
            "BatchKernels.range_euclid2_1536": rangeHelper(1536, v1536, { $0 }, run: {
                BatchKernels.range_euclid2_1536(query: $0, candidates: $1, range: $2, out: $3)
            }),
            "BatchKernels.range_euclid_384": rangeHelper(384, v384, { $0 }, run: {
                BatchKernels.range_euclid_384(query: $0, candidates: $1, range: $2, out: $3)
            }),
            "BatchKernels.range_euclid_512": rangeHelper(512, v512, { $0 }, run: {
                BatchKernels.range_euclid_512(query: $0, candidates: $1, range: $2, out: $3)
            }),
            "BatchKernels.range_euclid_768": rangeHelper(768, v768, { $0 }, run: {
                BatchKernels.range_euclid_768(query: $0, candidates: $1, range: $2, out: $3)
            }),
            "BatchKernels.range_euclid_1536": rangeHelper(1536, v1536, { $0 }, run: {
                BatchKernels.range_euclid_1536(query: $0, candidates: $1, range: $2, out: $3)
            }),
            "BatchKernels.range_cosine_preNorm_384": rangeHelper(384, v384, { $0 }, run: {
                BatchKernels.range_cosine_preNorm_384(query: $0, candidates: $1, range: $2, out: $3)
            }),
            "BatchKernels.range_cosine_preNorm_512": rangeHelper(512, v512, { $0 }, run: {
                BatchKernels.range_cosine_preNorm_512(query: $0, candidates: $1, range: $2, out: $3)
            }),
            "BatchKernels.range_cosine_preNorm_768": rangeHelper(768, v768, { $0 }, run: {
                BatchKernels.range_cosine_preNorm_768(query: $0, candidates: $1, range: $2, out: $3)
            }),
            "BatchKernels.range_cosine_preNorm_1536": rangeHelper(1536, v1536, { $0 }, run: {
                BatchKernels.range_cosine_preNorm_1536(query: $0, candidates: $1, range: $2, out: $3)
            }),
            "BatchKernels.range_cosine_fused_384": rangeHelper(384, v384, { $0 }, run: {
                BatchKernels.range_cosine_fused_384(query: $0, candidates: $1, range: $2, out: $3)
            }),
            "BatchKernels.range_cosine_fused_512": rangeHelper(512, v512, { $0 }, run: {
                BatchKernels.range_cosine_fused_512(query: $0, candidates: $1, range: $2, out: $3)
            }),
            "BatchKernels.range_cosine_fused_768": rangeHelper(768, v768, { $0 }, run: {
                BatchKernels.range_cosine_fused_768(query: $0, candidates: $1, range: $2, out: $3)
            }),
            "BatchKernels.range_cosine_fused_1536": rangeHelper(1536, v1536, { $0 }, run: {
                BatchKernels.range_cosine_fused_1536(query: $0, candidates: $1, range: $2, out: $3)
            })
        ]
    }

    private static func mixedPrecision() -> [String: RangeHelper] {
        let v512: ([Float]) throws -> Vector512Optimized = { try Vector512Optimized($0) }
        let v768: ([Float]) throws -> Vector768Optimized = { try Vector768Optimized($0) }
        let v1536: ([Float]) throws -> Vector1536Optimized = { try Vector1536Optimized($0) }
        let h512: (Vector512Optimized) -> MK.Vector512FP16 = { MK.Vector512FP16(from: $0) }
        let h768: (Vector768Optimized) -> MK.Vector768FP16 = { MK.Vector768FP16(from: $0) }
        let h1536: (Vector1536Optimized) -> MK.Vector1536FP16 = { MK.Vector1536FP16(from: $0) }
        return [
            "MixedPrecisionKernels.range_euclid2_mixed_512": rangeHelper(512, v512, h512, run: {
                MK.range_euclid2_mixed_512(query: $0, candidatesFP16: $1, range: $2, out: $3)
            }),
            "MixedPrecisionKernels.range_euclid2_mixed_768": rangeHelper(768, v768, h768, run: {
                MK.range_euclid2_mixed_768(query: $0, candidatesFP16: $1, range: $2, out: $3)
            }),
            "MixedPrecisionKernels.range_euclid2_mixed_1536": rangeHelper(1536, v1536, h1536, run: {
                MK.range_euclid2_mixed_1536(query: $0, candidatesFP16: $1, range: $2, out: $3)
            }),
            "MixedPrecisionKernels.range_cosine_mixed_512": rangeHelper(512, v512, h512, run: {
                MK.range_cosine_mixed_512(query: $0, candidatesFP16: $1, range: $2, out: $3)
            }),
            "MixedPrecisionKernels.range_cosine_mixed_768": rangeHelper(768, v768, h768, run: {
                MK.range_cosine_mixed_768(query: $0, candidatesFP16: $1, range: $2, out: $3)
            }),
            "MixedPrecisionKernels.range_cosine_mixed_1536": rangeHelper(1536, v1536, h1536, run: {
                MK.range_cosine_mixed_1536(query: $0, candidatesFP16: $1, range: $2, out: $3)
            }),
            "MixedPrecisionKernels.range_dot_mixed_512": rangeHelper(512, v512, h512, run: {
                MK.range_dot_mixed_512(query: $0, candidatesFP16: $1, range: $2, out: $3)
            }),
            "MixedPrecisionKernels.range_dot_mixed_768": rangeHelper(768, v768, h768, run: {
                MK.range_dot_mixed_768(query: $0, candidatesFP16: $1, range: $2, out: $3)
            }),
            "MixedPrecisionKernels.range_dot_mixed_1536": rangeHelper(1536, v1536, h1536, run: {
                MK.range_dot_mixed_1536(query: $0, candidatesFP16: $1, range: $2, out: $3)
            })
        ]
    }

    private static func registry() -> [String: RangeHelper] {
        batchKernels().merging(mixedPrecision()) { first, _ in first }
    }

    @Test func everyListedHelperHasAnEvaluator() {
        let names = Self.publicRangeHelpers + Self.internalRangeHelpers
        #expect(Set(names).count == names.count, "duplicate helper names")
        #expect(Set(Self.registry().keys) == Set(names))
        #expect(Self.publicRangeHelpers.count == 21)
    }

    @Test("Pair slots and odd tails score a candidate with the same bits",
          arguments: publicRangeHelpers + internalRangeHelpers)
    func pairSlotsEqualTheOddTail(_ name: String) throws {
        let helper = try #require(Self.registry()[name], "\(name) has no evaluator")
        var random = P01OpsRandom(state: 0xF111_0000 &+ UInt64(helper.width))
        let filler = (0..<helper.width).map { _ in random.unit() }
        var mismatches: [String] = []
        var compared = 0
        for fixture in closureFixtures(helper.width) {
            let values = try helper.positions(fixture.query, fixture.candidate, filler)
            let single = values[0]
            compared += values.count - 1
            for (position, value) in values.enumerated().dropFirst()
            where !(value.bitPattern == single.bitPattern || (value.isNaN && single.isNaN)) {
                mismatches.append("\(fixture.name) position \(position): \(t14Hex(value)) vs single \(t14Hex(single))")
            }
        }
        #expect(compared > 0)
        #expect(mismatches.isEmpty, "\(name): \(mismatches.count) of \(compared) positions differ: \(mismatches.prefix(6))")
    }

    /// The chunked parallel helpers of `BatchOperations` (`computeDistances_*_parallel(chunkSize:)`) give every candidate
    /// the bits of the whole-range helper for any chunk size, so the default chunking (from the core count and the
    /// configuration) cannot change a score. Near-duplicate rows, 47 of them, so most chunk sizes leave odd tails.
    @Test("Chunked parallel helpers do not depend on the chunk size", arguments: [512, 768, 1536])
    func chunkedHelpersDoNotDependOnChunkSize(_ dimension: Int) async throws {
        var random = P01OpsRandom(state: 0xC4C4_0000 &+ UInt64(dimension))
        let base = (0..<dimension).map { _ in random.unit() }
        let rows = (0..<47).map { _ in base.map { $0 + random.unit() * 1e-3 } }
        var mismatches: [String] = []
        func whole<V>(_ query: V, _ candidates: [V], _ run: (V, [V], Range<Int>, UnsafeMutableBufferPointer<Float>) -> Void) -> [Float] {
            var out = [Float](repeating: .nan, count: candidates.count)
            return out.withUnsafeMutableBufferPointer { buffer -> [Float] in
                run(query, candidates, 0..<candidates.count, buffer)
                return Array(buffer)
            }
        }
        func compare(_ label: String, _ chunked: [Float], _ reference: [Float]) {
            let differing = zip(chunked, reference).filter { $0.bitPattern != $1.bitPattern }.count
            if chunked.count != reference.count || differing > 0 {
                mismatches.append("\(label): \(differing) of \(reference.count) scores differ")
            }
        }
        for chunk in [1, 2, 3, 5, 7, 16, 47] {
            switch dimension {
            case 512:
                let q = try Vector512Optimized(base), c = try rows.map { try Vector512Optimized($0) }
                compare("cosine chunk \(chunk)", await BatchOperations.computeDistances_cosine_fused_parallel(
                    query: q, candidates: c, dim: 512, chunkSize: chunk, cores: 4),
                        whole(q, c) { BatchKernels.range_cosine_fused_512(query: $0, candidates: $1, range: $2, out: $3) })
                compare("euclid chunk \(chunk)", await BatchOperations.computeDistances_euclid_optimized_parallel(
                    query: q, candidates: c, dim: 512, chunkSize: chunk, cores: 4),
                        whole(q, c) { BatchKernels.range_euclid_512(query: $0, candidates: $1, range: $2, out: $3) })
            case 768:
                let q = try Vector768Optimized(base), c = try rows.map { try Vector768Optimized($0) }
                compare("cosine chunk \(chunk)", await BatchOperations.computeDistances_cosine_fused_parallel(
                    query: q, candidates: c, dim: 768, chunkSize: chunk, cores: 4),
                        whole(q, c) { BatchKernels.range_cosine_fused_768(query: $0, candidates: $1, range: $2, out: $3) })
                compare("euclid chunk \(chunk)", await BatchOperations.computeDistances_euclid_optimized_parallel(
                    query: q, candidates: c, dim: 768, chunkSize: chunk, cores: 4),
                        whole(q, c) { BatchKernels.range_euclid_768(query: $0, candidates: $1, range: $2, out: $3) })
            default:
                let q = try Vector1536Optimized(base), c = try rows.map { try Vector1536Optimized($0) }
                compare("cosine chunk \(chunk)", await BatchOperations.computeDistances_cosine_fused_parallel(
                    query: q, candidates: c, dim: 1536, chunkSize: chunk, cores: 4),
                        whole(q, c) { BatchKernels.range_cosine_fused_1536(query: $0, candidates: $1, range: $2, out: $3) })
                compare("euclid chunk \(chunk)", await BatchOperations.computeDistances_euclid_optimized_parallel(
                    query: q, candidates: c, dim: 1536, chunkSize: chunk, cores: 4),
                        whole(q, c) { BatchKernels.range_euclid_1536(query: $0, candidates: $1, range: $2, out: $3) })
            }
        }
        #expect(mismatches.isEmpty, "D=\(dimension): \(mismatches)")
    }
}
