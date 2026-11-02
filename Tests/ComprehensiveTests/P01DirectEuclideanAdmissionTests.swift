import Foundation
import Testing
@testable import VectorCore

// P01 final fix (I1): the seven direct optimized Euclidean routes, `TopKSelection.nearestEuclidean{384,512,768,1536}`
// and `Operations.findNearest` with `EuclideanDistance` on `Vector{512,768,1536}Optimized` and `k < n`, admit
// candidates by the SQUARED distance of `EuclideanKernels.squared{D}` and return its square root
// (`squaredAdmissionAtRootTiedBoundaries`). Two candidates whose squared distances differ but whose Float roots are
// equal therefore resolve at the k boundary by the squared value; an admission by the root would tie them and keep
// the smaller index. Batch Euclidean admits by the root (`P01BatchRouteTests.rootTieAdmission`); that contract is
// not asserted here.

/// One direct width: a row constructor, the routes' own squared kernel and the public routes under test.
struct P01DirectWidth<O: Sendable>: Sendable {
    let dim: Int
    let make: @Sendable ([Float]) throws -> O
    /// `EuclideanKernels.squared{D}`, the score both direct routes select on.
    let squared: @Sendable (O, O) -> Float
    /// `TopKSelection.nearestEuclidean{D}`.
    let nearest: @Sendable (Int, O, [O]) -> TopKResult
    /// `Operations.findNearest(to:in:k:)` with its default `EuclideanDistance`; nil for 384, which has no direct
    /// Operations route.
    let operations: (@Sendable (Int, O, [O]) async throws -> [NearestNeighborResult])?
}

let p01Direct384 = P01DirectWidth<Vector384Optimized>(
    dim: 384, make: { try Vector384Optimized($0) }, squared: { EuclideanKernels.squared384($0, $1) },
    nearest: { TopKSelection.nearestEuclidean384(k: $0, query: $1, candidates: $2) }, operations: nil)

let p01Direct512 = P01DirectWidth<Vector512Optimized>(
    dim: 512, make: { try Vector512Optimized($0) }, squared: { EuclideanKernels.squared512($0, $1) },
    nearest: { TopKSelection.nearestEuclidean512(k: $0, query: $1, candidates: $2) },
    operations: { try await Operations.findNearest(to: $1, in: $2, k: $0) })

let p01Direct768 = P01DirectWidth<Vector768Optimized>(
    dim: 768, make: { try Vector768Optimized($0) }, squared: { EuclideanKernels.squared768($0, $1) },
    nearest: { TopKSelection.nearestEuclidean768(k: $0, query: $1, candidates: $2) },
    operations: { try await Operations.findNearest(to: $1, in: $2, k: $0) })

let p01Direct1536 = P01DirectWidth<Vector1536Optimized>(
    dim: 1536, make: { try Vector1536Optimized($0) }, squared: { EuclideanKernels.squared1536($0, $1) },
    nearest: { TopKSelection.nearestEuclidean1536(k: $0, query: $1, candidates: $2) },
    operations: { try await Operations.findNearest(to: $1, in: $2, k: $0) })

/// The method of `p01RootTiePair` (P01BatchFixtures.swift), run through the direct routes' kernel: with query 0,
/// `a = [1, 0, ...]` and `b = [1, j·2^-14, 0, ...]` for j = 1, 2, ...; returns the first `b` whose squared distance
/// is larger than `a`'s while the two Float roots are bit-equal, and that `j`.
func p01DirectRootTiePair<O>(_ width: P01DirectWidth<O>) throws -> (b: [Float], a: [Float], j: Int) {
    let zero = try width.make([Float](repeating: 0, count: width.dim))
    var a = [Float](repeating: 0, count: width.dim)
    a[0] = 1
    let squaredA = width.squared(zero, try width.make(a))
    for j in 1...4_096 {
        var b = a
        b[1] = Float(j) * 0x1p-14
        let squaredB = width.squared(zero, try width.make(b))
        if squaredA < squaredB, squaredA.squareRoot().bitPattern == squaredB.squareRoot().bitPattern {
            return (b, a, j)
        }
    }
    throw NSError(domain: "P01.directRootTieSearch", code: width.dim)
}

func p01TopKEntries(_ result: TopKResult) -> [P01OpsEntry] {
    zip(result.indices, result.distances).map { P01OpsEntry(index: $0, bits: $1.bitPattern) }
}

@Suite("P01 direct Euclidean squared admission (final fix I1)")
struct P01DirectEuclideanAdmissionTests {
    static let count = 600
    /// Where each root-tied pair sits (`b` at `slot`, `a` at `slot + 1`) and its power-of-two scale. Slot 6
    /// straddles a range boundary of the chunks-of-7 plan; slots 255, 299 and 511 are placed to straddle the CPU
    /// parallel provider's partitions. The test asserts that each partitioned provider split at least one pair.
    static let slots: [(slot: Int, scale: Float)] = [(6, 4), (40, 0.125), (130, 8), (255, 1), (299, 2), (511, 0.5)]

    /// At every k boundary that falls between the two members of a root-tied pair, each direct route keeps the
    /// member with the smaller squared distance (the higher index), exactly as the squared-admission oracle does,
    /// index for index and bit for bit, while root admission would keep the other member. One past each boundary
    /// both members are kept and the output order (root, then index) is checked too. Operations runs under the CPU
    /// sequential and parallel providers and two partition plans, each wrapped in a recording provider.
    @Test(arguments: [384, 512, 768, 1536])
    func squaredAdmissionAtRootTiedBoundaries(dim: Int) async throws {
        switch dim {
        case 384: try await check(p01Direct384)
        case 512: try await check(p01Direct512)
        case 768: try await check(p01Direct768)
        default: try await check(p01Direct1536)
        }
    }

    private func check<O: Sendable>(_ width: P01DirectWidth<O>) async throws {
        let n = Self.count, dim = width.dim
        let (b, a, j) = try p01DirectRootTiePair(width)
        var random = P01OpsRandom(state: 0xD1 &+ UInt64(dim))
        // Random rows of expected norm about 2^((i mod 13) / 2 - 2), interleaved with the pairs' scales 0.125...8.
        var raw: [[Float]] = (0..<n).map { i in
            let factor = Float(pow(2, Double(i % 13) / 2 - 2)) / (Float(dim) / 3).squareRoot()
            return (0..<dim).map { _ in random.unit() * factor }
        }
        var pairs: [(b: Int, a: Int)] = []
        for (slot, scale) in Self.slots {
            raw[slot] = b.map { $0 * scale }
            raw[slot + 1] = a.map { $0 * scale }
            pairs.append((slot, slot + 1))
        }
        let query = try width.make([Float](repeating: 0, count: dim))
        let rows = try raw.map(width.make)
        let squared = rows.map { width.squared(query, $0) }
        let roots = squared.map { $0.squareRoot() }

        // Fixture precondition, on the built rows: every pair has distinct squared distances (a's smaller) and
        // bit-equal roots, and b directly follows a in the squared order, so k = rank(a) + 1 splits the pair.
        let order = squared.indices.sorted { P01OpsOracle.before(($0, squared[$0]), ($1, squared[$1])) }
        var boundaries: [(k: Int, b: Int, a: Int)] = []
        for pair in pairs {
            try #require(squared[pair.a] < squared[pair.b] && roots[pair.a].bitPattern == roots[pair.b].bitPattern,
                         "dim=\(dim) pair \(pair): squared \(squared[pair.a]) / \(squared[pair.b])")
            let rank = try #require(order.firstIndex(of: pair.a))
            try #require(rank + 2 < n && order[rank + 1] == pair.b, "dim=\(dim) pair \(pair) rank \(rank)")
            boundaries.append((rank + 1, pair.b, pair.a))
        }
        print("P01_DIRECT_TIE dim=\(dim) j=\(j) a2=0x\(String(squared[pairs[0].a].bitPattern, radix: 16))"
              + " b2=0x\(String(squared[pairs[0].b].bitPattern, radix: 16))"
              + " root=0x\(String(roots[pairs[0].a].bitPattern, radix: 16)) boundaries=\(boundaries.map { $0.k })")

        for boundary in boundaries {
            for k in [boundary.k, boundary.k + 1] {
                let expected = P01OpsOracle.select(squared, k: k, format: { $0.squareRoot() })
                let rootAdmission = P01OpsOracle.select(roots, k: k)
                let kept = Set(expected.map(\.index)), rootKept = Set(rootAdmission.map(\.index))
                if k == boundary.k {
                    // Discrimination precondition: the two admissions differ here, in members.
                    try #require(kept.contains(boundary.a) && !kept.contains(boundary.b)
                                 && rootKept.contains(boundary.b) && !rootKept.contains(boundary.a)
                                 && expected != rootAdmission, "dim=\(dim) k=\(k): fixture does not split the pair")
                } else {
                    try #require(kept.isSuperset(of: [boundary.a, boundary.b]) && expected == rootAdmission)
                }

                let label = "dim=\(dim) k=\(k) pair=(\(boundary.b),\(boundary.a))"
                let direct = p01TopKEntries(width.nearest(k, query, rows))
                #expect(direct == expected, "nearestEuclidean\(dim) \(label)")
                #expect(Set(direct.map(\.index)).contains(boundary.a), "nearestEuclidean\(dim) \(label): dropped a")

                guard let operations = width.operations else { continue }
                for (name, provider) in Self.providers() {
                    let result = try await Operations.$computeProvider.withValue(provider) {
                        try await operations(k, query, rows)
                    }
                    let entries = p01Entries(result)
                    #expect(entries == expected, "Operations.findNearest/\(dim) \(name) \(label)")
                    #expect(Set(entries.map(\.index)).contains(boundary.a),
                            "Operations.findNearest/\(dim) \(name) \(label): dropped a")
                    // The provider was used: one hint-free reduction with TopKBuffer partials of capacity
                    // min(k, range count) over a cover of 0..<n, combined by the Q5 rule.
                    let ledger = provider.ledger
                    #expect(ledger.reduceWithoutHint == 1 && ledger.reduceHints.isEmpty && ledger.executeEntries == 0,
                            "\(name) \(label): provider calls")
                    p01CheckStream(ledger, calls: nil, count: n, k: k, label: "\(name) \(label)")
                    let ranges = ledger.partials.map(\.range).filter { !$0.isEmpty }
                    let split = pairs.filter { pair in ranges.contains { $0.contains(pair.b) && !$0.contains(pair.a) } }
                    switch name {
                    case "cpu-sequential":
                        #expect(ranges == [0..<n], "\(name) \(label): \(ranges)")
                    case "cpu-parallel" where ProcessInfo.processInfo.activeProcessorCount > 1:
                        #expect(ranges.count > 1 && !split.isEmpty, "\(name) \(label): \(ranges)")
                    case "plan-singletons-balanced", "plan-chunks-of-7-right":
                        #expect(!split.isEmpty, "\(name) \(label): \(ranges.count) ranges")
                    default:
                        break
                    }
                }
            }
        }
    }

    /// Fresh recording providers (each ledger observes one search).
    private static func providers() -> [(String, P01RecordingProvider)] {
        func plan(_ name: String) -> @Sendable (Range<Int>) -> [Range<Int>] {
            P01PlanProvider.plans.first { $0.0 == name }!.1
        }
        return [
            ("cpu-sequential", P01RecordingProvider(inner: CPUComputeProvider(mode: .sequential))),
            ("cpu-parallel", P01RecordingProvider(inner: CPUComputeProvider(mode: .parallel))),
            ("plan-singletons-balanced", P01RecordingProvider(inner: P01PlanProvider(
                label: "singletons", plan: plan("singletons-with-empties"), fold: .balanced))),
            ("plan-chunks-of-7-right", P01RecordingProvider(inner: P01PlanProvider(
                label: "chunks-of-7", plan: plan("chunks-of-7"), fold: .right)))
        ]
    }

    // MARK: TopKSelection widths: independent finite references and exceptional parity

    /// `TopKSelection.nearestEuclidean{D}` (a synchronous API with no provider): finite fixtures agree with an
    /// independent Double reference (index order exactly, distances within a tolerance no NaN can pass), and
    /// complete results on finite and exceptional rows equal the squared-admission oracle over the kernel's own
    /// scores, bit for bit, for k in {1, 5, n/2, n - 1, n, n + 7}. Nonpositive k and empty candidates return [].
    @Test(arguments: [384, 512, 768, 1536])
    func nearestEuclideanMatchesReferences(dim: Int) throws {
        switch dim {
        case 384: try references(p01Direct384, tolerance: 1e-3)
        case 512: try references(p01Direct512, tolerance: 1e-3)
        case 768: try references(p01Direct768, tolerance: 1e-3)
        default: try references(p01Direct1536, tolerance: 2e-3)
        }
    }

    private func references<O: Sendable>(_ width: P01DirectWidth<O>, tolerance: Double) throws {
        let count = 300, dim = width.dim
        for exceptional in [false, true] {
            // Finite seed 8: the gap precondition below holds at all four widths (seed 42 has a near-tie at 384).
            var random = P01OpsRandom(state: (exceptional ? 99 : 8) &+ UInt64(dim))
            let rawQuery = (0..<dim).map { _ in random.unit() }
            let raw: [[Float]] = (0..<count).map { row in
                var v = (0..<dim).map { _ in random.unit() }
                guard exceptional else { return v }
                switch row % 11 {
                case 3: v[1] = .nan
                case 4: v[2] = Float(bitPattern: 0x7FC0_1234)
                case 5: v[3] = .infinity
                case 6: v[4] = -.infinity
                case 7: v = [Float](repeating: 0, count: dim)
                case 8: v = [Float](repeating: -0.0, count: dim)
                case 9: v[5] = 3e38
                case 10: v = rawQuery
                default: break
                }
                return v
            }
            let query = try width.make(rawQuery)
            let rows = try raw.map(width.make)
            let squared = rows.map { width.squared(query, $0) }
            for k in [1, 5, count / 2, count - 1, count, count + 7] {
                let expected = P01OpsOracle.select(squared, k: k, format: { $0.squareRoot() })
                #expect(p01TopKEntries(width.nearest(k, query, rows)) == expected,
                        "dim=\(dim) exceptional=\(exceptional) k=\(k)")
            }
            guard !exceptional else { continue }
            let k = 5
            let reference: [Double] = raw.map { row in
                zip(rawQuery, row).reduce(0.0) { $0 + pow(Double($1.0) - Double($1.1), 2) }.squareRoot()
            }
            let order = reference.indices.sorted {
                reference[$0] < reference[$1] || (reference[$0] == reference[$1] && $0 < $1)
            }
            // Fixture precondition: neighbours in the reference order are more than two tolerances apart.
            let gaps = (0..<k).map { reference[order[$0 + 1]] - reference[order[$0]] }
            #expect(gaps.allSatisfy { $0 > 2 * tolerance }, "dim=\(dim): fixture has near-ties \(gaps)")
            let result = width.nearest(k, query, rows)
            #expect(result.indices == Array(order.prefix(k)), "dim=\(dim)")
            for (index, distance) in zip(result.indices, result.distances) {
                #expect(Double(distance).isFinite && abs(Double(distance) - reference[index]) <= tolerance,
                        "dim=\(dim): \(distance) vs \(reference[index])")
            }
        }
        let query = try width.make([Float](repeating: 0, count: dim))
        let rows = try [[Float](repeating: 1, count: dim)].map(width.make)
        #expect(width.nearest(0, query, rows).isEmpty && width.nearest(-1, query, rows).isEmpty)
        #expect(width.nearest(3, query, []).isEmpty)
    }
}
