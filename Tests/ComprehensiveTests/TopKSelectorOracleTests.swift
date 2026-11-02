import Foundation
import Testing
@testable import VectorCore

// P01 phase D: oracle tests for the common Int-indexed selector (`TopKSelection.selectByScore`) and the public
// routes that share it: array `select`, generic `select<T>`, generic `nearest`, `batchNearest` and
// `VectorMath.nearestNeighbors`. Preserved controls: the pointer API's label mapping and the direct dot512
// route's descending policy.
//
// ORACLE. `SelectorOracle` never calls TopKBuffer, `TopKSelection.orderedBefore`/`orderedAscending`, a heap or
// any production selection. It ranks with an explicit key in an insertion sort: class (numeric 0, NaN 1), then
// the direction-adjusted Double value, then the original index (index policies). `literalTable` pins
// hand-written expectations and `oracleAgreesWithLiteralTable` checks the oracle against them. `.smallerValue`
// is checked only through admissible equivalence classes (count, value-class sequence, provenance, no repeated
// pair), never an index choice.
//
// SCORES are compared by exact bit pattern (`sameScore`): a NaN never matches a number, -0 never matches +0 and
// NaN payloads must agree. No tolerance is used anywhere (`bitComparisonRejectsNaNAndSignedZeroMismatches`).

// MARK: - Fixtures and oracle

private let nanA = Float(nan: 0x11, signaling: false)
private let nanB = Float(nan: 0x22, signaling: false)
private let nanC = Float(nan: 0x33, signaling: false)
private let allPolicies: [TieBreaker] = [.smallerIndex, .insertionOrder, .smallerValue]

/// Exact score identity (bit pattern).
private func sameScore(_ a: Float, _ b: Float) -> Bool { a.bitPattern == b.bitPattern }

private enum SelectorOracle {
    /// True when candidate `a` ranks strictly before `b`.
    static func before(_ a: (index: Int, score: Float), _ b: (index: Int, score: Float),
                       descending: Bool, byIndex: Bool) -> Bool {
        let aClass = a.score.isNaN ? 1 : 0
        let bClass = b.score.isNaN ? 1 : 0
        if aClass != bClass { return aClass < bClass }
        if aClass == 0 {
            let x = descending ? -Double(a.score) : Double(a.score)
            let y = descending ? -Double(b.score) : Double(b.score)
            if x < y { return true }
            if y < x { return false }
        }
        return byIndex && a.index < b.index
    }

    /// The complete best-first index order under the index policies (insertion sort, explicit key).
    static func order(_ scores: [Float], descending: Bool = false) -> [Int] {
        var sorted: [Int] = []
        sorted.reserveCapacity(scores.count)
        for index in scores.indices {
            var position = sorted.count
            while position > 0,
                  before((index, scores[index]), (sorted[position - 1], scores[sorted[position - 1]]),
                         descending: descending, byIndex: true) {
                position -= 1
            }
            sorted.insert(index, at: position)
        }
        return sorted
    }

    static func prefix(_ order: [Int], k: Int) -> [Int] { k > 0 ? Array(order.prefix(k)) : [] }

    /// `.smallerValue` equivalence class: nil for NaN, otherwise the value with the zero sign dropped.
    static func valueClass(_ value: Float) -> Double? {
        if value.isNaN { return nil }
        return value == 0 ? 0 : Double(value)
    }
}

/// Hand-written expectations (no computation): best-first indices for the index policies.
private struct LiteralRow: Sendable {
    let scores: [Float]
    let k: Int
    let ascending: [Int]
    let descending: [Int]
}

private let literalTable: [LiteralRow] = [
    LiteralRow(scores: [], k: 3, ascending: [], descending: []),
    LiteralRow(scores: [5], k: 1, ascending: [0], descending: [0]),
    LiteralRow(scores: [5], k: 0, ascending: [], descending: []),
    LiteralRow(scores: [nanA, 1, 0], k: 3, ascending: [2, 1, 0], descending: [1, 2, 0]),
    LiteralRow(scores: [nanA, 2, 1, nanB, 0], k: 5, ascending: [4, 2, 1, 0, 3], descending: [1, 2, 4, 0, 3]),
    LiteralRow(scores: [1, nanA, 0], k: 2, ascending: [2, 0], descending: [0, 2]),
    LiteralRow(scores: [nanA, 5, nanB], k: 3, ascending: [1, 0, 2], descending: [1, 0, 2]),
    LiteralRow(scores: [0, -0.0, 0], k: 3, ascending: [0, 1, 2], descending: [0, 1, 2]),
    LiteralRow(scores: [.infinity, -.infinity, nanA, 0], k: 4, ascending: [1, 3, 0, 2], descending: [0, 3, 1, 2]),
    LiteralRow(scores: [nanA, nanB, nanC], k: 2, ascending: [0, 1], descending: [0, 1]),
    LiteralRow(scores: [3, 1, 3, 1], k: 3, ascending: [1, 3, 0], descending: [0, 2, 1]),
    LiteralRow(scores: [2, 1, 2, 2], k: 2, ascending: [1, 0], descending: [0, 2]),
    LiteralRow(scores: [-0.0, 0, -1], k: 2, ascending: [2, 0], descending: [0, 1]),
    LiteralRow(scores: [nanA, .infinity, nanB, .infinity], k: 3, ascending: [1, 3, 0], descending: [1, 3, 0]),
    LiteralRow(scores: [4, nanA, 4], k: 7, ascending: [0, 2, 1], descending: [0, 2, 1])
]

/// Records at most 25 issues per test, plus the total.
private final class Failures {
    private(set) var count = 0
    func record(_ text: String) {
        count += 1
        if count <= 25 { Issue.record(Comment(rawValue: text)) }
    }
}

// MARK: - Routes under test

private struct Scored {
    let index: Int
    let score: Float
}

/// Spy witness: scores candidate `Int(b[0])` from the fixture and logs the index; never Euclidean.
private struct SpyMetric: DistanceMetric {
    let scores: [Float]
    let log: P01Log
    var name: String { "p01-selector-spy" }
    func distance<V: VectorProtocol>(_ a: V, _ b: V) -> Float where V.Scalar == Float {
        let index = Int(b.toArray()[0])
        log.append(index)
        return scores[index]
    }
}

private func candidates2(_ count: Int) -> [Vector<Dim2>] {
    (0..<count).map { Vector<Dim2>(x: Float($0), y: 0) }
}

private struct Route {
    let name: String
    let policies: [TieBreaker]
    /// Score callbacks per candidate for positive k and nonempty input (0: the API takes no callback).
    let passes: Int
    let run: (_ scores: [Float], _ k: Int, _ policy: TieBreaker, _ log: P01Log) -> [(index: Int, score: Float)]
}

private func pairs(_ result: TopKResult) -> [(index: Int, score: Float)] {
    zip(result.indices, result.distances).map { (index: $0, score: $1) }
}

private var routes: [Route] {
    [
        Route(name: "selectByScore", policies: allPolicies, passes: 1) { scores, k, policy, log in
            TopKSelection.selectByScore(count: scores.count, k: k, tieBreaker: policy) { index in
                log.append(index)
                return scores[index]
            }.map { (index: $0.index, score: $0.distance) }
        },
        Route(name: "select([Float])", policies: allPolicies, passes: 0) { scores, k, policy, _ in
            pairs(TopKSelection.select(k: k, from: scores, tieBreaker: policy))
        },
        Route(name: "select<T>", policies: allPolicies, passes: 1) { scores, k, policy, log in
            let elements = scores.enumerated().map { Scored(index: $0.offset, score: $0.element) }
            let selected = TopKSelection.select(k: k, from: elements, distance: { element in
                log.append(element.index)
                return element.score
            }, tieBreaker: policy)
            return selected.map { (index: $0.index, score: $0.score) }
        },
        Route(name: "nearest", policies: [.smallerIndex], passes: 1) { scores, k, _, log in
            pairs(TopKSelection.nearest(k: k, query: Vector<Dim2>(x: -1, y: 0), candidates: candidates2(scores.count),
                                        metric: SpyMetric(scores: scores, log: log)))
        },
        Route(name: "batchNearest", policies: [.smallerIndex], passes: 2) { scores, k, _, log in
            let queries = [Vector<Dim2>(x: -1, y: 0), Vector<Dim2>(x: -2, y: 0)]
            let results = TopKSelection.batchNearest(k: k, queries: queries, candidates: candidates2(scores.count),
                                                     metric: SpyMetric(scores: scores, log: log))
            guard results.count == 2 else {
                Issue.record("batchNearest returned \(results.count) results for 2 queries")
                return []
            }
            if results[0].indices != results[1].indices
                || results[0].distances.map(\.bitPattern) != results[1].distances.map(\.bitPattern) {
                Issue.record("batchNearest results differ between queries with identical scores")
            }
            return pairs(results[0])
        },
        Route(name: "VectorMath.nearestNeighbors", policies: [.smallerIndex], passes: 1) { scores, k, _, log in
            VectorMath.nearestNeighbors(query: Vector<Dim2>(x: -1, y: 0), in: candidates2(scores.count), k: k) { _, candidate in
                let index = Int(candidate[0])
                log.append(index)
                return scores[index]
            }.map { (index: $0.index, score: $0.distance) }
        }
    ]
}

/// Routes that are cheap enough for the largest exhaustive permutation set.
private var scalarRoutes: [Route] { routes.filter { ["selectByScore", "select([Float])", "select<T>"].contains($0.name) } }

/// Checks one call against the oracle, the count law, provenance and the callback law; returns failure texts.
private func violations(_ route: Route, scores: [Float], order: [Int], k: Int, policy: TieBreaker) -> [String] {
    let log = P01Log()
    let actual = route.run(scores, k, policy, log)
    let count = scores.count
    var found: [String] = []
    let expectedCount = k > 0 ? min(k, count) : 0
    if actual.count != expectedCount { found.append("count \(actual.count) != \(expectedCount)") }
    if route.passes > 0 {
        let expectedCalls = (k > 0 && count > 0) ? (0..<route.passes).flatMap { _ in Array(0..<count) } : []
        let calls = log.snapshot()
        if calls != expectedCalls {
            found.append("callbacks \(calls.count) \(Array(calls.prefix(16))) != \(expectedCalls.count) in index order")
        }
    }
    for entry in actual where !(0..<count).contains(entry.index) || !sameScore(entry.score, scores[entry.index]) {
        found.append("entry (\(entry.index), \(entry.score)) is not an input pair with identical bits")
        break
    }
    if Set(actual.map(\.index)).count != actual.count { found.append("repeated index in \(actual.map(\.index))") }
    let expected = SelectorOracle.prefix(order, k: k)
    if policy == .smallerValue {
        let got = actual.map { SelectorOracle.valueClass($0.score) }
        let want = expected.map { SelectorOracle.valueClass(scores[$0]) }
        if got != want { found.append("value classes \(got) != \(want)") }
    } else if actual.map(\.index) != expected {
        found.append("indices \(actual.map(\.index)) != \(expected)")
    }
    return found
}

private func checkAll(_ routeList: [Route], scores: [Float], ks: [Int], failures: Failures, label: String) {
    let order = SelectorOracle.order(scores)
    for route in routeList {
        for policy in route.policies {
            for k in ks {
                let found = violations(route, scores: scores, order: order, k: k, policy: policy)
                if !found.isEmpty {
                    failures.record("\(route.name) \(policy) k=\(k) \(label) scores=\(scores.prefix(24)): \(found)")
                }
            }
        }
    }
}

/// k classes: negative, zero, one, the strategy boundary neighbours, N-1, N, N+1 and Int.max.
private func kClasses(_ count: Int, extra: [Int] = []) -> [Int] {
    var ks: Set<Int> = [Int.min, -1, 0, 1, 2, count - 1, count, count + 1, Int.max]
    ks.formUnion([count / 10 - 1, count / 10, count / 10 + 1])
    ks.formUnion(extra)
    return ks.sorted()
}

/// All orderings of `values` (Heap's algorithm; repeated values give repeated sequences, which is harmless).
private func permutations(_ values: [Float]) -> [[Float]] {
    var result: [[Float]] = []
    var current = values
    func generate(_ size: Int) {
        if size <= 1 {
            result.append(current)
            return
        }
        generate(size - 1)
        for i in 0..<(size - 1) {
            current.swapAt(size % 2 == 0 ? i : 0, size - 1)
            generate(size - 1)
        }
    }
    generate(current.count)
    return result
}

private struct LCG {
    var state: UInt64
    mutating func next(_ bound: Int) -> Int {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return Int((state >> 33) % UInt64(bound))
    }
}

/// Score generators for the strategy-neighbour fixtures.
private func generatedScores(_ count: Int, kind: Int, seed: UInt64) -> [Float] {
    let pool: [Float] = [0, -0.0, 1, 1, 2, -1, .infinity, -.infinity, nanA, nanB, 3.5, 1, 0.5, -0.0]
    var rng = LCG(state: seed &+ UInt64(count) &* 31 &+ UInt64(kind))
    switch kind {
    case 0:  // mixed pool, NaN forced first, middle and last
        var scores = (0..<count).map { _ in pool[rng.next(pool.count)] }
        if count > 0 { scores[0] = nanC; scores[count / 2] = nanA; scores[count - 1] = nanB }
        return scores
    case 1:  // all NaN (with distinct payloads)
        return (0..<count).map { Float(nan: UInt32(0x100 + $0 % 64), signaling: false) }
    case 2:  // all equal
        return Array(repeating: 7, count: count)
    case 3:  // signed zeros only
        return (0..<count).map { $0 % 3 == 1 ? 0 : -0.0 }
    case 4:  // infinities, NaN and a few numerics
        return (0..<count).map { [Float.infinity, -.infinity, nanA, 1, -.infinity][($0 * 7 + 3) % 5] }
    default:  // distinct-ish numerics with ties, no NaN
        return (0..<count).map { _ in Float(rng.next(9)) - 4 }
    }
}

// MARK: - Suite

@Suite("Top-K selector oracle (P01 phase D)")
struct TopKSelectorOracleTests {

    // MARK: Oracle and comparison self-checks

    @Test func oracleAgreesWithLiteralTable() {
        for row in literalTable {
            #expect(SelectorOracle.prefix(SelectorOracle.order(row.scores), k: row.k) == row.ascending, "\(row.scores)")
            #expect(SelectorOracle.prefix(SelectorOracle.order(row.scores, descending: true), k: row.k) == row.descending,
                    "\(row.scores)")
        }
    }

    @Test func bitComparisonRejectsNaNAndSignedZeroMismatches() {
        #expect(!sameScore(.nan, 0))
        #expect(!sameScore(0, .nan))
        #expect(!sameScore(nanA, nanB), "distinct NaN payloads must not match")
        #expect(!sameScore(0, -0.0))
        #expect(!sameScore(.infinity, .greatestFiniteMagnitude))
        #expect(sameScore(nanA, nanA) && sameScore(-0.0, -0.0))
        #expect(SelectorOracle.valueClass(.nan) == nil && SelectorOracle.valueClass(0) == SelectorOracle.valueClass(-0.0))
    }

    /// Pins the unchanged array/generic strategy predicate `min(k, n) < n / 10` at its boundaries.
    @Test func strategyPredicateBoundaries() {
        let small: [(Int, Int)] = [(20, 1), (100, 9), (109, 9), (110, 10), (1000, 99)]
        let large: [(Int, Int)] = [(19, 1), (20, 2), (100, 10), (109, 10), (110, 11), (1000, 100), (1, 1), (10, 1)]
        for (count, k) in small { #expect(TopKSelection.usesSmallKStrategy(actualK: k, count: count), "N=\(count) k=\(k)") }
        for (count, k) in large { #expect(!TopKSelection.usesSmallKStrategy(actualK: k, count: count), "N=\(count) k=\(k)") }
    }

    // MARK: Literal table across routes

    @Test func literalTableAcrossRoutes() {
        let failures = Failures()
        for row in literalTable {
            for route in routes {
                for policy in route.policies {
                    let log = P01Log()
                    let actual = route.run(row.scores, row.k, policy, log)
                    if policy == .smallerValue {
                        let want = row.ascending.map { SelectorOracle.valueClass(row.scores[$0]) }
                        if actual.map({ SelectorOracle.valueClass($0.score) }) != want {
                            failures.record("\(route.name) smallerValue \(row.scores): classes differ from the table")
                        }
                    } else if actual.map(\.index) != row.ascending {
                        failures.record("\(route.name) \(policy) \(row.scores) k=\(row.k): \(actual.map(\.index)) != \(row.ascending)")
                    }
                    let provenance = actual.allSatisfy { entry in
                        row.scores.indices.contains(entry.index) && sameScore(entry.score, row.scores[entry.index])
                    }
                    if !provenance {
                        failures.record("\(route.name) \(policy) \(row.scores): entries are not input pairs with identical bits")
                    }
                }
            }
        }
        #expect(failures.count == 0, "\(failures.count) literal-table failures")
    }

    // MARK: Exhaustive small permutations

    /// Every ordering of small multisets (NaN payloads, signed zeros, infinities, ties) and every k class.
    /// These sizes always take the large-k branch (n / 10 <= 0); the small-k branch is exhausted below.
    @Test func exhaustivePermutationsOfSmallInputs() {
        let failures = Failures()
        let multisets: [[Float]] = [
            [], [3], [nanA], [-0.0],
            [0, -0.0], [nanA, 1],
            [nanA, -0.0, 0, 1, nanB, -.infinity],
            [0, 1, nanB, -.infinity, 1, .infinity],
            [2, 2, 2, nanA, nanB],
            [-0.0, 0, -0.0, 0],
            [nanA, nanB, nanC]
        ]
        var cases = 0
        for multiset in multisets {
            for scores in permutations(multiset) {
                checkAll(routes, scores: scores, ks: kClasses(scores.count, extra: Array(0...(scores.count + 1))),
                         failures: failures, label: "perm")
                cases += 1
            }
        }
        // One seven-element multiset over the scalar routes only (5040 orderings).
        for scores in permutations([3, 1, nanA, .infinity, -.infinity, -0.0, 1]) {
            checkAll(scalarRoutes, scores: scores, ks: [Int.min, -1, 0, 1, 2, 3, 6, 7, 8, Int.max],
                     failures: failures, label: "perm7")
            cases += 1
        }
        #expect(cases > 6000, "ran \(cases) orderings")
        #expect(failures.count == 0, "\(failures.count) permutation failures")
    }

    /// Every ordering of six contested values embedded at first, middle and last positions of inputs that take
    /// the small-k branch, with tied numeric fillers (ties across the k boundary) or NaN fillers (NaN fill).
    @Test(arguments: [(20, 1), (30, 2), (70, 6)])
    func exhaustivePermutationsOnTheSmallKBranch(count: Int, k: Int) {
        #expect(TopKSelection.usesSmallKStrategy(actualK: k, count: count))
        let failures = Failures()
        let positions = [0, count / 5, count / 2, count / 2 + 1, count - 2, count - 1]
        for filler in [Float(0.5), Float(nan: 0x77, signaling: false)] {
            for contested in permutations([nanA, -0.0, 0, -.infinity, 1, .infinity]) {
                var scores = Array(repeating: filler, count: count)
                for (position, value) in zip(positions, contested) { scores[position] = value }
                checkAll(routes, scores: scores, ks: [k], failures: failures, label: "filler=\(filler)")
            }
        }
        #expect(failures.count == 0, "\(failures.count) small-k permutation failures (N=\(count) k=\(k))")
    }

    // MARK: Strategy neighbours and k classes

    @Test(arguments: [(20, 1), (20, 2), (100, 9), (100, 10), (109, 9), (109, 10), (110, 10), (110, 11),
                      (1000, 99), (1000, 100)])
    func strategyNeighboursAgainstOracle(count: Int, k: Int) {
        let failures = Failures()
        for kind in 0...5 {
            for seed: UInt64 in [1, 2] where seed == 1 || kind == 0 || kind == 5 {
                let scores = generatedScores(count, kind: kind, seed: seed)
                checkAll(routes, scores: scores, ks: kClasses(count, extra: [k]), failures: failures,
                         label: "N=\(count) kind=\(kind) seed=\(seed)")
            }
        }
        #expect(failures.count == 0, "\(failures.count) strategy-neighbour failures (N=\(count) k=\(k))")
    }

    /// Ties that straddle the k boundary on both branches, and runs of equal values across the 256-candidate
    /// boundary used by the tiled batch paths (index 255/256) and across index 511/512.
    @Test func tiesAcrossChunkAndCapacityBoundaries() {
        let failures = Failures()
        var straddle = Array(repeating: Float(10), count: 600)
        for index in 250..<262 { straddle[index] = 1 }
        for index in 508..<516 { straddle[index] = -0.0 }
        straddle[0] = nanA
        straddle[599] = nanB
        checkAll(routes, scores: straddle, ks: [1, 5, 8, 9, 12, 19, 20, 21, 59, 60, 61, 598, 599, 600, 601],
                 failures: failures, label: "straddle")
        let periodic = (0..<1000).map { Float(($0 / 7) % 3) }
        checkAll(routes, scores: periodic, ks: [1, 7, 50, 99, 100, 334, 335, 999], failures: failures, label: "periodic")
        #expect(failures.count == 0, "\(failures.count) tie-boundary failures")
    }

    @Test func emptyAndSingletonInputs() {
        let failures = Failures()
        for scores: [Float] in [[], [nanA], [-0.0], [0], [.infinity], [-.infinity], [42]] {
            checkAll(routes, scores: scores, ks: [Int.min, -1, 0, 1, 2, Int.max], failures: failures, label: "tiny")
        }
        #expect(failures.count == 0)
    }

    // MARK: Callback law

    /// Score callbacks run once per candidate in index order (NaN and losing values included) on both
    /// branches, and never for k <= 0 or empty input. The batch route runs one pass per query.
    @Test func callbacksRunOncePerCandidateInIndexOrder() {
        var scores = (0..<50).map { Float(50 - $0) }
        scores[3] = nanA
        scores[49] = .infinity
        for route in routes where route.passes > 0 {
            for k in [1, 4, 5, 49, 50, 51, Int.max] {  // 1 and 4 take the small-k branch (N / 10 = 5)
                let log = P01Log()
                _ = route.run(scores, k, .smallerIndex, log)
                let expected = (0..<route.passes).flatMap { _ in Array(0..<50) }
                #expect(log.snapshot() == expected, "\(route.name) k=\(k): \(log.count) calls")
            }
            for k in [0, -1, Int.min] {
                let log = P01Log()
                #expect(route.run(scores, k, .smallerIndex, log).isEmpty)
                #expect(log.count == 0, "\(route.name) k=\(k) called the score callback")
            }
            let log = P01Log()
            #expect(route.run([], 3, .smallerIndex, log).isEmpty)
            #expect(log.count == 0, "\(route.name) empty input called the score callback")
        }
    }

    // MARK: Preserved controls

    /// The pointer API (unchanged) maps labels after selection: duplicate, negative and nonmonotonic labels
    /// never affect admission or order. Compared with the oracle and with the Int-indexed array API.
    @Test func pointerLabelMappingControl() {
        let failures = Failures()
        let fixtures: [([Float], [Int])] = [
            (generatedScores(20, kind: 0, seed: 1), [1, 2, 5, 19, 20, 21]),
            (generatedScores(100, kind: 0, seed: 2), [1, 9, 10, 99, 100]),
            ([nanA, 1, 0, -0.0, 1, .infinity, -.infinity], [1, 3, 7, 8])
        ]
        for (scores, ks) in fixtures {
            let labels: [Int32] = scores.indices.map { Int32(($0 * 37 + 11) % 5) - 2 }
            let order = SelectorOracle.order(scores)
            for policy in allPolicies {
                for k in ks {
                    let pointer = scores.withUnsafeBufferPointer { values in
                        labels.withUnsafeBufferPointer { ids in
                            TopKSelection.select(k: k, from: values.baseAddress!, count: values.count,
                                                 ids: ids.baseAddress!, tieBreaker: policy)
                        }
                    }
                    let array = TopKSelection.select(k: k, from: scores, tieBreaker: policy)
                    let expected = SelectorOracle.prefix(order, k: k)
                    if pointer.indices.count != expected.count || array.count != expected.count {
                        failures.record("counts pointer=\(pointer.indices.count) array=\(array.count) want=\(expected.count)")
                        continue
                    }
                    if policy == .smallerValue {
                        let want = expected.map { SelectorOracle.valueClass(scores[$0]) }
                        if pointer.distances.map(SelectorOracle.valueClass) != want {
                            failures.record("pointer smallerValue classes differ k=\(k)")
                        }
                    } else {
                        if pointer.indices != expected.map({ labels[$0] }) {
                            failures.record("pointer \(policy) k=\(k): labels \(pointer.indices) != \(expected.map { labels[$0] })")
                        }
                        if pointer.indices != array.indices.map({ labels[$0] }) {
                            failures.record("pointer \(policy) k=\(k) disagrees with the Int-indexed array API")
                        }
                        if !zip(pointer.distances, expected).allSatisfy({ sameScore($0, scores[$1]) }) {
                            failures.record("pointer \(policy) k=\(k): score bits")
                        }
                    }
                }
            }
        }
        #expect(failures.count == 0)
    }

    /// The direct optimized dot route keeps its descending policy (numeric descending, NaN last, smaller index).
    /// Candidate i is s_i * e0 and the query is e0, so the dot product is s_i exactly (no -0 in these fixtures).
    @Test func directDotKeepsDescendingPolicyControl() throws {
        let failures = Failures()
        var rows: [([Float], [Int])] = literalTable
            .filter { row in !row.scores.contains { value in value == 0 && value.sign == .minus } }
            .map { row in (row.scores, [row.k]) }
        rows.append((generatedScores(40, kind: 4, seed: 1), [1, 3, 4, 39, 40, 41]))
        var query = [Float](repeating: 0, count: 512)
        query[0] = 1
        let q = try Vector512Optimized(query)
        for (scores, ks) in rows {
            let candidates = try scores.map { value -> Vector512Optimized in
                var lanes = [Float](repeating: 0, count: 512)
                lanes[0] = value
                return try Vector512Optimized(lanes)
            }
            let order = SelectorOracle.order(scores, descending: true)
            for k in ks + [0, -1, Int.max] {
                let result = TopKSelection.nearestDotProduct512(k: k, query: q, candidates: candidates)
                let expected = SelectorOracle.prefix(order, k: k)
                if result.indices != expected {
                    failures.record("dot512 \(scores.prefix(12)) k=\(k): \(result.indices) != \(expected)")
                }
                for (index, score) in zip(result.indices, result.distances) {
                    guard scores.indices.contains(index) else {
                        failures.record("dot512 returned index \(index) outside the input")
                        continue
                    }
                    let reference = scores[index]
                    if reference.isNaN ? !score.isNaN : !sameScore(score, reference) {
                        failures.record("dot512 score for \(index): \(score) != \(reference)")
                    }
                }
            }
        }
        #expect(failures.count == 0)
    }

    /// Generic distance witnesses stay authoritative on optimized types: a custom metric whose order is the
    /// reverse of Euclidean decides the result, and its witness runs exactly once per candidate.
    @Test func genericNearestUsesTheSuppliedWitnessOnOptimizedTypes() throws {
        let count = 30
        let candidates = try (0..<count).map { index -> Vector512Optimized in
            var lanes = [Float](repeating: 0, count: 512)
            lanes[0] = Float(index)
            return try Vector512Optimized(lanes)
        }
        let query = try Vector512Optimized([Float](repeating: 0, count: 512))
        let scores = (0..<count).map { Float(count - $0) }  // reverse of the Euclidean order
        for k in [1, 2, 3, 29, 30, 31] {
            let log = P01Log()
            let custom = TopKSelection.nearest(k: k, query: query, candidates: candidates,
                                               metric: SpyMetric(scores: scores, log: log))
            let expected = SelectorOracle.prefix(SelectorOracle.order(scores), k: k)
            #expect(custom.indices == expected, "custom witness k=\(k)")
            #expect(log.snapshot() == Array(0..<count), "custom witness calls k=\(k)")
            let euclidean = TopKSelection.nearest(k: k, query: query, candidates: candidates, metric: EuclideanDistance())
            #expect(euclidean.indices == Array(0..<min(k, count)), "Euclidean control k=\(k)")
        }
    }
}
