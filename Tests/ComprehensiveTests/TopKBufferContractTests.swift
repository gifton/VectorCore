import Foundation
import Testing
@testable import VectorCore

// P01 phase C: TopKBuffer zero-capacity guard and `mergeIn(from:)`.
//
// ORACLE. Expected contents come from `Oracle`, which does not call TopKBuffer, `TopKSelection.orderedBefore`,
// or any heap: it sorts the candidate entries with its own explicit comparator (numeric class first with NaN
// last, then the direction-adjusted Double value, then the index) and takes a prefix. For `.smallerValue`,
// equal numeric values and two NaNs are equivalent classes, so only the sorted value-class sequence and the
// membership of every retained (value, index) pair in the input are asserted, never an index choice.

private struct Entry {
    let value: Float
    let index: Int
}

private enum Oracle {
    /// True when `a` ranks strictly before `b` (better first).
    static func before(_ a: Entry, _ b: Entry, maximize: Bool, byIndex: Bool) -> Bool {
        let aClass = a.value.isNaN ? 1 : 0
        let bClass = b.value.isNaN ? 1 : 0
        if aClass != bClass { return aClass < bClass }
        if aClass == 0 {
            let x = maximize ? -Double(a.value) : Double(a.value)
            let y = maximize ? -Double(b.value) : Double(b.value)
            if x < y { return true }
            if y < x { return false }
        }
        return byIndex && a.index < b.index
    }

    static func best(_ entries: [Entry], capacity: Int, maximize: Bool, policy: TieBreaker) -> [Entry] {
        let byIndex = policy != .smallerValue
        // Insertion sort: tiny inputs, no reliance on a library sort's stability or comparator.
        var sorted: [Entry] = []
        for entry in entries {
            var position = sorted.count
            while position > 0, before(entry, sorted[position - 1], maximize: maximize, byIndex: byIndex) { position -= 1 }
            sorted.insert(entry, at: position)
        }
        return Array(sorted.prefix(capacity))
    }
}

private let policies: [TieBreaker] = [.smallerIndex, .insertionOrder, .smallerValue]

private func live(_ buffer: TopKBuffer) -> [Entry] {
    (0..<buffer.size).map { Entry(value: buffer.vals[$0], index: buffer.idxs[$0]) }
}

private func sameValueClass(_ a: Float, _ b: Float) -> Bool {
    (a.isNaN && b.isNaN) || a == b
}

/// Compares a buffer with the oracle under the policy's notion of equivalence. Returns a failure text or nil.
private func mismatch(_ buffer: TopKBuffer, inputs: [Entry], capacity: Int, maximize: Bool, policy: TieBreaker) -> String? {
    let expected = Oracle.best(inputs, capacity: capacity, maximize: maximize, policy: policy)
    let byIndex = policy != .smallerValue
    let actual = Oracle.best(live(buffer), capacity: .max, maximize: maximize, policy: policy)
    if actual.count != expected.count { return "count \(actual.count) != \(expected.count)" }
    // Multiplicity: inputs carry distinct indices, so a retained index seen twice means one input pair was kept
    // twice (the per-entry presence check below cannot see that, notably for `.smallerValue`).
    if Set(actual.map(\.index)).count != actual.count {
        return "retained indices \(actual.map(\.index)) contain a duplicate"
    }
    for (a, e) in zip(actual, expected) {
        if !sameValueClass(a.value, e.value) { return "value \(a.value)@\(a.index) != \(e.value)@\(e.index)" }
        if byIndex && a.index != e.index { return "index \(a.index) != \(e.index) (value \(a.value))" }
        let present = inputs.contains { input in
            input.index == a.index && (input.value.bitPattern == a.value.bitPattern || (input.value.isNaN && a.value.isNaN))
        }
        guard present else {
            return "retained pair (\(a.value), \(a.index)) is not an input"
        }
    }
    return nil
}

private func sameBits(_ a: TopKBuffer, _ b: TopKBuffer) -> Bool {
    a.k == b.k && a.size == b.size && a.isMinHeap == b.isMinHeap && a.tieBreaker == b.tieBreaker
        && a.vals.map(\.bitPattern) == b.vals.map(\.bitPattern) && a.idxs == b.idxs
}

private func buffer(_ entries: [Entry], capacity: Int, maximize: Bool, policy: TieBreaker) -> TopKBuffer {
    var b = TopKBuffer(k: capacity, isMinHeap: maximize, tieBreaker: policy)
    for entry in entries { b.pushIfBetter(val: entry.value, idx: entry.index) }
    return b
}

private struct LCG {
    var state: UInt64
    mutating func next(_ bound: Int) -> Int {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return Int((state >> 33) % UInt64(bound))
    }
}

private let pool: [Float] = [0, -0.0, 1, 1, 2, -1, .infinity, -.infinity, .nan, .nan, 3.5, 1]

@Suite("TopKBuffer zero capacity and merge-in")
struct TopKBufferContractTests {

    // MARK: Zero capacity

    @Test func zeroCapacityIgnoresPushesWithoutGrowth() {
        for maximize in [false, true] {
            for policy in policies {
                var b = TopKBuffer(k: 0, isMinHeap: maximize, tieBreaker: policy)
                for (i, v) in ([1, .nan, -.infinity, .infinity, 0, -0.0] as [Float]).enumerated() {
                    b.pushIfBetter(val: v, idx: i)
                }
                #expect(b.size == 0 && b.k == 0)
                #expect(b.vals.isEmpty && b.idxs.isEmpty, "arrays must not grow (maximize=\(maximize) \(policy))")
            }
        }
    }

    @Test func mergeIntoAndFromZeroCapacity() {
        for maximize in [false, true] {
            for policy in policies {
                let empty = TopKBuffer(k: 0, isMinHeap: maximize, tieBreaker: policy)
                var zero = empty
                zero.mergeIn(from: empty)
                #expect(sameBits(zero, empty))
                let two = [Entry(value: 2, index: 0), Entry(value: .nan, index: 1)]
                var filled = buffer(two, capacity: 3, maximize: maximize, policy: policy)
                let before = filled
                filled.mergeIn(from: empty)                       // empty source
                #expect(sameBits(filled, before))
                #expect(mismatch(filled, inputs: two, capacity: 3, maximize: maximize, policy: policy) == nil)
                var zeroDest = TopKBuffer(k: 0, isMinHeap: maximize, tieBreaker: policy)
                zeroDest.mergeIn(from: empty)                     // 0 <- 0 is allowed
                #expect(zeroDest.size == 0)
            }
        }
    }

    // MARK: Class matrix against the oracle

    /// Random partitions (including empty ranges), capacities min(k, rangeCount) and full k, destination
    /// capacities from the largest partial up to full k, with and without a pre-seeded destination.
    @Test func mergeInMatchesOracleAcrossPartitionsCapacitiesAndPolicies() {
        var cases = 0
        for policy in policies {
            for maximize in [false, true] {
                for n in 0...9 {
                    for seed in 0..<24 {
                        var rng = LCG(state: UInt64(n * 1000 + seed) &+ (maximize ? 7 : 0))
                        let entries = (0..<n).map { Entry(value: pool[rng.next(pool.count)], index: $0) }
                        var cuts = (0..<rng.next(4)).map { _ in rng.next(n + 1) }
                        cuts.sort()
                        let bounds = [0] + cuts + [n]
                        let ranges = (0..<bounds.count - 1).map { bounds[$0]..<bounds[$0 + 1] }
                        for k in Set([0, 1, 2, n / 2, max(n - 1, 0), n, n + 1]).sorted() {
                            for fullK in [false, true] {
                                // fullK: every partial has capacity K (a clamped-to-count variant is the default)
                                let partials = ranges.map { r in
                                    buffer(entries[r].map { $0 }, capacity: fullK ? k : min(k, r.count), maximize: maximize, policy: policy)
                                }
                                let maxCap = partials.map(\.k).max() ?? 0
                                for destCap in Set([maxCap, k, min(k, partials.reduce(0) { $0 + $1.k })]).sorted() where destCap >= maxCap {
                                    for seeded in [false, true] {
                                        var dest: TopKBuffer
                                        var rest = partials[...]
                                        if seeded, !rest.isEmpty {
                                            dest = buffer(Array(entries[ranges[0]]), capacity: destCap,
                                                          maximize: maximize, policy: policy)
                                            rest = rest.dropFirst()
                                        } else {
                                            dest = TopKBuffer(k: destCap, isMinHeap: maximize, tieBreaker: policy)
                                        }
                                        for p in rest { dest.mergeIn(from: p) }
                                        if let why = mismatch(dest, inputs: entries, capacity: destCap,
                                                              maximize: maximize, policy: policy) {
                                            let head = "\(policy) max=\(maximize) n=\(n) seed=\(seed) k=\(k)"
                                            let tail = "fullK=\(fullK) dest=\(destCap) seeded=\(seeded) ranges=\(ranges)"
                                            Issue.record("\(head) \(tail): \(why)")
                                        }
                                        cases += 1
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        #expect(cases > 10_000, "matrix ran \(cases) cases")
    }

    @Test func differentlySizedPartialsIntoOneDestination() {
        for policy in policies {
            for maximize in [false, true] {
                let entries = (0..<40).map { Entry(value: pool[($0 * 7 + 3) % pool.count] + Float($0 % 5), index: $0) }
                let ranges = [0..<1, 1..<4, 4..<12, 12..<12, 12..<40]
                let caps = [1, 3, 8, 0, 8]
                var dest = TopKBuffer(k: 8, isMinHeap: maximize, tieBreaker: policy)
                for (r, c) in zip(ranges, caps) {
                    dest.mergeIn(from: buffer(entries[r].map { $0 }, capacity: c, maximize: maximize, policy: policy))
                }
                #expect(mismatch(dest, inputs: entries, capacity: 8, maximize: maximize, policy: policy) == nil)
            }
        }
    }

    @Test func repeatedMergesAreStable() {
        for policy in policies {
            for maximize in [false, true] {
                let entries = (0..<12).map { Entry(value: pool[$0 % pool.count], index: $0) }
                let a = buffer(Array(entries[0..<6]), capacity: 4, maximize: maximize, policy: policy)
                let b = buffer(Array(entries[6..<12]), capacity: 4, maximize: maximize, policy: policy)
                var dest = TopKBuffer(k: 4, isMinHeap: maximize, tieBreaker: policy)
                dest.mergeIn(from: a)
                dest.mergeIn(from: b)
                let once = dest
                #expect(mismatch(once, inputs: entries, capacity: 4, maximize: maximize, policy: policy) == nil)
                // A further disjoint source (indices 12-14) whose entries are all worse than the retained boundary
                // entry: a NaN, a value equal to the boundary value but with a larger index, and a strictly worse
                // number. Retained before this merge: ascending -inf@7, -1@5, 0@0, -0@1 (boundary value 0);
                // descending inf@6, 3.5@10, 2@4 and one 1 (1@2 under the index policies; `.smallerValue` may keep
                // 1@3 or 1@11), so the boundary value is 1. No pushed entry is strictly better than
                // the heap root under any policy (an equal value is never strictly better, even for
                // `.smallerValue`), so the buffer is bit-identical afterwards and still matches the union oracle.
                let worse = maximize
                    ? [Entry(value: .nan, index: 12), Entry(value: 1, index: 13), Entry(value: 0, index: 14)]
                    : [Entry(value: .nan, index: 12), Entry(value: 0, index: 13), Entry(value: 1, index: 14)]
                dest.mergeIn(from: buffer(worse, capacity: 3, maximize: maximize, policy: policy))
                #expect(sameBits(dest, once), "worse disjoint source changed the buffer (\(policy) max=\(maximize))")
                #expect(mismatch(dest, inputs: entries + worse, capacity: 4, maximize: maximize, policy: policy) == nil)
                // An empty source is also a no-op.
                dest.mergeIn(from: TopKBuffer(k: 4, isMinHeap: maximize, tieBreaker: policy))
                #expect(sameBits(dest, once))
            }
        }
    }

    // MARK: Explicit NaN and numeric-before-NaN ordering (literal expectations)

    @Test func numericsRankBeforeNaNInBothDirections() {
        let values: [Float] = [.nan, 5, .nan, 1, 2]
        for policy in policies {
            for maximize in [false, true] {
                let left = buffer((0..<2).map { Entry(value: values[$0], index: $0) }, capacity: 2, maximize: maximize, policy: policy)
                let right = buffer((2..<5).map { Entry(value: values[$0], index: $0) }, capacity: 3, maximize: maximize, policy: policy)
                var dest = TopKBuffer(k: 4, isMinHeap: maximize, tieBreaker: policy)
                dest.mergeIn(from: right)
                dest.mergeIn(from: left)
                let numerics = live(dest).filter { !$0.value.isNaN }.map(\.index).sorted()
                #expect(numerics == [1, 3, 4], "all three numerics are kept before any NaN (\(policy) max=\(maximize))")
                #expect(live(dest).filter { $0.value.isNaN }.count == 1, "one NaN fills the fourth slot")
                if policy != .smallerValue {
                    #expect(live(dest).first(where: { $0.value.isNaN })?.index == 0, "smaller index NaN wins the slot")
                }
            }
        }
    }

    // MARK: Alias and copy-on-write independence

    @Test func sourceAndDestinationStayIndependent() {
        for policy in policies {
            for maximize in [false, true] {
                let entriesA = (0..<6).map { Entry(value: Float($0), index: $0) }
                let entriesB = (6..<12).map { Entry(value: Float($0 % 4), index: $0) }
                let src = buffer(entriesB, capacity: 3, maximize: maximize, policy: policy)
                let srcBefore = src
                let original = buffer(entriesA, capacity: 4, maximize: maximize, policy: policy)
                let originalBefore = original

                var copy = original                  // shares storage with `original` until mutated
                copy.mergeIn(from: src)
                #expect(sameBits(original, originalBefore), "mutating the copy must not change the original")
                #expect(sameBits(src, srcBefore), "the source is never modified")
                #expect(mismatch(copy, inputs: entriesA + entriesB, capacity: 4, maximize: maximize, policy: policy) == nil)

                // destination that shares storage with the source
                var alias = src
                let sibling = buffer(entriesA, capacity: 3, maximize: maximize, policy: policy)
                alias.mergeIn(from: sibling)
                #expect(sameBits(src, srcBefore), "merging into a copy of src leaves src intact")

                // self-copy merge violates disjointness but must neither crash nor alter the original value
                var twin = src
                twin.mergeIn(from: twin)
                #expect(sameBits(src, srcBefore))
                #expect(twin.size <= twin.k)
            }
        }
    }
}

/// Precondition traps of `mergeIn(from:)`, each observed in a bounded child (a trap cannot be caught in-process).
@Suite("TopKBuffer mergeIn preconditions", .serialized)
struct TopKBufferMergeInTrapTests {
    static let childFilter = "TopKBufferMergeInTrapTests/child"

    private func expectTrap(_ scenario: String) throws {
        let child = try p01RunChild(scenario, filter: Self.childFilter)
        #expect(child.run.terminationReason == .uncaughtSignal, "\(scenario) status=\(child.run.statusDescription): \(child.tail)")
        #expect(child.output.contains("P01_OBSERVED \(scenario) before-merge"), "\(child.tail)")
        #expect(!child.output.contains("P01_OBSERVED \(scenario) after-merge"), "\(child.tail)")
    }

    @Test func directionMismatchTraps() throws { try expectTrap("mergein-direction") }
    @Test func tiePolicyMismatchTraps() throws { try expectTrap("mergein-policy") }
    @Test func smallerDestinationCapacityTraps() throws { try expectTrap("mergein-capacity") }

    @Test func child() {
        guard let scenario = ProcessInfo.processInfo.environment["P01_SCENARIO"], scenario.hasPrefix("mergein-") else { return }
        var dest = TopKBuffer(k: 2, isMinHeap: false)
        var src = TopKBuffer(k: 2, isMinHeap: false)
        switch scenario {
        case "mergein-direction": src = TopKBuffer(k: 2, isMinHeap: true)
        case "mergein-policy": src = TopKBuffer(k: 2, isMinHeap: false, tieBreaker: .smallerValue)
        case "mergein-capacity": src = TopKBuffer(k: 3, isMinHeap: false)
        default: Issue.record("Unknown scenario \(scenario)"); return
        }
        p01Marker("P01_OBSERVED \(scenario) before-merge")
        dest.mergeIn(from: src)
        p01Marker("P01_OBSERVED \(scenario) after-merge")
    }
}
