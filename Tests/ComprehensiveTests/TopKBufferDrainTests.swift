import Foundation
import Testing
@testable import VectorCore

// P02 phase B: mechanical tests of `TopKBuffer.drainWorstFirst` (Sources/VectorCore/Operations/Kernels/
// TopKBufferDrain.swift) against an independent sorted oracle. Phase 1 check map rows T26-T31 and T33.
//
// The oracle restates the documented selection law (numbers before NaNs in both directions; ascending scores
// for a max-heap buffer, `isMinHeap == false`, descending for a min-heap one; equal scores, including +0/-0 and two
// NaNs, by the smaller index for `.smallerIndex` and `.insertionOrder`, and equivalent for `.smallerValue`) with its
// own comparator and `sorted`; it uses no production comparator, heap or selector. Entries are compared by index
// and exact score bits. The cost law (one heapify, then one sift-down per removal) leaves every output unchanged, so
// no test here can see it; it is checked by the phase B scaling probe outside the repository
// (P02/phaseB/scaling), because the repository keeps no wall-clock assertion outside its performance gate.

struct P02DrainEntry: Equatable, CustomStringConvertible {
    let index: Int
    let bits: UInt32
    init(_ index: Int, _ value: Float) { self.index = index; self.bits = value.bitPattern }
    var value: Float { Float(bitPattern: bits) }
    var description: String { "\(index):\(value)[0x\(String(bits, radix: 16))]" }
}

enum P02DrainOracle {
    /// True when `a` ranks strictly before (is better than) `b`.
    static func before(_ a: P02DrainEntry, _ b: P02DrainEntry, largest: Bool, policy: TieBreaker) -> Bool {
        if a.value.isNaN != b.value.isNaN { return !a.value.isNaN }
        if !a.value.isNaN && a.value != b.value { return largest ? a.value > b.value : a.value < b.value }
        switch policy {
        case .smallerIndex, .insertionOrder: return a.index < b.index
        case .smallerValue: return false
        }
    }

    /// Best-first kept entries for a total-order policy: the first `k` of all pushed entries sorted best first.
    static func bestFirst(_ pushed: [P02DrainEntry], k: Int, largest: Bool, policy: TieBreaker) -> [P02DrainEntry] {
        Array(pushed.sorted { before($0, $1, largest: largest, policy: policy) }.prefix(k))
    }
}

/// Scores with many exact ties, signed zeros, infinities, NaN payloads (positive and negative sign) and a
/// subnormal, mixed with small integers.
func p02DrainValues(_ count: Int, seed: UInt64) -> [Float] {
    let palette: [Float] = [0, -0.0, 1, 1, 2, -3, .infinity, -.infinity, Float(bitPattern: 0x7FC0_1234),
                            Float(bitPattern: 0xFFC0_0001), Float(bitPattern: 0x0000_0003), 0.5]
    var random = P02SplitMix(state: seed)
    return (0..<count).map { _ in
        random.next() % 3 == 0 ? palette[Int(random.next() % UInt64(palette.count))] : Float(Int(random.next() % 9) - 4)
    }
}

/// Pushes `values` (index = position + offset) and returns the buffer.
func p02Buffer(k: Int, largest: Bool, policy: TieBreaker, values: [Float], offset: Int = 0) -> TopKBuffer {
    var buffer = TopKBuffer(k: k, isMinHeap: largest, tieBreaker: policy)
    for (i, v) in values.enumerated() { buffer.pushIfBetter(val: v, idx: i + offset) }
    return buffer
}

/// Drains `buffer`, returning the callbacks in order.
func p02Drain(_ buffer: inout TopKBuffer) -> [P02DrainEntry] {
    var seen: [P02DrainEntry] = []
    buffer.drainWorstFirst { seen.append(P02DrainEntry($0, $1)) }
    return seen
}

@Suite("TopKBuffer drainWorstFirst")
struct TopKBufferDrainTests {
    static let totalPolicies: [TieBreaker] = [.smallerIndex, .insertionOrder]

    /// T26/T28/T30/T33, full buffer (more pushes than capacity, so the buffer is a heap with evictions): the
    /// callbacks arrive worst first, i.e. exactly the reverse of the oracle's best-first kept entries, and the buffer
    /// ends empty with its capacity and policy unchanged.
    @Test func fullDrainMatchesSortedOracle() {
        for k in [1, 2, 3, 7, 16, 100] {
            for largest in [false, true] {
                for policy in Self.totalPolicies {
                    let values = p02DrainValues(3 * k + 5, seed: UInt64(k) &* 31 &+ (largest ? 7 : 0))
                    let pushed = values.enumerated().map { P02DrainEntry($0.offset, $0.element) }
                    var buffer = p02Buffer(k: k, largest: largest, policy: policy, values: values)
                    #expect(buffer.size == k)
                    let drained = p02Drain(&buffer)
                    let expected = P02DrainOracle.bestFirst(pushed, k: k, largest: largest, policy: policy)
                    #expect(drained == Array(expected.reversed()), "k\(k) largest \(largest) \(policy)")
                    #expect(buffer.size == 0 && buffer.k == k && buffer.isMinHeap == largest && buffer.tieBreaker == policy)
                }
            }
        }
    }

    /// T26, partly filled buffer (`size < k`, never heapified by `pushIfBetter`): the drain heapifies first, so the
    /// order is still exactly the oracle's. This is the case a drain without the entry heapify gets wrong.
    @Test func partialDrainHeapifiesFirst() {
        for k in [2, 3, 8, 33, 100] {
            for size in Set([1, 2, k / 2, k - 1]) where size >= 1 && size < k {
                for largest in [false, true] {
                    for policy in Self.totalPolicies {
                        let values = p02DrainValues(size, seed: UInt64(k * 1000 + size))
                        let pushed = values.enumerated().map { P02DrainEntry($0.offset, $0.element) }
                        var buffer = p02Buffer(k: k, largest: largest, policy: policy, values: values)
                        #expect(buffer.size == size)
                        let drained = p02Drain(&buffer)
                        let expected = P02DrainOracle.bestFirst(pushed, k: k, largest: largest, policy: policy)
                        #expect(drained == Array(expected.reversed()), "k\(k) size\(size) largest \(largest) \(policy)")
                        #expect(buffer.size == 0)
                    }
                }
            }
        }
    }

    /// T27: an empty buffer (and a zero-capacity one) makes no callback and stays empty.
    @Test func emptyDrainMakesNoCall() {
        for k in [0, 1, 5] {
            var buffer = TopKBuffer(k: k, isMinHeap: false)
            if k == 0 { buffer.pushIfBetter(val: 1, idx: 0) }
            var calls = 0
            buffer.drainWorstFirst { _, _ in calls += 1 }
            #expect(calls == 0 && buffer.size == 0, "k\(k)")
        }
    }

    /// T29: a drained buffer is reusable: refill (partly and fully) and drain again, three rounds; each round matches
    /// the oracle of that round's pushes only.
    @Test func drainedBufferIsReusable() {
        for largest in [false, true] {
            var buffer = TopKBuffer(k: 9, isMinHeap: largest)
            for round in 0..<3 {
                let count = [5, 30, 9][round]
                let values = p02DrainValues(count, seed: UInt64(100 + round))
                for (i, v) in values.enumerated() { buffer.pushIfBetter(val: v, idx: 1_000 * round + i) }
                let pushed = values.enumerated().map { P02DrainEntry(1_000 * round + $0.offset, $0.element) }
                let drained = p02Drain(&buffer)
                let expected = P02DrainOracle.bestFirst(pushed, k: 9, largest: largest, policy: .smallerIndex)
                #expect(drained == Array(expected.reversed()), "round \(round) largest \(largest)")
                #expect(buffer.size == 0)
            }
        }
    }

    /// T30, `.smallerValue` (equal scores are equivalent, so their relative order is unspecified): with no eviction
    /// the drained entries are exactly the pushed ones (index and bits), and no entry arrives after one it ranks
    /// worse than.
    @Test func smallerValuePolicyDrainsAWeakOrder() {
        for k in [1, 4, 17, 64] {
            for largest in [false, true] {
                for size in Set([1, k / 2, k]) where size >= 1 {
                    let values = p02DrainValues(size, seed: UInt64(7 * k + size))
                    let pushed = values.enumerated().map { P02DrainEntry($0.offset, $0.element) }
                    var buffer = p02Buffer(k: k, largest: largest, policy: .smallerValue, values: values)
                    let drained = p02Drain(&buffer)
                    let key: (P02DrainEntry) -> String = { "\($0.index):\($0.bits)" }
                    #expect(drained.map(key).sorted() == pushed.map(key).sorted(), "k\(k) size\(size) largest \(largest)")
                    let ordered = zip(drained, drained.dropFirst()).allSatisfy {
                        !P02DrainOracle.before($0.0, $0.1, largest: largest, policy: .smallerValue)
                    }
                    #expect(ordered, "k\(k) size\(size) largest \(largest): \(drained)")
                }
            }
        }
    }

    /// T30: score bits pass through unchanged: NaN payloads of both signs, +0 and -0, both infinities, a subnormal.
    @Test func scoreBitsArePreserved() {
        let values: [Float] = [Float(bitPattern: 0x7FC0_1234), -0.0, 0.0, .infinity, -.infinity,
                               Float(bitPattern: 0xFFC0_0001), Float(bitPattern: 0x0000_0003)]
        for largest in [false, true] {
            var buffer = p02Buffer(k: values.count, largest: largest, policy: .smallerIndex, values: values)
            let drained = p02Drain(&buffer)
            let pushed = values.enumerated().map { P02DrainEntry($0.offset, $0.element) }
            #expect(drained == Array(P02DrainOracle.bestFirst(pushed, k: values.count, largest: largest,
                                                              policy: .smallerIndex).reversed()))
            #expect(Set(drained.map(\.bits)) == Set(values.map(\.bitPattern)))
        }
    }

    /// T31: value-copy independence: a copy taken before the drain keeps its size and the exact bits of its arrays,
    /// compared with snapshots taken BEFORE the drain into independent storage (`map` builds new arrays; phase B review
    /// F4: comparing the copy with the undrained original compared one shared storage with itself), and draining it
    /// later gives the same sequence.
    @Test func copyTakenBeforeDrainIsUnchanged() {
        let original = p02Buffer(k: 12, largest: false, policy: .smallerIndex, values: p02DrainValues(40, seed: 99))
        let snapshotBits = original.vals.map(\.bitPattern)
        let snapshotIndices = original.idxs.map { $0 }
        let copy = original
        var drained = original
        let first = p02Drain(&drained)
        #expect(drained.size == 0 && first.count == 12)
        #expect(copy.size == 12)
        #expect(copy.vals.map(\.bitPattern) == snapshotBits && copy.idxs == snapshotIndices)
        var later = copy
        #expect(p02Drain(&later) == first)
    }

    /// The intended consumer: filling a K-entry row from its end to its start yields the best-first row.
    @Test func consumerFillingFromTheEndBuildsTheBestFirstRow() {
        let values = p02DrainValues(50, seed: 5)
        var buffer = p02Buffer(k: 10, largest: false, policy: .smallerIndex, values: values)
        var row = [NearestNeighborResult](repeating: NearestNeighborResult(index: -1, distance: 0), count: buffer.size)
        var slot = row.count
        buffer.drainWorstFirst { index, score in
            slot -= 1
            row[slot] = NearestNeighborResult(index: index, distance: score)
        }
        let pushed = values.enumerated().map { P02DrainEntry($0.offset, $0.element) }
        #expect(slot == 0)
        #expect(row.map { P02DrainEntry($0.index, $0.distance) }
                == P02DrainOracle.bestFirst(pushed, k: 10, largest: false, policy: .smallerIndex))
    }
}
