//
//  TopKTieBreakingTests.swift
//  VectorCore
//
//  Tests for configurable, deterministic Top-K tie-breaking
//  (beta-evolution-4, DOCUMENT-4 S3). Verifies that equal-distance ties are
//  resolved deterministically (default: smallerIndex) and consistently across
//  the array, pointer, large-k, and optimized selection paths.
//

import Testing
import Foundation
@testable import VectorCore

@Suite("Top-K tie-breaking determinism")
struct TopKTieBreakingTests {

    // MARK: - Default policy

    @Test("All-equal distances select the smallest indices, in order")
    func arrayAllEqual() {
        let distances = [Float](repeating: 1.0, count: 10)
        let r = TopKSelection.select(k: 3, from: distances)
        #expect(r.indices == [0, 1, 2])
    }

    @Test("Default tie-break equals explicit .smallerIndex")
    func defaultEqualsSmallerIndex() {
        let distances: [Float] = [5, 1, 1, 1, 2, 1]
        let a = TopKSelection.select(k: 3, from: distances)
        let b = TopKSelection.select(k: 3, from: distances, tieBreaker: .smallerIndex)
        #expect(a.indices == b.indices)
        #expect(a.distances == b.distances)
    }

    // MARK: - Boundary ties (which elements survive the cut)

    @Test("smallerIndex keeps the lowest indices among ties at the k boundary (large-k path)")
    func boundaryTiesLargeK() {
        // index 0 is the unique minimum; the remaining six tie at 2.0.
        // n=7, k=4 → k >= n/10 → large-k (sort) path.
        let distances: [Float] = [0.0, 2, 2, 2, 2, 2, 2]
        let r = TopKSelection.select(k: 4, from: distances, tieBreaker: .smallerIndex)
        #expect(r.indices == [0, 1, 2, 3])
    }

    @Test("smallerIndex is deterministic on the small-k heap path")
    func smallKHeapDeterministic() {
        // n=1000, k=50 → k < n/10 → heap path. All equal → indices 0..<50.
        let distances = [Float](repeating: 1.0, count: 1000)
        let r1 = TopKSelection.select(k: 50, from: distances, tieBreaker: .smallerIndex)
        let r2 = TopKSelection.select(k: 50, from: distances, tieBreaker: .smallerIndex)
        #expect(r1.indices == r2.indices)
        #expect(r1.indices == Array(0..<50))
    }

    // MARK: - Cross-path consistency

    @Test("Pointer path agrees with array path under ties")
    func pointerMatchesArray() {
        // Minima (value 1) sit at odd indices; expect the three smallest such indices.
        let distances: [Float] = [2, 1, 2, 1, 2, 1, 2, 1, 2, 1]
        let arr = TopKSelection.select(k: 3, from: distances, tieBreaker: .smallerIndex)
        let ptr = distances.withUnsafeBufferPointer { buf in
            TopKSelection.select(k: 3, from: buf.baseAddress!, count: buf.count,
                                 ids: nil, tieBreaker: .smallerIndex)
        }
        #expect(ptr.indices.map { Int($0) } == arr.indices)
        #expect(arr.indices == [1, 3, 5])
    }

    // MARK: - Optimized kernel path ordering

    @Test("Optimized nearestEuclidean512 orders identical candidates by smaller index")
    func optimizedDeterministicTies() {
        let q = try! Vector512Optimized([Float](repeating: 0, count: 512))
        let c = try! Vector512Optimized([Float](repeating: 1, count: 512))
        let candidates = Array(repeating: c, count: 8)  // all identical → equal distance
        let r = TopKSelection.nearestEuclidean512(k: 3, query: q, candidates: candidates)
        #expect(r.indices == [0, 1, 2])
    }

    // MARK: - Non-tie correctness is preserved

    @Test("Distinct distances are unaffected by the tie-breaker")
    func distinctUnaffected() {
        let distances: [Float] = [9, 3, 7, 1, 5]
        let r = TopKSelection.select(k: 3, from: distances, tieBreaker: .smallerIndex)
        #expect(r.indices == [3, 1, 4])           // values 1, 3, 5
        #expect(r.distances == [1, 3, 5])
    }
}
