//
//  PointerSeamRegressionTests.swift
//  VectorCore
//
//  Regression locks for the pointer-level seams VectorIndex depends on
//  (beta-evolution-4, DOCUMENT-4 S1 + S2). These APIs already shipped in 0.2.2;
//  these tests pin the exact behavior so downstream hot paths can rely on it.
//

import Testing
import Foundation
@testable import VectorCore

// MARK: - S1: Pointer-level Top-K selection

@Suite("S1 pointer Top-K (regression lock)")
struct PointerTopKRegressionTests {

    /// Convenience: run the pointer-based select over a Swift array.
    private func ptrSelect(
        _ distances: [Float], k: Int, ids: [Int32]? = nil,
        tieBreaker: TieBreaker = .smallerIndex
    ) -> (indices: [Int32], distances: [Float]) {
        distances.withUnsafeBufferPointer { db in
            if let ids {
                return ids.withUnsafeBufferPointer { ib in
                    TopKSelection.select(k: k, from: db.baseAddress!, count: db.count,
                                         ids: ib.baseAddress!, tieBreaker: tieBreaker)
                }
            }
            return TopKSelection.select(k: k, from: db.baseAddress!, count: db.count,
                                        ids: nil, tieBreaker: tieBreaker)
        }
    }

    @Test("Pointer select matches the array select (distinct distances)")
    func parityWithArraySelect() {
        let distances: [Float] = [9, 3, 7, 1, 5, 8, 2, 6, 4, 0]
        let arr = TopKSelection.select(k: 4, from: distances)
        let ptr = ptrSelect(distances, k: 4)
        #expect(ptr.indices.map { Int($0) } == arr.indices)
        #expect(ptr.distances == arr.distances)
        #expect(ptr.indices == [9, 3, 6, 1].map(Int32.init))   // values 0,1,2,3
    }

    @Test("ids buffer remaps result indices")
    func idsRemap() {
        let distances: [Float] = [5, 1, 4, 2, 3]
        let ids: [Int32] = [1000, 1001, 1002, 1003, 1004]
        let r = ptrSelect(distances, k: 3, ids: ids)
        // smallest distances are at positions 1,3,4 → remapped to 1001,1003,1004
        #expect(r.indices == [1001, 1003, 1004])
        #expect(r.distances == [1, 2, 3])
    }

    @Test("Results are sorted ascending by distance")
    func sortedAscending() {
        let distances: [Float] = (0..<200).map { Float(($0 * 37) % 200) }
        let r = ptrSelect(distances, k: 25)
        #expect(r.distances == r.distances.sorted())
        #expect(r.indices.count == 25)
    }

    @Test("k greater than count returns count results")
    func kExceedsCount() {
        let distances: [Float] = [3, 1, 2]
        let r = ptrSelect(distances, k: 10)
        #expect(r.indices.count == 3)
        #expect(r.distances == [1, 2, 3])
    }

    @Test("Empty input and non-positive k return empty")
    func emptyAndZeroK() {
        #expect(ptrSelect([], k: 5).indices.isEmpty)
        #expect(ptrSelect([1, 2, 3], k: 0).indices.isEmpty)
    }

    @Test("Heap path (small k) and sort path (large k) agree with the array reference")
    func bothPathsCorrect() {
        // n = 1000: k = 10 (< n/10 → heap path) and k = 600 (>= n/10 → sort path)
        let distances: [Float] = (0..<1000).map { Float(($0 * 911) % 1000) }
        for k in [10, 600] {
            let arr = TopKSelection.select(k: k, from: distances)
            let ptr = ptrSelect(distances, k: k)
            #expect(ptr.indices.map { Int($0) } == arr.indices, "mismatch at k=\(k)")
            #expect(ptr.distances == arr.distances, "distance mismatch at k=\(k)")
        }
    }
}

// MARK: - S2: In-place raw-buffer normalize

@Suite("S2 in-place normalize (regression lock)")
struct NormalizeUncheckedRegressionTests {

    /// L2 norm computed in Double for measurement accuracy.
    private func l2(_ v: [Float]) -> Double {
        sqrt(v.reduce(0.0) { $0 + Double($1) * Double($1) })
    }

    /// Allocate a 16-byte-aligned Float buffer initialized from `values`.
    /// `normalizeUnchecked` uses aligned SIMD4 loads, so callers must supply
    /// aligned memory — this mirrors VectorIndex's real usage.
    private func withAlignedCopy(_ values: [Float], _ body: (UnsafeMutablePointer<Float>, Int) -> Void) {
        let p = try! AlignedMemory.allocateAligned(count: values.count, alignment: 64)
        values.withUnsafeBufferPointer { p.update(from: $0.baseAddress!, count: $0.count) }
        defer { AlignedMemory.deallocate(p) }
        body(p, values.count)
    }

    @Test("Normalizes an aligned buffer to unit L2 norm")
    func unitNorm() {
        let values = (0..<512).map { Float(sin(Double($0) * 0.3)) + 1.5 }
        withAlignedCopy(values) { p, n in
            NormalizeKernels.normalizeUnchecked(p, dimension: n)
            let out = Array(UnsafeBufferPointer(start: p, count: n))
            #expect(abs(l2(out) - 1.0) < 1e-4)
        }
    }

    @Test("Preserves direction (parity with typed normalizedUnchecked)")
    func parityWithTyped() {
        let values = (0..<512).map { Float(cos(Double($0) * 0.17)) + 0.5 }
        let typed = try! Vector512Optimized(values).normalizedUnchecked().toArray()
        withAlignedCopy(values) { p, n in
            NormalizeKernels.normalizeUnchecked(p, dimension: n)
            let ptr = Array(UnsafeBufferPointer(start: p, count: n))
            #expect(abs(l2(typed) - 1.0) < 1e-4)
            #expect(abs(l2(ptr) - 1.0) < 1e-4)
            // Same direction ⇒ dot of two unit vectors ≈ 1.
            let dot = zip(typed, ptr).reduce(0.0) { $0 + Double($1.0) * Double($1.1) }
            #expect(abs(dot - 1.0) < 1e-4)
        }
    }

    @Test("Handles dimensions that are not a multiple of 4 (SIMD tail)")
    func nonMultipleOf4() {
        for dim in [300, 301, 303] {
            let values = (0..<dim).map { Float($0 % 17) + 1 }   // all nonzero
            withAlignedCopy(values) { p, n in
                NormalizeKernels.normalizeUnchecked(p, dimension: n)
                let out = Array(UnsafeBufferPointer(start: p, count: n))
                #expect(abs(l2(out) - 1.0) < 1e-4, "dim \(dim) not unit norm")
            }
        }
    }

    @Test("Large magnitudes do not overflow")
    func largeMagnitudeNoOverflow() {
        let values = (0..<512).map { Float(Double($0 + 1) * 1e17) }   // up to ~5e19
        withAlignedCopy(values) { p, n in
            NormalizeKernels.normalizeUnchecked(p, dimension: n)
            let out = Array(UnsafeBufferPointer(start: p, count: n))
            #expect(out.allSatisfy { $0.isFinite })
            #expect(abs(l2(out) - 1.0) < 1e-4)
        }
    }

    @Test("Subnormal-dominated vectors never produce NaN/Inf")
    func subnormalNoPoison() {
        let values = (0..<512).map { Float(Double($0 + 1) * 1e-41) }   // subnormal range
        withAlignedCopy(values) { p, n in
            NormalizeKernels.normalizeUnchecked(p, dimension: n)
            let out = Array(UnsafeBufferPointer(start: p, count: n))
            // Contract: leave unchanged rather than poison with Inf/NaN.
            #expect(out.allSatisfy { $0.isFinite })
        }
    }
}
