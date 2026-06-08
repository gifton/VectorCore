//
//  SoALayoutContractTests.swift
//  VectorCore
//
//  Locks the frozen SoA memory-layout contract (Docs/SoA_Layout_Contract.md) that
//  VectorAccelerate's zero-copy Metal kernels couple to:
//    1. SoALayout descriptor fields + the `lane * count + candidate` index formula.
//    2. A golden parity fixture — known vectors → page-aligned SoA → analytic distances —
//       so a downstream GPU shader can be validated against our CPU kernel's intent.
//
//  If a change breaks these tests, it is a BREAKING CHANGE to the published layout
//  contract, not a routine refactor. Update the contract doc and notify consumers.
//

import Testing
import Foundation
import simd
@testable import VectorCore

@Suite("SoA Layout Contract — descriptor")
struct SoALayoutDescriptorTests {

    @Test("lanes = dimension / 4 for every SoACompatible type (no tail lanes)")
    func lanesAreDimensionOverFour() {
        #expect(Vector384Optimized.lanes == 384 / 4)
        #expect(Vector512Optimized.lanes == 512 / 4)
        #expect(Vector768Optimized.lanes == 768 / 4)
        #expect(Vector1536Optimized.lanes == 1536 / 4)
        // Every dimension is divisible by 4 ⇒ lanes are exact, no partial lane.
        #expect(384 % 4 == 0 && 512 % 4 == 0 && 768 % 4 == 0 && 1536 % 4 == 0)
    }

    @Test("element stride is 16 bytes (SIMD4<Float>)")
    func elementStrideIs16() {
        #expect(SoALayout.elementStrideBytes == 16)
        #expect(SoALayout.elementStrideBytes == MemoryLayout<SIMD4<Float>>.stride)
    }

    @Test("derived byte counts follow the frozen formulas")
    func derivedByteCounts() {
        // 768-dim, N = 10: lanes = 192.
        let layout = SoALayout(lanes: 192, count: 10,
                               allocatedByteCount: PlatformConfiguration.roundUpToPage(192 * 10 * 16))
        #expect(layout.lanes == 192)
        #expect(layout.count == 10)
        #expect(layout.elementCount == 192 * 10)
        #expect(layout.laneStrideBytes == 10 * 16)          // count * elementStride
        #expect(layout.logicalByteCount == 192 * 10 * 16)   // lanes * count * elementStride
        #expect(layout.allocatedByteCount >= layout.logicalByteCount)
        #expect(layout.allocatedByteCount % PlatformConfiguration.pageSize == 0)
    }

    @Test("elementIndex is lane * count + candidate")
    func elementIndexFormula() {
        let layout = SoALayout(lanes: 128, count: 5, allocatedByteCount: 0)
        #expect(layout.elementIndex(lane: 0, candidate: 0) == 0)
        #expect(layout.elementIndex(lane: 0, candidate: 4) == 4)
        #expect(layout.elementIndex(lane: 1, candidate: 0) == 5)   // next lane starts at +count
        #expect(layout.elementIndex(lane: 3, candidate: 2) == 3 * 5 + 2)
        #expect(layout.elementIndex(lane: 127, candidate: 4) == 127 * 5 + 4)
    }

    @Test("forType derives the same layout the live SoA reports (page-aligned)")
    func forTypeMatchesInstancePageAligned() throws {
        let candidates = try (0..<7).map { k in
            try Vector768Optimized(Array(repeating: Float(k), count: 768))
        }
        let soa = SoA<Vector768Optimized>.build(from: candidates, pageAligned: true)
        let derived = SoALayout.forType(Vector768Optimized.self, count: 7, pageAligned: true)
        #expect(derived == soa.layoutDescriptor)
        #expect(derived.allocatedByteCount == soa.pageAlignedBytes?.byteCount)
    }

    @Test("forType matches the live SoA for the non-page-aligned path")
    func forTypeMatchesInstancePlain() throws {
        let candidates = try (0..<7).map { k in
            try Vector512Optimized(Array(repeating: Float(k), count: 512))
        }
        let soa = SoA<Vector512Optimized>.build(from: candidates, pageAligned: false)
        let derived = SoALayout.forType(Vector512Optimized.self, count: 7, pageAligned: false)
        #expect(derived == soa.layoutDescriptor)
        // Non-page-aligned: allocated == logical, and no zero-copy pointer is offered.
        #expect(derived.allocatedByteCount == derived.logicalByteCount)
        #expect(soa.pageAlignedBytes == nil)
    }

    @Test("allocatedByteCount is page-rounded slack, never a source for count")
    func allocatedIsSlackNotCount() throws {
        // N = 5 @ 512-dim ⇒ logical = 128*5*16 = 10240 B; rounds up to a whole page.
        let candidates = try (0..<5).map { _ in
            try Vector512Optimized(Array(repeating: Float(1), count: 512))
        }
        let soa = SoA<Vector512Optimized>.build(from: candidates, pageAligned: true)
        let d = soa.layoutDescriptor
        #expect(d.logicalByteCount == 10240)
        #expect(d.allocatedByteCount == PlatformConfiguration.roundUpToPage(10240))
        #expect(d.allocatedByteCount > d.logicalByteCount)   // there IS trailing slack
        // Reconstructing N from the padded byte count would be WRONG — prove it differs.
        let wrongN = d.allocatedByteCount / (d.lanes * SoALayout.elementStrideBytes)
        #expect(wrongN != d.count)   // padded bytes ⇒ inflated N; consumers must use `count`
    }
}

@Suite("SoA Layout Contract — golden parity fixture")
struct SoALayoutParityFixtureTests {

    // Fixture construction (portable, reproducible by any consumer):
    //   query     q = [1, 1, …, 1]            (512 dims)
    //   candidate c_k = [1+k, 1+k, …, 1+k]    (512 dims), k = 0 … N-1
    // Then (q − c_k) = [−k, …] over 512 dims, so the Euclidean SQUARED distance is
    //   ‖q − c_k‖² = 512 · k².
    private static let N = 5
    private static let dim = 512
    private static func makeQuery() throws -> Vector512Optimized {
        try Vector512Optimized(Array(repeating: Float(1), count: dim))
    }
    private static func makeCandidates() throws -> [Vector512Optimized] {
        try (0..<N).map { k in try Vector512Optimized(Array(repeating: Float(1 + k), count: dim)) }
    }
    /// Expected Euclidean SQUARED distances: 512 · k² = [0, 512, 2048, 4608, 8192].
    private static let expectedSquared: [Float] = (0..<N).map { Float(dim) * Float($0 * $0) }

    @Test("page-aligned SoA buffer holds candidate j at elementIndex(lane, j)")
    func rawBufferMatchesIndexFormula() throws {
        let soa = SoA<Vector512Optimized>.build(from: try Self.makeCandidates(), pageAligned: true)
        let layout = soa.layoutDescriptor

        soa.withUnsafeRawBuffer { raw in
            let elems = raw.bindMemory(to: SIMD4<Float>.self)
            // Candidate j is all (1+j); every lane should read back that constant SIMD4.
            for j in 0..<Self.N {
                let expected = SIMD4<Float>(repeating: Float(1 + j))
                for lane in [0, 1, layout.lanes / 2, layout.lanes - 1] {
                    let idx = layout.elementIndex(lane: lane, candidate: j)
                    #expect(elems[idx] == expected)
                }
            }
        }
    }

    @Test("CPU SoA kernel produces the analytic golden distances")
    func cpuKernelMatchesGolden() throws {
        let query = try Self.makeQuery()
        let soa = SoA<Vector512Optimized>.build(from: try Self.makeCandidates(), pageAligned: true)

        var out = [Float](repeating: .nan, count: Self.N)
        out.withUnsafeMutableBufferPointer { buf in
            BatchKernels_SoA.euclid2_512(query: query, soa: soa, out: buf)
        }

        for k in 0..<Self.N {
            #expect(abs(out[k] - Self.expectedSquared[k]) < 1e-3,
                    "k=\(k): kernel \(out[k]) vs golden \(Self.expectedSquared[k])")
        }
    }

    @Test("descriptor for the fixture matches the documented golden values")
    func descriptorGoldenValues() throws {
        let soa = SoA<Vector512Optimized>.build(from: try Self.makeCandidates(), pageAligned: true)
        let d = soa.layoutDescriptor
        // These exact values appear in Docs/SoA_Layout_Contract.md §Golden fixture.
        #expect(d.lanes == 128)
        #expect(d.count == 5)
        #expect(d.laneStrideBytes == 80)        // 5 * 16
        #expect(d.logicalByteCount == 10240)    // 128 * 5 * 16
        #expect(d.elementIndex(lane: 1, candidate: 0) == 5)
        #expect(d.elementIndex(lane: 127, candidate: 4) == 639)
    }
}
