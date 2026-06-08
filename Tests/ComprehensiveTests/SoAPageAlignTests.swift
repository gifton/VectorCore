//
//  SoAPageAlignTests.swift
//  VectorCore
//
//  R1/R2: opt-in page-aligned SoA buffer + public bytesNoCopy-ready accessor.
//

import Testing
import Foundation
@testable import VectorCore

@Suite("SoA page alignment (R1/R2)")
struct SoAPageAlignTests {
    private func makeCandidates(_ n: Int) -> [Vector512Optimized] {
        (0..<n).map { k in try! Vector512Optimized((0..<512).map { Float(($0 + k) % 13) }) }
    }

    @Test("Page-aligned SoA exposes a page-aligned, page-length pointer")
    func pageAlignedExposesBytes() {
        let page = PlatformConfiguration.pageSize
        let soa = SoA512.build(from: makeCandidates(300), pageAligned: true)
        let bytes = soa.pageAlignedBytes
        #expect(bytes != nil)
        if let (base, byteCount) = bytes {
            #expect(Int(bitPattern: base) % page == 0, "base must be page-aligned")
            #expect(byteCount % page == 0, "length must be a page multiple")
            #expect(byteCount >= 300 * 128 * 16)   // count * lanes * sizeof(SIMD4<Float>)
        }
    }

    @Test("Non-page-aligned SoA returns nil for pageAlignedBytes")
    func defaultReturnsNil() {
        let soa = SoA512.build(from: makeCandidates(300))   // pageAligned defaults false
        #expect(soa.pageAlignedBytes == nil)
    }

    @Test("Page alignment preserves the SoA data layout")
    func dataIntegrity() {
        let candidates = makeCandidates(64)
        let aligned = SoA512.build(from: candidates, pageAligned: true)
        let plain = SoA512.build(from: candidates, pageAligned: false)
        for lane in 0..<aligned.lanes {
            let a = aligned.lanePointer(lane)
            let p = plain.lanePointer(lane)
            for j in 0..<64 { #expect(a[j] == p[j], "lane \(lane) candidate \(j)") }
        }
    }

    @Test("withUnsafeRawBuffer exposes the logical data length")
    func rawBufferLength() {
        let soa = SoA512.build(from: makeCandidates(10), pageAligned: true)
        soa.withUnsafeRawBuffer { base, byteCount in
            #expect(byteCount == 10 * 128 * 16)
            #expect(Int(bitPattern: base) % 16 == 0)
        }
    }

    @Test("Empty page-aligned SoA has no bytesNoCopy pointer")
    func emptyAligned() {
        let soa = SoA512.build(from: [], pageAligned: true)
        #expect(soa.pageAlignedBytes == nil)
    }
}
