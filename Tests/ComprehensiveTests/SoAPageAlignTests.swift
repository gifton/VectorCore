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

    @Test("withUnsafeRawBuffer exposes the logical data length (both paths)")
    func rawBufferLength() {
        for aligned in [true, false] {
            let soa = SoA512.build(from: makeCandidates(10), pageAligned: aligned)
            soa.withUnsafeRawBuffer { raw in
                #expect(raw.count == 10 * 128 * 16)
                #expect(Int(bitPattern: raw.baseAddress!) % 16 == 0)
            }
        }
    }

    @Test("Already-page-multiple capacity rounds to itself")
    func exactPageMultiple() {
        let page = PlatformConfiguration.pageSize
        // count chosen so logical bytes == one page on the running arch.
        let count = page / (128 * 16)            // 512-dim: 16384/2048 = 8 on a 16KB page
        guard count > 0 else { return }          // skip on archs with a tiny page size
        let soa = SoA512.build(from: makeCandidates(count), pageAligned: true)
        if let (base, byteCount) = soa.pageAlignedBytes {
            #expect(Int(bitPattern: base) % page == 0)
            #expect(byteCount % page == 0)
            #expect(byteCount == count * 128 * 16)   // already a multiple → no extra padding
        } else {
            Issue.record("expected page-aligned bytes")
        }
    }

    @Test("Empty page-aligned SoA has no bytesNoCopy pointer")
    func emptyAligned() {
        let soa = SoA512.build(from: [], pageAligned: true)
        #expect(soa.pageAlignedBytes == nil)
    }
}
