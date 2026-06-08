//
//  UnifiedVectorBufferTests.swift
//  VectorCore
//
//  Tests for the zero-copy buffer contract (beta-evolution-4, DOCUMENT-4 S5):
//  UnifiedVectorBuffer protocol + PageAlignedBuffer page-aligned allocation path
//  for MTLDevice.makeBuffer(bytesNoCopy:) interop.
//

import Testing
import Foundation
@testable import VectorCore

@Suite("UnifiedVectorBuffer / PageAlignedBuffer")
struct UnifiedVectorBufferTests {

    // MARK: - Page alignment contract (the makeBuffer(bytesNoCopy:) requirement)

    @Test("Base address and length satisfy the bytesNoCopy contract")
    func pageAlignmentContract() {
        let page = PlatformConfiguration.pageSize
        // 512 floats = 2048 bytes — far smaller than a page, must round up.
        let buf = PageAlignedBuffer(elementCount: 512)

        buf.withUnsafeContiguousBytes { raw in
            let addr = Int(bitPattern: raw.baseAddress!)
            #expect(addr % page == 0, "base address must be page-aligned")
        }
        #expect(Int(bitPattern: buf.baseAddress) % page == 0)
        #expect(buf.allocatedByteCount % page == 0, "allocated length must be a page multiple")
        #expect(buf.allocatedByteCount >= 512 * MemoryLayout<Float>.stride)
        #expect(buf.alignment == page)
        #expect(buf.elementCount == 512)
        #expect(buf.byteCount == 512 * MemoryLayout<Float>.stride)
    }

    @Test("Logical length and padded length round correctly across page boundaries")
    func lengthRounding() {
        let page = PlatformConfiguration.pageSize
        func padded(_ floats: Int) -> Int {
            let logical = floats * MemoryLayout<Float>.stride
            return ((logical + page - 1) / page) * page
        }
        // Sub-page, multi-page, and exact-page cases.
        for floats in [1, 512, 4096, 5000, 8192] {
            let buf = PageAlignedBuffer(elementCount: floats)
            #expect(buf.allocatedByteCount == padded(floats))
            #expect(buf.allocatedByteCount % page == 0)
            #expect(buf.allocatedByteCount >= floats * MemoryLayout<Float>.stride)
        }
        // 4096 floats == exactly one 16KB page on Apple Silicon → no extra padding.
        if page == 16384 {
            #expect(PageAlignedBuffer(elementCount: 4096).allocatedByteCount == 16384)
        }
    }

    // MARK: - Zero initialization (padding must be zero for GPU reads)

    @Test("Buffer is zero-initialized including padding")
    func zeroInitialized() {
        let buf = PageAlignedBuffer(elementCount: 100)
        // Logical region zero.
        buf.withUnsafeContiguousBytes { raw in
            for i in 0..<100 {
                let v = raw.load(fromByteOffset: i * MemoryLayout<Float>.stride, as: Float.self)
                #expect(v == 0)
            }
        }
        // Padding region (beyond logical, up to allocatedByteCount) also zero.
        let logical = 100 * MemoryLayout<Float>.stride
        let rawBase = UnsafeRawBufferPointer(start: buf.baseAddress, count: buf.allocatedByteCount)
        for off in stride(from: logical, to: buf.allocatedByteCount, by: MemoryLayout<Float>.stride) {
            #expect(rawBase.load(fromByteOffset: off, as: Float.self) == 0)
        }
    }

    // MARK: - Mutate / read round-trip

    @Test("Mutable write is visible through the read contract")
    func mutateReadRoundTrip() {
        let buf = PageAlignedBuffer(elementCount: 256)
        buf.withUnsafeMutableBufferPointer { ptr in
            for i in 0..<256 { ptr[i] = Float(i) * 0.5 }
        }
        buf.withUnsafeContiguousBytes { raw in
            #expect(raw.count == 256 * MemoryLayout<Float>.stride)
            for i in 0..<256 {
                let v = raw.load(fromByteOffset: i * MemoryLayout<Float>.stride, as: Float.self)
                #expect(v == Float(i) * 0.5)
            }
        }
    }

    @Test("init(copying:) preserves values and count")
    func copyingInit() {
        let source = (0..<384).map { Float($0) * 1.25 }
        let buf = PageAlignedBuffer(copying: source)
        #expect(buf.elementCount == 384)
        buf.withUnsafeContiguousBytes { raw in
            for i in 0..<384 {
                let v = raw.load(fromByteOffset: i * MemoryLayout<Float>.stride, as: Float.self)
                #expect(v == source[i])
            }
        }
    }

    // MARK: - Ownership transfer (relinquish for Metal bytesNoCopy deallocator)

    @Test("consumeAllocation transfers ownership and preserves pointer identity")
    func consumeAllocation() {
        let buf = PageAlignedBuffer(elementCount: 128)
        #expect(buf.ownsAllocation == true)
        let before = buf.baseAddress

        let (base, length) = buf.consumeAllocation()
        #expect(buf.ownsAllocation == false, "buffer must relinquish ownership")
        #expect(base == before, "pointer identity preserved across transfer")
        #expect(length == buf.allocatedByteCount)

        // Caller is now responsible for freeing — deinit must NOT double-free.
        AlignedMemory.deallocate(base)
        // buf deinits here without freeing (verified by absence of double-free crash).
    }

    // MARK: - Conformance of optimized vector types (read contract, not page-aligned)

    @Test("Vector512Optimized conforms to the read contract")
    func optimizedConformance() {
        let values = (0..<512).map { Float($0) * 0.01 }
        let v = try! Vector512Optimized(values)
        let u: any UnifiedVectorBuffer = v

        #expect(u.elementCount == 512)
        #expect(u.alignment >= 16, "SIMD4 storage is at least 16-byte aligned")
        u.withUnsafeContiguousBytes { raw in
            #expect(raw.count == 512 * MemoryLayout<Float>.stride)
            let addr = Int(bitPattern: raw.baseAddress!)
            #expect(addr % u.alignment == 0)
            for i in 0..<512 {
                let got = raw.load(fromByteOffset: i * MemoryLayout<Float>.stride, as: Float.self)
                #expect(abs(got - values[i]) < 1e-6)
            }
        }
    }

    @Test("DynamicVector conforms to the read contract")
    func dynamicConformance() {
        let values = (0..<300).map { Float($0) * 0.1 }
        let v = DynamicVector(values)
        let u: any UnifiedVectorBuffer = v

        #expect(u.elementCount == 300)
        u.withUnsafeContiguousBytes { raw in
            #expect(raw.count == 300 * MemoryLayout<Float>.stride)
            for i in 0..<300 {
                let got = raw.load(fromByteOffset: i * MemoryLayout<Float>.stride, as: Float.self)
                #expect(abs(got - values[i]) < 1e-6)
            }
        }
    }
}
