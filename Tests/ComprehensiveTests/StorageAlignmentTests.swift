import Foundation
import Testing
@testable import VectorCore

@Suite("Storage Alignment")
struct StorageAlignmentSuite {

    @Test
    func testOptimalAlignmentIsPowerOfTwoAndAtLeastMinimum() {
        let a = AlignedMemory.optimalAlignment
        let m = AlignedMemory.minimumAlignment
        #expect(a >= m)
        #expect((a & (a - 1)) == 0) // power of two
    }

    @Test
    func testAllocateAlignedReturnsProperlyAlignedPointer_Default() throws {
        let ptr = try AlignedMemory.allocateAligned(count: 257) // non-power size
        defer { free(UnsafeMutableRawPointer(ptr)) }
        #expect(AlignedMemory.isAligned(ptr, to: AlignedMemory.optimalAlignment))
        #expect(AlignedMemory.isAligned(ptr, to: AlignedMemory.minimumAlignment))
    }

    @Test
    func testAllocateAlignedReturnsProperlyAlignedPointer_Custom() throws {
        let custom = max(AlignedMemory.minimumAlignment * 2, 32)
        let ptr = try AlignedMemory.allocateAligned(type: Float.self, count: 33, alignment: custom)
        defer { free(UnsafeMutableRawPointer(ptr)) }
        #expect(AlignedMemory.isAligned(ptr, to: custom))
    }

    @Test
    func testAlignedMemoryIsAlignedHelpersWorkForPointers() throws {
        let alignment = max(AlignedMemory.minimumAlignment * 2, 32)
        let base = try AlignedMemory.allocateAligned(type: Float.self, count: 8, alignment: alignment)
        defer { free(UnsafeMutableRawPointer(base)) }
        #expect(AlignedMemory.isAligned(base, to: alignment))
        // Offset by 1 element should generally break high alignment (>=32)
        let off = base.advanced(by: 1)
        #expect(!AlignedMemory.isAligned(off, to: alignment))
    }

    @Test
    func testAlignedValueStorageVerifyAlignmentReturnsTrue() {
        let storage = AlignedValueStorage(count: 128, alignment: 64)
        #expect(storage.verifyAlignment())
    }

    @Test
    func testVectorFixedDimBufferIsAligned() {
        let v = Vector<Dim512>(repeating: 1)
        v.withUnsafeBufferPointer { buf in
            if let p = buf.baseAddress {
                #expect(AlignedMemory.isAligned(p, to: AlignedMemory.minimumAlignment))
            } else {
                Issue.record("No baseAddress for Vector<Dim512> buffer")
            }
        }
    }

    @Test
    func testDynamicVectorBufferIsAlignedForLargeDimension() {
        let dv = DynamicVector(dimension: 4096, repeating: 0)
        dv.withUnsafeBufferPointer { buf in
            if let p = buf.baseAddress {
                #expect(AlignedMemory.isAligned(p, to: AlignedMemory.minimumAlignment))
            } else {
                Issue.record("No baseAddress for DynamicVector buffer")
            }
        }
    }

    @Test
    func testMinimumAlignmentSatisfiesAllAllocations() throws {
        let m = AlignedMemory.minimumAlignment
        for n in [1, 7, 15, 32, 255, 1024] {
            let ptr = try AlignedMemory.allocateAligned(type: UInt8.self, count: n, alignment: m)
            #expect(AlignedMemory.isAligned(ptr, to: m))
            free(UnsafeMutableRawPointer(ptr))
        }
    }

    @Test
    func testAlignmentConstantsAreReasonableForPlatform() {
        let a = AlignedMemory.optimalAlignment
        let m = AlignedMemory.minimumAlignment
        #expect(m >= 16)
        #expect(a >= m)
        #expect(a <= 256)
    }

}
