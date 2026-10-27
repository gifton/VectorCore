import Foundation
import Testing
import VectorCore
#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif

private protocol BorrowContractFP16 {
    init(fp16Values: [UInt16])
    func withUnsafeFP16BufferPointer<R>(_ body: (UnsafeBufferPointer<UInt16>) throws -> R) rethrows -> R
}
extension MixedPrecisionKernels.Vector512FP16: BorrowContractFP16 {}
extension MixedPrecisionKernels.Vector768FP16: BorrowContractFP16 {}
extension MixedPrecisionKernels.Vector1536FP16: BorrowContractFP16 {}
private final class BorrowSentinel: Error {}

private struct ExternalEightLaneVector: SoACompatible {
    static let dimension = 8
    static let lanes = 2
    let storage: ContiguousArray<SIMD4<Float>>

    init(candidate: Int) {
        storage = [SIMD4<Float>(Float(candidate * 10), Float(candidate * 10 + 1), Float(candidate * 10 + 2), Float(candidate * 10 + 3)),
                   SIMD4<Float>(Float(candidate * 10 + 4), Float(candidate * 10 + 5), Float(candidate * 10 + 6), Float(candidate * 10 + 7))]
        precondition(Self.dimension == 4 * Self.lanes && storage.count == Self.lanes)
    }
}

@Suite struct StorageBorrowContractTests {
    private func borrow<V: BorrowContractFP16>(_ type: V.Type, width: Int) throws {
        let patterns: [UInt16] = [0, 0x8000, 1, 0x03ff, 0x0400, 0x3c00, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7c01, 0x7e55, 0xfeab]
        let bits = (0..<width).map { patterns[$0 % patterns.count] }
        var owner: V? = V(fp16Values: bits)
        var calls = 0
        let sum = owner!.withUnsafeFP16BufferPointer { buffer in
            calls += 1
            #expect(buffer.count == width)
            #expect(Array(buffer) == bits)
            #expect(buffer[0] == bits[0] && buffer[width - 1] == bits[width - 1])
            return buffer.reduce(UInt64(0)) { $0 + UInt64($1) }
        }
        #expect(calls == 1 && sum == bits.reduce(UInt64(0)) { $0 + UInt64($1) })
        let reverse = Array(bits.reversed())
        let nested = owner!.withUnsafeFP16BufferPointer { outer in
            V(fp16Values: reverse).withUnsafeFP16BufferPointer { inner in
                #expect(Array(outer) == bits && Array(inner) == reverse)
                return zip(outer, inner).reduce(UInt64(0)) { $0 + UInt64($1.0 ^ $1.1) }
            }
        }
        #expect(nested == zip(bits, reverse).reduce(UInt64(0)) { $0 + UInt64($1.0 ^ $1.1) })
        let temporary = V(fp16Values: bits).withUnsafeFP16BufferPointer { Array($0) }
        #expect(temporary == bits)
        let sentinel = BorrowSentinel()
        var throwingCalls = 0
        do {
            try V(fp16Values: bits).withUnsafeFP16BufferPointer { buffer in
                throwingCalls += 1
                #expect(Array(buffer) == bits)
                throw sentinel
            }
            Issue.record("Throwing borrow returned normally")
        } catch let caught as BorrowSentinel {
            #expect(caught === sentinel)
        }
        #expect(throwingCalls == 1)
        let owned: [UInt16] = owner!.withUnsafeFP16BufferPointer { Array($0) }
        owner = V(fp16Values: reverse)
        #expect(owned == bits)
        owner = nil
        #expect(owned == bits)
    }

    @Test func fp16Borrow512() throws { try borrow(MixedPrecisionKernels.Vector512FP16.self, width: 512) }
    @Test func fp16Borrow768() throws { try borrow(MixedPrecisionKernels.Vector768FP16.self, width: 768) }
    @Test func fp16Borrow1536() throws { try borrow(MixedPrecisionKernels.Vector1536FP16.self, width: 1536) }

    private func kernels<V: BorrowContractFP16>(
        _ type: V.Type, width: Int,
        dot: (V, V) -> Float, distance: (V, V) -> Float, cosine: (V, V) -> Float
    ) {
        let first: [Float] = [1, -2, 3, -4], second: [Float] = [2, 1, -1, 3]
        let a = (0..<width).map { first[$0 % 4] }, b = (0..<width).map { second[$0 % 4] }
        // Exact binary16 bits of `first` / `second` (Float16 is unavailable on Intel macOS); the oracle below is Double.
        let firstBits: [UInt16] = [0x3C00, 0xC000, 0x4200, 0xC400], secondBits: [UInt16] = [0x4000, 0x3C00, 0xBC00, 0x4200]
        let lhs = V(fp16Values: (0..<width).map { firstBits[$0 % 4] })
        let rhs = V(fp16Values: (0..<width).map { secondBits[$0 % 4] })
        let zero = V(fp16Values: Array(repeating: 0, count: width))
        let product = zip(a, b).reduce(0.0) { $0 + Double($1.0) * Double($1.1) }
        let squares = zip(a, b).reduce(0.0) { $0 + pow(Double($1.0) - Double($1.1), 2) }
        let normA = a.reduce(0.0) { $0 + Double($1) * Double($1) }
        let normB = b.reduce(0.0) { $0 + Double($1) * Double($1) }
        #expect(dot(lhs, rhs) == Float(product))
        #expect(abs(distance(lhs, rhs) - Float(squares.squareRoot())) <= 2e-6 * Float(squares.squareRoot()))
        #expect(abs(cosine(lhs, rhs) - Float(1 - product / (normA * normB).squareRoot())) <= 2e-6)
        #expect(dot(zero, lhs) == 0 && distance(zero, zero) == 0)
        #expect(cosine(zero, zero) == 1 && cosine(zero, lhs) == 1)
    }
    @Test func directFP16KernelMatrix() {
        kernels(MixedPrecisionKernels.Vector512FP16.self, width: 512, dot: MixedPrecisionKernels.dotFP16_512,
                distance: MixedPrecisionKernels.euclidean512, cosine: MixedPrecisionKernels.cosine512)
        kernels(MixedPrecisionKernels.Vector768FP16.self, width: 768, dot: MixedPrecisionKernels.dotFP16_768,
                distance: MixedPrecisionKernels.euclidean768, cosine: MixedPrecisionKernels.cosine768)
        kernels(MixedPrecisionKernels.Vector1536FP16.self, width: 1536, dot: MixedPrecisionKernels.dotFP16_1536,
                distance: MixedPrecisionKernels.euclidean1536, cosine: MixedPrecisionKernels.cosine1536)
    }

    @Test func externalConformerLayoutMatrix() {
        #expect(ExternalEightLaneVector.dimension == 8 && ExternalEightLaneVector.lanes == 2)
        for count in [0, 1, 5] {
            let vectors = (0..<count).map { ExternalEightLaneVector(candidate: $0) }
            #expect(vectors.allSatisfy { $0.storage.count == 2 })
            for aligned in [false, true] {
                let soa = SoA(vectors: vectors, pageAligned: aligned)
                let layout = soa.layoutDescriptor
                let logical = 2 * count * MemoryLayout<SIMD4<Float>>.stride
                let page = Int(sysconf(Int32(_SC_PAGESIZE)))
                #expect(page > 0)
                let allocated = aligned && count > 0 ? ((logical + page - 1) / page) * page : logical
                #expect(layout.lanes == 2 && layout.count == count)
                #expect(layout.elementCount == 2 * count && layout.laneStrideBytes == 16 * count)
                #expect(layout.logicalByteCount == logical && layout.allocatedByteCount == allocated)
                soa.withUnsafeRawBuffer { raw in
                    #expect(raw.count == logical)
                    let lanes = raw.bindMemory(to: SIMD4<Float>.self)
                    for lane in 0..<2 {
                        for candidate in 0..<count {
                            #expect(layout.elementIndex(lane: lane, candidate: candidate) == lane * count + candidate)
                            for scalar in 0..<4 {
                                #expect(lanes[lane * count + candidate][scalar] == Float(candidate * 10 + lane * 4 + scalar))
                            }
                        }
                    }
                }
                if aligned && count > 0 {
                    #expect(soa.pageAlignedBytes?.byteCount == allocated)
                    #expect(Int(bitPattern: soa.pageAlignedBytes!.base) % page == 0)
                } else {
                    #expect(soa.pageAlignedBytes == nil)
                }
            }
        }
    }

    @Test func ownedDynamicSliceSurvivesSourceChange() {
        for count in [8, 17, 512] {
            var source = DynamicVector((0..<count).map(Float.init))
            var slice = source.slice(from: 1, to: count - 1)
            source = DynamicVector(dimension: count, repeating: -1)
            #expect(slice.toArray() == (1..<(count - 1)).map(Float.init))
            slice[0] = 99
            #expect(source.toArray() == Array(repeating: -1, count: count))
        }
    }
}

extension StorageBorrowContractTests {
    private func ownedSoA<V: OptimizedVector>(_ vectors: [V], width: Int) throws {
        var owner: MixedPrecisionKernels.SoAFP16<V>? = try .init(vectors: vectors)
        let owned: ContiguousArray<UInt16> = owner!.storage
        let expected = (0..<(8 * width)).map { index -> UInt16 in
            let group = index / (width * 4)
            let candidate = group * 4 + index % 4
            // Exact binary16 bits of 1...5 (Float16 is unavailable on Intel macOS).
            return candidate < 5 ? [0x3C00, 0x4000, 0x4200, 0x4400, 0x4500][candidate] : 0
        }
        #expect(Array(owned) == expected)
        owner = nil
        #expect(Array(owned) == expected)
    }

    @Test func fp16SoAStorageRemainsOwned() throws {
        try ownedSoA((0..<5).map { Vector512Optimized(repeating: Float($0 + 1)) }, width: 512)
        try ownedSoA((0..<5).map { Vector768Optimized(repeating: Float($0 + 1)) }, width: 768)
        try ownedSoA((0..<5).map { Vector1536Optimized(repeating: Float($0 + 1)) }, width: 1536)
    }
}

#if !(arch(x86_64) && (os(macOS) || targetEnvironment(macCatalyst)))
extension StorageBorrowContractTests {
    /// Pins the literal binary16 constants used above against the native type (arm64 / iOS family).
    @Test func float16FreeBitConstantsMatchNativeFloat16() {
        #expect([1, -2, 3, -4].map { Float16($0).bitPattern } == [0x3C00, 0xC000, 0x4200, 0xC400])
        #expect([2, 1, -1, 3].map { Float16($0).bitPattern } == [0x4000, 0x3C00, 0xBC00, 0x4200])
        #expect((1...5).map { Float16($0).bitPattern } == [0x3C00, 0x4000, 0x4200, 0x4400, 0x4500])
    }
}
#endif
