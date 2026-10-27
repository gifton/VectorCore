import Foundation
import Testing
@testable import VectorCore

// P02 phase F (root ruling RD7): `scalarCount` of Vector384/512/768/1536Optimized is computed, not stored, so each
// handle holds only its `_storage` reference. The layout pin is the only test of handle size; the golden tests show
// that the JSON and B01 binary encodings, which read the VALUE of `scalarCount`, keep their exact bytes. Expected bytes
// are built here independently of Core's encoders: the JSON text from the format, the binary frame from the B01 layout
// (Docs/Serialization.md) with a bit-at-a-time CRC32 checked against B01Golden's generator-made frame.

/// Reflected CRC32 (polynomial 0xEDB88320, initial value and final xor 0xFFFFFFFF), bit at a time; independent of
/// Core's `crc32()`.
func p02CRC32(_ bytes: [UInt8]) -> UInt32 {
    var crc: UInt32 = 0xFFFF_FFFF
    for byte in bytes {
        crc ^= UInt32(byte)
        for _ in 0..<8 { crc = (crc & 1) == 1 ? (crc >> 1) ^ 0xEDB8_8320 : crc >> 1 }
    }
    return crc ^ 0xFFFF_FFFF
}

/// The B01 version1 frame of `words` (little-endian magic 0x56454354, version 1, dimension, zero flags, the words, then
/// the CRC32 of everything before it).
func p02B01Frame(_ words: [UInt32]) -> [UInt8] {
    func le32(_ v: UInt32) -> [UInt8] { (0..<4).map { UInt8(truncatingIfNeeded: v >> (8 * $0)) } }
    var frame = le32(0x5645_4354) + [0x01, 0x00] + le32(UInt32(words.count)) + [0x00, 0x00]
    for word in words { frame += le32(word) }
    return frame + le32(p02CRC32(frame))
}

@Suite("P02 optimized vector handle layout (computed scalarCount)")
struct P02HandleLayoutTests {

    /// Coordinates for a `width`-lane fixture: the 16 B01 special words first (signed zeros, infinities, quiet and
    /// signaling NaN payloads, subnormals, extremes), then the lane index as a Float.
    static func words(width: Int) -> [UInt32] {
        B01Golden.specialWords + (B01Golden.specialWords.count..<width).map { Float($0).bitPattern }
    }

    /// The handle is exactly its storage reference: the stride of each optimized type equals the stride of
    /// `ContiguousArray<SIMD4<Float>>` (the stored `scalarCount` made it twice that before P02 phase F).
    @Test func handleStrideIsTheStorageStride() {
        let storage = MemoryLayout<ContiguousArray<SIMD4<Float>>>.stride
        #expect(MemoryLayout<Vector384Optimized>.stride == storage)
        #expect(MemoryLayout<Vector512Optimized>.stride == storage)
        #expect(MemoryLayout<Vector768Optimized>.stride == storage)
        #expect(MemoryLayout<Vector1536Optimized>.stride == storage)
    }

    /// `scalarCount` is the type's dimension, and reflection shows only the storage child (the CHANGELOG notes that
    /// `Mirror` and `dump` output lose `scalarCount`).
    @Test func scalarCountIsTheDimensionAndNotAStoredChild() throws {
        func check<V: VectorProtocol>(_ vector: V, _ dimension: Int) {
            #expect(vector.scalarCount == dimension)
            #expect(Mirror(reflecting: vector).children.map(\.label) == ["_storage"], "\(V.self)")
        }
        check(Vector384Optimized(), 384)
        check(Vector512Optimized(), 512)
        check(Vector768Optimized(), 768)
        check(Vector1536Optimized(), 1536)
    }

    /// JSON: each type encodes as a flat array of its coordinates (no `scalarCount` key), byte for byte
    /// `[0,1,2,...,D-1]` for integer coordinates, and decodes back to an equal value.
    @Test func jsonGoldenBytesAtEveryWidth() throws {
        func check<V: VectorProtocol & Codable & Equatable>(_: V.Type, width: Int) throws where V.Scalar == Float {
            let vector = try V((0..<width).map { Float($0) })
            let data = try JSONEncoder().encode(vector)
            let golden = "[" + (0..<width).map { String($0) }.joined(separator: ",") + "]"
            #expect(Array(data) == Array(golden.utf8), "\(V.self): \(String(bytes: data.prefix(40), encoding: .utf8) ?? "?")")
            #expect(try JSONDecoder().decode(V.self, from: data) == vector, "\(V.self)")
        }
        try check(Vector384Optimized.self, width: 384)
        try check(Vector512Optimized.self, width: 512)
        try check(Vector768Optimized.self, width: 768)
        try check(Vector1536Optimized.self, width: 1536)
    }

    /// B01 binary: `encodeBinary()` emits exactly the independently built version1 frame (header with the dimension,
    /// every stored word including NaN payloads and signed zeros, CRC32). Control: the test's CRC and frame builder
    /// reproduce B01Golden's generator-made `ordinaryFrame` and `specialFrame`.
    @Test func binaryGoldenFramesAtEveryWidth() throws {
        #expect(p02B01Frame(B01Golden.ordinaryWords) == B01Golden.ordinaryFrame)
        #expect(p02B01Frame(B01Golden.specialWords) == B01Golden.specialFrame)
        func check<V: VectorProtocol>(_: V.Type, width: Int) throws where V.Scalar == Float {
            let words = Self.words(width: width)
            let vector = try V(words.map(Float.init(bitPattern:)))
            let frame = Array(vector.encodeBinary())
            #expect(frame.count == 16 + 4 * width, "\(V.self)")
            #expect(frame == p02B01Frame(words), "\(V.self)")
        }
        try check(Vector384Optimized.self, width: 384)
        try check(Vector512Optimized.self, width: 512)
        try check(Vector768Optimized.self, width: 768)
        try check(Vector1536Optimized.self, width: 1536)
    }
}
