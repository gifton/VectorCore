import Foundation
import Testing
@testable import VectorCore

// Independent bit-at-a-time oracle, deliberately separate from Core CRC/writers.
enum B01Frame {
    static func crc(_ bytes: [UInt8]) -> UInt32 {
        var value: UInt32 = 0xffff_ffff
        for byte in bytes {
            value ^= UInt32(byte)
            for _ in 0..<8 { value = (value >> 1) ^ (value & 1 == 0 ? 0 : 0xedb8_8320) }
        }
        return value ^ 0xffff_ffff
    }
    static func append(_ value: UInt32, bytes: Int, to output: inout [UInt8]) {
        for offset in 0..<bytes { output.append(UInt8(truncatingIfNeeded: value >> (8 * offset))) }
    }
    static func make(_ words: [UInt32], dimension: UInt32? = nil, version: UInt32 = 1, flags: UInt32 = 0) -> Data {
        var bytes: [UInt8] = []
        append(0x5645_4354, bytes: 4, to: &bytes)
        append(version, bytes: 2, to: &bytes)
        append(dimension ?? UInt32(words.count), bytes: 4, to: &bytes)
        append(flags, bytes: 2, to: &bytes)
        for word in words { append(word, bytes: 4, to: &bytes) }
        append(crc(bytes), bytes: 4, to: &bytes)
        return Data(bytes)
    }
    static func expectError(_ kind: VectorError.ErrorKind, _ body: () throws -> Void) {
        do { try body(); Issue.record("Expected VectorError.\(kind)") }
        catch let error as VectorError { #expect(error.kind == kind) }
        catch { Issue.record("Wrong error family: \(error)") }
    }
    static func bothReject(_ data: Data, _ kind: VectorError.ErrorKind) {
        expectError(kind) { _ = try DynamicVector.decodeBinary(from: data) }
        expectError(kind) { _ = try Vector<Dim3>.decodeBinary(from: data) }
    }
}

@Suite struct B01BinaryContractTests {
    @Test(arguments: [0, 1, 2])
    func emptyDynamicRoundTrip(_ route: Int) throws {
        let vector: DynamicVector
        switch route {
        case 0: vector = DynamicVector()
        case 1: vector = DynamicVector([])
        default: vector = DynamicVector([Float(7)]).dropLast()
        }
        #expect(vector.scalarCount == 0)
        let data = vector.encodeBinary()
        #expect(data == Data(B01Golden.emptyFrame))
        #expect(data.count == 16)
        let decoded = try DynamicVector.decodeBinary(from: data)
        #expect(decoded.scalarCount == 0)
        #expect(decoded.toArray().isEmpty)
        let json = try JSONEncoder().encode(vector)
        #expect(try JSONDecoder().decode(DynamicVector.self, from: json).scalarCount == 0)
    }

    @Test func nonemptyGoldenAndDirectSliceControls() throws {
        let values = B01Golden.ordinaryWords.map(Float.init(bitPattern:))
        let fixed = try Vector<Dim3>(values)
        let golden = Data(B01Golden.ordinaryFrame)
        #expect(fixed.encodeBinary() == golden)
        #expect(DynamicVector(values).encodeBinary() == golden)
        #expect(try DynamicVector.decodeBinary(from: golden).toArray().map(\.bitPattern) == B01Golden.ordinaryWords)
        for offset in [1, 2, 3, 5] {
            let backing = Data(repeating: 0xa5, count: offset) + golden + Data([0xff])
            let slice = backing[offset..<(offset + golden.count)]
            #expect(slice.startIndex == offset)
            #expect(try DynamicVector.decodeBinary(from: slice).toArray().map(\.bitPattern) == B01Golden.ordinaryWords)
            #expect(try Vector<Dim3>.decodeBinary(from: slice).toArray().map(\.bitPattern) == B01Golden.ordinaryWords)
        }
    }

    @Test func pathologicalStoredBitsAndSlices() throws {
        #expect(B01Frame.crc(Array("123456789".utf8)) == 0xcbf4_3926)
        #expect(B01Frame.make(B01Golden.specialWords) == Data(B01Golden.specialFrame))
        let values = B01Golden.specialWords.map(Float.init(bitPattern:))
        let dynamic = DynamicVector(values)
        let fixed = try Vector<Dim16>(values)
        // These constructors retain the fixture words, including signaling NaN payloads.
        #expect(dynamic.toArray().map(\.bitPattern) == B01Golden.specialWords)
        #expect(fixed.toArray().map(\.bitPattern) == B01Golden.specialWords)
        let frame = Data(B01Golden.specialFrame)
        #expect(dynamic.encodeBinary() == frame)
        #expect(fixed.encodeBinary() == frame)
        for offset in [1, 2, 3, 5] {
            let backing = Data(repeating: 0x55, count: offset) + frame + Data([0xff])
            let slice = backing[offset..<(offset + frame.count)]
            #expect(slice.startIndex == offset)
            #expect(try DynamicVector.decodeBinary(from: slice).toArray().map(\.bitPattern) == B01Golden.specialWords)
            #expect(try Vector<Dim16>.decodeBinary(from: slice).toArray().map(\.bitPattern) == B01Golden.specialWords)
        }
    }

    @Test func completeTruncationAndExactFrameLength() {
        let frame = Data(B01Golden.ordinaryFrame)
        for length in 0..<frame.count { B01Frame.bothReject(Data(frame.prefix(length)), .insufficientData) }
        B01Frame.bothReject(frame + Data([0]), .insufficientData)
        B01Frame.bothReject(frame + frame, .insufficientData)
        B01Frame.bothReject(Data(), .insufficientData)
        B01Frame.bothReject(Data(B01Golden.emptyFrame) + Data([0]), .insufficientData)
    }

    @Test func validationOrderAndReservedFlags() throws {
        var magic = Data(B01Golden.ordinaryFrame); magic[0] ^= 1
        B01Frame.bothReject(magic, .invalidData)
        for version: UInt32 in [0, 2, 65_535] {
            B01Frame.bothReject(B01Frame.make(B01Golden.ordinaryWords, version: version), .invalidData)
        }
        var payload = Data(B01Golden.ordinaryFrame); payload[12] ^= 1
        B01Frame.bothReject(payload, .dataCorruption)
        var checksum = Data(B01Golden.ordinaryFrame); checksum[checksum.count - 1] ^= 1
        B01Frame.bothReject(checksum, .dataCorruption)
        var emptyChecksum = Data(B01Golden.emptyFrame); emptyChecksum[15] ^= 1
        B01Frame.bothReject(emptyChecksum, .dataCorruption)
        let flags = B01Frame.make(B01Golden.ordinaryWords, flags: 0xffff)
        #expect(try DynamicVector.decodeBinary(from: flags).toArray().map(\.bitPattern) == B01Golden.ordinaryWords)
        #expect(try Vector<Dim3>.decodeBinary(from: flags).toArray().map(\.bitPattern) == B01Golden.ordinaryWords)
        #expect(try DynamicVector.decodeBinary(from: B01Frame.make([], flags: 0xffff)).scalarCount == 0)
        B01Frame.expectError(.dimensionMismatch) { _ = try Vector<Dim4>.decodeBinary(from: Data(B01Golden.ordinaryFrame)) }
    }

    @Test func dimensionCapAndTinyHeaderBomb() throws {
        for count in [9_999, 10_000] {
            let vector = DynamicVector(dimension: count, repeating: 0.5)
            let frame = vector.encodeBinary()
            #expect(frame.count == 16 + 4 * count)
            #expect(try DynamicVector.decodeBinary(from: frame).toArray().map(\.bitPattern) == vector.toArray().map(\.bitPattern))
        }
        let oversized = DynamicVector(dimension: 10_001, repeating: 0).encodeBinary()
        #expect(oversized.count == 40_020)
        B01Frame.bothReject(oversized, .invalidDimension)
        let bomb = B01Frame.make([], dimension: UInt32.max)
        #expect(bomb.count == 16)
        B01Frame.bothReject(bomb, .invalidDimension)
    }

    @Test func everySupportedPositiveFixedDimensionRejectsEmpty() {
        func check<D: StaticDimension>(_ type: D.Type) {
            B01Frame.expectError(.dimensionMismatch) { _ = try Vector<D>.decodeBinary(from: Data(B01Golden.emptyFrame)) }
        }
        check(Dim2.self); check(Dim3.self); check(Dim4.self); check(Dim8.self)
        check(Dim16.self); check(Dim32.self); check(Dim64.self); check(Dim128.self)
        check(Dim256.self); check(Dim384.self); check(Dim512.self); check(Dim768.self)
        check(Dim1024.self); check(Dim1536.self); check(Dim2048.self); check(Dim3072.self)
    }
}
